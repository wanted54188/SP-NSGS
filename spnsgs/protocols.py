from __future__ import annotations
import csv,json,math,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import Snapshot,inspect_reference,load_observation_snapshots
from .solver import SpectralVorticitySolver
def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _dtype(name: str) -> torch.dtype:
    result = getattr(torch, name, None)
    if result not in (torch.float32, torch.float64):
        raise ValueError("paper solver dtype must be 'float32' or 'float64'")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_sensor_indices(
    ny: int, nx: int, count: int, seed: int
) -> np.ndarray:
    """Deterministic stratified sensor layout on the periodic grid."""
    if count >= ny * nx:
        return np.arange(ny * nx, dtype=np.int64)
    rng = np.random.default_rng(seed)
    aspect = nx / ny
    columns = max(1, min(nx, int(round(math.sqrt(count * aspect)))))
    rows = max(1, min(ny, int(math.ceil(count / columns))))
    while rows * columns < count:
        if columns < nx:
            columns += 1
        elif rows < ny:
            rows += 1
        else:
            break
    y_edges = np.linspace(0, ny, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, nx, columns + 1, dtype=np.int64)
    selected: list[int] = []
    for iy in range(rows):
        for ix in range(columns):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            row = int(rng.integers(y0, y1))
            column = int(rng.integers(x0, x1))
            selected.append(row * nx + column)
    selected_array = np.unique(np.asarray(selected, dtype=np.int64))
    if selected_array.size > count:
        selected_array = np.sort(
            rng.choice(selected_array, size=count, replace=False)
        )
    elif selected_array.size < count:
        remaining = np.setdiff1d(
            np.arange(ny * nx, dtype=np.int64), selected_array
        )
        extra = rng.choice(
            remaining, size=count - selected_array.size, replace=False
        )
        selected_array = np.sort(np.concatenate([selected_array, extra]))
    return selected_array


def _prepare_reference(
    config: Config, device: torch.device, dtype: torch.dtype
) -> tuple[list[Snapshot], dict[str, torch.Tensor]]:
    snapshots = load_observation_snapshots(config)
    snapshots = [
        snapshot
        for snapshot in snapshots
        if config.data.train_time_min_s - 1e-12
        <= snapshot.time_s
        <= config.data.train_time_max_s + 1e-12
    ]
    if len(snapshots) < 2:
        raise ValueError("paper solver requires at least two regularly spaced snapshots")
    times = np.asarray([snapshot.time_s for snapshot in snapshots])
    differences = np.diff(times)
    if not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-9):
        raise ValueError("paper solver requires a uniform observation time interval")
    shape = (len(snapshots), snapshots[0].ny, snapshots[0].nx)

    def stacked(name: str) -> torch.Tensor:
        values = [getattr(snapshot, name) for snapshot in snapshots]
        if any(value is None for value in values):
            raise ValueError(f"paper-solver reference field {name!r} is unavailable")
        array = np.stack(values).reshape(shape)
        return torch.as_tensor(array, device=device, dtype=dtype)

    reference = {
        "time": torch.as_tensor(times, device=device, dtype=dtype),
        "omega": stacked("omega"),
        "u": stacked("u"),
        "v": stacked("v"),
        "q_sgs_x": stacked("q_sgs_x"),
        "q_sgs_y": stacked("q_sgs_y"),
        "pi_sgs": stacked("pi_sgs"),
        "div_q_sgs": stacked("div_q_sgs"),
    }
    return snapshots, reference


def _time_varying_sparse_subset(
    parent_indices: np.ndarray,
    omega: torch.Tensor,
    count: int,
    ny: int,
    nx: int,
    importance_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    indices, _ = _time_varying_sparse_subset_with_weights(
        parent_indices,
        omega,
        count,
        ny,
        nx,
        importance_fraction,
        0.25,
        rng,
    )
    return indices


def _time_varying_sparse_subset_with_weights(
    parent_indices: np.ndarray,
    omega: torch.Tensor,
    count: int,
    ny: int,
    nx: int,
    importance_fraction: float,
    importance_pool_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the same sparse label budget from a fixed sensor cloud.

    The importance proxy is formed only from omega values already available
    at ``parent_indices``.  Consequently it changes where the unchanged label
    budget is spent, but it never inspects a dense SGS target field.
    """
    parent_indices = np.asarray(parent_indices, dtype=np.int64)
    if count < 1 or count > parent_indices.size:
        raise ValueError("sparse subset count must lie within its parent set")
    if not 0.0 <= importance_fraction <= 1.0:
        raise ValueError("closure_importance_fraction must be in [0, 1]")
    if not 0.0 < importance_pool_fraction < 1.0:
        raise ValueError("closure_importance_pool_fraction must be in (0, 1)")
    importance_count = int(round(count * importance_fraction))
    omega_at_parent = (
        omega.reshape(omega.shape[0], -1)[:, parent_indices]
        .detach()
        .cpu()
        .numpy()
    )
    parent_y = parent_indices // nx
    parent_x = parent_indices % nx
    delta_y = np.abs(parent_y[:, None] - parent_y[None, :])
    delta_x = np.abs(parent_x[:, None] - parent_x[None, :])
    delta_y = np.minimum(delta_y, ny - delta_y)
    delta_x = np.minimum(delta_x, nx - delta_x)
    distance_squared = (delta_x**2 + delta_y**2).astype(np.float64)
    np.fill_diagonal(distance_squared, np.inf)
    neighbour_count = min(4, max(1, parent_indices.size - 1))
    nearest = np.argsort(distance_squared, axis=1, kind="stable")[
        :, :neighbour_count
    ]
    rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    for time_index in range(omega.shape[0]):
        local_values = omega_at_parent[time_index]
        neighbour_values = local_values[nearest]
        proxy = np.max(
            np.abs(local_values[:, None] - neighbour_values), axis=1
        )
        if importance_count <= 0:
            chosen = rng.choice(parent_indices.size, size=count, replace=False)
            sample_weights = np.full(count, 1.0 / count, dtype=np.float64)
        else:
            high_size = max(
                importance_count,
                int(round(parent_indices.size * importance_pool_fraction)),
            )
            high_size = min(high_size, parent_indices.size - 1)
            high = np.argsort(proxy, kind="stable")[-high_size:]
            low = np.setdiff1d(
                np.arange(parent_indices.size, dtype=np.int64), high
            )
            high_count = min(importance_count, high.size)
            low_count = count - high_count
            if low_count > low.size:
                shortfall = low_count - low.size
                low_count = low.size
                high_count += shortfall
            chosen_high = rng.choice(high, size=high_count, replace=False)
            chosen_low = rng.choice(low, size=low_count, replace=False)
            chosen = np.concatenate([chosen_high, chosen_low])
            high_weight = high.size / (
                parent_indices.size * max(1, high_count)
            )
            low_weight = low.size / (
                parent_indices.size * max(1, low_count)
            )
            sample_weights = np.concatenate(
                [
                    np.full(high_count, high_weight),
                    np.full(low_count, low_weight),
                ]
            )
        order = np.argsort(parent_indices[chosen], kind="stable")
        rows.append(parent_indices[chosen][order])
        weights = sample_weights[order]
        weight_rows.append(weights / np.sum(weights))
    return np.stack(rows), np.stack(weight_rows)


def _sensor_protocol(
    config: Config,
    snapshots: list[Snapshot],
    reference: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    dict,
]:
    size = snapshots[0].size
    train_count = max(1, int(round(size * config.paper.train_sensor_fraction)))
    validation_count = max(
        1, int(round(size * config.paper.validation_sensor_fraction))
    )
    closure_count = max(
        1, int(round(size * config.paper.closure_sensor_fraction))
    )
    if closure_count > train_count:
        raise ValueError(
            "closure sensors must be a subset of the flow training sensors"
        )
    total_count = train_count + validation_count
    if total_count >= size:
        raise ValueError("train+validation sensor fractions must be below 1")
    all_indices = _select_sensor_indices(
        snapshots[0].ny, snapshots[0].nx, total_count, config.training.seed
    )
    rng = np.random.default_rng(config.training.seed + 91)
    permutation = rng.permutation(all_indices)
    train_np = np.sort(permutation[:train_count])
    validation_np = np.sort(permutation[train_count:])
    # Keep exactly ``closure_count`` labels per snapshot. The sampler may allocate a
    # configurable fraction with a high-gradient proxy formed exclusively
    # from the already available 4% omega sensors.  No dense SGS field or
    # additional label location is consulted when constructing this proxy.
    importance_fraction = float(config.paper.closure_importance_fraction)
    if not 0.0 <= importance_fraction <= 1.0:
        raise ValueError("closure_importance_fraction must be in [0, 1]")
    closure_np, closure_weight_np = _time_varying_sparse_subset_with_weights(
        train_np,
        reference["omega"],
        closure_count,
        snapshots[0].ny,
        snapshots[0].nx,
        importance_fraction,
        float(config.paper.closure_importance_pool_fraction),
        rng,
    )
    train = torch.as_tensor(train_np, device=device, dtype=torch.long)
    validation = torch.as_tensor(validation_np, device=device, dtype=torch.long)
    closure = torch.as_tensor(closure_np, device=device, dtype=torch.long)
    closure_weights = torch.as_tensor(
        closure_weight_np,
        device=device,
        dtype=reference["omega"].dtype,
    )

    flat_omega = reference["omega"].reshape(reference["omega"].shape[0], -1)
    flat_u = reference["u"].reshape(reference["u"].shape[0], -1)
    flat_v = reference["v"].reshape(reference["v"].shape[0], -1)
    scales = {
        "omega": float(torch.sqrt(torch.mean(flat_omega[:, train].square())).item()),
        "velocity": float(
            torch.sqrt(
                0.5
                * torch.mean(
                    flat_u[:, train].square() + flat_v[:, train].square()
                )
            ).item()
        ),
    }
    protocol = {
        "model": (
            "theory-faithful SP-NSGS: SPD dissipative branch plus bounded "
            "backscatter, trained by solver-embedded sparse rollout"
        ),
        "checkpoint_format_version": int(
            config.paper.checkpoint_format_version
        ),
        "closure_parameterization": config.paper.closure_parameterization,
        "initial_condition": (
            f"complete observed field at t={snapshots[0].time_s:g} s; "
            "not a future-time supervision label. This is not a t=0 "
            "STAR initial field unless that export exists"
        ),
        "future_training_labels": (
            "u, v and omega at fixed 4% sensors; div(q_sgs) and local "
            "q_sgs/transfer at a time-rotating 0.25% subset of those same "
            "positions"
        ),
        "closure_sensor_layout": (
            "25 labels per snapshot, deterministically rotated within the "
            "fixed 400 flow-training sensors; total label count unchanged"
        ),
        "state_observer": (
            "periodic multi-scale Poisson-consistent nudging only between "
            "autonomous training windows, using training u, v and omega "
            "sensors; validation sensors are never assimilated"
        ),
        "closure_training_state": (
            "the uncorrected multi-step forecast state inside each training "
            "window; sparse observations anchor only the next window"
        ),
        "checkpoint_selection": (
            "combined held-out assimilated score plus a complete autonomous "
            "five-second sparse-flow validation score"
        ),
        "sparse_moment_losses": (
            "velocity and vorticity second moments computed from the same "
            "training sensors; no additional labels"
        ),
        "sparse_increment_loss": (
            "periodic nearest-neighbour omega increments formed only from the "
            "same fixed flow sensors; no additional labels or dense fields"
        ),
        "dense_future_fields_used_for_training": False,
        "closure_labels_used_for_training": True,
        "train_sensor_count": train_count,
        "validation_sensor_count": validation_count,
        "closure_sensor_count_per_snapshot": closure_count,
        "closure_sensor_unique_count": int(np.unique(closure_np).size),
        "grid_point_count": size,
        "train_sensor_fraction": train_count / size,
        "validation_sensor_fraction": validation_count / size,
        "closure_sensor_fraction": closure_count / size,
        "closure_importance_fraction": importance_fraction,
        "closure_importance_pool_fraction": float(
            config.paper.closure_importance_pool_fraction
        ),
        "closure_loss_weighting": (
            "exact two-stratum inverse-probability weights; active sampling "
            "is unbiased for the fixed parent sensor cloud"
        ),
        "closure_importance_proxy": (
            "nearest-neighbour omega increments on fixed flow sensors only"
        ),
        "train_indices": train_np.tolist(),
        "validation_indices": validation_np.tolist(),
        "closure_indices_by_time": closure_np.tolist(),
        "closure_weights_by_time": closure_weight_np.tolist(),
        "normalisation_from_training_sensors_only": scales,
    }
    return train, validation, closure, closure_weights, scales, protocol


def _periodic_nearest_sensor_pairs(
    indices: torch.Tensor,
    ny: int,
    nx: int,
    neighbours: int,
) -> torch.Tensor:
    """Return unique local-index pairs on the periodic sensor point cloud.

    Pair entries index the sensor vectors, not the full grid.  Consequently the
    increment loss reuses exactly the same sparse labels as the pointwise loss.
    """
    count = int(indices.numel())
    if count < 2 or neighbours <= 0:
        return torch.empty((0, 2), device=indices.device, dtype=torch.long)
    flat = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    y = flat // nx
    x = flat % nx
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx, nx - dx)
    dy = np.minimum(dy, ny - dy)
    distance_squared = dx * dx + dy * dy
    np.fill_diagonal(distance_squared, np.iinfo(np.int64).max)
    nearest_count = min(int(neighbours), count - 1)
    pairs: set[tuple[int, int]] = set()
    for first in range(count):
        nearest = np.argsort(distance_squared[first], kind="stable")[:nearest_count]
        for second in nearest:
            pair = (first, int(second))
            pairs.add(pair if pair[0] < pair[1] else (pair[1], pair[0]))
    pair_array = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return torch.as_tensor(pair_array, device=indices.device, dtype=torch.long)


def _periodic_longitudinal_flux_pairs(
    indices: torch.Tensor,
    ny: int,
    nx: int,
    dx_m: float,
    dy_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sparse multi-scale longitudinal-flux pairs on a periodic grid.

    For every labelled point the candidate set contains its nearest neighbour,
    second-nearest neighbour, and the neighbour closest to the median periodic
    distance.  Pair entries index the existing sparse label vector; this does
    not create or query any additional SGS labels.
    """
    count = int(indices.numel())
    empty_pairs = torch.empty((0, 2), device=indices.device, dtype=torch.long)
    empty_directions = torch.empty(
        (0, 2), device=indices.device, dtype=torch.get_default_dtype()
    )
    if count < 2:
        return empty_pairs, empty_directions

    flat = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    y = flat // nx
    x = flat % nx
    signed_dx = (x[None, :] - x[:, None] + nx / 2.0) % nx - nx / 2.0
    signed_dy = (y[None, :] - y[:, None] + ny / 2.0) % ny - ny / 2.0
    displacement_x = float(dx_m) * signed_dx
    displacement_y = float(dy_m) * signed_dy
    distance = np.hypot(displacement_x, displacement_y)
    np.fill_diagonal(distance, np.inf)

    pairs: set[tuple[int, int]] = set()
    for first in range(count):
        ordered = np.argsort(distance[first], kind="stable")
        finite = ordered[np.isfinite(distance[first, ordered])]
        if finite.size == 0:
            continue
        selected = [int(finite[0])]
        if finite.size > 1:
            selected.append(int(finite[1]))
        finite_distances = distance[first, finite]
        medium_target = float(np.median(finite_distances))
        selected.append(
            int(finite[np.argmin(np.abs(finite_distances - medium_target))])
        )
        for second in selected:
            pair = (first, second)
            pairs.add(pair if first < second else (second, first))

    pair_array = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    if pair_array.size == 0:
        return empty_pairs, empty_directions
    first = pair_array[:, 0]
    second = pair_array[:, 1]
    direction = np.stack(
        [displacement_x[first, second], displacement_y[first, second]], axis=1
    )
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    return (
        torch.as_tensor(pair_array, device=indices.device, dtype=torch.long),
        torch.as_tensor(
            direction,
            device=indices.device,
            dtype=torch.get_default_dtype(),
        ),
    )


def _stable_weighted_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """Centred Pearson correlation with positive normalised weights."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if prediction.numel() < 2:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if weights is None:
        weights = torch.full_like(prediction, 1.0 / prediction.numel())
    else:
        weights = weights.reshape(-1).to(
            device=prediction.device, dtype=prediction.dtype
        )
        weights = weights / weights.sum().clamp_min(float(epsilon))
    prediction_fluctuation = prediction - torch.sum(weights * prediction)
    target_fluctuation = target - torch.sum(weights * target)
    covariance = torch.sum(
        weights * prediction_fluctuation * target_fluctuation
    )
    variance_product = torch.sum(
        weights * prediction_fluctuation.square()
    ) * torch.sum(weights * target_fluctuation.square())
    return covariance / torch.sqrt(variance_product + float(epsilon))


def _normalised_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-12)
    scale = torch.sqrt(torch.sum(weights * target.square())).detach().clamp_min(1e-6)
    error = (prediction - target) / scale
    absolute_error = torch.abs(error)
    robust_error = torch.where(
        absolute_error <= delta,
        0.5 * error.square(),
        delta * (absolute_error - 0.5 * delta),
    )
    return torch.sum(weights * robust_error)


def _sign_balanced_transfer_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Equal-weight positive/negative total-transfer supervision."""
    group_losses: list[torch.Tensor] = []
    for mask in (target >= 0.0, target < 0.0):
        if int(torch.count_nonzero(mask).item()) == 0:
            continue
        group_losses.append(
            _normalised_huber(
                prediction[mask], target[mask], weights[mask], delta
            )
        )
    if not group_losses:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if len(group_losses) == 1:
        return 0.5 * group_losses[0]
    return 0.5 * (group_losses[0] + group_losses[1])


def _periodic_fourier_quadrature_weights(
    indices: torch.Tensor, ny: int, nx: int, maximum_mode: int = 2
) -> torch.Tensor:
    """Positive coordinate-only quadrature for low periodic modes.

    The weights integrate sine/cosine modes through ``maximum_mode`` and stay
    close to a uniform sensor average.  Construction uses sensor coordinates
    only, never any flow or closure reference value.
    """
    flat = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    x = 2.0 * np.pi * (flat % nx) / nx
    y = 2.0 * np.pi * (flat // nx) / ny
    features = [np.ones_like(x)]
    for kx in range(maximum_mode + 1):
        for ky in range(-maximum_mode, maximum_mode + 1):
            if kx == 0 and ky <= 0:
                continue
            phase = kx * x + ky * y
            features.extend((np.cos(phase), np.sin(phase)))
    matrix = np.stack(features, axis=0)
    target = np.zeros(matrix.shape[0], dtype=np.float64)
    target[0] = 1.0
    uniform = np.full(flat.size, 1.0 / flat.size, dtype=np.float64)
    correction = matrix.T @ np.linalg.solve(
        matrix @ matrix.T + 1e-4 * np.eye(matrix.shape[0]),
        target - matrix @ uniform,
    )
    weights = np.maximum(uniform + correction, 0.0)
    weights /= np.sum(weights)
    return torch.as_tensor(
        weights, device=indices.device, dtype=torch.get_default_dtype()
    )


from .reference import load_training_snapshots as load_observation_snapshots
