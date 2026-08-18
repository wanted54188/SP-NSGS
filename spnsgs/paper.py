from __future__ import annotations

import csv
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import Snapshot, inspect_reference, load_observation_snapshots
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
            "complete prescribed t=0 field; it is an initial condition, not a "
            "future-time supervision label"
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


def _sensor_loss(
    solver: SpectralVorticitySolver,
    omega: torch.Tensor,
    target_index: int,
    indices: torch.Tensor,
    reference: dict[str, torch.Tensor],
    scales: dict[str, float],
    closure_indices: torch.Tensor | None = None,
    closure_weights: torch.Tensor | None = None,
    increment_pairs: torch.Tensor | None = None,
    closure_increment_pairs: torch.Tensor | None = None,
    closure_increment_directions: torch.Tensor | None = None,
    closure_omega: torch.Tensor | None = None,
    sensor_quadrature_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state = solver.state(omega)
    closure_state = (
        state if closure_omega is None else solver.state(closure_omega)
    )
    flat_omega = state["omega"].reshape(-1)
    flat_u = state["u"].reshape(-1)
    flat_v = state["v"].reshape(-1)
    omega_target = reference["omega"][target_index].reshape(-1)[indices]
    u_target = reference["u"][target_index].reshape(-1)[indices]
    v_target = reference["v"][target_index].reshape(-1)[indices]
    # Normalise at each target time.  A single all-time scale makes the later,
    # weaker vorticity snapshots artificially cheap and rewards collapse to a
    # smooth low-energy field.
    omega_scale_squared = torch.mean(omega_target.square()).detach().clamp_min(1e-12)
    velocity_scale_squared = (
        0.5 * torch.mean(u_target.square() + v_target.square())
    ).detach().clamp_min(1e-12)
    omega_prediction = flat_omega[indices]
    u_prediction = flat_u[indices]
    v_prediction = flat_v[indices]
    omega_loss = torch.mean((omega_prediction - omega_target).square()) / omega_scale_squared
    velocity_loss = 0.5 * torch.mean(
        (u_prediction - u_target).square()
        + (v_prediction - v_target).square()
    ) / velocity_scale_squared
    zero = torch.zeros((), device=omega.device, dtype=omega.dtype)
    omega_increment_loss = zero
    if increment_pairs is not None and increment_pairs.numel() > 0:
        first = increment_pairs[:, 0]
        second = increment_pairs[:, 1]
        predicted_increment = omega_prediction[first] - omega_prediction[second]
        target_increment = omega_target[first] - omega_target[second]
        increment_scale_squared = (
            torch.mean(target_increment.square()).detach().clamp_min(1e-12)
        )
        omega_increment_loss = torch.mean(
            (predicted_increment - target_increment).square()
        ) / increment_scale_squared

    def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.sum(prediction * target)
        denominator = torch.sqrt(
            torch.sum(prediction.square()) * torch.sum(target.square()) + 1e-12
        )
        return 1.0 - numerator / denominator

    correlation_loss = 0.5 * (
        cosine_loss(omega_prediction, omega_target)
        + cosine_loss(
            torch.cat([u_prediction, v_prediction]),
            torch.cat([u_target, v_target]),
        )
    )
    if sensor_quadrature_weights is None:
        quadrature_weights = torch.full_like(
            u_target, 1.0 / max(1, u_target.numel())
        )
    else:
        quadrature_weights = sensor_quadrature_weights.to(
            device=omega.device, dtype=omega.dtype
        )
        quadrature_weights = quadrature_weights / quadrature_weights.sum().clamp_min(
            1e-12
        )
    if solver.config.paper.coordinate_weighted_sparse_moments:
        # The target remains a sparse quadrature.  Only the prediction is
        # integrated on its available computational grid.
        predicted_energy = 0.5 * torch.mean(
            state["u"].square() + state["v"].square()
        )
        predicted_enstrophy = 0.5 * torch.mean(state["omega"].square())
    else:
        predicted_energy = 0.5 * torch.mean(
            u_prediction.square() + v_prediction.square()
        )
        predicted_enstrophy = 0.5 * torch.mean(omega_prediction.square())
    target_energy = 0.5 * torch.sum(
        quadrature_weights * (u_target.square() + v_target.square())
    )
    sparse_energy_loss = (
        (predicted_energy - target_energy) / target_energy.detach().clamp_min(1e-12)
    ).square()
    target_enstrophy = 0.5 * torch.sum(
        quadrature_weights * omega_target.square()
    )
    sparse_enstrophy_loss = (
        (predicted_enstrophy - target_enstrophy)
        / target_enstrophy.detach().clamp_min(1e-12)
    ).square()
    closure = solver.closure_fields(closure_state)
    divergence_field = solver.project(
        solver.derivative(closure["qx"], "x")
        + solver.derivative(closure["qy"], "y")
    )
    flux_scale = torch.mean(
        closure["qx"].square() + closure["qy"].square()
    ) / max(scales["omega"] ** 2 * scales["velocity"] ** 2, 1e-12)
    # Positive Pi means net forward enstrophy transfer.  Local backscatter is
    # allowed; only a negative spatial mean is weakly penalised.
    mean_pi = torch.mean(closure["pi"])
    dissipation_penalty = torch.relu(-mean_pi) / (
        torch.mean(torch.abs(closure["pi"])).detach().clamp_min(1e-6)
    )
    dissipation_coefficient_fraction = closure["c_theta"] / max(
        solver.config.paper.dissipation_coefficient_soft_limit, 1e-12
    )
    dissipation_coefficient_saturation = torch.mean(
        torch.relu(dissipation_coefficient_fraction - 0.9).square()
    )
    omega_spectrum = torch.fft.fft2(state["omega"])
    spectral_energy = omega_spectrum.real.square() + omega_spectrum.imag.square()
    axis_cutoff = solver.config.paper.dealias_fraction * torch.minimum(
        torch.amax(torch.abs(solver.kx)), torch.amax(torch.abs(solver.ky))
    )
    tail_start = (
        solver.config.paper.spectral_tail_start_fraction * axis_cutoff
    )
    tail_mask = (
        (torch.sqrt(solver.k_squared) >= tail_start) & solver.dealias
    )
    spectral_tail_fraction = torch.sum(spectral_energy[tail_mask]) / torch.sum(
        spectral_energy[solver.dealias]
    ).clamp_min(1e-12)
    spectral_tail_excess = torch.relu(
        spectral_tail_fraction
        / max(
            solver.config.paper.spectral_tail_max_enstrophy_fraction,
            1e-12,
        )
        - 1.0
    ).square()
    anisotropy_condition_regularization = torch.mean(
        torch.relu(
            closure["A_condition_number"]
            / max(solver.config.paper.anisotropy_condition_soft_limit, 1e-12)
            - 1.0
        ).square()
    )
    beta = max(solver.config.paper.maximum_backscatter_ratio, 1e-12)
    backscatter_saturation = torch.mean(
        torch.relu(closure["backscatter_ratio"] / beta - 0.9).square()
    )
    sparse_sgs_divergence = zero
    sparse_sgs_divergence_correlation = zero
    sparse_sgs_transfer = zero
    sparse_sgs_mean_transfer = zero
    sparse_sgs_branch_transfer = zero
    sparse_sgs_flux = zero
    sparse_sgs_longitudinal_increment = zero
    if closure_indices is not None and closure_indices.numel() > 0:
        if closure_weights is None:
            label_weights = torch.full_like(
                closure_indices,
                1.0 / closure_indices.numel(),
                dtype=omega.dtype,
            )
        else:
            label_weights = closure_weights.to(
                device=omega.device, dtype=omega.dtype
            )
            label_weights = label_weights / label_weights.sum().clamp_min(1e-12)
        divergence_prediction = divergence_field.reshape(-1)[closure_indices]
        divergence_target = reference["div_q_sgs"][target_index].reshape(-1)[
            closure_indices
        ]
        divergence_scale_squared = torch.sum(
            label_weights * divergence_target.square()
        ).detach().clamp_min(1e-12)
        sparse_sgs_divergence = torch.sum(
            label_weights * (divergence_prediction - divergence_target).square()
        ) / divergence_scale_squared
        if divergence_prediction.numel() >= 16:
            divergence_correlation = _stable_weighted_correlation(
                divergence_prediction,
                divergence_target,
                label_weights,
                solver.config.paper.closure_correlation_epsilon,
            )
            sparse_sgs_divergence_correlation = 1.0 - divergence_correlation

        transfer_prediction = closure["pi"].reshape(-1)[closure_indices]
        transfer_target = reference["pi_sgs"][target_index].reshape(-1)[
            closure_indices
        ]
        transfer_scale_squared = torch.sum(
            label_weights * transfer_target.square()
        ).detach().clamp_min(1e-12)
        sparse_sgs_transfer = _sign_balanced_transfer_loss(
            transfer_prediction,
            transfer_target,
            label_weights,
            max(
                float(solver.config.paper.sparse_sgs_transfer_huber_delta),
                1e-6,
            ),
        )
        # The same probability-corrected sparse labels also constrain the
        # signed mean transfer.  This prevents a low pointwise loss from
        # retaining the large forward-transfer bias seen in the previous run.
        sparse_sgs_mean_transfer = (
            torch.sum(label_weights * (transfer_prediction - transfer_target))
            / torch.sqrt(transfer_scale_squared)
        ).square()
        flux_x_prediction = closure["qx"].reshape(-1)[closure_indices]
        flux_y_prediction = closure["qy"].reshape(-1)[closure_indices]
        flux_x_target = reference["q_sgs_x"][target_index].reshape(-1)[
            closure_indices
        ]
        flux_y_target = reference["q_sgs_y"][target_index].reshape(-1)[
            closure_indices
        ]
        flux_scale_squared = (
            0.5
            * torch.sum(
                label_weights
                * (flux_x_target.square() + flux_y_target.square())
            )
        ).detach().clamp_min(1e-12)
        sparse_sgs_flux = 0.5 * torch.sum(
            label_weights
            * (
                (flux_x_prediction - flux_x_target).square()
                + (flux_y_prediction - flux_y_target).square()
            )
        ) / flux_scale_squared
        if (
            closure_increment_pairs is not None
            and closure_increment_directions is not None
            and closure_increment_pairs.numel() > 0
        ):
            first = closure_increment_pairs[:, 0]
            second = closure_increment_pairs[:, 1]
            directions = closure_increment_directions.to(
                device=omega.device, dtype=omega.dtype
            )
            predicted_increment = (
                (flux_x_prediction[second] - flux_x_prediction[first])
                * directions[:, 0]
                + (flux_y_prediction[second] - flux_y_prediction[first])
                * directions[:, 1]
            )
            target_increment = (
                (flux_x_target[second] - flux_x_target[first])
                * directions[:, 0]
                + (flux_y_target[second] - flux_y_target[first])
                * directions[:, 1]
            )
            increment_scale_squared = torch.mean(
                target_increment.square()
            ).detach().clamp_min(1e-12)
            sparse_sgs_longitudinal_increment = torch.mean(
                (predicted_increment - target_increment).square()
            ) / increment_scale_squared

    def reference_time_derivative(field: torch.Tensor) -> torch.Tensor:
        count = field.shape[0]
        if count < 2:
            raise ValueError("time-derivative supervision requires two snapshots")
        if target_index >= 2:
            dt = reference["time"][target_index] - reference["time"][target_index - 1]
            return (
                3.0 * field[target_index]
                - 4.0 * field[target_index - 1]
                + field[target_index - 2]
            ) / (2.0 * dt)
        dt = reference["time"][target_index] - reference["time"][target_index - 1]
        return (field[target_index] - field[target_index - 1]) / dt

    physics_inferred_divergence = zero
    sparse_energy_tendency = zero
    needs_resolved_budget = (
        solver.config.paper.physics_inferred_divergence_weight > 0.0
        or solver.config.paper.sparse_energy_tendency_weight > 0.0
    )
    if needs_resolved_budget:
        nonlinear = solver.project(
            closure_state["u"] * closure_state["omega_x"]
            + closure_state["v"] * closure_state["omega_y"]
        )
        forcing = solver.vorticity_forcing(
            float(reference["time"][target_index].item()),
            closure_state["omega"],
        )
        sparse_omega_history = reference["omega"].reshape(
            reference["omega"].shape[0], -1
        )[:, indices]
        omega_t_at_sensors = reference_time_derivative(sparse_omega_history)
        inferred_at_sensors = (
            -omega_t_at_sensors
            - nonlinear.reshape(-1)[indices]
            + solver.config.physics.kinematic_viscosity_m2_s
            * closure_state["laplacian_omega"].reshape(-1)[indices]
            + forcing.reshape(-1)[indices]
        ).detach()
        # The filtered STAR snapshots and the RK/pseudo-spectral solver are
        # not exactly discretely consistent.  Use the unchanged 0.25% direct
        # SGS labels to calibrate only the scalar amplitude of the inferred
        # source before extending it to all 4% flow sensors.
        if closure_indices is not None and closure_indices.numel() > 0:
            local_closure_indices = torch.searchsorted(indices, closure_indices)
            inferred_at_labels = inferred_at_sensors[local_closure_indices]
            direct_at_labels = reference["div_q_sgs"][target_index].reshape(-1)[
                closure_indices
            ]
            calibration = torch.sum(
                inferred_at_labels * direct_at_labels
            ) / torch.sum(inferred_at_labels.square()).clamp_min(1e-12)
            calibration = calibration.detach().clamp(0.5, 1.5)
            inferred_at_sensors = calibration * inferred_at_sensors
        predicted_at_sensors = divergence_field.reshape(-1)[indices]
        inferred_scale = torch.sqrt(
            torch.sum(quadrature_weights * inferred_at_sensors.square())
        ).detach().clamp_min(1e-6)
        normalised_error = (
            predicted_at_sensors - inferred_at_sensors
        ) / inferred_scale
        delta = max(
            float(solver.config.paper.physics_inferred_divergence_huber_delta),
            1e-6,
        )
        absolute_error = torch.abs(normalised_error)
        robust_error = torch.where(
            absolute_error <= delta,
            0.5 * normalised_error.square(),
            delta * (absolute_error - 0.5 * delta),
        )
        physics_inferred_divergence = torch.sum(
            quadrature_weights * robust_error
        )

        def sparse_reference_energy(time_index: int) -> torch.Tensor:
            target_u = reference["u"][time_index].reshape(-1)[indices]
            target_v = reference["v"][time_index].reshape(-1)[indices]
            return 0.5 * torch.sum(
                quadrature_weights * (target_u.square() + target_v.square())
            )

        if target_index >= 2:
            dt = reference["time"][target_index] - reference["time"][target_index - 1]
            energy_dt = (
                3.0 * sparse_reference_energy(target_index)
                - 4.0 * sparse_reference_energy(target_index - 1)
                + sparse_reference_energy(target_index - 2)
            ) / (2.0 * dt)
        else:
            energy_dt = (
                sparse_reference_energy(target_index)
                - sparse_reference_energy(target_index - 1)
            ) / (
                reference["time"][target_index]
                - reference["time"][target_index - 1]
            )
        predicted_rhs = solver.project(
            -nonlinear
            + solver.config.physics.kinematic_viscosity_m2_s
            * closure_state["laplacian_omega"]
            - divergence_field
            + forcing
        )
        predicted_energy_dt = torch.mean(closure_state["psi"] * predicted_rhs)
        duration = (
            reference["time"][-1] - reference["time"][0]
        ).detach().clamp_min(1e-6)
        energy_dt_scale = torch.maximum(
            torch.abs(energy_dt).detach(),
            (target_energy.detach() / duration).clamp_min(1e-6),
        )
        tendency_error = (predicted_energy_dt - energy_dt.detach()) / energy_dt_scale
        sparse_energy_tendency = torch.where(
            torch.abs(tendency_error) <= 1.0,
            0.5 * tendency_error.square(),
            torch.abs(tendency_error) - 0.5,
        )
    total = (
        solver.config.paper.omega_sensor_weight * omega_loss
        + solver.config.paper.velocity_sensor_weight * velocity_loss
        + solver.config.paper.sensor_correlation_weight * correlation_loss
        + solver.config.paper.omega_increment_weight * omega_increment_loss
        + solver.config.paper.sparse_energy_weight * sparse_energy_loss
        + solver.config.paper.sparse_enstrophy_weight * sparse_enstrophy_loss
        + solver.config.paper.flux_regularization_weight * flux_scale
        + solver.config.paper.anisotropy_condition_weight
        * anisotropy_condition_regularization
        + solver.config.paper.backscatter_saturation_weight
        * backscatter_saturation
        + solver.config.paper.mean_dissipation_weight * dissipation_penalty
        + solver.config.paper.dissipation_coefficient_saturation_weight
        * dissipation_coefficient_saturation
        + solver.config.paper.sparse_sgs_divergence_weight
        * sparse_sgs_divergence
        + solver.config.paper.sparse_sgs_divergence_correlation_weight
        * sparse_sgs_divergence_correlation
        + solver.config.paper.sparse_sgs_transfer_weight
        * sparse_sgs_transfer
        + solver.config.paper.sparse_sgs_mean_transfer_weight
        * sparse_sgs_mean_transfer
        + solver.config.paper.sparse_sgs_branch_transfer_weight
        * sparse_sgs_branch_transfer
        + solver.config.paper.sparse_sgs_flux_weight * sparse_sgs_flux
        + solver.config.paper.sparse_sgs_longitudinal_increment_weight
        * sparse_sgs_longitudinal_increment
        + solver.config.paper.physics_inferred_divergence_weight
        * physics_inferred_divergence
        + solver.config.paper.sparse_energy_tendency_weight
        * sparse_energy_tendency
        + solver.config.paper.spectral_tail_weight * spectral_tail_excess
    )
    return total, {
        "omega_sensor": omega_loss,
        "velocity_sensor": velocity_loss,
        "sensor_correlation": correlation_loss,
        "omega_increment": omega_increment_loss,
        "sparse_energy": sparse_energy_loss,
        "sparse_enstrophy": sparse_enstrophy_loss,
        "flux_regularization": flux_scale,
        "anisotropy_condition_regularization": anisotropy_condition_regularization,
        "backscatter_saturation": backscatter_saturation,
        "mean_anisotropy_condition": torch.mean(closure["A_condition_number"]),
        "maximum_backscatter_ratio_realised": torch.max(
            closure["backscatter_ratio"]
        ),
        "maximum_local_backscatter_transfer_ratio_realised": torch.max(
            closure["local_backscatter_transfer_ratio"]
        ),
        "mean_backscatter_transfer_limiter": torch.mean(
            closure["backscatter_transfer_limiter"]
        ),
        "mean_dissipative_transfer": torch.mean(closure["pi_d"]),
        "mean_backscatter_transfer": torch.mean(closure["pi_b"]),
        "mean_dissipation_penalty": dissipation_penalty,
        "dissipation_coefficient_saturation": dissipation_coefficient_saturation,
        "sparse_sgs_divergence": sparse_sgs_divergence,
        "sparse_sgs_divergence_correlation": sparse_sgs_divergence_correlation,
        "sparse_sgs_transfer": sparse_sgs_transfer,
        "sparse_sgs_mean_transfer": sparse_sgs_mean_transfer,
        "sparse_sgs_branch_transfer": sparse_sgs_branch_transfer,
        "sparse_sgs_flux": sparse_sgs_flux,
        "sparse_sgs_longitudinal_increment": sparse_sgs_longitudinal_increment,
        "physics_inferred_divergence": physics_inferred_divergence,
        "sparse_energy_tendency": sparse_energy_tendency,
        "spectral_tail_fraction": spectral_tail_fraction,
        "spectral_tail_excess": spectral_tail_excess,
    }


class _SparseVorticityNudger:
    """Lift sparse omega and velocity innovations to one consistent state.

    Omega innovations recover the smaller resolved structures.  A broader
    velocity innovation field is converted to vorticity through its curl, so
    that the corrected state remains exactly periodic and incompressible.
    """

    def __init__(
        self,
        solver: SpectralVorticitySolver,
        indices: torch.Tensor,
    ) -> None:
        self.solver = solver
        self.indices = indices
        mask = torch.zeros(
            solver.ny * solver.nx,
            device=indices.device,
            dtype=solver.kx.dtype,
        )
        mask[indices] = 1.0
        self.mask = mask.reshape(solver.ny, solver.nx)
        self.omega_transfer, self.omega_denominator = self._kernel(
            solver.config.paper.nudging_length_scale_m
        )
        self.velocity_transfer, self.velocity_denominator = self._kernel(
            solver.config.paper.velocity_nudging_length_scale_m
        )

    def _kernel(self, length_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        transfer = torch.exp(-0.5 * length_scale**2 * self.solver.k_squared)
        denominator = torch.fft.ifft2(
            torch.fft.fft2(self.mask) * transfer
        ).real
        denominator = denominator.clamp_min(
            max(float(torch.mean(denominator).item()) * 0.05, 1e-6)
        )
        return transfer, denominator

    def _lift(
        self,
        innovation: torch.Tensor,
        transfer: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        numerator = torch.zeros(
            self.solver.ny * self.solver.nx,
            device=innovation.device,
            dtype=innovation.dtype,
        ).scatter(0, self.indices, innovation)
        smooth_numerator = torch.fft.ifft2(
            torch.fft.fft2(numerator.reshape(self.solver.ny, self.solver.nx))
            * transfer
        ).real
        return smooth_numerator / denominator

    def __call__(
        self,
        omega: torch.Tensor,
        target_omega: torch.Tensor,
        target_u: torch.Tensor | None = None,
        target_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_omega = omega.reshape(-1)
        flat_target = target_omega.reshape(-1)
        innovation = flat_target[self.indices] - flat_omega[self.indices]
        omega_correction = self._lift(
            innovation, self.omega_transfer, self.omega_denominator
        )
        corrected = (
            omega
            + self.solver.config.paper.nudging_relaxation * omega_correction
        )
        if target_u is not None and target_v is not None:
            state = self.solver.state(omega)
            u_innovation = (
                target_u.reshape(-1)[self.indices]
                - state["u"].reshape(-1)[self.indices]
            )
            v_innovation = (
                target_v.reshape(-1)[self.indices]
                - state["v"].reshape(-1)[self.indices]
            )
            lifted_u = self._lift(
                u_innovation,
                self.velocity_transfer,
                self.velocity_denominator,
            )
            lifted_v = self._lift(
                v_innovation,
                self.velocity_transfer,
                self.velocity_denominator,
            )
            velocity_curl = self.solver.derivative(
                lifted_v, "x"
            ) - self.solver.derivative(lifted_u, "y")
            corrected = (
                corrected
                + self.solver.config.paper.velocity_nudging_relaxation
                * velocity_curl
            )
        return self.solver.project(corrected)


def _rollout(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    *,
    keep_states: bool,
) -> list[torch.Tensor]:
    states = [solver.project(initial_omega)]
    omega = states[0]
    for index in range(1, times.numel()):
        start = float(times[index - 1].item())
        dt = float((times[index] - times[index - 1]).item())
        omega = solver.advance(omega, start, dt)
        if not torch.isfinite(omega).all():
            raise FloatingPointError(f"non-finite rollout at t={times[index].item():g}")
        if keep_states:
            states.append(omega)
    return states if keep_states else [omega]


def _assimilated_rollout(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    reference: dict[str, torch.Tensor],
    nudger: _SparseVorticityNudger,
) -> list[torch.Tensor]:
    states = [solver.project(initial_omega)]
    omega = states[0]
    for index in range(1, times.numel()):
        start = float(times[index - 1].item())
        dt = float((times[index] - times[index - 1]).item())
        omega = solver.advance(omega, start, dt)
        omega = nudger(
            omega,
            reference["omega"][index],
            reference["u"][index],
            reference["v"][index],
        )
        if not torch.isfinite(omega).all():
            raise FloatingPointError(
                f"non-finite assimilated rollout at t={times[index].item():g}"
            )
        states.append(omega)
    return states


def _validation_score(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    indices: torch.Tensor,
    assimilation_indices: torch.Tensor,
    reference: dict[str, torch.Tensor],
    scales: dict[str, float],
    increment_pairs: torch.Tensor | None = None,
    closure_validation_schedule: torch.Tensor | None = None,
    closure_validation_weights: torch.Tensor | None = None,
    sensor_quadrature_weights: torch.Tensor | None = None,
    closure_increment_geometry: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, float]:
    solver.eval()
    losses = []
    flow_losses = []
    autonomous_flow_losses = []
    divergence_relative_l2: list[torch.Tensor] = []
    divergence_correlations: list[torch.Tensor] = []
    transfer_correlations: list[torch.Tensor] = []
    predicted_backscatter: list[torch.Tensor] = []
    reference_backscatter: list[torch.Tensor] = []
    nudger = _SparseVorticityNudger(solver, assimilation_indices)

    def flow_objective(components: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            solver.config.paper.omega_sensor_weight
            * components["omega_sensor"]
            + solver.config.paper.velocity_sensor_weight
            * components["velocity_sensor"]
            + solver.config.paper.sensor_correlation_weight
            * components["sensor_correlation"]
            + solver.config.paper.omega_increment_weight
            * components["omega_increment"]
            + solver.config.paper.sparse_energy_weight
            * components["sparse_energy"]
            + solver.config.paper.sparse_enstrophy_weight
            * components["sparse_enstrophy"]
            + solver.config.paper.sparse_energy_tendency_weight
            * components["sparse_energy_tendency"]
            + solver.config.paper.spectral_tail_weight
            * components["spectral_tail_excess"]
        )

    with torch.no_grad():
        omega = solver.project(initial_omega)
        for index in range(1, times.numel()):
            start = float(times[index - 1].item())
            dt = float((times[index] - times[index - 1]).item())
            omega = solver.advance(omega, start, dt)
            analysis_omega = nudger(
                omega,
                reference["omega"][index],
                reference["u"][index],
                reference["v"][index],
            )
            closure_indices = (
                None
                if closure_validation_schedule is None
                else closure_validation_schedule[index]
            )
            closure_weights = (
                None
                if closure_validation_weights is None
                else closure_validation_weights[index]
            )
            closure_pairs = None
            closure_directions = None
            if closure_increment_geometry is not None:
                closure_pairs, closure_directions = closure_increment_geometry[index]
            loss, components = _sensor_loss(
                solver,
                omega,
                index,
                indices,
                reference,
                scales,
                closure_indices=closure_indices,
                closure_weights=closure_weights,
                increment_pairs=increment_pairs,
                closure_increment_pairs=closure_pairs,
                closure_increment_directions=closure_directions,
                closure_omega=analysis_omega,
                sensor_quadrature_weights=sensor_quadrature_weights,
            )
            losses.append(loss)
            flow_losses.append(flow_objective(components))
            if closure_indices is not None and closure_indices.numel() > 0:
                closure_state = solver.state(analysis_omega)
                closure = solver.closure_fields(closure_state)
                divergence_prediction = solver.project(
                    solver.derivative(closure["qx"], "x")
                    + solver.derivative(closure["qy"], "y")
                ).reshape(-1)[closure_indices]
                divergence_target = reference["div_q_sgs"][index].reshape(-1)[
                    closure_indices
                ]
                transfer_prediction = closure["pi"].reshape(-1)[closure_indices]
                transfer_target = reference["pi_sgs"][index].reshape(-1)[
                    closure_indices
                ]
                metric_weights = closure_weights
                if metric_weights is None:
                    metric_weights = torch.full_like(
                        divergence_target,
                        1.0 / divergence_target.numel(),
                    )
                else:
                    metric_weights = metric_weights / metric_weights.sum().clamp_min(
                        1e-12
                    )
                divergence_relative_l2.append(
                    torch.sqrt(
                        torch.sum(
                            metric_weights
                            * (divergence_prediction - divergence_target).square()
                        )
                        / torch.sum(
                            metric_weights * divergence_target.square()
                        ).clamp_min(1e-12)
                    )
                )
                divergence_correlations.append(
                    _stable_weighted_correlation(
                        divergence_prediction,
                        divergence_target,
                        metric_weights,
                        solver.config.paper.closure_correlation_epsilon,
                    )
                )
                transfer_correlations.append(
                    _stable_weighted_correlation(
                        transfer_prediction,
                        transfer_target,
                        metric_weights,
                        solver.config.paper.closure_correlation_epsilon,
                    )
                )
                predicted_backscatter.append(transfer_prediction < 0.0)
                reference_backscatter.append(transfer_target < 0.0)
            omega = analysis_omega

        # Checkpoint selection must see the deployment mode as well as the
        # assimilated mode.  This trajectory receives no state correction and
        # is scored only at the held-out sparse flow sensors.
        omega = solver.project(initial_omega)
        for index in range(1, times.numel()):
            start = float(times[index - 1].item())
            dt = float((times[index] - times[index - 1]).item())
            omega = solver.advance(omega, start, dt)
            if not torch.isfinite(omega).all():
                autonomous_flow_losses.append(
                    torch.as_tensor(1e6, device=omega.device, dtype=omega.dtype)
                )
                break
            _, components = _sensor_loss(
                solver,
                omega,
                index,
                indices,
                reference,
                scales,
                increment_pairs=increment_pairs,
                closure_omega=omega,
                sensor_quadrature_weights=sensor_quadrature_weights,
            )
            autonomous_flow_losses.append(flow_objective(components))
    solver.train()
    assimilated_total = torch.mean(torch.stack(losses))
    assimilated_flow = torch.mean(torch.stack(flow_losses))
    autonomous_flow = torch.mean(torch.stack(autonomous_flow_losses))
    autonomous_weight = float(
        solver.config.paper.autonomous_validation_weight
    )
    if autonomous_weight < 0.0:
        raise ValueError("autonomous_validation_weight must be non-negative")
    if not divergence_relative_l2:
        raise RuntimeError("closure checkpoint metrics require held-out SGS labels")
    divergence_error_tensor = torch.stack(divergence_relative_l2)
    divergence_correlation = torch.mean(torch.stack(divergence_correlations))
    transfer_correlation = torch.mean(torch.stack(transfer_correlations))
    predicted_backscatter_fraction = torch.cat(predicted_backscatter).float().mean()
    reference_backscatter_fraction = torch.cat(reference_backscatter).float().mean()
    closure_score = (
        0.40 * torch.mean(divergence_error_tensor)
        + 0.15 * divergence_error_tensor[-1]
        + 0.15 * (1.0 - divergence_correlation)
        + 0.15 * (1.0 - transfer_correlation)
        + 0.15
        * torch.abs(
            predicted_backscatter_fraction - reference_backscatter_fraction
        )
    )
    rollout_score = assimilated_flow + autonomous_weight * autonomous_flow
    composite_score = closure_score + float(
        solver.config.paper.checkpoint_composite_rollout_weight
    ) * rollout_score
    return {
        "validation_score": float(composite_score.item()),
        "closure_score": float(closure_score.item()),
        "rollout_score": float(rollout_score.item()),
        "flow_score": float(assimilated_flow.item()),
        "autonomous_flow_score": float(autonomous_flow.item()),
        "assimilated_total": float(assimilated_total.item()),
        "divergence_relative_l2_mean": float(
            torch.mean(divergence_error_tensor).item()
        ),
        "divergence_relative_l2_endpoint": float(
            divergence_error_tensor[-1].item()
        ),
        "divergence_relative_l2_worst": float(
            torch.max(divergence_error_tensor).item()
        ),
        "divergence_correlation": float(divergence_correlation.item()),
        "transfer_correlation": float(transfer_correlation.item()),
        "backscatter_fraction_prediction": float(
            predicted_backscatter_fraction.item()
        ),
        "backscatter_fraction_reference": float(
            reference_backscatter_fraction.item()
        ),
        "backscatter_fraction_error": float(
            torch.abs(
                predicted_backscatter_fraction - reference_backscatter_fraction
            ).item()
        ),
    }


def _checkpoint(
    path: Path,
    solver: SpectralVorticitySolver,
    epoch: int,
    validation_score: float,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    scales: dict[str, float],
    history: list[dict],
    validation_metrics: dict[str, float] | None = None,
    training_state: dict | None = None,
) -> None:
    torch.save(
        {
            "version": int(solver.config.paper.checkpoint_format_version),
            "closure_parameterization": solver.config.paper.closure_parameterization,
            "closure_feature_set": solver.config.paper.closure_feature_set,
            "theory_model": "q_sgs = q_d + q_b",
            "experiment_variant": solver.config.paper.experiment_variant,
            "backscatter_factor": float(solver.closure.backscatter_factor),
            "maximum_backscatter_ratio": float(
                solver.closure.maximum_backscatter_ratio
            ),
            "maximum_local_backscatter_transfer_ratio": float(
                solver.closure.maximum_local_backscatter_transfer_ratio
            ),
            "gradient_support_coefficient_maximum": float(
                solver.closure.gradient_support_coefficient_maximum
            ),
            "dissipation_coefficient_initial_bias": float(
                solver.closure.dissipation_coefficient_initial_bias
            ),
            "dissipation_coefficient_soft_limit": float(
                solver.closure.dissipation_coefficient_soft_limit
            ),
            "maximum_anisotropy_condition": float(
                solver.closure.maximum_anisotropy_condition
            ),
            "autonomous_validation_weight": float(
                solver.config.paper.autonomous_validation_weight
            ),
            "coordinate_weighted_sparse_moments": bool(
                solver.config.paper.coordinate_weighted_sparse_moments
            ),
            "physics_inferred_divergence_weight": float(
                solver.config.paper.physics_inferred_divergence_weight
            ),
            "sparse_sgs_divergence_correlation_weight": float(
                solver.config.paper.sparse_sgs_divergence_correlation_weight
            ),
            "sparse_sgs_longitudinal_increment_weight": float(
                solver.config.paper.sparse_sgs_longitudinal_increment_weight
            ),
            "checkpoint_selection_uses_dense_test_fields": False,
            "sparse_energy_tendency_weight": float(
                solver.config.paper.sparse_energy_tendency_weight
            ),
            "sparse_sgs_mean_transfer_weight": float(
                solver.config.paper.sparse_sgs_mean_transfer_weight
            ),
            "sparse_sgs_branch_transfer_weight": float(
                solver.config.paper.sparse_sgs_branch_transfer_weight
            ),
            "closure_output_scale": float(solver.closure_output_scale),
            "model_state": solver.state_dict(),
            "epoch": epoch,
            "validation_score": validation_score,
            "validation_metrics": validation_metrics or {},
            "train_indices": train_indices.detach().cpu(),
            "validation_indices": validation_indices.detach().cpu(),
            "scales": scales,
            "history": history,
            "training_state": training_state or {},
        },
        path,
    )


def _sparse_closure_scale_proposal(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    reference: dict[str, torch.Tensor],
    assimilation_indices: torch.Tensor,
    closure_schedule: torch.Tensor,
    closure_weights: torch.Tensor,
) -> float:
    """Fit one source-amplitude scalar from existing sparse SGS labels."""
    solver.set_closure_output_scale(1.0)
    nudger = _SparseVorticityNudger(solver, assimilation_indices)
    numerator = torch.zeros((), device=initial_omega.device)
    denominator = torch.zeros_like(numerator)
    solver.eval()
    with torch.no_grad():
        states = _assimilated_rollout(
            solver, initial_omega, times, reference, nudger
        )
        for index in range(1, times.numel()):
            state = solver.state(states[index])
            closure = solver.closure_fields(state)
            predicted = solver.project(
                solver.derivative(closure["qx"], "x")
                + solver.derivative(closure["qy"], "y")
            ).reshape(-1)
            labels = closure_schedule[index]
            predicted = predicted[labels]
            target = reference["div_q_sgs"][index].reshape(-1)[labels]
            weights = closure_weights[index].to(predicted)
            numerator = numerator + torch.sum(weights * predicted * target)
            denominator = denominator + torch.sum(weights * predicted.square())
    solver.train()
    if denominator.item() <= 1e-20:
        return 1.0
    return float((numerator / denominator).item())


def _select_sparse_closure_scale(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    reference: dict[str, torch.Tensor],
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    train_closure_schedule: torch.Tensor,
    train_closure_weights: torch.Tensor,
    validation_closure_schedule: torch.Tensor | None,
    validation_closure_weights: torch.Tensor | None,
    scales: dict[str, float],
    validation_increment_pairs: torch.Tensor,
    validation_quadrature_weights: torch.Tensor,
) -> dict[str, object]:
    """Select or reject sparse amplitude calibration on held-out sensors."""
    proposal = _sparse_closure_scale_proposal(
        solver,
        initial_omega,
        times,
        reference,
        train_indices,
        train_closure_schedule,
        train_closure_weights,
    )
    lower = float(solver.config.paper.closure_scale_minimum)
    upper = float(solver.config.paper.closure_scale_maximum)
    if not 0.0 < lower <= 1.0 <= upper:
        raise ValueError(
            "closure scale bounds must satisfy 0 < minimum <= 1 <= maximum"
        )
    proposal = min(upper, max(lower, proposal))
    # The original model is always an admissible candidate.  Including a
    # midpoint makes the decision robust when the sparse LS estimate is noisy.
    candidates = sorted({1.0, proposal, 0.5 * (1.0 + proposal)})
    scores: list[dict[str, float]] = []
    print(
        "稀疏闭合校准开始：正在比较原比例、折中比例和稀疏拟合比例；"
        "此阶段会进行 3 次固定时域验证，不是程序卡住。"
    )
    for candidate in candidates:
        solver.set_closure_output_scale(candidate)
        validation_metrics = _validation_score(
            solver,
            initial_omega,
            times,
            validation_indices,
            train_indices,
            reference,
            scales,
            validation_increment_pairs,
            validation_closure_schedule,
            validation_closure_weights,
            validation_quadrature_weights,
        )
        scores.append(
            {
                "scale": float(candidate),
                "validation": validation_metrics["validation_score"],
                "flow": validation_metrics["flow_score"],
                "autonomous": validation_metrics["autonomous_flow_score"],
                "closure": validation_metrics["closure_score"],
            }
        )
        print(
            json.dumps(
                {
                    "stage": "sparse_closure_scale_validation",
                    "scale": float(candidate),
                    "validation_score": validation_metrics["validation_score"],
                    "flow_validation_score": validation_metrics["flow_score"],
                    "autonomous_validation_score": validation_metrics[
                        "autonomous_flow_score"
                    ],
                    "closure_validation_score": validation_metrics[
                        "closure_score"
                    ],
                },
                ensure_ascii=False,
            )
        )
    baseline = next(row for row in scores if row["scale"] == 1.0)
    tolerance = float(
        solver.config.paper.closure_scale_validation_flow_tolerance
    )
    admissible = [
        row
        for row in scores
        if row["flow"] <= baseline["flow"] * (1.0 + tolerance)
    ]
    selected = min(admissible, key=lambda row: row["validation"])
    solver.set_closure_output_scale(selected["scale"])
    print(
        "稀疏闭合校准完成："
        f"selected_scale={selected['scale']:.6g}; "
        f"baseline_retained={selected['scale'] == 1.0}"
    )
    return {
        "least_squares_proposal": proposal,
        "candidates": scores,
        "selected_scale": selected["scale"],
        "baseline_retained": selected["scale"] == 1.0,
        "uses_dense_reference": False,
    }


def _write_discrete_consistency_report(
    output_dir: Path,
    solver: SpectralVorticitySolver,
    reference: dict[str, torch.Tensor],
) -> None:
    """Audit whether STAR/filter data satisfy the discrete paper equation.

    This is a read-only diagnostic and never contributes to an optimiser
    loss.  A large value identifies a filter/sign/time-discretisation
    mismatch before the neural closure is blamed for it.
    """
    residual_rms: list[float] = []
    relative_rms: list[float] = []
    times: list[float] = []
    with torch.no_grad():
        for index in range(1, reference["time"].numel() - 1):
            dt = reference["time"][index + 1] - reference["time"][index - 1]
            omega_t = (
                reference["omega"][index + 1]
                - reference["omega"][index - 1]
            ) / dt
            state = solver.state(solver.project(reference["omega"][index]))
            advection = solver.project(
                state["u"] * state["omega_x"]
                + state["v"] * state["omega_y"]
            )
            residual = (
                omega_t
                + advection
                - solver.config.physics.kinematic_viscosity_m2_s
                * state["laplacian_omega"]
                + solver.project(reference["div_q_sgs"][index])
            )
            rms = torch.sqrt(torch.mean(residual.square()))
            scale = torch.sqrt(torch.mean(omega_t.square())).clamp_min(1e-12)
            residual_rms.append(float(rms.item()))
            relative_rms.append(float((rms / scale).item()))
            times.append(float(reference["time"][index].item()))
    report = {
        "purpose": "diagnostic only; excluded from all training losses",
        "equation": "omega_t + u.grad(omega) - nu.laplacian(omega) + div(q_sgs)",
        "rms_per_s2": _aggregate(residual_rms),
        "relative_to_omega_t_rms": _aggregate(relative_rms),
        "times_s": times,
        "per_time_rms_per_s2": residual_rms,
        "per_time_relative_rms": relative_rms,
    }
    (output_dir / "discrete_consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_history(output_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = list(history[0])
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = np.asarray([row["epoch"] for row in history])
    labels = {
        "total": "train total (changing horizon)",
        "omega_sensor": "omega sensor",
        "velocity_sensor": "velocity sensor",
        "omega_increment": "sparse omega increment",
        "sparse_energy": "sparse energy moment",
        "sparse_enstrophy": "sparse enstrophy moment",
        "sparse_sgs_divergence": "0.25% SGS divergence",
        "sparse_sgs_mean_transfer": "0.25% SGS mean transfer",
        "sparse_sgs_branch_transfer": "0.25% branch transfer split",
        "sparse_sgs_flux": "0.25% SGS flux",
        "physics_inferred_divergence": "4% physics-inferred SGS source",
        "sparse_energy_tendency": "sparse kinetic-energy tendency",
        "anisotropy_condition_regularization": "anisotropy condition regularisation",
        "backscatter_saturation": "backscatter saturation regularisation",
        "maximum_backscatter_ratio_realised": (
            "realised |q_b|/sqrt(|q_d|^2+|q_Clark|^2)"
        ),
        "mean_dissipative_transfer": "mean dissipative transfer",
        "mean_backscatter_transfer": "mean backscatter transfer",
        "spectral_tail_fraction": "predicted spectral-tail fraction",
        "backscatter_factor": "backscatter curriculum factor",
        "validation_score": "combined held-out validation",
        "autonomous_validation_score": "autonomous 5 s validation",
    }
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 1, figsize=(7.15, 5.4), sharex=True)
        flow_keys = (
            "total",
            "omega_sensor",
            "velocity_sensor",
            "omega_increment",
            "validation_score",
            "autonomous_validation_score",
        )
        closure_keys = (
            "sparse_sgs_divergence",
            "sparse_sgs_mean_transfer",
            "sparse_sgs_branch_transfer",
            "sparse_sgs_flux",
            "physics_inferred_divergence",
            "anisotropy_condition_regularization",
            "backscatter_saturation",
            "maximum_backscatter_ratio_realised",
            "mean_dissipative_transfer",
            "mean_backscatter_transfer",
            "sparse_energy",
            "sparse_energy_tendency",
            "sparse_enstrophy",
            "spectral_tail_fraction",
            "backscatter_factor",
        )
        for axis, selected in zip(axes, (flow_keys, closure_keys)):
            for key in selected:
                values = np.asarray([row.get(key, np.nan) for row in history], dtype=np.float64)
                valid = np.isfinite(values) & (values > 0.0)
                axis.semilogy(epochs[valid], values[valid], label=labels[key])
            axis.grid(True, which="both", alpha=0.20)
            axis.legend(ncol=2, frameon=False, loc="best")
        horizon_axis = axes[0].twinx()
        horizon_axis.plot(
            epochs,
            [row["maximum_time_s"] for row in history],
            color="0.25",
            linestyle=":",
            linewidth=1.0,
            label="training horizon",
        )
        horizon_axis.set_ylabel("training horizon (s)", color="0.25")
        horizon_axis.set_ylim(0.0, max(row["maximum_time_s"] for row in history) * 1.08)
        axes[0].set_ylabel("flow / validation loss")
        axes[1].set(xlabel="epoch", ylabel="closure / statistical loss")
        _panel_label(axes[0], "(a)")
        _panel_label(axes[1], "(b)")
        figure.suptitle("SP-NSGS training and fixed-horizon validation", y=0.995)
        figure.tight_layout()
        _save_publication_figure(figure, output_dir / "training_history")
        paper_dir = output_dir / "predictions" / "paper_figures"
        _save_publication_figure(figure, paper_dir / "figure_4_training_convergence")
        plt.close(figure)


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def train_paper(
    config: Config,
    resume_checkpoint: str | Path | None = None,
    stop_after_epoch: int | None = None,
    status_callback=None,
) -> Path:
    _seed_everything(config.training.seed)
    device = _device(config.training.device)
    dtype = _dtype(config.training.dtype)
    if device.type == "cuda" and config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = inspect_reference(config)
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    snapshots, reference = _prepare_reference(config, device, dtype)
    (
        train_indices,
        validation_indices,
        closure_indices,
        closure_weights,
        scales,
        protocol,
    ) = _sensor_protocol(config, snapshots, reference, device)
    solver = SpectralVorticitySolver(
        config,
        snapshots[0].ny,
        snapshots[0].nx,
        snapshots[0].dx,
        snapshots[0].dy,
        device,
        dtype,
    )
    _write_discrete_consistency_report(output_dir, solver, reference)
    optimizer = torch.optim.AdamW(
        solver.closure.parameters(),
        lr=config.paper.learning_rate,
        weight_decay=config.paper.closure_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.paper.epochs),
        eta_min=config.paper.minimum_learning_rate,
    )
    times = reference["time"]
    initial_omega = reference["omega"][0].detach()
    training_nudger = _SparseVorticityNudger(solver, train_indices)
    train_increment_pairs = _periodic_nearest_sensor_pairs(
        train_indices,
        snapshots[0].ny,
        snapshots[0].nx,
        config.paper.omega_increment_neighbours,
    )
    validation_increment_pairs = _periodic_nearest_sensor_pairs(
        validation_indices,
        snapshots[0].ny,
        snapshots[0].nx,
        config.paper.omega_increment_neighbours,
    )
    train_quadrature_weights = _periodic_fourier_quadrature_weights(
        train_indices, snapshots[0].ny, snapshots[0].nx
    ).to(dtype=dtype)
    validation_quadrature_weights = _periodic_fourier_quadrature_weights(
        validation_indices, snapshots[0].ny, snapshots[0].nx
    ).to(dtype=dtype)
    closure_validation_count = min(
        validation_indices.numel(),
        max(
            1,
            int(
                round(
                    snapshots[0].ny
                    * snapshots[0].nx
                    * config.paper.closure_sensor_fraction
                )
            ),
        ),
    )
    closure_validation_schedule = None
    closure_validation_weights = None
    if closure_validation_count > 0:
        validation_np = validation_indices.detach().cpu().numpy()
        # Validation is intentionally uniform.  Mirroring the active training
        # sampler here would hide its full-domain amplitude bias.
        closure_validation_np, closure_validation_weight_np = (
            _time_varying_sparse_subset_with_weights(
            validation_np,
            reference["omega"],
            closure_validation_count,
            snapshots[0].ny,
            snapshots[0].nx,
            0.0,
            float(config.paper.closure_importance_pool_fraction),
            np.random.default_rng(config.training.seed + 1901),
            )
        )
        closure_validation_schedule = torch.as_tensor(
            closure_validation_np, device=device, dtype=torch.long
        )
        closure_validation_weights = torch.as_tensor(
            closure_validation_weight_np, device=device, dtype=dtype
        )
    train_closure_increment_geometry = [
        _periodic_longitudinal_flux_pairs(
            closure_indices[index],
            snapshots[0].ny,
            snapshots[0].nx,
            snapshots[0].dx,
            snapshots[0].dy,
        )
        for index in range(times.numel())
    ]
    validation_closure_increment_geometry = (
        None
        if closure_validation_schedule is None
        else [
            _periodic_longitudinal_flux_pairs(
                closure_validation_schedule[index],
                snapshots[0].ny,
                snapshots[0].nx,
                snapshots[0].dx,
                snapshots[0].dy,
            )
            for index in range(times.numel())
        ]
    )
    protocol["closure_validation_sensor_count"] = int(
        closure_validation_count
    )
    protocol["closure_validation_unique_sensor_count"] = int(
        0
        if closure_validation_schedule is None
        else torch.unique(closure_validation_schedule).numel()
    )
    protocol["closure_validation_layout"] = (
        "time-varying uniformly sampled held-out subset; never assimilated, "
        "never optimised, and deliberately distinct from active training sampling"
    )
    protocol["closure_validation_is_training_data"] = False
    protocol["omega_increment_neighbours"] = int(
        config.paper.omega_increment_neighbours
    )
    protocol["omega_increment_train_pair_count"] = int(
        train_increment_pairs.shape[0]
    )
    protocol["omega_increment_validation_pair_count"] = int(
        validation_increment_pairs.shape[0]
    )
    protocol["omega_increment_weight"] = float(
        config.paper.omega_increment_weight
    )
    protocol["longitudinal_flux_increment_pairing"] = (
        "nearest, second-nearest, and median-distance periodic neighbours "
        "within the existing sparse SGS labels"
    )
    protocol["longitudinal_flux_increment_mean_train_pair_count"] = float(
        np.mean(
            [
                geometry[0].shape[0]
                for geometry in train_closure_increment_geometry
            ]
        )
    )
    protocol["longitudinal_flux_increment_weight"] = float(
        config.paper.sparse_sgs_longitudinal_increment_weight
    )
    protocol["coordinate_weighted_sparse_moments"] = bool(
        config.paper.coordinate_weighted_sparse_moments
    )
    protocol["train_moment_quadrature_effective_sensor_count"] = float(
        1.0 / torch.sum(train_quadrature_weights.square()).item()
    )
    protocol["physics_inferred_divergence"] = (
        "centred sparse omega_t plus resolved terms reconstructed from the "
        "same flow-sensor observer; no additional SGS labels"
    )
    protocol["physics_inferred_divergence_weight"] = float(
        config.paper.physics_inferred_divergence_weight
    )
    protocol["sparse_closure_scale_calibration"] = bool(
        config.paper.sparse_closure_scale_calibration
    )
    protocol["closure_training_state"] = (
        "sparse analysis reconstructed only from the unchanged 4% training "
        "u/v/omega sensors; flow rollout losses remain autonomous inside each "
        "window and dense STAR fields are never used for optimisation"
    )
    protocol["closure_scale_calibration_data"] = (
        "existing sparse training SGS divergence labels; candidate selection "
        "uses the disjoint sparse validation sensors; dense fields excluded"
    )
    (output_dir / "sparse_data_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history: list[dict] = []
    start_epoch = 0
    best_closure_score = float("inf")
    best_rollout_score = float("inf")
    best_composite_score = float("inf")
    best_closure_state = deepcopy(solver.state_dict())
    best_rollout_state = deepcopy(solver.state_dict())
    best_composite_state = deepcopy(solver.state_dict())
    best_closure_metrics: dict[str, float] = {}
    best_rollout_metrics: dict[str, float] = {}
    best_composite_metrics: dict[str, float] = {}
    ema_state: dict[str, torch.Tensor] | None = None
    if resume_checkpoint is not None:
        resume_payload = torch.load(
            # RNG states are device-specific byte tensors.  Loading a CPU
            # generator state through a CUDA map-location corrupts its type;
            # model/optimizer tensors are moved by their respective loaders.
            Path(resume_checkpoint), map_location="cpu", weights_only=False
        )
        resume_state = resume_payload.get("training_state")
        if not resume_state:
            raise ValueError("checkpoint lacks full training_state; exact resume unavailable")
        if int(resume_payload["epoch"]) >= config.paper.epochs:
            raise ValueError("resume checkpoint already reaches the requested epoch budget")
        solver.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        history = list(resume_payload["history"])
        start_epoch = int(resume_payload["epoch"])
        ema_state = resume_state["ema_state"]
        best_closure_score = float(resume_state["best_closure_score"])
        best_rollout_score = float(resume_state["best_rollout_score"])
        best_composite_score = float(resume_state["best_composite_score"])
        best_closure_state = resume_state["best_closure_state"]
        best_rollout_state = resume_state["best_rollout_state"]
        best_composite_state = resume_state["best_composite_state"]
        best_closure_metrics = resume_state["best_closure_metrics"]
        best_rollout_metrics = resume_state["best_rollout_metrics"]
        best_composite_metrics = resume_state["best_composite_metrics"]
        _restore_rng_state(resume_state["rng_state"])
    ema_decay = float(config.paper.parameter_ema_decay)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("parameter_ema_decay must be in [0, 1)")
    validation_tolerance = float(
        config.paper.validation_tolerance_fraction
    )
    if validation_tolerance < 0.0:
        raise ValueError("validation_tolerance_fraction must be non-negative")
    backscatter_warmup = int(config.paper.backscatter_warmup_epochs)
    backscatter_ramp = int(config.paper.backscatter_ramp_epochs)
    if backscatter_warmup < 0 or backscatter_ramp < 0:
        raise ValueError("backscatter warmup and ramp epochs must be non-negative")
    if (
        config.paper.experiment_variant == "full_sp_nsgs"
        and config.paper.epochs <= backscatter_warmup + backscatter_ramp
    ):
        raise ValueError(
            "epochs must exceed backscatter_warmup_epochs + "
            "backscatter_ramp_epochs so the full theory model is validated"
        )
    rollout_window_maximum = int(config.paper.rollout_window_snapshots)
    rollout_window_warmup = int(config.paper.rollout_window_warmup_epochs)
    rollout_window_ramp = int(config.paper.rollout_window_ramp_epochs)
    if rollout_window_maximum < 1:
        raise ValueError("rollout_window_snapshots must be positive")
    if rollout_window_warmup < 0 or rollout_window_ramp < 0:
        raise ValueError(
            "rollout window warmup and ramp epochs must be non-negative"
        )
    start_clock = time.perf_counter()

    for epoch in range(start_epoch + 1, config.paper.epochs + 1):
        solver.train()
        if epoch <= backscatter_warmup:
            backscatter_factor = 0.0
        elif backscatter_ramp == 0:
            backscatter_factor = 1.0
        else:
            backscatter_factor = min(
                1.0, (epoch - backscatter_warmup) / backscatter_ramp
            )
        solver.closure.set_backscatter_factor(backscatter_factor)
        if epoch <= rollout_window_warmup:
            rollout_window = 1
        elif rollout_window_ramp == 0:
            rollout_window = rollout_window_maximum
        else:
            window_progress = min(
                1.0,
                (epoch - rollout_window_warmup) / rollout_window_ramp,
            )
            rollout_window = 1 + int(
                math.floor(
                    (rollout_window_maximum - 1) * window_progress + 1e-12
                )
            )
        if config.paper.causal_curriculum_epochs > 0:
            progress = min(1.0, epoch / config.paper.causal_curriculum_epochs)
            maximum_time = config.paper.curriculum_start_time_s + progress * (
                float(times[-1].item()) - config.paper.curriculum_start_time_s
            )
        else:
            maximum_time = float(times[-1].item())
        maximum_index = int(
            torch.searchsorted(
                times, torch.as_tensor(maximum_time, device=device, dtype=dtype)
            ).item()
        )
        maximum_index = max(1, min(maximum_index, times.numel() - 1))
        omega = solver.project(initial_omega)
        window_losses: list[torch.Tensor] = []
        component_sums = {
            "omega_sensor": 0.0,
            "velocity_sensor": 0.0,
            "omega_increment": 0.0,
            "flux_regularization": 0.0,
            "anisotropy_condition_regularization": 0.0,
            "backscatter_saturation": 0.0,
            "mean_anisotropy_condition": 0.0,
            "maximum_backscatter_ratio_realised": 0.0,
            "maximum_local_backscatter_transfer_ratio_realised": 0.0,
            "mean_backscatter_transfer_limiter": 0.0,
            "mean_dissipative_transfer": 0.0,
            "mean_backscatter_transfer": 0.0,
            "mean_dissipation_penalty": 0.0,
            "sensor_correlation": 0.0,
            "sparse_energy": 0.0,
            "sparse_enstrophy": 0.0,
            "dissipation_coefficient_saturation": 0.0,
            "sparse_sgs_divergence": 0.0,
            "sparse_sgs_divergence_correlation": 0.0,
            "sparse_sgs_transfer": 0.0,
            "sparse_sgs_mean_transfer": 0.0,
            "sparse_sgs_branch_transfer": 0.0,
            "sparse_sgs_flux": 0.0,
            "sparse_sgs_longitudinal_increment": 0.0,
            "physics_inferred_divergence": 0.0,
            "sparse_energy_tendency": 0.0,
            "spectral_tail_fraction": 0.0,
            "spectral_tail_excess": 0.0,
        }
        update_count = 0
        for index in range(1, maximum_index + 1):
            start_time = float(times[index - 1].item())
            dt = float((times[index] - times[index - 1]).item())
            omega = solver.advance(omega, start_time, dt)
            if not torch.isfinite(omega).all():
                raise FloatingPointError(
                    "non-finite training rollout at "
                    f"epoch={epoch}, t={float(times[index].item()):.6g} s; "
                    "training stopped before corrupting the checkpoint"
                )
            # Direct SGS labels describe the closure at the STAR state, not at
            # an increasingly phase-shifted autonomous forecast.  Construct a
            # sparse analysis using only the unchanged 4% training sensors and
            # use it for the closure-label terms.  The flow loss is still
            # evaluated on the uncorrected forecast, so multi-step dynamics are
            # not replaced by supervised field fitting.
            analysis_omega = training_nudger(
                omega,
                reference["omega"][index],
                reference["u"][index],
                reference["v"][index],
            )
            if not torch.isfinite(analysis_omega).all():
                raise FloatingPointError(
                    "non-finite sparse analysis at "
                    f"epoch={epoch}, t={float(times[index].item()):.6g} s"
                )
            loss, components = _sensor_loss(
                solver,
                omega,
                index,
                train_indices,
                reference,
                scales,
                closure_indices=closure_indices[index],
                closure_weights=closure_weights[index],
                increment_pairs=train_increment_pairs,
                closure_increment_pairs=train_closure_increment_geometry[index][0],
                closure_increment_directions=train_closure_increment_geometry[index][1],
                closure_omega=analysis_omega,
                sensor_quadrature_weights=train_quadrature_weights,
            )
            window_losses.append(loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite sparse training loss at "
                    f"epoch={epoch}, t={float(times[index].item()):.6g} s"
                )
            for key, value in components.items():
                component_sums[key] += float(value.detach().item())
            at_window_end = (
                len(window_losses) >= rollout_window
                or index == maximum_index
            )
            if at_window_end:
                # Assimilate only at the end of an autonomous training
                # window.  All losses inside the window therefore constrain
                # true multi-step forecast dynamics, while the same 4% sparse
                # sensors provide stable anchors between windows.
                optimizer.zero_grad(set_to_none=True)
                window_loss = torch.mean(torch.stack(window_losses))
                if not torch.isfinite(window_loss):
                    raise FloatingPointError(
                        f"non-finite window loss at epoch={epoch}, index={index}"
                    )
                window_loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    solver.closure.parameters(), config.paper.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"non-finite closure gradient at epoch={epoch}, index={index}"
                    )
                optimizer.step()
                # The previous implementation computed this analysis state but
                # accidentally discarded it.  That turned every epoch into a
                # fully autonomous 5 s rollout and was the principal cause of
                # the rising loss and late numerical explosion.  Anchor only
                # after the complete autonomous window, using the unchanged 4%
                # sparse observations, and detach before the next window.
                omega = analysis_omega.detach()
                window_losses.clear()
                update_count += 1
        scheduler.step()
        if (
            ema_decay > 0.0
            and epoch >= config.paper.parameter_ema_start_epoch
        ):
            current_state = solver.state_dict()
            if ema_state is None:
                ema_state = deepcopy(current_state)
            else:
                with torch.no_grad():
                    for key, current_value in current_state.items():
                        ema_state[key].mul_(ema_decay).add_(
                            current_value, alpha=1.0 - ema_decay
                        )
        denominator = max(1, maximum_index)
        train_total = (
            config.paper.omega_sensor_weight
            * component_sums["omega_sensor"]
            + config.paper.velocity_sensor_weight
            * component_sums["velocity_sensor"]
            + config.paper.sensor_correlation_weight
            * component_sums["sensor_correlation"]
            + config.paper.omega_increment_weight
            * component_sums["omega_increment"]
            + config.paper.sparse_energy_weight
            * component_sums["sparse_energy"]
            + config.paper.sparse_enstrophy_weight
            * component_sums["sparse_enstrophy"]
            + config.paper.flux_regularization_weight
            * component_sums["flux_regularization"]
            + config.paper.anisotropy_condition_weight
            * component_sums["anisotropy_condition_regularization"]
            + config.paper.backscatter_saturation_weight
            * component_sums["backscatter_saturation"]
            + config.paper.mean_dissipation_weight
            * component_sums["mean_dissipation_penalty"]
            + config.paper.dissipation_coefficient_saturation_weight
            * component_sums["dissipation_coefficient_saturation"]
            + config.paper.sparse_sgs_divergence_weight
            * component_sums["sparse_sgs_divergence"]
            + config.paper.sparse_sgs_divergence_correlation_weight
            * component_sums["sparse_sgs_divergence_correlation"]
            + config.paper.sparse_sgs_transfer_weight
            * component_sums["sparse_sgs_transfer"]
            + config.paper.sparse_sgs_mean_transfer_weight
            * component_sums["sparse_sgs_mean_transfer"]
            + config.paper.sparse_sgs_branch_transfer_weight
            * component_sums["sparse_sgs_branch_transfer"]
            + config.paper.sparse_sgs_flux_weight
            * component_sums["sparse_sgs_flux"]
            + config.paper.sparse_sgs_longitudinal_increment_weight
            * component_sums["sparse_sgs_longitudinal_increment"]
            + config.paper.physics_inferred_divergence_weight
            * component_sums["physics_inferred_divergence"]
            + config.paper.sparse_energy_tendency_weight
            * component_sums["sparse_energy_tendency"]
            + config.paper.spectral_tail_weight
            * component_sums["spectral_tail_excess"]
        ) / denominator
        # ``None`` means "not evaluated this epoch" and serialises as JSON
        # ``null``.  Using NaN here previously made a skipped validation look
        # indistinguishable from a numerical failure in the PyCharm console.
        validation: float | None = None
        flow_validation: float | None = None
        autonomous_validation: float | None = None
        closure_validation: float | None = None
        rollout_validation: float | None = None
        validation_metrics: dict[str, float] | None = None
        validation_uses_ema = False
        validation_computed = (
            epoch == 1
            or epoch % config.paper.validation_every_epochs == 0
            or epoch == config.paper.epochs
        )
        if validation_computed:
            online_state = None
            if ema_state is not None:
                online_state = deepcopy(solver.state_dict())
                solver.load_state_dict(ema_state)
                validation_uses_ema = True
            validation_metrics = _validation_score(
                solver,
                initial_omega,
                times,
                validation_indices,
                train_indices,
                reference,
                scales,
                validation_increment_pairs,
                closure_validation_schedule,
                closure_validation_weights,
                validation_quadrature_weights,
                validation_closure_increment_geometry,
            )
            validation = validation_metrics["validation_score"]
            closure_validation = validation_metrics["closure_score"]
            rollout_validation = validation_metrics["rollout_score"]
            flow_validation = validation_metrics["flow_score"]
            autonomous_validation = validation_metrics["autonomous_flow_score"]
            # A dissipative-only warm-up checkpoint is not a valid checkpoint
            # for the complete manuscript model.  Selection starts only after
            # the bounded backscatter branch has reached its full amplitude.
            complete_variant_ready = (
                config.paper.experiment_variant == "iso_dissipative"
                or backscatter_factor >= 1.0 - 1e-12
            )
            if complete_variant_ready:
                if closure_validation < best_closure_score:
                    best_closure_score = closure_validation
                    best_closure_state = deepcopy(solver.state_dict())
                    best_closure_metrics = dict(validation_metrics)
                    _checkpoint(
                        output_dir / "best_closure.pt",
                        solver,
                        epoch,
                        closure_validation,
                        train_indices,
                        validation_indices,
                        scales,
                        history,
                        validation_metrics,
                    )
                if rollout_validation < best_rollout_score:
                    best_rollout_score = rollout_validation
                    best_rollout_state = deepcopy(solver.state_dict())
                    best_rollout_metrics = dict(validation_metrics)
                    _checkpoint(
                        output_dir / "best_rollout.pt",
                        solver,
                        epoch,
                        rollout_validation,
                        train_indices,
                        validation_indices,
                        scales,
                        history,
                        validation_metrics,
                    )
                if validation < best_composite_score:
                    best_composite_score = validation
                    best_composite_state = deepcopy(solver.state_dict())
                    best_composite_metrics = dict(validation_metrics)
                    _checkpoint(
                        output_dir / "best_composite.pt",
                        solver,
                        epoch,
                        validation,
                        train_indices,
                        validation_indices,
                        scales,
                        history,
                        validation_metrics,
                    )
                    # Backward-compatible alias; its selection semantics are
                    # now explicitly the held-out composite score.
                    _checkpoint(
                        output_dir / "checkpoint_best.pt",
                        solver,
                        epoch,
                        validation,
                        train_indices,
                        validation_indices,
                        scales,
                        history,
                        validation_metrics,
                    )
            if online_state is not None:
                solver.load_state_dict(online_state)
        row = {
            "epoch": epoch,
            "maximum_time_s": maximum_time,
            "updates": update_count,
            "total": train_total,
            "omega_sensor": component_sums["omega_sensor"] / denominator,
            "velocity_sensor": component_sums["velocity_sensor"] / denominator,
            "sensor_correlation": component_sums["sensor_correlation"] / denominator,
            "omega_increment": component_sums["omega_increment"] / denominator,
            "sparse_energy": component_sums["sparse_energy"] / denominator,
            "sparse_enstrophy": component_sums["sparse_enstrophy"] / denominator,
            "flux_regularization": component_sums["flux_regularization"] / denominator,
            "anisotropy_condition_regularization": component_sums[
                "anisotropy_condition_regularization"
            ]
            / denominator,
            "backscatter_saturation": component_sums[
                "backscatter_saturation"
            ]
            / denominator,
            "mean_anisotropy_condition": component_sums[
                "mean_anisotropy_condition"
            ]
            / denominator,
            "maximum_backscatter_ratio_realised": component_sums[
                "maximum_backscatter_ratio_realised"
            ]
            / denominator,
            "maximum_local_backscatter_transfer_ratio_realised": component_sums[
                "maximum_local_backscatter_transfer_ratio_realised"
            ]
            / denominator,
            "mean_backscatter_transfer_limiter": component_sums[
                "mean_backscatter_transfer_limiter"
            ]
            / denominator,
            "mean_dissipative_transfer": component_sums[
                "mean_dissipative_transfer"
            ]
            / denominator,
            "mean_backscatter_transfer": component_sums[
                "mean_backscatter_transfer"
            ]
            / denominator,
            "mean_dissipation_penalty": component_sums[
                "mean_dissipation_penalty"
            ]
            / denominator,
            "dissipation_coefficient_saturation": component_sums[
                "dissipation_coefficient_saturation"
            ]
            / denominator,
            "sparse_sgs_divergence": component_sums[
                "sparse_sgs_divergence"
            ]
            / denominator,
            "sparse_sgs_divergence_correlation": component_sums[
                "sparse_sgs_divergence_correlation"
            ]
            / denominator,
            "sparse_sgs_transfer": component_sums["sparse_sgs_transfer"]
            / denominator,
            "sparse_sgs_mean_transfer": component_sums[
                "sparse_sgs_mean_transfer"
            ]
            / denominator,
            "sparse_sgs_branch_transfer": component_sums[
                "sparse_sgs_branch_transfer"
            ]
            / denominator,
            "sparse_sgs_flux": component_sums["sparse_sgs_flux"]
            / denominator,
            "sparse_sgs_longitudinal_increment": component_sums[
                "sparse_sgs_longitudinal_increment"
            ]
            / denominator,
            "physics_inferred_divergence": component_sums[
                "physics_inferred_divergence"
            ]
            / denominator,
            "sparse_energy_tendency": component_sums[
                "sparse_energy_tendency"
            ]
            / denominator,
            "spectral_tail_fraction": component_sums[
                "spectral_tail_fraction"
            ]
            / denominator,
            "spectral_tail_excess": component_sums["spectral_tail_excess"]
            / denominator,
            "validation_score": validation,
            "closure_validation_score": closure_validation,
            "rollout_validation_score": rollout_validation,
            "flow_validation_score": flow_validation,
            "autonomous_validation_score": autonomous_validation,
            "validation_divergence_relative_l2_mean": (
                None
                if validation_metrics is None
                else validation_metrics["divergence_relative_l2_mean"]
            ),
            "validation_divergence_relative_l2_endpoint": (
                None
                if validation_metrics is None
                else validation_metrics["divergence_relative_l2_endpoint"]
            ),
            "validation_divergence_relative_l2_worst": (
                None
                if validation_metrics is None
                else validation_metrics["divergence_relative_l2_worst"]
            ),
            "validation_divergence_correlation": (
                None
                if validation_metrics is None
                else validation_metrics["divergence_correlation"]
            ),
            "validation_transfer_correlation": (
                None
                if validation_metrics is None
                else validation_metrics["transfer_correlation"]
            ),
            "validation_backscatter_fraction_error": (
                None
                if validation_metrics is None
                else validation_metrics["backscatter_fraction_error"]
            ),
            "validation_computed": validation_computed,
            "backscatter_factor": float(solver.closure.backscatter_factor),
            "rollout_window_snapshots": int(rollout_window),
            "validation_uses_parameter_ema": validation_uses_ema,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start_clock,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if status_callback is not None:
            status_callback(row)
        if epoch % config.paper.checkpoint_every_epochs == 0:
            resume_state = {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": ema_state,
                "best_closure_score": best_closure_score,
                "best_rollout_score": best_rollout_score,
                "best_composite_score": best_composite_score,
                "best_closure_state": best_closure_state,
                "best_rollout_state": best_rollout_state,
                "best_composite_state": best_composite_state,
                "best_closure_metrics": best_closure_metrics,
                "best_rollout_metrics": best_rollout_metrics,
                "best_composite_metrics": best_composite_metrics,
                "rng_state": _capture_rng_state(),
                "curriculum": {
                    "backscatter_factor": float(solver.closure.backscatter_factor),
                    "rollout_window_snapshots": int(rollout_window),
                },
            }
            _checkpoint(
                output_dir / "last_checkpoint.pt",
                solver,
                epoch,
                validation,
                train_indices,
                validation_indices,
                scales,
                history,
                validation_metrics,
                resume_state,
            )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return output_dir / "last_checkpoint.pt"
        _write_history(output_dir, history)

    solver.load_state_dict(best_composite_state)
    solver.closure.set_backscatter_factor(1.0)
    final_validation_metrics = dict(best_composite_metrics)
    calibration_report: dict[str, object] = {
        "enabled": False,
        "selected_scale": 1.0,
        "baseline_retained": True,
        "uses_dense_reference": False,
    }
    if config.paper.sparse_closure_scale_calibration:
        calibration_report = {
            "enabled": True,
            **_select_sparse_closure_scale(
                solver,
                initial_omega,
                times,
                reference,
                train_indices,
                validation_indices,
                closure_indices,
                closure_weights,
                closure_validation_schedule,
                closure_validation_weights,
                scales,
                validation_increment_pairs,
                validation_quadrature_weights,
            ),
        }
        selected_row = next(
            row
            for row in calibration_report["candidates"]
            if row["scale"] == calibration_report["selected_scale"]
        )
        best_composite_score = float(selected_row["validation"])
        final_validation_metrics = {
            **final_validation_metrics,
            "validation_score": best_composite_score,
            "flow_score": float(selected_row["flow"]),
            "autonomous_flow_score": float(selected_row["autonomous"]),
            "closure_score": float(selected_row["closure"]),
        }
    (output_dir / "closure_scale_calibration.json").write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_path = output_dir / "checkpoint_final.pt"
    _checkpoint(
        final_path,
        solver,
        config.paper.epochs,
        best_composite_score,
        train_indices,
        validation_indices,
        scales,
        history,
        final_validation_metrics,
    )
    runtime = {
        "version": int(solver.config.paper.checkpoint_format_version),
        "closure_parameterization": solver.config.paper.closure_parameterization,
        "theory_model": "q_sgs = q_d + q_b",
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "elapsed_seconds": time.perf_counter() - start_clock,
        "epochs": config.paper.epochs,
        "best_validation_score": best_composite_score,
        "best_closure_score": best_closure_score,
        "best_rollout_score": best_rollout_score,
        "best_composite_score": best_composite_score,
        "best_closure_metrics": best_closure_metrics,
        "best_rollout_metrics": best_rollout_metrics,
        "best_composite_metrics": best_composite_metrics,
        "validation_tolerance_fraction": validation_tolerance,
        "dissipation_coefficient_initial_bias": float(
            solver.closure.dissipation_coefficient_initial_bias
        ),
        "dissipation_coefficient_soft_limit": float(
            solver.closure.dissipation_coefficient_soft_limit
        ),
        "maximum_backscatter_ratio": float(
            solver.closure.maximum_backscatter_ratio
        ),
        "maximum_local_backscatter_transfer_ratio": float(
            solver.closure.maximum_local_backscatter_transfer_ratio
        ),
        "learned_gradient_support_coefficient": float(
            (
                solver.closure.gradient_support_coefficient_maximum
                * torch.sigmoid(solver.closure.gradient_support_coefficient_raw)
            ).detach().cpu().item()
        ),
        "maximum_anisotropy_condition": float(
            solver.closure.maximum_anisotropy_condition
        ),
        "autonomous_validation_weight": float(
            config.paper.autonomous_validation_weight
        ),
        "backscatter_warmup_epochs": backscatter_warmup,
        "backscatter_ramp_epochs": backscatter_ramp,
        "coordinate_weighted_sparse_moments": bool(
            config.paper.coordinate_weighted_sparse_moments
        ),
        "sparse_energy_tendency_weight": float(
            config.paper.sparse_energy_tendency_weight
        ),
        "sparse_sgs_mean_transfer_weight": float(
            config.paper.sparse_sgs_mean_transfer_weight
        ),
        "sparse_sgs_branch_transfer_weight": float(
            config.paper.sparse_sgs_branch_transfer_weight
        ),
        "sparse_sgs_divergence_correlation_weight": float(
            config.paper.sparse_sgs_divergence_correlation_weight
        ),
        "sparse_sgs_longitudinal_increment_weight": float(
            config.paper.sparse_sgs_longitudinal_increment_weight
        ),
        "physics_inferred_divergence_weight": float(
            config.paper.physics_inferred_divergence_weight
        ),
        "closure_output_scale": float(solver.closure_output_scale),
        "closure_scale_calibration": calibration_report,
    }
    (output_dir / "runtime_report.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_path


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def _correlation(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = prediction.ravel()
    reference = reference.ravel()
    if np.std(prediction) < 1e-30 or np.std(reference) < 1e-30:
        return float("nan")
    return float(np.corrcoef(prediction, reference)[0, 1])


def _radial_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = u.shape
    energy = 0.5 * (
        np.abs(np.fft.fft2(u) / (nx * ny)) ** 2
        + np.abs(np.fft.fft2(v) / (nx * ny)) ** 2
    )
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    shell = np.rint(np.sqrt(kkx**2 + kky**2)).astype(np.int64)
    spectrum = np.bincount(shell.ravel(), weights=energy.ravel())
    return np.arange(spectrum.size), spectrum


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "maximum": float("nan"), "final": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "final": float(array[-1]),
    }


def _paper_style() -> dict:
    """Compact journal-style Matplotlib settings with embedded vector fonts."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "lines.linewidth": 1.35,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 350,
    }


def _save_publication_figure(figure, base_path: Path) -> None:
    """Write a review-friendly raster image and a publication vector PDF."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _write_prediction_plots(
    output_dir: Path,
    records: list[dict],
    metrics: list[dict],
    autonomous_metrics: list[dict],
    closure_records: list[dict],
    final_spectrum: tuple[np.ndarray, np.ndarray, np.ndarray],
    spectral_tail_start: int,
    resolved_cutoff: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    closure_indices: np.ndarray,
    grid_shape: tuple[int, int],
    summary: dict,
    signed_error_color_limit: float,
    absolute_error_color_limit_per_s: float,
    show_absolute_closure_error_panels: bool,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return
    paper_dir = output_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([row["time_s"] for row in metrics])

    with mpl.rc_context(_paper_style()):
        # Figure 1: the primary claim - sparse-assimilated reconstruction.
        columns = len(records)
        figure, axes = plt.subplots(3, columns, figsize=(7.15, 4.6), squeeze=False)
        limit = max(float(np.max(np.abs(row["reference"]))) for row in records)
        normalized_errors = []
        for row in records:
            omega_rms = max(float(np.sqrt(np.mean(row["reference"] ** 2))), 1e-12)
            normalized_errors.append(
                (row["prediction"] - row["reference"]) / omega_rms
            )
        error_limit = float(signed_error_color_limit)
        if error_limit <= 0.0:
            raise ValueError("signed omega-error colour limit must be positive")
        field_image = error_image = None
        for column, record in enumerate(records):
            field_image = axes[0, column].imshow(record["reference"], origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[1, column].imshow(record["prediction"], origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            error_image = axes[2, column].imshow(
                normalized_errors[column],
                origin="lower",
                cmap="RdBu_r",
                vmin=-error_limit,
                vmax=error_limit,
            )
            relative_error = _relative_l2(
                record["prediction"], record["reference"]
            )
            axes[2, column].text(
                0.03,
                0.04,
                rf"$\epsilon_\omega={100.0 * relative_error:.1f}\%$",
                transform=axes[2, column].transAxes,
                color="0.1",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            )
            axes[0, column].set_title(rf"$t={record['time_s']:g}\,\mathrm{{s}}$")
            for axis in axes[:, column]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes[0, 0].set_ylabel("STAR reference")
        axes[1, 0].set_ylabel("SP-NSGS\n(4% assimilated)")
        axes[2, 0].set_ylabel("normalised\nsigned error")
        _panel_label(axes[0, 0], "(a)")
        _panel_label(axes[1, 0], "(b)")
        _panel_label(axes[2, 0], "(c)")
        figure.subplots_adjust(left=0.08, right=0.91, bottom=0.04, top=0.95, wspace=0.06, hspace=0.10)
        if field_image is not None:
            cax = figure.add_axes([0.925, 0.37, 0.012, 0.53])
            figure.colorbar(field_image, cax=cax, label=r"vorticity $\omega$ ($\mathrm{s}^{-1}$)")
        if error_image is not None:
            cax = figure.add_axes([0.925, 0.07, 0.012, 0.22])
            figure.colorbar(
                error_image,
                cax=cax,
                label=r"$\Delta\omega/\omega_{\mathrm{rms}}^{\mathrm{ref}}$",
            )
        _save_publication_figure(figure, paper_dir / "figure_1_sparse_reconstruction")
        figure.savefig(output_dir / "paper_comparison.png", dpi=350, bbox_inches="tight")
        plt.close(figure)

        # Supplementary absolute error: retained for full numerical transparency.
        figure, axes = plt.subplots(1, columns, figsize=(7.15, 1.65), squeeze=False)
        absolute_limit = float(absolute_error_color_limit_per_s)
        if absolute_limit <= 0.0:
            raise ValueError("absolute omega-error colour limit must be positive")
        absolute_image = None
        for column, record in enumerate(records):
            absolute_image = axes[0, column].imshow(
                np.abs(record["prediction"] - record["reference"]),
                origin="lower",
                cmap="cividis",
                vmin=0.0,
                vmax=absolute_limit,
            )
            axes[0, column].set_title(rf"$t={record['time_s']:g}\,\mathrm{{s}}$")
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
        axes[0, 0].set_ylabel("absolute error")
        figure.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.87, wspace=0.06)
        if absolute_image is not None:
            cax = figure.add_axes([0.925, 0.14, 0.012, 0.64])
            figure.colorbar(
                absolute_image,
                cax=cax,
                label=r"$|\Delta\omega|$ ($\mathrm{s}^{-1}$)",
            )
        _save_publication_figure(
            figure, paper_dir / "figure_s2_absolute_vorticity_error"
        )
        plt.close(figure)

        # Figure 2: global flow statistics and the resolved spectral range.
        figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.4))
        for key, label in (("u_relative_l2", r"$u$"), ("v_relative_l2", r"$v$"), ("omega_relative_l2", r"$\omega$")):
            axes[0, 0].plot(times, [row[key] for row in metrics], label=label)
        axes[0, 0].set(xlabel=r"$t$ (s)", ylabel="relative $L^2$ error")
        axes[0, 0].legend(frameon=False, ncol=3)
        axes[0, 1].plot(times, [row["kinetic_energy_ref"] for row in metrics], label="STAR reference")
        axes[0, 1].plot(times, [row["kinetic_energy_pred"] for row in metrics], "--", label="SP-NSGS")
        axes[0, 1].set(xlabel=r"$t$ (s)", ylabel="kinetic energy")
        axes[0, 1].legend(frameon=False)
        axes[1, 0].plot(times, [row["enstrophy_ref"] for row in metrics], label="STAR reference")
        axes[1, 0].plot(times, [row["enstrophy_pred"] for row in metrics], "--", label="SP-NSGS")
        axes[1, 0].set(xlabel=r"$t$ (s)", ylabel="enstrophy")
        axes[1, 0].legend(frameon=False)
        wave, reference_spectrum, prediction_spectrum = final_spectrum
        valid_ref = (wave > 0) & (wave <= resolved_cutoff) & (reference_spectrum > 0)
        valid_pred = (wave > 0) & (wave <= resolved_cutoff) & (prediction_spectrum > 0)
        axes[1, 1].loglog(wave[valid_ref], reference_spectrum[valid_ref], label="STAR reference")
        axes[1, 1].loglog(wave[valid_pred], prediction_spectrum[valid_pred], "--", label="SP-NSGS")
        axes[1, 1].axvspan(spectral_tail_start, resolved_cutoff, color="0.75", alpha=0.20, label="tail band")
        axes[1, 1].axvline(resolved_cutoff, color="0.3", linestyle=":", linewidth=0.9)
        axes[1, 1].set(xlabel="wave number $k$", ylabel="$E(k)$")
        axes[1, 1].legend(frameon=False)
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
            _panel_label(axis, label)
            axis.grid(True, which="both", alpha=0.18)
        figure.tight_layout()
        _save_publication_figure(figure, paper_dir / "figure_2_flow_statistics")
        figure.savefig(output_dir / "comparison_metrics.png", dpi=350, bbox_inches="tight")
        plt.close(figure)

        # Figure 3: a priori closure validation required by the manuscript.
        if closure_records:
            closure = closure_records[-1]
            figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.6))
            closure_rows = (
                ("div_q", r"$\nabla\!\cdot\mathbf{q}_{sgs}$"),
                ("pi", r"$\Pi_\omega$"),
            )
            normalised_absolute_errors: dict[str, np.ndarray] = {}
            for prefix, _ in closure_rows:
                reference_field = closure[f"{prefix}_reference"]
                reference_rms = max(
                    float(np.sqrt(np.mean(reference_field**2))), 1e-12
                )
                normalised_absolute_errors[prefix] = (
                    np.abs(
                        closure[f"{prefix}_prediction"] - reference_field
                    )
                    / reference_rms
                )
            closure_error_limit = max(
                float(
                    np.percentile(
                        np.concatenate(
                            [value.ravel() for value in normalised_absolute_errors.values()]
                        ),
                        99.5,
                    )
                ),
                1e-12,
            )
            error_image = None
            for row_index, (prefix, symbol) in enumerate(closure_rows):
                reference_field = closure[f"{prefix}_reference"]
                prediction_field = closure[f"{prefix}_prediction"]
                field_limit = max(float(np.max(np.abs(reference_field))), float(np.max(np.abs(prediction_field))))
                image = axes[row_index, 0].imshow(reference_field, origin="lower", cmap="coolwarm", vmin=-field_limit, vmax=field_limit)
                axes[row_index, 1].imshow(prediction_field, origin="lower", cmap="coolwarm", vmin=-field_limit, vmax=field_limit)
                error_image = axes[row_index, 2].imshow(
                    normalised_absolute_errors[prefix],
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=closure_error_limit,
                )
                relative_error = _relative_l2(
                    prediction_field, reference_field
                )
                axes[row_index, 2].text(
                    0.03,
                    0.04,
                    rf"$\epsilon_{{L^2}}={100.0 * relative_error:.1f}\%$",
                    transform=axes[row_index, 2].transAxes,
                    color="white",
                    bbox={
                        "facecolor": "black",
                        "edgecolor": "none",
                        "alpha": 0.48,
                        "pad": 1.2,
                    },
                )
                axes[row_index, 0].set_ylabel(symbol)
                figure.colorbar(image, ax=axes[row_index, :2], fraction=0.025, pad=0.02)
            if error_image is not None:
                error_cax = figure.add_axes([0.925, 0.16, 0.012, 0.66])
                figure.colorbar(
                    error_image,
                    cax=error_cax,
                    label=r"$|\Delta f|/f_{\rm ref,rms}$",
                )
            for column, title in enumerate((
                "STAR reference",
                "SP-NSGS closure",
                "normalised absolute error\n(99.5th percentile clipped)",
            )):
                axes[0, column].set_title(title)
            for axis in axes.flat:
                axis.set_xticks([])
                axis.set_yticks([])
            _panel_label(axes[0, 0], "(a)")
            _panel_label(axes[1, 0], "(b)")
            figure.suptitle(rf"A priori SGS validation at $t={closure['time_s']:g}\,\mathrm{{s}}$", y=0.995)
            figure.subplots_adjust(left=0.07, right=0.90, bottom=0.06, top=0.88, wspace=0.22, hspace=0.10)
            error_figure_name = (
                "figure_3_apriori_sgs_fields"
                if show_absolute_closure_error_panels
                else "figure_s3_apriori_sgs_absolute_errors"
            )
            _save_publication_figure(figure, paper_dir / error_figure_name)
            plt.close(figure)
            if not show_absolute_closure_error_panels:
                # Main-text field comparison.  The complete, quantitatively
                # annotated error map remains available as Figure S3.
                figure, axes = plt.subplots(2, 2, figsize=(5.15, 4.6))
                for row_index, (prefix, symbol) in enumerate(closure_rows):
                    reference_field = closure[f"{prefix}_reference"]
                    prediction_field = closure[f"{prefix}_prediction"]
                    field_limit = max(
                        float(np.max(np.abs(reference_field))),
                        float(np.max(np.abs(prediction_field))),
                    )
                    image = axes[row_index, 0].imshow(
                        reference_field,
                        origin="lower",
                        cmap="coolwarm",
                        vmin=-field_limit,
                        vmax=field_limit,
                    )
                    axes[row_index, 1].imshow(
                        prediction_field,
                        origin="lower",
                        cmap="coolwarm",
                        vmin=-field_limit,
                        vmax=field_limit,
                    )
                    axes[row_index, 0].set_ylabel(symbol)
                    figure.colorbar(
                        image,
                        ax=axes[row_index, :],
                        fraction=0.035,
                        pad=0.025,
                    )
                axes[0, 0].set_title("STAR reference")
                axes[0, 1].set_title("SP-NSGS closure")
                for axis in axes.flat:
                    axis.set_xticks([])
                    axis.set_yticks([])
                _panel_label(axes[0, 0], "(a)")
                _panel_label(axes[1, 0], "(b)")
                figure.suptitle(
                    rf"A priori SGS validation at $t={closure['time_s']:g}\,\mathrm{{s}}$",
                    y=0.995,
                )
                figure.subplots_adjust(
                    left=0.09,
                    right=0.93,
                    bottom=0.06,
                    top=0.88,
                    wspace=0.18,
                    hspace=0.10,
                )
                _save_publication_figure(
                    figure, paper_dir / "figure_3_apriori_sgs_fields"
                )
                plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
        closure_error_curves = (
            ("div_q_sgs_relative_l2", r"$\nabla\cdot\mathbf{q}_{sgs}$", "-", 1.0),
            ("pi_sgs_relative_l2", r"$\Pi_\omega$", "-", 1.0),
            ("q_sgs_relative_l2", r"$\mathbf{q}_{sgs}$", "--", 0.75),
        )
        closure_correlation_curves = (
            ("div_q_sgs_correlation", r"$\nabla\cdot\mathbf{q}_{sgs}$", "-", 1.0),
            ("pi_sgs_correlation", r"$\Pi_\omega$", "-", 1.0),
            ("q_sgs_correlation", r"$\mathbf{q}_{sgs}$", "--", 0.75),
        )
        for key, label, linestyle, alpha in closure_error_curves:
            axes[0].plot(
                times, [row[key] for row in metrics], label=label,
                linestyle=linestyle, alpha=alpha,
            )
        for key, label, linestyle, alpha in closure_correlation_curves:
            axes[1].plot(
                times, [row[key] for row in metrics], label=label,
                linestyle=linestyle, alpha=alpha,
            )
        for key, label in (("spectrum_low_k_relative_l1", "low $k$"), ("spectrum_mid_k_relative_l1", "mid $k$"), ("spectrum_high_k_relative_l1", "high $k$")):
            axes[2].semilogy(times, [row[key] for row in metrics], label=label)
        axes[0].set(xlabel=r"$t$ (s)", ylabel="relative $L^2$ error")
        axes[1].set(xlabel=r"$t$ (s)", ylabel="correlation", ylim=(-0.05, 1.05))
        axes[2].set(xlabel=r"$t$ (s)", ylabel="spectral relative $L^1$")
        for label, axis in zip(("(a)", "(b)", "(c)"), axes):
            _panel_label(axis, label)
            axis.legend(frameon=False)
            axis.grid(True, which="both", alpha=0.18)
        figure.tight_layout()
        _save_publication_figure(figure, paper_dir / "figure_3b_closure_metrics")
        plt.close(figure)

        # Figure 3c: direct audit of the manuscript's structural claims.
        figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.45))
        axes[0].plot(
            times, [row["pi_d_mean"] for row in metrics], label=r"$\langle\Pi_d\rangle$"
        )
        axes[0].plot(
            times, [row["pi_b_mean"] for row in metrics], label=r"$\langle\Pi_b\rangle$"
        )
        axes[0].axhline(0.0, color="0.35", linewidth=0.8)
        axes[0].set(xlabel=r"$t$ (s)", ylabel="mean transfer")
        axes[1].plot(
            times,
            [row["backscatter_ratio_maximum"] for row in metrics],
            label=(
                r"$\max |\mathbf{q}_b|/"
                r"\sqrt{|\mathbf{q}_d|^2+|\mathbf{q}_{G}|^2}$"
            ),
        )
        axes[1].axhline(
            float(summary["maximum_backscatter_ratio"]),
            color="C3",
            linestyle="--",
            label=r"prescribed $\beta$",
        )
        axes[1].set(xlabel=r"$t$ (s)", ylabel="backscatter ratio")
        axes[2].plot(
            times,
            [row["anisotropy_condition_mean"] for row in metrics],
            label=r"mean $\kappa(\mathbf{A})$",
        )
        axes[2].plot(
            times,
            [row["anisotropy_eigenvalue_minimum"] for row in metrics],
            label=r"min $\lambda(\mathbf{A})$",
        )
        axes[2].set(xlabel=r"$t$ (s)", ylabel="SPD diagnostic")
        for label, axis in zip(("(a)", "(b)", "(c)"), axes):
            _panel_label(axis, label)
            axis.legend(frameon=False)
            axis.grid(True, alpha=0.18)
        figure.tight_layout()
        _save_publication_figure(
            figure, paper_dir / "figure_3c_theory_constraints"
        )
        plt.close(figure)

        # Figure 4: make the sparse-data protocol auditable.
        ny, nx = grid_shape
        figure, axis = plt.subplots(figsize=(3.5, 3.7))
        def coordinates(indices):
            return indices % nx, indices // nx
        x_train, y_train = coordinates(train_indices)
        x_val, y_val = coordinates(validation_indices)
        x_closure, y_closure = coordinates(closure_indices)
        axis.scatter(x_train, y_train, s=9, facecolors="none", edgecolors="C0", linewidths=0.6, label=f"flow train: {len(train_indices)} (4%)")
        axis.scatter(x_val, y_val, s=12, marker="x", color="C1", linewidths=0.7, label=f"held-out: {len(validation_indices)} (1%)")
        axis.scatter(x_closure, y_closure, s=13, marker="s", color="C3", label=f"SGS labels/time: {len(closure_indices)} (0.25%)")
        axis.set(xlabel="grid index $i$", ylabel="grid index $j$", xlim=(-1, nx), ylim=(-1, ny), aspect="equal")
        axis.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=1,
        )
        axis.grid(True, alpha=0.15)
        _panel_label(axis, "(a)")
        figure.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.27)
        _save_publication_figure(figure, paper_dir / "figure_5_sparse_sensor_protocol")
        plt.close(figure)

        # Supplement: do not confuse assimilated reconstruction with autonomous closure prediction.
        if autonomous_metrics:
            figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharex=True)
            for axis, key, title in zip(axes, ("u_relative_l2", "v_relative_l2", "omega_relative_l2"), (r"$u$", r"$v$", r"$\omega$")):
                axis.plot(times, [row[key] for row in metrics], label="4% assimilated")
                axis.plot(times, [row[key] for row in autonomous_metrics], "--", label="autonomous")
                axis.set(title=title, xlabel=r"$t$ (s)", ylabel="relative $L^2$")
                axis.grid(True, alpha=0.18)
            axes[0].legend(frameon=False)
            for label, axis in zip(("(a)", "(b)", "(c)"), axes):
                _panel_label(axis, label)
            figure.tight_layout()
            _save_publication_figure(figure, paper_dir / "figure_s1_assimilation_vs_autonomous")
            plt.close(figure)

    # Machine-readable and LaTeX-ready quantitative table.
    table_rows = [
        ("u relative L2", summary["field_relative_l2"]["u_relative_l2"]["mean"]),
        ("v relative L2", summary["field_relative_l2"]["v_relative_l2"]["mean"]),
        ("omega relative L2", summary["field_relative_l2"]["omega_relative_l2"]["mean"]),
        ("u correlation", summary["field_correlation"]["u_correlation"]["mean"]),
        ("v correlation", summary["field_correlation"]["v_correlation"]["mean"]),
        ("omega correlation", summary["field_correlation"]["omega_correlation"]["mean"]),
        ("kinetic-energy MAPE", summary["kinetic_energy"]["mean_absolute_percentage_error"]),
        ("enstrophy MAPE", summary["enstrophy"]["mean_absolute_percentage_error"]),
        ("resolved-spectrum relative L1", summary["resolved_spectrum_relative_l1"]["mean"]),
        ("high-k spectrum relative L1", summary["resolved_spectrum_band_relative_l1"]["high_k"]["mean"]),
    ]
    with (paper_dir / "table_1_quantitative_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "mean value"))
        writer.writerows(table_rows)
    latex_lines = [r"\begin{tabular}{lr}", r"\hline", r"Metric & Mean value \\", r"\hline"]
    for name, value in table_rows:
        latex_name = name.replace("_", "\\_")
        latex_lines.append(f"{latex_name} & {value:.4g} \\\\")
    latex_lines.extend((r"\hline", r"\end{tabular}"))
    (paper_dir / "table_1_quantitative_metrics.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    report_lines = [
        "# SP-NSGS paper-results handoff",
        "",
        "## Supported primary claim",
        "",
        "Sparse-assimilated reconstruction using fixed 4% flow sensors and a "
        "time-rotating 0.25% subset of SGS labels. Dense STAR fields are used "
        "only after training for evaluation.",
        "",
        "## Quantitative summary (mean over the test interval)",
        "",
        f"- Relative L2: u={table_rows[0][1]:.4f}, v={table_rows[1][1]:.4f}, omega={table_rows[2][1]:.4f}",
        f"- Correlation: u={table_rows[3][1]:.4f}, v={table_rows[4][1]:.4f}, omega={table_rows[5][1]:.4f}",
        f"- Kinetic-energy MAPE={100.0 * table_rows[6][1]:.2f}%",
        f"- Enstrophy MAPE={100.0 * table_rows[7][1]:.2f}%",
        f"- Resolved-spectrum relative L1={table_rows[8][1]:.4f}",
        f"- High-k-band relative L1={table_rows[9][1]:.4f}",
        "",
        "## Claim boundary",
        "",
        "Figure S1 is an autonomous rollout audit. It must not be replaced by "
        "the assimilated reconstruction in any claim about closure-model "
        "predictive capability. The a priori SGS error and high-k-band error "
        "should be reported, not hidden.",
    ]
    (paper_dir / "README_results.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def _ablation_rollout(
    solver: SpectralVorticitySolver,
    initial_omega: torch.Tensor,
    times: torch.Tensor,
    reference: dict[str, torch.Tensor] | None = None,
    nudger: _SparseVorticityNudger | None = None,
) -> tuple[list[torch.Tensor], float, float | None]:
    """Roll out a baseline without allowing its instability to abort prediction."""
    states = [solver.project(initial_omega)]
    omega = states[0]
    failure_time = None
    for index in range(1, times.numel()):
        start = float(times[index - 1].item())
        dt = float((times[index] - times[index - 1]).item())
        omega = solver.advance(omega, start, dt)
        if nudger is not None:
            if reference is None:
                raise ValueError("an assimilated ablation requires reference sensors")
            omega = nudger(
                omega,
                reference["omega"][index],
                reference["u"][index],
                reference["v"][index],
            )
        if not torch.isfinite(omega).all():
            failure_time = float(times[index].item())
            break
        states.append(omega)
    stable_horizon = float(times[len(states) - 1].item())
    return states, stable_horizon, failure_time


def _evaluate_paper_ablation(
    solver: SpectralVorticitySolver,
    snapshots,
    reference: dict[str, torch.Tensor],
    times: torch.Tensor,
    train_indices: torch.Tensor,
    full_assimilated_states: list[torch.Tensor],
    full_autonomous_states: list[torch.Tensor],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compare closure structures under exactly the same sparse protocol."""
    original_scale = float(solver.closure_output_scale)
    original_backscatter_factor = float(solver.closure.backscatter_factor)
    nudger = _SparseVorticityNudger(solver, train_indices)
    final_time = float(times[-1].item())
    modes: dict[str, dict[str, object]] = {
        "SP-NSGS": {
            "assimilated": full_assimilated_states,
            "autonomous": full_autonomous_states,
            "assimilated_stable_horizon_s": final_time,
            "autonomous_stable_horizon_s": final_time,
            "assimilated_failure_time_s": None,
            "autonomous_failure_time_s": None,
        }
    }
    solver.eval()
    try:
        with torch.no_grad():
            # Same governing solver and same observations, but q_sgs = 0.
            solver.set_closure_output_scale(0.0)
            solver.closure.set_backscatter_factor(1.0)
            assimilated, assimilated_horizon, assimilated_failure = (
                _ablation_rollout(
                    solver,
                    reference["omega"][0],
                    times,
                    reference,
                    nudger,
                )
            )
            autonomous, autonomous_horizon, autonomous_failure = (
                _ablation_rollout(
                    solver, reference["omega"][0], times
                )
            )
            modes["no SGS"] = {
                "assimilated": assimilated,
                "autonomous": autonomous,
                "assimilated_stable_horizon_s": assimilated_horizon,
                "autonomous_stable_horizon_s": autonomous_horizon,
                "assimilated_failure_time_s": assimilated_failure,
                "autonomous_failure_time_s": autonomous_failure,
            }
            # Retain the learned SPD dissipative branch and suppress only the
            # bounded backscatter branch.  This is the direct theory ablation.
            solver.set_closure_output_scale(1.0)
            solver.closure.set_backscatter_factor(0.0)
            assimilated, assimilated_horizon, assimilated_failure = (
                _ablation_rollout(
                    solver,
                    reference["omega"][0],
                    times,
                    reference,
                    nudger,
                )
            )
            autonomous, autonomous_horizon, autonomous_failure = (
                _ablation_rollout(
                    solver, reference["omega"][0], times
                )
            )
            modes["dissipative only"] = {
                "assimilated": assimilated,
                "autonomous": autonomous,
                "assimilated_stable_horizon_s": assimilated_horizon,
                "autonomous_stable_horizon_s": autonomous_horizon,
                "assimilated_failure_time_s": assimilated_failure,
                "autonomous_failure_time_s": autonomous_failure,
            }
    finally:
        solver.set_closure_output_scale(original_scale)
        solver.closure.set_backscatter_factor(original_backscatter_factor)

    rows: list[dict[str, float | str]] = []
    curves: dict[str, dict[str, list[float]]] = {}
    for model_name, mode_states in modes.items():
        assimilated = mode_states["assimilated"]
        autonomous = mode_states["autonomous"]
        curves[model_name] = {
            "time_s": [],
            "assimilated_omega_relative_l2": [],
            "assimilated_energy_relative_error": [],
            "autonomous_omega_relative_l2": [],
            "autonomous_energy_relative_error": [],
        }
        for index, snapshot in enumerate(snapshots):
            if not (
                solver.config.data.test_time_min_s - 1e-12
                <= snapshot.time_s
                <= solver.config.data.test_time_max_s + 1e-12
            ):
                continue
            ref_omega = snapshot.omega.reshape(snapshot.ny, snapshot.nx)
            ref_u = snapshot.u.reshape(snapshot.ny, snapshot.nx)
            ref_v = snapshot.v.reshape(snapshot.ny, snapshot.nx)
            reference_energy = max(
                float(0.5 * np.mean(ref_u**2 + ref_v**2)), 1e-30
            )
            row: dict[str, float | str] = {
                "model": model_name,
                "time_s": float(snapshot.time_s),
            }
            for mode_name, states_for_mode in (
                ("assimilated", assimilated),
                ("autonomous", autonomous),
            ):
                if index >= len(states_for_mode):
                    omega_error = float("nan")
                    energy_error = float("nan")
                else:
                    state = solver.state(states_for_mode[index])
                    omega = state["omega"].detach().cpu().numpy()
                    u = state["u"].detach().cpu().numpy()
                    v = state["v"].detach().cpu().numpy()
                    omega_error = _relative_l2(omega, ref_omega)
                    energy = float(0.5 * np.mean(u**2 + v**2))
                    energy_error = (
                        abs(energy - reference_energy) / reference_energy
                    )
                row[f"{mode_name}_omega_relative_l2"] = omega_error
                row[f"{mode_name}_energy_relative_error"] = energy_error
                curves[model_name][f"{mode_name}_omega_relative_l2"].append(
                    omega_error
                )
                curves[model_name][f"{mode_name}_energy_relative_error"].append(
                    energy_error
                )
            curves[model_name]["time_s"].append(float(snapshot.time_s))
            rows.append(row)

    with (output_dir / "ablation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for model_name, model_curves in curves.items():
        aggregate[model_name] = {
            key: float(np.nanmean(values))
            for key, values in model_curves.items()
            if key != "time_s" and np.any(np.isfinite(values))
        }
        aggregate[model_name].update(
            {
                key: value
                for key, value in modes[model_name].items()
                if key.endswith("_horizon_s") or key.endswith("_failure_time_s")
            }
        )
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return aggregate
    paper_dir = output_dir / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.8), sharex=True)
        panels = (
            ("assimilated_omega_relative_l2", "4% assimilated", r"$\epsilon_\omega$"),
            ("assimilated_energy_relative_error", "4% assimilated", r"$|\Delta K|/K_{ref}$"),
            ("autonomous_omega_relative_l2", "autonomous", r"$\epsilon_\omega$"),
            ("autonomous_energy_relative_error", "autonomous", r"$|\Delta K|/K_{ref}$"),
        )
        styles = {
            "no SGS": (":", "0.45"),
            "dissipative only": ("--", "C1"),
            "SP-NSGS": ("-", "C0"),
        }
        for axis, (key, mode_title, ylabel) in zip(axes.flat, panels):
            for model_name, model_curves in curves.items():
                linestyle, color = styles[model_name]
                axis.plot(
                    model_curves["time_s"],
                    model_curves[key],
                    linestyle=linestyle,
                    color=color,
                    label=model_name,
                )
            axis.set(xlabel=r"$t$ (s)", ylabel=ylabel, title=mode_title)
            axis.grid(True, alpha=0.18)
        axes[0, 0].legend(frameon=False)
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
            _panel_label(axis, label)
        figure.tight_layout()
        _save_publication_figure(
            figure, paper_dir / "figure_6_closure_ablation"
        )
        plt.close(figure)
    return aggregate


def predict_paper(config: Config, checkpoint_path: str | Path) -> Path:
    device = _device(config.training.device)
    dtype = _dtype(config.training.dtype)
    snapshots, reference = _prepare_reference(config, device, dtype)
    solver = SpectralVorticitySolver(
        config,
        snapshots[0].ny,
        snapshots[0].nx,
        snapshots[0].dx,
        snapshots[0].dy,
        device,
        dtype,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_version = int(config.paper.checkpoint_format_version)
    if checkpoint.get("version") != expected_version:
        raise ValueError(
            "checkpoint format does not match this SP-NSGS paper solver: "
            f"{checkpoint.get('version')!r} != {expected_version!r}"
        )
    checkpoint_parameterization = checkpoint.get(
        "closure_parameterization", "flux"
    )
    if checkpoint_parameterization != config.paper.closure_parameterization:
        raise ValueError(
            "checkpoint closure parameterisation does not match the config: "
            f"{checkpoint_parameterization!r} != "
            f"{config.paper.closure_parameterization!r}"
        )
    checkpoint_feature_set = checkpoint.get(
        "closure_feature_set", "normalised"
    )
    if checkpoint_feature_set != config.paper.closure_feature_set:
        raise ValueError(
            "checkpoint closure feature set does not match the config: "
            f"{checkpoint_feature_set!r} != "
            f"{config.paper.closure_feature_set!r}"
        )
    solver.load_state_dict(checkpoint["model_state"])
    solver.closure.set_backscatter_factor(1.0)
    solver.set_closure_output_scale(
        float(checkpoint.get("closure_output_scale", 1.0))
    )
    solver.eval()
    train_indices = checkpoint["train_indices"].to(device=device, dtype=torch.long)
    validation_indices = checkpoint["validation_indices"].to(
        device=device, dtype=torch.long
    )
    _, _, closure_schedule, _, _, _ = _sensor_protocol(
        config, snapshots, reference, device
    )
    nudger = _SparseVorticityNudger(solver, train_indices)
    with torch.no_grad():
        autonomous_states = _rollout(
            solver, reference["omega"][0], reference["time"], keep_states=True
        )
        states = _assimilated_rollout(
            solver,
            reference["omega"][0],
            reference["time"],
            reference,
            nudger,
        )
    prediction_dir = Path(config.training.output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.get("history"):
        _write_history(Path(config.training.output_dir), checkpoint["history"])
    metrics: list[dict] = []
    records: list[dict] = []
    closure_records: list[dict] = []
    autonomous_metrics: list[dict] = []
    all_omega = []
    all_u = []
    all_v = []
    final_spectrum = None
    evaluation_times = np.asarray(config.paper.evaluation_times_s)

    autonomous_errors = {"u": [], "v": [], "omega": []}
    for index, (snapshot, omega_tensor) in enumerate(zip(snapshots, states)):
        if snapshot.time_s < config.data.test_time_min_s - 1e-12 or snapshot.time_s > config.data.test_time_max_s + 1e-12:
            continue
        with torch.no_grad():
            state = solver.state(omega_tensor)
            closure = solver.rhs(omega_tensor, snapshot.time_s)
        omega = state["omega"].cpu().numpy()
        u = state["u"].cpu().numpy()
        v = state["v"].cpu().numpy()
        ref_omega = snapshot.omega.reshape(snapshot.ny, snapshot.nx)
        ref_u = snapshot.u.reshape(snapshot.ny, snapshot.nx)
        ref_v = snapshot.v.reshape(snapshot.ny, snapshot.nx)
        with torch.no_grad():
            autonomous_state = solver.state(autonomous_states[index])
        autonomous_errors["omega"].append(
            _relative_l2(autonomous_state["omega"].cpu().numpy(), ref_omega)
        )
        autonomous_errors["u"].append(
            _relative_l2(autonomous_state["u"].cpu().numpy(), ref_u)
        )
        autonomous_errors["v"].append(
            _relative_l2(autonomous_state["v"].cpu().numpy(), ref_v)
        )
        autonomous_metrics.append(
            {
                "time_s": snapshot.time_s,
                "u_relative_l2": autonomous_errors["u"][-1],
                "v_relative_l2": autonomous_errors["v"][-1],
                "omega_relative_l2": autonomous_errors["omega"][-1],
            }
        )
        q_pred = np.stack(
            [closure["qx"].cpu().numpy(), closure["qy"].cpu().numpy()], axis=-1
        )
        q_ref = np.stack(
            [
                snapshot.q_sgs_x.reshape(snapshot.ny, snapshot.nx),
                snapshot.q_sgs_y.reshape(snapshot.ny, snapshot.nx),
            ],
            axis=-1,
        )
        div_pred = closure["div_q"].cpu().numpy()
        div_ref = snapshot.div_q_sgs.reshape(snapshot.ny, snapshot.nx)
        pi_pred = closure["pi"].cpu().numpy()
        pi_d_pred = closure["pi_d"].cpu().numpy()
        pi_b_pred = closure["pi_b"].cpu().numpy()
        backscatter_ratio_pred = closure["backscatter_ratio"].cpu().numpy()
        anisotropy_condition_pred = closure["A_condition_number"].cpu().numpy()
        anisotropy_eigenvalue_min_pred = closure["A_eigenvalue_min"].cpu().numpy()
        pi_ref = snapshot.pi_sgs.reshape(snapshot.ny, snapshot.nx)
        wave, spectrum_ref = _radial_spectrum(ref_u, ref_v)
        _, spectrum_pred = _radial_spectrum(u, v)
        cutoff = min(spectrum_ref.size, spectrum_pred.size, int(min(snapshot.nx, snapshot.ny) / 3) + 1)
        resolved_cutoff = cutoff - 1
        spectral_tail_start = max(
            2,
            int(
                math.ceil(
                    config.paper.spectral_tail_start_fraction
                    * resolved_cutoff
                )
            ),
        )
        spectrum_error = float(
            np.sum(np.abs(spectrum_pred[1:cutoff] - spectrum_ref[1:cutoff]))
            / max(np.sum(np.abs(spectrum_ref[1:cutoff])), 1e-30)
        )

        def spectrum_band_error(start: int, stop: int) -> float:
            stop = min(stop, cutoff - 1)
            if stop < start:
                return float("nan")
            band = slice(start, stop + 1)
            return float(
                np.sum(np.abs(spectrum_pred[band] - spectrum_ref[band]))
                / max(np.sum(np.abs(spectrum_ref[band])), 1e-30)
            )
        if 0 < index < len(states) - 1:
            dt = float(reference["time"][index + 1] - reference["time"][index - 1])
            omega_t = (states[index + 1] - states[index - 1]) / dt
            pde_rms = float(torch.sqrt(torch.mean((omega_t - closure["rhs"]).square())).item())
        else:
            pde_rms = float("nan")
        row = {
            "time_s": snapshot.time_s,
            "u_relative_l2": _relative_l2(u, ref_u),
            "v_relative_l2": _relative_l2(v, ref_v),
            "omega_relative_l2": _relative_l2(omega, ref_omega),
            "u_correlation": _correlation(u, ref_u),
            "v_correlation": _correlation(v, ref_v),
            "omega_correlation": _correlation(omega, ref_omega),
            "kinetic_energy_ref": float(0.5 * np.mean(ref_u**2 + ref_v**2)),
            "kinetic_energy_pred": float(0.5 * np.mean(u**2 + v**2)),
            "enstrophy_ref": float(0.5 * np.mean(ref_omega**2)),
            "enstrophy_pred": float(0.5 * np.mean(omega**2)),
            "q_sgs_relative_l2": _relative_l2(q_pred, q_ref),
            "q_sgs_correlation": _correlation(q_pred, q_ref),
            "pi_sgs_relative_l2": _relative_l2(pi_pred, pi_ref),
            "pi_sgs_correlation": _correlation(pi_pred, pi_ref),
            "pi_sgs_mean_ref": float(np.mean(pi_ref)),
            "pi_sgs_mean_pred": float(np.mean(pi_pred)),
            "pi_sgs_backscatter_fraction_ref": float(np.mean(pi_ref < 0.0)),
            "pi_sgs_backscatter_fraction_pred": float(np.mean(pi_pred < 0.0)),
            "pi_d_mean": float(np.mean(pi_d_pred)),
            "pi_d_minimum": float(np.min(pi_d_pred)),
            "pi_d_negative_fraction": float(np.mean(pi_d_pred < -1e-10)),
            "pi_b_mean": float(np.mean(pi_b_pred)),
            "pi_b_backscatter_fraction": float(np.mean(pi_b_pred < 0.0)),
            "backscatter_ratio_mean": float(np.mean(backscatter_ratio_pred)),
            "backscatter_ratio_maximum": float(np.max(backscatter_ratio_pred)),
            "anisotropy_condition_mean": float(
                np.mean(anisotropy_condition_pred)
            ),
            "anisotropy_condition_maximum": float(
                np.max(anisotropy_condition_pred)
            ),
            "anisotropy_eigenvalue_minimum": float(
                np.min(anisotropy_eigenvalue_min_pred)
            ),
            "div_q_sgs_relative_l2": _relative_l2(div_pred, div_ref),
            "div_q_sgs_correlation": _correlation(div_pred, div_ref),
            "resolved_spectrum_relative_l1": spectrum_error,
            "spectrum_low_k_relative_l1": spectrum_band_error(1, 10),
            "spectrum_mid_k_relative_l1": spectrum_band_error(
                11, spectral_tail_start - 1
            ),
            "spectrum_high_k_relative_l1": spectrum_band_error(
                spectral_tail_start, resolved_cutoff
            ),
            "pde_residual_rms_per_s2": pde_rms,
        }
        metrics.append(row)
        all_omega.append(omega)
        all_u.append(u)
        all_v.append(v)
        if np.any(np.isclose(snapshot.time_s, evaluation_times, atol=1e-9)):
            records.append(
                {"time_s": snapshot.time_s, "reference": ref_omega, "prediction": omega}
            )
            closure_records.append(
                {
                    "time_s": snapshot.time_s,
                    "div_q_reference": div_ref,
                    "div_q_prediction": div_pred,
                    "pi_reference": pi_ref,
                    "pi_prediction": pi_pred,
                    "pi_d_prediction": pi_d_pred,
                    "pi_b_prediction": pi_b_pred,
                }
            )
        final_spectrum = (wave, spectrum_ref, spectrum_pred)

    with (prediction_dir / "prediction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    energy_ref = np.asarray([row["kinetic_energy_ref"] for row in metrics])
    energy_pred = np.asarray([row["kinetic_energy_pred"] for row in metrics])
    enstrophy_ref = np.asarray([row["enstrophy_ref"] for row in metrics])
    enstrophy_pred = np.asarray([row["enstrophy_pred"] for row in metrics])
    ablation = _evaluate_paper_ablation(
        solver,
        snapshots,
        reference,
        reference["time"],
        train_indices,
        states,
        autonomous_states,
        prediction_dir,
    )
    summary = {
        "version": int(config.paper.checkpoint_format_version),
        "closure_parameterization": config.paper.closure_parameterization,
        "closure_feature_set": config.paper.closure_feature_set,
        "theory_model": "q_sgs = q_d + q_b",
        "dissipation_coefficient_initial_bias": float(
            solver.closure.dissipation_coefficient_initial_bias
        ),
        "dissipation_coefficient_soft_limit": float(
            solver.closure.dissipation_coefficient_soft_limit
        ),
        "maximum_backscatter_ratio": float(
            solver.closure.maximum_backscatter_ratio
        ),
        "maximum_local_backscatter_transfer_ratio": float(
            solver.closure.maximum_local_backscatter_transfer_ratio
        ),
        "learned_gradient_support_coefficient": float(
            (
                solver.closure.gradient_support_coefficient_maximum
                * torch.sigmoid(solver.closure.gradient_support_coefficient_raw)
            ).detach().cpu().item()
        ),
        "maximum_anisotropy_condition": float(
            solver.closure.maximum_anisotropy_condition
        ),
        "autonomous_validation_weight": float(
            config.paper.autonomous_validation_weight
        ),
        "coordinate_weighted_sparse_moments": bool(
            config.paper.coordinate_weighted_sparse_moments
        ),
        "sparse_energy_tendency_weight": float(
            config.paper.sparse_energy_tendency_weight
        ),
        "sparse_sgs_mean_transfer_weight": float(
            config.paper.sparse_sgs_mean_transfer_weight
        ),
        "sparse_sgs_branch_transfer_weight": float(
            config.paper.sparse_sgs_branch_transfer_weight
        ),
        "physics_inferred_divergence_weight": float(
            config.paper.physics_inferred_divergence_weight
        ),
        "physics_inferred_divergence_note": (
            "formed only at existing sparse flow sensors and amplitude-"
            "calibrated with the unchanged 0.25% direct SGS labels"
        ),
        "theory_compliance": (
            "A is trace-normalised SPD with a hard condition-number cap, "
            "c_theta is positive, "
            "and |q_b| <= beta sqrt(|q_d|^2 + |q_Clark|^2) holds "
            "algebraically at every grid point; q_Clark is only the bounded "
            "support scale/direction prior inside q_b, not a third flux branch"
        ),
        "test_protocol": (
            "main fields use only fixed 4% u/v/omega training-sensor nudging; "
            "0.25% SGS labels are used at a subset of those positions; dense "
            "fields are used only for post-training metrics"
        ),
        "prediction_mode": "sparse-assimilated reconstruction",
        "closure_ablation": ablation,
        "closure_identifiability_note": (
            "The vorticity equation primarily identifies div(q_sgs). Sparse "
            "vector-flux and transfer labels additionally constrain the "
            "theory-prescribed dissipative/backscatter representative."
        ),
        "field_relative_l2": {
            key: _aggregate([row[key] for row in metrics])
            for key in ("u_relative_l2", "v_relative_l2", "omega_relative_l2")
        },
        "field_correlation": {
            key: _aggregate([row[key] for row in metrics])
            for key in ("u_correlation", "v_correlation", "omega_correlation")
        },
        "kinetic_energy": {
            "mean_absolute_percentage_error": float(np.mean(np.abs(energy_pred - energy_ref) / np.maximum(np.abs(energy_ref), 1e-30))),
            "maximum_absolute_percentage_error": float(np.max(np.abs(energy_pred - energy_ref) / np.maximum(np.abs(energy_ref), 1e-30))),
        },
        "enstrophy": {
            "mean_absolute_percentage_error": float(np.mean(np.abs(enstrophy_pred - enstrophy_ref) / np.maximum(np.abs(enstrophy_ref), 1e-30))),
            "maximum_absolute_percentage_error": float(np.max(np.abs(enstrophy_pred - enstrophy_ref) / np.maximum(np.abs(enstrophy_ref), 1e-30))),
        },
        "closure": {
            key: _aggregate([row[key] for row in metrics])
            for key in (
                "q_sgs_relative_l2",
                "q_sgs_correlation",
                "pi_sgs_relative_l2",
                "pi_sgs_correlation",
                "div_q_sgs_relative_l2",
                "div_q_sgs_correlation",
            )
        },
        "theory_diagnostics": {
            key: _aggregate([row[key] for row in metrics])
            for key in (
                "pi_d_mean",
                "pi_d_minimum",
                "pi_d_negative_fraction",
                "pi_b_mean",
                "pi_b_backscatter_fraction",
                "backscatter_ratio_mean",
                "backscatter_ratio_maximum",
                "anisotropy_condition_mean",
                "anisotropy_condition_maximum",
                "anisotropy_eigenvalue_minimum",
            )
        },
        "primary_identifiable_closure_metrics": {
            key: _aggregate([row[key] for row in metrics])
            for key in (
                "div_q_sgs_relative_l2",
                "div_q_sgs_correlation",
                "pi_sgs_relative_l2",
                "pi_sgs_correlation",
            )
        },
        "sparse_vector_flux_diagnostic": {
            key: _aggregate([row[key] for row in metrics])
            for key in ("q_sgs_relative_l2", "q_sgs_correlation")
        },
        "resolved_spectrum_relative_l1": _aggregate([row["resolved_spectrum_relative_l1"] for row in metrics]),
        "resolved_spectrum_band_relative_l1": {
            "low_k": _aggregate([row["spectrum_low_k_relative_l1"] for row in metrics]),
            "mid_k": _aggregate([row["spectrum_mid_k_relative_l1"] for row in metrics]),
            "high_k": _aggregate([row["spectrum_high_k_relative_l1"] for row in metrics]),
            "high_k_definition": f"k={spectral_tail_start}..{resolved_cutoff}",
        },
        "pde_residual_rms_per_s2": _aggregate([row["pde_residual_rms_per_s2"] for row in metrics]),
        "autonomous_rollout_field_relative_l2": {
            key: _aggregate(values) for key, values in autonomous_errors.items()
        },
    }
    (prediction_dir / "quantitative_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        prediction_dir / "rollout_fields.npz",
        times_s=np.asarray([row["time_s"] for row in metrics]),
        omega=np.stack(all_omega),
        u=np.stack(all_u),
        v=np.stack(all_v),
    )
    np.savez_compressed(
        prediction_dir / "autonomous_rollout_fields.npz",
        times_s=np.asarray([snapshot.time_s for snapshot in snapshots]),
        omega=np.stack([state.cpu().numpy() for state in autonomous_states]),
    )
    if final_spectrum is not None and records:
        _write_prediction_plots(
            prediction_dir,
            records,
            metrics,
            autonomous_metrics,
            closure_records,
            final_spectrum,
            spectral_tail_start,
            resolved_cutoff,
            train_indices.detach().cpu().numpy(),
            validation_indices.detach().cpu().numpy(),
            closure_schedule[-1].detach().cpu().numpy(),
            (snapshots[0].ny, snapshots[0].nx),
            summary,
            config.paper.signed_omega_error_color_limit,
            config.paper.absolute_omega_error_color_limit_per_s,
            config.paper.paper_show_absolute_closure_error_panels,
        )
    return prediction_dir / "prediction_metrics.csv"
