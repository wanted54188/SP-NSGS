from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .filters import (
    filter_vorticity_snapshot,
    gaussian_filter_2d,
    periodic_streamfunction_from_vorticity,
    periodic_second_derivatives,
    periodic_divergence,
)


REQUIRED_COLUMNS = {
    "x": "Position[X] (m)",
    "y": "Position[Y] (m)",
    "u": "Velocity[i] (m/s)",
    "v": "Velocity[j] (m/s)",
    "omega": "Vorticity[k] (/s)",
}
OPTIONAL_COLUMNS = {
    "z": "Position[Z] (m)",
    "pressure": "Pressure (Pa)",
    "q_sgs_x": "SGS Vorticity Flux[i] (m/s^2)",
    "q_sgs_y": "SGS Vorticity Flux[j] (m/s^2)",
    "pi_sgs": "SGS Enstrophy Transfer (/s^3)",
}
TIME_PATTERN = re.compile(r"_([0-9]+(?:\.[0-9]+)?e[+-][0-9]+)\.csv$", re.IGNORECASE)


@dataclass
class StructuredGrid:
    x: np.ndarray
    y: np.ndarray
    fields: dict[str, np.ndarray]
    dx: float
    dy: float
    endpoint_grid: bool
    coordinate_max_error_m: float
    periodic_mismatch: dict[str, dict[str, float]]

    @property
    def nx(self) -> int:
        return self.x.size

    @property
    def ny(self) -> int:
        return self.y.size


@dataclass
class Snapshot:
    time_s: float
    source: Path
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    pressure: np.ndarray | None = None
    psi: np.ndarray | None = None
    omega_x: np.ndarray | None = None
    omega_y: np.ndarray | None = None
    omega_xx: np.ndarray | None = None
    omega_xy: np.ndarray | None = None
    omega_yy: np.ndarray | None = None
    u_x: np.ndarray | None = None
    u_y: np.ndarray | None = None
    v_x: np.ndarray | None = None
    v_y: np.ndarray | None = None
    q_sgs_x: np.ndarray | None = None
    q_sgs_y: np.ndarray | None = None
    pi_sgs: np.ndarray | None = None
    div_q_sgs: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.x.size


def time_from_filename(path: Path) -> float:
    match = TIME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse time from filename: {path.name}")
    return float(match.group(1))


def discover_files(data_dir: Path, file_glob: str) -> list[tuple[float, Path]]:
    found = [(time_from_filename(path), path) for path in data_dir.glob(file_glob)]
    if not found:
        raise FileNotFoundError(f"No files matching {file_glob!r} in {data_dir}")
    return sorted(found, key=lambda item: item[0])


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS.values() if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    numeric = list(REQUIRED_COLUMNS.values()) + [
        column for column in OPTIONAL_COLUMNS.values() if column in frame
    ]
    if frame[numeric].isna().any().any():
        raise ValueError(f"{path.name} contains NaN values")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return frame


def _relative_rms(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.sqrt(np.mean(left**2))
    return float(np.sqrt(np.mean((left - right) ** 2)) / max(scale, 1e-30))


def _periodic_mismatch(fields: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in ("u", "v", "omega"):
        array = fields[name]
        result[name] = {
            "x_relative_rms": _relative_rms(array[:, 0], array[:, -1]),
            "y_relative_rms": _relative_rms(array[0, :], array[-1, :]),
            "x_max_abs": float(np.max(np.abs(array[:, 0] - array[:, -1]))),
            "y_max_abs": float(np.max(np.abs(array[0, :] - array[-1, :]))),
        }
    return result


def _merge_periodic_endpoints(array: np.ndarray) -> np.ndarray:
    merged = array.copy()
    column_average = 0.5 * (merged[:, 0] + merged[:, -1])
    merged[:, 0] = column_average
    merged[:, -1] = column_average
    row_average = 0.5 * (merged[0, :] + merged[-1, :])
    merged[0, :] = row_average
    merged[-1, :] = row_average
    corner_average = float(
        np.mean([array[0, 0], array[0, -1], array[-1, 0], array[-1, -1]])
    )
    merged[0, 0] = merged[0, -1] = merged[-1, 0] = merged[-1, -1] = corner_average
    return merged


def read_structured_grid(path: Path, config: Config) -> StructuredGrid:
    """Snap STAR plane-table coordinates to the known periodic Cartesian grid."""
    frame = _read_frame(path)
    expected_x = config.data.grid_points_x
    expected_y = config.data.grid_points_y
    endpoint_rows = expected_x * expected_y
    unique_rows = (expected_x - 1) * (expected_y - 1)
    if len(frame) == endpoint_rows:
        nx, ny = expected_x, expected_y
        endpoint_grid = True
        dx = config.domain.lx / (nx - 1)
        dy = config.domain.ly / (ny - 1)
    elif len(frame) == unique_rows:
        nx, ny = expected_x - 1, expected_y - 1
        endpoint_grid = False
        dx = config.domain.lx / nx
        dy = config.domain.ly / ny
    else:
        raise ValueError(
            f"{path.name} has {len(frame)} rows; expected {endpoint_rows} "
            f"(with endpoints) or {unique_rows} (unique periodic grid)"
        )
    raw_x = frame[REQUIRED_COLUMNS["x"]].to_numpy(dtype=np.float64)
    raw_y = frame[REQUIRED_COLUMNS["y"]].to_numpy(dtype=np.float64)

    def snap_axis(
        values: np.ndarray,
        minimum: float,
        spacing: float,
        count: int,
        allow_half_shift: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        origins = [minimum]
        if allow_half_shift:
            origins.append(minimum + 0.5 * spacing)
        candidates = []
        for origin in origins:
            index = np.rint((values - origin) / spacing).astype(np.int64)
            snapped = origin + index * spacing
            valid = index.min() >= 0 and index.max() < count
            error = float(np.max(np.abs(values - snapped))) if valid else float("inf")
            candidates.append((error, index, snapped, origin))
        error, index, snapped, origin = min(candidates, key=lambda item: item[0])
        if not np.isfinite(error):
            raise ValueError(f"{path.name} contains coordinates outside the grid")
        return index, snapped, origin

    ix, snapped_x, x_origin = snap_axis(
        raw_x,
        config.domain.x_min_m,
        dx,
        nx,
        allow_half_shift=not endpoint_grid,
    )
    iy, snapped_y, y_origin = snap_axis(
        raw_y,
        config.domain.y_min_m,
        dy,
        ny,
        allow_half_shift=not endpoint_grid,
    )
    if (
        ix.min() < 0
        or ix.max() >= nx
        or iy.min() < 0
        or iy.max() >= ny
    ):
        raise ValueError(f"{path.name} contains coordinates outside the configured grid")
    coordinate_error = float(
        max(np.max(np.abs(raw_x - snapped_x)), np.max(np.abs(raw_y - snapped_y)))
    )
    if coordinate_error > config.data.coordinate_snap_tolerance_m:
        raise ValueError(
            f"{path.name} coordinate snap error {coordinate_error:.3e} m exceeds "
            f"{config.data.coordinate_snap_tolerance_m:.3e} m"
        )
    keys = iy * nx + ix
    if np.unique(keys).size != nx * ny:
        raise ValueError(f"{path.name} does not map one-to-one onto the Cartesian grid")
    order = np.argsort(keys)
    fields = {
        key: frame[column].to_numpy(dtype=np.float64)[order].reshape(ny, nx)
        for key, column in REQUIRED_COLUMNS.items()
        if key in ("u", "v", "omega")
    }
    for key, column in OPTIONAL_COLUMNS.items():
        if column in frame:
            fields[key] = frame[column].to_numpy(dtype=np.float64)[order].reshape(ny, nx)
    mismatch = _periodic_mismatch(fields) if endpoint_grid else {}
    if endpoint_grid and config.data.merge_periodic_endpoints:
        fields = {
            key: _merge_periodic_endpoints(value)
            for key, value in fields.items()
        }
    if endpoint_grid and config.data.drop_periodic_endpoint:
        fields = {key: value[:-1, :-1] for key, value in fields.items()}
        nx -= 1
        ny -= 1
    x = x_origin + np.arange(nx, dtype=np.float64) * dx
    y = y_origin + np.arange(ny, dtype=np.float64) * dy
    return StructuredGrid(
        x=x,
        y=y,
        fields=fields,
        dx=dx,
        dy=dy,
        endpoint_grid=endpoint_grid,
        coordinate_max_error_m=coordinate_error,
        periodic_mismatch=mismatch,
    )


def _flatten_grid(
    grid: StructuredGrid,
    time_s: float,
    source: Path,
    fields: dict[str, np.ndarray],
    exclude_layers: int,
) -> Snapshot:
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    mask = np.ones_like(xx, dtype=bool)
    if exclude_layers > 0:
        if 2 * exclude_layers >= min(grid.nx, grid.ny):
            raise ValueError("exclude_boundary_layers removes the entire grid")
        mask[:] = False
        mask[
            exclude_layers : grid.ny - exclude_layers,
            exclude_layers : grid.nx - exclude_layers,
        ] = True

    def optional(name: str) -> np.ndarray | None:
        return fields[name][mask] if name in fields else None

    return Snapshot(
        time_s=time_s,
        source=source,
        x=xx[mask],
        y=yy[mask],
        u=fields["u"][mask],
        v=fields["v"][mask],
        omega=fields["omega"][mask],
        nx=grid.nx - 2 * exclude_layers,
        ny=grid.ny - 2 * exclude_layers,
        dx=grid.dx,
        dy=grid.dy,
        pressure=optional("pressure"),
        psi=optional("psi"),
        omega_x=optional("omega_x"),
        omega_y=optional("omega_y"),
        omega_xx=optional("omega_xx"),
        omega_xy=optional("omega_xy"),
        omega_yy=optional("omega_yy"),
        u_x=optional("u_x"),
        u_y=optional("u_y"),
        v_x=optional("v_x"),
        v_y=optional("v_y"),
        q_sgs_x=optional("q_sgs_x"),
        q_sgs_y=optional("q_sgs_y"),
        pi_sgs=optional("pi_sgs"),
        div_q_sgs=optional("div_q_sgs"),
    )


def read_snapshot(path: Path, time_s: float, config: Config) -> Snapshot:
    grid = read_structured_grid(path, config)
    fields = dict(grid.fields)
    if not config.data.data_are_filtered:
        if not config.data.apply_filter_in_memory:
            raise ValueError(
                "Raw data requires apply_filter_in_memory=true or preprocessing first"
            )
        if config.data.filter_width_m <= 0.0:
            raise ValueError("filter_width_m must be positive")
        filtered = filter_vorticity_snapshot(
            fields["u"],
            fields["v"],
            fields["omega"],
            grid.dx,
            grid.dy,
            config.data.filter_width_m,
        )
        fields.update(filtered)
        if "pressure" in fields:
            pressure_bar, _, _ = gaussian_filter_2d(
                fields["pressure"],
                grid.dx,
                grid.dy,
                config.data.filter_width_m,
            )
            fields["pressure"] = pressure_bar - np.mean(pressure_bar)
    elif "pressure" in fields:
        fields["pressure"] = fields["pressure"] - np.mean(fields["pressure"])
    if "omega_x" not in fields or "omega_y" not in fields:
        _, fields["omega_x"], fields["omega_y"] = gaussian_filter_2d(
            fields["omega"], grid.dx, grid.dy, 0.0
        )
    if (
        "div_q_sgs" not in fields
        and "q_sgs_x" in fields
        and "q_sgs_y" in fields
    ):
        fields["div_q_sgs"] = periodic_divergence(
            fields["q_sgs_x"], fields["q_sgs_y"], grid.dx, grid.dy
        )
    if not all(name in fields for name in ("omega_xx", "omega_xy", "omega_yy")):
        (
            fields["omega_xx"],
            fields["omega_xy"],
            fields["omega_yy"],
        ) = periodic_second_derivatives(fields["omega"], grid.dx, grid.dy)
    for velocity_name in ("u", "v"):
        x_name, y_name = f"{velocity_name}_x", f"{velocity_name}_y"
        if x_name not in fields or y_name not in fields:
            _, fields[x_name], fields[y_name] = gaussian_filter_2d(
                fields[velocity_name], grid.dx, grid.dy, 0.0
            )
    if "psi" not in fields:
        fields["psi"] = periodic_streamfunction_from_vorticity(
            fields["omega"], grid.dx, grid.dy
        )
    return _flatten_grid(
        grid,
        time_s,
        path,
        fields,
        config.data.exclude_boundary_layers,
    )


def load_snapshots(
    config: Config, time_min: float, time_max: float
) -> list[Snapshot]:
    selected = [
        (time_s, path)
        for time_s, path in discover_files(config.data.data_dir, config.data.file_glob)
        if time_min - 1e-12 <= time_s <= time_max + 1e-12
    ]
    if not selected:
        raise ValueError(f"No snapshots in time interval [{time_min}, {time_max}] s")
    snapshots = [read_snapshot(path, time_s, config) for time_s, path in selected]
    x0, y0 = snapshots[0].x, snapshots[0].y
    for snapshot in snapshots[1:]:
        if (
            snapshot.size != snapshots[0].size
            or not np.array_equal(snapshot.x, x0)
            or not np.array_equal(snapshot.y, y0)
        ):
            raise ValueError(f"Grid changed in {snapshot.source.name}")
    return snapshots


def load_observation_snapshots(config: Config) -> list[Snapshot]:
    interval = config.data.observation_time_interval_s
    if interval <= 0.0:
        raise ValueError("observation_time_interval_s must be positive")
    selected = []
    for time_s, path in discover_files(config.data.data_dir, config.data.file_glob):
        if not (
            config.data.train_time_min_s - 1e-12
            <= time_s
            <= config.data.train_time_max_s + 1e-12
        ):
            continue
        position = (time_s - config.data.train_time_min_s) / interval
        if abs(position - round(position)) <= 1e-8:
            selected.append(read_snapshot(path, time_s, config))
    if not selected:
        raise ValueError("No snapshots match observation_time_interval_s")
    return selected


def stack_observations(
    snapshots: Iterable[Snapshot],
    spatial_fraction: float = 1.0,
    seed: int = 0,
    sampling: str = "random",
    sensor_layout: str = "independent",
) -> dict[str, np.ndarray]:
    if not 0.0 < spatial_fraction <= 1.0:
        raise ValueError("observation_spatial_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    items = list(snapshots)
    if sensor_layout not in {"independent", "fixed"}:
        raise ValueError(
            "observation_sensor_layout must be 'independent' or 'fixed'"
        )
    indices = []
    fixed_index = None
    for item_index, item in enumerate(items):
        count = max(1, int(round(item.size * spatial_fraction)))
        if sensor_layout == "fixed" and item_index > 0:
            if fixed_index is None or fixed_index.size != count:
                raise ValueError(
                    "Fixed sensor layout requires an unchanged grid"
                )
            index = fixed_index
        elif count == item.size:
            index = np.arange(item.size)
        elif sampling == "random":
            index = np.sort(rng.choice(item.size, size=count, replace=False))
        elif sampling == "stratified":
            index = _stratified_grid_indices(item, count, rng)
        else:
            raise ValueError("observation_sampling must be 'random' or 'stratified'")
        if sensor_layout == "fixed" and item_index == 0:
            fixed_index = index.copy()
        indices.append(index)
    result = {
        "x": np.concatenate([item.x[index] for item, index in zip(items, indices)]),
        "y": np.concatenate([item.y[index] for item, index in zip(items, indices)]),
        "t": np.concatenate(
            [
                np.full(index.size, item.time_s, dtype=np.float64)
                for item, index in zip(items, indices)
            ]
        ),
        "u": np.concatenate([item.u[index] for item, index in zip(items, indices)]),
        "v": np.concatenate([item.v[index] for item, index in zip(items, indices)]),
        "omega": np.concatenate(
            [item.omega[index] for item, index in zip(items, indices)]
        ),
    }
    for key in (
        "psi",
        "omega_x",
        "omega_y",
        "omega_xx",
        "omega_xy",
        "omega_yy",
        "u_x",
        "u_y",
        "v_x",
        "v_y",
        "q_sgs_x",
        "q_sgs_y",
        "pi_sgs",
        "div_q_sgs",
    ):
        if all(getattr(item, key) is not None for item in items):
            result[key] = np.concatenate(
                [getattr(item, key)[index] for item, index in zip(items, indices)]
            )
    return result


def _stratified_grid_indices(
    snapshot: Snapshot,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose one sensor per spatial block for uniform sparse coverage."""
    best = None
    target_aspect = snapshot.ny / snapshot.nx
    for rows in range(1, min(snapshot.ny, count) + 1):
        columns = int(np.ceil(count / rows))
        if columns > snapshot.nx:
            continue
        cells = rows * columns
        aspect_error = abs(np.log((rows / columns) / target_aspect))
        score = aspect_error + 2.0 * (cells - count) / count
        if best is None or score < best[0]:
            best = (score, rows, columns)
    if best is None:
        return np.sort(rng.choice(snapshot.size, size=count, replace=False))
    _, rows, columns = best
    y_edges = np.linspace(0, snapshot.ny, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, snapshot.nx, columns + 1, dtype=np.int64)
    selected = []
    for row in range(rows):
        for column in range(columns):
            iy = int(rng.integers(y_edges[row], y_edges[row + 1]))
            ix = int(rng.integers(x_edges[column], x_edges[column + 1]))
            selected.append(iy * snapshot.nx + ix)
    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.size > count:
        selected_array = rng.choice(selected_array, size=count, replace=False)
    return np.sort(selected_array)


def _initial_fields(x: np.ndarray, y: np.ndarray, amplitude: float) -> tuple[np.ndarray, ...]:
    p1 = 3.0 * x + 4.0 * y
    p2 = 5.0 * x - 2.0 * y + 0.7
    p3 = 4.0 * x + 5.0 * y + 1.3
    p4 = 6.0 * x - 3.0 * y + 2.1
    u = amplitude * (
        4.0 * np.cos(p1)
        - 2.0 * np.cos(p2)
        - 5.0 * np.sin(p3)
        + 3.0 * np.sin(p4)
    )
    v = amplitude * (
        -3.0 * np.cos(p1)
        - 5.0 * np.cos(p2)
        + 4.0 * np.sin(p3)
        + 6.0 * np.sin(p4)
    )
    omega = amplitude * (
        25.0 * np.sin(p1)
        + 29.0 * np.sin(p2)
        + 41.0 * np.cos(p3)
        + 45.0 * np.cos(p4)
    )
    return u, v, omega


def inspect_reference(config: Config) -> dict:
    files = discover_files(config.data.data_dir, config.data.file_glob)
    time_s, path = files[0]
    grid = read_structured_grid(path, config)
    fields = grid.fields
    periodic_by_time = []
    maximum_coordinate_error = grid.coordinate_max_error_m
    maximum_z_deviation = (
        float(np.max(np.abs(fields["z"] - np.mean(fields["z"]))))
        if "z" in fields
        else 0.0
    )
    for current_time, current_path in files:
        current_grid = (
            grid if current_path == path else read_structured_grid(current_path, config)
        )
        current_mismatch = max(
            max(values["x_relative_rms"], values["y_relative_rms"])
            for values in current_grid.periodic_mismatch.values()
        )
        periodic_by_time.append((current_time, current_mismatch))
        maximum_coordinate_error = max(
            maximum_coordinate_error, current_grid.coordinate_max_error_m
        )
        if "z" in current_grid.fields:
            maximum_z_deviation = max(
                maximum_z_deviation,
                float(
                    np.max(
                        np.abs(
                            current_grid.fields["z"]
                            - np.mean(current_grid.fields["z"])
                        )
                    )
                ),
            )
    worst_periodic_time, maximum_periodic_mismatch = max(
        periodic_by_time, key=lambda item: item[1]
    )
    u, v, omega = fields["u"], fields["v"], fields["omega"]
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * grid.dx)
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * grid.dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * grid.dx)
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * grid.dy)
    divergence = du_dx + dv_dy
    omega_from_uv = dv_dx - du_dy
    omega_ref = omega[1:-1, 1:-1]
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    expected_u, expected_v, expected_omega = _initial_fields(
        xx,
        yy,
        config.physics.initial_streamfunction_amplitude_m2_s,
    )
    prepared_initial_omega_relative_l2 = None
    if not config.data.data_are_filtered and config.data.apply_filter_in_memory:
        filtered_actual = filter_vorticity_snapshot(
            u,
            v,
            omega,
            grid.dx,
            grid.dy,
            config.data.filter_width_m,
        )["omega"]
        filtered_expected, _, _ = gaussian_filter_2d(
            expected_omega,
            grid.dx,
            grid.dy,
            config.data.filter_width_m,
        )
        prepared_initial_omega_relative_l2 = float(
            np.linalg.norm(filtered_actual - filtered_expected)
            / max(np.linalg.norm(filtered_expected), 1e-30)
        )

    def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
        return float(
            np.linalg.norm(prediction - reference)
            / max(np.linalg.norm(reference), 1e-30)
        )

    z_range = None
    if "z" in fields:
        z_range = [float(np.min(fields["z"])), float(np.max(fields["z"]))]
    return {
        "files": len(files),
        "times_s": [item[0] for item in files],
        "time_step_s": sorted(
            set(float(value) for value in np.round(np.diff([x[0] for x in files]), 12))
        ),
        "first_file": str(path),
        "first_file_time_s": time_s,
        "raw_rows": int(config.data.grid_points_x * config.data.grid_points_y),
        "prepared_shape": [grid.ny, grid.nx],
        "spacing_m": {"dx": grid.dx, "dy": grid.dy},
        "coordinate_max_error_m": maximum_coordinate_error,
        "z_range_m": z_range,
        "maximum_z_deviation_from_plane_m": maximum_z_deviation,
        "periodic_boundary": grid.periodic_mismatch,
        "maximum_periodic_relative_rms": maximum_periodic_mismatch,
        "worst_periodic_time_s": worst_periodic_time,
        "periodic_check_pass": bool(
            maximum_periodic_mismatch
            <= config.data.periodic_relative_rms_tolerance
        ),
        "interior_divergence": {
            "rms_per_s": float(np.sqrt(np.mean(divergence**2))),
            "max_abs_per_s": float(np.max(np.abs(divergence))),
        },
        "vorticity_consistency": {
            "relative_rmse": float(
                np.sqrt(np.mean((omega_from_uv - omega_ref) ** 2))
                / max(np.sqrt(np.mean(omega_ref**2)), 1e-30)
            ),
            "correlation": float(
                np.corrcoef(omega_from_uv.ravel(), omega_ref.ravel())[0, 1]
            ),
        },
        "initial_condition_relative_l2": {
            "u": relative_l2(u, expected_u),
            "v": relative_l2(v, expected_v),
            "omega": relative_l2(omega, expected_omega),
        },
        "prepared_initial_omega_relative_l2": (
            prepared_initial_omega_relative_l2
        ),
        "kinematic_viscosity_m2_s": config.physics.kinematic_viscosity_m2_s,
        "forcing_enabled": config.physics.forcing_enabled,
        "filter": {
            "kind": config.data.filter_kind,
            "width_m": config.data.filter_width_m,
            "data_are_filtered": config.data.data_are_filtered,
            "apply_filter_in_memory": config.data.apply_filter_in_memory,
            "sgs_supervision": (
                "computed_in_memory"
                if not config.data.data_are_filtered
                and config.data.apply_filter_in_memory
                else "from_columns"
            ),
        },
    }


def validate_for_training(config: Config, report: dict) -> None:
    problems = []
    if (
        not config.data.data_are_filtered
        and not config.data.apply_filter_in_memory
    ):
        problems.append(
            "raw STAR data requires apply_filter_in_memory=true or preprocessing"
        )
    if config.data.filter_width_m <= 0.0:
        problems.append("filter_width_m must be positive")
    if (
        not report["periodic_check_pass"]
        and not config.data.allow_nonperiodic_data
    ):
        problems.append("periodic boundary mismatch exceeds tolerance")
    if config.physics.initial_condition_kind != "multimode_decay":
        problems.append("initial_condition_kind must be 'multimode_decay'")
    if config.data.train_time_max_s <= config.data.train_time_min_s:
        problems.append("train_time_max_s must exceed train_time_min_s")
    stage_steps = (
        config.training.flow_pretrain_steps,
        config.training.sgs_pretrain_steps,
        config.training.steps,
        config.training.physics_polish_steps,
    )
    if any(steps < 0 for steps in stage_steps):
        problems.append("training stage steps cannot be negative")
    if sum(stage_steps) < 1:
        problems.append("at least one training stage must have a positive step count")
    if problems:
        raise ValueError("Training preflight failed:\n- " + "\n- ".join(problems))


def report_as_json(config: Config) -> str:
    return json.dumps(inspect_reference(config), ensure_ascii=False, indent=2)
