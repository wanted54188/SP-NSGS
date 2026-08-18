from __future__ import annotations

"""Post-training validation suite for the SP-NSGS manuscript.

This module does not train or modify the closure.  It turns one frozen
checkpoint into an auditable evidence chain for

    q_sgs = q_d + q_b,

with a trace-normalised SPD dissipative branch and a bounded backscatter
correction.  Dense STAR fields are read only after training and are used only
for evaluation/figures.  Every plotted curve is also written as CSV or NPZ.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .config import Config
from .paper import (
    _SparseVorticityNudger,
    _ablation_rollout,
    _assimilated_rollout,
    _correlation,
    _device,
    _dtype,
    _paper_style,
    _panel_label,
    _prepare_reference,
    _radial_spectrum,
    _relative_l2,
    _save_publication_figure,
)
from .solver import SpectralVorticitySolver


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _safe_relative_error(prediction: float, reference: float) -> float:
    return float(abs(prediction - reference) / max(abs(reference), 1e-30))


def _mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _maximum_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.max(array)) if array.size else float("nan")


def _spectral_band_relative_l2(
    prediction: np.ndarray,
    reference: np.ndarray,
    minimum_wave_number: int,
    maximum_wave_number: int,
) -> float:
    """Parseval-equivalent field error restricted to a radial Fourier band."""
    ny, nx = reference.shape
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    radius = np.sqrt(kkx**2 + kky**2)
    mask = (radius >= minimum_wave_number) & (radius <= maximum_wave_number)
    prediction_hat = np.fft.fft2(prediction)
    reference_hat = np.fft.fft2(reference)
    numerator = np.sum(np.abs(prediction_hat[mask] - reference_hat[mask]) ** 2)
    denominator = np.sum(np.abs(reference_hat[mask]) ** 2)
    return float(np.sqrt(numerator / max(float(denominator), 1e-30)))


def _negative_transfer_statistics(values: np.ndarray) -> dict[str, float]:
    negative = np.asarray(values, dtype=np.float64)
    negative = negative[np.isfinite(negative) & (negative < 0.0)]
    if negative.size == 0:
        return {
            "probability": 0.0,
            "conditional_mean_magnitude": 0.0,
            "fifth_percentile": 0.0,
        }
    return {
        "probability": float(negative.size / np.asarray(values).size),
        "conditional_mean_magnitude": float(np.mean(-negative)),
        "fifth_percentile": float(np.percentile(negative, 5.0)),
    }


def _state_metrics(
    solver: SpectralVorticitySolver,
    state: torch.Tensor,
    snapshot,
) -> dict[str, float]:
    fields = solver.state(state)
    omega = fields["omega"].detach().cpu().numpy()
    u = fields["u"].detach().cpu().numpy()
    v = fields["v"].detach().cpu().numpy()
    ref_omega = snapshot.omega.reshape(snapshot.ny, snapshot.nx)
    ref_u = snapshot.u.reshape(snapshot.ny, snapshot.nx)
    ref_v = snapshot.v.reshape(snapshot.ny, snapshot.nx)
    energy = float(0.5 * np.mean(u**2 + v**2))
    energy_ref = float(0.5 * np.mean(ref_u**2 + ref_v**2))
    enstrophy = float(0.5 * np.mean(omega**2))
    enstrophy_ref = float(0.5 * np.mean(ref_omega**2))
    return {
        "omega_relative_l2": _relative_l2(omega, ref_omega),
        "u_relative_l2": _relative_l2(u, ref_u),
        "v_relative_l2": _relative_l2(v, ref_v),
        "kinetic_energy": energy,
        "kinetic_energy_ref": energy_ref,
        "kinetic_energy_relative_error": _safe_relative_error(
            energy, energy_ref
        ),
        "enstrophy": enstrophy,
        "enstrophy_ref": enstrophy_ref,
        "enstrophy_relative_error": _safe_relative_error(
            enstrophy, enstrophy_ref
        ),
    }


def _test_snapshot_indices(config: Config, snapshots) -> list[int]:
    return [
        index
        for index, snapshot in enumerate(snapshots)
        if config.data.test_time_min_s - 1e-12
        <= snapshot.time_s
        <= config.data.test_time_max_s + 1e-12
    ]


def _load_solver(
    config: Config,
    checkpoint_path: Path,
) -> tuple[
    SpectralVorticitySolver,
    list,
    dict[str, torch.Tensor],
    dict[str, Any],
    torch.device,
]:
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
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    expected = int(config.paper.checkpoint_format_version)
    if checkpoint.get("version") != expected:
        raise ValueError(
            "validation checkpoint version mismatch: "
            f"{checkpoint.get('version')!r} != {expected!r}"
        )
    solver.load_state_dict(checkpoint["model_state"])
    solver.set_closure_output_scale(
        float(checkpoint.get("closure_output_scale", 1.0))
    )
    solver.closure.set_backscatter_factor(1.0)
    solver.eval()
    return solver, snapshots, reference, checkpoint, device


def _apriori_records(
    solver: SpectralVorticitySolver,
    snapshots,
    reference: dict[str, torch.Tensor],
    indices: Iterable[int],
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Evaluate the frozen closure on reference resolved states.

    This deliberately uses reference omega rather than a predicted rollout,
    so closure reconstruction error is not contaminated by accumulated flow
    forecast error.
    """
    records: list[dict[str, Any]] = []
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for index in indices:
            snapshot = snapshots[index]
            closure = solver.rhs(
                reference["omega"][index], float(snapshot.time_s)
            )
            arrays = {
                key: value.detach().cpu().numpy()
                for key, value in closure.items()
                if isinstance(value, torch.Tensor) and value.ndim >= 2
            }
            q_ref_x = snapshot.q_sgs_x.reshape(snapshot.ny, snapshot.nx)
            q_ref_y = snapshot.q_sgs_y.reshape(snapshot.ny, snapshot.nx)
            pi_ref = snapshot.pi_sgs.reshape(snapshot.ny, snapshot.nx)
            div_ref = snapshot.div_q_sgs.reshape(snapshot.ny, snapshot.nx)
            q_pred = np.stack([arrays["qx"], arrays["qy"]], axis=-1)
            q_ref = np.stack([q_ref_x, q_ref_y], axis=-1)
            trace = arrays["A11"] + arrays["A22"]
            qd_norm = np.sqrt(arrays["qd_x"] ** 2 + arrays["qd_y"] ** 2)
            qb_norm = np.sqrt(arrays["qb_x"] ** 2 + arrays["qb_y"] ** 2)
            ratio = qb_norm / np.maximum(qd_norm, 1e-12)
            active = qd_norm > max(
                float(np.sqrt(np.mean(qd_norm**2))) * 1e-4, 1e-12
            )
            ratio_active = ratio[active]
            # ``backscatter_ratio`` is the exact analytic amplitude ratio used
            # by the frozen closure: |q_b| / (beta * composite_support).  It
            # includes both the bounded gate and the transfer limiter.  In
            # contrast, |q_b| / |q_d| is retained only as a descriptive
            # diagnostic and must never be presented as the algebraic bound.
            beta = max(float(solver.closure.maximum_backscatter_ratio), 1e-12)
            bound_ratio = arrays["backscatter_ratio"] / beta
            pi_scale = max(float(np.sqrt(np.mean(pi_ref**2))), 1e-30)
            pi_d_tolerance = 1e-8 * pi_scale
            reference_negative = _negative_transfer_statistics(pi_ref)
            model_negative = _negative_transfer_statistics(arrays["pi"])
            resolved_cutoff = max(1, min(snapshot.nx, snapshot.ny) // 3)
            high_start = max(2, int(math.ceil(0.60 * resolved_cutoff)))
            record = {
                "time_s": float(snapshot.time_s),
                "q_ref_x": q_ref_x,
                "q_ref_y": q_ref_y,
                "pi_ref": pi_ref,
                "div_ref": div_ref,
                "ratio_active": ratio_active,
                **arrays,
            }
            records.append(record)
            rows.append(
                {
                    "time_s": float(snapshot.time_s),
                    "nu_d_minimum": float(np.min(arrays["nu_d"])),
                    "nu_d_p05": _finite_percentile(arrays["nu_d"], 5.0),
                    "nu_d_median": _finite_percentile(arrays["nu_d"], 50.0),
                    "nu_d_p95": _finite_percentile(arrays["nu_d"], 95.0),
                    "A_eigenvalue_minimum": float(
                        np.min(arrays["A_eigenvalue_min"])
                    ),
                    "A_eigenvalue_maximum": float(
                        np.max(arrays["A_eigenvalue_max"])
                    ),
                    "A_trace_mean": float(np.mean(trace)),
                    "A_trace_maximum_absolute_error": float(
                        np.max(np.abs(trace - 2.0))
                    ),
                    "pi_d_minimum": float(np.min(arrays["pi_d"])),
                    "pi_d_negative_fraction_beyond_tolerance": float(
                        np.mean(arrays["pi_d"] < -pi_d_tolerance)
                    ),
                    "pi_reference_backscatter_probability": float(
                        np.mean(pi_ref < 0.0)
                    ),
                    "pi_model_backscatter_probability": float(
                        np.mean(arrays["pi"] < 0.0)
                    ),
                    "qb_over_qd_median": _finite_percentile(
                        ratio_active, 50.0
                    ),
                    "qb_over_qd_p95": _finite_percentile(
                        ratio_active, 95.0
                    ),
                    "backscatter_bound_ratio_median": _finite_percentile(
                        bound_ratio, 50.0
                    ),
                    "backscatter_bound_ratio_p95": _finite_percentile(
                        bound_ratio, 95.0
                    ),
                    "backscatter_bound_ratio_maximum": float(np.max(bound_ratio)),
                    "backscatter_bound_violation_fraction": float(
                        np.mean(bound_ratio > 1.0 + 1e-10)
                    ),
                    "q_sgs_relative_l2": _relative_l2(q_pred, q_ref),
                    "q_sgs_correlation": _correlation(q_pred, q_ref),
                    "q_sgs_x_relative_l2": _relative_l2(
                        arrays["qx"], q_ref_x
                    ),
                    "q_sgs_x_correlation": _correlation(
                        arrays["qx"], q_ref_x
                    ),
                    "q_sgs_y_relative_l2": _relative_l2(
                        arrays["qy"], q_ref_y
                    ),
                    "q_sgs_y_correlation": _correlation(
                        arrays["qy"], q_ref_y
                    ),
                    "pi_relative_l2": _relative_l2(arrays["pi"], pi_ref),
                    "pi_correlation": _correlation(arrays["pi"], pi_ref),
                    "pi_reference_negative_conditional_mean_magnitude": (
                        reference_negative["conditional_mean_magnitude"]
                    ),
                    "pi_model_negative_conditional_mean_magnitude": (
                        model_negative["conditional_mean_magnitude"]
                    ),
                    "pi_reference_negative_fifth_percentile": (
                        reference_negative["fifth_percentile"]
                    ),
                    "pi_model_negative_fifth_percentile": (
                        model_negative["fifth_percentile"]
                    ),
                    "div_q_relative_l2": _relative_l2(
                        arrays["div_q"], div_ref
                    ),
                    "div_q_correlation": _correlation(
                        arrays["div_q"], div_ref
                    ),
                    "div_q_low_k_relative_l2": _spectral_band_relative_l2(
                        arrays["div_q"], div_ref, 1, 10
                    ),
                    "div_q_mid_k_relative_l2": _spectral_band_relative_l2(
                        arrays["div_q"], div_ref, 11, high_start - 1
                    ),
                    "div_q_high_k_relative_l2": _spectral_band_relative_l2(
                        arrays["div_q"], div_ref, high_start, resolved_cutoff
                    ),
                }
            )
    return records, rows


def _plot_figure_1(output: Path) -> dict[str, Any]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    metadata = {
        "identity": "q_sgs = q_d + q_b",
        "dissipative_branch": "q_d = -nu_d A_theta grad(omega)",
        "constraints": [
            "nu_d > 0",
            "A_theta is SPD",
            "trace(A_theta) = 2",
            "Pi_d = -q_d dot grad(omega) >= 0",
        ],
        "backscatter_branch": (
            "q_b = beta q_star tanh(s_theta) n_theta, "
            "|q_b| <= beta q_star"
        ),
        "note": (
            "The Clark-gradient field supplies only q_star/direction support; "
            "it is not an additive third closure branch."
        ),
    }
    with mpl.rc_context(_paper_style()):
        figure, axis = plt.subplots(figsize=(7.15, 3.35))
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 6)
        axis.axis("off")

        def box(x, y, width, height, text, color):
            patch = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.08,rounding_size=0.08",
                facecolor=color, edgecolor="0.25", linewidth=1.0,
            )
            axis.add_patch(patch)
            axis.text(x + width / 2, y + height / 2, text,
                      ha="center", va="center")

        def arrow(x0, y0, x1, y1):
            axis.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                linewidth=1.0, color="0.25",
            ))

        box(0.2, 2.25, 2.05, 1.5,
            r"resolved fields" "\n" r"$\omega,\,\nabla\omega,\,\nabla\mathbf{u}$",
            "#e8eef6")
        box(3.0, 3.45, 3.15, 1.75,
            r"dissipative branch" "\n" r"$\mathbf{q}_d=-\nu_d\mathbf{A}_\theta\nabla\omega$" "\n" r"$\nu_d>0,\ \mathbf{A}_\theta\succ0,\ \mathrm{tr}\mathbf{A}_\theta=2$",
            "#dff1df")
        box(3.0, 0.80, 3.15, 1.75,
            r"bounded backscatter branch" "\n" r"$\mathbf{q}_b=\beta q_*\tanh(s_\theta)\mathbf{n}_\theta$" "\n" r"$|\mathbf{q}_b|\leq\beta q_*$",
            "#f7e5d5")
        box(7.05, 2.25, 2.00, 1.50,
            r"additive closure" "\n" r"$\mathbf{q}_{sgs}=\mathbf{q}_d+\mathbf{q}_b$",
            "#eee5f5")
        box(9.85, 2.25, 1.90, 1.50,
            r"filtered equation" "\n" r"$-\nabla\!\cdot\mathbf{q}_{sgs}$",
            "#e8eef6")
        arrow(2.25, 3.0, 3.0, 4.3)
        arrow(2.25, 3.0, 3.0, 1.67)
        arrow(6.15, 4.3, 7.05, 3.35)
        arrow(6.15, 1.67, 7.05, 2.65)
        arrow(9.05, 3.0, 9.85, 3.0)
        axis.text(4.58, 5.55, r"$\Pi_d=-\mathbf{q}_d\!\cdot\nabla\omega\geq0$",
                  ha="center", color="#1b6e2b")
        axis.text(4.58, 0.25,
                  r"local backscatter retained, amplitude and transfer bounded",
                  ha="center", color="#a65014")
        axis.text(8.05, 1.35,
                  r"only two additive branches",
                  ha="center", fontweight="bold")
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_1_model_architecture")
        plt.close(figure)
    return metadata


def _plot_figure_2(
    output: Path, rows: list[dict[str, float]]
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    t = np.asarray([row["time_s"] for row in rows])
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.1))
        axes[0, 0].semilogy(
            t, np.maximum([row["nu_d_minimum"] for row in rows], 1e-18),
            label="minimum",
        )
        axes[0, 0].semilogy(
            t, np.maximum([row["nu_d_median"] for row in rows], 1e-18),
            label="median",
        )
        axes[0, 0].semilogy(
            t, np.maximum([row["nu_d_p95"] for row in rows], 1e-18),
            label="95th percentile",
        )
        axes[0, 0].set(xlabel=r"$t$ (s)", ylabel=r"$\nu_d$ ($\mathrm{m^2s^{-1}}$)")
        axes[0, 0].legend(frameon=False)

        axes[0, 1].plot(
            t, [row["A_eigenvalue_minimum"] for row in rows],
            label=r"$\lambda_{min}$",
        )
        axes[0, 1].plot(
            t, [row["A_eigenvalue_maximum"] for row in rows],
            label=r"$\lambda_{max}$",
        )
        axes[0, 1].axhline(0.0, color="0.35", linestyle=":")
        axes[0, 1].set(xlabel=r"$t$ (s)", ylabel=r"eigenvalues of $\mathbf{A}_\theta$")
        axes[0, 1].legend(frameon=False)

        axes[1, 0].semilogy(
            t,
            np.maximum(
                [row["A_trace_maximum_absolute_error"] for row in rows],
                1e-18,
            ),
        )
        axes[1, 0].set(
            xlabel=r"$t$ (s)",
            ylabel=r"$\max|\mathrm{tr}(\mathbf{A}_\theta)-2|$",
        )

        axes[1, 1].plot(
            t, [row["pi_d_minimum"] for row in rows], label=r"$\min\Pi_d$"
        )
        axes[1, 1].plot(
            t,
            [row["pi_d_negative_fraction_beyond_tolerance"] for row in rows],
            "--",
            label="negative fraction",
        )
        axes[1, 1].axhline(0.0, color="0.35", linestyle=":")
        axes[1, 1].set(xlabel=r"$t$ (s)", ylabel="dissipative-branch audit")
        axes[1, 1].legend(frameon=False)
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
            _panel_label(axis, label)
            axis.grid(True, which="both", alpha=0.18)
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_2_dissipative_branch")
        plt.close(figure)


def _plot_figure_3(
    output: Path,
    records: list[dict[str, Any]],
    rows: list[dict[str, float]],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    record = records[-1]
    pi_ref = record["pi_ref"]
    pi_model = record["pi"]
    scale = max(float(np.sqrt(np.mean(pi_ref**2))), 1e-30)
    limit = _finite_percentile(
        np.abs(np.concatenate([pi_ref.ravel(), pi_model.ravel()])), 99.5
    )
    bins = np.linspace(-5.0, 5.0, 121)
    ratios = np.concatenate(
        [np.asarray(item["ratio_active"]).ravel() for item in records]
    )
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    ratio_upper = max(_finite_percentile(ratios, 99.5), 1e-3)
    ratio_bins = np.logspace(-4, math.log10(ratio_upper), 80)
    t = np.asarray([row["time_s"] for row in rows])

    np.savez_compressed(
        output.parent / "data" / "figure_3_backscatter_fields.npz",
        time_s=np.asarray(record["time_s"]),
        pi_reference=pi_ref,
        pi_model=pi_model,
        pi_d=record["pi_d"],
        pi_b=record["pi_b"],
        qb_over_qd=ratios,
    )
    with mpl.rc_context(_paper_style()):
        figure = plt.figure(figsize=(7.15, 4.75))
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.9))
        ax_ref = figure.add_subplot(grid[0, 0])
        ax_model = figure.add_subplot(grid[0, 1])
        ax_pdf = figure.add_subplot(grid[0, 2])
        ax_prob = figure.add_subplot(grid[1, :2])
        ax_ratio = figure.add_subplot(grid[1, 2])
        image = ax_ref.imshow(pi_ref, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax_model.imshow(pi_model, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax_ref.set_title("STAR-derived reference")
        ax_model.set_title("SP-NSGS")
        for axis in (ax_ref, ax_model):
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=[ax_ref, ax_model], fraction=0.035, pad=0.03,
                        label=r"$\Pi_\omega$ ($\mathrm{s^{-3}}$)")

        ax_pdf.hist(pi_ref.ravel() / scale, bins=bins, density=True,
                    histtype="step", label="reference")
        ax_pdf.hist(pi_model.ravel() / scale, bins=bins, density=True,
                    histtype="step", label="SP-NSGS")
        ax_pdf.set(xlabel=r"$\Pi_\omega/\Pi_{ref,rms}$", ylabel="PDF")
        ax_pdf.set_yscale("log")
        ax_pdf.legend(frameon=False)

        ax_prob.plot(
            t, [row["pi_reference_backscatter_probability"] for row in rows],
            label="reference",
        )
        ax_prob.plot(
            t, [row["pi_model_backscatter_probability"] for row in rows],
            "--", label="SP-NSGS",
        )
        ax_prob.set(xlabel=r"$t$ (s)", ylabel=r"$P(\Pi_\omega<0)$")
        ax_prob.legend(frameon=False)
        ax_prob.grid(True, alpha=0.18)

        if ratios.size:
            ax_ratio.hist(ratios, bins=ratio_bins, density=True, histtype="step")
            ax_ratio.set_xscale("log")
        ax_ratio.set(xlabel=r"$|\mathbf{q}_b|/|\mathbf{q}_d|$", ylabel="PDF")
        ax_ratio.grid(True, which="both", alpha=0.18)
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)", "(e)"),
                               (ax_ref, ax_model, ax_pdf, ax_prob, ax_ratio)):
            _panel_label(axis, label)
        figure.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.93,
                               wspace=0.38, hspace=0.42)
        _save_publication_figure(figure, output / "figure_3_backscatter_validation")
        plt.close(figure)


def _plot_figure_4(
    output: Path,
    records: list[dict[str, Any]],
    rows: list[dict[str, float]],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    record = records[-1]
    components = (
        ("q_ref_x", "qx", r"$q_{sgs,x}$"),
        ("q_ref_y", "qy", r"$q_{sgs,y}$"),
        ("div_ref", "div_q", r"$\nabla\!\cdot\mathbf{q}_{sgs}$"),
    )
    np.savez_compressed(
        output.parent / "data" / "figure_4_apriori_flux_fields.npz",
        time_s=np.asarray(record["time_s"]),
        qx_reference=record["q_ref_x"], qx_model=record["qx"],
        qy_reference=record["q_ref_y"], qy_model=record["qy"],
        div_q_reference=record["div_ref"], div_q_model=record["div_q"],
    )
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(3, 3, figsize=(7.15, 6.2))
        for row_index, (ref_key, pred_key, symbol) in enumerate(components):
            ref = record[ref_key]
            pred = record[pred_key]
            limit = _finite_percentile(
                np.abs(np.concatenate([ref.ravel(), pred.ravel()])), 99.5
            )
            rms = max(float(np.sqrt(np.mean(ref**2))), 1e-30)
            residual = (pred - ref) / rms
            residual_limit = max(_finite_percentile(np.abs(residual), 99.0), 1e-6)
            image = axes[row_index, 0].imshow(
                ref, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[row_index, 1].imshow(
                pred, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
            )
            error_image = axes[row_index, 2].imshow(
                residual, origin="lower", cmap="RdBu_r",
                vmin=-residual_limit, vmax=residual_limit,
            )
            axes[row_index, 0].set_ylabel(symbol)
            figure.colorbar(image, ax=axes[row_index, :2], fraction=0.025, pad=0.02)
            figure.colorbar(error_image, ax=axes[row_index, 2], fraction=0.046, pad=0.03)
            axes[row_index, 2].text(
                0.03, 0.04,
                rf"$\epsilon_{{L^2}}={100*_relative_l2(pred, ref):.1f}\%$" "\n"
                rf"$r={_correlation(pred, ref):.2f}$",
                transform=axes[row_index, 2].transAxes,
                color="white",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.5},
            )
        for column, title in enumerate(("reference", "SP-NSGS", "normalised residual")):
            axes[0, column].set_title(title)
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        _panel_label(axes[0, 0], "(a)")
        _panel_label(axes[1, 0], "(b)")
        _panel_label(axes[2, 0], "(c)")
        figure.suptitle(rf"A priori SGS flux reconstruction at $t={record['time_s']:g}$ s")
        figure.subplots_adjust(left=0.07, right=0.98, bottom=0.04, top=0.91,
                               wspace=0.28, hspace=0.10)
        _save_publication_figure(figure, output / "figure_4_apriori_flux_reconstruction")
        plt.close(figure)


def _rollout_modes(
    solver: SpectralVorticitySolver,
    reference: dict[str, torch.Tensor],
    train_indices: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    original_scale = float(solver.closure_output_scale)
    original_factor = float(solver.closure.backscatter_factor)
    nudger = _SparseVorticityNudger(solver, train_indices)
    result: dict[str, list[torch.Tensor]] = {}
    try:
        with torch.no_grad():
            for name, scale, factor in (
                ("no SGS", 0.0, 1.0),
                ("dissipative only", original_scale, 0.0),
                ("SP-NSGS", original_scale, 1.0),
            ):
                solver.set_closure_output_scale(scale)
                solver.closure.set_backscatter_factor(factor)
                states, _, _ = _ablation_rollout(
                    solver, reference["omega"][0], reference["time"],
                    reference, nudger,
                )
                result[name] = states
    finally:
        solver.set_closure_output_scale(original_scale)
        solver.closure.set_backscatter_factor(original_factor)
    return result


def _comparison_rows(
    solver: SpectralVorticitySolver,
    snapshots,
    modes: dict[str, list[torch.Tensor]],
    indices: Iterable[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, states in modes.items():
        for index in indices:
            if index >= len(states):
                continue
            rows.append(
                {
                    "model": model,
                    "time_s": float(snapshots[index].time_s),
                    **_state_metrics(solver, states[index], snapshots[index]),
                }
            )
    return rows


def _plot_figure_5(
    output: Path,
    solver: SpectralVorticitySolver,
    snapshots,
    modes: dict[str, list[torch.Tensor]],
    rows: list[dict[str, Any]],
    final_index: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    names = ("no SGS", "dissipative only", "SP-NSGS")
    reference = snapshots[final_index].omega.reshape(
        snapshots[final_index].ny, snapshots[final_index].nx
    )
    limit = float(np.max(np.abs(reference)))
    saved_fields: dict[str, np.ndarray] = {
        "time_s": np.asarray(float(snapshots[final_index].time_s)),
        "omega_reference": reference,
    }
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.7))
        for column, name in enumerate(names):
            if final_index < len(modes[name]):
                omega = solver.state(modes[name][final_index])["omega"].detach().cpu().numpy()
                saved_fields[
                    "omega_" + name.lower().replace("-", "_").replace(" ", "_")
                ] = omega
                axes[0, column].imshow(
                    omega, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit
                )
            axes[0, column].set_title(name)
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
        curve_specs = (
            ("omega_relative_l2", r"$\epsilon_\omega$"),
            ("kinetic_energy_relative_error", r"$|\Delta K|/K_{ref}$"),
            ("enstrophy_relative_error", r"$|\Delta Z|/Z_{ref}$"),
        )
        styles = {
            "no SGS": (":", "0.35"),
            "dissipative only": ("--", "C1"),
            "SP-NSGS": ("-", "C0"),
        }
        for axis, (key, ylabel) in zip(axes[1], curve_specs):
            for name in names:
                selected = [row for row in rows if row["model"] == name]
                axis.plot(
                    [row["time_s"] for row in selected],
                    [row[key] for row in selected],
                    linestyle=styles[name][0], color=styles[name][1], label=name,
                )
            axis.set(xlabel=r"$t$ (s)", ylabel=ylabel)
            axis.grid(True, alpha=0.18)
        axes[1, 0].legend(frameon=False)
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"), axes.flat):
            _panel_label(axis, label)
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_5_model_comparison")
        plt.close(figure)
    np.savez_compressed(
        output.parent / "data" / "figure_5_model_comparison_fields.npz",
        **saved_fields,
    )


def _posterior_rows(
    solver: SpectralVorticitySolver,
    snapshots,
    states: list[torch.Tensor],
    indices: list[int],
) -> list[dict[str, Any]]:
    """Numerical a-posteriori products, independent of any plot layout."""

    return [
        {"time_s": float(snapshots[index].time_s),
         **_state_metrics(solver, states[index], snapshots[index])}
        for index in indices if index < len(states)
    ]


def _snapshot_fields(
    solver: SpectralVorticitySolver,
    snapshots,
    states: list[torch.Tensor],
    evaluation_indices: list[int],
) -> dict[str, np.ndarray]:
    """Extract exactly three frozen snapshot pairs for downstream plotting."""

    chosen = [index for index in evaluation_indices if index < len(states)]
    if not chosen:
        available = np.linspace(
            0, max(len(states) - 1, 0), num=min(3, len(states)), dtype=int
        )
        chosen = sorted(set(int(value) for value in available))
    if not chosen:
        raise FloatingPointError("a posteriori rollout produced no finite state")
    if len(chosen) > 3:
        chosen = [chosen[0], chosen[len(chosen) // 2], chosen[-1]]
    while len(chosen) < 3:
        chosen.append(chosen[-1])
    fields: dict[str, np.ndarray] = {}
    for index in chosen:
        time_value = snapshots[index].time_s
        fields[f"omega_reference_t{time_value:g}"] = snapshots[index].omega.reshape(
            snapshots[index].ny, snapshots[index].nx
        )
        fields[f"omega_model_t{time_value:g}"] = solver.state(
            states[index]
        )["omega"].detach().cpu().numpy()
    return fields


def _write_snapshot_npz(
    output_path: Path,
    solver: SpectralVorticitySolver,
    snapshots,
    states: list[torch.Tensor],
    evaluation_indices: list[int],
) -> None:
    np.savez_compressed(
        output_path,
        **_snapshot_fields(solver, snapshots, states, evaluation_indices),
    )


def _write_apriori_field_sources(
    data_dir: Path,
    records: list[dict[str, Any]],
    maximum_backscatter_ratio: float,
) -> None:
    """Write frozen a-priori arrays once, independently of plot routines."""

    if not records:
        raise ValueError("cannot write a-priori field sources without records")
    record = records[-1]
    beta = max(float(maximum_backscatter_ratio), 1e-12)
    bound_ratio = np.asarray(record["backscatter_ratio"], dtype=float) / beta
    np.savez_compressed(
        data_dir / "figure_3_backscatter_fields.npz",
        time_s=np.asarray(record["time_s"]),
        pi_reference=record["pi_ref"],
        pi_model=record["pi"],
        pi_d=record["pi_d"],
        pi_b=record["pi_b"],
        # Descriptive only: can be large wherever q_d is small.
        qb_over_qd=np.asarray(record["ratio_active"], dtype=float),
        # Exact normalized frozen-model amplitude-bound audit, R_b <= 1.
        backscatter_bound_ratio=bound_ratio,
    )
    np.savez_compressed(
        data_dir / "figure_4_apriori_flux_fields.npz",
        time_s=np.asarray(record["time_s"]),
        qx_reference=record["q_ref_x"], qx_model=record["qx"],
        qy_reference=record["q_ref_y"], qy_model=record["qy"],
        div_q_reference=record["div_ref"], div_q_model=record["div_q"],
    )


def _write_model_comparison_fields(
    data_dir: Path,
    solver: SpectralVorticitySolver,
    snapshots,
    modes: dict[str, list[torch.Tensor]],
    final_index: int,
) -> None:
    """Write Fig. 5 snapshot sources without rendering a legacy figure."""

    reference = snapshots[final_index].omega.reshape(
        snapshots[final_index].ny, snapshots[final_index].nx
    )
    fields: dict[str, np.ndarray] = {
        "time_s": np.asarray(float(snapshots[final_index].time_s)),
        "omega_reference": reference,
    }
    for name in ("no SGS", "dissipative only", "SP-NSGS"):
        if final_index < len(modes[name]):
            key = "omega_" + name.lower().replace("-", "_").replace(" ", "_")
            fields[key] = solver.state(modes[name][final_index])["omega"].detach().cpu().numpy()
    np.savez_compressed(data_dir / "figure_5_model_comparison_fields.npz", **fields)


def _plot_figure_6(
    output: Path,
    solver: SpectralVorticitySolver,
    snapshots,
    states: list[torch.Tensor],
    indices: list[int],
    evaluation_indices: list[int],
) -> list[dict[str, Any]]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rows = _posterior_rows(solver, snapshots, states, indices)
    fields = _snapshot_fields(solver, snapshots, states, evaluation_indices)
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(3, 3, figsize=(7.15, 6.2))
        times = [row["time_s"] for row in rows]
        axes[0, 0].plot(times, [row["kinetic_energy_ref"] for row in rows], label="reference")
        axes[0, 0].plot(times, [row["kinetic_energy"] for row in rows], "--", label="SP-NSGS")
        axes[0, 0].set(xlabel=r"$t$ (s)", ylabel="kinetic energy")
        axes[0, 1].plot(times, [row["enstrophy_ref"] for row in rows], label="reference")
        axes[0, 1].plot(times, [row["enstrophy"] for row in rows], "--", label="SP-NSGS")
        axes[0, 1].set(xlabel=r"$t$ (s)", ylabel="enstrophy")
        axes[0, 2].plot(times, [row["omega_relative_l2"] for row in rows])
        axes[0, 2].set(xlabel=r"$t$ (s)", ylabel=r"$\epsilon_\omega$")
        axes[0, 0].legend(frameon=False)
        axes[0, 1].legend(frameon=False)
        for axis in axes[0]:
            axis.grid(True, alpha=0.18)
        snapshot_pairs = []
        for time_value in sorted(
            float(key.rsplit("t", 1)[-1])
            for key in fields if key.startswith("omega_reference_t")
        ):
            reference_key = f"omega_reference_t{time_value:g}"
            model_key = f"omega_model_t{time_value:g}"
            snapshot_pairs.append((time_value, fields[reference_key], fields[model_key]))
        max_limit = max(float(np.max(np.abs(value))) for _, value, _ in snapshot_pairs)
        for column, (time_value, ref, pred) in enumerate(snapshot_pairs):
            axes[1, column].imshow(ref, origin="lower", cmap="coolwarm", vmin=-max_limit, vmax=max_limit)
            axes[2, column].imshow(pred, origin="lower", cmap="coolwarm", vmin=-max_limit, vmax=max_limit)
            axes[1, column].set_title(rf"$t={time_value:g}$ s")
            for axis in axes[1:, column]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes[1, 0].set_ylabel("STAR reference")
        axes[2, 0].set_ylabel("SP-NSGS\n(4% assimilated)")
        for label, axis in zip(("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"), axes.flat):
            _panel_label(axis, label)
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_6_aposteriori_validation")
        plt.close(figure)
    np.savez_compressed(output.parent / "data" / "figure_6_snapshots.npz", **fields)
    return rows


def _nested_sensor_subset(
    train_indices: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    if count >= train_indices.numel():
        return train_indices.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(train_indices.numel(), generator=generator)
    selected = train_indices.detach().cpu()[order[:count]]
    return torch.sort(selected).values.to(device=train_indices.device)


def _sparse_robustness(
    config: Config,
    solver: SpectralVorticitySolver,
    snapshots,
    reference: dict[str, torch.Tensor],
    train_indices: torch.Tensor,
    autonomous_states: list[torch.Tensor],
    primary_states: list[torch.Tensor],
    indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fractions = sorted(set(float(value) for value in config.paper.sparse_robustness_sensor_fractions))
    for fraction in fractions:
        if fraction < 0.0 or fraction > config.paper.train_sensor_fraction + 1e-12:
            raise ValueError(
                "sparse robustness fractions must lie between zero and the "
                "training sensor fraction"
            )
        for repeat in range(int(config.paper.sparse_robustness_repeats)):
            if fraction <= 0.0:
                states = autonomous_states
                count = 0
            elif (
                abs(fraction - config.paper.train_sensor_fraction) <= 1e-12
                and repeat == 0
            ):
                states = primary_states
                count = int(train_indices.numel())
            else:
                count = max(1, int(round(fraction * solver.nx * solver.ny)))
                count = min(count, train_indices.numel())
                subset = _nested_sensor_subset(
                    train_indices, count, config.training.seed + 1009 * repeat
                )
                nudger = _SparseVorticityNudger(solver, subset)
                with torch.no_grad():
                    states = _assimilated_rollout(
                        solver, reference["omega"][0], reference["time"],
                        reference, nudger,
                    )
            metrics = [
                _state_metrics(solver, states[index], snapshots[index])
                for index in indices if index < len(states)
            ]
            rows.append(
                {
                    "sensor_fraction": fraction,
                    "sensor_percent": 100.0 * fraction,
                    "sensor_count": count,
                    "repeat": repeat,
                    "omega_relative_l2_mean": _mean_or_nan(m["omega_relative_l2"] for m in metrics),
                    "omega_relative_l2_maximum": _maximum_or_nan(m["omega_relative_l2"] for m in metrics),
                    "kinetic_energy_relative_error_mean": _mean_or_nan(m["kinetic_energy_relative_error"] for m in metrics),
                    "enstrophy_relative_error_mean": _mean_or_nan(m["enstrophy_relative_error"] for m in metrics),
                }
            )
            print(
                "  Fig.7 sparse audit: "
                f"{100*fraction:g}% ({count} sensors), repeat {repeat + 1}"
            )
    return rows


def _plot_figure_7(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fractions = sorted(set(row["sensor_percent"] for row in rows))
    keys = (
        ("omega_relative_l2_mean", r"mean $\epsilon_\omega$"),
        ("kinetic_energy_relative_error_mean", r"mean $|\Delta K|/K_{ref}$"),
        ("enstrophy_relative_error_mean", r"mean $|\Delta Z|/Z_{ref}$"),
    )
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
        for axis, (key, ylabel) in zip(axes, keys):
            means, deviations = [], []
            for fraction in fractions:
                values = np.asarray(
                    [row[key] for row in rows if row["sensor_percent"] == fraction]
                )
                means.append(float(np.mean(values)))
                deviations.append(float(np.std(values)))
            axis.errorbar(fractions, means, yerr=deviations, marker="o", capsize=2)
            axis.set(xlabel="assimilated sensor fraction (%)", ylabel=ylabel)
            axis.grid(True, alpha=0.18)
        for label, axis in zip(("(a)", "(b)", "(c)"), axes):
            _panel_label(axis, label)
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_7_sparse_data_robustness")
        plt.close(figure)


def _ablation_suite(
    solver: SpectralVorticitySolver,
    snapshots,
    reference: dict[str, torch.Tensor],
    train_indices: torch.Tensor,
    indices: list[int],
    cached_modes: dict[str, list[torch.Tensor]] | None = None,
) -> list[dict[str, Any]]:
    original_scale = float(solver.closure_output_scale)
    original_factor = float(solver.closure.backscatter_factor)
    gradient_raw = solver.closure.gradient_support_coefficient_raw.detach().clone()
    nudger = _SparseVorticityNudger(solver, train_indices)
    modes = (
        ("no SGS", 0.0, 1.0, False),
        ("dissipative only", original_scale, 0.0, False),
        ("bounded qb without qG support", original_scale, 1.0, True),
        ("full SP-NSGS", original_scale, 1.0, False),
    )
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for name, scale, factor, remove_gradient_support in modes:
                solver.set_closure_output_scale(scale)
                solver.closure.set_backscatter_factor(factor)
                solver.closure.gradient_support_coefficient_raw.copy_(
                    torch.full_like(gradient_raw, -30.0)
                    if remove_gradient_support else gradient_raw
                )
                cached_key = "SP-NSGS" if name == "full SP-NSGS" else name
                if cached_modes is not None and cached_key in cached_modes:
                    states = cached_modes[cached_key]
                    stable_horizon = float(
                        reference["time"][len(states) - 1].item()
                    )
                    failure_time = None
                else:
                    states, stable_horizon, failure_time = _ablation_rollout(
                        solver, reference["omega"][0], reference["time"],
                        reference, nudger,
                    )
                flow_metrics = [
                    _state_metrics(solver, states[index], snapshots[index])
                    for index in indices if index < len(states)
                ]
                div_errors, div_correlations, pi_errors, pi_correlations = [], [], [], []
                for index in indices:
                    closure = solver.rhs(reference["omega"][index], snapshots[index].time_s)
                    div_pred = closure["div_q"].detach().cpu().numpy()
                    pi_pred = closure["pi"].detach().cpu().numpy()
                    div_ref = snapshots[index].div_q_sgs.reshape(solver.ny, solver.nx)
                    pi_ref = snapshots[index].pi_sgs.reshape(solver.ny, solver.nx)
                    div_errors.append(_relative_l2(div_pred, div_ref))
                    div_correlations.append(_correlation(div_pred, div_ref))
                    pi_errors.append(_relative_l2(pi_pred, pi_ref))
                    pi_correlations.append(_correlation(pi_pred, pi_ref))
                rows.append(
                    {
                        "ablation": name,
                        "omega_relative_l2_mean": _mean_or_nan(m["omega_relative_l2"] for m in flow_metrics),
                        "kinetic_energy_relative_error_mean": _mean_or_nan(m["kinetic_energy_relative_error"] for m in flow_metrics),
                        "enstrophy_relative_error_mean": _mean_or_nan(m["enstrophy_relative_error"] for m in flow_metrics),
                        "div_q_relative_l2_mean": float(np.mean(div_errors)),
                        "div_q_correlation_mean": _mean_or_nan(div_correlations),
                        "pi_relative_l2_mean": float(np.mean(pi_errors)),
                        "pi_correlation_mean": _mean_or_nan(pi_correlations),
                        "stable_horizon_s": stable_horizon,
                        "failure_time_s": failure_time,
                        "protocol": "post-training switch-off; identical 4% assimilation",
                    }
                )
                print(f"  Fig.8 ablation complete: {name}")
    finally:
        solver.set_closure_output_scale(original_scale)
        solver.closure.set_backscatter_factor(original_factor)
        with torch.no_grad():
            solver.closure.gradient_support_coefficient_raw.copy_(gradient_raw)
    return rows


def _plot_figure_8(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    labels = [row["ablation"] for row in rows]
    x = np.arange(len(labels))
    panels = (
        ("omega_relative_l2_mean", r"mean $\epsilon_\omega$", False),
        ("div_q_relative_l2_mean", r"mean $\epsilon_{\nabla\cdot q}$", False),
        ("pi_correlation_mean", r"mean $r(\Pi_\omega)$", True),
    )
    with mpl.rc_context(_paper_style()):
        figure, axes = plt.subplots(1, 3, figsize=(7.15, 3.1))
        colors = ["0.55", "C1", "C2", "C0"]
        for axis, (key, ylabel, correlation) in zip(axes, panels):
            axis.bar(x, [row[key] for row in rows], color=colors)
            axis.set_xticks(x, labels, rotation=28, ha="right")
            axis.set_ylabel(ylabel)
            if correlation:
                axis.set_ylim(0.0, 1.0)
            axis.grid(True, axis="y", alpha=0.18)
        for label, axis in zip(("(a)", "(b)", "(c)"), axes):
            _panel_label(axis, label)
        figure.tight_layout()
        _save_publication_figure(figure, output / "figure_8_ablation_study")
        plt.close(figure)


def generate_validation_suite(
    config: Config,
    checkpoint_path: str | Path,
) -> Path:
    """Generate numerical validation products from a selected checkpoint.

    Deliberately no publication PNG/PDF is written here.  CSV/NPZ outputs are
    kept separate from the solver so they can be post-processed by an external
    presentation workflow without changing the numerical evaluation.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_root = Path(config.training.output_dir) / "predictions" / "validation_suite"
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Generating manuscript validation suite (model remains frozen)...")
    solver, snapshots, reference, checkpoint, device = _load_solver(
        config, checkpoint_path
    )
    train_indices = checkpoint["train_indices"].to(device=device, dtype=torch.long)
    test_indices = _test_snapshot_indices(config, snapshots)
    evaluation_times = np.asarray(config.paper.evaluation_times_s)
    evaluation_indices = [
        index for index in test_indices
        if np.any(np.isclose(snapshots[index].time_s, evaluation_times, atol=1e-9))
    ]
    if not evaluation_indices:
        raise ValueError("no configured evaluation time exists in the STAR snapshots")

    print("  Frozen a priori branch/closure source data (Figs.2--4)")
    apriori_records, apriori_rows = _apriori_records(
        solver, snapshots, reference, test_indices
    )
    _write_csv(data_dir / "figure_2_dissipative_branch.csv", apriori_rows)
    _write_csv(data_dir / "figure_3_backscatter_statistics.csv", apriori_rows)
    _write_csv(data_dir / "figure_4_apriori_flux_metrics.csv", apriori_rows)
    _write_apriori_field_sources(
        data_dir,
        apriori_records,
        solver.closure.maximum_backscatter_ratio,
    )

    print("  Frozen controlled baseline source data (Fig.5)")
    modes = _rollout_modes(solver, reference, train_indices)
    comparison_rows = _comparison_rows(
        solver, snapshots, modes, test_indices
    )
    _write_csv(data_dir / "figure_5_model_comparison.csv", comparison_rows)
    _write_model_comparison_fields(
        data_dir, solver, snapshots, modes, max(test_indices)
    )

    print("  Frozen a posteriori source data (Fig.6)")
    full_states = modes["SP-NSGS"]
    posterior_rows = _posterior_rows(
        solver, snapshots, full_states, test_indices
    )
    _write_csv(data_dir / "figure_6_aposteriori_metrics.csv", posterior_rows)
    _write_snapshot_npz(
        data_dir / "figure_6_snapshots.npz",
        solver,
        snapshots,
        full_states,
        evaluation_indices,
    )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_format_version": int(config.paper.checkpoint_format_version),
        "checkpoint_validation_metrics": checkpoint.get(
            "validation_metrics", {}
        ),
        "checkpoint_selection_uses_dense_test_fields": bool(
            checkpoint.get("checkpoint_selection_uses_dense_test_fields", False)
        ),
        "model_frozen_during_suite": True,
        "model_identity": "q_sgs = q_d + q_b",
        "dissipative_constraints": (
            "nu_d > 0; A_theta SPD; trace(A_theta)=2; Pi_d >= 0"
        ),
        "backscatter_constraint": (
            "bounded correction; no third additive closure branch"
        ),
        "training_flow_sensor_fraction": float(config.paper.train_sensor_fraction),
        "training_sgs_sensor_fraction": float(config.paper.closure_sensor_fraction),
        "dense_reference_usage": "post-training evaluation and plotting only",
        "figure_5_protocol": "same frozen checkpoint and identical 4% sparse assimilation",
        "phase_status": {
            "figure_1": "manual author schematic; not generated by Python",
            "figures_2_to_6": "source data available for final plotting",
            "figure_7": "reserved for future independently retrained multi-seed sparse study",
            "figure_8": "reserved for future independently trained physics ablation",
            "figure_9": "reserved for future multi-filter-width generalisation study",
        },
    }
    (output_root / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validation suite complete: {output_root}")
    return output_root
