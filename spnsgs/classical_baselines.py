"""Parameter-frozen classical SGS vorticity-flux formulas.

This module contains no training, checkpoint loading, or evaluation workflow.
Its ``state`` argument must come from the same spectral solver state used by
the neural closure.  The Gaussian filter convention is
exp(-delta**2 * |k|**2 / 24), so its leading gradient coefficient is
delta**2 / 12.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch


def classical_fluxes(
    state: Mapping[str, torch.Tensor], delta_m: float, c: float
) -> dict[str, torch.Tensor]:
    """Return Leith, Clark, and their sum for a shared nonnegative scalar c.

    Formal calibration must fit separate scalars for Leith and mixed.  This
    function only evaluates a supplied scalar and is not a calibration path.
    """
    if not math.isfinite(delta_m) or delta_m <= 0.0:
        raise ValueError("Gaussian filter width must be finite and positive")
    if not math.isfinite(c) or c < 0.0:
        raise ValueError("Leith coefficient c=C_L**3 must be finite and nonnegative")
    gx, gy = state["omega_x"], state["omega_y"]
    grad_norm = torch.sqrt(gx.square() + gy.square())
    leith_scale = c * delta_m**3 * grad_norm
    ql_x, ql_y = -leith_scale * gx, -leith_scale * gy
    clark_scale = delta_m**2 / 12.0
    qc_x = clark_scale * (state["u_x"] * gx + state["u_y"] * gy)
    qc_y = clark_scale * (state["v_x"] * gx + state["v_y"] * gy)
    return {
        "leith_x": ql_x,
        "leith_y": ql_y,
        "clark_x": qc_x,
        "clark_y": qc_y,
        "mixed_x": ql_x + qc_x,
        "mixed_y": ql_y + qc_y,
        "pi_leith": -(ql_x * gx + ql_y * gy),
        "pi_clark": -(qc_x * gx + qc_y * gy),
        "pi_mixed": -((ql_x + qc_x) * gx + (ql_y + qc_y) * gy),
    }


def fit_nonnegative_leith_scalar(
    leith_shape: np.ndarray,
    target_flux: np.ndarray,
    weights_by_time: np.ndarray,
    *,
    clark_flux: np.ndarray | None = None,
) -> float:
    """Fit one nonnegative c to weighted sparse vector-flux labels.

    Arrays have shapes (time, site, 2) and (time, site).  The per-time
    inverse-probability weights must each sum to one; this gives every stored
    time equal weight.  For the mixed comparator, pass the *fixed analytic*
    Clark flux, which is subtracted from the target before the one-scalar fit.
    """
    shape = np.asarray(leith_shape, dtype=np.float64)
    target = np.asarray(target_flux, dtype=np.float64)
    weights = np.asarray(weights_by_time, dtype=np.float64)
    if shape.ndim != 3 or shape.shape[-1] != 2 or target.shape != shape.shape:
        raise ValueError("Leith shape and target must both be (time, site, 2)")
    if weights.shape != shape.shape[:2]:
        raise ValueError("Weights must be (time, site)")
    if clark_flux is not None:
        clark = np.asarray(clark_flux, dtype=np.float64)
        if clark.shape != shape.shape:
            raise ValueError("Fixed Clark flux must match Leith shape")
        target = target - clark
    if not (
        np.isfinite(shape).all()
        and np.isfinite(target).all()
        and np.isfinite(weights).all()
        and (weights >= 0).all()
        and np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("Sparse inputs and normalized nonnegative weights must be finite")
    numerator = float(np.sum(weights[..., None] * shape * target))
    denominator = float(np.sum(weights[..., None] * shape * shape))
    if denominator == 0.0:
        return 0.0
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("Nonfinite weighted least-squares sum")
    return max(0.0, numerator / denominator)
