from __future__ import annotations
import math
import numpy as np
import torch
from .metrics import *
from .metrics import _radial_spectrum
def sgs_budget_diagnostic(omega: np.ndarray, div: np.ndarray,
                          pi: np.ndarray, transfer: np.ndarray) -> dict:
    """Scale-aware Parseval/integration-by-parts check for float32 solver fields.

    A fixed absolute cutoff alone rejects valid large-amplitude fields even
    when their normalized identity error is at float32 roundoff level.
    """
    if any(not np.isfinite(field).all() for field in (omega,div,pi,transfer)):
        raise FloatingPointError("nonfinite field in SGS budget diagnostic")
    absolute=abs(float(transfer.sum()+pi.mean()))
    scale=max(float(np.mean(np.abs(pi))),float(np.mean(np.abs(omega*div))),1e-30)
    relative=absolute/scale
    return {"SGS_budget_error":absolute,
            "SGS_budget_relative_error":relative,
            "SGS_budget_scale":scale,
            "SGS_budget_pass":bool(absolute<=1e-5 or relative<=1e-5)}


def score(solver, states: list[torch.Tensor], times: np.ndarray,
          ref: dict, matched: np.ndarray, mode: str, regime: str, ic: str) -> list[dict]:
    rows = []
    for index, omega in enumerate(states):
        output = solver.rhs(omega, float(times[index]))
        get = lambda name: output[name].detach().cpu().numpy().astype(np.float64)
        qx, qy, div, pi = (get(name) for name in ("qx", "qy", "div_q", "pi"))
        current_omega = omega.detach().cpu().numpy().astype(np.float64)
        u, v = get("u"), get("v")
        _, E = _radial_spectrum(u,v)
        model_transfer=transfer_spectrum(current_omega,div)
        budget=sgs_budget_diagnostic(current_omega,div,pi,model_transfer)
        q_error = np.stack((qx-ref["q_sgs_x"][index], qy-ref["q_sgs_y"][index]))
        q_ref = np.stack((ref["q_sgs_x"][index], ref["q_sgs_y"][index]))
        row = {
            "time_s": float(times[index]), "mode": mode, "regime": regime, "ic": ic,
            "q_vector_relative_l2": float(np.linalg.norm(q_error.ravel())/max(np.linalg.norm(q_ref.ravel()),1e-30)),
            "div_relative_l2_physical": rel2(div,ref["div_q_sgs"][index]),
            "div_relative_l2_solver_matched": rel2(div,matched[index]),
            "Pi_Pearson": pearson(pi,ref["pi_sgs"][index]),
            "Pi_RMS_ratio": float(np.sqrt(np.mean(pi*pi))/max(np.sqrt(np.mean(ref["pi_sgs"][index]**2)),1e-30)),
            "reference_q_vector_RMS":float(np.sqrt(np.mean(q_ref[0]**2+q_ref[1]**2))),
            "reference_div_RMS":float(np.sqrt(np.mean(ref["div_q_sgs"][index]**2))),
            "reference_Pi_RMS":float(np.sqrt(np.mean(ref["pi_sgs"][index]**2))),
            "q_vector_absolute_RMS_error":float(np.sqrt(np.mean(q_error[0]**2+q_error[1]**2))),
            "div_absolute_RMS_error":float(np.sqrt(np.mean((div-ref["div_q_sgs"][index])**2))),
            "Pi_absolute_RMS_error":float(np.sqrt(np.mean((pi-ref["pi_sgs"][index])**2))),
            "relative_metric_ill_conditioned":bool(any(np.sqrt(np.mean(a*a))<1e-12 for a in
                (q_ref[0],q_ref[1],ref["div_q_sgs"][index],ref["pi_sgs"][index]))),
            "SGS_transfer_spectrum_L1_physical_k1_33": spectral_l1(
                model_transfer,ref["T_sgs"][index]),
            "omega_relative_l2": rel2(current_omega,ref["omega"][index]),
            "K_relative_difference": abs(.5*np.mean(u*u+v*v)-ref["K"][index])/max(abs(ref["K"][index]),1e-30),
            "Z_relative_difference": abs(.5*np.mean(current_omega**2)-ref["Z"][index])/max(abs(ref["Z"][index]),1e-30),
            "E_spectrum_L1_k1_33": spectral_l1(E,ref["E"][index]),
            "model_omega_RMS":float(np.sqrt(np.mean(current_omega**2))),
            "model_omega_max_abs":float(np.max(np.abs(current_omega))),
            "model_q_vector_RMS":float(np.sqrt(np.mean(qx*qx+qy*qy))),
            "model_div_RMS":float(np.sqrt(np.mean(div*div))),
            "model_Pi_RMS":float(np.sqrt(np.mean(pi*pi))),
            **budget,
            **event_metrics(pi,ref["pi_sgs"][index]),
        }
        if not budget["SGS_budget_pass"]:
            raise ValueError(
                f"SGS budget identity failed at t={times[index]}: "
                f"absolute={budget['SGS_budget_error']:.6g}, "
                f"relative={budget['SGS_budget_relative_error']:.6g}, "
                f"scale={budget['SGS_budget_scale']:.6g}"
            )
        rows.append(row)
    return rows


def mean_rows(rows: list[dict]) -> dict:
    keys = rows[0]
    out = {}
    for key in keys:
        values = [row[key] for row in rows if isinstance(row[key],(int,float))
                  and not isinstance(row[key],bool)
                  and row[key] is not None and math.isfinite(row[key])]
        if values and key != "time_s":
            out[key] = float(np.mean(values))
    return out

