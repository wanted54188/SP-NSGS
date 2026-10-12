from __future__ import annotations
import csv,json,math,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import Snapshot,inspect_reference,load_observation_snapshots
from .solver import SpectralVorticitySolver
from .protocols import _device,_dtype,_normalised_huber,_periodic_fourier_quadrature_weights,_periodic_longitudinal_flux_pairs,_periodic_nearest_sensor_pairs,_prepare_reference,_seed_everything,_select_sensor_indices,_sensor_protocol,_sign_balanced_transfer_loss,_stable_weighted_correlation,_time_varying_sparse_subset,_time_varying_sparse_subset_with_weights
from .evolution import _SparseVorticityNudger,_assimilated_rollout,_rollout
from .metrics import _aggregate
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


def _heldout_reference_closure_score(
    solver: SpectralVorticitySolver,
    reference: dict[str, torch.Tensor],
    closure_validation_schedule: torch.Tensor | None,
    closure_validation_weights: torch.Tensor | None,
) -> dict[str, float | None]:
    """V2 primary selection: sparse held-out SGS labels, no trajectory rollout."""
    if closure_validation_schedule is None:
        raise RuntimeError("revision-v2 primary checkpoint requires held-out SGS labels")
    was_training = solver.training
    solver.eval()
    divergence_errors: list[torch.Tensor] = []
    divergence_correlations: list[torch.Tensor] = []
    transfer_correlations: list[torch.Tensor] = []
    predicted_backscatter: list[torch.Tensor] = []
    reference_backscatter: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for index in range(1, closure_validation_schedule.shape[0]):
                sites = closure_validation_schedule[index]
                if sites.numel() == 0:
                    continue
                closure = solver.closure_fields(solver.state(reference["omega"][index]))
                divergence = solver.project(
                    solver.derivative(closure["qx"], "x")
                    + solver.derivative(closure["qy"], "y")
                ).reshape(-1)[sites]
                divergence_ref = reference["div_q_sgs"][index].reshape(-1)[sites]
                transfer = closure["pi"].reshape(-1)[sites]
                transfer_ref = reference["pi_sgs"][index].reshape(-1)[sites]
                weights = (
                    torch.full_like(divergence_ref, 1.0 / sites.numel())
                    if closure_validation_weights is None
                    else closure_validation_weights[index]
                )
                weights = weights / weights.sum().clamp_min(1e-12)
                divergence_errors.append(torch.sqrt(
                    torch.sum(weights * (divergence - divergence_ref).square())
                    / torch.sum(weights * divergence_ref.square()).clamp_min(1e-12)
                ))
                eps = solver.config.paper.closure_correlation_epsilon
                divergence_correlations.append(_stable_weighted_correlation(
                    divergence, divergence_ref, weights, eps
                ))
                transfer_correlations.append(_stable_weighted_correlation(
                    transfer, transfer_ref, weights, eps
                ))
                predicted_backscatter.append(transfer < 0.0)
                reference_backscatter.append(transfer_ref < 0.0)
    finally:
        solver.train(was_training)
    if not divergence_errors:
        raise RuntimeError("revision-v2 closure score has no held-out SGS samples")
    errors = torch.stack(divergence_errors)
    div_corr = torch.mean(torch.stack(divergence_correlations))
    pi_corr = torch.mean(torch.stack(transfer_correlations))
    pred_fraction = torch.cat(predicted_backscatter).float().mean()
    ref_fraction = torch.cat(reference_backscatter).float().mean()
    fraction_error = torch.abs(pred_fraction - ref_fraction)
    score = (
        0.40 * torch.mean(errors)
        + 0.15 * errors[-1]
        + 0.15 * (1.0 - div_corr)
        + 0.15 * (1.0 - pi_corr)
        + 0.15 * fraction_error
    )
    return {
        "validation_score": float(score.item()),
        "closure_score": float(score.item()),
        "rollout_score": None,
        "flow_score": None,
        "autonomous_flow_score": None,
        "assimilated_total": None,
        "divergence_relative_l2_mean": float(errors.mean().item()),
        "divergence_relative_l2_endpoint": float(errors[-1].item()),
        "divergence_relative_l2_worst": float(errors.max().item()),
        "divergence_correlation": float(div_corr.item()),
        "transfer_correlation": float(pi_corr.item()),
        "backscatter_fraction_prediction": float(pred_fraction.item()),
        "backscatter_fraction_reference": float(ref_fraction.item()),
        "backscatter_fraction_error": float(fraction_error.item()),
    }


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
    primary_rule = solver.config.paper.primary_checkpoint_rule
    if primary_rule not in {"legacy_composite", "heldout_closure_reference_state"}:
        raise ValueError(f"unsupported primary_checkpoint_rule: {primary_rule}")
    if primary_rule == "heldout_closure_reference_state":
        return _heldout_reference_closure_score(
            solver, reference, closure_validation_schedule, closure_validation_weights
        )
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
                # Revision v2 isolates closure ranking from the assimilated
                # trajectory. The STAR filtered state is a validation input;
                # only disjoint held-out SGS locations enter the score.
                closure_input = (
                    reference["omega"][index]
                    if primary_rule == "heldout_closure_reference_state"
                    else analysis_omega
                )
                closure_state = solver.state(closure_input)
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
        config.paper.experiment_variant in {"full_sp_nsgs", "isotropic_signed"}
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
                config.paper.experiment_variant in {
                    "iso_dissipative", "anisotropic_dissipative"
                }
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
                if rollout_validation is not None and rollout_validation < best_rollout_score:
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
                if (config.paper.primary_checkpoint_rule == "legacy_composite"
                        and validation < best_composite_score):
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

    closure_only_primary = (
        config.paper.primary_checkpoint_rule == "heldout_closure_reference_state"
    )
    solver.load_state_dict(
        best_closure_state if closure_only_primary else best_composite_state
    )
    solver.closure.set_backscatter_factor(1.0)
    final_validation_metrics = dict(
        best_closure_metrics if closure_only_primary else best_composite_metrics
    )
    calibration_report: dict[str, object] = {
        "enabled": False,
        "selected_scale": 1.0,
        "baseline_retained": True,
        "uses_dense_reference": False,
    }
    if config.paper.sparse_closure_scale_calibration and not closure_only_primary:
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
        best_closure_score if closure_only_primary else best_composite_score,
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
        "best_validation_score": (
            best_closure_score if closure_only_primary else best_composite_score
        ),
        "best_closure_score": best_closure_score,
        "best_rollout_score": None if closure_only_primary else best_rollout_score,
        "best_composite_score": None if closure_only_primary else best_composite_score,
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


from .reference import inspect_training_reference as inspect_reference
