from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    file_glob: str = "XYZ_内部表_table_*.csv"
    grid_points_x: int = 101
    grid_points_y: int = 101
    coordinate_snap_tolerance_m: float = 1e-5
    drop_periodic_endpoint: bool = True
    merge_periodic_endpoints: bool = True
    filter_width_m: float = 0.25132741228718345
    filter_kind: str = "gaussian"
    data_are_filtered: bool = False
    apply_filter_in_memory: bool = True
    exclude_boundary_layers: int = 0
    observation_time_interval_s: float = 0.2
    observation_spatial_fraction: float = 0.10
    observation_sampling: str = "stratified"
    observation_sensor_layout: str = "independent"
    # These are only two scalar statistics per snapshot, not dense pointwise
    # supervision.  Using the full grid avoids Monte-Carlo jitter in the
    # kinetic-energy and enstrophy targets.
    global_moments_use_full_snapshots: bool = False
    train_time_min_s: float = 0.0
    train_time_max_s: float = 10.0
    test_time_min_s: float = 1.0
    test_time_max_s: float = 5.0
    allow_nonperiodic_data: bool = False
    periodic_relative_rms_tolerance: float = 0.08


@dataclass
class DomainConfig:
    x_min_m: float = 0.0
    x_max_m: float = 6.283185307179586
    y_min_m: float = 0.0
    y_max_m: float = 6.283185307179586
    t_initial_s: float = 0.0

    @property
    def lx(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def ly(self) -> float:
        return self.y_max_m - self.y_min_m


@dataclass
class PhysicsConfig:
    density_kg_m3: float = 1000.0
    viscosity_value: float = 0.001
    viscosity_kind: str = "dynamic"
    forcing_enabled: bool = False
    momentum_source_amplitude: float = 0.0
    momentum_source_kind: str = "per_volume"
    forcing_wave_number_rad_m: float = 0.0
    forcing_filter_gain: float = 1.0
    initial_condition_kind: str = "multimode_decay"
    initial_streamfunction_amplitude_m2_s: float = 0.15
    # Bounded correction for STAR-CCM+ mesh/time-discretisation damping.
    # The physical viscosity above is kept unchanged.
    numerical_viscosity_initial_m2_s: float = 0.0015
    numerical_viscosity_max_m2_s: float = 0.003
    learn_numerical_viscosity: bool = True

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        kind = self.viscosity_kind.lower()
        if kind == "dynamic":
            return self.viscosity_value / self.density_kg_m3
        if kind == "kinematic":
            return self.viscosity_value
        raise ValueError("viscosity_kind must be 'dynamic' or 'kinematic'")

    @property
    def forcing_acceleration_amplitude_m_s2(self) -> float:
        kind = self.momentum_source_kind.lower()
        if kind == "per_volume":
            return self.momentum_source_amplitude / self.density_kg_m3
        if kind == "acceleration":
            return self.momentum_source_amplitude
        raise ValueError(
            "momentum_source_kind must be 'per_volume' or 'acceleration'"
        )


@dataclass
class ModelConfig:
    flow_hidden_width: int = 128
    flow_hidden_layers: int = 4
    sgs_hidden_width: int = 64
    sgs_hidden_layers: int = 4
    fourier_modes: int = 12
    cross_fourier_modes: int = 10
    spectral_modes: int = 15
    # soft_dual: independent psi/omega heads (legacy);
    # streamfunction_laplacian: predict psi and differentiate twice;
    # vorticity_inverse_laplacian: predict omega and recover psi spectrally.
    flow_representation: str = "soft_dual"
    hard_vorticity_from_streamfunction: bool = False
    # For hard spectral representations, evaluate all spatial derivatives
    # analytically from the Fourier basis.  This avoids repeatedly building
    # nested autograd graphs for quantities whose derivatives are known.
    analytic_spatial_derivatives: bool = False
    sgs_enabled: bool = True
    time_fourier_modes: int = 4
    beta_backscatter: float = 2.0
    cholesky_epsilon: float = 1e-6
    direction_epsilon: float = 1e-6
    # Optional local resolved features for the auxiliary SGS flux map.
    sgs_use_velocity_gradient: bool = True
    sgs_use_vorticity_hessian: bool = True
    # V16 options.  The adaptive residual decoder makes the temporal
    # coefficient map easier to optimise without changing the exact periodic
    # Fourier representation in space.  RBF time features are local in time
    # and reduce interference between early and late snapshots.
    flow_network_kind: str = "mlp"
    time_feature_kind: str = "fourier"
    time_rbf_centers: int = 21
    # When enabled, predict D = div(q_SGS) directly from a periodic local
    # (omega, psi) stencil.  D is the identifiable term in the filtered
    # vorticity equation; the legacy vector flux head remains as an optional
    # auxiliary objective.
    sgs_direct_source: bool = False
    # direct: V16 pointwise source; conservative_potential: predict a local
    # pseudoscalar potential and apply a fixed periodic discrete Laplacian.
    sgs_source_kind: str = "direct"
    sgs_patch_size: int = 1
    sgs_source_channels: int = 24


@dataclass
class TrainingConfig:
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"
    steps: int = 5000
    learning_rate: float = 2e-4
    joint_learning_rate: float = 1e-4
    physics_polish_learning_rate: float = 2e-6
    gradient_clip_norm: float = 1.0
    flow_pretrain_steps: int = 10000
    omega_head_warmup_steps: int = 0
    sgs_pretrain_steps: int = 5000
    physics_polish_steps: int = 0
    freeze_flow_during_joint: bool = True
    joint_flow_learning_rate: float = 1e-5
    causal_training_enabled: bool = False
    causal_joint_restart: bool = False
    causal_time_windows: int = 10
    causal_full_domain_fraction: float = 0.75
    collocation_batch_size: int = 256
    data_batch_size: int = 512
    data_time_groups_per_batch: int = 1
    validation_fraction_of_sparse_sensors: float = 0.0
    validation_include_derived_fields: bool = True
    validation_include_global_moments: bool = True
    # Extra model-selection emphasis on the last observed time.  This does
    # not enter the training objective or add observations.
    validation_endpoint_weight: float = 0.0
    # Model-selection penalty on the worst time-resolved sparse omega error.
    # Unlike a global average, this prevents late-time failures from being
    # hidden by many easier early snapshots.
    validation_time_worst_weight: float = 0.0
    # Weight of held-out sparse closure validation in joint checkpoint
    # selection.  Zero preserves the legacy flow-only selection rule.
    validation_sgs_weight: float = 0.0
    initial_batch_size: int = 256
    log_every: int = 100
    checkpoint_every: int = 1000
    lambda_pde: float = 1.0
    lambda_streamfunction: float = 0.0
    lambda_initial: float = 5.0
    lambda_data: float = 40.0
    lambda_velocity_data: float = 10.0
    lambda_vorticity_gradient_data: float = 5.0
    lambda_psi_data: float = 5.0
    lambda_energy_data: float = 50.0
    lambda_enstrophy_data: float = 3.0
    lambda_energy_temporal: float = 100.0
    lambda_enstrophy_temporal: float = 1.0
    # Label-free regularisation of the spectral-coefficient trajectory.
    # The loss is a scale-normalised second finite difference in time.
    lambda_spectral_temporal_smoothness: float = 0.0
    lambda_pde_polish: float = 0.2
    lambda_budget: float = 0.1
    lambda_sgs: float = 1.0
    # Relative weight of the legacy vector-flux label when a direct SGS
    # source head is used.  Setting this below one prevents the unidentifiable
    # divergence-free part of q from dominating the dynamical source target.
    lambda_sgs_q: float = 1.0
    lambda_transfer: float = 1.0
    # Match the time-resolved mean SGS transfer using only the same sparse
    # closure sensors used by lambda_sgs; no dense/global labels are read.
    lambda_sgs_mean_transfer: float = 0.0
    # Sparse-label closure-shape objectives. Direction targets the vector
    # flux orientation; backscatter targets the sign of local SGS transfer.
    lambda_sgs_direction: float = 0.0
    lambda_sgs_backscatter: float = 0.0
    # Match div(q_SGS), the quantity that actually enters the filtered
    # vorticity equation, at the same sparse closure-sensor locations.
    lambda_sgs_divergence: float = 0.0
    # During joint training/validation, feed the SGS model reconstructed-flow
    # features instead of exact reference features.  This removes the
    # train/deployment covariate mismatch while retaining reference-feature
    # pretraining for a stable initialization.
    sgs_predicted_features_during_joint: bool = False
    # V16 can pretrain the closure on reconstructed periodic stencils.  This
    # removes the reference-feature/deployment mismatch from the outset.
    sgs_predicted_features_during_pretrain: bool = False
    # A stable full pass over the same sparse closure sensors is inexpensive
    # for the small SGS network and reduces minibatch variance in pretraining.
    sgs_full_batch_pretrain: bool = False
    # Absolute fraction of the full spatial grid carrying q_SGS / Pi labels.
    # Zero preserves the legacy behaviour of using every flow-training sensor.
    sgs_supervision_spatial_fraction: float = 0.0
    # Allow TensorFloat-32 matrix multiplication on Ampere/Ada CUDA devices.
    # Coordinate derivatives remain float32/float64 as configured above.
    enable_tf32: bool = False
    output_dir: Path = Path("outputs")


@dataclass
class PaperConfig:
    """Theory-faithful solver-embedded SP-NSGS paper controls."""

    checkpoint_format_version: int = 7
    closure_parameterization: str = "theory_faithful"
    # The default is the manuscript model.  ``iso_dissipative`` is reserved
    # for the independently trained, deliberately simpler Figure-8 baseline:
    # q_iso = -nu_iso grad(omega), nu_iso >= 0.  It never changes the default
    # SP-NSGS physics or permits a post-training switch-off to be reported as
    # an independent baseline.
    experiment_variant: str = "full_sp_nsgs"
    # ``normalised`` is the historical seven-channel input.  ``invariant_v2``
    # appends three dimensionless, spatially constant state descriptors so a
    # closure can distinguish spectra with similar normalised morphology but
    # different gradient/curvature content during turbulent decay.
    closure_feature_set: str = "normalised"
    closure_channels: int = 32
    closure_layers: int = 4
    closure_dilations: list[int] = field(default_factory=lambda: [1])
    # q_d = -nu_d A grad(omega), nu_d = c_theta Delta^3 |grad(omega)|,
    # c_theta starts from a positive softplus map and is then passed through a
    # smooth numerical ceiling.  This keeps the dissipative/backscatter
    # decomposition intact while preventing the two branches from growing
    # without bound and cancelling one another during sparse training.
    dissipation_coefficient_initial_bias: float = -8.0
    dissipation_coefficient_soft_limit: float = 0.02
    # Generalised bounded backscatter branch.  The original |q_d|-only scale
    # made q_b vanish whenever sparse training correctly reduced excessive
    # dissipation.  A Clark-gradient magnitude is now used only as an
    # objective local SGS scale/direction prior; the complete closure still
    # has exactly two branches, q_sgs = q_d + q_b.
    gradient_support_coefficient_initial: float = 1.0
    gradient_support_coefficient_maximum: float = 2.0
    gradient_direction_prior_strength: float = 1.0
    backscatter_learned_direction_strength: float = 0.5
    backscatter_gate_initial: float = 0.25
    # |q_b| <= beta sqrt(|q_d|^2 + |q_Clark|^2); beta must be positive.
    maximum_backscatter_ratio: float = 2.0
    # Local negative transfer remains possible, but its magnitude cannot
    # exceed this multiple of the pointwise dissipative transfer.  Values > 1
    # retain genuine local backscatter in the complete SGS flux.
    maximum_local_backscatter_transfer_ratio: float = 1.25
    direction_epsilon: float = 1e-3
    cholesky_diagonal_minimum: float = 0.05
    cholesky_offdiagonal_limit: float = 1.0
    # Enforced algebraically after the Cholesky map, not merely penalised.
    maximum_anisotropy_condition: float = 8.0
    anisotropy_condition_soft_limit: float = 8.0
    # First learn the provably dissipative branch, then ramp in backscatter.
    backscatter_warmup_epochs: int = 15
    backscatter_ramp_epochs: int = 20
    integrator: str = "rk4"
    substeps_per_snapshot: int = 5
    dealias_fraction: float = 2.0 / 3.0
    rollout_window_snapshots: int = 3
    rollout_window_warmup_epochs: int = 10
    rollout_window_ramp_epochs: int = 40
    epochs: int = 60
    learning_rate: float = 5e-4
    minimum_learning_rate: float = 2e-5
    closure_weight_decay: float = 1e-6
    parameter_ema_decay: float = 0.0
    parameter_ema_start_epoch: int = 20
    gradient_clip_norm: float = 1.0
    validation_every_epochs: int = 2
    # Full autonomous sparse-flow validation is added to checkpoint selection.
    autonomous_validation_weight: float = 1.0
    # Sparse validation scores fluctuate because only 25 held-out SGS labels
    # are used at each snapshot.  Within this relative tolerance, select the
    # checkpoint with the lower held-out flow score instead of treating small
    # closure-score noise as a meaningful ordering.
    validation_tolerance_fraction: float = 0.0
    checkpoint_every_epochs: int = 5
    train_sensor_fraction: float = 0.04
    validation_sensor_fraction: float = 0.01
    # SGS labels are read only at this subset of the already selected flow
    # sensors.  0.0025 means 25 points on the 100 x 100 periodic grid.
    closure_sensor_fraction: float = 0.0025
    # Fraction of the unchanged closure-label budget allocated using a local
    # vorticity-increment proxy computed only from the 4% flow sensors.
    closure_importance_fraction: float = 0.0
    # Fraction of the parent flow-sensor cloud treated as the high-gradient
    # stratum. Active sampling is corrected with exact stratum weights, so it
    # improves rare-event coverage without biasing full-domain moments.
    closure_importance_pool_fraction: float = 0.25
    sensor_sampling: str = "stratified"
    omega_sensor_weight: float = 1.0
    velocity_sensor_weight: float = 2.0
    sensor_correlation_weight: float = 1.0
    # Sparse first-order structure-function loss.  It uses differences between
    # neighbouring members of the same sensor set, so the number of labels is
    # unchanged while small resolved scales receive a direct training signal.
    omega_increment_weight: float = 0.0
    omega_increment_neighbours: int = 4
    sparse_energy_weight: float = 4.0
    sparse_enstrophy_weight: float = 4.0
    # Estimate spatial moments with coordinate-only periodic Fourier
    # quadrature, then compare those estimates with the complete predicted
    # field.  No additional reference values enter this construction.
    coordinate_weighted_sparse_moments: bool = False
    # Match the modelled kinetic-energy tendency to a centred finite
    # difference formed from the same sparse u/v time series.
    sparse_energy_tendency_weight: float = 0.0
    flux_regularization_weight: float = 1e-5
    anisotropy_condition_weight: float = 0.002
    backscatter_saturation_weight: float = 0.002
    mean_dissipation_weight: float = 0.02
    dissipation_coefficient_saturation_weight: float = 0.1
    sparse_sgs_divergence_weight: float = 0.15
    # Complement the amplitude-sensitive divergence loss with a centred
    # correlation loss on the same sparse SGS labels.  The epsilon is used in
    # both training and held-out checkpoint metrics when either field has very
    # small variance.
    sparse_sgs_divergence_correlation_weight: float = 0.0
    closure_correlation_epsilon: float = 1e-8
    sparse_sgs_transfer_weight: float = 0.03
    # Robust scale for the sign-balanced total-transfer Huber loss.  Positive
    # and negative reference samples are normalised independently.
    sparse_sgs_transfer_huber_delta: float = 1.0
    sparse_sgs_mean_transfer_weight: float = 0.5
    # Legacy branch-specific target loss.  It must remain zero when q_b is the
    # sign-indefinite bounded correction used by the manuscript model.
    sparse_sgs_branch_transfer_weight: float = 0.25
    sparse_sgs_flux_weight: float = 0.0
    # Multi-scale longitudinal increments formed only from existing sparse SGS
    # flux labels (nearest, second-nearest, and one medium-distance neighbour).
    sparse_sgs_longitudinal_increment_weight: float = 0.0
    # Physics-inferred source supervision.  At every existing sparse flow
    # sensor, the vorticity equation supplies a noisy estimate of div(q_sgs)
    # from omega_t and the reconstructed resolved terms.  No new SGS labels
    # or sensor locations are introduced.
    physics_inferred_divergence_weight: float = 0.0
    physics_inferred_divergence_huber_delta: float = 1.0
    # Optional one-parameter calibration of the complete conservative SGS
    # flux.  The least-squares proposal uses only the existing sparse training
    # SGS labels, while a disjoint sparse validation set may reject it and
    # retain the uncalibrated value 1.0.  This cannot consume dense fields.
    sparse_closure_scale_calibration: bool = False
    closure_scale_minimum: float = 0.75
    closure_scale_maximum: float = 1.25
    closure_scale_validation_flow_tolerance: float = 0.01
    # The composite checkpoint keeps the closure score primary and adds a
    # modest held-out sparse-rollout term.  Dense final-test fields are never
    # consulted during selection.
    checkpoint_composite_rollout_weight: float = 0.1
    # Label-free spectral-tail prior.  It prevents closure/backscatter energy
    # from accumulating close to the de-aliasing cutoff while leaving the
    # energetic low modes untouched.
    spectral_tail_weight: float = 0.003
    spectral_tail_start_fraction: float = 0.60
    spectral_tail_max_enstrophy_fraction: float = 0.002
    nudging_relaxation: float = 0.25
    nudging_length_scale_m: float = 0.12
    velocity_nudging_relaxation: float = 0.40
    velocity_nudging_length_scale_m: float = 0.12
    causal_curriculum_epochs: int = 20
    curriculum_start_time_s: float = 1.0
    # Fixed paper-figure colour limits.  These affect presentation only; no
    # prediction, loss or quantitative metric uses them.
    signed_omega_error_color_limit: float = 1.0
    absolute_omega_error_color_limit_per_s: float = 4.0
    # Keep pointwise closure-error maps in supplementary material by default;
    # the primary figure can show the reference/prediction fields
    # without visually over-emphasising isolated derivative errors.
    paper_show_absolute_closure_error_panels: bool = True
    # Post-training robustness audit.  These fractions change only the number
    # of sparse observations supplied to the already trained solver; they do
    # not add training labels and do not alter the closure model.  Fractions
    # zero and train_sensor_fraction reuse the autonomous and primary rollouts.
    sparse_robustness_sensor_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.04]
    )
    sparse_robustness_repeats: int = 1
    generate_validation_suite: bool = True
    evaluation_times_s: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)


def _construct(cls: type, values: dict[str, Any]) -> Any:
    known = cls.__dataclass_fields__
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
    return cls(**values)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    data_values = dict(raw.get("data", {}))
    train_values = dict(raw.get("training", {}))
    if "data_dir" in data_values:
        candidate = Path(data_values["data_dir"])
        data_values["data_dir"] = (
            candidate if candidate.is_absolute() else (base / candidate).resolve()
        )
    if "output_dir" in train_values:
        candidate = Path(train_values["output_dir"])
        train_values["output_dir"] = (
            candidate if candidate.is_absolute() else (base / candidate).resolve()
        )
    return Config(
        data=_construct(DataConfig, data_values),
        domain=_construct(DomainConfig, raw.get("domain", {})),
        physics=_construct(PhysicsConfig, raw.get("physics", {})),
        model=_construct(ModelConfig, raw.get("model", {})),
        training=_construct(TrainingConfig, train_values),
        paper=_construct(PaperConfig, raw.get("paper", {})),
    )

