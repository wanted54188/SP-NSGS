from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


class PeriodicConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("periodic convolution dilation must be positive")
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.norm1 = nn.GroupNorm(4 if channels % 4 == 0 else 1, channels)
        self.norm2 = nn.GroupNorm(4 if channels % 4 == 0 else 1, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.silu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return F.silu(value + residual)


class SPNSGSClosure(nn.Module):
    """Theory-faithful dissipative/backscatter SGS flux.

    The network does not predict an unconstrained SGS source.  It predicts the
    seven scalar fields needed by the manuscript model:

    ``a, l11, l21, l22, s, gx, gy``.

    A trace-normalised Cholesky map makes ``A`` symmetric positive definite
    and ``c`` is positive.  The backscatter branch is bounded by a composite
    local SGS scale made from ``|q_d|`` and a Clark-gradient support field.
    The support field is not a third additive closure branch: it supplies a
    physically scaled prior to q_b, while the complete flux remains exactly
    ``q_sgs = q_d + q_b``.  Consequently the dissipative branch retains
    non-negative pointwise enstrophy transfer by construction without making
    the expressive power of q_b proportional to excessive dissipation.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.experiment_variant = config.paper.experiment_variant.lower()
        if self.experiment_variant not in {
            "full_sp_nsgs",
            "iso_dissipative",
            "no_sgs",
        }:
            raise ValueError(
                "experiment_variant must be 'full_sp_nsgs', "
                "'iso_dissipative', or 'no_sgs'"
            )
        channels = config.paper.closure_channels
        self.parameterization = config.paper.closure_parameterization.lower()
        if self.parameterization != "theory_faithful":
            raise ValueError(
                "closure_parameterization must be 'theory_faithful'"
            )
        self.dissipation_coefficient_initial_bias = float(
            config.paper.dissipation_coefficient_initial_bias
        )
        self.dissipation_coefficient_soft_limit = float(
            config.paper.dissipation_coefficient_soft_limit
        )
        if self.dissipation_coefficient_soft_limit <= 0.0:
            raise ValueError("dissipation_coefficient_soft_limit must be positive")
        self.maximum_backscatter_ratio = float(
            config.paper.maximum_backscatter_ratio
        )
        if self.maximum_backscatter_ratio <= 0.0:
            raise ValueError("maximum_backscatter_ratio must be positive")
        self.maximum_local_backscatter_transfer_ratio = float(
            config.paper.maximum_local_backscatter_transfer_ratio
        )
        if self.maximum_local_backscatter_transfer_ratio <= 1.0:
            raise ValueError(
                "maximum_local_backscatter_transfer_ratio must exceed one "
                "so the complete flux can retain genuine local backscatter"
            )
        self.gradient_support_coefficient_maximum = float(
            config.paper.gradient_support_coefficient_maximum
        )
        gradient_initial = float(
            config.paper.gradient_support_coefficient_initial
        )
        if not 0.0 < gradient_initial < self.gradient_support_coefficient_maximum:
            raise ValueError(
                "gradient_support_coefficient_initial must lie strictly "
                "between zero and gradient_support_coefficient_maximum"
            )
        gradient_fraction = gradient_initial / self.gradient_support_coefficient_maximum
        self.gradient_support_coefficient_raw = nn.Parameter(
            torch.tensor(math.log(gradient_fraction / (1.0 - gradient_fraction)))
        )
        self.gradient_direction_prior_strength = float(
            config.paper.gradient_direction_prior_strength
        )
        self.backscatter_learned_direction_strength = float(
            config.paper.backscatter_learned_direction_strength
        )
        if self.gradient_direction_prior_strength < 0.0:
            raise ValueError("gradient_direction_prior_strength must be non-negative")
        if self.backscatter_learned_direction_strength < 0.0:
            raise ValueError(
                "backscatter_learned_direction_strength must be non-negative"
            )
        backscatter_gate_initial = float(config.paper.backscatter_gate_initial)
        if not -1.0 < backscatter_gate_initial < 1.0:
            raise ValueError("backscatter_gate_initial must lie in (-1, 1)")
        self.direction_epsilon = float(config.paper.direction_epsilon)
        if self.direction_epsilon <= 0.0:
            raise ValueError("direction_epsilon must be positive")
        self.cholesky_diagonal_minimum = float(
            config.paper.cholesky_diagonal_minimum
        )
        self.cholesky_offdiagonal_limit = float(
            config.paper.cholesky_offdiagonal_limit
        )
        self.maximum_anisotropy_condition = float(
            config.paper.maximum_anisotropy_condition
        )
        if self.cholesky_diagonal_minimum <= 0.0:
            raise ValueError("cholesky_diagonal_minimum must be positive")
        if self.cholesky_offdiagonal_limit < 0.0:
            raise ValueError("cholesky_offdiagonal_limit must be non-negative")
        if self.maximum_anisotropy_condition <= 1.0:
            raise ValueError("maximum_anisotropy_condition must exceed one")
        self.backscatter_factor = 1.0
        self.feature_set = config.paper.closure_feature_set.lower()
        if self.feature_set not in {
            "normalised",
            "invariant_v2",
            "mixed_gradient_v3",
        }:
            raise ValueError(
                "closure_feature_set must be 'normalised', 'invariant_v2', "
                "or 'mixed_gradient_v3'"
            )
        self.filter_width = float(config.data.filter_width_m)
        dilations = config.paper.closure_dilations or [1]
        if any(int(value) < 1 for value in dilations):
            raise ValueError("all closure dilations must be positive")
        input_channels = {
            "normalised": 7,
            "invariant_v2": 10,
            "mixed_gradient_v3": 12,
        }[self.feature_set]
        self.input = nn.Conv2d(
            input_channels, channels, 3, padding=1, padding_mode="circular"
        )
        self.blocks = nn.Sequential(
            *[
                PeriodicConvBlock(
                    channels, int(dilations[index % len(dilations)])
                )
                for index in range(config.paper.closure_layers)
            ]
        )
        output_channels = 7
        self.output = nn.Conv2d(
            channels, output_channels, 3, padding=1, padding_mode="circular"
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # The Clark direction prior avoids the bilinear dead point between
        # tanh(s_theta) and g_theta when the backscatter curriculum begins.
        with torch.no_grad():
            self.output.bias[4] = math.atanh(backscatter_gate_initial)

    @staticmethod
    def _normalise(field: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(torch.mean(field.square(), dim=(-2, -1), keepdim=True))
        return field / scale.detach().clamp_min(1e-6)

    def set_backscatter_factor(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("backscatter factor must be finite and in [0, 1]")
        self.backscatter_factor = value

    def forward(
        self,
        omega: torch.Tensor,
        psi: torch.Tensor,
        omega_x: torch.Tensor,
        omega_y: torch.Tensor,
        strain: torch.Tensor,
        laplacian_omega: torch.Tensor,
        u_x: torch.Tensor,
        u_y: torch.Tensor,
        v_x: torch.Tensor,
        v_y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gradient_shape_x = self.filter_width**2 / 12.0 * (
            u_x * omega_x + u_y * omega_y
        )
        gradient_shape_y = self.filter_width**2 / 12.0 * (
            v_x * omega_x + v_y * omega_y
        )
        feature_fields = [
            self._normalise(omega),
            self._normalise(psi),
            self._normalise(omega_x),
            self._normalise(omega_y),
            self._normalise(strain),
            self._normalise(laplacian_omega),
            self._normalise(omega.square()),
        ]
        if self.feature_set in {"invariant_v2", "mixed_gradient_v3"}:
            # Independent normalisation of every field erases physically
            # relevant decay information.  These dimensionless descriptors
            # retain it without introducing time as an input or any labels.
            omega_rms = torch.sqrt(torch.mean(omega.square())).detach().clamp_min(1e-6)
            gradient_rms = torch.sqrt(
                torch.mean(omega_x.square() + omega_y.square())
            ).detach()
            laplacian_rms = torch.sqrt(
                torch.mean(laplacian_omega.square())
            ).detach()
            strain_rms = torch.sqrt(torch.mean(strain.square())).detach()
            ones = torch.ones_like(omega)
            feature_fields.extend(
                [
                    ones * (self.filter_width * gradient_rms / omega_rms),
                    ones
                    * (
                        self.filter_width**2
                        * laplacian_rms
                        / omega_rms
                    ),
                    ones * (strain_rms / omega_rms),
                ]
            )
        if self.feature_set == "mixed_gradient_v3":
            feature_fields.extend(
                [
                    self._normalise(gradient_shape_x),
                    self._normalise(gradient_shape_y),
                ]
            )
        features = torch.stack(feature_fields, dim=0).unsqueeze(0)
        hidden = F.silu(self.input(features))
        raw = self.output(self.blocks(hidden))[0]

        # Start from manuscript Eq. (12), then apply a smooth numerical ceiling.
        # The ceiling is an implementation-stability guard: it prevents the
        # dissipative and backscatter branches from increasing together and
        # learning an ill-conditioned cancellation under sparse supervision.
        c_theta_unbounded = F.softplus(
            raw[0] + self.dissipation_coefficient_initial_bias
        )
        c_theta = self.dissipation_coefficient_soft_limit * torch.tanh(
            c_theta_unbounded / self.dissipation_coefficient_soft_limit
        )
        gradient_norm = torch.sqrt(omega_x.square() + omega_y.square() + 1e-16)
        nu_d = c_theta * self.filter_width**3 * gradient_norm

        # Cholesky factor.  Positive diagonals make A_tilde strictly SPD.
        l11 = self.cholesky_diagonal_minimum + F.softplus(raw[1])
        l21 = self.cholesky_offdiagonal_limit * torch.tanh(raw[2])
        l22 = self.cholesky_diagonal_minimum + F.softplus(raw[3])
        a11_tilde = l11.square()
        a12_tilde = l11 * l21
        a22_tilde = l21.square() + l22.square()
        trace_tilde = (a11_tilde + a22_tilde).clamp_min(1e-12)
        a11_raw = 2.0 * a11_tilde / trace_tilde
        a12_raw = 2.0 * a12_tilde / trace_tilde
        a22_raw = 2.0 * a22_tilde / trace_tilde

        # The Cholesky map guarantees SPD, but sparse labels can still drive
        # one eigenvalue arbitrarily close to zero.  Project only the
        # trace-free anisotropic part through a smooth hard condition-number
        # cap.  Trace(A)=2 and SPD are preserved exactly, while pathological
        # needle tensors cannot interpolate isolated closure sensors.
        anisotropy_x = 0.5 * (a11_raw - a22_raw)
        anisotropy_y = a12_raw
        anisotropy_radius = torch.sqrt(
            anisotropy_x.square() + anisotropy_y.square() + 1e-24
        )
        maximum_radius = (
            (self.maximum_anisotropy_condition - 1.0)
            / (self.maximum_anisotropy_condition + 1.0)
        )
        bounded_radius = maximum_radius * torch.tanh(
            anisotropy_radius / maximum_radius
        )
        anisotropy_scale = bounded_radius / anisotropy_radius
        a11 = 1.0 + anisotropy_scale * anisotropy_x
        a12 = anisotropy_scale * anisotropy_y
        a22 = 1.0 - anisotropy_scale * anisotropy_x
        if self.experiment_variant == "iso_dissipative":
            # Formal ISO baseline: retain the same positive diffusivity map,
            # but enforce A=I exactly and remove q_b algebraically.
            a11 = torch.ones_like(a11)
            a12 = torch.zeros_like(a12)
            a22 = torch.ones_like(a22)

        qd_x = -nu_d * (a11 * omega_x + a12 * omega_y)
        qd_y = -nu_d * (a12 * omega_x + a22 * omega_y)
        # Keep the norm effectively exact at stagnation points.  A larger
        # epsilon here would create a spurious non-zero backscatter flux even
        # when the dissipative branch vanishes.
        qd_norm = torch.sqrt(qd_x.square() + qd_y.square() + 1e-16)

        # Generalised bounded backscatter.  The previous |q_d|-only scale
        # forced the learned correction to disappear as soon as the model
        # reduced an over-dissipative q_d.  The Clark-gradient field supplies
        # a label-free local SGS scale and direction prior, but is not added as
        # a third branch.  The complete model remains q_sgs = q_d + q_b.
        gradient_coefficient = self.gradient_support_coefficient_maximum * torch.sigmoid(
            self.gradient_support_coefficient_raw
        )
        gradient_support_x = gradient_coefficient * gradient_shape_x
        gradient_support_y = gradient_coefficient * gradient_shape_y
        gradient_support_norm = torch.sqrt(
            gradient_support_x.square() + gradient_support_y.square() + 1e-16
        )
        support_norm = torch.sqrt(
            qd_norm.square() + gradient_support_norm.square() + 1e-16
        )
        gradient_direction_denominator = torch.sqrt(
            gradient_support_x.square()
            + gradient_support_y.square()
            + self.direction_epsilon**2
        )
        gradient_direction_x = gradient_support_x / gradient_direction_denominator
        gradient_direction_y = gradient_support_y / gradient_direction_denominator
        backscatter_gate = torch.tanh(raw[4])
        direction_raw_x = (
            self.gradient_direction_prior_strength * gradient_direction_x
            + self.backscatter_learned_direction_strength * torch.tanh(raw[5])
        )
        direction_raw_y = (
            self.gradient_direction_prior_strength * gradient_direction_y
            + self.backscatter_learned_direction_strength * torch.tanh(raw[6])
        )
        direction_denominator = torch.sqrt(
            direction_raw_x.square()
            + direction_raw_y.square()
            + self.direction_epsilon**2
        )
        direction_x = direction_raw_x / direction_denominator
        direction_y = direction_raw_y / direction_denominator
        direction_norm = torch.sqrt(
            direction_x.square() + direction_y.square()
        )
        effective_beta = (
            self.backscatter_factor * self.maximum_backscatter_ratio
        )
        if self.experiment_variant == "iso_dissipative":
            effective_beta = 0.0
        backscatter_amplitude = effective_beta * support_norm * backscatter_gate
        qb_x_raw = backscatter_amplitude * direction_x
        qb_y_raw = backscatter_amplitude * direction_y
        pi_d = -(qd_x * omega_x + qd_y * omega_y)
        pi_b_raw = -(qb_x_raw * omega_x + qb_y_raw * omega_y)

        # A bounded amplitude alone does not bound energy injection when A is
        # anisotropic.  Limit only the locally anti-dissipative component by
        # scaling the complete backscatter vector.  Because the limit is > 1,
        # pi_d + pi_b may still be negative locally: the backscatter mechanism
        # is retained, while an arbitrarily strong anti-diffusive mode is not.
        negative_transfer = torch.relu(-pi_b_raw)
        # Use the same composite physical scale in the transfer limiter.  This
        # preserves a finite anti-diffusive bound even when q_d is correctly
        # small, while preventing arbitrary grid-scale injection.
        transfer_support = (
            torch.relu(pi_d) + gradient_support_norm * gradient_norm
        )
        allowed_negative_transfer = (
            self.maximum_local_backscatter_transfer_ratio * transfer_support
        )
        transfer_limiter = torch.clamp(
            allowed_negative_transfer / (negative_transfer + 1e-12),
            max=1.0,
        )
        qb_x = transfer_limiter * qb_x_raw
        qb_y = transfer_limiter * qb_y_raw
        qx = qd_x + qb_x
        qy = qd_y + qb_y
        pi_b = -(qb_x * omega_x + qb_y * omega_y)
        pi = pi_d + pi_b

        eigenvalue_min = 1.0 - bounded_radius
        eigenvalue_max = 1.0 + bounded_radius
        anisotropy_condition = eigenvalue_max / eigenvalue_min
        # Since n_theta has norm <= 1, this is the exact analytical ratio to
        # the composite support scale.  A separate diagnostic below retains
        # the traditional ratio to |q_d| for interpretation only.
        backscatter_ratio = torch.abs(
            effective_beta * backscatter_gate
        ) * direction_norm * transfer_limiter
        qb_norm = torch.sqrt(qb_x.square() + qb_y.square() + 1e-16)
        return {
            "qx": qx,
            "qy": qy,
            "qd_x": qd_x,
            "qd_y": qd_y,
            "qb_x": qb_x,
            "qb_y": qb_y,
            "nu_t": nu_d,
            "nu_d": nu_d,
            "c_theta": c_theta,
            "A11": a11,
            "A12": a12,
            "A22": a22,
            "A_eigenvalue_min": eigenvalue_min,
            "A_eigenvalue_max": eigenvalue_max,
            "A_condition_number": anisotropy_condition,
            "backscatter_gate": backscatter_gate,
            "backscatter_ratio": backscatter_ratio,
            "backscatter_to_dissipation_ratio": qb_norm / (qd_norm + 1e-12),
            "backscatter_support_norm": support_norm,
            "gradient_support_x": gradient_support_x,
            "gradient_support_y": gradient_support_y,
            "gradient_support_norm": gradient_support_norm,
            "gradient_support_coefficient": gradient_coefficient,
            "backscatter_transfer_limiter": transfer_limiter,
            "local_backscatter_transfer_ratio": torch.relu(-pi_b) / (
                transfer_support + 1e-12
            ),
            "backscatter_factor": torch.as_tensor(
                self.backscatter_factor, device=omega.device, dtype=omega.dtype
            ),
            "pi_d": pi_d,
            "pi_b": pi_b,
            "pi": pi,
        }


class SpectralVorticitySolver(nn.Module):
    """Differentiable periodic pseudo-spectral vorticity integrator."""

    def __init__(
        self,
        config: Config,
        ny: int,
        nx: int,
        dx: float,
        dy: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.ny = ny
        self.nx = nx
        kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=dx, device=device)
        ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=dy, device=device)
        kky, kkx = torch.meshgrid(ky, kx, indexing="ij")
        k_squared = kkx.square() + kky.square()
        inverse_k_squared = torch.zeros_like(k_squared)
        inverse_k_squared[k_squared > 0.0] = 1.0 / k_squared[k_squared > 0.0]

        fraction = config.paper.dealias_fraction
        mode_x = torch.fft.fftfreq(nx, d=1.0 / nx, device=device).abs()
        mode_y = torch.fft.fftfreq(ny, d=1.0 / ny, device=device).abs()
        cutoff_x = fraction * (nx // 2)
        cutoff_y = fraction * (ny // 2)
        dealias = (mode_y[:, None] <= cutoff_y) & (mode_x[None, :] <= cutoff_x)
        self.register_buffer("kx", kkx.to(dtype=dtype), persistent=False)
        self.register_buffer("ky", kky.to(dtype=dtype), persistent=False)
        self.register_buffer("k_squared", k_squared.to(dtype=dtype), persistent=False)
        self.register_buffer(
            "inverse_k_squared", inverse_k_squared.to(dtype=dtype), persistent=False
        )
        self.register_buffer("dealias", dealias, persistent=False)
        self.closure = SPNSGSClosure(config).to(device=device, dtype=dtype)
        # This scalar is deliberately not trainable. It may be accepted only
        # by disjoint sparse validation after training.
        self.closure_output_scale = 1.0

    def set_closure_output_scale(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("closure output scale must be finite and non-negative")
        self.closure_output_scale = value

    def project(self, field: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(field)
        spectrum = (spectrum * self.dealias).clone()
        spectrum[..., 0, 0] = 0.0
        return torch.fft.ifft2(spectrum).real

    def derivative(self, field: torch.Tensor, axis: str) -> torch.Tensor:
        spectrum = torch.fft.fft2(field)
        wave_number = self.kx if axis == "x" else self.ky
        return torch.fft.ifft2(1j * wave_number * spectrum).real

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(-self.k_squared * torch.fft.fft2(field)).real

    def state(self, omega: torch.Tensor) -> dict[str, torch.Tensor]:
        omega_hat = torch.fft.fft2(omega) * self.dealias
        psi_hat = (omega_hat * self.inverse_k_squared).clone()
        psi_hat[..., 0, 0] = 0.0
        psi = torch.fft.ifft2(psi_hat).real
        u = torch.fft.ifft2(1j * self.ky * psi_hat).real
        v = torch.fft.ifft2(-1j * self.kx * psi_hat).real
        omega_x = torch.fft.ifft2(1j * self.kx * omega_hat).real
        omega_y = torch.fft.ifft2(1j * self.ky * omega_hat).real
        laplacian_omega = torch.fft.ifft2(-self.k_squared * omega_hat).real
        u_x = self.derivative(u, "x")
        u_y = self.derivative(u, "y")
        v_x = self.derivative(v, "x")
        v_y = self.derivative(v, "y")
        strain = torch.sqrt(
            (u_x - v_y).square() + (u_y + v_x).square() + 1e-12
        )
        return {
            "omega": omega,
            "psi": psi,
            "u": u,
            "v": v,
            "omega_x": omega_x,
            "omega_y": omega_y,
            "laplacian_omega": laplacian_omega,
            "strain": strain,
            "u_x": u_x,
            "u_y": u_y,
            "v_x": v_x,
            "v_y": v_y,
        }

    def vorticity_forcing(
        self, time_s: float, like: torch.Tensor
    ) -> torch.Tensor:
        """Return the resolved vorticity forcing on the periodic grid."""
        del time_s  # Reserved for future time-dependent forcing laws.
        if not self.config.physics.forcing_enabled:
            return torch.zeros_like(like)
        y = torch.arange(
            self.ny, device=like.device, dtype=like.dtype
        ) * (self.config.domain.ly / self.ny)
        amplitude = self.config.physics.forcing_acceleration_amplitude_m_s2
        wave_number = self.config.physics.forcing_wave_number_rad_m
        forcing_y = (
            -self.config.physics.forcing_filter_gain
            * amplitude
            * wave_number
            * torch.cos(wave_number * y)
        )
        return forcing_y[:, None].expand_as(like)

    def rhs(self, omega: torch.Tensor, time_s: float) -> dict[str, torch.Tensor]:
        state = self.state(omega)
        nonlinear = state["u"] * state["omega_x"] + state["v"] * state["omega_y"]
        nonlinear = self.project(nonlinear)
        closure = self.closure_fields(state)
        divergence_q = self.project(
            self.derivative(closure["qx"], "x")
            + self.derivative(closure["qy"], "y")
        )
        forcing = self.vorticity_forcing(time_s, omega)
        rhs = (
            -nonlinear
            + self.config.physics.kinematic_viscosity_m2_s
            * state["laplacian_omega"]
            - divergence_q
            + forcing
        )
        return {
            **state,
            **closure,
            "div_q": divergence_q,
            "rhs": self.project(rhs),
            "time_s": torch.as_tensor(time_s, device=omega.device),
        }

    def closure_fields(
        self, state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Evaluate the manuscript SGS flux q_d + q_b."""
        closure = self.closure(
            state["omega"],
            state["psi"],
            state["omega_x"],
            state["omega_y"],
            state["strain"],
            state["laplacian_omega"],
            state["u_x"],
            state["u_y"],
            state["v_x"],
            state["v_y"],
        )
        if not self.config.model.sgs_enabled:
            # Formal No-SGS baseline: the filtered equation receives exactly
            # q_sgs=0.  The network is still instantiated only so a frozen
            # checkpoint can be loaded for a common evaluation path.
            closure = {
                **closure,
                "qx": torch.zeros_like(closure["qx"]),
                "qy": torch.zeros_like(closure["qy"]),
                "qd_x": torch.zeros_like(closure["qd_x"]),
                "qd_y": torch.zeros_like(closure["qd_y"]),
                "qb_x": torch.zeros_like(closure["qb_x"]),
                "qb_y": torch.zeros_like(closure["qb_y"]),
                "pi_d": torch.zeros_like(closure["pi_d"]),
                "pi_b": torch.zeros_like(closure["pi_b"]),
                "pi": torch.zeros_like(closure["pi"]),
            }
        scale = float(self.closure_output_scale)
        if scale != 1.0:
            # Scale the observable conservative closure as a whole.  q and Pi
            # remain algebraically consistent, and div(q) in rhs therefore
            # receives exactly the same scale.  Internal decomposition fields
            # are kept unscaled as diagnostics of the learned model itself.
            closure = {
                **closure,
                "qx": scale * closure["qx"],
                "qy": scale * closure["qy"],
                "pi": scale * closure["pi"],
            }
        return {
            **closure,
        }

    def step(self, omega: torch.Tensor, time_s: float, dt: float) -> torch.Tensor:
        if self.config.paper.integrator.lower() != "rk4":
            raise ValueError("the paper solver currently supports integrator='rk4' only")
        k1 = self.rhs(omega, time_s)["rhs"]
        k2 = self.rhs(omega + 0.5 * dt * k1, time_s + 0.5 * dt)["rhs"]
        k3 = self.rhs(omega + 0.5 * dt * k2, time_s + 0.5 * dt)["rhs"]
        k4 = self.rhs(omega + dt * k3, time_s + dt)["rhs"]
        return self.project(omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    def advance(
        self, omega: torch.Tensor, time_s: float, snapshot_dt: float
    ) -> torch.Tensor:
        substeps = self.config.paper.substeps_per_snapshot
        dt = snapshot_dt / substeps
        for index in range(substeps):
            omega = self.step(omega, time_s + index * dt, dt)
        return omega
