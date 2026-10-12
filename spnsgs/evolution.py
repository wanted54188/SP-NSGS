from __future__ import annotations
import csv,json,math,random,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .data import Snapshot,inspect_reference,load_observation_snapshots
from .solver import SpectralVorticitySolver
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

