from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class SpectralSolverTest(unittest.TestCase):
    def _solver(self):
        import torch

        from spnsgs.config import Config
        from spnsgs.solver import SpectralVorticitySolver

        config = Config()
        config.paper.closure_channels = 8
        config.paper.closure_layers = 1
        config.paper.substeps_per_snapshot = 1
        return SpectralVorticitySolver(
            config,
            16,
            16,
            2.0 * torch.pi / 16,
            2.0 * torch.pi / 16,
            torch.device("cpu"),
            torch.float64,
        )

    def _paper_solver(self):
        import torch

        from spnsgs.config import Config
        from spnsgs.solver import SpectralVorticitySolver

        config = Config()
        config.paper.checkpoint_format_version = 7
        config.paper.closure_parameterization = "theory_faithful"
        config.paper.closure_channels = 8
        config.paper.closure_layers = 2
        config.paper.closure_dilations = [1, 2]
        config.paper.substeps_per_snapshot = 1
        return SpectralVorticitySolver(
            config,
            16,
            16,
            2.0 * torch.pi / 16,
            2.0 * torch.pi / 16,
            torch.device("cpu"),
            torch.float64,
        )

    def test_sparse_closure_scale_preserves_conservative_identities(self):
        import torch

        solver = self._paper_solver()
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.sin(2.0 * xx) * torch.cos(3.0 * yy)
        state = solver.state(omega)
        baseline = solver.closure_fields(state)
        div_baseline = solver.derivative(baseline["qx"], "x") + solver.derivative(
            baseline["qy"], "y"
        )
        solver.set_closure_output_scale(0.8)
        calibrated = solver.closure_fields(state)
        div_calibrated = solver.derivative(
            calibrated["qx"], "x"
        ) + solver.derivative(calibrated["qy"], "y")
        self.assertTrue(torch.allclose(calibrated["qx"], 0.8 * baseline["qx"]))
        self.assertTrue(torch.allclose(calibrated["qy"], 0.8 * baseline["qy"]))
        self.assertTrue(torch.allclose(calibrated["pi"], 0.8 * baseline["pi"]))
        self.assertTrue(torch.allclose(div_calibrated, 0.8 * div_baseline))

    def test_theory_flux_is_spd_dissipative_and_backscatter_bounded(self):
        import torch

        solver = self._paper_solver()
        torch.manual_seed(17)
        with torch.no_grad():
            solver.closure.output.weight.normal_(mean=0.0, std=0.02)
            solver.closure.output.bias[4] = 1.5
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        closure = solver.closure_fields(solver.state(omega))
        trace = closure["A11"] + closure["A22"]
        determinant = (
            closure["A11"] * closure["A22"] - closure["A12"].square()
        )
        self.assertTrue(torch.allclose(trace, 2.0 * torch.ones_like(trace)))
        self.assertTrue(torch.all(determinant > 0.0))
        self.assertTrue(torch.all(closure["A_eigenvalue_min"] > 0.0))
        self.assertLessEqual(
            float(torch.max(closure["A_condition_number"])),
            solver.closure.maximum_anisotropy_condition + 1e-10,
        )
        self.assertTrue(torch.all(closure["pi_d"] >= -1e-12))
        self.assertLessEqual(
            float(torch.max(closure["backscatter_ratio"])),
            solver.closure.maximum_backscatter_ratio + 1e-12,
        )
        qb_norm = torch.sqrt(closure["qb_x"].square() + closure["qb_y"].square())
        self.assertTrue(
            torch.all(
                qb_norm
                <= solver.closure.maximum_backscatter_ratio
                * closure["backscatter_support_norm"]
                + 1e-12
            )
        )
        self.assertLessEqual(
            float(torch.max(closure["local_backscatter_transfer_ratio"])),
            solver.closure.maximum_local_backscatter_transfer_ratio + 1e-10,
        )

    def test_iso_baseline_is_exactly_isotropic_and_has_no_backscatter(self):
        import torch

        from spnsgs.config import Config
        from spnsgs.solver import SpectralVorticitySolver

        config = Config()
        config.paper.experiment_variant = "iso_dissipative"
        config.paper.closure_channels = 8
        config.paper.closure_layers = 1
        solver = SpectralVorticitySolver(
            config,
            16,
            16,
            2.0 * torch.pi / 16,
            2.0 * torch.pi / 16,
            torch.device("cpu"),
            torch.float64,
        )
        with torch.no_grad():
            solver.closure.output.weight.normal_(mean=0.0, std=0.02)
            solver.closure.output.bias[4] = 2.0
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        closure = solver.closure_fields(solver.state(omega))
        self.assertTrue(torch.allclose(closure["A11"], torch.ones_like(closure["A11"])))
        self.assertTrue(torch.allclose(closure["A12"], torch.zeros_like(closure["A12"])))
        self.assertTrue(torch.allclose(closure["A22"], torch.ones_like(closure["A22"])))
        self.assertTrue(torch.allclose(closure["qb_x"], torch.zeros_like(closure["qb_x"])))
        self.assertTrue(torch.allclose(closure["qb_y"], torch.zeros_like(closure["qb_y"])))
        self.assertTrue(torch.all(closure["pi_d"] >= -1e-12))

    def test_ablation_rollout_records_instability_without_raising(self):
        import torch

        from spnsgs.paper import _ablation_rollout

        solver = self._paper_solver()
        solver.advance = lambda omega, start, dt: torch.full_like(
            omega, float("nan")
        )
        times = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        states, stable_horizon, failure_time = _ablation_rollout(
            solver, torch.zeros(16, 16, dtype=torch.float64), times
        )
        self.assertEqual(len(states), 1)
        self.assertEqual(stable_horizon, 0.0)
        self.assertAlmostEqual(failure_time, 0.1)

    def test_velocity_is_discretely_incompressible(self) -> None:
        import torch

        solver = self._solver()
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.sin(3.0 * xx + 2.0 * yy)
        state = solver.state(omega)
        divergence = solver.derivative(state["u"], "x") + solver.derivative(
            state["v"], "y"
        )
        self.assertLess(float(torch.max(torch.abs(divergence))), 1e-10)

    def test_conservative_sgs_source_has_zero_mean(self) -> None:
        import torch

        solver = self._solver()
        torch.manual_seed(18)
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        residual = solver.rhs(omega, 0.0)
        self.assertLess(float(torch.abs(torch.mean(residual["div_q"]))), 1e-12)
        self.assertTrue(torch.all(residual["c_theta"] >= 0.0))
        self.assertTrue(torch.isfinite(residual["c_theta"]).all())

    def test_positive_dissipation_coefficient_has_smooth_stability_ceiling(self) -> None:
        import torch

        solver = self._paper_solver()
        with torch.no_grad():
            solver.closure.output.bias[0] = 10.0
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        closure = solver.closure_fields(solver.state(omega))
        self.assertTrue(torch.all(closure["c_theta"] > 0.0))
        self.assertLessEqual(
            float(torch.max(closure["c_theta"])),
            solver.closure.dissipation_coefficient_soft_limit + 1e-12,
        )

    def test_dissipative_branch_is_pointwise_enstrophy_dissipative(self) -> None:
        import torch

        solver = self._paper_solver()
        torch.manual_seed(19)
        with torch.no_grad():
            solver.closure.output.weight.normal_(mean=0.0, std=0.03)
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        closure = solver.closure_fields(solver.state(omega))
        explicit = -(
            closure["qd_x"] * solver.state(omega)["omega_x"]
            + closure["qd_y"] * solver.state(omega)["omega_y"]
        )
        self.assertTrue(torch.allclose(closure["pi_d"], explicit))
        self.assertGreaterEqual(float(torch.min(closure["pi_d"])), -1e-12)

    def test_backscatter_curriculum_switch_is_exact(self) -> None:
        import torch

        solver = self._paper_solver()
        torch.manual_seed(20)
        with torch.no_grad():
            solver.closure.output.weight.normal_(mean=0.0, std=0.03)
            solver.closure.output.bias[4] = 2.0
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        state = solver.state(omega)
        solver.closure.set_backscatter_factor(0.0)
        dissipative_only = solver.closure_fields(state)
        self.assertLess(
            float(torch.max(torch.abs(dissipative_only["qb_x"]))), 1e-14
        )
        self.assertTrue(
            torch.allclose(dissipative_only["qx"], dissipative_only["qd_x"])
        )
        solver.closure.set_backscatter_factor(1.0)
        full = solver.closure_fields(state)
        self.assertGreater(float(torch.linalg.vector_norm(full["qb_x"])), 0.0)
        self.assertLessEqual(
            float(torch.max(full["backscatter_ratio"])),
            solver.closure.maximum_backscatter_ratio + 1e-12,
        )

    def test_transfer_limiter_retains_genuine_local_backscatter(self) -> None:
        import torch

        solver = self._paper_solver()
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        omega = torch.sin(x).repeat(16, 1)
        with torch.no_grad():
            solver.closure.output.weight.zero_()
            solver.closure.output.bias.zero_()
            solver.closure.output.bias[4] = 10.0
            solver.closure.output.bias[5] = 1.0
        closure = solver.closure_fields(solver.state(omega))
        self.assertLess(float(torch.min(closure["pi"])), 0.0)
        self.assertLessEqual(
            float(torch.max(closure["local_backscatter_transfer_ratio"])),
            solver.closure.maximum_local_backscatter_transfer_ratio + 1e-10,
        )

    def test_gradient_support_keeps_backscatter_expressive_when_qd_is_small(self):
        import torch

        solver = self._paper_solver()
        torch.manual_seed(23)
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        with torch.no_grad():
            solver.closure.output.weight.zero_()
            solver.closure.output.bias.zero_()
            solver.closure.output.bias[0] = -20.0
            solver.closure.output.bias[4] = 2.0
        closure = solver.closure_fields(solver.state(omega))
        qd_norm = torch.linalg.vector_norm(
            torch.stack([closure["qd_x"], closure["qd_y"]])
        )
        qb_norm = torch.linalg.vector_norm(
            torch.stack([closure["qb_x"], closure["qb_y"]])
        )
        self.assertGreater(float(closure["gradient_support_norm"].mean()), 0.0)
        self.assertGreater(float(qb_norm), float(qd_norm))

    def test_theory_output_head_backpropagates_through_rollout(self) -> None:
        import torch

        solver = self._paper_solver()
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.sin(2.0 * xx + yy) + 0.2 * torch.cos(xx - 3.0 * yy)
        with torch.no_grad():
            solver.closure.output.bias[4] = 0.5
        prediction = solver.advance(omega, 0.0, 0.01)
        torch.mean(prediction.square()).backward()
        gradient = solver.closure.output.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        active_heads = torch.sum(torch.abs(gradient), dim=(1, 2, 3))
        self.assertTrue(torch.all(active_heads > 0.0))

    def test_invariant_v2_features_are_finite_and_differentiable(self) -> None:
        import torch

        from spnsgs.config import Config
        from spnsgs.solver import SpectralVorticitySolver

        config = Config()
        config.paper.closure_feature_set = "invariant_v2"
        config.paper.closure_channels = 8
        config.paper.closure_layers = 1
        config.paper.substeps_per_snapshot = 1
        solver = SpectralVorticitySolver(
            config,
            16,
            16,
            2.0 * torch.pi / 16,
            2.0 * torch.pi / 16,
            torch.device("cpu"),
            torch.float64,
        )
        self.assertEqual(solver.closure.input.in_channels, 10)
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        prediction = solver.advance(omega, 0.0, 0.01)
        torch.mean(prediction.square()).backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(solver.closure.input.weight.grad)

    def test_mixed_gradient_features_feed_theory_head(self) -> None:
        import torch

        from spnsgs.config import Config
        from spnsgs.solver import SpectralVorticitySolver

        config = Config()
        config.paper.closure_parameterization = "theory_faithful"
        config.paper.closure_feature_set = "mixed_gradient_v3"
        config.paper.closure_channels = 8
        config.paper.closure_layers = 1
        solver = SpectralVorticitySolver(
            config,
            16,
            16,
            2.0 * torch.pi / 16,
            2.0 * torch.pi / 16,
            torch.device("cpu"),
            torch.float64,
        )
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        state = solver.state(omega)
        closure = solver.closure_fields(state)
        self.assertEqual(solver.closure.input.in_channels, 12)
        self.assertEqual(solver.closure.output.out_channels, 7)
        self.assertTrue(torch.isfinite(closure["qx"]).all())
        self.assertTrue(torch.isfinite(closure["qy"]).all())
        divergence = solver.project(
            solver.derivative(closure["qx"], "x")
            + solver.derivative(closure["qy"], "y")
        )
        self.assertLess(float(torch.abs(torch.mean(divergence))), 1e-12)

    def test_time_varying_sparse_subset_preserves_budget_and_parent_set(self) -> None:
        import numpy as np
        import torch

        from spnsgs.paper import _time_varying_sparse_subset

        parent = np.arange(0, 16 * 16, 3, dtype=np.int64)
        omega = torch.randn(5, 16, 16, dtype=torch.float64)
        first = _time_varying_sparse_subset(
            parent, omega, 12, 16, 16, 0.5, np.random.default_rng(2026)
        )
        second = _time_varying_sparse_subset(
            parent, omega, 12, 16, 16, 0.5, np.random.default_rng(2026)
        )
        self.assertEqual(first.shape, (5, 12))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(np.isin(first, parent)))
        self.assertTrue(all(np.unique(row).size == 12 for row in first))

    def test_active_sparse_subset_has_normalised_stratum_weights(self) -> None:
        import numpy as np
        import torch

        from spnsgs.paper import _time_varying_sparse_subset_with_weights

        parent = np.arange(100, dtype=np.int64)
        omega = torch.zeros((2, 10, 10), dtype=torch.float64)
        omega[0].reshape(-1)[75:] = torch.linspace(0.0, 10.0, 25)
        selected, weights = _time_varying_sparse_subset_with_weights(
            parent,
            omega,
            20,
            10,
            10,
            0.5,
            0.25,
            np.random.default_rng(22),
        )
        self.assertEqual(selected.shape, (2, 20))
        self.assertEqual(weights.shape, (2, 20))
        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
        self.assertTrue(np.all(weights > 0.0))

    def test_cholesky_parameterisation_remains_spd_for_extreme_outputs(self) -> None:
        import torch

        solver = self._paper_solver()
        with torch.no_grad():
            solver.closure.output.bias[1] = -50.0
            solver.closure.output.bias[2] = 50.0
            solver.closure.output.bias[3] = -50.0
        omega = solver.project(torch.randn(16, 16, dtype=torch.float64))
        closure = solver.closure_fields(solver.state(omega))
        determinant = (
            closure["A11"] * closure["A22"] - closure["A12"].square()
        )
        self.assertTrue(torch.all(determinant > 0.0))
        self.assertTrue(torch.all(closure["A_eigenvalue_min"] > 0.0))
        self.assertLessEqual(
            float(torch.max(closure["A_condition_number"])),
            solver.closure.maximum_anisotropy_condition + 1e-10,
        )
        self.assertTrue(
            torch.allclose(
                closure["A11"] + closure["A22"],
                2.0 * torch.ones_like(closure["A11"]),
            )
        )

    def test_rollout_is_finite_and_differentiable(self) -> None:
        import torch

        solver = self._solver()
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.sin(2.0 * xx) + 0.2 * torch.cos(3.0 * yy)
        prediction = solver.advance(omega, 0.0, 0.01)
        loss = torch.mean(prediction.square())
        loss.backward()
        self.assertTrue(torch.isfinite(prediction).all())
        gradients = [
            parameter.grad
            for parameter in solver.closure.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_sparse_nudging_reduces_sensor_innovation(self) -> None:
        import torch

        from spnsgs.paper import _SparseVorticityNudger

        solver = self._solver()
        indices = torch.arange(0, 16 * 16, 11, dtype=torch.long)
        nudger = _SparseVorticityNudger(solver, indices)
        omega = torch.zeros(16, 16, dtype=torch.float64)
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        target = torch.sin(2.0 * xx + yy) + 0.2 * torch.cos(xx - 2.0 * yy)
        before = torch.mean(
            (omega.reshape(-1)[indices] - target.reshape(-1)[indices]).square()
        )
        after_field = nudger(omega, target)
        after = torch.mean(
            (
                after_field.reshape(-1)[indices]
                - target.reshape(-1)[indices]
            ).square()
        )
        self.assertLess(float(after), float(before))

    def test_multifield_nudging_reduces_velocity_and_vorticity_error(self) -> None:
        import torch

        from spnsgs.paper import _SparseVorticityNudger

        solver = self._solver()
        indices = torch.arange(0, 16 * 16, 9, dtype=torch.long)
        nudger = _SparseVorticityNudger(solver, indices)
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        target_omega = torch.sin(2.0 * xx + yy) + 0.3 * torch.cos(xx - 3.0 * yy)
        target = solver.state(target_omega)
        omega = torch.zeros_like(target_omega)

        def sensor_error(field: torch.Tensor) -> torch.Tensor:
            state = solver.state(field)
            return (
                torch.mean((state["omega"].reshape(-1)[indices] - target["omega"].reshape(-1)[indices]).square())
                + torch.mean((state["u"].reshape(-1)[indices] - target["u"].reshape(-1)[indices]).square())
                + torch.mean((state["v"].reshape(-1)[indices] - target["v"].reshape(-1)[indices]).square())
            )

        before = sensor_error(omega)
        corrected = nudger(omega, target["omega"], target["u"], target["v"])
        after = sensor_error(corrected)
        self.assertLess(float(after), float(before))

    def test_periodic_sensor_pairs_reuse_only_sparse_sensor_entries(self) -> None:
        import torch

        from spnsgs.paper import _periodic_nearest_sensor_pairs

        # Flat indices 0 and 15 are periodic x-neighbours on a 16 x 16 grid.
        indices = torch.tensor([0, 15, 8, 8 * 16], dtype=torch.long)
        pairs = _periodic_nearest_sensor_pairs(indices, 16, 16, neighbours=1)
        self.assertEqual(pairs.ndim, 2)
        self.assertEqual(pairs.shape[1], 2)
        self.assertTrue(torch.all(pairs >= 0))
        self.assertTrue(torch.all(pairs < indices.numel()))
        pair_set = {tuple(pair) for pair in pairs.tolist()}
        self.assertIn((0, 1), pair_set)

    def test_periodic_fourier_sensor_weights_form_quadrature(self) -> None:
        import torch

        from spnsgs.paper import _periodic_fourier_quadrature_weights

        indices = torch.tensor([0, 3, 12, 15], dtype=torch.long)
        weights = _periodic_fourier_quadrature_weights(
            indices, 4, 4, maximum_mode=1
        )
        self.assertEqual(tuple(weights.shape), (4,))
        self.assertTrue(torch.all(weights > 0.0))
        self.assertAlmostEqual(float(torch.sum(weights)), 1.0, places=7)

    def test_sparse_increment_loss_is_zero_for_exact_sensor_field(self) -> None:
        import torch

        from spnsgs.paper import _periodic_nearest_sensor_pairs, _sensor_loss

        solver = self._solver()
        x = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        y = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = solver.project(torch.sin(2.0 * xx + yy) + 0.2 * torch.cos(3.0 * yy))
        state = solver.state(omega)
        indices = torch.arange(0, 16 * 16, 7, dtype=torch.long)
        pairs = _periodic_nearest_sensor_pairs(indices, 16, 16, neighbours=3)
        zeros = torch.zeros((1, 16, 16), dtype=torch.float64)
        reference = {
            "omega": omega.unsqueeze(0),
            "u": state["u"].unsqueeze(0),
            "v": state["v"].unsqueeze(0),
            "div_q_sgs": zeros,
            "pi_sgs": zeros,
        }
        _, components = _sensor_loss(
            solver,
            omega,
            0,
            indices,
            reference,
            {"omega": 1.0, "velocity": 1.0},
            increment_pairs=pairs,
        )
        self.assertLess(float(torch.abs(components["omega_increment"])), 1e-14)

    def test_validation_sensor_subsets_are_nested_and_deterministic(self) -> None:
        import torch

        from spnsgs.validation import _nested_sensor_subset

        parent = torch.arange(0, 256, 3, dtype=torch.long)
        first = _nested_sensor_subset(parent, 20, seed=2026)
        repeated = _nested_sensor_subset(parent, 20, seed=2026)
        larger = _nested_sensor_subset(parent, 40, seed=2026)
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.isin(first, parent).all())
        self.assertTrue(torch.isin(first, larger).all())
        self.assertEqual(first.numel(), 20)


if __name__ == "__main__":
    unittest.main()
