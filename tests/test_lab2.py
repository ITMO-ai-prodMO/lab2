import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from constructive_number import ConstructiveNumber
from functions import (
    AckleyFunction,
    BadieFunction,
    EggholderFunction,
    HimmelblauFunction,
    RastriginFunction,
)
from stochastic_optimizers import simulated_annealing, particle_swarm


class ConstructiveNumberExtensionsTests(unittest.TestCase):
    def test_sin_full_range_when_interval_long_enough(self):
        cn = ConstructiveNumber.from_real(0.0, 5.0)  # ширина 10, > 2π
        result = cn.sin()
        self.assertAlmostEqual(float(result.a), -1.0, places=6)
        self.assertAlmostEqual(float(result.b), 1.0, places=6)

    def test_sin_max_only(self):
        cn = ConstructiveNumber.from_bounds(0.0, math.pi)  # пересекает π/2 (max), не 3π/2
        result = cn.sin()
        self.assertAlmostEqual(float(result.b), 1.0, places=6)
        self.assertGreaterEqual(float(result.a), 0.0)

    def test_cos_full_range(self):
        cn = ConstructiveNumber.from_bounds(-math.pi, math.pi)
        result = cn.cos()
        self.assertAlmostEqual(float(result.a), -1.0, places=6)
        self.assertAlmostEqual(float(result.b), 1.0, places=6)

    def test_exp_monotone(self):
        cn = ConstructiveNumber.from_bounds(0.0, 1.0)
        result = cn.exp()
        self.assertAlmostEqual(float(result.a), 1.0, places=6)
        self.assertAlmostEqual(float(result.b), math.e, places=6)

    def test_sqrt_clamps_negatives_to_zero(self):
        cn = ConstructiveNumber.from_bounds(-0.0001, 4.0)
        result = cn.sqrt()
        self.assertAlmostEqual(float(result.a), 0.0, places=6)
        self.assertAlmostEqual(float(result.b), 2.0, places=6)

    def test_round_widens_to_integer_endpoints(self):
        cn = ConstructiveNumber.from_bounds(0.4, 0.6)
        result = cn.round()
        self.assertEqual(int(result.a), 0)
        self.assertEqual(int(result.b), 1)


class FunctionAtKnownOptimumTests(unittest.TestCase):
    def test_rastrigin_zero(self):
        f = RastriginFunction(2)
        self.assertAlmostEqual(float(f.value([0.0, 0.0])), 0.0, places=10)

    def test_ackley_zero(self):
        f = AckleyFunction()
        self.assertAlmostEqual(float(f.value([0.0, 0.0])), 0.0, places=6)

    def test_himmelblau_zero(self):
        f = HimmelblauFunction()
        self.assertAlmostEqual(float(f.value([3.0, 2.0])), 0.0, places=10)

    def test_eggholder_known_minimum(self):
        f = EggholderFunction()
        v = float(f.value([512.0, 404.2319]))
        self.assertLess(v, -959.0)

    def test_badie_nonneg(self):
        f = BadieFunction()
        for x, y in [(0.0, 0.0), (1.0, 1.0), (2.0, -3.0), (-1.5, 0.7)]:
            self.assertGreaterEqual(float(f.value([x, y])), 0.0)


class StochasticOptimizerTests(unittest.TestCase):
    def test_pso_finds_himmelblau_minimum(self):
        f = HimmelblauFunction()
        result = particle_swarm(
            f, x0=[0.0, 0.0], epsilon=0.0, n_particles=30, max_iter=80, rng=np.random.default_rng(42)
        )
        self.assertLess(result.final_value, 1e-3)

    def test_pso_eggholder_finds_strong_local(self):
        f = EggholderFunction()
        result = particle_swarm(
            f,
            x0=[0.0, 0.0],
            epsilon=0.0,
            n_particles=50,
            max_iter=120,
            velocity_scale=0.05,
            rng=np.random.default_rng(43),
        )
        self.assertLess(result.final_value, -700.0)

    def test_sa_descends_on_rastrigin(self):
        f = RastriginFunction(2)
        x0 = np.array([3.0, -2.0])
        initial_value = float(f.value(x0))
        result = simulated_annealing(
            f, x0=x0, epsilon=0.0, max_iter=1000, rng=np.random.default_rng(42)
        )
        self.assertLess(result.final_value, initial_value)

    def test_sa_counts_calls(self):
        f = RastriginFunction(2)
        result = simulated_annealing(
            f, x0=[1.0, 1.0], epsilon=0.0, max_iter=50, rng=np.random.default_rng(0)
        )
        # 1 initial + 50 proposals = 51
        self.assertEqual(result.value_calls, 51)
        self.assertEqual(result.iterations, 50)

    def test_results_carry_memory_estimate(self):
        f = RastriginFunction(2)
        sa = simulated_annealing(f, x0=[0.0, 0.0], epsilon=0.0, max_iter=10, rng=np.random.default_rng(0))
        pso = particle_swarm(f, x0=[0.0, 0.0], epsilon=0.0, n_particles=10, max_iter=5, rng=np.random.default_rng(0))
        self.assertEqual(sa.memory_scalars, 2 * 2)
        self.assertEqual(pso.memory_scalars, (3 * 10 + 1) * 2)


class CNPropagationTests(unittest.TestCase):
    def test_ackley_propagates_eps(self):
        f = AckleyFunction()
        x = [
            ConstructiveNumber.from_real(0.5, 1e-4),
            ConstructiveNumber.from_real(-0.3, 1e-4),
        ]
        v = f.value(x)
        # Расширение интервала при ненулевом eps > 0
        self.assertGreater(float(v.epsilon), 0.0)

    def test_badie_eps_propagation_includes_round_jumps(self):
        f = BadieFunction()
        # Эпсилон достаточно большой, чтобы round мог перепрыгнуть
        x = [
            ConstructiveNumber.from_real(0.5, 0.05),
            ConstructiveNumber.from_real(0.5, 0.05),
        ]
        v = f.value(x)
        self.assertGreater(float(v.epsilon), 0.0)


if __name__ == "__main__":
    unittest.main()
