import unittest

from src.constructive_number import ConstructiveNumber
from src.deterministic_optimizers import gradient_descent, newton_method
from src.objectives import Ackley2D


class DeterministicBaselineTests(unittest.TestCase):
    def test_gradient_descent_reduces_ackley(self) -> None:
        objective = Ackley2D()
        start = [ConstructiveNumber.from_real(3.0, 1e-6), ConstructiveNumber.from_real(-3.0, 1e-6)]
        baseline = float(objective.value(start))
        objective.reset_counters()
        result = gradient_descent(objective, start, learning_rate=0.03, max_iter=80)
        self.assertLess(result.resolved_value, baseline)

    def test_newton_reduces_ackley(self) -> None:
        objective = Ackley2D()
        start = [1.0, -1.0]
        baseline = float(objective.value(start))
        objective.reset_counters()
        result = newton_method(objective, start, max_iter=8, damping=0.1)
        self.assertLess(result.resolved_value, baseline)


if __name__ == "__main__":
    unittest.main()
