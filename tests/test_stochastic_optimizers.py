import unittest

from src.constructive_number import ConstructiveNumber
from src.objectives import Ackley2D, Step2D
from src.stochastic_optimizers import particle_swarm_optimization, simulated_annealing


class StochasticOptimizerTests(unittest.TestCase):
    def test_simulated_annealing_improves_step(self) -> None:
        objective = Step2D()
        start = [ConstructiveNumber.from_real(4.2, 1e-4), ConstructiveNumber.from_real(-3.7, 1e-4)]
        baseline = float(objective.value(start))
        objective.reset_counters()
        result = simulated_annealing(objective, start, max_iter=120)
        self.assertLessEqual(result.resolved_value, baseline)

    def test_particle_swarm_improves_ackley(self) -> None:
        objective = Ackley2D()
        start = [ConstructiveNumber.from_real(3.0, 1e-4), ConstructiveNumber.from_real(-3.0, 1e-4)]
        baseline = float(objective.value(start))
        objective.reset_counters()
        result = particle_swarm_optimization(objective, start, max_iter=80)
        self.assertLess(result.resolved_value, baseline)


if __name__ == "__main__":
    unittest.main()
