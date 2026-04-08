import unittest

from src.objectives import Ackley2D, DesmosLike3D, Rastrigin2D, Step2D


class ObjectiveTests(unittest.TestCase):
    def test_step_has_zero_at_origin(self) -> None:
        self.assertEqual(float(Step2D().value([0.0, 0.0])), 0.0)

    def test_rastrigin_has_zero_at_origin(self) -> None:
        self.assertAlmostEqual(float(Rastrigin2D().value([0.0, 0.0])), 0.0, places=12)

    def test_ackley_has_zero_at_origin(self) -> None:
        self.assertAlmostEqual(float(Ackley2D().value([0.0, 0.0])), 0.0, places=8)

    def test_desmos_like_has_zero_at_origin(self) -> None:
        self.assertAlmostEqual(float(DesmosLike3D().value([0.0, 0.0, 0.0])), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
