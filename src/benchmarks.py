from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np

from src.constructive_number import ConstructiveNumber

Point = Iterable[ConstructiveNumber]
FloatObjective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BenchmarkFunction:
    name: str
    bounds: tuple[tuple[float, float], tuple[float, float]]
    optimum: tuple[float, float]
    optimum_value: float
    objective: FloatObjective
    start: tuple[float, float] | None = None

    def evaluate_float(self, point: np.ndarray | Iterable[float]) -> float:
        x = np.asarray(point, dtype=float)
        if x.shape != (2,):
            raise ValueError(f"{self.name} expects a two-dimensional point")
        value = float(self.objective(x))
        if not math.isfinite(value):
            return float("inf")
        return value

    def evaluate(self, point: Point) -> ConstructiveNumber:
        values = list(point)
        mids = np.array([v.mid for v in values], dtype=float)
        eps = float(np.linalg.norm([v.epsilon for v in values], ord=2))
        value = self.evaluate_float(mids)
        return ConstructiveNumber.from_value_eps(value, eps)

    def clamp(self, point: np.ndarray) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        return np.clip(np.asarray(point, dtype=float), lo, hi)

    def default_start(self) -> np.ndarray:
        if self.start is not None:
            return np.asarray(self.start, dtype=float)
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        return lo + 0.73 * (hi - lo)


def _rastrigin(v: np.ndarray) -> float:
    return 20.0 + float(np.sum(v * v - 10.0 * np.cos(2.0 * math.pi * v)))


def _ackley(v: np.ndarray) -> float:
    x, y = v
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x * x + y * y)))
        - math.exp(0.5 * (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)))
        + math.e
        + 20.0
    )


def _sphere(v: np.ndarray) -> float:
    return float(np.sum(v * v))


def _rosenbrock(v: np.ndarray) -> float:
    x, y = v
    return 100.0 * (y - x * x) ** 2 + (x - 1.0) ** 2


def _beale(v: np.ndarray) -> float:
    x, y = v
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y ** 3) ** 2


def _goldstein_price(v: np.ndarray) -> float:
    x, y = v
    a = 1.0 + (x + y + 1.0) ** 2 * (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y + 6.0 * x * y + 3.0 * y * y)
    b = 30.0 + (2.0 * x - 3.0 * y) ** 2 * (18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y - 36.0 * x * y + 27.0 * y * y)
    return a * b


def _booth(v: np.ndarray) -> float:
    x, y = v
    return (x + 2.0 * y - 7.0) ** 2 + (2.0 * x + y - 5.0) ** 2


def _bukin6(v: np.ndarray) -> float:
    x, y = v
    return 100.0 * math.sqrt(abs(y - 0.01 * x * x)) + 0.01 * abs(x + 10.0)


def _matyas(v: np.ndarray) -> float:
    x, y = v
    return 0.26 * (x * x + y * y) - 0.48 * x * y


def _levi13(v: np.ndarray) -> float:
    x, y = v
    return math.sin(3.0 * math.pi * x) ** 2 + (x - 1.0) ** 2 * (1.0 + math.sin(3.0 * math.pi * y) ** 2) + (y - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * y) ** 2)


def _himmelblau(v: np.ndarray) -> float:
    x, y = v
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def _three_hump_camel(v: np.ndarray) -> float:
    x, y = v
    return 2.0 * x * x - 1.05 * x ** 4 + x ** 6 / 6.0 + x * y + y * y


def _easom(v: np.ndarray) -> float:
    x, y = v
    return -math.cos(x) * math.cos(y) * math.exp(-((x - math.pi) ** 2 + (y - math.pi) ** 2))


def _cross_in_tray(v: np.ndarray) -> float:
    x, y = v
    inner = abs(math.sin(x) * math.sin(y) * math.exp(abs(100.0 - math.sqrt(x * x + y * y) / math.pi)))
    return -0.0001 * (inner + 1.0) ** 0.1


def _eggholder(v: np.ndarray) -> float:
    x, y = v
    return -(y + 47.0) * math.sin(math.sqrt(abs(x / 2.0 + y + 47.0))) - x * math.sin(math.sqrt(abs(x - (y + 47.0))))


def _holder_table(v: np.ndarray) -> float:
    x, y = v
    return -abs(math.sin(x) * math.cos(y) * math.exp(abs(1.0 - math.sqrt(x * x + y * y) / math.pi)))


def _mccormick(v: np.ndarray) -> float:
    x, y = v
    return math.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1.0


def _schaffer2(v: np.ndarray) -> float:
    x, y = v
    return 0.5 + (math.sin(x * x - y * y) ** 2 - 0.5) / (1.0 + 0.001 * (x * x + y * y)) ** 2


def _schaffer4(v: np.ndarray) -> float:
    x, y = v
    return 0.5 + (math.cos(math.sin(abs(x * x - y * y))) ** 2 - 0.5) / (1.0 + 0.001 * (x * x + y * y)) ** 2


def _desmos_constructive(v: np.ndarray, d: float = 0.047) -> float:
    x, y = v
    a = (x * (round(math.sin(10.0 * y)) + 2.0)) ** 2 + y - 10.0
    b = x + (y * (round(math.sin(7.0 * x)) + 2.0)) ** 2 - 7.0
    return d * (a * a + b * b)


BENCHMARKS: list[BenchmarkFunction] = [
    BenchmarkFunction("Rastrigin", ((-5.12, 5.12), (-5.12, 5.12)), (0.0, 0.0), 0.0, _rastrigin, (4.1, -3.7)),
    BenchmarkFunction("Ackley", ((-5.0, 5.0), (-5.0, 5.0)), (0.0, 0.0), 0.0, _ackley, (3.6, -4.1)),
    BenchmarkFunction("Sphere", ((-10.0, 10.0), (-10.0, 10.0)), (0.0, 0.0), 0.0, _sphere, (7.0, -6.0)),
    BenchmarkFunction("Rosenbrock", ((-3.0, 3.0), (-3.0, 3.0)), (1.0, 1.0), 0.0, _rosenbrock, (-1.7, 2.0)),
    BenchmarkFunction("Beale", ((-4.5, 4.5), (-4.5, 4.5)), (3.0, 0.5), 0.0, _beale, (-3.5, 4.0)),
    BenchmarkFunction("GoldsteinPrice", ((-2.0, 2.0), (-2.0, 2.0)), (0.0, -1.0), 3.0, _goldstein_price, (1.4, 1.7)),
    BenchmarkFunction("Booth", ((-10.0, 10.0), (-10.0, 10.0)), (1.0, 3.0), 0.0, _booth, (-8.0, 8.0)),
    BenchmarkFunction("Bukin6", ((-15.0, -5.0), (-3.0, 3.0)), (-10.0, 1.0), 0.0, _bukin6, (-14.0, -2.2)),
    BenchmarkFunction("Matyas", ((-10.0, 10.0), (-10.0, 10.0)), (0.0, 0.0), 0.0, _matyas, (8.0, -7.0)),
    BenchmarkFunction("Levi13", ((-10.0, 10.0), (-10.0, 10.0)), (1.0, 1.0), 0.0, _levi13, (-7.0, 8.0)),
    BenchmarkFunction("Himmelblau", ((-5.0, 5.0), (-5.0, 5.0)), (3.0, 2.0), 0.0, _himmelblau, (-4.0, 4.0)),
    BenchmarkFunction("ThreeHumpCamel", ((-5.0, 5.0), (-5.0, 5.0)), (0.0, 0.0), 0.0, _three_hump_camel, (4.0, -4.0)),
    BenchmarkFunction("Easom", ((-100.0, 100.0), (-100.0, 100.0)), (math.pi, math.pi), -1.0, _easom, (3.0, 3.0)),
    BenchmarkFunction("CrossInTray", ((-10.0, 10.0), (-10.0, 10.0)), (1.34941, 1.34941), -2.06261, _cross_in_tray, (8.0, -8.0)),
    BenchmarkFunction("Eggholder", ((-512.0, 512.0), (-512.0, 512.0)), (512.0, 404.2319), -959.6407, _eggholder, (-350.0, 300.0)),
    BenchmarkFunction("HolderTable", ((-10.0, 10.0), (-10.0, 10.0)), (8.05502, 9.66459), -19.2085, _holder_table, (-8.0, -8.0)),
    BenchmarkFunction("McCormick", ((-1.5, 4.0), (-3.0, 4.0)), (-0.54719, -1.54719), -1.9133, _mccormick, (3.0, 3.0)),
    BenchmarkFunction("Schaffer2", ((-100.0, 100.0), (-100.0, 100.0)), (0.0, 0.0), 0.0, _schaffer2, (80.0, -70.0)),
    BenchmarkFunction("Schaffer4", ((-100.0, 100.0), (-100.0, 100.0)), (0.0, 1.25313), 0.292579, _schaffer4, (70.0, -80.0)),
    BenchmarkFunction("DesmosConstructive", ((-5.0, 5.0), (-5.0, 5.0)), (0.0, 0.0), 0.0, _desmos_constructive, (3.5, -3.5)),
]
