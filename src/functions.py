from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from constructive_number import ConstructiveNumber


def is_constructive(x) -> bool:
    return isinstance(x, ConstructiveNumber)


def zero_like(example):
    return ConstructiveNumber.from_real(0.0, 0.0) if is_constructive(example) else 0.0


def one_like(example):
    return ConstructiveNumber.from_real(1.0, 0.0) if is_constructive(example) else 1.0


def to_real_scalar(x, alpha: float = 0.5) -> float:
    return x.value(alpha) if is_constructive(x) else float(x)


def to_real_vector(x: Sequence, alpha: float = 0.5) -> np.ndarray:
    return np.array([to_real_scalar(v, alpha=alpha) for v in x], dtype=float)


def vector_epsilons(x: Sequence) -> np.ndarray:
    out = []
    for v in x:
        if is_constructive(v):
            out.append(float(v.epsilon))
        else:
            out.append(0.0)
    return np.array(out, dtype=float)


def _sin(x):
    return x.sin() if is_constructive(x) else math.sin(x)


def _cos(x):
    return x.cos() if is_constructive(x) else math.cos(x)


def _exp(x):
    return x.exp() if is_constructive(x) else math.exp(x)


def _sqrt(x):
    return x.sqrt() if is_constructive(x) else math.sqrt(x)


def _abs(x):
    return x.abs() if is_constructive(x) else abs(x)


def _round(x):
    return x.round() if is_constructive(x) else float(round(x))


@dataclass
class EvaluationCounters:
    value_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0

    def reset(self) -> None:
        self.value_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0


class BlackBoxFunction:
    def __init__(self, name: str, dimension: int, optimum: Sequence[float], optimum_value: float = 0.0,
                 bounds: Sequence[Sequence[float]] | None = None):
        self.name = name
        self.dimension = dimension
        self.optimum = np.array(optimum, dtype=float)
        self._optimum_value = float(optimum_value)
        if bounds is None:
            bounds = [[-5.0, 5.0]] * dimension
        self.bounds = np.array(bounds, dtype=float)
        self.counters = EvaluationCounters()

    def reset_counters(self) -> None:
        self.counters.reset()

    def __call__(self, x: Sequence):
        return self.value(x)

    def value(self, x: Sequence):
        raise NotImplementedError

    def optimum_value(self) -> float:
        return self._optimum_value

    def section_value(self, x1: float, x2: float) -> float:
        point = self.optimum.copy()
        point[0] = x1
        if self.dimension > 1:
            point[1] = x2
        return float(self.value(point))


class RastriginFunction(BlackBoxFunction):
    def __init__(self, dimension: int = 2, A: float = 10.0):
        super().__init__(
            name=f"Rastrigin-{dimension}D",
            dimension=dimension,
            optimum=[0.0] * dimension,
            optimum_value=0.0,
            bounds=[[-5.12, 5.12]] * dimension,
        )
        self.A = A

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        total = ConstructiveNumber.from_real(self.A * self.dimension, 0.0) if is_constructive(x[0]) else self.A * self.dimension
        two_pi = 2 * math.pi
        for xi in x:
            total = total + xi * xi - self.A * _cos(two_pi * xi)
        return total


class AckleyFunction(BlackBoxFunction):
    def __init__(self):
        super().__init__(
            name="Ackley-2D",
            dimension=2,
            optimum=[0.0, 0.0],
            optimum_value=0.0,
            bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        )

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        x1, x2 = x[0], x[1]
        squared = x1 * x1 + x2 * x2
        sqrt_part = _sqrt(0.5 * squared)
        first = -20.0 * _exp(-0.2 * sqrt_part)
        two_pi = 2 * math.pi
        cos_part = _cos(two_pi * x1) + _cos(two_pi * x2)
        second = -_exp(0.5 * cos_part)
        return first + second + (math.e + 20.0)


class HimmelblauFunction(BlackBoxFunction):
    def __init__(self):
        super().__init__(
            name="Himmelblau-2D",
            dimension=2,
            optimum=[3.0, 2.0],
            optimum_value=0.0,
            bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        )

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        x1, x2 = x[0], x[1]
        term1 = x1 * x1 + x2 - 11
        term2 = x1 + x2 * x2 - 7
        return term1 * term1 + term2 * term2


class EggholderFunction(BlackBoxFunction):
    def __init__(self):
        super().__init__(
            name="Eggholder-2D",
            dimension=2,
            optimum=[512.0, 404.2319],
            optimum_value=-959.6407,
            bounds=[[-512.0, 512.0], [-512.0, 512.0]],
        )

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        x1, x2 = x[0], x[1]
        a = x2 + 47
        first = -a * _sin(_sqrt(_abs(x1 / 2.0 + a)))
        second = -x1 * _sin(_sqrt(_abs(x1 - a)))
        return first + second


class BadieFunction(BlackBoxFunction):
    """
    Из Desmos: badieFunction
    f(x,y) = d * (((x*(round(sin(10y))+2))^2 + y - 10)^2
                + (x + (y*(round(sin(7x))+2))^2 - 7)^2)
    Множители round(sin(...))+2 принимают значения {1,2,3} → функция кусочно-гладкая, но РАЗРЫВНАЯ
    в точках, где аргумент round(sin) проходит через 1/2 или -1/2.
    Это ровно тот случай, ради которого нужны стохастические методы.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__(
            name="Badie-2D",
            dimension=2,
            optimum=[0.0, 0.0],
            optimum_value=0.0,
            bounds=[[-5.0, 5.0], [-5.0, 5.0]],
        )
        self.scale = scale

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        x1, x2 = x[0], x[1]
        c1 = _round(_sin(10 * x2)) + 2
        c2 = _round(_sin(7 * x1)) + 2
        first_inside = (x1 * c1)
        first_inside = first_inside * first_inside + x2 - 10
        first = first_inside * first_inside
        second_inside = (x2 * c2)
        second_inside = x1 + second_inside * second_inside - 7
        second = second_inside * second_inside
        return self.scale * (first + second)


class QuadraticFunction(BlackBoxFunction):
    """Гладкая квадратичная функция (используется для сравнения с lab_01)."""

    def __init__(self, name: str, diagonal: Sequence[float], center: Sequence[float], bounds_pad: float = 6.0):
        diagonal = np.array(diagonal, dtype=float)
        center = np.array(center, dtype=float)
        bounds = [[c - bounds_pad, c + bounds_pad] for c in center]
        super().__init__(
            name=name,
            dimension=len(diagonal),
            optimum=center,
            optimum_value=0.0,
            bounds=bounds,
        )
        self.diagonal = diagonal
        self.center = center

    @property
    def condition_number(self) -> float:
        return float(self.diagonal.max() / self.diagonal.min())

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        diffs = [xi - ci for xi, ci in zip(x, self.center)]
        total = zero_like(diffs[0])
        for coeff, d in zip(self.diagonal, diffs):
            total = total + 0.5 * coeff * d * d
        return total

    def gradient(self, x: Sequence):
        self.counters.gradient_calls += 1
        return [coeff * (xi - ci) for coeff, xi, ci in zip(self.diagonal, x, self.center)]

    def hessian(self, x: Sequence):
        self.counters.hessian_calls += 1
        return np.diag(self.diagonal)


class Rosenbrock3D(BlackBoxFunction):
    def __init__(self):
        super().__init__(
            name="Rosenbrock-3D",
            dimension=3,
            optimum=[1.0, 1.0, 1.0],
            optimum_value=0.0,
            bounds=[[-2.0, 2.0]] * 3,
        )

    def value(self, x: Sequence):
        self.counters.value_calls += 1
        total = zero_like(x[0])
        for i in range(2):
            term1 = x[i + 1] - x[i] * x[i]
            term2 = 1 - x[i]
            total = total + 100 * term1 * term1 + term2 * term2
        return total

    def gradient(self, x: Sequence):
        self.counters.gradient_calls += 1
        x1, x2, x3 = x
        g1 = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1)
        g2 = 200 * (x2 - x1 * x1) - 400 * x2 * (x3 - x2 * x2) - 2 * (1 - x2)
        g3 = 200 * (x3 - x2 * x2)
        return [g1, g2, g3]

    def hessian(self, x: Sequence):
        self.counters.hessian_calls += 1
        x1, x2, x3 = [to_real_scalar(v) for v in x]
        h = np.zeros((3, 3), dtype=float)
        h[0, 0] = 1200 * x1 ** 2 - 400 * x2 + 2
        h[0, 1] = -400 * x1
        h[1, 0] = -400 * x1
        h[1, 1] = 1200 * x2 ** 2 - 400 * x3 + 202
        h[1, 2] = -400 * x2
        h[2, 1] = -400 * x2
        h[2, 2] = 200
        return h


def get_lab02_functions() -> List[BlackBoxFunction]:
    return [
        RastriginFunction(dimension=2),
        AckleyFunction(),
        HimmelblauFunction(),
        EggholderFunction(),
        BadieFunction(),
    ]


def get_lab01_functions() -> List[BlackBoxFunction]:
    quadratic_good = QuadraticFunction(
        name="Quadratic-6D-good",
        diagonal=[0.92, 0.97, 1.00, 1.05, 1.08, 1.12],
        center=[1.0, -2.0, 0.5, 3.0, -1.5, 2.0],
        bounds_pad=6.0,
    )
    quadratic_bad = QuadraticFunction(
        name="Quadratic-4D-bad",
        diagonal=[1.0, 10.0, 50.0, 100.0],
        center=[-1.0, 2.0, -3.0, 0.5],
        bounds_pad=6.0,
    )
    rosenbrock = Rosenbrock3D()
    return [quadratic_good, quadratic_bad, rosenbrock]


def get_default_functions() -> List[BlackBoxFunction]:
    return get_lab02_functions() + get_lab01_functions()
