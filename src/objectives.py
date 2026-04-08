from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


def _to_float(value: object) -> float:
    if hasattr(value, "resolve"):
        return float(value)
    return float(value)


Vector = Sequence[object]


def _ensure_vector(values: Iterable[object]) -> list[object]:
    return list(values)


@dataclass
class ObjectiveFunction:
    name: str
    dimension: int
    optimum_point: list[float]
    optimal_value: float
    bounds: list[tuple[float, float]]
    value_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def value(self, x: Vector) -> object:
        self.value_calls += 1
        return self._value(_ensure_vector(x))

    def gradient(self, x: Vector) -> list[object]:
        self.gradient_calls += 1
        return self._gradient(_ensure_vector(x))

    def hessian(self, x: Vector) -> list[list[object]]:
        self.hessian_calls += 1
        return self._hessian(_ensure_vector(x))

    def reset_counters(self) -> None:
        self.value_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0

    def counters(self) -> dict[str, int]:
        return {
            "value_calls": self.value_calls,
            "gradient_calls": self.gradient_calls,
            "hessian_calls": self.hessian_calls,
        }

    def _value(self, x: list[object]) -> object:
        raise NotImplementedError

    def _gradient(self, x: list[object]) -> list[object]:
        # Central finite differences keep the interface compatible with constructive numbers.
        h = 1e-5
        result = []
        base = [_to_float(v) for v in x]
        for index in range(self.dimension):
            left = base.copy()
            right = base.copy()
            left[index] -= h
            right[index] += h
            value = (self.value(right) - self.value(left)) / (2.0 * h)
            result.append(value)
        return result

    def _hessian(self, x: list[object]) -> list[list[object]]:
        h = 1e-4
        base = [_to_float(v) for v in x]
        matrix = []
        for i in range(self.dimension):
            row = []
            for j in range(self.dimension):
                x_pp = base.copy()
                x_pm = base.copy()
                x_mp = base.copy()
                x_mm = base.copy()
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                value = (self.value(x_pp) - self.value(x_pm) - self.value(x_mp) + self.value(x_mm)) / (4.0 * h * h)
                row.append(value)
            matrix.append(row)
        return matrix


class Step2D(ObjectiveFunction):
    def __init__(self) -> None:
        super().__init__(
            name="step_2d",
            dimension=2,
            optimum_point=[0.0, 0.0],
            optimal_value=0.0,
            bounds=[(-6.0, 6.0), (-6.0, 6.0)],
            metadata={"kind": "discontinuous"},
        )

    def _value(self, x: list[object]) -> object:
        return sum(math.floor(_to_float(xi) + 0.5) ** 2 for xi in x)


class Rastrigin2D(ObjectiveFunction):
    def __init__(self) -> None:
        super().__init__(
            name="rastrigin_2d",
            dimension=2,
            optimum_point=[0.0, 0.0],
            optimal_value=0.0,
            bounds=[(-5.12, 5.12), (-5.12, 5.12)],
            metadata={"kind": "multimodal"},
        )

    def _value(self, x: list[object]) -> object:
        return 20.0 + sum((_to_float(xi) ** 2) - 10.0 * math.cos(2.0 * math.pi * _to_float(xi)) for xi in x)


class Ackley2D(ObjectiveFunction):
    def __init__(self) -> None:
        super().__init__(
            name="ackley_2d",
            dimension=2,
            optimum_point=[0.0, 0.0],
            optimal_value=0.0,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            metadata={"kind": "multimodal"},
        )

    def _value(self, x: list[object]) -> object:
        x1, x2 = (_to_float(v) for v in x)
        term1 = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x1 * x1 + x2 * x2)))
        term2 = -math.exp(0.5 * (math.cos(2.0 * math.pi * x1) + math.cos(2.0 * math.pi * x2)))
        return term1 + term2 + math.e + 20.0


class DesmosLike3D(ObjectiveFunction):
    def __init__(self) -> None:
        super().__init__(
            name="desmos_like_3d",
            dimension=3,
            optimum_point=[0.0, 0.0, 0.0],
            optimal_value=0.0,
            bounds=[(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)],
            metadata={"kind": "nonsmooth"},
        )

    def _value(self, x: list[object]) -> object:
        x1, x2, x3 = (_to_float(v) for v in x)
        ripple = abs(math.sin(3.0 * x1) * math.cos(2.0 * x2)) + abs(math.sin(2.0 * x3))
        bowl = 0.3 * (x1 * x1 + 1.5 * x2 * x2 + 0.7 * x3 * x3)
        ridge = abs(x1 - x2) + 0.5 * abs(x2 + x3)
        return bowl + ridge + ripple
