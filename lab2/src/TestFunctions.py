from __future__ import annotations

import math
from typing import Sequence, Union

from .ConstructiveNumber import ConstructiveNumber


Scalar = Union[int, float, ConstructiveNumber]
Vector2 = Sequence[Scalar]


def _to_float(value: Scalar, alpha: float = 0.5) -> float:
    if isinstance(value, ConstructiveNumber):
        return value.to_real(alpha)
    return float(value)


def _xy(point: Vector2, alpha: float = 0.5) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError()
    return _to_float(point[0], alpha), _to_float(point[1], alpha)


def matyas(point: Vector2, alpha: float = 0.5) -> float:
    x, y = _xy(point, alpha)
    return 0.26 * (x * x + y * y) - 0.48 * x * y


def himmelblau(point: Vector2, alpha: float = 0.5) -> float:
    x, y = _xy(point, alpha)
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def rastrigin(point: Vector2, alpha: float = 0.5, A: float = 10.0) -> float:
    x, y = _xy(point, alpha)
    return 2.0 * A + (x * x - A * math.cos(2.0 * math.pi * x)) + (y * y - A * math.cos(2.0 * math.pi * y))


def desmos_round_sin(point: Vector2, alpha: float = 0.5) -> float:
    x, y = _xy(point, alpha)
    term1 = ((x * (round(math.sin(10.0 * y)) + 2.0)) ** 2 + y - 10.0) ** 2
    term2 = (x + (y * (round(math.sin(7.0 * x)) + 2.0)) ** 2 - 7.0) ** 2
    return term1 + term2
