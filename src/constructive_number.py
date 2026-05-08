from __future__ import annotations

from dataclasses import dataclass
from typing import Union

NumberLike = Union["ConstructiveNumber", float, int]


@dataclass(frozen=True)
class ConstructiveNumber:
    """A simple constructive number represented by a closed interval."""

    left: float
    right: float

    def __post_init__(self) -> None:
        if self.left > self.right:
            left, right = self.right, self.left
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)

    @classmethod
    def from_bounds(cls, a: float, b: float) -> "ConstructiveNumber":
        return cls(min(float(a), float(b)), max(float(a), float(b)))

    @classmethod
    def from_value_eps(cls, value: float, eps: float = 0.0) -> "ConstructiveNumber":
        eps = abs(float(eps))
        value = float(value)
        return cls(value - eps, value + eps)

    @classmethod
    def exact(cls, value: float) -> "ConstructiveNumber":
        return cls.from_value_eps(value, 0.0)

    @property
    def mid(self) -> float:
        return 0.5 * (self.left + self.right)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def epsilon(self) -> float:
        return 0.5 * self.width

    def real(self, alpha: float) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        return self.left + alpha * self.width

    @staticmethod
    def _coerce(other: NumberLike) -> "ConstructiveNumber":
        if isinstance(other, ConstructiveNumber):
            return other
        return ConstructiveNumber.exact(float(other))

    def __add__(self, other: NumberLike) -> "ConstructiveNumber":
        other = self._coerce(other)
        return ConstructiveNumber(self.left + other.left, self.right + other.right)

    __radd__ = __add__

    def __sub__(self, other: NumberLike) -> "ConstructiveNumber":
        other = self._coerce(other)
        return ConstructiveNumber(self.left - other.right, self.right - other.left)

    def __rsub__(self, other: NumberLike) -> "ConstructiveNumber":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other: NumberLike) -> "ConstructiveNumber":
        other = self._coerce(other)
        values = (
            self.left * other.left,
            self.left * other.right,
            self.right * other.left,
            self.right * other.right,
        )
        return ConstructiveNumber(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other: NumberLike) -> "ConstructiveNumber":
        other = self._coerce(other)
        if other.left <= 0.0 <= other.right:
            raise ZeroDivisionError("division by interval containing zero")
        return self * ConstructiveNumber.from_bounds(1.0 / other.left, 1.0 / other.right)

    def __rtruediv__(self, other: NumberLike) -> "ConstructiveNumber":
        return self._coerce(other).__truediv__(self)

    def __neg__(self) -> "ConstructiveNumber":
        return ConstructiveNumber(-self.right, -self.left)

    def __pow__(self, power: int) -> "ConstructiveNumber":
        if not isinstance(power, int):
            raise TypeError("constructive interval power supports only integers")
        if power == 0:
            return ConstructiveNumber.exact(1.0)
        if power < 0:
            return ConstructiveNumber.exact(1.0) / (self ** (-power))
        result = ConstructiveNumber.exact(1.0)
        base = self
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def to_float(self) -> float:
        return self.mid
