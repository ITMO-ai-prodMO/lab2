from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union


RealLike = Union[int, float, Fraction]
MAX_DENOMINATOR = 10**6


def _to_fraction(value: RealLike) -> Fraction:
    if isinstance(value, Fraction):
        return value.limit_denominator(MAX_DENOMINATOR)
    if isinstance(value, int):
        return Fraction(value, 1)
    return Fraction(str(value)).limit_denominator(MAX_DENOMINATOR)


@dataclass(frozen=True, order=False)
class ConstructiveNumber:
    a: Fraction
    b: Fraction

    def __post_init__(self) -> None:
        left = _to_fraction(self.a).limit_denominator(MAX_DENOMINATOR)
        right = _to_fraction(self.b).limit_denominator(MAX_DENOMINATOR)
        if left <= right:
            object.__setattr__(self, "a", left)
            object.__setattr__(self, "b", right)
        else:
            object.__setattr__(self, "a", right)
            object.__setattr__(self, "b", left)

    @classmethod
    def from_real(cls, x: RealLike, epsilon: RealLike) -> "ConstructiveNumber":
        center = _to_fraction(x)
        radius = abs(_to_fraction(epsilon))
        return cls(center - radius, center + radius)

    @classmethod
    def from_bounds(cls, a: RealLike, b: RealLike) -> "ConstructiveNumber":
        return cls(_to_fraction(a), _to_fraction(b))

    @staticmethod
    def _coerce(value: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        if isinstance(value, ConstructiveNumber):
            return value
        return ConstructiveNumber.from_bounds(value, value)

    @property
    def width(self) -> Fraction:
        return self.b - self.a

    @property
    def epsilon(self) -> Fraction:
        return self.width / 2

    @property
    def midpoint(self) -> Fraction:
        return (self.a + self.b) / 2

    def resolve(self, alpha: float = 0.5) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        alpha_fraction = Fraction(str(alpha))
        return float(self.a + (self.b - self.a) * alpha_fraction)

    def _ordering_key(self) -> tuple[Fraction, Fraction]:
        return (self.midpoint, self.width)

    def __float__(self) -> float:
        return self.resolve(0.5)

    def __add__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        right = self._coerce(other)
        return ConstructiveNumber(self.a + right.a, self.b + right.b)

    def __radd__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        return self + other

    def __sub__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        right = self._coerce(other)
        return ConstructiveNumber(self.a - right.b, self.b - right.a)

    def __rsub__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        return self._coerce(other) - self

    def __mul__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        right = self._coerce(other)
        products = (
            self.a * right.a,
            self.a * right.b,
            self.b * right.a,
            self.b * right.b,
        )
        return ConstructiveNumber(min(products), max(products))

    def __rmul__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        return self * other

    def reciprocal(self) -> "ConstructiveNumber":
        if self.a <= 0 <= self.b:
            raise ZeroDivisionError("interval spans zero")
        return ConstructiveNumber(Fraction(1, 1) / self.b, Fraction(1, 1) / self.a)

    def __truediv__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: Union["ConstructiveNumber", RealLike]) -> "ConstructiveNumber":
        return self._coerce(other) / self

    def __neg__(self) -> "ConstructiveNumber":
        return ConstructiveNumber(-self.b, -self.a)

    def __abs__(self) -> "ConstructiveNumber":
        if self.a >= 0:
            return self
        if self.b <= 0:
            return -self
        return ConstructiveNumber(0, max(-self.a, self.b))

    def __lt__(self, other: Union["ConstructiveNumber", RealLike]) -> bool:
        return self._ordering_key() < self._coerce(other)._ordering_key()

    def __le__(self, other: Union["ConstructiveNumber", RealLike]) -> bool:
        return self._ordering_key() <= self._coerce(other)._ordering_key()

    def __gt__(self, other: Union["ConstructiveNumber", RealLike]) -> bool:
        return self._ordering_key() > self._coerce(other)._ordering_key()

    def __ge__(self, other: Union["ConstructiveNumber", RealLike]) -> bool:
        return self._ordering_key() >= self._coerce(other)._ordering_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (ConstructiveNumber, int, float, Fraction)):
            return False
        right = self._coerce(other)
        return self.a == right.a and self.b == right.b

    def __repr__(self) -> str:
        return f"ConstructiveNumber(a={self.a}, b={self.b}, eps={self.epsilon})"
