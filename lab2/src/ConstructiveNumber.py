from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

Number = Union[int, float, Fraction]

@dataclass(frozen=True)
class ConstructiveNumber:
    a: Fraction
    b: Fraction

    def __post_init__(self) -> None:
        if self.a > self.b:
            raise ValueError()

    @classmethod
    def from_bounds(cls, a: Number, b: Number) -> "ConstructiveNumber":
        return cls(Fraction(a), Fraction(b))

    @classmethod
    def from_real(cls, x: Number, epsilon: Number) -> "ConstructiveNumber":
        xq = Fraction(str(x))
        eps = Fraction(str(epsilon))
        if eps < 0:
            raise ValueError()
        return cls(xq - eps, xq + eps)

    @staticmethod
    def _to_cn(value: Union["ConstructiveNumber", Number]) -> "ConstructiveNumber":
        if isinstance(value, ConstructiveNumber):
            return value
        q = Fraction(str(value))
        return ConstructiveNumber(q, q)

    def __add__(self, other: Union["ConstructiveNumber", Number]) -> "ConstructiveNumber":
        o = self._to_cn(other)
        return ConstructiveNumber(self.a + o.a, self.b + o.b)

    def __radd__(self, other: Number) -> "ConstructiveNumber":
        return self + other

    def __sub__(self, other: Union["ConstructiveNumber", Number]) -> "ConstructiveNumber":
        o = self._to_cn(other)
        return ConstructiveNumber(self.a - o.b, self.b - o.a)

    def __rsub__(self, other: Number) -> "ConstructiveNumber":
        return self._to_cn(other) - self

    def __mul__(self, other: Union["ConstructiveNumber", Number]) -> "ConstructiveNumber":
        o = self._to_cn(other)
        products = (self.a * o.a, self.a * o.b, self.b * o.a, self.b * o.b)
        return ConstructiveNumber(min(products), max(products))

    def __rmul__(self, other: Number) -> "ConstructiveNumber":
        return self * other

    def __truediv__(self, other: Union["ConstructiveNumber", Number]) -> "ConstructiveNumber":
        o = self._to_cn(other)
        if o.a <= 0 <= o.b:
            raise ZeroDivisionError()
        ratios = (self.a / o.a, self.a / o.b, self.b / o.a, self.b / o.b)
        return ConstructiveNumber(min(ratios), max(ratios))

    def __rtruediv__(self, other: Number) -> "ConstructiveNumber":
        return self._to_cn(other) / self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (ConstructiveNumber, int, float, Fraction)):
            return NotImplemented
        o = self._to_cn(other)
        return self.a == o.a and self.b == o.b

    def __lt__(self, other: Union["ConstructiveNumber", Number]) -> bool:
        o = self._to_cn(other)
        return self.b < o.a

    def __le__(self, other: Union["ConstructiveNumber", Number]) -> bool:
        o = self._to_cn(other)
        return self.b <= o.a

    def __gt__(self, other: Union["ConstructiveNumber", Number]) -> bool:
        o = self._to_cn(other)
        return self.a > o.b

    def __ge__(self, other: Union["ConstructiveNumber", Number]) -> bool:
        o = self._to_cn(other)
        return self.a >= o.b

    def to_real(self, alpha: float = 0.5) -> float:
        if not (0 <= alpha <= 1):
            raise ValueError()
        alpha_q = Fraction(str(alpha))
        value = (1 - alpha_q) * self.a + alpha_q * self.b
        return float(value)

    def __repr__(self) -> str:
        return f"ConstructiveNumber(a={self.a}, b={self.b})"