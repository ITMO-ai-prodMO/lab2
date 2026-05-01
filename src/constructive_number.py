from __future__ import annotations

from dataclasses import dataclass
import math
from decimal import Decimal
from fractions import Fraction
from typing import Union

NumberLike = Union[int, float, str, Decimal, Fraction, "ConstructiveNumber"]

MAX_DENOMINATOR = 10**6


def _to_fraction(value: Union[int, float, str, Decimal, Fraction]) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot convert non-finite float {value} to Fraction")
        return Fraction(Decimal(str(value)))
    if isinstance(value, str):
        return Fraction(Decimal(value))
    raise TypeError(f"Unsupported type for fraction conversion: {type(value)!r}")


@dataclass(frozen=True)
class ConstructiveNumber:
    """
    Constructive number represented by a rational interval [a, b], a <= b.
    Lab_02 extends lab_01 implementation with sin/cos/exp/sqrt/round
    so that non-smooth and non-continuous test functions can also propagate
    uncertainty through interval arithmetic.
    """

    a: Fraction
    b: Fraction

    def __post_init__(self) -> None:
        a = self.a.limit_denominator(MAX_DENOMINATOR)
        b = self.b.limit_denominator(MAX_DENOMINATOR)
        if a > b:
            a, b = b, a
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @classmethod
    def from_real(cls, x: NumberLike, epsilon: NumberLike) -> "ConstructiveNumber":
        x_q = _to_fraction(x)
        eps_q = _to_fraction(epsilon)
        if eps_q < 0:
            raise ValueError("epsilon must be non-negative")
        return cls(x_q - eps_q, x_q + eps_q)

    @classmethod
    def from_bounds(cls, a: NumberLike, b: NumberLike) -> "ConstructiveNumber":
        a_q = _to_fraction(a)
        b_q = _to_fraction(b)
        if a_q <= b_q:
            return cls(a_q, b_q)
        return cls(b_q, a_q)

    @classmethod
    def _from_floats(cls, lo: float, hi: float) -> "ConstructiveNumber":
        if lo > hi:
            lo, hi = hi, lo
        return cls(_to_fraction(lo), _to_fraction(hi))

    @staticmethod
    def _coerce(other: NumberLike) -> "ConstructiveNumber":
        if isinstance(other, ConstructiveNumber):
            return other
        q = _to_fraction(other)
        return ConstructiveNumber(q, q)

    @property
    def midpoint(self) -> Fraction:
        return (self.a + self.b) / 2

    @property
    def width(self) -> Fraction:
        return self.b - self.a

    @property
    def epsilon(self) -> Fraction:
        return self.width / 2

    def value(self, alpha: float = 0.5) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        alpha_q = Fraction(Decimal(str(alpha)))
        result = (Fraction(1, 1) - alpha_q) * self.a + alpha_q * self.b
        return float(result)

    def __float__(self) -> float:
        return self.value(0.5)

    def __repr__(self) -> str:
        return f"ConstructiveNumber(a={self.a}, b={self.b}, eps={self.epsilon})"

    def __neg__(self) -> "ConstructiveNumber":
        return ConstructiveNumber(-self.b, -self.a)

    def __add__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        return ConstructiveNumber(self.a + other_c.a, self.b + other_c.b)

    def __radd__(self, other: NumberLike) -> "ConstructiveNumber":
        return self.__add__(other)

    def __sub__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        return ConstructiveNumber(self.a - other_c.b, self.b - other_c.a)

    def __rsub__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        return other_c.__sub__(self)

    def __mul__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        products = (
            self.a * other_c.a,
            self.a * other_c.b,
            self.b * other_c.a,
            self.b * other_c.b,
        )
        return ConstructiveNumber(min(products), max(products))

    def __rmul__(self, other: NumberLike) -> "ConstructiveNumber":
        return self.__mul__(other)

    def __truediv__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        if other_c.a <= 0 <= other_c.b:
            raise ZeroDivisionError("Cannot divide by interval containing zero")
        reciprocals = (
            Fraction(1, 1) / other_c.a,
            Fraction(1, 1) / other_c.b,
        )
        reciprocal_interval = ConstructiveNumber(min(reciprocals), max(reciprocals))
        return self * reciprocal_interval

    def __rtruediv__(self, other: NumberLike) -> "ConstructiveNumber":
        other_c = self._coerce(other)
        return other_c.__truediv__(self)

    def __pow__(self, exponent: int) -> "ConstructiveNumber":
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Only non-negative integer exponents are supported")
        if exponent == 0:
            return ConstructiveNumber(Fraction(1, 1), Fraction(1, 1))
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def _comparison_key(self) -> float:
        return self.value(0.5)

    def __lt__(self, other: NumberLike) -> bool:
        other_c = self._coerce(other)
        if self.b < other_c.a:
            return True
        if self.a >= other_c.b:
            return False
        return self._comparison_key() < other_c._comparison_key()

    def __le__(self, other: NumberLike) -> bool:
        other_c = self._coerce(other)
        return self < other_c or self == other_c

    def __gt__(self, other: NumberLike) -> bool:
        other_c = self._coerce(other)
        return not self <= other_c

    def __ge__(self, other: NumberLike) -> bool:
        other_c = self._coerce(other)
        return not self < other_c

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (ConstructiveNumber, int, float, str, Decimal, Fraction)):
            return False
        other_c = self._coerce(other)
        return self.a == other_c.a and self.b == other_c.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def contains_zero(self) -> bool:
        return self.a <= 0 <= self.b

    def abs(self) -> "ConstructiveNumber":
        if self.a >= 0:
            return self
        if self.b <= 0:
            return -self
        return ConstructiveNumber(Fraction(0, 1), max(-self.a, self.b))

    def sin(self) -> "ConstructiveNumber":
        return _interval_trig(self, math.sin, period=2 * math.pi, max_offset=math.pi / 2, min_offset=-math.pi / 2)

    def cos(self) -> "ConstructiveNumber":
        return _interval_trig(self, math.cos, period=2 * math.pi, max_offset=0.0, min_offset=math.pi)

    def exp(self) -> "ConstructiveNumber":
        lo = math.exp(float(self.a))
        hi = math.exp(float(self.b))
        return ConstructiveNumber._from_floats(lo, hi)

    def sqrt(self) -> "ConstructiveNumber":
        a = float(self.a)
        b = float(self.b)
        if b < 0:
            raise ValueError("sqrt of strictly-negative interval")
        a_clamped = max(0.0, a)
        return ConstructiveNumber._from_floats(math.sqrt(a_clamped), math.sqrt(b))

    def round(self) -> "ConstructiveNumber":
        a_round = round(float(self.a))
        b_round = round(float(self.b))
        lo, hi = (a_round, b_round) if a_round <= b_round else (b_round, a_round)
        return ConstructiveNumber(Fraction(lo, 1), Fraction(hi, 1))


def _interval_trig(
    cn: ConstructiveNumber,
    func,
    period: float,
    max_offset: float,
    min_offset: float,
) -> ConstructiveNumber:
    a = float(cn.a)
    b = float(cn.b)
    if b - a >= period:
        return ConstructiveNumber._from_floats(-1.0, 1.0)
    candidates = [func(a), func(b)]
    k_max_low = math.ceil((a - max_offset) / period)
    k_max_high = math.floor((b - max_offset) / period)
    for k in range(k_max_low, k_max_high + 1):
        candidates.append(func(max_offset + k * period))
    k_min_low = math.ceil((a - min_offset) / period)
    k_min_high = math.floor((b - min_offset) / period)
    for k in range(k_min_low, k_min_high + 1):
        candidates.append(func(min_offset + k * period))
    return ConstructiveNumber._from_floats(min(candidates), max(candidates))
