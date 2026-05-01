from fractions import Fraction


class ConstructiveNumber:
    def __init__(self, a, b=None):
        if b is None:
            b = a

        self.a = self._to_fraction(a)
        self.b = self._to_fraction(b)

        if self.a > self.b:
            self.a, self.b = self.b, self.a

    @staticmethod
    def _to_fraction(x):
        if isinstance(x, Fraction):
            return x
        if isinstance(x, ConstructiveNumber):
            raise TypeError("cannot convert ConstructiveNumber to Fraction directly")
        if isinstance(x, int):
            return Fraction(x)
        if isinstance(x, float):
            return Fraction(str(x))
        return Fraction(x)

    @classmethod
    def from_real(cls, x, eps):
        x = cls._to_fraction(x)
        eps = cls._to_fraction(eps)

        if eps < 0:
            raise ValueError("eps must be non-negative")

        return cls(x - eps, x + eps)

    @staticmethod
    def _to_constructive(value):
        if isinstance(value, ConstructiveNumber):
            return value
        return ConstructiveNumber(value, value)

    @property
    def width(self):
        return self.b - self.a

    @property
    def radius(self):
        return (self.b - self.a) / 2

    @property
    def midpoint(self):
        return (self.a + self.b) / 2

    def value(self, alpha=Fraction(1, 2)):
        alpha = self._to_fraction(alpha)
        if not (Fraction(0) <= alpha <= Fraction(1)):
            raise ValueError("alpha must be in [0, 1]")
        return (Fraction(1) - alpha) * self.a + alpha * self.b

    def to_float(self, alpha=Fraction(1, 2)):
        return float(self.value(alpha))

    def __add__(self, right):
        right = self._to_constructive(right)
        return ConstructiveNumber(self.a + right.a, self.b + right.b)

    def __radd__(self, left):
        return self.__add__(left)

    def __sub__(self, right):
        right = self._to_constructive(right)
        return ConstructiveNumber(self.a - right.b, self.b - right.a)

    def __rsub__(self, left):
        left = self._to_constructive(left)
        return left.__sub__(self)

    def __mul__(self, right):
        right = self._to_constructive(right)
        vals = [
            self.a * right.a,
            self.a * right.b,
            self.b * right.a,
            self.b * right.b,
        ]
        return ConstructiveNumber(min(vals), max(vals))

    def __rmul__(self, left):
        return self.__mul__(left)

    def __truediv__(self, right):
        right = self._to_constructive(right)

        if right.a <= 0 <= right.b:
            raise ZeroDivisionError("division by interval containing zero")

        inv_a = Fraction(1, 1) / right.a
        inv_b = Fraction(1, 1) / right.b
        return self * ConstructiveNumber(min(inv_a, inv_b), max(inv_a, inv_b))

    def __rtruediv__(self, left):
        left = self._to_constructive(left)
        return left.__truediv__(self)

    def __neg__(self):
        return ConstructiveNumber(-self.b, -self.a)

    def __abs__(self):
        if self.a >= 0:
            return ConstructiveNumber(self.a, self.b)
        if self.b <= 0:
            return ConstructiveNumber(-self.b, -self.a)
        return ConstructiveNumber(0, max(-self.a, self.b))

    def __eq__(self, right):
        right = self._to_constructive(right)
        return self.a == right.a and self.b == right.b

    def __lt__(self, right):
        right = self._to_constructive(right)
        return self.b < right.a

    def __le__(self, right):
        right = self._to_constructive(right)
        return self.b <= right.a

    def __gt__(self, right):
        right = self._to_constructive(right)
        return self.a > right.b

    def __ge__(self, right):
        right = self._to_constructive(right)
        return self.a >= right.b

    def __str__(self):
        return f"[{self.a}, {self.b}]"

    def __repr__(self):
        return f"ConstructiveNumber({self.a}, {self.b})"