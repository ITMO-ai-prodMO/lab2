import math

def _scalar_value(x, alpha=0.5):
    """
    Если x = ConstructiveNumber, берем представительное значение.
    Иначе обычный float.
    """
    if hasattr(x, "value"):
        return float(x.value(alpha))
    return float(x)


class BaseFunction:
    def __init__(self):
        self.func_calls = 0
        self.grad_calls = 0

    def reset_counters(self):
        self.func_calls = 0
        self.grad_calls = 0

    def stats(self):
        return {
            "name": self.name(),
            "dimension": self.dimension(),
            "func_calls": self.func_calls,
            "grad_calls": self.grad_calls,
            "minimum_point": self.minimum_point(),
            "minimum_value": self.minimum_value(),
        }


class SphereFunction(BaseFunction):
    """
    f(x) = sum(x_i^2)
    min at (0,...,0)
    """

    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def func(self, x):
        self.func_calls += 1

        result = 0
        for xi in x:
            result = result + xi * xi

        return result

    def name(self):
        return f"Sphere function ({self.n} variables)"

    def dimension(self):
        return self.n

    def minimum_point(self):
        return [0] * self.n

    def minimum_value(self):
        return 0


class RosenbrockFunction(BaseFunction):
    """
    f(x)=sum(100(x[i+1]-x[i]^2)^2 + (1-x[i])^2)
    min at (1,...,1)
    """

    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def func(self, x):
        self.func_calls += 1

        result = 0
        for i in range(len(x) - 1):
            t1 = x[i + 1] - x[i] * x[i]
            t2 = 1 - x[i]

            result = result + 100 * t1 * t1
            result = result + t2 * t2

        return result

    def name(self):
        return f"Rosenbrock function ({self.n} variables)"

    def dimension(self):
        return self.n

    def minimum_point(self):
        return [1] * self.n

    def minimum_value(self):
        return 0


class RastriginFunction(BaseFunction):
    """
    f(x)=10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    min at (0,...,0)
    """

    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def func(self, x):
        self.func_calls += 1

        result = 10 * self.n

        for xi in x:
            xf = _scalar_value(xi)
            result = result + xi * xi - 10 * math.cos(2 * math.pi * xf)

        return result

    def name(self):
        return f"Rastrigin function ({self.n} variables)"

    def dimension(self):
        return self.n

    def minimum_point(self):
        return [0] * self.n

    def minimum_value(self):
        return 0


class AckleyFunction(BaseFunction):
    """
    2D Ackley
    min at (0,0)
    """

    def func(self, x):
        self.func_calls += 1

        x1 = _scalar_value(x[0])
        x2 = _scalar_value(x[1])

        part1 = -20 * math.exp(
            -0.2 * math.sqrt(0.5 * (x1 * x1 + x2 * x2))
        )

        part2 = -math.exp(
            0.5 * (
                math.cos(2 * math.pi * x1) +
                math.cos(2 * math.pi * x2)
            )
        )

        return part1 + part2 + math.e + 20

    def name(self):
        return "Ackley function"

    def dimension(self):
        return 2

    def minimum_point(self):
        return [0, 0]

    def minimum_value(self):
        return 0


class HimmelblauFunction(BaseFunction):
    """
    f(x,y)=(x^2+y-11)^2 + (x+y^2-7)^2
    several global minima
    """

    def func(self, x):
        self.func_calls += 1

        x1 = x[0]
        x2 = x[1]

        t1 = x1 * x1 + x2 - 11
        t2 = x1 + x2 * x2 - 7

        return t1 * t1 + t2 * t2

    def name(self):
        return "Himmelblau function"

    def dimension(self):
        return 2

    def minimum_point(self):
        return [
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ]

    def minimum_value(self):
        return 0


class DesmosDiscontinuousFunction(BaseFunction):
    """
    Функция из Desmos.

    Разрывная функция из-за round(sin(...)).
    """

    def __init__(self, d=1):
        super().__init__()
        self.d = d

    def func(self, point):
        self.func_calls += 1

        x = _scalar_value(point[0])
        y = _scalar_value(point[1])

        a1 = round(math.sin(10 * y)) + 2
        a2 = round(math.sin(7 * x)) + 2

        temp1 = x * a1
        temp2 = y * a2

        part1 = temp1 * temp1 + y - 10
        part2 = x + temp2 * temp2 - 7

        return self.d * (part1 * part1 + part2 * part2)

    def name(self):
        return "Desmos discontinuous function"

    def dimension(self):
        return 2

    def minimum_point(self):
        return None

    def minimum_value(self):
        return None