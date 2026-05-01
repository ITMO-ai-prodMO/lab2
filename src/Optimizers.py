import time


def _scalar_value(x, alpha=0.5):
    """
    Преобразует число в обычный float для сравнений / норм.
    Если это ConstructiveNumber, берется представительное значение value(alpha).
    """
    if hasattr(x, "value"):
        return float(x.value(alpha))
    return float(x)


def _vector_to_float(x, alpha=0.5):
    return [_scalar_value(xi, alpha) for xi in x]


def _make_zero_like(x):
    return [0 for _ in x]


def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def vec_scale(s, x):
    return [s * xi for xi in x]


def dot(a, b):
    result = 0
    for ai, bi in zip(a, b):
        result = result + ai * bi
    return result


def norm2(x, alpha=0.5):
    xf = _vector_to_float(x, alpha)
    return sum(xi * xi for xi in xf) ** 0.5


def distance(a, b, alpha=0.5):
    return norm2(vec_sub(a, b), alpha)


def centroid(points):
    n = len(points)
    dim = len(points[0])

    c = []
    for j in range(dim):
        s = 0
        for i in range(n):
            s = s + points[i][j]
        c.append(s / n)
    return c

def solve_linear_system(A, b, alpha=0.5):
    """
    Решение Ax = b методом Гаусса с частичным выбором главного элемента.
    Для выбора ведущей строки используется представительное float-значение.
    """
    n = len(A)

    M = [row[:] for row in A]
    rhs = b[:]

    for k in range(n):
        pivot_row = k
        pivot_val = abs(_scalar_value(M[k][k], alpha))

        for i in range(k + 1, n):
            current = abs(_scalar_value(M[i][k], alpha))
            if current > pivot_val:
                pivot_val = current
                pivot_row = i

        if pivot_val == 0:
            raise ValueError("Matrix is singular or nearly singular")

        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        pivot = M[k][k]

        for i in range(k + 1, n):
            factor = M[i][k] / pivot
            for j in range(k, n):
                M[i][j] = M[i][j] - factor * M[k][j]
            rhs[i] = rhs[i] - factor * rhs[k]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, n):
            s = s - M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x

class NelderMead:
    """
    Метод 0-го порядка: деформируемый многогранник (Нелдер–Мид).
    """

    def __init__(
        self,
        step=1.0,
        alpha_reflect=1.0,
        gamma_expand=2.0,
        rho_contract=0.5,
        sigma_shrink=0.5,
        max_iter=300,
        tol=1e-6,
        alpha=0.5,
    ):
        self.step = step
        self.alpha_reflect = alpha_reflect
        self.gamma_expand = gamma_expand
        self.rho_contract = rho_contract
        self.sigma_shrink = sigma_shrink
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def optimize(self, objective, x0):
        n = len(x0)
        simplex = [x0[:]]

        for i in range(n):
            point = x0[:]
            point[i] = point[i] + self.step
            simplex.append(point)

        path = [x0[:]]
        func_values = []
        eps_history = []
        start_time = time.perf_counter()

        for iteration in range(self.max_iter):
            simplex = sorted(
                simplex,
                key=lambda p: _scalar_value(objective.func(p), self.alpha)
            )

            best = simplex[0]
            worst = simplex[-1]
            second_worst = simplex[-2]

            f_best = _scalar_value(objective.func(best), self.alpha)
            f_worst = _scalar_value(objective.func(worst), self.alpha)
            func_values.append(f_best)
            eps_history.append(self._extract_eps(best))

            if self._simplex_size(simplex) < self.tol:
                break

            c = centroid(simplex[:-1])

            reflected = vec_add(c, vec_scale(self.alpha_reflect, vec_sub(c, worst)))
            f_reflected = _scalar_value(objective.func(reflected), self.alpha)

            f_second_worst = _scalar_value(objective.func(second_worst), self.alpha)

            if f_best <= f_reflected < f_second_worst:
                simplex[-1] = reflected
                path.append(simplex[0][:])
                continue

            if f_reflected < f_best:
                expanded = vec_add(c, vec_scale(self.gamma_expand, vec_sub(reflected, c)))
                f_expanded = _scalar_value(objective.func(expanded), self.alpha)

                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                else:
                    simplex[-1] = reflected

                path.append(simplex[0][:])
                continue

            contracted = vec_add(c, vec_scale(self.rho_contract, vec_sub(worst, c)))
            f_contracted = _scalar_value(objective.func(contracted), self.alpha)

            if f_contracted < f_worst:
                simplex[-1] = contracted
                path.append(simplex[0][:])
                continue

            best = simplex[0]
            new_simplex = [best]
            for i in range(1, len(simplex)):
                shrunk = vec_add(best, vec_scale(self.sigma_shrink, vec_sub(simplex[i], best)))
                new_simplex.append(shrunk)
            simplex = new_simplex
            path.append(simplex[0][:])

        simplex = sorted(
            simplex,
            key=lambda p: _scalar_value(objective.func(p), self.alpha)
        )
        x_best = simplex[0]
        elapsed = time.perf_counter() - start_time

        return {
            "method": "NelderMead",
            "x_best": x_best,
            "f_best": objective.func(x_best),
            "iterations": len(path) - 1,
            "time_sec": elapsed,
            "path": path,
            "func_values": func_values,
            "eps_history": eps_history,
            "stats": objective.stats(),
        }

    def _simplex_size(self, simplex):
        best = simplex[0]
        return max(distance(best, p, self.alpha) for p in simplex)

    def _extract_eps(self, x):
        result = []
        for xi in x:
            if hasattr(xi, "width"):
                result.append(float(xi.width))
            elif hasattr(xi, "radius"):
                result.append(float(2 * xi.radius))
            else:
                result.append(0.0)
        return result

class GradientDescent:
    """
    Метод 1-го порядка: градиентный спуск.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, alpha=0.5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def optimize(self, objective, x0):
        x = x0[:]
        path = [x[:]]
        func_values = []
        grad_norms = []
        eps_history = []
        start_time = time.perf_counter()

        for iteration in range(self.max_iter):
            fx = objective.func(x)
            g = objective.grad(x)

            func_values.append(_scalar_value(fx, self.alpha))
            grad_norm = norm2(g, self.alpha)
            grad_norms.append(grad_norm)

            eps_history.append(self._extract_eps(x))

            if grad_norm < self.tol:
                break

            step = vec_scale(self.learning_rate, g)
            x = vec_sub(x, step)
            path.append(x[:])

        elapsed = time.perf_counter() - start_time

        return {
            "method": "GradientDescent",
            "x_best": x,
            "f_best": objective.func(x),
            "iterations": len(path) - 1,
            "time_sec": elapsed,
            "path": path,
            "func_values": func_values,
            "grad_norms": grad_norms,
            "eps_history": eps_history,
            "stats": objective.stats(),
        }

    def _extract_eps(self, x):
        result = []
        for xi in x:
            if hasattr(xi, "width"):
                result.append(float(xi.width))
            elif hasattr(xi, "radius"):
                result.append(float(2 * xi.radius))
            else:
                result.append(0.0)
        return result


class NewtonMethod:
    """
    Метод 2-го порядка: метод Ньютона.
    """

    def __init__(self, max_iter=100, tol=1e-8, damping=1.0, alpha=0.5):
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.alpha = alpha

    def optimize(self, objective, x0):
        if not hasattr(objective, "hess"):
            raise ValueError("Objective must implement hess(x) for Newton's method")

        x = x0[:]
        path = [x[:]]
        func_values = []
        grad_norms = []
        eps_history = []
        start_time = time.perf_counter()

        for iteration in range(self.max_iter):
            fx = objective.func(x)
            g = objective.grad(x)
            H = objective.hess(x)

            func_values.append(_scalar_value(fx, self.alpha))
            grad_norm = norm2(g, self.alpha)
            grad_norms.append(grad_norm)
            eps_history.append(self._extract_eps(x))

            if grad_norm < self.tol:
                break

            minus_g = vec_scale(-1, g)
            direction = solve_linear_system(H, minus_g, self.alpha)

            x = vec_add(x, vec_scale(self.damping, direction))
            path.append(x[:])

        elapsed = time.perf_counter() - start_time

        return {
            "method": "NewtonMethod",
            "x_best": x,
            "f_best": objective.func(x),
            "iterations": len(path) - 1,
            "time_sec": elapsed,
            "path": path,
            "func_values": func_values,
            "grad_norms": grad_norms,
            "eps_history": eps_history,
            "stats": objective.stats(),
        }

    def _extract_eps(self, x):
        result = []
        for xi in x:
            if hasattr(xi, "width"):
                result.append(float(xi.width))
            elif hasattr(xi, "radius"):
                result.append(float(2 * xi.radius))
            else:
                result.append(0.0)
        return result