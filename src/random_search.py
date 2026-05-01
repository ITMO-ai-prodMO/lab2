import random
import time
import tracemalloc

from ConstructiveNumber import ConstructiveNumber

def _scalar_value(x, alpha=0.5):
    if hasattr(x, "value"):
        return float(x.value(alpha))
    return float(x)

def _random_constructive_in_bounds(left, right, eps=0.0, alpha=0.5):
    a = _scalar_value(left, alpha)
    b = _scalar_value(right, alpha)

    if a > b:
        a, b = b, a

    value = random.uniform(a, b)
    return ConstructiveNumber.from_real(value, eps)


def _extract_eps(x):
    result = []
    for xi in x:
        if hasattr(xi, "width"):
            result.append(float(xi.width))
        elif hasattr(xi, "radius"):
            result.append(float(2 * xi.radius))
        else:
            result.append(0.0)
    return result


class RandomSearch:
    """
    Случайный поиск.

    bounds:
        [(left_1, right_1), ..., (left_n, right_n)]
    """

    def __init__(self, max_iter=1000, eps=0.0, alpha=0.5, seed=None, tol=None):
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.seed = seed
        self.tol = tol

    def optimize(self, objective, bounds, x0=None):
        if self.seed is not None:
            random.seed(self.seed)

        if len(bounds) == 0:
            raise ValueError("bounds must not be empty")

        tracemalloc.start()
        start_time = time.perf_counter()

        if x0 is None:
            x_best = [
                _random_constructive_in_bounds(left, right, self.eps, self.alpha)
                for left, right in bounds
            ]
        else:
            x_best = x0[:]

        f_best = objective.func(x_best)
        f_best_scalar = _scalar_value(f_best, self.alpha)

        path = [x_best[:]]
        func_values = [f_best_scalar]
        eps_history = [_extract_eps(x_best)]

        for _ in range(self.max_iter):
            x_candidate = [
                _random_constructive_in_bounds(left, right, self.eps, self.alpha)
                for left, right in bounds
            ]

            f_candidate = objective.func(x_candidate)
            f_candidate_scalar = _scalar_value(f_candidate, self.alpha)

            if f_candidate_scalar < f_best_scalar:
                x_best = x_candidate[:]
                f_best = f_candidate
                f_best_scalar = f_candidate_scalar

            path.append(x_best[:])
            func_values.append(f_best_scalar)
            eps_history.append(_extract_eps(x_best))

            if self.tol is not None and f_best_scalar <= self.tol:
                break

        elapsed = time.perf_counter() - start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method": "RandomSearch",
            "x_best": x_best,
            "f_best": f_best,
            "iterations": len(path) - 1,
            "time_sec": elapsed,
            "memory_current_bytes": current_memory,
            "memory_peak_bytes": peak_memory,
            "path": path,
            "func_values": func_values,
            "eps_history": eps_history,
            "stats": objective.stats(),
        }