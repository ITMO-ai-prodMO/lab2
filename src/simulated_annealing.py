import random
import time
import math
import tracemalloc

from ConstructiveNumber import ConstructiveNumber

def _scalar_value(x, alpha=0.5):
    if hasattr(x, "value"):
        return float(x.value(alpha))
    return float(x)


def _bound_to_float(value, alpha=0.5):
    return _scalar_value(value, alpha)


def _random_constructive_in_bounds(left, right, eps=0.0, alpha=0.5):
    a = _bound_to_float(left, alpha)
    b = _bound_to_float(right, alpha)

    if a > b:
        a, b = b, a

    value = random.uniform(a, b)
    return ConstructiveNumber.from_real(value, eps)


def _clip_float(value, left, right, alpha=0.5):
    a = _bound_to_float(left, alpha)
    b = _bound_to_float(right, alpha)

    if a > b:
        a, b = b, a

    return min(max(value, a), b)


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


class SimulatedAnnealing:
    """
    Метод имитации отжига.

    bounds:
        [(left_1, right_1), ..., (left_n, right_n)]
    """

    def __init__(
        self,
        max_iter=1000,
        initial_temperature=1.0,
        cooling_rate=0.995,
        step_size=1.0,
        min_temperature=1e-8,
        eps=0.0,
        alpha=0.5,
        seed=None,
        tol=None,
    ):
        self.max_iter = max_iter
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.min_temperature = min_temperature
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
            x_current = [
                _random_constructive_in_bounds(left, right, self.eps, self.alpha)
                for left, right in bounds
            ]
        else:
            x_current = x0[:]

        f_current = objective.func(x_current)
        f_current_scalar = _scalar_value(f_current, self.alpha)

        x_best = x_current[:]
        f_best = f_current
        f_best_scalar = f_current_scalar

        temperature = self.initial_temperature

        path = [x_best[:]]
        func_values = [f_best_scalar]
        temperatures = [temperature]
        eps_history = [_extract_eps(x_best)]

        for _ in range(self.max_iter):
            if temperature < self.min_temperature:
                break

            x_candidate = []

            for i, (left, right) in enumerate(bounds):
                current_float = _scalar_value(x_current[i], self.alpha)
                shift = random.uniform(-self.step_size, self.step_size)
                candidate_float = _clip_float(
                    current_float + shift,
                    left,
                    right,
                    self.alpha,
                )
                x_candidate.append(
                    ConstructiveNumber.from_real(candidate_float, self.eps)
                )

            f_candidate = objective.func(x_candidate)
            f_candidate_scalar = _scalar_value(f_candidate, self.alpha)

            delta = f_candidate_scalar - f_current_scalar

            if delta <= 0:
                accept = True
            else:
                probability = math.exp(-delta / temperature)
                accept = random.random() < probability

            if accept:
                x_current = x_candidate[:]
                f_current = f_candidate
                f_current_scalar = f_candidate_scalar

                if f_current_scalar < f_best_scalar:
                    x_best = x_current[:]
                    f_best = f_current
                    f_best_scalar = f_current_scalar

            path.append(x_best[:])
            func_values.append(f_best_scalar)
            eps_history.append(_extract_eps(x_best))
            temperatures.append(temperature)

            if self.tol is not None and f_best_scalar <= self.tol:
                break

            temperature = temperature * self.cooling_rate

        elapsed = time.perf_counter() - start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "method": "SimulatedAnnealing",
            "x_best": x_best,
            "f_best": f_best,
            "iterations": len(path) - 1,
            "time_sec": elapsed,
            "memory_current_bytes": current_memory,
            "memory_peak_bytes": peak_memory,
            "path": path,
            "func_values": func_values,
            "temperatures": temperatures,
            "eps_history": eps_history,
            "stats": objective.stats(),
        }