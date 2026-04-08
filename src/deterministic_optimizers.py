from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .constructive_number import ConstructiveNumber


def _to_float(value: object) -> float:
    if isinstance(value, ConstructiveNumber):
        return float(value)
    return float(value)


def _vector_to_float(values: Sequence[object]) -> list[float]:
    return [_to_float(value) for value in values]


def _mean_epsilon(values: Sequence[object]) -> float:
    epsilons = [float(value.epsilon) for value in values if isinstance(value, ConstructiveNumber)]
    if not epsilons:
        return 0.0
    return sum(epsilons) / len(epsilons)


def _norm(values: Sequence[object]) -> float:
    return math.sqrt(sum(_to_float(value) ** 2 for value in values))


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        if abs(pivot_value) < 1e-12:
            raise ValueError("singular hessian")
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][-1] for i in range(n)]


@dataclass
class OptimizationResult:
    method: str
    point: list[object]
    value: object
    iterations: int
    elapsed_seconds: float
    history: list[dict[str, object]]
    value_calls: int
    gradient_calls: int
    hessian_calls: int
    learning_rate: float | None = None

    @property
    def resolved_point(self) -> list[float]:
        return _vector_to_float(self.point)

    @property
    def resolved_value(self) -> float:
        return _to_float(self.value)


def _record(history: list[dict[str, object]], iteration: int, x: Sequence[object], value: object) -> None:
    history.append(
        {
            "iteration": iteration,
            "x": _vector_to_float(x),
            "value": _to_float(value),
            "epsilon": _mean_epsilon(x),
        }
    )


def gradient_descent(
    objective,
    x0: Sequence[object],
    learning_rate: float,
    max_iter: int = 500,
    tolerance: float = 1e-8,
    backtracking: bool = True,
) -> OptimizationResult:
    start = time.perf_counter()
    x = list(x0)
    history: list[dict[str, object]] = []
    value = objective.value(x)
    _record(history, 0, x, value)

    for iteration in range(1, max_iter + 1):
        gradient = objective.gradient(x)
        grad_norm = _norm(gradient)
        if grad_norm < tolerance:
            break

        step = learning_rate
        candidate = [xi - step * gi for xi, gi in zip(x, gradient)]
        candidate_value = objective.value(candidate)
        if backtracking:
            while _to_float(candidate_value) > _to_float(value) and step > 1e-10:
                step *= 0.5
                candidate = [xi - step * gi for xi, gi in zip(x, gradient)]
                candidate_value = objective.value(candidate)
        x = candidate
        value = candidate_value
        _record(history, iteration, x, value)
        new_gradient = objective.gradient(x)
        if _norm(new_gradient) < tolerance:
            break

    elapsed = time.perf_counter() - start
    return OptimizationResult(
        method="gradient_descent",
        point=x,
        value=value,
        iterations=len(history) - 1,
        elapsed_seconds=elapsed,
        history=history,
        value_calls=objective.value_calls,
        gradient_calls=objective.gradient_calls,
        hessian_calls=objective.hessian_calls,
        learning_rate=learning_rate,
    )


def nelder_mead(
    objective,
    x0: Sequence[object],
    initial_step: float = 0.4,
    max_iter: int = 400,
    tolerance: float = 1e-8,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> OptimizationResult:
    start = time.perf_counter()
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = point[i] + initial_step
        simplex.append(point)

    values = [objective.value(point) for point in simplex]
    history: list[dict[str, object]] = []
    best_index = min(range(len(simplex)), key=lambda idx: _to_float(values[idx]))
    _record(history, 0, simplex[best_index], values[best_index])

    for iteration in range(1, max_iter + 1):
        order = sorted(range(len(simplex)), key=lambda idx: _to_float(values[idx]))
        simplex = [simplex[idx] for idx in order]
        values = [values[idx] for idx in order]
        best_point = simplex[0]
        best_value = values[0]
        worst_point = simplex[-1]
        second_worst_value = values[-2]

        centroid = []
        for coord in range(n):
            total = 0
            for point in simplex[:-1]:
                total = total + point[coord]
            centroid.append(total / n)

        reflected = [c + alpha * (c - w) for c, w in zip(centroid, worst_point)]
        reflected_value = objective.value(reflected)

        if _to_float(values[0]) <= _to_float(reflected_value) < _to_float(second_worst_value):
            simplex[-1] = reflected
            values[-1] = reflected_value
        elif _to_float(reflected_value) < _to_float(values[0]):
            expanded = [c + gamma * (r - c) for c, r in zip(centroid, reflected)]
            expanded_value = objective.value(expanded)
            if _to_float(expanded_value) < _to_float(reflected_value):
                simplex[-1] = expanded
                values[-1] = expanded_value
            else:
                simplex[-1] = reflected
                values[-1] = reflected_value
        else:
            contracted = [c + rho * (w - c) for c, w in zip(centroid, worst_point)]
            contracted_value = objective.value(contracted)
            if _to_float(contracted_value) < _to_float(values[-1]):
                simplex[-1] = contracted
                values[-1] = contracted_value
            else:
                best = simplex[0]
                simplex = [best] + [
                    [best[j] + sigma * (point[j] - best[j]) for j in range(n)]
                    for point in simplex[1:]
                ]
                values = [objective.value(point) for point in simplex]

        order = sorted(range(len(simplex)), key=lambda idx: _to_float(values[idx]))
        simplex = [simplex[idx] for idx in order]
        values = [values[idx] for idx in order]
        _record(history, iteration, simplex[0], values[0])

        max_distance = max(
            math.dist(_vector_to_float(simplex[0]), _vector_to_float(point))
            for point in simplex[1:]
        )
        if max_distance < tolerance and abs(_to_float(values[-1]) - _to_float(values[0])) < tolerance:
            break

    elapsed = time.perf_counter() - start
    return OptimizationResult(
        method="nelder_mead",
        point=simplex[0],
        value=values[0],
        iterations=len(history) - 1,
        elapsed_seconds=elapsed,
        history=history,
        value_calls=objective.value_calls,
        gradient_calls=objective.gradient_calls,
        hessian_calls=objective.hessian_calls,
    )


def newton_method(
    objective,
    x0: Sequence[object],
    max_iter: int = 100,
    tolerance: float = 1e-8,
    damping: float = 1.0,
) -> OptimizationResult:
    start = time.perf_counter()
    x = list(x0)
    history: list[dict[str, object]] = []
    value = objective.value(x)
    _record(history, 0, x, value)

    for iteration in range(1, max_iter + 1):
        gradient = objective.gradient(x)
        grad_norm = _norm(gradient)
        if grad_norm < tolerance:
            break
        hessian = objective.hessian(x)
        step = _solve_linear_system([[float(cell) for cell in row] for row in hessian], _vector_to_float(gradient))
        step_scale = damping
        candidate = [xi - step_scale * si for xi, si in zip(x, step)]
        candidate_value = objective.value(candidate)
        while _to_float(candidate_value) > _to_float(value) and step_scale > 1e-8:
            step_scale *= 0.5
            candidate = [xi - step_scale * si for xi, si in zip(x, step)]
            candidate_value = objective.value(candidate)
        x = candidate
        value = candidate_value
        _record(history, iteration, x, value)
        if _norm(step) < tolerance:
            break

    elapsed = time.perf_counter() - start
    return OptimizationResult(
        method="newton_method",
        point=x,
        value=value,
        iterations=len(history) - 1,
        elapsed_seconds=elapsed,
        history=history,
        value_calls=objective.value_calls,
        gradient_calls=objective.gradient_calls,
        hessian_calls=objective.hessian_calls,
    )


def tune_learning_rate(
    objective_factory: Callable[[], object],
    x0: Sequence[object],
    candidates: Sequence[float],
    max_iter: int = 150,
) -> float:
    best_lr = candidates[0]
    best_value = math.inf
    for candidate in candidates:
        objective = objective_factory()
        result = gradient_descent(
            objective=objective,
            x0=x0,
            learning_rate=candidate,
            max_iter=max_iter,
            tolerance=1e-6,
            backtracking=True,
        )
        if result.resolved_value < best_value:
            best_value = result.resolved_value
            best_lr = candidate
    return best_lr
