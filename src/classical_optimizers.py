"""
Классические методы оптимизации (gradient descent, Nelder-Mead, Newton)
из lab_01, адаптированные под OptimizationResult из stochastic_optimizers
(добавлено поле memory_scalars).
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence

import numpy as np

from constructive_number import ConstructiveNumber
from functions import BlackBoxFunction, is_constructive, to_real_scalar, to_real_vector, vector_epsilons
from stochastic_optimizers import (
    OptimizationResult,
    make_constructive_vector,
    point_eps_max,
    scalar_eps,
    vector_to_floats,
)


def _record(history_points, history_values, history_value_eps, history_point_eps, point, value):
    history_points.append(vector_to_floats(point))
    history_values.append(to_real_scalar(value))
    history_value_eps.append(scalar_eps(value))
    history_point_eps.append(point_eps_max(point))


def _gradient_norm(grad: Sequence) -> float:
    return float(np.linalg.norm([to_real_scalar(g) for g in grad]))


def gradient_descent(
    func: BlackBoxFunction,
    x0: Sequence[float],
    learning_rate: float,
    epsilon: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> OptimizationResult:
    func.reset_counters()
    x = make_constructive_vector(x0, epsilon) if epsilon > 0 else [float(v) for v in x0]
    history_points, history_values, history_value_eps, history_point_eps = [], [], [], []
    start = time.perf_counter()

    value = func.value(x)
    _record(history_points, history_values, history_value_eps, history_point_eps, x, value)

    iteration = 0
    for iteration in range(1, max_iter + 1):
        grad = func.gradient(x)
        gnorm = _gradient_norm(grad)
        if not math.isfinite(gnorm) or gnorm < tol:
            break
        x = [xi - learning_rate * gi for xi, gi in zip(x, grad)]
        value = func.value(x)
        _record(history_points, history_values, history_value_eps, history_point_eps, x, value)

        current_point = vector_to_floats(x)
        if not np.all(np.isfinite(current_point)) or np.linalg.norm(current_point) > 1e6:
            break
        if abs(to_real_scalar(value)) > 1e12:
            break
        if np.linalg.norm(current_point - func.optimum) < tol:
            break

    elapsed = time.perf_counter() - start
    final_point = vector_to_floats(x)
    return OptimizationResult(
        method_name="Gradient descent",
        function_name=func.name,
        initial_point=np.array(x0, dtype=float),
        final_point=final_point,
        final_value=to_real_scalar(value),
        distance_to_optimum=float(np.linalg.norm(final_point - func.optimum)),
        iterations=iteration,
        elapsed_time=elapsed,
        value_calls=func.counters.value_calls,
        gradient_calls=func.counters.gradient_calls,
        hessian_calls=func.counters.hessian_calls,
        memory_scalars=2 * func.dimension,
        history_points=history_points,
        history_values=history_values,
        history_value_eps=history_value_eps,
        history_point_eps=history_point_eps,
        extra={"learning_rate": learning_rate},
    )


def nelder_mead(
    func: BlackBoxFunction,
    x0: Sequence[float],
    epsilon: float = 0.0,
    step: float = 0.5,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    max_iter: int = 250,
    tol: float = 1e-8,
) -> OptimizationResult:
    func.reset_counters()
    dim = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(dim):
        point = np.array(x0, dtype=float)
        point[i] += step
        simplex.append(point)

    if epsilon > 0:
        simplex = [make_constructive_vector(p, epsilon) for p in simplex]
    else:
        simplex = [[float(v) for v in p] for p in simplex]

    history_points, history_values, history_value_eps, history_point_eps = [], [], [], []
    start = time.perf_counter()

    values = [func.value(p) for p in simplex]
    best_index = int(np.argmin([to_real_scalar(v) for v in values]))
    _record(history_points, history_values, history_value_eps, history_point_eps, simplex[best_index], values[best_index])

    iteration = 0
    for iteration in range(1, max_iter + 1):
        order = np.argsort([to_real_scalar(v) for v in values])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        current_values = np.array([to_real_scalar(v) for v in values], dtype=float)
        if np.std(current_values) < tol:
            break

        centroid = []
        for j in range(dim):
            total = simplex[0][j]
            for i in range(1, dim):
                total = total + simplex[i][j]
            centroid.append(total / dim)

        worst = simplex[-1]
        reflected = [c + alpha * (c - w) for c, w in zip(centroid, worst)]
        reflected_value = func.value(reflected)

        if to_real_scalar(values[0]) <= to_real_scalar(reflected_value) < to_real_scalar(values[-2]):
            simplex[-1], values[-1] = reflected, reflected_value
        elif to_real_scalar(reflected_value) < to_real_scalar(values[0]):
            expanded = [c + gamma * (r - c) for c, r in zip(centroid, reflected)]
            expanded_value = func.value(expanded)
            if to_real_scalar(expanded_value) < to_real_scalar(reflected_value):
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        else:
            contracted = [c + rho * (w - c) for c, w in zip(centroid, worst)]
            contracted_value = func.value(contracted)
            if to_real_scalar(contracted_value) < to_real_scalar(values[-1]):
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                best = simplex[0]
                new_simplex = [best]
                for i in range(1, len(simplex)):
                    shrunk = [best[j] + sigma * (simplex[i][j] - best[j]) for j in range(dim)]
                    new_simplex.append(shrunk)
                simplex = new_simplex
                values = [func.value(p) for p in simplex]

        best_index = int(np.argmin([to_real_scalar(v) for v in values]))
        _record(history_points, history_values, history_value_eps, history_point_eps, simplex[best_index], values[best_index])

    elapsed = time.perf_counter() - start
    best_index = int(np.argmin([to_real_scalar(v) for v in values]))
    best_point = simplex[best_index]
    best_value = values[best_index]
    final_point = vector_to_floats(best_point)

    return OptimizationResult(
        method_name="Nelder-Mead",
        function_name=func.name,
        initial_point=np.array(x0, dtype=float),
        final_point=final_point,
        final_value=to_real_scalar(best_value),
        distance_to_optimum=float(np.linalg.norm(final_point - func.optimum)),
        iterations=iteration,
        elapsed_time=elapsed,
        value_calls=func.counters.value_calls,
        gradient_calls=func.counters.gradient_calls,
        hessian_calls=func.counters.hessian_calls,
        memory_scalars=(dim + 1) * dim,
        history_points=history_points,
        history_values=history_values,
        history_value_eps=history_value_eps,
        history_point_eps=history_point_eps,
        extra={"step": step},
    )


def newton_method(
    func: BlackBoxFunction,
    x0: Sequence[float],
    epsilon: float = 0.0,
    damping: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> OptimizationResult:
    func.reset_counters()
    x = make_constructive_vector(x0, epsilon) if epsilon > 0 else [float(v) for v in x0]
    history_points, history_values, history_value_eps, history_point_eps = [], [], [], []
    start = time.perf_counter()

    value = func.value(x)
    _record(history_points, history_values, history_value_eps, history_point_eps, x, value)

    iteration = 0
    for iteration in range(1, max_iter + 1):
        grad = func.gradient(x)
        grad_real = np.array([to_real_scalar(g) for g in grad], dtype=float)
        if np.linalg.norm(grad_real) < tol:
            break
        hess = func.hessian(x)
        try:
            step = np.linalg.solve(hess, grad_real)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad_real, rcond=None)[0]

        current_value = to_real_scalar(value)
        step_scale = damping
        accepted = False
        while step_scale > 1e-6:
            candidate = [xi - step_scale * si for xi, si in zip(x, step)]
            candidate_value = func.value(candidate)
            if to_real_scalar(candidate_value) < current_value:
                x = candidate
                value = candidate_value
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            x = [xi - 1e-3 * gi for xi, gi in zip(x, grad)]
            value = func.value(x)

        _record(history_points, history_values, history_value_eps, history_point_eps, x, value)
        current_point = vector_to_floats(x)
        if not np.all(np.isfinite(current_point)) or np.linalg.norm(current_point) > 1e6:
            break
        if abs(to_real_scalar(value)) > 1e12:
            break
        if np.linalg.norm(current_point - func.optimum) < tol:
            break

    elapsed = time.perf_counter() - start
    final_point = vector_to_floats(x)
    return OptimizationResult(
        method_name="Newton",
        function_name=func.name,
        initial_point=np.array(x0, dtype=float),
        final_point=final_point,
        final_value=to_real_scalar(value),
        distance_to_optimum=float(np.linalg.norm(final_point - func.optimum)),
        iterations=iteration,
        elapsed_time=elapsed,
        value_calls=func.counters.value_calls,
        gradient_calls=func.counters.gradient_calls,
        hessian_calls=func.counters.hessian_calls,
        memory_scalars=func.dimension * func.dimension + 2 * func.dimension,
        history_points=history_points,
        history_values=history_values,
        history_value_eps=history_value_eps,
        history_point_eps=history_point_eps,
    )
