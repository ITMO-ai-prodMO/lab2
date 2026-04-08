from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

from .constructive_number import ConstructiveNumber


def _to_float(value: object) -> float:
    if isinstance(value, ConstructiveNumber):
        return float(value)
    return float(value)


def _vector_to_float(values: Sequence[object]) -> list[float]:
    return [_to_float(value) for value in values]


def _mean_epsilon(values: Sequence[object]) -> float:
    eps = [float(value.epsilon) for value in values if isinstance(value, ConstructiveNumber)]
    return sum(eps) / len(eps) if eps else 0.0


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _wrap_point(point: Sequence[float], epsilon: float) -> list[ConstructiveNumber]:
    return [ConstructiveNumber.from_real(value, epsilon) for value in point]


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
    memory_units: int

    @property
    def resolved_point(self) -> list[float]:
        return _vector_to_float(self.point)

    @property
    def resolved_value(self) -> float:
        return _to_float(self.value)


def simulated_annealing(
    objective,
    x0: Sequence[object],
    temperature: float = 2.0,
    cooling: float = 0.97,
    step_scale: float = 0.35,
    max_iter: int = 250,
    seed: int = 42,
) -> OptimizationResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    current = _vector_to_float(x0)
    epsilon = _mean_epsilon(x0)
    current_wrapped = _wrap_point(current, epsilon)
    current_value = objective.value(current_wrapped)
    best = current[:]
    best_value = current_value
    history = [{"iteration": 0, "x": current[:], "value": _to_float(current_value), "epsilon": epsilon}]

    temp = temperature
    for iteration in range(1, max_iter + 1):
        candidate = []
        for index, value in enumerate(current):
            lower, upper = objective.bounds[index]
            proposal = value + rng.gauss(0.0, step_scale * max(temp, 0.05))
            candidate.append(_clip(proposal, (lower, upper)))
        candidate_wrapped = _wrap_point(candidate, epsilon)
        candidate_value = objective.value(candidate_wrapped)
        delta = _to_float(candidate_value) - _to_float(current_value)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-8)):
            current = candidate
            current_value = candidate_value
        if _to_float(current_value) < _to_float(best_value):
            best = current[:]
            best_value = current_value
        history.append(
            {"iteration": iteration, "x": best[:], "value": _to_float(best_value), "epsilon": epsilon}
        )
        temp *= cooling

    elapsed = time.perf_counter() - start
    best_wrapped = _wrap_point(best, epsilon)
    return OptimizationResult(
        method="simulated_annealing",
        point=best_wrapped,
        value=best_value,
        iterations=max_iter,
        elapsed_seconds=elapsed,
        history=history,
        value_calls=objective.value_calls,
        gradient_calls=objective.gradient_calls,
        hessian_calls=objective.hessian_calls,
        memory_units=len(current) * 4,
    )


def particle_swarm_optimization(
    objective,
    x0: Sequence[object],
    swarm_size: int = 18,
    inertia: float = 0.72,
    cognitive: float = 1.49,
    social: float = 1.49,
    max_iter: int = 180,
    seed: int = 7,
) -> OptimizationResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    center = _vector_to_float(x0)
    epsilon = _mean_epsilon(x0)
    dimension = len(center)

    positions = []
    velocities = []
    personal_best_positions = []
    personal_best_values = []

    for _ in range(swarm_size):
        point = []
        for idx, bounds in enumerate(objective.bounds):
            spread = 0.35 * (bounds[1] - bounds[0])
            point.append(_clip(center[idx] + rng.uniform(-spread, spread), bounds))
        positions.append(point)
        velocities.append([rng.uniform(-0.1, 0.1) for _ in range(dimension)])
        wrapped = _wrap_point(point, epsilon)
        value = objective.value(wrapped)
        personal_best_positions.append(point[:])
        personal_best_values.append(value)

    best_index = min(range(swarm_size), key=lambda idx: _to_float(personal_best_values[idx]))
    global_best = personal_best_positions[best_index][:]
    global_best_value = personal_best_values[best_index]

    history = [{"iteration": 0, "x": global_best[:], "value": _to_float(global_best_value), "epsilon": epsilon}]
    for iteration in range(1, max_iter + 1):
        for particle in range(swarm_size):
            for dim in range(dimension):
                r1 = rng.random()
                r2 = rng.random()
                velocities[particle][dim] = (
                    inertia * velocities[particle][dim]
                    + cognitive * r1 * (personal_best_positions[particle][dim] - positions[particle][dim])
                    + social * r2 * (global_best[dim] - positions[particle][dim])
                )
                positions[particle][dim] = _clip(
                    positions[particle][dim] + velocities[particle][dim],
                    objective.bounds[dim],
                )
            wrapped = _wrap_point(positions[particle], epsilon)
            value = objective.value(wrapped)
            if _to_float(value) < _to_float(personal_best_values[particle]):
                personal_best_positions[particle] = positions[particle][:]
                personal_best_values[particle] = value
                if _to_float(value) < _to_float(global_best_value):
                    global_best = positions[particle][:]
                    global_best_value = value
        history.append(
            {"iteration": iteration, "x": global_best[:], "value": _to_float(global_best_value), "epsilon": epsilon}
        )

    elapsed = time.perf_counter() - start
    best_wrapped = _wrap_point(global_best, epsilon)
    return OptimizationResult(
        method="particle_swarm",
        point=best_wrapped,
        value=global_best_value,
        iterations=max_iter,
        elapsed_seconds=elapsed,
        history=history,
        value_calls=objective.value_calls,
        gradient_calls=objective.gradient_calls,
        hessian_calls=objective.hessian_calls,
        memory_units=swarm_size * dimension * 4,
    )
