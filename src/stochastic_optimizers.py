from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from constructive_number import ConstructiveNumber
from functions import (
    BlackBoxFunction,
    is_constructive,
    to_real_scalar,
    to_real_vector,
    vector_epsilons,
)


def make_constructive_vector(x: Sequence[float], epsilon: float) -> List[ConstructiveNumber]:
    return [ConstructiveNumber.from_real(float(v), float(epsilon)) for v in x]


def vector_to_floats(x: Sequence) -> np.ndarray:
    return to_real_vector(x, alpha=0.5)


def clip_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, bounds[:, 0]), bounds[:, 1])


def scalar_eps(value) -> float:
    if is_constructive(value):
        return float(value.epsilon)
    return 0.0


def point_eps_max(x: Sequence) -> float:
    eps = vector_epsilons(x)
    return float(eps.max()) if len(eps) else 0.0


@dataclass
class OptimizationResult:
    method_name: str
    function_name: str
    initial_point: np.ndarray
    final_point: np.ndarray
    final_value: float
    distance_to_optimum: float
    iterations: int
    elapsed_time: float
    value_calls: int
    gradient_calls: int
    hessian_calls: int
    memory_scalars: int
    history_points: List[np.ndarray] = field(default_factory=list)
    history_values: List[float] = field(default_factory=list)
    history_value_eps: List[float] = field(default_factory=list)
    history_point_eps: List[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _record(history_points, history_values, history_value_eps, history_point_eps, point, value):
    history_points.append(vector_to_floats(point))
    history_values.append(to_real_scalar(value))
    history_value_eps.append(scalar_eps(value))
    history_point_eps.append(point_eps_max(point))


def simulated_annealing(
    func: BlackBoxFunction,
    x0: Sequence[float],
    epsilon: float = 0.0,
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.95,
    sigma_scale: float = 0.1,
    max_iter: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> OptimizationResult:
    """
    Метод имитации отжига над ConstructiveNumber.

    Идея: на каждой итерации возмущаем текущую точку гауссовским шумом, всегда
    принимаем улучшение и принимаем ухудшение с вероятностью exp(-delta/T).
    Температура T убывает геометрически: T_{k+1} = cooling_rate * T_k.

    Не использует производных — поэтому подходит для разрывных функций.
    """
    func.reset_counters()
    if rng is None:
        rng = np.random.default_rng(0)

    x0 = np.asarray(x0, dtype=float)
    bounds = func.bounds
    sigma = sigma_scale * (bounds[:, 1] - bounds[:, 0])

    current_floats = clip_to_bounds(x0.copy(), bounds)
    current = make_constructive_vector(current_floats, epsilon) if epsilon > 0 else [float(v) for v in current_floats]
    current_value = func.value(current)

    best_floats = current_floats.copy()
    best = current
    best_value = current_value

    history_points, history_values, history_value_eps, history_point_eps = [], [], [], []
    _record(history_points, history_values, history_value_eps, history_point_eps, best, best_value)

    temperature = float(initial_temperature)
    accepted = 0
    start = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        proposal_floats = clip_to_bounds(current_floats + sigma * rng.standard_normal(len(current_floats)), bounds)
        proposal = make_constructive_vector(proposal_floats, epsilon) if epsilon > 0 else [float(v) for v in proposal_floats]
        proposal_value = func.value(proposal)

        delta = to_real_scalar(proposal_value) - to_real_scalar(current_value)
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current = proposal
            current_floats = proposal_floats
            current_value = proposal_value
            accepted += 1
            if to_real_scalar(current_value) < to_real_scalar(best_value):
                best = current
                best_floats = current_floats.copy()
                best_value = current_value

        _record(history_points, history_values, history_value_eps, history_point_eps, best, best_value)
        temperature *= cooling_rate

    elapsed = time.perf_counter() - start

    final_point = vector_to_floats(best)
    return OptimizationResult(
        method_name="Simulated Annealing",
        function_name=func.name,
        initial_point=x0,
        final_point=final_point,
        final_value=to_real_scalar(best_value),
        distance_to_optimum=float(np.linalg.norm(final_point - func.optimum)),
        iterations=max_iter,
        elapsed_time=elapsed,
        value_calls=func.counters.value_calls,
        gradient_calls=func.counters.gradient_calls,
        hessian_calls=func.counters.hessian_calls,
        memory_scalars=2 * func.dimension,
        history_points=history_points,
        history_values=history_values,
        history_value_eps=history_value_eps,
        history_point_eps=history_point_eps,
        extra={"accepted": accepted, "T0": initial_temperature, "cooling_rate": cooling_rate, "sigma_scale": sigma_scale},
    )


def particle_swarm(
    func: BlackBoxFunction,
    x0: Sequence[float],
    epsilon: float = 0.0,
    n_particles: int = 30,
    inertia: float = 0.7,
    cognitive: float = 1.5,
    social: float = 1.5,
    max_iter: int = 100,
    velocity_scale: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> OptimizationResult:
    """
    Particle Swarm Optimization над ConstructiveNumber.

    Популяция частиц движется по пространству:
        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x <- x + v
    Позиции хранятся как CN (с фиксированным input eps), скорости — обычные float.
    Не использует производных, естественно глобален.
    """
    func.reset_counters()
    if rng is None:
        rng = np.random.default_rng(0)

    bounds = func.bounds
    dim = func.dimension
    span = bounds[:, 1] - bounds[:, 0]
    v_max = velocity_scale * span * 5

    positions_float = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, dim))
    positions_float[0] = clip_to_bounds(np.asarray(x0, dtype=float), bounds)
    velocities = rng.uniform(-1.0, 1.0, size=(n_particles, dim)) * velocity_scale * span

    def wrap(point: np.ndarray):
        return make_constructive_vector(point, epsilon) if epsilon > 0 else [float(v) for v in point]

    positions = [wrap(p) for p in positions_float]
    values = [func.value(p) for p in positions]
    real_values = np.array([to_real_scalar(v) for v in values], dtype=float)

    pbest_float = positions_float.copy()
    pbest = [list(p) for p in positions]
    pbest_values = list(values)
    pbest_real = real_values.copy()

    gbest_idx = int(np.argmin(pbest_real))
    gbest_float = pbest_float[gbest_idx].copy()
    gbest = pbest[gbest_idx]
    gbest_value = pbest_values[gbest_idx]

    history_points, history_values, history_value_eps, history_point_eps = [], [], [], []
    _record(history_points, history_values, history_value_eps, history_point_eps, gbest, gbest_value)

    start = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest_float - positions_float)
            + social * r2 * (gbest_float - positions_float)
        )
        velocities = np.clip(velocities, -v_max, v_max)
        positions_float = clip_to_bounds(positions_float + velocities, bounds)

        for i in range(n_particles):
            positions[i] = wrap(positions_float[i])
            values[i] = func.value(positions[i])
            real_v = to_real_scalar(values[i])
            real_values[i] = real_v
            if real_v < pbest_real[i]:
                pbest[i] = list(positions[i])
                pbest_float[i] = positions_float[i].copy()
                pbest_values[i] = values[i]
                pbest_real[i] = real_v
                if real_v < to_real_scalar(gbest_value):
                    gbest = list(positions[i])
                    gbest_float = positions_float[i].copy()
                    gbest_value = values[i]

        _record(history_points, history_values, history_value_eps, history_point_eps, gbest, gbest_value)

    elapsed = time.perf_counter() - start
    final_point = vector_to_floats(gbest)
    return OptimizationResult(
        method_name="Particle Swarm",
        function_name=func.name,
        initial_point=np.asarray(x0, dtype=float),
        final_point=final_point,
        final_value=to_real_scalar(gbest_value),
        distance_to_optimum=float(np.linalg.norm(final_point - func.optimum)),
        iterations=max_iter,
        elapsed_time=elapsed,
        value_calls=func.counters.value_calls,
        gradient_calls=func.counters.gradient_calls,
        hessian_calls=func.counters.hessian_calls,
        memory_scalars=(3 * n_particles + 1) * dim,
        history_points=history_points,
        history_values=history_values,
        history_value_eps=history_value_eps,
        history_point_eps=history_point_eps,
        extra={
            "n_particles": n_particles,
            "inertia": inertia,
            "cognitive": cognitive,
            "social": social,
        },
    )
