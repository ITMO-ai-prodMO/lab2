from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence, Union

from .ConstructiveNumber import ConstructiveNumber

Scalar = Union[int, float, ConstructiveNumber]
Vector = Sequence[Scalar]


@dataclass
class StochasticOptimizationResult:
    method: str
    x_best: list[float]
    f_best: float
    iterations: int
    func_evals: int
    converged: bool
    history: list[float]


def _to_float(value: Scalar, alpha: float = 0.5) -> float:
    if isinstance(value, ConstructiveNumber):
        return value.to_real(alpha)
    return float(value)


def _vector_to_float(x: Vector, alpha: float = 0.5) -> list[float]:
    return [_to_float(v, alpha) for v in x]


def _to_cn_vector(x: Sequence[float], epsilon: float) -> list[ConstructiveNumber]:
    return [ConstructiveNumber.from_real(xi, epsilon) for xi in x]


def _evaluate(
    box: Any,
    x: Sequence[float],
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> float:
    x_cn = _to_cn_vector(x, epsilon)
    return _to_float(box(x_cn), alpha)


def _project_bounds(x: list[float], bounds: Sequence[tuple[float, float]]) -> list[float]:
    return [min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, bounds)]


def simulated_annealing(
    box: Any,
    x0: Vector,
    bounds: Sequence[tuple[float, float]],
    *,
    max_iter: int = 2000,
    temp_start: float = 1.0,
    temp_end: float = 1e-4,
    cooling_rate: float = 0.995,
    step_scale: float = 0.25,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
    f_target: float | None = None,
    f_tol: float = 1e-6,
    seed: int | None = None,
) -> StochasticOptimizationResult:
    if temp_start <= 0 or temp_end <= 0:
        raise ValueError()
    if not (0 < cooling_rate < 1):
        raise ValueError()

    rng = random.Random(seed)
    x_curr = _project_bounds(_vector_to_float(x0, alpha), bounds)
    f_curr = _evaluate(box, x_curr, alpha=alpha, epsilon=epsilon)
    x_best = x_curr.copy()
    f_best = f_curr

    history = [f_best]
    func_evals = 1
    temperature = temp_start

    for it in range(1, max_iter + 1):
        proposal = []
        for xi, (lo, hi) in zip(x_curr, bounds):
            span = hi - lo
            sigma = max(span * step_scale, 1e-12)
            proposal.append(xi + rng.gauss(0.0, sigma))
        proposal = _project_bounds(proposal, bounds)

        f_prop = _evaluate(box, proposal, alpha=alpha, epsilon=epsilon)
        func_evals += 1

        delta = f_prop - f_curr
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            x_curr = proposal
            f_curr = f_prop
            if f_curr < f_best:
                x_best = x_curr.copy()
                f_best = f_curr

        history.append(f_best)
        temperature *= cooling_rate

        if f_target is not None and f_best <= f_target + f_tol:
            return StochasticOptimizationResult(
                method="simulated_annealing",
                x_best=x_best,
                f_best=f_best,
                iterations=it,
                func_evals=func_evals,
                converged=True,
                history=history,
            )

        if temperature <= temp_end:
            return StochasticOptimizationResult(
                method="simulated_annealing",
                x_best=x_best,
                f_best=f_best,
                iterations=it,
                func_evals=func_evals,
                converged=False,
                history=history,
            )

    return StochasticOptimizationResult(
        method="simulated_annealing",
        x_best=x_best,
        f_best=f_best,
        iterations=max_iter,
        func_evals=func_evals,
        converged=False,
        history=history,
    )


def differential_evolution(
    box: Any,
    bounds: Sequence[tuple[float, float]],
    *,
    pop_size: int = 30,
    max_iter: int = 500,
    mutation: float = 0.8,
    crossover: float = 0.9,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
    seed: int | None = None,
    tol: float = 1e-8,
    min_iter: int = 30,
    patience: int = 25,
    f_target: float | None = None,
    f_tol: float = 1e-6,
) -> StochasticOptimizationResult:
    dim = len(bounds)
    if dim == 0:
        raise ValueError()
    if pop_size < 4:
        raise ValueError()
    if not (0 < mutation <= 2):
        raise ValueError()
    if not (0 <= crossover <= 1):
        raise ValueError()
    if min_iter < 1:
        raise ValueError()
    if patience < 1:
        raise ValueError()

    rng = random.Random(seed)

    def random_point() -> list[float]:
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    population = [random_point() for _ in range(pop_size)]
    scores = [_evaluate(box, x, alpha=alpha, epsilon=epsilon) for x in population]
    func_evals = pop_size

    best_idx = min(range(pop_size), key=lambda i: scores[i])
    x_best = population[best_idx].copy()
    f_best = scores[best_idx]
    history = [f_best]
    stagnant_iters = 0

    for it in range(1, max_iter + 1):
        best_before_iter = f_best
        for i in range(pop_size):
            candidates = [j for j in range(pop_size) if j != i]
            a_idx, b_idx, c_idx = rng.sample(candidates, 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]

            mutant = [a[d] + mutation * (b[d] - c[d]) for d in range(dim)]
            mutant = _project_bounds(mutant, bounds)

            j_rand = rng.randrange(dim)
            trial = []
            for d in range(dim):
                if rng.random() < crossover or d == j_rand:
                    trial.append(mutant[d])
                else:
                    trial.append(population[i][d])

            f_trial = _evaluate(box, trial, alpha=alpha, epsilon=epsilon)
            func_evals += 1

            if f_trial <= scores[i]:
                population[i] = trial
                scores[i] = f_trial
                if f_trial < f_best:
                    x_best = trial.copy()
                    f_best = f_trial

        history.append(f_best)
        if f_target is not None and f_best <= f_target + f_tol:
            return StochasticOptimizationResult(
                method="differential_evolution",
                x_best=x_best,
                f_best=f_best,
                iterations=it,
                func_evals=func_evals,
                converged=True,
                history=history,
            )

        improvement = best_before_iter - f_best
        if improvement > tol:
            stagnant_iters = 0
        else:
            stagnant_iters += 1

        if it >= min_iter and stagnant_iters >= patience:
            converged = True if f_target is None else (f_best <= f_target + f_tol)
            return StochasticOptimizationResult(
                method="differential_evolution",
                x_best=x_best,
                f_best=f_best,
                iterations=it,
                func_evals=func_evals,
                converged=converged,
                history=history,
            )

    return StochasticOptimizationResult(
        method="differential_evolution",
        x_best=x_best,
        f_best=f_best,
        iterations=max_iter,
        func_evals=func_evals,
        converged=False,
        history=history,
    )


def optimize_stochastic(
    box: Any,
    *,
    method: str,
    x0: Vector | None = None,
    bounds: Sequence[tuple[float, float]],
    **kwargs: Any,
) -> StochasticOptimizationResult:
    method_name = method.lower()
    if method_name == "simulated_annealing":
        if x0 is None:
            raise ValueError()
        return simulated_annealing(box, x0, bounds, **kwargs)
    if method_name == "differential_evolution":
        return differential_evolution(box, bounds, **kwargs)
    raise ValueError()
