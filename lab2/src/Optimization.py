from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Union

from .ConstructiveNumber import ConstructiveNumber


Scalar = Union[int, float, ConstructiveNumber]
Vector = Sequence[Scalar]


@dataclass
class OptimizationResult:
    method: str
    x_best: list[float]
    f_best: float
    iterations: int
    converged: bool
    history: list[float]


def _to_float(value: Scalar, alpha: float = 0.5) -> float:
    if isinstance(value, ConstructiveNumber):
        return value.to_real(alpha)
    return float(value)


def _vector_to_float(x: Vector, alpha: float = 0.5) -> list[float]:
    return [_to_float(v, alpha) for v in x]


def _evaluate(box: Any, x: list[float], alpha: float = 0.5) -> float:
    return _to_float(box(x), alpha)


def _gradient_to_float(box: Any, x: list[float], alpha: float = 0.5) -> list[float]:
    return [_to_float(g, alpha) for g in box.gradient(x)]


def _norm2(v: Sequence[float]) -> float:
    return sum(vi * vi for vi in v) ** 0.5


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _scale(v: Sequence[float], s: float) -> list[float]:
    return [s * vi for vi in v]


def gradient_descent(
    box: Any,
    x0: Vector,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha: float = 0.5,
    c1: float = 1e-4,
    backtracking: float = 0.5,
) -> OptimizationResult:

    x = _vector_to_float(x0, alpha)
    history: list[float] = []

    for it in range(1, max_iter + 1):
        f_x = _evaluate(box, x, alpha)
        grad = _gradient_to_float(box, x, alpha)
        grad_norm = _norm2(grad)
        history.append(f_x)

        if grad_norm <= tol:
            return OptimizationResult(
                method="gradient_descent",
                x_best=x,
                f_best=f_x,
                iterations=it,
                converged=True,
                history=history,
            )

        step = learning_rate
        while step > 1e-14:
            x_candidate = _sub(x, _scale(grad, step))
            f_candidate = _evaluate(box, x_candidate, alpha)

            # Условие Армихо.
            if f_candidate <= f_x - c1 * step * grad_norm * grad_norm:
                x = x_candidate
                break
            step *= backtracking

        if step <= 1e-14:
            return OptimizationResult(
                method="gradient_descent",
                x_best=x,
                f_best=f_x,
                iterations=it,
                converged=False,
                history=history,
            )

    return OptimizationResult(
        method="gradient_descent",
        x_best=x,
        f_best=_evaluate(box, x, alpha),
        iterations=max_iter,
        converged=False,
        history=history,
    )


def nelder_mead(
    box: Any,
    x0: Vector,
    step: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha: float = 0.5,
    reflection: float = 1.0,
    expansion: float = 2.0,
    contraction: float = 0.5,
    shrink: float = 0.5,
) -> OptimizationResult:

    x_start = _vector_to_float(x0, alpha)
    n = len(x_start)

    simplex: list[list[float]] = [x_start]
    for i in range(n):
        point = x_start.copy()
        point[i] += step
        simplex.append(point)

    history: list[float] = []

    for it in range(1, max_iter + 1):
        simplex.sort(key=lambda p: _evaluate(box, p, alpha))
        values = [_evaluate(box, p, alpha) for p in simplex]

        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        f_best = values[0]
        f_worst = values[-1]
        history.append(f_best)

        if abs(f_worst - f_best) <= tol:
            return OptimizationResult(
                method="nelder_mead",
                x_best=best,
                f_best=f_best,
                iterations=it,
                converged=True,
                history=history,
            )

        centroid = [0.0] * n
        for p in simplex[:-1]:
            centroid = _add(centroid, p)
        centroid = _scale(centroid, 1.0 / n)

        reflected = _add(centroid, _scale(_sub(centroid, worst), reflection))
        f_reflected = _evaluate(box, reflected, alpha)

        if values[0] <= f_reflected < values[-2]:
            simplex[-1] = reflected
            continue

        if f_reflected < values[0]:
            expanded = _add(centroid, _scale(_sub(reflected, centroid), expansion))
            f_expanded = _evaluate(box, expanded, alpha)
            simplex[-1] = expanded if f_expanded < f_reflected else reflected
            continue

        if f_reflected < values[-1]:
            contracted = _add(centroid, _scale(_sub(reflected, centroid), contraction))
        else:
            contracted = _add(centroid, _scale(_sub(worst, centroid), contraction))

        f_contracted = _evaluate(box, contracted, alpha)
        if f_contracted < min(f_worst, f_reflected):
            simplex[-1] = contracted
            continue

        new_simplex = [best]
        for p in simplex[1:]:
            new_simplex.append(_add(best, _scale(_sub(p, best), shrink)))
        simplex = new_simplex

    simplex.sort(key=lambda p: _evaluate(box, p, alpha))
    x_best = simplex[0]
    return OptimizationResult(
        method="nelder_mead",
        x_best=x_best,
        f_best=_evaluate(box, x_best, alpha),
        iterations=max_iter,
        converged=False,
        history=history,
    )


def optimize(
    box: Any,
    x0: Vector,
    method: str = "gradient_descent",
    **kwargs: Any,
) -> OptimizationResult:

    method_name = method.lower()
    if method_name in {"gradient_descent"}:
        return gradient_descent(box, x0, **kwargs)
    if method_name in {"nelder_mead"}:
        return nelder_mead(box, x0, **kwargs)
    raise ValueError()
