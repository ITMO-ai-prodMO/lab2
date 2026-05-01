from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Sequence, Union

import pandas as pd

from .ConstructiveNumber import ConstructiveNumber
from .Optimization import optimize
from .StochasticOptimization import optimize_stochastic
from .TestFunctions import desmos_round_sin, himmelblau, matyas, rastrigin


Scalar = Union[int, float, ConstructiveNumber]
Function2D = Callable[[Sequence[Scalar], float], float]


@dataclass(frozen=True)
class TestProblem:
    name: str
    fn: Function2D
    bounds: tuple[tuple[float, float], tuple[float, float]]
    x0: tuple[float, float]
    ref_value: float | None = None


TEST_PROBLEMS: tuple[TestProblem, ...] = (
    TestProblem(
        name="matyas",
        fn=matyas,
        bounds=((-10.0, 10.0), (-10.0, 10.0)),
        x0=(4.0, -4.0),
        ref_value=0.0,
    ),
    TestProblem(
        name="himmelblau",
        fn=himmelblau,
        bounds=((-6.0, 6.0), (-6.0, 6.0)),
        x0=(-4.0, 0.0),
        ref_value=0.0,
    ),
    TestProblem(
        name="rastrigin",
        fn=rastrigin,
        bounds=((-5.12, 5.12), (-5.12, 5.12)),
        x0=(3.0, 3.0),
        ref_value=0.0,
    ),
    TestProblem(
        name="desmos_round_sin",
        fn=desmos_round_sin,
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
        x0=(1.0, 1.0),
    ),
)


DETERMINISTIC_METHODS = {"gradient_descent", "nelder_mead"}
STOCHASTIC_METHODS = {"simulated_annealing", "differential_evolution"}
DEFAULT_AUTO_LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _to_float(v: Scalar, alpha: float = 0.5) -> float:
    if isinstance(v, ConstructiveNumber):
        return v.to_real(alpha)
    return float(v)


class CallableFunctionBox:
    def __init__(self, problem: TestProblem, alpha: float = 0.5, grad_eps: float = 1e-6):
        self.problem = problem
        self.arity = 2
        self.name = problem.name
        self.alpha = alpha
        self.grad_eps = grad_eps
        self.objective_calls = 0

    def __call__(self, x: Sequence[Scalar]) -> float:
        return self.value(x)

    def value(self, x: Sequence[Scalar]) -> float:
        self.objective_calls += 1
        return float(self.problem.fn(x, self.alpha))

    def gradient(self, x: Sequence[Scalar]) -> list[float]:
        x0 = [_to_float(x[0], self.alpha), _to_float(x[1], self.alpha)]
        h = self.grad_eps
        grad = []
        for i in range(2):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h
            xm[i] -= h
            fp = self.value(xp)
            fm = self.value(xm)
            grad.append((fp - fm) / (2.0 * h))
        return grad

    def hessian(self, x: Sequence[Scalar]) -> list[list[float]]:
        x0 = [_to_float(x[0], self.alpha), _to_float(x[1], self.alpha)]
        h = self.grad_eps
        hess = [[0.0, 0.0], [0.0, 0.0]]
        f0 = self.value(x0)

        for i in range(2):
            xp = x0.copy()
            xm = x0.copy()
            xp[i] += h
            xm[i] -= h
            fp = self.value(xp)
            fm = self.value(xm)
            hess[i][i] = (fp - 2.0 * f0 + fm) / (h * h)

        xpp = [x0[0] + h, x0[1] + h]
        xpm = [x0[0] + h, x0[1] - h]
        xmp = [x0[0] - h, x0[1] + h]
        xmm = [x0[0] - h, x0[1] - h]
        cross = (self.value(xpp) - self.value(xpm) - self.value(xmp) + self.value(xmm)) / (4.0 * h * h)
        hess[0][1] = cross
        hess[1][0] = cross
        return hess


def _run_one(
    problem: TestProblem,
    method: str,
    deterministic_params: dict[str, dict] | None,
    stochastic_params: dict[str, dict] | None,
    alpha: float,
    grad_eps: float,
    cn_epsilon: float,
    success_f_tol: float,
    seed: int,
):
    box = CallableFunctionBox(problem, alpha=alpha, grad_eps=grad_eps)
    method_key = method.lower()

    tracemalloc.start()
    t0 = time.perf_counter()
    if method_key in DETERMINISTIC_METHODS:
        params = dict((deterministic_params or {}).get(method_key, {}))
        if method_key == "gradient_descent":
            params = dict(params)
            lr_value = params.get("learning_rate")
            if lr_value is True or (isinstance(lr_value, str) and lr_value.strip().lower() == "auto"):
                params.pop("learning_rate", None)
            params.pop("learning_rate_grid", None)
            params.pop("auto_lr_max_iter", None)
            params.pop("auto_lr_tol", None)
            params.pop("learning_rate_by_problem", None)
        res = optimize(box, list(problem.x0), method=method_key, **params)
    elif method_key in STOCHASTIC_METHODS:
        params = dict((stochastic_params or {}).get(method_key, {}))
        params.setdefault("seed", seed)
        params.setdefault("epsilon", cn_epsilon)
        if problem.ref_value is not None:
            params.setdefault("f_target", float(problem.ref_value))
            params.setdefault("f_tol", success_f_tol)
        res = optimize_stochastic(
            box,
            method=method_key,
            x0=list(problem.x0),
            bounds=list(problem.bounds),
            **params,
        )
    else:
        tracemalloc.stop()
        raise ValueError(f"Unknown optimization method: {method}")
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    f_best = float(res.f_best)
    if problem.ref_value is None:
        quality_success = float("nan")
    else:
        quality_success = float(abs(f_best - float(problem.ref_value)) <= success_f_tol)

    row = {
        "problem": problem.name,
        "method": method_key,
        "converged": bool(res.converged),
        "f_best": f_best,
        "quality_success": quality_success,
        "iterations": int(res.iterations),
        "time_sec": float(dt),
        "func_calls": int(box.objective_calls),
        "peak_memory_kib": float(peak / 1024.0),
    }
    return row, res


def run_comparative_report(
    methods: Sequence[str] | None = None,
    runs_per_method: int = 10,
    base_seed: int = 42,
    deterministic_params: dict[str, dict] | None = None,
    stochastic_params: dict[str, dict] | None = None,
    alpha: float = 0.5,
    grad_eps: float = 1e-6,
    cn_epsilon: float = 1e-8,
    success_f_tol: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_methods = [m.lower() for m in (methods or sorted(DETERMINISTIC_METHODS | STOCHASTIC_METHODS))]
    rows: list[dict] = []

    for p_idx, problem in enumerate(TEST_PROBLEMS):
        for m_idx, method in enumerate(selected_methods):
            for run_idx in range(runs_per_method):
                run_seed = base_seed + 1000 * p_idx + 100 * m_idx + run_idx
                row, _ = _run_one(
                    problem=problem,
                    method=method,
                    deterministic_params=deterministic_params,
                    stochastic_params=stochastic_params,
                    alpha=alpha,
                    grad_eps=grad_eps,
                    cn_epsilon=cn_epsilon,
                    success_f_tol=success_f_tol,
                    seed=run_seed,
                )
                row["run"] = run_idx
                rows.append(row)

    detailed = pd.DataFrame(rows)

    summary = (
        detailed.groupby(["problem", "method"], as_index=False)
        .agg(
            converged_rate=("converged", "mean"),
            success_rate=("quality_success", "mean"),
            success_support=("quality_success", "count"),
            f_best_mean=("f_best", "mean"),
            f_best_std=("f_best", "std"),
            iterations_mean=("iterations", "mean"),
            iterations_std=("iterations", "std"),
            time_sec_mean=("time_sec", "mean"),
            time_sec_std=("time_sec", "std"),
            func_calls_mean=("func_calls", "mean"),
            func_calls_std=("func_calls", "std"),
            peak_memory_kib_mean=("peak_memory_kib", "mean"),
            peak_memory_kib_std=("peak_memory_kib", "std"),
        )
        .sort_values(["problem", "iterations_mean", "time_sec_mean"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    return detailed, summary


def tune_learning_rate(
    problem_name: str,
    learning_rate_grid: Sequence[float] | str = "auto",
    max_iter: int = 3000,
    tol: float = 1e-8,
    alpha: float = 0.5,
    grad_eps: float = 1e-6,
) -> tuple[pd.DataFrame, float]:
    problem = next((p for p in TEST_PROBLEMS if p.name == problem_name), None)
    if problem is None:
        raise ValueError(f"Unknown problem: {problem_name}")

    if isinstance(learning_rate_grid, str):
        if learning_rate_grid.strip().lower() != "auto":
            raise ValueError("learning_rate_grid must be 'auto' or a sequence of floats")
        lr_candidates = DEFAULT_AUTO_LR_GRID
    else:
        lr_candidates = tuple(float(lr) for lr in learning_rate_grid)

    rows = []
    for lr in lr_candidates:
        box = CallableFunctionBox(problem, alpha=alpha, grad_eps=grad_eps)
        tracemalloc.start()
        t0 = time.perf_counter()
        res = optimize(
            box,
            list(problem.x0),
            method="gradient_descent",
            learning_rate=lr,
            max_iter=max_iter,
            tol=tol,
        )
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows.append(
            {
                "problem": problem.name,
                "learning_rate": float(lr),
                "converged": bool(res.converged),
                "iterations": int(res.iterations),
                "time_sec": float(dt),
                "func_calls": int(box.objective_calls),
                "peak_memory_kib": float(peak / 1024.0),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["converged", "iterations", "time_sec", "func_calls"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    return df, float(df.iloc[0]["learning_rate"])
