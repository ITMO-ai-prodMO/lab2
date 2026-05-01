"""
Эксперименты lab_02:
- стохастика (SA, PSO) на всех 5 функциях lab_02 + 3 функциях lab_01;
- классика (GD, NM, Newton) на функциях lab_01 для прямого сравнения;
- NM также на гладких мультимодальных (Rastrigin/Ackley/Himmelblau/Eggholder),
  чтобы показать, как 0-порядок без рестартов застревает в локальных минимумах.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions import (
    BlackBoxFunction,
    BadieFunction,
    EggholderFunction,
    HimmelblauFunction,
    Rosenbrock3D,
    QuadraticFunction,
    AckleyFunction,
    RastriginFunction,
    get_lab01_functions,
    get_lab02_functions,
)
from stochastic_optimizers import (
    OptimizationResult,
    particle_swarm,
    simulated_annealing,
)
from classical_optimizers import gradient_descent, nelder_mead, newton_method
from utils import seed_everything

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPSILON = 1e-4

START_POINTS = {
    "Quadratic-6D-good": np.array([4.0, -4.0, 2.0, 0.0, 0.0, 5.0]),
    "Quadratic-4D-bad": np.array([3.0, -3.0, 1.0, 4.0]),
    "Rosenbrock-3D": np.array([-1.2, 1.0, 0.8]),
    "Rastrigin-2D": np.array([3.0, -2.0]),
    "Ackley-2D": np.array([2.5, -2.5]),
    "Himmelblau-2D": np.array([0.0, 0.0]),
    "Eggholder-2D": np.array([0.0, 0.0]),
    "Badie-2D": np.array([1.0, 1.0]),
}

CLASSIC_LR = {
    "Quadratic-6D-good": 1.0,
    "Quadratic-4D-bad": 0.015,
    "Rosenbrock-3D": 3e-5,
}

GD_MAX_ITER = {
    "Quadratic-6D-good": 80,
    "Quadratic-4D-bad": 300,
    "Rosenbrock-3D": 250,
}

SA_PARAMS = {
    "default": dict(initial_temperature=10.0, cooling_rate=0.995, sigma_scale=0.1, max_iter=1500),
    "Eggholder-2D": dict(initial_temperature=200.0, cooling_rate=0.995, sigma_scale=0.05, max_iter=1500),
    "Rosenbrock-3D": dict(initial_temperature=5.0, cooling_rate=0.995, sigma_scale=0.05, max_iter=1500),
    "Quadratic-6D-good": dict(initial_temperature=5.0, cooling_rate=0.995, sigma_scale=0.05, max_iter=1500),
    "Quadratic-4D-bad": dict(initial_temperature=20.0, cooling_rate=0.995, sigma_scale=0.05, max_iter=1500),
}

PSO_PARAMS = {
    "default": dict(n_particles=30, inertia=0.7, cognitive=1.5, social=1.5, max_iter=80, velocity_scale=0.1),
    "Eggholder-2D": dict(n_particles=50, inertia=0.7, cognitive=1.5, social=1.5, max_iter=120, velocity_scale=0.05),
    "Rosenbrock-3D": dict(n_particles=40, inertia=0.7, cognitive=1.5, social=1.5, max_iter=120, velocity_scale=0.05),
    "Quadratic-6D-good": dict(n_particles=40, inertia=0.7, cognitive=1.5, social=1.5, max_iter=120, velocity_scale=0.05),
    "Quadratic-4D-bad": dict(n_particles=40, inertia=0.7, cognitive=1.5, social=1.5, max_iter=120, velocity_scale=0.05),
}


def _params(table: dict, name: str) -> dict:
    return table.get(name, table["default"])


def _sanitize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _result_row(result: OptimizationResult, group: str) -> dict:
    return {
        "group": group,
        "function": result.function_name,
        "method": result.method_name,
        "iterations": result.iterations,
        "elapsed_time_sec": result.elapsed_time,
        "value_calls": result.value_calls,
        "gradient_calls": result.gradient_calls,
        "hessian_calls": result.hessian_calls,
        "memory_scalars": result.memory_scalars,
        "final_value": result.final_value,
        "distance_to_optimum": result.distance_to_optimum,
        "final_value_eps": result.history_value_eps[-1] if result.history_value_eps else 0.0,
        "final_point_eps": result.history_point_eps[-1] if result.history_point_eps else 0.0,
    }


def run_stochastic_on(func: BlackBoxFunction, x0: np.ndarray, seed: int) -> List[OptimizationResult]:
    sa_params = _params(SA_PARAMS, func.name)
    pso_params = _params(PSO_PARAMS, func.name)
    sa = simulated_annealing(func, x0, epsilon=EPSILON, rng=np.random.default_rng(seed), **sa_params)
    pso = particle_swarm(func, x0, epsilon=EPSILON, rng=np.random.default_rng(seed + 1), **pso_params)
    return [sa, pso]


def run_classical_smooth(func: BlackBoxFunction, x0: np.ndarray) -> List[OptimizationResult]:
    """GD, NM, Newton — только для гладких функций lab_01 (где есть gradient/hessian)."""
    out: List[OptimizationResult] = []
    out.append(gradient_descent(func, x0, learning_rate=CLASSIC_LR[func.name], epsilon=EPSILON, max_iter=GD_MAX_ITER[func.name]))
    out.append(nelder_mead(func, x0, epsilon=EPSILON, step=0.5, max_iter=250))
    out.append(newton_method(func, x0, epsilon=EPSILON, max_iter=80))
    return out


def run_nm_only(func: BlackBoxFunction, x0: np.ndarray) -> OptimizationResult:
    """NM на мультимодальных функциях, где gradient/hessian не определены."""
    return nelder_mead(func, x0, epsilon=EPSILON, step=1.0, max_iter=250)


def plot_convergence(function_name: str, runs: List[OptimizationResult]) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in runs:
        history = np.array(result.history_values, dtype=float)
        ax.plot(history, label=f"{result.method_name} (calls={result.value_calls})")
    ax.set_xlabel("Iteration / record")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Convergence: {function_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = f"convergence_{_sanitize(function_name)}.png"
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    return filename


def plot_value_eps(function_name: str, runs: List[OptimizationResult]) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in runs:
        history = np.array(result.history_value_eps, dtype=float)
        if history.max() == 0:
            continue
        ax.plot(history, label=result.method_name)
    ax.set_xlabel("Iteration / record")
    ax.set_ylabel("epsilon of f(x)")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_title(f"Value epsilon dynamics: {function_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = f"value_eps_{_sanitize(function_name)}.png"
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    return filename


def plot_trajectory_2d(function_name: str, func: BlackBoxFunction, runs: List[OptimizationResult]) -> Optional[str]:
    if func.dimension != 2:
        return None
    bounds = func.bounds
    x_grid = np.linspace(bounds[0, 0], bounds[0, 1], 120)
    y_grid = np.linspace(bounds[1, 0], bounds[1, 1], 120)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = float(func.value([float(X[i, j]), float(Y[i, j])]))
    Z_clipped = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X, Y, Z_clipped, levels=30, cmap="viridis", alpha=0.7)
    for result in runs:
        traj = np.array(result.history_points, dtype=float)
        ax.plot(traj[:, 0], traj[:, 1], marker=".", markersize=3, linewidth=0.6, label=result.method_name)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], marker="*", s=120)
    ax.scatter([func.optimum[0]], [func.optimum[1]], marker="x", s=120, color="white", linewidths=2, label="known optimum")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Trajectories: {function_name}")
    ax.legend()
    fig.tight_layout()
    filename = f"trajectory_{_sanitize(function_name)}.png"
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    return filename


def run_all_experiments(seed: int = 42) -> Dict[str, object]:
    seed_everything(seed)
    rows: List[dict] = []
    runs_by_function: Dict[str, List[OptimizationResult]] = {}
    funcs_by_name: Dict[str, BlackBoxFunction] = {}

    # lab_01-style smooth functions: classical + stochastic
    for func in get_lab01_functions():
        funcs_by_name[func.name] = func
        x0 = START_POINTS[func.name]
        classic = run_classical_smooth(func, x0)
        stochastic = run_stochastic_on(func, x0, seed)
        all_runs = classic + stochastic
        runs_by_function[func.name] = all_runs
        for r in classic:
            rows.append(_result_row(r, "smooth"))
        for r in stochastic:
            rows.append(_result_row(r, "smooth"))

    # lab_02 multimodal / discontinuous: NM (где можно) + SA + PSO
    for func in get_lab02_functions():
        funcs_by_name[func.name] = func
        x0 = START_POINTS[func.name]
        runs: List[OptimizationResult] = []
        runs.append(run_nm_only(func, x0))
        runs.extend(run_stochastic_on(func, x0, seed))
        runs_by_function[func.name] = runs
        for r in runs:
            rows.append(_result_row(r, "multimodal" if func.name != "Badie-2D" else "discontinuous"))

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    plot_index = {}
    for function_name, runs in runs_by_function.items():
        plot_index[function_name] = {
            "convergence": plot_convergence(function_name, runs),
            "value_eps": plot_value_eps(function_name, runs),
        }
        traj = plot_trajectory_2d(function_name, funcs_by_name[function_name], runs)
        if traj:
            plot_index[function_name]["trajectory"] = traj

    with open(RESULTS_DIR / "plots.json", "w", encoding="utf-8") as f:
        json.dump(plot_index, f, indent=2, ensure_ascii=False)

    return {"summary": summary, "runs": runs_by_function, "plots": plot_index}


if __name__ == "__main__":
    result = run_all_experiments(seed=42)
    print(result["summary"].to_string(index=False))
