from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .constructive_number import ConstructiveNumber
from .deterministic_optimizers import gradient_descent, nelder_mead, newton_method, tune_learning_rate
from .objectives import Ackley2D, DesmosLike3D, Rastrigin2D, Step2D
from .stochastic_optimizers import particle_swarm_optimization, simulated_annealing


ROOT = Path(__file__).resolve().parents[1]
LAB1_ROOT = ROOT.parent / "metopt-lab1"
if str(LAB1_ROOT) not in sys.path:
    sys.path.append(str(LAB1_ROOT))

sns.set_theme(style="whitegrid")

OBJECTIVES = {
    "step_2d": Step2D,
    "rastrigin_2d": Rastrigin2D,
    "ackley_2d": Ackley2D,
    "desmos_like_3d": DesmosLike3D,
}

START_POINTS = {
    "step_2d": [4.2, -3.7],
    "rastrigin_2d": [3.5, -4.0],
    "ackley_2d": [3.0, -3.0],
    "desmos_like_3d": [2.8, -2.2, 1.6],
}

EPS_VALUES = (1e-2, 1e-4, 1e-6)
LR_CANDIDATES = (1e-3, 1e-2, 3e-2, 1e-1)


def make_constructive_vector(point: list[float], epsilon: float) -> list[ConstructiveNumber]:
    return [ConstructiveNumber.from_real(value, epsilon) for value in point]


def _slice_grid(objective, center: list[float], radius: float = 5.0, steps: int = 100):
    x = np.linspace(center[0] - radius, center[0] + radius, steps)
    y = np.linspace(center[1] - radius, center[1] + radius, steps)
    z = np.zeros((steps, steps))
    base = [0.0] * objective.dimension
    for idx in range(objective.dimension):
        base[idx] = objective.optimum_point[idx]
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            point = base.copy()
            point[0] = xi
            point[1] = yj
            z[j, i] = float(objective.value(point))
    objective.reset_counters()
    return x, y, z


def _plot_trajectory(output_dir: Path, objective_name: str, traces: dict[str, list[dict[str, object]]]) -> None:
    objective = OBJECTIVES[objective_name]()
    start = START_POINTS[objective_name]
    x_grid, y_grid, z_grid = _slice_grid(objective, start)
    plt.figure(figsize=(8, 6))
    levels = np.linspace(np.min(z_grid), np.max(z_grid), 18)
    plt.contour(x_grid, y_grid, z_grid, levels=levels, cmap="plasma")
    for label, history in traces.items():
        points = np.array([step["x"][:2] for step in history])
        plt.plot(points[:, 0], points[:, 1], marker="o", markersize=3, label=label)
    plt.title(f"Траектории методов: {objective_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"trajectory_{objective_name}.png", dpi=180)
    plt.close()


def _plot_values(output_dir: Path, objective_name: str, traces: dict[str, list[dict[str, object]]]) -> None:
    plt.figure(figsize=(8, 5))
    for label, history in traces.items():
        plt.plot(
            [step["iteration"] for step in history],
            [max(step["value"], 1e-12) for step in history],
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("Итерация")
    plt.ylabel("Лучшее значение функции")
    plt.title(f"Сходимость методов: {objective_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"convergence_{objective_name}.png", dpi=180)
    plt.close()


def _distance(point: list[float], optimum: list[float]) -> float:
    return float(np.linalg.norm(np.array(point) - np.array(optimum)))


def _append_row(rows: list[dict[str, object]], objective, epsilon: float, result, family: str) -> None:
    rows.append(
        {
            "objective": objective.name,
            "epsilon": epsilon,
            "family": family,
            "method": result.method,
            "iterations": result.iterations,
            "time_seconds": result.elapsed_seconds,
            "value": result.resolved_value,
            "distance_to_optimum": _distance(result.resolved_point, objective.optimum_point),
            "value_calls": result.value_calls,
            "gradient_calls": result.gradient_calls,
            "hessian_calls": result.hessian_calls,
            "memory_units": getattr(result, "memory_units", len(result.history) * objective.dimension),
            "final_epsilon": result.history[-1]["epsilon"],
        }
    )


def run_all(output_dir: str | Path = "img") -> list[dict[str, object]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for objective_name, builder in OBJECTIVES.items():
        for epsilon in EPS_VALUES:
            x0 = make_constructive_vector(START_POINTS[objective_name], epsilon)

            objective_sa = builder()
            sa_result = simulated_annealing(objective_sa, x0, max_iter=250)
            _append_row(rows, objective_sa, epsilon, sa_result, "stochastic")

            objective_pso = builder()
            pso_result = particle_swarm_optimization(objective_pso, x0, max_iter=180)
            _append_row(rows, objective_pso, epsilon, pso_result, "stochastic")

            learning_rate = tune_learning_rate(builder, x0, LR_CANDIDATES, max_iter=50)

            objective_nm = builder()
            nm_result = nelder_mead(objective_nm, x0, initial_step=0.4, max_iter=160)
            nm_result.memory_units = len(x0) * (len(x0) + 1)
            _append_row(rows, objective_nm, epsilon, nm_result, "deterministic")

            objective_gd = builder()
            gd_result = gradient_descent(objective_gd, x0, learning_rate=learning_rate, max_iter=120)
            gd_result.memory_units = len(x0) * 3
            _append_row(rows, objective_gd, epsilon, gd_result, "deterministic")

            objective_newton = builder()
            newton_result = newton_method(objective_newton, x0, max_iter=25, damping=0.5 if objective_name != "step_2d" else 0.25)
            newton_result.memory_units = len(x0) * len(x0) + len(x0)
            _append_row(rows, objective_newton, epsilon, newton_result, "deterministic")

            if epsilon == 1e-4:
                _plot_trajectory(
                    output_path,
                    objective_name,
                    {
                        "Simulated Annealing": sa_result.history,
                        "Particle Swarm": pso_result.history,
                        "Gradient Descent": gd_result.history,
                    },
                )
                _plot_values(
                    output_path,
                    objective_name,
                    {
                        "Simulated Annealing": sa_result.history,
                        "Particle Swarm": pso_result.history,
                        "Gradient Descent": gd_result.history,
                        "Newton": newton_result.history,
                    },
                )

    fieldnames = list(rows[0].keys())
    with (output_path / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    return rows


if __name__ == "__main__":
    run_all(ROOT / "img")
