from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.benchmarks import BENCHMARKS, BenchmarkFunction
from src.optimizers import OptimizationResult, ParticleSwarm, SimulatedAnnealing, StochasticOptimizer


def default_optimizers() -> list[StochasticOptimizer]:
    return [SimulatedAnnealing(), ParticleSwarm()]


def run_all(seed: int = 42) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, fn in enumerate(BENCHMARKS):
        for optimizer in default_optimizers():
            result = optimizer.optimize(fn, seed=seed + index * 1009 + len(optimizer.name))
            rows.append(_row(fn, result))
    return rows


def _row(fn: BenchmarkFunction, result: OptimizationResult) -> dict[str, object]:
    return {
        "function": fn.name,
        "method": result.method,
        "best_x": result.best_point[0],
        "best_y": result.best_point[1],
        "best_value": result.best_value,
        "known_optimum_value": fn.optimum_value,
        "abs_value_error": abs(result.best_value - fn.optimum_value),
        "distance_to_known_optimum": np.linalg.norm(result.best_point - np.asarray(fn.optimum, dtype=float)),
        "function_calls": result.calls,
        "elapsed_sec": result.elapsed_sec,
        "memory_bytes": result.memory_bytes,
    }


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _method_names(rows: list[dict[str, object]]) -> list[str]:
    return sorted({str(r["method"]) for r in rows})


def _function_names() -> list[str]:
    return [fn.name for fn in BENCHMARKS]


def _value(rows: list[dict[str, object]], function_name: str, method: str, column: str) -> float:
    return float(next(r[column] for r in rows if r["function"] == function_name and r["method"] == method))


def save_summary_plot(rows: list[dict[str, object]], path: Path) -> None:
    save_metric_bar_plot(
        rows,
        column="abs_value_error",
        ylabel="|f(best) - f*|, log scale",
        path=path,
        log_scale=True,
    )


def save_metric_bar_plot(
    rows: list[dict[str, object]],
    column: str,
    ylabel: str,
    path: Path,
    log_scale: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = _method_names(rows)
    functions = _function_names()
    x = np.arange(len(functions))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(15, 7))
    start = -0.4 + width / 2
    for i, method in enumerate(methods):
        values = [_value(rows, fn, method, column) for fn in functions]
        if log_scale:
            values = [v + 1e-12 for v in values]
        ax.bar(x + start + i * width, values, width=width, label=method)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(functions, rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_error_heatmap(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = _method_names(rows)
    functions = _function_names()
    matrix = np.array(
        [[_value(rows, fn, method, "abs_value_error") + 1e-12 for method in methods] for fn in functions],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.5, 10))
    image = ax.imshow(np.log10(matrix), cmap="magma_r", aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(functions)))
    ax.set_yticklabels(functions)
    ax.set_title("log10 absolute value error")
    fig.colorbar(image, ax=ax, label="log10(|f(best)-f*|)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_winner_plot(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = _method_names(rows)
    wins = {method: 0 for method in methods}
    for fn in _function_names():
        best_method = min(methods, key=lambda method: _value(rows, fn, method, "abs_value_error"))
        wins[best_method] += 1
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#326273", "#e39774", "#6a994e", "#bc4749"]
    ax.bar(list(wins.keys()), list(wins.values()), color=colors[: len(wins)])
    ax.set_ylabel("Number of benchmark wins")
    ax.set_title("Which method found the closer function value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_all_plots(rows: list[dict[str, object]], out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    save_summary_plot(rows, out_dir / "value_error.png")
    save_metric_bar_plot(rows, "distance_to_known_optimum", "Distance to known optimum", plots_dir / "distance_to_optimum.png", log_scale=True)
    save_metric_bar_plot(rows, "function_calls", "Function calls", plots_dir / "function_calls.png")
    save_metric_bar_plot(rows, "elapsed_sec", "Elapsed time, seconds", plots_dir / "elapsed_time.png")
    save_metric_bar_plot(rows, "memory_bytes", "Estimated memory, bytes", plots_dir / "memory_usage.png")
    save_error_heatmap(rows, plots_dir / "error_heatmap.png")
    save_winner_plot(rows, plots_dir / "method_wins.png")


def save_method_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    lines = [
        "| Метод | Средняя ошибка значения | Медиана вызовов f | Среднее время, c | Средняя память, байт |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, items in sorted(grouped.items()):
        errors = np.array([float(r["abs_value_error"]) for r in items], dtype=float)
        calls = np.array([int(r["function_calls"]) for r in items], dtype=float)
        times = np.array([float(r["elapsed_sec"]) for r in items], dtype=float)
        memory = np.array([int(r["memory_bytes"]) for r in items], dtype=float)
        lines.append(f"| {method} | {errors.mean():.6g} | {np.median(calls):.0f} | {times.mean():.6f} | {memory.mean():.0f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
