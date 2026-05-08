from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import numpy as np

from src.benchmarks import BenchmarkFunction
from src.constructive_number import ConstructiveNumber


@dataclass
class OptimizationResult:
    method: str
    best_point: np.ndarray
    best_value: float
    history_best: list[float]
    history_distance: list[float]
    calls: int
    elapsed_sec: float
    memory_bytes: int


class StochasticOptimizer:
    name = "base"

    def optimize(self, fn: BenchmarkFunction, seed: int) -> OptimizationResult:
        raise NotImplementedError

    @staticmethod
    def _cn_point(point: np.ndarray, eps: float) -> list[ConstructiveNumber]:
        return [ConstructiveNumber.from_value_eps(float(v), eps) for v in point]

    @staticmethod
    def _distance(point: np.ndarray, optimum: tuple[float, float]) -> float:
        return float(np.linalg.norm(point - np.asarray(optimum, dtype=float)))


class SimulatedAnnealing(StochasticOptimizer):
    name = "SimulatedAnnealing"

    def __init__(self, iterations: int = 900, t0: float = 1.0, cooling: float = 0.992, eps: float = 1e-6):
        self.iterations = iterations
        self.t0 = t0
        self.cooling = cooling
        self.eps = eps

    def optimize(self, fn: BenchmarkFunction, seed: int) -> OptimizationResult:
        rng = random.Random(seed)
        lo = np.array([b[0] for b in fn.bounds], dtype=float)
        hi = np.array([b[1] for b in fn.bounds], dtype=float)
        span = hi - lo
        current = fn.default_start()
        current_value = fn.evaluate(self._cn_point(current, self.eps)).mid
        best = current.copy()
        best_value = current_value
        history_best = [best_value]
        history_distance = [self._distance(best, fn.optimum)]
        calls = 1
        t0 = time.perf_counter()

        for k in range(self.iterations):
            temp = max(1e-12, self.t0 * (self.cooling ** k))
            step_scale = 0.2 * span * (0.15 + temp / self.t0)
            candidate = current + np.array([rng.gauss(0.0, step_scale[0]), rng.gauss(0.0, step_scale[1])])
            candidate = fn.clamp(candidate)
            candidate_value = fn.evaluate(self._cn_point(candidate, self.eps)).mid
            calls += 1
            delta = candidate_value - current_value
            if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
                current = candidate
                current_value = candidate_value
            if candidate_value < best_value:
                best = candidate.copy()
                best_value = candidate_value
            history_best.append(best_value)
            history_distance.append(self._distance(best, fn.optimum))

        memory = int(np.array(history_best).nbytes + np.array(history_distance).nbytes + best.nbytes)
        return OptimizationResult(self.name, best, best_value, history_best, history_distance, calls, time.perf_counter() - t0, memory)


class ParticleSwarm(StochasticOptimizer):
    name = "ParticleSwarm"

    def __init__(self, particles: int = 40, iterations: int = 260, inertia: float = 0.72, cognitive: float = 1.45, social: float = 1.45, eps: float = 1e-6):
        self.particles = particles
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.eps = eps

    def optimize(self, fn: BenchmarkFunction, seed: int) -> OptimizationResult:
        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in fn.bounds], dtype=float)
        hi = np.array([b[1] for b in fn.bounds], dtype=float)
        span = hi - lo
        positions = rng.uniform(lo, hi, size=(self.particles, 2))
        positions[0] = fn.default_start()
        velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(self.particles, 2))
        personal_best = positions.copy()
        personal_values = np.array([fn.evaluate(self._cn_point(p, self.eps)).mid for p in positions], dtype=float)
        calls = self.particles
        best_index = int(np.argmin(personal_values))
        global_best = personal_best[best_index].copy()
        global_value = float(personal_values[best_index])
        history_best = [global_value]
        history_distance = [self._distance(global_best, fn.optimum)]
        t0 = time.perf_counter()

        for _ in range(self.iterations):
            r1 = rng.random(size=(self.particles, 2))
            r2 = rng.random(size=(self.particles, 2))
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best - positions)
                + self.social * r2 * (global_best - positions)
            )
            velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
            positions = np.clip(positions + velocities, lo, hi)

            for i in range(self.particles):
                value = fn.evaluate(self._cn_point(positions[i], self.eps)).mid
                calls += 1
                if value < personal_values[i]:
                    personal_values[i] = value
                    personal_best[i] = positions[i].copy()
                    if value < global_value:
                        global_value = float(value)
                        global_best = positions[i].copy()
            history_best.append(global_value)
            history_distance.append(self._distance(global_best, fn.optimum))

        memory = int(positions.nbytes + velocities.nbytes + personal_best.nbytes + personal_values.nbytes + np.array(history_best).nbytes + np.array(history_distance).nbytes)
        return OptimizationResult(self.name, global_best, global_value, history_best, history_distance, calls, time.perf_counter() - t0, memory)
