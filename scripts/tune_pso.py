"""Bayesian-Optimisation hyperparameter tuning for the PSO driver.

Wraps a truncated PSO (50 particles, 250 iterations) inside a
gp_minimize loop to select (c1, c2, w) — cognitive, social, and
inertia weights.

Why BO here
-----------
Each "evaluation" is itself a full PSO run, so it takes minutes.
That is exactly the regime where BO's sample efficiency pays off:
10 random + 20 model-guided trials are enough to land near the
optimum.

Usage
-----
    python scripts/tune_pso.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyswarms as ps
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.objective import (  # noqa: E402
    OptimizationTracker, make_objective,
    LOWER_BOUNDS, UPPER_BOUNDS, snap_pso_particle,
)


tracker = OptimizationTracker("PSO_tuning")
objective_max = make_objective(tracker)


def pso_objective_wrapper(particles: np.ndarray) -> np.ndarray:
    n_particles = particles.shape[0]
    scores = np.zeros(n_particles)
    for i in range(n_particles):
        snap_pso_particle(particles[i])
        scores[i] = -objective_max(particles[i])
    return scores


def pso_trial(hp):
    """Run a single inner PSO with the given (c1, c2, w)."""
    c1, c2, w = hp
    options = {"c1": c1, "c2": c2, "w": w}
    bounds = (LOWER_BOUNDS, UPPER_BOUNDS)
    optimiser = ps.single.GlobalBestPSO(
        n_particles=50, dimensions=11, options=options, bounds=bounds,
    )
    best_cost, _ = optimiser.optimize(pso_objective_wrapper, iters=250)
    return best_cost  # already a minimisation value


def main():
    search_space = [
        Real(0.5, 2.5, name="c1"),
        Real(0.5, 2.5, name="c2"),
        Real(0.4, 0.9, name="w"),
    ]

    print("Running Bayesian Optimisation over PSO hyperparameters...")
    result = gp_minimize(
        pso_trial,
        search_space,
        n_calls=30,
        n_initial_points=10,
        random_state=42,
    )

    print("\n=== PSO hyperparameter tuning complete ===")
    print(f"Best (c1, c2, w): {result.x}")
    print(f"Best inner-PSO best_cost: {result.fun:.4f}")

    tracker.save_to_csv("PSO_tuning_history.csv")

    plot_convergence(result)
    plt.show()


if __name__ == "__main__":
    main()
