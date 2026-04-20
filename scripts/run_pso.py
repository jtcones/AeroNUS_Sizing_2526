"""Particle Swarm Optimisation driver for the AIAA DBF sizing problem.

Uses PySwarms' ``GlobalBestPSO`` with cognitive/social/inertia weights
tuned via Bayesian Optimisation (see ``scripts/tune_pso.py``).

Discrete variable handling
--------------------------
PSO is fundamentally continuous. Without intervention, most particles
spend their time at fractional values for ``n_pucks``, ``motor_power``,
etc., which the RCPlane constructor coerces silently — wasting
evaluations. The wrapper below snaps each particle's discrete-valued
dimensions *before* scoring via :func:`scripts.objective.snap_pso_particle`.

Usage
-----
From the repository root:

    python scripts/run_pso.py

Output
------
* ``PSO_optimization_history.csv`` — one row per fitness evaluation.
* A matplotlib cost-history plot (skip with ``--no-plot``).
* A full best-plane summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.objective import (  # noqa: E402
    OptimizationTracker, make_objective,
    LOWER_BOUNDS, UPPER_BOUNDS, snap_pso_particle,
    pretty_print_inputs, pretty_print_plane,
)
from AIAA_DBF_2526 import RCPlane  # noqa: E402


# --- PSO hyperparameters (from the BO tuning run) ---
C1 = 1.738336075497409   # cognitive
C2 = 1.3969107799046967  # social
W = 0.4500706027319868   # inertia
N_PARTICLES = 250
N_ITERS = 1250


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the cost-history plot at the end.")
    parser.add_argument("--iters", type=int, default=N_ITERS,
                        help=f"Number of PSO iterations (default: {N_ITERS}).")
    parser.add_argument("--particles", type=int, default=N_PARTICLES,
                        help=f"Swarm size (default: {N_PARTICLES}).")
    args = parser.parse_args()

    tracker = OptimizationTracker("PSO")
    objective_max = make_objective(tracker)

    def pso_objective_wrapper(particles: np.ndarray) -> np.ndarray:
        """Adapter: PySwarms passes a batch of particles, wants a
        minimisation objective. ``make_objective`` produces a max
        objective, so we negate before returning."""
        n_particles = particles.shape[0]
        scores = np.zeros(n_particles)
        for i in range(n_particles):
            snap_pso_particle(particles[i])
            scores[i] = -objective_max(particles[i])
        return scores

    bounds = (LOWER_BOUNDS, UPPER_BOUNDS)
    options = {"c1": C1, "c2": C2, "w": W}

    optimiser = ps.single.GlobalBestPSO(
        n_particles=args.particles,
        dimensions=11,
        options=options,
        bounds=bounds,
    )

    print(f"Running PSO: {args.particles} particles x {args.iters} iterations "
          f"= {args.particles * args.iters:,} evaluations")
    start = time.time()
    best_cost, best_pos = optimiser.optimize(pso_objective_wrapper, iters=args.iters)
    print(f"PSO complete. Wall time: {time.time() - start:.1f} s.")

    # best_cost is negative (we minimised -fitness), so the actual fitness is -best_cost.
    print(f"\nBest fitness: {-best_cost:.4f}")
    print(f"Best decision vector: {best_pos}")
    pretty_print_inputs(best_pos)

    (m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio,
     m1_battery, m2_battery, banner_length, banner_AR, m3_battery) = best_pos
    best_plane = RCPlane(
        m_struct=m_struct, wing_span=wing_span, wing_AR=wing_AR,
        motor_power=motor_power, m1_battery=m1_battery,
        n_pucks=int(n_pucks), passenger_cargo_ratio=int(passenger_cargo_ratio),
        m2_battery=m2_battery, banner_length=banner_length,
        banner_AR=banner_AR, m3_battery=m3_battery,
    )
    pretty_print_plane(best_plane)

    tracker.save_to_csv()

    if not args.no_plot:
        try:
            plot_cost_history(cost_history=optimiser.cost_history)
            plt.show()
        except Exception as e:
            print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
