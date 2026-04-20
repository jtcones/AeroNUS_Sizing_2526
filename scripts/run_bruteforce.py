"""Brute-force grid-search driver for the AIAA DBF sizing problem.

Iterates every combination on a coarse 11-dimensional grid (see
``grid_spaces`` below) and writes one row per evaluation to
``Brute_Force_optimization_history.csv``.

Purpose
-------
Brute force is not the fastest way to optimise this problem, but it
provides a **provably exhaustive** baseline over the chosen grid and
is useful as a sanity check against GA/PSO results on any new
objective function or rule set.

Usage
-----
From the repository root:

    python scripts/run_bruteforce.py

Output
------
A CSV file ``Brute_Force_optimization_history.csv`` in the working
directory, one row per grid point, plus a console printout of the
best score and elapsed wall time.

Tuning
------
Shrink the grid if runtime is a concern; the default grid has
~60k-100k combinations and takes on the order of 30 minutes on a
modern laptop. Coarsening ``n_pucks``, ``banner_length`` or
``motor_power`` has the largest payoff.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running the script from anywhere — add repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Must chdir into the repo root so that "clarkY.dat" resolves.
import os  # noqa: E402
os.chdir(REPO_ROOT)

from scripts.objective import OptimizationTracker, make_objective  # noqa: E402


def main():
    tracker = OptimizationTracker("Brute_Force")
    objective = make_objective(tracker)

    # Grid definition — tune these lists to trade runtime for coverage.
    # Order MUST match the canonical decision-vector order.
    grid_spaces = [
        np.linspace(0.5, 4, 7),      # m_struct
        np.linspace(0.92, 1.52, 6),  # wing_span
        np.linspace(300, 900, 4),    # motor_power
        np.linspace(4, 8, 7),        # wing_AR
        [1, 3, 5],                   # n_pucks
        [3, 5, 7],                   # passenger_cargo_ratio
        [50, 75, 100],               # m1_battery
        [50, 75, 100],               # m2_battery
        [200, 350, 500],             # banner_length
        [3, 5],                      # banner_AR
        [50, 75, 100],               # m3_battery
    ]

    all_combinations = list(itertools.product(*grid_spaces))
    print(f"Starting brute force over {len(all_combinations):,} combinations...")

    start = time.time()
    best_score = -np.inf

    pbar = tqdm(all_combinations, desc="Optimising")
    for combination in pbar:
        curr = objective(combination)
        if curr > best_score:
            best_score = curr
        pbar.set_postfix({"Curr": f"{curr:.2f}", "Best": f"{best_score:.2f}"})

    tracker.save_to_csv()
    print(f"\nDone. Best fitness: {best_score:.4f}. Wall time: {time.time() - start:.1f} s.")


if __name__ == "__main__":
    main()
