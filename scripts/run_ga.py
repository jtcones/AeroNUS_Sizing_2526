"""Genetic Algorithm driver for the AIAA DBF sizing problem.

Uses PyGAD with tournament selection, single-point crossover,
random mutation, and a small elite carry-over.

Hyperparameters
---------------
The defaults below were selected via Bayesian Optimisation over the
outer-loop hyperparameter space (see ``scripts/tune_ga.py``). Re-run
that tuning if you materially change the objective function or the
bounds.

Usage
-----
From the repository root:

    python scripts/run_ga.py

Output
------
* ``GA_optimization_history.csv`` — one row per fitness evaluation.
* Console printout of the best design vector and its full RCPlane
  performance summary.
* A matplotlib window showing the best-fitness-per-generation curve
  (skip with ``--no-plot``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pygad

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.objective import (  # noqa: E402
    OptimizationTracker, make_objective, GENE_SPACE,
    pretty_print_inputs, pretty_print_plane,
)
from AIAA_DBF_2526 import RCPlane  # noqa: E402


# --- GA hyperparameters (from the BO tuning run) ---
SOL_PER_POP = 450
CROSSOVER_PROB = 0.65
MUTATION_PROB = 0.15814129005182623
K_TOURNAMENT = 5
NUM_GENERATIONS = 500
NUM_PARENTS_MATING = 30
KEEP_ELITISM = 2
MUTATION_PERCENT_GENES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the fitness plot at the end.")
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS,
                        help=f"Number of GA generations (default: {NUM_GENERATIONS}).")
    args = parser.parse_args()

    tracker = OptimizationTracker("GA")
    objective = make_objective(tracker)

    def fitness_func(ga_instance, solution, solution_idx):
        return objective(solution)

    ga = pygad.GA(
        num_generations=args.generations,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=SOL_PER_POP,
        num_genes=11,
        gene_space=GENE_SPACE,
        mutation_type="random",
        mutation_percent_genes=MUTATION_PERCENT_GENES,
        crossover_type="single_point",
        parent_selection_type="tournament",
        K_tournament=K_TOURNAMENT,
        keep_elitism=KEEP_ELITISM,
        mutation_probability=MUTATION_PROB,
        crossover_probability=CROSSOVER_PROB,
    )

    print(f"Running GA: {SOL_PER_POP} individuals x {args.generations} generations "
          f"= {SOL_PER_POP * args.generations:,} evaluations (upper bound)")
    start = time.time()
    ga.run()
    print(f"GA complete. Wall time: {time.time() - start:.1f} s.")

    solution, solution_fitness, _ = ga.best_solution()

    print(f"\nBest fitness: {solution_fitness:.4f}")
    print(f"Best decision vector: {solution}")
    pretty_print_inputs(solution)

    # Re-run the best plane through the constructor for a full summary.
    (m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio,
     m1_battery, m2_battery, banner_length, banner_AR, m3_battery) = solution
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
            ga.plot_fitness()
        except Exception as e:
            print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
