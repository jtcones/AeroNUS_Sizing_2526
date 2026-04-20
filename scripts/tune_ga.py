"""Bayesian-Optimisation hyperparameter tuning for the GA driver.

Wraps the GA in an outer gp_minimize loop (scikit-optimize). Each
"evaluation" runs a full but truncated GA (50 generations) and returns
its best fitness to the BO acquisition function. The BO surrogate
picks the next hyperparameter quadruple most likely to improve on the
best seen so far.

When to re-run
--------------
* After materially changing the objective function.
* After materially changing the bounds / gene_space.
* After moving to a new version of PyGAD that changes default operators.

Otherwise, the pre-tuned hyperparameters in :mod:`scripts.run_ga`
should be reused.

Usage
-----
    python scripts/tune_ga.py

Output
------
Prints the best ``(sol_per_pop, crossover_prob, mutation_prob, K_tournament)``
and renders a BO convergence plot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pygad
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from scripts.objective import OptimizationTracker, make_objective, GENE_SPACE  # noqa: E402


# Tuning-stage tracker (records every inner GA evaluation).
tracker = OptimizationTracker("GA_tuning")
objective = make_objective(tracker)


def ga_trial(hp):
    """Run a single inner GA with the given hyperparameters.

    Returns the *negative* best fitness, because gp_minimize is a
    minimiser and we want to maximise fitness.
    """
    sol_per_pop, crossover_probability, mutation_probability, K_tournament = hp
    sol_per_pop = int(sol_per_pop)

    def fitness_func(ga_instance, solution, solution_idx):
        return objective(solution)

    ga = pygad.GA(
        num_generations=50,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=11,
        gene_space=GENE_SPACE,
        mutation_type="random",
        mutation_percent_genes=10,
        crossover_type="single_point",
        parent_selection_type="tournament",
        K_tournament=K_tournament,
        keep_elitism=2,
        mutation_probability=mutation_probability,
        crossover_probability=crossover_probability,
    )
    ga.run()
    _, solution_fitness, _ = ga.best_solution()
    return -solution_fitness


def main():
    search_space = [
        Integer(100, 500, name="sol_per_pop"),
        Categorical([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                    name="crossover_probability"),
        Real(0.01, 0.2, name="mutation_probability"),
        Integer(3, 7, name="K_tournament"),
    ]

    print("Running Bayesian Optimisation over GA hyperparameters...")
    result = gp_minimize(
        ga_trial,
        search_space,
        n_calls=40,
        n_initial_points=10,
        random_state=42,
    )

    print("\n=== GA hyperparameter tuning complete ===")
    print(f"Best (sol_per_pop, crossover_prob, mutation_prob, K_tournament): {result.x}")
    print(f"Best inner-GA fitness: {-result.fun:.4f}")

    tracker.save_to_csv("GA_tuning_history.csv")

    plot_convergence(result)
    plt.show()


if __name__ == "__main__":
    main()
