"""Quick summary statistics over optimisation-history CSVs.

Prints total evaluations, convergence counts, wall time, and best
score for any optimisation-history CSV produced by
:class:`scripts.objective.OptimizationTracker`.

Usage
-----
From the repository root:

    python scripts/analyze_results.py GA_optimization_history.csv
    python scripts/analyze_results.py GA_optimization_history.csv PSO_optimization_history.csv

You can pass one or many CSVs — each is reported in turn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def analyse(df: pd.DataFrame) -> dict:
    """Compute summary stats for a single history DataFrame."""
    total_combinations = len(df)
    converged_mask = df["is_converged"].astype(bool) == True  # noqa: E712
    num_converged = int(converged_mask.sum())

    no_error_mask = df["error_msg"].isna() | (df["error_msg"] == "")
    converged_no_error = int((converged_mask & no_error_mask).sum())

    total_time_seconds = float(df["time_elapsed_sec"].max())
    best_score = float(df["final_score"].max())

    return {
        "total_combinations": total_combinations,
        "converged_count": num_converged,
        "converged_and_scoring_count": converged_no_error,
        "feasibility_rate_percent": 100 * converged_no_error / total_combinations
                                    if total_combinations else 0.0,
        "total_time_seconds": total_time_seconds,
        "best_final_score": best_score,
    }


def report(path: Path) -> None:
    """Load a CSV and print a labelled summary."""
    if not path.exists():
        print(f"[skip] {path} not found.")
        return

    print(f"\n=== {path.name} ===")
    df = pd.read_csv(path)
    stats = analyse(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<32s} {value:>12.2f}")
        else:
            print(f"  {key:<32s} {value:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="Paths to optimisation_history CSVs.")
    args = parser.parse_args()
    for csv in args.csvs:
        report(Path(csv))


if __name__ == "__main__":
    main()
