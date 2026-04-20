"""Evaluate a single RCPlane configuration and print the full summary.

Useful for:

* Re-running the report's reference results on a fresh checkout.
* Quickly scoring a hand-picked configuration without touching
  the optimisation loop.

Usage
-----
Run with the default configuration (best known GA solution):

    python scripts/evaluate_config.py

Pass your own 11-value decision vector (m_struct, span, motor_power,
wing_AR, n_pucks, pax_cargo_ratio, m1_batt, m2_batt, banner_len,
banner_AR, m3_batt) as positional arguments:

    python scripts/evaluate_config.py 0.44 0.95 820 7.5 1 3 65 65 351 4.9 95
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from AIAA_DBF_2526 import RCPlane  # noqa: E402
from scripts.objective import (  # noqa: E402
    OptimizationTracker, make_objective,
    pretty_print_inputs, pretty_print_plane,
)


# Best-known solutions from the final report (Stage C).
REFERENCE_CONFIGS = {
    "GA":  (0.4411255024983447, 0.950, 820.0, 7.5, 1, 3, 65.0, 65.0, 351.0, 4.9, 95),
    "PSO": (0.6607541203176531, 1.17, 586, 8.04, 1, 3, 68.07125875830295,
            54.417155370284334, 233.30466944397486, 2.42, 55),
    "BF":  (0.5, 0.92, 900.0, 4.666667, 1, 3, 75, 75, 350, 5, 100),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("values", nargs="*", type=float,
                        help="Optional: 11 numeric decision-variable values.")
    parser.add_argument("--config", choices=list(REFERENCE_CONFIGS.keys()),
                        default="GA",
                        help="Reference config to use if no values are given.")
    args = parser.parse_args()

    if args.values:
        if len(args.values) != 11:
            parser.error(f"Expected 11 values, got {len(args.values)}.")
        x = tuple(args.values)
    else:
        x = REFERENCE_CONFIGS[args.config]
        print(f"Using reference config: {args.config}")

    (m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio,
     m1_battery, m2_battery, banner_length, banner_AR, m3_battery) = x

    # Score with the real objective to populate mission scores.
    tracker = OptimizationTracker("single_eval")
    objective = make_objective(tracker)
    fitness = objective(x)
    print(f"\nFitness: {fitness:.4f}")

    # Build a clean RCPlane for the full printout.
    plane = RCPlane(
        m_struct=m_struct, wing_span=wing_span, wing_AR=wing_AR,
        motor_power=motor_power, m1_battery=m1_battery,
        n_pucks=int(n_pucks), passenger_cargo_ratio=int(passenger_cargo_ratio),
        m2_battery=m2_battery, banner_length=banner_length,
        banner_AR=banner_AR, m3_battery=m3_battery,
    )
    pretty_print_inputs(x)
    pretty_print_plane(plane)


if __name__ == "__main__":
    main()
