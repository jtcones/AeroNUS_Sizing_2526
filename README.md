# AeroNUS DBF 2025/26 — Sizing & Optimisation Framework

A physics-based performance model and optimisation framework for the
AIAA Design/Build/Fly (DBF) 2025/26 competition aircraft, built as
the optimisation component of an NUS Mechanical Engineering ME4101A
final-year project.

**What this gives you**

- A black-box aircraft performance function `F(x)` that takes 11
  design variables and returns per-mission scores.
- Three optimisation drivers over `F(x)`: brute-force grid search,
  Genetic Algorithm (PyGAD), and Particle Swarm (PySwarms).
- Bayesian-Optimisation (scikit-optimize) wrappers for tuning the
  GA and PSO hyperparameters.
- A reproducible CSV logging layer for post-hoc analysis.

**Intended audience**

Future AeroNUS DBF teams. The system is deliberately split into a
reusable RC-UAV performance core (`Components.py`) and a
competition-specific wrapper (`AIAA_DBF_2526.py`) so you can reuse
the modelling work and only rewrite the scoring rules when the
rulebook changes.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Project layout](#project-layout)
3. [Installation](#installation)
4. [How it works — 60 seconds](#how-it-works--60-seconds)
5. [Running the optimisers](#running-the-optimisers)
6. [Evaluating a single configuration](#evaluating-a-single-configuration)
7. [Analysing results](#analysing-results)
8. [Re-tuning hyperparameters](#re-tuning-hyperparameters)
9. [Optional: XFOIL integration](#optional-xfoil-integration)
10. [Extending for next season](#extending-for-next-season)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/jtcones/AeroNUS_Sizing_2526.git
cd AeroNUS_Sizing_2526

# 2. Install (use a virtualenv or conda env)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Sanity check — evaluate the best-known GA config
python scripts/evaluate_config.py

# 4. Run the real thing (several minutes each)
python scripts/run_ga.py
python scripts/run_pso.py
```

If `evaluate_config.py` prints an aircraft summary with
`Convergence Status: PASS`, the installation is working.

---

## Project layout

```
AeroNUS_Sizing_2526/
├── README.md                    # You are here
├── REPLICATION.md               # Step-by-step to reproduce the report
├── CONTRIBUTING.md              # How to extend / upgrade
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── Components.py                # Reusable RC-UAV component abstractions
├── AIAA_DBF_2526.py             # AIAA 2025/26 aircraft (concrete subclasses)
├── clarkY.dat                   # Clark Y airfoil coordinates
│
├── material_properties/         # Constants (density, UTS, areal mass, E)
│   ├── __init__.py
│   ├── density.py
│   ├── UTS.py
│   ├── Young_Modulus.py
│   └── areal_mass.py
│
└── scripts/
    ├── __init__.py
    ├── objective.py             # Shared objective, tracker, bounds
    ├── run_bruteforce.py        # Grid-search driver
    ├── run_ga.py                # Genetic Algorithm driver
    ├── run_pso.py               # Particle Swarm Optimisation driver
    ├── tune_ga.py               # BO hyperparameter tuning for GA
    ├── tune_pso.py              # BO hyperparameter tuning for PSO
    ├── evaluate_config.py       # Score a single design vector
    └── analyze_results.py       # Summary stats from history CSVs
```

---

## Installation

### Python version

Tested on **Python 3.11** and **3.12**. Newer versions should also
work; older ones may struggle with `pygad >= 3.x`.

### Step-by-step

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# or:  .venv\Scripts\activate     # Windows (PowerShell or CMD)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### About the `numpy < 2.0` pin

`scikit-optimize` (used for the BO tuning scripts) has internal
histogram code that breaks against numpy 2.x as of version 0.10.x.
If you upgrade `scikit-optimize` to a release that supports numpy 2,
feel free to drop the pin in `requirements.txt`.

### AeroPy and XFOIL

**Not required** for the shipped configuration (which uses the
`clarkY.dat` coordinates). They are only needed if you want to swap
in NACA airfoils generated on the fly, or extend the model with
real XFOIL polars. See [Optional: XFOIL integration](#optional-xfoil-integration)
below.

---

## How it works — 60 seconds

### The design vector

Every candidate aircraft is described by 11 numbers in a fixed order:

| # | Name               | Units | Type |
|--:|--------------------|-------|------|
| 1 | `m_struct`         | kg    | real |
| 2 | `wing_span`        | m     | real |
| 3 | `motor_power`      | W     | int  |
| 4 | `wing_AR`          | —     | real |
| 5 | `n_pucks`          | —     | int  |
| 6 | `pax_cargo_ratio`  | —     | int  |
| 7 | `m1_battery`       | Wh    | int  |
| 8 | `m2_battery`       | Wh    | int  |
| 9 | `banner_length`    | in    | int  |
|10 | `banner_AR`        | —     | real |
|11 | `m3_battery`       | Wh    | int  |

The bounds are defined once in `scripts/objective.py` (`LOWER_BOUNDS`,
`UPPER_BOUNDS`, `GENE_SPACE`) and reused by every driver.

### The performance model

Given a design vector, `RCPlane(**kwargs)`:

1. Sizes payloads for each mission (Mission 2 = pucks + ducks,
   Mission 3 = ripstop banner).
2. Builds avionics, propulsion, and a rectangular Clark Y wing.
3. Runs per-mission performance analysis — cruise CL from power
   balance (`brentq`), stall speed, sustained turn, lap count, flight
   time.
4. Sizes the main wing spar for the worst-case cruise speed across
   all missions.
5. Sizes the fuselage (semi-monocoque) and tail (inverted-T foam).
6. Runs a **mass-coherence check**: the assumed structural mass
   (`m_struct`) must match the sum of component masses within 1%.
   Designs that fail this are flagged `is_converged = False`.

### The objective function

`scripts/objective.py` turns a design vector into a scalar fitness:

- Sum of DBF mission scores (M1 + M2 + M3 + GM), with the rulebook's
  propagation gates (M2 only counts if M1 passed, etc.).
- Plus a small engineering-bonus term (L/D, stall margin, AR, flight
  time) to keep solutions physically credible.
- Designs that fail the mass-coherence check receive a heavy penalty
  but still see a small slice of their performance score so the
  optimiser has a gradient to follow.

### Why this design space

Mid-fidelity physics models can be exploited by aggressive
optimisers — for instance, pushing aspect ratio toward 50 because the
drag model rewards it. The bounds in `objective.py` are **deliberately
tight** around the region where the model is trustworthy. Loosen
them at your own risk; see the report for the full rationale.

---

## Running the optimisers

All scripts are run from the repository root. They:

- Print a progress bar / live best score.
- Save a CSV of every evaluation at the end.
- Print a full best-plane summary.

### Brute force (baseline)

```bash
python scripts/run_bruteforce.py
```

Default grid has roughly 60-100k combinations; typical runtime is
20-45 minutes. Writes `Brute_Force_optimization_history.csv`.

Use this as a **sanity check** for GA/PSO — if the GA can't find
something on par with brute force's best, the objective function
probably has a bug.

### Genetic Algorithm

```bash
python scripts/run_ga.py
```

Default: 450 individuals x 500 generations ≈ 225k evaluations. Uses
pre-tuned hyperparameters; override with e.g.
`--generations 100` for a quick smoke test.

Writes `GA_optimization_history.csv`.

### Particle Swarm Optimisation

```bash
python scripts/run_pso.py
```

Default: 250 particles x 1250 iterations ≈ 312k evaluations.

Writes `PSO_optimization_history.csv`.

**Flags**

- `--no-plot` — skip the final matplotlib window (useful on servers).
- `--iters N`, `--particles N` — trim the budget for quick runs.
- GA: `--generations N`.

---

## Evaluating a single configuration

### The best-known configs from the report

```bash
# Use the best GA result (default)
python scripts/evaluate_config.py

# Use the best PSO or brute-force result
python scripts/evaluate_config.py --config PSO
python scripts/evaluate_config.py --config BF
```

### Your own configuration

Pass 11 numbers in the canonical order:

```bash
python scripts/evaluate_config.py \
    0.44 0.95 820 7.5 1 3 65 65 351 4.9 95
```

Output includes wing geometry, spar sizing, tail geometry, fuselage
mass, and a full M1/M2/M3 performance breakdown.

---

## Analysing results

After any run, feed the CSV(s) to the analysis helper:

```bash
python scripts/analyze_results.py GA_optimization_history.csv
python scripts/analyze_results.py GA_optimization_history.csv PSO_optimization_history.csv
```

You'll get, per file:

- Total evaluations
- Converged (mass-coherent) count
- Converged-and-scoring count (feasibility rate)
- Wall time
- Best score

For deeper analysis, load the CSVs into pandas directly — each row
carries the design vector plus 50+ performance metrics per mission.

---

## Re-tuning hyperparameters

Only necessary if you've changed the objective function, the bounds,
or you've moved to a materially different version of PyGAD/PySwarms.

```bash
python scripts/tune_ga.py     # ~1-2 hours
python scripts/tune_pso.py    # ~1-2 hours
```

Each runs Bayesian Optimisation over an outer hyperparameter space,
with each trial being a truncated inner GA/PSO run. When done,
paste the printed best values back into the constants at the top of
`scripts/run_ga.py` or `scripts/run_pso.py`.

---

## Optional: XFOIL integration

The shipped setup uses Clark Y airfoil coordinates directly from
`clarkY.dat`, so XFOIL is **not** a runtime dependency. If you want
to experiment with NACA airfoils or swap in real XFOIL polars
(recommended — see "Extending for next season" below), follow these
steps:

### 1. Build or download XFOIL

- **Windows**: grab the `xfoil.exe` from
  [https://web.mit.edu/drela/Public/web/xfoil/](https://web.mit.edu/drela/Public/web/xfoil/)
  and place it somewhere on disk.
- **macOS / Linux**: clone the source and build. Easiest path on a
  Mac:
  ```bash
  brew install gcc
  git clone https://github.com/dgorodnichy/XFOIL-for-Mac.git
  cd XFOIL-for-Mac && make
  ```
  You'll end up with an `xfoil` binary. Move it somewhere
  permanent and make sure it is executable.

### 2. Install AeroPy

Uncomment the AeroPy line in `requirements.txt` and re-install:

```bash
pip install "git+https://github.com/leal26/AeroPy.git"
```

### 3. Point `AircraftConfig` at your XFOIL binary

Edit the default path in `Components.py`:

```python
@dataclass
class AircraftConfig:
    ...
    xfoil_path: str = r"../xfoil.exe"     # <-- change this
```

Or pass a custom `config` instance when constructing the wing.

### 4. Use a NACA airfoil

```python
wing = XPSRectangularWing(airfoil_type="naca2412", aspect_ratio=8, span=1.2)
```

The `Wing` class will lazily import AeroPy and shell out to your
XFOIL binary to generate the coordinates.

---

## Extending for next season

A few high-leverage places to start, roughly in order of
cost-to-benefit:

1. **Update the mission scoring formulas** in
   `scripts/objective.py` (`mission_2`, `mission_3`, `gm`) to the
   new rulebook. This is usually a 10-minute change.

2. **Update the design-variable bounds** in
   `scripts/objective.py` (`LOWER_BOUNDS`, `UPPER_BOUNDS`,
   `GENE_SPACE`) if the rule changes open up new payload or geometry
   regimes.

3. **Calibrate the empirical constants** in `AIAA_DBF_2526.py`
   against your prototype data — especially the fuselage sizing
   (`SemiMono._size_fuselage`) and drag (`CD0 = 0.13 + 0.01 *
   sqrt(n_ducks)` in `RCPlane.__post_init__`). These were fitted to
   previous AeroNUS builds and are the single biggest source of
   "simulation vs reality" gap.

4. **Swap parasite drag for XFOIL polars** at low Reynolds numbers.
   The current `CD0` model tends to overestimate drag (the report's
   L/D ratios consistently came out below 2), which distorts the
   optimum aspect ratio. A proper XFOIL-fed CD0 is the next big
   fidelity upgrade.

5. **Upgrade the optimiser** only *after* the above are settled.
   Candidate routes: multi-objective GA (NSGA-II) to get a Pareto
   front instead of a single point, or a surrogate-assisted BO if
   your physics model gets expensive enough to justify it.

See `CONTRIBUTING.md` for coding conventions and pull-request
expectations if you're working in a team.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'AeroPy'`**

You hit a NACA-airfoil code path without AeroPy installed. Either
install AeroPy (see [Optional: XFOIL integration](#optional-xfoil-integration))
or switch your wing to a `.dat` airfoil.

**`FileNotFoundError: Airfoil source 'clarkY.dat' not found`**

The scripts chdir into the repository root before running so that
`clarkY.dat` resolves relatively. If you're running code outside the
repo root, either chdir yourself or pass an absolute path to
`XPSRectangularWing(airfoil_type=...)`.

**`ValueError: N_max must be greater than 1`**

The aircraft is too underpowered to sustain a level turn at the
specified CL. Normal during an optimiser run — the tracker logs it
as a physical failure and the fitness function returns a heavy
penalty. If you're seeing it in `evaluate_config.py` for a
configuration that should be feasible, double-check the motor power
and wing area.

**PSO particles stuck at integer-boundary values**

That's the oscillation behaviour described in the report. The
`snap_pso_particle` helper in `objective.py` is the fix — make sure
the PSO wrapper is calling it before scoring each particle.

**`numpy.ndarray is not subscriptable` / weird scikit-optimize errors**

You've got numpy 2.x installed. The requirements file caps numpy at
`<2.0` for this reason. Recreate your virtualenv from
`requirements.txt`.

---

## References

- AIAA. AIAA Design/Build/Fly 2025–2026 Rules.
  [https://www.aiaa.org/dbf](https://www.aiaa.org/dbf)
- Raymer, D. P. *Aircraft Design: A Conceptual Approach*, 6th ed.,
  AIAA, 2018.
- Anderson, J. D. *Introduction to Flight*, McGraw-Hill.
- Phillips, W. F. *Mechanics of Flight*, Wiley, 2004.
- Kennedy, J. and Eberhart, R. C. "Particle Swarm Optimization",
  IEEE ICNN, 1995.
- Holland, J. H. *Adaptation in Natural and Artificial Systems*,
  University of Michigan Press, 1975.
- Gad, A. F. "PyGAD: An Intuitive Genetic Algorithm Python Library",
  *Multimedia Tools and Applications*, 2023.
- Miranda, L. J. V. and Dario, J. M. "PySwarms: A Research Toolkit
  for Particle Swarm Optimization in Python", JOSS, 2018.
- AeroPy: [https://github.com/leal26/AeroPy](https://github.com/leal26/AeroPy)

For the full methodology, results, and discussion, see the ME4101A
final report submitted alongside this repository.

---

## License

Internal NUS AeroNUS project code. Contact the original author before
redistributing outside the team.
