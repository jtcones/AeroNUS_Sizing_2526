# Replication guide

How to reproduce the headline results from the ME4101A final report
on a fresh checkout. If you followed the Quick Start in `README.md`
you're already set up — skip to step 3.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm with:

```bash
python -c "import numpy, scipy, pandas, pygad, pyswarms, skopt; \
           print('OK:', numpy.__version__, pygad.__version__)"
```

## 2. Sanity check

Run the best-known GA configuration through the performance model:

```bash
python scripts/evaluate_config.py --config GA
```

Expected: `Convergence Status : PASS` and a full mission summary.
If this fails, fix the environment before continuing.

## 3. Reproduce the optimiser comparison

Run all three optimisers. Each writes a history CSV next to the
script. All three are fully deterministic except for internal RNG
state inside PyGAD and PySwarms — seed them if you need bit-exact
reproducibility.

```bash
# Brute force baseline (grid search)
python scripts/run_bruteforce.py
# -> Brute_Force_optimization_history.csv

# Genetic Algorithm
python scripts/run_ga.py --no-plot
# -> GA_optimization_history.csv

# Particle Swarm Optimisation
python scripts/run_pso.py --no-plot
# -> PSO_optimization_history.csv
```

**Runtimes (reference — Apple M2 / 16 GB / single process):**

| Stage               | Evaluations | Wall time   |
|---------------------|------------:|------------:|
| Brute force         | ~85 000     | ~30 min     |
| Genetic Algorithm   | ~225 000    | ~35 min     |
| Particle Swarm      | ~312 000    | ~45 min     |

The reported wall times are approximate and scale roughly linearly
with the number of evaluations.

## 4. Summary statistics

```bash
python scripts/analyze_results.py \
    Brute_Force_optimization_history.csv \
    GA_optimization_history.csv \
    PSO_optimization_history.csv
```

Prints total evaluations, convergence count, feasibility rate, wall
time, and best score for each file.

## 5. Reproducing the report's "best design" table

The best-known design vectors are stored in
`scripts/evaluate_config.py` under `REFERENCE_CONFIGS`. To
regenerate the per-mission breakdown for each:

```bash
python scripts/evaluate_config.py --config GA  > ga_best.txt
python scripts/evaluate_config.py --config PSO > pso_best.txt
python scripts/evaluate_config.py --config BF  > bf_best.txt
```

The output is identical to the summaries quoted in the report
(modulo floating-point noise in Python versions newer than the one
the report was written against).

## 6. Reproducing the hyperparameter tuning runs

Only needed if you've changed the objective function or the bounds.

```bash
python scripts/tune_ga.py    # 40 BO trials over (sol_per_pop, p_c, p_m, K)
python scripts/tune_pso.py   # 30 BO trials over (c1, c2, w)
```

Each BO trial runs a truncated inner GA/PSO and returns its best
fitness. Expect 1-2 hours per tuning run. When done, the best
hyperparameters are printed — paste them into the constants at the
top of `run_ga.py` or `run_pso.py`.

## 7. Troubleshooting differences

If your best score differs from the report's:

- **Minor (< 1%)** — expected. PyGAD and PySwarms have internal
  RNG state that isn't seeded by default; you won't get bit-exact
  reproducibility without explicit `numpy.random.seed()` and
  `random.seed()` calls in the drivers.
- **Major (> 5%)** — check:
  1. That you ran from the repository root (so `clarkY.dat`
     resolves).
  2. That `requirements.txt` installed cleanly (especially that
     `pygad >= 3.3.1` and `pyswarms >= 1.3.0` are present).
  3. That the mass-coherence check threshold in
     `AIAA_DBF_2526.RCPlane.check_mass_coherence` is still 1%.
  4. That no upstream library behaviour has changed (e.g.
     `brentq` tolerance, `pygad` default operators).

If you still can't reproduce, open an issue on the repository with
your `pip freeze` output and the best-score lines from each
history CSV.
