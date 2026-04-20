# Contributing

This repository is maintained by the NUS AeroNUS DBF team. Future
team members will inherit and extend it each season.

This guide exists so the next person picking this up can be
productive in a day, not a week.

## Ground rules

1. **Keep `Components.py` rule-set-agnostic.** Anything that depends
   on the DBF rulebook (scoring, payloads, mission geometry) goes in
   `AIAA_DBF_2526.py` or `scripts/objective.py`. This keeps the
   reusable core stable across seasons.
2. **No silent breaking changes to the design vector.** The 11
   variables are ordered in exactly one way, everywhere. If you add
   a twelfth, update every driver, every tracker column, and every
   reference config in the same commit.
3. **Every new model assumption gets a comment.** The empirical
   constants in `SemiMono._size_fuselage` and the `CD0` formula in
   `RCPlane.__post_init__` are the dangerous kind — they look like
   magic numbers but are actually fitted to specific prototypes.
   Leave a trail for whoever comes next.
4. **Use docstrings.** NumPy-style. All public classes and methods
   already have them; keep the standard when you add new ones.

## Where to make the common changes

### "We have new rules this season"

1. `scripts/objective.py` — `mission_2`, `mission_3`, `gm`,
   scoring gates in `make_objective`.
2. `scripts/objective.py` — `LOWER_BOUNDS`, `UPPER_BOUNDS`,
   `GENE_SPACE`.
3. `AIAA_DBF_2526.py` — payload sizing in `RCPlane.__post_init__`
   (pucks, ducks, banner materials).
4. `Components.py` — `FlightRouteParams` if the course geometry
   changes.

### "We have a new airframe"

1. `AIAA_DBF_2526.py` — replace `XPSRectangularWing`, `SemiMono`,
   or `CompressedFoamRectangleInvertedTTail` with new concrete
   subclasses. Copy the existing one as a starting template.
2. `material_properties/` — add any new materials. Each sub-module
   is a flat list of module-level constants; follow the existing
   units (kg, m, Pa).

### "We want better drag / aero predictions"

This is where XFOIL polars help. The current `CD0` estimate
(``0.13 + 0.01 * sqrt(n_ducks)``) is a conservative empirical fit.
Replacing it with an XFOIL-derived value per design vector is the
single biggest fidelity upgrade available. See the "Optional: XFOIL
integration" section of `README.md`.

### "We want to try a different optimiser"

Add a `scripts/run_<name>.py` in the same shape as the existing
drivers:

1. Construct a fresh `OptimizationTracker(<name>)`.
2. Call `make_objective(tracker)` once to get the fitness function.
3. Hand it to your optimiser library.
4. `tracker.save_to_csv()` at the end.
5. Print a best-plane summary via `pretty_print_plane`.

Candidate libraries worth trying: PyMOO (for NSGA-II and
multi-objective), Optuna (if you want TPE as an alternative to BO),
or a surrogate-assisted BO over the whole design vector if your
physics model gets expensive.

## Style

- 4-space indent, PEP 8 line length (soft 100).
- Type hints where they help readability; don't decorate every
  trivial helper.
- Prefer keyword arguments at call sites that use >3 positional
  arguments. `RCPlane(...)` is a good example.
- Imports: stdlib, then third-party, then local. Use absolute
  imports (`from scripts.objective import ...`, not relative).

## Testing

The repository currently has no unit tests — everything is
integration-tested by running the driver scripts. If you're adding
a subtle change (e.g. the sustained-turn calculation), consider
adding a `tests/` directory with `pytest` and pinning a few
known-good input/output pairs. The reference configurations in
`scripts/evaluate_config.py` are good candidates.

## Handing over

When you graduate, please:

1. Commit a final working `requirements.txt` frozen from a clean
   `pip install -r requirements.txt && pip freeze`.
2. Update `REPLICATION.md` with this year's wall times and any new
   quirks.
3. Add a short "What I'd do differently" note to the README or the
   report for the next person.

Good luck. Build planes, not powerpoints.
