"""Shared components for the GA / PSO / brute-force optimisation scripts.

Centralises the pieces that every search driver reuses:

* :func:`round_inches`, :func:`mission_2`, :func:`mission_3`, :func:`gm`
  — per-mission scoring from the DBF 2025/26 rulebook.
* :func:`engineering_bonus` — a small shaping term that nudges
  solutions toward physically credible configurations (L/D, stall
  margin, aspect ratio, flight-time envelope).
* :class:`OptimizationTracker` — logs every function evaluation for
  post-hoc analysis.
* :func:`make_objective` — returns a closure ``f(x) -> fitness`` that
  the individual drivers can hand to PyGAD / PySwarms / brute force,
  with a dedicated tracker attached.

Maximisation convention
-----------------------
``make_objective`` returns a function to be **maximised** (i.e. the
higher the value, the better). PSO expects a minimisation objective,
so the PSO driver wraps the returned function with a negation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from AIAA_DBF_2526 import RCPlane


# =====================================================================
# Mission scoring — AIAA DBF 2025/26 rulebook
# =====================================================================

def round_inches(x: float) -> float:
    """Round a measurement (in) down to the nearest 0, 0.25, 0.50, 0.75.

    Mission 3's banner length is scored on 0.25-inch increments, so
    this helper snaps the real-valued optimiser output to a legal
    banner size before scoring.

    Examples
    --------
    >>> round_inches(10.37)
    10.25
    >>> round_inches(5.81)
    5.75
    >>> round_inches(7.99)
    7.75
    """
    whole = int(x)
    frac = x - whole
    targets = [0.00, 0.25, 0.50, 0.75]
    rounded_frac = max([t for t in targets if t <= frac], default=0.00)
    return whole + rounded_frac


def mission_2(num_passengers: int, num_cargo: int, m2_laps: int, battery_capacity: float) -> float:
    """Mission 2 score: ``1 + NI / NI_max``.

    Net income ``NI = Income - Cost``, normalised by the best historical
    NI (``1940`` here — update if newer seasons publish a different cap).

    Parameters
    ----------
    num_passengers, num_cargo : int
        Payload counts for M2.
    m2_laps : int
        Completed laps within the 5-minute mission window.
    battery_capacity : float
        M2 battery capacity (Wh) — enters the energy-factor cost term.
    """
    income = (num_passengers * (6 + 2 * m2_laps)) + (num_cargo * (10 + 8 * m2_laps))
    EF = battery_capacity / 100
    cost = m2_laps * (10 + (num_passengers * 0.5) + (num_cargo * 2)) * EF
    net_income = income - cost
    return 1 + (net_income / 1940)


def mission_3(banner_length: float, number_of_laps: int, wing_span: float) -> float:
    """Mission 3 score: ``2 + m3 / m3_best``.

    ``m3 = rounded_banner_in * laps / RAC``, where RAC depends on the
    wing span in feet.
    """
    wing_span_inches = wing_span * 39.3701
    pre = round(wing_span_inches) / 12 * 0.05 + 0.75
    RAC = pre if pre >= 0.9 else 0.9
    rounded_banner = round_inches(banner_length)
    m3 = rounded_banner * number_of_laps / RAC
    m3_best = 1200
    return 2 + (m3 / m3_best)


def gm(num_passengers: int, num_cargo: int) -> float:
    """Ground-mission score. ``32.2 / (1.8*(pax+cargo) + 25)``."""
    return 32.2 / (1.8 * (num_passengers + num_cargo) + 25)


# =====================================================================
# Engineering-bonus shaping term
# =====================================================================

def engineering_bonus(plane: RCPlane) -> float:
    """Reward credible designs, penalise out-of-envelope ones.

    Components:

    * ``+0.1 * L/D_cruise`` — small pull toward efficient wings.
    * Penalty proportional to ``|stall_margin - 1.3|`` — keep cruise a
      healthy ~30% above stall.
    * Small bonus for higher AR, capped at AR=10 to avoid runaway
      aspect ratios.
    * Flight-time penalty above 335 s — the mission-window cap.
    """
    ld_bonus = 0.1 * plane.missions["1"].performance.L_D_cruise

    stall_val = plane.missions["1"].performance.stall_margin
    stall_bonus = -abs(stall_val - 1.3) * 0.5

    ar_bonus = 0.05 * plane.wing.aspect_ratio if plane.wing.aspect_ratio < 10 else 0

    ft_penalty = 0
    for i in ["1", "2", "3"]:
        flight_time = plane.missions[i].performance.flight_time
        if flight_time > 335:
            ft_penalty -= 2 + 0.1 * (flight_time - 335)

    return ld_bonus + stall_bonus + ar_bonus + ft_penalty


# =====================================================================
# Tracker
# =====================================================================

@dataclass
class OptimizationTracker:
    """Records every objective-function evaluation and writes to CSV.

    Each call to :meth:`log_evaluation` appends one row with the 11
    design variables, the fitness, any error message, and — if the
    RCPlane constructed successfully — a wide set of per-mission
    performance metrics for offline analysis.

    Parameters
    ----------
    algorithm_name : str
        Used as the CSV filename prefix.

    Examples
    --------
    >>> tracker = OptimizationTracker("GA")
    >>> # ... after running ...
    >>> tracker.save_to_csv()  # -> GA_optimization_history.csv
    """
    algorithm_name: str
    history: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def log_evaluation(self, x, plane: RCPlane | None, final_score: float,
                       bonus_score: float, error_msg: str = ""):
        """Append one evaluation to the history buffer.

        Parameters
        ----------
        x : sequence of 11 floats
            Decision vector in the canonical order.
        plane : RCPlane or None
            The constructed plane, or None if construction failed.
        final_score : float
            Competition score (M1 + M2 + M3 + GM, with propagation
            rules from the rulebook).
        bonus_score : float
            Engineering-bonus shaping term.
        error_msg : str, optional
            Free-form diagnostic.
        """
        (m_struct, wing_span, motor_power, wing_AR, n_pucks, pax_cargo_ratio,
         m1_batt, m2_batt, banner_len, banner_AR, m3_batt) = x

        record = {
            "iteration": len(self.history) + 1,
            "time_elapsed_sec": time.time() - self.start_time,
            "m_struct": m_struct, "wing_span": wing_span, "motor_power": motor_power,
            "wing_AR": wing_AR, "n_pucks": n_pucks, "pax_cargo_ratio": pax_cargo_ratio,
            "m1_battery": m1_batt, "m2_battery": m2_batt,
            "banner_length": banner_len, "banner_AR": banner_AR, "m3_battery": m3_batt,
            "error_msg": error_msg,
            "final_score": final_score,
            "bonus_score": bonus_score,
            "total_score": final_score + bonus_score,
        }

        if plane:
            record.update({
                "is_converged": plane.is_converged,
                "relative_error": plane.relative_error,
                "m_max": plane.m_max,
                "wing_loading_N_m2": plane.wing_loading,
                "power_to_weight_W_N": plane.power_to_weight,
                "gm_score": plane.missions.get("gm", 0),
            })
            for m_idx in ["1", "2", "3"]:
                if m_idx in plane.missions:
                    mission = plane.missions[m_idx]
                    perf = mission.performance
                    prefix = f"m{m_idx}_"
                    record.update({
                        f"{prefix}score": mission.score,
                        f"{prefix}payload_mass": mission.payload,
                        f"{prefix}battery_mass": mission.avionics.mass_battery,
                        f"{prefix}CL_cruise": perf.CL_cruise,
                        f"{prefix}V_cruise": perf.V_cruise,
                        f"{prefix}V_stall": perf.V_stall,
                        f"{prefix}n_turn": perf.n_turn,
                        f"{prefix}v_turn": perf.v_turn,
                        f"{prefix}turn_radius": perf.turn_radius,
                        f"{prefix}bank_angle_rad": perf.bank_angle,
                        f"{prefix}n_max": perf.n_max,
                        f"{prefix}max_v_turn": perf.max_v_turn,
                        f"{prefix}max_bank_angle_rad": perf.max_bank_angle,
                        f"{prefix}max_turn_radius": perf.max_turn_radius,
                        f"{prefix}flight_time": perf.flight_time,
                        f"{prefix}number_of_laps": perf.number_of_laps,
                        f"{prefix}L_D": perf.L_D_cruise,
                        f"{prefix}stall_margin": perf.stall_margin,
                        f"{prefix}load_factor_margin": perf.load_factor_margin,
                    })
        else:
            record.update({
                "is_converged": False, "relative_error": None, "m_max": None,
                "wing_loading_N_m2": None, "power_to_weight_W_N": None,
            })

        self.history.append(record)

    def save_to_csv(self, filename: str | None = None) -> pd.DataFrame:
        """Write the recorded history to CSV and return the DataFrame."""
        df = pd.DataFrame(self.history)
        if filename is None:
            filename = f"{self.algorithm_name}_optimization_history.csv"
        df.to_csv(filename, index=False)
        elapsed = self.history[-1]["time_elapsed_sec"] if self.history else 0.0
        print(f"Saved {len(self.history)} iterations to {filename}. "
              f"Total time: {elapsed:.1f} s")
        return df


# =====================================================================
# Objective function factory
# =====================================================================

def make_objective(tracker: OptimizationTracker) -> Callable:
    """Return a fitness function that closes over the given tracker.

    The returned function:

    * Accepts a length-11 decision vector ``x``.
    * Builds an :class:`RCPlane`, catching any physical-constraint
      exceptions (e.g. underpowered turns) and logging them.
    * Applies the DBF mission-score propagation rules (M1 gate, then
      M2, then M3, then GM).
    * Adds the engineering-bonus shaping term.
    * Returns a scalar **to be maximised**.

    Fitness rules of thumb
    ----------------------
    * Physical failure (exception raised)       -> very negative score
      scaled by relative mass error.
    * Mass not coherent (``is_converged`` False) -> scaled penalty with
      a small slice of performance to keep a smooth gradient.
    * Fully feasible                            -> ``performance * 10``.
    """

    def objective(x) -> float:
        (m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio,
         m1_battery, m2_battery, banner_length, banner_AR, m3_battery) = x

        try:
            plane = RCPlane(
                m_struct=m_struct, wing_span=wing_span, wing_AR=wing_AR,
                motor_power=motor_power, m1_battery=m1_battery,
                n_pucks=n_pucks, passenger_cargo_ratio=passenger_cargo_ratio,
                m2_battery=m2_battery, banner_length=banner_length,
                banner_AR=banner_AR, m3_battery=m3_battery,
            )

            # --- Per-mission scoring with propagation gates ---
            m1_score = 1 if plane.missions["1"].performance.number_of_laps > 2 else 0
            m2_score = mission_2(
                n_pucks * passenger_cargo_ratio, n_pucks,
                plane.missions["2"].performance.number_of_laps, m2_battery,
            )
            m3_score = mission_3(banner_length, plane.missions["3"].performance.number_of_laps, wing_span)
            gm_score = gm(n_pucks * passenger_cargo_ratio, n_pucks)

            plane.missions["1"].update_score(m1_score)
            plane.missions["2"].update_score(m2_score)
            plane.missions["3"].update_score(m3_score)
            plane.missions["gm"] = gm_score

            if not m1_score:
                final_score, error_msg = 0, "M1 Failed"
            elif not m2_score:
                final_score, error_msg = m1_score, "M2 Failed"
            elif not m3_score:
                final_score, error_msg = m1_score + m2_score, "M3 Failed"
            else:
                final_score = m1_score + m2_score + m3_score + gm_score
                error_msg = ""

            bonus = engineering_bonus(plane)
            rel_error = plane.relative_error
            convergence_penalty = rel_error * 1000
            total_performance = final_score + bonus

            if not plane.is_converged:
                tracker.log_evaluation(x, None, -1, 0, error_msg="Mass not Coherent")
                return -convergence_penalty + (total_performance * 0.1)
            else:
                tracker.log_evaluation(x, plane, final_score, bonus, error_msg=error_msg)
                return -convergence_penalty + (total_performance * 10)

        except Exception as error:
            tracker.log_evaluation(x, None, -1, 0, error_msg=str(error))
            return -100.0

    return objective


# =====================================================================
# Decision-vector bounds — single source of truth
# =====================================================================

# Canonical order. Used by every driver.
VAR_NAMES = [
    "m_struct", "wing_span", "motor_power", "wing_AR", "n_pucks",
    "pax_cargo_ratio", "m1_battery", "m2_battery", "banner_length",
    "banner_AR", "m3_battery",
]

# Domain-informed bounds (the "Stage C" bounds from the report).
# These reflect what the physics model can evaluate credibly; shrinking
# them further speeds convergence but risks missing corner configurations.
LOWER_BOUNDS = np.array([0.2, 0.92, 40,   4,  1, 3, 20, 20, 10,  1,  20])
UPPER_BOUNDS = np.array([13,  1.52, 4000, 10, 20, 10, 100, 100, 1000, 5, 100])

# PyGAD-style gene_space (per-variable dicts with low/high/step).
GENE_SPACE = [
    {"low": 0.2,  "high": 13},              # m_struct
    {"low": 0.92, "high": 1.52, "step": 0.01},  # wing_span
    {"low": 40,   "high": 4000, "step": 10},    # motor_power
    {"low": 4,    "high": 10,   "step": 0.1},   # wing_AR
    {"low": 1,    "high": 20,   "step": 1},     # n_pucks
    {"low": 3,    "high": 10,   "step": 1},     # pax_cargo_ratio
    {"low": 20,   "high": 100,  "step": 5},     # m1_battery
    {"low": 20,   "high": 100,  "step": 5},     # m2_battery
    {"low": 10,   "high": 1000, "step": 1},     # banner_length
    {"low": 1,    "high": 5,    "step": 0.1},   # banner_AR
    {"low": 20,   "high": 100,  "step": 5},     # m3_battery
]

# Indices that PSO should snap to discrete levels on each particle update.
# PSO is fundamentally continuous — without this, most particles live on
# fractional motor_power, fractional puck counts, etc. and never score.
INTEGER_INDICES = [2, 4, 5, 6, 7, 8, 10]          # motor_power, n_pucks, pax_cargo_ratio, batteries, banner_length
TWO_DECIMAL_INDICES = [3, 9]             # wing_AR, banner_AR (snap to 0.01)


def snap_pso_particle(particle: np.ndarray) -> np.ndarray:
    """Snap a PSO particle's discrete-valued dimensions to legal values.

    This addresses integer-boundary oscillation: without snapping,
    PSO spends many iterations on non-integer ``n_pucks`` values that
    the RCPlane constructor then silently coerces, wasting the search.
    """
    for idx in INTEGER_INDICES:
        particle[idx] = np.round(particle[idx])
    for idx in TWO_DECIMAL_INDICES:
        particle[idx] = np.round(particle[idx] * 100) / 100
    return particle


# =====================================================================
# Pretty-printing — useful after a run
# =====================================================================

def pretty_print_inputs(x):
    """Print the 11 design variables in a labelled column."""
    param_names = [
        "m_struct (kg)", "wing_span (m)", "motor_power (W)", "wing_AR",
        "n_pucks", "P/C_ratio", "m1_battery (Wh)", "m2_battery (Wh)",
        "banner_length (in)", "banner_AR", "m3_battery (Wh)",
    ]
    print("\n--- Best RCPlane Configuration ---")
    for name, value in zip(param_names, x):
        print(f"  {name:<25}: {value:>12}")


def pretty_print_plane(plane: RCPlane):
    """Print a human-readable summary of a constructed RCPlane."""
    import math
    print("=" * 55)
    print(" AIAA DBF 25/26 - AIRCRAFT DESIGN SUMMARY")
    print("=" * 55)

    status = "PASS" if plane.is_converged else "FAIL"
    print("\n[ SYSTEM CONVERGENCE ]")
    print(f"  Target Struct Mass : {plane.m_struct:.3f} kg")
    print(f"  Convergence Status : {status} (Error: {plane.relative_error * 100:.3f}%)")

    print("\n[ AEROSTRUCTURES ]")
    print("  Wing:")
    print(f"    - Airfoil        : {plane.wing.airfoil_type}")
    print(f"    - Dimensions     : Span = {plane.wing.span:.2f} m | "
          f"Chord = {plane.wing.chord:.3f} m")
    print(f"    - Aspect Ratio   : {plane.wing.aspect_ratio:.2f}  | "
          f"Area  = {plane.wing.surface_area:.3f} m^2")
    print(f"    - Mass           : {plane.wing.mass:.3f} kg")
    print("  Main Spar:")
    print(f"    - Material       : {plane.wing.spar.material} "
          f"({'Sized' if plane.wing.spar.is_sized else 'Unsized'})")
    print(f"    - Dimensions     : OD = {plane.wing.spar.outer_diameter * 1000:.1f} mm | "
          f"Wall = {plane.wing.spar.wall_thickness * 1000:.1f} mm")
    print(f"    - Mass           : {plane.wing.spar.mass:.3f} kg")
    print("  Empennage (Inverted T-Tail):")
    print(f"    - H-Tail         : Span = {plane.tail.span_H:.2f} m | "
          f"Area = {plane.tail.area_H:.3f} m^2")
    print(f"    - V-Tail         : Span = {plane.tail.span_V:.2f} m | "
          f"Area = {plane.tail.area_V:.3f} m^2")
    print(f"    - Mass           : {plane.tail.mass:.3f} kg")
    print("  Fuselage:")
    print(f"    - Mass           : {plane.fuselage.mass:.3f} kg")
    print(f"    - Wing AC Pos.   : {plane.fuselage.wing_ac_from_nose:.3f} m from nose")

    print("\n[ PROPULSION & POWER ]")
    print(f"  - Motor Max Power  : {plane.propulsion.motor_power:.1f} W")
    print(f"  - Effective Power  : {plane.propulsion.effective_power:.1f} W "
          f"(Motor {plane.propulsion.motor_eff*100:.0f}% / "
          f"Prop {plane.propulsion.prop_eff*100:.0f}%)")
    print(f"  - Propulsion Mass  : {plane.propulsion.mass:.3f} kg")

    print("\n" + "=" * 55)
    print(" FLIGHT MISSIONS ANALYSIS")
    print("=" * 55)

    for m_id, mission in plane.missions.items():
        if m_id == "gm":
            continue
        bank_deg = math.degrees(mission.performance.bank_angle)
        print(f"\n>>> MISSION {m_id} <<<")
        print(f"  - Payload        : {mission.payload:.3f} kg")
        print(f"  - Battery Mass   : {mission.avionics.mass_battery:.3f} kg "
              f"(Capacity: {mission.avionics.capacity:.1f})")
        print(f"  - Speeds         : Cruise = {mission.performance.V_cruise:.2f} m/s | "
              f"Stall = {mission.performance.V_stall:.2f} m/s")
        print(f"  - Aero           : CL_cruise = {mission.performance.CL_cruise:.3f} | "
              f"L/D = {mission.performance.L_D_cruise:.2f}")
        print(f"  - Safety Margins : Stall = {mission.performance.stall_margin:.2f}x | "
              f"Load Factor = {mission.performance.load_factor_margin:.2f}x")
        print(f"  - Turn Radius    : {mission.performance.turn_radius:.2f} m at "
              f"{mission.performance.v_turn:.2f} m/s")
        print(f"  - Bank Angle     : {bank_deg:.1f} deg")
        print(f"  - Load Factor    : {mission.performance.n_turn:.2f} g "
              f"(Max: {mission.performance.n_max:.2f} g)")
        print(f"  - Est. Time      : {mission.performance.flight_time:.1f} s")
        print(f"  - Laps in 5 min  : {mission.performance.number_of_laps:.0f} laps")
