"""Concrete AIAA DBF 2025/26 aircraft model.

This module plugs a specific aircraft configuration — an XPS-foam,
rectangular-wing, semi-monocoque, inverted-T-tail RC UAV — into the
abstract component interfaces declared in :mod:`Components`. It also
assembles the three competition missions and closes the mass-coherence
loop.

Design-variable interface
-------------------------
An ``RCPlane`` instance is fully parameterised by the eleven design
variables listed in the report (see Table 3.1):

==============  =========  =============================================
Name            Type       Meaning
==============  =========  =============================================
m_struct        Real (kg)  Assumed structural mass (convergence target)
wing_span       Real (m)   Tip-to-tip wing span
motor_power     Int  (W)   Motor rated input power
wing_AR         Real       Wing aspect ratio b^2 / S
n_pucks         Int        Number of hockey pucks (Mission 2)
pax_cargo_ratio Int        Passengers per puck (Mission 2)
m1_battery      Int  (Wh)  Mission 1 battery capacity
m2_battery      Int  (Wh)  Mission 2 battery capacity
banner_length   Int  (in)  Banner length (Mission 3)
banner_AR       Real       Banner aspect ratio (Mission 3)
m3_battery      Int  (Wh)  Mission 3 battery capacity
==============  =========  =============================================

Usage
-----
>>> from AIAA_DBF_2526 import RCPlane
>>> plane = RCPlane(
...     m_struct=0.66, wing_span=1.17, wing_AR=8.04,
...     motor_power=586, n_pucks=1, passenger_cargo_ratio=3,
...     m1_battery=68.07, m2_battery=54.42,
...     banner_length=233, banner_AR=2.42, m3_battery=55,
... )
>>> plane.is_converged
True

See ``scripts/evaluate_config.py`` for a ready-to-run example.
"""

from Components import (
    Wing, RectangularWing, Fuselage, Tail, Performance, Propulsion,
    FlightMission, LandingGear, Avionics, Plane,
)
from dataclasses import dataclass, field, InitVar
import numpy as np
from material_properties import areal_mass, density, UTS  # noqa: F401 (UTS used by Components)
from scipy.optimize import brentq


# =====================================================================
# Wing — XPS foam core, rectangular planform, shrink-wrap skin
# =====================================================================

@dataclass
class XPSRectangularWing(RectangularWing):
    """Rectangular wing built from blue XPS foam with a shrink-wrap skin.

    Mass model: wing volume (airfoil area x chord^2 x span) in XPS foam
    plus a shrink-wrap area term (approximated as four times the
    airfoil perimeter ratio times chord times span, to cover both
    surfaces and the leading/trailing edges conservatively).
    """

    def _calculate_mass(self) -> float:
        """Estimate wing mass (kg) as foam core + shrink-wrap skin.

        Excludes the main spar, which is tracked separately on
        ``self.spar`` and added in the mass-coherence check.
        """
        wing_volume = self.airfoil_area_ratio * self.chord ** 2 * self.span
        wrap_area = self.airfoil_circum_ratio * self.chord * 4 * self.span

        mass = wing_volume * density.blue_xps_foam
        mass += wrap_area * areal_mass.shrink_wrap
        return mass


# =====================================================================
# Fuselage — semi-monocoque box
# =====================================================================

@dataclass
class SemiMono(Fuselage):
    """Simple box-fuselage sizing driven by wing span and chord.

    Empirical scaling laws (``length_constant`` etc.) were calibrated
    to past AeroNUS DBF prototypes and should be re-measured against
    any new airframe. Also reports the wing / tail aerodynamic-centre
    positions along the fuselage for tail-moment-arm calculations.

    Parameters
    ----------
    wing_span : float
        Wing span (m). Drives fuselage length.
    wing_c : float
        Wing mean chord (m). Used to place the wing AC along the
        fuselage.

    Computed attributes
    -------------------
    mass : float
        Total fuselage mass (kg) = skin + internal structure.
    wing_ac_from_nose : float
        Position of wing aerodynamic centre from the nose (m).
    tail_ac_from_nose : float
        Position of tail aerodynamic centre from the nose (m).
    wing_ac_to_tail_ac : float
        Moment arm between wing AC and tail AC (m).
    """
    wing_span: InitVar[float] = None
    wing_c: InitVar[float] = None

    def __post_init__(self, wing_span, wing_c):
        self.mass = self._size_fuselage(wing_span, wing_c)

    def _size_fuselage(self, wing_span, wing_c) -> float:
        """Calibrated empirical sizing. See source for constants."""
        # Empirical fuselage sizing constants.
        int_struc_percentage = 0.1
        length_constant = 0.85
        width_constant = 0.09
        height_constant = 0.09
        AC_chord_ratio = 0.25
        nose_ratio = 0.15
        wing_ac_to_tail_ac = 0.88

        length = wing_span * length_constant
        width = length * width_constant
        height = length * height_constant

        # External box surface area -> skin volume -> skin mass.
        front_area = width * height
        bottom_area = length * width
        left_area = length * height
        fuselage_skin_area = 2 * front_area + 2 * bottom_area + 2 * left_area
        fuselage_skin_volume = fuselage_skin_area * 0.005
        fuselage_skin_mass = fuselage_skin_volume * density.white_foam

        # Internal bulkheads / stringers as a fraction of gross volume.
        structural_volume = length * width * height * int_struc_percentage
        structural_mass = structural_volume * density.basswood

        total_mass = fuselage_skin_mass + structural_mass

        # Aerodynamic-centre positions, for tail moment-arm calc.
        wing_ac_from_nose = length * nose_ratio + wing_c * AC_chord_ratio
        tail_ac_from_nose = length * wing_ac_to_tail_ac
        wing_ac_to_tail_ac = tail_ac_from_nose - wing_ac_from_nose

        self.wing_ac_from_nose = wing_ac_from_nose
        self.tail_ac_from_nose = tail_ac_from_nose
        self.wing_ac_to_tail_ac = wing_ac_to_tail_ac
        return total_mass


# =====================================================================
# Tail — inverted T, compressed foam + carbon spar + fibre tape
# =====================================================================

@dataclass
class CompressedFoamRectangleInvertedTTail(Tail):
    """Inverted-T empennage sized by tail volume coefficients.

    Uses the classical tail-volume approach (Raymer Ch. 6):
    ``S_HT = V_H * c_wing * S_wing / L_HT`` etc. Mass estimation
    includes flat compressed-foam surfaces, a small carbon spar for
    each tail panel, and fibre tape along the span.

    Parameters
    ----------
    tail_coefficient_H : float
        Horizontal tail volume coefficient (V_H).
    tail_coefficient_V : float
        Vertical tail volume coefficient (V_V).
    taper_ratio : float
        Taper ratio (tip_chord / root_chord). 1.0 for a rectangular tail.
    moment_arm : float
        Wing-AC to tail-AC distance (m).
    wing_area : float
        Wing reference area (m^2).
    wing_span : float
        Wing span (m).
    wing_AR : float
        Wing aspect ratio.
    wing_chord : float
        Wing mean chord (m).
    """
    tail_coefficient_H: float
    tail_coefficient_V: float
    taper_ratio: float

    moment_arm: InitVar[float] = None
    wing_area: InitVar[float] = None
    wing_span: InitVar[float] = None
    wing_AR: InitVar[float] = None
    wing_chord: InitVar[float] = None

    def __post_init__(self, moment_arm, wing_area, wing_span, wing_AR, wing_chord):
        (self.span_H, self.chord_H, self.area_H, self.AR_H,
         self.span_V, self.chord_V, self.area_V, self.AR_V) = \
            self._calculate_geometry(moment_arm, wing_area, wing_span, wing_AR, wing_chord)
        self.mass = self._calculate_mass()

    def _calculate_geometry(self, moment_arm, wing_area, wing_span, wing_AR, wing_chord):
        """Tail geometry from volume coefficients and wing reference data."""
        # Horizontal tail.
        area_H = self.tail_coefficient_H * wing_chord * wing_area / moment_arm
        AR_H = wing_AR - 0.5                          # convention: lag wing AR slightly
        span_H = (area_H * AR_H) ** 0.5
        chord_H = area_H / (span_H / 2 * (1 + self.taper_ratio))

        # Vertical tail.
        area_V = self.tail_coefficient_V * wing_span * wing_area / moment_arm
        AR_V = AR_H                                   # matched for simplicity
        span_V = (area_V * AR_V) ** 0.5
        chord_V = area_V / (span_V / 2 * (1 + self.taper_ratio))

        return span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V

    def _calculate_mass(self) -> float:
        """Mass = foam panels + small tail spars + fibre tape."""
        mass_H = self.area_H * areal_mass.compressed_foam
        mass_V = self.area_V * areal_mass.compressed_foam
        spar_mass = (self.span_H + self.span_V) * np.pi * (0.0015 ** 2) * density.carbon_spar
        tape_mass = (self.chord_H * self.span_H + self.chord_V * self.span_V) * areal_mass.fibre_tape
        return mass_H + mass_V + spar_mass + tape_mass


# =====================================================================
# Propulsion — single electric motor + propeller
# =====================================================================

@dataclass
class SinglePropellerMotor(Propulsion):
    """Single-motor tractor propulsion with a fixed-pitch prop.

    Effective power stack: ``P_eff = P_rated * eta_motor * eta_prop * eta_ESC``.
    Motor mass is back-calculated from the motor power and a target
    power density (default 3000 W/kg).
    """

    def _propulsion_effectiveness(self) -> float:
        esc_eff = 0.98
        return self.motor_power * self.motor_eff * self.prop_eff * esc_eff

    def _size_propulsion(self) -> float:
        return self.motor_power / self.power_density


# =====================================================================
# Avionics — AIAA-spec servos, ESC, LiPo
# =====================================================================

@dataclass
class AIAAAvionics(Avionics):
    """AIAA DBF avionics sizing.

    * Avionics mass = N_servo * servo_mass + ESC_mass (no harness term).
    * Battery mass  = capacity / energy_density.
    """

    def _mass_avionics(self):
        return self.servo_mass * self.num_servo + self.ESC_mass

    def _mass_battery(self):
        return self.capacity / self.energy_density


# =====================================================================
# Per-mission performance
# =====================================================================

@dataclass
class AIAA2526Performance(Performance):
    """AIAA DBF 2025/26 performance analysis.

    Implements:

    * ``flight_time_one_lap`` — lap time on the DBF course geometry.
    * ``find_CL_cruise`` — solve ``P_eff == P_req`` on CL via a grid
      search for a sign change followed by ``scipy.optimize.brentq``.
    * ``find_turning_conditions`` — closed-form steady-turn analysis
      at a given CL with the effective power as the limiter.
    * ``analyse_performance`` — sequences cruise, stall, turn, lap,
      and flight time into the 16-tuple consumed by the base class.
    """

    def flight_time_one_lap(self, radius, v_straight, v_turn):
        """Return lap time (s) on the DBF 2025/26 course.

        Parameters
        ----------
        radius : float
            Turn radius (m).
        v_straight : float
            Cruise velocity (m/s).
        v_turn : float
            Turn velocity (m/s).

        Notes
        -----
        The course has two straight segments and equivalent turning
        distance of 2 circles, ``2 * (2*pi*r)``. A 1.5x multiplier is applied to
        the *total* lap time to account for real-world imperfect turns
        (the aircraft is rarely at the theoretical minimum radius).
        """
        TURNING_DISTANCE_RATIO = 2 * (2 * np.pi)
        total_time = (
            self.config.route.straight_distance / v_straight
            + TURNING_DISTANCE_RATIO * radius / v_turn
        ) * 1.5
        return total_time

    def find_CL_cruise(self, W, P_effective, CD0, k, S, CL_min, CL_max, CD_extra):
        """Find CL where required power equals effective power.

        Uses a 2000-point grid from ``CL_min`` to ``CL_max`` to
        bracket sign changes of ``P_eff - P_req(CL)``, then calls
        :func:`scipy.optimize.brentq` on the first bracket found.
        If no sign change is present (e.g. underpowered aircraft),
        returns the CL that minimises ``|P_eff - P_req|``.

        Raises
        ------
        ValueError
            If brentq cannot locate a root in the bracketed range.
        """

        def current_power_required(CL):
            return self.power_required(CL, W, CD0, k, S, CD_extra)

        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([current_power_required(CL) for CL in CL_grid])

        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]

        if len(sign_change_indices) == 0:
            # No balanced cruise CL exists in range — best-effort fallback.
            idx_best = np.argmin(np.abs(f_vals))
            return CL_grid[idx_best]

        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]

        try:
            CL_solution = brentq(lambda CL: P_effective - current_power_required(CL), a, b)
            return CL_solution
        except ValueError as error:
            raise ValueError(
                f"CL_cruise search error: {error}. "
                f"Could not find a root between {a:.2f} and {b:.2f}."
            )

    def find_turning_conditions(self, CL, W, CD0, k, P_effective, S, CD_extra):
        """Sustained-turn analysis at a fixed CL, power-limited.

        Solves the steady-turn equations (level turn at constant CL)
        with the effective motor power as the load-factor ceiling.

        Parameters
        ----------
        CL : float
            Lift coefficient to turn at (e.g. ``CL_turn`` or ``CL_max``).
        W : float
            Weight (N).
        CD0 : float
        k : float
            Induced-drag factor.
        P_effective : float
            Effective (prop-thrust) power (W).
        S : float
            Wing area (m^2).
        CD_extra : float
            Additional drag (e.g. banner, M3).

        Returns
        -------
        (n_max, v, angle_rad, radius) : tuple[float, float, float, float]
            Load factor, velocity (m/s), bank angle (rad), and turn
            radius (m).

        Raises
        ------
        ValueError
            If the available power yields ``n_max <= 1``.
        """
        rho = self.config.physics.rho
        g = self.config.physics.g
        CD = CD0 + CD_extra + k * CL ** 2

        # A = power required at n=1; turn power scales as n^1.5.
        A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))

        # Maximum load factor sustainable with the available power.
        n_max = (P_effective / A) ** (2.0 / 3.0)
        v = self.velocity(CL, n_max, W, S)

        if n_max > 1:
            angle = np.arccos(1 / n_max)
        else:
            raise ValueError("N_max must be greater than 1")
        radius = v ** 2 / (g * np.tan(angle))

        return n_max, v, angle, radius

    def analyse_performance(self, m_total, battery_cap, wing_area, wing_AR,
                            depth_of_discharge, P_effective, power, CD0, CD_extra=0):
        """Orchestrate cruise, stall, turn, and lap analysis.

        Parameters
        ----------
        m_total : float
            Total mass for this mission (kg) — structure + payload
            + battery + propulsion.
        battery_cap : float
            Battery capacity (Wh) for this mission.
        wing_area : float
        wing_AR : float
        depth_of_discharge : float
            Usable fraction of battery capacity.
        P_effective : float
            Effective (thrust) power (W).
        power : float
            Rated motor input power (W) — drives flight-time calc.
        CD0 : float
            Zero-lift drag coefficient.
        CD_extra : float, optional
            Additional drag (e.g. banner). Default 0.

        Returns
        -------
        16-tuple
            See base class for order. Flight time caps at 270 s
            (5-minute mission window minus 60 s takeoff/landing).
        """
        rho = self.config.physics.rho  # noqa: F841
        g = self.config.physics.g
        e = self.config.physics.e
        k = 1 / (np.pi * wing_AR * e)
        W = m_total * g

        CL_min = self.config.aero.cl_min
        CL_max = self.config.aero.cl_max
        CL_turn = self.config.aero.cl_turn

        try:
            # Turning performance.
            n_turn, v_turn, bank_angle, turn_radius = self.find_turning_conditions(
                CL_turn, W, CD0, k, P_effective, wing_area, CD_extra,
            )
            n_max, max_v_turn, max_bank_angle, max_turn_radius = self.find_turning_conditions(
                CL_max, W, CD0, k, P_effective, wing_area, CD_extra,
            )
            # Cruise performance.
            CL_cruise = self.find_CL_cruise(W, P_effective, CD0, k, wing_area, CL_min, CL_max, CD_extra)
            V_cruise = self.velocity(CL_cruise, 1, W, wing_area)
            # Stall speed.
            V_stall = self.velocity(CL_max, 1, W, wing_area)

            # Derived metrics.
            CD_cruise = CD0 + CD_extra + k * (CL_cruise ** 2)
            L_D_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else 0
            stall_margin = V_cruise / V_stall if V_stall > 0 else 0
            load_factor_margin = n_max / n_turn if n_turn > 0 else 0

            # Flight time / lap count.
            flight_time = battery_cap * depth_of_discharge * 3600 / power
            effective_flying = flight_time - 60          # exclude takeoff/landing
            max_time = min(effective_flying, 270)        # within the 5-minute window
            one_lap = self.flight_time_one_lap(turn_radius, V_cruise, v_turn)
            number_of_laps = max_time // one_lap

            return (CL_cruise, V_cruise, V_stall, n_turn, v_turn, turn_radius,
                    bank_angle, n_max, max_v_turn, max_bank_angle, max_turn_radius,
                    flight_time, number_of_laps, L_D_cruise, stall_margin, load_factor_margin)

        except Exception as error:
            raise error


# =====================================================================
# Plane — wiring design variables to components
# =====================================================================

@dataclass(kw_only=True)
class RCPlane(Plane):
    """The full AIAA DBF 2025/26 aircraft, parameterised by 11 design vars.

    Instantiating an ``RCPlane`` runs the full sizing pipeline:

    1. Resolve payload masses for each mission (pucks, ducks, banner).
    2. Instantiate avionics and propulsion.
    3. Build the wing, size it aerodynamically.
    4. Compute per-mission performance (M1, M2, M3) — M3 includes
       banner drag.
    5. Size the main wing spar for the worst-case cruise speed.
    6. Size fuselage and tail from the wing geometry.
    7. Call :meth:`check_mass_coherence` to verify the assumed
       structural mass matches the sum of component masses within
       1%.

    After construction, inspect ``plane.is_converged`` and
    ``plane.relative_error`` to check feasibility.

    Parameters
    ----------
    m_struct : float
        Assumed structural mass (kg). Convergence target.
    wing_span : float
        Wing span (m).
    wing_AR : float
        Wing aspect ratio.
    motor_power : float
        Motor rated power (W).
    m1_battery, m2_battery, m3_battery : float
        Battery capacities per mission (Wh).
    n_pucks : int
        Number of hockey pucks (M2).
    passenger_cargo_ratio : int
        Ducks per puck (M2).
    banner_length : float
        Banner length (inches, M3).
    banner_AR : float
        Banner aspect ratio (M3).
    """
    m_struct: float
    wing_span: InitVar[float] = None
    wing_AR: InitVar[int] = None
    motor_power: InitVar[int] = None
    m1_battery: InitVar[int] = None
    n_pucks: InitVar[int] = None
    passenger_cargo_ratio: InitVar[int] = None
    m2_battery: InitVar[int] = None
    banner_length: InitVar[int] = None
    banner_AR: InitVar[int] = None
    m3_battery: InitVar[int] = None

    relative_error: float = field(init=False)
    is_converged: bool = field(init=False)

    def __post_init__(self, wing_span, wing_AR, motor_power, m1_battery, n_pucks,
                      passenger_cargo_ratio, m2_battery, banner_length, banner_AR, m3_battery):
        # --- Mission 2 payload: hockey pucks + "ducks" (passenger proxy) ---
        m_duck = 0.0184   # kg, per duck
        m_puck = 0.170    # kg, per puck
        n_ducks = n_pucks * passenger_cargo_ratio
        CD0 = 0.13 + 0.01 * np.sqrt(n_ducks)    # empirical duck-count drag bump
        m2_payload = n_pucks * m_puck + n_ducks * m_duck

        # --- Mission 3 payload: banner ---
        banner_length_m = banner_length * 0.0254
        banner_width = banner_length / banner_AR
        banner_width_m = banner_width * 0.0254
        area_banner = banner_length_m * banner_width_m
        m3_payload = 1.2 * area_banner * areal_mass.lightweight_ripstop  # 1.2x safety factor

        # --- Avionics & Propulsion ---
        m1_avionics = AIAAAvionics(m1_battery)
        m2_avionics = AIAAAvionics(m2_battery)
        m3_avionics = AIAAAvionics(m3_battery)
        self.propulsion = SinglePropellerMotor(motor_power)

        # --- Wing geometry (uses Clark Y by default) ---
        self.wing = XPSRectangularWing(airfoil_type="clarkY.dat", aspect_ratio=wing_AR, span=wing_span)

        # --- Per-mission total mass = structure + prop + payload + battery ---
        m1_payload = 0
        m_M1 = self.m_struct + self.propulsion.mass + m1_payload + m1_avionics.mass_battery
        m_M2 = self.m_struct + self.propulsion.mass + m2_payload + m2_avionics.mass_battery
        m_M3 = self.m_struct + self.propulsion.mass + m3_payload + m3_avionics.mass_battery

        self.m_max = max(m_M1, m_M2, m_M3)

        # --- Mission 1: unloaded laps ---
        m1_performance = AIAA2526Performance(
            m_M1, m1_battery, self.wing.surface_area, wing_AR,
            m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0,
        )
        # --- Mission 2: pucks + ducks ---
        m2_performance = AIAA2526Performance(
            m_M2, m2_battery, self.wing.surface_area, wing_AR,
            m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0,
        )
        # --- Mission 3: banner drag ---
        CD_banner = 0.5 * np.power(banner_AR, -0.5)
        m3_performance = AIAA2526Performance(
            m_M3, m3_battery, self.wing.surface_area, wing_AR,
            m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0, CD_banner,
        )

        self.missions["1"] = FlightMission(m1_avionics, m1_performance, m1_payload)
        self.missions["2"] = FlightMission(m2_avionics, m2_performance, m2_payload)
        self.missions["3"] = FlightMission(m3_avionics, m3_performance, m3_payload)

        # --- Spar sizing against worst-case cruise speed ---
        max_v_cruise = max(m1_performance.V_cruise, m2_performance.V_cruise, m3_performance.V_cruise)
        self.wing.size_spar(wing_span, max_v_cruise)

        # --- Fuselage & tail ---
        self.fuselage = SemiMono(self.wing.span, self.wing.chord)
        self.tail = CompressedFoamRectangleInvertedTTail(
            0.5, 0.04, 1,                            # V_H, V_V, taper
            self.fuselage.wing_ac_to_tail_ac,        # moment arm
            self.wing.surface_area, wing_span, wing_AR, self.wing.chord,
        )

        # --- Mass coherence check ---
        self.check_mass_coherence()

    def check_mass_coherence(self):
        """Sanity-check the assumed vs computed structural mass.

        Sums the masses of the wing (foam + skin), spar, fuselage,
        tail, and avionics (M1 pack as a proxy); applies a 6% margin
        for items not modelled (wiring harness, glue, fasteners,
        paint); and compares to the design-variable ``m_struct``.

        Populates ``self.relative_error`` and ``self.is_converged``
        (True when ``relative_error <= 1%``).
        """
        summed_mass = (
            self.missions["1"].avionics.mass_avionics
            + self.tail.mass
            + self.wing.mass
            + self.wing.spar.mass
            + self.fuselage.mass
        ) * 1.06

        relative_error = abs(summed_mass - self.m_struct) / self.m_struct
        self.relative_error = relative_error
        self.is_converged = relative_error <= 0.01

    def converged(self):
        """Return True iff the mass-coherence check passed."""
        return self.is_converged

    @property
    def wing_loading(self) -> float:
        """Wing loading W/S (N/m^2) at maximum takeoff weight."""
        g = self.wing.config.physics.g
        weight_max = self.m_max * g
        return weight_max / self.wing.surface_area

    @property
    def power_to_weight(self) -> float:
        """Power-to-weight ratio P/W (W/N) at maximum takeoff weight."""
        g = self.wing.config.physics.g
        weight_max = self.m_max * g
        return self.propulsion.motor_power / weight_max
