from dataclasses import dataclass, field
import numpy as np
from math import ceil
from scipy.optimize import brentq

from fixed_params.general import *
from fixed_params import *
import fixed_params.wing as fp_wing
from material_properties import *

"""
"""


# ========== WING ==========
@dataclass(frozen=True)
class WingGeometry:
    span: float
    area: float
    chord: float
    AR: float


@dataclass(frozen=True)
class WingStructure(WingGeometry):
    mass: float
    spar_radius: float


def design_wing_geometry(wing_AR: float, wing_span: float) -> WingGeometry:
    """Dimension wing geometry."""
    wing_c = wing_span / wing_AR
    wing_S = wing_span * wing_c

    return WingGeometry(span=wing_span, area=wing_S, chord=wing_c, AR=wing_AR)


def estimate_wing_structure(wing: WingGeometry, v_cruise: float, CL_max: float) -> WingStructure:
    """Add structural mass estimation to a WingGeometry object."""
    force = 0.5 * rho * CL_max * wing.area * v_cruise ** 2
    spar_radius = spar_dimension(wing.span, force, safety_factor=2)

    wing_volume = fp_wing.airfoil_constant[fp_wing.airfoil_type][0] * wing.chord ** 2 * wing.span
    wrap_area = fp_wing.airfoil_constant[fp_wing.airfoil_type][1] * wing.chord * 4 * wing.span

    mass = 0
    mass += wing_volume * density.blue_xps_foam
    mass += wrap_area * areal_mass.shrink_wrap
    mass += np.pi * wing.span * (spar_radius ** 2 - (spar_radius - 0.002) ** 2) * density.carbon_spar

    return WingStructure(
        span=wing.span,
        area=wing.area,
        chord=wing.chord,
        AR=wing.AR,
        mass=mass,
        spar_radius=spar_radius
    )


# ========== FUSELAGE ==========
@dataclass(frozen=True)
class Fuselage:
    length: float
    width: float
    height: float
    wing_ac_from_nose: float
    tail_ac_from_nose: float
    wing_ac_to_tail_ac: float
    mass: float


def size_fuselage(wing_span: float, wing_c: float) -> Fuselage:
    length = wing_span * fuselage.length_constant
    width = length * fuselage.width_constant
    height = length * fuselage.height_constant

    front_area = width * height
    bottom_area = length * width
    left_area = length * height
    fuselage_skin_area = 2 * front_area + 2 * bottom_area + 2 * left_area
    fuselage_skin_volume = fuselage_skin_area * 0.005
    fuselage_skin_mass = fuselage_skin_volume * density.white_foam

    structural_volume = length * width * height * fuselage.int_struc_percentage
    structural_mass = structural_volume * density.basswood

    total_mass = fuselage_skin_mass + structural_mass

    wing_ac_from_nose = length * fuselage.nose_ratio + wing_c * fuselage.AC_chord_ratio
    tail_ac_from_nose = length * fuselage.wing_ac_to_tail_ac
    wing_ac_to_tail_ac = tail_ac_from_nose - wing_ac_from_nose

    return Fuselage(length, width, height, wing_ac_from_nose, tail_ac_from_nose, wing_ac_to_tail_ac, total_mass)


# ========== TAIL ==========
@dataclass(frozen=True)
class Tail:
    span_H: float
    chord_H: float
    area_H: float
    AR_H: float
    span_V: float
    chord_V: float
    area_V: float
    AR_V: float
    mass: float


def design_tail(wing: WingGeometry, fus: Fuselage) -> Tail:
    moment_arm = fus.wing_ac_to_tail_ac

    area_H = tail.tail_coefficient_H * wing.chord * wing.area / moment_arm
    AR_H = wing.AR / 2
    span_H = (area_H * AR_H) ** 0.5
    chord_H = area_H / (span_H / 2 * (1 + tail.taper_ratio))

    area_V = tail.tail_coefficient_V * wing.span * wing.area / moment_arm
    AR_V = AR_H
    span_V = (area_V * AR_V) ** 0.5
    chord_V = area_V / (span_V / 2 * (1 + tail.taper_ratio))

    # Simple flat foam tails
    mass_H = area_H * areal_mass.compressed_foam
    mass_V = area_V * areal_mass.compressed_foam
    spar_mass = (span_H + span_V) * np.pi * (0.0015 ** 2) * density.carbon_spar
    tape_mass = (chord_H * span_H + chord_V * span_V) * areal_mass.fibre_tape
    mass = mass_H + mass_V + spar_mass + tape_mass

    return Tail(span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V, mass)


# ========== PROPULSION ==========
@dataclass(frozen=True)
class Propulsion:
    motor_power: float
    effective_power: float
    mass: float


def design_propulsion(motor_power: float) -> Propulsion:
    eff = propulsion.motor_eff * propulsion.prop_eff * avionics.esc_eff
    m_prop = motor_power / propulsion.power_density
    return Propulsion(motor_power, motor_power * eff, m_prop)


# ========== PERFORMANCE ==========
@dataclass(frozen=True)
class Performance:
    CL_cruise: float
    V_cruise: float
    V_stall: float
    n_turn: float
    v_turning: float
    turn_radius: float
    bank_angle: float
    n_max: float
    max_v_turn: float
    max_bank_angle: float
    max_turn_radius: float


@dataclass(frozen=True)
class M2(Performance):
    flight_time: float
    num_laps: int
    num_ducks: int
    num_pucks: int


@dataclass(frozen=True)
class M3(Performance):
    flight_time: float
    num_laps: int
    banner_length: float
    banner_width: float


# ------------------ Performance Functions -----------------------------
def velocity(CL, n, W, S):
    """
    Calculates flight velocity.

    Args:
        CL (float): Lift coefficient.
        n (float): Load factor.
        W (float): Weight (Newtons).
        S (float): Wing area (m^2).

    Returns:
        float: Velocity (m/s).
    """
    return np.sqrt((2 * n * W) / (rho * S * CL))


def power_required(CL, W, CD0, k, S, CD_banner=0):
    """
    Calculates required power for straight and level flight (n=1).

    Args:
        CL (float): Lift coefficient.
        W (float): Weight (Newtons).
        CD0 (float): Zero-lift drag coefficient.
        k (float): Induced drag constant.
        S (float): Wing area (m^2).
        CD_banner (float): Additional banner drag coefficient (default 0).

    Returns:
        float: Required Power (Watts).
    """
    CD = CD0 + CD_banner + k * CL ** 2
    v = velocity(CL, 1, W, S)
    return (CD * W / CL) * v


def find_CL_cruise(W, P_effective, CD0, k, S, CL_min, CL_max, CD_banner=0):
    """
    Finds the lift coefficient for cruise flight where P_effective = P_required.

    Args:
        W (float): Weight (Newtons).
        P_effective (float): Effective motor power (Watts).
        CD0 (float): Zero-lift drag coefficient.
        k (float): Induced drag constant.
        S (float): Wing area (m^2).
        CL_min (float): Minimum searchable lift coefficient.
        CL_max (float): Maximum searchable lift coefficient.
        CD_banner (float): Additional banner drag coefficient (default 0).

    Returns:
        float: Lift coefficient (CL) solution.

    Raises:
        ValueError: If brentq cannot find a root in the specified range.
    """

    # Define the specific power function needed for the search, closing over parameters
    def current_power_required(CL):
        return power_required(CL, W, CD0, k, S, CD_banner)

    CL_grid = np.linspace(CL_min, CL_max, 2000)
    f_vals = P_effective - np.array([current_power_required(CL) for CL in CL_grid])

    # Find bracket where sign changes
    sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]

    if len(sign_change_indices) == 0:
        # If no sign change, return the CL that minimizes the absolute difference
        idx_best = np.argmin(np.abs(f_vals))
        # Note: If min abs difference is still far from zero, the solution is approximate
        return CL_grid[idx_best]

    # Use first bracket found for brentq
    a = CL_grid[sign_change_indices[0]]
    b = CL_grid[sign_change_indices[0] + 1]

    try:
        # Pass a lambda that calls the closed-over function
        CL_solution = brentq(lambda CL: P_effective - current_power_required(CL), a, b)
        return CL_solution
    except ValueError as error:
        raise ValueError(f"CL_cruise search error: {error}. Could not find a root between {a:.2f} and {b:.2f}.")


def find_turning_conditions(CL, W, CD0, k, P_effective, S, CD_extra=0):
    """
    Calculates turning performance (n, V, angle, radius) for a given CL and W.

    Args:
        CL (float): Lift coefficient used for the turn.
        W (float): Weight (Newtons).
        CD0 (float): Zero-lift drag coefficient.
        k (float): Induced drag constant.
        P_effective (float): Effective motor power (Watts).
        S (float): Wing area (m^2).

    Returns:
        tuple: (n_max, v, angle_rad, radius)
    """
    CD = CD0 + CD_extra + k * CL ** 2

    # Calculate the power required for straight flight at this CL (A)
    A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))

    # Calculate the maximum load factor (n_max) possible with P_effective
    n_max = (P_effective / A) ** (2.0 / 3.0)

    # Calculate velocity at n_max
    v = velocity(CL, n_max, W, S)

    # Calculate bank angle (rad) and turn radius
    if n_max > 1:
        angle = np.arccos(1 / n_max)
    else:
        raise ValueError("3: N_max must be greater than 1")
    radius = v ** 2 / (g * np.tan(angle))

    return n_max, v, angle, radius


def flight_time_one_lap(radius, v_straight, v_turn):
    """
    Calculates time for one lap of the track.

    Args:
        radius (float): Turn radius (m).
        v_straight (float): Cruise velocity (m/s).
        v_turn (float): Turning velocity (m/s).
        straight_distance (float): Length of the straight segment (m).

    Returns:
        float: Time for one lap (seconds).
    """
    # Assuming turning segment is 4*pi*r total distance (2 turns of 2*pi*r each)
    TURNING_DISTANCE_RATIO = 2 * (2 * np.pi)

    # The original logic included a multiplier of 1.5
    total_time = (flight_track.straight_distance / v_straight + TURNING_DISTANCE_RATIO * radius / v_turn) * 1.5
    return total_time


def analyse_performance_m2(m_total: float, battery_cap: float, wing: WingGeometry, propulsion: Propulsion, CD0: float,
                           ducks: int, n_pucks: int) -> M2:
    k = 1 / (np.pi * wing.AR * e)
    W = m_total * g
    P_effective = propulsion.effective_power
    power = propulsion.motor_power

    try:
        # Turning performance
        n_turn, v_turn, bank_angle, turn_radius = find_turning_conditions(CL_turn, W, CD0, k,
                                                                          P_effective, wing.area)
        n_max, max_v_turn, max_bank_angle, max_turn_radius = find_turning_conditions(CL_max, W, CD0, k,
                                                                                     P_effective, wing.area)
        # Cruise performance
        CL_cruise = find_CL_cruise(W, P_effective, CD0, k, wing.area, CL_min, CL_max)
        V_cruise = velocity(CL_cruise, 1, W, wing.area)

        # stall
        V_stall = velocity(CL_max, 1, W, wing.area)

        # flight time
        flight_time = battery_cap * avionics.depth_of_discharge * 3600 / power
        effective_flying = min(flight_time - 60, 270)
        one_lap = flight_time_one_lap(turn_radius, V_cruise, v_turn)
        number_of_laps = effective_flying // one_lap

        if number_of_laps < 1:
            raise ValueError("Number of Laps cannot be less than 1")
        return M2(CL_cruise, V_cruise, V_stall, n_turn, v_turn, turn_radius, bank_angle, n_max, max_v_turn,
                  max_bank_angle, max_turn_radius, flight_time, number_of_laps, ducks, n_pucks)

    except Exception as error:
        raise error


def analyse_performance_m3(m_total: float, battery_cap: float, wing: WingGeometry, propulsion: Propulsion, CD0: float, banner_length: float, banner_AR: float) -> M3:
    k = 1 / (np.pi * wing.AR * e)
    W = m_total * g
    P_effective = propulsion.effective_power
    power = propulsion.motor_power

    # Power required function for M3 (with banner drag)
    CD_banner = 0.5 * np.power(banner_AR, -0.5)

    try:
        # Turning performance
        n_turn, v_turn, bank_angle, turn_radius = find_turning_conditions(CL_turn, W, CD0, k,
                                                                          P_effective, wing.area, CD_extra=CD_banner)
        n_max, max_v_turn, max_bank_angle, max_turn_radius = find_turning_conditions(CL_max, W, CD0, k,
                                                                                     P_effective, wing.area, CD_extra=CD_banner)
        # Cruise performance
        CL_cruise = find_CL_cruise(W, P_effective, CD0, k, wing.area, CL_min, CL_max, CD_banner=CD_banner)
        V_cruise = velocity(CL_cruise, 1, W, wing.area)

        # stall
        V_stall = velocity(CL_max, 1, W, wing.area)

        # flight time
        flight_time = battery_cap * avionics.depth_of_discharge * 3600 / power
        effective_flying = min(flight_time - 60, 270)
        one_lap = flight_time_one_lap(turn_radius, V_cruise, v_turn)
        number_of_laps = effective_flying // one_lap

        if number_of_laps < 1:
            raise ValueError("Number of Laps cannot be less than 1")

        return M3(CL_cruise, V_cruise, V_stall, n_turn, v_turn, turn_radius, bank_angle, n_max, max_v_turn,
                  max_bank_angle, max_turn_radius, flight_time, number_of_laps, banner_length, banner_length / banner_AR)

    except ValueError as error:
        print(f"Error: {error}")
        raise error


# ========== AVIONICS ==========
@dataclass(frozen=True)
class Avionics:
    m2_capacity: int
    m3_capacity: int
    m2_mass_battery: float
    m3_mass_battery: float
    mass_avionics: float


def design_avionics(m2_battery: int, m3_battery: int) -> Avionics:
    m2_capacity = m2_battery
    m2_mass_battery = m2_capacity / avionics.energy_density

    m3_capacity = m3_battery
    m3_mass_battery = m3_capacity / avionics.energy_density

    mass_avionics = (avionics.servo_mass * avionics.num_servo +
                     avionics.ESC_mass)
    return Avionics(m2_capacity, m3_capacity, m2_mass_battery, m3_mass_battery, mass_avionics)


# ========== RCPlane Orchestrator ==========
@dataclass
class RCPlane:
    m_struct: float
    wing_span: float
    motor_power: int
    wing_AR: int
    n_pucks: int
    passenger_cargo_ratio: int
    m2_battery: int
    banner_length: int
    banner_AR: int
    m3_battery: int

    m_max: float = field(init=False)
    m2_payload: float = field(init=False)
    m3_payload: float = field(init=False)
    wing: WingStructure | None = None
    fuselage: Fuselage | None = None
    tail: Tail | None = None
    propulsion: Propulsion | None = None
    m2: M2 | None = None
    m3: M3 | None = None
    avionics: Avionics | None = None

    def __post_init__(self):
        m_duck = general.duck
        m_puck = general.puck
        ducks = self.n_pucks * self.passenger_cargo_ratio
        self.m2_payload = self.n_pucks * m_puck + ducks * m_duck

        self.CD0 = 0.13 + 0.01 * np.sqrt(ducks)

        # inches to m
        banner_length_m = self.banner_length * 0.0254
        banner_width = self.banner_length / self.banner_AR
        banner_width_m = banner_width * 0.0254
        area_banner = banner_length_m * banner_width_m
        self.m3_payload = 1.2 * area_banner * areal_mass.lightweight_ripstop

        # Avionics & Propulsion
        self.avionics = design_avionics(self.m2_battery, self.m3_battery)
        self.propulsion = design_propulsion(self.motor_power)

        # Wing Geom
        wing = design_wing_geometry(self.wing_AR, self.wing_span)

        self.m_max = self.m_struct + self.propulsion.mass + max(self.m2_payload + self.avionics.m2_mass_battery,
                                                                self.m3_payload + self.avionics.m3_mass_battery)

        # Mission 2 Performance
        m_M2 = self.m_struct + self.propulsion.mass + self.m2_payload + self.avionics.m2_mass_battery
        self.m2 = analyse_performance_m2(m_M2, self.avionics.m2_capacity, wing, self.propulsion,
                                         self.CD0, ducks, self.n_pucks)

        # Mission 3 Performance
        m_M3 = self.m_struct + self.propulsion.mass + self.m3_payload + self.avionics.m3_mass_battery
        self.m3 = analyse_performance_m3(m_M3, self.avionics.m3_capacity, wing, self.propulsion,
                                         self.CD0, self.banner_length, self.banner_AR)

        # Wing Structure
        max_v_cruise = max(self.m2.V_cruise, self.m3.V_cruise)
        self.wing = estimate_wing_structure(wing, max_v_cruise, CL_max)

        # Size Fuselage, Tail, Landing Gear
        self.fuselage = size_fuselage(self.wing.span, self.wing.chord)
        self.tail = design_tail(self.wing, self.fuselage)

        if not self.is_mass_coherent():
            raise ValueError("Sized Mass deviates from Input Mass by too much")


    def is_mass_coherent(self):
        # multiplied by 1.06 as 6% of mass to landing gear
        summed_mass = (self.avionics.mass_avionics + self.tail.mass + self.wing.mass + self.fuselage.mass) * 1.06

        # # allow for 2.5% deviation. Due to assumed overestimates
        # deviation = abs(summed_mass - self.m_struct)/summed_mass
        return summed_mass <= self.m_struct * 1.04


def spar_dimension(wing_span, force, safety_factor, wall_thickness=0.002):
    uts_with_safety_factor = UTS.carbon_spar / safety_factor

    '''
    Returns the outer radius of spar.

    Inputs:
    wing_span: wing full span
    force: force on full wing
    safety_factor: safety factor 
    wall_thickness: thickness of the spar
    '''
    # force is total lift for BOTH wings, each panel takes half
    panel_length = wing_span / 2
    panel_force = force / 2

    # Max bending moment at root of one panel occurs at the centre for uniform distribution(cantilever assumption)
    M = panel_force * panel_length / 2

    rad_o = 0.0025  # starting guess outer radius (m)
    while True:
        r_i = rad_o - wall_thickness
        I = np.pi * (rad_o ** 4 - r_i ** 4) / 4
        sigma = M * rad_o / I
        if sigma <= uts_with_safety_factor:
            break
        rad_o += 0.0005  # increment 0.5 mm
    return rad_o


# # m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio, m2_battery, banner_length, banner_AR, m3_battery
# plane = RCPlane(1.33, 1.15, 350, 7, 5, 6, 50, 250, 5, 50)
#
# print(plane)