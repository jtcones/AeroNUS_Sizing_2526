from Components import Wing, RectangularWing, Fuselage, Tail, Performance, Propulsion, FlightMission, LandingGear, Avionics, Plane
from dataclasses import dataclass, field, InitVar
import numpy as np
from material_properties import *
import subprocess
import os
from scipy.optimize import brentq

@dataclass
class XPSRectangularWing(RectangularWing):
    def _calculate_mass(self) -> float:
        """Add structural mass estimation. Excluding spars required"""
        wing_volume = self.airfoil_area_ratio * self.chord ** 2 * self.span
        wrap_area = self.airfoil_circum_ratio * self.chord * 4 * self.span

        mass = wing_volume * density.blue_xps_foam
        mass += wrap_area * areal_mass.shrink_wrap
        return mass

@dataclass
class SemiMono(Fuselage):
    wing_span: InitVar[float] = None
    wing_c: InitVar[float] = None
    def __post_init__(self, wing_span, wing_c):
        self.mass = self._size_fuselage(wing_span, wing_c)

    def _size_fuselage(self, wing_span, wing_c) -> float:
        #fuselage constants
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

        front_area = width * height
        bottom_area = length * width
        left_area = length * height
        fuselage_skin_area = 2 * front_area + 2 * bottom_area + 2 * left_area
        fuselage_skin_volume = fuselage_skin_area * 0.005
        fuselage_skin_mass = fuselage_skin_volume * density.white_foam

        structural_volume = length * width * height * int_struc_percentage
        structural_mass = structural_volume * density.basswood

        total_mass = fuselage_skin_mass + structural_mass

        wing_ac_from_nose = length * nose_ratio + wing_c * AC_chord_ratio
        tail_ac_from_nose = length * wing_ac_to_tail_ac
        wing_ac_to_tail_ac = tail_ac_from_nose - wing_ac_from_nose

        self.wing_ac_from_nose = wing_ac_from_nose
        self.tail_ac_from_nose = tail_ac_from_nose
        self.wing_ac_to_tail_ac = wing_ac_to_tail_ac
        return total_mass

@dataclass
class CompressedFoamRectangleInvertedTTail(Tail):
    tail_coefficient_H : float
    tail_coefficient_V : float
    taper_ratio: float

    moment_arm: InitVar[float] = None
    wing_area: InitVar[float] = None
    wing_span: InitVar[float] = None
    wing_AR: InitVar[float] = None
    wing_chord: InitVar[float] = None

    def __post_init__(self, moment_arm, wing_area, wing_span, wing_AR, wing_chord):
        self.span_H, self.chord_H, self.area_H, self.AR_H, self.span_V, self.chord_V, self.area_V, self.AR_V = self._calculate_geometry(moment_arm, wing_area, wing_span, wing_AR, wing_chord)

        self.mass = self._calculate_mass()

    def _calculate_geometry(self, moment_arm, wing_area, wing_span, wing_AR, wing_chord) -> tuple[float, float, float, float, float, float, float, float]:
        get_v_span = lambda a, AR, S: np.sqrt(S * AR * np.cos(a))
        area_H = self.tail_coefficient_H * wing_chord * wing_area / moment_arm
        AR_H = wing_AR - 0.5
        span_H = (area_H * AR_H) ** 0.5
        chord_H = area_H / (span_H / 2 * (1 + self.taper_ratio))

        area_V = self.tail_coefficient_V * wing_span * wing_area / moment_arm
        AR_V = AR_H
        span_V = (area_V * AR_V) ** 0.5
        chord_V = area_V / (span_V / 2 * (1 + self.taper_ratio))

        return span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V

    def _calculate_mass(self) -> float:
        # Simple flat foam tails
        mass_H = self.area_H * areal_mass.compressed_foam
        mass_V = self.area_V * areal_mass.compressed_foam
        spar_mass = (self.span_H + self.span_V) * np.pi * (0.0015 ** 2) * density.carbon_spar
        tape_mass = (self.chord_H * self.span_H + self.chord_V * self.span_V) * areal_mass.fibre_tape
        mass = mass_H + mass_V + spar_mass + tape_mass
        return mass

@dataclass
class SinglePropellerMotor(Propulsion):
    def _propulsion_effectiveness(self) -> float:
        esc_eff = 0.98
        return self.motor_power * self.motor_eff * self.prop_eff * esc_eff

    def _size_propulsion(self) -> float:
        return self.motor_power / self.power_density

@dataclass
class AIAAAvionics(Avionics):

    def _mass_avionics(self):
        return self.servo_mass * self.num_servo + self.ESC_mass

    def _mass_battery(self):
        return self.capacity / self.energy_density

@dataclass
class AIAA2526Performance(Performance):
    def flight_time_one_lap(self, radius, v_straight, v_turn):
        """
        Calculates time for one lap of the track.

        Args:
            radius (float): Turn radius (m).
            v_straight (float): Cruise velocity (m/s).
            v_turn (float): Turning velocity (m/s).

        Returns:
            float: Time for one lap (seconds).
        """
        # Assuming turning segment is 4*pi*r total distance (2 turns of 2*pi*r each)
        TURNING_DISTANCE_RATIO = 2 * (2 * np.pi)

        # The original logic included a multiplier of 1.5
        total_time = (self.config.route.straight_distance / v_straight + TURNING_DISTANCE_RATIO * radius / v_turn) * 1.5
        # multiple 1.5 to account for real life inperfect turning ( not always at the best radius)
        return total_time

    def find_CL_cruise(self, W, P_effective, CD0, k, S, CL_min, CL_max, CD_extra):
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
            CD_extra (float): Additional banner drag coefficient (default 0).

        Returns:
            float: Lift coefficient (CL) solution.

        Raises:
            ValueError: If brentq cannot find a root in the specified range.
        """

        # Define the specific power function needed for the search, closing over parameters
        def current_power_required(CL):
            return self.power_required(CL, W, CD0, k, S, CD_extra)

        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([current_power_required(CL) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]

        if len(sign_change_indices) == 0:
            # print("no sign changes")
            # If no sign change, return the CL that minimizes the absolute difference
            idx_best = np.argmin(np.abs(f_vals))
            # Note: If min abs difference is still far from zero, the solution is approximate
            return CL_grid[idx_best]

        # Use first bracket found for brentq
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]

        try:
            # Pass a lambda that calls the closed-over function
            # print("solution is between:", a, " and ", b)
            CL_solution = brentq(lambda CL: P_effective - current_power_required(CL), a, b)
            # print("CL is ", CL_solution)
            return CL_solution
        except ValueError as error:
            raise ValueError(f"CL_cruise search error: {error}. Could not find a root between {a:.2f} and {b:.2f}.")

    def find_turning_conditions(self, CL, W, CD0, k, P_effective, S, CD_extra):
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
        rho = self.config.physics.rho
        g = self.config.physics.g
        CD = CD0 + CD_extra + k * CL ** 2

        # Calculate the power required for straight flight at this CL (A)
        A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))

        # Calculate the maximum load factor (n_max) possible with P_effective
        n_max = (P_effective / A) ** (2.0 / 3.0)

        # Calculate velocity at n_max
        v = self.velocity(CL, n_max, W, S)

        # Calculate bank angle (rad) and turn radius
        if n_max > 1:
            angle = np.arccos(1 / n_max)
        else:
            raise ValueError("N_max must be greater than 1")
        radius = v ** 2 / (g * np.tan(angle))

        return n_max, v, angle, radius

    def analyse_performance(self, m_total, battery_cap, wing_area, wing_AR, depth_of_discharge, P_effective, power,
                               CD0, CD_extra=0):
        rho = self.config.physics.rho
        g = self.config.physics.g
        e = self.config.physics.e
        k = 1 / (np.pi * wing_AR * e)
        W = m_total * g

        CL_min = self.config.aero.cl_min
        CL_max = self.config.aero.cl_max
        CL_turn = self.config.aero.cl_turn

        try:
            # Turning performance
            n_turn, v_turn, bank_angle, turn_radius = self.find_turning_conditions(CL_turn, W, CD0, k,
                                                                              P_effective, wing_area, CD_extra)
            n_max, max_v_turn, max_bank_angle, max_turn_radius = self.find_turning_conditions(CL_max, W, CD0, k,
                                                                                         P_effective, wing_area, CD_extra)
            # Cruise performance
            CL_cruise = self.find_CL_cruise(W, P_effective, CD0, k, wing_area, CL_min, CL_max, CD_extra)
            # print("In Performance Calculations, CL_cruise is: ", CL_cruise)
            V_cruise = self.velocity(CL_cruise, 1, W, wing_area)

            # stall
            V_stall = self.velocity(CL_max, 1, W, wing_area)

            # 1. L/D Ratio at Cruise
            CD_cruise = CD0 + CD_extra + k * (CL_cruise ** 2)
            L_D_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else 0

            # 2. Stall Margin (How much faster are we flying than stall speed)
            stall_margin = V_cruise / V_stall if V_stall > 0 else 0

            # 3. Load Factor Margin (How much extra G-pulling capability do we have in a turn)
            load_factor_margin = n_max / n_turn if n_turn > 0 else 0

            # flight time
            flight_time = battery_cap * depth_of_discharge * 3600 / power
            effective_flying = flight_time - 60 #excluding take off and landing
            one_lap = self.flight_time_one_lap(turn_radius, V_cruise, v_turn)
            number_of_laps = effective_flying // one_lap
            if number_of_laps < 1:
                raise ValueError("Number of Laps Flown is less than 1")
            return CL_cruise, V_cruise, V_stall, n_turn, v_turn, turn_radius, bank_angle, n_max, max_v_turn, max_bank_angle, max_turn_radius, flight_time, number_of_laps, L_D_cruise, stall_margin, load_factor_margin

        except Exception as error:
            raise error

@dataclass(kw_only=True)
class RCPlane(Plane):
    #inputs
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

    def __post_init__(self, wing_span, wing_AR, motor_power, m1_battery, n_pucks, passenger_cargo_ratio, m2_battery, banner_length, banner_AR, m3_battery):
        m_duck = 0.0184 #kg
        m_puck = 0.170 #kg
        n_ducks = n_pucks * passenger_cargo_ratio
        CD0 = 0.13 + 0.01 * np.sqrt(n_ducks)
        m2_payload = n_pucks * m_puck + n_ducks * m_duck

        # inches to m
        banner_length_m = banner_length * 0.0254
        banner_width = banner_length / banner_AR
        banner_width_m = banner_width * 0.0254
        area_banner = banner_length_m * banner_width_m
        m3_payload = 1.2 * area_banner * areal_mass.lightweight_ripstop

        # Avionics & Propulsion
        m1_avionics = AIAAAvionics(m1_battery)
        m2_avionics = AIAAAvionics(m2_battery)
        m3_avionics = AIAAAvionics(m3_battery)
        self.propulsion = SinglePropellerMotor(motor_power)

        self.wing = XPSRectangularWing(airfoil_type="clarkY.dat", aspect_ratio=wing_AR, span=wing_span)
        m1_payload = 0

        m_M1 = self.m_struct + self.propulsion.mass + m1_payload + m1_avionics.mass_battery
        m_M2 = self.m_struct + self.propulsion.mass + m2_payload + m2_avionics.mass_battery
        m_M3 = self.m_struct + self.propulsion.mass + m3_payload + m3_avionics.mass_battery

        self.m_max = max(m_M1, m_M2, m_M3)

        # print("M1 Performance")
        # Mission 1 Performance
        m1_performance = AIAA2526Performance(m_M1, m1_battery, self.wing.surface_area, wing_AR, m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0)
        # print("M2 Performance")
        # Mission 2 Performance
        m2_performance = AIAA2526Performance(m_M2, m2_battery, self.wing.surface_area, wing_AR, m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0)
        # print("M3 Performance")
        # Mission 3 Performance
        # Power required function for M3 (with banner drag)
        CD_banner = 0.5 * np.power(banner_AR, -0.5)
        m3_performance = AIAA2526Performance(m_M3, m3_battery, self.wing.surface_area, wing_AR, m1_avionics.depth_of_discharge, self.propulsion.effective_power, motor_power, CD0, CD_banner)

        self.missions["1"] = FlightMission(m1_avionics, m1_performance, m1_payload)
        self.missions["2"] = FlightMission(m2_avionics, m2_performance, m2_payload)
        self.missions["3"] = FlightMission(m3_avionics, m3_performance, m3_payload)

        # Wing Structure
        max_v_cruise = max(m1_performance.V_cruise, m2_performance.V_cruise, m3_performance.V_cruise)
        self.wing.size_spar(wing_span, max_v_cruise)

        # Size Fuselage, Tail, Landing Gear
        self.fuselage = SemiMono(self.wing.span, self.wing.chord)
        self.tail = CompressedFoamRectangleInvertedTTail(0.5, 0.04, 1, self.fuselage.wing_ac_to_tail_ac, self.wing.surface_area, wing_span, wing_AR, self.wing.chord)

        self.check_mass_coherence()

    def check_mass_coherence(self):
        summed_mass = (self.missions["1"].avionics.mass_avionics + self.tail.mass + self.wing.mass + self.wing.spar.mass + self.fuselage.mass) * 1.06

        # Calculate the relative difference
        # We want the Guess (m_struct) to be very close to the Result (actual_mass)
        relative_error = abs(summed_mass - self.m_struct) / self.m_struct

        # A 1% tolerance is usually sufficient for engineering convergence
        is_converged = relative_error <= 0.01

        self.relative_error = relative_error
        self.is_converged = is_converged

    def converged(self):
        return self.is_converged

    @property
    def wing_loading(self) -> float:
        """Returns Wing Loading (W/S) in N/m^2 based on Max Takeoff Weight."""
        g = self.wing.config.physics.g
        weight_max = self.m_max * g
        return weight_max / self.wing.surface_area

    @property
    def power_to_weight(self) -> float:
        """Returns Power-to-Weight ratio (P/W) in W/N based on Max Takeoff Weight."""
        g = self.wing.config.physics.g
        weight_max = self.m_max * g
        return self.propulsion.motor_power / weight_max

# wing = XPSRectangularWing("clarkY.dat", 1, 7)
#
# body = SemiMono(wing.span, wing.chord)
# m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio, m1_battery, m2_battery, banner_length, banner_AR, m3_battery = 0.5,1.0,1000.0,4,10,5,10,30,500,4,30
# plane = RCPlane(m_struct=m_struct, wing_span=wing_span, wing_AR=wing_AR, motor_power=motor_power, m1_battery=m1_battery, n_pucks=n_pucks, passenger_cargo_ratio=passenger_cargo_ratio, m2_battery=m2_battery, banner_length=banner_length, banner_AR=banner_AR, m3_battery=m3_battery)
# print(plane)