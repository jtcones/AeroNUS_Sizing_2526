import numpy as np
import matplotlib.pyplot as plt
from fixed_params import *
from fixed_params.general import *
from fixed_params.propulsion import *
from fixed_params.avionics import *
from fixed_params.flight_track import *
from fixed_params.wing import *
from material_properties import *
from skopt import gp_minimize
from skopt.space import Real, Integer
from scipy.optimize import brentq

"""
max min wingspan = [3, 5] feet = [0.9144, 1.524] m
length of banner >= 10 inches
height of banner to length ration <= 5
passenger per cargo >= 3
wingspan fixed at 1.2
structure to weight ratio = 0.7
"""


def round_inches(x: float) -> float:
    """
    Round a measurement in inches down to the nearest 0.00, 0.25, 0.50, or 0.75.

    Example:
        10.37 -> 10.25
        5.81  -> 5.75
        7.99  -> 7.75
    """
    # Whole inch part
    whole = int(x)

    # Fractional part
    frac = x - whole

    # Define breakpoints
    targets = [0.00, 0.25, 0.50, 0.75]

    # Find the largest target <= frac
    rounded_frac = max([t for t in targets if t <= frac], default=0.00)

    return whole + rounded_frac


def mission_2(num_passengers, num_cargo, m2_laps, battery_capacity):

    income = (num_passengers * (6 + 2 * m2_laps)) + (num_cargo * (10 + 8 * m2_laps))
    EF = battery_capacity / 100
    cost = m2_laps * (10 + (num_passengers*0.5) + (num_cargo*2)) * EF
    net_income = income - cost
    m2 = 1 + (net_income / 766)
    return m2

def mission_3(banner_length, number_of_laps, wing_span):
    """
    :param banner_length: length in inches
    :param number_of_laps: number of laps
    :param wing_span: span in metres
    :return: mission 3 score with best score of 410
    """
    wing_span_inches = wing_span * 39.3701
    pre = round(wing_span_inches)/12 * 0.05 + 0.75
    RAC = pre if pre >= 0.9 else 0.9
    rounded_banner = round_inches(banner_length)
    m3 = (rounded_banner * number_of_laps / RAC)
    m3_best = 410
    return 2 + (m3 / m3_best)


def objective_m2(x):
    WS, pucks, PC_ratio, fly_time = x
    m_duck = general.duck
    m_puck = general.puck
    ducks = pucks * PC_ratio
    payload = pucks * m_puck + ducks * m_duck

    power_ratio = general.power_ratio  # power required per kg of plane
    structure_ratio = 0.7
    m = (1 + structure_ratio) * payload
    usable_energy = 0.8
    energy_density = avionics.energy_density  # energy per kg of battery
    power_density = propulsion.power_density  # power per kg of motor

    # take off is included in 5 min flight time. landing is not included
    # this means we are fixing every flight for a full 5 mins. is this optimum
    time_TO = 30  # secs
    time_landing = 30  # secs
    time_flight = fly_time + time_landing + time_TO

    # Avionics & Propulsion
    num = power_ratio * (1 + structure_ratio) * payload
    den = 1 - power_ratio * (1 + structure_ratio) * (
                time_flight / 3600 / usable_energy / energy_density + 1 / power_density)
    power = num / den

    m_motor = power / power_density
    m_battery = power * (time_flight / 3600 / usable_energy / energy_density)

    battery_cap = power * (time_flight / 3600 / usable_energy)

    if battery_cap > 100:
        return 0

    P_effective = power * motor_eff

    # Wing
    S = m / wing.wing_loading
    c = S / WS
    AR = WS ** 2 / S

    # Flight performance
    m = (1 + structure_ratio) * payload

    W = m * g
    k = 1 / (np.pi * AR * e)
    q = 0.5 * rho * S

    def power_required(CL):
        return (CD0 + k * CL ** 2) * (W / CL) * np.sqrt((2 * W) / (rho * S * CL))

    def velocity(CL, n):
        return np.sqrt((2 * n * W) / (rho * S * CL))

    def find_CL_cruise():
        # Create a fine CL grid to look for sign changes
        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([power_required(CL) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]
        if len(sign_change_indices) == 0:
            idx_best = np.argmin(np.abs(f_vals))
            CL_approx = CL_grid[idx_best]
            return CL_approx

        # Use first bracket found
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]
        try:
            CL_solution = brentq(lambda CL: P_effective - power_required(CL), a, b)
            return CL_solution
        except ValueError as error:
            raise error

    def find_turning_conditions(CL):
        def find_G(CL):
            CD = CD0 + k * CL ** 2
            A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))
            n_max = (P_effective / A) ** (2.0 / 3.0)
            return n_max

        def find_bank_angle_rad(g):
            return np.arccos(1 / g)

        def find_turn_radius(v, angle):
            return v ** 2 / (g * np.tan(angle))

        n = find_G(CL)
        v = velocity(CL, n)
        angle = find_bank_angle_rad(n)
        radius = find_turn_radius(v, angle)
        return n, v, angle, radius

    def flight_time_one_lap(radius, v_straight, v_turn):
        turning_distance = 2 * (2 * np.pi * radius)

        total_time = (flight_track.straight_distance / v_straight + turning_distance / v_turn) * 1.5
        return total_time

    # calculate maximum turning functionality
    n_max, v_turning_max, bank_angle_rad_max, turn_radius_max = find_turning_conditions(CL_max)

    # calculate turning under CL_turn
    n_turning, v_turning, bank_angle_rad, turn_radius = find_turning_conditions(CL_turn)

    max_bank_angle = np.degrees(bank_angle_rad_max)
    turning_bank_angle = np.degrees(bank_angle_rad)
    V_stall = velocity(CL_max, 1)

    try:
        CL_cruise = find_CL_cruise()
        V_cruise = velocity(CL_cruise, 1)

    except ValueError as error:
        print(f"Error: {error}")

    # number of laps in 5 minutes
    one_lap_time = flight_time_one_lap(turn_radius, V_cruise, v_turning)
    effective_flight_time = fly_time
    number_of_laps = effective_flight_time // one_lap_time
    if number_of_laps == 0:
        return 0
    print("number of laps: ", number_of_laps)

    try:
        m2_score = mission_2(ducks, pucks, number_of_laps, battery_cap)
        if not np.isfinite(m2_score):
            return 1e6  # penalize invalid cases
        return -m2_score
    except Exception as error:
        print(f"Error for params {ducks, pucks, number_of_laps, battery_cap}: {error}")
        return 1e6


def objective_m3(x):
    WS, banner_length, banner_AR, fly_time = x
    # inches to m
    banner_length_m = banner_length * 0.0254
    banner_width = banner_length / banner_AR
    banner_width_m = banner_width * 0.0254
    area_banner = banner_length_m * banner_width_m
    payload = 1.2 * area_banner * areal_mass.lightweight_ripstop

    power_ratio = general.power_ratio  # power required per kg of plane
    structure_ratio = 0.7
    m = (1 + structure_ratio) * payload
    usable_energy = 0.8
    energy_density = avionics.energy_density  # energy per kg of battery
    power_density = propulsion.power_density  # power per kg of motor

    # take off is included in 5 min flight time. landing is not included
    # this means we are fixing every flight for a full 5 mins. is this optimum
    time_TO = 30  # secs
    time_landing = 30  # secs
    time_flight = fly_time + time_landing + time_TO

    # Avionics & Propulsion
    num = power_ratio * (1 + structure_ratio) * payload
    den = 1 - power_ratio * (1 + structure_ratio) * (
                time_flight / 3600 / usable_energy / energy_density + 1 / power_density)
    power = num / den

    m_motor = power / power_density
    m_battery = power * (time_flight / 3600 / usable_energy / energy_density)

    battery_cap = power * (time_flight / 3600 / usable_energy)

    if battery_cap > 100:
        return 0

    P_effective = power * motor_eff

    # Wing
    S = m / wing.wing_loading
    c = S / WS
    AR = WS ** 2 / S

    # Flight performance
    m = (1 + structure_ratio) * payload

    W = m * g
    k = 1 / (np.pi * AR * e)
    q = 0.5 * rho * S

    def velocity(CL, n):
        return np.sqrt((2 * n * W) / (rho * S * CL))

    def power_required(CL):
        v = velocity(CL, 1)
        # more power required because of additional drag. Add D friction, D pressure of banner.
        # drag friction
        Re = v * banner_length_m / general.nu
        cf = 0.074 * np.power(Re, -0.2)
        surface_area = 2 * banner_length_m * banner_width_m
        Df = 0.5 * rho * cf * surface_area * (v ** 2)
        Dp = 0.5 * rho * CDP * 0.021 * banner_width_m * (v ** 2)
        return ((CD0 + k * CL ** 2) * (W / CL) + 2 * (Df + Dp)) * v

    def find_CL_cruise():
        # Create a fine CL grid to look for sign changes
        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([power_required(CL) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]
        if len(sign_change_indices) == 0:
            idx_best = np.argmin(np.abs(f_vals))
            CL_approx = CL_grid[idx_best]
            return CL_approx

        # Use first bracket found
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]
        try:
            CL_solution = brentq(lambda CL: P_effective - power_required(CL), a, b)
            return CL_solution
        except ValueError as error:
            raise error

    def find_turning_conditions(CL):
        def find_G(CL):
            CD = CD0 + k * CL ** 2
            A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))
            n_max = (P_effective / A) ** (2.0 / 3.0)
            return n_max

        def find_bank_angle_rad(g):
            return np.arccos(1 / g)

        def find_turn_radius(v, angle):
            return v ** 2 / (g * np.tan(angle))

        n = find_G(CL)
        v = velocity(CL, n)
        angle = find_bank_angle_rad(n)
        radius = find_turn_radius(v, angle)
        return n, v, angle, radius

    def flight_time_one_lap(radius, v_straight, v_turn):
        turning_distance = 2 * (2 * np.pi * radius)

        total_time = (flight_track.straight_distance / v_straight + turning_distance / v_turn) * 1.5
        return total_time

    # calculate maximum turning functionality
    n_max, v_turning_max, bank_angle_rad_max, turn_radius_max = find_turning_conditions(CL_max)

    # calculate turning under CL_turn
    n_turning, v_turning, bank_angle_rad, turn_radius = find_turning_conditions(CL_turn)

    max_bank_angle = np.degrees(bank_angle_rad_max)
    turning_bank_angle = np.degrees(bank_angle_rad)
    V_stall = velocity(CL_max, 1)

    try:
        CL_cruise = find_CL_cruise()
        V_cruise = velocity(CL_cruise, 1)

    except ValueError as error:
        print(f"Error: {error}")

    # number of laps in 5 minutes
    one_lap_time = flight_time_one_lap(turn_radius, V_cruise, v_turning)
    effective_flight_time = fly_time
    number_of_laps = effective_flight_time // one_lap_time
    if number_of_laps == 0:
        return 0
    print("number of laps: ", number_of_laps)

    try:
        m3_score = mission_3(banner_length, number_of_laps, WS)
        if not np.isfinite(m3_score):
            return 1e6  # penalize invalid cases
        return -m3_score
    except Exception as error:
        print(f"Error for params {banner_length, banner_AR, number_of_laps, battery_cap}: {error}")
        return 1e6

def objective_total(x):
    # Unpack decision variables
    WS, pucks, PC_ratio, banner_length, banner_AR, m2_fly_time, m3_fly_time = x
    m_duck = general.duck
    m_puck = general.puck
    ducks = pucks * PC_ratio
    m2_payload = pucks * m_puck + ducks * m_duck
    CD0 = 0.13 + 0.01 * np.sqrt(ducks)
    banner_length_m = banner_length * 0.0254
    banner_width = banner_length / banner_AR
    banner_width_m = banner_width * 0.0254
    area_banner = banner_length_m * banner_width_m
    m3_payload = 1.2 * area_banner * areal_mass.lightweight_ripstop

    payload = m2_payload if m2_payload > m3_payload else m3_payload

    power_ratio = general.power_ratio  # power required per kg of plane
    structure_ratio = 1.5
    m = (1 + structure_ratio) * payload
    usable_energy = 0.8
    energy_density = avionics.energy_density  # energy per kg of battery
    power_density = propulsion.power_density  # power per kg of motor

    # take off is included in 5 min flight time. landing is not included
    # this means we are fixing every flight for a full 5 mins. is this optimum
    time_TO = 30  # secs
    time_landing = 30  # secs
    m2_time_flight = m2_fly_time + time_landing + time_TO
    m3_time_flight = m3_fly_time + time_landing + time_TO

    # Avionics & Propulsion
    # m2_num = power_ratio * (1 + structure_ratio) * payload
    # den = 1 - power_ratio * (1 + structure_ratio) * (
    #         time_flight / 3600 / usable_energy / energy_density + 1 / power_density)
    # power = num / den

    power = power_ratio * m
    m_motor = power / power_density
    m_battery = power * (m2_time_flight / 3600 / usable_energy / energy_density)

    battery_cap = power * (m2_time_flight / 3600 / usable_energy)

    if battery_cap >= 100:
        return 0

    P_effective = power * motor_eff

    # Wing
    S = m / wing.wing_loading
    c = S / WS

    print("chord: ", c)
    AR = WS ** 2 / S

    # Flight performance
    m_struct = payload * structure_ratio
    print("mstruct: ", m_struct)
    print("m2_payload", m2_payload)
    print("m3_payload", m3_payload)
    m_2 = m_struct + m2_payload
    m_3 = m_struct + m3_payload
    W_2 = m_2 * g
    W_3 = m_3 * g
    k = 1 / (np.pi * AR * e)
    q = 0.5 * rho * S

    def power_required(CL, W):
        return (CD0 + k * CL ** 2) * (W / CL) * np.sqrt((2 * W) / (rho * S * CL))

    def velocity(CL, n, W):
        return np.sqrt((2 * n * W) / (rho * S * CL))

    def find_CL_cruise(W):
        # Create a fine CL grid to look for sign changes
        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([power_required(CL,W) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]
        if len(sign_change_indices) == 0:
            idx_best = np.argmin(np.abs(f_vals))
            CL_approx = CL_grid[idx_best]
            return CL_approx

        # Use first bracket found
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]
        try:
            CL_solution = brentq(lambda CL: P_effective - power_required(CL, W), a, b)
            return CL_solution
        except ValueError as error:
            raise error

    def find_turning_conditions(CL, W):
        def find_G(CL):
            CD = CD0 + k * CL ** 2
            A = (CD * W / CL) * np.sqrt(2.0 * W / (rho * S * CL))
            n_max = (P_effective / A) ** (2.0 / 3.0)
            return n_max

        def find_bank_angle_rad(g):
            return np.arccos(1 / g)

        def find_turn_radius(v, angle):
            return v ** 2 / (g * np.tan(angle))

        n = find_G(CL)
        v = velocity(CL, n, W)
        angle = find_bank_angle_rad(n)
        radius = find_turn_radius(v, angle)
        return n, v, angle, radius

    def flight_time_one_lap(radius, v_straight, v_turn):
        turning_distance = 2 * (2 * np.pi * radius)

        total_time = (flight_track.straight_distance / v_straight + turning_distance / v_turn) * 1.5
        return total_time

    # calculate maximum turning functionality
    n_max, v_turning_max, bank_angle_rad_max, turn_radius_max = find_turning_conditions(CL_max, W_2)

    # calculate turning under CL_turn
    n_turning, v_turning, bank_angle_rad, turn_radius = find_turning_conditions(CL_turn, W_2)

    max_bank_angle = np.degrees(bank_angle_rad_max)
    turning_bank_angle = np.degrees(bank_angle_rad)
    V_stall = velocity(CL_max, 1, W_2)

    try:
        CL_cruise = find_CL_cruise(W_2)
        V_cruise = velocity(CL_cruise, 1, W_2)

    except ValueError as error:
        print(f"Error: {error}")

    # number of laps in 5 minutes
    one_lap_time = flight_time_one_lap(turn_radius, V_cruise, v_turning)
    effective_flight_time = m2_fly_time
    number_of_laps = effective_flight_time // one_lap_time
    if number_of_laps == 0:
        return 1e6
    print("number of laps: ", number_of_laps)

    try:
        m2_score = mission_2(ducks, pucks, number_of_laps, battery_cap)
        if not np.isfinite(m2_score):
            return 1e6  # penalize invalid cases
    except Exception as error:
        print(f"Error for params {ducks, pucks, number_of_laps, battery_cap}: {error}")
        return 1e6

    def power_required_m3(CL, W):
        v = velocity(CL, 1, W)
        # more power required because of additional drag. Add D friction, D pressure of banner.
        # drag friction
        CD_banner = 0.5 * np.power(banner_AR, -0.5)
        return ((CD0 + CD_banner + k * CL ** 2) * (W / CL)) * v

    def find_CL_cruise_m3(W):
        # Create a fine CL grid to look for sign changes
        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([power_required_m3(CL, W) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]
        if len(sign_change_indices) == 0:
            idx_best = np.argmin(np.abs(f_vals))
            CL_approx = CL_grid[idx_best]
            return CL_approx

        # Use first bracket found
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]
        try:
            CL_solution = brentq(lambda CL: P_effective - power_required(CL, W), a, b)
            return CL_solution
        except ValueError as error:
            raise error

    # calculate turning under CL_turn
    n_turning, v_turning, bank_angle_rad, turn_radius = find_turning_conditions(CL_turn, W_3)

    turning_bank_angle = np.degrees(bank_angle_rad)
    V_stall = velocity(CL_max, 1, W_3)

    try:
        CL_cruise = find_CL_cruise_m3(W_3)
        V_cruise = velocity(CL_cruise, 1, W_3)

    except ValueError as error:
        print(f"Error: {error}")

    # number of laps in 5 minutes
    one_lap_time = flight_time_one_lap(turn_radius, V_cruise, v_turning)
    effective_flight_time = m3_fly_time
    number_of_laps = effective_flight_time // one_lap_time
    if number_of_laps == 0:
        return 1e6
    print("number of laps: ", number_of_laps)

    try:
        m3_score = mission_3(banner_length, number_of_laps, WS)
        if not np.isfinite(m2_score):
            return 1e6  # penalize invalid cases
    except Exception as error:
        print(f"Error for params {ducks, pucks, number_of_laps, battery_cap}: {error}")
        return 1e6

    return - (m3_score + m2_score)

#WS, pucks, PC_ratio, banner_length, banner_AR, m2_fly_time, m3_fly_time
search_space = [
    Real(0.9, 1.52, name="wing_span"),      # m
    Integer(0, 30, name="pucks"),
    Integer(3, 15, name="passenger_cargo_ratio"),
    Integer(10, 1000, name="banner_length"), #inch
    Integer(1, 6, name="banner_AR"),
    Integer(45, 270, name="m2_fly_time"), #s
    Integer(45, 270, name="m3_fly_time") #s
]

# Run Bayesian Optimization
result = gp_minimize(
    objective_total,
    search_space,
    n_calls=1000,
    n_initial_points=10,
    random_state=42
)

# Best design
best_params = result.x
#WS, pucks, PC_ratio, banner_length, banner_AR, m2_fly_time, m3_fly_time
print("Best parameters found:", best_params)

print(objective_total(best_params))
# x = (0.9, 26, 10, 45)
# print(objective_bo(x))