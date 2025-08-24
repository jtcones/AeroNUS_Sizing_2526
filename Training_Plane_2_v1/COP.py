import numpy as np
from fixed_params import *
from scipy.optimize import minimize
from RC_Plane import RC_Plane
from scipy.optimize import NonlinearConstraint

#plane input params
domains = {
    "m_total": list(np.arange(0, 10.01, 0.01)), #kg
    "wing_span": list(np.arange(1, 3.1, 0.1)),
    "motor_power": list(np.arange(300, 1001, 100)),
    "battery_cell": [2, 3, 4, 6]
}

constraints = {
    ("m_total", "m_struct"): lambda m, t: abs(m - general.payload - t) > 0.1,
}


def objective(x):
    assign = {"m_total": x[0], "wing_span": x[1], "motor_power": x[2], "battery_cell": x[3]}
    plane = RC_Plane(assign)
    final_plane = plane.wing_dimensioning().semi_monocoque_mass().tail_mass_and_dimension().landing_gear().propulsion().performance().calc_mass_wing(
        "Fibre Composite Blue Foam").calc_avionics()

    return -mission_score(final_plane)  # negative because we maximize

def consistent(c, assignment):
    func = c[("m_total", "m_struct")]
    m = assignment["m_total"]
    t = assignment["m_struct"]
    return func(m, t)

def mission_score(plane):
    #RAC
    mass_plane = plane.params["m_struct"] / 5
    #assembly_time - for every once the wing span exceeds 1.2m, assembly of wing will double. once fuselage exceeds 1.2m, asssembly time for fuselage doubles
    wing_span = plane.params["wing_span"]
    fuselage_length = plane.params["fuselage_length"]
    assembly_time = 1 + 0.5 * (wing_span / 1.2) + 0.5 * (fuselage_length / 1.2)
    SCF = 1 / (mass_plane * assembly_time)

    time = plane.params["max_laps_flight_time"]

    total_score = SCF + SCF/time
    return total_score


def mass_consistency(x):
    assign = {"m_total": x[0], "wing_span": x[1], "motor_power": x[2], "battery_cell": x[3]}
    plane = RC_Plane(assign)
    final_plane = plane.wing_dimensioning().semi_monocoque_mass().tail_mass_and_dimension().landing_gear().propulsion().performance().calc_mass_wing(
        "Fibre Composite Blue Foam").calc_avionics()
    assignment = final_plane.params
    m = assignment["m_total"]
    t = assignment["m_struct"]
    return m - (general.payload + t)

nlc = NonlinearConstraint(mass_consistency, -0.1, 0.1)
x0 = [1.2, 1.2, 400, 6]
bounds = [(1, 10), (1, 3), (300, 1000), (2, 6)]
result = minimize(objective, x0, bounds=bounds, constraints=[nlc])
if result.success:
    x = result.x
    assign = {"m_total": x[0], "wing_span": x[1], "motor_power": x[2], "battery_cell": x[3]}
    plane = RC_Plane(assign)
    final_plane = plane.wing_dimensioning().semi_monocoque_mass().tail_mass_and_dimension().landing_gear().propulsion().performance().calc_mass_wing(
        "Fibre Composite Blue Foam").calc_avionics()
    print(final_plane.params)
print(result)