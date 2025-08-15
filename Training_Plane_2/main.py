"""
This is the main file for the AeroNUS sizing program.

Overview of algo
we will guess the first mass ratio.
Through the sizing program, we will size the plane in the following order
    wings,
    tail,
    propulsion,
    fuselage,
    flight performance,
    avionics,
    structural properties,
    landing gear

the result of sizing each part will return the mass of each part such that it can carry our desired payload.

With this total mass, we can identify the true mass ratio and see if it matches our input. From there we will tune the
mass ratio and run the sizing program again till we achieve a sizing within 0.001 range.
"""
import numpy as np
from scipy.optimize import brentq, fixed_point
from RC_plane import RC_plane

wing_details = {
    "wing_span": 1.2,
    "taper_ratio": 1,
    "wing_loading": 5.5,
    "airfoil_type": "clark Y"
}

tail_details = {
    "taper_ratio": 1,
    "tail_coefficient_H": 0.5,
    "tail_coefficient_V": 0.025
}

# size wings

# #mission 1 (empty payload)
m1_laps = 2
#
# #mission 2 (full payload)
m2_laps = 4

payload = 0.6
mass_structure = 1.1


mass_total = mass_structure + payload
print(RC_plane(max(m2_laps, m1_laps), mass_total, wing_details, tail_details))







# tolerance = 1e-3
# max_iterations = 100
# for i in range(max_iterations):
#     mass_total = mass_structure + payload
#     tp1 = RC_plane(max(m2_laps, m1_laps), mass_total, wing_details, tail_details)
#     calc_structure_mass = tp1.get_calculated_mass()
#     print(f"Iter {i}: Input guess = {mass_structure:.3f} kg, Calculated structure mass = {calc_structure_mass:.3f} kg")
#     print(tp1.debug_mass())
#     if abs(calc_structure_mass - mass_structure) < tolerance:
#         print(f"Converged after {i + 1} iterations")
#         print(tp1)
#         with open("results.txt", "w", encoding="utf-8") as f:
#             f.write(tp1.__str__())
#         break
#     mass_structure = 0.5 * (calc_structure_mass + mass_structure)



# def mass_structure_func(m_struct):
#     mass_total = m_struct + payload
#     tp1 = RC_plane(max(m2_laps, m1_laps), mass_total, wing_details, tail_details)
#     return tp1.get_calculated_mass() - payload
#
# mass_structure_solution = fixed_point(mass_structure_func, 1.0)  # initial guess 1.0 kg

# tp1 = RC_plane(max(m2_laps, m1_laps), mass_structure_solution + payload, wing_details, tail_details)




