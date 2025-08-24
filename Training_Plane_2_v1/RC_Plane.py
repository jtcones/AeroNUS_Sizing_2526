from fixed_params.general import *
from fixed_params import *
from material_properties import *
import numpy as np
from scipy.optimize import brentq
from math import ceil

class RC_Plane():
    def __init__(self, params):
        self.params = params
        self.params.setdefault("m_struct", 0)

    def assign_param(self, assignment):
        key, value = assignment
        self.params[key] = value
        return RC_Plane(self.params)

    def propulsion(self):
        """
        Step 1 of RC Plane Sizing,
        After choosing a motor power
        :return: an RC_Plane object with new params
        """
        power = self.params["motor_power"]
        overall_propulsion_efficiency = propulsion.motor_eff * propulsion.prop_eff * avionics.esc_eff
        self.params["m_propulsion"] = power * propulsion.mass_per_W
        self.params["effective_power"] = power * overall_propulsion_efficiency
        self.params["m_struct"] = self.params["m_struct"] + self.params["m_propulsion"]
        return RC_Plane(self.params)

    def wing_dimensioning(self):
        """
        To size wing dimensions
        :return: updated RC_Plane object with new params
        """
        wing_span = self.params["wing_span"]
        m_total = self.params["m_total"]
        wing_S = m_total / wing.wing_loading
        wing_c = wing_S / wing_span
        wing_AR = wing_span ** 2 / wing_S
        self.params["wing_S"] = wing_S
        self.params["wing_c"] = wing_c
        self.params["wing_AR"] = wing_AR
        return RC_Plane(self.params)

    def performance(self):
        """
        flight performance calculations
        :return:
        """
        m = self.params["m_total"]
        AR = self.params["wing_AR"]
        S = self.params["wing_S"]
        P_effective = self.params["effective_power"]


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
                # No bracket found — return best approximate (min residual) and flag
                idx_best = np.argmin(np.abs(f_vals))
                CL_approx = CL_grid[idx_best]
                return CL_approx

                # print("No Bracket")
                # return {"success": False, "reason": "no_bracket", "CL": CL_approx, "residual_W": residual, "V": V, "D": D,
                #         "P_req": P_req}

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

        def flight_time(radius, v_straight, v_turn):
            turning_distance = 2 * (2 * np.pi * radius)

            total_time = (flight_track.straight_distance / v_straight + turning_distance / v_turn) * flight_track.max_laps + flight_track.take_off_and_land_time
            return total_time

        #calculate maximum turning functionality
        n_max, v_turning_max, bank_angle_rad_max, turn_radius_max = find_turning_conditions(CL_max)

        #calculate turning under CL_turn
        n_turning, v_turning, bank_angle_rad, turn_radius = find_turning_conditions(CL_turn)


        self.params["n_max"] = n_max
        self.params["v_turning_max"] = v_turning_max
        self.params["bank_angle_max"] = np.degrees(bank_angle_rad_max)
        self.params["turn_radius_max"] = turn_radius_max
        self.params["n_turning"] = n_turning
        self.params["v_turning"] = v_turning
        self.params["bank_angle"] = np.degrees(bank_angle_rad)
        self.params["turn_radius"] = turn_radius
        self.params["V_stall"] = velocity(CL_max, 1)

        try:
            CL_cruise = find_CL_cruise()
            V_cruise = velocity(CL_cruise, 1)

            self.params["CL_cruise"] = CL_cruise
            self.params["V_cruise"] = V_cruise
            self.params["max_laps_flight_time"] = flight_time(turn_radius, V_cruise, v_turning)

        except ValueError as error:
            print(f"Error: {error}")

        return RC_Plane(self.params)

    def calc_mass_wing(self, construction_method):
        wing_span, c, S, AR = self.get_wing_dimensions()
        v_cruise = self.params["V_cruise"]

        def calc_wing_volume(wing_type, chord, wing_span):
            volume = wing.airfoil_constant[wing_type][0] * chord ** 2 * wing_span
            return volume

        def calc_total_surface_area(wing_type, chord, wing_span):
            area = wing.airfoil_constant[wing_type][1] * chord * 4 * wing_span
            return area

        mass = 0
        force = 0.5 * density.air * CL_max * S * (v_cruise) ** 2
        wing_spar_rad_outer = spar_dimension(wing_span, force, safety_factor=2)
        wing_volume = calc_wing_volume(wing.airfoil_type, c, wing_span)
        wrap_area = calc_total_surface_area(wing.airfoil_type, c, wing_span)
        if construction_method == "Fibre Composite Blue Foam":
            mass += wing_volume * density.blue_xps_foam
            mass += wrap_area * areal_mass.shrink_wrap
            # mass += wing.wrap_area * ((areal_mass["fibreglass_light"] + areal_mass["epoxy"]) * number_of_ply)
            mass += np.pi * wing_span * ((wing_spar_rad_outer) ** 2 - (wing_spar_rad_outer - 0.002) ** 2) * \
                    density.carbon_spar

        self.params["m_wing"] = mass
        self.params["wing_spar_rad_outer"] = wing_spar_rad_outer
        self.params["m_struct"] = self.params["m_struct"] + mass
        return RC_Plane(self.params)

    def semi_monocoque_mass(self, fuselage_skin_thickness=0.005):  # Compressed foam + wood structures
        length = self.params["wing_span"] * fuselage.length_constant
        width = length * fuselage.width_constant
        height = length * fuselage.height_constant
        # Calculate skin mass
        front_area = width * height
        rear_area = front_area
        bottom_area = length * width
        top_area = bottom_area
        left_area = length * height
        right_area = left_area
        fuselage_skin_area = front_area + rear_area + bottom_area + top_area + left_area + right_area
        fuselage_skin_volume = fuselage_skin_area * fuselage_skin_thickness
        fuselage_skin_mass = fuselage_skin_volume * density.white_foam

        # calculate structural mass
        structural_volume = length * width * height * fuselage.int_struc_percentage
        structural_mass = structural_volume * density.basswood

        # Sum total mass
        total_mass = fuselage_skin_mass + structural_mass
        self.params["m_fuselage"] = total_mass
        self.params["fuselage_width"] = width
        self.params["fuselage_length"] = length
        self.params["fuselage_height"] = height
        self.params["wing_ac_from_nose"] = length * 0.15 + self.params["wing_c"] * 0.25
        self.params["tail_ac_from_nose"] = self.params["wing_ac_from_nose"] + self.params["wing_span"] * 0.45
        self.params["wing_ac_to_tail_ac"] = self.params["tail_ac_from_nose"] - self.params["wing_ac_from_nose"]
        self.params["m_struct"] = self.params["m_struct"] + total_mass
        return RC_Plane(self.params)

    def tail_mass_and_dimension(self):
        def calc_mass_tail_H(construction_method, thickness=0.01):
            #force = 0.5 * density.air * CL * tail_surface_area_H * (v_cruise) ** 2 * 0.5
            if construction_method == "Flat Plate White Foam":
                foam_area = tail_span_H * tail_c_H
                foam_mass = foam_area * areal_mass.compressed_foam
                spar_volume = tail_span_H * np.pi * (0.0015 ** 2)
                spar_mass = spar_volume * density.carbon_spar
                fibre_tape_mass = (0.5 * (tail_c_H + (
                            tail.taper_ratio * tail_c_H)) * tail_span_H) * areal_mass.fibre_tape
                # epoxy_spar_area = np.pi * (2 * 0.0015) * self.tail.tail_span_H
                # epoxy_spar_mass = epoxy_spar_area * areal_mass["epoxy"]
                mass = foam_mass + spar_mass + fibre_tape_mass  # epoxy_spar_mass
                return (mass, 0.0015)

        def calc_mass_tail_V( construction_method, thickness=0.01):
            #force = 0.5 * density.air * CL * tail_surface_area_V * (v_cruise) ** 2 * 0.5
            if construction_method == "Flat Plate White Foam":
                foam_area = tail_span_V * tail_c_V
                foam_mass = foam_area * areal_mass.compressed_foam
                spar_volume = tail_span_V * np.pi * (0.0015 ** 2)
                spar_mass = spar_volume * density.carbon_spar
                fibre_tape_mass = (0.5 * (tail_c_V + (
                            tail.taper_ratio * tail_c_V)) * tail_span_V) * areal_mass.fibre_tape
                # epoxy_spar_area = np.pi * (2 * 0.0015) * tail_span_V
                # epoxy_spar_mass = epoxy_spar_area * areal_mass["epoxy"]
                mass = foam_mass + spar_mass + fibre_tape_mass
                return (mass, 0.0015)

        moment_arm = self.params["wing_ac_to_tail_ac"]
        wing_span, wing_chord, wing_surface_area, wing_AR = self.get_wing_dimensions()
        tail_surface_area_H = tail.tail_coefficient_H * wing_chord * wing_surface_area / moment_arm
        tail_AR_H = wing_AR / 2
        tail_span_H = (tail_surface_area_H * tail_AR_H) ** (0.5)
        tail_c_H = tail_surface_area_H / (tail_span_H / 2 * (1 + tail.taper_ratio))

        tail_surface_area_V = tail.tail_coefficient_V * wing_span * wing_surface_area / moment_arm
        tail_AR_V = tail_AR_H
        tail_span_V = (tail_surface_area_V * tail_AR_V) ** (0.5)
        tail_c_V = tail_surface_area_V / (tail_span_V / 2 * (1 + tail.taper_ratio))

        self.params["tail_surface_area_H"] = tail_surface_area_H
        self.params["tail_AR_H"] = tail_AR_H
        self.params["tail_span_H"] = tail_span_H
        self.params["tail_c_H"] = tail_c_H
        self.params["tail_surface_area_V"] = tail_surface_area_V
        self.params["tail_AR_V"] = tail_AR_V
        self.params["tail_span_V"] = tail_span_V
        self.params["tail_c_V"] = tail_c_V
        self.params["m_tail"] = calc_mass_tail_V("Flat Plate White Foam")[0] + calc_mass_tail_H("Flat Plate White Foam")[0]
        self.params["m_struct"] = self.params["m_struct"] + self.params["m_tail"]

        return RC_Plane(self.params)

    def calc_avionics(self):
        n_cells = self.params["battery_cell"]
        p_motor = self.params["motor_power"]
        total_time_seconds = self.params["max_laps_flight_time"]
        current = p_motor / (n_cells * avionics.volts_p_cell)
        current_mA = current * 1000
        capacity = (total_time_seconds/3600 * current_mA) / avionics.depth_of_discharge
        rounded_capacity = ceil(capacity/100) * 100
        mass = rounded_capacity * avionics.mass_p_mAh + avionics.servo_mass * avionics.num_servo + avionics.ESC_mass

        self.params["current"] = current
        self.params["capacity"] = rounded_capacity
        self.params["m_avionics"] = mass
        self.params["m_struct"] = self.params["m_struct"] + mass
        return RC_Plane(self.params)

    def landing_gear(self):
        self.params["m_landing_gear"] = landing_gear.landing_gear_mass_percentage * self.params["m_total"]
        self.params["m_struct"] = self.params["m_struct"] + self.params["m_landing_gear"]
        return RC_Plane(self.params)

    def get_fuselage_dimensions(self):
        """
        Returns Fuselage Dimensions
        :return: length, width, height, wing_ac_from_nose, tail_ac_from_nose, wing_ac_to_tail_ac
        """
        return self.params["fuselage_length"], self.params["fuselage_width"], self.params["fuselage_height"], self.params["wing_ac_from_nose"], self.params["tail_ac_from_nose"], self.params["wing_ac_to_tail_ac"]

    def get_tail_H_dimensions(self):
        return self.params["tail_span_H"], self.params["tail_c_H"], self.params["tail_surface_area_H"], self.params["tail_AR_H"]

    def get_tail_V_dimensions(self):
        return self.params["tail_span_V"], self.params["tail_c_V"], self.params["tail_surface_area_V"], self.params["tail_AR_V"]

    def get_wing_dimensions(self):
        """
        :return: wing dimensions in the order
        wing span, wing chord, wing surface area, wing aspect ratio
        """
        wing_span = self.params["wing_span"]
        wing_S = self.params["wing_S"]
        wing_c = self.params["wing_c"]
        wing_AR = self.params["wing_AR"]
        return wing_span, wing_c, wing_S, wing_AR

    def get_propulsion(self):
        return self.params["motor_power"], self.params["effective_power"]

    def get_performance(self):
        """
        :return: CL_cruise, V_cruise, V_stall, n_max, v_turning_max, bank_angle_max, turn_radius_max, n_turning,
        v_turning, bank_angle, turn radius, flight_time
        """
        return self.params["CL_cruise"], self.params["V_cruise"], self.params["V_stall"], self.params["n_max"], self.params[
            "v_turning_max"], self.params["bank_angle_max"], self.params[
            "turn_radius_max"], self.params["n_turning"], self.params["v_turning"], self.params[
            "bank_angle"], self.params["turn_radius"], self.params["max_laps_flight_time"]

    def get_avionics(self):
        return self.params["capacity"], self.params["battery_cell"], self.params["current"]

    def print_plane_specs(self):
        plane_dict = self.params
        # --- Overall Aircraft ---
        print("=== OVERALL AIRCRAFT ===")
        print(f"Total mass (m_total): {plane_dict['m_total']:.3f} kg  # includes payload")
        print(f"Structural mass (m_struct): {plane_dict['m_struct']:.3f} kg")
        print(f"Wing mass (m_wing): {plane_dict['m_wing']:.3f} kg")
        print(f"Fuselage mass (m_fuselage): {plane_dict['m_fuselage']:.3f} kg")
        print(f"Tail mass (m_tail): {plane_dict['m_tail']:.3f} kg")
        print(f"Landing gear mass (m_landing_gear): {plane_dict['m_landing_gear']:.3f} kg")
        print(f"Propulsion mass (m_propulsion): {plane_dict['m_propulsion']:.3f} kg")
        print(f"Avionics mass (m_avionics): {plane_dict['m_avionics']:.3f} kg")

        # --- Wing ---
        print("\n=== WING ===")
        print(f"Span (wing_span): {plane_dict['wing_span']:.3f} m")
        print(f"Area (wing_S): {plane_dict['wing_S']:.3f} m²")
        print(f"Chord (wing_c): {plane_dict['wing_c']:.3f} m")
        print(f"Aspect ratio (wing_AR): {plane_dict['wing_AR']:.3f}")
        print(f"Aerodynamic center from nose (wing_ac_from_nose): {plane_dict['wing_ac_from_nose']:.3f} m")
        print(f"Wing spar radius (wing_spar_rad_outer): {plane_dict['wing_spar_rad_outer']:.3f} m")

        # --- Tail ---
        print("\n=== TAIL ===")
        print("Horizontal tail:")
        print(f"  Area (tail_surface_area_H): {plane_dict['tail_surface_area_H']:.3f} m²")
        print(f"  Span (tail_span_H): {plane_dict['tail_span_H']:.3f} m")
        print(f"  Chord (tail_c_H): {plane_dict['tail_c_H']:.3f} m")
        print(f"  Aspect ratio (tail_AR_H): {plane_dict['tail_AR_H']:.3f}")
        print("Vertical tail:")
        print(f"  Area (tail_surface_area_V): {plane_dict['tail_surface_area_V']:.3f} m²")
        print(f"  Span (tail_span_V): {plane_dict['tail_span_V']:.3f} m")
        print(f"  Chord (tail_c_V): {plane_dict['tail_c_V']:.3f} m")
        print(f"  Aspect ratio (tail_AR_V): {plane_dict['tail_AR_V']:.3f}")
        print(f"Tail AC from nose (tail_ac_from_nose): {plane_dict['tail_ac_from_nose']:.3f} m")
        print(f"Wing AC to tail AC distance (wing_ac_to_tail_ac): {plane_dict['wing_ac_to_tail_ac']:.3f} m")

        # --- Fuselage ---
        print("\n=== FUSELAGE ===")
        print(f"Length (fuselage_length): {plane_dict['fuselage_length']:.3f} m")
        print(f"Width (fuselage_width): {plane_dict['fuselage_width']:.3f} m")
        print(f"Height (fuselage_height): {plane_dict['fuselage_height']:.3f} m")

        # --- Propulsion & Battery ---
        print("\n=== PROPULSION & BATTERY ===")
        print(f"Motor power (motor_power): {plane_dict['motor_power']:.1f} W")
        print(f"Battery cells (battery_cell): {plane_dict['battery_cell']:.2f}")
        print(f"Effective power (effective_power): {plane_dict['effective_power']:.2f} W")
        print(f"Current (current): {plane_dict['current']:.2f} A")
        print(f"Battery capacity (capacity): {plane_dict['capacity']} mAh")

        # --- Aerodynamic Performance ---
        print("\n=== AERODYNAMIC PERFORMANCE ===")
        print(f"Max load factor (n_max): {plane_dict['n_max']:.3f}")
        print(f"Max turning velocity (v_turning_max): {plane_dict['v_turning_max']:.2f} m/s")
        print(f"Max bank angle (bank_angle_max): {plane_dict['bank_angle_max']:.2f}°")
        print(f"Max turn radius (turn_radius_max): {plane_dict['turn_radius_max']:.2f} m")
        print(f"Turn load factor (n_turning): {plane_dict['n_turning']:.3f}")
        print(f"Turn velocity (v_turning): {plane_dict['v_turning']:.2f} m/s")
        print(f"Bank angle during turn (bank_angle): {plane_dict['bank_angle']:.2f}°")
        print(f"Turn radius (turn_radius): {plane_dict['turn_radius']:.2f} m")
        print(f"Stall speed (V_stall): {plane_dict['V_stall']:.2f} m/s")
        print(f"Cruise speed (V_cruise): {plane_dict['V_cruise']:.2f} m/s")
        print(f"Cruise lift coefficient (CL_cruise): {plane_dict['CL_cruise']:.3f}")
        print(f"Max laps / flight time (max_laps_flight_time): {plane_dict['max_laps_flight_time']:.2f} s")


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

    rad_o = 0.001  # starting guess outer radius (m)
    while True:
        r_i = rad_o - wall_thickness
        I = np.pi * (rad_o ** 4 - r_i ** 4) / 4
        sigma = M * rad_o / I
        if sigma <= uts_with_safety_factor:
            break
        rad_o += 0.0005  # increment 0.5 mm
    return rad_o

def velocity(CL, n, W, S):
    return np.sqrt((2 * n * W) / (rho * S * CL))

# assign = {"m_total": 1, "wing_span": 1, "motor_power": 300, "battery_cell": 6}
# plane = RC_Plane(assign)
# final_plane = plane.wing_dimensioning().semi_monocoque_mass().tail_mass_and_dimension().landing_gear().propulsion().performance().calc_mass_wing(
#     "Fibre Composite Blue Foam").calc_avionics()
# print(final_plane.params)


