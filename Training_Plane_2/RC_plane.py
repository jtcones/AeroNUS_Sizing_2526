from Avionics import Avionics
from Landing_gear import Landing_gear
from Propulsion import Propulsion
from Tail import Tail
from Wing import Wing
from Fuselage import Fuselage
import numpy as np
from scipy.optimize import brentq
# volumetric density (kg/m^3) – your values are fine
density = {
    "blue xps foam": 35.0,
    "white foam": 17.0,
    "compressed foam": 49.2,
    "EPS foam": 20.0,
    "carbon spar": 1500.0,
    "basswood": 400.0,
    "air": 1.225
}

# areal mass for skins/tapes/coat (kg/m^2)
areal_mass = {
    "fibreglass_light": 0.03200,     # 32 g/m^2 cloth
    "fibreglass_heavy": 0.24691,     # 247 g/m^2 cloth
    "epoxy": 0.27500,       # 275 g/m^2 per coat (tune!)
    "fibre_tape": 0.20227,           # 202 g/m^2
    "skin_tape": 0.20000,             # 200 g/m^2
    "shrink wrap": 0.05,            # 50 g/m^2
    "compressed foam": 0.21         # 210 g/m^2 for 5mm thick (THP Foam Board 5mm)
}

# mechanical properties in SI (Pa)
UTS_Pa = {
    "carbon_spar": 8.0e8,            # ~800 MPa (typical along fibres; tune to your spec)
    "fibreglass_laminate": 6.0e8,    # ~600 MPa (order-of-magnitude)
    "basswood": 6.0e7                # ~60 MPa
}

young_modulus_Pa = {
    "carbon_spar": 1.4e11,           # ~140 GPa (pultruded CF rod range 120–160 GPa)
    "fibreglass_laminate": 3.5e10,   # ~35 GPa
    "basswood": 1.0e10               # ~10 GPa
}


number_of_ply = 1
safety_factor = 2
g = 9.81

class RC_plane:
    def __init__(self, max_laps, mass_total, wing_details, tail_details):
        self.parasitic_drag_coefficient = 0.17
        self.max_g = 3.5
        self.propulsion = Propulsion()
        self.wing = Wing(wing_details, mass_total)
        self.performance = self.flight_performance(P_effective=self.propulsion.effective_power, m=mass_total, S=self.wing.surface_area, AR=self.wing.aspect_ratio, CD0=self.parasitic_drag_coefficient)
        self.avionics = Avionics(6, self.propulsion.motor_rated_power, self.performance["one_lap_timing"] * max_laps)
        self.fuselage = Fuselage(self.wing, density, areal_mass)
        self.tail = Tail(tail_details, self.wing, self.fuselage.wing_ac_to_tail_ac)
        self.landing = Landing_gear()

        self.mass_wing = self.calc_mass_wing("Fibre Composite Blue Foam", self.performance["V_Cruise"])[0]
        self.mass_tail = self.calc_mass_tail_H("Flat Plate White Foam", self.performance["V_Cruise"])[0] + self.calc_mass_tail_V("Flat Plate White Foam", self.performance["V_Cruise"])[0]
        self.mass_avionics = self.avionics.mass
        self.mass_propulsion = self.propulsion.mass
        self.mass_landing_gear = self.landing.mass
        self.mass_fuselage = self.fuselage.mass
        self.calc_mass_total = self.mass_wing + self.mass_tail + self.mass_fuselage + self.mass_avionics + self.mass_propulsion + self.mass_landing_gear


    def calc_mass_wing(self, construction_method, v_cruise, CL_max=1.3):
        number_of_ply = 1
        mass = 0
        force = 0.5 * density["air"] * CL_max * self.wing.surface_area * (v_cruise)**2
        wing_spar_rad_outer = self.spar_dimension(self.wing.wing_span, force, safety_factor=2)
        self.wing.spar_rad = wing_spar_rad_outer
        if construction_method == "Fibre Composite Blue Foam":
            mass += self.wing.wing_volume * density["blue xps foam"]
            mass += self.wing.wrap_area * areal_mass["shrink wrap"]
            # mass += self.wing.wrap_area * ((areal_mass["fibreglass_light"] + areal_mass["epoxy"]) * number_of_ply)
            mass += np.pi * self.wing.wing_span * ((wing_spar_rad_outer)**2-(wing_spar_rad_outer-0.002)**2) * density["carbon spar"]
        return (mass, wing_spar_rad_outer)

    def spar_dimension(self, wing_span, force, safety_factor, wall_thickness=0.002):
        uts_with_safety_factor = UTS_Pa["carbon_spar"] / safety_factor

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

    def flight_performance(self, P_effective, m, S, AR, g=9.81, rho=1.225, e=0.8, CD0=0.17, CL_min=0.02, CL_max=1.3):
        W = m * g
        k = 1 / (np.pi * AR * e)

        def power_required(CL):
            return (CD0 + k * CL ** 2) * (W / CL) * np.sqrt((2 * W) / (rho * S * CL))

        # Create a fine CL grid to look for sign changes
        CL_grid = np.linspace(CL_min, CL_max, 2000)
        f_vals = P_effective - np.array([power_required(CL) for CL in CL_grid])

        # Find bracket where sign changes
        sign_change_indices = np.where(np.sign(f_vals[:-1]) * np.sign(f_vals[1:]) < 0)[0]
        if len(sign_change_indices) == 0:
            # No bracket found — return best approximate (min residual) and flag
            idx_best = np.argmin(np.abs(f_vals))
            CL_approx = CL_grid[idx_best]
            residual = f_vals[idx_best]
            V = np.sqrt((2 * W) / (rho * S * CL_approx))
            D = W / CL_approx * (CD0 + k * CL_approx ** 2)
            P_req = (CD0 + k * CL_approx ** 2) * (W / CL_approx) * V
            print("No Bracket")
            return {"success": False, "reason": "no_bracket", "CL": CL_approx, "residual_W": residual, "V": V, "D": D,
                    "P_req": P_req}

        # Use first bracket found
        a = CL_grid[sign_change_indices[0]]
        b = CL_grid[sign_change_indices[0] + 1]
        try:
            CL_solution = brentq(lambda CL: P_effective - power_required(CL), a, b)
        except ValueError:
            out = {"success": False, "reason": "brentq_fail", "CL": None}
            print(out)
            return out

        velocity = lambda CL, n: np.sqrt((2 * n * W) / (rho * S * CL))
        V = velocity(CL_solution, 1)
        D = W / CL_solution * (CD0 + k * CL_solution ** 2)
        P_req = (CD0 + k * CL_solution ** 2) * (W / CL_solution) * V

        result = {"success": True, "CL": CL_solution, "V_Cruise": V, "D": D, "P_req": P_req}

        ####Find Max G, Banking Angle and Banking Velocity
        CD = CD0 + k * CL_max ** 2

        A = (CD * W / CL_max) * np.sqrt(2.0 * W / (rho * S * CL_max))
        n_max = (P_effective / A) ** (2.0 / 3.0)
        self.max_g = n_max if n_max < self.max_g else self.max_g
        V_turning = velocity(CL_max, self.max_g)
        bank_angle_rad = np.arccos(1/self.max_g)
        bank_angle_deg = np.degrees(bank_angle_rad)
        turn_radii = V_turning**2 / (g * np.tan(bank_angle_rad))

        result["V_turning"] = V_turning
        result["bank_angle"] = bank_angle_deg
        result["turn_radius"] = turn_radii
        result["max_g"] = self.max_g

        ####Find flight timing and distance of track.
        straight_distance = 152.4 * 4
        turning_distance = 2 * (2 * np.pi * turn_radii)
        take_off_and_land_time = 60
        one_lap_time = straight_distance/V + turning_distance/V_turning + take_off_and_land_time

        result["one_lap_timing"] = one_lap_time

        return result

    def calc_mass_tail_H(self, construction_method, v_cruise, CL=0.5, thickness=0.01):
        force = 0.5 * density["air"] * CL * self.tail.surface_area_H * (v_cruise) ** 2 * 0.5
        if construction_method == "Flat Plate White Foam":
            foam_area = self.tail.wing_span_H * self.tail.chord_root_H
            foam_mass = foam_area * areal_mass["compressed foam"]
            spar_volume = self.tail.wing_span_H * np.pi * (0.0015 ** 2)
            spar_mass = spar_volume * density["carbon spar"]
            fibre_tape_mass = (0.5 * (self.tail.chord_root_H + (self.tail.taper_ratio * self.tail.chord_root_H)) * self.tail.wing_span_H) * areal_mass["fibre_tape"]
            # epoxy_spar_area = np.pi * (2 * 0.0015) * self.tail.wing_span_H
            # epoxy_spar_mass = epoxy_spar_area * areal_mass["epoxy"]
            mass = foam_mass + spar_mass + fibre_tape_mass  #epoxy_spar_mass
            return (mass, 0.0015)

    def calc_mass_tail_V(self, construction_method, v_cruise, CL=0.5, thickness=0.01):
        force = 0.5 * density["air"] * CL * self.tail.surface_area_V * (v_cruise) ** 2 * 0.5
        if construction_method == "Flat Plate White Foam":
            foam_area = self.tail.wing_span_V * self.tail.chord_root_V
            foam_mass = foam_area * areal_mass["compressed foam"]
            spar_volume = self.tail.wing_span_V* np.pi * (0.0015 ** 2)
            spar_mass = spar_volume * density["carbon spar"]
            fibre_tape_mass = (0.5 * (self.tail.chord_root_V + (self.tail.taper_ratio * self.tail.chord_root_V)) * self.tail.wing_span_V) * areal_mass["fibre_tape"]
            # epoxy_spar_area = np.pi * (2 * 0.0015) * self.tail.wing_span_V
            # epoxy_spar_mass = epoxy_spar_area * areal_mass["epoxy"]
            mass = foam_mass + spar_mass + fibre_tape_mass
            return (mass, 0.0015)

    def get_calculated_mass(self):
        return self.calc_mass_total

    def debug_mass(self):
        summary = [
            "--- Mass Breakdown ---",
            f"Wing: {self.mass_wing:.4f} kg",
            f"Tail: {self.mass_tail:.4f} kg",
            f"Fuselage: {self.mass_fuselage:.4f} kg",
            f"Avionics: {self.mass_avionics:.4f} kg",
            f"Propulsion: {self.mass_propulsion:.4f} kg",
            f"Landing Gear: {self.mass_landing_gear:.4f} kg",
            f"Total Calculated Mass: {self.calc_mass_total:.4f} kg",
        ]
        return "\n".join(summary)

    def __str__(self):
        summary = [
            "===== RC Plane Summary =====",
            f"Parasitic Drag Coefficient (CD0): {self.parasitic_drag_coefficient:.3f}",
            f"Max Load Factor (G): {self.max_g}",
            "",
            "--- Performance ---",
            f"Cruise Speed (V_cruise): {self.performance['V_Cruise']:.2f} m/s",
            f"Turning Speed (V_turn): {self.performance['V_turning']:.2f} m/s",
            f"Bank Angle: {self.performance['bank_angle']:.2f} degree",
            f"One Lap Time: {self.performance['one_lap_timing']:.2f} s",
            f"Effective Power: {self.propulsion.effective_power:.2f} W",
            "",
            "--- Mass Breakdown ---",
            f"Wing: {self.mass_wing:.2f} kg",
            f"Tail: {self.mass_tail:.2f} kg",
            f"Fuselage: {self.mass_fuselage:.2f} kg",
            f"Avionics: {self.mass_avionics:.2f} kg",
            f"Propulsion: {self.mass_propulsion:.2f} kg",
            f"Landing Gear: {self.mass_landing_gear:.2f} kg",
            f"Total Calculated Mass: {self.calc_mass_total:.2f} kg",
            "",
            "--- Components ---",
            f"{self.wing}",
            "",
            f"{self.tail}",
            "",
            f"{self.fuselage}",
            "",
            f"{self.avionics}",
            "",
            f"{self.propulsion}",
            "",
            f"{self.landing}",
        ]
        return "\n".join(summary)
