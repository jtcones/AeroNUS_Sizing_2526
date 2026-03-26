from AeroPy.aeropy import xfoil_module
from dataclasses import dataclass, field, InitVar
from material_properties import UTS, density
from abc import ABC, abstractmethod
import numpy as np
import os
import subprocess

#plane wing params
airfoil_type = "clark Y"
wing_loading = 14 #kg/m^2
airfoil_constant = { # area, circumference
    "clark Y": (0.0809931, 0.5111733)
}

@dataclass(frozen=True)
class PhysicsEnv:
    g: float = 9.81
    rho: float = 1.225
    e: float = 0.8

@dataclass(frozen=True)
class AeroLimits:
    cl_min: float = 0.2
    cl_max: float = 1.5
    cl_turn: float = 1.0

@dataclass(frozen=True)
class DragParams:
    cd00: float = 0.17
    cdp: float = 1.0

@dataclass(frozen=True)
class FlightRouteParams:
    straight_distance: float = 152.4 * 4 # 500 ft * 4

@dataclass
class AircraftConfig:
    """Master configuration that holds the substructures."""
    physics: PhysicsEnv = field(default_factory=PhysicsEnv)
    aero: AeroLimits = field(default_factory=AeroLimits)
    drag: DragParams = field(default_factory=DragParams)
    route: FlightRouteParams = field(default_factory=FlightRouteParams)
    xfoil_path: str = r"../xfoil.exe"


@dataclass
class Spar:
    material: str = "Carbon Fiber"

    # These remain 0 or an initial guess until sized
    outer_diameter: float = 0.0
    wall_thickness: float = 0.002
    mass: float = field(init=False, default=0.0)
    is_sized: bool = field(init=False, default=False)

    def spar_dimension(self, wing_span, force, safety_factor=2):
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
            r_i = rad_o - self.wall_thickness
            I = np.pi * (rad_o ** 4 - r_i ** 4) / 4
            sigma = M * rad_o / I
            if sigma <= uts_with_safety_factor:
                break
            rad_o += 0.0005  # increment 0.5 mm
        self.outer_diameter= rad_o
        self.mass = np.pi * wing_span * (rad_o ** 2 - (rad_o - self.wall_thickness) ** 2) * density.carbon_spar
        self.is_sized = True


@dataclass
class Wing(ABC):
    airfoil_type: str #dat file link unless it is a naca airfoil.
    span: float
    aspect_ratio: float
    spar: Spar = field(default_factory=Spar)
    config: AircraftConfig = field(default_factory=AircraftConfig, repr=False)

    # Calculated fields
    surface_area: float = field(init=False)
    root_chord: float = field(init=False)
    tip_chord: float = field(init=False)
    chord: float = field(init=False)
    mass: float = field(init=False)
    airfoil_area_ratio: float = field(init=False)
    airfoil_circum_ratio: float = field(init=False)

    def __post_init__(self):
        # 1. Calculate Surface Area
        self.surface_area = (self.span ** 2) / self.aspect_ratio

        # 2. Calculate Mean Chord
        self.chord = self.surface_area / self.span

        # Trigger the specific geometry processing
        self.root_chord, self.tip_chord = self._calculate_geometry()

        # airfoil processing
        coords = self._fetch_coordinates()
        self.airfoil_area_ratio, self.airfoil_circum_ratio = self.analyze_airfoil_ratios(coords)

        #Trigger the specific mass processing
        self.mass = self._calculate_mass()

    def _fetch_coordinates(self):
        """Uses aeropy to generate NACA or load a .dat file."""
        # 1. Check if it's a NACA string
        if self.airfoil_type.lower().startswith("naca"):
            return self._generate_naca_coords(self.airfoil_type)

        # 2. Otherwise, treat as a file path
        if os.path.exists(self.airfoil_type):
            return np.loadtxt(self.airfoil_type, skiprows=1)

        raise FileNotFoundError(f"Airfoil source '{self.airfoil_type}' not found.")

    def _generate_naca_coords(self, naca_code):
        """
        Manually tells XFOIL to generate NACA and save to temp.
        This bypasses the limitations of the aeropy.call function.
        """
        temp_file = f"{naca_code}_temp.dat"
        # XFOIL commands to generate and save
        commands = f"{naca_code}\nsave {temp_file}\ny\nquit\n"

        # We use subprocess here because aeropy.call is optimized for polars,
        # not geometry export.
        subprocess.run([self.config.xfoil_path], input=commands, text=True, capture_output=True)

        data = np.loadtxt(temp_file, skiprows=1)
        if os.path.exists(temp_file):
            os.remove(temp_file)  # Clean up
        return data

    def analyze_airfoil_ratios(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]

        # 1. Calculate Area using the Shoelace Formula (Trapezoidal)
        # This works for any closed polygon
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # 2. Calculate Circumference (Perimeter)
        # Distance between each consecutive point
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx ** 2 + dy ** 2)
        circumference = np.sum(distances)

        # Add distance from last point back to first point to close the loop
        last_dist = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
        circumference += last_dist

        # 3. Calculate Ratios (Assuming chord = 1.0)
        # Square Area = 1*1, Square Perimeter = 4*1
        area_ratio = area / 1.0
        circum_ratio = circumference / 4.0

        return area_ratio, circum_ratio

    def size_spar(self, wing_span, max_v_cruise):
        force = self._force_distribution(max_v_cruise)
        self.spar.spar_dimension(wing_span, force)

    @abstractmethod
    def _calculate_geometry(self) -> tuple[float, float]:
        """Subclasses must implement how root and tip chord are handled. Return root then tip chord"""
        pass

    @abstractmethod
    def _force_distribution(self, v_cruise) -> float:
        pass

    @abstractmethod
    def _calculate_mass(self) -> float:
        """Subclasses must implement how mass of wing is estimated via materials used excluding the spars"""
        pass



@dataclass
class RectangularWing(Wing):
    def _calculate_geometry(self) -> tuple[float, float]:
        return self.chord, self.chord

    def _force_distribution(self, v_cruise) -> float:
        rho = self.config.physics.rho
        CL_max = self.config.aero.cl_max
        force = 0.5 * rho * CL_max * self.surface_area * v_cruise ** 2
        return force

@dataclass
class Fuselage(ABC):
    mass: float = field(init=False)
    wing_ac_from_nose: float = field(init=False)
    tail_ac_from_nose: float = field(init=False)
    wing_ac_to_tail_ac: float = field(init=False)

    @abstractmethod
    def _size_fuselage(self, a, b) -> float:
        """Subclasses must implement how fuselage mass is estimated."""
        pass

@dataclass
class Tail(ABC):
    span_H: float = field(init=False)
    chord_H: float = field(init=False)
    area_H: float = field(init=False)
    AR_H: float = field(init=False)
    span_V: float = field(init=False)
    chord_V: float = field(init=False)
    area_V: float = field(init=False)
    AR_V: float = field(init=False)
    mass: float = field(init=False)

    def __post_init__(self):
        self.span_H, self.chord_H, self.area_H, self.AR_H, self.span_V, self.chord_V, self.area_V, self.AR_V = self._calculate_geometry()

        self.mass = self._calculate_mass()

    @abstractmethod
    def _calculate_geometry(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Subclasses must implement how span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V
        are calculated.

        return it in the order span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V"""
        pass

    @abstractmethod
    def _calculate_mass(self) -> float:
        """Subclasses must implement how mass of tail is estimated via materials used"""
        pass

@dataclass
class Propulsion(ABC):
    motor_power: float
    motor_eff: float = 0.85# electrical -> mechanical
    prop_eff: float = 0.6
    mass_per_W: float = field(repr=False, default=0.0003)# kg/W?
    power_density: float = 3000 # W/kg number of W per kg of motor

    effective_power: float = field(init=False)
    mass: float = field(init=False)

    def __post_init__(self):
        self.effective_power = self._propulsion_effectiveness()
        self.mass = self._size_propulsion()

    @abstractmethod
    def _propulsion_effectiveness(self) -> float:
        """Subclasses must implement what is the effective power of the propulsion"""
        pass

    @abstractmethod
    def _size_propulsion(self) -> float:
        """Subclasses must implement what is the mass of the propulsion"""
        pass

@dataclass
class LandingGear(ABC):
    mass: float = field(init=False)

    def __post_init__(self):
        self.mass = self._size_landing_gear()

    @abstractmethod
    def _size_landing_gear(self) -> float:
        """Subclasses must implement what is the mass of the landing gear"""
        pass

@dataclass
class Avionics(ABC):
    capacity: float
    esc_eff: float = 0.98
    servo_mass: float = 0.015
    num_servo: float = 4
    ESC_mass: float = 0.0403
    # wire_mass_per_m: float = 0.015  # kg/m
    depth_of_discharge: float = 0.8
    volts_p_cell: float = 3.7
    energy_density: float = 100  # Whr/kg number of Whr per kg of lipo battery

    mass_battery: float = field(init=False)
    mass_avionics: float = field(init=False)

    def __post_init__(self):
        self.mass_avionics = self._mass_avionics()
        self.mass_battery = self._mass_battery()

    @abstractmethod
    def _mass_avionics(self):
        """Subclasses must implement what is the mass of the avionics"""
        pass

    @abstractmethod
    def _mass_battery(self):
        """Subclasses must implement what is the mass of the battery used"""
        pass

@dataclass
class Performance(ABC):
    CL_cruise: float = field(init=False)
    V_cruise: float = field(init=False)
    V_stall: float = field(init=False)
    n_turn: float = field(init=False)
    v_turn: float = field(init=False)
    turn_radius: float = field(init=False)
    bank_angle: float = field(init=False)
    n_max: float = field(init=False)
    max_v_turn: float = field(init=False)
    max_bank_angle: float = field(init=False)
    max_turn_radius: float = field(init=False)
    flight_time: float = field(init=False)
    number_of_laps: int = field(init=False)
    L_D_cruise: float = field(init=False)
    stall_margin: float = field(init=False)
    load_factor_margin: float = field(init=False)

    m_total: InitVar[float] = None
    battery_cap: InitVar[float] = None
    wing_area: InitVar[float] = None
    wing_AR: InitVar[float] = None
    depth_of_discharge: InitVar[float] = None
    P_effective: InitVar[float] = None
    power: InitVar[float] = None
    CD0: InitVar[float] = None
    CD_extra: InitVar[float] = 0.0

    #adjust how performance uses CD0, either as a DragParam or added
    config: AircraftConfig = field(default_factory=AircraftConfig, repr=False)

    def __post_init__(self, m_total, battery_cap, wing_area, wing_AR, depth_of_discharge, P_effective, power, CD0, CD_extra):
        self.CL_cruise, self.V_cruise, self.V_stall, self.n_turn, self.v_turn, self.turn_radius, \
            self.bank_angle, self.n_max, self.max_v_turn, self.max_bank_angle, self.max_turn_radius, \
            self.flight_time, self.number_of_laps, self.L_D_cruise, self.stall_margin, self.load_factor_margin = \
            self.analyse_performance(m_total, battery_cap, wing_area, wing_AR, depth_of_discharge, P_effective, power,
                                     CD0, CD_extra)
    @abstractmethod
    def flight_time_one_lap(self, radius, v_straight, v_turn):
        """
        Subclass to calculates time for one lap of the track.
        """
        pass

    def velocity(self, CL, n, W, S):
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
        rho = self.config.physics.rho
        return np.sqrt((2 * n * W) / (rho * S * CL))

    def power_required(self, CL, W, CD0, k, S, CD_extra):
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
        CD = CD0 + CD_extra + k * CL ** 2
        v = self.velocity(CL, 1, W, S)
        return (CD * W / CL) * v

    @abstractmethod
    def find_CL_cruise(self, W, P_effective, CD0, k, S, CL_min, CL_max, CD_banner):
        """
        Subclass to implement how to get turning conditions
        """
        pass

    @abstractmethod
    def find_turning_conditions(self, CL, W, CD0, k, P_effective, S, CD_extra):
        """
        Subclass to implement how to get turning conditions.

        Return
        n_max: maximum multiple of G experienced by the aircraft at inputted CL
        v: Velocity of turn carried out
        angle_rad:  angle of turn with respect to horizontal in radians
        radius: the radius of the turn
        """
        pass

    @abstractmethod
    def analyse_performance(self, m_total, battery_cap, wing_area, wing_AR, depth_of_discharge, P_effective, power,
                               CD0, CD_extra):
        """
        Subclass to implement how to analyse the performance of aircraft.

        Return
        """
        pass
@dataclass
class FlightMission:
    avionics: Avionics
    performance: Performance
    payload: float
    score: float = 0.0
    is_scored: bool = field(init=False, default=False)

    def update_score(self, score: float):
        self.score = score
        self.is_scored = True

@dataclass(kw_only=True)
class Plane(ABC):
    wing: Wing = field(init=False)
    tail: Tail = field(init=False)
    fuselage: Fuselage = field(init=False)
    propulsion: Propulsion = field(init=False)
    #landing gear

    # Grouped Missions
    missions: dict[str, FlightMission] = field(default_factory=dict)

    def print_mission_result(self):
        for name, m in self.missions.items():
            print(f"{name} Payload: {m.payload} kg | Score: {m.score}")
            print(m)

    @abstractmethod
    def check_mass_coherence(self):
        pass

# class StrictConfig(ABC):
