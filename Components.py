"""Abstract base classes for the reusable RC-UAV performance model.

This module defines the generic building blocks of a fixed-wing RC
aircraft (wing, fuselage, tail, propulsion, avionics, landing gear,
performance, plane) as a hierarchy of ``@dataclass`` ABCs. Concrete,
competition-specific subclasses live in ``AIAA_DBF_2526.py``.

The split is deliberate:

* **Components.py** contains everything *any* fixed-wing RC UAV needs —
  generic geometry, spar sizing, aerodynamic/performance interfaces,
  and the mission-container abstraction. It is intended to be reusable
  across future AIAA DBF rule sets and other projects.
* **AIAA_DBF_2526.py** contains the concrete implementations that plug
  the 2025/26 rules, materials, and configuration choices into those
  abstract interfaces.

External dependencies
---------------------
* ``AeroPy`` and ``xfoil.exe``: only required if you use NACA airfoils
  via ``Wing._generate_naca_coords``. If you use a ``.dat`` file
  (as the shipped Clark Y configuration does), these are not called.
* ``material_properties``: local package providing density, UTS, etc.

See Also
--------
AIAA_DBF_2526.py : Concrete subclasses implementing the 2025/26 aircraft.
README.md        : Repository overview and setup instructions.
"""

from dataclasses import dataclass, field, InitVar
from material_properties import UTS, density
from abc import ABC, abstractmethod
import numpy as np
import os
import subprocess

# Note: AeroPy / XFOIL are imported lazily inside ``Wing._generate_naca_coords``
# so that this module can be used with plain ``.dat`` airfoils without
# requiring the AeroPy package or an ``xfoil`` binary to be installed.
# See README.md ("Optional: XFOIL integration") for setup notes.


# =====================================================================
# Environment / Configuration dataclasses
# =====================================================================

@dataclass(frozen=True)
class PhysicsEnv:
    """Physical environment constants.

    Attributes
    ----------
    g : float
        Gravitational acceleration (m/s^2).
    rho : float
        Air density (kg/m^3). ISA sea level.
    e : float
        Oswald efficiency factor used in the induced-drag term
        ``k = 1 / (pi * AR * e)``. 0.8 is a reasonable default for
        small rectangular-wing UAVs (Raymer, Ch. 12).
    """
    g: float = 9.81
    rho: float = 1.225
    e: float = 0.8


@dataclass(frozen=True)
class AeroLimits:
    """Aerodynamic operating envelope (lift coefficients).

    Attributes
    ----------
    cl_min : float
        Lowest usable cruise CL. Prevents the CL search from drifting
        into unrealistically fast, low-lift flight states.
    cl_max : float
        Maximum usable lift coefficient. Corresponds to the stall
        limit of the chosen airfoil (Clark Y ~ 1.5).
    cl_turn : float
        Nominal CL used when evaluating "standard" turn performance
        (``find_turning_conditions``). Distinct from ``cl_max`` so
        that turn-radius reporting uses a realistic steady value
        rather than the on-stall limit.
    """
    cl_min: float = 0.2
    cl_max: float = 1.5
    cl_turn: float = 1.0


@dataclass(frozen=True)
class DragParams:
    """Parasite drag parameters.

    Attributes
    ----------
    cd00 : float
        Base zero-lift drag coefficient (clean airframe, no payload).
    cdp : float
        Payload drag scale factor (currently unused in performance
        routines; reserved for future per-payload drag modelling).
    """
    cd00: float = 0.17
    cdp: float = 1.0


@dataclass(frozen=True)
class FlightRouteParams:
    """Course geometry for the DBF flight path.

    Attributes
    ----------
    straight_distance : float
        Total straight-line distance per lap (m). The DBF 2025/26
        course is 500 ft per straight segment — two runway straights
        plus two 'crossings' equivalent to 4 x 500 ft = 609.6 m.
    """
    straight_distance: float = 152.4 * 4  # 500 ft * 4


@dataclass
class AircraftConfig:
    """Master configuration bundle passed into downstream components.

    Groups physics, aerodynamics, drag, and route parameters so every
    component references the same environmental assumptions.

    Attributes
    ----------
    physics : PhysicsEnv
    aero : AeroLimits
    drag : DragParams
    route : FlightRouteParams
    xfoil_path : str
        Filesystem path to the XFOIL executable. Only used when a
        NACA airfoil is requested via ``Wing._generate_naca_coords``.
        Defaults to ``"../xfoil.exe"`` for the repo layout; override
        per-platform as needed.
    """
    physics: PhysicsEnv = field(default_factory=PhysicsEnv)
    aero: AeroLimits = field(default_factory=AeroLimits)
    drag: DragParams = field(default_factory=DragParams)
    route: FlightRouteParams = field(default_factory=FlightRouteParams)
    xfoil_path: str = r"../xfoil.exe"


# =====================================================================
# Structural components
# =====================================================================

@dataclass
class Spar:
    """Main wing spar — sized by peak bending stress.

    The spar is treated as a hollow circular tube running along the
    full span. For each half-wing (panel), the model assumes a
    uniformly distributed lift resulting in a peak bending moment at
    the root of ``M = F_panel * L_panel / 2``. The outer radius is
    increased in 0.5 mm steps until bending stress falls below
    ``UTS / safety_factor``.

    Attributes
    ----------
    material : str
        Label only — material strength is pulled from
        ``material_properties.UTS.carbon_spar``.
    outer_diameter : float
        Outer radius (m) after sizing. The name is historical —
        geometrically this is the outer *radius*.
    wall_thickness : float
        Tube wall thickness (m). Fixed input; defaults to 2 mm.
    mass : float
        Computed spar mass (kg). 0 until :meth:`spar_dimension` is called.
    is_sized : bool
        True once :meth:`spar_dimension` has run.
    """
    material: str = "Carbon Fiber"

    outer_diameter: float = 0.0
    wall_thickness: float = 0.002
    mass: float = field(init=False, default=0.0)
    is_sized: bool = field(init=False, default=False)

    def spar_dimension(self, wing_span, force, safety_factor=2):
        """Size the spar to withstand peak wing-root bending stress.

        Parameters
        ----------
        wing_span : float
            Full wing span (m), tip-to-tip.
        force : float
            Total lift force on both wings (N) at the sizing
            condition. Typically ``0.5 * rho * CL_max * S * V^2``
            evaluated at the highest cruise speed across all missions.
        safety_factor : float, optional
            Divides the material UTS. Default is 2.

        Notes
        -----
        Modifies ``outer_diameter``, ``mass``, and ``is_sized``
        in place. The starting guess for the outer radius is 2.5 mm
        and grows in 0.5 mm steps until the stress criterion is met.
        """
        uts_with_safety_factor = UTS.carbon_spar / safety_factor

        # Each half-wing takes half of the total lift and has half the span.
        panel_length = wing_span / 2
        panel_force = force / 2

        # Peak bending moment at the root of a uniformly-loaded
        # cantilever of length L with total load P is P*L/2.
        M = panel_force * panel_length / 2

        rad_o = 0.0025  # 2.5 mm starting guess for outer radius
        while True:
            r_i = rad_o - self.wall_thickness
            I = np.pi * (rad_o ** 4 - r_i ** 4) / 4
            sigma = M * rad_o / I
            if sigma <= uts_with_safety_factor:
                break
            rad_o += 0.0005  # increment 0.5 mm

        self.outer_diameter = rad_o
        self.mass = (
            np.pi * wing_span
            * (rad_o ** 2 - (rad_o - self.wall_thickness) ** 2)
            * density.carbon_spar
        )
        self.is_sized = True


# =====================================================================
# Wing (abstract)
# =====================================================================

@dataclass
class Wing(ABC):
    """Abstract wing component.

    Subclasses provide planform geometry (``_calculate_geometry``),
    structural mass (``_calculate_mass``), and a root-lift
    distribution for spar sizing (``_force_distribution``). The base
    class handles the reference-area/chord arithmetic and loads the
    airfoil coordinates.

    Parameters (init)
    -----------------
    airfoil_type : str
        Either a ``.dat`` file path (SELIG-style, first line is a
        title) or a string like ``"naca2412"`` — in the latter case
        XFOIL is invoked to generate coordinates on the fly.
    span : float
        Full tip-to-tip span (m).
    aspect_ratio : float
        Wing aspect ratio ``b^2 / S``.
    spar : Spar, optional
        Spar instance. Defaults to an unsized ``Spar()``.
    config : AircraftConfig, optional
        Environment / limits bundle. Defaults to ``AircraftConfig()``.

    Computed attributes
    -------------------
    surface_area, root_chord, tip_chord, chord, mass,
    airfoil_area_ratio, airfoil_circum_ratio
        All set in :meth:`__post_init__`.
    """
    airfoil_type: str      # .dat file path, or a NACA designation like "naca2412"
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
        # 1. Reference area and mean chord from span and AR.
        self.surface_area = (self.span ** 2) / self.aspect_ratio
        self.chord = self.surface_area / self.span

        # 2. Subclass-specific planform (e.g. rectangular vs tapered).
        self.root_chord, self.tip_chord = self._calculate_geometry()

        # 3. Airfoil geometry analysis for foam-volume and skin-area calcs.
        coords = self._fetch_coordinates()
        self.airfoil_area_ratio, self.airfoil_circum_ratio = self.analyze_airfoil_ratios(coords)

        # 4. Subclass-specific mass estimate.
        self.mass = self._calculate_mass()

    def _fetch_coordinates(self):
        """Return airfoil coordinates as an (N, 2) array.

        Routes to either NACA generation (XFOIL subprocess) or a
        ``.dat`` file loader depending on ``airfoil_type``.

        Raises
        ------
        FileNotFoundError
            If ``airfoil_type`` is not a NACA code and no matching
            ``.dat`` file is found on disk.
        """
        if self.airfoil_type.lower().startswith("naca"):
            return self._generate_naca_coords(self.airfoil_type)

        if os.path.exists(self.airfoil_type):
            return np.loadtxt(self.airfoil_type, skiprows=1)

        raise FileNotFoundError(f"Airfoil source '{self.airfoil_type}' not found.")

    def _generate_naca_coords(self, naca_code):
        """Shell out to XFOIL to export a NACA airfoil's geometry.

        Imports ``AeroPy.aeropy.xfoil_module`` lazily so that projects
        using a ``.dat`` file (the default in this repo) do not need
        AeroPy or a built XFOIL binary installed. The import is
        retained to preserve the historical coupling — future work
        may swap the raw subprocess call for AeroPy's wrappers.

        The AeroPy wrapper is geared toward polar analysis rather
        than geometry export, so a direct subprocess call is used
        here. The intermediate temp file is deleted on success.

        Parameters
        ----------
        naca_code : str
            e.g. ``"naca2412"``.

        Returns
        -------
        np.ndarray
            Airfoil coordinates, shape (N, 2).

        Raises
        ------
        ImportError
            If AeroPy is not installed on this machine.
        FileNotFoundError
            If ``self.config.xfoil_path`` does not resolve.
        """
        # Lazy import — only required when a NACA airfoil is actually used.
        from AeroPy.aeropy import xfoil_module  # noqa: F401

        temp_file = f"{naca_code}_temp.dat"
        commands = f"{naca_code}\nsave {temp_file}\ny\nquit\n"
        subprocess.run([self.config.xfoil_path], input=commands, text=True, capture_output=True)

        data = np.loadtxt(temp_file, skiprows=1)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return data

    def analyze_airfoil_ratios(self, coords):
        """Compute unit-chord area and perimeter ratios of the airfoil.

        The area is returned relative to a unit square (1x1 m^2)
        and the perimeter relative to the unit square's 4 m perimeter.
        Both are dimensionless and scale cleanly with actual chord.

        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates (N, 2), ordered around the outline.

        Returns
        -------
        (area_ratio, circum_ratio) : tuple[float, float]
        """
        x = coords[:, 0]
        y = coords[:, 1]

        # Shoelace formula — works for any simple closed polygon.
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Perimeter by summing segment lengths.
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx ** 2 + dy ** 2)
        circumference = np.sum(distances)

        # Close the loop: last point back to first.
        last_dist = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
        circumference += last_dist

        area_ratio = area / 1.0      # unit square area = 1
        circum_ratio = circumference / 4.0  # unit square perimeter = 4

        return area_ratio, circum_ratio

    def size_spar(self, wing_span, max_v_cruise):
        """Size the main spar for the worst-case cruise condition.

        Parameters
        ----------
        wing_span : float
            Full wing span (m). Passed through rather than re-reading
            ``self.span`` so that callers can size using a modified span.
        max_v_cruise : float
            Largest cruise velocity across all missions (m/s). Used
            as the reference speed for the lift-force distribution.
        """
        force = self._force_distribution(max_v_cruise)
        self.spar.spar_dimension(wing_span, force)

    @abstractmethod
    def _calculate_geometry(self) -> tuple[float, float]:
        """Return (root_chord, tip_chord) for this planform (m)."""
        pass

    @abstractmethod
    def _force_distribution(self, v_cruise) -> float:
        """Return total lift force (N) used to size the spar.

        Parameters
        ----------
        v_cruise : float
            Reference speed (m/s).
        """
        pass

    @abstractmethod
    def _calculate_mass(self) -> float:
        """Return wing mass (kg) excluding the spar."""
        pass


@dataclass
class RectangularWing(Wing):
    """Constant-chord rectangular planform.

    Both root and tip chords equal the mean chord. Spar sizing
    assumes the wing is at CL_max during the reference speed.
    """

    def _calculate_geometry(self) -> tuple[float, float]:
        return self.chord, self.chord

    def _force_distribution(self, v_cruise) -> float:
        """Total wing lift at CL_max and the given cruise speed."""
        rho = self.config.physics.rho
        CL_max = self.config.aero.cl_max
        force = 0.5 * rho * CL_max * self.surface_area * v_cruise ** 2
        return force


# =====================================================================
# Fuselage & Tail & Propulsion & Landing Gear & Avionics (abstract)
# =====================================================================

@dataclass
class Fuselage(ABC):
    """Abstract fuselage.

    Subclasses set ``mass``, ``wing_ac_from_nose``,
    ``tail_ac_from_nose``, and ``wing_ac_to_tail_ac`` in their
    ``__post_init__`` using whatever sizing law they adopt.
    """
    mass: float = field(init=False)
    wing_ac_from_nose: float = field(init=False)
    tail_ac_from_nose: float = field(init=False)
    wing_ac_to_tail_ac: float = field(init=False)

    @abstractmethod
    def _size_fuselage(self, a, b) -> float:
        """Return fuselage mass (kg)."""
        pass


@dataclass
class Tail(ABC):
    """Abstract empennage.

    Subclasses must populate the full set of ``span_H``, ``chord_H``,
    ``area_H``, ``AR_H`` (horizontal tail) and the matching vertical
    tail fields, plus ``mass``.
    """
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
        self.span_H, self.chord_H, self.area_H, self.AR_H, \
            self.span_V, self.chord_V, self.area_V, self.AR_V = self._calculate_geometry()
        self.mass = self._calculate_mass()

    @abstractmethod
    def _calculate_geometry(self) -> tuple[float, float, float, float, float, float, float, float]:
        """Return (span_H, chord_H, area_H, AR_H, span_V, chord_V, area_V, AR_V)."""
        pass

    @abstractmethod
    def _calculate_mass(self) -> float:
        """Return tail mass (kg)."""
        pass


@dataclass
class Propulsion(ABC):
    """Abstract electric propulsion unit.

    Attributes
    ----------
    motor_power : float
        Rated electrical input power to the motor (W).
    motor_eff : float
        Electrical-to-mechanical efficiency of the motor.
    prop_eff : float
        Propeller efficiency (thrust power / shaft power).
    mass_per_W : float
        Legacy sizing constant (kg/W). Retained for backwards
        compatibility; concrete subclasses may prefer
        ``power_density`` instead.
    power_density : float
        Motor power density (W/kg). Used by
        ``SinglePropellerMotor._size_propulsion`` to give
        ``mass = motor_power / power_density``.
    """
    motor_power: float
    motor_eff: float = 0.85         # electrical -> mechanical
    prop_eff: float = 0.6
    mass_per_W: float = field(repr=False, default=0.0003)  # kg/W
    power_density: float = 3000     # W/kg

    effective_power: float = field(init=False)
    mass: float = field(init=False)

    def __post_init__(self):
        self.effective_power = self._propulsion_effectiveness()
        self.mass = self._size_propulsion()

    @abstractmethod
    def _propulsion_effectiveness(self) -> float:
        """Return effective (useful) thrust-power output (W)."""
        pass

    @abstractmethod
    def _size_propulsion(self) -> float:
        """Return motor + propeller mass (kg)."""
        pass


@dataclass
class LandingGear(ABC):
    """Abstract landing gear stub.

    Present for completeness; not instantiated by the shipped
    2025/26 RCPlane. Future iterations can hook concrete gear
    masses and drag contributions in here.
    """
    mass: float = field(init=False)

    def __post_init__(self):
        self.mass = self._size_landing_gear()

    @abstractmethod
    def _size_landing_gear(self) -> float:
        """Return landing gear mass (kg)."""
        pass


@dataclass
class Avionics(ABC):
    """Abstract avionics & battery pack.

    Bundles servos + ESC + battery. Subclasses define how
    ``mass_avionics`` (servos + ESC + harness) and ``mass_battery``
    (battery pack alone) are computed.

    Parameters
    ----------
    capacity : float
        Battery capacity (Wh).
    esc_eff : float
        ESC efficiency (electrical). Applied in propulsion, not here.
    servo_mass : float
        Mass of a single servo (kg).
    num_servo : float
        Number of servos on the aircraft.
    ESC_mass : float
        ESC mass (kg).
    depth_of_discharge : float
        Usable fraction of battery capacity at landing.
    volts_p_cell : float
        Nominal voltage per LiPo cell (V).
    energy_density : float
        Battery-level energy density (Wh/kg) used to back out pack
        mass from capacity.
    """
    capacity: float
    esc_eff: float = 0.98
    servo_mass: float = 0.015
    num_servo: float = 4
    ESC_mass: float = 0.0403
    depth_of_discharge: float = 0.8
    volts_p_cell: float = 3.7
    energy_density: float = 100     # Wh/kg for LiPo

    mass_battery: float = field(init=False)
    mass_avionics: float = field(init=False)

    def __post_init__(self):
        self.mass_avionics = self._mass_avionics()
        self.mass_battery = self._mass_battery()

    @abstractmethod
    def _mass_avionics(self):
        """Return mass (kg) of servos + ESC + harness, excluding battery."""
        pass

    @abstractmethod
    def _mass_battery(self):
        """Return battery-pack mass (kg) alone."""
        pass


# =====================================================================
# Performance (abstract)
# =====================================================================

@dataclass
class Performance(ABC):
    """Abstract per-mission performance analyser.

    Subclasses implement ``analyse_performance`` to populate cruise
    conditions, turn conditions, flight time, and lap count. The
    generic helpers ``velocity`` and ``power_required`` are provided
    as concrete methods because they do not depend on the subclass.

    Call pattern
    ------------
    Instances are created with all required inputs as ``InitVar``
    arguments, which :meth:`__post_init__` forwards to
    :meth:`analyse_performance` to fill in every public attribute.
    """
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

    config: AircraftConfig = field(default_factory=AircraftConfig, repr=False)

    def __post_init__(self, m_total, battery_cap, wing_area, wing_AR, depth_of_discharge,
                      P_effective, power, CD0, CD_extra):
        (self.CL_cruise, self.V_cruise, self.V_stall, self.n_turn, self.v_turn,
         self.turn_radius, self.bank_angle, self.n_max, self.max_v_turn,
         self.max_bank_angle, self.max_turn_radius, self.flight_time,
         self.number_of_laps, self.L_D_cruise, self.stall_margin,
         self.load_factor_margin) = \
            self.analyse_performance(m_total, battery_cap, wing_area, wing_AR,
                                     depth_of_discharge, P_effective, power, CD0, CD_extra)

    @abstractmethod
    def flight_time_one_lap(self, radius, v_straight, v_turn):
        """Return time (s) to fly one lap of the DBF course."""
        pass

    def velocity(self, CL, n, W, S):
        """Flight velocity from steady-state lift balance.

        Parameters
        ----------
        CL : float
            Lift coefficient.
        n : float
            Load factor (1.0 for level flight).
        W : float
            Weight (N).
        S : float
            Wing reference area (m^2).

        Returns
        -------
        float
            Velocity (m/s).
        """
        rho = self.config.physics.rho
        return np.sqrt((2 * n * W) / (rho * S * CL))

    def power_required(self, CL, W, CD0, k, S, CD_extra):
        """Power required for straight-and-level flight at the given CL.

        Parameters
        ----------
        CL : float
        W : float
            Weight (N).
        CD0 : float
            Zero-lift drag coefficient (clean airframe).
        k : float
            Induced-drag factor, ``1 / (pi * AR * e)``.
        S : float
            Wing area (m^2).
        CD_extra : float
            Additional parasite drag term (e.g. banner drag in M3).

        Returns
        -------
        float
            Required power (W).
        """
        CD = CD0 + CD_extra + k * CL ** 2
        v = self.velocity(CL, 1, W, S)
        return (CD * W / CL) * v

    @abstractmethod
    def find_CL_cruise(self, W, P_effective, CD0, k, S, CL_min, CL_max, CD_banner):
        """Return CL that balances ``P_effective == P_required``."""
        pass

    @abstractmethod
    def find_turning_conditions(self, CL, W, CD0, k, P_effective, S, CD_extra):
        """Return (n_max, v, angle_rad, radius) for a sustained turn at CL."""
        pass

    @abstractmethod
    def analyse_performance(self, m_total, battery_cap, wing_area, wing_AR,
                            depth_of_discharge, P_effective, power, CD0, CD_extra):
        """Compute every public performance attribute and return as a tuple.

        Returns
        -------
        tuple
            ``(CL_cruise, V_cruise, V_stall, n_turn, v_turn, turn_radius,
            bank_angle, n_max, max_v_turn, max_bank_angle, max_turn_radius,
            flight_time, number_of_laps, L_D_cruise, stall_margin,
            load_factor_margin)``
        """
        pass


# =====================================================================
# Mission & Plane containers
# =====================================================================

@dataclass
class FlightMission:
    """A single competition mission: payload + performance + score.

    Attributes
    ----------
    avionics : Avionics
        Mission-specific battery/avionics pack.
    performance : Performance
        Mission-specific performance analysis.
    payload : float
        Payload mass for this mission (kg).
    score : float
        Mission score. Set via :meth:`update_score`.
    is_scored : bool
        True once :meth:`update_score` has been called.
    """
    avionics: Avionics
    performance: Performance
    payload: float
    score: float = 0.0
    is_scored: bool = field(init=False, default=False)

    def update_score(self, score: float):
        """Record the final mission score and mark it as scored."""
        self.score = score
        self.is_scored = True


@dataclass(kw_only=True)
class Plane(ABC):
    """Abstract top-level aircraft container.

    Holds the structural components and a dictionary of missions.
    Concrete subclasses (e.g. ``RCPlane``) wire design variables to
    component constructors inside ``__post_init__``.
    """
    wing: Wing = field(init=False)
    tail: Tail = field(init=False)
    fuselage: Fuselage = field(init=False)
    propulsion: Propulsion = field(init=False)
    # landing gear reserved for future use

    missions: dict[str, FlightMission] = field(default_factory=dict)

    def print_mission_result(self):
        """Print one line per mission — payload + final score."""
        for name, m in self.missions.items():
            print(f"{name} Payload: {m.payload} kg | Score: {m.score}")
            print(m)

    @abstractmethod
    def check_mass_coherence(self):
        """Verify that the guessed structural mass matches the sum of component masses."""
        pass
