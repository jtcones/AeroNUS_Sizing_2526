import numpy as np
import itertools
import pandas as pd
from RC_Plane_v3 import RCPlane

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

def mission_2raw(num_passengers, num_cargo, m2_laps, battery_capacity):

    income = (num_passengers * (6 + 2 * m2_laps)) + (num_cargo * (10 + 8 * m2_laps))
    EF = battery_capacity / 100
    cost = m2_laps * (10 + (num_passengers*0.5) + (num_cargo*2)) * EF
    net_income = income - cost
    return net_income
def mission_3raw(banner_length, number_of_laps, wing_span):
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
    return m3
# ----------------------------------------------------------------------
# 2. DISCRETIZED SEARCH SPACE
# ----------------------------------------------------------------------

# Real(0.2, 20, name="m_struct"), #kg
M_STRUCT_VALUES = np.linspace(0.5, 20)

# Real(0.92, 1.52, name="wing_span"), # m
WING_SPAN_VALUES = np.linspace(0.92, 1.52)

# Integer(40, 4000, name="motor_power"), #W
MOTOR_POWER_VALUES = np.linspace(40, 4000, 100)

# Integer(1, 8, name="wing_AR"),
WING_AR_VALUES = [i for i in range(1, 9)]

# Integer(1, 30, name="pucks"),
PUCK_VALUES = [i for i in range(1, 31)]

# Integer(3, 15, name="passenger_cargo_ratio"),
PASSENGER_CARGO_RATIO_VALUES = [i for i in range(3, 16)]

# Integer(1, 100, name="m2_battery"), #Wh
M2_BATTERY_VALUES = np.linspace(1, 100, 100)

# Integer(1, 1000, name="banner_length"), #inch
BANNER_LENGTH_VALUES = np.linspace(1, 1000, 200)

# Integer(1, 5, name="banner_AR"),
BANNER_AR_VALUES = [i for i in range(1, 6)]

# Integer(1, 100, name="m3_battery")
M3_BATTERY_VALUES = np.linspace(1, 100, 100)

# Generate the full grid of parameters
PARAMETER_GRID = itertools.product(
    M_STRUCT_VALUES, WING_SPAN_VALUES, MOTOR_POWER_VALUES, WING_AR_VALUES,
    PUCK_VALUES, PASSENGER_CARGO_RATIO_VALUES, M2_BATTERY_VALUES,
    BANNER_LENGTH_VALUES, BANNER_AR_VALUES, M3_BATTERY_VALUES
)

# ----------------------------------------------------------------------
# 3. GRID SEARCH EXECUTION
# ----------------------------------------------------------------------

RESULTS = []

for i, params in enumerate(PARAMETER_GRID):
    m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio, \
        m2_battery, banner_length, banner_AR, m3_battery = params

    try:
        # Instantiate the plane
        # NOTE: This line requires your RCPlane class to be defined/imported

        # --- Check Constraints ---

        # 1. Mass Coherence (Handled internally by RCPlane __post_init__)
        # If __post_init__ raises ValueError, the design is skipped by the 'except' block.

        # 2. Minimum Laps
        # If __post_init__ raises ValueError, the design is skipped by the 'except' block.
        plane = RCPlane(m_struct, wing_span, motor_power, wing_AR, n_pucks, passenger_cargo_ratio,
                        m2_battery, banner_length, banner_AR, m3_battery)
        # --- Calculate Score ---
        m2_score = mission_2raw(plane.m2.num_ducks, plane.m2.num_pucks, plane.m2.num_laps, m2_battery)
        m3_score = mission_3raw(banner_length, plane.m3.num_laps, wing_span)

        # Store results
        RESULTS.append({
            # --- Score and Performance ---
            'M2 Laps': plane.m2.num_laps,
            'M3 Laps': plane.m3.num_laps,
            'V_cruise M2 (m/s)': plane.m2.V_cruise,
            'V_cruise M3 (m/s)': plane.m3.V_cruise,
            'Turn Rad M2 (m)': plane.m2.turn_radius,
            'Turn Rad M3 (m)': plane.m3.turn_radius,
            'Total Mass M2 (kg)': m_struct + plane.propulsion.mass + plane.m2_payload + plane.avionics.m2_mass_battery,
            'Total Mass M3 (kg)': m_struct + plane.propulsion.mass + plane.m3_payload + plane.avionics.m3_mass_battery,

            # --- Input Parameters ---
            'm_struct (kg)': m_struct,
            'wing_span (m)': wing_span,
            'motor_power (W)': motor_power,
            'wing_AR': wing_AR,
            'n_pucks': n_pucks,
            'P/C_ratio': passenger_cargo_ratio,
            'm2_battery (Wh)': m2_battery,
            'banner_length (in)': banner_length,
            'banner_AR': banner_AR,
            'm3_battery (Wh)': m3_battery,
        })
        # # Keep best 5
        # sorted(RESULTS, key=lambda x: x["score"])
        # RESULTS.pop()

    except ValueError as e:
        # Catch mass deviation or CL_cruise search errors (invalid designs)
        continue
    except RuntimeWarning as e:
        # Catch invalid arccos or other math warnings (invalid designs)
        continue
    except Exception as e:
        # Catch any other unexpected errors and move on
        # print(f"Skipping combination {params} due to error: {e}")
        continue

# ----------------------------------------------------------------------
# 4. OUTPUT RESULTS
# ----------------------------------------------------------------------
df_results = pd.DataFrame(RESULTS)
print("\nGrid Search Complete.")
print(f"Found {len(df_results)} valid designs.")

if not df_results.empty:
    # Sort the DataFrame by the calculated Score
    df_results_sorted = df_results.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_results_sorted.index += 1  # Start index at 1 for Rank

    print("\n--- Top 5 Configurations by Score ---")
    # Display the most relevant columns for ranking
    display_cols = ['Score', 'M2 Laps', 'M3 Laps', 'm_struct (kg)', 'motor_power (W)',
                    'wing_span (m)', 'wing_AR', 'n_pucks', 'm2_battery (Wh)', 'm3_battery (Wh)']
    print(df_results_sorted[display_cols].head())

    # Save the full DataFrame (optional, but highly recommended)
    # df_results_sorted.to_csv('rc_plane_grid_search_results.csv', index_label='Rank')
    # print("\nFull results saved to 'rc_plane_grid_search_results.csv'")
else:
    print("No valid designs were found within the specified grid and constraints.")