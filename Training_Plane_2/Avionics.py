from math import ceil
class Avionics:

    def __init__(self, n_cells, p_motor, total_time_seconds):
        self.volts_p_cell = 3.7
        self.n_cells = n_cells
        self.mass_per_6s_2800 = 0.38 #Tattu R-Line 750mAh 14.8V 95C 4S1P Lipo Battery Pack With XT30 Plug
        self.current = p_motor / (self.n_cells * self.volts_p_cell)
        self.depth_of_discharge = 0.8
        self.current_mA = self.current * 1000
        self.capacity = (total_time_seconds/3600 * self.current_mA) / self.depth_of_discharge
        self.num_battery = ceil(self.capacity/2800)
        self.servo_mass = 0.015
        self.num_servo = 5
        self.ESC_mass = 0.0403
        self.wire_mass_per_m = 0.015  # kg/m
        self.mass = self.mass_per_6s_2800 * self.num_battery + self.servo_mass * self.num_servo + self.ESC_mass

    def __str__(self):
        return (f"Avionics:\n"
                f"  Cells: {self.n_cells} ({self.volts_p_cell} V each)\n"
                f"  Current Draw: {self.current:.2f} A\n"
                f"  Required Capacity: {self.capacity:.0f} mAh\n"
                f"  Batteries Needed: {self.num_battery}\n"
                f"  Total Mass: {self.mass:.3f} kg")


