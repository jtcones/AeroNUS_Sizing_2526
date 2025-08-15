class Propulsion:
    def __init__(self):
        self.motor_eff = 0.85 #electrical -> mechanical
        self.esc_eff = 0.98
        self.prop_eff = 0.6
        self.motor_rated_power = 400
        # self.max_watt_hour = 100
        self.overall_propulsion_efficiency = self.motor_eff * self.esc_eff * self.prop_eff
        self.mass = self.motor_rated_power * 0.0003
        self.effective_power = self.motor_rated_power*self.overall_propulsion_efficiency

    def __str__(self):
        return (f"Propulsion:\n"
                f"  Rated Power: {self.motor_rated_power} W\n"
                f"  Efficiency (motor/ESC/prop): {self.motor_eff}/{self.esc_eff}/{self.prop_eff}\n"
                f"  Overall Efficiency: {self.overall_propulsion_efficiency:.3f}\n"
                f"  Effective Power: {self.effective_power:.2f} W\n"
                f"  Mass: {self.mass:.3f} kg")