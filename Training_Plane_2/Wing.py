area_constant = {
    "clark Y": 0.0809931
}

circumference_constant = {
    "clark Y": 0.5111733
}
class Wing:
    def __init__(self, wing_details, m_total):
        self.airfoil_type = wing_details["airfoil_type"]
        self.wing_span = wing_details["wing_span"]
        self.taper_ratio = wing_details["taper_ratio"]
        self.wing_loading = wing_details["wing_loading"] # 4.4 to 5.5 kg/m^2
        self.surface_area = m_total / self.wing_loading
        self.aspect_ratio = self.wing_span**2 / self.surface_area
        self.chord_tip = (self.surface_area / self.wing_span) # if taper_ratio == 1 else
        self.chord_root = self.chord_tip  # if taper_ratio == 1 else
        self.chord_mac = self.chord_root * (2/3) * ((1 + self.taper_ratio + self.taper_ratio**2)/(1 + self.taper_ratio))
        self.wing_volume = self.calc_wing_volume(self.airfoil_type)
        self.wrap_area = self.calc_total_surface_area(self.airfoil_type)

    def calc_wing_volume(self, wing_type):
        volume = area_constant[wing_type] * self.chord_tip**2 * self.wing_span
        return volume

    def calc_total_surface_area(self, wing_type):
        area = circumference_constant[wing_type] * self.chord_tip*4 * self.wing_span
        return area

    def __str__(self):
        return (f"Wing:\n"
                f"  Airfoil: {self.airfoil_type}\n"
                f"  Span: {self.wing_span:.3f} m\n"
                f"  Aspect Ratio: {self.aspect_ratio:.3f}\n"
                f"  Surface Area: {self.surface_area:.3f} m²\n"
                f"  Chord (root/tip): {self.chord_root:.3f} m / {self.chord_tip:.3f} m\n"
                f"  Mean Aerodynamic Chord: {self.chord_mac:.3f} m")