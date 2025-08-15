class Fuselage:

    def __init__(self, wing, density):
        self.int_struc_percentage = 0.1
        self.length = wing.wing_span * 0.85
        self.width = self.length * 0.09
        self.height = self.length * 0.09
        self.wing_ac_from_nose = self.length * 0.15 + wing.chord_root * 0.25
        self.tail_ac_from_nose = self.wing_ac_from_nose + wing.wing_span * 0.45
        self.wing_ac_to_tail_ac = self.tail_ac_from_nose - self.wing_ac_from_nose
        self.mass = self.semi_monocoque_mass(density)


    def semi_monocoque_mass(self, density, fuselage_skin_thickness=0.005):  # Compressed foam + wood structures
        # Calculate skin mass
        front_area = self.width * self.height
        rear_area = front_area
        bottom_area = self.length * self.width
        top_area = bottom_area
        left_area = self.length * self.height
        right_area = left_area
        fuselage_skin_area = front_area + rear_area + bottom_area + top_area + left_area + right_area
        fuselage_skin_volume = fuselage_skin_area * fuselage_skin_thickness
        fuselage_skin_mass = fuselage_skin_volume * density["white foam"]

        # calculate structural mass
        structural_volume = self.length * self.width * self.height * self.int_struc_percentage
        structural_mass = structural_volume * density["basswood"]

        # Sum total mass
        total_mass = fuselage_skin_mass + structural_mass
        return total_mass

    def __str__(self):
        return (f"Fuselage:\n"
                f"  Length: {self.length} m\n"
                f"  Width: {self.width} m\n"
                f"  Height: {self.height} m\n"
                f"  Wing AC from nose: {self.wing_ac_from_nose} m\n"
                f"  Tail AC from nose: {self.tail_ac_from_nose} m\n"
                f"  Wing AC to Tail AC: {self.wing_ac_to_tail_ac} m\n"
                f"  Mass: {self.mass:.3f} kg")
