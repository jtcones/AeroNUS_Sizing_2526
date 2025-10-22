class Tail:
    def __init__(self, tail_details, wing, moment_arm):
        # self.mass
        self.tail_coefficient_H = tail_details["tail_coefficient_H"]
        self.tail_coefficient_V = tail_details["tail_coefficient_V"]
        self.surface_area_H = self.tail_coefficient_H * wing.chord_mac * wing.surface_area / moment_arm
        self.surface_area_V = self.tail_coefficient_V * wing.wing_span * wing.surface_area / moment_arm

        self.taper_ratio = tail_details["taper_ratio"]
        self.aspect_ratio_H = wing.aspect_ratio / 2
        self.aspect_ratio_V = self.aspect_ratio_H
        self.wing_span_H = (self.surface_area_H * self.aspect_ratio_H)**(0.5)
        self.chord_root_H = self.surface_area_H / (self.wing_span_H / 2 * (1 + self.taper_ratio))

        self.wing_span_V = (self.surface_area_V * self.aspect_ratio_V)**(0.5)
        self.chord_root_V = self.surface_area_V / (self.wing_span_V / 2 * (1 + self.taper_ratio))
        #self.wing_loading

    def __str__(self):
        return (f"Tail:\n"
                f"  Horizontal Tail Area: {self.surface_area_H:.3f} m², Span: {self.wing_span_H:.3f} m, Root Chord: {self.chord_root_H:.3f} m\n"
                f"  Vertical Tail Area: {self.surface_area_V:.3f} m², Span: {self.wing_span_V:.3f} m, Root Chord: {self.chord_root_V:.3f} m")