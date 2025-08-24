✈️ NUS AeroNUS Plane Sizing Program

This repository contains a sizing program for the NUS AeroNUS team, developed for the AIAA DBF 2025/2026 competition.

The program helps automate aircraft sizing decisions by combining:
Fixed parameters (fixed_params/)
Material properties (material_properties/)
Mission-specific requirements

📂 Repository Structure
.
├── .idea/                   # IDE-specific files (ignore)
├── Training_Plane_2/        # Deprecated approach (use v1 instead)
├── Training_Plane_2_v1/     # Active development version
├── fixed_params/            # Fixed design parameters (constants, configs)
├── material_properties/     # Material databases and properties
├── 2009_dbf_rules.pdf       # Reference DBF rules (2009 baseline)
└── README.md                # This file
⚠️ Note: Training_Plane_2 is deprecated. Please use Training_Plane_2_v1 together with fixed_params/ and material_properties/.

Mission Profile
For the training plane, two representative missions are modeled:
Mission 1: Fly 2 laps with no payload, focus on minimizing flight time.
Mission 2: Fly 4 laps with payload (0.6 kg for current revision), no penalty on timing.
Scoring is based on:
System Complexity Factor (SCF) → favors ease of assembly
Relative Flight Time → favors planes optimized for their weight class

🚀 Getting Started
Clone this repository:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Use the Training_Plane_2_v1/ folder as the main entry point.
Modify parameters in fixed_params/ and material_properties/ for your design iteration.

📌 Notes
Repository is still under active development.
Contributions from team members are welcome – especially for refining aerodynamic models and adding mission scoring functions.
