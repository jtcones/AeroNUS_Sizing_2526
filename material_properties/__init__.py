"""Material property constants used by the performance model.

Submodules
----------
areal_mass      : kg/m^2 — skin, wrap, tape, ripstop
density         : kg/m^3 — foams, basswood, spar, air
UTS             : Pa     — ultimate tensile strengths
Young_Modulus   : Pa     — Young's moduli

Notes
-----
These constants should be updated as the team characterises new materials
or switches suppliers. Any change here propagates through the performance
model (see Components.py and AIAA_DBF_2526.py).
"""
from . import areal_mass
from . import density
from . import UTS
from . import Young_Modulus
