"""Ultimate Tensile Strengths (Pa).

All values are conservative design numbers — tune to your own supplier
datasheets where possible. Used by the spar sizing routine in
``Components.Spar.spar_dimension``.
"""

carbon_spar = 8.0e8             # ~800 MPa (typical along fibres; tune to your spec)
fibreglass_laminate = 6.0e8     # ~600 MPa (order-of-magnitude)
basswood = 6.0e7                # ~60 MPa
