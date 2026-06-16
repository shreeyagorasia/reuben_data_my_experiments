"""
models/chapman_richards/model.py
==================================
The Chapman-Richards growth curve formula.

This is also reused by models/pinn_baseline/model.py as a "physics
prior" (the PINN's predicted growth rate is compared against this
curve's derivative).
"""

import numpy as np


def chapman_richards(t, y_max, k, p):
    """Chapman-Richards growth curve.

        H(t) = y_max * (1 - exp(-k * t)) ** p

    t      : age (years)
    y_max  : asymptotic maximum height
    k, p   : shape parameters
    """
    return y_max * (1 - np.exp(-k * t)) ** p
