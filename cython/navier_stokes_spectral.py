import numpy as np
import matplotlib.pyplot as plt
import build.navier_stokes_spectral_c as ns_c
from utils.timings import wtime

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


# @wtime #NOTE: timing wrapper
def run_navier_stokes():
    return ns_c.main()


def main():
    """Navier-Stokes Simulation"""

    result = run_navier_stokes()


if __name__ == "__main__":
    main()
