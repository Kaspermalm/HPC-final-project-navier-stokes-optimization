# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
import matplotlib.pyplot as plt

import pyfftw
pyfftw.interfaces.cache.enable()

cimport numpy as cnp

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


cdef inline cnp.ndarray[cnp.float64_t, ndim=2] poisson_solve(
    cnp.ndarray[cnp.float64_t, ndim=2] rho,
    cnp.ndarray[cnp.float64_t, ndim=2] kSq_inv,
):
    """solve the Poisson equation, given source field rho"""

    cdef cnp.ndarray[cnp.complex128_t, ndim=2] V_hat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] V

    V_hat = -(fft.fftn(rho)) * kSq_inv
    V = np.real(fft.ifftn(V_hat))
    return V


cdef inline cnp.ndarray[cnp.float64_t, ndim=2] diffusion_solve(
    cnp.ndarray[cnp.float64_t, ndim=2] v,
    double dt,
    double nu,
    cnp.ndarray[cnp.float64_t, ndim=2] kSq,
):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v_new

    v_hat = (fft.fftn(v)) / (1.0 + dt * nu * kSq)
    v_new = np.real(fft.ifftn(v_hat))
    return v_new


cdef inline tuple grad(
    cnp.ndarray[cnp.float64_t, ndim=2] v,
    cnp.ndarray[cnp.float64_t, ndim=2] kx,
    cnp.ndarray[cnp.float64_t, ndim=2] ky,
):
    """return gradient of v"""

    cdef cnp.ndarray[cnp.complex128_t, ndim=2] v_hat
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx, dvy

    v_hat = fft.fftn(v)
    dvx = np.real(fft.ifftn(1j * kx * v_hat))
    dvy = np.real(fft.ifftn(1j * ky * v_hat))
    return dvx, dvy


cdef inline cnp.ndarray[cnp.float64_t, ndim=2] div(
    cnp.ndarray[cnp.float64_t, ndim=2] vx,
    cnp.ndarray[cnp.float64_t, ndim=2] vy,
    cnp.ndarray[cnp.float64_t, ndim=2] kx,
    cnp.ndarray[cnp.float64_t, ndim=2] ky,
):
    """return divergence of (vx,vy)"""

    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx_x, dvy_y

    dvx_x = np.real(fft.ifftn(1j * kx * fft.fftn(vx)))
    dvy_y = np.real(fft.ifftn(1j * ky * fft.fftn(vy)))
    return dvx_x + dvy_y


cdef inline cnp.ndarray[cnp.float64_t, ndim=2] curl(
    cnp.ndarray[cnp.float64_t, ndim=2] vx,
    cnp.ndarray[cnp.float64_t, ndim=2] vy,
    cnp.ndarray[cnp.float64_t, ndim=2] kx,
    cnp.ndarray[cnp.float64_t, ndim=2] ky,
):
    """return curl of (vx,vy)"""

    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx_y, dvy_x

    dvx_y = np.real(fft.ifftn(1j * ky * fft.fftn(vx)))
    dvy_x = np.real(fft.ifftn(1j * kx * fft.fftn(vy)))
    return dvy_x - dvx_y


cdef inline cnp.ndarray[cnp.float64_t, ndim=2] apply_dealias(
    cnp.ndarray[cnp.float64_t, ndim=2] f,
    cnp.ndarray[cnp.npy_bool, ndim=2] dealias,
):
    """apply 2/3 rule dealias to field f"""

    cdef cnp.ndarray[cnp.complex128_t, ndim=2] f_hat

    f_hat = dealias * fft.fftn(f)
    return np.real(fft.ifftn(f_hat))


cpdef main():
    """Navier-Stokes Simulation"""

    # Simulation parameters
    cdef int N = 400  # Spatial resolution
    cdef double t = 0  # current time of the simulation
    cdef double tEnd = 1  # time at which simulation ends
    cdef double dt = 0.001  # timestep
    cdef double tOut = 0.01  # draw frequency
    cdef double nu = 0.001  # viscosity
    cdef bool plotRealTime = False  # switch on for plotting as the simulation goes along

    # Type annotations
    cdef int Nt, i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xlin, klin
    cdef cnp.ndarray[cnp.float64_t, ndim=2] xx, yy, vx, vy, kx, ky, kSq, kSq_inv
    cdef cnp.ndarray[cnp.npy_bool, ndim=2] dealias
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dvx_x, dvx_y, dvy_x, dvy_y
    cdef cnp.ndarray[cnp.float64_t, ndim=2] rhs_x, rhs_y, div_rhs, P, dPx, dPy, wz



    # Domain [0,1] x [0,1]
    cdef double L = 1
    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]  # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Initial Condition (vortex)
    vx = -np.sin(2 * np.pi * yy)
    vy = np.sin(2 * np.pi * xx * 2)

    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)

    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = fft.ifftshift(kx)
    ky = fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    # dealias with the 2/3 rule
    dealias = (np.abs(kx) < (2.0 / 3.0) * kmax) & (np.abs(ky) < (2.0 / 3.0) * kmax)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)

        # Correction (to eliminate divergence component of velocity)
        vx += -dt * dPx
        vy += -dt * dPy

        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)

        # vorticity (for plotting)
        wz = curl(vx, vy, kx, ky)

        # update time
        t += dt
        # print(t)

        # plot in real time
        
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(wz, cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1
        

    # Save figure
    plt.savefig("navier-stokes-spectral.png", dpi=240)
    plt.show()

    return 0