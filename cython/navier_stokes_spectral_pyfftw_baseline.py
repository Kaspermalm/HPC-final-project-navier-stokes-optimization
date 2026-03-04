import numpy as np
import matplotlib.pyplot as plt

import pyfftw


"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0
"""


pyfftw.interfaces.cache.enable()
# pyfft refactor fft:s to real fft
rfft2 = pyfftw.interfaces.numpy_fft.rfft2
irfft2 = pyfftw.interfaces.numpy_fft.irfft2
# irfftshift = pyfftw.interfaces.numpy_fft.ifftshift


def main():
    """Navier-Stokes Simulation"""

    # Simulation parameters
    N = 400  # Spatial resolution
    t = 0  # current time of the simulation
    tEnd = 1  # time at which simulation ends
    dt = 0.001  # timestep
    tOut = 0.01  # draw frequency
    nu = 0.001  # viscosity
    plotRealTime = False  # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
    L = 1
    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]  # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Initial Condition (vortex)
    vx = -np.sin(2 * np.pi * yy)
    vy = np.sin(2 * np.pi * xx * 2)

    # Fourier Space Variables for rfft2:
    # axis 0 keeps full spectrum (fftfreq), axis 1 is half-spectrum (rfftfreq)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)   # y-frequencies (axis 0)
    kx_1d = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)  # x-frequencies (axis 1)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing="xy")

    # Match the original full-FFT code: kmax = max(klin) with klin in [-N/2, ..., N/2-1]
    kmax = np.max(ky_1d[ky_1d > 0.0])

    # kx = np.fft.ifftshift(kx)
    # ky = np.fft.ifftshift(ky)
    ikx = 1j * kx
    iky = 1j * ky

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

        """return gradient of v"""
        # dvx_x, dvx_y = grad(vx, kx, ky)
        # dvy_x, dvy_y = grad(vy, kx, ky)

        vx_hat = rfft2(vx)
        vy_hat = rfft2(vy)

        dvx_x = irfft2(ikx * vx_hat)
        dvx_y = irfft2(iky * vx_hat)

        dvy_x = irfft2(ikx * vy_hat)
        dvy_y = irfft2(iky * vy_hat)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        """apply 2/3 rule dealias to field f"""
        # rhs_x = apply_dealias(rhs_x, dealias)
        # rhs_y = apply_dealias(rhs_y, dealias)

        rhs_x = irfft2(dealias * rfft2(rhs_x))
        rhs_y = irfft2(dealias * rfft2(rhs_y))

        vx += dt * rhs_x
        vy += dt * rhs_y

        # Poisson solve for pressure
        """return divergence of (rhs_x,rhs_y)"""
        # div_rhs = div(rhs_x, rhs_y, kx, ky)
        dvx_x = irfft2(ikx * rfft2(rhs_x))
        dvy_y = irfft2(iky * rfft2(rhs_y))
        div_rhs = dvx_x + dvy_y

        """solve the Poisson equation, given source field rho"""
        # P = poisson_solve(div_rhs, kSq_inv)
        V_hat = -(rfft2(div_rhs)) * kSq_inv
        P = irfft2(V_hat)

        # Grad P
        div_hat = rfft2(P)

        dPx = irfft2(ikx * div_hat)
        dPy = irfft2(iky * div_hat)

        # Correction (to eliminate divergence component of velocity)
        vx += -dt * dPx
        vy += -dt * dPy

        # Diffusion solve (implicit)
        """solve the diffusion equation over a timestep dt, given viscosity nu"""
        # vx = diffusion_solve(vx, dt, nu, kSq)
        # vy = diffusion_solve(vy, dt, nu, kSq)

        vx_hat = (rfft2(vx)) / (1.0 + dt * nu * kSq)
        vy_hat = (rfft2(vy)) / (1.0 + dt * nu * kSq)

        vx = irfft2(vx_hat)
        vy = irfft2(vy_hat)

        # vorticity (for plotting)
        """return curl of (vx,vy)"""
        # wz = curl(vx, vy, kx, ky)

        dvx_y = irfft2(iky * rfft2(vx))
        dvy_x = irfft2(ikx * rfft2(vy))

        wz = dvy_x - dvx_y

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
    plt.savefig("navier_stokes_spectral.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
