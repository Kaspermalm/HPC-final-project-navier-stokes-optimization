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
fftn = pyfftw.interfaces.numpy_fft.fftn
ifftn = pyfftw.interfaces.numpy_fft.ifftn


def poisson_solve(rho_hat, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    return -rho_hat * kSq_inv


def diffusion_solve(vx_hat, vy_hat, diffuse_denom):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    vx_hat = vx_hat / diffuse_denom
    vy_hat = vy_hat / diffuse_denom
    return vx_hat, vy_hat


def grad(v_hat, ikx, iky):
    """return gradient of v"""
    dvx = ifftn(ikx * v_hat).real
    dvy = ifftn(iky * v_hat).real
    return dvx, dvy


def div(vx_hat, vy_hat, ikx, iky):
    """return divergence of (vx,vy)"""
    return ikx * vx_hat + iky * vy_hat


def curl(vx_hat, vy_hat, ikx, iky):
    """return curl of (vx,vy)"""
    return ifftn(ikx * vy_hat - iky * vx_hat).real


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = fftn(f)
    return dealias * f_hat


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

    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)

    # FIXED: zero division for kSq
    kSq = kx**2 + ky**2
    # kSq_inv = 1.0 / kSq
    # kSq_inv[kSq == 0] = 1
    kSq_inv = np.zeros_like(kSq)
    mask = kSq != 0
    kSq_inv[mask] = 1.0 / kSq[mask]

    ikx = 1j * kx
    iky = 1j * ky

    # dealias with the 2/3 rule
    dealias = (np.abs(kx) < (2.0 / 3.0) * kmax) & (np.abs(ky) < (2.0 / 3.0) * kmax)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Precompute vx/vy_hat
    vx_hat = fftn(vx)
    vy_hat = fftn(vy)

    # precompute diffusion denom
    diffuse_denom = 1.0 + dt * nu * kSq

    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y = grad(vx_hat, ikx, iky)
        dvy_x, dvy_y = grad(vy_hat, ikx, iky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx_hat += dt * rhs_x
        vy_hat += dt * rhs_y

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, ikx, iky)
        P = poisson_solve(div_rhs, kSq_inv)

        # Correction (to eliminate divergence component of velocity)
        vx_hat += -dt * (ikx * P)
        vy_hat += -dt * (iky * P)

        # Diffusion solve (implicit)
        vx_hat, vy_hat = diffusion_solve(vx_hat, vy_hat, diffuse_denom)

        # new real space fields
        vx = ifftn(vx_hat).real
        vy = ifftn(vy_hat).real

        # update time
        t += dt
        # print(t)

        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            # vorticity (for plotting)
            wz = curl(vx_hat, vy_hat, ikx, iky)

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
