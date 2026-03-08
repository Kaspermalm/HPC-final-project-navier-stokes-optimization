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
# fftn = pyfftw.interfaces.numpy_fft.fftn
# ifftn = pyfftw.interfaces.numpy_fft.ifftn


def poisson_solve(rho_hat, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    return -rho_hat * kSq_inv


# def diffusion_solve(vx_hat, vy_hat, diffuse_denom):
#     """solve the diffusion equation over a timestep dt, given viscosity nu"""
#     vx_hat /= diffuse_denom
#     vy_hat /= diffuse_denom
#     return vx_hat, vy_hat


# def grad(v_hat, ikx, iky):
#     """return gradient of v"""
#     dvx = ifftn(ikx * v_hat).real
#     dvy = ifftn(iky * v_hat).real
#     return dvx, dvy


def div(vx_hat, vy_hat, ikx, iky):
    """return divergence of (vx,vy)"""
    return ikx * vx_hat + iky * vy_hat


def curl(vx_hat, vy_hat, ikx, iky, diff_hat, wz, ifft_wz):
    """return curl of (vx,vy)"""
    diff_hat[:] = ikx * vy_hat - iky * vx_hat
    ifft_wz()
    return wz.real


def apply_dealias(f_hat, dealias, fft_f):
    """apply 2/3 rule dealias to field f in Fourier space (via planned FFT)"""
    fft_f()
    f_hat *= dealias
    return f_hat


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

    # Allocate aligned arrays for fftw plans
    Nh = N // 2 + 1
    # real variables
    vx = pyfftw.empty_aligned((N, N), dtype="float64")
    vy = pyfftw.empty_aligned((N, N), dtype="float64")
    wz = pyfftw.empty_aligned((N, N), dtype="float64")
    rhs_x = pyfftw.empty_aligned((N, N), dtype="float64")
    rhs_y = pyfftw.empty_aligned((N, N), dtype="float64")
    # spectral variables
    vx_hat = pyfftw.empty_aligned((N, Nh), dtype="complex128")
    vy_hat = pyfftw.empty_aligned((N, Nh), dtype="complex128")
    rhs_x_hat = pyfftw.empty_aligned((N, Nh), dtype="complex128")
    rhs_y_hat = pyfftw.empty_aligned((N, Nh), dtype="complex128")
    diff_hat = pyfftw.empty_aligned((N, Nh), dtype="complex128")

    # create pyfftw.FFTW objects with plans for each fft operations
    fft_vx = pyfftw.FFTW(vx, vx_hat, axes=(0, 1), direction="FFTW_FORWARD")
    fft_vy = pyfftw.FFTW(vy, vy_hat, axes=(0, 1), direction="FFTW_FORWARD")
    fft_rhs_x = pyfftw.FFTW(rhs_x, rhs_x_hat, axes=(0, 1), direction="FFTW_FORWARD")
    fft_rhs_y = pyfftw.FFTW(rhs_y, rhs_y_hat, axes=(0, 1), direction="FFTW_FORWARD")
    ifft_wz = pyfftw.FFTW(
        diff_hat,
        wz,
        axes=(0, 1),
        direction="FFTW_BACKWARD",
        normalise_idft=True,
    )
    ifft_vx = pyfftw.FFTW(
        vx_hat,
        vx,
        axes=(0, 1),
        direction="FFTW_BACKWARD",
        normalise_idft=True,
    )
    ifft_vy = pyfftw.FFTW(
        vy_hat,
        vy,
        axes=(0, 1),
        direction="FFTW_BACKWARD",
        normalise_idft=True,
    )

    # Domain [0,1] x [0,1]
    L = 1
    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]  # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Initial Condition (vortex)
    vx[:] = -np.sin(2 * np.pi * yy)
    vy[:] = np.sin(2 * np.pi * xx * 2)

    # Fourier Space Variables (half-spectrum on axis 1)
    dx = L / N
    kx = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)[:, None]
    kmax = np.max(np.abs(2.0 * np.pi * np.fft.fftfreq(N, d=dx)))

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
    # vx_hat =fftn(vx)
    # vy_hat = fftn(vy)
    fft_vx()
    fft_vy()

    # precompute diffusion denom
    diffuse_denom = 1.0 + dt * nu * kSq

    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        # dvx_x, dvx_y = grad(vx_hat, ikx, iky)
        # dvy_x, dvy_y = grad(vy_hat, ikx, iky)

        # rhs_x = -(vx * dvx_x + vy * dvx_y)
        # rhs_y = -(vx * dvy_x + vy * dvy_y)

        # vorticity formualation
        wz_real = curl(vx_hat, vy_hat, ikx, iky, diff_hat, wz, ifft_wz)
        rhs_x[:] = vy * wz_real
        rhs_y[:] = -vx * wz_real

        rhs_x_hat = apply_dealias(rhs_x_hat, dealias, fft_rhs_x)
        rhs_y_hat = apply_dealias(rhs_y_hat, dealias, fft_rhs_y)

        vx_hat += dt * rhs_x_hat
        vy_hat += dt * rhs_y_hat

        # Poisson solve for pressure
        # Keep this identical to navier_stokes_spectral_pyfftw_algo_optim_vortex.py:
        # project nonlinear increment (rhs), not the full updated velocity field.
        div_rhs = div(rhs_x_hat, rhs_y_hat, ikx, iky)
        P = poisson_solve(div_rhs, kSq_inv)

        # Correction (to eliminate divergence component of velocity)
        vx_hat += -dt * (ikx * P)
        vy_hat += -dt * (iky * P)

        # Diffusion solve (implicit)
        # diffusion_solve(vx_hat, vy_hat, diffuse_denom) # removed to keep inline
        vx_hat /= diffuse_denom
        vy_hat /= diffuse_denom

        # new real space fields
        ifft_vx()
        ifft_vy()

        # update time
        t += dt
        # print(t)

        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(wz_real, cmap="RdBu")
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
