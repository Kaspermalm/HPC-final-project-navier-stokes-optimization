import math

import matplotlib.pyplot as plt
import torch

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid)
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""

try:
    profile
except NameError:

    def profile(func):
        return func


@profile
def poisson_solve(rho_hat, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    return -rho_hat * kSq_inv


@profile
def div(vx_hat, vy_hat, ikx, iky):
    """return divergence of (vx,vy)"""
    return ikx * vx_hat + iky * vy_hat


@profile
def curl(vx_hat, vy_hat, ikx, iky, diff_hat, work_hat, wz, grid_shape):
    """return curl of (vx,vy)"""
    _ = work_hat
    diff_hat[:] = ikx * vy_hat - iky * vx_hat
    wz[:] = torch.fft.irfft2(diff_hat, s=grid_shape).real
    return wz


@profile
def apply_dealias(f, f_hat, dealias):
    """apply 2/3 rule dealias to field f in Fourier space"""
    f_hat[:] = torch.fft.rfft2(f)
    f_hat *= dealias
    return f_hat


@torch.no_grad()
@profile
def main(N=400):
    """Navier-Stokes Simulation"""
    # Simulation parameters
    t = 0.0  # current time of the simulation
    tEnd = 1.0  # time at which simulation ends
    dt = 0.001  # timestep
    tOut = 0.01  # draw frequency
    nu = 0.001  # viscosity
    plotRealTime = False  # switch on for plotting as the simulation goes along

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        real_dtype = torch.float32
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        real_dtype = torch.float32
    else:
        device = torch.device("cpu")
        real_dtype = torch.float64
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    N_half = N // 2 + 1
    grid_shape = (N, N)

    # real variables
    vx = torch.empty(grid_shape, device=device, dtype=real_dtype)
    vy = torch.empty(grid_shape, device=device, dtype=real_dtype)
    wz = torch.empty(grid_shape, device=device, dtype=real_dtype)
    rhs_x = torch.empty(grid_shape, device=device, dtype=real_dtype)
    rhs_y = torch.empty(grid_shape, device=device, dtype=real_dtype)
    # spectral variables
    vx_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    vy_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    rhs_x_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    rhs_y_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    diff_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    work_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    div_rhs_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    P_hat = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    vx_hat_ifft = torch.empty((N, N_half), device=device, dtype=complex_dtype)
    vy_hat_ifft = torch.empty((N, N_half), device=device, dtype=complex_dtype)

    # Domain [0,1] x [0,1]
    L = 1.0
    xlin = torch.linspace(0.0, L, steps=N + 1, device=device, dtype=real_dtype)
    xlin = xlin[0:N]
    yy, xx = torch.meshgrid(xlin, xlin, indexing="ij")

    # Initial Condition (vortex)
    vx[:] = -torch.sin(2.0 * math.pi * yy)
    vy[:] = torch.sin(2.0 * math.pi * xx * 2.0)

    # Fourier Space Variables (half-spectrum on axis 1)
    dx = L / N
    kx = (2.0 * math.pi * torch.fft.rfftfreq(N, d=dx, device=device, dtype=real_dtype))[
        None, :
    ]
    ky = (2.0 * math.pi * torch.fft.fftfreq(N, d=dx, device=device, dtype=real_dtype))[
        :, None
    ]
    kmax = torch.max(
        torch.abs(
            2.0 * math.pi * torch.fft.fftfreq(N, d=dx, device=device, dtype=real_dtype)
        )
    )

    kSq = kx.square() + ky.square()
    kSq_inv = torch.zeros_like(kSq)
    mask = kSq != 0
    kSq_inv[mask] = 1.0 / kSq[mask]
    ikx = (1j * kx).to(complex_dtype)
    iky = (1j * ky).to(complex_dtype)

    # dealias with the 2/3 rule
    dealias = (torch.abs(kx) < (2.0 / 3.0) * kmax) & (
        torch.abs(ky) < (2.0 / 3.0) * kmax
    )

    # number of timesteps
    Nt = math.ceil(tEnd / dt)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Precompute vx/vy_hat
    vx_hat[:] = torch.fft.rfft2(vx)
    vy_hat[:] = torch.fft.rfft2(vy)

    # precompute diffusion denom
    diffuse_denom = 1.0 + dt * nu * kSq

    # Main Loop
    for i in range(Nt):
        # vorticity formulation
        wz = curl(vx_hat, vy_hat, ikx, iky, diff_hat, work_hat, wz, grid_shape)
        rhs_x[:] = vy * wz
        rhs_y[:] = -vx * wz

        rhs_x_hat = apply_dealias(rhs_x, rhs_x_hat, dealias)
        rhs_y_hat = apply_dealias(rhs_y, rhs_y_hat, dealias)

        vx_hat += dt * rhs_x_hat
        vy_hat += dt * rhs_y_hat

        # Poisson solve for pressure
        div_rhs_hat[:] = div(rhs_x_hat, rhs_y_hat, ikx, iky)
        P_hat[:] = poisson_solve(div_rhs_hat, kSq_inv)

        vx_hat -= dt * ikx * P_hat
        vy_hat -= dt * iky * P_hat

        # Diffusion solve
        vx_hat /= diffuse_denom
        vy_hat /= diffuse_denom

        # copy to ifft temp variables
        vx_hat_ifft[:] = vx_hat
        vy_hat_ifft[:] = vy_hat
        vx[:] = torch.fft.irfft2(vx_hat_ifft, s=grid_shape).real
        vy[:] = torch.fft.irfft2(vy_hat_ifft, s=grid_shape).real

        # update time
        t += dt

        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(wz.detach().cpu().numpy(), cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1

    # Save figure
    plt.show()

    return 0


if __name__ == "__main__":
    main()
