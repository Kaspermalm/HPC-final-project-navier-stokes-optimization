## GPU (pytorch)
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pdb
from torch.fft import fftn, rfft2, irfft2, fftshift, ifftshift, ifftshift
#%time
"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""

def gpu_load(data,device):
  return torch.from_numpy(data.astype(np.float32)).to(device)

def poisson_solve(rho_hat, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    return -rho_hat * kSq_inv


def div(vx_hat, vy_hat, kx, ky):
    """return divergence of (vx,vy)"""
    return 1j*kx*vx_hat + 1j*ky*vy_hat


def curl(vx_hat, vy_hat, kx, ky):
    """return curl of (vx,vy)"""
    return irfft2(1j*kx*vy_hat - 1j*ky*vx_hat)


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * rfft2(f)
    return f_hat


def main(N=400):
    """Navier-Stokes Simulation"""
    # Initialize pytorch
    device = torch.device('mps')
    #device = torch.device('cuda')
    #print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Simulation parameters
    #N = 400  # Spatial resolution
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
    xx, yy = gpu_load(xx,device), gpu_load(yy,device)

    # Initial Conditions (vortex)
    vx = -torch.sin(2 * np.pi * yy)
    vy = torch.sin(2 * np.pi * xx * 2)
    
    #if False:
    dx = L / N
    kx = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)[:, None]
    kmax = np.max(np.abs(2.0 * np.pi * np.fft.fftfreq(N, d=dx)))
    kx, ky = gpu_load(kx,device), gpu_load(ky,device)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1
    
    # precompute diffusion denom
    diffuse_denom = 1.0 + dt * nu * kSq

    # dealias with the 2/3 rule
    dealias = (torch.abs(kx) < (2.0 / 3.0) * kmax) & (torch.abs(ky) < (2.0 / 3.0) * kmax)

    # initialize the velocity coefficients
    vx_hat = rfft2(vx)
    vy_hat = rfft2(vy)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1
    
    # Main Loop
    for i in range(Nt):
        wz = curl(vx_hat,vy_hat,kx,ky)

        rhs_x = vy * wz
        rhs_y = -vx * wz

        rhs_x_hat = apply_dealias(rhs_x,dealias)
        rhs_y_hat = apply_dealias(rhs_y,dealias)

        vx_hat += dt * rhs_x_hat
        vy_hat += dt * rhs_y_hat

        # Poisson
        div_rhs_hat = div(rhs_x_hat,rhs_y_hat,kx,ky)
        P_hat = poisson_solve(div_rhs_hat,kSq_inv)
        vx_hat -= dt * 1j*kx * P_hat
        vy_hat -= dt * 1j*ky * P_hat

        # Diffusion solve
        vx_hat /= diffuse_denom
        vy_hat /= diffuse_denom

        # Back to time domain
        vx = irfft2(vx_hat)
        vy = irfft2(vy_hat)

        # update time
        t += dt

        # plot in real time
        #if False:
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(wz.cpu().numpy(), cmap="RdBu")
            plt.clim(-20, 20)
            plt.colorbar()
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1

    # Save figure
    #plt.savefig("navier_stokes_spectral.png", dpi=240)
    plt.show()
    wz = curl(vx_hat,vy_hat,kx,ky)
    return wz.cpu().numpy(), vx.cpu().numpy(), vy.cpu().numpy()


if __name__ == "__main__":
    main()
