import argparse
import math
import time

import matplotlib.pyplot as plt


try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise ImportError(
        "PyTorch is required for cython/navier_stokes_spectral_torch_mps.py. "
        "Install torch with MPS support to run this solver."
    ) from exc


def _resolve_device(device):
    if device is None:
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is False."
        )
    return resolved


def _real_dtype_for_device(device):
    if device.type == "mps":
        return torch.float32
    return torch.float64


def _complex_dtype_for_real(real_dtype):
    return torch.complex64 if real_dtype == torch.float32 else torch.complex128


def poisson_solve(rho_hat, k_sq_inv):
    return -rho_hat * k_sq_inv


def div(vx_hat, vy_hat, ikx, iky):
    return ikx * vx_hat + iky * vy_hat


def curl(vx_hat, vy_hat, ikx, iky, grid_shape):
    return torch.fft.irfft2(ikx * vy_hat - iky * vx_hat, s=grid_shape).real


def apply_dealias(field, dealias):
    return torch.fft.rfft2(field) * dealias


def _plot_vorticity(wz, save_path=None):
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = fig.gca()
    image = ax.imshow(wz.detach().cpu().numpy(), cmap="RdBu")
    image.set_clim(-20, 20)
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")

    if save_path is not None:
        fig.savefig(save_path, dpi=240, bbox_inches="tight")

    plt.show()
    return fig


@torch.no_grad()
def main(
    N=400,
    t_end=1.0,
    dt=0.001,
    t_out=0.01,
    nu=0.001,
    plot_real_time=False,
    plot_final=True,
    save_path=None,
    device=None,
    return_state=False,
):
    resolved_device = _resolve_device(device)
    real_dtype = _real_dtype_for_device(resolved_device)
    complex_dtype = _complex_dtype_for_real(real_dtype)

    t = 0.0
    output_count = 1
    n_half = N // 2 + 1
    grid_shape = (N, N)
    L = 1.0

    xlin = torch.linspace(
        0.0, L, steps=N + 1, device=resolved_device, dtype=real_dtype
    )[:-1]
    yy, xx = torch.meshgrid(xlin, xlin, indexing="ij")

    vx = -torch.sin(2.0 * torch.pi * yy)
    vy = torch.sin(4.0 * torch.pi * xx)

    dx = L / N
    kx = (
        2.0
        * torch.pi
        * torch.fft.rfftfreq(N, d=dx, device=resolved_device, dtype=real_dtype)
    ).view(1, n_half)
    ky = (
        2.0
        * torch.pi
        * torch.fft.fftfreq(N, d=dx, device=resolved_device, dtype=real_dtype)
    ).view(N, 1)
    kmax = torch.max(
        torch.abs(
            2.0
            * torch.pi
            * torch.fft.fftfreq(N, d=dx, device=resolved_device, dtype=real_dtype)
        )
    )

    k_sq = kx.square() + ky.square()
    k_sq_inv = torch.zeros_like(k_sq)
    mask = k_sq != 0
    k_sq_inv[mask] = 1.0 / k_sq[mask]

    ikx = (1j * kx).to(complex_dtype)
    iky = (1j * ky).to(complex_dtype)
    dealias = (
        (torch.abs(kx) < (2.0 / 3.0) * kmax) & (torch.abs(ky) < (2.0 / 3.0) * kmax)
    ).to(complex_dtype)
    diffuse_denom = (1.0 + dt * nu * k_sq).to(complex_dtype)

    nt = math.ceil(t_end / dt)

    vx_hat = torch.fft.rfft2(vx)
    vy_hat = torch.fft.rfft2(vy)
    wz = torch.zeros(grid_shape, device=resolved_device, dtype=real_dtype)

    for i in range(nt):
        current_wz = curl(vx_hat, vy_hat, ikx, iky, grid_shape)
        rhs_x = vy * current_wz
        rhs_y = -vx * current_wz

        rhs_x_hat = apply_dealias(rhs_x, dealias)
        rhs_y_hat = apply_dealias(rhs_y, dealias)

        vx_hat = vx_hat + dt * rhs_x_hat
        vy_hat = vy_hat + dt * rhs_y_hat

        div_rhs_hat = div(rhs_x_hat, rhs_y_hat, ikx, iky)
        p_hat = poisson_solve(div_rhs_hat, k_sq_inv)

        vx_hat = vx_hat - dt * (ikx * p_hat)
        vy_hat = vy_hat - dt * (iky * p_hat)

        vx_hat = vx_hat / diffuse_denom
        vy_hat = vy_hat / diffuse_denom

        vx = torch.fft.irfft2(vx_hat, s=grid_shape).real
        vy = torch.fft.irfft2(vy_hat, s=grid_shape).real

        t += dt
        plot_this_turn = t + dt > output_count * t_out
        if plot_this_turn:
            output_count += 1
            if plot_real_time:
                wz = curl(vx_hat, vy_hat, ikx, iky, grid_shape)
                _plot_vorticity(wz)

    wz = curl(vx_hat, vy_hat, ikx, iky, grid_shape)

    if plot_final:
        _plot_vorticity(wz, save_path=save_path)

    if return_state:
        return {
            "device": str(resolved_device),
            "dtype": str(real_dtype),
            "t": t,
            "vx": vx.detach().cpu(),
            "vy": vy.detach().cpu(),
            "wz": wz.detach().cpu(),
            "vx_hat": vx_hat.detach().cpu(),
            "vy_hat": vy_hat.detach().cpu(),
        }

    return 0


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Torch Navier-Stokes spectral solver with MPS support."
    )
    parser.add_argument("--N", type=int, default=400)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t-out", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument("--no-plot-final", action="store_true")
    parser.add_argument("--save-path", default=None, help="Optional image output path.")
    parser.add_argument(
        "--device", default=None, help="Torch device, for example mps or cpu."
    )
    parser.add_argument("--return-state", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    start = time.perf_counter()
    result = main(
        N=args.N,
        t_end=args.t_end,
        dt=args.dt,
        t_out=args.t_out,
        nu=args.nu,
        plot_final=not args.no_plot_final,
        save_path=args.save_path,
        device=args.device,
        return_state=args.return_state,
    )
    elapsed = time.perf_counter() - start

    if args.return_state:
        print(
            {
                "device": result["device"],
                "dtype": result["dtype"],
                "t": result["t"],
                "elapsed_s": elapsed,
            }
        )
    else:
        print(elapsed)
