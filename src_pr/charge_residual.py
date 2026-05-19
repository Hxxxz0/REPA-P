import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("FONTCONFIG_PATH", "/tmp/fontconfig-cache")
os.makedirs(os.environ["FONTCONFIG_PATH"], exist_ok=True)

import argparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve (-Δ) U = ρ via DST and visualize the potential/residual on a square grid."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None ⇒ new realization).")
    parser.add_argument("--N", type=int, default=256, help="Interior grid size (excludes boundary).")
    parser.add_argument("--step", type=float, default=1.0, help="Uniform grid spacing (dx=dy).")
    parser.add_argument("--margin", type=int, default=32, help="Keep charges at least this far from boundary.")
    return parser.parse_args()


def main():
    """Run Poisson residual diagnostics for two random Coulomb point charges and solve via DST."""
    args = parse_args()
    N = args.N
    dx = dy = args.step
    rng = np.random.default_rng(args.seed)
    margin = min(args.margin, max(1, N // 2 - 1))  # keep charges away from boundaries

    q1, q2 = random_charge(rng), random_charge(rng)
    x1, y1 = rng.integers(margin, N - margin), rng.integers(margin, N - margin)
    x2, y2 = rng.integers(margin, N - margin), rng.integers(margin, N - margin)

    rho = np.zeros((N, N), dtype=float)
    rho[x1, y1] += q1 / (dx * dy)
    rho[x2, y2] += q2 / (dx * dy)

    U = solve_poisson_dst(rho, dx)

    U_full = np.zeros((N + 2, N + 2), float)
    U_full[1:-1, 1:-1] = U
    lapU = (
        4 * U_full[1:-1, 1:-1]
        - U_full[0:-2, 1:-1]
        - U_full[2:, 1:-1]
        - U_full[1:-1, 0:-2]
        - U_full[1:-1, 2:]
    ) / (dx * dy)
    r = lapU - rho
    L_inf = np.abs(r).max()
    L2 = np.sqrt((r**2).sum() * (dx * dy))

    print(f"q1 = {q1:.4f} at (x1,y1)=({x1},{y1})")
    print(f"q2 = {q2:.4f} at (x2,y2)=({x2},{y2})")
    print(f"Residual ||r||_inf = {L_inf:.6f}")
    print(f"Residual ||r||_2   = {L2:.6f}")

    if args.seed is None:
        print("Seed not provided: random generation on each run.")
    else:
        print(f"Seed = {args.seed}")

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)
    plot_field(U, x1, y1, x2, y2, q1, q2, dx, outdir)
    plot_residual(r, outdir)


def random_charge(rng):
    sign = rng.choice([-1.0, 1.0])
    mag = rng.uniform(0.5, 1.5)
    return sign * mag


def dst1(x):
    x = np.asarray(x)
    n = x.shape[-1]
    y = np.zeros(x.shape[:-1] + (2 * (n + 1),), float)
    y[..., 1:n+1] = x
    y[..., n+2:] = -x[..., ::-1]
    Y = np.fft.fft(y, axis=-1)
    return -Y.imag[..., 1:n+1]


def idst1(X):
    return dst1(X) / (2 * (X.shape[-1] + 1))


def solve_poisson_dst(rho, h):
    N = rho.shape[0]
    rho_hat = dst1(dst1(rho.T).T)
    m = np.arange(1, N + 1)
    n = np.arange(1, N + 1)
    lam_m = 2.0 * (1 - np.cos(np.pi * m / (N + 1))) / (h * h)
    lam_n = 2.0 * (1 - np.cos(np.pi * n / (N + 1))) / (h * h)
    lam_2d = lam_m[:, None] + lam_n[None, :]
    U_hat = rho_hat / lam_2d
    return idst1(idst1(U_hat.T).T)


def plot_field(U, x1, y1, x2, y2, q1, q2, dx, outdir):
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(U.shape[0]) - 0.5
    x = np.arange(U.shape[1]) - 0.5
    levels = np.linspace(U.min(), U.max(), 40)
    cn = ax.contour(
        x,
        y,
        U.T,
        levels=levels,
        cmap="viridis",
        linewidths=0.9,
        linestyles="-",
        antialiased=True,
    )
    ax.clabel(cn, fmt="%.2f", inline=True, fontsize=6)
    grad_x, grad_y = np.gradient(U, dx, dx)
    Ex = -grad_x
    Ey = -grad_y
    X, Y = np.meshgrid(x, y, indexing="xy")
    speed = np.hypot(Ey, Ex)
    strm = ax.streamplot(
        X,
        Y,
        Ey,
        Ex,
        color=speed,
        cmap="winter",
        density=1.2,
        linewidth=0.8,
        arrowsize=1.0,
        arrowstyle="->",
    )
    scatter = ax.scatter(
        [y1, y2],
        [x1, x2],
        c=["red" if q > 0 else "blue" for q in (q1, q2)],
        edgecolor="black",
        linewidth=1.0,
        s=140,
        marker="o",
        zorder=5,
        label="point charges",
    )
    for label, (cx, cy, charge) in zip(
        ["charge 1", "charge 2"], [(x1, y1, q1), (x2, y2, q2)]
    ):
        ax.text(
            cy + 3,
            cx + 3,
            f"{label} ({'+' if charge > 0 else '-'})",
            color="black",
            fontsize=8,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"),
        )
    ax.set_title("Equipotential + Electric Field Lines", fontsize=14)
    ax.set_xlabel("y grid index")
    ax.set_ylabel("x grid index")
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(alpha=0.25, linestyle=":")
    cbar = fig.colorbar(cn, ax=ax, shrink=0.75, label="U")
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.7)
    legend = ax.legend(loc="upper right", framealpha=0.8)
    legend.get_frame().set_linewidth(0.7)
    strm.lines.set_alpha(0.65)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "potential.png"), dpi=150)
    plt.close(fig)


def plot_residual(r, outdir):
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(r.shape[0]) - 0.5
    x = np.arange(r.shape[1]) - 0.5
    max_abs = float(np.max(np.abs(r)))
    if max_abs == 0.0:
        max_abs = 1e-12
    levels = np.linspace(-max_abs, max_abs, 80)
    cf = ax.contourf(x, y, r.T, levels=levels, cmap="RdBu_r", extend="both")
    cn = ax.contour(
        x, y, r.T, levels=np.linspace(-max_abs, max_abs, 20), colors="black", linewidths=0.4
    )
    ax.clabel(cn, fmt="%.3f", inline=True, fontsize=6)
    ax.set_title("Residual r = div(∇U) - ρ (interior)")
    ax.set_xlabel("y (interior)")
    ax.set_ylabel("x (interior)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linestyle=":")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.78, label="residual")
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "residual.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
