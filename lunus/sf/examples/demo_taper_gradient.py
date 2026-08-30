"""
Demonstrates, directly and numerically, that a hard density cutoff makes
the autograd-style gradient (differentiate the smooth Gaussian, holding
the mask fixed -- exactly what torch.where's backward rule does) diverge
from the TRUE gradient (finite difference on the actual masked function,
which correctly captures voxels crossing the boundary as the atom
moves) -- and that a smooth taper fixes this.

Uses a simplified, unit-cell-free setup (plain Euclidean 3D, single
atom) since the point is about the cutoff mechanism itself, not
crystallographic bookkeeping.
"""
import numpy as np
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import precalculate_density_iso, cutoff_radius


def hard_mask_density(atom_pos, grid_points, A, lam, R):
    d = grid_points - atom_pos
    r2 = np.sum(d * d, axis=-1)
    dens = np.sum(A * np.exp(-lam * r2[..., None]), axis=-1)
    return np.where(r2 <= R * R, dens, 0.0)


def smooth_taper_density(atom_pos, grid_points, A, lam, R, w):
    """Same Gaussian sum, but multiplied by a C^1 raised-cosine taper
    that goes from 1 at r=R-w to 0 at r=R, instead of a hard step."""
    d = grid_points - atom_pos
    r = np.sqrt(np.sum(d * d, axis=-1))
    dens = np.sum(A * np.exp(-lam * r[..., None] ** 2), axis=-1)
    taper = np.ones_like(r)
    edge = (r > R - w) & (r < R)
    taper[edge] = 0.5 * (1 + np.cos(np.pi * (r[edge] - (R - w)) / w))
    taper[r >= R] = 0.0
    return dens * taper


def total_density(atom_pos, grid_points, A, lam, R, mode, w=None):
    if mode == "hard":
        return hard_mask_density(atom_pos, grid_points, A, lam, R).sum()
    else:
        return smooth_taper_density(atom_pos, grid_points, A, lam, R, w).sum()


def analytic_masked_gradient(atom_pos, grid_points, A, lam, R):
    """Replicates exactly what autograd computes for torch.where(r2<=R^2, dens, 0):
    differentiate the smooth Gaussian analytically, but hold the mask FIXED
    at the current position -- never accounting for boundary crossings."""
    d = grid_points - atom_pos  # (N,3)
    r2 = np.sum(d * d, axis=-1)
    mask = (r2 <= R * R).astype(np.float64)
    # d(dens)/d(pos) = sum_i A_i * exp(-lam_i r2) * (-lam_i) * d(r2)/d(pos)
    # d(r2)/d(pos) = -2*d   (since d = grid - pos, r2 = |d|^2)
    dens_terms = A * np.exp(-lam * r2[..., None] ** 0 * r2[..., None])  # (N,5) -- r2 already squared dist
    ddens_dr2 = np.sum(dens_terms * (-lam), axis=-1)  # (N,)
    grad_per_point = ddens_dr2[:, None] * (-2 * d) * mask[:, None]  # (N,3)
    return grad_per_point.sum(axis=0)


def analytic_taper_gradient(atom_pos, grid_points, A, lam, R, w):
    """Same idea, but now differentiating through the smooth taper too --
    this is what autograd WOULD correctly compute for a smooth cutoff,
    since every piece is genuinely differentiable."""
    d = grid_points - atom_pos
    r2 = np.sum(d * d, axis=-1)
    r = np.sqrt(r2)
    dens = np.sum(A * np.exp(-lam * r2[..., None]), axis=-1)
    ddens_dr2 = np.sum(A * np.exp(-lam * r2[..., None]) * (-lam), axis=-1)
    ddens_dpos = ddens_dr2[:, None] * (-2 * d)  # (N,3), chain rule through r2

    taper = np.ones_like(r)
    dtaper_dr = np.zeros_like(r)
    edge = (r > R - w) & (r < R)
    taper[edge] = 0.5 * (1 + np.cos(np.pi * (r[edge] - (R - w)) / w))
    dtaper_dr[edge] = -0.5 * np.sin(np.pi * (r[edge] - (R - w)) / w) * (np.pi / w)
    taper[r >= R] = 0.0
    # dr/dpos = d / r  (avoid div by zero at r=0, negligible contribution there anyway)
    r_safe = np.where(r > 1e-8, r, 1.0)
    dr_dpos = -d / r_safe[:, None]
    dtaper_dpos = dtaper_dr[:, None] * dr_dpos

    # product rule: d(dens*taper)/dpos = ddens_dpos*taper + dens*dtaper_dpos
    grad_per_point = ddens_dpos * taper[:, None] + dens[:, None] * dtaper_dpos
    return grad_per_point.sum(axis=0)


def main():
    b_iso = 16.2
    coef9 = IT92_COEFFS["C"]
    A, lam = precalculate_density_iso(coef9, b_iso)
    R = cutoff_radius(A, lam, cutoff=0.01)
    w = 0.4  # taper transition width, Angstrom -- modest fraction of R

    print(f"cutoff radius R = {R:.3f} A, taper width w = {w} A")

    # Much finer grid than the production grid (0.3 A) -- this is a
    # deliberately over-resolved quadrature grid purely to get a clean,
    # well-converged picture of the continuum boundary effect, not a
    # claim about what grid resolution xtraj.py should actually use.
    spacing = 0.04
    n = int(np.ceil((R + 1.0) / spacing))
    ax = np.arange(-n, n + 1) * spacing
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    grid_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    print(f"quadrature grid: {grid_points.shape[0]} points, spacing = {spacing} A")

    rng = np.random.default_rng(0)
    eps = 1e-4

    print("\n--- HARD CUTOFF (averaged over 5 random atom positions/directions) ---")
    rel_diffs = []
    for trial in range(5):
        atom_pos = rng.uniform(-0.15, 0.15, size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        f_p = total_density(atom_pos + eps * direction, grid_points, A, lam, R, "hard")
        f_m = total_density(atom_pos - eps * direction, grid_points, A, lam, R, "hard")
        fd_grad = (f_p - f_m) / (2 * eps)
        ag_grad = analytic_masked_gradient(atom_pos, grid_points, A, lam, R) @ direction
        rel_diff = abs(fd_grad - ag_grad) / max(abs(fd_grad), 1e-8)
        rel_diffs.append(rel_diff)
        print(f"  trial {trial}: finite-diff (true) = {fd_grad:10.5f}   "
              f"autograd-style = {ag_grad:10.5f}   rel.diff = {rel_diff:.2%}")
    print(f"  mean relative difference = {np.mean(rel_diffs):.2%}")

    print("\n--- SMOOTH TAPER (same 5 trials) ---")
    rng = np.random.default_rng(0)  # same trials for a fair comparison
    rel_diffs_t = []
    for trial in range(5):
        atom_pos = rng.uniform(-0.15, 0.15, size=3)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        f_p = total_density(atom_pos + eps * direction, grid_points, A, lam, R, "taper", w)
        f_m = total_density(atom_pos - eps * direction, grid_points, A, lam, R, "taper", w)
        fd_grad = (f_p - f_m) / (2 * eps)
        ag_grad = analytic_taper_gradient(atom_pos, grid_points, A, lam, R, w) @ direction
        rel_diff = abs(fd_grad - ag_grad) / max(abs(fd_grad), 1e-8)
        rel_diffs_t.append(rel_diff)
        print(f"  trial {trial}: finite-diff = {fd_grad:10.5f}   "
              f"autograd-style = {ag_grad:10.5f}   rel.diff = {rel_diff:.2%}")
    print(f"  mean relative difference = {np.mean(rel_diffs_t):.2%}")


if __name__ == "__main__":
    main()
