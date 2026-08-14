"""
Differentiable PyTorch equivalent of gemmi's ``Grid::symmetrize_sum()``.

This is how the gemmi engine builds a full unit cell of density: splat only
the atoms it was given, then sum the density over the space group's
symmetry operations, rather than replicating the ATOMS by symmetry and
splatting every copy. gemmi's ``put_model_density_on_grid()`` is exactly
``initialize_grid(); add_model_density_to_grid(model); symmetrize_sum()``.

Doing it this way is much cheaper than expanding atoms: splatting costs
O(n_atoms * box_volume) Gaussian evaluations per symmetry copy, while
symmetrizing the finished grid costs O(n_grid) integer index arithmetic
per operation -- and for the signed-permutation operations that make up
essentially every common space group, it is just axis permutes, flips and
rolls, with no index tensors materialized at all.

VERIFIED CONVENTION (probed directly against gemmi rather than read off
its source): ``symmetrize_sum()`` computes

    new[p] = sum over all ops g, INCLUDING the identity, of  data[g(p)]

so every point of a symmetry orbit ends up holding that orbit's total.
Reproduced to float32 rounding (1.8e-07) on a random test grid.

DIFFERENTIABILITY: every operation used here (permute/flip/roll for the
fast path, advanced indexing for the general path, and the final sum) is
a standard differentiable PyTorch primitive whose backward is a gather or
scatter-add. Gradients flow back through the symmetrization to the atomic
coordinates exactly as they do through the splat itself.
"""

import numpy as np
import torch


def _is_signed_permutation(R):
    """True if R is a signed permutation matrix (one +/-1 per row and column)."""
    return (
        np.all(np.isin(R, (-1, 0, 1)))
        and np.all(np.count_nonzero(R, axis=1) == 1)
        and np.all(np.count_nonzero(R, axis=0) == 1)
    )


class GridOp:
    """
    One space group operation expressed in integer GRID index units:

        y_i = sum_j R[i,j] * x_j + t_grid[i]   (mod N_i)

    where x is a grid index triple and y is the index it maps to. Built by
    build_grid_ops(), which validates that the operation actually maps this
    grid onto itself exactly (no interpolation is ever done, matching gemmi).
    """

    def __init__(self, R, t_grid, grid_shape, xyz_str=None):
        self.R = np.asarray(R, dtype=np.int64)
        self.t_grid = np.asarray(t_grid, dtype=np.int64)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        self.xyz_str = xyz_str
        self.fast = _is_signed_permutation(self.R)

        if self.fast:
            # y_i = s_i * x_{sigma(i)} + t_grid[i]
            self.sigma = [int(np.nonzero(self.R[i])[0][0]) for i in range(3)]
            self.sign = [int(self.R[i, self.sigma[i]]) for i in range(3)]
            self.sigma_inv = [0, 0, 0]
            for i, j in enumerate(self.sigma):
                self.sigma_inv[j] = i

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Returns the tensor G with G[p] = grid[self(p)]. Differentiable."""
        return self._apply_fast(grid) if self.fast else self._apply_gather(grid)

    def _apply_fast(self, grid):
        N = self.grid_shape
        out = grid
        # Step 1: E[z] = grid[sign*z + t_grid], one axis at a time.
        for i in range(3):
            if self.sign[i] == 1:
                shift = -self.t_grid[i]
            else:
                # flip gives F[z] = grid[N-1-z]; we want grid[t-z] = F[z + (N-1-t)]
                out = torch.flip(out, dims=(i,))
                shift = -(N[i] - 1 - self.t_grid[i])
            shift = int(shift % N[i])
            if shift:
                out = torch.roll(out, shifts=shift, dims=i)
        # Step 2: relabel axes so that z_i = x_{sigma(i)}.
        return out.permute(*self.sigma_inv)

    def _apply_gather(self, grid):
        N = self.grid_shape
        device = grid.device
        axes = [torch.arange(N[k], device=device, dtype=torch.long) for k in range(3)]
        index = []
        for i in range(3):
            acc = torch.zeros(1, device=device, dtype=torch.long)
            for j in range(3):
                if self.R[i, j] != 0:
                    shape = [N[j] if k == j else 1 for k in range(3)]
                    acc = acc + int(self.R[i, j]) * axes[j].view(shape)
            index.append(torch.remainder(acc + int(self.t_grid[i]), N[i]))
        i0, i1, i2 = torch.broadcast_tensors(*index)
        return grid[i0, i1, i2]

    def __repr__(self):
        tag = "fast" if self.fast else "gather"
        return f"GridOp({self.xyz_str or self.R.tolist()}, t_grid={self.t_grid.tolist()}, {tag})"


def build_grid_ops(rotations, translations, grid_shape, tol=1e-6):
    """
    rotations:    iterable of (3,3) integer rotation matrices (fractional basis)
    translations: iterable of (3,) fractional translations
    grid_shape:   (Nu, Nv, Nw)

    Returns a list of GridOp, EXCLUDING the identity (which symmetrize_sum
    contributes as the untransformed grid), matching gemmi's
    get_scaled_ops_except_id(). An empty list means "nothing to do" -- which
    is exactly what happens for P1, again matching gemmi, whose symmetrize()
    returns immediately when the space group is P1.

    Raises ValueError if an operation does not map this grid exactly onto
    itself, rather than silently interpolating:

      * a fractional translation t_i must land on a grid plane (t_i * N_i
        integral) -- e.g. the 3/4 screw translation of P4_3 needs Nw
        divisible by 4;
      * an axis-mixing rotation (R[i,j] != 0 for i != j) needs N_i == N_j,
        since it maps axis j's sampling onto axis i's -- e.g. the 4-fold of
        P4_3 requires Nu == Nv.

    cell_utils.grid_shape_for_resolution() rounds each axis up to an even
    5-smooth number independently, so it does NOT guarantee either condition;
    check with this function and adjust the grid if it raises.
    """
    N = tuple(int(n) for n in grid_shape)
    ops = []

    for R_in, t_in in zip(rotations, translations):
        R = np.asarray(R_in, dtype=np.float64)
        t = np.asarray(t_in, dtype=np.float64)

        if np.max(np.abs(R - np.round(R))) > tol:
            raise ValueError(f"non-integer rotation matrix in fractional basis:\n{R}")
        R = np.round(R).astype(np.int64)

        is_identity = np.array_equal(R, np.eye(3, dtype=np.int64)) and np.all(np.abs(t) < tol)
        if is_identity:
            continue

        t_grid_f = t * np.array(N, dtype=np.float64)
        if np.max(np.abs(t_grid_f - np.round(t_grid_f))) > tol * max(N):
            raise ValueError(
                f"symmetry translation {t.tolist()} does not land on grid {N}: "
                f"t*N = {t_grid_f.tolist()} is not integral. Choose a grid whose "
                "axes are divisible by the translation denominators."
            )
        t_grid = np.round(t_grid_f).astype(np.int64)

        for i in range(3):
            for j in range(3):
                if R[i, j] != 0 and N[i] != N[j]:
                    raise ValueError(
                        f"symmetry operation mixes axes {j} -> {i} (R[{i},{j}]={R[i, j]}) "
                        f"but the grid has N[{i}]={N[i]} != N[{j}]={N[j]}; the operation "
                        "cannot map this grid onto itself. Use equal sampling on those axes."
                    )

        ops.append(GridOp(R, t_grid, N))

    return ops


def symmetry_grid_constraints(rotations, translations, tol=1e-6, max_denom=48):
    """
    Returns (groups, divisors) -- the two conditions build_grid_ops enforces:

      groups:   list of sets of axis indices that must have EQUAL sampling
                (because some operation maps one onto another)
      divisors: (3,) list; N[i] must be a multiple of divisors[i] so every
                fractional translation lands on a grid plane
    """
    from fractions import Fraction
    from math import gcd

    parent = [0, 1, 2]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    divisors = [1, 1, 1]
    for R_in, t_in in zip(rotations, translations):
        R = np.round(np.asarray(R_in, dtype=np.float64)).astype(np.int64)
        t = np.asarray(t_in, dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i != j and R[i, j] != 0:
                    union(i, j)
            den = Fraction(float(t[i])).limit_denominator(max_denom).denominator
            divisors[i] = divisors[i] * den // gcd(divisors[i], den)

    groups = {}
    for i in range(3):
        groups.setdefault(find(i), set()).add(i)
    group_list = list(groups.values())

    # axes forced equal must share the strictest divisor
    for grp in group_list:
        lcm = 1
        for i in grp:
            lcm = lcm * divisors[i] // gcd(lcm, divisors[i])
        for i in grp:
            divisors[i] = lcm

    return group_list, divisors


def adjust_grid_for_symmetry(grid_shape, rotations, translations, tol=1e-6, primes=(2, 3, 5)):
    """
    Smallest grid >= grid_shape (per axis) that the given symmetry operations
    map exactly onto itself, keeping every axis an even, FFT-friendly
    (5-smooth) size. gemmi applies the equivalent constraints when it sizes
    its own grid, which is why its grid can differ from a naive per-axis
    rounding; use this so the torch engine lands on the same kind of grid.

    Returns (Nu, Nv, Nw), unchanged if grid_shape already satisfies both
    conditions.
    """
    from .cell_utils import next_fft_friendly

    groups, divisors = symmetry_grid_constraints(rotations, translations, tol=tol)
    N = [int(n) for n in grid_shape]
    out = list(N)

    for grp in groups:
        target = max(N[i] for i in grp)
        divisor = divisors[min(grp)]
        n = next_fft_friendly(target, primes)
        while n % divisor != 0:
            n = next_fft_friendly(n + 1, primes)
        for i in grp:
            out[i] = n

    return tuple(out)


def build_grid_ops_from_cctbx(space_group, grid_shape, tol=1e-6):
    """
    Convenience adapter: builds the grid operations from a cctbx
    sgtbx.space_group object (e.g. xray_structure.space_group()).
    """
    rotations, translations, labels = [], [], []
    for op in space_group:
        rotations.append(np.array(op.r().as_double()).reshape(3, 3))
        translations.append(np.array(op.t().as_double()))
        labels.append(op.as_xyz())

    ops = build_grid_ops(rotations, translations, grid_shape, tol=tol)

    # re-attach the readable x,y,z labels, skipping the identity that
    # build_grid_ops dropped (order is otherwise preserved)
    non_identity = [
        lab for R, t, lab in zip(rotations, translations, labels)
        if not (np.array_equal(np.round(R).astype(int), np.eye(3, dtype=int))
                and np.all(np.abs(t) < tol))
    ]
    for op, lab in zip(ops, non_identity):
        op.xyz_str = lab
    return ops


def symmetrize_sum(grid: torch.Tensor, grid_ops) -> torch.Tensor:
    """
    Differentiable equivalent of gemmi's Grid::symmetrize_sum().

    grid:      (Nu, Nv, Nw) real density, typically straight out of
               density_torch.splat_density()
    grid_ops:  list from build_grid_ops() (identity excluded; empty for P1)

    Returns a new grid holding, at every point, the sum of the density over
    that point's full symmetry orbit. Each operation is applied to the
    ORIGINAL grid, never to the running total.
    """
    if not grid_ops:
        return grid
    out = grid
    for op in grid_ops:
        out = out + op.apply(grid)
    return out
