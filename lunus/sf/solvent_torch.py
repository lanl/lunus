"""
Flat / mask-based bulk solvent (Jiang & Brunger 1994), differentiable.

    F_total(h) = k_overall * exp(-h.U.h) * [ F_protein(h)
                                             + k_sol * exp(-b_sol*s^2/4) * F_mask(h) ]

where F_mask is the transform of a real-space solvent mask. The mask here is a
DENSITY THRESHOLD -- solvent is wherever the model's density falls below a
cutoff -- rather than the geometric (vdW radius + probe, then shrink) mask that
Phenix and REFMAC build. That choice, and its consequences, are argued in
docs/solvent-design.md; the short version is that splat_density() already
produces exactly the grid a threshold needs, so F_mask costs one extra FFT and
inherits the protein density's grid and symmetry conventions unchanged.

The model is EXPLICIT ORDERED WATERS PLUS A MASK FOR THE REMAINDER, the same
division crystallographic refinement makes. A threshold mask suits that well:
explicit waters raise the density where they sit and so exclude themselves from
the solvent region, with no special handling of the boundary between the two
models.

Four things in here are not obvious, all of them from that design note:

  * THE DENSITY IS SMOOTHED BEFORE IT IS THRESHOLDED, and that is not a
    cosmetic filter -- it is this model's stand-in for the PROBE RADIUS a
    geometric mask is built with. Phenix and REFMAC roll a solvent-sized probe
    over the vdW surface and then shrink the result, which smooths the boundary
    and closes the crevices between neighbouring atoms. A bare density
    threshold has no equivalent and follows every one of them, which measured
    on real data is not a small effect: it puts ~1.5x too much amplitude into
    F_mask beyond 2 A, uncorrelated with a geometric mask's, and drives a fit
    of k_sol/b_sol to 0.554/199 against the conventional 0.35/46. See
    mask_blur, and "What the R-factor found" in docs/solvent-design.md.

  * THE MASK MUST BE BUILT FROM THE POST-SYMMETRIZATION DENSITY.
    symmetrize_sum() SUMS over the symmetry orbit, so a mask built first and
    summed after reaches |orbit| rather than 1 and means nothing. Callers in
    structure_factor_torch.py do this correctly; anyone calling solvent_mask()
    directly must too.

  * THE MASK SHOULD NOT SEE B-FACTORS. Raising B spreads density and lowers it
    near the surface, so a mask that differentiates through B grows into the
    protein as B rises -- backwards, since the solvent envelope is a property
    of the molecular boundary, not of thermal motion. It also hands an
    optimizer a route to low-resolution amplitudes through surface B alone.
    B is not in the autograd graph today, so this costs nothing now; the
    detach_mask default is what keeps it that way when it is.

  * OCCUPANCY IS THE DIAGNOSTIC THAT MATTERS. A model whose atoms already fill
    the cell -- an all-atom MD frame with explicit bulk water, say -- produces
    an empty mask, F_mask ~ 0, and a feature that silently does nothing. The
    atom selection cannot detect this; the mask can. See mask_occupancy().
"""

import math
import warnings

import torch

from .density_torch import taper_factor
from .structure_factor_torch import compute_fcalc, reciprocal_inv_d2

# Phenix's starting values, and the conventional defaults elsewhere. They are
# normally REFINED per dataset; fixed values are a starting model, not a fit.
K_SOL_DEFAULT = 0.35     # e/A^3
B_SOL_DEFAULT = 46.0     # A^2

# The mask's probe radius, as a B factor in A^2 (sigma = sqrt(B / 8 pi^2), so
# this is sigma = 1.13 A). Unlike the cutoff, this is NOT a per-structure
# calibration: it is a property of the solvent being modelled, the same way
# Phenix's solvent_radius is, and it is expressed as a physical length so it
# transfers across grids unchanged.
#
# 1.13 A is where R-work turns over on 7FPV against deposited amplitudes, and
# it is arrived at without being told what it should be -- which makes its
# agreement with Phenix's 1.11 A solvent radius a corroboration rather than a
# fit. mask_blur=0.0 restores the unsmoothed threshold, and changes results.
MASK_BLUR_DEFAULT = 100.0    # A^2

# Occupancies outside this band are reported. The band is deliberately much
# wider than the 0.3-0.7 a protein crystal actually shows: the point is to
# catch a model that is doing NOTHING, not to second-guess an unusual but
# deliberate one. A small molecule alone in a large cell legitimately sits
# near 0.9.
OCCUPANCY_MIN = 0.05
OCCUPANCY_MAX = 0.99


class SolventMaskWarning(UserWarning):
    """The solvent mask is degenerate: empty, or covering the whole cell.

    Raised once per SolventModel rather than per configuration. Escalate it
    with warnings.simplefilter("error", SolventMaskWarning) to fail instead --
    worth doing in a pipeline, since both degenerate cases produce numbers
    rather than exceptions.
    """


def grid_inv_d2(grid_shape, orth_matrix):
    """s^2 for every voxel of the reciprocal grid, shape `grid_shape`.

    Through the FULL reciprocal metric rather than 1/a^2 + 1/b^2 + 1/c^2 --
    those agree only for an orthogonal cell and silently do not for a
    monoclinic or triclinic one.
    """
    Nu, Nv, Nw = grid_shape
    dev, dt = orth_matrix.device, orth_matrix.dtype
    inv_orth = torch.linalg.inv(orth_matrix)
    fu = torch.fft.fftfreq(Nu, d=1.0 / Nu, device=dev, dtype=dt)
    fv = torch.fft.fftfreq(Nv, d=1.0 / Nv, device=dev, dtype=dt)
    fw = torch.fft.fftfreq(Nw, d=1.0 / Nw, device=dev, dtype=dt)
    H = torch.stack(torch.meshgrid(fu, fv, fw, indexing="ij"), dim=-1)
    s_cart = H @ inv_orth
    return (s_cart * s_cart).sum(-1)


def blur_density(density, orth_matrix, blur_b, inv_d2_grid=None):
    """
    density convolved with a Gaussian, as density * exp(-B s^2 / 4).

    THE PROBE RADIUS, see the module docstring. B is in A^2, the same
    convention as b_sol and as compute_fcalc's blur, so the real-space Gaussian
    has sigma = sqrt(B / (8 pi^2)) -- B = 100 is sigma = 1.13 A.

    Done in reciprocal space because on a periodic grid that is exact and
    costs one FFT pair, where a real-space convolution would need a truncated
    kernel and would reintroduce exactly the boundary artifacts this is here to
    remove. It is differentiable, though the default detach_mask means it does
    not normally carry gradients.

    IT MUST BE A GAUSSIAN, NOT A LOW-PASS, and that is not a matter of taste.
    Truncate the filter and it RINGS: a hard low-pass at 3 A gives a minimum of
    -0.65 on 7FPV, -24% of peak, over 22% of voxels. That would not be a
    cosmetic defect, because negative density sits below ANY cutoff -- every
    ringing lobe becomes a spurious solvent pocket in the middle of the
    protein, while the mask's occupancy still looks entirely reasonable.

    A Gaussian avoids that, but the guarantee is asymptotic rather than exact,
    and the difference is worth stating because the obvious argument overshoots.
    The multiplier's inverse transform is a positive kernel, so a non-negative
    density -- which splat_density's output is by construction -- would stay
    non-negative if the DFT summed over ALL frequencies. It does not: it stops
    at Nyquist, and discarding that tail is a truncation, in miniature. So the
    undershoot is bounded by how much Gaussian is left at the grid edge, and it
    disappears once the grid resolves sigma. Measured, same 40-atom model:

        voxel 1.25 A   min/peak  -2.9e-03      (sigma barely sampled)
        voxel 0.83 A             -1.9e-05
        voxel 0.62 A             -3.9e-09
        voxel 0.42 A             +1.3e-08      (roundoff; sign is now positive)

    In practice this never matters: sigma = 1.13 A needs voxels below ~0.6 A to
    be resolved, grid_shape_for_resolution gives d_min/3, and so any d_min
    finer than ~1.8 A is already there -- 7FPV at 1.04 A has 0.34 A voxels and a
    blurred minimum of +2.7e-6, positive in float32 and float64 alike. On a
    deliberately coarse grid the ripples are still ~2 orders of magnitude below
    a cutoff sitting at ~20% of peak. tests/test_solvent.py pins both the bound
    and its grid dependence.

    Pass inv_d2_grid to reuse a cached frequency grid; it depends only on the
    grid shape and the cell, so recomputing it per configuration is pure waste.
    """
    if blur_b == 0.0:
        return density
    if inv_d2_grid is None:
        inv_d2_grid = grid_inv_d2(density.shape, orth_matrix)
    weight = torch.exp(-0.25 * blur_b * inv_d2_grid).to(density.dtype)
    return torch.fft.ifftn(torch.fft.fftn(density) * weight).real


def solvent_mask(density, cutoff, taper_width):
    """
    (Nu,Nv,Nw) density -> smooth solvent mask in [0, 1], same shape.

    1 where the density is well below `cutoff` (solvent), 0 where it is at or
    above it (protein), with a C^1 raised-cosine transition of width
    `taper_width` between. A hard `density < cutoff` would have zero gradient
    almost everywhere and an undefined one at the boundary; this is the same
    fix, and the same function, that density_torch uses for the atomic cutoff.

    CUTOFF AND TAPER_WIDTH ARE IN DENSITY UNITS (e/A^3), not Angstroms. The
    width is structurally the same kind of hyperparameter as the atomic
    taper_width and numerically nothing like it -- do not carry 0.1 across.

    Sizing the width: d(mask)/d(rho) = (pi/2w) sin(pi x), so the gradient
    scales as 1/w and lives entirely in the shell where
    cutoff - w < rho < cutoff. Too narrow and that shell is thinner than a
    voxel, which makes gradients depend on which voxels happen to land in it.
    Aim for a shell 2-3 voxels thick; shell_voxels() measures it.

    Differentiable in `density`, and therefore in whatever `density` was
    differentiable in -- see the module docstring on why the caller should
    usually hand this a detached grid.
    """
    return taper_factor(density, cutoff, taper_width)


def mask_occupancy(mask):
    """
    Fraction of the cell the mask calls solvent: a scalar in [0, 1].

    This is the sanity check on the whole model, and it is cheap enough to run
    every time:

        ~0.3-0.7   a plausible crystal solvent content; the model is working
        ~0         the atoms already fill the cell, so F_mask ~ 0 and the
                   solvent contribution is nothing at all -- SILENTLY. An
                   all-atom MD frame with explicit bulk water does this.
        ~1         cutoff far too high, or almost nothing was selected

    The atom selection cannot tell these apart ("water is selected" is not the
    failure; a fully solvated box is), which is why the check is on the mask.
    """
    return mask.mean()


def shell_voxels(mask):
    """
    Number of voxels inside the taper shell, i.e. strictly between solvent and
    protein. The count the taper width should be chosen to control: too few and
    the mask is effectively a hard threshold sampled on a grid, with gradients
    and -- for a variance observable like diffuse -- frame-to-frame noise that
    depend on grid alignment rather than on the structure.
    """
    return int(((mask > 1e-6) & (mask < 1.0 - 1e-6)).sum())


def calibrate_cutoff(density, target_occupancy, taper_width_frac=0.5,
                     iterations=60):
    """
    The cutoff at which the mask calls `target_occupancy` of the cell solvent.

    THE CUTOFF IS A CALIBRATION, NOT A CONSTANT. It is not transferable between
    systems, B-factor conventions or grids -- measured, it moves by 21x between
    B = 20 and B = 60 on one structure -- and, more surprisingly, the fraction
    of peak density it corresponds to depends on how tightly the model packs:

        sparse model, true vacuum between atoms   ~1e-3 of peak
        a real protein crystal (4WOR, P4_1)       ~0.11 of peak

    because a real structure has no vacuum, only overlapping tails, so a much
    higher threshold is needed to carve out the same volume. Picking a fixed
    fraction of peak instead measured twice as far from gemmi's mask on 4WOR
    (R 0.053 against 0.025) AND gave a shell an order of magnitude thinner.

    target_occupancy is what a crystal's solvent content is expected to be --
    0.3-0.7 for a protein, from the Matthews coefficient, or the occupancy of a
    geometric mask if one is available to match.

    Bisection on a monotone quantity, so it is deterministic and needs no
    gradients; run once at setup, not per configuration.
    """
    if not 0.0 < target_occupancy < 1.0:
        raise ValueError("target_occupancy must be in (0, 1), got %r"
                         % (target_occupancy,))

    # The reachable range first. It is NOT (0, 1): a model with real vacuum
    # between its atoms has an occupancy FLOOR at the vacuum fraction, since
    # no positive cutoff can call a zero-density voxel protein. Bisecting past
    # that returns the smallest cutoff tried and an occupancy nothing like the
    # one asked for -- silently, which is how this was found.
    lo, hi = 1e-12, float(density.max())

    def occ_at(c):
        return float(mask_occupancy(
            solvent_mask(density, c, taper_width_frac * c)))

    floor, ceiling = occ_at(lo), occ_at(hi)
    if not floor - 1e-6 <= target_occupancy <= ceiling + 1e-6:
        raise ValueError(
            "occupancy %.4f is out of reach for this density: it can only be "
            "driven between %.4f (cutoff -> 0) and %.4f (cutoff = peak). A "
            "floor well above zero means the model has genuine vacuum in it -- "
            "%.1f%% of voxels have no density at all -- so no threshold can "
            "call that region protein. Either the target is wrong for this "
            "system, or the model is missing the atoms that should fill it."
            % (target_occupancy, floor, ceiling, 100 * floor))

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        occ = float(mask_occupancy(
            solvent_mask(density, mid, taper_width_frac * mid)))
        if occ > target_occupancy:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def f_solvent(mask, cell_volume, hkl, orth_matrix=None):
    """
    Solvent mask grid -> (n_refl,) complex F_mask(hkl).

    Just compute_fcalc on the mask: the mask lives on the same grid as the
    density, so the transform, the Miller-index extraction and the wrapping
    conventions are identical and cannot drift apart. No blur -- the mask is
    not a density and has no B to undo.

    Units: compute_fcalc returns ifftn(grid) * cell_volume, so for a
    dimensionless mask F_mask carries a volume, and k_sol * F_mask is in
    electrons (e/A^3 * A^3). That is why f_total needs no extra normalization.
    """
    return compute_fcalc(mask, cell_volume, hkl, orth_matrix, blur=0.0)


def f_total(F_protein, F_mask, inv_d2, k_sol=K_SOL_DEFAULT, b_sol=B_SOL_DEFAULT,
            k_overall=1.0, u_aniso=None, hkl=None, density_scale=1.0):
    """
    Combine the protein and solvent contributions:

        F_total = k_overall * exp(-h.U.h) * [ F_protein
                                              + k_sol*exp(-b_sol*s^2/4)*F_mask ]

    inv_d2 is s^2 = 1/d^2 per reflection, from reciprocal_inv_d2(). The
    exp(-b_sol * inv_d2 / 4) convention matches compute_fcalc's blur term
    (exp(+blur * 0.25 * inv_d2)) exactly, so the two cannot disagree by the
    factor of four that this formula invites.

    u_aniso, when given, is the (3,3) symmetric anisotropic scaling tensor in
    the same reciprocal basis as hkl, which must then also be given. Omit both
    for isotropic scaling, which is the usual starting point.

    k_sol, b_sol and k_overall may be plain floats or 0-d tensors; passing
    tensors with requires_grad=True is what makes them refinable later. They
    are NOT refined here, and fixed values will not reproduce a published
    refined R -- see docs/solvent-design.md on validation.

    density_scale is how many unit cells' worth of contents F_protein
    represents, and it multiplies k_sol. It exists because the mask is
    dimensionless -- 1 in solvent, whatever the density -- while F_protein is
    not: fold an N-cell supercell into one cell and the protein density is N
    times a single cell's, so a k_sol of 0.35 e/A^3 buys N times too little
    solvent. Measured on the 7FPV trajectory folded from its 2x2x2 box, with
    ~6 cells' worth of atoms in the cell: |F_total|/|F_protein| came out 0.9970
    against 0.9176 for the same model as a single crystallographic cell.

    The pipeline sets it from the supercell factors when it knows them. It
    CANNOT know them when the folding happens implicitly, through the modulo in
    the splat's scatter, which is what happens if coordinates outside [0,1) are
    passed with no supercell= argument -- pass density_scale explicitly there.
    """
    k_eff = _as_tensor(k_sol, inv_d2) * _as_tensor(density_scale, inv_d2)
    scale = torch.exp(-0.25 * _as_tensor(b_sol, inv_d2) * inv_d2)
    F = F_protein + (k_eff * scale).to(F_protein.dtype) * F_mask

    if u_aniso is not None:
        if hkl is None:
            raise ValueError("u_aniso requires hkl to evaluate exp(-h.U.h)")
        h = hkl.to(u_aniso.dtype)
        # h.U.h per reflection, without forming the (n,3,3) outer products.
        hUh = ((h @ u_aniso) * h).sum(-1)
        F = F * torch.exp(-hUh).to(F.dtype)

    return _as_tensor(k_overall, inv_d2).to(F.dtype) * F


def _as_tensor(x, like):
    """Accept floats or tensors for the scale parameters, without promoting a
    float32 pipeline to float64 through a Python scalar."""
    if torch.is_tensor(x):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


class SolventModel:
    """
    The solvent settings, as one object, so that structure_factors_one_config()
    and _batch() grow a single `solvent=` argument rather than six.

    solvent=None everywhere means today's behaviour, bit for bit: the solvent
    code is not merely multiplied by zero, it does not run.

    Fields:
      cutoff        density below which a voxel is solvent, e/A^3
      taper_width   width of the C^1 transition, e/A^3 (see solvent_mask)
      k_sol         solvent electron density, e/A^3
      b_sol         solvent B-factor, A^2
      k_overall     overall scale
      u_aniso       (3,3) anisotropic scaling tensor, or None
      mask_blur     B in A^2 smoothing the density before it is thresholded --
                    this model's probe radius, see the module docstring.
                    Defaults to MASK_BLUR_DEFAULT (sigma 1.13 A); 0.0 restores
                    the unsmoothed threshold and does NOT reproduce
                    conventional solvent scales on real data.
      detach_mask   build the mask from a detached copy of the density
                    (default True). Keeps the solvent envelope out of the
                    gradient of every parameter the density depends on --
                    in particular B, once B is differentiable. See the module
                    docstring.
      check_occupancy
                    measure the mask ONCE, on the first configuration, and
                    warn if it is degenerate (default True). The check costs
                    one reduction per run and catches the failure mode that
                    otherwise has no symptom at all.

    After the first apply(), last_occupancy and last_shell_voxels hold that
    measurement, for callers that would rather report it than be warned.
    """

    __slots__ = ("cutoff", "taper_width", "k_sol", "b_sol", "k_overall",
                 "u_aniso", "detach_mask", "check_occupancy",
                 "density_scale", "mask_blur", "last_occupancy",
                 "last_shell_voxels", "_inv_d2_grid", "_inv_d2_key")

    def __init__(self, cutoff, taper_width, k_sol=K_SOL_DEFAULT,
                 b_sol=B_SOL_DEFAULT, k_overall=1.0, u_aniso=None,
                 detach_mask=True, check_occupancy=True, density_scale=1.0,
                 mask_blur=MASK_BLUR_DEFAULT):
        if taper_width <= 0:
            raise ValueError(
                "taper_width must be > 0; a hard threshold has no usable "
                "gradient (that is the whole reason for the taper)")
        if mask_blur < 0:
            raise ValueError("mask_blur must be >= 0 A^2; a negative B "
                             "sharpens the density and amplifies exactly the "
                             "boundary roughness the blur exists to remove")
        self.cutoff = cutoff
        self.taper_width = taper_width
        self.k_sol = k_sol
        self.b_sol = b_sol
        self.k_overall = k_overall
        self.u_aniso = u_aniso
        self.detach_mask = detach_mask
        self.check_occupancy = check_occupancy
        self.density_scale = density_scale
        self.mask_blur = mask_blur
        self.last_occupancy = None
        self.last_shell_voxels = None
        self._inv_d2_grid = None
        self._inv_d2_key = None

    def _cached_inv_d2(self, density, orth_matrix):
        """The frequency grid for the blur, built once per (shape, device,
        dtype). It is a full extra grid, so it is not built at all when
        mask_blur is 0."""
        key = (tuple(density.shape), str(density.device), density.dtype)
        if self._inv_d2_key != key:
            self._inv_d2_grid = grid_inv_d2(density.shape, orth_matrix)
            self._inv_d2_key = key
        return self._inv_d2_grid

    def mask_source(self, density, orth_matrix):
        """The grid the mask is thresholded from: detached if detach_mask, and
        smoothed by mask_blur.

        Public because CALIBRATION MUST SEE THE SAME GRID. Calibrating a cutoff
        on the raw density and then thresholding the blurred one asks for an
        occupancy from one distribution and applies it to another -- measured
        on 7FPV, that misses the requested occupancy by more than a third.
        calibrate() below does this correctly; anything hand-rolled must too.
        """
        rho = density.detach() if self.detach_mask else density
        if self.mask_blur == 0.0:
            return rho
        return blur_density(rho, orth_matrix, self.mask_blur,
                            self._cached_inv_d2(rho, orth_matrix))

    def build_mask(self, density, orth_matrix):
        """density -> the solvent mask, including the one-time occupancy check.

        The single place a mask is built. structure_factor_torch's supercell
        path used to inline its own copy of this, which is how a mask_blur
        added in one place would have silently not applied in the other.
        """
        mask = solvent_mask(self.mask_source(density, orth_matrix),
                            self.cutoff, self.taper_width)
        if self.check_occupancy and self.last_occupancy is None:
            self._check(mask)
        return mask

    def calibrate(self, density, orth_matrix, target_occupancy,
                  taper_width_frac=0.5):
        """Set cutoff and taper_width so the mask calls target_occupancy of the
        cell solvent, measured on the grid the mask will actually see.

        Returns the cutoff. Run once at setup, on a representative
        configuration -- the cutoff is a calibration and not a constant, but it
        is not so unstable that it needs redoing per frame.
        """
        source = self.mask_source(density, orth_matrix)
        self.cutoff = calibrate_cutoff(source, target_occupancy,
                                       taper_width_frac=taper_width_frac)
        self.taper_width = taper_width_frac * self.cutoff
        return self.cutoff

    def _check(self, mask):
        """Once per model. Reads back two scalars from the device, which is a
        synchronisation, so it is not done per configuration."""
        self.last_occupancy = float(mask_occupancy(mask))
        self.last_shell_voxels = shell_voxels(mask)
        occ = self.last_occupancy

        if occ < OCCUPANCY_MIN:
            warnings.warn(
                "solvent mask is essentially empty (occupancy %.4f): the "
                "atoms already fill the cell, so F_mask is ~0 and the solvent "
                "term contributes NOTHING -- silently. This is what an "
                "all-atom model with explicit bulk water does. Either exclude "
                "the bulk solvent from the atom set, or lower cutoff (=%.4g, "
                "in e/A^3)." % (occ, self.cutoff),
                SolventMaskWarning, stacklevel=3)
        elif occ > OCCUPANCY_MAX:
            warnings.warn(
                "solvent mask covers almost the whole cell (occupancy %.4f): "
                "hardly anything is being called protein. Check cutoff "
                "(=%.4g, in e/A^3): it is not transferable between B-factor "
                "conventions, grids or systems, and must be calibrated for "
                "this one." % (occ, self.cutoff),
                SolventMaskWarning, stacklevel=3)

    def apply(self, density, F_protein, cell_volume, hkl, orth_matrix,
              density_scale=1.0):
        """
        density must ALREADY be symmetry-expanded -- see the module docstring.

        Returns (F_total, mask), the mask being handed back so callers can
        report mask_occupancy() without recomputing it.
        """
        mask = self.build_mask(density, orth_matrix)
        F_mask = f_solvent(mask, cell_volume, hkl, orth_matrix)
        inv_d2 = reciprocal_inv_d2(orth_matrix, hkl)
        F = f_total(
            F_protein, F_mask, inv_d2,
            k_sol=self.k_sol, b_sol=self.b_sol, k_overall=self.k_overall,
            u_aniso=self.u_aniso, hkl=hkl,
            # The caller's factor (the supercell fold, which it knows) times
            # the model's own (anything the caller cannot know).
            density_scale=self.density_scale * density_scale,
        )
        return F, mask

    def __repr__(self):
        return ("SolventModel(cutoff=%r, taper_width=%r, k_sol=%r, b_sol=%r, "
                "k_overall=%r, u_aniso=%s, detach_mask=%r, occupancy=%s)"
                % (self.cutoff, self.taper_width, self.k_sol, self.b_sol,
                   self.k_overall,
                   "None" if self.u_aniso is None else "set", self.detach_mask,
                   "unmeasured" if self.last_occupancy is None
                   else "%.4f" % self.last_occupancy))
