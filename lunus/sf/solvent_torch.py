"""
Flat / mask-based bulk solvent (Jiang & Brunger 1994), differentiable.

    F_total(h) = k_overall * exp(-h.U.h) * [ F_protein(h)
                                             + k_sol * exp(-b_sol*s^2/4) * F_mask(h) ]

where F_mask is the transform of a real-space solvent mask. The mask here is a
THRESHOLD ON A FIELD placed like the density -- solvent is wherever the field
falls below a cutoff -- rather than the geometric (vdW radius + probe, then
shrink) mask that Phenix and REFMAC build. That choice, and its consequences,
are argued in docs/solvent-design.md; the short version is that
compute_density() already produces exactly the grid a threshold needs, so
F_mask costs one extra FFT and inherits the protein density's grid and symmetry
conventions unchanged. The field is, by default, soft spheres or an
atom-presence field built from the atoms (SolventModel.set_atoms; the
benchmark ranks them in that order), or the electron density itself on
request (mask_source="density"); most of what follows was learned on the
density and holds for every field.

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
SOURCE_B_DEFAULT = 20.0      # A^2; see mask_source_weights() and SolventModel.source_b

# The fixed level of the soft-sphere field: one sphere's plateau, in "sphere
# overlaps". Measured, against calibration to gemmi's occupancy, ONLY for
# united-atom spheres at the default offset and edge, with no extra smoothing,
# on models without hydrogens -- fit_solvent_rfactor.mask_level_rule has the
# numbers. Every other mask source and setting is calibrated or given.
SPHERE_LEVEL_FIXED = 1.0
# The atom-presence field's kernels are truncated where the atom's density
# falls below this, e/A^3: the R-factor tool's --density-cutoff default, with
# which the field was measured.
SOURCE_DENSITY_CUTOFF = 1e-4

MASK_SOURCES = ("auto", "spheres", "atoms", "density")

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
    remove. It is differentiable, and carries gradients under the default
    detach_mask=False; detach_mask=True suppresses them.

    IT MUST BE A GAUSSIAN, NOT A LOW-PASS, and that is not a matter of taste.
    Truncate the filter and it RINGS: a hard low-pass at 3 A gives a minimum of
    -0.65 on 7FPV, -24% of peak, over 22% of voxels. That would not be a
    cosmetic defect, because negative density sits below ANY cutoff -- every
    ringing lobe becomes a spurious solvent pocket in the middle of the
    protein, while the mask's occupancy still looks entirely reasonable.

    A Gaussian avoids that, but the guarantee is asymptotic rather than exact,
    and the difference is worth stating because the obvious argument overshoots.
    The multiplier's inverse transform is a positive kernel, so a non-negative
    density -- which compute_density's output is by construction -- would stay
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


def mask_source_weights(elements_per_atom, coeff_table):
    """Per-atom weights that turn a density placement into an ATOM-PRESENCE
    field: 1 / f(0), the electron count the kernel integrates to, so every
    atom contributes the same unit Gaussian whatever its element.

    THE MASK SHOULD NOT FOLLOW THE ELECTRON DENSITY. A geometric mask is
    built from atom positions and radii; it does not care that a sulfur has
    16 electrons and a carbon 6, nor what each atom's B is. A threshold on
    the electron density does both, and at low resolution and high B the
    boundary it draws is far from the geometric one. Placing every atom with
    the same weight and the same B (SOURCE_B_DEFAULT) and thresholding THAT
    took the binned R-free gap to mmtbx on 20 entries at 2.6-3.0 A from
    +0.0105 to +0.0046 (every entry better; gemmi's geometric mask on the
    same grid reaches -0.0015), on 20 entries at 1.8-2.6 A from +0.0062 to
    +0.0017, and on 39 SW40 entries at 0.9-2.5 A from +0.0010 to -0.0005
    (35 better), measured 2026-11-10. Each atom is its element's IT92 sum of
    Gaussians divided by its electron count, so the field keeps a slight
    element dependence in shape, none in weight. Multiply the occupancies by
    these before placing the mask's source with uniform-B kernels. The width
    that matters is the TOTAL, source_b plus mask_blur: with the default
    smoothing of 100 A^2, totals of 70-110 A^2 measure alike at low resolution
    and 120 is marginally better on SW40, which is where 20 comes from."""
    import numpy as np

    return np.array([1.0 / (sum(coeff_table[e][0:4]) + coeff_table[e][8])
                     for e in elements_per_atom], dtype=float)


def close_mask(mask, orth_matrix, blur_b, level, taper=0.1, inv_d2_grid=None):
    """The shrink step a geometric mask has and a density threshold lacks.

    A geometric mask dilates the atoms by the probe radius and then shrinks
    the protein back, so crevices narrower than the probe stay filled while
    the outer surface returns towards the atoms. The threshold mask's
    smoothing (mask_blur) plays the probe's role but has no shrink: it moves
    the whole boundary out, and the calibration then pulls it back everywhere
    at once. This is the shrink, differentiably: the soft mask (1 = solvent)
    is smoothed by exp(-blur_b s^2/4) and thresholded again at `level`: a
    voxel is protein where no more than `level` of its neighbourhood was
    solvent, fully solvent from `level + taper` up, with a C^1 taper (in mask
    units) between -- the window lies above the level, as solvent_mask's lies
    below its cutoff. level < 0.5 grows the solvent
    (erodes the protein), the shrink; 0.5 smooths without bias; > 0.5 grows
    the protein. The first-stage cutoff is calibrated through this step
    (calibrate_cutoff's `finish`), so the FINAL occupancy meets the target
    and the first stage over-dilates by exactly what the shrink takes back.

    MEASURED AND NOT SHIPPED. On 20 entries at 2.6-3.0 A every setting tried
    (B2 20-100 A^2, level 0.2-0.5) made the gap to mmtbx worse by 0.003-0.011,
    for the same reason a wider mask_blur does: more smoothing of the boundary
    is the wrong direction. Kept, with its tests, for a later look; the
    atom-presence source (mask_source_weights) is what worked.
    """
    if blur_b <= 0.0:
        return mask
    smoothed = blur_density(mask, orth_matrix, blur_b, inv_d2_grid)
    # taper_factor is 1 well below its cutoff: feed it the protein fraction
    return taper_factor(1.0 - smoothed, 1.0 - level, taper)


def calibrate_cutoff(density, target_occupancy, taper_width_frac=0.5,
                     iterations=60, finish=None):
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

    finish, if given, maps the thresholded mask to the mask actually used
    (close_mask), and the occupancy that meets the target is the finished
    one's; it must keep the occupancy monotone in the cutoff, which a
    smoothing followed by a threshold does.
    """
    if finish is None:
        def finish(mask):
            return mask
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
        return float(mask_occupancy(finish(
            solvent_mask(density, c, taper_width_frac * c))))

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
        occ = occ_at(mid)
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
    the density calculation's scatter, which is what happens if coordinates outside [0,1) are
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

    WHICH FIELD THE MASK IS CUT FROM is mask_source:

      "auto"        (default) soft spheres when set_atoms was given the
                    atoms' names (atom_names and residue_names), else the
                    atom-presence field when it was given only their
                    elements. Decided by set_atoms alone: kernels built by
                    hand and passed as mask_kernels= go with an explicit
                    "spheres" or "atoms". Knowing NOTHING about the atoms is
                    an error, not a quiet fall back to the electron density,
                    which measured worst of the three; pass
                    mask_source="density" to ask for it.
      "spheres"     the soft-sphere field (lunus.sf.mask_spheres); the same as
                    spheres=True.
      "atoms"       the atom-presence field at source_b (default
                    SOURCE_B_DEFAULT); the same as source_b=B.
      "density"     the electron density itself, thresholded in e/A^3.

    Measured on the R-factor benchmark (binned R-free gap to mmtbx; see
    docs/solvent-design.md), united-atom spheres beat the atom-presence field,
    which beats the density, at every resolution tried.

    Fields:
      cutoff        the level below which a voxel is solvent, in the units of
                    the field it is cut from: e/A^3 for the density, sphere
                    overlaps for the spheres, unit-Gaussian overlaps for the
                    atom-presence field. None (default) is SPHERE_LEVEL_FIXED
                    for spheres on a model without hydrogens -- the one case
                    with a measured fixed level -- and an error everywhere else
                    until calibrate() sets it.
      taper_width   width of the C^1 transition, in the cutoff's units (see
                    solvent_mask). None (default) is half the cutoff.
      k_sol         solvent electron density, e/A^3
      b_sol         solvent B-factor, A^2
      k_overall     overall scale
      u_aniso       (3,3) anisotropic scaling tensor, or None
      mask_blur     B in A^2 smoothing the density before it is thresholded --
                    this model's probe radius, see the module docstring.
                    Defaults to MASK_BLUR_DEFAULT (sigma 1.13 A), or to 0.0
                    for the sphere field (with "auto", decided when the field
                    is); 0.0 on the density restores the unsmoothed threshold
                    and does NOT reproduce conventional solvent scales on
                    real data.
      detach_mask   build the mask from a detached copy of the density
                    (default FALSE). False is the consistent choice: the
                    forward pass rebuilds the mask from the current density
                    every call, so a backward pass that treats it as constant
                    is the gradient of a DIFFERENT model. Measured on 7FPV,
                    coordinate refinement against a known target reaches 4.8x
                    lower loss with the mask live. See "What detaching the mask
                    costs" in docs/solvent-design.md.

                    True is not merely worse -- it is the correct gradient of
                    the model where the mask is HELD FIXED, which is what
                    Phenix and REFMAC do per macro-cycle. That model is the
                    better long-term design and is not implementable here yet,
                    because nothing can pin a mask across calls.
      spheres       False (default). True is mask_source="spheres" (the
                    attribute records that request; with "auto" it stays
                    False whatever resolved_source() decides, and assigning
                    it, or source_b, after construction does not change the
                    source -- choose it when constructing the model): a
                    soft-sphere field for the mask (lunus.sf.mask_spheres:
                    every atom a flat-top sphere of half its Lennard-Jones
                    sigma, edge 0.5 A), from set_atoms, or placed by the caller
                    and handed in as `source`; it is placed WITHOUT
                    the pipeline blur, so pipeline_blur is not subtracted
                    from mask_blur, and mask_blur defaults to 0 (its own edge
                    is all the smoothing it wants: 25 A^2 more cost 0.003).
                    On the R-factor benchmark it beat the atom-presence field
                    at every resolution; see docs/solvent-design.md. Exclusive
                    with source_b.
      source_b      None (default). A number is mask_source="atoms" at that
                    B: a SECOND field for the mask, every atom a unit-weight
                    Gaussian at this B (mask_source_weights, uniform-B
                    kernels), from set_atoms or placed by the caller and
                    handed in as `source`; see mask_source_weights for why and
                    what it buys. The atom-presence field uses
                    SOURCE_B_DEFAULT, the measured value, when this is None.
      check_occupancy
                    measure the mask ONCE, on the first configuration, and
                    warn if it is degenerate (default True). The check costs
                    one reduction per run and catches the failure mode that
                    otherwise has no symptom at all.

    After the first apply(), last_occupancy and last_shell_voxels hold that
    measurement, for callers that would rather report it than be warned.

    THE ATOMS, for the two atom-based fields, are told to the model once with
    set_atoms(); structure_factors_one_config and _batch then place the mask's
    field themselves, from kernels built on first use and cached, and
    mask_field() places the same field for calibrate().
    """

    __slots__ = ("cutoff", "taper_width", "k_sol", "b_sol", "k_overall",
                 "u_aniso", "detach_mask", "check_occupancy",
                 "density_scale", "mask_blur", "source_b", "spheres", "source_kind",
                 "atom_elements", "atom_radii", "last_occupancy",
                 "last_shell_voxels", "_inv_d2_grid", "_inv_d2_key",
                 "_warned_blur", "_kernels", "_kernels_key", "_kernels_orth",
                 "_n_hydrogens")

    def __init__(self, cutoff=None, taper_width=None, k_sol=K_SOL_DEFAULT,
                 b_sol=B_SOL_DEFAULT, k_overall=1.0, u_aniso=None,
                 detach_mask=False, check_occupancy=True, density_scale=1.0,
                 mask_blur=None, source_b=None, spheres=False, mask_source="auto"):
        if mask_source not in MASK_SOURCES:
            raise ValueError("mask_source must be one of %s, got %r"
                             % (", ".join(MASK_SOURCES), mask_source))
        if spheres and source_b is not None:
            raise ValueError("spheres and source_b are two different mask "
                             "sources; set one")
        if spheres:
            if mask_source not in ("auto", "spheres"):
                raise ValueError("spheres=True is mask_source='spheres', which "
                                 "contradicts mask_source=%r" % (mask_source,))
            mask_source = "spheres"
        if source_b is not None:
            if source_b <= 0:
                raise ValueError("source_b must be > 0 A^2 or None")
            if mask_source not in ("auto", "atoms"):
                raise ValueError("source_b is mask_source='atoms', which "
                                 "contradicts mask_source=%r" % (mask_source,))
            mask_source = "atoms"
        if taper_width is not None and taper_width <= 0:
            raise ValueError(
                "taper_width must be > 0; a hard threshold has no usable "
                "gradient (that is the whole reason for the taper)")
        if cutoff is not None and cutoff <= 0:
            raise ValueError("cutoff must be > 0, or None")
        if mask_blur is None and mask_source != "auto":
            # the sphere field's own soft edge is all the smoothing it wants;
            # with "auto" this waits until the field is known (effective_mask_blur)
            mask_blur = 0.0 if mask_source == "spheres" else MASK_BLUR_DEFAULT
        if mask_blur is not None and mask_blur < 0:
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
        self.source_b = source_b
        self.spheres = mask_source == "spheres"
        # the requested source; "auto" is resolved per call (resolved_source)
        self.source_kind = mask_source
        self.atom_elements = None
        self.atom_radii = None
        self.last_occupancy = None
        self.last_shell_voxels = None
        self._inv_d2_grid = None
        self._inv_d2_key = None
        self._warned_blur = False
        self._kernels = None
        self._kernels_key = None
        self._kernels_orth = None
        self._n_hydrogens = None

    # ------------------------------------------------------------------
    # the atoms, and what follows from them

    def set_atoms(self, elements, atom_names=None, residue_names=None,
                  residue_ids=None):
        """Tell the model which atoms the mask is built from, once.

        elements: one symbol per atom, in the order the coordinates will
        come (any case, charges allowed: 'C', 'ZN', 'O1-'). With atom_names
        and residue_names as well -- PDB names, any case and padding -- the
        atoms become soft spheres of united-atom radius
        (mask_spheres.sphere_radii_from_names), which is what
        mask_source="auto" then uses; with elements only, the atom-presence
        field. residue_ids (optional, anything hashable per atom, e.g. a
        (chain, number, insertion code) tuple) lets the radii's hydrogen
        rule work per residue; without it a single hydrogen anywhere makes
        the whole model explicit-hydrogen (sphere_radii_from_names).

        Plain sequences, so any structure library will do; from cctbx
        scatterer labels, mask_spheres.names_from_cctbx_labels gives the
        three name columns. The kernels are built on first use, for the grid,
        cell, device and dtype of that call, and cached; calling set_atoms
        again replaces the atoms and drops the cache. It is the only way in:
        assigning atom_elements or atom_radii directly leaves the cache and
        the hydrogen count describing the old atoms.

        With the atom-presence field, structure_factors_one_config multiplies
        the occupancies by the field's weights itself; a mask_occ= passed
        alongside is taken as already weighted, as it always has been."""
        from .mask_spheres import is_hydrogen, sphere_radii_from_names

        elements = [str(e) for e in elements]
        if (atom_names is None) != (residue_names is None):
            raise ValueError("set_atoms needs atom_names AND residue_names for "
                             "the sphere field (or neither, for the atom-presence "
                             "field)")
        radii = None
        if atom_names is not None and self.source_kind != "atoms":
            radii = sphere_radii_from_names(
                elements, list(atom_names), list(residue_names),
                None if residue_ids is None else list(residue_ids))
        elif self.source_kind == "spheres":
            raise ValueError("mask_source='spheres' needs the atoms' names: "
                             "set_atoms(elements, atom_names, residue_names, "
                             "residue_ids)")
        if radii is None and self.source_kind != "density":
            # the atom-presence field needs every element's IT92 coefficients;
            # an unknown symbol fails here, not at the first structure-factor call
            from .elements import it92_coefficients
            from .mask_spheres import bare_element
            try:
                it92_coefficients(sorted({bare_element(e).capitalize() for e in elements}))
            except KeyError as e:
                raise ValueError("set_atoms: %s; the atom-presence field needs a "
                                 "scattering factor for every element" % e)
        self.atom_elements = elements
        self.atom_radii = radii
        # counted once here, not per configuration in threshold()
        self._n_hydrogens = sum(is_hydrogen(e) for e in elements)
        self._kernels = None
        self._kernels_key = None
        self._kernels_orth = None

    def resolved_source(self, mask_kernels=None):
        """The field this model's mask is cut from, "spheres", "atoms" or
        "density". With "auto" that is decided by set_atoms ALONE -- names
        give the spheres, elements only the atom-presence field -- and never
        by the kernels a caller passes: hand-built mask_kernels= go with an
        explicit mask_source="spheres" or "atoms" (spheres=True, source_b=B),
        so that calibrate() and the structure-factor call cannot read one
        field two ways. mask_kernels says only whether the caller supplies
        the field's kernels, which an explicit source accepts instead of
        set_atoms. Raises when the source needs atoms the model has not been
        given."""
        kind = self.source_kind
        if kind == "density":
            return kind
        if kind == "spheres":
            if mask_kernels is None and self.atom_radii is None:
                raise ValueError(
                    "mask_source='spheres' needs the atoms: call set_atoms("
                    "elements, atom_names, residue_names, residue_ids), or pass "
                    "the 6-tuple mask_spheres.sphere_kernels() returns as "
                    "mask_kernels=")
            return kind
        if kind == "atoms":
            if mask_kernels is None and self.atom_elements is None:
                raise ValueError(
                    "mask_source='atoms' needs the atoms: call set_atoms(elements), "
                    "or pass mask_kernels= and mask_occ= (see "
                    "solvent_torch.mask_source_weights)")
            return kind
        if self.atom_radii is not None:
            return "spheres"
        if self.atom_elements is not None:
            return "atoms"
        if mask_kernels is not None:
            raise ValueError(
                "SolventModel(mask_source='auto') takes its field from "
                "set_atoms(), not from the kernels passed: with hand-built "
                "mask_kernels=, say which field they are, mask_source='spheres' "
                "or mask_source='atoms' (spheres=True or source_b=B), or call "
                "set_atoms() instead.")
        raise ValueError(
            "SolventModel(mask_source='auto') builds its mask from the atoms, "
            "and it has not been told what they are. Either call "
            "set_atoms(elements, atom_names, residue_names, residue_ids) -- soft "
            "spheres, the measured best -- or set_atoms(elements) for the "
            "atom-presence field; or pass mask_source='density' to threshold "
            "the electron density, with its cutoff in e/A^3, as SolventModel "
            "did by default before 2026-11-18.")

    def effective_source_b(self):
        """The atom-presence field's B, A^2."""
        return SOURCE_B_DEFAULT if self.source_b is None else self.source_b

    def effective_mask_blur(self, kind):
        """mask_blur, with its default for `kind` when it was left None."""
        if self.mask_blur is not None:
            return self.mask_blur
        return 0.0 if kind == "spheres" else MASK_BLUR_DEFAULT

    def threshold(self, kind):
        """(cutoff, taper_width) for a mask cut from `kind`: as given, or as
        calibrate() set them; with cutoff None, SPHERE_LEVEL_FIXED for the
        sphere field of a model without hydrogens at the settings it was
        measured with (no mask_blur, taper half the level), and an error
        otherwise; with taper_width None, half the cutoff."""
        if self.cutoff is not None:
            taper = self.taper_width if self.taper_width is not None else 0.5 * self.cutoff
            return self.cutoff, taper
        if kind != "spheres":
            raise ValueError(
                "SolventModel(cutoff=None): the %s field has no fixed level -- "
                "what it should be depends on the system, its B factors and "
                "the grid. Give cutoff=%s, or call calibrate() with a target "
                "solvent fraction."
                % ({"atoms": "atom-presence", "density": "electron-density"}[kind],
                   " in e/A^3" if kind == "density" else ""))
        if self.atom_elements is None:
            raise ValueError(
                "SolventModel(cutoff=None) on the sphere field: the fixed level "
                "%.1f holds only for a model without hydrogens, and without "
                "set_atoms() the model cannot tell. Call set_atoms(), give "
                "cutoff=, or calibrate()." % SPHERE_LEVEL_FIXED)
        n_h = self._n_hydrogens
        if n_h is None:                 # atom_elements assigned without set_atoms
            from .mask_spheres import is_hydrogen
            n_h = sum(is_hydrogen(e) for e in self.atom_elements)
        if n_h:
            raise ValueError(
                "SolventModel(cutoff=None) on the sphere field: the model has %d "
                "hydrogens, whose spheres pack the field deeper and move its "
                "level (calibrated levels ran 1.12-1.46 on such models), so the "
                "fixed level %.1f does not apply. Call calibrate() with a target "
                "solvent fraction, or give cutoff=." % (n_h, SPHERE_LEVEL_FIXED))
        if self.effective_mask_blur(kind) != 0.0:
            raise ValueError(
                "SolventModel(cutoff=None) on the sphere field: the fixed level "
                "%.1f was measured with no extra smoothing, and mask_blur is "
                "%g A^2. Calibrate, or give cutoff=."
                % (SPHERE_LEVEL_FIXED, self.effective_mask_blur(kind)))
        taper = 0.5 * SPHERE_LEVEL_FIXED
        if self.taper_width is not None and self.taper_width != taper:
            raise ValueError(
                "SolventModel(cutoff=None) on the sphere field: the fixed level "
                "%.1f was measured with taper_width %.2f, and %g moves where the "
                "mask crosses it. Calibrate, or give cutoff=."
                % (SPHERE_LEVEL_FIXED, taper, self.taper_width))
        return SPHERE_LEVEL_FIXED, taper

    def cached_mask_kernels(self, kind, grid_shape, orth_matrix, blur=0.0):
        """(mask_kernels, weights) for `kind` from the set_atoms atoms, in the
        form structure_factors_one_config's mask_kernels= takes -- a 6-tuple
        whose last entry is the field's own element index -- and, for the
        atom-presence field, the per-atom weights the occupancies are
        multiplied by (None for the spheres). Built for this grid, cell,
        device, dtype and (atom field only) pipeline blur, and cached until
        any of those changes. orth_matrix is the torch tensor of the call."""
        import numpy as np

        if kind not in ("spheres", "atoms"):
            raise ValueError("no mask kernels for the %r source" % (kind,))
        cheap = (kind, tuple(int(g) for g in grid_shape), str(orth_matrix.device),
                 orth_matrix.dtype, float(blur) if kind == "atoms" else 0.0,
                 self.effective_source_b())
        # the same, unmodified orth tensor as last time, ON A GPU: a hit without
        # copying it to the host, which there is a synchronisation per
        # configuration. Not on the CPU, where the byte comparison below is
        # cheap and catches what the version counter misses (a tensor sharing
        # a numpy array's memory, or .data changed in place). On a GPU the
        # .data route still slips past; a cell is not rewritten that way in
        # practice.
        if (orth_matrix.device.type != "cpu"
                and self._kernels_key is not None and self._kernels_key[0] == cheap
                and self._kernels_orth is not None
                and self._kernels_orth[0] is orth_matrix
                and self._kernels_orth[1] == orth_matrix._version):
            return self._kernels
        orth_np = orth_matrix.detach().cpu().numpy()
        key = (cheap, orth_np.tobytes())
        if self._kernels_key == key:
            self._kernels_orth = (orth_matrix, orth_matrix._version)
            return self._kernels
        dev, dt = orth_matrix.device, orth_matrix.dtype
        if kind == "spheres":
            from .mask_spheres import sphere_kernels
            kernels = sphere_kernels(self.atom_radii, tuple(grid_shape),
                                     orth_np.astype(float), device=dev, dtype=dt)
            weights = None
        else:
            from .elements import it92_coefficients
            from .kernel_torch import build_atom_kernels_torch
            from .mask_spheres import bare_element
            elements = [bare_element(e).capitalize() for e in self.atom_elements]
            present = sorted(set(elements))
            coeffs = it92_coefficients(present)
            A, lam, offsets, radius, taper_w, e2i = build_atom_kernels_torch(
                elements, present, coeffs,
                np.full(len(elements), float(self.effective_source_b())), float(blur),
                tuple(grid_shape), orth_np.astype(float), cutoff=SOURCE_DENSITY_CUTOFF,
                device=dev, dtype=dt)
            idx = torch.tensor([e2i[e] for e in elements], dtype=torch.long, device=dev)
            kernels = (A, lam, offsets, radius, taper_w, idx)
            weights = torch.as_tensor(mask_source_weights(elements, coeffs),
                                      device=dev, dtype=dt)
        self._kernels = (kernels, weights)
        self._kernels_key = key
        self._kernels_orth = (orth_matrix, orth_matrix._version)
        return self._kernels

    def _check_atom_count(self, n_atoms):
        """Raise unless set_atoms was told n_atoms atoms: its kernels and
        weights are per atom, in the coordinates' order."""
        if self.atom_elements is not None and len(self.atom_elements) != n_atoms:
            raise ValueError(
                "set_atoms() was told %d atoms but the coordinates have %d; the "
                "mask's kernels are per atom, in the coordinates' order"
                % (len(self.atom_elements), n_atoms))

    def mask_field(self, frac_coords, occ, grid_shape, orth_matrix, blur=0.0,
                   grid_ops=None):
        """The mask's field for one configuration, from the set_atoms atoms:
        placed on the grid and symmetry-expanded as
        structure_factors_one_config places it (uncompiled; a compiled call
        agrees to roundoff). What calibrate() needs as
        `source` when the mask is not cut from the density:

            field = model.mask_field(frac, occ, grid, orth, blur, grid_ops)
            model.calibrate(density, orth, target, pipeline_blur=blur,
                            source=field)

        blur is the pipeline's FFT-artifact blur, the same number passed as
        blur= to the structure-factor call (the atom-presence field carries
        it, the sphere field does not). Raises for the density source, which
        has no field of its own. Not for a supercell= model, whose field the
        structure-factor call places on the supercell's grid before folding;
        calibrate such a model on a field placed the same way."""
        from .density_torch import compute_density
        from .symmetry_torch import symmetrize_sum

        kind = self.resolved_source()
        if kind == "density":
            raise ValueError("the density source has no field of its own: "
                             "calibrate on the density itself")
        self._check_atom_count(frac_coords.shape[0])
        kernels, weights = self.cached_mask_kernels(kind, grid_shape, orth_matrix, blur)
        A, lam, offsets, radius, taper_w, idx = kernels
        field = compute_density(frac_coords, idx, occ if weights is None else occ * weights,
                                A, lam, offsets, radius, tuple(grid_shape), orth_matrix,
                                taper_w, compile_core=False)
        if grid_ops:
            field = symmetrize_sum(field, grid_ops)
        return field

    def _cached_inv_d2(self, density, orth_matrix):
        """The frequency grid for the blur, built once per (shape, device,
        dtype). It is a full extra grid, so it is not built at all when
        mask_blur is 0."""
        key = (tuple(density.shape), str(density.device), density.dtype)
        if self._inv_d2_key != key:
            self._inv_d2_grid = grid_inv_d2(density.shape, orth_matrix)
            self._inv_d2_key = key
        return self._inv_d2_grid

    def field_kind(self, source):
        """The source kind for a mask cut from `source` (None: the density),
        the decision mask_source(), build_mask() and calibrate() share. With
        "auto" and a field handed in, the atoms told to set_atoms say which
        field it is, as they do for structure_factors_one_config."""
        if source is not None and self.source_kind != "auto":
            return self.source_kind
        if source is None and self.source_kind in ("spheres", "atoms"):
            raise ValueError(
                "SolventModel.%s is set, so the mask expects its field as "
                "`source`; structure_factors_one_config builds it from "
                "set_atoms() or mask_kernels=, and mask_field() builds it for "
                "calibrate() (see %s)"
                % (("spheres", "mask_spheres.sphere_kernels")
                   if self.source_kind == "spheres"
                   else ("source_b", "mask_source_weights")))
        kind = self.resolved_source()
        if source is None and kind != "density":
            raise ValueError(
                "SolventModel's mask is cut from the %s field, which must be "
                "handed in as `source`: mask_field() places it, for calibrate() "
                "and build_mask()" % kind)
        return kind

    def mask_source(self, density, orth_matrix, pipeline_blur=0.0, source=None,
                    _kind=None):
        """The grid the mask is thresholded from: `source` when given (the
        atom-presence field of source_b, placed by the caller), else the
        density; detached if detach_mask, and smoothed to a TOTAL of mask_blur.

        pipeline_blur is the blur already folded into the density by the
        caller -- the FFT-artifact trick, where atoms are placed on the grid with an
        extra B and compute_fcalc removes it again in reciprocal space. THAT
        IS THE SAME OPERATION AS mask_blur. Adding B to an atomic form factor
        convolves that atom with a Gaussian, density is a linear sum of atoms,
        and convolution is linear, so placing density with blur=B is exactly
        convolving the grid by B -- verified to 1e-4 of peak, the residual
        being the real-space taper truncation.

        So the two ADD, silently, and a pipeline running blur=50 with the
        default mask_blur=100 would give the mask 150 A^2 of smoothing: 1.5x
        the probe radius that was calibrated against two crystals. mask_blur is
        therefore a TOTAL, and only the remainder is applied here. The mask is
        then invariant to the blur trick, which is what you want from a
        numerical device that is supposed to cancel out of the answer.

        Public because CALIBRATION MUST SEE THE SAME GRID. Calibrating a cutoff
        on the raw density and then thresholding the blurred one asks for an
        occupancy from one distribution and applies it to another -- measured
        on 7FPV, that misses the requested occupancy by more than a third.
        calibrate() below does this correctly; anything hand-rolled must too.
        """
        kind = self.field_kind(source) if _kind is None else _kind
        mask_blur = self.effective_mask_blur(kind)
        field = density if source is None else source
        rho = field.detach() if self.detach_mask else field
        if kind == "spheres":
            pipeline_blur = 0.0             # the sphere field carries none
        remaining = mask_blur - pipeline_blur
        if remaining <= 0.0:
            if pipeline_blur > mask_blur and not self._warned_blur:
                self._warned_blur = True
                warnings.warn(
                    "the pipeline blur (%.4g A^2) already exceeds mask_blur "
                    "(%.4g), so the mask is smoother than asked for and "
                    "nothing here can sharpen it back -- the blur is baked "
                    "into the density. Either lower the pipeline blur or "
                    "accept the wider effective probe."
                    % (pipeline_blur, mask_blur),
                    SolventMaskWarning, stacklevel=3)
            return rho
        return blur_density(rho, orth_matrix, remaining,
                            self._cached_inv_d2(rho, orth_matrix))

    def build_mask(self, density, orth_matrix, pipeline_blur=0.0, source=None,
                   _kind=None):
        """density -> the solvent mask, including the one-time occupancy check.

        The single place a mask is built. structure_factor_torch's supercell
        path used to inline its own copy of this, which is how a mask_blur
        added in one place would have silently not applied in the other.
        `source`, when given, is thresholded instead of the density (the
        sphere or atom-presence field; mask_field() places it).
        """
        kind = self.field_kind(source) if _kind is None else _kind
        cutoff, taper = self.threshold(kind)
        mask = solvent_mask(
            self.mask_source(density, orth_matrix, pipeline_blur, source, _kind=kind),
            cutoff, taper)
        if self.check_occupancy and self.last_occupancy is None:
            self._check(mask, cutoff, kind)
        return mask

    def calibrate(self, density, orth_matrix, target_occupancy,
                  taper_width_frac=0.5, pipeline_blur=0.0, source=None):
        """Set cutoff and taper_width so the mask calls target_occupancy of the
        cell solvent, measured on the grid the mask will actually see.

        Returns the cutoff. Run once at setup, on a representative
        configuration -- the cutoff is a calibration and not a constant, but it
        is not so unstable that it needs redoing per frame.

        For a mask not cut from the density, `source` is its field:
        mask_field() places it from the set_atoms atoms.
        """
        grid = self.mask_source(density, orth_matrix, pipeline_blur, source)
        self.cutoff = calibrate_cutoff(grid, target_occupancy,
                                       taper_width_frac=taper_width_frac)
        self.taper_width = taper_width_frac * self.cutoff
        return self.cutoff

    def _check(self, mask, cutoff, kind):
        """Once per model. Reads back two scalars from the device, which is a
        synchronisation, so it is not done per configuration."""
        units = "e/A^3" if kind == "density" else "in the %s field's units" % kind
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
                "%s)." % (occ, cutoff, units),
                SolventMaskWarning, stacklevel=3)
        elif occ > OCCUPANCY_MAX:
            warnings.warn(
                "solvent mask covers almost the whole cell (occupancy %.4f): "
                "hardly anything is being called protein. Check cutoff "
                "(=%.4g, %s): it is not transferable between B-factor "
                "conventions, grids or systems, and must be calibrated for "
                "this one." % (occ, cutoff, units),
                SolventMaskWarning, stacklevel=3)

    def apply(self, density, F_protein, cell_volume, hkl, orth_matrix,
              density_scale=1.0, pipeline_blur=0.0, source=None, _kind=None):
        """
        density must ALREADY be symmetry-expanded -- see the module docstring;
        so must `source`, the mask's field when it is not the density.

        Returns (F_total, mask, F_mask): the mask so callers can report
        mask_occupancy() without recomputing it, and F_mask -- f_solvent of
        that mask, the term F_total was built from, WITHOUT the density_scale
        factors -- so a caller that wants the solvent term on its own does not
        pay for a second FFT.
        """
        mask = self.build_mask(density, orth_matrix, pipeline_blur, source, _kind=_kind)
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
        return F, mask, F_mask

    def __repr__(self):
        return ("SolventModel(mask_source=%r, cutoff=%r, taper_width=%r, k_sol=%r, "
                "b_sol=%r, k_overall=%r, u_aniso=%s, detach_mask=%r, source_b=%r, "
                "atoms=%s, occupancy=%s)"
                % (self.source_kind, self.cutoff, self.taper_width, self.k_sol,
                   self.b_sol, self.k_overall,
                   "None" if self.u_aniso is None else "set", self.detach_mask,
                   self.source_b,
                   "none" if self.atom_elements is None else "%d%s" % (
                       len(self.atom_elements),
                       " named" if self.atom_radii is not None else ""),
                   "unmeasured" if self.last_occupancy is None
                   else "%.4f" % self.last_occupancy))
