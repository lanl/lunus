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

Three things in here are not obvious, all of them from that design note:

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

import torch

from .density_torch import taper_factor
from .structure_factor_torch import compute_fcalc, reciprocal_inv_d2

# Phenix's starting values, and the conventional defaults elsewhere. They are
# normally REFINED per dataset; fixed values are a starting model, not a fit.
K_SOL_DEFAULT = 0.35     # e/A^3
B_SOL_DEFAULT = 46.0     # A^2


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
            k_overall=1.0, u_aniso=None, hkl=None):
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
    """
    scale = torch.exp(-0.25 * _as_tensor(b_sol, inv_d2) * inv_d2)
    F = F_protein + (_as_tensor(k_sol, inv_d2) * scale).to(F_protein.dtype) * F_mask

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
      detach_mask   build the mask from a detached copy of the density
                    (default True). Keeps the solvent envelope out of the
                    gradient of every parameter the density depends on --
                    in particular B, once B is differentiable. See the module
                    docstring.
    """

    __slots__ = ("cutoff", "taper_width", "k_sol", "b_sol", "k_overall",
                 "u_aniso", "detach_mask")

    def __init__(self, cutoff, taper_width, k_sol=K_SOL_DEFAULT,
                 b_sol=B_SOL_DEFAULT, k_overall=1.0, u_aniso=None,
                 detach_mask=True):
        if taper_width <= 0:
            raise ValueError(
                "taper_width must be > 0; a hard threshold has no usable "
                "gradient (that is the whole reason for the taper)")
        self.cutoff = cutoff
        self.taper_width = taper_width
        self.k_sol = k_sol
        self.b_sol = b_sol
        self.k_overall = k_overall
        self.u_aniso = u_aniso
        self.detach_mask = detach_mask

    def apply(self, density, F_protein, cell_volume, hkl, orth_matrix):
        """
        density must ALREADY be symmetry-expanded -- see the module docstring.

        Returns (F_total, mask), the mask being handed back so callers can
        report mask_occupancy() without recomputing it.
        """
        rho = density.detach() if self.detach_mask else density
        mask = solvent_mask(rho, self.cutoff, self.taper_width)
        F_mask = f_solvent(mask, cell_volume, hkl, orth_matrix)
        inv_d2 = reciprocal_inv_d2(orth_matrix, hkl)
        F = f_total(
            F_protein, F_mask, inv_d2,
            k_sol=self.k_sol, b_sol=self.b_sol, k_overall=self.k_overall,
            u_aniso=self.u_aniso, hkl=hkl,
        )
        return F, mask

    def __repr__(self):
        return ("SolventModel(cutoff=%r, taper_width=%r, k_sol=%r, b_sol=%r, "
                "k_overall=%r, u_aniso=%s, detach_mask=%r)"
                % (self.cutoff, self.taper_width, self.k_sol, self.b_sol,
                   self.k_overall,
                   "None" if self.u_aniso is None else "set", self.detach_mask))
