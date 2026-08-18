"""
Isotropic-background removal: the anisotropic component of a diffuse map.

Diffuse intensity is conventionally compared after its isotropic part is
removed. The radial component is dominated by contributions a coordinate model
does not describe -- solvent scattering, incoherent background, absorption --
so the anisotropic residual is what reports on correlated motion.

The background is estimated by averaging the data in spherical shells of
reciprocal space, fitting a cubic spline through the shell means, and
evaluating that spline at each reflection. Shell thickness is conventionally
the reciprocal-cell diagonal, d* of (1,1,1), which makes a shell about one
reciprocal lattice repeat thick whatever the cell size.

This is the crystallography-library-free core of what xtraj.py's to_aniso()
has always done. It takes d* and the data as plain arrays, so a caller can get
d* from cctbx, from gemmi, or from its own reciprocal-cell arithmetic; to_aniso
is now a thin wrapper that adds the cctbx-shaped work around it -- unpacking a
miller_array and merging in Laue symmetry afterwards -- which does belong
there.

LINEARITY
---------
The operation is linear in the data. Shell averaging is a mean over a fixed
partition of the reflections, and a cubic spline's values are a linear function
of its knot values, so

    aniso(I) = I - S @ shell_mean(I)

with S an (n_refl, n_bins) matrix fixed by d* and the shell thickness alone.
Nothing in S depends on the data, so a caller evaluating this repeatedly on one
reflection list -- guided sampling, or a trajectory scored frame by frame --
can build S once and reduce each evaluation to a scatter-add and a small
matmul. spline_basis() exposes S for that, and aniso_torch.IsotropicBackground
is the differentiable version built on it. A single call is better served by
subtract_isotropic(), which never materializes S.
"""

import numpy as np


# A cubic spline through three knots is not determined. xtraj has always
# refused fewer than four shells, and so does this.
MIN_SHELLS = 4


def shell_bins(d_star, shell_thickness):
    """
    Assign reflections to spherical shells.

    d_star:          (n_refl,) magnitudes in reciprocal space
    shell_thickness: shell width, conventionally d* of (1,1,1)

    Returns (bin_index, knots, counts):
      bin_index  (n_refl,) index into knots, one per reflection
      knots      (n_bins,) shell centres, in d* units
      counts     (n_bins,) reflections per shell

    Shells containing no reflections at all are not represented: unique() is
    taken over the bin numbers that actually occur. That matters at low
    resolution, where the shells are geometrically small and one can easily
    hold nothing.

    The bin number is floor(d*/thickness + 0.5), reproducing the C cast in the
    original lunus implementation. This is NOT np.round, which breaks ties to
    even; they differ only for exact halves, but shell edges are exactly where
    those land.
    """
    d_star = np.asarray(d_star, dtype=np.float64)
    raw = np.floor(d_star / shell_thickness + 0.5).astype(np.int64)
    unique, bin_index = np.unique(raw, return_inverse=True)
    counts = np.bincount(bin_index, minlength=len(unique))
    return bin_index, unique.astype(np.float64) * shell_thickness, counts


def spline_basis(d_star, knots):
    """
    The linear map from shell means to interpolated background.

    Returns an (n_refl, n_bins) array S for which

        background = S @ shell_means

    holds for ANY shell means, so a caller can apply it repeatedly without
    re-solving the spline. Column j is the not-a-knot cubic spline through knot
    values e_j evaluated at every d*, which is the definition of the linear
    map -- obtained by solving the spline n_bins times rather than by
    reimplementing its algebra, so it cannot drift from what scipy does.

    Costs n_bins spline solves once, and (n_refl * n_bins) doubles of storage.
    For a single evaluation use subtract_isotropic().

    Raises ValueError for fewer than MIN_SHELLS knots.
    """
    from scipy.interpolate import CubicSpline

    d_star = np.asarray(d_star, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    n_bins = len(knots)
    if n_bins < MIN_SHELLS:
        raise ValueError(
            "Only %d shells; a minimum of %d is required for cubic spline "
            "interpolation." % (n_bins, MIN_SHELLS)
        )

    basis = np.empty((len(d_star), n_bins), dtype=np.float64)
    identity = np.eye(n_bins)
    for j in range(n_bins):
        spline = CubicSpline(knots, identity[j], bc_type="not-a-knot", extrapolate=True)
        basis[:, j] = spline(d_star)
    return basis


def subtract_isotropic(data, d_star, shell_thickness, mask_value=np.nan):
    """
    Remove the radially-averaged component from data on one reflection list.

    data:            (n_refl,) intensities; a copy is returned, the input is
                     not modified
    d_star:          (n_refl,) magnitudes in reciprocal space
    shell_thickness: shell width, conventionally d* of (1,1,1)
    mask_value:      value denoting unmeasured data, np.nan by default. Masked
                     points take no part in the shell means and are left
                     untouched by the subtraction, so a masked map comes back
                     masked in the same places.

    Returns a new (n_refl,) float64 array.

    Raises ValueError if fewer than MIN_SHELLS shells hold valid data.

    A shell whose reflections are all masked contributes no knot: a knot with
    no data has no defensible value, and inventing one would bend the spline
    through it.
    """
    from scipy.interpolate import CubicSpline

    data = np.array(data, dtype=np.float64, copy=True)
    d_star = np.asarray(d_star, dtype=np.float64)

    if np.isnan(mask_value):
        valid = ~np.isnan(data)
    else:
        valid = data != mask_value

    bin_index, knots, _ = shell_bins(d_star, shell_thickness)

    centres = []
    means = []
    for j in range(len(knots)):
        in_shell = valid & (bin_index == j)
        if in_shell.any():
            centres.append(knots[j])
            means.append(data[in_shell].mean())

    if len(centres) < MIN_SHELLS:
        raise ValueError(
            "Only found %d populated shells. A minimum of %d is required for "
            "cubic spline interpolation." % (len(centres), MIN_SHELLS)
        )

    spline = CubicSpline(centres, means, bc_type="not-a-knot", extrapolate=True)
    background = spline(d_star)

    data[valid] -= background[valid]
    return data
