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

The Laue merge to_aniso() applies afterwards is linear and fixed in the same
way. segment_average() applies such a partition once a caller supplies one,
and AnisoOperator is the two steps together: to_aniso(), precomputed and
batched. xtraj.laue_segment_index() builds the partition.
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


def segment_spline_basis(d_star, knots, segment_index, counts):
    """spline_basis() averaged over each segment, (n_segments, n_bins).

    Built one column at a time, so the full (n_refl, n_bins) basis is never
    materialised -- at a million reflections that is 500 MB it does not need.
    """
    from scipy.interpolate import CubicSpline

    d_star = np.asarray(d_star, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    n_bins = len(knots)
    if n_bins < MIN_SHELLS:
        raise ValueError(
            "Only %d shells; a minimum of %d is required for cubic spline "
            "interpolation." % (n_bins, MIN_SHELLS))

    basis = np.empty((len(counts), n_bins), dtype=np.float64)
    identity = np.eye(n_bins)
    for j in range(n_bins):
        column = CubicSpline(knots, identity[j], bc_type="not-a-knot",
                             extrapolate=True)(d_star)
        basis[:, j] = np.bincount(segment_index, weights=column,
                                  minlength=len(counts)) / counts
    return basis


def segment_counts(segment_index, n_segments=None):
    """Multiplicity of each segment, as float64, ready to divide sums by."""
    segment_index = np.asarray(segment_index, dtype=np.int64)
    if n_segments is None:
        n_segments = int(segment_index.max()) + 1 if segment_index.size else 0
    counts = np.bincount(segment_index, minlength=n_segments).astype(np.float64)
    if not counts.all():
        raise ValueError("segment ids must be dense: %d of %d segments are empty"
                         % (int((counts == 0).sum()), n_segments))
    return counts


def segment_average(data, segment_index, counts):
    """Replace every value by the mean over its segment.

    data is (..., n_refl); leading axes are an independent batch. This is what
    a symmetry merge followed by a map back onto the original indices does to
    the data, once the partition is precomputed.

    np.bincount sums each segment sequentially in reflection order, which is
    the same order -- and so the same float64 result to the last bit -- as
    cctbx's merge_equivalents_real. NaN propagates across a segment, as
    merging does. np.add.reduceat is the more obvious spelling but
    reassociates, and measured 7x slower.
    """
    data = np.asarray(data, dtype=np.float64)
    segment_index = np.asarray(segment_index, dtype=np.int64)
    n_refl = segment_index.shape[0]
    if data.shape[-1] != n_refl:
        raise ValueError(
            "data has %d reflections but the partition has %d; both must be on "
            "the same list, in the same order." % (data.shape[-1], n_refl))

    flat = data.reshape(-1, n_refl)
    means = np.empty((flat.shape[0], len(counts)), dtype=np.float64)
    for row in range(flat.shape[0]):
        means[row] = np.bincount(segment_index, weights=flat[row],
                                 minlength=len(counts))
    means /= counts
    return means[:, segment_index].reshape(data.shape)


class AnisoOperator:
    """Subtract the isotropic background, then average over a fixed partition.

    The precomputed, batched form of xtraj.py's to_aniso(): build once per
    reflection list, then each call is a scatter-add, one (n_bins) matmul and
    a gather. Agreement with subtract_isotropic() is ~1e-9 rather than exact,
    since the background becomes S @ shell_means -- see LINEARITY above.

    segment_index gives the partition to average over, or None for no
    averaging; valid is a FIXED mask, unlike subtract_isotropic()'s per-call
    mask_value, so the same points must be masked on every call.

    reduced=True returns the SEGMENT MEANS, (..., n_segments), instead of
    gathering them back to (..., n_refl). Since A(I - S m) = A I - (A S) m,
    only the segment-averaged basis is needed, so the full-length background,
    output and gather all disappear -- for a caller that goes straight on to
    reduce over reflections, all three are pure overhead. Requires a partition
    and rejects a mask.
    """

    def __init__(self, d_star, shell_thickness, segment_index=None, valid=None,
                 reduced=False):
        d_star = np.asarray(d_star, dtype=np.float64)
        n_refl = len(d_star)
        bin_index, knots, _ = shell_bins(d_star, shell_thickness)

        if valid is None:
            valid_idx = None
            valid_bin = bin_index
        else:
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != (n_refl,):
                raise ValueError("valid must be (%d,), got %r" % (n_refl, valid.shape))
            valid_idx = np.nonzero(valid)[0]
            valid_bin = bin_index[valid_idx]

        # A shell with no valid data contributes no knot, as in subtract_isotropic.
        populated_counts = np.bincount(valid_bin, minlength=len(knots))
        populated = populated_counts > 0
        if int(populated.sum()) < MIN_SHELLS:
            raise ValueError(
                "Only found %d populated shells. A minimum of %d is required for "
                "cubic spline interpolation." % (int(populated.sum()), MIN_SHELLS))

        remap = np.full(len(knots), -1, dtype=np.int64)
        remap[populated] = np.arange(int(populated.sum()))

        self.n_bins = int(populated.sum())
        self.d_star = d_star
        self.shell_thickness = float(shell_thickness)
        self._valid_idx = valid_idx
        self._bin_index = remap[valid_bin]
        self._counts = populated_counts[populated].astype(np.float64)
        self._basis = None          # set below, once the partition is known

        self.segment_index = None
        self.segment_counts = None
        self.reduced = bool(reduced)
        if segment_index is not None:
            self.segment_index = np.asarray(segment_index, dtype=np.int64)
            if self.segment_index.shape != (n_refl,):
                raise ValueError("segment_index must be (%d,), got %r"
                                 % (n_refl, self.segment_index.shape))
            self.segment_counts = segment_counts(self.segment_index)
        elif self.reduced:
            raise ValueError("reduced=True needs a segment_index")
        if self.reduced and valid is not None:
            raise ValueError("reduced=True does not support a mask")

        if self.reduced:
            self._basis = segment_spline_basis(d_star, knots[populated],
                                               self.segment_index,
                                               self.segment_counts)
        else:
            self._basis = spline_basis(d_star, knots[populated])

    def __call__(self, data):
        """data is (..., n_refl).

        Returns the same shape, or (..., n_segments) when reduced.
        """
        data = np.asarray(data, dtype=np.float64)
        n_refl = len(self.d_star)
        if data.shape[-1] != n_refl:
            raise ValueError(
                "data has %d reflections but this operator was built for %d; "
                "both must be on the same list, in the same order."
                % (data.shape[-1], n_refl))

        flat = data.reshape(-1, n_refl)
        values = flat if self._valid_idx is None else flat[:, self._valid_idx]

        shell_sums = np.empty((flat.shape[0], self.n_bins), dtype=np.float64)
        for row in range(flat.shape[0]):
            shell_sums[row] = np.bincount(self._bin_index, weights=values[row],
                                          minlength=self.n_bins)
        shell_means = shell_sums / self._counts

        if self.reduced:
            segment_sums = np.empty((flat.shape[0], len(self.segment_counts)),
                                    dtype=np.float64)
            for row in range(flat.shape[0]):
                segment_sums[row] = np.bincount(
                    self.segment_index, weights=flat[row],
                    minlength=len(self.segment_counts))
            out = segment_sums / self.segment_counts - shell_means @ self._basis.T
            return out.reshape(data.shape[:-1] + (len(self.segment_counts),))

        background = shell_means @ self._basis.T

        if self._valid_idx is None:
            out = flat - background
        else:
            out = flat.copy()
            out[:, self._valid_idx] -= background[:, self._valid_idx]
        out = out.reshape(data.shape)

        if self.segment_index is None:
            return out
        return segment_average(out, self.segment_index, self.segment_counts)
