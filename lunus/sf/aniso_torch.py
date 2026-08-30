"""
Differentiable isotropic-background removal, for repeated use on one hkl set.

aniso.subtract_isotropic() solves a spline every call, which is right for a
one-shot analysis and wrong for a loss evaluated thousands of times on the same
reflections. This builds the linear operator once -- see the LINEARITY note in
aniso.py -- and each evaluation is then a scatter-add over shells and one
(n_bins) matmul, differentiable in the intensities with no spline in the hot
path.

Gradients flow to the intensities; the operator itself has no parameters, and
the shells and spline basis are constants determined by the reflection list.

The one behaviour this does NOT carry over is masking. subtract_isotropic()
takes a mask_value and excludes those points from the shell means; here every
reflection counts. That fits the calculated side of a fit, where the model
produces a value everywhere, and does not fit masked experimental data --
transform that once with the numpy path instead.
"""

import torch

from .aniso import shell_bins, spline_basis


class IsotropicBackground:
    """
    Subtract the shell-averaged, spline-interpolated background from
    intensities on a fixed reflection list.

    Build once per reflection list, then call per evaluation.

    d_star:          (n_refl,) magnitudes in reciprocal space
    shell_thickness: shell width, conventionally d* of (1,1,1)
    device, dtype:   placement for the precomputed operator. float64 matches
                     the numpy path exactly; float32 halves the storage of the
                     (n_refl, n_bins) basis and is usually what a torch
                     pipeline is running in anyway.

    Attributes:
      n_bins   number of populated shells, i.e. spline knots
      d_star   the resolution coordinate, kept for callers that want to bin
               something else the same way (a resolution-shell loss, say)
    """

    def __init__(self, d_star, shell_thickness, device="cpu", dtype=torch.float32):
        bin_index, knots, counts = shell_bins(d_star, shell_thickness)
        basis = spline_basis(d_star, knots)

        self.n_bins = len(knots)
        self.shell_thickness = float(shell_thickness)
        self.d_star = torch.as_tensor(d_star, dtype=dtype, device=device)
        self._bin_index = torch.as_tensor(bin_index, dtype=torch.long, device=device)
        self._basis = torch.as_tensor(basis, dtype=dtype, device=device)
        self._counts = torch.as_tensor(counts, dtype=dtype, device=device)

    def __call__(self, intensities):
        """
        intensities: (..., n_refl) on the reflection list this was built for

        Returns the same shape with the radial component removed.
        """
        n_refl = self._bin_index.shape[0]
        if intensities.shape[-1] != n_refl:
            raise ValueError(
                "intensities has %d reflections but this operator was built for "
                "%d; both must be on the same list, in the same order."
                % (intensities.shape[-1], n_refl)
            )

        flat = intensities.reshape(-1, n_refl)
        shell_sums = torch.zeros(
            flat.shape[0], self.n_bins, dtype=flat.dtype, device=flat.device
        )
        # index_add_ over the shell axis: differentiable, and the gradient it
        # implies (each reflection receives 1/count of its shell's) is exactly
        # the transpose of the averaging, which is what makes the whole
        # operator's backward correct without writing one.
        shell_sums.index_add_(1, self._bin_index, flat)
        shell_means = shell_sums / self._counts

        background = shell_means @ self._basis.T
        return (flat - background).reshape(intensities.shape)

    def to(self, device):
        """Move the precomputed operator to device, in place; returns self."""
        self.d_star = self.d_star.to(device)
        self._bin_index = self._bin_index.to(device)
        self._basis = self._basis.to(device)
        self._counts = self._counts.to(device)
        return self
