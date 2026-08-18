"""
Isotropic-background removal: parity with the original xtraj implementation,
and agreement between the three ways of applying it.

The reference below is the loop that lived in xtraj.py's to_aniso() before
sf/aniso.py existed, copied verbatim apart from the miller_array unpacking that
surrounded it. Keeping it here is the point: the refactor is only safe if the
new path reproduces it exactly, and once the original is gone there is nothing
else to compare against.

Parity is asserted as EXACT equality, not a tolerance. Both paths do the same
float64 arithmetic in the same order, so anything less than exact means the
behaviour moved.
"""
import numpy as np
import pytest

from lunus.sf.aniso import shell_bins, spline_basis, subtract_isotropic


def reference_to_aniso(data_np, d_star_np, shell_thickness, mask_value=np.nan):
    """The original loop from xtraj.to_aniso(), verbatim. Do not tidy."""
    from scipy.interpolate import CubicSpline

    data_np = data_np.copy()

    normalized_d_star = d_star_np / shell_thickness
    bin_indices = np.floor(normalized_d_star + 0.5).astype(int)

    bin_centers = []
    bin_averages = []
    unique_r_bins = np.unique(bin_indices)

    for r in unique_r_bins:
        mask = (bin_indices == r)
        bin_data = data_np[mask]
        if np.isnan(mask_value):
            valid_data = bin_data[~np.isnan(bin_data)]
        else:
            valid_data = bin_data[bin_data != mask_value]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            center = r * shell_thickness
            bin_centers.append(center)
            bin_averages.append(mean_val)

    if len(bin_centers) < 4:
        raise ValueError("Only found %d populated shells." % len(bin_centers))

    spline = CubicSpline(bin_centers, bin_averages, bc_type='not-a-knot',
                         extrapolate=True)
    isotropic_bg_np = spline(d_star_np)

    if np.isnan(mask_value):
        valid_mask = ~np.isnan(data_np)
    else:
        valid_mask = (data_np != mask_value)

    data_np[valid_mask] -= isotropic_bg_np[valid_mask]
    return data_np


def synthetic_map(n_refl=4000, seed=0, shell_thickness=0.05):
    """
    A diffuse-like map: a strong radial component plus anisotropic structure,
    which is the signal the operation is supposed to separate.
    """
    rng = np.random.default_rng(seed)
    d_star = rng.uniform(0.02, 0.55, n_refl)
    isotropic = 500.0 * np.exp(-8.0 * d_star)
    anisotropic = rng.normal(0.0, 20.0, n_refl)
    return d_star, isotropic + anisotropic, shell_thickness


class TestParityWithOriginal:
    def test_matches_reference_exactly(self):
        d_star, data, thickness = synthetic_map()
        np.testing.assert_array_equal(
            subtract_isotropic(data, d_star, thickness),
            reference_to_aniso(data, d_star, thickness),
        )

    def test_matches_reference_with_nan_mask(self):
        d_star, data, thickness = synthetic_map()
        data = data.copy()
        data[::7] = np.nan
        result = subtract_isotropic(data, d_star, thickness)
        np.testing.assert_array_equal(result, reference_to_aniso(data, d_star, thickness))
        assert np.isnan(result[::7]).all(), "masked points must stay masked"

    def test_matches_reference_with_sentinel_mask(self):
        d_star, data, thickness = synthetic_map()
        data = data.copy()
        data[::11] = -32768.0
        np.testing.assert_array_equal(
            subtract_isotropic(data, d_star, thickness, mask_value=-32768.0),
            reference_to_aniso(data, d_star, thickness, mask_value=-32768.0),
        )

    def test_does_not_modify_its_input(self):
        d_star, data, thickness = synthetic_map()
        before = data.copy()
        subtract_isotropic(data, d_star, thickness)
        np.testing.assert_array_equal(data, before)

    def test_too_few_shells_raises(self):
        # Three shells: under the minimum a cubic spline needs.
        d_star = np.array([0.10, 0.11, 0.20, 0.21, 0.30, 0.31])
        data = np.arange(6, dtype=np.float64)
        with pytest.raises(ValueError, match="minimum of 4"):
            subtract_isotropic(data, d_star, 0.1)


class TestBasisForm:
    """The precomputed linear operator must agree with the one-shot path."""

    def test_basis_reproduces_subtract_isotropic(self):
        d_star, data, thickness = synthetic_map()
        bin_index, knots, counts = shell_bins(d_star, thickness)
        basis = spline_basis(d_star, knots)

        shell_means = np.bincount(bin_index, weights=data, minlength=len(knots)) / counts
        by_basis = data - basis @ shell_means

        # Not exact: the one-shot sums each shell's values in reflection order
        # while bincount accumulates in the same order but through a different
        # reduction, so the means differ in the last bits.
        np.testing.assert_allclose(by_basis, subtract_isotropic(data, d_star, thickness),
                                   rtol=0, atol=1e-9)

    def test_removes_a_purely_radial_signal(self):
        """A map that is exactly a function of d* should come back near zero."""
        d_star, _, thickness = synthetic_map()
        radial = 500.0 * np.exp(-8.0 * d_star)
        residual = subtract_isotropic(radial, d_star, thickness)
        assert np.abs(residual).max() < 0.05 * radial.max()


class TestTorchOperator:
    """The numpy-only tests above must still run where torch is absent, so the
    skip is per-test rather than at module scope."""

    @staticmethod
    def _operator(d_star, thickness):
        torch = pytest.importorskip("torch", reason="PyTorch not installed")
        from lunus.sf.aniso_torch import IsotropicBackground

        return torch, IsotropicBackground(d_star, thickness, dtype=torch.float64)

    def test_matches_numpy_core(self):
        d_star, data, thickness = synthetic_map()
        torch, operator = self._operator(d_star, thickness)

        result = operator(torch.as_tensor(data, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(result, subtract_isotropic(data, d_star, thickness),
                                   rtol=0, atol=1e-9)

    def test_batched_matches_single(self):
        d_star, data, thickness = synthetic_map()
        torch, operator = self._operator(d_star, thickness)

        one = torch.as_tensor(data, dtype=torch.float64)
        rows = [one, 2.0 * one, -one]
        batched = operator(torch.stack(rows))
        for i, row in enumerate(rows):
            torch.testing.assert_close(batched[i], operator(row))

    def test_is_linear(self):
        """aniso(a*x + b*y) == a*aniso(x) + b*aniso(y) -- the property the
        precomputed operator depends on for its correctness."""
        d_star, data, thickness = synthetic_map()
        torch, operator = self._operator(d_star, thickness)

        x = torch.as_tensor(data, dtype=torch.float64)
        y = torch.as_tensor(synthetic_map(seed=1)[1], dtype=torch.float64)
        torch.testing.assert_close(operator(3.0 * x - 2.0 * y),
                                   3.0 * operator(x) - 2.0 * operator(y))

    def test_gradients_reach_the_intensities(self):
        d_star, data, thickness = synthetic_map()
        torch, operator = self._operator(d_star, thickness)

        x = torch.as_tensor(data, dtype=torch.float64).requires_grad_(True)
        operator(x).pow(2).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() > 0

    def test_wrong_reflection_count_raises(self):
        d_star, data, thickness = synthetic_map()
        torch, operator = self._operator(d_star, thickness)

        with pytest.raises(ValueError, match="same list"):
            operator(torch.zeros(len(data) - 1, dtype=torch.float64))
