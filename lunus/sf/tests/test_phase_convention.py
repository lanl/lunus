"""The sign of the phases.

compute_fcalc must return F(hkl) = integral rho(r) exp(+2 pi i h.r) d^3r, the
crystallographic convention that gemmi's and cctbx's direct summations use,
so that maps built from its phases put the density where the atoms are. A
conjugate after the ifftn (which already carries the + sign) returns F(-hkl)
instead: the same amplitudes, every phase with its sign flipped, and a map
that is the inversion of the model through the origin. Amplitude-only
comparisons cannot see this, so the check here is against a direct discrete
Fourier sum with the sign written out, on a density that is NOT
centrosymmetric about the origin.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunus.sf.structure_factor_torch import compute_fcalc


def blob_density(grid_shape, pos_frac, width_frac=0.05):
    """A Gaussian blob at fractional position pos_frac on a periodic grid."""
    axes = [np.arange(n) / n for n in grid_shape]
    u, v, w = np.meshgrid(*axes, indexing="ij")
    d2 = 0.0
    for coord, p in zip((u, v, w), pos_frac):
        delta = (coord - p + 0.5) % 1.0 - 0.5          # minimum image
        d2 = d2 + delta * delta
    return np.exp(-d2 / (2.0 * width_frac ** 2))


def direct_sum(density, cell_volume, hkl):
    """F(h) = (V/N) sum_r rho(r) exp(+2 pi i h.r), written out."""
    grid_shape = density.shape
    axes = [np.arange(n) / n for n in grid_shape]
    u, v, w = np.meshgrid(*axes, indexing="ij")
    n_points = density.size
    out = []
    for h, k, l in hkl:
        phase = np.exp(2j * np.pi * (h * u + k * v + l * w))
        out.append((density * phase).sum() * cell_volume / n_points)
    return np.array(out)


def test_phase_sign_matches_direct_sum():
    grid_shape = (24, 24, 24)
    pos_frac = np.array([0.1, 0.2, 0.3])
    cell_volume = 1000.0
    density = blob_density(grid_shape, pos_frac)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [2, -1, 1]])

    F_direct = direct_sum(density, cell_volume, hkl)
    F = compute_fcalc(torch.tensor(density), cell_volume, torch.tensor(hkl)).numpy()

    # The blob is real and positive, so its phase is exactly +2 pi h.x.
    expected_phase = 2 * np.pi * (hkl @ pos_frac)
    phase_err = (np.angle(F) - expected_phase + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(phase_err)) < 1e-6, (
        "phase(F) should be +2 pi h.x; got " + str(np.angle(F)) +
        " vs " + str((expected_phase + np.pi) % (2 * np.pi) - np.pi))
    np.testing.assert_allclose(F, F_direct, rtol=1e-9, atol=1e-9 * np.abs(F_direct).max())


def test_phase_sign_through_symmetry_expansion():
    """The same check through symmetrize_sum() in a space group with screw
    axes (P4_3), against cctbx direct summation on the asymmetric unit. The
    examples used by the smoke test are P1, so without this the symmetrized
    path's phases are never compared with an independent reference."""
    pytest.importorskip("cctbx")
    from cctbx import crystal, miller, xray
    from cctbx.array_family import flex
    from cctbx.eltbx import xray_scattering
    from lunus.sf.symmetry_torch import build_grid_ops_from_cctbx, symmetrize_sum

    cs = crystal.symmetry(unit_cell=(40.0, 40.0, 36.0, 90, 90, 90),
                          space_group_symbol="P 43")
    sites = np.random.default_rng(1).random((6, 3))
    grid_shape = (48, 48, 48)
    b_iso = 30.0

    # A unit point scatterer with Debye-Waller B is the Gaussian
    # (4 pi / B)^1.5 exp(-4 pi^2 r^2 / B) in real space.
    M = np.array(cs.unit_cell().orthogonalization_matrix()).reshape(3, 3)
    axes = [np.arange(n) / n for n in grid_shape]
    u, v, w = np.meshgrid(*axes, indexing="ij")
    density = np.zeros(grid_shape)
    for site in sites:
        delta = np.stack([(c - s + 0.5) % 1.0 - 0.5 for c, s in zip((u, v, w), site)], -1)
        r2 = ((delta @ M.T) ** 2).sum(-1)
        density += (4 * np.pi / b_iso) ** 1.5 * np.exp(-4 * np.pi ** 2 * r2 / b_iso)
    ops = build_grid_ops_from_cctbx(cs.space_group(), grid_shape)
    density = symmetrize_sum(torch.tensor(density), ops)

    ms = miller.build_set(cs, anomalous_flag=False, d_min=4.0)
    hkl = np.array(ms.indices())
    F = compute_fcalc(density, cs.unit_cell().volume(), torch.tensor(hkl)).numpy()

    scatterers = flex.xray_scatterer([
        xray.scatterer(label="X%d" % i, site=tuple(s), b=b_iso, scattering_type="const")
        for i, s in enumerate(sites)])
    xs = xray.structure(crystal_symmetry=cs, scatterers=scatterers)
    xs.scattering_type_registry(custom_dict={"const": xray_scattering.gaussian(1.0)})
    F_direct = ms.structure_factors_from_scatterers(
        xs, algorithm="direct").f_calc().data().as_numpy_array()

    weight = np.abs(F) * np.abs(F_direct)
    agreement = float(np.sum(weight * np.cos(np.angle(F) - np.angle(F_direct))) / weight.sum())
    assert agreement > 0.99, (
        "P4_3: weighted mean cos(phase difference) vs cctbx direct summation "
        "is %.4f; a flipped sign gives about 0.07" % agreement)
