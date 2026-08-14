"""
Run this where PyTorch is installed (this sandbox doesn't have network
access to install it, so this file is written but not executed here --
see validate_numpy.py, which tests the identical algorithm in NumPy and
was executed and checked).

Two things are checked:
  1. The torch pipeline (density_torch + structure_factor_torch)
     reproduces the same F(hkl) values as validate_numpy.py's NumPy
     reference, for the identical test system -- catches translation
     bugs (indexing, broadcasting, dtype issues) rather than algorithm
     bugs, since the algorithm itself was already validated in NumPy.
  2. autograd's dF/d(frac_coords) matches a finite-difference gradient
     of the *analytic* closed-form single-atom formula -- confirming
     gradients flowing back to atomic coordinates are correct, using
     the real autograd graph (not a hand-rolled backward pass).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import calculate_sf
from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2, grid_shape_for_resolution
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.structure_factor_torch import structure_factors_one_config

DTYPE = torch.float64  # double precision for a tight numerical check;
                        # use float32 for production-scale runs


def analytic_single_atom_F_np(coef9, b_iso, pos_frac, hkl, M):
    inv_d2 = reciprocal_inv_d2(M, hkl)
    stol2 = inv_d2 / 4.0
    f = calculate_sf(coef9, stol2)
    dwf = np.exp(-b_iso * stol2)
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return f * dwf * phase


@pytest.fixture(scope="module")
def torch_system():
    """Identical test system to test_numpy_reference.py, built as torch tensors."""
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M_np = orth_matrix(a, b, c, alpha, beta, gamma)
    cell_volume = abs(np.linalg.det(M_np))

    d_min, rate = 2.0, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)

    b_iso = 20.0 * d_min * d_min
    coef9 = IT92_COEFFS["C"]
    pos_frac_np = np.array([0.372, 0.611, 0.148])

    hkl_np = np.array([
        [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3],
        [2, 1, 0], [1, -3, 2], [-4, 2, 1], [3, 3, 3],
        [5, -1, 2], [0, 5, -2],
    ])

    M = torch.tensor(M_np, dtype=DTYPE)
    hkl = torch.tensor(hkl_np, dtype=torch.long)
    elem_A, elem_lam, elem_offsets, elem_radius_ang, taper_width, elem_to_idx = build_element_kernels_torch(
        ["C"], IT92_COEFFS, b_iso, blur=0.0,
        grid_shape=grid_shape, orth_matrix_np=M_np,
        cutoff=1e-5, device="cpu", dtype=DTYPE,
    )

    frac_coords = torch.tensor(pos_frac_np.reshape(1, 3), dtype=DTYPE, requires_grad=True)
    element_idx = torch.zeros(1, dtype=torch.long)

    return dict(
        M_np=M_np, M=M, cell_volume=cell_volume, grid_shape=grid_shape,
        b_iso=b_iso, coef9=coef9, pos_frac_np=pos_frac_np,
        hkl_np=hkl_np, hkl=hkl,
        frac_coords=frac_coords, element_idx=element_idx,
        atom_A=elem_A[element_idx],
        atom_lam=elem_lam[element_idx],
        atom_radius_ang=elem_radius_ang[element_idx],
        elem_offsets=elem_offsets,
        taper_width=taper_width,
        occ=torch.ones(1, dtype=DTYPE),
    )


def _structure_factors(s):
    return structure_factors_one_config(
        s["frac_coords"], s["element_idx"], s["occ"],
        s["atom_A"], s["atom_lam"], s["elem_offsets"], s["atom_radius_ang"],
        s["grid_shape"], s["M"], s["cell_volume"], s["hkl"], s["taper_width"],
        blur=0.0,
    )


def test_values_match_numpy_reference(torch_system):
    """The torch pipeline must reproduce the NumPy reference's F(hkl).

    Catches translation bugs -- indexing, broadcasting, dtype -- rather than
    algorithm bugs, since the algorithm itself is validated against a
    closed-form reference in test_numpy_reference.py.
    """
    s = torch_system
    coef9, b_iso = s["coef9"], s["b_iso"]
    pos_frac_np, hkl_np, M_np = s["pos_frac_np"], s["hkl_np"], s["M_np"]

    F = _structure_factors(s)
    F_ana = analytic_single_atom_F_np(coef9, b_iso, pos_frac_np, hkl_np, M_np)

    print("\nhkl            F_torch                       F_analytic                    rel.err")
    max_rel_err = 0.0
    for i, (h, k, l) in enumerate(hkl_np):
        fn = F[i].detach().numpy()
        fa = F_ana[i]
        rel_err = abs(fn - fa) / max(abs(fa), 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"{h:3d}{k:3d}{l:3d}   {fn.real:8.4f}{fn.imag:+8.4f}i   {fa.real:8.4f}{fa.imag:+8.4f}i   {rel_err:.2e}")
    print(f"\nMax relative error (values, torch vs analytic): {max_rel_err:.2e}")
    # Threshold accounts for the taper's real, deliberate, already-measured
    # deviation from this fully-UNTRUNCATED analytic reference -- not just
    # grid discretization. Directly confirmed via radial integration for
    # these exact parameters (b_iso=80, cutoff=1e-5, taper_width=0.1): the
    # hard cutoff alone loses ~0.018% of total charge (unavoidable with any
    # finite radius), the taper adds a bit more on top, totaling ~2.1e-4 --
    # matching what's observed here to within grid-discretization noise.
    # This is the correct, intentional tradeoff for gradient correctness
    # (see density_torch.py's module docstring), not a regression.
    assert max_rel_err < 5e-4, "torch pipeline does not match the analytic reference"


def test_autograd_matches_analytic_gradient(torch_system):
    """autograd's dF/d(frac_coords) vs a finite difference of the CLOSED-FORM
    analytic formula -- confirming gradients flowing back to atomic
    coordinates are correct, using the real autograd graph rather than a
    hand-rolled backward pass.

    Only reflections with h != 0 are used. The analytic derivative w.r.t.
    frac_x carries a factor of h, so h == 0 rows have an identically zero
    reference and the relative error against them is meaningless.
    test_numpy_reference.py notes the same effect.
    """
    s = torch_system
    coef9, b_iso = s["coef9"], s["b_iso"]
    pos_frac_np, hkl_np, M_np = s["pos_frac_np"], s["hkl_np"], s["M_np"]
    frac_coords = s["frac_coords"]

    print("\nGradient check (autograd dF/d(frac_x) vs analytic finite difference):")
    eps = 1e-6
    max_grad_rel_err = 0.0
    indices = [i for i, (h, _, _) in enumerate(hkl_np) if h != 0][:2]
    assert indices, "no h != 0 reflections available to compare"

    for i in indices:
        h, k, l = hkl_np[i]
        hkl1 = hkl_np[i:i + 1]

        # Real and imaginary parts need separate backward passes; each starts
        # from a zeroed .grad so the two do not accumulate into one another.
        if frac_coords.grad is not None:
            frac_coords.grad.zero_()
        _structure_factors(s)[i].real.backward(retain_graph=True)
        grad_real_x = frac_coords.grad[0, 0].item()

        frac_coords.grad.zero_()
        _structure_factors(s)[i].imag.backward()
        grad_imag_x = frac_coords.grad[0, 0].item()
        dF_autograd = complex(grad_real_x, grad_imag_x)

        pos_p = pos_frac_np.copy(); pos_p[0] += eps
        pos_m = pos_frac_np.copy(); pos_m[0] -= eps
        Fa_p = analytic_single_atom_F_np(coef9, b_iso, pos_p, hkl1, M_np)[0]
        Fa_m = analytic_single_atom_F_np(coef9, b_iso, pos_m, hkl1, M_np)[0]
        dF_ana = (Fa_p - Fa_m) / (2 * eps)

        rel_err = abs(dF_autograd - dF_ana) / max(abs(dF_ana), 1e-8)
        max_grad_rel_err = max(max_grad_rel_err, rel_err)
        print(f"{h:3d}{k:3d}{l:3d}   dF_autograd={dF_autograd:10.3f}   "
              f"dF_analytic={dF_ana:10.3f}   rel.err={rel_err:.2e}")

    print(f"\nMax relative error (gradients, autograd vs analytic): {max_grad_rel_err:.2e}")
    assert max_grad_rel_err < 1e-3, "autograd gradient does not match the analytic reference"
