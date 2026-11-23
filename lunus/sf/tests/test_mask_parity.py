"""
SolventModel's default mask is the R-factor tool's mask.

The tool (fit_solvent_rfactor.py, as pdb_rfactor.py runs it) places the
sphere field, thresholds it at --mask-level auto and transforms it itself.
Here the tool runs in process with spies on the functions it calls, and the
library builds F_mask from the very inputs the tool used -- coordinates,
cctbx symmetry weights, grid, cell, symmetry operations, reflections --
through SolventModel() + set_atoms(names), cutoff=None, and
structure_factors_one_config(..., return_parts=True). The two must be the
same numbers, bit for bit: one model without hydrogens (3I7M, level 1.0)
and one with them (2YL0, level 1.0 + 0.89 h = 1.278), cut to 3 A to keep the
suite quick; the same holds at full resolution (measured 2026-11-22). The
two entries' files are not tracked (only 7FPV is): the test skips without
them, and `pdb_rfactor.py 3I7M --dir lunus/sf/examples/compare_gemmi` (and
2YL0) fetches them from the wwPDB.

compile_core=False, as the tool places the field. The compiled CPU core
agrees with it to 1e-15 in a process without cctbx, but this branch predates
the fix for a cctbx-loaded OpenMP stub that makes the compiled CPU cores
wrong (master fbaf33f), and this test has cctbx loaded.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from conftest import REPO_ROOT, requires_cctbx  # noqa: E402

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
COMPARE = os.path.join(REPO_ROOT, "lunus", "sf", "examples", "compare_gemmi")


def run_tool_and_capture(monkeypatch, pdb_id, d_min="3.0"):
    """fit_solvent_rfactor.run on the compare_gemmi files, default mask
    settings, with what it placed and transformed for its own mask."""
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import fit_solvent_rfactor as fsr

    import lunus.sf.density_torch as density_torch
    import lunus.sf.mask_spheres as mask_spheres
    import lunus.sf.solvent_torch as solvent_torch
    import lunus.sf.symmetry_torch as symmetry_torch

    cap = {}
    real_kernels = mask_spheres.sphere_kernels
    real_density = density_torch.compute_density
    real_sym = symmetry_torch.symmetrize_sum
    real_mask = solvent_torch.solvent_mask
    real_fsolvent = solvent_torch.f_solvent

    def kernels(radii, *a, **kw):
        cap["kernels"] = real_kernels(radii, *a, **kw)
        cap["radii"] = np.array(radii)
        return cap["kernels"]

    def density(frac, idx, occ, *a, **kw):
        out = real_density(frac, idx, occ, *a, **kw)
        if "protein" not in cap:          # the protein density comes first
            cap["protein"] = (frac, idx, occ) + a
        if "kernels" in cap and idx is cap["kernels"][5]:
            cap["field_asu"] = out
        return out

    def sym(field, ops):
        out = real_sym(field, ops)
        if "field_asu" in cap and field is cap["field_asu"]:
            cap["ops"] = ops
        return out

    def mask(source, cutoff, taper):
        cap["mask"] = real_mask(source, cutoff, taper)
        cap["level"] = (float(cutoff), float(taper))
        return cap["mask"]

    def fsolvent(m, volume, hkl, orth):
        out = real_fsolvent(m, volume, hkl, orth)
        if m is cap.get("mask") and "F_mask" not in cap:     # not gemmi's mask
            cap["F_mask"] = out
            cap["transform"] = (volume, hkl, orth)
        return out

    monkeypatch.setattr(mask_spheres, "sphere_kernels", kernels)
    monkeypatch.setattr(density_torch, "compute_density", density)
    monkeypatch.setattr(symmetry_torch, "symmetrize_sum", sym)
    monkeypatch.setattr(solvent_torch, "solvent_mask", mask)
    monkeypatch.setattr(solvent_torch, "f_solvent", fsolvent)
    path = os.path.join(COMPARE, pdb_id + ".pdb")
    args = fsr.build_parser().parse_args(
        [path, os.path.join(COMPARE, pdb_id + "-sf.cif"), "--aniso-adp", "--device", "cpu",
         "--d-min", d_min])
    with contextlib.redirect_stdout(io.StringIO()):
        r = fsr.run(args)
    monkeypatch.undo()
    xrs = fsr.read_model(path, log=io.StringIO())[0]
    cap["labels"] = [sc.label for sc in xrs.scatterers()]
    cap["elements"] = [sc.scattering_type for sc in xrs.scatterers()]
    cap["result"] = r
    return cap


@requires_cctbx
@pytest.mark.parametrize("pdb_id, level", [("3I7M", 1.0), ("2YL0", 1.0 + 0.89 * 729 / 2332)])
def test_library_default_mask_is_the_tool_mask(monkeypatch, pdb_id, level):
    from lunus.sf import mask_spheres
    from lunus.sf.solvent_torch import SolventModel
    from lunus.sf.structure_factor_torch import structure_factors_one_config

    # the compare_gemmi PDB files are gitignored (only 7FPV is tracked), so a
    # fresh clone does not have them
    for name in (pdb_id + ".pdb", pdb_id + "-sf.cif"):
        if not os.path.exists(os.path.join(COMPARE, name)):
            pytest.skip("examples/compare_gemmi/%s not present" % name)
    cap = run_tool_and_capture(monkeypatch, pdb_id)
    assert cap["result"]["mask_level"] == level and cap["level"] == (level, 0.5 * level)
    frac, idx, occ, A, lam, offsets, radius, grid, orth_p, taper_w = cap["protein"]
    volume, hkl, orth = cap["transform"]
    assert torch.equal(orth, orth_p)

    model = SolventModel(check_occupancy=False)
    model.set_atoms(cap["elements"], *mask_spheres.names_from_cctbx_labels(cap["labels"]))
    assert np.array_equal(model.atom_radii, cap["radii"])
    assert model.threshold("spheres") == cap["level"]
    for blur in (0.0, cap["result"]["protein_blur"]):
        # the sphere field carries no pipeline blur, so F_mask does not
        # depend on the blur= a caller passes for the protein
        parts = structure_factors_one_config(
            frac, idx, occ, A, lam, offsets, radius, grid, orth, volume, hkl, taper_w,
            blur=blur, grid_ops=cap["ops"], compile_core=False, solvent=model,
            return_parts=True)
        assert torch.equal(parts.F_mask, cap["F_mask"]), (
            "library F_mask differs from the tool's (blur %g): max |d| %.3g"
            % (blur, float((parts.F_mask - cap["F_mask"]).abs().max())))
    # and the mask grid itself, from the same field
    field = model.mask_field(frac, occ, grid, orth, grid_ops=cap["ops"])
    assert torch.equal(model.build_mask(None, orth, source=field), cap["mask"])
