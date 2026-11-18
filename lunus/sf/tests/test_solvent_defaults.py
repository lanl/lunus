"""
SolventModel's mask source follows what it knows about the atoms.

mask_source="auto" (the default) cuts the mask from soft spheres when the
model has been told the atoms' names (set_atoms with atom_names and
residue_names), from the atom-presence field when it has only their elements,
and refuses -- naming both fixes -- when it has been told nothing: the
electron density, which measured worst, is never a silent fallback and has to
be asked for as mask_source="density". cutoff=None is the one measured fixed
level (1.0, spheres, no hydrogens) and an error everywhere else until
calibrate() sets it.

The kernels set_atoms() builds must be exactly the ones a caller would build
by hand and pass as mask_kernels=, so every comparison against the hand-built
path below is bit for bit. The radii from names must be exactly the radii
from cctbx labels.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from conftest import REPO_ROOT, requires_cctbx  # noqa: E402

from lunus.sf import mask_spheres  # noqa: E402
from lunus.sf.cell_utils import orth_matrix  # noqa: E402
from lunus.sf.density_torch import compute_density  # noqa: E402
from lunus.sf.elements import it92_coefficients  # noqa: E402
from lunus.sf.kernel_torch import build_atom_kernels_torch  # noqa: E402
from lunus.sf.solvent_torch import (  # noqa: E402
    SOURCE_DENSITY_CUTOFF, SPHERE_LEVEL_FIXED, SolventModel, mask_occupancy,
    mask_source_weights)
from lunus.sf.structure_factor_torch import (  # noqa: E402
    structure_factors_batch, structure_factors_one_config)

CELL = (20.0, 22.0, 24.0, 90.0, 90.0, 90.0)
GRID = (20, 22, 24)
DTYPE = torch.float64

# a few residues' worth of named atoms; the coordinates are random, the names
# are what the radii and the hydrogen rule read
RESIDUES = [
    ("ALA", ["N", "CA", "C", "O", "CB"]),
    ("SER", ["N", "CA", "C", "O", "CB", "OG"]),
    ("VAL", ["N", "CA", "C", "O", "CB", "CG1", "CG2"]),
    ("HOH", ["O"]),
]


def named_system(with_h=False, blur=0.0, repeat=3, seed=0):
    """Named atoms in a small cell, with the protein's per-atom kernels."""
    elements, atom_names, residue_names, residue_ids = [], [], [], []
    for r in range(repeat):
        for resname, names in RESIDUES:
            names = list(names) + (["HA"] if with_h and resname == "SER" else [])
            for name in names:
                elements.append("H" if name.startswith("H") else name[0])
                atom_names.append(name)
                residue_names.append(resname)
    # one residue id per residue, shared by its atoms
    k = 0
    for r in range(repeat):
        for resname, names in RESIDUES:
            n = len(names) + (1 if with_h and resname == "SER" else 0)
            residue_ids += [("A", k, "")] * n
            k += 1
    rng = np.random.default_rng(seed)
    n = len(elements)
    M = orth_matrix(*CELL)
    present = sorted(set(elements))
    coeffs = it92_coefficients(present)
    A, lam, offsets, radius, taper_w, e2i = build_atom_kernels_torch(
        elements, present, coeffs, np.full(n, 20.0), blur, GRID, M,
        cutoff=SOURCE_DENSITY_CUTOFF, device="cpu", dtype=DTYPE)
    return dict(
        elements=elements, atom_names=atom_names, residue_names=residue_names,
        residue_ids=residue_ids,
        frac=torch.tensor(rng.random((n, 3)), dtype=DTYPE),
        element_idx=torch.tensor([e2i[e] for e in elements], dtype=torch.long),
        occ=torch.tensor(0.5 + 0.5 * rng.random(n), dtype=DTYPE),
        atom_A=A, atom_lam=lam, elem_offsets=offsets, atom_radius_ang=radius,
        taper_width=taper_w, orth=torch.tensor(M, dtype=DTYPE), M=M,
        cell_volume=float(np.linalg.det(M)), blur=blur,
        hkl=torch.tensor([[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
                          for l in range(-3, 4) if (h, k, l) != (0, 0, 0)]))


def rest(s):
    return (s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"], s["elem_offsets"],
            s["atom_radius_ang"], GRID, s["orth"], s["cell_volume"], s["hkl"],
            s["taper_width"])


def sf(s, **kw):
    return structure_factors_one_config(s["frac"], *rest(s), blur=s["blur"],
                                        compile_core=False, **kw)


def sf_batch(s, frac, occ, **kw):
    r = list(rest(s))
    r[1] = occ
    return structure_factors_batch(frac, *r, blur=s["blur"], compile_core=False, **kw)


def hand_built_spheres(s):
    radii = mask_spheres.sphere_radii_from_names(
        s["elements"], s["atom_names"], s["residue_names"], s["residue_ids"])
    return mask_spheres.sphere_kernels(radii, GRID, s["M"], dtype=DTYPE)


def hand_built_atoms(s, source_b=20.0):
    """The atom-presence field's kernels as a caller builds them, and the
    weights its occupancies are multiplied by."""
    present = sorted(set(s["elements"]))
    coeffs = it92_coefficients(present)
    A, lam, offsets, radius, taper_w, e2i = build_atom_kernels_torch(
        s["elements"], present, coeffs, np.full(len(s["elements"]), source_b),
        s["blur"], GRID, s["M"], cutoff=SOURCE_DENSITY_CUTOFF, device="cpu", dtype=DTYPE)
    idx = torch.tensor([e2i[e] for e in s["elements"]], dtype=torch.long)
    weights = torch.tensor(mask_source_weights(s["elements"], coeffs), dtype=DTYPE)
    return (A, lam, offsets, radius, taper_w, idx), weights


def told(model, s, names=True):
    if names:
        model.set_atoms(s["elements"], s["atom_names"], s["residue_names"],
                        s["residue_ids"])
    else:
        model.set_atoms(s["elements"])
    return model


QUIET = dict(check_occupancy=False)


# ----------------------------------------------------------------- auto

def test_auto_picks_spheres_with_names_and_the_atom_field_with_elements():
    s = named_system()
    assert told(SolventModel(), s).resolved_source() == "spheres"
    assert told(SolventModel(), s, names=False).resolved_source() == "atoms"
    # set_atoms alone decides: kernels passed by hand do not, with "auto"
    sk = hand_built_spheres(s)
    with pytest.raises(ValueError, match="say which field"):
        SolventModel().resolved_source(sk)
    with pytest.raises(ValueError, match="say which field"):
        sf(s, solvent=SolventModel(cutoff=1.0, taper_width=0.5), mask_kernels=sk)
    assert told(SolventModel(), s, names=False).resolved_source(sk) == "atoms"
    assert SolventModel(spheres=True).resolved_source(sk) == "spheres"
    assert SolventModel(mask_source="density").resolved_source(sk) == "density"


def test_auto_with_no_atoms_raises_naming_both_fixes():
    """No silent fallback to the density: sampleworks 0.35.0's calls land here."""
    s = named_system()
    model = SolventModel(cutoff=0.3, taper_width=0.15)
    for call in (lambda: sf(s, solvent=model),
                 lambda: model.build_mask(torch.zeros(GRID, dtype=DTYPE), s["orth"]),
                 lambda: model.calibrate(torch.rand(GRID, dtype=DTYPE), s["orth"], 0.5)):
        with pytest.raises(ValueError) as e:
            call()
        assert "set_atoms(" in str(e.value) and "mask_source='density'" in str(e.value)


# ------------------------------------------- set_atoms = the hand-built kernels

def test_set_atoms_spheres_equal_hand_built_kernels_bit_for_bit():
    s = named_system()
    auto = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
    ref = SolventModel(cutoff=1.0, taper_width=0.5, spheres=True, **QUIET)
    sk = hand_built_spheres(s)
    assert torch.equal(sf(s, solvent=auto), sf(s, solvent=ref, mask_kernels=sk))
    # an explicit mask_kernels= still wins over the cached ones
    other = mask_spheres.sphere_kernels([1.4] * len(s["elements"]), GRID, s["M"], dtype=DTYPE)
    assert torch.equal(sf(s, solvent=auto, mask_kernels=other),
                       sf(s, solvent=ref, mask_kernels=other))
    assert not torch.equal(sf(s, solvent=auto, mask_kernels=other), sf(s, solvent=auto))


@pytest.mark.parametrize("blur", [0.0, 5.0])
def test_set_atoms_atom_field_equals_hand_built_kernels_bit_for_bit(blur):
    """The atom-presence field carries the pipeline blur, so the cached
    kernels are keyed on it; mask_occ is occ times the weights."""
    s = named_system(blur=blur)
    auto = told(SolventModel(cutoff=0.05, taper_width=0.025, **QUIET), s, names=False)
    ref = SolventModel(cutoff=0.05, taper_width=0.025, source_b=20.0, **QUIET)
    ak, w = hand_built_atoms(s)
    assert torch.equal(sf(s, solvent=auto),
                       sf(s, solvent=ref, mask_kernels=ak, mask_occ=s["occ"] * w))


@pytest.mark.parametrize("names", [True, False])
def test_set_atoms_through_the_batch_checkpointed_or_not(names):
    """Per-configuration occupancies: the atom field's weights are applied
    to each configuration's row."""
    s = named_system()
    frac = torch.stack([s["frac"], s["frac"] + 0.01, s["frac"] - 0.02])
    occ = torch.stack([s["occ"], 0.9 * s["occ"], s["occ"].flip(0)])
    if names:
        auto = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
        ref = SolventModel(cutoff=1.0, taper_width=0.5, spheres=True, **QUIET)
        kw = dict(mask_kernels=hand_built_spheres(s))
    else:
        auto = told(SolventModel(cutoff=0.05, taper_width=0.025, **QUIET), s, names=False)
        ref = SolventModel(cutoff=0.05, taper_width=0.025, source_b=20.0, **QUIET)
        ak, w = hand_built_atoms(s)
        kw = dict(mask_kernels=ak, mask_occ=occ * w)
    for ck in (False, True):
        assert torch.equal(sf_batch(s, frac, occ, solvent=auto, use_checkpoint=ck),
                           sf_batch(s, frac, occ, solvent=ref, use_checkpoint=ck, **kw))


def test_return_parts_with_cached_kernels():
    s = named_system()
    auto = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
    ref = SolventModel(cutoff=1.0, taper_width=0.5, spheres=True, **QUIET)
    got = sf(s, solvent=auto, return_parts=True)
    want = sf(s, solvent=ref, mask_kernels=hand_built_spheres(s), return_parts=True)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert torch.equal(got.F_total, sf(s, solvent=auto))


def test_cached_kernels_follow_the_grid():
    """A second call on another grid must not reuse the first grid's kernels."""
    s = named_system()
    model = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
    k1, _ = model.cached_mask_kernels("spheres", GRID, s["orth"])
    assert model.cached_mask_kernels("spheres", GRID, s["orth"])[0] is k1
    k2, _ = model.cached_mask_kernels("spheres", (40, 44, 48), s["orth"])
    # twice the grid points over the same cell: finer voxels, more offsets
    assert k2 is not k1 and k2[2][0].offsets.shape[0] > k1[2][0].offsets.shape[0]


# ------------------------------------------------------------ the aliases

def test_spheres_and_source_b_are_aliases():
    s = named_system()
    sk = hand_built_spheres(s)
    a = SolventModel(cutoff=1.0, taper_width=0.5, spheres=True, **QUIET)
    b = SolventModel(cutoff=1.0, taper_width=0.5, mask_source="spheres", **QUIET)
    assert a.mask_blur == b.mask_blur == 0.0
    assert torch.equal(sf(s, solvent=a, mask_kernels=sk), sf(s, solvent=b, mask_kernels=sk))
    ak, w = hand_built_atoms(s)
    a = SolventModel(cutoff=0.05, taper_width=0.025, source_b=20.0, **QUIET)
    b = SolventModel(cutoff=0.05, taper_width=0.025, mask_source="atoms", **QUIET)
    assert torch.equal(sf(s, solvent=a, mask_kernels=ak, mask_occ=s["occ"] * w),
                       sf(s, solvent=b, mask_kernels=ak, mask_occ=s["occ"] * w))
    # auto told the names, with the same kernels passed: the same field,
    # mask_blur and all
    c = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
    assert torch.equal(sf(s, solvent=c, mask_kernels=sk),
                       sf(s, solvent=SolventModel(cutoff=1.0, taper_width=0.5, spheres=True,
                                                  **QUIET), mask_kernels=sk))


def test_contradictory_sources_are_refused():
    with pytest.raises(ValueError, match="mask_source must be one of"):
        SolventModel(mask_source="gemmi")
    with pytest.raises(ValueError, match="contradicts"):
        SolventModel(spheres=True, mask_source="atoms")
    with pytest.raises(ValueError, match="contradicts"):
        SolventModel(source_b=20.0, mask_source="density")
    with pytest.raises(ValueError, match="atom_names AND residue_names"):
        SolventModel().set_atoms(["C"], atom_names=["CA"])
    with pytest.raises(ValueError, match="needs the atoms' names"):
        SolventModel(mask_source="spheres").set_atoms(["C"])


# ---------------------------------------------------------- cutoff=None

def test_cutoff_none_is_the_fixed_level_for_spheres_without_hydrogens():
    s = named_system()
    model = told(SolventModel(**QUIET), s)
    assert model.threshold("spheres") == (SPHERE_LEVEL_FIXED, 0.5 * SPHERE_LEVEL_FIXED)
    assert torch.equal(sf(s, solvent=model),
                       sf(s, solvent=told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)))
    # the taper it was measured with may be given; another one moves the level
    assert told(SolventModel(taper_width=0.5), s).threshold("spheres") == (1.0, 0.5)
    with pytest.raises(ValueError, match="taper_width 0.50"):
        told(SolventModel(taper_width=0.2), s).threshold("spheres")
    # extra smoothing moves the level, so it is not the measured case
    with pytest.raises(ValueError, match="no extra smoothing"):
        told(SolventModel(mask_blur=25.0), s).threshold("spheres")
    # sphere kernels passed by hand may be built any way: no fixed level,
    # with set_atoms or without
    with pytest.raises(ValueError, match="passed by hand"):
        sf(s, solvent=SolventModel(spheres=True, **QUIET), mask_kernels=hand_built_spheres(s))
    with pytest.raises(ValueError, match="passed by hand"):
        sf(s, solvent=model, mask_kernels=hand_built_spheres(s))
    # and a spheres model never told the atoms cannot tell about hydrogens
    with pytest.raises(ValueError, match="cannot tell"):
        SolventModel(spheres=True).threshold("spheres")


def test_cutoff_none_with_hydrogens_needs_calibration():
    s = named_system(with_h=True)
    model = told(SolventModel(), s)
    with pytest.raises(ValueError, match="hydrogens"):
        sf(s, solvent=model)
    field = model.mask_field(s["frac"], s["occ"], GRID, s["orth"])
    model.calibrate(None, s["orth"], 0.6, source=field)
    sf(s, solvent=model)
    assert abs(model.last_occupancy - 0.6) < 2e-3


@pytest.mark.parametrize("blur", [0.0, 5.0])
def test_cutoff_none_on_the_atom_field_needs_calibration(blur):
    s = named_system(blur=blur)
    model = told(SolventModel(), s, names=False)
    with pytest.raises(ValueError, match="no fixed level"):
        sf(s, solvent=model)
    field = model.mask_field(s["frac"], s["occ"], GRID, s["orth"], blur=blur)
    model.calibrate(None, s["orth"], 0.6, pipeline_blur=blur, source=field)
    F = sf(s, solvent=model)
    assert abs(model.last_occupancy - 0.6) < 2e-3
    # the field mask_field places is the one the structure-factor call places
    ak, w = hand_built_atoms(s)
    ref = SolventModel(cutoff=model.cutoff, taper_width=model.taper_width, source_b=20.0, **QUIET)
    assert torch.equal(F, sf(s, solvent=ref, mask_kernels=ak, mask_occ=s["occ"] * w))


def test_density_needs_an_explicit_cutoff_and_taper_defaults_to_half():
    s = named_system()
    with pytest.raises(ValueError, match="e/A"):
        sf(s, solvent=SolventModel(mask_source="density"))
    model = SolventModel(mask_source="density", cutoff=0.3)
    assert model.threshold("density") == (0.3, 0.15)
    rho = compute_density(s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
                          s["elem_offsets"], s["atom_radius_ang"], GRID, s["orth"],
                          s["taper_width"], compile_core=False)
    model = SolventModel(mask_source="density", mask_blur=0.0, **QUIET)
    model.calibrate(rho, s["orth"], 0.6)
    assert abs(float(mask_occupancy(model.build_mask(rho, s["orth"]))) - 0.6) < 2e-3
    with pytest.raises(ValueError, match="no field of its own"):
        model.mask_field(s["frac"], s["occ"], GRID, s["orth"])


# ------------------------------------------------------ names and labels

def test_cctbx_label_helper_splits_a_label():
    an, rn, ri = mask_spheres.names_from_cctbx_labels(['pdb=" CA AMET A   1 "',
                                                       'pdb="HD21BASN B  12A"'])
    assert an == ["CA", "HD21"]
    assert rn == ["MET", "ASN"]
    assert ri == ["MET A   1 ", "ASN B  12A"]


def test_without_residue_ids_one_hydrogen_decides_for_the_whole_model():
    el = ["C", "C", "H"]
    an = ["CB", "CB", "H1"]
    rn = ["ALA", "ALA", "HOH"]
    per_res = mask_spheres.sphere_radii_from_names(el, an, rn, [1, 2, 3])
    whole = mask_spheres.sphere_radii_from_names(el, an, rn)
    # per residue, the alanines have no hydrogen and keep the united-atom CB
    assert per_res[0] == 0.5 * mask_spheres.SIGMA_UA["CH3B"]
    # without ids, the water's hydrogen makes every residue explicit
    assert whole[0] == 0.5 * mask_spheres.SIGMA_UA["C"]
    with pytest.raises(ValueError, match="one entry per atom"):
        mask_spheres.sphere_radii_from_names(el, an[:2], rn)


@requires_cctbx
def test_names_give_the_label_radii_on_a_model_with_hydrogens():
    """8VPW, hydrogens kept: radii from names equal sphere_radii(labels) bit
    for bit; without residue_ids they agree on every residue that holds a
    hydrogen (the whole-model rule makes every residue explicit), and on the
    same model without its hydrogens they agree everywhere. The helper's
    names and ids match the PDB records."""
    import os

    import iotbx.pdb

    inp = iotbx.pdb.input(file_name=os.path.join(
        REPO_ROOT, "lunus", "sf", "examples", "compare_gemmi", "8VPW.pdb"))
    xrs = inp.xray_structure_simple()
    labels = [sc.label for sc in xrs.scatterers()]
    elements = [sc.scattering_type for sc in xrs.scatterers()]
    an, rn, ri = mask_spheres.names_from_cctbx_labels(labels)

    records = list(inp.atoms_with_labels())
    assert len(records) == len(labels)
    assert an == [a.name.strip() for a in records]
    assert rn == [a.resname.strip() for a in records]
    assert ri == ["%3s%2s%4s%1s" % (a.resname, a.chain_id, a.resseq, a.icode)
                  for a in records]

    want = mask_spheres.sphere_radii(labels, elements)
    assert np.array_equal(mask_spheres.sphere_radii_from_names(elements, an, rn, ri), want)

    is_h = np.array([mask_spheres.is_hydrogen(e) for e in elements])
    assert is_h.sum() > 1000
    with_h = {k for k, h in zip(ri, is_h) if h}
    in_h_residue = np.array([k in with_h for k in ri])
    whole = mask_spheres.sphere_radii_from_names(elements, an, rn)
    assert np.array_equal(whole[in_h_residue], want[in_h_residue])

    keep = ~is_h
    sub = [list(np.asarray(x, dtype=object)[keep]) for x in (labels, elements, an, rn)]
    want_heavy = mask_spheres.sphere_radii(sub[0], sub[1])
    assert np.array_equal(mask_spheres.sphere_radii_from_names(sub[1], sub[2], sub[3]),
                          want_heavy)


def test_element_case_and_charges_do_not_matter():
    """biotite gives 'CL', 'FE'; cctbx gives 'Cl1-': the same atom field."""
    orth = torch.tensor(orth_matrix(*CELL), dtype=DTYPE)
    a = SolventModel()
    a.set_atoms(["C", "CL", "FE", "O"])
    b = SolventModel()
    b.set_atoms(["c", "Cl1-", "Fe", "O"])
    ka, wa = a.cached_mask_kernels("atoms", GRID, orth)
    kb, wb = b.cached_mask_kernels("atoms", GRID, orth)
    assert torch.equal(wa, wb) and torch.equal(ka[0], kb[0]) and torch.equal(ka[5], kb[5])


def test_explicit_atoms_source_skips_the_radii():
    """mask_source='atoms' never uses sphere radii, so set_atoms does not
    compute them (an unknown element would otherwise need gemmi)."""
    s = named_system()
    model = told(SolventModel(mask_source="atoms"), s)
    assert model.atom_radii is None and model.resolved_source() == "atoms"


def test_set_atoms_count_must_match_the_coordinates():
    s = named_system()
    model = SolventModel(cutoff=1.0, taper_width=0.5, **QUIET)
    model.set_atoms(s["elements"][:-1])
    with pytest.raises(ValueError, match="told %d atoms" % (len(s["elements"]) - 1)):
        sf(s, solvent=model)
    with pytest.raises(ValueError, match="told"):
        model.mask_field(s["frac"], s["occ"], GRID, s["orth"])


@pytest.mark.parametrize("names", [True, False])
def test_set_atoms_on_a_supercell_with_parts(names):
    """The cached kernels are built for the TARGET grid, as hand-built ones
    are, and the supercell places them on the larger grid unchanged."""
    s = named_system()
    s["frac"] = s["frac"] * torch.tensor([2.0, 1.0, 1.0], dtype=DTYPE)
    if names:
        auto = told(SolventModel(cutoff=1.0, taper_width=0.5, **QUIET), s)
        ref = SolventModel(cutoff=1.0, taper_width=0.5, spheres=True, **QUIET)
        kw = dict(mask_kernels=hand_built_spheres(s))
    else:
        auto = told(SolventModel(cutoff=0.05, taper_width=0.025, **QUIET), s, names=False)
        ref = SolventModel(cutoff=0.05, taper_width=0.025, source_b=20.0, **QUIET)
        ak, w = hand_built_atoms(s)
        kw = dict(mask_kernels=ak, mask_occ=s["occ"] * w)
    got = sf(s, solvent=auto, supercell=(2, 1, 1), return_parts=True)
    want = sf(s, solvent=ref, supercell=(2, 1, 1), return_parts=True, **kw)
    for a, b in zip(got, want):
        assert torch.equal(a, b)


def test_gradients_through_the_cached_kernels():
    """Coordinates and per-configuration occupancies, batch and checkpoint:
    the cached path's gradients are the hand-built path's, bit for bit."""
    s = named_system()
    auto = told(SolventModel(cutoff=0.05, taper_width=0.025, **QUIET), s, names=False)
    ref = SolventModel(cutoff=0.05, taper_width=0.025, source_b=20.0, **QUIET)
    ak, w = hand_built_atoms(s)
    out = {}
    for name, model in (("auto", auto), ("ref", ref)):
        for ck in (False, True):
            frac = torch.stack([s["frac"], s["frac"] + 0.01]).requires_grad_(True)
            occ = torch.stack([s["occ"], 0.8 * s["occ"]]).requires_grad_(True)
            kw = {} if name == "auto" else dict(mask_kernels=ak, mask_occ=occ * w)
            parts = sf_batch(s, frac, occ, solvent=model, use_checkpoint=ck,
                             return_parts=True, **kw)
            (parts.F_total.abs().sum() + parts.F_mask.abs().sum()).backward()
            out[name, ck] = (parts, frac.grad, occ.grad)
    for ck in (False, True):
        for a, b in zip(out["auto", ck][0], out["ref", ck][0]):
            assert torch.equal(a, b)
        assert torch.equal(out["auto", ck][1], out["ref", ck][1])
        assert torch.equal(out["auto", ck][2], out["ref", ck][2])
        assert bool(out["auto", ck][2].abs().sum() > 0)


def test_cached_kernels_follow_blur_dtype_source_b_and_cell():
    s = named_system()
    model = told(SolventModel(**QUIET), s, names=False)
    k0 = model.cached_mask_kernels("atoms", GRID, s["orth"], blur=0.0)
    assert model.cached_mask_kernels("atoms", GRID, s["orth"], blur=0.0) is k0
    assert model.cached_mask_kernels("atoms", GRID, s["orth"], blur=5.0) is not k0
    assert model.cached_mask_kernels("atoms", GRID, s["orth"].float())[0][0].dtype == torch.float32
    model.source_b = 40.0
    k40 = model.cached_mask_kernels("atoms", GRID, s["orth"])
    assert not torch.equal(k40[0][0], k0[0][0])
    # a cell rewritten in place through shared numpy memory: the CPU path
    # compares the values, not the tensor's version counter
    M = np.array(s["M"], dtype=float)
    orth = torch.from_numpy(M)
    k1 = model.cached_mask_kernels("atoms", GRID, orth)
    M *= 1.5
    k2 = model.cached_mask_kernels("atoms", GRID, orth)
    # a 1.5x cell on the same grid: coarser voxels, fewer offsets per atom
    assert k2 is not k1 and k2[0][2][0].offsets.shape[0] < k1[0][2][0].offsets.shape[0]


def test_an_unknown_element_fails_at_set_atoms():
    with pytest.raises(ValueError, match="scattering factor"):
        SolventModel().set_atoms(["C", "Xx"])
