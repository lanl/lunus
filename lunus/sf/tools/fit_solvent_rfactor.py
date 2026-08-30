#!/usr/bin/env python
"""
Fit the bulk-solvent scales against deposited amplitudes, and report R by shell.

    PYTHONPATH=<repo root> python tools/fit_solvent_rfactor.py \\
        examples/compare_gemmi/7FPV.pdb examples/compare_gemmi/7FPV-sf.cif

This is the validation gate named in docs/solvent-design.md: every other check
on the solvent model compares it against itself, against gemmi, or against a
constructed fixture. This one compares it against measurements, which is the
only test that can find a mask model that is self-consistently wrong.

WHAT IS FITTED, AND WHY THAT IS NOT CHEATING. k_sol, b_sol and the overall
scale are refined HERE, in the harness, while the shipped SolventModel defaults
stay fixed at 0.35 / 46 / 1. That split is deliberate and is argued in the
design note: a published refined R comes from refined solvent scales plus
anisotropic scaling, so comparing fixed defaults against it measures the
missing scaling rather than the mask. Fitting three parameters against 70,000
observations is standard practice and is what makes the residual attributable
to the model.

WHAT THE ANSWER MEANS, AND WHAT IT DOES NOT.

  * The target is NOT the deposited R = 0.126 / R-free = 0.145. That came from
    a fully refined model with refined individual anisotropic ADPs. Two
    independent reasons this harness cannot reach it are printed in the caveats
    at the end of every run, and the larger one is not the solvent model.
  * What IS diagnostic: R should fall substantially when the solvent term is
    switched on, the fall should be concentrated below ~5 A, and the fitted
    k_sol / b_sol should land near the conventional 0.35 e/A^3 and 46 A^2.
    Scales that fit well but land somewhere unphysical mean the mask is
    absorbing an error rather than modelling the solvent.

THE CALIBRATION TARGET IS MEASURED, NOT ASSUMED. The cutoff is calibrated so
that our threshold mask calls the same fraction of the cell solvent as gemmi's
geometric mask on the same model (--radii-set, default Refmac). The alternative
-- picking a target from the deposited solvent content, 42.44% for 7FPV -- is
WRONG in a way worth stating, because the deposited figure counts all solvent
while the bulk mask covers only the disordered remainder: the ordered waters
and ligands in the model occupy the rest. gemmi's mask on the same coordinates
already makes that subtraction, which is why it is the target.

--mask gemmi runs the entire fit on gemmi's geometric mask instead of ours.
Comparing the two R-factors separates "the mask model is wrong" from
"everything downstream of the mask is wrong", which no single number can.

--mask-blur B smooths the density with exp(-B s^2/4) BEFORE thresholding, which
is the threshold mask's analogue of the probe radius a geometric mask is built
with. It defaults to SolventModel's own value, so a plain run reports the
SHIPPED configuration; --mask-blur 0 disables it and reproduces what the model
did before this existed, which is where the unphysical scales come from.

--reference prints what this dataset can actually support, computed with cctbx
and mmtbx and no lunus code at all: the R that mmtbx's own bulk solvent reaches
on the same model, and the R that the isotropic-ADP approximation costs on its
own. Both are needed to read the numbers above them, and neither is obtainable
from inside this pipeline.
"""

import argparse
import contextlib
import io
import sys

import numpy as np
import torch


# ---------------------------------------------------------------- observations

def read_observations(path, d_min=None, log=sys.stdout):
    """
    Deposited structure factors -> (miller_array of F, free-set boolean mask).

    The reading path here is verified and fiddly enough to be worth spelling
    out, because three of its four steps are easy to skip and none of them
    fails loudly if you do:

      1. The deposited data are INTENSITIES, not amplitudes, and they are
         ANOMALOUS -- 142,372 Bijvoet-separate observations with sigmas.
      2. merge_equivalents() then average_bijvoet_mates() reduces that to one
         observation per unique reflection.
      3. french_wilson() converts I to F properly, i.e. without simply
         discarding the negative intensities that a weak reflection legitimately
         produces. It rejects a couple of hundred outright and is chatty about
         it on stdout even with log=None, hence the redirect.
      4. The R-free flags are NOT boolean. They are CCP4-convention bins 0-19,
         and the test set is bin 0 -- 3,716 reflections, 5.0%, matching the
         5.010% in the deposited REMARK 3. Reading the column as a boolean
         would silently put 95% of the data in the test set.
    """
    from iotbx.reflection_file_reader import any_reflection_file

    arrays = any_reflection_file(file_name=path).as_miller_arrays()
    intensities = [a for a in arrays if a.is_xray_intensity_array()]
    flags = [a for a in arrays if "r_free_flag" in a.info().label_string()]
    if not intensities:
        raise SystemExit("no intensity array in %s" % path)

    I = intensities[0]
    with contextlib.redirect_stdout(io.StringIO()):
        F = I.merge_equivalents().array().average_bijvoet_mates()
        F = F.french_wilson(log=None)
    if d_min is not None:
        F = F.resolution_filter(d_min=d_min)

    if flags:
        free = flags[0].merge_equivalents().array().average_bijvoet_mates()
        F, free = F.common_sets(free)
        free_mask = np.array(free.data()) == 0
    else:
        print("no R-free flags found: R-free will not be reported", file=log)
        free_mask = np.zeros(F.size(), dtype=bool)

    # (0,0,0) carries no measurable amplitude and would sit alone in the
    # lowest-resolution shell with an infinite d.
    keep = np.array([hkl != (0, 0, 0) for hkl in F.indices()])
    if not keep.all():
        F = F.select(keep.tolist())
        free_mask = free_mask[keep]

    return F, free_mask


# ------------------------------------------------------------------- the model

def read_model(path, log=sys.stdout):
    """Coordinates, isotropic-equivalent B, elements and occupancies, with the
    crystal symmetry taken from the file's own CRYST1 rather than the command
    line -- this tool is pointed at a deposited structure, which knows its own
    cell."""
    from iotbx.pdb import hierarchy

    from cctbx import adptbx

    pdb_in = hierarchy.input(file_name=path, sort_atoms=False)
    xrs = pdb_in.input.xray_structure_simple()

    # BEFORE convert_to_isotropic, which discards it. u_star is in the
    # fractional reciprocal basis; the kernel measures displacements in
    # Cartesian angstroms, and the two coincide only for a cubic cell with
    # a = 1 -- so this conversion is invisible on a toy and wrong on a crystal.
    is_aniso = np.array(xrs.use_u_aniso(), dtype=bool)
    unit_cell = xrs.unit_cell()
    u_cart = np.zeros((xrs.scatterers().size(), 3, 3))
    for i, sc in enumerate(xrs.scatterers()):
        if is_aniso[i]:
            c = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
            u_cart[i] = [[c[0], c[3], c[4]],
                         [c[3], c[1], c[5]],
                         [c[4], c[5], c[2]]]

    n_aniso = int(is_aniso.sum())
    xrs.convert_to_isotropic()

    elements = [s.scattering_type for s in xrs.scatterers()]
    b_per_atom = np.array(xrs.extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    occ = np.array(xrs.scatterers().extract_occupancies())
    frac = np.array(xrs.sites_frac()).reshape(-1, 3)
    # Isotropic scatterers get u_iso * I, which the anisotropic kernel
    # reproduces exactly -- but is_aniso keeps them on the fast path.
    u_cart[~is_aniso] = (np.array(xrs.extract_u_iso_or_u_equiv())[~is_aniso]
                         [:, None, None] * np.eye(3))

    print("%s: %d atoms, B %.1f-%.1f, occupancy %.2f-%.2f, %d with ANISOU"
          % (path, len(elements), b_per_atom.min(), b_per_atom.max(),
             occ.min(), occ.max(), n_aniso), file=log)
    return xrs, elements, b_per_atom, occ, frac, n_aniso, u_cart, is_aniso


def build_density(xrs, elements, b_per_atom, occ, frac, grid, device, dtype,
                  expand_symmetry, log=sys.stdout, u_cart=None,
                  is_aniso=None):
    """Splat, then symmetry-expand on the GRID.

    expand_symmetry is True here and False for a trajectory, and the difference
    is a property of the input model rather than a preference: a deposited PDB
    is one asymmetric unit whose other copies genuinely need generating, while
    an MD box already contains them. Getting it backwards does not raise --
    it symmetry-averages a model that was already complete.
    """
    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.density_torch import splat_density
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.kernel_torch import (build_atom_kernels_aniso_torch,
                                       build_atom_kernels_torch)
    from lunus.sf.symmetry_torch import build_grid_ops_from_cctbx, symmetrize_sum

    cell = xrs.unit_cell().parameters()
    M_np = orth_matrix(*cell)
    present = sorted(set(elements))
    atom_A, atom_lam, offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        elements, present, IT92_COEFFS, b_per_atom, 0.0, grid, M_np,
        cutoff=0.01, device=device, dtype=dtype)

    atom_L6 = aniso_mask = None
    if u_cart is not None:
        # The anisotropic kernels REPLACE atom_A and the radii for every atom,
        # including the isotropic ones -- for which U = u_iso*I gives exactly
        # the isotropic values back, so the two paths stay consistent. Only
        # atom_lam and the dispatch mask distinguish them.
        atom_A, atom_L6, offsets, atom_r, taper_w, _ = \
            build_atom_kernels_aniso_torch(
                elements, present, IT92_COEFFS, u_cart, 0.0, grid, M_np,
                cutoff=0.01, device=device, dtype=dtype)
        aniso_mask = torch.tensor(is_aniso, device=device)
        print("anisotropic ADPs: %d of %d atoms on the tensor kernel"
              % (int(is_aniso.sum()), len(elements)), file=log)

    frac_t = torch.tensor(frac, dtype=dtype, device=device)
    element_idx = torch.tensor([e2i[e] for e in elements], dtype=torch.long,
                               device=device)
    occ_t = torch.tensor(occ, dtype=dtype, device=device)
    orth = torch.tensor(M_np, dtype=dtype, device=device)

    density = splat_density(frac_t, element_idx, occ_t, atom_A, atom_lam,
                            offsets, atom_r, grid, orth, taper_w,
                            compile_core=False, atom_L6=atom_L6,
                            aniso_mask=aniso_mask)
    if expand_symmetry:
        grid_ops = build_grid_ops_from_cctbx(xrs.space_group(), grid)
        density = symmetrize_sum(density, grid_ops)
        print("symmetry-expanded on the grid: %d operations beyond the identity"
              % len(grid_ops), file=log)
    return density, orth, float(np.linalg.det(M_np))


def blur_density(density, orth, blur_b):
    """density * exp(-B s^2 / 4), applied in reciprocal space on the grid.

    THE THRESHOLD MASK'S MISSING PROBE RADIUS. A geometric mask is built from
    vdW radius + probe and then shrunk, and those two steps are what make its
    boundary smooth and simply connected. A density threshold has no equivalent:
    it follows the density's own boundary, including every crevice between
    neighbouring atoms, so the mask carries far more high-resolution structure
    than a geometric one -- measured on 7FPV, 1.5x the |F_mask| in every shell
    beyond 2 A, essentially uncorrelated with gemmi's.

    Smoothing the density first is the cheapest thing that plays the probe's
    role, and it is exact and differentiable on a periodic grid: one FFT pair.
    B is in the same A^2 convention as everything else here, so the Gaussian
    sigma is sqrt(B / (8 pi^2)) -- B = 100 is sigma = 1.13 A, close to a water
    probe radius, which is where the fitted scales come right.
    """
    if blur_b == 0.0:
        return density
    Nu, Nv, Nw = density.shape
    # s^2 for every grid frequency, through the full reciprocal metric rather
    # than 1/a^2 + 1/b^2 + 1/c^2 -- those agree only for an orthogonal cell,
    # and silently do not for a monoclinic or triclinic one.
    Mstar = torch.linalg.inv(orth).T
    fu = torch.fft.fftfreq(Nu, d=1.0 / Nu, device=density.device)
    fv = torch.fft.fftfreq(Nv, d=1.0 / Nv, device=density.device)
    fw = torch.fft.fftfreq(Nw, d=1.0 / Nw, device=density.device)
    H = torch.stack(torch.meshgrid(fu, fv, fw, indexing="ij"), dim=-1)
    s_cart = H.to(Mstar.dtype) @ Mstar.T
    s2 = (s_cart * s_cart).sum(-1).to(density.dtype)
    G = torch.fft.fftn(density) * torch.exp(-0.25 * blur_b * s2)
    return torch.fft.ifftn(G).real


def gemmi_mask_occupancy(pdb_path, grid, radii_set):
    """The occupancy of gemmi's GEOMETRIC mask on the same model and grid.

    This is the calibration target. It is a measurement of the same quantity
    our threshold mask produces -- the fraction of the cell left to the bulk
    once the model's own atoms, ordered waters and ligands included, are
    accounted for -- computed by the conventional method rather than ours.
    """
    import gemmi

    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    g = gemmi.FloatGrid()
    g.set_size(*grid)
    g.set_unit_cell(st.cell)
    g.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    masker = gemmi.SolventMasker(getattr(gemmi.AtomicRadiiSet, radii_set))
    masker.put_mask_on_float_grid(g, st[0])
    arr = np.array(g, copy=False)
    return float(arr.mean()), arr


# --------------------------------------------------------------------- scaling

UNIQUE_IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _scaled_amplitudes(F_calc, inv_d2, s_cart, log_k, u_params, aniso):
    """|F_calc| after the overall scale.

    Isotropic: k * exp(-B s^2 / 4), the same convention as everything else in
    the pipeline.

    Anisotropic: k * exp(-s.B_cart.s / 4), with B_cart a symmetric (3,3) in the
    CARTESIAN reciprocal frame and s = Mstar h in A^-1. Note this is not the
    h.U.h of f_total, deliberately. In the integer-h basis the diagonal
    components differ by the ratio of the cell edges squared -- on 7FPV that is
    70x between a and c -- so an optimizer stepping them together either does
    nothing to one or overflows exp() on another, which is exactly what it did.
    In this frame every component is a B in A^2, the isotropic answer is
    B_cart = B*I, and the problem is well scaled. Converting back to U for
    f_total is U = Mstar^T B_cart Mstar / 4.
    """
    mag = torch.abs(F_calc)
    if aniso:
        B = torch.zeros(3, 3, dtype=inv_d2.dtype, device=inv_d2.device)
        for value, (i, j) in zip(u_params, UNIQUE_IJ):
            B = B + value * _unit_sym(i, j, inv_d2)
        exponent = -0.25 * ((s_cart @ B) * s_cart).sum(-1)
    else:
        exponent = -0.25 * u_params[0] * inv_d2
    # A line search may propose a B that overflows exp() before it is rejected;
    # an inf here becomes a nan gradient and poisons the whole fit rather than
    # producing the large loss that would have made the step be rejected.
    return torch.exp(log_k) * torch.exp(exponent.clamp(max=60.0)) * mag


def _unit_sym(i, j, like):
    E = torch.zeros(3, 3, dtype=like.dtype, device=like.device)
    E[i, j] = 1.0
    E[j, i] = 1.0
    return E


def _f_calc(F_protein, F_mask, inv_d2, k_sol, b_sol):
    """F_protein + k_sol * exp(-b_sol s^2 / 4) * F_mask -- f_total's inner
    bracket, with the outer scale left to _scaled_amplitudes so that the fit
    can treat the two independently."""
    if F_mask is None:
        return F_protein
    weight = k_sol * torch.exp(-0.25 * b_sol * inv_d2)
    return F_protein + weight.to(F_protein.dtype) * F_mask


def optimal_log_k(F_obs, model_amplitudes):
    """The overall scale that minimises sum (Fo - k*g)^2, in closed form.

    Solved rather than refined because it is exactly linear, which keeps it out
    of the search and makes the coarse scan below a scan over two parameters
    instead of four.
    """
    num = (F_obs * model_amplitudes).sum()
    den = (model_amplitudes * model_amplitudes).sum()
    return torch.log(torch.clamp(num / den, min=1e-12))


def r_factor(F_obs, F_model):
    """R = sum||Fo| - |Fc|| / sum|Fo|, the crystallographic convention."""
    return float((F_obs - F_model).abs().sum() / F_obs.abs().sum())


def fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso,
               fit_solvent=True, log=sys.stdout):
    """
    Least squares on amplitudes over the work set. Returns a parameter dict.

    Two stages, because the objective is not convex in (k_sol, b_sol) and a
    gradient method started at the conventional values can and does settle into
    a local minimum that depends on where it began:

      1. A coarse scan over (k_sol, b_sol), with the overall scale solved
         exactly at every node and the overall B scanned alongside. Cheap:
         everything is a vector op over ~70k reflections and the two structure
         factor sets are computed once, before the scan.
      2. LBFGS on every parameter at once from the best node.

    Fitting on amplitudes rather than intensities matches the quantity R is
    computed from. The fit is unweighted -- weighting by 1/sigma^2 would tilt
    it toward the strong low-resolution terms, which is where the solvent lives
    and therefore exactly where an unweighted comparison is more honest.
    """
    Fo = F_obs[work]
    Fp = F_protein[work]
    Fm = None if F_mask is None else F_mask[work]
    s2 = inv_d2[work]
    s = s_cart[work]

    b_grid = torch.linspace(-20.0, 80.0, 26, dtype=inv_d2.dtype,
                            device=inv_d2.device)
    if fit_solvent:
        k_grid = torch.linspace(0.0, 1.2, 25, dtype=inv_d2.dtype,
                                device=inv_d2.device)
        # Deliberately wider than any physical b_sol. A scan that stops at 200
        # returns 199 for a mask carrying too much high-resolution content,
        # which reads as a fit rather than as the pinned boundary it is.
        bs_grid = torch.linspace(0.0, 400.0, 41, dtype=inv_d2.dtype,
                                 device=inv_d2.device)
    else:
        k_grid = torch.zeros(1, dtype=inv_d2.dtype, device=inv_d2.device)
        bs_grid = torch.zeros(1, dtype=inv_d2.dtype, device=inv_d2.device)

    best = None
    with torch.no_grad():
        for k_sol in k_grid:
            for b_sol in bs_grid:
                mag = torch.abs(_f_calc(Fp, Fm, s2, k_sol, b_sol))
                for b_ov in b_grid:
                    g = torch.exp(-0.25 * b_ov * s2) * mag
                    log_k = optimal_log_k(Fo, g)
                    resid = float(((Fo - torch.exp(log_k) * g) ** 2).sum())
                    if best is None or resid < best[0]:
                        best = (resid, float(k_sol), float(b_sol),
                                float(b_ov), float(log_k))
    _, k_sol0, b_sol0, b_ov0, log_k0 = best
    print("  coarse scan: k_sol %.3f, b_sol %.1f, B_overall %.1f"
          % (k_sol0, b_sol0, b_ov0), file=log)

    dtype, device = inv_d2.dtype, inv_d2.device
    log_k = torch.tensor(log_k0, dtype=dtype, device=device, requires_grad=True)
    k_sol = torch.tensor(k_sol0, dtype=dtype, device=device,
                         requires_grad=fit_solvent)
    b_sol = torch.tensor(b_sol0, dtype=dtype, device=device,
                         requires_grad=fit_solvent)
    if aniso:
        # Initialised at the isotropic answer, B_cart = B*I, so the refinement
        # starts where the isotropic fit finished and the off-diagonals start
        # at exactly zero -- which is where orthorhombic symmetry requires them
        # to stay. See the note this prints at the end.
        u_params = [torch.tensor(b_ov0 if i == j else 0.0, dtype=dtype,
                                 device=device, requires_grad=True)
                    for i, j in UNIQUE_IJ]
    else:
        u_params = [torch.tensor(b_ov0, dtype=dtype, device=device,
                                 requires_grad=True)]

    params = [log_k] + u_params + ([k_sol, b_sol] if fit_solvent else [])
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        F_c = _f_calc(Fp, Fm, s2, k_sol, b_sol)
        model = _scaled_amplitudes(F_c, s2, s, log_k, u_params, aniso)
        loss = ((Fo - model) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    return {
        "log_k": log_k.detach(),
        "u_params": [p.detach() for p in u_params],
        "aniso": aniso,
        "k_sol": k_sol.detach(),
        "b_sol": b_sol.detach(),
        "fit_solvent": fit_solvent,
    }



def model_amplitudes(fit, F_protein, F_mask, inv_d2, s_cart):
    with torch.no_grad():
        F_c = _f_calc(F_protein, F_mask if fit["fit_solvent"] else None,
                      inv_d2, fit["k_sol"], fit["b_sol"])
        return _scaled_amplitudes(F_c, inv_d2, s_cart, fit["log_k"],
                                  fit["u_params"], fit["aniso"])


# ----------------------------------------------------------------- diagnostics

def compare_masks(ours, gemmi_grid, F_ours, F_gemmi, d, log=sys.stdout):
    """How our threshold mask differs from the geometric one, in the two places
    where it shows: the voxels, and the transform by resolution shell.

    Worth reporting even when the fit looks fine, because the two disagree in a
    way that a single overall agreement number hides -- they match where the
    solvent term is large and are uncorrelated where it is small.
    """
    gm = torch.tensor(np.asarray(gemmi_grid), dtype=ours.dtype,
                      device=ours.device)
    agreement = float(((ours > 0.5) == (gm > 0.5)).to(ours.dtype).mean())
    print("  binarised voxel agreement with gemmi's mask: %.4f" % agreement,
          file=log)

    a_ours = np.abs(F_ours.detach().cpu().numpy())
    a_gem = np.abs(F_gemmi.detach().cpu().numpy())
    edges = np.percentile(d, np.linspace(0, 100, 6))
    print("  %-16s %8s %10s %10s %8s"
          % ("d range (A)", "n", "|Fm| ours", "/ gemmi", "corr"), file=log)
    for lo, hi in zip(edges[:-1][::-1], edges[1:][::-1]):
        sel = (d >= lo) & (d <= hi)
        corr = np.corrcoef(a_ours[sel], a_gem[sel])[0, 1]
        print("  %6.2f - %-7.2f %8d %10.1f %10.3f %8.4f"
              % (hi, lo, sel.sum(), a_ours[sel].mean(),
                 a_ours[sel].mean() / a_gem[sel].mean(), corr), file=log)
    return agreement


def external_reference(pdb, cif, d_min, log=sys.stdout):
    """What this dataset supports, measured without any lunus code.

    Two numbers, and the harness's output cannot be read without them:

      * mmtbx.f_model's own bulk solvent and scaling on the SAME model. That is
        the R a correct solvent model reaches here, and it is the honest
        target -- not the deposited R, which additionally had every coordinate
        and ADP refined against this data.
      * the same, on the model converted to isotropic ADPs, which is what
        lunus.sf can represent. The gap between the two rows is the cost of the
        missing anisotropic kernel, and on a 1.04 A structure it is large.
    """
    import contextlib
    import io as _io

    import mmtbx.f_model
    from cctbx.array_family import flex
    from iotbx.pdb import hierarchy

    F_obs, free_np = read_observations(cif, d_min=d_min, log=_io.StringIO())
    free_flags = F_obs.customized_copy(
        data=flex.bool(free_np.tolist())).set_observation_type(None)

    xrs_aniso = hierarchy.input(
        file_name=pdb, sort_atoms=False).input.xray_structure_simple()
    xrs_iso = xrs_aniso.deep_copy_scatterers()
    xrs_iso.convert_to_isotropic()

    print("\nexternal reference (cctbx/mmtbx; no lunus.sf code involved)",
          file=log)
    print("  %-34s %8s %8s %8s %8s"
          % ("", "R-work", "R-free", "k_sol", "b_sol"), file=log)
    for name, xrs in (("deposited model, anisotropic ADPs", xrs_aniso),
                      ("same model, isotropic ADPs", xrs_iso)):
        fmodel = mmtbx.f_model.manager(
            f_obs=F_obs, r_free_flags=free_flags, xray_structure=xrs)
        with contextlib.redirect_stdout(_io.StringIO()):
            fmodel.update_all_scales(remove_outliers=False)
        k_sol, b_sol = fmodel.k_sol_b_sol_from_k_mask()
        print("  %-34s %8.4f %8.4f %8.3f %8.1f"
              % (name, fmodel.r_work(), fmodel.r_free(), k_sol, b_sol),
              file=log)
    print("  Which row is the target depends on --aniso-adp: without it the",
          file=log)
    print("  second, with it the first. The gap between them is the",
          file=log)
    print("  anisotropic ADPs.", file=log)


# --------------------------------------------------------------------- report

def shell_table(d, F_obs, model_no_solvent, model_solvent, work, free,
                n_shells, solvent_share, log=sys.stdout):
    """R by resolution shell, lowest resolution first.

    THE LAST COLUMN IS THE ONE THAT SAYS WHERE THE SOLVENT ACTS. It is
    |k_sol exp(-b_sol s^2/4) F_mask| / |F_protein|, the size of the solvent
    term itself, and it dies away by ~4 A as bulk solvent must.

    Read it before reading dR, because dR alone is misleading here: the
    no-solvent column is not a clean baseline but a fit DISTORTED by the
    missing term. With no solvent to absorb the low-resolution deficit the
    overall B compensates, which throws off every shell including the ones
    solvent never reaches -- so dR is large at high resolution while the
    solvent contribution there is nil. That is the scaling recovering, not
    bulk solvent working at 1 A.
    """
    edges = np.percentile(d, np.linspace(0, 100, n_shells + 1))
    print("\n  %-17s %7s  %-15s  %-15s  %8s %8s"
          % ("d range (A)", "n", "R-work  no/with", "R-free  no/with",
             "dR-work", "solvent"), file=log)
    for lo, hi in zip(edges[:-1][::-1], edges[1:][::-1]):
        sel = (d >= lo) & (d <= hi)
        rows = []
        for mask in (sel & work, sel & free):
            if mask.sum() < 2:
                rows.append((float("nan"), float("nan")))
                continue
            rows.append((r_factor(F_obs[mask], model_no_solvent[mask]),
                         r_factor(F_obs[mask], model_solvent[mask])))
        (rw0, rw1), (rf0, rf1) = rows
        print("  %7.2f - %-7.2f %7d  %6.4f / %6.4f  %6.4f / %6.4f  %+8.4f %7.1f%%"
              % (hi, lo, int(sel.sum()), rw0, rw1, rf0, rf1, rw1 - rw0,
                 100.0 * float(solvent_share[sel].mean())), file=log)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("cif", help="deposited structure factors (mmCIF or MTZ)")
    p.add_argument("--d-min", type=float, default=None,
                   help="truncate the observations (default: use them all)")
    p.add_argument("--rate", type=float, default=1.5,
                   help="grid oversampling; spacing is d_min/(2*rate)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--mask", default="threshold", choices=["threshold", "gemmi"],
                   help="ours, or gemmi's geometric mask through the same fit")
    p.add_argument("--radii-set", default="Refmac",
                   choices=["Refmac", "Cctbx", "VanDerWaals"],
                   help="gemmi radii set, for the calibration target and --mask gemmi")
    p.add_argument("--target-occupancy", type=float, default=None,
                   help="override the gemmi-measured calibration target")
    p.add_argument("--taper-frac", type=float, default=0.5,
                   help="taper width as a fraction of the cutoff")
    p.add_argument("--mask-blur", type=float, default=None, metavar="B",
                   help="smooth the density by exp(-B s^2/4) before "
                        "thresholding; the threshold mask's probe radius "
                        "(default: SolventModel's own, so the tool reports "
                        "the shipped configuration; 0 disables it)")
    p.add_argument("--aniso-adp", action="store_true",
                   help="use each atom's ANISOTROPIC DISPLACEMENT PARAMETER "
                        "instead of its isotropic equivalent. Distinct from "
                        "--aniso, which is the overall SCALE tensor: that is a "
                        "fitted correction to |F|, this is the atomic model. "
                        "The gap --reference quantifies is this one -- on 7FPV "
                        "mmtbx reaches 0.130/0.151 with ADPs and 0.179/0.194 "
                        "without.")
    p.add_argument("--reference", action="store_true",
                   help="also report what mmtbx reaches on the same model")
    p.add_argument("--aniso", action="store_true",
                   help="fit a full anisotropic scale tensor instead of one B")
    p.add_argument("--no-expand-symmetry", action="store_true",
                   help="the model already contains every symmetry copy")
    p.add_argument("--shells", type=int, default=10)
    args = p.parse_args()

    from lunus.sf.cell_utils import grid_shape_for_resolution
    from lunus.sf.solvent_torch import (
        MASK_BLUR_DEFAULT, calibrate_cutoff, f_solvent, mask_occupancy,
        shell_voxels, solvent_mask,
    )
    if args.mask_blur is None:
        args.mask_blur = MASK_BLUR_DEFAULT
    from lunus.sf.structure_factor_torch import compute_fcalc, reciprocal_inv_d2
    from lunus.sf.symmetry_torch import adjust_grid_for_symmetry

    device = args.device
    dtype = getattr(torch, args.dtype)

    print("=" * 78)
    print("bulk solvent against deposited amplitudes")
    print("=" * 78)

    F_obs_array, free_np = read_observations(args.cif, d_min=args.d_min)
    d_np = np.array(F_obs_array.d_spacings().data())
    hkl_np = np.array(F_obs_array.indices())
    fobs_np = np.array(F_obs_array.data())
    print("%s: %d amplitudes, %.2f-%.2f A, %d free (%.1f%%)"
          % (args.cif, len(fobs_np), d_np.max(), d_np.min(), free_np.sum(),
             100.0 * free_np.mean()))

    xrs, elements, b_per_atom, occ, frac, n_aniso, u_cart, is_aniso = \
        read_model(args.pdb)
    cell = xrs.unit_cell().parameters()
    d_min_grid = args.d_min if args.d_min is not None else float(d_np.min())
    grid = adjust_grid_for_symmetry(
        grid_shape_for_resolution(cell[0], cell[1], cell[2], d_min_grid,
                                  rate=args.rate),
        [np.array(o.r().as_double()).reshape(3, 3) for o in xrs.space_group()],
        [np.array(o.t().as_double()) for o in xrs.space_group()])
    print("cell %.3f %.3f %.3f %.1f %.1f %.1f, %s, grid %s (%.1f M voxels), %s"
          % (cell + (xrs.space_group().type().lookup_symbol(), grid,
                     np.prod(grid) / 1e6, args.dtype)))

    density, orth, volume = build_density(
        xrs, elements, b_per_atom, occ, frac, grid, device, dtype,
        expand_symmetry=not args.no_expand_symmetry,
        u_cart=u_cart if args.aniso_adp else None, is_aniso=is_aniso)

    hkl = torch.tensor(hkl_np, dtype=torch.long, device=device)
    F_obs = torch.tensor(fobs_np, dtype=dtype, device=device)
    work = ~free_np
    inv_d2 = reciprocal_inv_d2(orth, hkl).to(dtype)
    # s = Mstar h in A^-1, with Mstar = inv(orth).T, so s_cart = hkl @ inv(orth).
    # The same quantity reciprocal_inv_d2 squares and sums; kept as a vector
    # here because the anisotropic scale needs its direction, not just |s|.
    s_cart = hkl.to(dtype) @ torch.linalg.inv(orth)

    with torch.no_grad():
        F_protein = compute_fcalc(density, volume, hkl, orth)

    # ---- the mask
    print("\nsolvent mask")
    target, gemmi_grid = gemmi_mask_occupancy(args.pdb, grid, args.radii_set)
    print("  gemmi %s geometric mask on the same model and grid: occupancy %.4f"
          % (args.radii_set, target))
    if args.target_occupancy is not None:
        target = args.target_occupancy
        print("  overridden by --target-occupancy %.4f" % target)

    if args.mask == "gemmi":
        mask = torch.tensor(np.array(gemmi_grid), dtype=dtype, device=device)
        print("  using gemmi's mask directly (--mask gemmi)")
        occupancy, shell, cutoff = target, 0, float("nan")
    else:
        with torch.no_grad():
            source = blur_density(density, orth, args.mask_blur)
            if args.mask_blur:
                sigma = np.sqrt(args.mask_blur / (8.0 * np.pi ** 2))
                print("  density smoothed by B = %.1f A^2 (sigma %.2f A) "
                      "before thresholding" % (args.mask_blur, sigma))
            cutoff = calibrate_cutoff(source, target,
                                      taper_width_frac=args.taper_frac)
            mask = solvent_mask(source, cutoff, args.taper_frac * cutoff)
        occupancy, shell = float(mask_occupancy(mask)), shell_voxels(mask)
        print("  threshold mask calibrated to it: cutoff %.4g e/A^3 "
              "(%.3f of peak %.3f)" % (cutoff, cutoff / float(source.max()),
                                       float(source.max())))
        print("  occupancy %.4f, taper shell %d voxels (%.2f%% of the grid)"
              % (occupancy, shell, 100.0 * shell / np.prod(grid)))

    with torch.no_grad():
        F_mask = f_solvent(mask, volume, hkl, orth)
        low = d_np > 5.0
        print("  |F_mask|/|F_protein|: %.4f overall, %.4f beyond 5 A"
              % (float(F_mask.abs().mean() / F_protein.abs().mean()),
                 float(F_mask[low].abs().mean() / F_protein[low].abs().mean())))
        if args.mask != "gemmi":
            F_gemmi = f_solvent(
                torch.tensor(np.array(gemmi_grid), dtype=dtype, device=device),
                volume, hkl, orth)
            compare_masks(mask, gemmi_grid, F_mask, F_gemmi, d_np)

    # ---- the fits
    print("\nfitting %d work reflections (%d free held out)"
          % (work.sum(), free_np.sum()))
    print("no solvent:")
    fit0 = fit_scales(F_obs, F_protein, None, inv_d2, s_cart, work, args.aniso,
                      fit_solvent=False)
    print("with solvent:")
    fit1 = fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, args.aniso,
                      fit_solvent=True)

    m0 = model_amplitudes(fit0, F_protein, None, inv_d2, s_cart)
    m1 = model_amplitudes(fit1, F_protein, F_mask, inv_d2, s_cart)

    print("\nfitted parameters")
    print("  %-22s %14s %14s" % ("", "no solvent", "with solvent"))
    print("  %-22s %14.4f %14.4f"
          % ("k_overall", float(torch.exp(fit0["log_k"])),
             float(torch.exp(fit1["log_k"]))))
    if args.aniso:
        names = ["B11", "B22", "B33", "B12", "B13", "B23"]
        for i, name in enumerate(names):
            print("  %-22s %14.2f %14.2f"
                  % (name + " (A^2, cart)", float(fit0["u_params"][i]),
                     float(fit1["u_params"][i])))
    else:
        print("  %-22s %14.2f %14.2f"
              % ("B_overall (A^2)", float(fit0["u_params"][0]),
                 float(fit1["u_params"][0])))
    print("  %-22s %14s %14.4f"
          % ("k_sol (e/A^3)", "-", float(fit1["k_sol"])))
    print("  %-22s %14s %14.4f"
          % ("b_sol (A^2)", "-", float(fit1["b_sol"])))
    print("  %-22s %14s %14s"
          % ("conventional", "-", "0.35 / 46"))

    print("\noverall")
    for name, sel in (("R-work", work), ("R-free", free_np)):
        if sel.sum() < 2:
            continue
        r0, r1 = r_factor(F_obs[sel], m0[sel]), r_factor(F_obs[sel], m1[sel])
        print("  %-8s no solvent %.4f   with solvent %.4f   change %+.4f (%+.1f%%)"
              % (name, r0, r1, r1 - r0, 100.0 * (r1 - r0) / r0))

    with torch.no_grad():
        contribution = (fit1["k_sol"] * torch.exp(-0.25 * fit1["b_sol"] * inv_d2)
                        * F_mask.abs())
        solvent_share = (contribution / F_protein.abs()).cpu().numpy()
    shell_table(d_np, F_obs, m0, m1, work, free_np, args.shells, solvent_share)

    # Equal-count shells put four fifths of a 1.04 A dataset above 2 A, which
    # is precisely where bulk solvent does nothing. The acceptance criterion is
    # about the low-resolution end, so it gets bands of its own.
    print("\nlow resolution, where the acceptance criterion lives")
    print("  %-16s %7s  %-15s  %8s %8s"
          % ("d range (A)", "n", "R-work  no/with", "dR-work", "solvent"))
    bands = [(np.inf, 8.0), (8.0, 5.0), (5.0, 4.0), (4.0, 3.0), (3.0, 2.0)]
    for hi, lo in bands:
        sel = (d_np < hi) & (d_np >= lo) & work
        if sel.sum() < 2:
            continue
        r0, r1 = r_factor(F_obs[sel], m0[sel]), r_factor(F_obs[sel], m1[sel])
        print("  %6.1f - %-7.1f %7d  %6.4f / %6.4f  %+8.4f %7.1f%%"
              % (min(hi, d_np.max()), lo, int(sel.sum()), r0, r1, r1 - r0,
                 100.0 * float(solvent_share[sel].mean())))

    if args.reference:
        external_reference(args.pdb, args.cif, args.d_min)

    print("\ncaveats, in descending order of how much they cost")
    if args.aniso_adp:
        print("  * %d of %d atoms use their full ANISOTROPIC ADP. The largest"
              % (n_aniso, len(elements)))
        print("    approximation in this pipeline is therefore NOT in play, and")
        print("    the target is the deposited-model row of --reference.")
    else:
        print("  * %d of %d atoms were refined with ANISOTROPIC ADPs and are used"
              % (n_aniso, len(elements)))
        print("    here as their isotropic equivalents. This inflates R at HIGH")
        print("    resolution and is unrelated to the solvent model. It is the")
        print("    largest single reason the overall R here is not the deposited")
        print("    one -- pass --aniso-adp to remove it.")
    print("  * No positional, ADP or occupancy refinement happens here at all:")
    print("    the deposited coordinates are used exactly as given, against")
    print("    which only 3-4 scale parameters are fitted.")
    print("  * The fit is unweighted least squares on amplitudes.")
    if args.aniso:
        print("  * The off-diagonal B_cart components are unconstrained. In an")
        print("    ORTHORHOMBIC space group symmetry requires them to be zero,")
        print("    so their fitted magnitudes above are a direct check on the")
        print("    u_aniso code path rather than a modelling result.")
    print("\n  The diagnostic quantity is the CHANGE from solvent and where it")
    print("  falls, not the absolute R. See docs/solvent-design.md, Validation.")


if __name__ == "__main__":
    main()
