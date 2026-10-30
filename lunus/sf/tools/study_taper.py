#!/usr/bin/env python
"""
What the density taper buys an optimizer, measured on a real crystal.

    PYTHONPATH=<repo root> python tools/study_taper.py \\
        examples/compare_gemmi/7FPV.pdb --device cuda --out taper.json

The taper is a C^1 raised cosine over the outermost `taper_width` angstroms
of each atom's truncated density. It is not what makes the calculation
differentiable -- a hard cutoff is differentiable almost everywhere -- but
autograd differentiates the Gaussian with the cutoff sphere held fixed,
so it omits the term from voxels crossing the boundary, and a hard edge is
a sharp feature for the FFT to represent. This script measures how much
either matters, three ways, over a sweep of taper widths and cutoff
densities. The width HARD (1e-6 A) is narrower than any voxel and is a
hard cutoff in all but name; `taper_factor` divides by the width, so a
literal zero is not available.

  fidelity   At the displaced starting structure, compare the autograd
             gradient of the loss with a central finite difference, for a
             sample of atoms, as the step shrinks. Answers "how wrong is
             the gradient at a point?". Read the MAX error as well as the
             median: for one atom a voxel crossing inside a 1e-3 A step is
             a rare event, so a hard cutoff mostly agrees with autograd here
             and shows its boundary term only in the few samples that
             straddle a crossing (a smoke run on 7FPV gave 3e-7 median for
             the hard cutoff). The tapered density, by contrast, shows a
             LARGER finite-difference error at coarse steps (1e-2 A), which
             is the taper's curvature over 0.1 A entering the O(eps^2)
             truncation error, and converges below it. The collective
             effect of thousands of atoms crossing is what the next stage
             measures.

  roughness  Walk the whole structure along one direction in sub-voxel
             steps and record the loss and the directional derivative.
             Voxels crossing the cutoff spheres put jumps in the derivative
             and kinks in the loss; the taper removes them. Answers "is
             the optimum well defined at the voxel scale?". The walk step
             must be well under the taper width to resolve the difference:
             the default 200 steps over one voxel is ~5e-3 A per step.

  optimize   Displace the coordinates and refine them back against
             amplitudes of the true structure with Adam (learning-rate
             sweep) and with L-BFGS. Answers "does it cost steps, or
             convergence?". L-BFGS is the one to watch: quasi-Newton
             methods assume a smooth gradient and stall on a kinked one,
             where Adam averages over the noise.

  schedule   The two-stage refinement the first sweep suggested. On 7FPV
             with the self-consistent target the taper was decisive at a
             1e-2 cutoff (L-BFGS stalled without it) and immaterial at
             1e-4, because the density jump at the boundary IS the cutoff
             value. But an evaluation at 1e-2 costs about a quarter of one
             at 1e-4 (mean radius 1.93 against 3.14 A, work ~ r^3), and the
             tapered 1e-2 optimum was as good as the 1e-4 one. So: L-BFGS
             at the coarse cutoff for a budget of iterations, then rebuild
             the kernels at the fine cutoff, reset the optimizer (its
             curvature pairs describe the coarse surface) and continue.
             Compared against a one-stage run at the fine cutoff with the
             same total iterations, by final loss, rmsd and wall time. The
             target is the FINE model's own amplitudes, or the exact ones
             with --target exact, so both schedules aim at the same point;
             the loss under the fine model at the switch says how far the
             coarse optimum sits from it. Run with and without the taper,
             since the claim is that the taper is what makes stage one
             usable.

The target amplitudes are by default computed with the SAME kernels as the
model (`--target self`), so that the only difference between runs is the
gradient, and the optimum is exactly recoverable. `--target exact` uses
cctbx direct summation of the undisplaced structure instead -- no grid,
no cutoff, no taper -- which is the realistic case and includes the
taper's forward-model error in the residual. Run both and read them
against each other.

Precision. On CUDA the atomic scatter is not bit-reproducible (4e-5
relative on Icalc, docs/performance.md), which a finite difference at a
1e-4 A step would measure instead of the gradient. The fidelity and
roughness stages therefore run in float64 unless `--dtype float32` is
given, and `--deterministic` switches torch to its deterministic (slower)
kernels. The optimize stage follows `--dtype`.

Every stage writes its raw numbers into `--out` (JSON) so the figures can
be made without re-running.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# tools/ is not a package; the model reader lives next door.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_solvent_rfactor import read_model, scattering_table  # noqa: E402

from lunus.sf.cell_utils import (grid_shape_for_resolution, orth_matrix,  # noqa: E402
                                 reciprocal_inv_d2)
from lunus.sf.kernel_torch import build_atom_kernels_torch  # noqa: E402
from lunus.sf.structure_factor_torch import structure_factors_one_config  # noqa: E402
from lunus.sf.symmetry_torch import (adjust_grid_for_symmetry,  # noqa: E402
                                     build_grid_ops_from_cctbx)

HARD = 1e-6          # A; a "taper" no voxel can resolve, i.e. a hard cutoff
DEFAULT_WIDTHS = [HARD, 0.01, 0.03, 0.1, 0.3]
DEFAULT_CUTOFFS = [1e-2, 1e-4]
DEFAULT_EPS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


def width_label(w):
    return "hard" if w <= HARD else "%g" % w


# --------------------------------------------------------------------- setup
def load_system(pdb, d_min, device, dtype, log):
    """Everything about the crystal that does not depend on the taper or the
    cutoff: coordinates, B, occupancies, grid, symmetry operations and the
    full reflection list to d_min."""
    from cctbx import miller

    xrs, elements, b_per_atom, occ, frac, _, _, _ = read_model(pdb, log)
    cell = xrs.unit_cell().parameters()
    M_np = orth_matrix(*cell)
    sg = xrs.space_group()
    rots = [np.array(o.r().as_double()).reshape(3, 3) for o in sg]
    trans = [np.array(o.t().as_double()) for o in sg]
    grid = adjust_grid_for_symmetry(
        grid_shape_for_resolution(cell[0], cell[1], cell[2], d_min), rots, trans)
    grid_ops = build_grid_ops_from_cctbx(sg, grid)

    # The complete unique set to d_min, which is what a refinement sees.
    ms = miller.build_set(crystal_symmetry=xrs.crystal_symmetry(),
                          anomalous_flag=False, d_min=d_min)
    hkl_np = np.array(list(ms.indices()), dtype=np.int64).reshape(-1, 3)
    keep = np.any(hkl_np != 0, axis=1)
    hkl_np = hkl_np[keep]
    from scitbx.array_family import flex
    ms = ms.select(flex.bool(keep.tolist()))

    spacing = min(np.linalg.norm(M_np[:, i]) / grid[i] for i in range(3))
    print("cell %.3f %.3f %.3f %.1f %.1f %.1f, %s, grid %s, voxel %.3f A, "
          "%d reflections to %.2f A, %d symmetry ops beyond the identity"
          % (cell + (sg.type().lookup_symbol(), grid, spacing, len(hkl_np),
                     d_min, len(grid_ops))), file=log)

    present = sorted(set(elements))
    return dict(
        xrs=xrs, ms=ms, elements=elements, present=present,
        coeffs=scattering_table(present, log), b=b_per_atom,
        occ=torch.tensor(occ, dtype=dtype, device=device),
        frac0=torch.tensor(frac, dtype=dtype, device=device),
        grid=grid, grid_ops=grid_ops, M_np=M_np, Minv_np=np.linalg.inv(M_np),
        orth=torch.tensor(M_np, dtype=dtype, device=device),
        Minv=torch.tensor(np.linalg.inv(M_np), dtype=dtype, device=device),
        volume=float(np.linalg.det(M_np)),
        hkl=torch.tensor(hkl_np, dtype=torch.long, device=device),
        spacing=spacing, device=device, dtype=dtype,
    )


def build_kernels(s, cutoff, width):
    """Per-atom kernels for one (cutoff, taper width). Cheap: a few thousand
    atoms. The candidate boxes are sized from the cutoff radius, so they are
    rebuilt per cutoff; the width only changes where inside that radius the
    taper starts."""
    atom_A, atom_lam, offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        s["elements"], s["present"], s["coeffs"], s["b"], 0.0, s["grid"],
        s["M_np"], cutoff=cutoff, taper_width=width,
        device=s["device"], dtype=s["dtype"])
    element_idx = torch.tensor([e2i[e] for e in s["elements"]],
                               dtype=torch.long, device=s["device"])
    return dict(A=atom_A, lam=atom_lam, offsets=offsets, r=atom_r,
                width=taper_w, element_idx=element_idx,
                cutoff=cutoff, label="cutoff=%g width=%s" % (cutoff, width_label(width)))


def forward(s, k, frac, compile_core=False):
    return structure_factors_one_config(
        frac, k["element_idx"], s["occ"], k["A"], k["lam"], k["offsets"],
        k["r"], s["grid"], s["orth"], s["volume"], s["hkl"], k["width"],
        grid_ops=s["grid_ops"], compile_core=compile_core)


def loss_fn(F, target):
    """Normalised amplitude least squares: 0 at the target, 1 is "nothing"."""
    return ((F.abs() - target) ** 2).sum() / (target ** 2).sum()


def exact_amplitudes(s, log):
    """|F| of the undisplaced structure by cctbx direct summation on the same
    reflection list: no grid, no cutoff, no taper. The torch engine's F uses
    the opposite sign convention to cctbx's; amplitudes do not care."""
    xrs = s["xrs"]
    xrs.scattering_type_registry(table="it1992")
    t0 = time.time()
    fc = s["ms"].structure_factors_from_scatterers(
        xray_structure=xrs, algorithm="direct").f_calc()
    amp = np.abs(np.array(fc.data()))
    print("exact target: %d amplitudes by direct summation in %.1f s"
          % (len(amp), time.time() - t0), file=log)
    return torch.tensor(amp, dtype=s["dtype"], device=s["device"])


def cart_to_frac_delta(s, delta_cart):
    """(n,3) Cartesian displacement in A -> fractional displacement."""
    return delta_cart @ s["Minv"].T


def frac_grad_to_cart(s, g_frac):
    """dL/dfrac -> dL/dx_cart. frac = Minv x, so dL/dx = Minv^T dL/dfrac."""
    return g_frac @ s["Minv"]


# ------------------------------------------------------------------ fidelity
def stage_fidelity(s, k, frac_start, target, atoms, eps_list, log):
    """Autograd against central finite differences on a sample of atoms."""
    frac = frac_start.clone().requires_grad_(True)
    L0 = loss_fn(forward(s, k, frac), target)
    L0.backward()
    g_cart = frac_grad_to_cart(s, frac.grad).detach()

    def loss_at(frac_pt):
        with torch.no_grad():
            return float(loss_fn(forward(s, k, frac_pt), target))

    rows = []
    for eps in eps_list:
        auto, fd = [], []
        for i in atoms:
            for a in range(3):
                step = torch.zeros(3, dtype=s["dtype"], device=s["device"])
                step[a] = eps
                dfrac = step @ s["Minv"].T
                fp = frac_start.clone(); fp[i] += dfrac
                fm = frac_start.clone(); fm[i] -= dfrac
                fd.append((loss_at(fp) - loss_at(fm)) / (2 * eps))
                auto.append(float(g_cart[i, a]))
        auto, fd = np.array(auto), np.array(fd)
        rel = np.abs(fd - auto) / np.maximum(np.abs(auto), 1e-300)
        cos = float(auto @ fd / (np.linalg.norm(auto) * np.linalg.norm(fd)))
        rows.append(dict(eps=eps, median_rel_err=float(np.median(rel)),
                         max_rel_err=float(rel.max()), cosine=cos,
                         rms_rel_err=float(np.sqrt(np.mean(rel ** 2)))))
        print("  %-28s eps %-7g  median |fd-auto|/|auto| %.2e  max %.2e  cos %.6f"
              % (k["label"], eps, rows[-1]["median_rel_err"],
                 rows[-1]["max_rel_err"], cos), file=log)
    return dict(loss0=L0.item(), grad_norm=g_cart.norm().item(), rows=rows)


# ----------------------------------------------------------------- roughness
def stage_roughness(s, k, frac_start, target, direction_cart, t_max, n_steps, log):
    """Loss and directional derivative along one line, in sub-voxel steps."""
    ts = np.linspace(0.0, t_max, n_steps + 1)
    d_frac = cart_to_frac_delta(s, direction_cart)
    losses, dd = [], []
    for t in ts:
        frac = (frac_start + t * d_frac).requires_grad_(True)
        L = loss_fn(forward(s, k, frac), target)
        L.backward()
        g_cart = frac_grad_to_cart(s, frac.grad)
        losses.append(float(L.detach()))
        dd.append(float((g_cart * direction_cart).sum()))
    losses, dd = np.array(losses), np.array(dd)
    h = ts[1] - ts[0]
    # Central difference of the loss profile is the derivative the optimizer
    # actually descends; the autograd directional derivative is what it is told.
    fd = (losses[2:] - losses[:-2]) / (2 * h)
    auto = dd[1:-1]
    scale = np.mean(np.abs(auto))
    mismatch = float(np.sqrt(np.mean((fd - auto) ** 2)) / scale)
    jumps = np.abs(np.diff(dd))
    roughness = float(np.sqrt(np.mean(jumps ** 2)) / scale)
    largest = float(jumps.max() / scale)
    print("  %-28s  derivative: rms(fd-auto)/mean|auto| %.3e   "
          "step-to-step rms jump %.3e   largest jump %.3e"
          % (k["label"], mismatch, roughness, largest), file=log)
    return dict(t=ts.tolist(), loss=losses.tolist(), directional=dd.tolist(),
                mismatch=mismatch, roughness=roughness, largest_jump=largest)


# ------------------------------------------------------------------ optimize
def rmsd_to_truth(delta, disp_cart):
    """delta is what the optimizer added; -disp_cart is what it should add."""
    return float(((delta + disp_cart) ** 2).sum(dim=1).mean().sqrt())


def run_adam(s, k, frac_start, disp_cart, target, lr, steps, reach, log):
    delta = torch.zeros_like(disp_cart, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    hist, L_first, hit = [], None, None
    for it in range(steps):
        opt.zero_grad()
        L = loss_fn(forward(s, k, frac_start + cart_to_frac_delta(s, delta)), target)
        L.backward()
        opt.step()
        hist.append((float(L), rmsd_to_truth(delta.detach(), disp_cart)))
        L_first = L_first if L_first is not None else float(L)
        if hit is None and float(L) < reach * L_first:
            hit = it
    final_L, final_rmsd = hist[-1]
    best_L = min(h[0] for h in hist)
    print("  %-28s Adam lr %-6g  loss %.3e -> %.3e (best %.3e)  rmsd %.3f -> %.3f A  "
          "reached %gx start at step %s"
          % (k["label"], lr, hist[0][0], final_L, best_L,
             hist[0][1], final_rmsd, reach, hit), file=log)
    return dict(lr=lr, loss=[h[0] for h in hist], rmsd=[h[1] for h in hist],
                final_loss=final_L, best_loss=best_L, final_rmsd=final_rmsd,
                steps_to_reach=hit)


def run_lbfgs(s, k, frac_start, disp_cart, target, steps, reach, log,
              delta_init=None, label=None):
    """One L-BFGS run of `steps` iterations. delta_init continues from an
    earlier run's displacement (the schedule stage); the optimizer state is
    always fresh. Returns (record, delta)."""
    delta = (torch.zeros_like(disp_cart) if delta_init is None
             else delta_init.detach().clone()).requires_grad_(True)
    opt = torch.optim.LBFGS([delta], lr=1.0, max_iter=steps, history_size=20,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=0.0, tolerance_change=0.0)
    evals = []

    def closure():
        opt.zero_grad()
        L = loss_fn(forward(s, k, frac_start + cart_to_frac_delta(s, delta)), target)
        L.backward()
        evals.append((float(L), rmsd_to_truth(delta.detach(), disp_cart)))
        return L

    opt.step(closure)
    st = opt.state[opt._params[0]]
    n_iter, n_evals = int(st.get("n_iter", 0)), int(st.get("func_evals", len(evals)))
    with torch.no_grad():
        final_L = float(loss_fn(forward(s, k, frac_start + cart_to_frac_delta(s, delta)), target))
    final_rmsd = rmsd_to_truth(delta.detach(), disp_cart)
    hit = next((i for i, e in enumerate(evals) if e[0] < reach * evals[0][0]), None)
    # L-BFGS with max_iter=steps stops early only on a failed line search
    # (or an exactly zero gradient, with the tolerances off), so iterations
    # short of the budget are the stall count that this stage exists to see.
    print("  %-28s L-BFGS  loss %.3e -> %.3e  rmsd %.3f -> %.3f A  "
          "iterations %d of %d  evaluations %d  reached %gx start at eval %s%s"
          % (label or k["label"], evals[0][0], final_L, evals[0][1], final_rmsd,
             n_iter, steps, n_evals, reach, hit,
             "  STOPPED EARLY" if n_iter < steps else ""), file=log)
    return dict(loss=[e[0] for e in evals], rmsd=[e[1] for e in evals],
                final_loss=final_L, final_rmsd=final_rmsd, iterations=n_iter,
                evaluations=n_evals, stopped_early=n_iter < steps,
                evals_to_reach=hit), delta.detach()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------------ schedule
def stage_schedule(s, frac_start, disp_cart, exact_target, width, coarse,
                   fine, coarse_steps, steps, reach, log):
    """One-stage L-BFGS at `fine` against two-stage coarse-then-fine, same
    total iteration budget, same target, timed."""
    k_c = build_kernels(s, coarse, width)
    k_f = build_kernels(s, fine, width)
    with torch.no_grad():
        target = exact_target if exact_target is not None else \
            forward(s, k_f, s["frac0"]).abs()
    tag = "width=%s fine=%g" % (width_label(width), fine)
    print("\n== schedule %s: coarse %g (radius %.2f A) -> fine %g (radius %.2f A), "
          "work ratio ~%.1fx" % (tag, coarse, float(k_c["r"].mean()), fine,
                                 float(k_f["r"].mean()),
                                 (float(k_f["r"].mean()) / float(k_c["r"].mean())) ** 3),
          file=log)

    def timed(fn):
        sync(s["device"]); t0 = time.time()
        out = fn()
        sync(s["device"]); return out, time.time() - t0

    (one, _), t_one = timed(lambda: run_lbfgs(
        s, k_f, frac_start, disp_cart, target, steps, reach, log,
        label="one-stage %s" % tag))

    (st1, delta1), t_c = timed(lambda: run_lbfgs(
        s, k_c, frac_start, disp_cart, target, coarse_steps, reach, log,
        label="stage 1 (coarse) %s" % tag))
    with torch.no_grad():                   # where the coarse optimum sits,
        L_switch = float(loss_fn(forward(   # seen by the fine model
            s, k_f, frac_start + cart_to_frac_delta(s, delta1)), target))
    (st2, _), t_f = timed(lambda: run_lbfgs(
        s, k_f, frac_start, disp_cart, target, steps - coarse_steps, reach, log,
        delta_init=delta1, label="stage 2 (fine) %s" % tag))
    print("  %-28s one-stage %.3e / %.3f A in %.1f s;  two-stage %.3e / %.3f A "
          "in %.1f s (%.1f + %.1f);  loss under the fine model at the switch %.3e"
          % (tag, one["final_loss"], one["final_rmsd"], t_one,
             st2["final_loss"], st2["final_rmsd"], t_c + t_f, t_c, t_f, L_switch),
          file=log)
    return dict(width=width, width_label=width_label(width), coarse=coarse,
                fine=fine, coarse_steps=coarse_steps, steps=steps,
                radius_coarse=float(k_c["r"].mean()), radius_fine=float(k_f["r"].mean()),
                one_stage=dict(one, seconds=t_one),
                stage1=dict(st1, seconds=t_c), loss_at_switch_fine=L_switch,
                stage2=dict(st2, seconds=t_f), two_stage_seconds=t_c + t_f)


# ---------------------------------------------------------------------- main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb", help="one asymmetric unit, e.g. examples/compare_gemmi/7FPV.pdb")
    p.add_argument("--d-min", type=float, default=2.0)
    p.add_argument("--widths", type=float, nargs="+", default=DEFAULT_WIDTHS,
                   help="taper widths, A; %g means a hard cutoff (default %s)"
                        % (HARD, " ".join(width_label(w) for w in DEFAULT_WIDTHS)))
    p.add_argument("--cutoffs", type=float, nargs="+", default=DEFAULT_CUTOFFS,
                   help="density cutoffs, e/A^3 (default %s)" % DEFAULT_CUTOFFS)
    p.add_argument("--stages", nargs="+", default=["fidelity", "roughness", "optimize"],
                   choices=["fidelity", "roughness", "optimize", "schedule"],
                   help="schedule is not in the default set; it ignores "
                        "--widths/--cutoffs and uses the --schedule-* options")
    p.add_argument("--target", default="self", choices=["self", "exact"],
                   help="amplitudes to fit: the model's own kernels on the true "
                        "coordinates (self), or cctbx direct summation (exact)")
    p.add_argument("--displace", type=float, default=0.3,
                   help="RMS displacement of the starting structure, A")
    p.add_argument("--seed", type=int, default=0)
    # fidelity
    p.add_argument("--fd-atoms", type=int, default=30,
                   help="atoms sampled for the finite-difference check")
    p.add_argument("--eps", type=float, nargs="+", default=DEFAULT_EPS,
                   help="finite-difference steps, A")
    # roughness
    p.add_argument("--walk-steps", type=int, default=200)
    p.add_argument("--walk-length", type=float, default=None,
                   help="length of the walk, A (default: one voxel)")
    # optimize
    p.add_argument("--steps", type=int, default=200, help="per optimizer run")
    p.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03],
                   help="Adam learning rates, A per step")
    p.add_argument("--no-lbfgs", action="store_true")
    p.add_argument("--reach", type=float, default=0.1,
                   help="loss fraction of the start that counts as 'reached'")
    # schedule
    p.add_argument("--schedule-widths", type=float, nargs="+", default=[HARD, 0.1],
                   help="taper widths for the schedule stage (default: hard "
                        "and 0.1, since the claim is that the taper makes "
                        "the coarse stage usable)")
    p.add_argument("--schedule-coarse", type=float, default=1e-2,
                   help="stage-one cutoff, e/A^3")
    p.add_argument("--schedule-fine", type=float, nargs="+", default=[1e-4, 1e-5],
                   help="stage-two cutoffs, each also run one-stage for comparison")
    p.add_argument("--schedule-coarse-steps", type=int, default=None,
                   help="L-BFGS iterations in stage one (default: half of --steps)")
    # machinery
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                   help="for the optimize stage; fidelity and roughness always "
                        "use float64 unless this is float32")
    p.add_argument("--deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms(True), with the "
                        "CUBLAS_WORKSPACE_CONFIG it needs; slower on CUDA but "
                        "makes finite differences trustworthy there")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the density core; worth it only for a "
                        "long optimize stage")
    p.add_argument("--out", default="taper_study.json")
    return p


def main():
    args = build_parser().parse_args()
    log = sys.stdout
    if args.deterministic:
        # cuBLAS is only deterministic with a fixed workspace, and it reads
        # this variable when its handle is created, i.e. at the first matmul,
        # so it must be set here and not merely documented. Without it the
        # very first cart_to_frac_delta() raises on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    dtype = dict(float32=torch.float32, float64=torch.float64)[args.dtype]
    if args.device == "mps" and dtype is torch.float64:
        raise SystemExit("MPS has no float64; pass --dtype float32 (and do not "
                         "trust the fidelity stage below eps ~1e-2 there)")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    s = load_system(args.pdb, args.d_min, args.device, dtype, log)
    n_atoms = s["frac0"].shape[0]

    # One displacement and one walk direction, shared by every configuration
    # so the sweep compares like with like.
    disp_np = rng.normal(size=(n_atoms, 3)) * (args.displace / math.sqrt(3.0))
    disp_cart = torch.tensor(disp_np, dtype=dtype, device=args.device)
    frac_start = (s["frac0"] + cart_to_frac_delta(s, disp_cart)).detach()
    direction_np = rng.normal(size=(n_atoms, 3))
    direction_np /= np.sqrt((direction_np ** 2).sum(axis=1).mean())   # RMS 1 A per unit t
    direction_cart = torch.tensor(direction_np, dtype=dtype, device=args.device)
    fd_atoms = sorted(rng.choice(n_atoms, size=args.fd_atoms, replace=False).tolist())
    walk_length = args.walk_length or s["spacing"]
    print("starting structure displaced by %.3f A rms; walk of %.3f A in %d steps; "
          "%d atoms in the finite-difference sample"
          % (args.displace, walk_length, args.walk_steps, len(fd_atoms)), file=log)

    exact_target = exact_amplitudes(s, log) if args.target == "exact" else None

    results = dict(args=vars(args), grid=list(s["grid"]), n_atoms=n_atoms,
                   n_refl=int(s["hkl"].shape[0]), voxel=s["spacing"], runs=[])
    sweep_stages = [st for st in args.stages if st != "schedule"]
    for cutoff in (args.cutoffs if sweep_stages else []):
        for width in args.widths:
            k = build_kernels(s, cutoff, width)
            with torch.no_grad():
                target = exact_target if exact_target is not None else \
                    forward(s, k, s["frac0"]).abs()
                L_start = float(loss_fn(forward(s, k, frac_start), target))
            print("\n== %s  (loss at the displaced start %.3e, mean radius %.2f A)"
                  % (k["label"], L_start, float(k["r"].mean())), file=log)
            run = dict(cutoff=cutoff, width=width, width_label=width_label(width),
                       loss_start=L_start, mean_radius=float(k["r"].mean()))

            if "fidelity" in args.stages:
                t0 = time.time()
                run["fidelity"] = stage_fidelity(s, k, frac_start, target, fd_atoms,
                                                 args.eps, log)
                print("  fidelity: %.0f s" % (time.time() - t0), file=log)
            if "roughness" in args.stages:
                t0 = time.time()
                run["roughness"] = stage_roughness(s, k, frac_start, target,
                                                   direction_cart, walk_length,
                                                   args.walk_steps, log)
                print("  roughness: %.0f s" % (time.time() - t0), file=log)
            if "optimize" in args.stages:
                t0 = time.time()
                run["adam"] = [run_adam(s, k, frac_start, disp_cart, target, lr,
                                        args.steps, args.reach, log)
                               for lr in args.lrs]
                if not args.no_lbfgs:
                    run["lbfgs"], _ = run_lbfgs(s, k, frac_start, disp_cart,
                                                target, args.steps, args.reach, log)
                print("  optimize: %.0f s" % (time.time() - t0), file=log)
            results["runs"].append(run)
            with open(args.out, "w") as f:          # after every run, so a
                json.dump(results, f, indent=1)     # killed sweep keeps its data

    if "schedule" in args.stages:
        coarse_steps = args.schedule_coarse_steps or args.steps // 2
        results["schedule"] = []
        for width in args.schedule_widths:
            for fine in args.schedule_fine:
                results["schedule"].append(stage_schedule(
                    s, frac_start, disp_cart, exact_target, width,
                    args.schedule_coarse, fine, coarse_steps, args.steps,
                    args.reach, log))
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)

    # ---- summary tables -------------------------------------------------
    runs = results["runs"]
    if "fidelity" in args.stages:
        eps_last = args.eps[-1]
        print("\nFidelity: median |fd - auto| / |auto| at the smallest step (%g A)"
              % eps_last, file=log)
        print("  %-10s" % "cutoff" + "".join("%12s" % r["width_label"] for r in runs
                                             if r["cutoff"] == args.cutoffs[0]), file=log)
        for c in args.cutoffs:
            print("  %-10g" % c + "".join(
                "%12.2e" % r["fidelity"]["rows"][-1]["median_rel_err"]
                for r in runs if r["cutoff"] == c), file=log)
    if "roughness" in args.stages:
        print("\nRoughness: rms step-to-step jump in the directional derivative, "
              "relative to its mean magnitude", file=log)
        for c in args.cutoffs:
            print("  %-10g" % c + "".join(
                "%12.2e" % r["roughness"]["roughness"]
                for r in runs if r["cutoff"] == c), file=log)
    if "optimize" in args.stages:
        print("\nOptimize: best Adam final rmsd (A) over learning rates"
              + ("" if args.no_lbfgs else "; L-BFGS final rmsd and iterations"), file=log)
        for c in args.cutoffs:
            cells = []
            for r in runs:
                if r["cutoff"] != c:
                    continue
                best = min(a["final_rmsd"] for a in r["adam"])
                cell = "%.3f" % best
                if not args.no_lbfgs:
                    cell += "/%.3f(%d)" % (r["lbfgs"]["final_rmsd"], r["lbfgs"]["iterations"])
                cells.append("%18s" % cell)
            print("  %-10g" % c + "".join(cells), file=log)
    if "schedule" in args.stages:
        print("\nSchedule: L-BFGS, %d iterations total; two-stage spends %d at "
              "cutoff %g first" % (args.steps, coarse_steps, args.schedule_coarse),
              file=log)
        print("  %-6s %-7s %14s %14s %12s %14s" % (
            "width", "fine", "one-stage", "two-stage", "at switch", "seconds"), file=log)
        print("  %-6s %-7s %14s %14s %12s %14s" % (
            "", "", "loss / rmsd", "loss / rmsd", "fine loss", "one / two"), file=log)
        for r in results["schedule"]:
            print("  %-6s %-7g %8.2e/%.3f %8.2e/%.3f %12.2e %6.1f / %6.1f"
                  % (r["width_label"], r["fine"],
                     r["one_stage"]["final_loss"], r["one_stage"]["final_rmsd"],
                     r["stage2"]["final_loss"], r["stage2"]["final_rmsd"],
                     r["loss_at_switch_fine"], r["one_stage"]["seconds"],
                     r["two_stage_seconds"]), file=log)
    print("\nwrote %s" % args.out, file=log)


if __name__ == "__main__":
    main()
