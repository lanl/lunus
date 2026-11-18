"""
Bragg scaling: the model between |F_calc| and |F_obs|, its least-squares fit,
and R-work / R-free. Tensors in, tensors out; torch and numpy only, no cctbx.

    |F_model(h)| = k_bin(h) exp(-s.B_cart.s / 4) |F_protein(h) + k_mask(h) F_mask(h)|

k_bin is one overall scale per resolution bin (or one scale); B_cart a
symmetric 3x3 in the Cartesian reciprocal frame (B*I when isotropic);
k_mask one solvent scale per bin, as mmtbx fits it, or the exponential
k_sol exp(-b_sol s^2 / 4). See docs/scaling-design.md.

The flat functions are the arithmetic of tools/fit_solvent_rfactor.py, moved
here unchanged so that the tool and a guidance loss use one implementation;
the tool imports them back. BraggScaler bundles the per-reflection-list state
(geometry, bins, the fitted parameters) for a caller that scales many
evaluations against one data set: fit once, refit the bin scales in closed
form per evaluation, and apply() the model differentiably in F_protein and
F_mask.

solvent_torch.f_total is this model's single-scale exponential case with
fixed parameters (U = Mstar^T B_cart Mstar / 4); it is SolventModel's forward
model, and this module is what fits the parameters to data. Neither calls the
other.
"""

import io
import math
import numbers
import sys
import numpy as np
import torch


UNIQUE_IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def scaled_amplitudes(F_calc, inv_d2, s_cart, log_k, u_params, aniso):
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


def inner_f_calc(F_protein, F_mask, inv_d2, k_sol, b_sol):
    """F_protein + k_sol * exp(-b_sol s^2 / 4) * F_mask -- f_total's inner
    bracket, with the outer scale left to scaled_amplitudes so that the fit
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


def resolution_bins(inv_d2, n_bins):
    """Equal-count resolution bins: an index 0..n_bins-1 per reflection, 0 the
    lowest resolution. Free reflections are binned with the rest, so that a
    scale fitted on the work set applies to them unchanged."""
    order = torch.argsort(inv_d2)
    bins = torch.empty_like(order)
    bins[order] = (torch.arange(len(order), device=order.device) * n_bins
                   // len(order))
    return bins


# Derived from cctbx; see lunus/sf/THIRD_PARTY_NOTICES for its license.
def log_resolution_bins(d, n_lowest=100, min_per_bin=50, max_bins=30,
                        eps=1e-4, device="cpu"):
    """mmtbx's resolution bins: an index per reflection, 0 the lowest
    resolution, from d spacings in A.

    A port of cctbx.miller.set.log_binning (Urzhumtsev et al. 2009), which is
    what mmtbx.f_model bins its scales by, with the same defaults: the first
    bin holds the n_lowest lowest-resolution reflections, every later bin is a
    step of equal width in ln(1/d^2) set by the second n_lowest, the step is
    widened if that would make more than max_bins, a last bin under a quarter
    of the one before is merged into it, and any bin under min_per_bin
    reflections is merged into the one before. Fine at low resolution, where
    the solvent term and the scale change fastest; few and wide at high
    resolution; and fewer bins when there are fewer reflections, which
    equal-count bins (resolution_bins) do not do. Fewer than 2*n_lowest
    reflections is one bin.

    Pass the d spacings cctbx itself computed for these reflections
    (F_obs_array.d_spacings()) to get mmtbx's bins exactly: limits fall on
    data values by construction, so recomputing d some other way can move a
    reflection that sits on a limit. Free reflections are binned with the
    rest, as in resolution_bins.

    One deliberate difference from cctbx: if reflections n_lowest to
    2*n_lowest share one d spacing the bin step is zero, and cctbx's loop
    never ends; this raises ValueError instead. Merged data do not do that."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if 2 * n_lowest >= n:
        return torch.zeros(n, dtype=torch.long, device=device)
    assert max_bins > 1
    d_sorted = np.sort(d)[::-1]               # lowest resolution first

    # Scalar math, not numpy: only these four ln(1/d^2) are ever used, and
    # the limits sit on data values, so a reflection on a limit lands in the
    # bin cctbx puts it in only if the roundoff is libm's, as in cctbx --
    # numpy's vectorised log/exp can differ in the last place.
    def lnss(x):
        return math.log(1.0 / (x * x))

    def d_of(x):
        return math.sqrt(1.0 / math.exp(x))

    s0, s1, s2, top = (lnss(float(d_sorted[i]))
                       for i in (0, n_lowest, 2 * n_lowest, n - 1))
    if s2 == s1:
        raise ValueError("log_resolution_bins: reflections %d to %d share one "
                         "d spacing, so the bin step is zero"
                         % (n_lowest, 2 * n_lowest))

    def limits_for(step, cap=None):
        # cap: stop as soon as there are more limits than max_bins. The
        # first call's result is then discarded in favour of the rescaled
        # step, so a step that is tiny but not zero cannot loop for as long
        # as cctbx's would.
        limits = [d_of(s0)]
        x = s1
        while x <= top:
            if cap is not None and len(limits) > cap:
                return limits
            limits.append(d_of(x))
            x += step
        limits.append(d_of(min(top, x)))
        return limits

    limits = limits_for(s2 - s1, cap=max_bins)
    if len(limits) > max_bins:
        limits = limits_for((top - s1) / (max_bins - 1))
    pairs = []
    for i in range(len(limits)):
        if i < len(limits) - 2:
            pairs.append((limits[i], limits[i + 1]))
        elif i == len(limits) - 1:
            pairs.append((limits[i - 1], limits[i]))
    selections = []
    for i, (d_max, d_min) in enumerate(pairs):
        if i == 0:
            sel = (d <= d_max + eps) & (d > d_min)
        elif i == len(pairs) - 1:
            sel = (d <= d_max) & (d > d_min - eps)
        else:
            sel = (d <= d_max) & (d > d_min)
        selections.append(sel)
    last, before = selections[-1], selections[-2]
    if last.sum() < before.sum() / 4:
        selections = selections[:-2] + [before | last]
    merged = []
    for sel in selections:
        if sel.sum() < min_per_bin and merged:
            merged[-1] = merged[-1] | sel
        else:
            merged.append(sel)
    bins = np.full(n, -1, dtype=np.int64)
    for i, sel in enumerate(merged):
        bins[sel] = i
    if (bins < 0).any():
        raise ValueError("log_resolution_bins left %d reflections unbinned"
                         % int((bins < 0).sum()))
    return torch.as_tensor(bins, device=device)


def assign_scale_bins(spec, inv_d2, d):
    """The bin index per reflection for a scale_bins_arg value, or None for 0.
    inv_d2 is the torch tensor the fit uses; d the cctbx d spacings of the
    same reflections, in the same order."""
    if spec == "log":
        return log_resolution_bins(d, device=inv_d2.device)
    if spec:
        return resolution_bins(inv_d2, spec)
    return None


# Derived from cctbx; see lunus/sf/THIRD_PARTY_NOTICES for its license.
def mmtbx_moving_average(x):
    """mmtbx.bulk_solvent.scaler.moving_average, ported: five passes of a
    three-point mean applied only at local extrema, ends padded by
    repetition. Takes and returns a list of floats."""
    x_ = [x[0]] + list(x) + [x[-1]]
    for _ in range(5):
        result = x_[:]
        extremum = [i for i in range(1, len(result) - 1)
                    if ((result[i - 1] < result[i] and result[i + 1] < result[i]) or
                        (result[i - 1] > result[i] and result[i + 1] > result[i]))]
        for i in extremum:
            result[i] = (x_[i - 1] + x_[i] + x_[i + 1]) / 3.0
        x_ = result[:]
    return result[1:-1]


def solvent_curve_rule(k_bins, d_min_bins, smooth=True, d_cut=3.0):
    """mmtbx's regularisation of a per-bin solvent scale (scaler.smooth and
    the non-negative grid it searches): negatives to zero; from the first
    bin that is zero AND whose high-resolution edge is below d_cut, zero in
    every bin from there on; and the moving average at extrema. The solvent
    term has no signal at high resolution and a free bin fits noise
    (unregularised, 36 of 40 SW40 entries went negative there).

    Values under 0.005, half of mmtbx's grid step, count as zero.

    One deliberate difference in order: mmtbx smooths first and truncates
    after, which works for its grid search because that lands on exact
    zeros in runs. A least-squares bin lands on noise around zero instead,
    and smoothing first would average a clamped zero with its neighbours and
    re-admit the noise; so here the cut is found on the clamped values and
    the smoothing is applied to the bins that survive it. k_bins is a
    tensor, low resolution first; d_min_bins the bins' high-resolution edges
    in A."""
    # Values below half of mmtbx's grid step (0.01) are zero, as the grid
    # would have them. This is what makes the cut fire on a numerical zero:
    # a least-squares bin in a region with no solvent lands at +-1e-6 with a
    # sign the optimizer decides, and without the snap the cut would depend
    # on that sign. It also keeps a 0.001 bin next to a zero from being a
    # strict extremum that the moving average lifts off the floor. The edge
    # therefore sits at 0.005: a bin at 0.0051 is kept and smoothed, one at
    # 0.0049 is a zero, which is the same jump mmtbx's grid has at its step.
    k = [float(v) if float(v) >= 0.005 else 0.0 for v in k_bins]
    keep = solvent_cut_index(k, d_min_bins, d_cut)
    head = k[:keep]
    if smooth and len(head) > 2:
        head = mmtbx_moving_average(head)      # a mean of >= 0 values is >= 0
    out = head + [0.0] * (len(k) - keep)
    return torch.tensor(out, dtype=k_bins.dtype, device=k_bins.device)


def solvent_cut_index(k_bins, d_min_bins, d_cut=3.0):
    """Index of the first bin that is non-positive AND has its high-resolution
    edge below d_cut; bins from there on carry no solvent. len(k_bins) when
    there is none."""
    for i, (v, d) in enumerate(zip(k_bins, d_min_bins)):
        if float(v) <= 0.0 and d < d_cut:
            return i
    return len(k_bins)


def exponential_from_curve(k_bins, bins, inv_d2, d_cut=4.0):
    """(k_sol, b_sol, n): the exponential k exp(-b s^2/4) closest, in least
    squares over the reflections at d >= d_cut, to the per-bin curve they
    see; n is how many reflections that was. The same quantity mmtbx reports
    as k_sol/b_sol from its k_mask curve (f_model.k_sol_b_sol_from_k_mask:
    reflections at d >= 4 A, k in [0, 0.6], b in [0, 150]; mmtbx searches
    locally around its own starting guess, this is a global grid at the same
    bounds, so the two can differ in the second digit). A diagnostic, not a
    model parameter. With fewer than 2 reflections at d >= d_cut there is
    nothing to fit: n is 0 and the result is (max of the curve, 0)."""
    s2 = inv_d2.detach().to(torch.float64).cpu()
    k_refl = k_bins.detach().to(torch.float64).cpu()[bins.cpu()]
    sel = s2 <= 1.0 / d_cut ** 2
    n = int(sel.sum())
    if n < 2:
        return min(0.6, max(0.0, float(k_bins.max()))), 0.0, 0
    k, s2 = k_refl[sel], s2[sel]
    k_grid = torch.arange(0.0, 0.6001, 0.005, dtype=torch.float64)
    b_grid = torch.arange(0.0, 150.001, 1.0, dtype=torch.float64)
    # sum over reflections of (k_g e^{-b s2/4} - k)^2, expanded so that the
    # per-reflection work is two reductions rather than a (121, 151, n) grid
    e = torch.exp(-0.25 * b_grid[:, None] * s2[None, :])           # (151, n)
    ee = (e * e).sum(1)                                             # (151,)
    ek = (e * k[None, :]).sum(1)                                    # (151,)
    cost = (k_grid[:, None] ** 2) * ee[None, :] - 2.0 * k_grid[:, None] * ek[None, :]
    i, j = divmod(int(torch.argmin(cost)), len(b_grid))
    return float(k_grid[i]), float(b_grid[j]), n


def solvent_scheme(fit):
    """"per bin" or "exponential": which solvent scale a fit_scales result
    used, for printed output and the result dict."""
    return "per bin" if fit.get("k_mask_bins") is not None else "exponential"


def solvent_weight(fit, inv_d2):
    """The factor on F_mask the fit applies, per reflection: k_mask[bin] for
    a per-bin solvent fit, k_sol exp(-b_sol s^2/4) otherwise, zero for a fit
    without solvent. model_amplitudes and the shell table's solvent column
    both take it from here, so they cannot disagree about which solvent
    model was fitted -- the k_sol/b_sol of a per-bin fit are a diagnostic
    the model never used."""
    if fit.get("k_mask_bins") is not None:
        return fit["k_mask_bins"][fit["bins"]]
    if not fit["fit_solvent"]:
        return torch.zeros_like(inv_d2)
    return fit["k_sol"] * torch.exp(-0.25 * fit["b_sol"] * inv_d2)


def scan_grids(fit_solvent, like):
    """The (k_sol, b_sol, B_overall) grids of fit_scales' coarse scan, on
    like's dtype and device. One place, so that the equivalence tests scan
    the grid the fit scans. Without the solvent the first two are the single
    node 0."""
    dtype, device = like.dtype, like.device
    b_grid = torch.linspace(-20.0, 80.0, 26, dtype=dtype, device=device)
    if fit_solvent:
        k_grid = torch.linspace(0.0, 1.2, 25, dtype=dtype, device=device)
        # Deliberately wider than any physical b_sol. A scan that stops at 200
        # returns 199 for a mask carrying too much high-resolution content,
        # which reads as a fit rather than as the pinned boundary it is.
        bs_grid = torch.linspace(0.0, 400.0, 41, dtype=dtype, device=device)
    else:
        k_grid = torch.zeros(1, dtype=dtype, device=device)
        bs_grid = torch.zeros(1, dtype=dtype, device=device)
    return k_grid, bs_grid, b_grid


def coarse_scan(Fo, Fp, Fm, s2, k_grid, bs_grid, b_grid):
    """Stage 1 of fit_scales: the node of the (k_sol, b_sol, B_overall) grid
    with the smallest least-squares residual, the overall scale solved
    exactly at every node. Returns (k_sol, b_sol, B_overall, log_k) as
    floats, the same node coarse_scan_reference finds.

    Batched: for each b_sol the amplitudes |F_p + k_sol w(b_sol) F_m| are
    formed for every k_sol at once, (n_k, n_refl), and the two sums the
    residual needs at every (k_sol, B_overall) node,

        num = sum_h Fo e^{-B s^2/4} |F_c|,   den = sum_h e^{-B s^2/4 * 2} |F_c|^2,

    are two matrix products against the (n_B, n_refl) table of overall-B
    factors. The residual at the exact scale k = num/den is
    sum Fo^2 - num^2/den, which is the loop's sum (Fo - k g)^2 expanded, so
    nothing is fitted differently. What changes is the shape of the work:
    the loop ran 26,650 nodes as separate small kernels and read each
    residual back as a Python float, which on a GPU is a device
    synchronisation per node and was ~4 s of every entry's wall time there
    (measured on the torchref1000 pilot, 2026-11-10); here it is 41 passes
    of a few batched operations and one synchronisation at the end. The
    minimum is taken in the loop's own order (k_sol outermost, B_overall
    innermost) so a tie resolves to the same node.

    With a mask, the amplitudes come from three real arrays formed once,

        |F_p + c F_m|^2 = |F_p|^2 + 2c Re(F_p F_m*) + c^2 |F_m|^2,

    rather than a complex (n_k, n) table and its modulus per b_sol. The same
    function, rounded differently, so the node can differ from the complex
    form's only where two nodes tie to rounding; on the 140 SW40 and TR1000
    entries of the 2026-11-15 validation (float64) it picked the same node
    every time and every printed row was unchanged. The returned log_k is
    rounded differently too (last bit), and LBFGS carries that into the fit:
    in float64 the fitted scales move by ~1e-8, far below what is printed;
    in float32 (--dtype float32) the optimizer amplifies it and printed R,
    k_sol and b_sol can differ from earlier runs. Where the solvent term
    cancels the protein the expanded sum can round below zero, hence the
    clamp. On the 140 entries the single-scale fit with solvent took a
    median 0.75 (SW40) and 0.82 (TR1000) of its earlier time."""
    with torch.no_grad():
        n_k, n_bs, n_b = len(k_grid), len(bs_grid), len(b_grid)
        E = torch.exp(-0.25 * b_grid[:, None] * s2[None, :])        # (n_b, n)
        E2 = E * E
        Fo_sq = (Fo * Fo).sum()
        num = torch.empty(n_k, n_bs, n_b, dtype=s2.dtype, device=s2.device)
        den = torch.empty_like(num)
        if Fm is None:
            # one (1, n) row, which the assignments broadcast over every
            # (k_sol, b_sol) node of the grid
            mag = torch.abs(Fp)[None, :]
            num[:] = ((mag * Fo[None, :]) @ E.T)[:, None, :]
            den[:] = ((mag * mag) @ E2.T)[:, None, :]
        else:
            P = Fp.real * Fp.real + Fp.imag * Fp.imag                # |F_p|^2
            Q = Fp.real * Fm.real + Fp.imag * Fm.imag                # Re(F_p F_m*)
            M = Fm.real * Fm.real + Fm.imag * Fm.imag                # |F_m|^2
            for j in range(n_bs):
                # (n_k, 1) k_sol against the (1, n) solvent weight: c is the
                # (n_k, n) table of k_sol exp(-b_sol s^2/4), as in inner_f_calc
                c = k_grid[:, None] * torch.exp(-0.25 * bs_grid[j] * s2)[None, :]
                # clamped: rounding can take a near-cancelling sum below 0
                mag2 = (P[None, :] + 2.0 * c * Q[None, :]
                        + c * c * M[None, :]).clamp(min=0.0)
                num[:, j, :] = (torch.sqrt(mag2) * Fo[None, :]) @ E.T  # (n_k, n_b)
                den[:, j, :] = mag2 @ E2.T
        k = torch.clamp(num / den, min=1e-12)
        # sum (Fo - k g)^2 at the clamped k, exactly as the loop evaluated it
        resid = Fo_sq - 2.0 * k * num + k * k * den
        idx = torch.argmin(resid.reshape(-1))                        # first minimum, loop order
        i = idx // (n_bs * n_b)
        j = idx // n_b % n_bs
        m = idx % n_b
        # One read-back for all four, the only synchronisation in the scan.
        return tuple(torch.stack([k_grid[i], bs_grid[j], b_grid[m],
                                  torch.log(k.reshape(-1)[idx])]).tolist())


def coarse_scan_reference(Fo, Fp, Fm, s2, k_grid, bs_grid, b_grid):
    """The scan as a Python loop over nodes, kept as the reference the
    vectorised coarse_scan is tested against. Not used by the fit."""
    best = None
    with torch.no_grad():
        for k_sol in k_grid:
            for b_sol in bs_grid:
                mag = torch.abs(inner_f_calc(Fp, Fm, s2, k_sol, b_sol))
                for b_ov in b_grid:
                    g = torch.exp(-0.25 * b_ov * s2) * mag
                    log_k = optimal_log_k(Fo, g)
                    resid = float(((Fo - torch.exp(log_k) * g) ** 2).sum())
                    if best is None or resid < best[0]:
                        best = (resid, float(k_sol), float(b_sol),
                                float(b_ov), float(log_k))
    return best[1:]


def fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso,
               fit_solvent=True, bins=None, solvent_bins=False, log=sys.stdout,
               init=None):
    """
    Least squares on amplitudes over the work set. Returns a parameter dict.

    Two stages, because the objective is not convex in (k_sol, b_sol) and a
    gradient method started at the conventional values can and does settle into
    a local minimum that depends on where it began:

      1. A coarse scan over (k_sol, b_sol), with the overall scale solved
         exactly at every node and the overall B scanned alongside
         (coarse_scan: batched, one synchronisation; the two structure
         factor sets are computed once, before the scan).
      2. LBFGS on every parameter at once from the best node.

      init: a single-scale fit (bins=None) that this function returned for
         the same inputs, fit_solvent and aniso. Stages 1 and 2 are skipped
         and the rest starts from its log_k, u_params, k_sol and b_sol, which
         is exactly where they would have finished, since both stages are
         deterministic on the same inputs. The R-factor tool (tools/
         fit_solvent_rfactor.py) passes the single-scale fits
         to the binned main fit, and the one with solvent to the binned row,
         this way; each would otherwise repeat the
         scan and the LBFGS on the same structure factors; with the scan's
         real-array form (coarse_scan) and the per-bin solver's arrays
         gathered once (_fit_solvent_bins), the binned row takes about a
         third of its earlier time (SW40 and TR1000, 2026-11-15).

      3. Only with bins (resolution_bins or log_resolution_bins): the
         single overall scale becomes one scale per bin, refined by LBFGS
         jointly with everything else from where stage 2 finished. This is
         mmtbx's binned k_isotropic, and it is what lets the model follow a
         falloff that no Gaussian B can -- 8VQ1, room-temperature XFEL data,
         goes from R-work 0.198 to 0.138 with 20 equal-count bins, against
         mmtbx's 0.137. k_sol and b_sol stay single numbers, so
         they remain the diagnostic they are without bins.

      4. Only with bins AND solvent_bins=True: the solvent scale becomes one
         value per bin as well, k_mask[bin] in place of k_sol exp(-b_sol s^2/4),
         which is what mmtbx fits (a free k_mask per bin). Least squares
         jointly with the bin scales, started from the exponential of stage
         3, B_cart held where stage 3 left it, and regularised by mmtbx's own
         rule (solvent_curve_rule): no negative values, a moving average at
         local extrema, and zero from the first zero below 3 A outward. The
         surviving bins are refit from the regularised curve and the bin
         scales are then solved in closed form. (In practice the second pass
         returns to the least-squares optimum of the bins it frees, so its
         effect is to refit after the cut has zeroed bins that carried
         weight.) The curve is a step per bin; mmtbx additionally chooses,
         per bin, between the step and a linear interpolation to the next
         bin, which this does not do. Measured on SW40 (stage 0,
         2026-11-09) this closes the lunus-vs-mmtbx R-free gap from a median
         +0.0020 to +0.0011, which is where mmtbx's complete scaling on the
         same structure factors lands; the exponential's shape -- too little
         solvent at 6-4 A, too much at 3-2 A -- was the whole remaining
         scaling difference. The k_sol/b_sol returned by such a fit are an
         exponential fitted to the curve at d >= 4 A, as mmtbx reports them,
         not parameters the model used.

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
    dtype, device = inv_d2.dtype, inv_d2.device

    if init is not None:
        # Stages 1 and 2 were done by the fit handed in (see the docstring).
        # Its values are copied, not shared, so this fit cannot move them.
        if (init["bins"] is not None or bool(init["aniso"]) != bool(aniso)
                or bool(init["fit_solvent"]) != bool(fit_solvent)):
            raise ValueError("fit_scales: init must be a single-scale fit "
                             "(bins=None) with the same aniso and fit_solvent")
        log_k = init["log_k"].detach().clone().requires_grad_(True)
        k_sol = init["k_sol"].detach().clone().requires_grad_(fit_solvent)
        b_sol = init["b_sol"].detach().clone().requires_grad_(fit_solvent)
        u_params = [p.detach().clone().requires_grad_(True)
                    for p in init["u_params"]]
        if fit_solvent:
            print("  started from the single-scale fit: k_sol %.3f, b_sol %.1f"
                  % (float(init["k_sol"]), float(init["b_sol"])), file=log)
        else:
            print("  started from the single-scale fit", file=log)
    else:
        k_sol0, b_sol0, b_ov0, log_k0 = coarse_scan(Fo, Fp, Fm, s2,
                                                    *scan_grids(fit_solvent, inv_d2))
        print("  coarse scan: k_sol %.3f, b_sol %.1f, B_overall %.1f"
              % (k_sol0, b_sol0, b_ov0), file=log)

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

        _refine_single(Fo, Fp, Fm, s2, s, aniso, fit_solvent, log_k, u_params,
                       k_sol, b_sol)

    log_k_bins = None
    if bins is not None:
        # The overall scale is absorbed into the bins, and the isotropic part
        # of B is then degenerate with them: left free, the pair drifts
        # together (B_overall -34 against bin scales falling to 0.01 on 8VQ1)
        # and the printed B means nothing. So the isotropic part is held at
        # the single-scale answer and only the anisotropy refines with the
        # bins; an isotropic fit has no B left to refine at all.
        b = bins[work]
        n_bins = int(bins.max()) + 1
        log_k_bins = torch.full((n_bins,), float(log_k.detach()), dtype=dtype,
                                device=device, requires_grad=True)
        n_diag = 3 if aniso else 1
        b_iso0 = sum(float(p.detach()) for p in u_params[:n_diag]) / n_diag

        def held(u):
            shift = sum(u[:n_diag]) / n_diag - b_iso0
            return [p - shift for p in u[:n_diag]] + list(u[n_diag:])

        params = [log_k_bins] + (u_params if aniso else []) + (
            [k_sol, b_sol] if fit_solvent else [])
        opt = torch.optim.LBFGS(params, max_iter=300,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            F_c = inner_f_calc(Fp, Fm, s2, k_sol, b_sol)
            model = scaled_amplitudes(F_c, s2, s, log_k_bins[b],
                                       held(u_params), aniso)
            loss = ((Fo - model) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            u_params = [p.clone() for p in held(u_params)]
        k = torch.exp(log_k_bins.detach())
        print("  %d resolution bins: scale %.3f-%.3f (low to high resolution: "
              "%s)" % (n_bins, float(k.min()), float(k.max()),
                       " ".join("%.2f" % float(v) for v in k)), file=log)
        log_k_bins = log_k_bins.detach()

    k_mask_bins, n_fit = None, 0
    k_sol_out, b_sol_out = k_sol.detach(), b_sol.detach()
    if bins is not None and fit_solvent and solvent_bins:
        # Bin edges and centres from every reflection in the bin, not the
        # work subset: a bin is never empty that way, and it is what
        # mmtbx's ss_bin_values are.
        d_min_bins = [float(1.0 / torch.sqrt(inv_d2[bins == i].max()))
                      for i in range(n_bins)]
        s2_bins = [float(inv_d2[bins == i].mean()) for i in range(n_bins)]
        k_mask_bins, log_k_bins = _fit_solvent_bins(
            Fo, Fp, Fm, s2, s, b, n_bins, log_k_bins, u_params, aniso,
            k_sol_out, b_sol_out, d_min_bins, s2_bins)
        ks, bs, n_fit = exponential_from_curve(k_mask_bins, bins, inv_d2)
        k_sol_out = torch.tensor(ks, dtype=dtype, device=device)
        b_sol_out = torch.tensor(bs, dtype=dtype, device=device)
        print("  solvent scale per bin (low to high resolution): %s"
              % " ".join("%.2f" % float(v) for v in k_mask_bins), file=log)
        if n_fit:
            print("  exponential closest to it over the %d reflections at d >= 4 A: "
                  "k_sol %.3f, b_sol %.1f (reported as k_sol/b_sol; not used by "
                  "the model)" % (n_fit, ks, bs), file=log)
        else:
            print("  fewer than 2 reflections at d >= 4 A to fit an exponential to; "
                  "k_sol is reported as the curve's maximum and b_sol as 0", file=log)

    return {
        "log_k": log_k.detach(),
        "log_k_bins": log_k_bins,
        "bins": bins,
        "u_params": [p.detach() for p in u_params],
        "aniso": aniso,
        "k_sol": k_sol_out,
        "b_sol": b_sol_out,
        "fit_solvent": fit_solvent,
        "k_mask_bins": k_mask_bins,
        "n_refl_exp_fit": n_fit if k_mask_bins is not None else None,
    }


# THE PER-BIN SOLVENT SCALE IS BOUNDED AT 1.0. mmtbx's initial per-bin search
# runs on a grid from 0 to 1.0 (bulk_solvent/scaler.py, k_mask_grid_search);
# its later cycles are not strictly bounded but only accept a curve that lowers
# R. A free least-squares bin can run away: on 4Y0B (data to 1.83 A, in the
# torchref1000 run of 2026-11-10, once the soft-sphere mask had given F_mask
# real high-resolution content) the high-resolution bins found a degenerate
# solution in which the solvent stood in for the protein, solvent scales 2,
# 5.6, 16, 49, 147 with the bin scales going to zero, binned R-free 0.507
# against mmtbx's 0.324. Bounded, the last bin sits AT the bound (curve
# ... 0.21 0.24 0.35 1.00) and R-free is 0.331: the bound binds, it does not
# make the curve fall to zero. The solve is exact per bin (_fit_solvent_bins).
# On the 20 + 20 + 39 entries of the low-, mid-resolution and SW40 sets no
# other entry moves by more than 0.0009 and no median moves.
SOLVENT_BIN_MAX = 1.0
# Measured and rejected, kept for a later look: zeroing every bin whose
# high-resolution edge is below this many A (with a 0.6 bound) fixed 4Y0B
# equally but made 9 of those 79 entries worse by more than 0.001 and none
# better; it discards real solvent signal beyond 3 A. mmtbx itself cuts only
# from the first zero bin below 3 A, which solvent_curve_rule does. None = off.
SOLVENT_BIN_FORCE_CUT = None


def _fit_solvent_bins(Fo, Fp, Fm, s2, s, b, n_bins, log_k_bins, u_params, aniso,
                      k_sol, b_sol, d_min_bins, s2_bins):
    """Stage 4 of fit_scales, on the work set: one solvent scale per bin.
    d_min_bins and s2_bins are each bin's high-resolution edge (A) and mean
    s^2, over every reflection in it. Returns (k_mask_bins, log_k_bins),
    both detached. A bin with no work reflections (possible with many
    equal-count bins on little data) keeps its stage-3 scale and starts the
    solvent at the exponential's value at the bin centre."""
    dtype, device = s2.dtype, s2.device
    u = [p.detach() for p in u_params]
    in_bin = [b == i for i in range(n_bins)]
    with torch.no_grad():
        km0 = k_sol * torch.exp(-0.25 * b_sol * s2)
        km = torch.stack([
            km0[sel].mean() if bool(sel.any())
            else k_sol * torch.exp(-0.25 * b_sol * s2_bins[i])
            for i, sel in enumerate(in_bin)])

    def solve_lbfgs(active, km_start, lk_start):
        # the unbounded joint solve (SOLVENT_BIN_MAX = None), as before
        km_p = km_start.clone().requires_grad_(True)
        lk_p = lk_start.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([km_p, lk_p], max_iter=300,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            km_eff = torch.where(active, km_p, torch.zeros_like(km_p))
            F_c = Fp + km_eff[b].to(Fp.dtype) * Fm
            model = scaled_amplitudes(F_c, s2, s, lk_p[b], u, aniso)
            loss = ((Fo - model) ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return km_p.detach(), lk_p.detach()

    # THE BOUNDED SOLVE IS EXACT, PER BIN. With the anisotropy u held fixed
    # the model in bin i is exp(lk_i) * A * |Fp + k_i Fm| with A per
    # reflection, so the bins are independent and lk_i has a closed form:
    # what is left is a 1-D minimisation of the residual over k_i in
    # [0, SOLVENT_BIN_MAX], done as mmtbx's initial search does it (a 0.01
    # grid) followed by golden-section refinement in the bracketing cell.
    # Not a clamp inside an optimizer: that was tried first and a bin that
    # overshot below 0 got no gradient back, came out as 0, and fired the
    # curve rule's cut on 4Y0B -- the right answer for the wrong reason.
    with torch.no_grad():
        A = scaled_amplitudes(torch.ones_like(Fp), s2, s,
                               torch.zeros((), dtype=dtype, device=device), u, aniso)

        # Each bin's reflections gathered once: the golden-section search
        # evaluates a bin ~40 times per pass, and gathering by boolean mask on
        # every evaluation was a large part of the solver's time. The same
        # arithmetic on the same values, so the result is unchanged.
        parts = [(A[sel], Fp[sel], Fm[sel], Fo[sel]) for sel in in_bin]

    def bin_residual(k, i):
        # residual sum of squares at the closed-form scale in bin i, for a
        # tensor of k
        Ai, Fpi, Fmi, Foi = parts[i]
        g = Ai[None, :] * (Fpi[None, :] + k[:, None].to(Fpi.dtype) * Fmi[None, :]).abs()
        fo = Foi[None, :]
        num = (fo * g).sum(1)
        den = (g * g).sum(1).clamp(min=1e-300)
        return (fo * fo).sum(1) - num * num / den

    def solve_bounded(active, km_start, lk_start):
        km_out = torch.zeros_like(km_start)
        k_max = float(SOLVENT_BIN_MAX)
        grid = torch.arange(0.0, k_max + 1e-9, 0.01, dtype=dtype, device=device)
        with torch.no_grad():
            for i in range(n_bins):
                if not bool(active[i]):
                    continue
                if len(parts[i][0]) == 0:
                    # no work reflections: keep the starting value (the
                    # exponential's at the bin centre), as the unbounded
                    # solve did; a 0 here would fire the curve rule's cut
                    km_out[i] = km_start[i].clamp(0.0, k_max)
                    continue
                r = bin_residual(grid, i)
                j0 = int(torch.argmin(r))
                lo, hi = float(grid[max(j0 - 1, 0)]), float(grid[min(j0 + 1, len(grid) - 1)])
                gr = 0.5 * (5 ** 0.5 - 1)
                a, c = lo, hi
                x1, x2 = c - gr * (c - a), a + gr * (c - a)
                f1, f2 = (float(v) for v in bin_residual(
                    torch.tensor([x1, x2], dtype=dtype, device=device), i))
                for _ in range(40):
                    if f1 <= f2:
                        c, x2, f2 = x2, x1, f1
                        x1 = c - gr * (c - a)
                        f1 = float(bin_residual(torch.tensor([x1], dtype=dtype, device=device), i)[0])
                    else:
                        a, x1, f1 = x1, x2, f2
                        x2 = a + gr * (c - a)
                        f2 = float(bin_residual(torch.tensor([x2], dtype=dtype, device=device), i)[0])
                best = min((float(r[j0]), float(grid[j0])), (f1, x1), (f2, x2))
                km_out[i] = best[1]
        return km_out, lk_start

    def solve(active, km_start, lk_start):
        if SOLVENT_BIN_MAX is None:
            return solve_lbfgs(active, km_start, lk_start)
        return solve_bounded(active, km_start, lk_start)

    active = torch.ones(n_bins, dtype=torch.bool, device=device)
    if SOLVENT_BIN_FORCE_CUT is not None:
        active = torch.tensor([float(d) >= SOLVENT_BIN_FORCE_CUT for d in d_min_bins],
                              dtype=torch.bool, device=device)
    km, lk = solve(active, km, log_k_bins)
    km = solvent_curve_rule(km, d_min_bins)
    # Second pass: every bin before the cut stays free, including one the
    # first pass clamped to zero above 3 A, so that it can come back positive
    # once its neighbours are regularised (mmtbx's grid allows the same);
    # bins beyond the cut are held at zero.
    cut = solvent_cut_index(km, d_min_bins)
    active = (torch.arange(n_bins, device=device) < cut) & active
    if cut > 0:
        km, lk = solve(active, km, lk)
        km = solvent_curve_rule(torch.where(active, km, torch.zeros_like(km)), d_min_bins)
    # Bin scales in closed form with the curve fixed: exactly linear per bin.
    with torch.no_grad():
        F_c = Fp + km[b].to(Fp.dtype) * Fm
        zero = torch.zeros((), dtype=dtype, device=device)
        g = scaled_amplitudes(F_c, s2, s, zero, u, aniso)
        lk = torch.stack([optimal_log_k(Fo[sel], g[sel]) if bool(sel.any())
                          else lk[i] for i, sel in enumerate(in_bin)])
    return km, lk


def model_f_c(fit, F_protein, F_mask, inv_d2):
    """F_c = F_protein + k_mask F_mask with the fit's solvent, before the
    overall scale; F_protein alone for a fit without solvent."""
    if fit["fit_solvent"] and F_mask is not None:
        return F_protein + solvent_weight(fit, inv_d2).to(F_protein.dtype) * F_mask
    return F_protein


def model_log_k(fit):
    """The fit's log overall scale, per reflection for a binned fit."""
    if fit.get("log_k_bins") is not None:
        return fit["log_k_bins"][fit["bins"]]
    return fit["log_k"]


def model_amplitudes(fit, F_protein, F_mask, inv_d2, s_cart):
    with torch.no_grad():
        return scaled_amplitudes(model_f_c(fit, F_protein, F_mask, inv_d2), inv_d2,
                                  s_cart, model_log_k(fit), fit["u_params"], fit["aniso"])


def model_scale(fit, inv_d2, s_cart):
    """The overall scale model_amplitudes puts on F_c, per reflection: real
    and positive, so F_model keeps F_c's phase."""
    return scaled_amplitudes(torch.ones_like(inv_d2), inv_d2, s_cart, model_log_k(fit),
                              fit["u_params"], fit["aniso"])


def model_structure_factors(fit, F_protein, F_mask, inv_d2, s_cart):
    """Complex F_model = k_total (F_protein + k_mask F_mask) on the Fobs
    scale: model_amplitudes is its modulus, and its phase is F_c's."""
    with torch.no_grad():
        F_c = model_f_c(fit, F_protein, F_mask, inv_d2)
        return model_scale(fit, inv_d2, s_cart).to(F_c.dtype) * F_c


def _refine_single(Fo, Fp, Fm, s2, s, aniso, fit_solvent, log_k, u_params,
                   k_sol, b_sol):
    """Stage 2 of fit_scales: LBFGS on every single-scale parameter at once,
    in place, from wherever the leaf tensors given now are (the scan's best
    node, or BraggScaler's current values for a warm fit)."""
    params = [log_k] + u_params + ([k_sol, b_sol] if fit_solvent else [])
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        F_c = inner_f_calc(Fp, Fm, s2, k_sol, b_sol)
        model = scaled_amplitudes(F_c, s2, s, log_k, u_params, aniso)
        loss = ((Fo - model) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)


def refine_single_scale(start, F_obs, F_protein, F_mask, inv_d2, s_cart, work):
    """A single-scale fit (bins=None), as fit_scales returns one, refined by
    stage 2 alone from the values of `start`, a single-scale fit made for
    other inputs (an earlier step's structure factors, a displaced model).
    The warm start: no coarse scan, so the fit stays in start's basin of
    (k_sol, b_sol)."""
    if start["bins"] is not None:
        raise ValueError("refine_single_scale: start must be a single-scale fit")
    aniso, fit_solvent = start["aniso"], start["fit_solvent"]
    Fo = F_obs[work]
    Fp = F_protein[work]
    Fm = None if F_mask is None else F_mask[work]
    log_k = start["log_k"].detach().clone().requires_grad_(True)
    k_sol = start["k_sol"].detach().clone().requires_grad_(fit_solvent)
    b_sol = start["b_sol"].detach().clone().requires_grad_(fit_solvent)
    u_params = [p.detach().clone().requires_grad_(True) for p in start["u_params"]]
    _refine_single(Fo, Fp, Fm, inv_d2[work], s_cart[work], aniso, fit_solvent,
                   log_k, u_params, k_sol, b_sol)
    return {
        "log_k": log_k.detach(), "log_k_bins": None, "bins": None,
        "u_params": [p.detach() for p in u_params], "aniso": aniso,
        "k_sol": k_sol.detach(), "b_sol": b_sol.detach(),
        "fit_solvent": fit_solvent, "k_mask_bins": None, "n_refl_exp_fit": None,
    }


def shell_rows(d, F_obs, model_no_solvent, model_solvent, work, free, n_shells,
               solvent_share):
    """R by resolution shell, lowest resolution first, as rows
    (d_max, d_min, n, R-work no solvent, R-work with, R-free no solvent,
    R-free with, mean solvent share), d in A: the numbers the R-factor tool's shell table prints.
    Equal-count shells by percentile of d; a shell with fewer than 2 work (or
    free) reflections has nan for those R. d, work, free and solvent_share
    are numpy arrays or tensors on the reflections of F_obs."""
    edges = np.percentile(d, np.linspace(0, 100, n_shells + 1))
    rows = []
    for lo, hi in zip(edges[:-1][::-1], edges[1:][::-1]):
        sel = (d >= lo) & (d <= hi)
        rs = []
        for mask in (sel & work, sel & free):
            if mask.sum() < 2:
                rs.append((float("nan"), float("nan")))
                continue
            rs.append((r_factor(F_obs[mask], model_no_solvent[mask]),
                       r_factor(F_obs[mask], model_solvent[mask])))
        (rw0, rw1), (rf0, rf1) = rs
        rows.append((float(hi), float(lo), int(sel.sum()), rw0, rw1, rf0, rf1,
                     float(solvent_share[sel].mean())))
    return rows


# ------------------------------------------------------------- BraggScaler

def _copy_fit(fit, device=None):
    """A fit_scales dict with every tensor (and list of tensors) detached and
    cloned, onto device if given; None for None."""
    if fit is None:
        return None
    out = {}
    for k, v in fit.items():
        if torch.is_tensor(v):
            v = v.detach().clone().to(device)
        elif isinstance(v, list):
            v = [p.detach().clone().to(device) for p in v]
        out[k] = v
    return out


class BraggScaler:
    """
    Scale state for one reflection list. Build once; call per evaluation.

        scaler = BraggScaler(hkl, orth_matrix, F_obs, work)
        scaler.fit(F_protein, F_mask)                 # once: scan + LBFGS
        ...
        scaler.refit(F_protein, F_mask)               # per step: bin scales, closed form
        F_model = scaler.apply(F_protein, F_mask)     # differentiable in F_protein, F_mask
        loss = ((F_obs - F_model)[work] ** 2).sum()
        r_work, r_free = scaler.r_factors(F_model)

    hkl (n,3) integer indices and orth_matrix (3,3, Cartesian = orth @ frac)
    give s = Mstar h and s^2 = 1/d^2. F_obs (n,) amplitudes; work (n,) bool,
    True for the reflections the scales are fitted on -- the rest are the free
    set, which fit and refit never read.

    bins: "log" (default: mmtbx's log bins, log_resolution_bins), N for N
    equal-count bins, or 0 for one overall scale. solvent_bins: "auto"
    (default: a per-bin solvent scale with log bins, the exponential
    otherwise -- the R-factor tool's --solvent-bins auto), True or False. d:
    the d spacings (A) to bin by; pass cctbx's own (F_obs_array.d_spacings())
    for mmtbx's log bins exactly, since a reflection on a bin limit can move
    when d is recomputed. Default 1/sqrt(s^2).

    fit() is what the R-factor tool does: the single-scale fit (coarse scan,
    LBFGS) and, with bins, the binned fit started from it. refit() moves only
    some parameters (see refit). apply() and apply_complex() use the fitted
    parameters as plain tensors, set under no_grad, so the gradient of
    anything computed from them flows through F_protein and F_mask only --
    the detached-refit policy of docs/scaling-design.md, which is the exact
    gradient of the profiled loss when the refit parameters are at a
    stationary point (refit("bins") always is).

    Inputs are moved to `device` and real tensors to `dtype`; F_protein and
    F_mask passed later must be on the same device (complex, any precision).
    """

    def __init__(self, hkl, orth_matrix, F_obs, work, *, bins="log",
                 solvent_bins="auto", aniso=False, fit_solvent=True, d=None,
                 device="cpu", dtype=torch.float64):
        # numbers.Integral so a numpy integer passes; bool excluded, since
        # True would quietly mean one bin rather than an error.
        if not (bins == "log" or (isinstance(bins, numbers.Integral)
                                  and not isinstance(bins, bool) and bins >= 0)):
            raise ValueError("bins must be 'log' or a whole number >= 0, got %r"
                             % (bins,))
        if solvent_bins not in ("auto", True, False):
            raise ValueError("solvent_bins must be 'auto', True or False")
        self.device = torch.device(device)
        self.dtype = dtype
        hkl = torch.as_tensor(hkl, device=self.device)
        orth = torch.as_tensor(orth_matrix, dtype=dtype, device=self.device)
        # s = Mstar h with Mstar = inv(orth).T, so s_cart = h @ inv(orth): the
        # R-factor tool's expression, so that the two compute the same s.
        self.s_cart = hkl.to(dtype) @ torch.linalg.inv(orth)
        self.inv_d2 = (self.s_cart * self.s_cart).sum(-1)
        self.F_obs = torch.as_tensor(F_obs, dtype=dtype, device=self.device)
        self.work = torch.as_tensor(work, dtype=torch.bool, device=self.device)
        if self.work.shape != self.F_obs.shape or self.F_obs.shape != self.inv_d2.shape:
            raise ValueError("hkl, F_obs and work must cover the same reflections")
        if int(self.work.sum()) == 0:
            raise ValueError("no work reflections to fit on")
        self.free = ~self.work
        # The bins from float64 on the CPU whatever dtype and device the
        # scaler runs in: reflections at equal or nearly equal s^2 are common,
        # and float32 rounding (or a CUDA sort) orders them differently, which
        # moves reflections between bins -- and a state_dict from a float64
        # scaler would then not load into a float32 one for the same list.
        s_64 = hkl.cpu().to(torch.float64) @ torch.linalg.inv(
            torch.as_tensor(orth_matrix).detach().cpu().to(torch.float64))
        inv_d2_64 = (s_64 * s_64).sum(-1)
        if d is None:
            d = 1.0 / torch.sqrt(inv_d2_64)
        if torch.is_tensor(d):
            d = d.detach().cpu().numpy()
        self.bin_spec = bins
        self.bins = assign_scale_bins(bins, inv_d2_64, np.asarray(d))
        self.d_min_bins = self.s2_bins = None
        if self.bins is not None:
            self.bins = self.bins.to(self.device)
            # Each bin's high-resolution edge and mean s^2 over every
            # reflection in it, as fit_scales computes them: fixed for the
            # reflection list, so refit("solvent") need not recompute them.
            n_bins = int(self.bins.max()) + 1
            self.d_min_bins = [float(1.0 / torch.sqrt(self.inv_d2[self.bins == i].max()))
                               for i in range(n_bins)]
            self.s2_bins = [float(self.inv_d2[self.bins == i].mean())
                            for i in range(n_bins)]
        self.solvent_bins = (bins == "log") if solvent_bins == "auto" else bool(solvent_bins)
        self.aniso = bool(aniso)
        self.fit_solvent = bool(fit_solvent)
        self.single = None      # the single-scale fit (fit_scales dict)
        self.state = None       # the model in force (fit_scales dict)
        self.n_eval = 0         # calls to apply / apply_complex
        self.last_refit = {"fit": None, "fit_warm": None, "bins": None,
                           "solvent": None, "all": None, "what": None}

    # -- inputs ---------------------------------------------------------

    def _inputs(self, F_protein, F_mask):
        if self.fit_solvent and F_mask is None:
            raise ValueError("this scaler fits a solvent scale (fit_solvent=True); "
                             "pass F_mask")
        return F_protein, (F_mask if self.fit_solvent else None)

    def _need_fit(self):
        if self.state is None:
            raise RuntimeError("BraggScaler: call fit() first")

    # -- fitting --------------------------------------------------------

    def fit(self, F_protein, F_mask=None, *, warm=False, log=None):
        """Fit every scale parameter on the work set, under no_grad (the
        LBFGS inside builds its own graph on detached inputs). Returns the
        model's fit_scales dict and sets the state.

        warm=False: the R-factor tool's fit -- coarse scan and LBFGS for the
        single scale, then with bins the bin fit from it (fit_scales, init).
        warm=True: no scan; the single-scale LBFGS starts from the current
        single-scale values (refine_single_scale), so the fit stays in the
        basin the previous fit found. With bins those are the last fit()'s
        single-scale values: refit() moves the binned model, not them.
        log: a stream for fit_scales' progress lines; None is silent."""
        F_protein, F_mask = self._inputs(F_protein, F_mask)
        F_protein, F_mask = F_protein.detach(), None if F_mask is None else F_mask.detach()
        out = log if log is not None else io.StringIO()
        args = (self.F_obs, F_protein, F_mask, self.inv_d2, self.s_cart, self.work)
        if warm:
            if self.single is None:
                raise RuntimeError("BraggScaler.fit(warm=True) needs an earlier fit()")
            single = refine_single_scale(self.single, *args)
        else:
            single = fit_scales(*args, self.aniso, fit_solvent=self.fit_solvent,
                                log=out)
        if self.bins is None:
            state = single
        else:
            state = fit_scales(*args, self.aniso, fit_solvent=self.fit_solvent,
                               bins=self.bins, solvent_bins=self.solvent_bins,
                               log=out, init=single)
        self.single, self.state = single, state
        self.last_refit["fit"] = self.n_eval
        self.last_refit["fit_warm"] = bool(warm)
        self.last_refit["what"] = "fit"
        return state

    def refit(self, F_protein, F_mask=None, *, what="bins", refit_aniso=False):
        """Move some of the parameters, detached, from where they are.

        "bins": the overall scale per bin (or the single scale), in closed
          form, everything else held. Exactly linear, so this is the
          parameters' least-squares optimum, a stationary point, and the
          gradient through apply() afterwards is the exact gradient of the
          profiled loss. The one meant for every step.
        "solvent": the solvent scale and the bin scales. Per-bin solvent: the
          bounded per-bin solve, mmtbx's curve rule, then the bin scales in
          closed form (as the end of fit_scales; a bin with no work
          reflections starts from the reported exponential rather than the
          one fit_scales seeds it with, so only a list with such a bin can
          come out different from fit()'s -- that bin, and through the curve
          rule's moving average and cut its neighbours and the reported
          exponential). Exponential: LBFGS on
          k_sol, b_sol and the scales from the current values. refit_aniso
          adds the anisotropy of B_cart (its isotropic part held when binned,
          as fit_scales holds it).
          THE GRADIENT AFTER A PER-BIN SOLVENT REFIT IS NOT THE PROFILED ONE.
          The bound and mmtbx's curve rule (cut, moving average) leave k_mask
          off the per-bin least-squares optimum, so autograd through apply()
          with k_mask held differs from the derivative of the loss re-minimised
          over k_mask (by 16-18% along a random F_protein direction on
          synthetic data like test_scaling_torch's, log bins, anisotropic,
          measured 2026-11-17). It is the exact gradient of the loss with
          k_mask held, which is what the policy of docs/scaling-design.md
          runs between refits; the exponential refit, an LBFGS to a stationary
          point, does give the profiled gradient.
        "all": fit(warm=True); last_refit then records it under "fit" (with
          fit_warm True) as well as under "all"."""
        self._need_fit()
        if what == "all":
            self.fit(F_protein, F_mask, warm=True)
            self.last_refit["all"] = self.n_eval
            self.last_refit["what"] = "all"
            return
        if what not in ("bins", "solvent"):
            raise ValueError("what must be 'bins', 'solvent' or 'all'")
        F_protein, F_mask = self._inputs(F_protein, F_mask)
        w = self.work
        Fo, Fp = self.F_obs[w], F_protein.detach()[w]
        Fm = None if F_mask is None else F_mask.detach()[w]
        s2, s = self.inv_d2[w], self.s_cart[w]
        b = None if self.bins is None else self.bins[w]
        st = dict(self.state)
        if what == "bins":
            st = self._refit_bins(st, Fo, Fp, Fm, s2, s, b)
        else:
            st = self._refit_solvent(st, Fo, Fp, Fm, s2, s, b, refit_aniso)
        self.state = st
        if self.bins is None:
            self.single = st
        self.last_refit[what] = self.n_eval
        self.last_refit["what"] = what

    def _f_c_work(self, st, Fp, Fm, s2):
        """F_c on the work set under the state's solvent model."""
        if not st["fit_solvent"] or Fm is None:
            return Fp
        if st["k_mask_bins"] is not None:
            return Fp + st["k_mask_bins"][self.bins[self.work]].to(Fp.dtype) * Fm
        return inner_f_calc(Fp, Fm, s2, st["k_sol"], st["b_sol"])

    def _refit_bins(self, st, Fo, Fp, Fm, s2, s, b):
        with torch.no_grad():
            zero = torch.zeros((), dtype=s2.dtype, device=s2.device)
            g = scaled_amplitudes(self._f_c_work(st, Fp, Fm, s2), s2, s, zero,
                                  st["u_params"], st["aniso"])
            if b is None:
                st["log_k"] = optimal_log_k(Fo, g)
            else:
                # optimal_log_k per bin, all bins in one pass: no Python loop
                # and no host sync per bin on the path meant for every step.
                # A bin with no work reflections keeps its scale.
                old = st["log_k_bins"]
                num = torch.zeros_like(old).index_add_(0, b, (Fo * g).to(old.dtype))
                den = torch.zeros_like(old).index_add_(0, b, (g * g).to(old.dtype))
                n_in = torch.bincount(b, minlength=len(old))
                st["log_k_bins"] = torch.where(
                    n_in > 0, torch.log(torch.clamp(num / den, min=1e-12)), old)
        return st

    def _refit_solvent(self, st, Fo, Fp, Fm, s2, s, b, refit_aniso):
        if not st["fit_solvent"]:
            raise ValueError("refit('solvent') on a scaler without solvent")
        if st["k_mask_bins"] is not None:
            if refit_aniso and st["aniso"]:
                st = self._lbfgs(st, Fo, Fp, Fm, s2, s, b, move_u=True,
                                 move_solvent=False)
            n_bins = len(st["log_k_bins"])
            km, lk = _fit_solvent_bins(Fo, Fp, Fm, s2, s, b, n_bins,
                                       st["log_k_bins"], st["u_params"],
                                       st["aniso"], st["k_sol"], st["b_sol"],
                                       self.d_min_bins, self.s2_bins)
            ks, bs, n_fit = exponential_from_curve(km, self.bins, self.inv_d2)
            st["k_mask_bins"], st["log_k_bins"] = km, lk
            st["k_sol"] = torch.tensor(ks, dtype=self.dtype, device=self.device)
            st["b_sol"] = torch.tensor(bs, dtype=self.dtype, device=self.device)
            st["n_refl_exp_fit"] = n_fit
            return st
        return self._lbfgs(st, Fo, Fp, Fm, s2, s, b, move_u=refit_aniso and st["aniso"],
                           move_solvent=True)

    def _lbfgs(self, st, Fo, Fp, Fm, s2, s, b, move_u, move_solvent):
        """LBFGS on the scales (per bin or single), plus the anisotropy of
        B_cart if move_u and k_sol / b_sol if move_solvent, from the state's
        values; a per-bin solvent curve is held. Binned, the isotropic part of
        B_cart is held as fit_scales holds it; single-scale, move_u moves all
        of B_cart."""
        aniso = st["aniso"]
        binned = b is not None
        log_k = (st["log_k_bins"] if binned else st["log_k"]).detach().clone().requires_grad_(True)
        u = [p.detach().clone().requires_grad_(move_u) for p in st["u_params"]]
        k_sol = st["k_sol"].detach().clone().requires_grad_(move_solvent)
        b_sol = st["b_sol"].detach().clone().requires_grad_(move_solvent)
        km = st["k_mask_bins"]
        n_diag = 3 if aniso else 1
        b_iso0 = sum(float(p.detach()) for p in u[:n_diag]) / n_diag

        def held(u):
            if not (binned and move_u):
                return u
            shift = sum(u[:n_diag]) / n_diag - b_iso0
            return [p - shift for p in u[:n_diag]] + list(u[n_diag:])

        params = [log_k] + (u if move_u else []) + ([k_sol, b_sol] if move_solvent else [])
        opt = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            if km is not None:
                F_c = Fp + km[b].to(Fp.dtype) * Fm
            else:
                F_c = inner_f_calc(Fp, Fm, s2, k_sol, b_sol)
            model = scaled_amplitudes(F_c, s2, s, log_k[b] if binned else log_k,
                                      held(u), aniso)
            loss = ((Fo - model) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            st["u_params"] = [p.detach().clone() for p in held(u)]
        if binned:
            st["log_k_bins"] = log_k.detach()
        else:
            st["log_k"] = log_k.detach()
        if km is None:
            st["k_sol"], st["b_sol"] = k_sol.detach(), b_sol.detach()
        return st

    # -- the model ------------------------------------------------------

    def apply(self, F_protein, F_mask=None):
        """|F_model| on every reflection, differentiable in F_protein and
        F_mask (the parameters carry no gradient)."""
        return self.apply_complex(F_protein, F_mask).abs()

    def apply_complex(self, F_protein, F_mask=None):
        """F_model = k_total (F_protein + k_mask F_mask) on the F_obs scale,
        with F_c's phase: model_structure_factors without its no_grad, the
        synthetic data an MTZ is written from. Differentiable in F_protein
        and F_mask."""
        self._need_fit()
        F_protein, F_mask = self._inputs(F_protein, F_mask)
        self.n_eval += 1
        st = self.state
        F_c = model_f_c(st, F_protein, F_mask, self.inv_d2)
        return model_scale(st, self.inv_d2, self.s_cart).to(F_c.dtype) * F_c

    def r_factors(self, F_model):
        """(R-work, R-free) of |F_model| against F_obs; R-free is nan with no
        free reflections."""
        Fm = F_model.detach().abs() if torch.is_complex(F_model) else F_model.detach()
        r_work = r_factor(self.F_obs[self.work], Fm[self.work])
        r_free = (r_factor(self.F_obs[self.free], Fm[self.free])
                  if bool(self.free.any()) else float("nan"))
        return r_work, r_free

    # -- state ----------------------------------------------------------

    def state_dict(self):
        """The fitted parameters and the refit record, as plain values and
        tensors (copies): bins, the model in force ('state', a fit_scales
        dict: log_k / log_k_bins, u_params = B_cart, k_sol, b_sol,
        k_mask_bins), the single-scale fit, and last_refit -- the evaluation
        count (calls to apply/apply_complex) at the last fit and at the last
        refit of each kind, and the last kind."""
        return {"bin_spec": self.bin_spec,
                "bins": None if self.bins is None else self.bins.clone(),
                "solvent_bins": self.solvent_bins, "aniso": self.aniso,
                "fit_solvent": self.fit_solvent,
                "state": _copy_fit(self.state), "single": _copy_fit(self.single),
                "n_eval": self.n_eval, "last_refit": dict(self.last_refit)}

    def load_state_dict(self, state):
        """Restore what state_dict() returned, onto this scaler's device and
        dtype. The reflection list and the model settings (bins,
        solvent_bins, aniso, fit_solvent) must be the ones this scaler was
        built with; a state for other settings raises."""
        bins = state["bins"]
        if (bins is None) != (self.bins is None) or (
                bins is not None and not torch.equal(bins.to(self.device), self.bins)):
            raise ValueError("load_state_dict: the bins differ; the state is "
                             "for another reflection list or bin setting")
        # solvent_bins means nothing with one scale, so it is compared only
        # when there are bins
        names = ("aniso", "fit_solvent") + (("solvent_bins",) if self.bins is not None else ())
        for name in names:
            if bool(state[name]) != bool(getattr(self, name)):
                raise ValueError("load_state_dict: the state has %s=%r, this "
                                 "scaler %r" % (name, state[name], getattr(self, name)))
        self.state = self._moved(state["state"])
        # one scale: the single fit is the model, one dict, as fit() leaves it
        self.single = (self.state if self.bins is None
                       else self._moved(state["single"]))
        self.n_eval = state["n_eval"]
        self.last_refit = dict(state["last_refit"])

    def _moved(self, fit):
        """A copy of fit on this scaler's device, its real parameters in
        this scaler's dtype, and its bins this scaler's own."""
        out = _copy_fit(fit, self.device)
        if out is None:
            return None
        for k, v in out.items():
            if torch.is_tensor(v) and v.is_floating_point():
                out[k] = v.to(self.dtype)
            elif isinstance(v, list):
                out[k] = [p.to(self.dtype) if p.is_floating_point() else p for p in v]
        if out.get("bins") is not None:
            out["bins"] = self.bins
        return out

    def to(self, device):
        """Move every tensor to device (the dtype stays); returns self."""
        self.device = torch.device(device)
        for name in ("s_cart", "inv_d2", "F_obs", "work", "free"):
            setattr(self, name, getattr(self, name).to(self.device))
        if self.bins is not None:
            self.bins = self.bins.to(self.device)
        self.state = self._moved(self.state)
        self.single = (self.state if self.bins is None
                       else self._moved(self.single))
        return self
