# Design note: Bragg scaling for lunus.sf

Status: **proposed; being implemented on branch `sf-scaling-module`.**
First written 2026-11-06 from the state of `tools/fit_solvent_rfactor.py`, the
SW40 R-factor report and the sampleworks `diffuse_bragg` loss as of its commit
`dc238d7`. Updated 2026-11-17 to the code as it then stood:

- the R-factor tool's main fit is binned by default, with cctbx's log bins and
  a solvent scale per bin;
- `fit_scales` takes a warm start (`init=`);
- `--write-mtz` writes the scaled model as synthetic data;
- it also answers the sampleworks requirements raised in conversation with
  Doris (the section after "Why").

Numbers quoted are from runs recorded in `docs/solvent-design.md`, the SW40
report, or the sessions that wrote this. The sampleworks side is described
from that commit and from Doris's notes; it was not re-read for this update.

## Why

Guidance against experimental Bragg amplitudes needs a scaling model
between |F_calc| and |F_obs|. Today that model exists in two places and in
neither form is it usable for guidance:

- `tools/fit_solvent_rfactor.py` fits by least squares on amplitudes and
  reports R-work and R-free by shell. Its fit includes:
  - an overall scale, or one per resolution bin;
  - an overall isotropic or anisotropic B;
  - a solvent scale: k_sol / b_sol, or one k_mask per bin.

  Its main fit has been binned since 2026-11-17, using mmtbx's log bins and a
  bounded per-bin k_mask, and it reaches mmtbx's R on deposited models:
  - 8VPW: 0.1277 / 0.1502 against mmtbx's 0.1280 / 0.1522;
  - 2YL0 with hydrogens: 0.1374 / 0.1527 against 0.1391 / 0.1520.

  But `tools/` is not a package. `pyproject.toml` ships `lunus`, `lunus.sf`,
  `lunus.md` and `lunus.command_line` only, so a pip install of lunus,
  including the v0.2.1 sampleworks pins, cannot import any of it. The tool also
  reads observations through cctbx, which the `sf` extra deliberately does not
  require.
- sampleworks' `DiffuseBraggLoss._residual` fits one scalar per term, closed
  form, detached, at every evaluation. It has no B, no bins, no solvent fit and
  no R-factor. Its `per_conformer` solvent goes through
  `solvent_torch.SolventModel` / `f_total` at lunus's default k_sol and b_sol.

The SW40 swap test puts two thirds of the single-scale R gap between lunus.sf
and mmtbx in the scaling model and none in the structure factors. A loss
with a single scalar scale against experimental Fo is therefore dominated by
the radial falloff and the solvent term, neither of which is what guidance
should be moving atoms to fix.

## Requirements from the sampleworks side

From a conversation with Doris (kept from the notes at the head of the first
draft), each with where it stands.

1. **After the binned scaling method is decided, an option to export the
   synthetic data as .mtz.**
   - **Decided:** the binned method is cctbx log bins with a per-bin k_mask,
     the R-factor tool's default.
   - **Exists:** `fit_solvent_rfactor.py --write-mtz` (and `pdb_rfactor.py
     --write-mtz`) already writes the scaled model as synthetic data:
     - FP / SIGFP: |F_model| with the observed sigmas;
     - PHIC;
     - the deposited FreeR_flag;
     - optionally FMODEL over the complete set, FPROTEIN / FMASK unscaled, and
       FOBS.

     It writes the binned model by default. The writer needs cctbx, and stays
     in the tool.
   - **What moves:** the arithmetic that makes F_model (`model_structure_factors`)
     moves into the module. A cctbx-free client can then form the same
     synthetic F_model and write it with gemmi or reciprocalspaceship.
2. **Name the files the change touches.**
   - New: `lunus/sf/scaling_torch.py`, `lunus/sf/tests/test_scaling_torch.py`.
   - Changed:
     - `lunus/sf/tools/fit_solvent_rfactor.py`: the scaling arithmetic moves
       out and is imported back. The printed output and the MTZ are unchanged.
     - `lunus/sf/structure_factor_torch.py`: `return_parts`.
     - `lunus/sf/__init__.py`: the new public names.
     - `lunus/sf/tests/test_fit_scales.py`: its monkeypatches point at the
       module, where the functions now live.
     - `lunus/sf/README.md` and this note.
   - Not changed: `pdb_rfactor.py`, `solvent_torch.py`.
3. **The function and signature changes needed, for example giving the forward
   calculation the experimental data.**
   - The forward structure-factor call does not need F_obs. F_obs, the free set
     and the reflection geometry go to the `BraggScaler` constructor, once per
     reflection list.
   - The forward call's one change is `return_parts=True`, so the caller gets
     F_protein and F_mask separately ("The forward call", under API).
   - `fit_scales` and the other moved functions keep their signatures.
4. **The SW forward model takes a single .mtz, synthetic or Fo, and a flag that
   turns scaling on.**
   - **Sampleworks side, open.** The lunus side is what that flag switches.
   - On: a `BraggScaler` built from the MTZ's amplitudes and free flags, `fit`
     once, `refit("bins")` and `apply` per evaluation.
   - Off: compare unscaled amplitudes. That is meaningful only for a target
     written by the same pipeline on the same scale.
   - Synthetic data written by `--write-mtz` are on the observed data's
     scale, k_total(h)·|F_protein + k_mask F_mask|. A forward model scoring
     against them still needs scaling on, with the target's own free flags.
5. **A script that generates scaled synthetic data, whose functions
   sampleworks guidance reuses.**
   - The script is `fit_solvent_rfactor.py --write-mtz` (or `pdb_rfactor.py ID
     --write-mtz`).
   - After this change the functions it uses to scale are the module's:
     - `fit_scales`, which `BraggScaler.fit` calls;
     - `model_structure_factors`, which `BraggScaler.apply_complex` is,
       differentiably.

     So the synthetic data and the guidance loss come from one implementation.
   - Reading the observations (French-Wilson, free flags) and writing the MTZ
     stay cctbx-based in the tool. Sampleworks reads and writes with its own
     gemmi / reciprocalspaceship.

## The model

As in the tool:

    |F_model(h)| = k_bin(h) · exp(−¼ s·B_cart·s) · | F_protein(h) + k_mask(h) · F_mask(h) |

with s = M* h the Cartesian reciprocal vector in Å⁻¹ and s² = 1/d².

- `B_cart` is a symmetric 3×3 in the Cartesian reciprocal frame, every
  component a B in Å². The tool learned this the hard way: in the integer-h
  basis the diagonal components differ by the squared cell-edge ratio (70× on
  7FPV) and an optimizer stepping them together overflowed `exp()`. The
  isotropic model is B_cart = B·I. Conversion to `f_total`'s U is
  U = M*ᵀ B_cart M* / 4.
- `k_bin` is one scale per resolution bin, or a single scale when bins are
  off. The bins are mmtbx's by default: `log_resolution_bins`, a port of
  `cctbx.miller.set.log_binning`, fine at low resolution where the solvent and
  the scale change fastest, few and wide at high resolution, and fewer when
  there are fewer reflections. N equal-count bins (`resolution_bins`) remain
  available. Either way the bins are assigned on the whole reflection list,
  free reflections included, so a scale fitted on the work set applies to the
  free set unchanged.
- `k_mask` is, by default with log bins, one solvent scale per bin. That is
  what mmtbx fits, and it closed the median lunus-vs-mmtbx R-free gap on SW40
  from +0.0020 to +0.0011. It is bounded to [0, 1] (`SOLVENT_BIN_MAX`; 4Y0B
  ran away without the bound) and regularised by mmtbx's rule
  (`solvent_curve_rule`): no negatives, zero from the first zero bin below
  3 Å, and a moving average at local extrema. With `solvent_bins` off it is
  the exponential k_sol · exp(−¼ b_sol s²).
  - The k_sol / b_sol a per-bin fit reports are the exponential nearest the
    curve at d ≥ 4 Å, as mmtbx reports them. They are a diagnostic, not
    parameters, and not comparable to the conventional 0.35 / 46.
  - The single-scale fit's k_sol / b_sol remain that diagnostic.
- With bins on, the isotropic part of B is degenerate with the bin scales and
  drifts (B −34 against bin scales falling to 0.01 on 8VQ1). The tool holds
  the isotropic part at the single-scale answer and refines only the
  anisotropy alongside the bins. The module keeps that rule.
- The fit is unweighted least squares on amplitudes over the work set, as the
  tool's is and for the reason its docstring gives. A 1/σ² weighting is a
  later option, not a default.
- F_total is linear in F_protein and F_mask for fixed solvent scale, so the
  ensemble-mean Bragg amplitude is the model applied to the ensemble means.
  The module takes means. It never sees configurations.

## Which parameters move when: the policy

The loss is L(x, θ): x the coordinates, θ the scale parameters. There are
three ways to treat θ.

**Fixed θ**, fitted once against the starting model. The loss is a fixed
function of x, so its value is comparable across steps. But if the
ensemble's effective B drifts during sampling, the only way the loss can
reduce the mismatch is through x: it will contract or expand the ensemble to
fake a B change. Early in diffusion, when structures are far from the data,
a fixed scale makes the loss enormous and uninformative.

**Refit θ at every step, detached** (what `fit_scale` does for its one
scalar). Solve θ*(x) under `no_grad`, then backpropagate ∂L/∂x with θ* held
constant. The sampler then minimizes the profiled loss L*(x) = min_θ L(x, θ).
Its exact derivative is ∂L/∂x + (∂L/∂θ)(dθ*/dx), and ∂L/∂θ = 0 at a
minimizer, so the second term vanishes: detaching is the exact gradient of
the profiled loss, not an approximation. Four conditions:

- **The inner fit must reach a stationary point.** Closed-form parameters do by
  construction. An LBFGS stopped early does not, and the gradient is then
  slightly off.
- **Bounds do not break it.** A b_sol pinned at its floor (mmtbx's sits at 1.9
  on 5IMV), or a k_mask bin at 0 or at 1, is a constrained minimizer, and the
  constraint does not depend on x.
- **The per-bin curve rule does break it.** The cut and the moving average move
  k_mask away from the per-bin minimizer, so after `refit("solvent")` the
  k_mask bins are not stationary, and the zero threshold (0.005) can switch
  bins in or out between refits. That is one reason the per-bin solvent is held
  between refits rather than refit every step: held, the loss is an ordinary
  function of x and its gradient is exact.
- **Jumps between basins break it.** With the exponential solvent, (k_sol,
  b_sol) is non-convex, which is why the tool scans before LBFGS. If a step's
  scan lands in a different basin from the previous step's, the profiled loss
  has a kink and the sampler sees a sudden change of direction. A warm start
  prevents the jumps at the price of tracking one basin. The per-bin solve is
  a bounded one-dimensional search per bin with no basin to track, but the
  curve rule's discontinuity above plays the same role.

**Refit θ with gradients through the fit** gives the same first-order
gradient as the detached case when the inner fit converges, at a much
higher cost. Not pursued.

What the envelope theorem does not say is that the profiled loss is the right
objective. Every degree of freedom in θ is a component of the residual the
coordinates are no longer asked to explain:
- a scalar removes the overall magnitude;
- B removes the mean falloff;
- bins remove the entire radial profile;
- a per-bin k_mask removes the radial profile of the solvent contrast;
- an anisotropic B_cart removes the direction-dependent falloff, which may be
  genuine anisotropic disorder.

The SW40 result says these components are mostly not coordinate errors, which
argues for absorbing them. The `fit_scale` docstring's caveat, that scaling
discards the magnitude of disorder the data carries, is the counter-argument
for one of them.

**The split this note proposes**, by how x-dependent and how well conditioned
each parameter is:

| parameter | per step | why |
|---|---|---|
| k_bin (or single k) | closed form, detached, every step: `refit("bins")` | exactly linear given the rest: log k_bin = log(Σ_bin Fo·g / Σ_bin g²). O(N), no optimizer, cannot jump, stationary by construction |
| isotropic B | held from the preliminary fit | degenerate with bins when bins are on; when off (`bins=0`), refit with the rest by `refit("all")` |
| k_mask per bin (default) | held; `refit("solvent")` every N steps, detached | the bounded per-bin solve, the curve rule, then the bin scales in closed form. The rule's cut and smoothing are not smooth in x, so it is not run per step |
| k_sol, b_sol (exponential: `solvent_bins` off, or `bins=0`) | held; `refit("solvent")` every N steps, LBFGS from the current values, detached | physical, slowly varying, non-convex, prone to basin changes |
| anisotropic B_cart | held by default; `refit_aniso=True` adds it to the periodic refit | the open choice: an instrument effect or a disorder signal |

Defaults for guidance, all to be measured (open questions 1 and 2):
- log bins with a per-bin k_mask, the tool's default, so the loss scores
  within-shell structure and agrees with the R that is reported;
- a preliminary full `fit` at step 0;
- `refit("bins")` every step;
- `refit("solvent")` never after step 0.

The single-scale model (`bins=0`) stays available, because its k_sol / b_sol
remain the diagnostic they are without bins.

## API

Flat functions carry the arithmetic, as `solvent_torch.py` does. One class,
`BraggScaler`, bundles the per-reflection-list state that is built once and
reused on every call, the pattern `IsotropicBackground` already sets. There is
no new class elsewhere.

The flat functions are the tool's, moved with their names, signatures and
numbers unchanged. The two that were underscore-private in the tool become
public, since another module now calls them. The tool keeps the old names as
aliases.

```python
# scaling_torch.py: moved from tools/fit_solvent_rfactor.py

# the model
scaled_amplitudes(F_calc, inv_d2, s_cart, log_k, u_params, aniso)   # was _scaled_amplitudes
inner_f_calc(F_protein, F_mask, inv_d2, k_sol, b_sol)                 # was _f_calc
optimal_log_k(F_obs, model_amplitudes)                                # closed-form overall scale
r_factor(F_obs, F_model)

# bins
resolution_bins(inv_d2, n_bins)                                       # equal-count, 0 = lowest resolution
log_resolution_bins(d, n_lowest=100, min_per_bin=50, max_bins=30,
                    eps=1e-4, device="cpu")                           # mmtbx's bins
assign_scale_bins(spec, inv_d2, d)                                    # "log" | N | 0 -> bins or None

# the fit
scan_grids(fit_solvent, like); coarse_scan(...); coarse_scan_reference(...)
fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso,
           fit_solvent=True, bins=None, solvent_bins=False,
           log=sys.stdout, init=None) -> dict                         # unchanged
solvent_curve_rule(...); solvent_cut_index(...); mmtbx_moving_average(...)
exponential_from_curve(k_bins, bins, inv_d2, d_cut=4.0)
SOLVENT_BIN_MAX, SOLVENT_BIN_FORCE_CUT, UNIQUE_IJ

# a fit's model (fit = the dict fit_scales returns)
solvent_weight(fit, inv_d2); solvent_scheme(fit)
model_f_c(...); model_log_k(fit); model_scale(...)
model_amplitudes(fit, F_protein, F_mask, inv_d2, s_cart)              # no_grad, as now
model_structure_factors(fit, F_protein, F_mask, inv_d2, s_cart)       # no_grad, as now

# reporting
shell_rows(d, F_obs, model_no_solvent, model_solvent, work, free,
           n_shells, solvent_share) -> list[tuple]                    # the rows the tool's shell_table prints


class BraggScaler:
    """Scale state for one reflection list. Build once; call per evaluation."""

    def __init__(self, hkl, orth_matrix, F_obs, work, *, bins="log",
                 solvent_bins="auto", aniso=False, fit_solvent=True, d=None,
                 device="cpu", dtype=torch.float64)
    def fit(self, F_protein, F_mask=None, *, warm=False, log=None) -> dict
    def refit(self, F_protein, F_mask=None, *, what="bins", refit_aniso=False) -> None
    def apply(self, F_protein, F_mask=None) -> Tensor          # |F_model|, differentiable in F_*
    def apply_complex(self, F_protein, F_mask=None) -> Tensor  # F_model with F_c's phase, differentiable
    def r_factors(self, F_model) -> tuple[float, float]         # R-work, R-free
    def state_dict(self) -> dict
    def load_state_dict(self, state) -> None
    def to(self, device) -> "BraggScaler"
```

Notes on the signature:

- **Constructor arguments** are fixed per reflection list: `hkl`, `orth_matrix`,
  `F_obs` and `work` (a boolean mask; `~work` is the free set). From them the
  constructor computes s, s² and the bins, once.
  - `bins` takes the tool's `--scale-bins` values: `"log"` (default), N
    equal-count bins, or 0 for one scale.
  - `solvent_bins="auto"` means a per-bin k_mask with log bins and the
    exponential otherwise, the tool's `--solvent-bins auto`.
  - `d`, optional, is the d spacings to bin by. Pass cctbx's own
    (`F_obs_array.d_spacings()`) to get mmtbx's log bins exactly, since a
    reflection sitting on a limit can move if d is recomputed some other way.
    The default is 1/√s² from `hkl` and `orth_matrix`.
- **`fit`** runs `fit_scales` on the inputs: the single-scale fit (scan,
  LBFGS), then with bins the bin fit from it (`init=`), as the tool's run
  does. It sets the state and returns the binned fit's dict.
  - `warm=True` skips the scan and runs the single-scale LBFGS from the
    current single-scale values, then the bins and the per-bin solvent. The
    first draft asked for a narrow scan around the current (k_sol, b_sol). This
    drops it: an LBFGS from the current values tracks the current basin, which
    is what warm start is for, and a scan, however narrow, is what lets a fit
    leave it.
  - `log=None` is silent. Pass a stream to see the tool's progress lines.
- **`refit`**, with `F_protein`, `F_mask` the per-evaluation inputs:
  - `what="bins"` is the closed-form bin (or single) scale with everything
    else held. It is the only thing meant to run every step: one reduction
    per bin, under `no_grad`.
  - `what="solvent"` refits the solvent scale with the bin scales:
    - per bin: the bounded per-bin solve, the curve rule, the bin scales in
      closed form;
    - exponential: LBFGS on k_sol, b_sol and the scales from the current
      values.

    `refit_aniso=True` adds B_cart to it.
  - `what="all"` is `fit(warm=True)`.
- **`apply` and `apply_complex`** never detach anything inside the forward. The
  parameters they use are plain tensors without `requires_grad`, set by
  `fit`/`refit` under `no_grad`, so the caller's gradient flows through
  `F_protein` and `F_mask` only. This is the detached policy made structural
  rather than a `.detach()` at the call site. `apply_complex` is the
  differentiable `model_structure_factors`: synthetic data for an MTZ come
  from it.
- **`F_mask=None` with `fit_solvent=False`** is the no-solvent model.
- **The free set** is never used in `fit` or `refit`. `r_factors` reports both.
- **`state_dict()`** holds:
  - the bins;
  - log_k (single or per bin);
  - B_cart (`u_params`);
  - k_sol, b_sol;
  - k_mask per bin (or None);
  - the single-scale fit;
  - `last_refit`: the evaluation count (calls to `apply`/`apply_complex`) at
    the last `fit` (and whether it was warm) and at the last refit of each
    kind, and the last `what`.

  The SW40 study holds the scale parameters between refits and wants its run
  record to say whether the solvent mask was live or frozen over the same
  interval. The mask is rebuilt on every forward call inside
  `structure_factors_one_config` today. A frozen-mask mode is on
  `docs/solvent-design.md`'s list and is not this module's to build, but the
  scaler can report its own cadence, so that the record is complete whichever
  way the mask was run.
- **float64** is the default because the tool's numbers were measured in it.
  float32 works (a test checks that the R differs by less than the printed
  precision), but is a measurement to make, not a default to assume.

**The forward call** (`structure_factor_torch.py`):

```python
structure_factors_one_config(..., solvent=model, return_parts=True)
    -> SolventParts(F_total, F_protein, F_mask)      # a namedtuple
structure_factors_batch(..., solvent=model, return_parts=True)
    -> SolventParts of (n_configs, n_refl) tensors
```

F_mask is returned with the density scale applied, the supercell fold's
n_cells times `SolventModel.density_scale`, so that `F_protein + k·F_mask`
with a fitted k is on the same per-cell scale as k_sol, and F_total reassembles
as `f_total(F_protein, F_mask, inv_d2, k_sol, b_sol, k_overall, u_aniso, hkl)`
with density_scale 1. For an ensemble, the means of F_protein and F_mask over
configurations go to the scaler (linearity, above).

**The tool** keeps calling the flat functions as it does now, imported from the
module. Its printed output and its MTZ are unchanged, and `pdb_rfactor.py` does
not change. `BraggScaler.fit` calls the same `fit_scales` the tool calls, so
the two cannot drift apart.

**For sampleworks** (theirs to make; listed so it can be passed on):
`DiffuseBraggLoss` would build a `BraggScaler` in `prepare()` from its
reflection list and free flags. On each evaluation it would:
- call the forward with `return_parts=True`;
- `fit` once on the first evaluation;
- `refit("bins")`;
- `apply` to form the model amplitudes in place of today's scalar;
- log `r_factors` alongside the loss.

Its `exclude_free_reflections` keeps its meaning: the scaler never fits on the
free set, and the loss can still drop it from the residual.

## Cost

Per step, `refit("bins")` plus `apply` are a handful of vector operations over
the reflection list (56k to 326k reflections on SW40), and one small reduction
per bin. That is well under the structure-factor cost.

The preliminary `fit` costs more:
- the batched coarse scan (25 × 41 × 26 nodes, one device synchronisation);
- the single-scale LBFGS, the bin LBFGS and the per-bin solvent solve.

In the tool the binned fit with solvent, started from the single-scale fit,
takes about a third of its time before the warm start and the batched scan
(SW40 and TR1000, 2026-11-15). So the fit is a few seconds once per entry on
CPU, and a warm or solvent refit a fraction of that.

## Tests

- **Parity with the tool.** `BraggScaler` reproduces, to the printed digits on
  7FPV, the tool's single-scale and binned (log bins, per-bin k_mask) R-work
  and R-free, k_sol and b_sol, on the same F_protein and F_mask. The tool's own
  acceptance is unchanged output on 8VPW and 2YL0 with hydrogens, default and
  `--single-scale`, and an unchanged `--write-mtz`.
- **Closed form equals the optimizer.** Bin scales from `optimal_log_k`
  against bin scales from LBFGS on the same objective, to tolerance.
- **The envelope gradient.** The finite difference of the profiled loss with
  respect to F_protein, against the autograd gradient through `apply` after
  `refit("bins")`: equal to first order. The same check with a deliberately
  unconverged solvent refit, to show the size of the error that case makes.
- **Free set untouched.** Perturb F_obs on the free set only: the `fit` and
  `refit` state must not change, and `r_factors`' second value must.
- **Bins assigned on the full list.** A free reflection falls in the same
  bin whether or not the free set is present in the fit.
- **Warm start.** `fit(warm=True)` after a small displacement lands in the same
  basin as the cold fit's.
- **The forward call.** `return_parts=True` reassembles to the default
  F_total, and the default return is bit-identical to before.
- **f_total.** The module's single-scale exponential model equals `f_total`
  with U converted from B_cart.
- **Shapes and placement.** CPU float32 and float64, and the float32 R differs
  from float64 by less than the printed precision. CUDA when available.

## First refinement client: displacement refinement on SW40

The first use of `BraggScaler` outside the tool is a study the manuscript
needs: on every entry of the SW40 test set, do coordinates displaced from
the deposited model return to it when refined against the deposited
amplitudes with lunus.sf's gradients? The taper study answered this with
the model's own amplitudes as the target (`docs/design.md`, "What the taper
study measured"). This is the same test with the real target, which is only
possible once scaling and solvent sit inside the loss. It is a tool and may
depend on cctbx. It is a separate job from the module.

**Per entry.**

- **Model:**
  - the deposited coordinates, heavy protein atoms, conformer A at full
    occupancy, hydrogens excluded;
  - selections applied in file order (`atom_order.selection_in_file_order`;
    the `--select` / `--occ-one` path `tools/study_taper.py` already has);
  - ADPs as deposited through the anisotropic kernel, or flattened to
    isotropic. The record states which.
- **Target:** French-Wilson amplitudes and the deposited free set from the
  entry's structure-factor file, through `fit_solvent_rfactor.read_observations`.
- **Baseline:** `BraggScaler.fit` on the undisplaced model; record R-work and
  R-free. These must match the R-factor tool's rows for the entry to four
  decimals. If they do not, the study stops there.
- **Start:** displace every free coordinate by 0.3 Å rms, with one recorded
  seed, then `fit(warm=True)` on the displaced model, so the optimizer is not
  started from scales fitted to a different structure.
- **Refine:** L-BFGS over the coordinates, 300 iterations, strong-Wolfe line
  search, tolerances off so that an early stop is a failed line search. The
  loss is on the work set only. Each evaluation runs
  `structure_factors_one_config` with `solvent=` on and `return_parts=True`,
  then `refit("bins")`, `apply`, and the residual against F_obs. k_mask (or
  k_sol and b_sol) and B_cart are held from the start-of-run fit by default,
  with `refit("solvent")` every N iterations as a variable.
- **Record,** one JSON row per entry:
  - atoms free, and reflections per coordinate;
  - rmsd to the deposited coordinates at start and end;
  - R-work and R-free at baseline, start and end;
  - iterations completed, and whether the line search failed;
  - the scale parameters at baseline and end;
  - the scaler's `last_refit` record.

**What is read from it.**

- **Per entry:** how much of the 0.3 Å is recovered, and whether R-free ends
  near the baseline. An entry where R-work falls while R-free rises is a
  statement about what the amplitudes alone determine at that resolution
  with no restraints, not about the engine. The table is ordered by
  reflections per coordinate so that reading is available.
- **For the module:**
  - open question 2, the solvent refit cadence: run a subset of entries at
    never, every 50 and every 10 iterations, and compare end R-free and rmsd;
  - open question 1, B_cart held or refit: the same comparison with
    `refit_aniso` on and off.

  These are the measurements the policy table says it is waiting for.
- **For the taper,** optionally: each entry at the production taper and with
  the hard cutoff, which ties this to the taper study's conclusion that where
  the taper matters depends on the system.

**Cost.** The taper study's 7FPV refinements ran at `d_min` 2.0 on 11,011
reflections and 1,270 free atoms. The SW40 entries at full resolution carry
56,000 to 326,000 reflections and, from the 12 cached locally, 1,500 to
5,200 heavy atoms, with the largest entries of the forty several times that
again. Expect minutes per entry on the GPU rather than seconds, and an
afternoon for forty entries at one condition.

## Open questions

1. **Anisotropic B_cart during guidance.** Held or refit, and whether a
   non-orthorhombic cell makes the off-diagonals meaningful enough to matter.
   The tool's note that orthorhombic symmetry forces them to zero is a cheap
   correctness check in either mode. To be measured by the SW40 refinement
   study ("First refinement client", above), with `refit_aniso` on against off.
2. **Refit cadence for the solvent scale** (k_mask per bin, or k_sol and
   b_sol). Never, every N steps, or on a schedule tied to the sampler's noise
   level. To be measured by the SW40 refinement study at never, every 50 and
   every 10 iterations. A guidance run, where the noise level matters too,
   remains to be measured after it.
3. **Amplitudes versus intensities.** The tool, the R-factor and the
   sampleworks Bragg term are all on amplitudes. If a guidance loss on
   intensities is wanted later, the closed-form bin scale changes (least
   squares on I is still linear in k² per bin), and R stays on |F|.
4. **Weighting.** Unweighted now; 1/σ² when a target supplies σ. The
   observation reader, not this module, decides whether σ is available.
5. **Where French-Wilson lives for sampleworks:** `rs.algorithms` or
   `gemmi.french_wilson`. The tool's cctbx path stays as the reference the two
   are checked against.
6. **The SW forward model's single-MTZ input and its scaling flag**
   (requirement 4): sampleworks' design, with the lunus side above.
