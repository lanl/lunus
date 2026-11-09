# Design note: Bragg scaling for lunus.sf

Status: **proposed, nothing implemented.** Written 2026-11-06 from the state
of `tools/fit_solvent_rfactor.py` (the arithmetic this note moves), the
SW40 R-factor report, and the sampleworks `diffuse_bragg` loss as of its
commit `dc238d7`. Every number quoted is from a run recorded in
`docs/solvent-design.md`, the report, or the session that wrote this.

## Why

Guidance against experimental Bragg amplitudes needs a scaling model
between |F_calc| and |F_obs|. Today that model exists in two places and in
neither form is it usable for guidance:

- `tools/fit_solvent_rfactor.py` fits an overall scale, an overall isotropic
  or anisotropic B, optional per-resolution-bin scales, and k_sol / b_sol by
  least squares on amplitudes, and reports R-work and R-free by shell. It
  reaches mmtbx's R on deposited models (7FPV: 0.1297 / 0.1508 binned against
  mmtbx's 0.1298 / 0.1510). But `tools/` is not a package. `pyproject.toml`
  ships `lunus`, `lunus.sf`, `lunus.md` and `lunus.command_line` only, so a
  pip install of lunus, including the v0.2.1 sampleworks pins, cannot import
  any of it. It also reads observations through cctbx, which the `sf` extra
  deliberately does not require.
- sampleworks' `DiffuseBraggLoss._residual` fits one scalar per term, closed
  form, detached, at every evaluation. No B, no bins, no solvent fit (its
  `per_conformer` solvent uses lunus's default k_sol and b_sol), no R-factor.

The SW40 swap test puts two thirds of the single-scale R gap between lunus.sf
and mmtbx in the scaling model and none in the structure factors. A loss
with a single scalar scale against experimental Fo is therefore dominated by
the radial falloff and the solvent term, neither of which is what guidance
should be moving atoms to fix.

## What the module is

A torch module, `lunus/sf/scaling_torch.py`, cctbx-free, that takes
amplitudes and reflection geometry as tensors and provides:

1. the scaled model amplitude, differentiable with respect to F_protein and
   F_mask, so gradients reach coordinates through the existing pipeline;
2. a full fit of the scale parameters against F_obs on the work set;
3. a cheap per-step refit of the parameters that are closed form;
4. R-work and R-free, and a per-shell table, from the same amplitudes.

Reading observations (French-Wilson, free flags, merging) stays outside the
module: the tool keeps its cctbx reader, and sampleworks already has gemmi
and reciprocalspaceship, both of which read MTZ and mmCIF and do
French-Wilson (`gemmi.french_wilson`, `rs.algorithms.scale_merged_intensities`).
The module's contract is tensors in, tensors out.

## The model

Unchanged from the tool and from `solvent_torch.f_total`:

    |F_model(h)| = k_bin(h) · exp(−¼ s·B_cart·s) · | F_protein(h) + k_sol · exp(−¼ b_sol s²) · F_mask(h) |

with s = M* h the Cartesian reciprocal vector in Å⁻¹ and s² = 1/d².

- `B_cart` is a symmetric 3×3 in the Cartesian reciprocal frame, every
  component a B in Å². The tool learned this the hard way: in the integer-h
  basis the diagonal components differ by the squared cell-edge ratio (70× on
  7FPV) and an optimizer stepping them together overflowed `exp()`. The
  isotropic model is B_cart = B·I. Conversion to `f_total`'s U is
  U = M*ᵀ B_cart M* / 4.
- `k_bin` is one scale per equal-count resolution bin (bins assigned on the
  whole reflection list, free included, so a scale fitted on the work set
  applies to the free set unchanged), or a single scale when bins are off.
- With bins on, the isotropic part of B is degenerate with the bin scales and
  drifts (B −34 against bin scales falling to 0.01 on 8VQ1). The tool holds
  the isotropic part at the single-scale answer and refines only the
  anisotropy alongside the bins. The module keeps that rule.
- The fit is unweighted least squares on amplitudes over the work set, as the
  tool's is and for the reason its docstring gives. A 1/σ² weighting is a
  later option, not a default.
- F_total is linear in F_protein and F_mask for fixed k_sol and b_sol, so the
  ensemble-mean Bragg amplitude is the model applied to the ensemble means.
  The module takes means; it never sees configurations.

## Which parameters move when: the policy

The loss is L(x, θ): x the coordinates, θ the scale parameters. Three ways
to treat θ.

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
the profiled loss, not an approximation. Three conditions:

- the inner fit must reach a stationary point. Closed-form parameters do by
  construction; an LBFGS stopped early does not, and the gradient is then
  slightly off;
- bounds do not break it. A b_sol pinned at its floor (mmtbx's sits at 1.9 on
  5IMV) is a constrained minimizer, and the constraint does not depend on x;
- jumps between basins do. (k_sol, b_sol) is non-convex, which is why the
  tool scans before LBFGS. If a step's scan lands in a different basin from
  the previous step's, the profiled loss has a kink and the sampler sees a
  sudden change of direction. A warm start prevents the jumps at the price of
  tracking one basin.

**Refit θ with gradients through the fit** gives the same first-order
gradient as the detached case when the inner fit converges, at a much
higher cost. Not pursued.

What the envelope theorem does not say is that the profiled loss is the right
objective. Every degree of freedom in θ is a component of the residual the
coordinates are no longer asked to explain: a scalar removes the overall
magnitude, B the mean falloff, bins the entire radial profile, an anisotropic
B_cart the direction-dependent falloff, which may be genuine anisotropic
disorder. The SW40 result says these components are mostly not coordinate
errors, which argues for absorbing them; the `fit_scale` docstring's caveat,
that scaling discards the magnitude of disorder the data carries, is the
counter-argument for one of them.

**The split this note proposes**, by how x-dependent and how well conditioned
each parameter is:

| parameter | per step | why |
|---|---|---|
| k_bin (or single k) | closed form, detached, every step | exactly linear given the rest: log k_bin = log(Σ_bin Fo·g / Σ_bin g²). O(N), no optimizer, cannot jump |
| isotropic B | held from the preliminary fit | degenerate with bins when bins are on; when off, cheap to refit by one-dimensional LBFGS with warm start |
| k_sol, b_sol | held; refit every N steps with warm start, detached | physical, slowly varying, non-convex, prone to basin changes |
| anisotropic B_cart | held by default; `refit_aniso=True` to include it in the periodic refit | the open choice: between an instrument effect and a disorder signal |

Defaults for guidance: bins on (20), so the loss scores within-shell
structure and agrees with the R that is reported; a preliminary full fit at
step 0; per-step closed-form bin refit; k_sol and b_sol refit every
`refit_every` steps (default: never after step 0, until measured). The
single-scale case stays available because its k_sol / b_sol remain the
diagnostic they are without bins.

## API

Flat functions carry the arithmetic, as `solvent_torch.py` does; one class
bundles the per-reflection-list state that is built once and reused on every
call, the pattern `IsotropicBackground` already sets. No new class elsewhere.

```python
# scaling_torch.py

def scaled_amplitudes(F_calc, inv_d2, s_cart, log_k, B_cart) -> Tensor       # |F| after the overall scale
def inner_f_calc(F_protein, F_mask, inv_d2, k_sol, b_sol) -> Tensor           # f_total's bracket
def optimal_log_k(F_obs, g, bins=None, n_bins=None) -> Tensor                 # closed form, per bin when bins given
def resolution_bins(inv_d2, n_bins) -> LongTensor                              # equal-count, 0 = lowest resolution
def r_factor(F_obs, F_model, mask=None) -> float
def shell_table(d, F_obs, F_model, work, free, n_shells) -> list[tuple]       # what the tool prints

class BraggScaler:
    """Scale state for one reflection list. Build once; call per evaluation."""
    def __init__(self, hkl, orth_matrix, F_obs, work, *, n_bins=20, aniso=False,
                 fit_solvent=True, device="cpu", dtype=torch.float64)
    def fit(self, F_protein, F_mask=None, log=None) -> dict                     # scan + LBFGS; sets state; no_grad
    def refit(self, F_protein, F_mask=None, *, what="bins") -> None             # per-step; "bins" closed form, "solvent" warm LBFGS, "all"
    def apply(self, F_protein, F_mask=None) -> Tensor                           # |F_model|, differentiable in F_*
    def r_factors(self, F_model) -> tuple[float, float]                          # R-work, R-free
    def state_dict(self) / load_state_dict(self, d)                              # k_bins, B_cart, k_sol, b_sol, bins
    def to(self, device)
```

Notes on the signature:

- `F_obs` and `work` (a boolean mask; `~work` is the free set) are fixed per
  reflection list and so belong to the constructor. `F_protein`, `F_mask` are
  the per-evaluation inputs. `F_mask=None` with `fit_solvent=False` is the
  no-solvent model.
- `apply` never detaches anything inside the forward: the parameters it uses
  are plain tensors without `requires_grad`, set by `fit`/`refit` under
  `no_grad`. The caller's gradient flows through `F_protein` and `F_mask`
  only. This is the detached policy made structural rather than a `.detach()`
  at the call site.
- `refit(what="bins")` is the only thing meant to run every step. It is one
  reduction per bin and allocates nothing.
- `n_bins=0` gives the single-scale model and lets `refit(what="all")` move
  the isotropic B as well.
- The free set is never used in `fit` or `refit`. `r_factors` reports both.
- float64 is the default because the tool's numbers were measured in it;
  float32 is a measurement to make, not a default to assume.

Two additions for the first client (the SW40 refinement study, below):

- `fit(..., warm=True)` starts the scan and the LBFGS from the current
  state instead of from the conventional values. The study refits after
  displacing the coordinates and wants the same basin the baseline found;
  the warm-start remark under `refit` covers `refit` but not `fit`. With
  `warm=True` the scan is a narrow one around the current (k_sol, b_sol)
  rather than the full grid.
- The scaler keeps a record of what it holds and when it last moved each
  parameter: the iteration or call count at the last `fit`, at the last
  `refit` of each kind, and the mode in force (`what`), exposed as
  `last_refit` in `state_dict()`. The study holds the scale parameters
  between refits and wants its run record to say whether the solvent mask
  was live or frozen over the same interval. The mask is rebuilt on every
  forward call inside `structure_factors_one_config` today; a frozen-mask
  mode is on `docs/solvent-design.md`'s list and is not this module's to
  build. What the scaler can do is report its own cadence, so that the
  record is complete whichever way the mask was run.

Neither addition changes the sampleworks path.

The tool becomes a client: `fit_solvent_rfactor.run()` builds a
`BraggScaler`, calls `fit`, prints its own tables from `shell_table` and
`r_factors`. `pdb_rfactor.py` does not change. The module ships; the tool
still does not.

For sampleworks, `DiffuseBraggLoss` would hold a `BraggScaler` built in
`prepare()` from its reflection list and free flags, call `fit` once on the
first evaluation, `refit("bins")` on each, `apply` to form the model
amplitudes in place of today's scalar, and log `r_factors` alongside the
loss. Its `exclude_free_reflections` keeps its meaning: the scaler never fits
on the free set, and the loss can still drop it from the residual.

## Cost

Per step: `refit("bins")` plus `apply` are a handful of vector operations
over the reflection list (56k to 326k reflections on SW40), microseconds on
a GPU and well under the structure-factor cost. The preliminary `fit` is the
tool's scan, 25 × 41 × 26 nodes each a vector op over the work set, plus two
LBFGS solves; the tool runs the whole thing in 10 to 28 s per SW40 entry on
four CPU cores including both structure-factor calculations, so the fit
alone is a few seconds once, and a warm-started solvent refit is a fraction
of that.

## Tests

- **Parity with the tool.** 7FPV with `--aniso-adp --aniso`: single-scale
  0.1312 / 0.1522, binned 0.1297 / 0.1508 (R-work / R-free), reproduced
  through `BraggScaler` to four decimals before the tool is switched over to
  it. The SW40 `table_data.json` values are the wider check, and 12 of those
  entries are cached in `examples/compare_gemmi` after the hydrogen study.
- **Closed form equals the optimizer.** Bin scales from `optimal_log_k`
  against bin scales from LBFGS on the same objective, to tolerance.
- **The envelope gradient.** Finite difference of the profiled loss with
  respect to F_protein against the autograd gradient through `apply` after
  `refit("bins")`: equal to first order. The same check with a deliberately
  unconverged solvent refit, to show the size of the error that case makes.
- **Free set untouched.** Perturb F_obs on the free set only; `fit` and
  `refit` state must not change, `r_factors`' second value must.
- **Bins assigned on the full list.** A free reflection falls in the same
  bin whether or not the free set is present in the fit.
- **Shapes and placement.** Works on CUDA and CPU, float32 and float64, and
  the float32 R differs from float64 by less than the printed precision.

## First refinement client: displacement refinement on SW40

The first use of `BraggScaler` outside the tool is a study the manuscript
needs: on every entry of the SW40 test set, do coordinates displaced from
the deposited model return to it when refined against the deposited
amplitudes with lunus.sf's gradients? The taper study answered this with
the model's own amplitudes as the target (`docs/design.md`, "What the taper
study measured"). This is the same test with the real target, which is only
possible once scaling and solvent sit inside the loss. It is a tool and may
depend on cctbx.

**Per entry.**

- Model: deposited coordinates, heavy protein atoms, conformer A at full
  occupancy, hydrogens excluded, selections applied in file order
  (`atom_order.selection_in_file_order`; the `--select` / `--occ-one` path
  `tools/study_taper.py` already has). ADPs as deposited through the
  anisotropic kernel, or flattened to isotropic; the record states which.
- Target: French-Wilson amplitudes and the deposited free set from the
  entry's structure-factor file, through `fit_solvent_rfactor.read_observations`.
- Baseline: `BraggScaler.fit` on the undisplaced model; record R-work and
  R-free. These must match the R-factor tool's rows for the entry in the
  SW40 report's `table_data.json` to four decimals. If they do not, the
  study stops there.
- Start: displace every free coordinate by 0.3 Å rms, one recorded seed,
  then `fit(warm=True)` on the displaced model, so the optimizer is not
  started from scales fitted to a different structure.
- Refine: L-BFGS over the coordinates, 300 iterations, strong-Wolfe line
  search, tolerances off so that an early stop is a failed line search.
  Loss on the work set only. Each evaluation: `structure_factors_one_config`
  with `solvent=` on, `refit("bins")`, `apply`, residual against F_obs.
  k_sol, b_sol and B_cart held from the start-of-run fit by default;
  `refit("solvent")` every N iterations as a variable.
- Record, one JSON row per entry: atoms free, reflections per coordinate,
  rmsd to the deposited coordinates at start and end, R-work and R-free at
  baseline, start and end, iterations completed, whether the line search
  failed, the scale parameters at baseline and end, and the scaler's
  `last_refit` record.

**What is read from it.**

- Per entry: how much of the 0.3 Å is recovered, and whether R-free ends
  near the baseline. An entry where R-work falls while R-free rises is a
  statement about what the amplitudes alone determine at that resolution
  with no restraints, not about the engine; the table is ordered by
  reflections per coordinate so that reading is available.
- For the module: open question 2, the refit cadence for k_sol and b_sol,
  by running a subset of entries at never, every 50 and every 10
  iterations and comparing end R-free and rmsd; and open question 1, the
  anisotropic B_cart held or refit, by the same comparison with
  `refit_aniso` on and off. These are the measurements the policy table
  says it is waiting for.
- For the taper: optionally, each entry at the production taper and with
  the hard cutoff, which ties this to the taper study's conclusion that
  where the taper matters depends on the system.

**Cost.** The taper study's 7FPV refinements ran at `d_min` 2.0 on 11,011
reflections and 1,270 free atoms. The SW40 entries at full resolution carry
56,000 to 326,000 reflections and, from the 12 cached locally, 1,500 to
5,200 heavy atoms, with the largest entries of the forty several times that
again, so expect minutes per entry on the GPU rather than seconds, and an
afternoon for forty entries at one condition.

## Open questions

1. **Anisotropic B_cart during guidance.** Held or refit, and whether a
   non-orthorhombic cell makes the off-diagonals meaningful enough to matter.
   The tool's note that orthorhombic symmetry forces them to zero is a cheap
   correctness check in either mode. To be measured by the SW40 refinement
   study ("First refinement client", above), `refit_aniso` on against off.
2. **Refit cadence for k_sol and b_sol.** Never, every N steps, or on a
   schedule tied to the sampler's noise level. To be measured by the SW40
   refinement study at never, every 50 and every 10 iterations; a guidance
   run, where the noise level matters too, remains to be measured after it.
3. **Amplitudes versus intensities.** The tool, the R-factor and the
   sampleworks Bragg term are all on amplitudes. If a guidance loss on
   intensities is wanted later, the closed-form bin scale changes
   (least squares on I is still linear in k² per bin) and R stays on |F|.
4. **Weighting.** Unweighted now; 1/σ² when a target supplies σ. The
   observation reader, not this module, decides whether σ is available.
5. **Where French-Wilson lives for sampleworks.** `rs.algorithms` or
   `gemmi.french_wilson`; the tool's cctbx path stays as the reference the
   two are checked against.
