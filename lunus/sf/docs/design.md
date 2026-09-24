# Design: what matches gemmi, and what deliberately does not

`lunus.sf` implements a gemmi-like engine — splat atomic Gaussian density onto
a real-space grid, symmetrize, FFT, extract F(hkl) — in PyTorch,
differentiable with respect to atomic fractional coordinates. Gemmi is the
reference for what agreement is measured against, and the places where this
deliberately differs from it are recorded below.

## Provenance

This is an independent implementation, written from the published
mathematics and the published parameterization. It contains no code taken
from gemmi. Where it agrees with gemmi it does so because both compute the
same well-defined quantity, and that agreement is asserted by tests rather
than obtained by construction.

Concretely:

- **The density kernel** is the closed-form Fourier transform of the
  Cromer-Mann sum. `kernel.py`'s module docstring carries the derivation from
  first principles, and `tests/test_density_kernel.py` checks the result
  against direct numerical integration of *f(s)*, against the normalization
  ∫ρ(r)d³r = f(0) = Z, and against the requirement that total charge not
  depend on B.
- **The scattering coefficients** are not stored here at all. `elements.py`
  reads them at run time from gemmi or cctbx — ordinary use of a library's
  API. `tests/test_scattering_table.py` requires those two independent
  implementations to agree, and separately checks the coefficients against
  physics via f(0) = Z.

gemmi is used as a **dependency and as a test oracle**, which creates no
derivation: `tests/test_symmetry.py` compares against gemmi's own
`symmetrize_sum()`, and `examples/compare_gemmi/` runs both engines
end to end.

## Matched by construction

- the blur/unblur bookkeeping
- the `ifftn` → volume-scale → conjugate structure-factor convention
- the IT92 parameterization, read from a shared source

## Deliberately not identical

**The cutoff radius.** `kernel.cutoff_radius` finds the largest r at which
|ρ(r)| ≥ cutoff by a direct radial scan. The only requirement is that it not
be too *small*: density beyond it is discarded, so an under-estimate loses
real charge, while an over-estimate merely evaluates extra points whose
density is below the cutoff by construction. Structure factors are unchanged
by any radius at or above the true one, so this need not agree exactly with
the radius another implementation picks.

That last sentence holds in the limit and is misleading in practice: it is
conditional on the cutoff being tight enough that the discarded density really
is negligible. At `xtraj`'s default of 0.01 e/Å³ it is not, and the two
implementations' differing radius conventions are then the **largest** source
of disagreement with gemmi — larger than the taper below. See "Of those two
terms, the cutoff dominates".

**The taper.** A hard cutoff is unusable for gradients: atoms crossing the
boundary make the density discontinuous, so the finite-difference derivative
swings erratically. A smooth taper over the outermost 0.1 Å fixes that at the
cost of a small, deliberate deviation from an untruncated reference.
`examples/demo_taper_gradient.py` shows the contrast directly, and
`tests/test_taper.py` asserts that the tapered gradient converges monotonically
as eps shrinks. See `density_torch.py`'s module docstring for the full argument.

## Assumptions carried over from xtraj.py

- **B-factors.** One uniform B across the structure (matching
  `xrs.convert_to_isotropic()` + `xrs.set_b_iso(...)`), per-atom isotropic B
  (`use_top_bfacs=True`), or — since the anisotropic work — per-atom
  **anisotropic ADPs**, through the tensor-valued kernel derived at the end of
  `kernel.py`. The anisotropic path is opt-in and isotropic models never touch
  it; see "Anisotropic ADPs" below.
- **Only atomic coordinates are differentiated.** B-factor and blur are fixed
  hyperparameters, which is what lets the per-atom kernel/radius/offset setup
  happen once and be reused for every frame. If that changes, the B-dependence
  in `kernel.py` has to move into the torch graph instead of being precomputed
  in NumPy.

## Per-atom B-factors

B may vary per atom, still assumed FIXED across configurations.

- `kernel.py` — `precalculate_density_iso_batch` and `cutoff_radius_batch`
  (vectorized, chunked for memory) compute per-atom (A, lam, radius) instead of
  one shared set per element; `build_atom_kernels` ties them together. The
  batched (A, lam) formula matches the scalar version to machine precision;
  radius differs by ~0.2–0.5% from a coarser search grid, the same
  conservative-not-exact tolerance `cutoff_radius` has always had.
- `kernel_torch.py` — `build_atom_kernels_torch` returns per-atom
  `atom_A`/`atom_lam`/`atom_radius_ang`. The offset candidate list stays
  per-ELEMENT, sized using the MAXIMUM radius among that element's atoms so it
  safely contains every atom of that type.
- `density_torch.py` — `splat_density` takes per-atom values gathered by atom
  index within each element-grouped batch. For the uniform-B path
  (`build_element_kernels_torch`), callers broadcast with
  `atom_A = elem_A[element_idx]` first.
- `xtraj.py` — `use_top_bfacs=True` extracts B via the same
  `extract_u_iso_or_u_equiv() * 8*pi^2` conversion the gemmi branch uses. A
  per-frame assertion checks the atom count is stable, since the per-atom
  B/element arrays are built once at setup.

### Why per-atom B does not match an untruncated reference to better than ~4%

`tests/test_per_atom_b.py` measures ~4.1e-02 against an analytic reference, and
that is expected rather than a defect. A higher-B atom is more spread out, so
the same fixed density-cutoff THRESHOLD truncates a larger fraction of its
charge — measured directly, a B=85 carbon loses ~9.3% of its charge to the
cutoff before any taper, against ~3.1% for the same element at B=12. This is a
property of the fixed-threshold cutoff itself, which gemmi shares, not of
per-atom B or of this implementation.

> **Caveat, worth fixing.** What would actually establish correctness here is a
> grid-refinement convergence check — F(000) stable to <0.01% across a 6x
> refinement, confirming the splat self-consistently implements the intended
> (truncated + tapered) physics. That check was performed manually during
> development but is **not implemented in the test file**, which currently only
> guards the characterized ~4.1e-02 number. Implementing it would be a real
> improvement.

## Space-group handling: symmetry expansion of the DENSITY

Gemmi expands symmetry on the **density grid**, not on the atoms:
`put_model_density_on_grid()` is `initialize_grid()` +
`add_model_density_to_grid(model)` + `symmetrize_sum()`. `symmetry_torch.py`
does the same.

This took three iterations, so the history is worth recording:

1. The first version called `xrs_sel.expand_to_p1()` and re-splatted every
   symmetry copy of every atom — correct in spirit but slow, since splatting
   costs `O(n_atoms * box_volume)` Gaussian evaluations per copy.
2. The second removed symmetry handling altogether, reasoning that `xrs_sel`
   already held a complete atom set. That made the torch engine disagree with
   gemmi by a factor equal to the space-group order (measured: 4.06x on the
   P4₃ case, whole-grid density correlation only 0.496).
3. **Neither was right.** Expanding the *density* gets the same answer as
   expanding the atoms at a fraction of the cost: `O(n_grid)` integer index
   arithmetic per operation, and for signed-permutation operations — which is
   every operation of most space groups — it reduces to axis permutes, flips
   and rolls with no index tensors materialized at all.

`build_grid_ops()` / `build_grid_ops_from_cctbx()` turn a space group into
validated integer **grid** operations, and `symmetrize_sum()` applies them
differentiably. The convention, `new[p] = Σ_g data[g(p)]` over all ops
*including* the identity, was pinned by probing gemmi directly rather than by
reading its source. Gradients flow back through the symmetrization, so an atom
receives the gradient contributions of all of its symmetry images.

`adjust_grid_for_symmetry()` handles the grid-sizing constraints the operations
impose (a 4-fold in the ab plane needs `Nu == Nv`; a 3/4 screw translation needs
`Nw` divisible by 4), which `grid_shape_for_resolution()` does not guarantee on
its own since it rounds each axis independently.

### One physical note

This symmetrizes the density of a **single configuration**, which is what gemmi
does and therefore what parity requires. If the atoms fed in are a genuine
asymmetric unit, that is simply correct crystallography. If instead they already
span the full cell, symmetrizing folds the structure onto itself and suppresses
the per-configuration deviations from crystallographic symmetry that *are* the
diffuse signal. Which case you are in is a property of the input model, not of
this code.

The example in `examples/compare_gemmi/` is the second kind: an MD box whose
crystal is PDB 7FPV, P2₁2₁2₁ with cell 34.196, 45.558, 99.044. The symmetry is
real and approximately obeyed; it is passed on the command line only because
the topology's CRYST1 records the P1 simulation box. The box is an exact 2×2×2
supercell of that cell, so it already contains every symmetry copy — which is
why `expand_symmetry` defaults to False. The figures in the next section
predate this correction and were taken with P4₃ / 88.451, 88.451, 39.823; see
docs/performance.md.

## Measured agreement with gemmi

System: `examples/compare_gemmi/top_bfac.pdb.gz`, 135,834 atoms, per-residue B
on [10, 100], cell 88.451, 88.451, 39.823, space group P4₃ (superseded; see
docs/performance.md), `d_min` 0.9, grid
300 × 300 × 144, cutoff 0.01.

**Read the first row carefully**: it is torch NOT expanding while gemmi does,
i.e. a genuine mismatch, and not a description of `expand_symmetry=False`.
With that default both engines skip the expansion and they agree — the
corrected-cell parity run gives correlation 0.999919.

| | density correlation | density sum ratio | Icalc R-factor |
|---|---|---|---|
| torch not expanding, gemmi expanding | 0.496 | 4.06 | — |
| symmetrize the density | 0.99977 | 1.016887 | 0.0092 |
| …and taper 0.1 → 0.001 Å | 0.99990 | 1.0073 | 0.0044 |

Re-measured 2026-08-11 on the shipped configuration (which is the middle row,
taper 0.1 Å): density sums 77647450 (gemmi) / 76357936 (torch) = ratio
**1.016887**, Icalc correlation 0.999989, R-factor 0.0077 over all 227,566
reflections. Those came from the superseded cell; on the corrected 7FPV
configuration the same comparison gives **correlation 0.999919, R-factor
0.004997** over 115,348 reflections, with the same resolution dependence —
0.9999 beyond 1.55 Å falling to 0.9125 in the 0.900–0.933 Å shell.

The residual disagreement is the taper and the cutoff-radius convention, not
symmetry. It is monotonic in resolution — Icalc correlation 1.0000 at low
resolution falling to 0.9099 in the 0.900–0.932 Å shell — which is the
signature of a real-space truncation difference, and it halves when the taper is
narrowed. Reproduce with `examples/compare_gemmi/run_xtraj.sh`.

### Of those two terms, the cutoff dominates

Measured 2026-09-20 on a 7FPV MD trajectory at `d_min` 1.2 (supercell
68.392 × 91.116 × 198.088, 1,496,008 reflections, ten equal-count resolution
shells), comparing `engine=torch` against `engine=gemmi` through
`tools/compare_icalc_mtz.py`. R-factor on Icalc per shell:

| d_mid (Å) | A: taper 0.1, cutoff 0.01 | B: taper 0.001, cutoff 0.01 | C: taper 0.1, cutoff 1e-3 torch / 1e-4 gemmi | D: taper 0.1, cutoff 1e-4 both |
|---|---|---|---|---|
| 1.222 | 0.1633 | 0.0624 | 0.0164 | **0.0048** |
| 1.268 | 0.1452 | 0.0578 | 0.0271 | **0.0038** |
| 1.323 | 0.1022 | 0.0425 | 0.0283 | **0.0027** |
| 1.388 | 0.0564 | 0.0250 | 0.0177 | **0.0022** |
| 1.468 | 0.0187 | 0.0100 | 0.0067 | **0.0020** |
| 1.571 | 0.0106 | 0.0041 | 0.0049 | **0.0016** |
| 1.711 | 0.0186 | 0.0070 | 0.0057 | **0.0012** |
| 1.922 | 0.0119 | 0.0052 | 0.0024 | **0.0010** |
| 2.319 | 0.0034 | 0.0014 | 0.0025 | **0.0007** |
| 3.793 | 0.0039 | 0.0016 | 0.0015 | **0.0004** |
| **mean** | **0.0534** | **0.0217** | **0.0113** | **0.0020** |

Three things this settles, none of which was obvious from the `d_min` 0.9
parity run above.

**The taper acts as a resolution-independent factor; the cutoff carries the
resolution dependence.** Narrowing the taper (A → B) improves every shell by
**2.41 ± 0.22×**, including the low-resolution shells where R was already
0.003, and leaves the monotonic shape intact. Tightening the cutoff removes the
shape itself. So the steep high-resolution disagreement that motivates this
section is the *cutoff*, and narrowing the taper treats a symptom the loose
cutoff amplifies.

**Matched tight cutoffs beat a narrowed taper by 13× at the resolution limit**
(D 0.0048 against B 0.0624), at the default taper. Both knobs help, but they
are not comparable in size, and they are not independent: the taper acts over
the outermost 0.1 Å of the cutoff radius, where at 1e-4 the density is 100×
smaller than at 1e-2, so B's 2.41× should not be expected to transfer on top of
D.

**Mismatched cutoffs cancel, and the cancellation is legible in the shape.** C
sets the two engines to different cutoffs and looks good at the resolution
limit (0.0164), but it is the only configuration whose R is *not* monotonic —
it peaks at 1.323 Å with the outermost shell better than the two inside it.
Two densities truncated at different radii differ by a thin spherical shell,
whose transform oscillates in s rather than growing with it, and that hump is
the tell. Matching the cutoffs (D) is 3.4× better at the edge and 5.6× better
in the mean. **Do not tune the two engines' cutoffs independently to minimise
their disagreement**: agreement between two differently-truncated calculations
is not evidence that either is right.

Both engines take the cutoff from the same `gemmi_cutoff=` argument
(`xtraj.py` passes it to the torch kernel and to `calc.cutoff` alike), so D is
the configuration to run and A is only the default. The cost is the one
`6099334` measured: **1.76× on the torch splat** (2501 frames, `d_min` 1.8,
CUDA, 176 s → 310 s), 1.16–1.47× for gemmi's calculator on CPU, since the work
goes as the cutoff radius cubed. A Gaussian radius model — r ∝ √ln(ρ₀/c),
work ∝ r³ — reproduces that 1.76× to within 5% and predicts ~1.40× for 1e-3,
if a cheaper setting is wanted.

This is an independent confirmation of that commit's conclusion that "0.01 is a
reasonable default and a poor production setting", reached through engine
parity rather than through the solvent R-factor. `docs/solvent-design.md`'s
"What the density cutoff costs" reaches 1e-4 from the other direction.

### Settled against exact direct summation

The configurations above are engine-against-engine, which establishes
consistency and not accuracy. Both were therefore run against
`engine=cctbx cctbx_method=direct` — exact structure factors, no grid, no FFT,
no cutoff, no taper — on the same trajectory and reflection set. R-factor on
Icalc against that reference:

| d_mid (Å) | gemmi vs exact | torch vs exact | torch/gemmi | torch vs gemmi (D) |
|---|---|---|---|---|
| 1.222 | 0.0021 | 0.0050 | 2.4 | 0.0048 |
| 1.268 | 0.0014 | 0.0044 | 3.1 | 0.0038 |
| 1.323 | 0.0015 | 0.0034 | 2.3 | 0.0027 |
| 1.388 | 0.0011 | 0.0025 | 2.3 | 0.0022 |
| 1.468 | 0.0005 | 0.0020 | 4.0 | 0.0020 |
| 1.571 | 0.0004 | 0.0018 | 4.5 | 0.0016 |
| 1.711 | 0.0003 | 0.0013 | 4.3 | 0.0012 |
| 1.922 | 0.0002 | 0.0010 | 5.0 | 0.0010 |
| 2.319 | 0.0002 | 0.0008 | 4.0 | 0.0007 |
| 3.793 | 0.0002 | 0.0005 | 2.5 | 0.0004 |
| **overall** | **0.000564** | **0.001651** | **2.9** | **0.001519** |

correlation 0.999999 (gemmi), 0.999996 (torch vs exact) and 0.999996
(torch vs gemmi) overall. The last column is the same five frames (0–4) as
the reference runs; against the larger frame set of the table above it moves
by at most 0.0002 in any shell, so none of this depends on frame count.

**Both engines are accurate; gemmi is 2.9× the more accurate of the two.** The
ratio is stable across every shell (2.3–5.0), and both residuals span the same
10× from the resolution limit to low resolution — the same mechanism at
different magnitudes, which is what two truncated-and-gridded calculations
should look like.

**D's agreement was real, not cancellation.** If the two engines' errors were
independent, the torch-vs-gemmi R would be their quadrature sum. Observed
**0.001519 against a predicted 0.001745** — a ratio of 0.87, i.e. independent
to within a small common-mode term, which is expected since both truncate on
the same threshold. So the C-versus-D contrast above is exactly what it
appeared to be: C cancelled, D did not.

**A corollary worth keeping: gemmi is a slightly flattering reference.**
torch-vs-gemmi (0.001519) is 0.92× torch-vs-exact (0.001651), so benchmarking
the torch engine against gemmi understates its true error by ~8% — the
common-mode truncation the two share cancels in that comparison and does not
cancel against reality. Small here, but it is the reason the parity tables
above are a consistency check and this section is the accuracy one.

The practical reading: at `gemmi_cutoff=1e-4` the torch engine is within
**0.5% of exact at the resolution limit and 0.17% overall**, and gemmi within
0.2% and 0.06%. Nothing in this pipeline is limited by that.

## Anisotropic ADPs

Implemented: `precalculate_density_aniso[_batch]` in `kernel.py`,
`build_atom_kernels_aniso_torch`, and `_density_core_aniso` behind
`splat_density(atom_L6=..., aniso_mask=...)`. The derivation is in `kernel.py`
and the conventions are pinned in `tests/test_adp_aniso.py`.

**What it is worth**, which is why it was done: on 7FPV against deposited
amplitudes, `tools/fit_solvent_rfactor.py` reaches **R-work 0.1322, R-free
0.1528** with anisotropic ADPs against 0.1810 / 0.1947 with their isotropic
equivalents. mmtbx on the same model and data reaches 0.1298 / 0.1510, so
this closes a 0.049 gap to 0.0024. It was the largest single error in the
pipeline's agreement with experimental data.

**What it costs**: 2.37x on the splat for 7FPV as deposited, measured in situ
by `tools/bench_aniso_splat.py`. Note that is close to the all-anisotropic
worst case of 2.45x — only 54% of atoms carry ANISOU, but they are the
expensive ones (every non-hydrogen), **78% of the work** weighted by r³, which
is what candidate-voxel count scales as. Hydrogens are 46% of the atoms and
21.6% of the cost, so keeping them on the fast path buys ~0.08x rather than
halving the penalty. Proportion of atoms is not proportion of cost. A model
with **no** anisotropic ADPs pays nothing at all: `aniso_mask=None` runs none
of the code, asserted by `torch.equal` rather than a tolerance.

### Three things that were not obvious, and cost measurements to settle

**The shared eigenvectors are a precompute trick, not a hot-loop one.** All
five `Λ_i` are diagonal in `U`'s eigenbasis, which suggests rotating the
displacement into the principal axes once and reducing each Gaussian to a
three-term sum — fewest flops. That is the slower choice. It materializes
`(m,K,3)` intermediates where expanding `d·Λ_i·d` per Gaussian keeps everything
at `(m,K)`, and this core is memory-bound: measured on CUDA, expanded is
1.65-1.70x the isotropic core eager and the rotation is 2.61-2.70x. What the
shared eigenvectors actually buy is one eigendecomposition per atom at setup
instead of five 3x3 inversions.

**Compiling nearly erases the difference and does not change the answer.**
Compiled, the two forms are 2.81-2.95x and 2.99-3.11x — a ~5% margin where
eager showed 40%, because inductor fuses away most of the traffic the rotation
pays for. Expanded still wins, on speed and by 59 MB of peak memory. The
in-situ figure is 2.37x, so the isolated core did not badly overstate here.

**`u_star` is not `u_cart`.** A PDB stores anisotropic ADPs in the fractional
reciprocal basis; the kernel measures displacements in Cartesian angstroms.
They coincide only for a cubic cell with a = 1, so feeding `u_star` straight in
gives a correct answer on a toy fixture and a wrong one on every real crystal.
The conversion must also happen **before** `convert_to_isotropic()`, which
discards it. `tests/test_adp_aniso.py` pins the convention by splatting one
anisotropic atom and transforming it against the closed form — a check that a
transposed `U`, eigenvectors as rows, or the wrong basis all fail, and that the
isotropic-limit and total-charge checks all pass regardless.

**Not yet wired into `xtraj.py`** — the kernel, the splat and the R-factor tool
support it; the trajectory driver does not.
