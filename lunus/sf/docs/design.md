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

**The taper.** A hard cutoff is unusable for gradients: atoms crossing the
boundary make the density discontinuous, so the finite-difference derivative
swings erratically. A smooth taper over the outermost 0.1 Å fixes that at the
cost of a small, deliberate deviation from an untruncated reference.
`examples/demo_taper_gradient.py` shows the contrast directly, and
`tests/test_taper.py` asserts that the tapered gradient converges monotonically
as eps shrinks. See `density_torch.py`'s module docstring for the full argument.

## Assumptions carried over from xtraj.py

- **Isotropic B-factors.** Either one uniform B across the structure (matching
  `xrs.convert_to_isotropic()` + `xrs.set_b_iso(...)`), or per-atom isotropic B
  (`use_top_bfacs=True`). **Anisotropic ADPs are not implemented** —
  the anisotropic case needs a different, tensor-valued kernel, derived the
  same way as the isotropic one in `kernel.py` but from the anisotropic
  Debye-Waller factor. A separate piece of work if ever needed.
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

## Known limitation

Anisotropic ADPs are still not implemented; see "Assumptions" above.
