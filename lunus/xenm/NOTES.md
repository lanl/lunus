# Development Notes

Summary of the work that brought `xenm` into the lunus repository. These are
the essential notes behind the initial commit; see the READMEs for usage.

The code was first used in Wall, Wolff & Fraser (2018), "Bringing diffuse X-ray
scattering into focus," *Current Opinion in Structural Biology* 50, 109-116
(https://doi.org/10.1016/j.sbi.2018.01.009).

## Integration and modernization

- Tidied the original ENM refinement code and integrated it under `lunus/xenm/`
  as a proper package (`__init__.py`, top-level `README.md`, example docs).
- Modernized Python 2 -> 3: `print`, integer division, `xrange` -> `range`,
  `np.complex` -> builtin `complex`.
- Fixed cctbx atom selection for the modern API. After the structure is
  expanded to P1 and its scatterers are reordered, selections are built by
  parsing scatterer labels (` CA `, backbone ` N `) into `flex.bool` arrays.
- Switched the parallel diffuse calculation from `multiprocessing.Pool` to
  `multiprocessing.pool.ThreadPool` so worker functions share module globals
  (the original code relied on fork-inherited globals).

## `calcenm.sh` workflow

- Runs with plain `python` (cctbx is expected in the active environment).
- Uses the `lunus.`-prefixed C lattice tools (`lunus.hkl2lat`, `lunus.symlt`,
  `lunus.anisolt`, `lunus.corrlt`) and adds `<lunus_root>/c/bin` to `PATH`
  automatically.
- `resolution` is exposed via the `RES` environment variable (default 4.0 for
  fast testing; ~1.6 matches the full experimental grid).
- Decompresses the experimental `.lat.gz` on demand.

## Experimental data

- The 4WOR example ships the experimental diffuse data as a gzipped lunus
  lattice, `snc_exafel_mean_mirror_sym_aniso_cull.lat.gz` (smaller than the raw
  hkl and directly usable as both the `hkl2lat` template and correlation
  target).
- Regenerate a `.lat` from raw `h k l I` with `lunus.makelt` (build a template
  at the target cell and resolution) followed by `lunus.hkl2lat`.

## 4WOR identity and unit cell

- PDB 4WOR is **staphylococcal nuclease** (thermonuclease) in complex with
  Ca2+ and thymidine-3',5'-diphosphate (pdTp) at room temperature -- not
  nitrogenase reductase.
- The authoritative unit cell is the CRYST1 record:
  **a = b = 48.499 A, c = 63.430 A, alpha = beta = gamma = 90 deg, space group P41.**
  The experimental diffuse data uses this same cell. All cell references (docs,
  the `anisolt` call in `calcenm.sh`, the `makelt` template) are aligned to it.

## Diffuse-scattering vectorization (~3x faster)

The per-reciprocal-point loops spent nearly all their time on the coupling
contraction `fr^H . dc . fr`, where `dc = exp(qsq * V) - 1` on the (nsel x nsel)
covariance matrix, evaluated at every grid point.

Key observation: `dc` depends only on the scalar `qsq`, and the grid has
~14-17x fewer distinct `qsq` values than points (h^2/k^2/l^2 degeneracy plus
the a = b cell symmetry). `calc_diffuse_vectorized` exploits this:

- Coarse-grained scattering factors `fr` are computed in large vectorized
  blocks, using a precomputed residue-grouping matrix `G` (`fr = f @ G^T`) that
  replaces the inner per-atom accumulation loop.
- Points are grouped by unique `qsq`; `dc` is computed once per group and
  applied to all its points with a single BLAS matrix multiply.
- The intra-residue correction is folded into the block loop to bound memory.

Numerically identical to the original loops (relative difference ~1e-9). The CA
path with `kpoints` 0 or 1 (used by `calcenm.sh` and the refinement) routes
through the fast path; the all-atom and multi-R `kpoints > 1` cases keep the
original loops.

**Memory note:** at 1.6 A the `fr` array is ~2.7 GB (npts x nsel complex). Fine
on a workstation, but larger cells would need the `qsq` grouping itself blocked.

## Verified

- End-to-end `calcenm.sh 4wor.pdb 0.16` gives an anisotropic correlation of
  **0.36105** at `resolution=4.0`.
- The diffuse calculation dropped from ~50 s to ~14 s at 4 A; also verified at
  1.6 A (307520 reflections).
