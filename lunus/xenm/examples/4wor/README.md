# Example: 4WOR - Staphylococcal Nuclease

This example demonstrates xenm calculations using the staphylococcal nuclease structure (PDB code: 4WOR).

## Files

- `4wor.pdb` - Protein structure (symlink to lunus examples directory)
- `snc_exafel_mean_mirror_sym_aniso_cull.lat.gz` - Experimental diffuse scattering data in
  lunus `.lat` format (gzipped). This serves as both the `hkl2lat` template and the
  correlation target during refinement. `calcenm.sh` decompresses it automatically.

## Structure Information

**PDB Code:** 4WOR  
**Title:** Staphylococcal nuclease (thermonuclease) in complex with Ca2+ and pdTp, room temperature  
**Crystal System:** Tetragonal  
**Space Group:** P41  
**Unit Cell:** a=48.499 Å, b=48.499 Å, c=63.430 Å, α=β=γ=90° (from the CRYST1 record)

## Running the Example

> With cctbx installed in the active (conda) environment, invoke the scripts with
> plain `python`. `resolution` sets the reciprocal-space grid cutoff in Angstroms;
> the full experimental grid corresponds to ~1.6 Å, while 4.0 Å is a fast setting
> for testing.

### Basic ENM Calculation

```bash
cd examples/4wor
python ../../xenm.py input.pdb=4wor.pdb model=ANMEXP \
    anmexp.lambda=0.157 cutoff=10.0 resolution=4.0 output.hkl=test.hkl nproc=8
```

### With K-point Sampling

```bash
python ../../xenm.py input.pdb=4wor.pdb model=ANMEXP \
    anmexp.lambda=0.157 kpoints=1 resolution=4.0 nproc=8 output.hkl=test_kp.hkl
```

### Calculate Covariance Matrix

```bash
python ../../xenm.py input.pdb=4wor.pdb model=ANMEXP \
    anmexp.lambda=0.157 output.V=covariance.npy
```

### Generate PDB with Anisotropic B-factors

```bash
python ../../xenm.py input.pdb=4wor.pdb model=ANMEXP \
    anmexp.lambda=0.157 output.pdb=4wor_enm.pdb
```

### Single Correlation Evaluation

`calcenm.sh` runs one full cycle: compute diffuse scattering for a given lambda,
convert to a lattice, apply symmetry, subtract the anisotropic (radial) component,
and correlate against the experimental data. It prints `0.0 <anisotropic_correlation>`.

```bash
# Usage: calcenm.sh <pdb_file> <lambda_value>
../../scripts/calcenm.sh 4wor.pdb 0.16
# -> "0.0 0.361532"  at resolution=4.0
```

The resolution defaults to 4.0 Å (set in the script). Override it via the `RES`
environment variable, e.g. `RES=1.6 ../../scripts/calcenm.sh 4wor.pdb 0.16`.

### Refine Lambda Parameter

`refine_enm.py` drives a Powell optimization that repeatedly calls `calcenm.sh`
to maximize the anisotropic correlation. Run it from this directory:

```bash
python ../../refine_enm.py
```

## Expected Output

At `resolution=4.0`, one `calcenm.sh` evaluation takes roughly a minute (with
`nproc` set for your machine) and yields an anisotropic correlation near 0.36 for
lambda = 0.16. A finer grid (`RES=1.6`) increases both runtime and the reflection
count substantially.

## Notes

- With cctbx in the active environment, use `python` (not `cctbx.python`).
- K-point sampling (`kpoints=1`) is used for the crystalline correlation calculation.
- The lambda value of 0.157 is a reasonable starting point but can be optimized.
- Set `nproc` to the number of cores available for the parallel diffuse calculation.

## Preparing / Regenerating the Experimental Data

The experimental lattice `snc_exafel_mean_mirror_sym_aniso_cull.lat.gz` is included
in this directory. To regenerate a `.lat` from a raw `h k l I` file, build a
template lattice with `lunus.makelt` (same unit cell and resolution as the target
grid), then map the reflections in with `lunus.hkl2lat`:

```bash
CELL=48.499,48.499,63.430,90.0,90.0,90.0
lunus.makelt template.lat "$CELL" 1.6
lunus.hkl2lat experimental.hkl experimental.lat template.lat
```

The lunus tools (`lunus.makelt`, `lunus.hkl2lat`, `lunus.symlt`, `lunus.anisolt`,
`lunus.corrlt`) live in `<lunus_root>/c/bin`; `calcenm.sh` adds that directory to
`PATH` automatically.

## References

This example uses data from:
- Experimental diffuse scattering from XFEL experiments on staphylococcal nuclease
- Structure: PDB 4WOR (staphylococcal nuclease in complex with Ca2+ and pdTp)

The code was first used in:

Wall, M. E., Wolff, A. M., & Fraser, J. S. (2018). Bringing diffuse X-ray scattering into focus. *Current Opinion in Structural Biology*, 50, 109-116. https://doi.org/10.1016/j.sbi.2018.01.009
