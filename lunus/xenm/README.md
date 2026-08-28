# XENM - Elastic Network Model Refinement

This module provides tools for refining elastic network models (ENM) against diffuse X-ray scattering data from protein crystals.

## Overview

The xenm package implements Anisotropic Network Model (ANM) calculations to predict diffuse scattering from protein dynamics, following the approach described in:

**Riccardi, D., Cui, Q., & Phillips, G. N. (2010).** "Application of elastic network models to proteins in the crystalline state." *Biophysical Journal*, 99(8), 2616-2625.

The code calculates:
- Hessian matrices for various ENM force field models
- Covariance matrices from pseudoinverse of Hessian
- Predicted diffuse X-ray scattering patterns
- Refinement of ENM parameters against experimental data

## Requirements

- Python 3.x
- cctbx (Computational Crystallography Toolbox)
  - iotbx.pdb for structure handling
  - cctbx.array_family for crystallographic calculations
- NumPy
- SciPy (for optimization in refine_enm.py)
- multiprocessing (standard library)

## Programs

### xenm.py

Main program for calculating ENM and predicting diffuse scattering.

**Key parameters:**
- `input.pdb=<file>` - Input PDB structure file
- `model=<ANM|ANMEXP|EXPCUT|ANM2|ANM6|LLM>` - ENM force field model
  - `ANM`: Standard 1/r elastic network
  - `ANMEXP`: Exponential falloff model
  - `EXPCUT`: Exponentially smoothed cutoff
  - `ANM2`, `ANM6`: Power law variants
  - `LLM`: Lattice-based local model
- `cutoff=<distance>` - Distance cutoff for interactions (default: 25 Å)
- `anmexp.lambda=<value>` - Decay length for ANMEXP model (default: 0.157)
- `kpoints=<N>` - Use k-point sampling (0=off, 1+=grid size)
- `nproc=<N>` - Number of processors for parallel diffuse calculation
- `output.hkl=<file>` - Write predicted diffuse intensities
- `output.V=<file>` - Write covariance matrix
- `output.pdb=<file>` - Write structure with anisotropic B-factors
- `atom.selection=<CA|heavy>` - Atoms to include in ENM (default: CA)

- `resolution=<value>` - Reciprocal-space grid cutoff in Angstroms (default: 1.6)

**Example** (with cctbx in the active environment, use plain `python`):
```bash
python xenm.py input.pdb=protein.pdb model=ANMEXP anmexp.lambda=0.16 \
    cutoff=25.0 resolution=4.0 nproc=8 output.hkl=diffuse.hkl output.V=covar.npy
```

### refine_enm.py

Refine ENM lambda parameter against experimental diffuse scattering data.

Uses Powell optimization to maximize correlation between predicted and experimental diffuse scattering. Automatically locates and calls `scripts/calcenm.sh` wrapper for each trial.

**Usage:**
The script should be run from a directory containing:
- `4wor.pdb` (or edit line 16 to specify your PDB file)
- Experimental diffuse data accessible to calcenm.sh

Edit the script to adjust:
- PDB filename (line 16)
- Initial lambda guess x0 (line 29, default: 0.16)
- Maximum iterations (line 30, default: 10000)

Then run:
```bash
python refine_enm.py
```

The script will automatically find `scripts/calcenm.sh` relative to its own location, so it works regardless of where you run it from.

### calc_pinv.py / calc_pinv_complex.py

Compute pseudoinverse of Hessian matrices using the null-space projection method to handle the six rigid-body modes.

**Usage:**
```bash
python calc_pinv.py input.X=hessian.npy output.Xtilde=pinv.npy
```

The complex version handles complex-valued matrices for k-point calculations.

### expand_V_to_all_atoms.py

Expand covariance matrix from CA atoms to all atoms in the structure.

**Usage:**
```bash
python expand_V_to_all_atoms.py input.pdb=protein.pdb \
    input.V=covar_ca.txt output.V=covar_allatom.txt output.pdb=reordered.pdb
```

## Scripts

### calcenm.sh

Wrapper script that runs xenm.py and computes correlation with experimental data using the lunus lattice tools (`lunus.hkl2lat`, `lunus.symlt`, `lunus.anisolt`, `lunus.corrlt`). It adds `<lunus_root>/c/bin` to `PATH` automatically and defaults to `resolution=4.0` (override with the `RES` environment variable).

**Usage:**
```bash
scripts/calcenm.sh protein.pdb 0.16
```

Where:
- First argument: PDB file
- Second argument: lambda value for ANMEXP model

## Examples

See `examples/4wor/` for a test case using staphylococcal nuclease (PDB 4WOR).

## Models

**ANM (Standard):** Spring constant k = 1/r for all atom pairs within cutoff

**ANMEXP (Exponential):** k = (1/r) * exp(-r/2λ), where λ controls decay

**EXPCUT (Smooth cutoff):** k = (1/r) * sqrt(exp(-r+thresh/λ) / (1 + exp(-r+thresh/λ)))

**ANM2, ANM6:** Power law variants with k = 1/r² or k = 1/r⁶

**LLM (Lattice Local Model):** Distance-dependent correlation without explicit dynamics

## Output Files

- **HKL files:** Predicted diffuse intensities on reciprocal lattice grid
- **NPY files:** NumPy arrays (Hessian, covariance, pseudoinverse)
- **PDB files:** Structures with calculated anisotropic displacement parameters

## Notes

- Periodic boundary conditions (PBC) are enabled by default
- K-point sampling is recommended for crystalline calculations
- The code handles alternative conformations by keeping only the 'A' conformer
- Multiprocessing is used for the diffuse scattering calculation
- Memory usage scales as O(N²) where N is the number of selected atoms

## References

Riccardi, D., Cui, Q., & Phillips, G. N. (2010). Application of elastic network models to proteins in the crystalline state. *Biophysical Journal*, 99(8), 2616-2625.

Wall, M. E. (2009). Methods and software for diffuse X-ray scattering from protein crystals. *Methods in Molecular Biology*, 544, 269-279.

Wall, M. E., Wolff, A. M., & Fraser, J. S. (2018). Bringing diffuse X-ray scattering into focus. *Current Opinion in Structural Biology*, 50, 109-116. https://doi.org/10.1016/j.sbi.2018.01.009 — this code was first used in this work.

## License

See LICENSE file in the lunus repository root.

## Author

Michael Wall  
Los Alamos National Laboratory
