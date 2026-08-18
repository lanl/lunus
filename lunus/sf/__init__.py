"""Differentiable structure factor / diffuse intensity calculator.

A PyTorch implementation of a gemmi-like engine: splat atomic Gaussian
density onto a real-space grid -> symmetrize -> FFT -> extract F(hkl),
differentiable with respect to atomic fractional coordinates.

Gemmi-like rather than a clone: the density cutoff is tapered smoothly rather
than truncated hard, so that gradients are well behaved. See docs/design.md.

Import it as an ordinary subpackage; the repository root needs to be on
sys.path, which a cctbx module build arranges for you:

    from lunus.sf.kernel_torch import build_element_kernels_torch
    from lunus.sf.structure_factor_torch import structure_factors_one_config

Only xtraj.py and the cctbx-based comparison tools in tools/ require cctbx.
Everything else needs no more than numpy and torch, and lunus/__init__.py
deliberately keeps the lunus_ext Boost extension optional so that stays true.

The public API is available directly from the package:

    from lunus.sf import structure_factors_one_config, mean_and_diffuse

Names are resolved LAZILY (PEP 562), so `import lunus.sf` itself imports
nothing and costs nothing -- the multi-second torch import happens on first
use of a name that needs it, and never at all for someone who only wanted
lunus.md. Submodules remain importable directly if you prefer
(`from lunus.sf.kernel_torch import ...`).

See README.md for the layout, docs/design.md for the derivation and
assumptions, and docs/performance.md for benchmarks and the torch.compile /
MPI notes.
"""

# name -> submodule it lives in. Kept explicit rather than star-importing so
# that `dir(lunus.sf)` and tab-completion work without importing anything,
# and so the public surface is a deliberate list rather than whatever the
# submodules happen to define.
_EXPORTS = {
    # cell geometry and grid sizing (numpy only)
    "orth_matrix": "cell_utils",
    "reciprocal_inv_d2": "cell_utils",
    "next_fft_friendly": "cell_utils",
    "grid_shape_for_resolution": "cell_utils",
    # scattering coefficients (needs gemmi or cctbx)
    "it92_coefficients": "elements",
    "IT92_COEFFS": "elements",
    # density kernel (numpy only)
    "precalculate_density_iso": "kernel",
    "precalculate_density_iso_batch": "kernel",
    "density_at": "kernel",
    "cutoff_radius": "kernel",
    "cutoff_radius_batch": "kernel",
    "calculate_sf": "kernel",
    "build_element_kernels": "kernel",
    "build_atom_kernels": "kernel",
    # torch kernels
    "build_element_kernels_torch": "kernel_torch",
    "build_atom_kernels_torch": "kernel_torch",
    # the differentiable splat
    "splat_density": "density_torch",
    "precompute_element_offsets": "density_torch",
    "taper_factor": "density_torch",
    "warmup_compile": "density_torch",
    # symmetry
    "build_grid_ops": "symmetry_torch",
    "build_grid_ops_from_cctbx": "symmetry_torch",
    "adjust_grid_for_symmetry": "symmetry_torch",
    "symmetry_grid_constraints": "symmetry_torch",
    "symmetrize_sum": "symmetry_torch",
    "supercell_factors": "symmetry_torch",
    "fold_supercell": "symmetry_torch",
    # structure factors and the ensemble reduction
    "compute_fcalc": "structure_factor_torch",
    "structure_factors_one_config": "structure_factor_torch",
    "structure_factors_batch": "structure_factor_torch",
    "mean_and_diffuse": "structure_factor_torch",
    # anisotropic component (isotropic-background removal)
    "shell_bins": "aniso",
    "spline_basis": "aniso",
    "subtract_isotropic": "aniso",
    "IsotropicBackground": "aniso_torch",
    # bulk solvent
    "SolventModel": "solvent_torch",
    "SolventMaskWarning": "solvent_torch",
    "solvent_mask": "solvent_torch",
    "calibrate_cutoff": "solvent_torch",
    "mask_occupancy": "solvent_torch",
    "shell_voxels": "solvent_torch",
    "f_solvent": "solvent_torch",
    "f_total": "solvent_torch",
    # MPI training step
    "training_step": "training_mpi",
    "local_forward": "training_mpi",
    "allreduce_sums": "training_mpi",
    "compute_upstream_gradients": "training_mpi",
    "seed_local_backward": "training_mpi",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Import the owning submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))

    import importlib

    module = importlib.import_module("." + module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache; __getattr__ won't run again
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
