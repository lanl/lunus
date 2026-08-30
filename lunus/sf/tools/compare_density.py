"""
Direct comparison of the real-space electron density grids saved by
xtraj.py's gemmi and torch engines (density_gemmi.npy / density_torch.npy),
sharper than comparing F(hkl) because it isolates whether a discrepancy
is in atom placement/splatting versus the FFT/extraction step.

Usage: python compare_density.py [gemmi_path] [torch_path]
Defaults to density_gemmi.npy / density_torch.npy in the current directory.
"""

import sys
import numpy as np


def compare_densities(path_a, path_b, label_a="gemmi", label_b="torch"):
    a = np.load(path_a)
    b = np.load(path_b)

    print(f"{label_a}: shape={a.shape}, dtype={a.dtype}")
    print(f"{label_b}: shape={b.shape}, dtype={b.dtype}")

    if a.shape != b.shape:
        print("SHAPES DIFFER -- cannot compare voxel-by-voxel. Check grid sizing first")
        print("(this would mean the earlier grid-matching fix didn't actually take effect "
              "for this run -- re-check the printed grid dims in both engine logs).")
        return

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)

    print(f"\n{label_a}: sum={a64.sum():.4f}, max={a64.max():.4f}, min={a64.min():.4f}")
    print(f"{label_b}: sum={b64.sum():.4f}, max={b64.max():.4f}, min={b64.min():.4f}")
    print(f"sum ratio ({label_a}/{label_b}) = {a64.sum() / b64.sum():.6f}  "
          f"(if this is close to 1.0, total integrated density/electron count roughly agrees)")

    diff = a64 - b64
    print(f"\nmax abs diff = {np.max(np.abs(diff)):.6f}")
    print(f"RMS diff     = {np.sqrt(np.mean(diff**2)):.6f}")
    corr = np.corrcoef(a64.ravel(), b64.ravel())[0, 1]
    print(f"Pearson correlation (whole grid) = {corr:.8f}  "
          f"(close to 1.0 = same shape/placement, just possibly different scale/precision; "
          f"far from 1.0 = genuinely different atom placement)")

    idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    print(f"\nlargest abs diff at voxel {idx}: {label_a}={a64[idx]:.4f}, {label_b}={b64[idx]:.4f}")

    thresh = 1e-6
    mask_a = np.abs(a64) > thresh
    mask_b = np.abs(b64) > thresh
    na, nb = mask_a.sum(), mask_b.sum()
    overlap = (mask_a & mask_b).sum()
    only_a = (mask_a & ~mask_b).sum()
    only_b = (mask_b & ~mask_a).sum()
    print(f"\nvoxels with |density| > {thresh}:")
    print(f"  {label_a} only: {only_a}, {label_b} only: {only_b}, both: {overlap}")
    print(f"  ({label_a} total nonzero: {na}, {label_b} total nonzero: {nb})")
    if only_a > 0.01 * na or only_b > 0.01 * nb:
        print("  -> a meaningful fraction of nonzero voxels don't overlap between the two "
              "grids -- suggests atoms are landing in different places (or a different set "
              "of atoms is contributing), not just a scale/precision difference.")

    # correlation restricted to voxels where at least one grid has meaningful density,
    # to avoid the huge shared "both near zero" background dominating the statistic
    interesting = mask_a | mask_b
    if interesting.sum() > 1:
        corr_interesting = np.corrcoef(a64[interesting], b64[interesting])[0, 1]
        print(f"\nPearson correlation (voxels with meaningful density in either grid) = "
              f"{corr_interesting:.8f}")


if __name__ == "__main__":
    path_a = sys.argv[1] if len(sys.argv) > 1 else "density_gemmi.npy"
    path_b = sys.argv[2] if len(sys.argv) > 2 else "density_torch.npy"
    compare_densities(path_a, path_b)
