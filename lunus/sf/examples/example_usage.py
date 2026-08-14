"""
Illustrates the intended workflow for matching diffuse intensities:
several configurations -> per-config F(hkl) -> <F> and diffuse=<|F|^2>-|<F>|^2
-> scalar loss against target diffuse data -> backward() populates
gradients on every atom's fractional coordinates, in every configuration,
in one pass.

Run where PyTorch is installed. Uses a tiny synthetic system (a handful
of random atoms, a few configurations) purely to demonstrate the shape
of the computation -- swap in real MD frames, elements, and target
diffuse data for actual use.
"""

import torch

from lunus.sf.elements import IT92_COEFFS
from lunus.sf.cell_utils import orth_matrix, grid_shape_for_resolution
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.structure_factor_torch import structure_factors_batch, mean_and_diffuse

DTYPE = torch.float32
DEVICE = "cpu"  # switch to "cuda" for real-scale runs


def main():
    torch.manual_seed(0)

    # --- toy system setup ---
    a, b, c = 40.0, 40.0, 40.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    M_np = orth_matrix(a, b, c, alpha, beta, gamma)
    M = torch.tensor(M_np, dtype=DTYPE, device=DEVICE)
    cell_volume = abs(torch.det(M).item())

    d_min, rate = 2.5, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)

    b_iso = 20.0 * d_min * d_min
    elements_present = ["C", "N", "O", "H"]
    elem_A, elem_lam, elem_offsets, elem_radius_ang, taper_width, elem_to_idx = build_element_kernels_torch(
        elements_present, IT92_COEFFS, b_iso, blur=0.0,
        grid_shape=grid_shape, orth_matrix_np=M_np,
        cutoff=1e-5, device=DEVICE, dtype=DTYPE,
    )

    n_atoms = 200
    n_configs = 8
    element_choices = torch.randint(0, len(elements_present), (n_atoms,))
    atom_A = elem_A[element_choices]
    atom_lam = elem_lam[element_choices]
    atom_radius_ang = elem_radius_ang[element_choices]
    occ = torch.ones(n_atoms, dtype=DTYPE, device=DEVICE)

    # a small resolution shell of test reflections
    hh, kk, ll = torch.meshgrid(
        torch.arange(-3, 4), torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij"
    )
    hkl = torch.stack([hh, kk, ll], dim=-1).reshape(-1, 3)
    hkl = hkl[(hkl.abs().sum(dim=-1) > 0)]  # drop (0,0,0)

    # --- the ensemble: one (n_configs, n_atoms, 3) tensor ---
    # A single leaf tensor rather than a list, so one backward() populates
    # .grad for every member at once -- the shape a guided step wants, where
    # all N members are updated together.
    mean_frac = torch.rand(n_atoms, 3, device=DEVICE)
    frac_coords = torch.stack([
        mean_frac + 0.01 * torch.randn(n_atoms, 3, device=DEVICE)
        for _ in range(n_configs)
    ]).clone().detach().requires_grad_(True)

    # --- forward: F(hkl) for every configuration in one call ---
    # use_checkpoint=True recomputes each member's splat during backward
    # instead of retaining it. Memory then stays flat as n_configs grows,
    # which is what makes a large ensemble feasible at production grid sizes;
    # the cost is roughly 2.4x the time, and gradients are bit-identical.
    # See docs/design.md.
    F_configs = structure_factors_batch(
        frac_coords, element_choices, occ,
        atom_A, atom_lam, elem_offsets, atom_radius_ang,
        grid_shape, M, cell_volume, hkl, taper_width, blur=0.0,
        use_checkpoint=False,
    )  # (n_configs, n_refl)

    mean_F, diffuse = mean_and_diffuse(F_configs)

    # --- toy loss: match a (here, arbitrary placeholder) target diffuse pattern ---
    target_diffuse = torch.ones_like(diffuse) * diffuse.mean().item()
    loss = torch.nn.functional.mse_loss(diffuse, target_diffuse)

    loss.backward()

    print(f"n_atoms={n_atoms}, n_configs={n_configs}, n_reflections={hkl.shape[0]}")
    print(f"loss = {loss.item():.6f}")
    # frac_coords.grad is (n_configs, n_atoms, 3): every member got its own
    # gradient from the one backward pass.
    for i, gnorm in enumerate(frac_coords.grad.reshape(n_configs, -1).norm(dim=1)):
        print(f"  config {i}: grad norm over all atom coordinates = {gnorm.item():.4f}")


if __name__ == "__main__":
    main()
