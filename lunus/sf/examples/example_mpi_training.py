"""
Runnable example of training_mpi.py's training_step, distributing toy
configurations across MPI ranks. Run with:

    mpirun -n 4 python example_mpi_training.py

Each rank gets a slice of the total configurations and optimizes its own
local atoms; the loss (and every rank's view of it) is identical since
it's built from Allreduce'd aggregate data. Swap the toy system/loss for
your real trajectory frames and target diffuse data.
"""

import torch
from mpi4py import MPI

from lunus.sf.elements import IT92_COEFFS
from lunus.sf.cell_utils import orth_matrix, grid_shape_for_resolution
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.training_mpi import training_step

DTYPE = torch.float32
DEVICE = "cpu"  # switch to "cuda" per-rank via the gpu_wrapper.sh pattern for real runs


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Avoid CPU oversubscription: by default PyTorch tries to use ALL
    # available cores for its own internal threading, per process. With
    # `size` MPI ranks each doing that simultaneously, you get severe
    # contention -- often much SLOWER than a single process, especially
    # on CPU-limited/shared nodes. One thread per rank is a reasonable
    # starting point when ranks roughly match the core count; tune if
    # you have cores to spare per rank.
    if DEVICE == "cpu":
        torch.set_num_threads(1)

    torch.manual_seed(0)  # same initial atoms on every rank before we split configs

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

    n_atoms = 100
    n_total_configs = 16  # total across ALL ranks
    element_choices = torch.randint(0, len(elements_present), (n_atoms,))
    atom_A = elem_A[element_choices]
    atom_lam = elem_lam[element_choices]
    atom_radius_ang = elem_radius_ang[element_choices]
    occ = torch.ones(n_atoms, dtype=DTYPE, device=DEVICE)

    hh, kk, ll = torch.meshgrid(
        torch.arange(-3, 4), torch.arange(-3, 4), torch.arange(-3, 4), indexing="ij"
    )
    hkl = torch.stack([hh, kk, ll], dim=-1).reshape(-1, 3)
    hkl = hkl[(hkl.abs().sum(dim=-1) > 0)]

    mean_frac = torch.rand(n_atoms, 3, device=DEVICE)

    # split configs across ranks as evenly as possible
    configs_per_rank = [n_total_configs // size + (1 if r < n_total_configs % size else 0) for r in range(size)]
    my_n_configs = configs_per_rank[rank]

    torch.manual_seed(rank + 1)  # distinct perturbations per rank
    frac_coords_list = [
        (mean_frac + 0.01 * torch.randn(n_atoms, 3)).clone().detach().requires_grad_(True)
        for _ in range(my_n_configs)
    ]
    element_idx_list = [element_choices for _ in range(my_n_configs)]
    occ_list = [occ for _ in range(my_n_configs)]

    # toy target: match a flat diffuse pattern, ignore Bragg for this example
    target_diffuse = None  # set on first forward pass, once we know the reflection count

    def loss_fn(mean_F, diffuse):
        nonlocal target_diffuse
        if target_diffuse is None:
            target_diffuse = diffuse.detach().mean() * torch.ones_like(diffuse)
        return torch.nn.functional.mse_loss(diffuse, target_diffuse)

    optimizer = torch.optim.Adam(frac_coords_list, lr=1e-3)

    for step in range(5):
        loss_value = training_step(
            comm, frac_coords_list, element_idx_list, occ_list,
            atom_A, atom_lam, elem_offsets, atom_radius_ang,
            grid_shape, M, cell_volume, hkl, taper_width,
            n_total_configs, loss_fn, optimizer,
        )
        if rank == 0:
            print(f"step {step}: loss = {loss_value:.6f}")

    grad_norms = [fc.grad.norm().item() for fc in frac_coords_list]
    print(f"rank {rank}: handled {my_n_configs} configs, final grad norms = {grad_norms}")


if __name__ == "__main__":
    main()
