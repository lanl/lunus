"""
MPI-based training step for matching diffuse (and/or Bragg) intensities
across many configurations distributed over MPI ranks, without needing
torch.distributed or per-atom gradient synchronization.

Validated in validate_gradient_seeding.py: seeding each rank's local
backward with a gradient computed from a tiny reduced-aggregate graph is
mathematically IDENTICAL to one combined backward pass over everything
at once (matched to 5.55e-17, i.e. machine precision, against both a
direct combined computation and finite differences).

Why no per-atom gradient Allreduce is needed: each configuration's atoms
are unique to that configuration -- nothing is shared/replicated across
ranks the way model weights are in ordinary distributed training. Once
every rank knows how the loss depends on the GLOBAL aggregate (mean_F,
mean_I -- small per-hkl arrays, not per-atom), each rank can complete
its own backward pass and optimizer step using only its own local
computational graph. The only communication needed is one small
Allreduce of those aggregate sums -- reusing the exact same MPI pattern
already used for xtraj.py's forward-only sig_fcalc/sig_icalc reduction.

Usage sketch (see example_mpi_training.py):

    for step in range(n_steps):
        optimizer.zero_grad()
        local_sum_F, local_sum_I = local_forward(my_configs, ...)
        global_sum_F, global_sum_I = allreduce_sums(comm, local_sum_F, local_sum_I)
        loss_value, grad_F, grad_I = compute_upstream_gradients(
            global_sum_F, global_sum_I, n_total_configs, loss_fn
        )
        seed_local_backward(local_sum_F, local_sum_I, grad_F, grad_I)
        optimizer.step()
"""

import numpy as np
import torch


def local_forward(frac_coords_list, element_idx_list, occ_list, atom_A, atom_lam,
                   elem_offsets, atom_radius_ang, grid_shape, orth_matrix,
                   cell_volume, hkl, taper_width, blur=0.0, max_atoms_per_batch=50_000,
                   grid_ops=None):
    """
    Computes F_c for every LOCAL configuration on this rank and returns
    their (still autograd-tracked) sums. Each element of frac_coords_list
    is one configuration's (n_atoms_c, 3) tensor with requires_grad=True.

    grid_ops (from symmetry_torch.build_grid_ops) symmetry-expands each
    configuration's density grid before its FFT, the same way the forward-only
    path in xtraj.py does. Gradients flow back through the symmetrization, so
    an atom in the asymmetric unit correctly receives the gradient
    contributions of all of its symmetry images. None/empty = no expansion.
    """
    from .structure_factor_torch import structure_factors_one_config

    local_sum_F = None
    local_sum_I = None
    for frac_coords, element_idx, occ in zip(frac_coords_list, element_idx_list, occ_list):
        F_c = structure_factors_one_config(
            frac_coords, element_idx, occ,
            atom_A, atom_lam, elem_offsets, atom_radius_ang,
            grid_shape, orth_matrix, cell_volume,
            hkl, taper_width, blur=blur, max_atoms_per_batch=max_atoms_per_batch,
            grid_ops=grid_ops,
        )
        I_c = F_c.abs() ** 2
        local_sum_F = F_c if local_sum_F is None else local_sum_F + F_c
        local_sum_I = I_c if local_sum_I is None else local_sum_I + I_c
    return local_sum_F, local_sum_I


def allreduce_sums(comm, local_sum_F: torch.Tensor, local_sum_I: torch.Tensor):
    """
    Detaches local_sum_F/local_sum_I and Allreduces them across MPI
    ranks -- identical in spirit to xtraj.py's existing sig_fcalc/
    sig_icalc reduction. Returns plain numpy arrays (every rank gets the
    same, globally-summed result). Complex tensors go to CPU as
    complex128, real ones as float64, for a clean, unambiguous MPI
    buffer type regardless of the compute dtype/device used upstream.
    """
    from mpi4py import MPI

    local_F_np = local_sum_F.detach().cpu().numpy().astype(np.complex128)
    local_I_np = local_sum_I.detach().cpu().numpy().astype(np.float64)

    global_F_np = np.zeros_like(local_F_np)
    global_I_np = np.zeros_like(local_I_np)
    comm.Allreduce(local_F_np, global_F_np, op=MPI.SUM)
    comm.Allreduce(local_I_np, global_I_np, op=MPI.SUM)
    return global_F_np, global_I_np


def compute_upstream_gradients(global_sum_F_np, global_sum_I_np, n_total_configs, loss_fn):
    """
    Builds a tiny, independent graph from the globally-reduced sums
    (every rank does this redundantly -- it's cheap, and avoids a
    broadcast since every rank already has identical Allreduce'd data),
    computes the scalar loss, and backpropagates through JUST this tiny
    graph to get the "upstream" gradients that seed each rank's real
    local backward.

    loss_fn(mean_F, diffuse) -> scalar real torch tensor. mean_F is
    complex, diffuse is real (mean_I - |mean_F|^2 is computed here so
    loss_fn only ever needs to combine Bragg/diffuse quantities into one
    number, per the stated requirement that the target is always scalar).
    """
    global_sum_F = torch.tensor(global_sum_F_np, dtype=torch.complex128, requires_grad=True)
    global_sum_I = torch.tensor(global_sum_I_np, dtype=torch.float64, requires_grad=True)

    mean_F = global_sum_F / n_total_configs
    mean_I = global_sum_I / n_total_configs
    diffuse = mean_I - mean_F.abs() ** 2

    loss = loss_fn(mean_F, diffuse)
    loss.backward()

    return loss.item(), global_sum_F.grad.numpy(), global_sum_I.grad.numpy()


def seed_local_backward(local_sum_F: torch.Tensor, local_sum_I: torch.Tensor,
                         grad_F_np, grad_I_np):
    """
    Completes each rank's REAL backward pass, seeded with the upstream
    gradients from the tiny shared graph. This touches only this rank's
    own local computational graph (built by local_forward) -- no further
    communication. Populates .grad on every atom in every local config,
    combining contributions through both the Bragg and diffuse channels
    correctly (autograd sums gradient contributions wherever a value has
    multiple downstream consumers, exactly as it would in one combined
    graph -- that's the property validate_gradient_seeding.py confirms).
    """
    grad_F = torch.tensor(grad_F_np, dtype=local_sum_F.dtype, device=local_sum_F.device)
    grad_I = torch.tensor(grad_I_np, dtype=local_sum_I.dtype, device=local_sum_I.device)
    torch.autograd.backward([local_sum_F, local_sum_I], grad_tensors=[grad_F, grad_I])


def training_step(comm, frac_coords_list, element_idx_list, occ_list,
                   atom_A, atom_lam, elem_offsets, atom_radius_ang,
                   grid_shape, orth_matrix, cell_volume, hkl, taper_width,
                   n_total_configs, loss_fn, optimizer,
                   blur=0.0, max_atoms_per_batch=50_000, grid_ops=None):
    """
    One full training step: local forward on this rank's configs, one
    small Allreduce, tiny shared backward for the upstream gradient,
    seeded local backward, optimizer step. Returns the scalar loss value
    (identical on every rank, since it's computed from Allreduce'd data).
    """
    optimizer.zero_grad()

    local_sum_F, local_sum_I = local_forward(
        frac_coords_list, element_idx_list, occ_list,
        atom_A, atom_lam, elem_offsets, atom_radius_ang,
        grid_shape, orth_matrix, cell_volume, hkl, taper_width,
        blur=blur, max_atoms_per_batch=max_atoms_per_batch, grid_ops=grid_ops,
    )

    global_sum_F_np, global_sum_I_np = allreduce_sums(comm, local_sum_F, local_sum_I)

    loss_value, grad_F_np, grad_I_np = compute_upstream_gradients(
        global_sum_F_np, global_sum_I_np, n_total_configs, loss_fn,
    )

    seed_local_backward(local_sum_F, local_sum_I, grad_F_np, grad_I_np)

    optimizer.step()

    return loss_value
