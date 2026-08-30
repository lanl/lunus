"""
Writes a copy of a PDB file with a random, UNIFORM-PER-RESIDUE B-factor
in the B column -- one draw per residue, shared by every atom in that
residue -- for exercising the per-atom B-factor path (use_top_bfacs=True)
with genuinely varying B, instead of the all-zeros column the MD topology
comes with.

Residues are identified as CONTIGUOUS BLOCKS of (chainID, resSeq+iCode),
not by the key alone: a large MD system recycles chain IDs once the 62
single-character IDs run out (and leaves many residues with a blank
chain ID), and water resSeq numbering repeats, so the same key legitimately
reappears for different residues later in the file. Block boundaries are
unambiguous because residues are always written contiguously.

Usage:
    python make_random_bfacs.py in.pdb out.pdb [--bmin 10] [--bmax 100] [--seed 0]
"""

import argparse
import numpy as np

# PDB fixed-column fields (0-indexed slices)
CHAIN_ID = 21
RES_ID = slice(22, 27)   # resSeq (22:26) + iCode (26)
B_FACTOR = slice(60, 66)  # written as %6.2f


def randomize_bfactors(in_path, out_path, bmin=10.0, bmax=100.0, seed=0):
    rng = np.random.default_rng(seed)

    n_atoms = 0
    n_residues = 0
    prev_key = None
    b_current = 0.0
    b_values = []

    with open(in_path) as fi, open(out_path, "w") as fo:
        for line in fi:
            if not line.startswith(("ATOM", "HETATM")):
                fo.write(line)
                continue

            line = line.rstrip("\n").ljust(80)
            key = (line[CHAIN_ID], line[RES_ID])
            if key != prev_key:
                b_current = float(rng.uniform(bmin, bmax))
                b_values.append(b_current)
                n_residues += 1
                prev_key = key

            fo.write(line[:B_FACTOR.start] + f"{b_current:6.2f}" + line[B_FACTOR.stop:] + "\n")
            n_atoms += 1

    b_values = np.array(b_values)
    return n_atoms, n_residues, b_values


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_pdb")
    p.add_argument("out_pdb")
    p.add_argument("--bmin", type=float, default=10.0)
    p.add_argument("--bmax", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0,
                   help="fixed by default so the same B-factors are reproducible run to run")
    args = p.parse_args()

    n_atoms, n_residues, b = randomize_bfactors(
        args.in_pdb, args.out_pdb, args.bmin, args.bmax, args.seed
    )

    print(f"wrote {args.out_pdb}")
    print(f"  atoms    = {n_atoms}")
    print(f"  residues = {n_residues}  (one B draw each, seed={args.seed})")
    print(f"  B range  = {b.min():.2f} - {b.max():.2f}, mean {b.mean():.2f} "
          f"(requested {args.bmin} - {args.bmax})")


if __name__ == "__main__":
    main()
