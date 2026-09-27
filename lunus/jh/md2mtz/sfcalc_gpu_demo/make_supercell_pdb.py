#!/usr/bin/env ccp4-python
"""
make_supercell_pdb.py
=====================
Tile a PDB unit cell into a P1 supercell by replicating along a, b, c.

The input PDB should contain the *complete unit cell* (all symmetry-related
copies already in P1).  Each replica is shifted by (ia*a, ib*b, ic*c).

Usage
-----
  make_supercell_pdb.py  input.pdb  [super_mult=na,nb,nc]  [out=supercell.pdb]

  super_mult   comma-separated multipliers along a, b, c  (default 2,2,2)
  out          output filename (default supercell.pdb)
"""

import sys
import gemmi


def parse_args(argv):
    args = {'pdb': None, 'na': 2, 'nb': 2, 'nc': 2, 'out': 'supercell.pdb'}
    for arg in argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower().strip()
            if key in ('super_mult', 'mult', 'multipliers'):
                parts = [p.strip() for p in val.replace('x', ',').split(',')]
                if len(parts) != 3:
                    sys.exit("ERROR: super_mult must be na,nb,nc  e.g. 3,3,3")
                args['na'], args['nb'], args['nc'] = int(parts[0]), int(parts[1]), int(parts[2])
            elif key in ('out', 'outfile', 'output'):
                args['out'] = val
        elif arg.endswith('.pdb') or arg.endswith('.cif'):
            args['pdb'] = arg
    return args


def main():
    args = parse_args(sys.argv)
    if args['pdb'] is None:
        print(__doc__)
        sys.exit("ERROR: no input PDB specified")

    na, nb, nc = args['na'], args['nb'], args['nc']

    st = gemmi.read_structure(args['pdb'])
    cell = st.cell
    sg = st.find_spacegroup()

    n_in = sum(1 for ch in st[0] for res in ch for _ in res)
    print(f"Input:  {args['pdb']}")
    print(f"  Cell      : {cell.a:.3f} {cell.b:.3f} {cell.c:.3f}  "
          f"{cell.alpha:.2f} {cell.beta:.2f} {cell.gamma:.2f}")
    print(f"  Space grp : {sg.xhm() if sg else 'P1'}")
    print(f"  Atoms     : {n_in}")
    print(f"  Supercell : {na} x {nb} x {nc}")

    sup_cell = gemmi.UnitCell(
        na * cell.a, nb * cell.b, nc * cell.c,
        cell.alpha, cell.beta, cell.gamma
    )

    # Collect atoms from original model as (fractional_x, fractional_y, fractional_z, Atom)
    atoms_frac = []
    for ch in st[0]:
        for res in ch:
            for atom in res:
                frac = cell.fractionalize(atom.pos)
                atoms_frac.append((frac.x, frac.y, frac.z, res.name, atom))

    # Build output structure
    out_st = gemmi.Structure()
    out_st.cell = sup_cell
    out_st.spacegroup_hm = 'P 1'
    out_model = gemmi.Model('1')

    # Use single-letter chain IDs cycling through A-Z then a-z then 0-9
    _chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    chain_counter = [0]

    def next_chain():
        c = chain_counter[0]
        chain_counter[0] += 1
        # Two-character chain id for large supercells
        if c < len(_chars):
            return _chars[c]
        return _chars[(c // len(_chars)) - 1] + _chars[c % len(_chars)]

    res_serial = 1
    for ia in range(na):
        for ib in range(nb):
            for ic in range(nc):
                chain = gemmi.Chain(next_chain())
                res = gemmi.Residue()
                res.name = 'SUP'
                res.seqid = gemmi.SeqId(res_serial, ' ')
                for (fx, fy, fz, resname, orig_atom) in atoms_frac:
                    new_atom = orig_atom.clone()
                    sup_frac = gemmi.Fractional(
                        (fx + ia) / na,
                        (fy + ib) / nb,
                        (fz + ic) / nc,
                    )
                    new_atom.pos = sup_cell.orthogonalize(sup_frac)
                    res.add_atom(new_atom)
                res_serial += 1
                chain.add_residue(res)
                out_model.add_chain(chain)

    out_st.add_model(out_model)
    n_out = sum(1 for ch in out_st[0] for r in ch for _ in r)
    print(f"Output: {args['out']}")
    print(f"  Cell      : {sup_cell.a:.3f} {sup_cell.b:.3f} {sup_cell.c:.3f}  "
          f"{sup_cell.alpha:.2f} {sup_cell.beta:.2f} {sup_cell.gamma:.2f}")
    print(f"  Atoms     : {n_out}  ({na}*{nb}*{nc} = {na*nb*nc} copies of {n_in})")
    out_st.write_pdb(args['out'])
    print(f"Done.")


if __name__ == '__main__':
    main()
