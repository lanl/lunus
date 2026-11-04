"""Atom order at the iotbx boundary.

iotbx's hierarchy and the scatterers xray_structure_simple() builds hold the
same atoms in different orders, so anything computed on one and applied to
the other must be mapped. numpy only at import; the iotbx objects are passed
in.
"""

import numpy as np


def selection_in_file_order(pdb_in, selection_text):
    """An iotbx atom selection as a boolean mask in FILE order, the order
    xray_structure_simple's scatterers (and so read_model's arrays) are in.

    pdb_in is an iotbx.pdb hierarchy.input. Its hierarchy groups atoms by
    altloc within each residue, so on a file whose altlocs are interleaved
    atom by atom (7FPV) the hierarchy's order is not the file's, and a mask
    from atom_selection_cache() applied to the scatterers lands on the wrong
    atoms. The hierarchy's i_seq does not help: it is the hierarchy's own
    order. The hierarchy's atoms are the input's own atom objects, reordered,
    and iotbx's hierarchy_to_input_atom_permutation() maps them by tagging
    those shared objects. Labels would not do: an MD box repeats them (the
    compare_gemmi box has 21,690 duplicates once water residue numbers
    wrap)."""
    # file index -> hierarchy index
    perm = pdb_in.hierarchy_to_input_atom_permutation().as_numpy_array()
    # A copy whose file records and hierarchy no longer share atoms reads back
    # untagged (all zeros) rather than raising, so check it is a permutation.
    if not np.array_equal(np.sort(perm), np.arange(len(perm))):
        raise ValueError(
            "the hierarchy and the file records do not share atoms (a "
            "hierarchy.input that was pickled and unpickled, for example): "
            "compute the selection before copying it")
    in_hierarchy_order = pdb_in.hierarchy.atom_selection_cache() \
        .selection(selection_text).as_numpy_array()
    return in_hierarchy_order[perm]
