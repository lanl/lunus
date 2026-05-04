#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
source /home/jamesh/projects/fft_symmetry/claude_test/setup_cuda.sh

# Extract first 10000 ATOM/HETATM records from P1pdb1.pdb, keep header
ccp4-python - <<'PYEOF'
import random, gemmi
random.seed(42)
st = gemmi.read_structure('P1pdb1.pdb')
count = 0
done = False
for model in st:
    for chain in model:
        for res in chain:
            for atom in res:
                if count >= 10000:
                    done = True
                    break
                atom.b_iso = random.uniform(2.0, 999.0)
                count += 1
            if done: break
        if done: break
    if done: break
# Write only the first 10000 atoms
out = gemmi.Structure()
out.cell = st.cell
out.spacegroup_hm = st.spacegroup_hm
out_model = gemmi.Model('1')
n = 0
for model in st:
    for chain in model:
        out_chain = gemmi.Chain(chain.name)
        for res in chain:
            out_res = gemmi.Residue()
            out_res.name = res.name
            out_res.seqid = res.seqid
            for atom in res:
                if n >= 10000: break
                out_res.add_atom(atom)
                n += 1
            if len(out_res) > 0:
                out_chain.add_residue(out_res)
            if n >= 10000: break
        if len(out_chain) > 0:
            out_model.add_chain(out_chain)
        if n >= 10000: break
    break
out.add_model(out_model)
out.write_pdb('P1pdb1_10k.pdb')
print(f"Written P1pdb1_10k.pdb with {n} atoms, B uniform in [2,999]")
PYEOF

gemmi sfcalc --dmin=1.5 --rate=2.5 --to-mtz=P1pdb1_10k_gemmi.mtz P1pdb1_10k.pdb
ccp4-python sfcalc_gpu.py P1pdb1_10k.pdb outmtz=P1pdb1_10k_gpu.mtz outmap= bmax=0
ccp4-python compare_mtz.py P1pdb1_10k_gemmi.mtz P1pdb1_10k_gpu.mtz
