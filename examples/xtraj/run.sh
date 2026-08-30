OMP_NUM_THREADS=1 mpirun -np 4 python ~/packages/lunus/lunus/command_line/xtraj.py top=top_ref.pdb traj=traj_ref.xtc d_min=1.8 first=0 last=10 chunk=2 engine=gemmi gemmi_cutoff=0.01
