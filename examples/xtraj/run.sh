# gemmi_cutoff is PINNED to 0.01, the old default, rather than inheriting the
# current one of 1e-4. This example doubles as the byte-comparison smoke test
# against locally kept *.ref.hkl / *.ref.mtz outputs, and a looser cutoff is a
# different calculation -- inheriting the new default would make every one of
# those comparisons fail for a reason that is not a regression. Drop the
# argument to see what a default run now does; expect the outputs to differ
# and to cost ~1.2-1.5x more.
OMP_NUM_THREADS=1 mpirun -np 4 python ~/packages/lunus/lunus/command_line/xtraj.py top=top_ref.pdb traj=traj_ref.xtc d_min=1.8 first=0 last=10 chunk=2 engine=gemmi gemmi_cutoff=0.01
