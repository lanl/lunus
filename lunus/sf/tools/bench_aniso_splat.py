#!/usr/bin/env python
"""Deprecated name: the benchmark is tools/bench_aniso_density.py. This runs it with the
same arguments."""
import os
import runpy
import sys

print("bench_aniso_splat.py is deprecated; use bench_aniso_density.py", file=sys.stderr)
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_aniso_density.py"),
               run_name="__main__")
