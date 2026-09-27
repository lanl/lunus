#!/usr/bin/env ccp4-python
"""
test_all_sg.py
==============
Run test_one_sg.py for all 230 space groups and print a summary.

Usage:
  ccp4-python test_all_sg.py [dmin=2.0] [workdir=sg_tests] [jobs=4] [start=1] [end=230]

Output:
  One PASS/FAIL line per SG (from test_one_sg.py), then a summary table.
  Full logs are in workdir/<NNN_sgname>/test.log
"""

import sys
import os
import subprocess
import concurrent.futures

HERE    = os.path.dirname(os.path.abspath(__file__))
SCRIPT  = os.path.join(HERE, 'test_one_sg.py')


def parse_args(argv):
    args = {'dmin': 2.0, 'workdir': 'sg_tests', 'jobs': 4, 'start': 1, 'end': 230}
    for a in argv:
        for key in args:
            if a.startswith(f'{key}='):
                val = a.split('=', 1)[1]
                args[key] = type(args[key])(val)
    return args


def run_one(sgnum, dmin, workdir):
    """Run test_one_sg.py for one SG; return (sgnum, line, passed)."""
    result = subprocess.run(
        ['ccp4-python', SCRIPT, str(sgnum),
         f'dmin={dmin}', f'workdir={workdir}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = result.stdout.strip()
    passed = result.returncode == 0
    return sgnum, output, passed


def main():
    args = parse_args(sys.argv[1:])
    dmin    = args['dmin']
    workdir = args['workdir']
    jobs    = args['jobs']
    start   = args['start']
    end     = args['end']

    os.makedirs(workdir, exist_ok=True)
    sgnums = list(range(start, end + 1))

    print(f"Testing SG {start}–{end}  (dmin={dmin}, workdir={workdir}, jobs={jobs})")
    print()

    results = []   # list of (sgnum, line, passed)

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(run_one, n, dmin, workdir): n for n in sgnums}
        for fut in concurrent.futures.as_completed(futures):
            sgnum, line, passed = fut.result()
            print(line)
            sys.stdout.flush()
            results.append((sgnum, line, passed))

    # Sort by SG number and write results file
    results.sort(key=lambda r: r[0])
    results_path = os.path.join(workdir, 'results.txt')
    with open(results_path, 'w') as f:
        for _, line, _ in results:
            f.write(line + '\n')

    n_pass = sum(1 for _, _, p in results if p)
    n_fail = sum(1 for _, _, p in results if not p)

    print()
    print('=== Summary ===')
    fail_lines = [line for _, line, p in results if not p]
    if fail_lines:
        print('FAILs:')
        for line in fail_lines:
            print(' ', line)
    print(f'Total: {len(results)}   PASS: {n_pass}   FAIL: {n_fail}')
    print(f'Results written to {results_path}')

    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
