#!/usr/bin/env python3
"""Differential-test PufferLib env Lean reference models against the C impl.

Both the C trace driver and the Lean trace exe emit TSV (a header row + one row
per step) on identical CLI args. We compare column-by-column: integer columns
exactly, float columns within tolerance. Drivers drive from a fixed initial state
(bypassing reset RNG) and stop at the first terminal, so traces are deterministic.

Env registry is data-driven from tools/env_cases.json:
    { "<env>": { "exe": "<lake exe name>", "cases": [[arg, ...], ...] } }
The C binary is ctest/bin/<env>_trace; the Lean binary is .lake/build/bin/<exe>.

Usage:  tools/difftest.py [env|all]
"""
import json, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-4

with open(os.path.join(ROOT, "tools", "env_cases.json")) as fh:
    _CASES = json.load(fh)

REGISTRY = {
    env: (f"{ROOT}/ctest/bin/{env}_trace", f"{ROOT}/.lake/build/bin/{spec['exe']}", spec["cases"])
    for env, spec in _CASES.items()
}


def run(binary, args):
    out = subprocess.run([binary, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{binary} {' '.join(args)} exited {out.returncode}: {out.stderr.strip()}")
    return out.stdout.strip().splitlines()


def cell_eq(a, b):
    try:
        return abs(float(a) - float(b)) <= TOL
    except ValueError:
        return a == b


def compare_case(cbin, lbin, case):
    c, l = run(cbin, case), run(lbin, case)
    if not c or not l:
        return False, "empty output"
    if c[0].split("\t") != l[0].split("\t"):
        return False, f"header: C={c[0]!r} Lean={l[0]!r}"
    if len(c) != len(l):
        return False, f"row count: C={len(c)-1} Lean={len(l)-1}"
    header = c[0].split("\t")
    for ri in range(1, len(c)):
        cr, lr = c[ri].split("\t"), l[ri].split("\t")
        for ci, (cv, lv) in enumerate(zip(cr, lr)):
            if not cell_eq(cv, lv):
                return False, f"row {ri-1} col {header[ci]}: C={cv} Lean={lv}"
    return True, ""


def run_env(env):
    cbin, lbin, cases = REGISTRY[env]
    p = f = 0
    for case in cases:
        try:
            ok, detail = compare_case(cbin, lbin, case)
        except RuntimeError as e:
            ok, detail = False, str(e)
        if ok:
            p += 1
        else:
            f += 1
            print(f"  FAIL {env} {' '.join(case)}: {detail}")
    print(f"{env:12} PASS={p} FAIL={f}")
    return f == 0


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    envs = list(REGISTRY) if which == "all" else [which]
    ok = all(run_env(e) for e in envs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
