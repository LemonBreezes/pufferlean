#!/usr/bin/env python3
"""Cross-check the VERIFIED forward-mode AD gradient (puffer's error-bound mode) against exact ℝ.

`puffer verify-grad` runs the machine-checked forward-mode AD (`Puffer/Float/AutoDiffR.lean`) on
fixed expressions: for each, it emits the expression tree, the Float environment, the Float
gradient `dF`, and its PROVEN per-component error bound `derivErrBndF` (the Float mirror of the
ℝ bound `derivErrBnd`, certified by the theorem `dF_error`).

This script reconstructs the same f64s, computes the IDEAL real gradient `derivR` in Python
`decimal` at 60 digits (an ℝ proxy — exp/log are transcendental, so unlike the rational kernels
this needs `decimal`, not `fractions`), and — instead of trusting the emitted Float mirror
`derivErrBndF` (which rounds and can fall slightly below the ℝ bound) — RECOMPUTES the proven ℝ
bound `derivErrBnd` EXACTLY (`derivErrBnd_R`: each `toReal(Float subexpr)` via a Python-float
replica with `math.exp/log` = Lean's libm, the genuine reals in `decimal`). It then asserts, per k:

    |toReal(dF e σ k) − ∂(evalR e)/∂(var k)| ≤ derivErrBnd e σ k,

which IS the theorem `dF_error` evaluated on the emitted inputs — no Float-mirror seam. A pass
confirms the running gradient tracks the exact real derivative inside the proven interval —
closing the trifecta on the AD. (Residual assumptions: Python float == Lean Float; `math.exp/log`
share Lean's libm; `decimal` at 60 digits ≈ ℝ — the same the whole harness already rests on.)

Usage:  tools/grad_ref.py                     # build+run `puffer verify-grad`, check it
        puffer verify-grad | tools/grad_ref.py -
"""
import json, math, os, struct, subprocess, sys
from decimal import Decimal, getcontext

getcontext().prec = 60
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U64 = Decimal(1) / 2 ** 53          # Lean u64 = 2⁻⁵³
EXP_EPS = Decimal(1) / 2 ** 52      # Lean expEps = 2⁻⁵²
LOG_EPS = Decimal(1) / 2 ** 52      # Lean logEps = 2⁻⁵²


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def D(bits):
    return Decimal(f64(bits))


def evalR(e, env):
    """Exact-ℝ (high-precision decimal) evaluation, mirroring AutoDiffR.evalR."""
    op = e["op"]
    if op == "var":   return env[e["i"]]
    if op == "const": return D(e["c"])
    if op == "add":   return evalR(e["a"], env) + evalR(e["b"], env)
    if op == "sub":   return evalR(e["a"], env) - evalR(e["b"], env)
    if op == "mul":   return evalR(e["a"], env) * evalR(e["b"], env)
    if op == "scale": return D(e["c"]) * evalR(e["a"], env)
    if op == "exp":   return evalR(e["a"], env).exp()
    if op == "log":   return evalR(e["a"], env).ln()
    if op == "relu":  return max(evalR(e["a"], env), Decimal(0))
    if op == "max":   return max(evalR(e["a"], env), evalR(e["b"], env))
    if op == "min":   return min(evalR(e["a"], env), evalR(e["b"], env))
    raise ValueError(f"unknown op {op!r}")


def derivR(e, env, k):
    """∂/∂(var k) of evalR, mirroring AutoDiffR.derivR (standard rules)."""
    op = e["op"]
    if op == "var":   return Decimal(1) if e["i"] == k else Decimal(0)
    if op == "const": return Decimal(0)
    if op == "add":   return derivR(e["a"], env, k) + derivR(e["b"], env, k)
    if op == "sub":   return derivR(e["a"], env, k) - derivR(e["b"], env, k)
    if op == "mul":   return derivR(e["a"], env, k) * evalR(e["b"], env) + evalR(e["a"], env) * derivR(e["b"], env, k)
    if op == "scale": return D(e["c"]) * derivR(e["a"], env, k)
    if op == "exp":   return evalR(e["a"], env).exp() * derivR(e["a"], env, k)
    if op == "log":   return derivR(e["a"], env, k) / evalR(e["a"], env)
    if op == "relu":  return derivR(e["a"], env, k) if evalR(e["a"], env) > 0 else Decimal(0)
    # max/min: derivative of the selected argument (Lean: max a b = b, min a b = a, when a ≤ b).
    if op == "max":   return derivR(e["b"], env, k) if evalR(e["a"], env) <= evalR(e["b"], env) else derivR(e["a"], env, k)
    if op == "min":   return derivR(e["a"], env, k) if evalR(e["a"], env) <= evalR(e["b"], env) else derivR(e["b"], env, k)
    raise ValueError(f"unknown op {op!r}")


# ── EXACT-ℝ recomputation of the PROVEN Lean bound derivErrBnd (AutoDiffR.lean) ──────────────
# Replaces the emitted Float mirror derivErrBndF (which rounds and can fall slightly below the ℝ
# bound). Every toReal(<Float subexpr>) = the exact value of the Lean executable replicated in
# PYTHON FLOAT (Float ops == IEEE double; math.exp/log == Lean Float.exp/log, same libm), widened
# losslessly to Decimal; the genuine reals (Real.exp/log, evalR/derivR) are Decimal at prec=60.

def evalF_f(e, ef):
    """Lean evalF replicated in Python float."""
    op = e["op"]
    if op == "var":   return ef[e["i"]]
    if op == "const": return f64(e["c"])
    if op == "add":   return evalF_f(e["a"], ef) + evalF_f(e["b"], ef)
    if op == "sub":   return evalF_f(e["a"], ef) - evalF_f(e["b"], ef)
    if op == "mul":   return evalF_f(e["a"], ef) * evalF_f(e["b"], ef)
    if op == "scale": return f64(e["c"]) * evalF_f(e["a"], ef)
    if op == "exp":   return math.exp(evalF_f(e["a"], ef))
    if op == "log":   return math.log(evalF_f(e["a"], ef))
    if op == "relu":  return 0.0 if evalF_f(e["a"], ef) < 0.0 else evalF_f(e["a"], ef)
    if op == "max":   return (evalF_f(e["b"], ef) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else evalF_f(e["a"], ef))
    if op == "min":   return (evalF_f(e["a"], ef) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else evalF_f(e["b"], ef))
    raise ValueError(op)


def dF_f(e, ef, k):
    """Lean dF replicated in Python float."""
    op = e["op"]
    if op == "var":   return 1.0 if e["i"] == k else 0.0
    if op == "const": return 0.0
    if op == "add":   return dF_f(e["a"], ef, k) + dF_f(e["b"], ef, k)
    if op == "sub":   return dF_f(e["a"], ef, k) - dF_f(e["b"], ef, k)
    if op == "mul":   return dF_f(e["a"], ef, k) * evalF_f(e["b"], ef) + evalF_f(e["a"], ef) * dF_f(e["b"], ef, k)
    if op == "scale": return f64(e["c"]) * dF_f(e["a"], ef, k)
    if op == "exp":   return math.exp(evalF_f(e["a"], ef)) * dF_f(e["a"], ef, k)
    if op == "log":   return dF_f(e["a"], ef, k) / evalF_f(e["a"], ef)
    if op == "relu":  return 0.0 if evalF_f(e["a"], ef) < 0.0 else dF_f(e["a"], ef, k)
    if op == "max":   return dF_f(e["b"], ef, k) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else dF_f(e["a"], ef, k)
    if op == "min":   return dF_f(e["a"], ef, k) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else dF_f(e["b"], ef, k)
    raise ValueError(op)


def evalErrBnd_R(e, ef, ed):
    """Exact ℝ value of Lean evalErrBnd (ef = float env, ed = Decimal env)."""
    op = e["op"]
    if op in ("var", "const"): return Decimal(0)
    if op in ("add", "sub"):
        ta, tb = Decimal(evalF_f(e["a"], ef)), Decimal(evalF_f(e["b"], ef))
        s = (ta + tb) if op == "add" else (ta - tb)
        return U64 * abs(s) + evalErrBnd_R(e["a"], ef, ed) + evalErrBnd_R(e["b"], ef, ed)
    if op == "mul":
        ta, tb = Decimal(evalF_f(e["a"], ef)), Decimal(evalF_f(e["b"], ef))
        return (U64 * abs(ta * tb) + abs(ta) * evalErrBnd_R(e["b"], ef, ed)
                + abs(evalR(e["b"], ed)) * evalErrBnd_R(e["a"], ef, ed))
    if op == "scale":
        c, ta = Decimal(f64(e["c"])), Decimal(evalF_f(e["a"], ef))
        return U64 * abs(c * ta) + abs(c) * evalErrBnd_R(e["a"], ef, ed)
    if op == "exp":
        ta, ea = Decimal(evalF_f(e["a"], ef)), evalErrBnd_R(e["a"], ef, ed)
        return EXP_EPS * ta.exp() + ta.exp() * (ea.exp() - 1)
    if op == "log":
        ta, ea = Decimal(evalF_f(e["a"], ef)), evalErrBnd_R(e["a"], ef, ed)
        return LOG_EPS * abs(ta.ln()) + ea / (ta - ea)
    if op == "relu":  # reluF is exact + 1-Lipschitz: value error inherited from the argument.
        return evalErrBnd_R(e["a"], ef, ed)
    if op in ("max", "min"):  # exact; value error bounded by the sum of the arguments' errors.
        return evalErrBnd_R(e["a"], ef, ed) + evalErrBnd_R(e["b"], ef, ed)
    raise ValueError(op)


def derivErrBnd_R(e, ef, ed, k):
    """Exact ℝ value of Lean derivErrBnd — the PROVEN per-component gradient error bound."""
    op = e["op"]
    if op in ("var", "const"): return Decimal(0)
    if op in ("add", "sub"):
        da, db = Decimal(dF_f(e["a"], ef, k)), Decimal(dF_f(e["b"], ef, k))
        s = (da + db) if op == "add" else (da - db)
        return U64 * abs(s) + derivErrBnd_R(e["a"], ef, ed, k) + derivErrBnd_R(e["b"], ef, ed, k)
    if op == "mul":
        af, bf, daf, dbf = evalF_f(e["a"], ef), evalF_f(e["b"], ef), dF_f(e["a"], ef, k), dF_f(e["b"], ef, k)
        ta, tb, tda, tdb = Decimal(af), Decimal(bf), Decimal(daf), Decimal(dbf)
        return (U64 * abs(Decimal(daf * bf) + Decimal(af * dbf))
                + (U64 * abs(tda * tb) + abs(tda) * evalErrBnd_R(e["b"], ef, ed)
                   + abs(evalR(e["b"], ed)) * derivErrBnd_R(e["a"], ef, ed, k))
                + (U64 * abs(ta * tdb) + abs(ta) * derivErrBnd_R(e["b"], ef, ed, k)
                   + abs(derivR(e["b"], ed, k)) * evalErrBnd_R(e["a"], ef, ed)))
    if op == "scale":
        c, tda = Decimal(f64(e["c"])), Decimal(dF_f(e["a"], ef, k))
        return U64 * abs(c * tda) + abs(c) * derivErrBnd_R(e["a"], ef, ed, k)
    if op == "exp":
        af, tda = evalF_f(e["a"], ef), Decimal(dF_f(e["a"], ef, k))
        te = Decimal(math.exp(af))                    # toReal(Float.exp(evalF a))
        return (U64 * abs(te * tda) + abs(te) * derivErrBnd_R(e["a"], ef, ed, k)
                + abs(derivR(e["a"], ed, k)) * evalErrBnd_R(e, ef, ed))   # evalErrBnd (.exp a)
    if op == "log":
        af, daf = evalF_f(e["a"], ef), dF_f(e["a"], ef, k)
        ta, ea = Decimal(af), evalErrBnd_R(e["a"], ef, ed)
        numer = derivErrBnd_R(e["a"], ef, ed, k) + abs(derivR(e["a"], ed, k) / evalR(e["a"], ed)) * ea
        # Lean ℝ derivErrBnd log: u64·|toReal(dF a) / toReal(evalF a)| — division in ℝ (widen each,
        # then divide), NOT toReal(dF a / evalF a) which would be the Float mirror's float division.
        return U64 * abs(Decimal(daf) / Decimal(af)) + numer / (ta - ea)
    if op == "relu":  # away from kink the subgradient (0 or 1) is exact: error inherited.
        return derivErrBnd_R(e["a"], ef, ed, k)
    # max/min: away from the kink the selected argument's derivative error is inherited.
    if op == "max":
        return derivErrBnd_R(e["b"], ef, ed, k) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else derivErrBnd_R(e["a"], ef, ed, k)
    if op == "min":
        return derivErrBnd_R(e["a"], ef, ed, k) if evalF_f(e["a"], ef) <= evalF_f(e["b"], ef) else derivErrBnd_R(e["b"], ef, ed, k)
    raise ValueError(op)


def expr_str(e):
    """Compact human-readable form for the report label."""
    op = e["op"]
    if op == "var":   return f"x{e['i']}"
    if op == "const": return f"{f64(e['c']):g}"
    if op == "scale": return f"{f64(e['c']):g}*{expr_str(e['a'])}"
    if op in ("exp", "log", "relu"): return f"{op}({expr_str(e['a'])})"
    if op in ("max", "min"): return f"{op}({expr_str(e['a'])},{expr_str(e['b'])})"
    sym = {"add": "+", "sub": "-", "mul": "*"}[op]
    return f"({expr_str(e['a'])}{sym}{expr_str(e['b'])})"


def check(rec):
    env = [D(b) for b in rec["env"]]                       # Decimal env (exact-ℝ)
    env_f = [f64(b) for b in rec["env"]]                   # float env (for the toReal(Float) parts)
    grad = [D(b) for b in rec["grad"]]
    n = int(rec["nvars"])
    # PROVEN ℝ bound recomputed exactly (not the emitted Float mirror rec["bound"])
    bound = [derivErrBnd_R(rec["expr"], env_f, env, k) for k in range(n)]
    worst, fails = 0.0, 0
    for k in range(n):
        ideal = derivR(rec["expr"], env, k)
        err = abs(grad[k] - ideal)
        ok = err <= bound[k]
        if not ok:
            fails += 1
        r = float(err / bound[k]) if bound[k] != 0 else (0.0 if err == 0 else float("inf"))
        if r > worst:
            worst = r
    label = expr_str(rec["expr"])
    return fails == 0, (label[:34] if len(label) > 34 else label), worst, f"{n} components  worst slack={worst:6.1%}"


def main(argv):
    if argv and argv[0] == "-":
        lines = sys.stdin.read().splitlines()
    else:
        subprocess.run(["lake", "build", "puffer"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        lines = subprocess.run([f"{ROOT}/.lake/build/bin/puffer", "verify-grad"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    fails, worst, worst_lbl = 0, 0.0, ""
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n += 1
        ok, label, ratio, detail = check(json.loads(line))
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:36s} {detail}")
        if not ok:
            fails += 1
        if ratio > worst:
            worst, worst_lbl = ratio, label
    print(f"\n{n - fails}/{n} gradients within the proven bound"
          + (f"; tightest at {worst_lbl} (used {worst:.1%})" if worst_lbl else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
