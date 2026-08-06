#!/usr/bin/env python3
"""Cross-check the runnable Adam step (`puffer verify-adam`) against exact ℝ.

`puffer verify-adam` runs one Adam parameter update per case from fixed inputs and emits the
weight `p`, its 1st/2nd-moment state `m`,`v`, gradient `g`, the Adam hyper-parameters, and the
Float result `adamStepF` (`Puffer/Float/Exec.lean`, the Mathlib-free exec certified by
`Puffer.RL.AdamStep.adamStepF_error` through the rfl bridge `Puffer.RL.AdamExecBridge`) — all as
exact f64 bit patterns.

This script reconstructs the same f64s, computes the IDEAL ℝ step `adamStepR` in `decimal` at 60
digits (an ℝ proxy; sqrt is irrational so — like the exp/log kernels — this needs `decimal`, not
`fractions`), and — instead of trusting an emitted Float bound mirror (which rounds and can fall
below the ℝ bound) — RECOMPUTES the PROVEN ℝ bound `adamStepF_error` EXACTLY: every
`toReal(<Float subexpr>)` is the Lean executable replicated in PYTHON FLOAT (Float ops == IEEE
double, `math.sqrt` == Lean `Float.sqrt`, same libm), widened losslessly to `Decimal`; the genuine
reals (`Real.sqrt`, `adamM1R`/`adamM2R`/`adamDirR`) are `Decimal`. It then asserts, per case:

    |toReal(adamStepF …) − adamStepR …| ≤ (adamStepF_error's RHS),

which IS the theorem `adamStepF_error` (composed with `adamM1_error`/`adamM2_error`/`adamDir_error`)
evaluated on the emitted inputs — no Float-mirror seam. The inputs are the exact starting state, so
every input error (εp, εm, εv, εg) is 0 and the bound is the pure rounding error of the update
circuit. The denominator floor `dmin = toReal(Float.sqrt v' + ε)` (equality, so tightest). A pass
confirms the runnable Adam step tracks the exact real update inside its proven interval.

Usage:  tools/adam_ref.py                     # build+run `puffer verify-adam`, check it
        puffer verify-adam | tools/adam_ref.py -
"""
import json, math, os, struct, subprocess, sys
from decimal import Decimal, getcontext

getcontext().prec = 60
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U64 = Decimal(1) / 2 ** 53          # Lean u64 = 2⁻⁵³


def f64(bits):
    return struct.unpack("<d", struct.pack("<Q", int(bits)))[0]


def D(bits):
    """The emitted f64 as an exact Decimal (= toReal of the literal)."""
    return Decimal(f64(bits))


def Df(x):
    """A Python float (= a Lean Float value) widened losslessly to Decimal."""
    return Decimal(x)


def check(rec):
    # ── inputs: exact f64 → Python float (Lean Float) and Decimal (toReal) ───────────
    p, m, v, g = f64(rec["p"]), f64(rec["m"]), f64(rec["v"]), f64(rec["g"])
    lr, eps = f64(rec["lr"]), f64(rec["eps"])
    b1, c1, b2, c2 = f64(rec["b1"]), f64(rec["c1"]), f64(rec["b2"]), f64(rec["c2"])
    result = D(rec["result"])

    # ── Float circuit replicated in Python float (== toReal(Float subexpr)) ──────────
    b1m_f = b1 * m; c1g_f = c1 * g; m1_f = b1m_f + c1g_f               # adamM1F
    gg_f = g * g; b2v_f = b2 * v; c2gg_f = c2 * gg_f; m2_f = b2v_f + c2gg_f  # adamM2F
    sq_f = math.sqrt(m2_f); den_f = sq_f + eps; dir_f = m1_f / den_f   # adamDirF
    ld_f = lr * dir_f                                                  # lr·dir

    # ── ideal ℝ update (Decimal), hyper-params exact (their toReal) ──────────────────
    pR, mR, vR, gR = Df(p), Df(m), Df(v), Df(g)
    lrR, epsR = Df(lr), Df(eps)
    b1R, c1R, b2R, c2R = Df(b1), Df(c1), Df(b2), Df(c2)
    m1R = b1R * mR + c1R * gR                                          # adamM1R
    m2R = b2R * vR + c2R * (gR * gR)                                   # adamM2R
    denR = m2R.sqrt() + epsR
    dirR = m1R / denR                                                  # adamDirR
    stepR = pR - lrR * dirR                                            # adamStepR

    # ── PROVEN ℝ bound = adamStepF_error RHS (εp=εm=εv=εg=0) ─────────────────────────
    # adamM1_error RHS (εm=εg=0):
    em1 = U64 * abs(Df(b1m_f) + Df(c1g_f)) + U64 * abs(Df(b1) * Df(m)) + U64 * abs(Df(c1) * Df(g))
    # adamM2_error RHS (εv=εg=0):
    em2 = (U64 * abs(Df(b2v_f) + Df(c2gg_f)) + U64 * abs(Df(b2) * Df(v))
           + U64 * abs(Df(c2) * Df(gg_f)) + abs(Df(c2)) * (U64 * abs(Df(g) * Df(g))))
    # adamDir_error RHS (m→m1, v→m2, εm→em1, εv→em2, dmin = |toReal(√v'+ε)|):
    dmin = Df(den_f)
    sqrt_fac = (U64 * abs(Df(sq_f) + Df(eps)) + (U64 * Df(m2_f).sqrt() + em2.sqrt()))
    ed = U64 * abs(Df(m1_f) / Df(den_f)) + (em1 + abs(m1R / denR) * sqrt_fac) / dmin
    # adamStep_error RHS (εp=0):
    bound = U64 * abs(Df(p) - Df(ld_f)) + U64 * abs(Df(lr) * Df(dir_f)) + abs(Df(lr)) * ed

    err = abs(result - stepR)
    ok = err <= bound
    slack = float(err / bound) if bound != 0 else (0.0 if err == 0 else float("inf"))
    label = f"p={p:+.3g} m={m:+.3g} v={v:.3g} g={g:+.3g}"
    return ok, label, slack


def main(argv):
    if argv and argv[0] == "-":
        lines = sys.stdin.read().splitlines()
    else:
        subprocess.run(["lake", "build", "puffer"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        lines = subprocess.run([f"{ROOT}/.lake/build/bin/puffer", "verify-adam"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    fails, worst, worst_lbl, n = 0, 0.0, "", 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n += 1
        ok, label, slack = check(json.loads(line))
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:34s} slack={slack:7.2%}")
        if not ok:
            fails += 1
        if slack > worst:
            worst, worst_lbl = slack, label
    print(f"\n{n - fails}/{n} Adam steps within the proven bound"
          + (f"; tightest at ({worst_lbl}) used {worst:.2%}" if worst_lbl else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
