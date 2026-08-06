#!/usr/bin/env python3
"""Cross-check Lean's proven per-op error bounds against an independent exact-ℝ reference.

`puffer verify` emits one JSON object per kernel case (dot / gae / nsscalar). Each value
is an IEEE-754 f64 *bit pattern* (a decimal uint64), so Python reconstructs the exact same
f64 Lean computed — no decimal-parsing drift. For each case we:

  1. rebuild the f64 inputs and Lean's f64 `result`;
  2. compute the IDEAL real value with `fractions.Fraction` (every kernel is rational — no
     transcendentals — so Fraction is EXACT ℝ, the same object the Lean spec layer reasons about);
  3. recompute the PROVEN ℝ bound EXACTLY (`dot_bnd_R`/`gae_bnd_R`/`nsscalar_bnd_R`/`z1ErrBnd_R`) —
     NOT the emitted Float mirror, which rounds and can fall slightly below the ℝ bound — so the
     assertion `|result − ideal| ≤ ℝbnd` IS the proven theorem (dotF_error / gaeHeadF_error /
     nsScalarF_error / neuron_error+logit_error) evaluated on the emitted inputs, with no seam.

This closes the trifecta at runtime: the Float exec layer (`result`), the ℝ spec layer
(`ideal`, via exact rationals), and the exact-ℝ proven bound meet on real numbers produced by
two independent implementations at two different precisions.

Usage:  tools/verify_ref.py            # builds+runs `puffer verify`, checks every case
        puffer verify | tools/verify_ref.py -   # check JSONL from stdin
"""
import json, os, struct, subprocess, sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f64(bitpattern):
    """Exact f64 from its uint64 bit pattern (matches Lean's Float.toBits round-trip)."""
    return struct.unpack("<d", struct.pack("<Q", int(bitpattern)))[0]


def frac(bitpattern):
    """The f64 as an EXACT rational (Fraction of a float is lossless)."""
    return Fraction(f64(bitpattern))


def ideal_dot(x, w):
    """Σ xᵢ·wᵢ in exact ℝ."""
    return sum((frac(a) * frac(b) for a, b in zip(x, w)), Fraction(0))


def ideal_gae(w, deltas):
    """Backward recurrence  A_last = δ_last,  Aᵢ = δᵢ + w·A_{i+1}  in exact ℝ."""
    wq = frac(w)
    acc = Fraction(0)
    for d in reversed(deltas):
        acc = frac(d) + wq * acc
    return acc


def ideal_nsscalar(a, b, c, sigma):
    """Newton–Schulz scalar map  σ·(a + b·σ² + c·σ⁴)  in exact ℝ."""
    s = frac(sigma)
    aq, bq, cq = frac(a), frac(b), frac(c)
    return s * (aq + bq * s * s + cq * (s * s) * (s * s))


def dotq(w, v):
    """Exact real dot Σ wᵢ·vᵢ of two Fraction vectors."""
    return sum((a * b for a, b in zip(w, v)), Fraction(0))


# ── EXACT-ℝ recomputation of the PROVEN Lean bounds ─────────────────────────────────────────
# The emitted "bound" field is the Float MIRROR (…ErrBndF), which rounds when it evaluates the
# bound formula and can come out slightly SMALLER than the ℝ bound the proof establishes. So we
# recompute the exact ℝ bound here: every `toReal(<Float subexpr>)` = the exact value of that
# subexpression evaluated in PYTHON FLOAT (Python float IS IEEE double = Lean Float), taken as a
# lossless Fraction; every bare `toReal x · toReal y` = the exact rational product; u64 = 2⁻⁵³.
# Then `|error| ≤ ℝbnd` IS the proven theorem (dotF_error / gaeHeadF_error / nsScalarF_error /
# neuron_error+logit_error) evaluated on the emitted inputs — no Float-mirror seam.
U64 = Fraction(1, 2 ** 53)


def dotF_float(x, w):
    """Lean `dotF` replicated in Python float: right-nested fl(x₀·w₀ + dotF xs ws)."""
    if not x or not w:
        return 0.0
    return x[0] * w[0] + dotF_float(x[1:], w[1:])


def dotErrBnd_R(w, x):
    """Exact ℝ `dotErrBnd w x` (w, x: Python-float lists)."""
    if not w or not x:
        return Fraction(0)
    return (U64 * abs(Fraction(w[0] * x[0]) + Fraction(dotF_float(w[1:], x[1:])))
            + U64 * abs(Fraction(w[0]) * Fraction(x[0])) + dotErrBnd_R(w[1:], x[1:]))


def dot_bnd_R(rec):
    return dotErrBnd_R([f64(b) for b in rec["x"]], [f64(b) for b in rec["w"]])


def gae_bnd_R(rec):
    """Exact ℝ `gaeErrBnd w deltas`."""
    w = f64(rec["w"]); d = [f64(b) for b in rec["deltas"]]
    heads = [0.0] * (len(d) + 1)
    for i in range(len(d) - 1, -1, -1):
        heads[i] = d[i] + w * heads[i + 1]                # gaeHeadF replica (float)
    wq, bnd = Fraction(w), Fraction(0)
    for i in range(len(d) - 1, -1, -1):
        A = heads[i + 1]
        bnd = (U64 * abs(Fraction(d[i]) + Fraction(w * A))        # add uses the ROUNDED float w·A
               + U64 * abs(wq * Fraction(A)) + abs(wq) * bnd)     # this uses the EXACT real w·A
    return bnd


def nsscalar_bnd_R(rec):
    """Exact ℝ `nsScalarErrBnd a b c σ` (the e2..eT2 chain)."""
    af, bf, cf, sf = (f64(rec[k]) for k in ("a", "b", "c", "sigma"))
    ss, = (sf * sf,); ss_ss = ss * ss; b_ss = bf * ss; c_ss4 = cf * ss_ss
    a_p_bss = af + b_ss; inner = a_p_bss + c_ss4
    aR, bR, cR, sR = Fraction(af), Fraction(bf), Fraction(cf), Fraction(sf)
    ssQ, ss_ssQ, b_ssQ, c_ss4Q = Fraction(ss), Fraction(ss_ss), Fraction(b_ss), Fraction(c_ss4)
    a_p_bssQ, innerQ, sRsR = Fraction(a_p_bss), Fraction(inner), sR * sR
    e2 = U64 * abs(sRsR)
    e4 = U64 * abs(ssQ * ssQ) + abs(ssQ) * e2 + abs(sRsR) * e2
    eBS = U64 * abs(bR * ssQ) + abs(bR) * e2
    eCS = U64 * abs(cR * ss_ssQ) + abs(cR) * e4
    eT1 = U64 * abs(aR + b_ssQ) + eBS
    eT2 = U64 * abs(a_p_bssQ + c_ss4Q) + eT1 + eCS
    return U64 * abs(sR * innerQ) + abs(sR) * eT2


def z1ErrBnd_R(w, b, x):
    """Exact ℝ `z1ErrBnd w b x` (forward linear-unit bound; w, x float lists, b float)."""
    return U64 * abs(Fraction(b) + Fraction(dotF_float(w, x))) + dotErrBnd_R(w, x)


def check_fwd(rec):
    """MLP forward pass (dot→bias→relu→dot→bias). The forward pass is rational, so Fraction
    is exact ℝ. Checks two PROVEN bounds against the EXACT-ℝ bound (recomputed, not the emitted
    Float mirror):
      (1) hidden neuron: |hFⱼ − relu(b1ⱼ + W1ⱼ·x)| ≤ z1ErrBnd(W1ⱼ, b1ⱼ, x)            [neuron_error]
      (2) output logit:  |logitFₖ − (b2ₖ + Σ W2ₖᵢ·hRᵢ)| ≤ z1ErrBnd(W2ₖ, b2ₖ, hF) + dotDiffₖ  [logit_error]
          where hR = ideal real hidden and dotDiffₖ = Σ |W2ₖᵢ|·|hFᵢ − hRᵢ| (folds hidden error in).
    Returns (ok, label, worst_ratio, detail)."""
    W1f = [[f64(v) for v in row] for row in rec["W1"]]; b1f = [f64(v) for v in rec["b1"]]
    W2f = [[f64(v) for v in row] for row in rec["W2"]]; b2f = [f64(v) for v in rec["b2"]]
    xf = [f64(v) for v in rec["x"]]; hFf = [f64(v) for v in rec["hidden"]]
    W1 = [[Fraction(v) for v in row] for row in W1f]; b1 = [Fraction(v) for v in b1f]
    W2 = [[Fraction(v) for v in row] for row in W2f]; b2 = [Fraction(v) for v in b2f]
    x = [Fraction(v) for v in xf]; hF = [Fraction(v) for v in hFf]
    logitF = [frac(v) for v in rec["logits"]]
    hR = [max(b1[j] + dotq(W1[j], x), Fraction(0)) for j in range(len(b1))]  # ideal real hidden

    worst, fails = 0.0, 0

    def track(err, bnd):
        nonlocal worst, fails
        ok = err <= bnd
        if not ok:
            fails += 1
        r = float(err / bnd) if bnd != 0 else (0.0 if err == 0 else float("inf"))
        if r > worst:
            worst = r

    for j in range(len(hF)):                          # (1) hidden neuron bound (exact ℝ)
        track(abs(hF[j] - hR[j]), z1ErrBnd_R(W1f[j], b1f[j], xf))
    for k in range(len(logitF)):                      # (2) end-to-end logit bound (exact ℝ)
        ideal_real = b2[k] + dotq(W2[k], hR)
        dot_diff = sum((abs(W2[k][i]) * abs(hF[i] - hR[i]) for i in range(len(hF))), Fraction(0))
        track(abs(logitF[k] - ideal_real), z1ErrBnd_R(W2f[k], b2f[k], hFf) + dot_diff)

    label = f"fwd  {len(hF)}h/{len(logitF)}o"
    detail = f"{len(hF)+len(logitF)} checks  worst slack={worst:6.1%}"
    return fails == 0, label, worst, detail


def check(rec):
    """Return (ok, label, err_ratio, detail) for one emitted case."""
    op = rec["op"]
    if op == "dot":
        result, bound, ideal = frac(rec["result"]), dot_bnd_R(rec), ideal_dot(rec["x"], rec["w"])
        label = f"dot   n={len(rec['x']):2d}"
    elif op == "gae":
        result, bound, ideal = frac(rec["result"]), gae_bnd_R(rec), ideal_gae(rec["w"], rec["deltas"])
        label = f"gae   n={len(rec['deltas']):2d}"
    elif op == "nsscalar":
        result, bound = frac(rec["result"]), nsscalar_bnd_R(rec)
        ideal = ideal_nsscalar(rec["a"], rec["b"], rec["c"], rec["sigma"])
        label = f"nsscalar σ={f64(rec['sigma']):+.4f}"
    elif op == "fwd":
        return check_fwd(rec)
    else:
        raise ValueError(f"unknown op {op!r}")
    err = abs(result - ideal)
    ok = err <= bound
    ratio = float(err / bound) if bound != 0 else (0.0 if err == 0 else float("inf"))
    detail = f"|err|={float(err):.3e}  bound={float(bound):.3e}  slack={ratio:6.1%}"
    return ok, label, ratio, detail


def main(argv):
    if argv and argv[0] == "-":
        lines = sys.stdin.read().splitlines()
    else:
        subprocess.run(["lake", "build", "puffer"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        out = subprocess.run([f"{ROOT}/.lake/build/bin/puffer", "verify"],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        lines = out.stdout.splitlines()

    fails = worst = 0
    worst_line = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ok, label, ratio, detail = check(json.loads(line))
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {label:22s} {detail}")
        if not ok:
            fails += 1
        if ratio > worst:
            worst, worst_line = ratio, label
    n = sum(1 for l in lines if l.strip())
    print(f"\n{n - fails}/{n} within proven bound"
          + (f"; tightest headroom at {worst_line} (used {worst:.1%} of bound)" if worst_line else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
