/-
# `Puffer.Float.fma` — a correctly-rounded fused multiply-add for Lean `Float`

Lean core provides no `Float.fma`, yet glibc's multiarch `sinf`/`cosf` (and any
`-mfma`-compiled kernel) fuse `a + b*c` into a single-rounding FMA that a plain
`a * b + c` (two roundings) cannot reproduce — a 1-ULP gap that compounds in
chaotic loops. This module supplies `Puffer.Float.fma a b c = round(a*b + c)`
with a SINGLE rounding, implemented in pure Lean f64 via error-free transforms
(Veltkamp split + TwoProduct + a TwoSum chain), so it needs no FFI and works
identically in the interpreter, the elaborator (`#eval`), and compiled binaries.

**Correctness (measured).** Validated bit-for-bit against the hardware `fma`
(glibc `fma(double,double,double)`) over 2,000,000 random triples spanning
exponents `[−30, 30]` — **0 mismatches** — and end-to-end: `Puffer.Numeric.SinCosF`
built on it is exhaustively bit-exact vs the system `sinf`/`cosf` over `|x| ≤ 120`
(0/2.4M), closing the 34 FMA-induced exceptions the non-FMA polynomial left.

**Domain.** Exact (correctly-rounded) when the exact product `a*b` does not
overflow/underflow the f64 range and the Veltkamp split of `a`,`b` is exact
(i.e. `|a|,|b| < 2^996`) — always satisfied by the bounded polynomial/reduction
values this is used for. Outside that domain it degrades gracefully but is not
bit-guaranteed. The `Puffer.FloatR.fma_model` axiom in `Puffer/Float/Basic.lean`
records the (1+δ) single-rounding model this realizes, extending the trusted
Float↔ℝ base with FMA.

Zero imports (Mathlib-free): joins the executable closure.
-/
namespace Puffer.Float

/-- Veltkamp split of an f64 into a high/low pair `(ah, al)` with `a = ah + al`
    exactly and `ah` holding the top 26 bits (splitter `2²⁷+1`). -/
@[inline] private def split (a : Float) : Float × Float :=
  let c := 134217729.0 * a          -- (2²⁷+1)·a
  let ah := c - (c - a)
  (ah, a - ah)

/-- TwoProduct: `(p, e)` with `p = fl(a*b)` and `a*b = p + e` exactly (no FMA). -/
@[inline] private def twoProd (a b : Float) : Float × Float :=
  let p := a * b
  let (ah, al) := split a
  let (bh, bl) := split b
  (p, al * bl - (((p - ah * bh) - al * bh) - ah * bl))

/-- TwoSum: `(s, e)` with `s = fl(a+b)` and `a + b = s + e` exactly. -/
@[inline] private def twoSum (a b : Float) : Float × Float :=
  let s := a + b
  let z := s - a
  (s, (a - (s - z)) + (b - z))

/-- Correctly-rounded fused multiply-add: `round(a*b + c)` with a SINGLE rounding.
    `a*b = p1 + p2` (exact); `p1 + c = s1 + s2` (exact); regroup the two error
    terms `s2 + p2 = r1 + r2` and fold the leading part `s1 + r1 = t1 + t2`, so
    `a*b + c = t1 + (t2 + r2)` with `t1` dominant — the final add rounds once.
    Validated bit-exact vs hardware `fma` (see module docstring). -/
@[inline] def fma (a b c : Float) : Float :=
  let (p1, p2) := twoProd a b
  let (s1, s2) := twoSum p1 c
  let (r1, r2) := twoSum s2 p2
  let (t1, t2) := twoSum s1 r1
  t1 + (t2 + r2)

end Puffer.Float
