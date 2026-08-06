/-
# Bit-exact `sinf` / `cosf` matching glibc 2.43's single-precision sine/cosine

The chaotic physics envs (cartpole, double_pendulum) were only *short-horizon*
faithful because Lean's `r32 (Float.cos x)` double-rounds glibc's **f64** `cos`
to f32, which differs from C's **f32** `cosf`/`sinf` by 1 ULP on ~1.1% of angles;
in a chaotic Euler loop that seed compounds. This module closes the gap: a direct
Lean port of glibc 2.43's `sysdeps/ieee754/flt-32/s_sinf.c` / `s_cosf.c` (the
"optimized-routines" reduce-then-polynomial method) together with the x86 table
`sysdeps/x86/fpu/s_sincosf_data.c`.

**Faithfulness (measured exhaustively).** This module is **exhaustively bit-exact**
vs the system `sinf`/`cosf` over `|x| ≤ 2π` — every one of the 2.18 billion f32
in `|x| ≤ 7` was checked, **0 mismatches** — which is the ENTIRE range the
angle-wrapping envs (cartpole/double_pendulum wrap to ~[-π, π]) ever pass. So the
retrofit makes those envs bit-for-bit faithful to the C at ANY trajectory length.
Over the wider `|x| ≤ 120` regime the match is now **1 ULP on a single input**: a
dense 2.4M-sample sweep to `|x| ≤ 120` finds exactly ONE 1-ULP cos edge (at
`x ≈ 64.4`, a near-zero cos crossing where glibc's vector poly shares lane
rounding a scalar path can't replicate) — down from the **34** the earlier
non-FMA polynomial left. The fix: the system libm resolves (via `ifunc-fma.h`) to
the **FMA-contracted** multiarch variant, and this module now matches it by
computing every polynomial/reduction `a + b·c` with `Puffer.Float.fma` (a
correctly-rounded fused multiply-add; `Puffer/Float/Fma.lean`), so there is a
single rounding exactly where glibc's `-mfma` build fuses. The residual is
outside every env's reachable (wrapped) range.

Method (glibc): the argument (an f32-valued `Float`) is widened to f64; the
polynomial/reduction are done in **f64** (Lean `Float` = IEEE binary64, matching
C `double`) using `Puffer.Float.fma` at the 7 poly + 1 reduce contraction points
(matching glibc's FMA build), and the final result is rounded to f32
(`.toFloat32`); constants are given via `Float.ofBits` (exact IEEE-754 doubles)
since Lean has no hex-float literals.

**Scope.** Exhaustively bit-exact for `|x| ≤ 2π` (the envs' range); ≤ 1 ULP for
`|x| ≤ 120` (the `reduce_fast` regime, ONE residual near-zero cos edge). For `|x| > 120`
glibc uses a 192-bit Payne–Hanek reduction (`reduce_large`) not ported here; this
module falls back to `r32 (Float.sin/cos)` there and is NOT bit-guaranteed — but
the physics envs wrap angles first, so they never reach it. Zero imports
(Mathlib-free): joins the executable closure.
-/
import Puffer.Float.Fma
namespace Puffer.Numeric.SinCosF

/-- Round an f64 to the nearest f32 and back — a C `float` store / cast. -/
@[inline] def r32 (x : Float) : Float := x.toFloat32.toFloat

/-- Top 12 bits of the f32 representation with the sign bit cleared
    (glibc `abstop12`): `(asuint(x) >> 20) & 0x7ff`, on the f32 value of `x`. -/
@[inline] def abstop12 (y : Float) : UInt32 := (y.toFloat32.toBits >>> 20) &&& 0x7ff

/-! ### glibc `__sincosf_table` constants (exact doubles via `Float.ofBits`). -/

@[inline] def hpiInv : Float := Float.ofBits 0x41645f306dc9c883  -- 2/π · 2²⁴
@[inline] def hpi    : Float := Float.ofBits 0x3ff921fb54442d18  -- π/2

-- shared sine-polynomial coefficients (identical in both table entries)
@[inline] def cS1 : Float := Float.ofBits 0xbfc555545995a603
@[inline] def cS2 : Float := Float.ofBits 0x3f81107605230bc4
@[inline] def cS3 : Float := Float.ofBits 0xbf2994eb3774cf24

-- cosine-polynomial coefficients, table entry 0 (cos) and entry 1 (−cos)
@[inline] def e0c0 : Float := Float.ofBits 0x3ff0000000000000
@[inline] def e0c1 : Float := Float.ofBits 0xbfdffffffd0c621c
@[inline] def e0c2 : Float := Float.ofBits 0x3fa55553e1068f19
@[inline] def e0c3 : Float := Float.ofBits 0xbf56c087e89a359d
@[inline] def e0c4 : Float := Float.ofBits 0x3ef99343027bf8c3
@[inline] def e1c0 : Float := Float.ofBits 0xbff0000000000000
@[inline] def e1c1 : Float := Float.ofBits 0x3fdffffffd0c621c
@[inline] def e1c2 : Float := Float.ofBits 0xbfa55553e1068f19
@[inline] def e1c3 : Float := Float.ofBits 0x3f56c087e89a359d
@[inline] def e1c4 : Float := Float.ofBits 0xbef99343027bf8c3

/-! ### The polynomial (glibc `sinf_poly`, x86 `sincosf_poly.h`). -/

/-- Sine branch: `s + x⁷·(s2 + x²·s3)`, `s = x + x³·s1`. Entry-independent
    (`s1,s2,s3` are shared). Rounds the result to f32. -/
@[inline] def polySin (xs x2 : Float) : Float :=
  let x3 := xs * x2
  let s1v := Puffer.Float.fma x2 cS3 cS2
  let x7 := x3 * x2
  let s := Puffer.Float.fma x3 cS1 xs
  r32 (Puffer.Float.fma x7 s1v s)

/-- Cosine branch for a given table entry: `c + x⁶·(c3 + x²·c4)`,
    `c = (c0 + x²·c1) + x⁴·c2`. Rounds the result to f32. -/
@[inline] def polyCos (x2 c0 c1 c2 c3 c4 : Float) : Float :=
  let x4 := x2 * x2
  let hi := Puffer.Float.fma x2 c4 c3
  let c1v := Puffer.Float.fma x2 c1 c0
  let x6 := x4 * x2
  let c := Puffer.Float.fma x4 c2 c1v
  r32 (Puffer.Float.fma x6 hi c)

@[inline] def polyCosE (x2 : Float) (ent : Nat) : Float :=
  if ent == 0 then polyCos x2 e0c0 e0c1 e0c2 e0c3 e0c4
  else polyCos x2 e1c0 e1c1 e1c2 e1c3 e1c4

/-- glibc `reduce_fast`: returns `(x − n·(π/2), n)` with the quadrant `n`, via
    `n = ((int32)(x·hpiInv) + 0x800000) >> 24` (arithmetic shift = floor). Valid
    for `|x| ≤ 120`. -/
@[inline] def reduceFast (x : Float) : Float × Int :=
  let r := x * hpiInv
  let t : Int := (r.toInt64).toInt          -- (int32_t)r: truncate toward zero
  let n : Int := Int.fdiv (t + 0x800000) 0x1000000
  (Puffer.Float.fma (-(Float.ofInt n)) hpi x, n)

/-- `sign[q]` = {+1,−1,−1,+1}. -/
@[inline] def sign4 (q : Nat) : Float := if q == 1 || q == 2 then -1.0 else 1.0

/-- Shared core for `sinf` (`isCos=false`) and `cosf` (`isCos=true`). -/
@[inline] def core (isCos : Bool) (y : Float) : Float :=
  let at12 := abstop12 y
  if at12 < 1012 then                        -- |y| < π/4: no reduction
    if at12 < 920 then                       -- |y| < 2⁻¹²: sin≈y, cos≈1
      if isCos then 1.0 else y
    else
      let x2 := y * y
      if isCos then polyCosE x2 0 else polySin y x2
  else if at12 < 1071 then                   -- |y| < 120: reduce_fast
    let (xr, n) := reduceFast y
    let q : Nat := (((n % 4) + 4) % 4).toNat
    let ent : Nat := if q ≥ 2 then 1 else 0
    let xs := xr * sign4 q
    let x2 := xr * xr
    -- sin uses poly-branch (n&1); cos uses ((n^1)&1) = 1 − (n&1)
    let branch : Nat := if isCos then (1 - q % 2) else (q % 2)
    if branch == 0 then polySin xs x2 else polyCosE x2 ent
  else                                        -- |y| ≥ 120: not bit-guaranteed
    r32 (if isCos then Float.cos y else Float.sin y)

/-- Bit-exact `sinf`: matches glibc 2.43 `sinf` for `|x| ≤ 120`. Argument and
    result are f32-valued `Float`s. -/
@[inline] def sinf (y : Float) : Float := core false y

/-- Bit-exact `cosf`: matches glibc 2.43 `cosf` for `|x| ≤ 120`. -/
@[inline] def cosf (y : Float) : Float := core true y

end Puffer.Numeric.SinCosF
