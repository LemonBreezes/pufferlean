/-
The Adam optimizer step — one parameter update, with a proven Float↔ℝ accuracy bound.

Muon (the optimizer PufferLib actually ships) has its full trifecta here (ℝ convergence, runnable Float,
bounded error). This file adds a STANDALONE textbook Adam step — not a PufferLib code path (PufferLib exposes
no Adam; its `beta1/beta2` fields are a vestigial API, `beta1` reused as Muon's momentum) but the field's
default optimizer and a natural piece of a verified-ML-in-Lean stack. It supplies: the executable single-
parameter Adam update over `Float`, its ℝ meaning, and a machine-checked bound `|toReal(exec) − ℝ| ≤ ε`
built op-by-op through the update circuit (the `nsScalarF_error` technique: `mul/add/sub/div/sqrtApprox_error`
composed, with the hyper-parameters exact and the state `m`,`v`,`g`,`p` carrying their input errors).

The circuit, per weight (`m`,`v` = the PREVIOUS moment state; `c₁ = 1−β₁`, `c₂ = 1−β₂` precomputed constants):
  m' = β₁·m + c₁·g            (1st moment)
  v' = β₂·v + c₂·(g·g)        (2nd moment)
  Δ  = lr · (m' / (√v' + ε))  (Adam direction)
  p' = p − Δ                  (update)
`adamStep_error` bounds `p'`; `adamM1_error`/`adamM2_error`/`adamDir_error` are the reusable circuit stages.
`divApprox_error` needs a positive denominator floor `dmin ≤ |toReal(√v' + ε)|`, supplied as a hypothesis:
with `ε > 0` and `v' ≥ 0` one exists (e.g. `dmin = ε·(1−u64)` — the raw `ε` is NOT provable, since the Float
`√v'+ε` carries an outer `(1+δ)` and can round a hair below `ε`). Axiom-clean beyond the trusted Float base.

The RAW form (`adamStepF`) is Adam's `t → ∞` limit (and what several implementations use); the BIAS-CORRECTED
form (`adamStepBcF`, the second half of this file) applies the standard rescale `m̂ = m'/(1−β₁ᵗ)`,
`v̂ = v'/(1−β₂ᵗ)` — each an exact scalar divide by the constant `c₁ₜ = 1−β₁ᵗ`, `c₂ₜ = 1−β₂ᵗ`, bounded by ONE
more `divApprox_error` (`adamM1Hat_error`/`adamM2Hat_error`) that feeds straight into the reused `adamDirF`.
`adamDirBc_error` composes the two rescales onto `adamDir_error` (bias-corrected direction error from the raw
moment errors); `adamStepBcF_error` closes the bias-corrected update. The rescale divisors `c₁ₜ`,`c₂ₜ` need a
positive denominator floor (`dmin ≤ |toReal c₁ₜ|`; `c₁ₜ = 1−β₁ᵗ ∈ (0,1]` for `β₁ ∈ [0,1)`, `t ≥ 1`), supplied
as a hypothesis like `ε`'s.

The hyper-parameters `β₁`,`β₂`,`c₁`,`c₂`,`lr`,`ε` (fixed literals) and `c₁ₜ`,`c₂ₜ` (the bias-correction
divisors `1−βᵗ`, runtime-computed) are all treated as EXACT — their `toReal` IS taken as the ℝ value (error
0), so the ℝ reference divides by the SAME rounded `c₁ₜ`,`c₂ₜ`, not an idealized `1−βᵗ`; only the accumulated
state `m`,`v`,`g`,`p` carries error.
-/
import Puffer.Float.Basic

namespace Puffer.RL.AdamStep

open Puffer.FloatR

/-! ### Executable Float step + ℝ meaning -/

/-- 1st-moment update `β₁·m + c₁·g` (`c₁ = 1−β₁`). -/
def adamM1F (m g b1 c1 : Float) : Float := b1 * m + c1 * g
/-- 2nd-moment update `β₂·v + c₂·g²` (`c₂ = 1−β₂`). -/
def adamM2F (v g b2 c2 : Float) : Float := b2 * v + c2 * (g * g)
/-- Adam direction `m / (√v + ε)`. -/
def adamDirF (m v eps : Float) : Float := m / (Float.sqrt v + eps)
/-- One Adam parameter update `p − lr·(m'/(√v'+ε))`. -/
def adamStepF (p m v g lr b1 c1 b2 c2 eps : Float) : Float :=
  p - lr * adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps

/-- ℝ 1st-moment update. -/
noncomputable def adamM1R (m g b1 c1 : ℝ) : ℝ := b1 * m + c1 * g
/-- ℝ 2nd-moment update. -/
noncomputable def adamM2R (v g b2 c2 : ℝ) : ℝ := b2 * v + c2 * (g * g)
/-- ℝ Adam direction. -/
noncomputable def adamDirR (m v eps : ℝ) : ℝ := m / (Real.sqrt v + eps)

/-! ### Structural ℝ properties of the Adam step -/

/-- **The Adam second-moment estimate stays nonnegative.** For nonnegative inputs (`v ≥ 0`, decay `β₂ ≥ 0`,
    weight `c₂ ≥ 0`), the updated second moment `β₂·v + c₂·g²` is `≥ 0` — the running variance estimate never goes
    negative, so `√v'` in the direction is always well-defined (the `g²` term is `≥ 0` for any gradient sign). -/
theorem adamM2R_nonneg (v g b2 c2 : ℝ) (hv : 0 ≤ v) (hb2 : 0 ≤ b2) (hc2 : 0 ≤ c2) :
    0 ≤ adamM2R v g b2 c2 := by
  unfold adamM2R
  have h1 : 0 ≤ b2 * v := mul_nonneg hb2 hv
  have h2 : 0 ≤ c2 * (g * g) := mul_nonneg hc2 (mul_self_nonneg g)
  linarith

/-- **The Adam direction is bounded by `|m|/ε`.** `|m/(√v + ε)| ≤ |m|/ε` for `ε > 0` — since `√v ≥ 0` the
    denominator is at least `ε`, so the `ε`-floor caps the update magnitude regardless of how small the variance
    estimate `v` is. This is why Adam's `ε` prevents the unbounded steps that a vanishing second moment would
    otherwise produce. -/
theorem adamDirR_abs_le (m v eps : ℝ) (heps : 0 < eps) :
    |adamDirR m v eps| ≤ |m| / eps := by
  unfold adamDirR
  have hden : 0 < Real.sqrt v + eps := by positivity
  rw [abs_div, abs_of_pos hden]
  gcongr
  linarith [Real.sqrt_nonneg v]

/-! ### Per-op circuit bounds -/

private theorem h0 (z : Float) : |toReal z - toReal z| ≤ 0 := by simp

/-- **1st-moment update error.** `|toReal(β₁·m + c₁·g) − (β₁·mR + c₁·gR)|`, hyper-params exact. -/
theorem adamM1_error (m g b1 c1 : Float) (mR gR εm εg : ℝ)
    (hm : |toReal m - mR| ≤ εm) (hg : |toReal g - gR| ≤ εg) :
    |toReal (adamM1F m g b1 c1) - adamM1R mR gR (toReal b1) (toReal c1)|
      ≤ u64 * |toReal (b1 * m) + toReal (c1 * g)|
        + (u64 * |toReal b1 * toReal m| + |toReal b1| * εm)
        + (u64 * |toReal c1 * toReal g| + |toReal c1| * εg) := by
  have hbm : |toReal (b1 * m) - toReal b1 * mR| ≤ u64 * |toReal b1 * toReal m| + |toReal b1| * εm := by
    simpa using mulApprox_error b1 m (toReal b1) mR 0 εm (h0 b1) hm
  have hcg : |toReal (c1 * g) - toReal c1 * gR| ≤ u64 * |toReal c1 * toReal g| + |toReal c1| * εg := by
    simpa using mulApprox_error c1 g (toReal c1) gR 0 εg (h0 c1) hg
  simpa [adamM1F, adamM1R] using
    addApprox_error (b1 * m) (c1 * g) (toReal b1 * mR) (toReal c1 * gR) _ _ hbm hcg

/-- **2nd-moment update error.** `|toReal(β₂·v + c₂·g²) − (β₂·vR + c₂·gR²)|`, hyper-params exact. -/
theorem adamM2_error (v g b2 c2 : Float) (vR gR εv εg : ℝ)
    (hv : |toReal v - vR| ≤ εv) (hg : |toReal g - gR| ≤ εg) :
    |toReal (adamM2F v g b2 c2) - adamM2R vR gR (toReal b2) (toReal c2)|
      ≤ u64 * |toReal (b2 * v) + toReal (c2 * (g * g))|
        + (u64 * |toReal b2 * toReal v| + |toReal b2| * εv)
        + (u64 * |toReal c2 * toReal (g * g)|
            + |toReal c2| * (u64 * |toReal g * toReal g| + |toReal g| * εg + |gR| * εg)) := by
  have hgg : |toReal (g * g) - gR * gR| ≤ u64 * |toReal g * toReal g| + |toReal g| * εg + |gR| * εg :=
    mulApprox_error g g gR gR εg εg hg hg
  have hbv : |toReal (b2 * v) - toReal b2 * vR| ≤ u64 * |toReal b2 * toReal v| + |toReal b2| * εv := by
    simpa using mulApprox_error b2 v (toReal b2) vR 0 εv (h0 b2) hv
  have hcgg : |toReal (c2 * (g * g)) - toReal c2 * (gR * gR)|
      ≤ u64 * |toReal c2 * toReal (g * g)|
        + |toReal c2| * (u64 * |toReal g * toReal g| + |toReal g| * εg + |gR| * εg) := by
    simpa using mulApprox_error c2 (g * g) (toReal c2) (gR * gR) 0
      (u64 * |toReal g * toReal g| + |toReal g| * εg + |gR| * εg) (h0 c2) hgg
  simpa [adamM2F, adamM2R] using
    addApprox_error (b2 * v) (c2 * (g * g)) (toReal b2 * vR) (toReal c2 * (gR * gR)) _ _ hbv hcgg

/-- **Adam-direction error** `m / (√v + ε)`. Combines `sqrtApprox_error` (the `√v`), `addApprox_error`
    (the `+ε` denominator) and `divApprox_error`; `dmin` is the caller-supplied positive denominator floor
    (`0 < dmin ≤ |toReal(√v + ε)|`; e.g. `ε·(1−u64)` when `ε > 0`, `v ≥ 0`). -/
theorem adamDir_error (m v eps : Float) (mR vR εm εv dmin : ℝ)
    (hm : |toReal m - mR| ≤ εm) (hv : |toReal v - vR| ≤ εv)
    (hvnn : 0 ≤ toReal v) (hvRnn : 0 ≤ vR)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt v + eps)|)
    (hyR : Real.sqrt vR + toReal eps ≠ 0) :
    |toReal (adamDirF m v eps) - adamDirR mR vR (toReal eps)|
      ≤ u64 * |toReal m / toReal (Float.sqrt v + eps)|
        + (εm + |mR / (Real.sqrt vR + toReal eps)|
              * (u64 * |toReal (Float.sqrt v) + toReal eps|
                  + (u64 * Real.sqrt (toReal v) + Real.sqrt εv))) / dmin := by
  have hsq : |toReal (Float.sqrt v) - Real.sqrt vR| ≤ u64 * Real.sqrt (toReal v) + Real.sqrt εv :=
    sqrtApprox_error v vR εv hv hvnn hvRnn
  have hden' : |toReal (Float.sqrt v + eps) - (Real.sqrt vR + toReal eps)|
      ≤ u64 * |toReal (Float.sqrt v) + toReal eps| + (u64 * Real.sqrt (toReal v) + Real.sqrt εv) := by
    simpa using addApprox_error (Float.sqrt v) eps (Real.sqrt vR) (toReal eps)
      (u64 * Real.sqrt (toReal v) + Real.sqrt εv) 0 hsq (h0 eps)
  simpa [adamDirF, adamDirR] using
    divApprox_error m (Float.sqrt v + eps) mR (Real.sqrt vR + toReal eps) εm
      (u64 * |toReal (Float.sqrt v) + toReal eps| + (u64 * Real.sqrt (toReal v) + Real.sqrt εv))
      dmin hm hden' hdmin hden hyR

/-- ℝ Adam parameter update `p − lr·(m'/(√v'+ε))`. -/
noncomputable def adamStepR (p m v g lr b1 c1 b2 c2 eps : ℝ) : ℝ :=
  p - lr * adamDirR (adamM1R m g b1 c1) (adamM2R v g b2 c2) eps

/-- **Adam step trust-region displacement bound.** One Adam update moves the parameter by at most
    `|lr|·|m'|/ε`: `|adamStepR − p| ≤ |lr|·|adamM1R m g b1 c1|/ε` for `ε > 0`. The displacement is the learning
    rate times the ε-floored direction (`adamDirR_abs_le`), so the ε-floor bounds how far a single step can move
    a weight regardless of the second moment — the parameter-space trust region of the optimizer, composing the
    step definition onto the direction bound (a182). -/
theorem adamStepR_dist_le (p m v g lr b1 c1 b2 c2 eps : ℝ) (heps : 0 < eps) :
    |adamStepR p m v g lr b1 c1 b2 c2 eps - p| ≤ |lr| * |adamM1R m g b1 c1| / eps := by
  unfold adamStepR
  have hrw : p - lr * adamDirR (adamM1R m g b1 c1) (adamM2R v g b2 c2) eps - p
      = -(lr * adamDirR (adamM1R m g b1 c1) (adamM2R v g b2 c2) eps) := by ring
  rw [hrw, abs_neg, abs_mul]
  have hd := adamDirR_abs_le (adamM1R m g b1 c1) (adamM2R v g b2 c2) eps heps
  calc |lr| * |adamDirR (adamM1R m g b1 c1) (adamM2R v g b2 c2) eps|
      ≤ |lr| * (|adamM1R m g b1 c1| / eps) := mul_le_mul_of_nonneg_left hd (abs_nonneg lr)
    _ = |lr| * |adamM1R m g b1 c1| / eps := by ring

/-- **Adam anti-alignment (descent-along-the-first-moment) law.** One Adam update displaces the parameter in
    the direction OPPOSITE to its accumulated first moment: the displacement `adamStepR − p` and the first-moment
    estimate `adamM1R m g b1 c1` have opposite signs, i.e. their product is `≤ 0`, for a nonnegative learning rate
    (`0 ≤ lr`) and a positive `ε` (`0 < eps`). This is the sign/directional counterpart of the magnitude bound
    `adamStepR_dist_le` — the structural guarantee that Adam moves each weight AGAINST its running momentum.
    Because `disp = −lr·(m'/(√v'+ε))`, we get `disp·m' = −lr·(m')²/(√v'+ε)`, which is `≤ 0` since `lr ≥ 0`,
    `(m')² ≥ 0`, and the denominator `√v'+ε > 0` (the `√·` term is nonnegative for ANY second-moment value `v'`,
    so `ε > 0` makes the denominator strictly positive regardless of `v'`). Both hypotheses are load-bearing:
    with `lr < 0` (e.g. `lr=−1, m'=1, ε=1e-8`) the product becomes `+1e8 > 0`, and with `ε` negative enough to
    flip the denominator's sign the product likewise turns positive whenever `m' ≠ 0`. -/
theorem adamStepR_descent (p m v g lr b1 c1 b2 c2 eps : ℝ) (hlr : 0 ≤ lr) (heps : 0 < eps) :
    (adamStepR p m v g lr b1 c1 b2 c2 eps - p) * adamM1R m g b1 c1 ≤ 0 := by
  unfold adamStepR adamDirR
  set m' := adamM1R m g b1 c1 with hm'
  set v' := adamM2R v g b2 c2 with hv'
  have hD : 0 < Real.sqrt v' + eps := by have := Real.sqrt_nonneg v'; linarith
  have key : (p - lr * (m' / (Real.sqrt v' + eps)) - p) * m'
      = - (lr * (m' * m' / (Real.sqrt v' + eps))) := by ring
  rw [key]
  have hnn : 0 ≤ lr * (m' * m' / (Real.sqrt v' + eps)) :=
    mul_nonneg hlr (div_nonneg (mul_self_nonneg m') hD.le)
  linarith

/-- **Adam outer-step error.** Given the direction error `εd`, one update `p − lr·dir` is within
    `u64·|…| + εp + (u64·|lr·dir| + |lr|·εd)` of the ideal — the `mul`+`sub` closing the circuit. Chain
    `adamM1_error`/`adamM2_error` → `adamDir_error` to supply `εd`; then `adamStepF` = `p − lr·dir`. -/
theorem adamStep_error (p lr dir : Float) (pR dirR εp εd : ℝ)
    (hp : |toReal p - pR| ≤ εp) (hd : |toReal dir - dirR| ≤ εd) :
    |toReal (p - lr * dir) - (pR - toReal lr * dirR)|
      ≤ u64 * |toReal p - toReal (lr * dir)| + εp
        + (u64 * |toReal lr * toReal dir| + |toReal lr| * εd) := by
  have hmul : |toReal (lr * dir) - toReal lr * dirR|
      ≤ u64 * |toReal lr * toReal dir| + |toReal lr| * εd := by
    simpa using mulApprox_error lr dir (toReal lr) dirR 0 εd (h0 lr) hd
  simpa using subApprox_error p (lr * dir) pR (toReal lr * dirR) εp
    (u64 * |toReal lr * toReal dir| + |toReal lr| * εd) hp hmul

/-- **The Adam step accuracy bound (end-to-end).** `|toReal(adamStepF …) − adamStepR …| ≤ …`, given the
    Adam-direction error `εd` (from `adamM1_error`/`adamM2_error` → `adamDir_error`) and the param error `εp`.
    Every hyper-parameter is exact; the whole update circuit's Float rounding is bounded against the ℝ step. -/
theorem adamStepF_error (p m v g lr b1 c1 b2 c2 eps : Float) (pR mR vR gR εp εd : ℝ)
    (hp : |toReal p - pR| ≤ εp)
    (hd : |toReal (adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps)
            - adamDirR (adamM1R mR gR (toReal b1) (toReal c1))
                (adamM2R vR gR (toReal b2) (toReal c2)) (toReal eps)| ≤ εd) :
    |toReal (adamStepF p m v g lr b1 c1 b2 c2 eps)
        - adamStepR pR mR vR gR (toReal lr) (toReal b1) (toReal c1) (toReal b2) (toReal c2) (toReal eps)|
      ≤ u64 * |toReal p - toReal (lr * adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps)| + εp
        + (u64 * |toReal lr * toReal (adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps)|
            + |toReal lr| * εd) := by
  simpa [adamStepF, adamStepR] using
    adamStep_error p lr (adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps) pR
      (adamDirR (adamM1R mR gR (toReal b1) (toReal c1)) (adamM2R vR gR (toReal b2) (toReal c2)) (toReal eps))
      εp εd hp hd

/-! ### Bias correction (`m̂ = m'/(1−β₁ᵗ)`, `v̂ = v'/(1−β₂ᵗ)`) — the standard Adam rescale -/

/-- Bias-corrected 1st moment `m̂ = m'/c₁ₜ` (`c₁ₜ = 1−β₁ᵗ`). -/
def adamM1HatF (m c1t : Float) : Float := m / c1t
/-- Bias-corrected 2nd moment `v̂ = v'/c₂ₜ` (`c₂ₜ = 1−β₂ᵗ`). -/
def adamM2HatF (v c2t : Float) : Float := v / c2t
/-- ℝ bias-corrected 1st moment. -/
noncomputable def adamM1HatR (m c1t : ℝ) : ℝ := m / c1t
/-- ℝ bias-corrected 2nd moment. -/
noncomputable def adamM2HatR (v c2t : ℝ) : ℝ := v / c2t

/-- **Bias-correct 1st moment error.** `m̂ = m'/c₁ₜ`, an exact scalar divide (`c₁ₜ` exact, error 0); one
    `divApprox_error` with the denominator floor `dmin1 ≤ |toReal c₁ₜ|` (`c₁ₜ = 1−β₁ᵗ > 0`). -/
theorem adamM1Hat_error (m c1t : Float) (mR εm dmin1 : ℝ)
    (hm : |toReal m - mR| ≤ εm)
    (hdmin : 0 < dmin1) (hden : dmin1 ≤ |toReal c1t|) (hyR : toReal c1t ≠ 0) :
    |toReal (adamM1HatF m c1t) - adamM1HatR mR (toReal c1t)|
      ≤ u64 * |toReal m / toReal c1t| + εm / dmin1 := by
  have h := divApprox_error m c1t mR (toReal c1t) εm 0 dmin1 hm (h0 c1t) hdmin hden hyR
  simpa [adamM1HatF, adamM1HatR] using h

/-- **Bias-correct 2nd moment error.** `v̂ = v'/c₂ₜ`, an exact scalar divide; one `divApprox_error`. -/
theorem adamM2Hat_error (v c2t : Float) (vR εv dmin2 : ℝ)
    (hv : |toReal v - vR| ≤ εv)
    (hdmin : 0 < dmin2) (hden : dmin2 ≤ |toReal c2t|) (hyR : toReal c2t ≠ 0) :
    |toReal (adamM2HatF v c2t) - adamM2HatR vR (toReal c2t)|
      ≤ u64 * |toReal v / toReal c2t| + εv / dmin2 := by
  have h := divApprox_error v c2t vR (toReal c2t) εv 0 dmin2 hv (h0 c2t) hdmin hden hyR
  simpa [adamM2HatF, adamM2HatR] using h

/-- **Bias-corrected Adam-direction error** `m̂/(√v̂+ε)`. Composes the two `Hat` rescales onto
    `adamDir_error`, producing the bias-corrected direction error from the RAW moment-output errors `εm'`,
    `εv'` (from `adamM1_error`/`adamM2_error`). `dmin1`/`dmin2` floor the rescale divisors, `dmin` the
    `√v̂+ε` denominator. -/
theorem adamDirBc_error (m' v' eps c1t c2t : Float) (m'R v'R εm' εv' dmin1 dmin2 dmin : ℝ)
    (hm' : |toReal m' - m'R| ≤ εm') (hv' : |toReal v' - v'R| ≤ εv')
    (hd1min : 0 < dmin1) (hd1 : dmin1 ≤ |toReal c1t|) (hc1t : toReal c1t ≠ 0)
    (hd2min : 0 < dmin2) (hd2 : dmin2 ≤ |toReal c2t|) (hc2t : toReal c2t ≠ 0)
    (hvhatnn : 0 ≤ toReal (adamM2HatF v' c2t)) (hvhatRnn : 0 ≤ adamM2HatR v'R (toReal c2t))
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt (adamM2HatF v' c2t) + eps)|)
    (hyR : Real.sqrt (adamM2HatR v'R (toReal c2t)) + toReal eps ≠ 0) :
    |toReal (adamDirF (adamM1HatF m' c1t) (adamM2HatF v' c2t) eps)
        - adamDirR (adamM1HatR m'R (toReal c1t)) (adamM2HatR v'R (toReal c2t)) (toReal eps)|
      ≤ u64 * |toReal (adamM1HatF m' c1t) / toReal (Float.sqrt (adamM2HatF v' c2t) + eps)|
        + ((u64 * |toReal m' / toReal c1t| + εm' / dmin1)
              + |adamM1HatR m'R (toReal c1t) / (Real.sqrt (adamM2HatR v'R (toReal c2t)) + toReal eps)|
                * (u64 * |toReal (Float.sqrt (adamM2HatF v' c2t)) + toReal eps|
                    + (u64 * Real.sqrt (toReal (adamM2HatF v' c2t))
                        + Real.sqrt (u64 * |toReal v' / toReal c2t| + εv' / dmin2)))) / dmin := by
  have hmhat := adamM1Hat_error m' c1t m'R εm' dmin1 hm' hd1min hd1 hc1t
  have hvhat := adamM2Hat_error v' c2t v'R εv' dmin2 hv' hd2min hd2 hc2t
  exact adamDir_error (adamM1HatF m' c1t) (adamM2HatF v' c2t) eps
    (adamM1HatR m'R (toReal c1t)) (adamM2HatR v'R (toReal c2t))
    (u64 * |toReal m' / toReal c1t| + εm' / dmin1)
    (u64 * |toReal v' / toReal c2t| + εv' / dmin2)
    dmin hmhat hvhat hvhatnn hvhatRnn hdmin hden hyR

/-- Bias-corrected Float step `p − lr·(m̂/(√v̂+ε))`, `m̂ = m'/c₁ₜ`, `v̂ = v'/c₂ₜ`. -/
def adamStepBcF (p m v g lr b1 c1 b2 c2 eps c1t c2t : Float) : Float :=
  p - lr * adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t) (adamM2HatF (adamM2F v g b2 c2) c2t) eps

/-- ℝ bias-corrected step. -/
noncomputable def adamStepBcR (p m v g lr b1 c1 b2 c2 eps c1t c2t : ℝ) : ℝ :=
  p - lr * adamDirR (adamM1HatR (adamM1R m g b1 c1) c1t) (adamM2HatR (adamM2R v g b2 c2) c2t) eps

/-- **The bias-corrected Adam step accuracy bound (end-to-end).** Mirrors `adamStepF_error` for the
    bias-corrected update: given the bias-corrected direction error `εd` (from `adamM1_error`/`adamM2_error`
    → `adamDirBc_error`) and the param error `εp`, one update `p − lr·(m̂/(√v̂+ε))` is bounded against the
    ℝ step. -/
theorem adamStepBcF_error (p m v g lr b1 c1 b2 c2 eps c1t c2t : Float) (pR mR vR gR εp εd : ℝ)
    (hp : |toReal p - pR| ≤ εp)
    (hd : |toReal (adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t) (adamM2HatF (adamM2F v g b2 c2) c2t) eps)
            - adamDirR (adamM1HatR (adamM1R mR gR (toReal b1) (toReal c1)) (toReal c1t))
                (adamM2HatR (adamM2R vR gR (toReal b2) (toReal c2)) (toReal c2t)) (toReal eps)| ≤ εd) :
    |toReal (adamStepBcF p m v g lr b1 c1 b2 c2 eps c1t c2t)
        - adamStepBcR pR mR vR gR (toReal lr) (toReal b1) (toReal c1) (toReal b2) (toReal c2)
            (toReal eps) (toReal c1t) (toReal c2t)|
      ≤ u64 * |toReal p
              - toReal (lr * adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t)
                  (adamM2HatF (adamM2F v g b2 c2) c2t) eps)| + εp
        + (u64 * |toReal lr * toReal (adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t)
              (adamM2HatF (adamM2F v g b2 c2) c2t) eps)| + |toReal lr| * εd) := by
  simpa [adamStepBcF, adamStepBcR] using
    adamStep_error p lr (adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t)
        (adamM2HatF (adamM2F v g b2 c2) c2t) eps) pR
      (adamDirR (adamM1HatR (adamM1R mR gR (toReal b1) (toReal c1)) (toReal c1t))
        (adamM2HatR (adamM2R vR gR (toReal b2) (toReal c2)) (toReal c2t)) (toReal eps)) εp εd hp hd

end Puffer.RL.AdamStep
