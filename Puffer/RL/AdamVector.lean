/-
# Vector (whole-parameter-array) Adam step — the piece scalar `AdamStep` lacks vs Muon's vector update

`Puffer/RL/AdamStep.lean` gives the SCALAR Adam trifecta: one weight's update `adamStepF`, its ℝ
meaning `adamStepR`, and the op-by-op error bridge `adamStepF_error` (plus the bias-corrected
`adamStepBcF`/`adamStepBcF_error`). Real Adam keeps per-weight moment state and updates a whole
parameter array; Muon's train step already operates on a weight list. This module lifts Adam to
arrays, mirroring `BackwardFinite.updateVec`'s house style but over the four coupled arrays Adam
needs (params `p`, 1st-moment `m`, 2nd-moment `v`, gradient `g`).

**Representation.** A parameter array is a `List (Float × Float × Float × Float)` of per-coordinate
tuples `(p, m, v, g)` — the moment state travels WITH each weight, so one `List.map` of the scalar
`adamStepF` is the whole-array update (`updateVec` uses `zipWith` for its two arrays; four coupled
arrays are cleanest as one list of tuples, and `List.map` gives clean `Forall₂` reasoning). The
bias-corrected variant `adamStepBcVecF` adds the two runtime rescale constants `c₁ₜ`,`c₂ₜ`.

**The bound.** `adamStepVec_error` is the per-coordinate (L∞) statement: as a `List.Forall₂`
between the array and its ℝ mirror, EVERY coordinate is within the exact scalar `adamStepF_error`
bound of its ℝ meaning. `εp`,`εd` are a single UNIFORM (common-budget) pair: the `Forall₂` hypothesis
requires every coordinate's parameter error to be `≤ εp` and every coordinate's Adam-direction error
to be `≤ εd` — an L∞ budget, not a heterogeneous `εp_i`/`εd_i` per coordinate (a heterogeneous
version would take `εp εd : (coordinate) → ℝ`; the uniform form is the stronger hypothesis, so the
theorem it yields is safely weaker). Proof: `List.Forall₂.imp` lifts the pointwise scalar bound — the
per-coordinate input-error relation (param `≤ εp` + Adam-direction `≤ εd`) implies the output bound on
that coordinate by `adamStepF_error`. `adamStepBcVec_error` is the same for the bias-corrected step.
`forall₂_iff_get` converts either to the indexed `∀ i, |…[i] − …[i]| ≤ …` form, and `adamStepVecF_get`
rewrites `…[i]` through the vector def `adamStepVecF` itself.

**Scope (honestly disclosed, same convention as `AdamStep`).** The uniform direction-error budget
`εd` is taken as a hypothesis (exactly as scalar `adamStepF_error` does — it is itself dischargeable
by `adamM1_error`/`adamM2_error`→`adamDir_error` with a denominator floor, per coordinate). The
moment state `m`,`v`,`g`,`p` carries input error; every hyper-parameter (`β₁`,`β₂`,`c₁`,`c₂`,`lr`,
`ε`, and the bias divisors `c₁ₜ`,`c₂ₜ`) is EXACT — its `toReal` IS the ℝ value. Axiom-clean beyond
the trusted Float base (the bound reuses only the scalar theorems).
-/
import Puffer.RL.AdamStep

namespace Puffer.RL.AdamVector

open Puffer.FloatR (toReal u64)
open Puffer.RL.AdamStep

/-! ### Vector executable step + ℝ meaning

A coordinate tuple is `(p, m, v, g)`: `.1` = param, `.2.1` = 1st moment, `.2.2.1` = 2nd moment,
`.2.2.2` = gradient. -/

/-- The whole-array raw Adam update: `adamStepF` mapped over every `(p,m,v,g)` coordinate. -/
def adamStepVecF (cs : List (Float × Float × Float × Float)) (lr b1 c1 b2 c2 eps : Float) :
    List Float :=
  cs.map (fun c => adamStepF c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps)

/-- The whole-array bias-corrected Adam update (`m̂ = m'/c₁ₜ`, `v̂ = v'/c₂ₜ`). -/
def adamStepBcVecF (cs : List (Float × Float × Float × Float)) (lr b1 c1 b2 c2 eps c1t c2t : Float) :
    List Float :=
  cs.map (fun c => adamStepBcF c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps c1t c2t)

/-- ℝ meaning of the whole-array raw Adam update. -/
noncomputable def adamStepVecR (csR : List (ℝ × ℝ × ℝ × ℝ)) (lr b1 c1 b2 c2 eps : ℝ) : List ℝ :=
  csR.map (fun c => adamStepR c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps)

/-- ℝ meaning of the whole-array bias-corrected Adam update. -/
noncomputable def adamStepBcVecR (csR : List (ℝ × ℝ × ℝ × ℝ)) (lr b1 c1 b2 c2 eps c1t c2t : ℝ) :
    List ℝ :=
  csR.map (fun c => adamStepBcR c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps c1t c2t)

/-- The updated array has one output weight per input coordinate. -/
@[simp] theorem adamStepVecF_length (cs : List (Float × Float × Float × Float))
    (lr b1 c1 b2 c2 eps : Float) :
    (adamStepVecF cs lr b1 c1 b2 c2 eps).length = cs.length := by
  simp [adamStepVecF]

@[simp] theorem adamStepBcVecF_length (cs : List (Float × Float × Float × Float))
    (lr b1 c1 b2 c2 eps c1t c2t : Float) :
    (adamStepBcVecF cs lr b1 c1 b2 c2 eps c1t c2t).length = cs.length := by
  simp [adamStepBcVecF]

/-- Coordinate `i` of the updated array IS the scalar `adamStepF` on coordinate `i`'s own
    `(p,m,v,g)` — so the `Forall₂` bound below, read at index `i` via `forall₂_iff_get`, is a
    statement directly about `(adamStepVecF …)[i]`. -/
@[simp] theorem adamStepVecF_get (cs : List (Float × Float × Float × Float))
    (lr b1 c1 b2 c2 eps : Float) (i : Nat) (hi : i < cs.length) :
    (adamStepVecF cs lr b1 c1 b2 c2 eps)[i]'(by simpa using hi)
      = adamStepF cs[i].1 cs[i].2.1 cs[i].2.2.1 cs[i].2.2.2 lr b1 c1 b2 c2 eps := by
  simp [adamStepVecF]

/-! ### Per-coordinate error bound (the vector lift of `adamStepF_error`) -/

/-- **The vector Adam step accuracy bound.** As a `List.Forall₂` between the updated array and its
    ℝ mirror: given, per coordinate, the parameter error `εp` and that coordinate's Adam-direction
    error `εd`, EVERY output weight is within the exact scalar `adamStepF_error` bound of its ℝ
    meaning. Proved by `List.Forall₂.imp` — the pointwise scalar bound lifted over the array. Use
    `List.forall₂_iff_get` to read it as the indexed `∀ i, |out[i] − outR[i]| ≤ …` form. -/
theorem adamStepVec_error
    (cs : List (Float × Float × Float × Float)) (csR : List (ℝ × ℝ × ℝ × ℝ))
    (lr b1 c1 b2 c2 eps : Float) (εp εd : ℝ)
    (h : List.Forall₂ (fun c cR =>
        (|toReal c.1 - cR.1| ≤ εp) ∧
        (|toReal (adamDirF (adamM1F c.2.1 c.2.2.2 b1 c1) (adamM2F c.2.2.1 c.2.2.2 b2 c2) eps)
            - adamDirR (adamM1R cR.2.1 cR.2.2.2 (toReal b1) (toReal c1))
                (adamM2R cR.2.2.1 cR.2.2.2 (toReal b2) (toReal c2)) (toReal eps)| ≤ εd)) cs csR) :
    List.Forall₂ (fun c cR =>
        |toReal (adamStepF c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps)
          - adamStepR cR.1 cR.2.1 cR.2.2.1 cR.2.2.2 (toReal lr) (toReal b1) (toReal c1)
              (toReal b2) (toReal c2) (toReal eps)|
          ≤ u64 * |toReal c.1
                - toReal (lr * adamDirF (adamM1F c.2.1 c.2.2.2 b1 c1)
                    (adamM2F c.2.2.1 c.2.2.2 b2 c2) eps)| + εp
            + (u64 * |toReal lr * toReal (adamDirF (adamM1F c.2.1 c.2.2.2 b1 c1)
                (adamM2F c.2.2.1 c.2.2.2 b2 c2) eps)| + |toReal lr| * εd)) cs csR :=
  h.imp (fun c cR hc =>
    adamStepF_error c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps
      cR.1 cR.2.1 cR.2.2.1 cR.2.2.2 εp εd hc.1 hc.2)

/-- **The bias-corrected vector Adam step accuracy bound.** Same as `adamStepVec_error` for the
    bias-corrected update `p − lr·(m̂/(√v̂+ε))`, lifting the scalar `adamStepBcF_error` coordinatewise;
    the per-coordinate direction error `εd` is the bias-corrected one (`adamDirBc_error`). -/
theorem adamStepBcVec_error
    (cs : List (Float × Float × Float × Float)) (csR : List (ℝ × ℝ × ℝ × ℝ))
    (lr b1 c1 b2 c2 eps c1t c2t : Float) (εp εd : ℝ)
    (h : List.Forall₂ (fun c cR =>
        (|toReal c.1 - cR.1| ≤ εp) ∧
        (|toReal (adamDirF (adamM1HatF (adamM1F c.2.1 c.2.2.2 b1 c1) c1t)
              (adamM2HatF (adamM2F c.2.2.1 c.2.2.2 b2 c2) c2t) eps)
            - adamDirR (adamM1HatR (adamM1R cR.2.1 cR.2.2.2 (toReal b1) (toReal c1)) (toReal c1t))
                (adamM2HatR (adamM2R cR.2.2.1 cR.2.2.2 (toReal b2) (toReal c2)) (toReal c2t))
                (toReal eps)| ≤ εd)) cs csR) :
    List.Forall₂ (fun c cR =>
        |toReal (adamStepBcF c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps c1t c2t)
          - adamStepBcR cR.1 cR.2.1 cR.2.2.1 cR.2.2.2 (toReal lr) (toReal b1) (toReal c1)
              (toReal b2) (toReal c2) (toReal eps) (toReal c1t) (toReal c2t)|
          ≤ u64 * |toReal c.1
                - toReal (lr * adamDirF (adamM1HatF (adamM1F c.2.1 c.2.2.2 b1 c1) c1t)
                    (adamM2HatF (adamM2F c.2.2.1 c.2.2.2 b2 c2) c2t) eps)| + εp
            + (u64 * |toReal lr * toReal (adamDirF (adamM1HatF (adamM1F c.2.1 c.2.2.2 b1 c1) c1t)
                (adamM2HatF (adamM2F c.2.2.1 c.2.2.2 b2 c2) c2t) eps)| + |toReal lr| * εd)) cs csR :=
  h.imp (fun c cR hc =>
    adamStepBcF_error c.1 c.2.1 c.2.2.1 c.2.2.2 lr b1 c1 b2 c2 eps c1t c2t
      cR.1 cR.2.1 cR.2.2.1 cR.2.2.2 εp εd hc.1 hc.2)

/-! ### Demo: one whole-array Adam step (real hyper-parameters β₁=0.9, β₂=0.999, lr=1e-3, ε=1e-8) -/

-- two weights `p=[0.5, −0.3]`, zero moment state, gradient `g=[0.1, −0.2]`; each weight moves a
-- learning-rate-sized step opposite its gradient (ε-floored direction ≈ ±1).
/-- info: [0.496838, -0.296838] -/
#guard_msgs in #eval adamStepVecF [(0.5, 0, 0, 0.1), (-0.3, 0, 0, -0.2)] 1e-3 0.9 0.1 0.999 0.001 1e-8

-- bias-corrected first step (t = 1): c₁ₜ = 1−β₁¹ = 0.1, c₂ₜ = 1−β₂¹ = 0.001.
/-- info: [0.499000, -0.299000] -/
#guard_msgs in
  #eval adamStepBcVecF [(0.5, 0, 0, 0.1), (-0.3, 0, 0, -0.2)] 1e-3 0.9 0.1 0.999 0.001 1e-8 0.1 0.001

end Puffer.RL.AdamVector
