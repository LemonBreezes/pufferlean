/-
# C93: Adam overflow-freedom — the finiteness certificate scalar `AdamStep` lacked

`AdamStep.lean` gives the scalar Adam step its full accuracy trifecta (Float exec, ℝ meaning,
op-by-op error bound), but nothing about *overflow*. Muon got both: `muon_train_step_all_finite`
(C61) and the runnable Bool capstone (C88). This file supplies the Adam analogue.

The Adam circuit, per weight (`m'`,`v'` the updated moments, `d = √v' + ε` the denominator):
  m' = β₁·m + c₁·g            v' = β₂·v + c₂·g²
  dir = m' / (√v' + ε)        p' = p − lr·dir

**The `ε`-floor is load-bearing.** `dir`'s magnitude is bounded by `|m'|/ε` regardless of how
small the second moment `v'` becomes — this is exactly why Adam's `ε` prevents unbounded steps
(the ℝ statement is `AdamStep.adamDirR_abs_le`). So the finiteness proof needs a positive
denominator floor `dmin ≤ |toReal (√v' + ε)|` — the SAME floor `adamDir_error` already requires
for the accuracy bound. Given that floor and magnitude bounds on the inputs, every stage's
`toReal` is `(1+u64)`-controlled and the final update is within `overflowBound`, so
`isFinite_of_bounded` (C43) closes it. **No new axiom** — the second moment / `√v'` never needs
an upper bound (the floor caps the quotient from below), so no `sqrt` finiteness fact is invoked.

* `adamStepF_mag_le` — `|toReal (adamStepF …)| ≤ adamMag …`, the closed-form propagated bound.
* `adamStepF_isFinite` — `adamMag … ≤ overflowBound → (adamStepF …).isFinite = true`.
* `adamStepCF` / `adamStepF_isFinite_runnable` — the C81/C88-style runnable Bool certificate:
  a Float mirror of `adamMag` (one `slackF` per rounding node, `slackF_key` domination). Its two
  genuine runtime Bool gates are `checkLe (adamStepCF …) capF` (the overflow budget) and the C89
  `minWdF`-style `checkLe adamMinF dminF` (the positive floor pin). The six input-magnitude
  bounds `|toReal x| ≤ toReal xF` are stated as ℝ inequalities (each Bool-*dischargeable* via
  `checkAbsLe_sound`, exactly as C81/C88 wrap theirs — left un-wrapped here so callers can supply
  either form), and the floor *coupling* `toReal dminF ≤ |toReal (√v'+ε)|` stays caller-side (it
  names the runtime denominator — there is no Float data to check it against, exactly as C89
  discloses for its ball couplings). So the overflow budget and the floor positivity — the only
  data not fixed by the input witnesses — are the Bool-gated part.

**Scope.** Scalar (one weight). The vector lift over a whole parameter array is `AdamVector`
(C92); this file's `isFinite` composes coordinatewise there. Slack compounds ≈0.1% per `slackF`
node — 6 nodes: 2 moment muls + 1 moment add + 1 direction div + 1 step mul + 1 step sub (≈0.6%).
-/
import Puffer.RL.AdamStep
import Puffer.RL.FiniteBound
import Puffer.RL.BudgetEval
import Puffer.RL.RunConstEval
import Puffer.RL.MarginCheck

namespace Puffer.RL.AdamFinite

open Puffer.FloatR (toReal u64 u64_pos u64_lt_one div_model mul_model toReal_ofScientific_close)
open Puffer.RL.AdamStep (adamM1F adamM2F adamDirF adamStepF)
open Puffer.RL.FiniteBound (overflowBound isFinite_of_bounded mul_mag_le add_mag_le)
open Puffer.RL.LossFinite (sub_mag_le)
open Puffer.RL.BudgetEval (slackF slackF_key slackF_nonneg capF capF_le)
open Puffer.RL.RunConstEval (unit_mul_dominates unit_add_dominates unit_scale_dominates)
open Puffer.RL.MarginCheck (checkLe checkLe_sound)

/-! ### The closed-form magnitude bound (ℝ) -/

/-- `(1+u64)` roundoff factor is nonnegative. -/
private theorem u1_nonneg : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith

/-- Propagated magnitude bound for the 1st moment `m' = β₁·m + c₁·g`. -/
noncomputable def adamM1Mag (Bm Bg Bb1 Bc1 : ℝ) : ℝ :=
  (1 + u64) * ((1 + u64) * (Bb1 * Bm) + (1 + u64) * (Bc1 * Bg))

/-- Propagated magnitude bound for the Adam direction `m'/(√v'+ε)` under a denominator floor `dmin`. -/
noncomputable def adamDirMag (Bm Bg Bb1 Bc1 dmin : ℝ) : ℝ :=
  (1 + u64) * (adamM1Mag Bm Bg Bb1 Bc1 / dmin)

/-- Propagated magnitude bound for the whole Adam step `p − lr·dir`. -/
noncomputable def adamMag (Bp Bm Bg Blr Bb1 Bc1 dmin : ℝ) : ℝ :=
  (1 + u64) * (Bp + (1 + u64) * (Blr * adamDirMag Bm Bg Bb1 Bc1 dmin))

/-- `adamM1Mag ≥ 0` for nonnegative operand bounds. -/
theorem adamM1Mag_nonneg (Bm Bg Bb1 Bc1 : ℝ)
    (hBm : 0 ≤ Bm) (hBg : 0 ≤ Bg) (hBb1 : 0 ≤ Bb1) (hBc1 : 0 ≤ Bc1) :
    0 ≤ adamM1Mag Bm Bg Bb1 Bc1 := by
  unfold adamM1Mag
  have := u64_pos
  have h1 : 0 ≤ Bb1 * Bm := mul_nonneg hBb1 hBm
  have h2 : 0 ≤ Bc1 * Bg := mul_nonneg hBc1 hBg
  positivity

/-! ### Per-stage magnitude propagation -/

/-- **1st-moment magnitude bound.** `|toReal (β₁·m + c₁·g)| ≤ adamM1Mag`. -/
theorem adamM1F_mag_le (m g b1 c1 : Float) (Bm Bg Bb1 Bc1 : ℝ)
    (hm : |toReal m| ≤ Bm) (hg : |toReal g| ≤ Bg)
    (hb1 : |toReal b1| ≤ Bb1) (hc1 : |toReal c1| ≤ Bc1)
    (_hBm : 0 ≤ Bm) (_hBg : 0 ≤ Bg) (hBb1 : 0 ≤ Bb1) (hBc1 : 0 ≤ Bc1) :
    |toReal (adamM1F m g b1 c1)| ≤ adamM1Mag Bm Bg Bb1 Bc1 := by
  unfold adamM1F adamM1Mag
  have hbm : |toReal (b1 * m)| ≤ (1 + u64) * (Bb1 * Bm) :=
    (mul_mag_le b1 m).trans
      (mul_le_mul_of_nonneg_left (mul_le_mul hb1 hm (abs_nonneg _) hBb1) u1_nonneg)
  have hcg : |toReal (c1 * g)| ≤ (1 + u64) * (Bc1 * Bg) :=
    (mul_mag_le c1 g).trans
      (mul_le_mul_of_nonneg_left (mul_le_mul hc1 hg (abs_nonneg _) hBc1) u1_nonneg)
  calc |toReal (b1 * m + c1 * g)|
      ≤ (1 + u64) * (|toReal (b1 * m)| + |toReal (c1 * g)|) := add_mag_le _ _
    _ ≤ (1 + u64) * ((1 + u64) * (Bb1 * Bm) + (1 + u64) * (Bc1 * Bg)) :=
        mul_le_mul_of_nonneg_left (add_le_add hbm hcg) u1_nonneg

/-- **Adam-direction magnitude bound.** `|toReal (m'/(√v'+ε))| ≤ (1+u64)·(Am/dmin)` given the numerator
    bound `|toReal m'| ≤ Am` and the positive denominator floor `dmin ≤ |toReal (√v'+ε)|` — the `ε`-floor
    caps the quotient WITHOUT any upper bound on the second moment `v'` (the finiteness analogue of
    `AdamStep.adamDirR_abs_le`). -/
theorem adamDirF_mag_le (m' v' eps : Float) (Am dmin : ℝ)
    (hm' : |toReal m'| ≤ Am) (hAm : 0 ≤ Am)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt v' + eps)|) :
    |toReal (adamDirF m' v' eps)| ≤ (1 + u64) * (Am / dmin) := by
  unfold adamDirF
  obtain ⟨δ, hδ, hdiv⟩ := div_model m' (Float.sqrt v' + eps)
  rw [hdiv, abs_mul]
  have h1δ : |1 + δ| ≤ 1 + u64 := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  have hq : |toReal m' / toReal (Float.sqrt v' + eps)| ≤ Am / dmin := by
    rw [abs_div]
    gcongr
  calc |toReal m' / toReal (Float.sqrt v' + eps)| * |1 + δ|
      ≤ (Am / dmin) * (1 + u64) := mul_le_mul hq h1δ (abs_nonneg _) (div_nonneg hAm hdmin.le)
    _ = (1 + u64) * (Am / dmin) := by ring

/-- **Adam step magnitude bound (end-to-end).** `|toReal (adamStepF …)| ≤ adamMag …`, composing the
    moment / direction / update stages, each `(1+u64)`-controlled, under the `ε`-floor. -/
theorem adamStepF_mag_le (p m v g lr b1 c1 b2 c2 eps : Float)
    (Bp Bm Bg Blr Bb1 Bc1 dmin : ℝ)
    (hp : |toReal p| ≤ Bp) (hm : |toReal m| ≤ Bm) (hg : |toReal g| ≤ Bg)
    (hlr : |toReal lr| ≤ Blr) (hb1 : |toReal b1| ≤ Bb1) (hc1 : |toReal c1| ≤ Bc1)
    (_hBp : 0 ≤ Bp) (hBm : 0 ≤ Bm) (hBg : 0 ≤ Bg) (hBlr : 0 ≤ Blr) (hBb1 : 0 ≤ Bb1) (hBc1 : 0 ≤ Bc1)
    (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (Float.sqrt (adamM2F v g b2 c2) + eps)|) :
    |toReal (adamStepF p m v g lr b1 c1 b2 c2 eps)| ≤ adamMag Bp Bm Bg Blr Bb1 Bc1 dmin := by
  unfold adamStepF adamMag
  set dir := adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps with hdir
  have hm' : |toReal (adamM1F m g b1 c1)| ≤ adamM1Mag Bm Bg Bb1 Bc1 :=
    adamM1F_mag_le m g b1 c1 Bm Bg Bb1 Bc1 hm hg hb1 hc1 hBm hBg hBb1 hBc1
  have hAm : 0 ≤ adamM1Mag Bm Bg Bb1 Bc1 := adamM1Mag_nonneg _ _ _ _ hBm hBg hBb1 hBc1
  have hdir_le : |toReal dir| ≤ adamDirMag Bm Bg Bb1 Bc1 dmin := by
    unfold adamDirMag
    exact adamDirF_mag_le (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps
      (adamM1Mag Bm Bg Bb1 Bc1) dmin hm' hAm hdmin hden
  have hprod : |toReal (lr * dir)| ≤ (1 + u64) * (Blr * adamDirMag Bm Bg Bb1 Bc1 dmin) :=
    (mul_mag_le lr dir).trans
      (mul_le_mul_of_nonneg_left (mul_le_mul hlr hdir_le (abs_nonneg _) hBlr) u1_nonneg)
  calc |toReal (p - lr * dir)|
      ≤ (1 + u64) * (|toReal p| + |toReal (lr * dir)|) := sub_mag_le _ _
    _ ≤ (1 + u64) * (Bp + (1 + u64) * (Blr * adamDirMag Bm Bg Bb1 Bc1 dmin)) :=
        mul_le_mul_of_nonneg_left (add_le_add hp hprod) u1_nonneg

/-- **THE ADAM NO-OVERFLOW CERTIFICATE.** One Adam step over magnitude-bounded inputs, with the positive
    denominator floor `dmin` (the `ε`-floor `adamDir_error` also needs), whose propagated bound stays
    within `overflowBound`, is overflow-free. The magnitude bound is PROVED (`adamStepF_mag_le`); the
    finiteness step is C43's one axiom (`isFinite_of_bounded`). No new axiom. -/
theorem adamStepF_isFinite (p m v g lr b1 c1 b2 c2 eps : Float)
    (Bp Bm Bg Blr Bb1 Bc1 dmin : ℝ)
    (hp : |toReal p| ≤ Bp) (hm : |toReal m| ≤ Bm) (hg : |toReal g| ≤ Bg)
    (hlr : |toReal lr| ≤ Blr) (hb1 : |toReal b1| ≤ Bb1) (hc1 : |toReal c1| ≤ Bc1)
    (hBp : 0 ≤ Bp) (hBm : 0 ≤ Bm) (hBg : 0 ≤ Bg) (hBlr : 0 ≤ Blr) (hBb1 : 0 ≤ Bb1) (hBc1 : 0 ≤ Bc1)
    (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (Float.sqrt (adamM2F v g b2 c2) + eps)|)
    (hbound : adamMag Bp Bm Bg Blr Bb1 Bc1 dmin ≤ overflowBound) :
    (adamStepF p m v g lr b1 c1 b2 c2 eps).isFinite = true :=
  isFinite_of_bounded _
    ((adamStepF_mag_le p m v g lr b1 c1 b2 c2 eps Bp Bm Bg Blr Bb1 Bc1 dmin
      hp hm hg hlr hb1 hc1 hBp hBm hBg hBlr hBb1 hBc1 hdmin hden).trans hbound)

/-! ### The runnable Bool certificate (C81/C88 style) -/

/-- **Division unit** (the C83/C89 direction-reversal): `(1+u64)·(x/y) ≤ toReal (slackF · (xF/yF))`
    with the numerator dominated above (`x ≤ toReal xF`) and the denominator by a positive LOWER
    witness (`0 < toReal yF ≤ y`). One `slackF` absorbs the div rounding + the outer mul rounding
    (`slackF_key`). -/
theorem unit_div_dominates (xF yF : Float) (x y : ℝ)
    (hx0 : 0 ≤ x) (hxf : x ≤ toReal xF) (hyF0 : 0 < toReal yF) (hyf : toReal yF ≤ y) :
    (1 + u64) * (x / y) ≤ toReal (slackF * (xF / yF)) := by
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hs := slackF_nonneg
  obtain ⟨δ₁, hδ₁, e₁⟩ := div_model xF yF
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (xF / yF)
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hy0 : 0 ≤ y := le_trans hyF0.le hyf
  have hxq : x / y ≤ toReal xF / toReal yF := by gcongr; exact hx0.trans hxf
  have hxq0 : 0 ≤ toReal xF / toReal yF := div_nonneg (hx0.trans hxf) hyF0.le
  have hq0 : 0 ≤ x / y := div_nonneg hx0 hy0
  have h₁ : (x / y) * (1 - u64) ≤ toReal (xF / yF) := by
    rw [e₁]
    calc (x / y) * (1 - u64) ≤ (toReal xF / toReal yF) * (1 - u64) :=
          mul_le_mul_of_nonneg_right hxq hu1
      _ ≤ (toReal xF / toReal yF) * (1 + δ₁) := mul_le_mul_of_nonneg_left hd₁ hxq0
  have h₁0 : 0 ≤ toReal (xF / yF) := by rw [e₁]; exact mul_nonneg hxq0 (by linarith)
  rw [e₂]
  calc (1 + u64) * (x / y)
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (x / y) :=
        mul_le_mul_of_nonneg_right slackF_key hq0
    _ = (toReal slackF * ((x / y) * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * toReal (xF / yF)) * (1 - u64) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₁ hs) hu1
    _ ≤ (toReal slackF * toReal (xF / yF)) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs h₁0)

/-- Float mirror of `adamM1Mag` (`slackF` per node). -/
def adamM1MagF (BmF BgF Bb1F Bc1F : Float) : Float :=
  slackF * (slackF * (Bb1F * BmF) + slackF * (Bc1F * BgF))
/-- Float mirror of `adamDirMag` (denominator = the positive lower witness `dminF`). -/
def adamDirMagF (BmF BgF Bb1F Bc1F dminF : Float) : Float :=
  slackF * (adamM1MagF BmF BgF Bb1F Bc1F / dminF)
/-- Float mirror of `adamMag` — the whole-step overflow budget. -/
def adamStepCF (BpF BmF BgF BlrF Bb1F Bc1F dminF : Float) : Float :=
  slackF * (BpF + slackF * (BlrF * adamDirMagF BmF BgF Bb1F Bc1F dminF))

/-- Positive floor literal for the `√v'+ε` denominator (C89's `minWdF` pin). -/
def adamMinF : Float := 1e-30

/-- `0 < toReal adamMinF` — the one-time pin via `toReal_ofScientific_close` (C89's `minWdF` pattern). -/
theorem adamMinF_pos : 0 < toReal adamMinF := by
  have h := toReal_ofScientific_close 1 true 30
  have habs : |(1e-30 : ℝ)| = (1e-30 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h1 := (abs_le.mp h).1
  have hnum : (0 : ℝ) < 1e-30 := by norm_num
  unfold adamMinF
  nlinarith [u64_lt_one]

/-- `adamDirMag ≥ 0` for nonneg operand bounds and a positive floor. -/
theorem adamDirMag_nonneg (Bm Bg Bb1 Bc1 dmin : ℝ)
    (hBm : 0 ≤ Bm) (hBg : 0 ≤ Bg) (hBb1 : 0 ≤ Bb1) (hBc1 : 0 ≤ Bc1) (hdmin : 0 < dmin) :
    0 ≤ adamDirMag Bm Bg Bb1 Bc1 dmin := by
  unfold adamDirMag
  exact mul_nonneg u1_nonneg (div_nonneg (adamM1Mag_nonneg _ _ _ _ hBm hBg hBb1 hBc1) hdmin.le)

/-- **1st-moment budget domination**: `adamM1Mag ≤ toReal (adamM1MagF …)`. -/
theorem adamM1Mag_dominates (BmF BgF Bb1F Bc1F : Float) (Bm Bg Bb1 Bc1 : ℝ)
    (hBm0 : 0 ≤ Bm) (hBmf : Bm ≤ toReal BmF) (hBg0 : 0 ≤ Bg) (hBgf : Bg ≤ toReal BgF)
    (hBb10 : 0 ≤ Bb1) (hBb1f : Bb1 ≤ toReal Bb1F) (hBc10 : 0 ≤ Bc1) (hBc1f : Bc1 ≤ toReal Bc1F) :
    adamM1Mag Bm Bg Bb1 Bc1 ≤ toReal (adamM1MagF BmF BgF Bb1F Bc1F) := by
  unfold adamM1Mag adamM1MagF
  have hx := unit_mul_dominates Bb1F BmF Bb1 Bm hBb10 hBb1f hBm0 hBmf
  have hy := unit_mul_dominates Bc1F BgF Bc1 Bg hBc10 hBc1f hBg0 hBgf
  have hx0 : 0 ≤ (1 + u64) * (Bb1 * Bm) := mul_nonneg u1_nonneg (mul_nonneg hBb10 hBm0)
  have hy0 : 0 ≤ (1 + u64) * (Bc1 * Bg) := mul_nonneg u1_nonneg (mul_nonneg hBc10 hBg0)
  exact unit_add_dominates (slackF * (Bb1F * BmF)) (slackF * (Bc1F * BgF)) _ _ hx0 hx hy0 hy

/-- **Direction budget domination**: `adamDirMag ≤ toReal (adamDirMagF …)` (division reversed via
    the positive lower floor `dminF`). -/
theorem adamDirMag_dominates (BmF BgF Bb1F Bc1F dminF : Float) (Bm Bg Bb1 Bc1 dmin : ℝ)
    (hBm0 : 0 ≤ Bm) (hBmf : Bm ≤ toReal BmF) (hBg0 : 0 ≤ Bg) (hBgf : Bg ≤ toReal BgF)
    (hBb10 : 0 ≤ Bb1) (hBb1f : Bb1 ≤ toReal Bb1F) (hBc10 : 0 ≤ Bc1) (hBc1f : Bc1 ≤ toReal Bc1F)
    (hdminF0 : 0 < toReal dminF) (hdminf : toReal dminF ≤ dmin) :
    adamDirMag Bm Bg Bb1 Bc1 dmin ≤ toReal (adamDirMagF BmF BgF Bb1F Bc1F dminF) := by
  unfold adamDirMag adamDirMagF
  have hx0 : 0 ≤ adamM1Mag Bm Bg Bb1 Bc1 := adamM1Mag_nonneg _ _ _ _ hBm0 hBg0 hBb10 hBc10
  have hxf := adamM1Mag_dominates BmF BgF Bb1F Bc1F Bm Bg Bb1 Bc1
    hBm0 hBmf hBg0 hBgf hBb10 hBb1f hBc10 hBc1f
  exact unit_div_dominates (adamM1MagF BmF BgF Bb1F Bc1F) dminF
    (adamM1Mag Bm Bg Bb1 Bc1) dmin hx0 hxf hdminF0 hdminf

/-- **Whole-step budget domination**: `adamMag ≤ toReal (adamStepCF …)`. -/
theorem adamMag_dominates (BpF BmF BgF BlrF Bb1F Bc1F dminF : Float) (Bp Bm Bg Blr Bb1 Bc1 dmin : ℝ)
    (hBp0 : 0 ≤ Bp) (hBpf : Bp ≤ toReal BpF) (hBm0 : 0 ≤ Bm) (hBmf : Bm ≤ toReal BmF)
    (hBg0 : 0 ≤ Bg) (hBgf : Bg ≤ toReal BgF) (hBlr0 : 0 ≤ Blr) (hBlrf : Blr ≤ toReal BlrF)
    (hBb10 : 0 ≤ Bb1) (hBb1f : Bb1 ≤ toReal Bb1F) (hBc10 : 0 ≤ Bc1) (hBc1f : Bc1 ≤ toReal Bc1F)
    (hdminF0 : 0 < toReal dminF) (hdminf : toReal dminF ≤ dmin) :
    adamMag Bp Bm Bg Blr Bb1 Bc1 dmin ≤ toReal (adamStepCF BpF BmF BgF BlrF Bb1F Bc1F dminF) := by
  unfold adamMag adamStepCF
  have hdirf := adamDirMag_dominates BmF BgF Bb1F Bc1F dminF Bm Bg Bb1 Bc1 dmin
    hBm0 hBmf hBg0 hBgf hBb10 hBb1f hBc10 hBc1f hdminF0 hdminf
  have hdir0 : 0 ≤ adamDirMag Bm Bg Bb1 Bc1 dmin :=
    adamDirMag_nonneg _ _ _ _ _ hBm0 hBg0 hBb10 hBc10 (lt_of_lt_of_le hdminF0 hdminf)
  have hinner := unit_mul_dominates BlrF (adamDirMagF BmF BgF Bb1F Bc1F dminF)
    Blr (adamDirMag Bm Bg Bb1 Bc1 dmin) hBlr0 hBlrf hdir0 hdirf
  have hinner0 : 0 ≤ (1 + u64) * (Blr * adamDirMag Bm Bg Bb1 Bc1 dmin) :=
    mul_nonneg u1_nonneg (mul_nonneg hBlr0 hdir0)
  exact unit_add_dominates BpF (slackF * (BlrF * adamDirMagF BmF BgF Bb1F Bc1F dminF))
    _ _ hBp0 hBpf hinner0 hinner

/-- **THE RUNNABLE ADAM NO-OVERFLOW CERTIFICATE.** The input magnitude bounds are Float upper
    witnesses (`|toReal x| ≤ toReal xF`, each Bool-dischargeable via `checkAbsLe_sound`); the
    denominator floor is a positive Float lower witness `dminF` (positivity from ONE
    `checkLe adamMinF dminF` pin, the runtime coupling `toReal dminF ≤ |√v'+ε|` caller-side, exactly
    as C89); and ONE `checkLe (adamStepCF …) capF` gates the whole overflow budget. The two Bool
    gates plus the input witnesses ⟹ the Adam step is overflow-free. -/
theorem adamStepF_isFinite_runnable
    (p m v g lr b1 c1 b2 c2 eps : Float)
    (BpF BmF BgF BlrF Bb1F Bc1F dminF : Float)
    (hp : |toReal p| ≤ toReal BpF) (hm : |toReal m| ≤ toReal BmF) (hg : |toReal g| ≤ toReal BgF)
    (hlr : |toReal lr| ≤ toReal BlrF) (hb1 : |toReal b1| ≤ toReal Bb1F) (hc1 : |toReal c1| ≤ toReal Bc1F)
    (hpin : checkLe adamMinF dminF = true)
    (hden : toReal dminF ≤ |toReal (Float.sqrt (adamM2F v g b2 c2) + eps)|)
    (hchk : checkLe (adamStepCF BpF BmF BgF BlrF Bb1F Bc1F dminF) capF = true) :
    (adamStepF p m v g lr b1 c1 b2 c2 eps).isFinite = true := by
  have hBp0 : 0 ≤ toReal BpF := (abs_nonneg _).trans hp
  have hBm0 : 0 ≤ toReal BmF := (abs_nonneg _).trans hm
  have hBg0 : 0 ≤ toReal BgF := (abs_nonneg _).trans hg
  have hBlr0 : 0 ≤ toReal BlrF := (abs_nonneg _).trans hlr
  have hBb10 : 0 ≤ toReal Bb1F := (abs_nonneg _).trans hb1
  have hBc10 : 0 ≤ toReal Bc1F := (abs_nonneg _).trans hc1
  have hdminF0 : 0 < toReal dminF := lt_of_lt_of_le adamMinF_pos (checkLe_sound hpin)
  have hdmin0 : 0 < |toReal (Float.sqrt (adamM2F v g b2 c2) + eps)| := lt_of_lt_of_le hdminF0 hden
  have hdom := adamMag_dominates BpF BmF BgF BlrF Bb1F Bc1F dminF
    (toReal BpF) (toReal BmF) (toReal BgF) (toReal BlrF) (toReal Bb1F) (toReal Bc1F)
    (|toReal (Float.sqrt (adamM2F v g b2 c2) + eps)|)
    hBp0 le_rfl hBm0 le_rfl hBg0 le_rfl hBlr0 le_rfl hBb10 le_rfl hBc10 le_rfl hdminF0 hden
  have hbound := hdom.trans ((checkLe_sound hchk).trans capF_le)
  exact adamStepF_isFinite p m v g lr b1 c1 b2 c2 eps
    (toReal BpF) (toReal BmF) (toReal BgF) (toReal BlrF) (toReal Bb1F) (toReal Bc1F)
    (|toReal (Float.sqrt (adamM2F v g b2 c2) + eps)|)
    hp hm hg hlr hb1 hc1 hBp0 hBm0 hBg0 hBlr0 hBb10 hBc10 hdmin0 le_rfl hbound

/-! ### Demos: the budget gate on real Adam hyperparameters -/

-- Real Adam config (β₁=0.9, β₂=0.999, c₁=0.1, lr=1e-3, ε=1e-8), unit state bounds, floor 1e-8.
/-- info: true -/
#guard_msgs in #eval checkLe (adamStepCF 1.0 1.0 1.0 0.001 0.9 0.1 1e-8) capF

-- Negative control: a huge state bound drives the budget past the cap.
/-- info: false -/
#guard_msgs in #eval checkLe (adamStepCF 1.0 1e300 1.0 0.001 0.9 0.1 1e-8) capF

end Puffer.RL.AdamFinite
