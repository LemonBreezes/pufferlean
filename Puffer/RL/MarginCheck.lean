/-
# Verified runtime margin checker: the data-dependent whole-run conditions, runnably checked with proven soundness

The whole-run program (C27–C67) reduced everything structural to proofs and left a small slate of DATA-DEPENDENT
per-step conditions — the param-region bounds (`hRegθ : ∀ p i, |θ p i| ≤ R`, C35/C38/C41), the clip-interior /
clip-margin conditions (`hIntθ`/`hmarginθ'`: `ratio ∈ (lo, hi)` resp. `ratio ∈ (lo + d, hi − d)`, C35/C36/C38), and
the fixed-point ratio margin (C49/C58) — "exactly the quantities the `puffer verify` harness checks numerically."
This module closes that loop: COMPUTABLE `Bool` checkers over the runtime `Float` data, with PROVEN soundness
bridges — if the checker returns `true`, the ℝ-level premise the whole-run theorems consume HOLDS.

**Comparison-semantics route (NO new axiom).** The trusted base already derives that Float comparison respects the
real order: `le_of_float_le : a ≤ b → toReal a ≤ toReal b` (from the exact `toReal_min`) and IEEE negation is exact
(`toReal_neg`). So the checkers use plain Float `≤` (decidable) and their soundness needs NOTHING new. Float
comparisons yield only NON-strict ℝ facts, so STRICT margins are obtained honestly by splitting each condition into
* a per-step DATA comparison against a fixed Float THRESHOLD (`tLo ≤ x ≤ tHi`, sound and runnable), and
* a ONE-TIME proof-side GAP hypothesis on the fixed constants (`L + e < toReal tLo`, `toReal tHi + e < H`) —
  about hyperparameters (`lo`/`hi`/margins/thresholds), established once offline, not per step.
The gap `e ≥ 0` also absorbs the FORWARD ERROR between the runtime Float value and the exact ℝ evaluation
(`|v − toReal x| ≤ e`, supplied by the proven forward bounds, e.g. `RatioForward`), so the conclusion lands on the
EXACT value `v = evalR (ratioE …) σ` the theorems quantify over.

* `checkLe`/`checkGe`/`checkInterval`/`checkAbsLe` — the computable comparisons, each with a soundness theorem.
* `checkRegion` — all params in `[−R, R]` (`List.all`), sound for every member and every index: the runtime-data
  form of `hRegθ p`.
* `checkClipMargin` + `checkClipMargin_sound`/`checkClipMargin_sound_exact` — the threshold check with the
  one-time-gap soundness, in both the `toReal x` form and the forward-error-absorbing EXACT-value form.
* `ratio_margin_certificate` / `step_certificate` (capstones) — instantiated at the PPO ratio: a passing check
  certifies C35/C38's `hIntθ p`-conjunct (bare interval) resp. the `hmargin p`-conjunct (margin interval) at that
  step, and the region check certifies the `hRegθ p` data.

**Scope (honestly disclosed).** SOUND, not complete: the Float comparisons are conservative — a borderline-true
condition may fail the check, but `true` always implies the ℝ-level premise. Certificates are PER-STEP (per data
point); the `∀ p` lift is the runtime loop iterating the check at every step (as `puffer verify` does). The
one-time gap hypotheses (`L + e < toReal tLo`, …) are the honest offline obligations about the fixed
hyperparameters and the forward-error/margin budgets; the per-step obligation is exactly one runnable `Bool`. The
region soundness is stated over the runtime `List Float` (membership and positional forms) — connecting a list
entry to the abstract `Nat`-indexed parameter vector `θ p` is the caller's data plumbing (note: C35's
`hRegθ` quantifies over ALL `Nat` indices, so the caller must also bound the indices beyond the list —
e.g. zero-padding, which needs `0 ≤ toReal R`).
-/
import Puffer.RL.BudgetDischarge

open Puffer.FloatR
open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateExpr (ratioE)

namespace Puffer.RL.MarginCheck

/-! ### The computable comparisons -/

/-- Runnable `toReal x ≤ toReal b` witness: plain Float `≤` (decidable). -/
def checkLe (x b : Float) : Bool := decide (x ≤ b)

/-- Runnable `toReal b ≤ toReal x` witness. -/
def checkGe (x b : Float) : Bool := decide (b ≤ x)

/-- Runnable two-sided interval membership `toReal lo ≤ toReal x ≤ toReal hi`. -/
def checkInterval (x lo hi : Float) : Bool := checkGe x lo && checkLe x hi

/-- Runnable `|toReal x| ≤ toReal R` via the EXACT IEEE negation (`toReal_neg`): `−R ≤ x ∧ x ≤ R`. -/
def checkAbsLe (x R : Float) : Bool := checkInterval x (-R) R

/-- `checkLe` is sound: a passing Float comparison implies the real one (`le_of_float_le`, derived —
    no new axiom). -/
theorem checkLe_sound {x b : Float} (h : checkLe x b = true) : toReal x ≤ toReal b :=
  le_of_float_le (of_decide_eq_true h)

theorem checkGe_sound {x b : Float} (h : checkGe x b = true) : toReal b ≤ toReal x :=
  le_of_float_le (of_decide_eq_true h)

theorem checkInterval_sound {x lo hi : Float} (h : checkInterval x lo hi = true) :
    toReal lo ≤ toReal x ∧ toReal x ≤ toReal hi := by
  rw [checkInterval, Bool.and_eq_true] at h
  exact ⟨checkGe_sound h.1, checkLe_sound h.2⟩

/-- `checkAbsLe` is sound: `|toReal x| ≤ toReal R` (the negation side exact via `toReal_neg`). -/
theorem checkAbsLe_sound {x R : Float} (h : checkAbsLe x R = true) : |toReal x| ≤ toReal R := by
  obtain ⟨h1, h2⟩ := checkInterval_sound h
  rw [toReal_neg] at h1
  exact abs_le.mpr ⟨h1, h2⟩

/-! ### The region check (`hRegθ p`'s runtime data) -/

/-- Runnable region membership: every parameter in `[−R, R]`. -/
def checkRegion (θrow : List Float) (R : Float) : Bool := θrow.all (fun x => checkAbsLe x R)

/-- `checkRegion` is sound for every member: the runtime-data form of the `hRegθ p` premise
    (C35/C38/C41's param-boundedness at one step). -/
theorem checkRegion_sound {θrow : List Float} {R : Float} (h : checkRegion θrow R = true) :
    ∀ x ∈ θrow, |toReal x| ≤ toReal R := by
  rw [checkRegion, List.all_eq_true] at h
  exact fun x hx => checkAbsLe_sound (h x hx)

/-- Positional form: every index of the runtime row is bounded. -/
theorem checkRegion_sound_idx {θrow : List Float} {R : Float} (h : checkRegion θrow R = true) :
    ∀ i (hi : i < θrow.length), |toReal θrow[i]| ≤ toReal R :=
  fun _ hi => checkRegion_sound h _ (θrow.getElem_mem hi)

/-! ### The margin check (`hIntθ`/`hmargin`'s runtime data) -/

/-- Runnable threshold membership `tLo ≤ x ≤ tHi` — the per-step clip/margin data comparison. The
    STRICTNESS and the margin/forward-error budgets live in the one-time gap hypotheses of the
    soundness theorems below, not in the runnable check. -/
def checkClipMargin (x tLo tHi : Float) : Bool := checkInterval x tLo tHi

/-- **Margin soundness (`toReal x` form).** A passing check plus the ONE-TIME gaps on the fixed
    threshold constants (`L < toReal tLo`, `toReal tHi < H` — offline obligations about
    hyperparameters, not per-step data) give the STRICT open-interval membership
    `L < toReal x < H`. Instantiating `L := toReal lo` and `H := toReal hi` yields exactly the
    `hIntθ p`-conjunct (C35/C38/C41); `L := toReal lo + d`, `H := toReal hi − d` yields the
    `hmargin p`-conjunct (C36/C38). -/
theorem checkClipMargin_sound {x tLo tHi : Float} {L H : ℝ}
    (h : checkClipMargin x tLo tHi = true)
    (hLo : L < toReal tLo) (hHi : toReal tHi < H) :
    L < toReal x ∧ toReal x < H := by
  obtain ⟨h1, h2⟩ := checkInterval_sound h
  exact ⟨lt_of_lt_of_le hLo h1, lt_of_le_of_lt h2 hHi⟩

/-- **Margin soundness (EXACT-value form).** The runtime check runs on the FLOAT value `x`, but the
    whole-run premises quantify over the EXACT ℝ evaluation `v` (e.g. `evalR (ratioE …) σ`). Given
    the proven forward error `|v − toReal x| ≤ e` (from the forward-bound theorems) and the gaps
    WIDENED by `e` (`L + e < toReal tLo`, `toReal tHi + e < H` — still one-time constants-only
    obligations), a passing check certifies the strict membership of the EXACT value:
    `L < v < H`. -/
theorem checkClipMargin_sound_exact {x tLo tHi : Float} {v e L H : ℝ}
    (h : checkClipMargin x tLo tHi = true)
    (herr : |v - toReal x| ≤ e)
    (hLo : L + e < toReal tLo) (hHi : toReal tHi + e < H) :
    L < v ∧ v < H := by
  obtain ⟨h1, h2⟩ := checkInterval_sound h
  obtain ⟨hd1, hd2⟩ := abs_le.mp herr
  constructor
  · linarith
  · linarith

/-! ### The instantiated per-step certificates -/

/-- **The ratio-margin certificate.** Instantiated at the PPO ratio: if the runtime Float ratio `rF`
    passes the threshold check, the forward error to the exact ratio is within `e`, and the one-time
    gaps hold, then the EXACT ratio at this step satisfies the clip-interior premise
    `toReal lo < evalR (ratioE …) σ < toReal hi` — the `hIntθ p`-conjunct of
    C35's `ppo_whole_run_reachable` / C38's `ppo_whole_run_from_barriers` at this step. (This theorem
    hard-codes `L := toReal lo`, `H := toReal hi`; for the MARGIN-strengthened `hmargin p`-conjunct
    (C36/C38, `L := toReal lo + d`, `H := toReal hi − d`), apply the underlying
    `checkClipMargin_sound_exact` directly at those endpoints.) -/
theorem ratio_margin_certificate (chosen eE : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (σ : Nat → ℝ) (rF tLo tHi : Float) (e : ℝ)
    (h : checkClipMargin rF tLo tHi = true)
    (herr : |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ - toReal rF| ≤ e)
    (hLo : toReal lo + e < toReal tLo) (hHi : toReal tHi + e < toReal hi) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ
      ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ < toReal hi :=
  checkClipMargin_sound_exact h herr hLo hHi

/-- **The per-step certificate (capstone).** One step's runtime data — the parameter row and the
    Float ratio — passing BOTH runnable checks (plus the one-time gap/forward-error hypotheses)
    certifies BOTH per-step premises the whole-run theorems consume at that step: the region data
    (`hRegθ p`'s row, every entry `≤ toReal R`) and the exact-ratio clip interior (`hIntθ p`).
    The `∀ p` lift is the runtime loop running these two `Bool`s at every step — exactly what the
    `puffer verify` harness does, now with a machine-checked guarantee behind each passing check. -/
theorem step_certificate (chosen eE : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (σ : Nat → ℝ) (θrow : List Float) (Rf rF tLo tHi : Float) (e : ℝ)
    (hreg : checkRegion θrow Rf = true)
    (hclip : checkClipMargin rF tLo tHi = true)
    (herr : |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ - toReal rF| ≤ e)
    (hLo : toReal lo + e < toReal tLo) (hHi : toReal tHi + e < toReal hi) :
    (∀ x ∈ θrow, |toReal x| ≤ toReal Rf)
      ∧ (toReal lo < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ
          ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) σ < toReal hi) :=
  ⟨checkRegion_sound hreg,
   ratio_margin_certificate chosen eE es oldLogp lo hi σ rF tLo tHi e hclip herr hLo hHi⟩

end Puffer.RL.MarginCheck
