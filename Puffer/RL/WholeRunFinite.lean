/-
# Whole-run finiteness: composing the per-step overflow-free certificate across n training steps

C57 (`BackwardFinite`) certified ONE linear-layer training step overflow-free with the SGD-shaped update, and
C61 (`MuonUpdateFinite`) with the Muon-shaped (Newton–Schulz) update; both disclosed that whole-run finiteness
"composes per-step". This module performs that composition: an `n`-step training run stays overflow-free at
EVERY step — weights and forward pass alike — while the propagated magnitude budget stays `≤ overflowBound`.

The key structural fact: the per-step weight-budget map is AFFINE with a step-invariant constant. The backward
gradient `gradW gseed x` depends only on the (fixed) inputs and cotangent seed, NOT on the weights, so one step
maps a weight budget `B` to `stepBound C B := (1+u64)·(B + C)` where `C` packages the whole gradient-through-
update term — `sgdStepC` for the SGD update, `muonStepC` (with the NS circuit budget `nsScalarFBound`) for the
Muon-shaped one. The `n`-step budget is the `n`-fold iterate `runBound C n B`, and `stepBound` is monotone and
inflationary (`B ≤ stepBound C B` for nonneg budgets), so intermediate steps are dominated by the final budget
(`runBound_mono_steps`).

* `stepBound`/`runBound` (+ monotonicity) — the shared one-step/n-step budget maps.
* `trainRun`/`muonTrainRun` — the `n`-step runs, folding C57's `linearTrainStep` / C61's `muonTrainStep`.
* `trainRun_mag`/`muonTrainRun_mag` — the invariant: initial weights `≤ B₀` ⟹ step-`n` weights `≤ runBound C n B₀`.
* `trainRun_all_finite`/`muonTrainRun_all_finite` (capstones) — EVERY weight at EVERY step `m ≤ n` is `isFinite`,
  given the single checkable budget condition `runBound C n B₀ ≤ overflowBound`.
* `trainRun_forward_all_finite`/`muonTrainRun_forward_all_finite` — the forward pass `dotF x (weights at step m)`
  is `isFinite` at every step `m ≤ n` (the run's weight lists never grow — `trainRun_length` — so one `dotBound`
  budget at the max of the input and final-weight bounds covers all steps via `dotBound_mono`).

**Scope (honestly disclosed).** The budget `runBound C n B₀` GROWS with `n` (each step inflates by `(1+u64)` and
adds the gradient-through-update constant), so for fixed budgets this certifies a FINITE horizon — the honest
price of the crude per-step composition. A uniform-in-`n` weight budget would need a CONTRACTION on the weight
magnitudes (weight decay, C32-style `|1−wd|`-damping of the `B` term) — the natural refinement, not done here.
The model is a FIXED-BATCH repeated step (the same `x`/`gseed` each step); a varying batch with UNIFORM per-step
bounds `Bx`/`Bgs` gives the same budgets — the bound argument only ever uses the per-step input bounds, never the
identity of the batch — but the fold is stated with a fixed batch for concreteness. The backward gradient
`gradW gseed x` is step-invariant here (fixed batch), so its finiteness is exactly C57's `gradW_isFinite`,
unchanged — no per-step restatement needed. Reuses C43's single trusted no-overflow axiom `isFinite_of_bounded`
+ the `(1+δ)` base — NO new axiom. As in C57/C61: the backward is the self-contained linear-layer `gradW` (not
the full `ADReverse` tape), and the Muon update is the scalar per-coordinate NS circuit.
-/
import Puffer.RL.MuonUpdateFinite
open Puffer.FloatR
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound dotBound dotF_isFinite)
open Puffer.RL.ForwardFinite (dotBound_mono)
open Puffer.RL.BackwardFinite (updateVec updateVec_mag gradW gradW_mag linearTrainStep)
open Puffer.RL.MuonUpdateFinite (nsScalarFBound muonUpdateVec muonUpdateVec_mag muonTrainStep
  muonUpdateBound)

namespace Puffer.RL.WholeRunFinite

/-! ### The shared one-step / n-step budget maps -/

/-- **The one-step weight-budget map** `B ↦ (1+u64)·(B + C)`: one training step takes a `B`-bounded weight row to
    a `stepBound C B`-bounded one, where the constant `C` packages the (step-invariant) gradient-through-update
    term — `sgdStepC` for SGD, `muonStepC` for the Muon-shaped update. -/
noncomputable def stepBound (C B : ℝ) : ℝ := (1 + u64) * (B + C)

/-- **The n-step budget**: the `n`-fold iterate of `stepBound C`, matching the fold direction of the runs
    (`runBound C (n+1) B = runBound C n (stepBound C B)`). Grows with `n` — the honest crude composition. -/
noncomputable def runBound (C : ℝ) : Nat → ℝ → ℝ
  | 0, B => B
  | n + 1, B => runBound C n (stepBound C B)

/-- `stepBound C` is monotone in the budget. -/
theorem stepBound_mono (C : ℝ) {B B' : ℝ} (h : B ≤ B') : stepBound C B ≤ stepBound C B' := by
  unfold stepBound
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  exact mul_le_mul_of_nonneg_left (by linarith) h1

/-- `stepBound C` is inflationary on nonneg budgets: `B ≤ stepBound C B` (`(1+u)·(B+C) ≥ B+C ≥ B`). -/
theorem le_stepBound (C B : ℝ) (hC : 0 ≤ C) (hB : 0 ≤ B) : B ≤ stepBound C B := by
  unfold stepBound
  nlinarith [u64_pos]

/-- `runBound C n` is monotone in the budget (iterated `stepBound_mono`). -/
theorem runBound_mono (C : ℝ) : ∀ (n : Nat) {B B' : ℝ}, B ≤ B' → runBound C n B ≤ runBound C n B'
  | 0, _, _, h => h
  | n + 1, _, _, h => runBound_mono C n (stepBound_mono C h)

/-- One more step never shrinks the budget (nonneg `B`, `C`). -/
theorem runBound_le_succ (C : ℝ) (hC : 0 ≤ C) (n : Nat) (B : ℝ) (hB : 0 ≤ B) :
    runBound C n B ≤ runBound C (n + 1) B :=
  runBound_mono C n (le_stepBound C B hC hB)

/-- **Intermediate budgets are dominated by the final one**: `m ≤ n ⟹ runBound C m B ≤ runBound C n B`
    (nonneg `B`, `C`) — what lets one final budget condition certify every intermediate step. -/
theorem runBound_mono_steps (C : ℝ) (hC : 0 ≤ C) (B : ℝ) (hB : 0 ≤ B) {m n : Nat} (h : m ≤ n) :
    runBound C m B ≤ runBound C n B := by
  induction h with
  | refl => exact le_refl _
  | step _ ih => exact ih.trans (runBound_le_succ C hC _ B hB)

/-- C61's NS circuit budget is nonneg for nonneg inputs (products/sums of nonnegs, `1+u64 ≥ 0`). -/
theorem nsScalarFBound_nonneg (Ba Bb Bc Bσ : ℝ)
    (ha : 0 ≤ Ba) (hb : 0 ≤ Bb) (hc : 0 ≤ Bc) (hσ : 0 ≤ Bσ) :
    0 ≤ nsScalarFBound Ba Bb Bc Bσ := by
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  have hσσ : 0 ≤ (1 + u64) * (Bσ * Bσ) := mul_nonneg h1 (mul_nonneg hσ hσ)
  unfold Puffer.RL.MuonUpdateFinite.nsScalarFBound
  exact mul_nonneg h1 (mul_nonneg hσ (mul_nonneg h1 (add_nonneg
    (mul_nonneg h1 (add_nonneg ha (mul_nonneg h1 (mul_nonneg hb hσσ))))
    (mul_nonneg h1 (mul_nonneg hc (mul_nonneg h1 (mul_nonneg hσσ hσσ)))))))

/-! ### The SGD run -/

/-- **The SGD step constant**: the gradient-through-update term `(1+u64)·(Blr·((1+u64)·(Bgs·Bx)))` — C57's
    `updateVec` budget with the `gradW` gradient budget threaded. Step-invariant (the gradient depends only on
    the fixed inputs/seed). -/
noncomputable def sgdStepC (Bx Bgs Blr : ℝ) : ℝ := (1 + u64) * (Blr * ((1 + u64) * (Bgs * Bx)))

theorem sgdStepC_nonneg (Bx Bgs Blr : ℝ) (hx : 0 ≤ Bx) (hgs : 0 ≤ Bgs) (hlr : 0 ≤ Blr) :
    0 ≤ sgdStepC Bx Bgs Blr := by
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  exact mul_nonneg h1 (mul_nonneg hlr (mul_nonneg h1 (mul_nonneg hgs hx)))

/-- **The n-step SGD training run**: fold C57's `linearTrainStep` (fixed batch `x`/`gseed` per step — see the
    module scope note). -/
def trainRun (x : List Float) (gseed lr : Float) : Nat → List Float → List Float
  | 0, w => w
  | n + 1, w => trainRun x gseed lr n (linearTrainStep x w gseed lr)

/-- One step never lengthens the weight row (`updateVec` is a `zipWith`). -/
theorem linearTrainStep_length (x w : List Float) (gseed lr : Float) :
    (linearTrainStep x w gseed lr).length ≤ w.length := by
  simp only [Puffer.RL.BackwardFinite.linearTrainStep, Puffer.RL.BackwardFinite.updateVec,
    List.length_zipWith]
  exact min_le_left _ _

/-- The run never lengthens the weight row. -/
theorem trainRun_length (x : List Float) (gseed lr : Float) :
    ∀ (n : Nat) (w : List Float), (trainRun x gseed lr n w).length ≤ w.length
  | 0, _ => le_refl _
  | n + 1, w => (trainRun_length x gseed lr n _).trans (linearTrainStep_length x w gseed lr)

/-- **One SGD step maps a `B`-budget to `stepBound (sgdStepC …) B`** (C57's `updateVec_mag` at the `gradW`
    gradient budget — definitionally the affine step map). -/
theorem linearTrainStep_mag (x w : List Float) (gseed lr : Float) (Bx B Bgs Blr : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) :
    ∀ u ∈ linearTrainStep x w gseed lr, |toReal u| ≤ stepBound (sgdStepC Bx Bgs Blr) B := by
  intro u hu
  have h := updateVec_mag lr B ((1 + u64) * (Bgs * Bx)) Blr hlr w (gradW gseed x) hw
    (gradW_mag gseed x Bgs Bx hgs hx) u hu
  unfold stepBound sgdStepC
  exact h

/-- **The run invariant**: initial weights `≤ B` ⟹ every weight after `n` steps `≤ runBound (sgdStepC …) n B`
    (induction over the fold, one `linearTrainStep_mag` per step). -/
theorem trainRun_mag (x : List Float) (gseed lr : Float) (Bx Bgs Blr : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) :
    ∀ (n : Nat) (w : List Float) (B : ℝ), (∀ wi ∈ w, |toReal wi| ≤ B) →
      ∀ u ∈ trainRun x gseed lr n w, |toReal u| ≤ runBound (sgdStepC Bx Bgs Blr) n B
  | 0, _, _, hw => hw
  | n + 1, w, B, hw =>
      trainRun_mag x gseed lr Bx Bgs Blr hx hgs hlr n (linearTrainStep x w gseed lr)
        (stepBound (sgdStepC Bx Bgs Blr) B)
        (linearTrainStep_mag x w gseed lr Bx B Bgs Blr hx hw hgs hlr)

/-- **THE SGD WHOLE-RUN IS OVERFLOW-FREE.** For a fixed-batch `n`-step SGD run on bounded inputs/weights/seed/
    step-size, with the single checkable budget condition `runBound (sgdStepC …) n B₀ ≤ overflowBound`, EVERY
    weight at EVERY step `m ≤ n` is `isFinite` (intermediate budgets dominated by the final one). -/
theorem trainRun_all_finite (x w : List Float) (gseed lr : Float) (n : Nat) (Bx B0 Bgs Blr : ℝ)
    (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (hbound : runBound (sgdStepC Bx Bgs Blr) n B0 ≤ overflowBound) :
    ∀ m, m ≤ n → ∀ u ∈ trainRun x gseed lr m w, u.isFinite = true := by
  intro m hm u hu
  have hC : 0 ≤ sgdStepC Bx Bgs Blr :=
    sgdStepC_nonneg Bx Bgs Blr hBx0 ((abs_nonneg _).trans hgs) ((abs_nonneg _).trans hlr)
  exact isFinite_of_bounded _
    (((trainRun_mag x gseed lr Bx Bgs Blr hx hgs hlr m w B0 hw u hu).trans
      (runBound_mono_steps (sgdStepC Bx Bgs Blr) hC B0 hB00 hm)).trans hbound)

/-- **The forward pass is overflow-free at every step of the SGD run**: `dotF x (weights at step m)` is
    `isFinite` for all `m ≤ n`, from one `dotBound` budget at the max of the input bound and the final weight
    budget (the run never lengthens the weight row, so `dotBound_mono` covers every step's length). -/
theorem trainRun_forward_all_finite (x w : List Float) (gseed lr : Float) (n : Nat)
    (Bx B0 Bgs Blr : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (hfwd : dotBound (min x.length w.length)
        (max Bx (runBound (sgdStepC Bx Bgs Blr) n B0)) ≤ overflowBound) :
    ∀ m, m ≤ n → (Puffer.RL.FiniteBound.dotF x (trainRun x gseed lr m w)).isFinite = true := by
  intro m hm
  have hC : 0 ≤ sgdStepC Bx Bgs Blr :=
    sgdStepC_nonneg Bx Bgs Blr hBx0 ((abs_nonneg _).trans hgs) ((abs_nonneg _).trans hlr)
  have hBmax0 : 0 ≤ max Bx (runBound (sgdStepC Bx Bgs Blr) n B0) := hBx0.trans (le_max_left _ _)
  have hx' : ∀ xi ∈ x, |toReal xi| ≤ max Bx (runBound (sgdStepC Bx Bgs Blr) n B0) :=
    fun xi hxi => (hx xi hxi).trans (le_max_left _ _)
  have hw' : ∀ wi ∈ trainRun x gseed lr m w,
      |toReal wi| ≤ max Bx (runBound (sgdStepC Bx Bgs Blr) n B0) := fun wi hwi =>
    ((trainRun_mag x gseed lr Bx Bgs Blr hx hgs hlr m w B0 hw wi hwi).trans
      (runBound_mono_steps (sgdStepC Bx Bgs Blr) hC B0 hB00 hm)).trans (le_max_right _ _)
  have hlen : min x.length (trainRun x gseed lr m w).length ≤ min x.length w.length :=
    min_le_min (le_refl _) (trainRun_length x gseed lr m w)
  exact dotF_isFinite _ hBmax0 x (trainRun x gseed lr m w) hx' hw'
    ((dotBound_mono _ hlen).trans hfwd)

/-! ### The Muon-shaped run -/

/-- **The Muon step constant**: the gradient-through-NS-through-update term — C61's `muonUpdateBound` tail with
    the `gradW` gradient budget threaded through the NS circuit budget. Step-invariant, like `sgdStepC`. -/
noncomputable def muonStepC (Bx Bgs Blr Ba Bb Bc : ℝ) : ℝ :=
  (1 + u64) * (Blr * nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx)))

theorem muonStepC_nonneg (Bx Bgs Blr Ba Bb Bc : ℝ) (hx : 0 ≤ Bx) (hgs : 0 ≤ Bgs) (hlr : 0 ≤ Blr)
    (ha : 0 ≤ Ba) (hb : 0 ≤ Bb) (hc : 0 ≤ Bc) : 0 ≤ muonStepC Bx Bgs Blr Ba Bb Bc := by
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  exact mul_nonneg h1 (mul_nonneg hlr (nsScalarFBound_nonneg Ba Bb Bc _ ha hb hc
    (mul_nonneg h1 (mul_nonneg hgs hx))))

/-- **The n-step Muon-shaped training run**: fold C61's `muonTrainStep` (fixed batch per step). -/
def muonTrainRun (x : List Float) (gseed lr a b c : Float) : Nat → List Float → List Float
  | 0, w => w
  | n + 1, w => muonTrainRun x gseed lr a b c n (muonTrainStep x w gseed lr a b c)

/-- One Muon-shaped step never lengthens the weight row. -/
theorem muonTrainStep_length (x w : List Float) (gseed lr a b c : Float) :
    (muonTrainStep x w gseed lr a b c).length ≤ w.length := by
  simp only [Puffer.RL.MuonUpdateFinite.muonTrainStep, Puffer.RL.MuonUpdateFinite.muonUpdateVec,
    Puffer.RL.BackwardFinite.updateVec, List.length_zipWith]
  exact min_le_left _ _

/-- The Muon-shaped run never lengthens the weight row. -/
theorem muonTrainRun_length (x : List Float) (gseed lr a b c : Float) :
    ∀ (n : Nat) (w : List Float), (muonTrainRun x gseed lr a b c n w).length ≤ w.length
  | 0, _ => le_refl _
  | n + 1, w =>
      (muonTrainRun_length x gseed lr a b c n _).trans (muonTrainStep_length x w gseed lr a b c)

/-- **One Muon-shaped step maps a `B`-budget to `stepBound (muonStepC …) B`** (C61's `muonUpdateVec_mag` at the
    `gradW` gradient budget — definitionally the affine step map). -/
theorem muonTrainStep_mag (x w : List Float) (gseed lr a b c : Float) (Bx B Bgs Blr Ba Bb Bc : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ u ∈ muonTrainStep x w gseed lr a b c,
      |toReal u| ≤ stepBound (muonStepC Bx Bgs Blr Ba Bb Bc) B := by
  intro u hu
  have h := muonUpdateVec_mag w (gradW gseed x) lr a b c B ((1 + u64) * (Bgs * Bx)) Blr Ba Bb Bc
    hw (gradW_mag gseed x Bgs Bx hgs hx) hlr ha hb hc u hu
  unfold Puffer.RL.MuonUpdateFinite.muonUpdateBound at h
  unfold stepBound muonStepC
  exact h

/-- **The Muon-run invariant**: initial weights `≤ B` ⟹ every weight after `n` steps
    `≤ runBound (muonStepC …) n B`. -/
theorem muonTrainRun_mag (x : List Float) (gseed lr a b c : Float) (Bx Bgs Blr Ba Bb Bc : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ (n : Nat) (w : List Float) (B : ℝ), (∀ wi ∈ w, |toReal wi| ≤ B) →
      ∀ u ∈ muonTrainRun x gseed lr a b c n w,
        |toReal u| ≤ runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B
  | 0, _, _, hw => hw
  | n + 1, w, B, hw =>
      muonTrainRun_mag x gseed lr a b c Bx Bgs Blr Ba Bb Bc hx hgs hlr ha hb hc n
        (muonTrainStep x w gseed lr a b c) (stepBound (muonStepC Bx Bgs Blr Ba Bb Bc) B)
        (muonTrainStep_mag x w gseed lr a b c Bx B Bgs Blr Ba Bb Bc hx hw hgs hlr ha hb hc)

/-- **THE MUON WHOLE-RUN IS OVERFLOW-FREE.** For a fixed-batch `n`-step Muon-shaped run on bounded inputs/
    weights/seed/step-size/NS-coefficients, with the single budget condition `runBound (muonStepC …) n B₀ ≤
    overflowBound`, EVERY weight at EVERY step `m ≤ n` is `isFinite`. C57/C61's per-step certificate composed
    across the run. -/
theorem muonTrainRun_all_finite (x w : List Float) (gseed lr a b c : Float) (n : Nat)
    (Bx B0 Bgs Blr Ba Bb Bc : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hbound : runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B0 ≤ overflowBound) :
    ∀ m, m ≤ n → ∀ u ∈ muonTrainRun x gseed lr a b c m w, u.isFinite = true := by
  intro m hm u hu
  have hC : 0 ≤ muonStepC Bx Bgs Blr Ba Bb Bc :=
    muonStepC_nonneg Bx Bgs Blr Ba Bb Bc hBx0 ((abs_nonneg _).trans hgs)
      ((abs_nonneg _).trans hlr) ((abs_nonneg _).trans ha) ((abs_nonneg _).trans hb)
      ((abs_nonneg _).trans hc)
  exact isFinite_of_bounded _
    (((muonTrainRun_mag x gseed lr a b c Bx Bgs Blr Ba Bb Bc hx hgs hlr ha hb hc m w B0 hw u
      hu).trans (runBound_mono_steps (muonStepC Bx Bgs Blr Ba Bb Bc) hC B0 hB00 hm)).trans hbound)

/-- **The forward pass is overflow-free at every step of the Muon run** (mirror of
    `trainRun_forward_all_finite`). -/
theorem muonTrainRun_forward_all_finite (x w : List Float) (gseed lr a b c : Float) (n : Nat)
    (Bx B0 Bgs Blr Ba Bb Bc : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hfwd : dotBound (min x.length w.length)
        (max Bx (runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B0)) ≤ overflowBound) :
    ∀ m, m ≤ n →
      (Puffer.RL.FiniteBound.dotF x (muonTrainRun x gseed lr a b c m w)).isFinite = true := by
  intro m hm
  have hC : 0 ≤ muonStepC Bx Bgs Blr Ba Bb Bc :=
    muonStepC_nonneg Bx Bgs Blr Ba Bb Bc hBx0 ((abs_nonneg _).trans hgs)
      ((abs_nonneg _).trans hlr) ((abs_nonneg _).trans ha) ((abs_nonneg _).trans hb)
      ((abs_nonneg _).trans hc)
  have hBmax0 : 0 ≤ max Bx (runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B0) :=
    hBx0.trans (le_max_left _ _)
  have hx' : ∀ xi ∈ x, |toReal xi| ≤ max Bx (runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B0) :=
    fun xi hxi => (hx xi hxi).trans (le_max_left _ _)
  have hw' : ∀ wi ∈ muonTrainRun x gseed lr a b c m w,
      |toReal wi| ≤ max Bx (runBound (muonStepC Bx Bgs Blr Ba Bb Bc) n B0) := fun wi hwi =>
    ((muonTrainRun_mag x gseed lr a b c Bx Bgs Blr Ba Bb Bc hx hgs hlr ha hb hc m w B0 hw wi
      hwi).trans (runBound_mono_steps (muonStepC Bx Bgs Blr Ba Bb Bc) hC B0 hB00 hm)).trans
      (le_max_right _ _)
  have hlen : min x.length (muonTrainRun x gseed lr a b c m w).length ≤ min x.length w.length :=
    min_le_min (le_refl _) (muonTrainRun_length x gseed lr a b c m w)
  exact dotF_isFinite _ hBmax0 x (muonTrainRun x gseed lr a b c m w) hx' hw'
    ((dotBound_mono _ hlen).trans hfwd)

end Puffer.RL.WholeRunFinite
