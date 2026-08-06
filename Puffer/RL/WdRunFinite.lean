/-
# Horizon-free whole-run finiteness under weight decay: C64's named refinement

C64 (`WholeRunFinite`) composed the per-step overflow-free certificate across `n` training steps, but its budget
`runBound C n B₀` GROWS with `n` — the plain update's budget map `stepBound C B = (1+u64)·(B + C)` has slope
`(1+u64) > 1`, so fixed budgets certify only a FINITE horizon. C64 named the refinement: a uniform-in-`n` bound
needs a CONTRACTION on the weight magnitudes — weight decay. This module delivers it.

With the weight-decay update `w ← d·w − lr·g` (decay factor `d ≈ 1 − wd`, taken as a Float parameter — real
trainers precompute it once; `decayFactor_mag` bridges the literally-computed `1.0 − wd`), the per-step
weight-budget map becomes AFFINE WITH SLOPE `wdStepRho Bd = (1+u64)²·Bd`:

    B  ↦  wdStepRho Bd · B + wdStepC Bx Bgs Blr

and is CONTRACTIVE when `wdStepRho Bd < 1` — i.e. `(1+u64)²·Bd < 1`, satisfied whenever the decay-factor bound
obeys `Bd ≤ 1 − 3·u64` (`wdStepRho_lt_one`). Since `u64 = 2⁻⁵³ ≈ 1.1·10⁻¹⁶`, ANY real weight decay
(`wd ≳ 10⁻¹⁵`) dominates the rounding inflation — the condition is mild and realistic. Under the contraction,
C2/C32's `affine_recur_uniform` gives the UNIFORM budget

    wdRunBound ρ C n B₀  ≤  wdUniformBound ρ C B₀  :=  B₀ + C/(1−ρ)      for ALL n

— so ONE `n`-independent budget check `wdUniformBound … ≤ overflowBound` certifies an ARBITRARILY LONG run:

* `wdUpdateF`/`wdUpdateVec`/`wdTrainStep`/`wdTrainRun` — the weight-decay update, its vector form, the training
  step (backward `gradW` as C57, then the decayed update), and the `n`-step run (fixed-batch model, as C64).
* `wdRunBound`/`wdRunBound_succ_eq`/`wdRunBound_uniform` — the budget iterate, its last-step recurrence
  (`wdRunBound ρ C (n+1) B = ρ·wdRunBound ρ C n B + C` — iterates of an affine map commute), and the UNIFORM
  bound via `affine_recur_uniform`.
* `wdTrainRun_all_finite_uniform` (capstone) — **the horizon-free certificate**: EVERY weight at EVERY step of an
  UNBOUNDED run is `isFinite`, from ONE `n`-independent budget condition.
* `wdTrainRun_forward_all_finite_uniform` — the forward pass `dotF x (weights at step m)` is `isFinite` at every
  step, likewise from one `n`-independent condition.
* `decayFactor_mag`/`wdStepRho_lt_one` — the literal `1.0 − wd` decay factor's magnitude bound (via the exact
  `sub_model`, NOT the crude triangle bound — the cancellation in `1 − wd` is the point), and the mildness of the
  contraction condition.

This mirrors, at the FLOAT-BUDGET level, the ℝ-level weight-decay contraction story: C32 (the horizon-free
whole-run ERROR interval under weight decay), C46/C49 (the clip-interior trapping under the contraction) — here
the same `|1−wd| < 1`-damping makes the overflow-freedom horizon-free.

**Scope (honestly disclosed).** The decay factor `d` is a Float parameter modeling the precomputed `1 − wd`
(`decayFactor_mag` supplies its bound `Bd = (1+u64)·(1−wdlo)` from `toReal wd ∈ [wdlo, 1]`); the contraction
`wdStepRho Bd < 1` is a hypothesis with the mild sufficient condition `Bd ≤ 1 − 3·u64` proved
(`wdStepRho_lt_one`). The FIXED-BATCH model and the C57 lineage persist (self-contained linear-layer backward
`gradW`, not the full `ADReverse` tape); the update is the SGD-shaped decayed `d·w − lr·g` — the Muon-shaped
weight-decay variant (`d·w − lr·NS(g)`) composes identically (thread C61's `nsScalarFBound` into `wdStepC` as
C64's `muonStepC` does) but is not restated here. Reuses C43's single trusted no-overflow axiom
`isFinite_of_bounded` + the `(1+δ)` base — NO new axiom.
-/
import Puffer.RL.WholeRunFinite
import Puffer.RL.MuonTrainBound
open Puffer.FloatR
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound dotBound dotF_isFinite)
open Puffer.RL.ForwardFinite (dotBound_mono)
open Puffer.RL.BackwardFinite (gradW gradW_mag)
open Puffer.RL.LossForwardFinite (sub_bound mul_bound)
open Puffer.RL.MuonTrainBound (affine_recur_uniform)

namespace Puffer.RL.WdRunFinite

/-! ### The contractive step-budget map: slope and constant -/

/-- **The weight-decay budget slope** `(1+u64)²·Bd`: the decay factor's bound `Bd`, inflated by one rounding for
    the product `d·w` and one for the outer subtraction. The step map is contractive iff this is `< 1`. -/
noncomputable def wdStepRho (Bd : ℝ) : ℝ := (1 + u64) ^ 2 * Bd

/-- **The weight-decay step constant**: the gradient-through-update term `(1+u64)²·(Blr·((1+u64)·(Bgs·Bx)))` —
    the `lr·g` budget (gradient bound from `gradW_mag`) through the product and the outer subtraction.
    Step-invariant (the fixed-batch gradient does not depend on the weights), exactly as C64's `sgdStepC`. -/
noncomputable def wdStepC (Bx Bgs Blr : ℝ) : ℝ :=
  (1 + u64) ^ 2 * (Blr * ((1 + u64) * (Bgs * Bx)))

theorem wdStepRho_nonneg (Bd : ℝ) (hd : 0 ≤ Bd) : 0 ≤ wdStepRho Bd :=
  mul_nonneg (sq_nonneg _) hd

theorem wdStepC_nonneg (Bx Bgs Blr : ℝ) (hx : 0 ≤ Bx) (hgs : 0 ≤ Bgs) (hlr : 0 ≤ Blr) :
    0 ≤ wdStepC Bx Bgs Blr := by
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  exact mul_nonneg (sq_nonneg _) (mul_nonneg hlr (mul_nonneg h1 (mul_nonneg hgs hx)))

/-- **The contraction condition is mild**: any decay-factor bound `Bd ≤ 1 − 3·u64` gives `wdStepRho Bd < 1`.
    Since `u64 = 2⁻⁵³ ≈ 1.1·10⁻¹⁶`, any real weight decay (`wd ≳ 10⁻¹⁵`) satisfies it — weight decay dominates
    the rounding inflation. -/
theorem wdStepRho_lt_one (Bd : ℝ) (h : Bd ≤ 1 - 3 * u64) : wdStepRho Bd < 1 := by
  unfold wdStepRho
  have key : (1 + u64) ^ 2 * Bd ≤ (1 + u64) ^ 2 * (1 - 3 * u64) :=
    mul_le_mul_of_nonneg_left h (sq_nonneg _)
  nlinarith [u64_pos, mul_pos u64_pos u64_pos, mul_pos (mul_pos u64_pos u64_pos) u64_pos]

/-- **The literally-computed decay factor** `1.0 − wd` is bounded by `(1+u64)·(1−wdlo)` when
    `toReal wd ∈ [wdlo, 1]` — via the EXACT `sub_model` (the crude triangle bound `(1+u64)·(1+Bwd) ≥ 1` cannot
    see the cancellation in `1 − wd`; the exact model can). Feeding `Bd := (1+u64)·(1−wdlo)` to `wdStepRho`
    gives the contraction for any `wdlo` comfortably above `u64`-scale. -/
theorem decayFactor_mag (wd : Float) (wdlo : ℝ)
    (hlo : wdlo ≤ toReal wd) (hhi : toReal wd ≤ 1) :
    |toReal ((1.0 : Float) - wd)| ≤ (1 + u64) * (1 - wdlo) := by
  obtain ⟨δ, hδ, heq⟩ := sub_model (1.0 : Float) wd
  rw [heq, toReal_oneLit, abs_mul]
  have h1 : |1 - toReal wd| ≤ 1 - wdlo := by
    rw [abs_of_nonneg (by linarith)]
    linarith
  have h2 : |1 + δ| ≤ 1 + u64 := by
    have := abs_le.mp hδ
    rw [abs_le]
    constructor <;> linarith [this.1, this.2]
  calc |1 - toReal wd| * |1 + δ|
      ≤ (1 - wdlo) * (1 + u64) :=
        mul_le_mul h1 h2 (abs_nonneg _) (by linarith [(abs_nonneg (1 - toReal wd)).trans h1])
    _ = (1 + u64) * (1 - wdlo) := by ring

/-! ### The weight-decay update, step, and run -/

/-- **One weight-decay update** `w' = d·w − lr·g` (Float): decay factor `d` (the precomputed `1 − wd`), then the
    gradient step. -/
def wdUpdateF (w g lr d : Float) : Float := d * w - lr * g

/-- **Update magnitude — AFFINE in the weight bound with slope `wdStepRho Bd`**:
    `|toReal (d·w − lr·g)| ≤ wdStepRho Bd·Bw + (1+u64)²·(Blr·Bg)` (two `mul_bound`s and a `sub_bound` over the
    op tree, regrouped). The decayed weight term carries the CONTRACTIVE slope — the whole point. -/
theorem wdUpdateF_mag_le (w g lr d : Float) (Bw Bg Blr Bd : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (hd : |toReal d| ≤ Bd) :
    |toReal (wdUpdateF w g lr d)| ≤ wdStepRho Bd * Bw + (1 + u64) ^ 2 * (Blr * Bg) := by
  have h := sub_bound (d * w) (lr * g) ((1 + u64) * (Bd * Bw)) ((1 + u64) * (Blr * Bg))
    (mul_bound d w Bd Bw hd hw) (mul_bound lr g Blr Bg hlr hg)
  calc |toReal (wdUpdateF w g lr d)|
      = |toReal (d * w - lr * g)| := rfl
    _ ≤ (1 + u64) * ((1 + u64) * (Bd * Bw) + (1 + u64) * (Blr * Bg)) := h
    _ = wdStepRho Bd * Bw + (1 + u64) ^ 2 * (Blr * Bg) := by unfold wdStepRho; ring

/-- **The vectorized weight-decay update** of a weight row by a gradient row. -/
def wdUpdateVec (w g : List Float) (lr d : Float) : List Float :=
  List.zipWith (fun wi gi => wdUpdateF wi gi lr d) w g

/-- Vector-update magnitude: every updated weight obeys the affine single-update bound. -/
theorem wdUpdateVec_mag (lr d : Float) (Bw Bg Blr Bd : ℝ) (hlr : |toReal lr| ≤ Blr)
    (hd : |toReal d| ≤ Bd) :
    ∀ (w g : List Float), (∀ wi ∈ w, |toReal wi| ≤ Bw) → (∀ gi ∈ g, |toReal gi| ≤ Bg) →
      ∀ u ∈ wdUpdateVec w g lr d, |toReal u| ≤ wdStepRho Bd * Bw + (1 + u64) ^ 2 * (Blr * Bg)
  | [], _, _, _ => by simp [wdUpdateVec]
  | _, [], _, _ => by simp [wdUpdateVec]
  | wi :: ws, gi :: gs, hw, hg => by
      intro u hu
      have hstep : wdUpdateVec (wi :: ws) (gi :: gs) lr d
          = wdUpdateF wi gi lr d :: wdUpdateVec ws gs lr d := rfl
      rw [hstep, List.mem_cons] at hu
      rcases hu with rfl | hrest
      · exact wdUpdateF_mag_le wi gi lr d Bw Bg Blr Bd
          (hw wi (List.mem_cons.mpr (Or.inl rfl))) (hg gi (List.mem_cons.mpr (Or.inl rfl)))
          hlr hd
      · exact wdUpdateVec_mag lr d Bw Bg Blr Bd hlr hd ws gs
          (fun q hq => hw q (List.mem_cons.mpr (Or.inr hq)))
          (fun q hq => hg q (List.mem_cons.mpr (Or.inr hq))) u hrest

/-- **One weight-decay training step**: backward gradient `gradW gseed x` (C57), then the decayed update. -/
def wdTrainStep (x w : List Float) (gseed lr d : Float) : List Float :=
  wdUpdateVec w (gradW gseed x) lr d

/-- One step never lengthens the weight row (`wdUpdateVec` is a `zipWith`). -/
theorem wdTrainStep_length (x w : List Float) (gseed lr d : Float) :
    (wdTrainStep x w gseed lr d).length ≤ w.length := by
  simp only [wdTrainStep, wdUpdateVec, List.length_zipWith]
  exact min_le_left _ _

/-- **One weight-decay step maps a `B`-budget to `wdStepRho Bd·B + wdStepC …`** — the CONTRACTIVE affine step
    map (C57's `gradW_mag` gradient budget threaded into the decayed update). -/
theorem wdTrainStep_mag (x w : List Float) (gseed lr d : Float) (Bx B Bgs Blr Bd : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal d| ≤ Bd) :
    ∀ u ∈ wdTrainStep x w gseed lr d, |toReal u| ≤ wdStepRho Bd * B + wdStepC Bx Bgs Blr := by
  intro u hu
  have h := wdUpdateVec_mag lr d B ((1 + u64) * (Bgs * Bx)) Blr Bd hlr hd w (gradW gseed x) hw
    (gradW_mag gseed x Bgs Bx hgs hx) u hu
  unfold wdStepC
  exact h

/-- **The n-step weight-decay training run** (fixed batch per step, as C64 — see the module scope note). -/
def wdTrainRun (x : List Float) (gseed lr d : Float) : Nat → List Float → List Float
  | 0, w => w
  | n + 1, w => wdTrainRun x gseed lr d n (wdTrainStep x w gseed lr d)

/-- The run never lengthens the weight row. -/
theorem wdTrainRun_length (x : List Float) (gseed lr d : Float) :
    ∀ (n : Nat) (w : List Float), (wdTrainRun x gseed lr d n w).length ≤ w.length
  | 0, _ => le_refl _
  | n + 1, w => (wdTrainRun_length x gseed lr d n _).trans (wdTrainStep_length x w gseed lr d)

/-! ### The budget iterate and its UNIFORM bound -/

/-- **The n-step budget iterate** of the affine map `B ↦ ρ·B + C` (inside-first, matching the run's fold). -/
noncomputable def wdRunBound (ρ C : ℝ) : Nat → ℝ → ℝ
  | 0, B => B
  | n + 1, B => wdRunBound ρ C n (ρ * B + C)

/-- **The last-step recurrence**: `wdRunBound ρ C (n+1) B = ρ·wdRunBound ρ C n B + C` — iterates of an affine
    map commute, converting the inside-first fold into the `a(n+1) ≤ ρ·a(n) + C` shape `affine_recur_uniform`
    consumes. -/
theorem wdRunBound_succ_eq (ρ C : ℝ) : ∀ (n : Nat) (B : ℝ),
    wdRunBound ρ C (n + 1) B = ρ * wdRunBound ρ C n B + C
  | 0, _ => rfl
  | n + 1, B => wdRunBound_succ_eq ρ C n (ρ * B + C)

/-- **The uniform budget** `B + C/(1−ρ)` — an `n`-INDEPENDENT cap on the whole run's weight budgets. -/
noncomputable def wdUniformBound (ρ C B : ℝ) : ℝ := B + C / (1 - ρ)

/-- **THE UNIFORM BOUND (the crux).** Under the contraction `ρ < 1` (nonneg `ρ`, `C`, `B`), the budget iterate
    is bounded UNIFORMLY IN `n`: `wdRunBound ρ C n B ≤ wdUniformBound ρ C B` — C2/C32's `affine_recur_uniform`
    at the sequence `a n := wdRunBound ρ C n B` (the last-step recurrence supplies `hrec` exactly). -/
theorem wdRunBound_uniform (ρ C : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : 0 ≤ C) (B : ℝ)
    (hB : 0 ≤ B) (n : Nat) : wdRunBound ρ C n B ≤ wdUniformBound ρ C B :=
  affine_recur_uniform (fun k => wdRunBound ρ C k B) ρ C hρ0 hρ1 hC hB
    (fun k => le_of_eq (wdRunBound_succ_eq ρ C k B)) n

/-- **The run invariant**: initial weights `≤ B` ⟹ every weight after `n` steps
    `≤ wdRunBound (wdStepRho Bd) (wdStepC …) n B` (induction over the fold, one `wdTrainStep_mag` per step). -/
theorem wdTrainRun_mag (x : List Float) (gseed lr d : Float) (Bx Bgs Blr Bd : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (hd : |toReal d| ≤ Bd) :
    ∀ (n : Nat) (w : List Float) (B : ℝ), (∀ wi ∈ w, |toReal wi| ≤ B) →
      ∀ u ∈ wdTrainRun x gseed lr d n w,
        |toReal u| ≤ wdRunBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) n B
  | 0, _, _, hw => hw
  | n + 1, w, B, hw =>
      wdTrainRun_mag x gseed lr d Bx Bgs Blr Bd hx hgs hlr hd n (wdTrainStep x w gseed lr d)
        (wdStepRho Bd * B + wdStepC Bx Bgs Blr)
        (wdTrainStep_mag x w gseed lr d Bx B Bgs Blr Bd hx hw hgs hlr hd)

/-! ### The horizon-free capstones -/

/-- **THE WEIGHT-DECAY WHOLE-RUN IS OVERFLOW-FREE — HORIZON-FREE.** For a fixed-batch weight-decay run on
    bounded inputs/weights/seed/step-size/decay-factor, under the contraction `wdStepRho Bd < 1` (mild:
    `Bd ≤ 1 − 3·u64` suffices, `wdStepRho_lt_one`), with the SINGLE `n`-INDEPENDENT budget condition
    `wdUniformBound … ≤ overflowBound`, EVERY weight at EVERY step of an ARBITRARILY LONG run is `isFinite`.
    C64's named refinement delivered: the weight-decay contraction makes the finiteness certificate
    horizon-free, mirroring the ℝ-level C32 story at the Float-budget level. -/
theorem wdTrainRun_all_finite_uniform (x w : List Float) (gseed lr d : Float)
    (Bx B0 Bgs Blr Bd : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal d| ≤ Bd)
    (hcontract : wdStepRho Bd < 1)
    (hbound : wdUniformBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) B0 ≤ overflowBound) :
    ∀ m, ∀ u ∈ wdTrainRun x gseed lr d m w, u.isFinite = true := by
  intro m u hu
  have hρ0 : 0 ≤ wdStepRho Bd := wdStepRho_nonneg Bd ((abs_nonneg _).trans hd)
  have hC : 0 ≤ wdStepC Bx Bgs Blr :=
    wdStepC_nonneg Bx Bgs Blr hBx0 ((abs_nonneg _).trans hgs) ((abs_nonneg _).trans hlr)
  exact isFinite_of_bounded _
    (((wdTrainRun_mag x gseed lr d Bx Bgs Blr Bd hx hgs hlr hd m w B0 hw u hu).trans
      (wdRunBound_uniform (wdStepRho Bd) (wdStepC Bx Bgs Blr) hρ0 hcontract hC B0 hB00 m)).trans
      hbound)

/-- **The forward pass is overflow-free at every step — horizon-free**: `dotF x (weights at step m)` is
    `isFinite` for ALL `m`, from one `n`-INDEPENDENT `dotBound` budget at the max of the input bound and the
    uniform weight budget (the run never lengthens the weight row, so `dotBound_mono` covers every step). -/
theorem wdTrainRun_forward_all_finite_uniform (x w : List Float) (gseed lr d : Float)
    (Bx B0 Bgs Blr Bd : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal d| ≤ Bd)
    (hcontract : wdStepRho Bd < 1)
    (hfwd : dotBound (min x.length w.length)
        (max Bx (wdUniformBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) B0)) ≤ overflowBound) :
    ∀ m, (Puffer.RL.FiniteBound.dotF x (wdTrainRun x gseed lr d m w)).isFinite = true := by
  intro m
  have hρ0 : 0 ≤ wdStepRho Bd := wdStepRho_nonneg Bd ((abs_nonneg _).trans hd)
  have hC : 0 ≤ wdStepC Bx Bgs Blr :=
    wdStepC_nonneg Bx Bgs Blr hBx0 ((abs_nonneg _).trans hgs) ((abs_nonneg _).trans hlr)
  have hBmax0 : 0 ≤ max Bx (wdUniformBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) B0) :=
    hBx0.trans (le_max_left _ _)
  have hx' : ∀ xi ∈ x,
      |toReal xi| ≤ max Bx (wdUniformBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) B0) :=
    fun xi hxi => (hx xi hxi).trans (le_max_left _ _)
  have hw' : ∀ wi ∈ wdTrainRun x gseed lr d m w,
      |toReal wi| ≤ max Bx (wdUniformBound (wdStepRho Bd) (wdStepC Bx Bgs Blr) B0) := fun wi hwi =>
    ((wdTrainRun_mag x gseed lr d Bx Bgs Blr Bd hx hgs hlr hd m w B0 hw wi hwi).trans
      (wdRunBound_uniform (wdStepRho Bd) (wdStepC Bx Bgs Blr) hρ0 hcontract hC B0 hB00 m)).trans
      (le_max_right _ _)
  have hlen : min x.length (wdTrainRun x gseed lr d m w).length ≤ min x.length w.length :=
    min_le_min (le_refl _) (wdTrainRun_length x gseed lr d m w)
  exact dotF_isFinite _ hBmax0 x (wdTrainRun x gseed lr d m w) hx' hw'
    ((dotBound_mono _ hlen).trans hfwd)

/-- Satisfiability of the contraction: the boundary decay bound `1 − 3·u64` itself contracts. -/
example : wdStepRho (1 - 3 * u64) < 1 := wdStepRho_lt_one _ (le_refl _)

end Puffer.RL.WdRunFinite
