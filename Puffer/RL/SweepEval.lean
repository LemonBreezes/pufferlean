/-
# Runnable sweep-budget evaluator + the all-Bool tape certificate

C78 (`BudgetEval`) made the dot/run/horizon-free budget checks runnable (computable Float
evaluators with PROVEN domination) and disclosed C68's `sweepBound` — the tape-sweep budget, a
two-level `edgeBound`/`nodeBound` recursion — as the remaining ℝ-side check. This module completes
it, and composes the result with C71/C77's comp-tape capstones into tape certificates whose budget
hypotheses are runtime `Bool`s.

**The sweep evaluator (C78's named remainder).** The ℝ recursions are
`edgeBound A D C (k+1) = (1+u64)·(edgeBound k + (1+u64)·(A·D))` (TWO ℝ-side `(1+u64)` factors per
edge), `nodeBound E D C = edgeBound C D C E`, and the inside-first
`sweepBound E D (m+1) C = sweepBound E D m (nodeBound E D C)`. The Float mirrors replace each
`(1+u64)` by C78's `slackF`; the per-edge rounding accounting is exactly C78's `dotBoundF` shape:
four adverse roundings per edge step (`A·D`, the inner `slackF·`, the add, the outer `slackF·`),
covered by two `slackF_key` spends (each absorbing one ℝ-side factor + two roundings). The
generic step is factored once (`slack_step`) and reused everywhere. `sweepBoundF`'s domination
generalizes over the running budget exactly as C78's `runBoundF_dominates` — no monotonicity
lemma needed. `sweepBound_le_overflow_of_check` turns ONE `checkLe … capF` Bool into the
ℝ-side `sweepBound E D n C ≤ overflowBound` hypothesis the tape capstones consume.

**The weight-budget evaluator (log-free).** C71's `compWeightBound B e` is an all-`max` recursion
over exact (`toReal_max`) structure with three non-trivial entries: constants `|toReal c|`
(mirrored EXACTLY by `absF c := max c (−c)` — `toReal_max`/`toReal_neg` are exact, no slack),
`fwdBound` sibling values (mirrored by `fwdBoundF`, one `slackF` per rounded op), and the exp
entry `Real.exp(fwdBound)·(1+expEps)` (mirrored by `slackF · Float.exp (fwdBoundF …)`, covered by
the second key `slackF_key_exp : (1+expEps) ≤ toReal slackF·((1−expEps)·(1−u64))` — the exp
model's `expEps` rounding + the mul's `u64` rounding in one slack). `compWeightBoundF_dominates`
gives the full log-free domination.

**The capstones.**
* `adGrad_isFinite_comp_runnable` — the ALL-BOOL(+plumbing) log-free tape certificate: the inputs
  as a recorded list (C73's zero-padded `padRow` representation `hrep` — the caller's data
  plumbing), THREE runtime Bools (`checkRegion inputs Bf`, `checkLe 0.0 Bf`, and ONE
  `checkLe (sweepBoundF 2 (compWeightBoundF Bf e) n 1.0) capF` — the weight budget COMPUTED by the
  harness from the expression, nothing plumbed), and the syntactic `LogFree e` ⟹ every gradient
  the ACTUAL `grads` engine produces on the compiled tape is overflow-free.
* `adGrad_isFinite_comp_partition_runnable` — the log-inclusive (C77 `PartitionLogs`) version:
  the same Bools, with the weight budget's Float representative `Df` supplied under the ONE
  plumbed hypothesis `compWeightBoundL … ≤ toReal Df`.

**Scope (honestly disclosed).** The evaluators are upward-slacked (sound-not-complete, C78's
philosophy; the conservatism compounds through the sweep exactly as C78's `runBoundF`). The
log-free weight budget is FULLY runnable; the log-INCLUSIVE `compWeightBoundL` mirror is NOT
built here because its log entry `(1/c)·(1+u64)` needs a LOWER Float representative of the floor
`c = floorC (…)` — the DIRECTION REVERSAL: domination of `1/c` requires `toReal cF ≤ c` (a
smaller Float floor gives a LARGER, dominating `1/cF`), the opposite direction from every other
representative in the chain, and `floorC`'s own evaluation (`exp(−M)·…`) would need
downward-rounded Float transcendentals — the disclosed remaining enhancement. Until then the
log-inclusive capstone takes `Df` as the caller's ONE plumbed constant. NO new axiom: C78's
`slackF`/`capF` machinery, the pre-existing models, and the exact `toReal_max`/`toReal_neg`/
`toReal_oneLit`/`toReal_zeroLit`.
-/
import Puffer.RL.BudgetEval
import Puffer.RL.CompTapeFloor
import Puffer.RL.TraceCheck

namespace Puffer.RL.SweepEval

open Puffer.FloatR (toReal u64 u64_pos u64_lt_one expEps expEps_pos exp_model add_model mul_model
  toReal_max toReal_neg toReal_oneLit toReal_zeroLit)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr)
open Puffer.FloatR.ADReverse (comp)
open Puffer.RL.FiniteBound (overflowBound)
open Puffer.RL.ADTapeFinite (LogFree fwdBound edgeBound nodeBound sweepBound edgeBound_nonneg
  edgeBound_succ)
open Puffer.RL.CompTapeWeights (compWeightBound compWeightBound_nonneg adGrad_isFinite_comp)
open Puffer.RL.CompTapeLog (compWeightBoundL compWeightBoundL_nonneg)
open Puffer.RL.CompTapeFloor (PartitionLogs logArgBound fwdBound_nonneg
  adGrad_isFinite_comp_partition)
open Puffer.RL.FloatFloor (floorC)
open Puffer.RL.BudgetEval (slackF slackF_lower slackF_nonneg slackF_key capF capF_le)
open Puffer.RL.MarginCheck (checkLe checkLe_sound checkRegion)
open Puffer.RL.TraceCheck (padRow padRow_abs_le)

/-! ### Small shared facts -/

private theorem one_sub_u64' : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]

/-- `expEps = 2⁻⁵² ≤ 10⁻⁶` (from the definition, as C78 derived the analogous `u64` fact). -/
theorem expEps_le_small : expEps ≤ (1 : ℝ) / 1000000 := by
  unfold Puffer.FloatR.expEps; norm_num

theorem expEps_lt_one : expEps < 1 :=
  lt_of_le_of_lt expEps_le_small (by norm_num)

private theorem one_sub_expEps' : (0 : ℝ) ≤ 1 - expEps := by linarith [expEps_lt_one]

/-- **The second slack key** (proved once, for the exp entry): `(1+expEps) ≤ toReal slackF ·
    ((1−expEps)·(1−u64))` — one `slackF` factor covers the ℝ-side `(1+expEps)` exp budget, the
    exp model's adverse `(1−expEps)` rounding, AND the multiplication's adverse `(1−u64)`
    rounding. Numerically comfortable: `expEps = 2⁻⁵²`, `u64 = 2⁻⁵³`, `slackF ≈ 1.001`. -/
theorem slackF_key_exp : (1 + expEps) ≤ toReal slackF * ((1 - expEps) * (1 - u64)) := by
  have h := slackF_lower
  have hprod : (0 : ℝ) ≤ (1 - expEps) * (1 - u64) := mul_nonneg one_sub_expEps' one_sub_u64'
  have step : (1.001 : ℝ) * (1 - u64) * ((1 - expEps) * (1 - u64))
      ≤ toReal slackF * ((1 - expEps) * (1 - u64)) := mul_le_mul_of_nonneg_right h hprod
  refine le_trans ?_ step
  have hu : u64 ≤ (1 : ℝ) / 1000000 := by unfold Puffer.FloatR.u64; norm_num
  have he : expEps ≤ (1 : ℝ) / 1000000 := expEps_le_small
  nlinarith [u64_pos, expEps_pos, hu, he, mul_nonneg u64_pos.le expEps_pos.le,
    sq_nonneg u64, sq_nonneg expEps,
    mul_nonneg (mul_nonneg u64_pos.le u64_pos.le) expEps_pos.le,
    mul_nonneg (mul_nonneg u64_pos.le expEps_pos.le) expEps_pos.le]

/-- **The generic slack step** (factored once, reused by every evaluator case): if the ℝ value
    `y` is dominated by the Float op's pre-rounding value (`y ≤ Y` with `toReal opF = Y·(1+δ)`,
    one adverse rounding), then `slackF · opF` dominates `(1+u64)·y` — `slackF_key` absorbing the
    ℝ-side factor, the op's rounding, and the outer multiplication's rounding. -/
theorem slack_step (opF : Float) (y Y : ℝ) (hy0 : 0 ≤ y) (hyY : y ≤ Y)
    (hop : ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal opF = Y * (1 + δ)) :
    (1 + u64) * y ≤ toReal (slackF * opF) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := hop
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF opF
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hY0 : 0 ≤ Y := le_trans hy0 hyY
  have h1δ : (0 : ℝ) ≤ 1 + δ₁ := by
    have := (abs_le.mp hδ₁).1; linarith [u64_lt_one]
  have hop0 : 0 ≤ toReal opF := by rw [e₁]; exact mul_nonneg hY0 h1δ
  rw [e₂]
  calc (1 + u64) * y
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * y :=
        mul_le_mul_of_nonneg_right slackF_key hy0
    _ = (toReal slackF * (y * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * (Y * (1 + δ₁))) * (1 - u64) := by
        have h1 : y * (1 - u64) ≤ Y * (1 + δ₁) :=
          le_trans (mul_le_mul_of_nonneg_right hyY one_sub_u64')
            (mul_le_mul_of_nonneg_left hd₁ hY0)
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h1 slackF_nonneg)
          one_sub_u64'
    _ = (toReal slackF * toReal opF) * (1 - u64) := by rw [← e₁]
    _ ≤ (toReal slackF * toReal opF) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg slackF_nonneg hop0)

/-- **The exp slack step**: `Real.exp a·(1+expEps) ≤ toReal (slackF · Float.exp af)` whenever
    `a ≤ toReal af` — `slackF_key_exp` absorbing the exp budget's `(1+expEps)`, the exp model's
    adverse rounding, and the multiplication's rounding. -/
theorem slack_exp_step (af : Float) (a : ℝ) (haa : a ≤ toReal af) :
    Real.exp a * (1 + expEps) ≤ toReal (slackF * Float.exp af) := by
  obtain ⟨δe, hδe, ee⟩ := exp_model af
  obtain ⟨δm, hδm, em⟩ := mul_model slackF (Float.exp af)
  have hde : 1 - expEps ≤ 1 + δe := by have := (abs_le.mp hδe).1; linarith
  have hdm : 1 - u64 ≤ 1 + δm := by have := (abs_le.mp hδm).1; linarith
  have h1δe : (0 : ℝ) ≤ 1 + δe := by
    have := (abs_le.mp hδe).1; linarith [expEps_lt_one]
  have hE0 : 0 ≤ toReal (Float.exp af) := by
    rw [ee]; exact mul_nonneg (Real.exp_pos _).le h1δe
  have hmono : Real.exp a ≤ Real.exp (toReal af) := Real.exp_le_exp.mpr haa
  rw [em]
  calc Real.exp a * (1 + expEps)
      = (1 + expEps) * Real.exp a := by ring
    _ ≤ (toReal slackF * ((1 - expEps) * (1 - u64))) * Real.exp a :=
        mul_le_mul_of_nonneg_right slackF_key_exp (Real.exp_pos _).le
    _ = (toReal slackF * (Real.exp a * (1 - expEps))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * (Real.exp (toReal af) * (1 + δe))) * (1 - u64) := by
        have h1 : Real.exp a * (1 - expEps) ≤ Real.exp (toReal af) * (1 + δe) :=
          le_trans (mul_le_mul_of_nonneg_right hmono one_sub_expEps')
            (mul_le_mul_of_nonneg_left hde (Real.exp_pos _).le)
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h1 slackF_nonneg)
          one_sub_u64'
    _ = (toReal slackF * toReal (Float.exp af)) * (1 - u64) := by rw [← ee]
    _ ≤ (toReal slackF * toReal (Float.exp af)) * (1 + δm) :=
        mul_le_mul_of_nonneg_left hdm (mul_nonneg slackF_nonneg hE0)

/-! ### The sweep-budget evaluator (C78's named remainder) -/

/-- Computable evaluator of C68's `edgeBound`, `slackF` in place of each `(1+u64)`:
    `edgeBoundF Af Df Cf (k+1) = slackF·(edgeBoundF k + slackF·(Af·Df))`. -/
def edgeBoundF (A D C : Float) : Nat → Float
  | 0 => C
  | k + 1 => slackF * (edgeBoundF A D C k + slackF * (A * D))

/-- **Edge-budget domination**: four adverse roundings per step (`A·D`, the inner `slackF·`, the
    add, the outer `slackF·`) against the ℝ side's two `(1+u64)` factors — two `slack_step`
    spends, exactly C78's `dotBoundF` accounting. -/
theorem edgeBoundF_dominates (Af Df Cf : Float) (A D C : ℝ)
    (hA0 : 0 ≤ A) (hAa : A ≤ toReal Af) (hD0 : 0 ≤ D) (hDd : D ≤ toReal Df)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) :
    ∀ k, edgeBound A D C k ≤ toReal (edgeBoundF Af Df Cf k)
  | 0 => hCc
  | k + 1 => by
      have IH := edgeBoundF_dominates Af Df Cf A D C hA0 hAa hD0 hDd hC0 hCc k
      have hAD0 : 0 ≤ A * D := mul_nonneg hA0 hD0
      have hADd : A * D ≤ toReal Af * toReal Df :=
        mul_le_mul hAa hDd hD0 (hA0.trans hAa)
      have hinner : (1 + u64) * (A * D) ≤ toReal (slackF * (Af * Df)) :=
        slack_step (Af * Df) (A * D) (toReal Af * toReal Df) hAD0 hADd (mul_model Af Df)
      have hinner0 : 0 ≤ (1 + u64) * (A * D) :=
        mul_nonneg (by linarith [u64_pos]) hAD0
      have hedge0 : 0 ≤ edgeBound A D C k := edgeBound_nonneg A D C hA0 hD0 hC0 k
      have hsum0 : 0 ≤ edgeBound A D C k + (1 + u64) * (A * D) := add_nonneg hedge0 hinner0
      have hsumd : edgeBound A D C k + (1 + u64) * (A * D)
          ≤ toReal (edgeBoundF Af Df Cf k) + toReal (slackF * (Af * Df)) :=
        add_le_add IH hinner
      have houter := slack_step (edgeBoundF Af Df Cf k + slackF * (Af * Df))
        (edgeBound A D C k + (1 + u64) * (A * D))
        (toReal (edgeBoundF Af Df Cf k) + toReal (slackF * (Af * Df)))
        hsum0 hsumd (add_model _ _)
      rw [edgeBound_succ]
      exact houter

/-- Computable evaluator of C68's `nodeBound E D C = edgeBound C D C E`. -/
def nodeBoundF (E : Nat) (D C : Float) : Float := edgeBoundF C D C E

theorem nodeBoundF_dominates (E : Nat) (Df Cf : Float) (D C : ℝ)
    (hD0 : 0 ≤ D) (hDd : D ≤ toReal Df) (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) :
    nodeBound E D C ≤ toReal (nodeBoundF E Df Cf) :=
  edgeBoundF_dominates Cf Df Cf C D C hC0 hCc hD0 hDd hC0 hCc E

/-- Computable evaluator of C68's `sweepBound` (inside-first iterate, as the original). -/
def sweepBoundF (E : Nat) (D : Float) : Nat → Float → Float
  | 0, C => C
  | m + 1, C => sweepBoundF E D m (nodeBoundF E D C)

/-- **Sweep-budget domination** — the induction generalizes over the running budget, re-seeding
    with the dominated per-node pair, exactly as C78's `runBoundF_dominates`. -/
theorem sweepBoundF_dominates (E : Nat) (Df : Float) (D : ℝ)
    (hD0 : 0 ≤ D) (hDd : D ≤ toReal Df) :
    ∀ (m : Nat) (Cf : Float) (C : ℝ), 0 ≤ C → C ≤ toReal Cf →
      sweepBound E D m C ≤ toReal (sweepBoundF E Df m Cf)
  | 0, _, _, _, hCc => hCc
  | m + 1, Cf, C, hC0, hCc => by
      have hnode := nodeBoundF_dominates E Df Cf D C hD0 hDd hC0 hCc
      have hnode0 : 0 ≤ nodeBound E D C := edgeBound_nonneg C D C hC0 hD0 hC0 E
      show sweepBound E D m (nodeBound E D C)
          ≤ toReal (sweepBoundF E Df m (nodeBoundF E Df Cf))
      exact sweepBoundF_dominates E Df D hD0 hDd m (nodeBoundF E Df Cf) (nodeBound E D C)
        hnode0 hnode

/-- **The runnable sweep discharge**: one `checkLe … capF` Bool certifies the ℝ-side
    `sweepBound E D n C ≤ overflowBound` — the exact hypothesis C68/C71/C75/C77's tape
    capstones consume. -/
theorem sweepBound_le_overflow_of_check (E : Nat) (Df Cf : Float) (D C : ℝ) (n : Nat)
    (hD0 : 0 ≤ D) (hDd : D ≤ toReal Df) (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf)
    (hchk : checkLe (sweepBoundF E Df n Cf) capF = true) :
    sweepBound E D n C ≤ overflowBound :=
  ((sweepBoundF_dominates E Df D hD0 hDd n Cf C hC0 hCc).trans
    (checkLe_sound hchk)).trans capF_le

/-! ### The log-free weight-budget evaluator -/

/-- Exact Float absolute value: `max c (−c)` — `toReal (absF c) = |toReal c|` with NO rounding
    (the `max`/negation embeddings are exact axioms). -/
def absF (c : Float) : Float := max c (-c)

theorem toReal_absF (c : Float) : toReal (absF c) = |toReal c| := by
  unfold absF
  rw [toReal_max, toReal_neg]
  exact (abs_eq_max_neg (a := toReal c)).symm

/-- Computable evaluator of C68's `fwdBound` on the log-free grammar: `slackF` per rounded op,
    `absF` for the exact constant magnitudes, `slackF · Float.exp` for the exp budget, exact
    `max` for relu/max/min. (`log` gets a junk `0.0`; domination requires `LogFree`.) -/
def fwdBoundF (B : Float) : Expr → Float
  | .var _ => B
  | .const c => absF c
  | .add a b => slackF * (fwdBoundF B a + fwdBoundF B b)
  | .sub a b => slackF * (fwdBoundF B a + fwdBoundF B b)
  | .mul a b => slackF * (fwdBoundF B a * fwdBoundF B b)
  | .scale c a => slackF * (absF c * fwdBoundF B a)
  | .exp a => slackF * Float.exp (fwdBoundF B a)
  | .log _ => 0.0
  | .relu a => fwdBoundF B a
  | .max a b => max (fwdBoundF B a) (fwdBoundF B b)
  | .min a b => max (fwdBoundF B a) (fwdBoundF B b)

/-- **Forward-budget domination (log-free)**: `fwdBound B e ≤ toReal (fwdBoundF Bf e)`. One
    `slack_step` per rounded op (add/sub/mul/scale), `slack_exp_step` for exp, exact transport
    for var/const/relu/max/min. -/
theorem fwdBoundF_dominates (Bf : Float) (B : ℝ) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    ∀ e : Expr, LogFree e → fwdBound B e ≤ toReal (fwdBoundF Bf e) := by
  intro e
  induction e with
  | var i => intro _; exact hBb
  | const c => intro _; exact le_of_eq (toReal_absF c).symm
  | add a b iha ihb =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      exact slack_step _ _ _
        (add_nonneg (fwdBound_nonneg B hB0 a) (fwdBound_nonneg B hB0 b))
        (add_le_add (iha hlf.1) (ihb hlf.2)) (add_model _ _)
  | sub a b iha ihb =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      exact slack_step _ _ _
        (add_nonneg (fwdBound_nonneg B hB0 a) (fwdBound_nonneg B hB0 b))
        (add_le_add (iha hlf.1) (ihb hlf.2)) (add_model _ _)
  | mul a b iha ihb =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      exact slack_step _ _ _
        (mul_nonneg (fwdBound_nonneg B hB0 a) (fwdBound_nonneg B hB0 b))
        (mul_le_mul (iha hlf.1) (ihb hlf.2) (fwdBound_nonneg B hB0 b)
          ((fwdBound_nonneg B hB0 a).trans (iha hlf.1)))
        (mul_model _ _)
  | scale c a iha =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      refine slack_step _ _ _
        (mul_nonneg (abs_nonneg _) (fwdBound_nonneg B hB0 a)) ?_ (mul_model _ _)
      rw [toReal_absF]
      exact mul_le_mul_of_nonneg_left (iha hlf) (abs_nonneg _)
  | exp a iha =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      exact slack_exp_step _ _ (iha hlf)
  | log a _ => intro hlf; exact hlf.elim
  | relu a iha => intro hlf; exact iha hlf
  | max a b iha ihb =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      rw [toReal_max]
      exact max_le_max (iha hlf.1) (ihb hlf.2)
  | min a b iha ihb =>
      intro hlf
      simp only [fwdBound, fwdBoundF]
      rw [toReal_max]
      exact max_le_max (iha hlf.1) (ihb hlf.2)

/-- Computable evaluator of C71's `compWeightBound` on the log-free grammar — the all-`max`
    mirror, with `absF` (exact), `fwdBoundF` (sibling values), and the slacked exp entry. -/
def compWeightBoundF (B : Float) : Expr → Float
  | .var _ => 1.0
  | .const _ => 1.0
  | .add a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .sub a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .mul a b => max (max (fwdBoundF B a) (fwdBoundF B b))
      (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .scale c a => max (absF c) (compWeightBoundF B a)
  | .exp a => max (slackF * Float.exp (fwdBoundF B a)) (compWeightBoundF B a)
  | .log _ => 0.0
  | .relu a => max 1.0 (compWeightBoundF B a)
  | .max a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .min a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))

/-- **Weight-budget domination (log-free)**: `compWeightBound B e ≤ toReal (compWeightBoundF Bf
    e)` — the `max` spine transports exactly (`toReal_max`), the leaves by `toReal_oneLit`/
    `toReal_absF` (exact), `fwdBoundF_dominates`, and `slack_exp_step`. -/
theorem compWeightBoundF_dominates (Bf : Float) (B : ℝ) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    ∀ e : Expr, LogFree e → compWeightBound B e ≤ toReal (compWeightBoundF Bf e) := by
  intro e
  induction e with
  | var i => intro _; exact le_of_eq toReal_oneLit.symm
  | const c => intro _; exact le_of_eq toReal_oneLit.symm
  | add a b iha ihb =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hlf.1) (ihb hlf.2))
  | sub a b iha ihb =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hlf.1) (ihb hlf.2))
  | mul a b iha ihb =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_max, toReal_max]
      exact max_le_max
        (max_le_max (fwdBoundF_dominates Bf B hB0 hBb a hlf.1)
          (fwdBoundF_dominates Bf B hB0 hBb b hlf.2))
        (max_le_max (iha hlf.1) (ihb hlf.2))
  | scale c a iha =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max]
      exact max_le_max (le_of_eq (toReal_absF c).symm) (iha hlf)
  | exp a iha =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max]
      exact max_le_max (slack_exp_step _ _ (fwdBoundF_dominates Bf B hB0 hBb a hlf)) (iha hlf)
  | log a _ => intro hlf; exact hlf.elim
  | relu a iha =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (iha hlf)
  | max a b iha ihb =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hlf.1) (ihb hlf.2))
  | min a b iha ihb =>
      intro hlf
      simp only [compWeightBound, compWeightBoundF]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hlf.1) (ihb hlf.2))

/-! ### The all-Bool tape certificates -/

/-- **THE ALL-BOOL LOG-FREE TAPE CERTIFICATE (capstone).** For a `LogFree` expression compiled by
    the ACTUAL `comp` from `Tape.empty`, with the inputs recorded as a list (`hrep` — C73's
    zero-padded representation, the caller's data plumbing): THREE runtime Bools —
    `checkRegion inputs Bf`, `checkLe 0.0 Bf`, and ONE sweep-budget check whose weight budget
    `compWeightBoundF Bf e` is COMPUTED by the harness from the expression — certify that EVERY
    gradient the actual `grads` engine produces is overflow-free. Nothing else is plumbed: the
    domination and the threshold's safety are proved offline, once. -/
theorem adGrad_isFinite_comp_runnable (inputs : List Float) (σ : Nat → Float) (Bf : Float)
    (hrep : ∀ i, σ i = padRow inputs i)
    (hin : checkRegion inputs Bf = true) (hB0 : checkLe 0.0 Bf = true)
    (e : Expr) (hlf : LogFree e) (root : V)
    (hbudget : checkLe (sweepBoundF 2 (compWeightBoundF Bf e)
        (comp σ e Tape.empty).2.val.size 1.0) capF = true) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true := by
  have hB0' : (0 : ℝ) ≤ toReal Bf := by
    have := checkLe_sound hB0
    rwa [toReal_zeroLit] at this
  have hσ : ∀ i, |toReal (σ i)| ≤ toReal Bf := by
    intro i
    rw [hrep i]
    exact padRow_abs_le hin hB0' i
  exact adGrad_isFinite_comp σ (toReal Bf) hσ e hlf root
    (sweepBound_le_overflow_of_check 2 (compWeightBoundF Bf e) 1.0
      (compWeightBound (toReal Bf) e) 1 _
      (compWeightBound_nonneg (toReal Bf) e)
      (compWeightBoundF_dominates Bf (toReal Bf) hB0' le_rfl e hlf)
      zero_le_one (le_of_eq toReal_oneLit.symm) hbudget)

/-- **The log-inclusive (partition-shaped) runnable certificate.** As above for C77's
    `PartitionLogs` capstone, with the log-inclusive weight budget's Float representative `Df`
    supplied under the ONE plumbed hypothesis `hDf` (the log-inclusive Float mirror needs the
    floor's LOWER representative — the disclosed direction reversal — and remains the
    enhancement; every other hypothesis is a runtime Bool or C73-style plumbing). -/
theorem adGrad_isFinite_comp_partition_runnable (inputs : List Float) (σ : Nat → Float)
    (Bf Df : Float)
    (hrep : ∀ i, σ i = padRow inputs i)
    (hin : checkRegion inputs Bf = true) (hB0 : checkLe 0.0 Bf = true)
    (e : Expr) (hp : PartitionLogs e) (root : V)
    (hDf : compWeightBoundL (toReal Bf) (floorC (logArgBound (toReal Bf) e)) e ≤ toReal Df)
    (hbudget : checkLe (sweepBoundF 2 Df
        (comp σ e Tape.empty).2.val.size 1.0) capF = true) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true := by
  have hB0' : (0 : ℝ) ≤ toReal Bf := by
    have := checkLe_sound hB0
    rwa [toReal_zeroLit] at this
  have hσ : ∀ i, |toReal (σ i)| ≤ toReal Bf := by
    intro i
    rw [hrep i]
    exact padRow_abs_le hin hB0' i
  exact adGrad_isFinite_comp_partition σ (toReal Bf) hσ e hp root
    (sweepBound_le_overflow_of_check 2 Df 1.0
      (compWeightBoundL (toReal Bf) (floorC (logArgBound (toReal Bf) e)) e) 1 _
      (compWeightBoundL_nonneg _ _ e) hDf
      zero_le_one (le_of_eq toReal_oneLit.symm) hbudget)

end Puffer.RL.SweepEval
