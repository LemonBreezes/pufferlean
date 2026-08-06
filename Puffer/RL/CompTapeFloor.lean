/-
# The tape certificate from VALUE BOUNDS alone: C75's floor discharged for partition-shaped logs

C75 (`CompTapeLog`) delivered the full-grammar comp-tape certificate `adGrad_isFinite_comp_log`,
but under the per-input floor hypothesis `LogFloored σ c e` — every `log` argument's Float value
`≥ c > 0`, a semantic side-condition left to the caller. C76 (`FloatFloor`) supplied the
quantitative floor for PARTITION-SHAPED log arguments (`log (add (exp a) b)` — the compiled
softmax log-partition, the ONLY `log` shape the PPO loss uses): `evalF_partition_floor` floors
such an argument at `floorC M` once the inner logit's value is `≥ −M`. This module is their
COMPOSITION — the `LogFloored` hypothesis is DISCHARGED from plain input bounds:

* `NonnegShape b` — the partition tail's syntax (`exp` of a `LogFree` term, a NONNEGATIVE
  constant, or an `add` of two such shapes: covers both `exp l₁ + (exp l₂ + …)` and the repo's
  canonical `SoftmaxExpr.expSumE` fold, which terminates in `.const 0` — its side condition
  `0 ≤ toReal 0` discharged by `toReal_zero`); `nonnegShape_logFree` and `nonnegShape_nonneg`
  (the tail's Float value is nonnegative — `expF_pos` per term, the const's side condition,
  C76's `add_floor` at `x = 0` per add).
* `PartitionLogs e` — every `log` in `e` has a partition-shaped argument `add (exp a) b` with a
  `LogFree` inner logit `a` and a `NonnegShape` tail `b` (structural everywhere else). This is
  exactly the PPO loss's log usage: the log-partition of a softmax over network logits.
* `logArgBound B e` — the recursive bound on the log arguments (at a `log` node, the argument's
  own `fwdBound`; `fwdBound_le_addExp` shows the inner logit's budget is below it), and
  `partitionLogs_logFloored` — THE DISCHARGE: input bounds `|toReal (σ i)| ≤ B` + `PartitionLogs e`
  yield `LogFloored σ (floorC (logArgBound B e)) e`, by structural induction with
  `logFloored_mono`/`floorC_antitone` reconciling the per-node floors to the single global one.
* **`adGrad_isFinite_comp_partition`** (capstone) — the final tape certificate for the PPO-shaped
  loss grammar: for a `PartitionLogs` full-grammar expression compiled by the ACTUAL compiler from
  `Tape.empty` on inputs bounded by `B`, EVERY adjoint the ACTUAL `grads` engine produces is
  overflow-free — hypotheses ONLY the input bounds, the shape, and the single budget check.
  VALUE BOUNDS IN, GRADIENTS-FINITE OUT: the `LogFloored`/`0 < c` hypotheses are GONE (discharged
  at `floorC (logArgBound B e)`, positive by `floorC_pos`).

**Scope (honestly disclosed).** The certificate covers expressions whose logs are ALL
partition-shaped — exactly the PPO loss's log usage (its only `log` is the softmax log-partition,
cf. C9/C51); other log shapes keep C75's `LogFloored` interface. The floor is C76's conservative
head-term floor (`floorC` of the whole argument's forward budget — the true argument value is at
least the head exp's floored contribution). NO new axiom: everything composes C75's capstone,
C76's floor lemmas, and C68's log-free forward magnitude.
-/
import Puffer.RL.CompTapeLog
import Puffer.RL.FloatFloor

namespace Puffer.RL.CompTapeFloor

open Puffer.FloatR (toReal u64 u64_pos expEps expEps_pos)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr evalF)
open Puffer.FloatR.ADReverse (comp)
open Puffer.RL.ADTapeFinite (LogFree fwdBound evalF_mag sweepBound)
open Puffer.RL.FiniteBound (overflowBound)
open Puffer.RL.LossForwardFinite (expF_pos)
open Puffer.RL.LogTapeFinite (LogFloored logFree_logFloored)
open Puffer.RL.CompTapeLog (compWeightBoundL adGrad_isFinite_comp_log)
open Puffer.RL.FloatFloor (floorC floorC_pos evalF_partition_floor add_floor
  one_sub_expEps_pos one_sub_u64_pos)

/-! ### The partition shapes -/

/-- **The partition tail's syntax**: an `exp` of a `LogFree` term, a NONNEGATIVE constant, or an
    `add` of two such shapes — covering both the hand-written tail `exp l₁ + (exp l₂ + …)` AND the
    repo's canonical `SoftmaxExpr.expSumE` fold, which terminates in `.const 0` (the `.const c`
    case carries the semantic side condition `0 ≤ toReal c`, discharged for the canonical `0` by
    `toReal_zero`). Syntactically guarantees a nonnegative Float value (`nonnegShape_nonneg`) and
    log-freeness (`nonnegShape_logFree`). -/
def NonnegShape : Expr → Prop
  | .exp a => LogFree a
  | .const c => 0 ≤ Puffer.FloatR.toReal c
  | .add a b => NonnegShape a ∧ NonnegShape b
  | _ => False

/-- A `NonnegShape` tail is `LogFree` (its exps' arguments are, consts trivially, adds of such). -/
theorem nonnegShape_logFree : ∀ b : Expr, NonnegShape b → LogFree b := by
  intro b
  induction b with
  | var i => exact fun h => h.elim
  | const c => exact fun _ => trivial
  | add a b iha ihb => exact fun h => ⟨iha h.1, ihb h.2⟩
  | sub a b _ _ => exact fun h => h.elim
  | mul a b _ _ => exact fun h => h.elim
  | scale c a _ => exact fun h => h.elim
  | exp a _ => exact fun h => h
  | log a _ => exact fun h => h.elim
  | relu a _ => exact fun h => h.elim
  | max a b _ _ => exact fun h => h.elim
  | min a b _ _ => exact fun h => h.elim

/-- A `NonnegShape` tail's Float value is nonnegative at every input: each `exp` term is positive
    (`expF_pos`), and one rounded add of nonnegatives is nonnegative (C76's `add_floor` at
    `x = 0`). -/
theorem nonnegShape_nonneg (σ : Nat → Float) :
    ∀ b : Expr, NonnegShape b → 0 ≤ toReal (evalF b σ) := by
  intro b
  induction b with
  | var i => exact fun h => h.elim
  | const c => exact fun h => h
  | add a b iha ihb =>
      intro h
      have := add_floor (evalF a σ) (evalF b σ) (iha h.1) (ihb h.2) 0 (iha h.1)
      simpa using this
  | sub a b _ _ => exact fun h => h.elim
  | mul a b _ _ => exact fun h => h.elim
  | scale c a _ => exact fun h => h.elim
  | exp a _ => exact fun _ => (expF_pos _).le
  | log a _ => exact fun h => h.elim
  | relu a _ => exact fun h => h.elim
  | max a b _ _ => exact fun h => h.elim
  | min a b _ _ => exact fun h => h.elim

/-- **The PPO-shaped log predicate**: structural on every op; at `log arg` it requires the
    argument to be PARTITION-SHAPED — `add (exp a) b` with a `LogFree` inner logit `a` and a
    `NonnegShape` tail `b`. This is exactly the PPO loss's log usage (its only `log` is the
    softmax log-partition over network logits); everything below a `log` is log-free, so no
    nested-log recursion arises. -/
def PartitionLogs : Expr → Prop
  | .var _ => True
  | .const _ => True
  | .add a b => PartitionLogs a ∧ PartitionLogs b
  | .sub a b => PartitionLogs a ∧ PartitionLogs b
  | .mul a b => PartitionLogs a ∧ PartitionLogs b
  | .scale _ a => PartitionLogs a
  | .exp a => PartitionLogs a
  | .log arg => ∃ a b, arg = .add (.exp a) b ∧ LogFree a ∧ NonnegShape b
  | .relu a => PartitionLogs a
  | .max a b => PartitionLogs a ∧ PartitionLogs b
  | .min a b => PartitionLogs a ∧ PartitionLogs b

/-! ### The global floor value -/

/-- **The recursive log-argument bound**: at a `log` node, the argument's own forward budget
    (`fwdBound` — the argument is log-free under `PartitionLogs`); elsewhere the max over the
    subtrees. `floorC (logArgBound B e)` is the single global floor at which every log argument
    of a `PartitionLogs` expression is floored (`partitionLogs_logFloored`). -/
noncomputable def logArgBound (B : ℝ) : Expr → ℝ
  | .var _ => 0
  | .const _ => 0
  | .add a b => max (logArgBound B a) (logArgBound B b)
  | .sub a b => max (logArgBound B a) (logArgBound B b)
  | .mul a b => max (logArgBound B a) (logArgBound B b)
  | .scale _ a => logArgBound B a
  | .exp a => logArgBound B a
  | .log arg => fwdBound B arg
  | .relu a => logArgBound B a
  | .max a b => max (logArgBound B a) (logArgBound B b)
  | .min a b => max (logArgBound B a) (logArgBound B b)

/-- `floorC` is antitone: a larger magnitude bound gives a smaller (but still positive) floor. -/
theorem floorC_antitone {M M' : ℝ} (h : M ≤ M') : floorC M' ≤ floorC M := by
  unfold Puffer.RL.FloatFloor.floorC
  have hexp : Real.exp (-M') ≤ Real.exp (-M) := Real.exp_le_exp.mpr (neg_le_neg h)
  have h1 := one_sub_expEps_pos.le
  have h2 := one_sub_u64_pos.le
  have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hexp h1) h2
  linarith

/-- `LogFloored` is monotone in the floor: a smaller floor is implied by a larger one. -/
theorem logFloored_mono (σ : Nat → Float) (c c' : ℝ) (h : c' ≤ c) :
    ∀ e, LogFloored σ c e → LogFloored σ c' e := by
  intro e
  induction e with
  | var i => exact fun _ => trivial
  | const cc => exact fun _ => trivial
  | add a b iha ihb => exact fun hf => ⟨iha hf.1, ihb hf.2⟩
  | sub a b iha ihb => exact fun hf => ⟨iha hf.1, ihb hf.2⟩
  | mul a b iha ihb => exact fun hf => ⟨iha hf.1, ihb hf.2⟩
  | scale cc a iha => exact fun hf => iha hf
  | exp a iha => exact fun hf => iha hf
  | log a iha => exact fun hf => ⟨iha hf.1, h.trans hf.2⟩
  | relu a iha => exact fun hf => iha hf
  | max a b iha ihb => exact fun hf => ⟨iha hf.1, ihb hf.2⟩
  | min a b iha ihb => exact fun hf => ⟨iha hf.1, ihb hf.2⟩

/-- The forward budget is nonnegative (inputs bounded by a nonnegative `B`). -/
theorem fwdBound_nonneg (B : ℝ) (hB : 0 ≤ B) : ∀ e, 0 ≤ fwdBound B e := by
  intro e
  induction e with
  | var i => exact hB
  | const c => exact abs_nonneg _
  | add a b iha ihb =>
      simp only [fwdBound]
      have hu : (0:ℝ) ≤ 1 + u64 := by linarith [u64_pos]
      exact mul_nonneg hu (by linarith)
  | sub a b iha ihb =>
      simp only [fwdBound]
      have hu : (0:ℝ) ≤ 1 + u64 := by linarith [u64_pos]
      exact mul_nonneg hu (by linarith)
  | mul a b iha ihb =>
      simp only [fwdBound]
      have hu : (0:ℝ) ≤ 1 + u64 := by linarith [u64_pos]
      exact mul_nonneg hu (mul_nonneg iha ihb)
  | scale c a iha =>
      simp only [fwdBound]
      have hu : (0:ℝ) ≤ 1 + u64 := by linarith [u64_pos]
      exact mul_nonneg hu (mul_nonneg (abs_nonneg _) iha)
  | exp a _ =>
      simp only [fwdBound]
      exact mul_nonneg (Real.exp_pos _).le (by linarith [expEps_pos])
  | log a _ => simp only [fwdBound]; exact le_refl 0
  | relu a iha => simpa only [fwdBound] using iha
  | max a b iha _ => simp only [fwdBound]; exact iha.trans (le_max_left _ _)
  | min a b iha _ => simp only [fwdBound]; exact iha.trans (le_max_left _ _)

/-- **The inner logit's budget sits below its partition argument's budget**:
    `fwdBound B a ≤ fwdBound B (add (exp a) b)` (via `x ≤ exp x`, the `(1+expEps)`/`(1+u64)`
    inflations `≥ 1`, and the tail budget's nonnegativity). This lets the single per-log floor
    `floorC (fwdBound B arg)` cover the inner logit's value bound. -/
theorem fwdBound_le_addExp (B : ℝ) (hB : 0 ≤ B) (a b : Expr) :
    fwdBound B a ≤ fwdBound B (.add (.exp a) b) := by
  show fwdBound B a ≤ (1 + u64) * (Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b)
  have hxe : fwdBound B a ≤ Real.exp (fwdBound B a) := by
    linarith [Real.add_one_le_exp (fwdBound B a)]
  have hb0 : 0 ≤ fwdBound B b := fwdBound_nonneg B hB b
  have hexp0 := (Real.exp_pos (fwdBound B a)).le
  have h1 : Real.exp (fwdBound B a) ≤ Real.exp (fwdBound B a) * (1 + expEps) :=
    le_mul_of_one_le_right hexp0 (by linarith [expEps_pos])
  have h0 : (0:ℝ) ≤ Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b :=
    add_nonneg (mul_nonneg hexp0 (by linarith [expEps_pos])) hb0
  calc fwdBound B a ≤ Real.exp (fwdBound B a) := hxe
    _ ≤ Real.exp (fwdBound B a) * (1 + expEps) := h1
    _ ≤ Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b := le_add_of_nonneg_right hb0
    _ ≤ (1 + u64) * (Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b) :=
        le_mul_of_one_le_left h0 (by linarith [u64_pos])

/-! ### The discharge: PartitionLogs + input bounds ⟹ LogFloored at a concrete positive floor -/

/-- **THE FLOOR DISCHARGE.** For a `PartitionLogs` expression on inputs bounded by `B`, the
    `LogFloored` hypothesis of C75's capstone HOLDS at the concrete positive floor
    `floorC (logArgBound B e)`: by structural induction, each `log`'s partition-shaped argument
    is floored via C76's `evalF_partition_floor` (the inner logit's value bounded below by C68's
    log-free `evalF_mag` through `fwdBound_le_addExp`; the tail nonnegative by
    `nonnegShape_nonneg`), and the per-node floors reconcile to the single global one by
    `floorC_antitone`/`logFloored_mono`. -/
theorem partitionLogs_logFloored (σ : Nat → Float) (B : ℝ)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) :
    ∀ e, PartitionLogs e → LogFloored σ (floorC (logArgBound B e)) e := by
  intro e
  induction e with
  | var i => exact fun _ => trivial
  | const c => exact fun _ => trivial
  | add a b iha ihb =>
      intro hp
      exact ⟨logFloored_mono σ _ _ (floorC_antitone (le_max_left _ _)) a (iha hp.1),
             logFloored_mono σ _ _ (floorC_antitone (le_max_right _ _)) b (ihb hp.2)⟩
  | sub a b iha ihb =>
      intro hp
      exact ⟨logFloored_mono σ _ _ (floorC_antitone (le_max_left _ _)) a (iha hp.1),
             logFloored_mono σ _ _ (floorC_antitone (le_max_right _ _)) b (ihb hp.2)⟩
  | mul a b iha ihb =>
      intro hp
      exact ⟨logFloored_mono σ _ _ (floorC_antitone (le_max_left _ _)) a (iha hp.1),
             logFloored_mono σ _ _ (floorC_antitone (le_max_right _ _)) b (ihb hp.2)⟩
  | scale c a iha => exact fun hp => iha hp
  | exp a iha => exact fun hp => iha hp
  | log arg _ =>
      intro hp
      obtain ⟨a, b, rfl, hlfa, hnb⟩ := hp
      have hB0 : (0:ℝ) ≤ B := (abs_nonneg _).trans (hσ 0)
      refine ⟨logFree_logFloored σ _ _ ⟨hlfa, nonnegShape_logFree b hnb⟩, ?_⟩
      -- the floor conjunct at c = floorC (fwdBound B (add (exp a) b))
      have hmag : |toReal (evalF a σ)| ≤ fwdBound B a := evalF_mag σ B hσ a hlfa
      have hgrow : fwdBound B a ≤ fwdBound B (.add (.exp a) b) := fwdBound_le_addExp B hB0 a b
      have ha : -(fwdBound B (.add (.exp a) b)) ≤ toReal (evalF a σ) := by
        have h1 := (abs_le.mp hmag).1
        linarith
      exact evalF_partition_floor σ a b _ ha (nonnegShape_nonneg σ b hnb)
  | relu a iha => exact fun hp => iha hp
  | max a b iha ihb =>
      intro hp
      exact ⟨logFloored_mono σ _ _ (floorC_antitone (le_max_left _ _)) a (iha hp.1),
             logFloored_mono σ _ _ (floorC_antitone (le_max_right _ _)) b (ihb hp.2)⟩
  | min a b iha ihb =>
      intro hp
      exact ⟨logFloored_mono σ _ _ (floorC_antitone (le_max_left _ _)) a (iha hp.1),
             logFloored_mono σ _ _ (floorC_antitone (le_max_right _ _)) b (ihb hp.2)⟩

/-! ### The capstone: value bounds in, gradients-finite out -/

/-- **CAPSTONE: the tape certificate from value bounds alone.** For a full-grammar expression
    whose logs are all PARTITION-SHAPED (`PartitionLogs e` — the PPO loss's log usage), compiled
    by the ACTUAL compiler `comp` from `Tape.empty` on inputs bounded by `B`, EVERY adjoint the
    ACTUAL imperative `grads` engine produces is overflow-free — hypotheses ONLY the input
    bounds, the shape, and the single checkable budget. C75's `LogFloored`/`0 < c` hypotheses are
    DISCHARGED at the concrete floor `floorC (logArgBound B e)` (positive by `floorC_pos`,
    floored by `partitionLogs_logFloored`). VALUE BOUNDS IN, GRADIENTS-FINITE OUT. -/
theorem adGrad_isFinite_comp_partition (σ : Nat → Float) (B : ℝ)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) (e : Expr) (hp : PartitionLogs e) (root : V)
    (hbound : sweepBound 2 (compWeightBoundL B (floorC (logArgBound B e)) e)
      (comp σ e Tape.empty).2.val.size 1 ≤ overflowBound) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite_comp_log σ B (floorC (logArgBound B e)) (floorC_pos _) hσ e
    (partitionLogs_logFloored σ B hσ e hp) root hbound

/-! ### Non-vacuity: the genuine 2-logit log-partition -/

/-- The genuine two-logit log-partition `log (exp x₀ + exp x₁)` is `PartitionLogs`. -/
example : PartitionLogs (.log (.add (.exp (.var 0)) (.exp (.var 1)))) :=
  ⟨.var 0, .exp (.var 1), rfl, trivial, trivial⟩

/-- THE CANONICAL SHAPE: the repo's own `SoftmaxExpr.expSumE`-built log-partition — whose fold
    terminates in `.const 0` — is `PartitionLogs` (the const's side condition `0 ≤ toReal 0`
    discharged by `toReal_zero`). So the capstone applies to the ACTUAL compiled softmax
    log-partition, not only hand-written const-free forms. -/
example : PartitionLogs
    (.log (.add (.exp (.var 0)) (.add (.exp (.var 1)) (.const 0)))) :=
  ⟨.var 0, .add (.exp (.var 1)) (.const 0), rfl, trivial,
    ⟨trivial, le_of_eq Puffer.FloatR.toReal_zero.symm⟩⟩

/-- The capstone instantiates on the genuine two-logit log-partition: input bounds + the budget
    check alone certify every gradient of the compiled `log (exp x₀ + exp x₁)` finite. -/
example (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B) (root : V)
    (hbound : sweepBound 2
        (compWeightBoundL B
          (floorC (logArgBound B (.log (.add (.exp (.var 0)) (.exp (.var 1))))))
          (.log (.add (.exp (.var 0)) (.exp (.var 1)))))
        (comp σ (.log (.add (.exp (.var 0)) (.exp (.var 1)))) Tape.empty).2.val.size 1
      ≤ overflowBound) :
    ∀ j, j < (comp σ (.log (.add (.exp (.var 0)) (.exp (.var 1)))) Tape.empty).2.val.size →
      ((grads (comp σ (.log (.add (.exp (.var 0)) (.exp (.var 1)))) Tape.empty).2 root)[j]!).isFinite
        = true :=
  adGrad_isFinite_comp_partition σ B hσ (.log (.add (.exp (.var 0)) (.exp (.var 1))))
    ⟨.var 0, .exp (.var 1), rfl, trivial, trivial⟩ root hbound

end Puffer.RL.CompTapeFloor
