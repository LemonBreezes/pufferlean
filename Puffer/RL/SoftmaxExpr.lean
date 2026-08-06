/-
# Softmax log-partition → `Expr` compile

The PPO policy ratio is `r = exp(logp_new − logp_old)` where `logp_new` is a log-softmax of the network's
logits: `log π(a) = logit_a − log(Σⱼ exp(logitⱼ))`. This module compiles that log-softmax — and its
log-partition (log-sum-exp) `log(Σⱼ exp(logitⱼ))` — into the AD grammar `Expr`, so `newLogp` in the C8
surrogate compiler (`ratioE`/`ppoSurrogateE`) can be an ACTUAL softmax over the logits rather than an
abstract sub-expression.

The grammar has no n-ary sum, so `Σⱼ exp(logitⱼ)` is compiled as a fold of `add` over the mapped `exp`
nodes (`expSumE`). Two things fall out:

* **Forward correctness.** `evalR (logSoftmaxE chosen logits) σ = evalR chosen σ − log(Σ exp(evalR · σ))`,
  and (given the logit list realizes a `(s, l)` softmax) `evalR (logSoftmaxE …) σ = log(Net.softmax s l i)` —
  the compiled log-softmax equals the pre-existing ℝ softmax spec.
* **The C6 `log`-floor discharges automatically.** A sum of exponentials is strictly positive
  (`expSumE_pos`), and on a bounded region `|σ i| ≤ R` with a `Smooth` head logit it is bounded below by the
  concrete positive constant `exp(−vMag R e)` (`expSumE_floor`). This is exactly the semantic positive floor
  `c > 0` that C6's `derivR_log_lip` needs for the log-partition node — so for softmax it is FREE (a partition
  can never approach 0), removing the one hypothesis that made `log` a composable-only node in C6.

**Scope (honestly disclosed):** this compiles the log-softmax VALUE (and its exact ℝ-spec match + the
partition's positive floor); the logits themselves are supplied as arbitrary `Expr`s (a linear-layer→`Expr`
builder for the network's pre-softmax outputs is a separate front-end piece). The gradient-Lipschitz of the
whole log-softmax node combines C5 (`exp` in `expSumE`, all `Smooth`) with C6 (`log` of the partition, floor
now discharged) — assembled downstream.
-/
import Puffer.Float.AutoDiffR
import Puffer.Net.Forward

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)

namespace Puffer.RL.SoftmaxExpr

/-- **Σ exp(logit) as an `Expr`.** The softmax partition, compiled as a fold of `add` over the `exp` of each
    logit expression (the grammar has no n-ary sum). Empty list ↦ `const 0`. -/
def expSumE : List Expr → Expr
  | [] => .const 0
  | e :: es => .add (.exp e) (expSumE es)

/-- `evalR (expSumE logits) σ = Σ exp(evalR eⱼ σ)` (list sum). -/
theorem evalR_expSumE (logits : List Expr) (σ : Nat → ℝ) :
    evalR (expSumE logits) σ = (logits.map (fun e => Real.exp (evalR e σ))).sum := by
  induction logits with
  | nil => simp [expSumE, evalR]
  | cons e es ih => simp only [expSumE, evalR, List.map_cons, List.sum_cons, ih]

/-- The partition is nonnegative (a sum of `exp`s ≥ 0, empty ↦ 0). -/
theorem expSumE_nonneg (logits : List Expr) (σ : Nat → ℝ) :
    0 ≤ evalR (expSumE logits) σ := by
  induction logits with
  | nil => simp [expSumE, evalR]
  | cons e es ih => simp only [expSumE, evalR]; exact add_nonneg (Real.exp_pos _).le ih

/-- The partition dominates any single `exp` term (here the head) — the seed of the positive floor. -/
theorem expSumE_ge_exp_head (e : Expr) (es : List Expr) (σ : Nat → ℝ) :
    Real.exp (evalR e σ) ≤ evalR (expSumE (e :: es)) σ := by
  simp only [expSumE, evalR]
  exact le_add_of_nonneg_right (expSumE_nonneg es σ)

/-- **The partition is strictly positive** (nonempty logits): `0 < Σ exp`. A sum of exponentials can never
    reach `0`, which is why softmax's `log`-partition discharges C6's positive-floor hypothesis for free. -/
theorem expSumE_pos (e : Expr) (es : List Expr) (σ : Nat → ℝ) :
    0 < evalR (expSumE (e :: es)) σ :=
  lt_of_lt_of_le (Real.exp_pos _) (expSumE_ge_exp_head e es σ)

/-- **Log-partition (log-sum-exp) as an `Expr`.** `log(Σⱼ exp(logitⱼ))` — the softmax normalizer / cumulant. -/
def logPartitionE (logits : List Expr) : Expr := .log (expSumE logits)

theorem evalR_logPartitionE (logits : List Expr) (σ : Nat → ℝ) :
    evalR (logPartitionE logits) σ = Real.log ((logits.map (fun e => Real.exp (evalR e σ))).sum) := by
  simp only [logPartitionE, evalR, evalR_expSumE]

/-- **Log-softmax as an `Expr`.** `log π(a) = logit_a − log(Σⱼ exp(logitⱼ))`, the new-policy log-prob that
    feeds the PPO ratio `ratioE`. -/
def logSoftmaxE (chosen : Expr) (logits : List Expr) : Expr := .sub chosen (logPartitionE logits)

/-- `evalR (logSoftmaxE chosen logits) σ = evalR chosen σ − log(Σ exp(evalR · σ))` — the log-softmax formula. -/
theorem evalR_logSoftmaxE (chosen : Expr) (logits : List Expr) (σ : Nat → ℝ) :
    evalR (logSoftmaxE chosen logits) σ
      = evalR chosen σ - Real.log ((logits.map (fun e => Real.exp (evalR e σ))).sum) := by
  simp only [logSoftmaxE, evalR, evalR_logPartitionE]

/-- **Match to the ℝ softmax spec.** Given the logit list realizes a softmax `(s, l)` at `σ` (its chosen
    logit evaluates to `l i` and its `Σ exp` matches `softmaxDenom s l`), the compiled log-softmax evaluates
    to exactly the real-valued `log(Net.softmax s l i)`. Uses `Real.log_div` + `Real.log_exp` and the
    established `softmaxDenom_pos` (nonempty ⇒ denominator ≠ 0). -/
theorem evalR_logSoftmaxE_eq_logSoftmax {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i : ι)
    (hs : s.Nonempty) (chosen : Expr) (logits : List Expr) (σ : Nat → ℝ)
    (hchosen : evalR chosen σ = l i)
    (hsum : (logits.map (fun e => Real.exp (evalR e σ))).sum = Puffer.Net.softmaxDenom s l) :
    evalR (logSoftmaxE chosen logits) σ = Real.log (Puffer.Net.softmax s l i) := by
  rw [evalR_logSoftmaxE, hchosen, hsum, Puffer.Net.softmax,
      Real.log_div (Real.exp_ne_zero _) (Puffer.Net.softmaxDenom_pos s l hs).ne', Real.log_exp]

/-- Clean `Fin n` instantiation: logits given as a vector `logitE`, softmax over `Finset.univ`. The
    list-sum↔`softmaxDenom` match (`hsum`) is discharged here via `List.map_ofFn`/`List.sum_ofFn`. -/
theorem evalR_logSoftmaxE_ofFn {n : ℕ} [NeZero n] (logitE : Fin n → Expr) (i : Fin n) (σ : Nat → ℝ) :
    evalR (logSoftmaxE (logitE i) (List.ofFn logitE)) σ
      = Real.log (Puffer.Net.softmax Finset.univ (fun j => evalR (logitE j) σ) i) := by
  apply evalR_logSoftmaxE_eq_logSoftmax
  · exact Finset.univ_nonempty
  · rfl
  · rw [Puffer.Net.softmaxDenom, List.map_ofFn, List.sum_ofFn]; rfl

/-- **C6 floor discharge for the softmax partition.** For a `Smooth` head logit `e`, on the region
    `|σ i| ≤ R` the partition is bounded below by the CONCRETE positive constant `exp(−vMag R e) > 0`:
    `exp(−vMag R e) ≤ evalR (expSumE (e :: es)) σ`. This supplies the positive floor `c` that C6's
    `derivR_log_lip` requires of the `log`-partition node — automatically, from the region bound on the
    logits (via `evalR_mag`) and the fact that the partition dominates a single `exp` term. So for softmax
    the `log`-node's semantic floor hypothesis is never a real obstruction. -/
theorem expSumE_floor (e : Expr) (es : List Expr) (he : Smooth e) (σ : Nat → ℝ)
    (R : ℝ) (hσ : ∀ i, |σ i| ≤ R) :
    Real.exp (-(vMag R e)) ≤ evalR (expSumE (e :: es)) σ := by
  refine le_trans ?_ (expSumE_ge_exp_head e es σ)
  apply Real.exp_le_exp.mpr
  have := evalR_mag he σ R hσ
  linarith [abs_le.mp this]

end Puffer.RL.SoftmaxExpr
