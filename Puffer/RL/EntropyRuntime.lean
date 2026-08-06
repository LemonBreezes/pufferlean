/-
The DISCRETE-policy entropy bonus `H = −Σᵢ pᵢ·log pᵢ` — the runnable `Float` reduction within a certified
bound of its ℝ meaning. The discrete-action counterpart to the Gaussian entropy (`GaussianRuntime`), and the
central case (PufferLib's categorical policies) for the PPO/A2C entropy-bonus term.

Unlike the scalar op-chains, this is a REDUCTION: the running sum `Σ pᵢ·log pᵢ` (via `SoftmaxBound.sumIdxFGo`)
of the per-term products, each rounding at `log` (`log_error`, needs `pᵢ > 0`) and the multiply
(`mulApprox_error`), accumulated by `sumIdxFGo_error`, then negated (`toReal_neg`).

  • `entropyF` — the runnable `−(Σᵢ pᵢ · Float.log pᵢ)` over a probability function `p : Nat → Float`.
  • `entropyProbR` — the ℝ entropy `−(Σᵢ pᵢ·log pᵢ)` of the real probabilities.
  • `entTermBnd` — the per-term budget `u64·|pᵢ·log pᵢ| + |pᵢ|·(logEps·|log pᵢ|)` (the `log` rounding scaled
    by `pᵢ` + the product rounding).
  • `entropyF_error` — the running entropy is within the accumulated `sumIdxErrBnd` over the per-term
    budgets, given every probability is positive (the `log` floor).
  • `entropyProbR_softmax` / `entropyF_softmax_error` — the SPEC BRIDGE: `entropyProbR` over the softmax
    probabilities IS `EntropyPerturb.entropy` (a80's categorical Shannon entropy, `rfl`), so the running
    entropy over probs whose `toReal` is the ideal softmax lands on the categorical-entropy spec (the log
    floor discharged from `softmax_pos`).

The probabilities are treated as exact `Float` inputs (their `toReal` is the reference), so this bounds the
entropy REDUCTION's own rounding; composing the softmax's rounding (`SoftmaxExec.train_softmax_error`, a67)
onto `p` is a separate step (as `log2π` was parameterized in `GaussianRuntime`).

Axiom-clean beyond the trusted Float base (`add/mul/log_model` + `toReal` + `toReal_neg`/`toReal_zeroLit`).
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.RL.SoftmaxBound
import Puffer.RL.EntropyPerturb

namespace Puffer.RL.EntropyRuntime

open Puffer.FloatR
open Puffer.RL.SoftmaxBound (sumIdxFGo sumIdxErrBnd sumIdxFGo_error sumIdxRGo_eq_sum)
open Finset
open Puffer.Net (softmax softmax_pos)
open Puffer.RL.EntropyPerturb (entropy entropy_le_log_card)

/-- Executable discrete-policy entropy `H = −Σᵢ pᵢ·log pᵢ` over a probability function `p`. -/
def entropyF (p : Nat → Float) (n : Nat) : Float :=
  -(sumIdxFGo (fun i => p i * Float.log (p i)) 0.0 (List.range n))

/-- ℝ discrete entropy of the (real) probabilities. -/
noncomputable def entropyProbR (pR : Nat → ℝ) (n : Nat) : ℝ :=
  -(∑ i ∈ Finset.range n, pR i * Real.log (pR i))

/-- Per-term error budget for `pᵢ·log pᵢ` (the `log` rounding scaled by `pᵢ`, plus the product rounding). -/
noncomputable def entTermBnd (p : Nat → Float) (i : Nat) : ℝ :=
  u64 * |toReal (p i) * toReal (Float.log (p i))|
    + |toReal (p i)| * (logEps * |Real.log (toReal (p i))|)

/-- **Discrete-entropy runtime error.** The running `entropyF` deviates from the ℝ entropy of the
    probabilities by at most the accumulated `sumIdxErrBnd` over the per-term budgets, given every
    probability is positive (`log` floor). -/
theorem entropyF_error (p : Nat → Float) (n : Nat) (hp : ∀ i, i < n → 0 < toReal (p i)) :
    |toReal (entropyF p n) - entropyProbR (fun i => toReal (p i)) n|
      ≤ sumIdxErrBnd (fun i => p i * Float.log (p i)) (entTermBnd p) 0.0 (List.range n) 0 := by
  have h0acc : |toReal (0.0 : Float) - (0 : ℝ)| ≤ 0 := by rw [toReal_zeroLit]; simp
  have hterm : ∀ k ∈ List.range n,
      |toReal (p k * Float.log (p k)) - toReal (p k) * Real.log (toReal (p k))| ≤ entTermBnd p k := by
    intro k hk
    have hkn : k < n := List.mem_range.mp hk
    have hlog : |toReal (Float.log (p k)) - Real.log (toReal (p k))|
        ≤ logEps * |Real.log (toReal (p k))| := log_error (p k) (hp k hkn)
    have := mulApprox_error (p k) (Float.log (p k)) (toReal (p k)) (Real.log (toReal (p k)))
      0 (logEps * |Real.log (toReal (p k))|) (by simp) hlog
    simpa [entTermBnd] using this
  have hsum := sumIdxFGo_error (fun i => p i * Float.log (p i))
    (fun i => toReal (p i) * Real.log (toReal (p i))) (entTermBnd p)
    (List.range n) 0.0 0 0 h0acc hterm
  rw [sumIdxRGo_eq_sum] at hsum
  rw [entropyF, entropyProbR, toReal_neg]
  calc |(-toReal (sumIdxFGo (fun i => p i * Float.log (p i)) 0.0 (List.range n)))
          - (-(∑ i ∈ Finset.range n, toReal (p i) * Real.log (toReal (p i))))|
      = |toReal (sumIdxFGo (fun i => p i * Float.log (p i)) 0.0 (List.range n))
          - (∑ i ∈ Finset.range n, toReal (p i) * Real.log (toReal (p i)))| := by
        rw [← abs_neg]; ring_nf
    _ ≤ _ := hsum

/-! ### Bridge to the softmax-entropy spec (a80) -/

/-- **Spec bridge.** `entropyProbR` over the softmax probabilities IS the categorical (softmax) entropy
    `EntropyPerturb.entropy` — the two entropy formulations coincide. -/
theorem entropyProbR_softmax (n : Nat) (l : Nat → ℝ) :
    entropyProbR (fun i => softmax (Finset.range n) l i) n = entropy (Finset.range n) l := rfl

/-- **Running categorical entropy on the softmax spec.** Given a `Float` prob function whose `toReal` is the
    ideal softmax (the softmax's OWN rounding scoped out), the running `entropyF` is within `entropyF_error`'s
    budget of the categorical entropy `EntropyPerturb.entropy`. -/
theorem entropyF_softmax_error (p : Nat → Float) (n : Nat) (l : Nat → ℝ)
    (hpsm : ∀ i, i < n → toReal (p i) = softmax (Finset.range n) l i) (hne : (Finset.range n).Nonempty) :
    |toReal (entropyF p n) - entropy (Finset.range n) l|
      ≤ Puffer.RL.SoftmaxBound.sumIdxErrBnd (fun i => p i * Float.log (p i)) (entTermBnd p) 0.0
          (List.range n) 0 := by
  have hp : ∀ i, i < n → 0 < toReal (p i) := by
    intro i hi; rw [hpsm i hi]; exact softmax_pos (Finset.range n) l i hne
  have hbridge : entropyProbR (fun i => toReal (p i)) n = entropy (Finset.range n) l := by
    rw [← entropyProbR_softmax n l]
    unfold entropyProbR
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    simp only [hpsm i (Finset.mem_range.mp hi)]
  have h := entropyF_error p n hp
  rwa [hbridge] at h

/-- **Runtime entropy bonus is bounded by `log(#actions)`.** The running `entropyF` over a policy whose `toReal`
    is the softmax of `l` is at most `log n + sumIdxErrBnd` — the maximum-entropy bound `entropy ≤ log|s|` (a160)
    transported to the executable, up to the reduction's rounding budget. So the trainer's entropy term can never
    exceed `log(#actions)` beyond arithmetic error. -/
theorem entropyF_le_log_card (p : Nat → Float) (n : Nat) (l : Nat → ℝ)
    (hpsm : ∀ i, i < n → toReal (p i) = softmax (Finset.range n) l i)
    (hne : (Finset.range n).Nonempty) :
    toReal (entropyF p n)
      ≤ Real.log n + sumIdxErrBnd (fun i => p i * Float.log (p i)) (entTermBnd p) 0.0
          (List.range n) 0 := by
  have herr := entropyF_softmax_error p n l hpsm hne
  have hmax := entropy_le_log_card (Finset.range n) l hne
  rw [Finset.card_range] at hmax
  have hb := (abs_le.mp herr).2
  linarith

/-- **Runtime entropy bonus is nonnegative up to arithmetic error.** For any policy whose `toReal`
    probabilities are genuine (each `pᵢ ∈ (0,1]`), the executed entropy `entropyF p n` can dip at most
    the reduction's own `sumIdxErrBnd` budget below `0`: `−budget ≤ toReal (entropyF p n)`. The ℝ
    entropy of a sub/probability vector is `≥ 0` (each `pᵢ·log pᵢ ≤ 0` by `Real.log_nonpos`), and
    `entropyF_error` transports that floor to the `Float` reduction. This is the missing LOWER half
    bracketing the runtime entropy term — the upper half being `entropyF_le_log_card` — so together the
    executed entropy bonus lives in `[−budget, log n + budget]` and the trainer's `entCoef·H` term is
    essentially nonnegative. Both hypotheses are load-bearing: without `0 < toReal (p i)` the `log`
    floor of `entropyF_error` fails; without `toReal (p i) ≤ 1` a probability `> 1` makes the ℝ entropy
    negative (a single `p₀ = e` gives `entropy = −e`, far below the tiny budget). -/
theorem neg_budget_le_entropyF (p : Nat → Float) (n : Nat)
    (hpos : ∀ i, i < n → 0 < toReal (p i))
    (hle : ∀ i, i < n → toReal (p i) ≤ 1) :
    -(sumIdxErrBnd (fun i => p i * Float.log (p i)) (entTermBnd p) 0.0 (List.range n) 0)
      ≤ toReal (entropyF p n) := by
  have hnonneg : 0 ≤ entropyProbR (fun i => toReal (p i)) n := by
    unfold entropyProbR
    rw [neg_nonneg]
    apply Finset.sum_nonpos
    intro i hi
    have hik := Finset.mem_range.mp hi
    exact mul_nonpos_of_nonneg_of_nonpos (hpos i hik).le
      (Real.log_nonpos (hpos i hik).le (hle i hik))
  have herr := entropyF_error p n hpos
  have hb := (abs_le.mp herr).1
  linarith

end Puffer.RL.EntropyRuntime
