/-
The running policy's LOG-probabilities within a proven bound of the ideal — the objective-facing companion
to `PolicyBound` (which bounds the raw action probabilities).

The REINFORCE objective the trainer optimizes is `adv · log π(a|s)`, with the full-softmax policy
`π(·|s) = softmax(logits)` (`NNTrain.policyProbs`); so the quantity flowing into the policy gradient is the
LOG-policy `log π(a|s)`, and its accuracy matters. (The PPO actor-critic path is DIFFERENT — `policyAndValue`
splits the last logit off as the value head and softmaxes only the first `A`, so its stored `oldLogp` is a
shorter softmax; this file bounds the full-softmax `policyProbs` log-policy, not PPO's `oldLogp`.) Log-softmax
perturbs even more gently than softmax:

  • `log_softmax_eq` — the exact identity `log (softmax s l i) = lᵢ − log (Σⱼ exp lⱼ)` (log-softmax).
  • `logDenom_perturb` — `|log (Σ exp a) − log (Σ exp b)| ≤ ε` (the log-sum-exp moves within `±ε`, since the
    sum itself scales within `[e^{−ε}, e^{ε}]`; `denom_ub`/`denom_lb` + `log` monotone).
  • `logSoftmax_perturb` — the clean **2-Lipschitz** log-policy bound: `|log softmax(a)ᵢ − log softmax(b)ᵢ|
    ≤ 2ε` (numerator `±ε` + log-denominator `±ε`). NO floors, NO derivatives.
  • `softmax_tv_perturb` — the whole-distribution **total-variation** bound: `Σᵢ |softmax(a)ᵢ − softmax(b)ᵢ|
    ≤ e^{2ε} − 1` (the tight per-component bound summed, using `Σ softmax = 1`).
  • `abs_log_sub_le` — `log`'s Lipschitzness with a floor: `|log u − log v| ≤ |u − v| / d` for `u,v ≥ d > 0`.
  • `logPolicy_error` — the capstone on the actual runnable policy: `Float.log ((policyProbs p size s)[a]!)`
    (the running full-softmax log-policy, the REINFORCE objective input) within
    `logEps·|log prob| + (Bsm + (e^{2ε} − 1))/d` of the ideal log-policy `log π_ideal(a|s)`, composing
    `log_error` (the `Float.log` rounding), the a72 probability bound, and `abs_log_sub_le` (with a
    probability floor `d`).

The ℝ perturbation lemmas are pure Mathlib (axiom-clean, no Float axioms); the capstone inherits the trusted
Float base (`log_model` via `log_error`, plus `PolicyBound`'s footprint).
-/
import Puffer.RL.PolicyBound

namespace Puffer.RL.LogPolicyBound

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.PolicyBound

/-! ### Log-softmax and log-sum-exp perturbation over ℝ -/

variable {ι : Type*}

/-- **Log-softmax identity.** `log (softmax s l i) = lᵢ − log (Σⱼ exp lⱼ)`. -/
theorem log_softmax_eq (s : Finset ι) (l : ι → ℝ) (i : ι) (hs : s.Nonempty) :
    Real.log (softmax s l i) = l i - Real.log (softmaxDenom s l) := by
  rw [softmax, Real.log_div (Real.exp_ne_zero _) (ne_of_gt (softmaxDenom_pos s l hs)), Real.log_exp]

/-- **Log-sum-exp perturbation.** `|log (Σ exp a) − log (Σ exp b)| ≤ ε` under an `ε`-perturbation. -/
theorem logDenom_perturb (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hs : s.Nonempty)
    (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b)| ≤ ε := by
  have hSa := softmaxDenom_pos s a hs
  have hSb := softmaxDenom_pos s b hs
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · have h := Real.log_le_log (by positivity) (denom_lb s a b ε hab)
    rw [Real.log_mul (Real.exp_ne_zero _) hSb.ne', Real.log_exp] at h
    linarith
  · have h := Real.log_le_log hSa (denom_ub s a b ε hab)
    rw [Real.log_mul (Real.exp_ne_zero _) hSb.ne', Real.log_exp] at h
    linarith

/-- **Log-policy is 2-Lipschitz.** `|log softmax(a)ᵢ − log softmax(b)ᵢ| ≤ 2ε` under an `ε`-perturbation of
    the logits — no floors, no derivatives. The clean log-policy sensitivity used by REINFORCE/PPO. -/
theorem logSoftmax_perturb (s : Finset ι) (a b : ι → ℝ) (i : ι) (ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |Real.log (softmax s a i) - Real.log (softmax s b i)| ≤ 2*ε := by
  rw [log_softmax_eq s a i hs, log_softmax_eq s b i hs]
  have h1 : |a i - b i| ≤ ε := hab i hi
  have hd := logDenom_perturb s a b ε hs hab
  have hkey : |a i - Real.log (softmaxDenom s a) - (b i - Real.log (softmaxDenom s b))|
      ≤ |a i - b i| + |Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b)| := by
    rw [show a i - Real.log (softmaxDenom s a) - (b i - Real.log (softmaxDenom s b))
        = (a i - b i) - (Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b)) by ring]
    exact abs_sub _ _
  linarith

/-- **Whole-policy total-variation bound.** The `ℓ₁` distance between the two action distributions is
    `≤ e^{2ε} − 1` — the tight per-component bound summed, using `Σᵢ softmax = 1`. -/
theorem softmax_tv_perturb (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hs : s.Nonempty)
    (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    ∑ i ∈ s, |softmax s a i - softmax s b i| ≤ Real.exp (2*ε) - 1 := by
  calc ∑ i ∈ s, |softmax s a i - softmax s b i|
      ≤ ∑ i ∈ s, softmax s b i * (Real.exp (2*ε) - 1) :=
        Finset.sum_le_sum (fun i hi => softmax_input_perturb s a b i ε hs hi hab)
    _ = (∑ i ∈ s, softmax s b i) * (Real.exp (2*ε) - 1) := by rw [← Finset.sum_mul]
    _ = Real.exp (2*ε) - 1 := by rw [softmax_sum_one s b hs, one_mul]

/-- **`log` is Lipschitz above a floor.** `|log u − log v| ≤ |u − v| / d` for `u, v ≥ d > 0`
    (`log x ≤ x − 1` on each side). -/
theorem abs_log_sub_le (u v d : ℝ) (hd : 0 < d) (hu : d ≤ u) (hv : d ≤ v) :
    |Real.log u - Real.log v| ≤ |u - v| / d := by
  have hu0 : 0 < u := lt_of_lt_of_le hd hu
  have hv0 : 0 < v := lt_of_lt_of_le hd hv
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · have h := Real.log_le_sub_one_of_pos (div_pos hv0 hu0)
    rw [Real.log_div hv0.ne' hu0.ne', show v/u - 1 = (v-u)/u by field_simp] at h
    have h2 : (v - u)/u ≤ |u - v| / d := by
      calc (v - u)/u ≤ |v - u|/u := by gcongr; exact le_abs_self _
        _ ≤ |v - u|/d := by gcongr
        _ = |u - v|/d := by rw [abs_sub_comm]
    linarith [h.trans h2]
  · have h := Real.log_le_sub_one_of_pos (div_pos hu0 hv0)
    rw [Real.log_div hu0.ne' hv0.ne', show u/v - 1 = (u-v)/v by field_simp] at h
    have h2 : (u - v)/v ≤ |u - v| / d := by
      calc (u - v)/v ≤ |u - v|/v := by gcongr; exact le_abs_self _
        _ ≤ |u - v|/d := by gcongr
    linarith [h.trans h2]

/-! ### Capstone: the running log-policy bound -/

open Puffer.RL.NNTrain
open Puffer.RL.ForwardExec (hRList)

/-- **Running log-policy bound (capstone).** The running full-softmax log-policy
    `Float.log ((policyProbs p size s)[a]!)` (`policyProbs`'s log-probability, the REINFORCE objective input —
    NOT PPO's actor-critic `oldLogp`, which is a shorter softmax) is within
    `logEps·|log prob| + (Bsm + (e^{2ε} − 1))/d` of the ideal log-policy `log π_ideal(a|s) =
    Real.log (Net.softmax idealLogitR a)`. Composes the `Float.log` rounding (`log_error`), the a72
    probability bound `Bsm + (e^{2ε} − 1)`, and `log`'s Lipschitzness (`abs_log_sub_le`) with a probability
    floor `d ≤ π(a|s)` (both running and ideal ≥ `d > 0`). -/
theorem logPolicy_error (p : MLP) (size s a : Nat) (ε Bsm d : ℝ)
    (ha : a < p.b2.size) (hpos : 0 < p.b2.size)
    (hsm : |toReal ((policyProbs p size s)[a]!)
             - softmax (Finset.range p.b2.size)
                 (fun j => toReal (forwardAll p (oneHot size s)).2.2[j]!) a| ≤ Bsm)
    (hε : 0 ≤ ε)
    (hεbnd : ∀ k, k < p.b2.size →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p (oneHot size s)).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p (oneHot size s)).2.1).toList (hRList p (oneHot size s))
        ≤ ε)
    (hd : 0 < d)
    (hprun : d ≤ toReal ((policyProbs p size s)[a]!))
    (hpideal : d ≤ softmax (Finset.range p.b2.size) (idealLogitR p size s) a) :
    |toReal (Float.log ((policyProbs p size s)[a]!))
        - Real.log (softmax (Finset.range p.b2.size) (idealLogitR p size s) a)|
      ≤ logEps * |Real.log (toReal ((policyProbs p size s)[a]!))|
        + (Bsm + (Real.exp (2*ε) - 1)) / d := by
  have hP0 : 0 < toReal ((policyProbs p size s)[a]!) := lt_of_lt_of_le hd hprun
  have hle := log_error ((policyProbs p size s)[a]!) hP0
  have h72 := policyProbs_error p size s a ε Bsm ha hpos hsm hε hεbnd
  have hlip := abs_log_sub_le (toReal ((policyProbs p size s)[a]!))
    (softmax (Finset.range p.b2.size) (idealLogitR p size s) a) d hd hprun hpideal
  have hdiv : |toReal ((policyProbs p size s)[a]!)
        - softmax (Finset.range p.b2.size) (idealLogitR p size s) a| / d
      ≤ (Bsm + (Real.exp (2*ε) - 1)) / d := by gcongr
  calc |toReal (Float.log ((policyProbs p size s)[a]!))
          - Real.log (softmax (Finset.range p.b2.size) (idealLogitR p size s) a)|
      ≤ |toReal (Float.log ((policyProbs p size s)[a]!))
            - Real.log (toReal ((policyProbs p size s)[a]!))|
        + |Real.log (toReal ((policyProbs p size s)[a]!))
            - Real.log (softmax (Finset.range p.b2.size) (idealLogitR p size s) a)| := abs_sub_le _ _ _
    _ ≤ logEps * |Real.log (toReal ((policyProbs p size s)[a]!))|
        + (Bsm + (Real.exp (2*ε) - 1)) / d := by
        have := hlip.trans hdiv; linarith

end Puffer.RL.LogPolicyBound
