/-
The running policy's action probabilities within a proven bound of the ideal ℝ softmax — the softmax-of-
logits composition that caps the forward-pass trifecta.

`ForwardExec` bounds the runnable `forwardAll` logits; `SoftmaxExec` bounds the runnable `Train.softmax`
of a FIXED Float logits array against `Net.softmax` of `toReal` of THOSE logits. What was missing is how
the softmax MOVES when its inputs (the logits) are themselves perturbed from their ideal real values — a
softmax input-sensitivity (Lipschitz) fact. This file supplies it and composes:

  • `softmax_input_perturb_le` — the ℝ core: if `|aₖ − bₖ| ≤ ε` on all of `s`, then
    `|softmax s a i − softmax s b i| ≤ e^{2ε} − 1`. Elementary (NO derivatives): the softmax ratio stays in
    `[e^{−2ε}, e^{2ε}]` because the denominator `Σ exp` scales within `[e^{−ε}, e^{ε}]` and the numerator
    within `[e^{−ε}, e^{ε}]`; the two-sided `e^{±2ε}·softmax` sandwich then gives the bound (`softmax ≤ 1`,
    `exp(2ε)+exp(−2ε) ≥ 2` via `add_one_le_exp`).
  • `softmax_logits_error` — the triangle composition: `Train.softmax`'s Float rounding (`Bsm`, from
    `SoftmaxExec.train_softmax_error`) PLUS the input perturbation `e^{2ε}−1` bound `(Train.softmax logits)[i]!`
    against `Net.softmax` of ARBITRARY ideal real logits `LR` within `ε` of `toReal ∘ logits`.
  • `policyProbs_error` — the capstone on the actual runnable policy: `(NNTrain.policyProbs p size s)[i]!`
    (the action probability the trainer samples from) within `Bsm + (e^{2ε} − 1)` of `Net.softmax` of the
    ideal real logits `idealLogitR` (`= toReal b2ₖ + dotRm W2ₖ hR`), given a uniform logit-error bound `ε`
    (from `ForwardExec.forwardAll_logit_error`) and the softmax denominator floor.

Axiom-clean beyond the trusted Float base — the perturbation lemmas are pure ℝ (Mathlib), the composition
inherits `SoftmaxExec`/`ForwardExec`'s footprint.
-/
import Puffer.RL.SoftmaxExec
import Puffer.RL.ForwardExec

namespace Puffer.RL.PolicyBound

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.NNTrain
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-! ### The softmax input-perturbation (Lipschitz) bound over ℝ -/

variable {ι : Type*}

/-- The softmax denominator `Σ exp` grows by at most `exp ε` under an `ε`-perturbation of the logits. -/
theorem denom_ub (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    softmaxDenom s a ≤ Real.exp ε * softmaxDenom s b := by
  rw [softmaxDenom, softmaxDenom, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro j hj
  rw [← Real.exp_add]
  exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hab j hj)).2])

/-- The softmax denominator shrinks by at most `exp (−ε)` under an `ε`-perturbation. -/
theorem denom_lb (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    Real.exp (-ε) * softmaxDenom s b ≤ softmaxDenom s a := by
  rw [softmaxDenom, softmaxDenom, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro j hj
  rw [← Real.exp_add]
  exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hab j hj)).1])

/-- Upper sandwich: `softmax s a i ≤ e^{2ε}·softmax s b i`. -/
theorem softmax_ub (s : Finset ι) (a b : ι → ℝ) (i : ι) (ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    softmax s a i ≤ Real.exp (2*ε) * softmax s b i := by
  have hSbpos : 0 < softmaxDenom s b := softmaxDenom_pos s b hs
  have hdenpos : 0 < Real.exp (-ε) * softmaxDenom s b := mul_pos (Real.exp_pos _) hSbpos
  have hnum : Real.exp (a i) ≤ Real.exp ε * Real.exp (b i) := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hab i hi)).2])
  have hstep : softmax s a i ≤ (Real.exp ε * Real.exp (b i)) / (Real.exp (-ε) * softmaxDenom s b) := by
    rw [softmax]; gcongr; exact denom_lb s a b ε hab
  refine hstep.trans (le_of_eq ?_)
  rw [softmax, show (2:ℝ)*ε = ε + ε by ring, Real.exp_add, Real.exp_neg]
  field_simp

/-- Lower sandwich: `e^{−2ε}·softmax s b i ≤ softmax s a i`. -/
theorem softmax_lb (s : Finset ι) (a b : ι → ℝ) (i : ι) (ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    Real.exp (-(2*ε)) * softmax s b i ≤ softmax s a i := by
  have hSapos : 0 < softmaxDenom s a := softmaxDenom_pos s a hs
  have hSbpos : 0 < softmaxDenom s b := softmaxDenom_pos s b hs
  have hnum : Real.exp (-ε) * Real.exp (b i) ≤ Real.exp (a i) := by
    rw [← Real.exp_add]; exact Real.exp_le_exp.mpr (by linarith [(abs_le.mp (hab i hi)).1])
  have heq : Real.exp (-(2*ε)) * softmax s b i
      = (Real.exp (-ε) * Real.exp (b i)) / (Real.exp ε * softmaxDenom s b) := by
    rw [softmax, show -(2*ε) = -ε + -ε by ring, Real.exp_add, Real.exp_neg]
    field_simp
  rw [heq, softmax]
  gcongr
  exact denom_ub s a b ε hab

/-- **Softmax input-perturbation (tight form).** `|softmax s a i − softmax s b i| ≤ softmax s b i·(e^{2ε}−1)`
    when `|aₖ − bₖ| ≤ ε` on `s`. -/
theorem softmax_input_perturb (s : Finset ι) (a b : ι → ℝ) (i : ι) (ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |softmax s a i - softmax s b i| ≤ softmax s b i * (Real.exp (2*ε) - 1) := by
  have hub := softmax_ub s a b i ε hs hi hab
  have hlb := softmax_lb s a b i ε hs hi hab
  have hsb : 0 < softmax s b i := softmax_pos s b i hs
  have hAM : (2:ℝ) ≤ Real.exp (2*ε) + Real.exp (-(2*ε)) := by
    have h1 := Real.add_one_le_exp (2*ε); have h2 := Real.add_one_le_exp (-(2*ε)); linarith
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · nlinarith [hlb, hsb, hAM,
      mul_nonneg hsb.le (by linarith : (0:ℝ) ≤ Real.exp (2*ε) + Real.exp (-(2*ε)) - 2)]
  · nlinarith [hub, hsb]

/-- Each softmax output is `≤ 1` (a probability). -/
theorem softmax_le_one (s : Finset ι) (l : ι → ℝ) (i : ι) (hs : s.Nonempty) (hi : i ∈ s) :
    softmax s l i ≤ 1 := by
  rw [← softmax_sum_one s l hs]
  exact Finset.single_le_sum (fun j _ => (softmax_pos s l j hs).le) hi

/-- **Softmax is strictly `< 1` when there is another action.** If some `j ≠ i` also lies in `s`, then
    `softmax s l i < 1` — a softmax probability is strictly sub-one whenever the support has ≥ 2 elements (the
    other action carries positive mass, so `πᵢ < Σ π = 1`). The softmax never concentrates fully on one action. -/
theorem softmax_lt_one (s : Finset ι) (l : ι → ℝ) (i j : ι) (hs : s.Nonempty)
    (hi : i ∈ s) (hj : j ∈ s) (hij : j ≠ i) : softmax s l i < 1 := by
  have h := Finset.single_lt_sum hij hi hj (softmax_pos s l j hs)
    (fun k _ _ => (softmax_pos s l k hs).le)
  rwa [softmax_sum_one s l hs] at h

/-- **Softmax probability floor from a logit ceiling.** If every logit on `s` is `≤ c`, then the softmax
    probability of any action `i` is bounded below by `e^{lᵢ − c} / |s|`:
    `Real.exp (l i − c) / s.card ≤ softmax s l i`. The denominator `Σⱼ e^{lⱼ}` can be no larger than
    `|s|·e^{c}` (sum of `|s|` terms each `≤ e^c` by `exp` monotonicity), so the ratio `e^{lᵢ}/Σ` is at least
    `e^{lᵢ}/(|s|·e^{c}) = e^{lᵢ−c}/|s|` — a concrete strictly-positive floor on every action probability, the
    lower-bound counterpart to `softmax_le_one`/`softmax_lt_one`. This is exactly the shape needed to DISCHARGE
    the abstract probability-floor hypotheses assumed downstream (e.g. `LogPolicyBound.logPolicy_error` takes
    `d ≤ softmax …` with `0 < d` as given — instantiating this at `c = max logit` supplies such a
    `d = e^{lᵢ−c}/|s| > 0` concretely). Holds for all `s` (the empty case collapses to `0 ≤ 0`). The ceiling
    hypothesis `hc` is load-bearing: without it a single large competing logit drives the probability below any
    fixed floor (e.g. `s = {0,1}`, `l 0 = 0`, `l 1 = 100`, `c = 0` gives floor `e^0/2 = 0.5` but
    `softmax₀ = 1/(1+e^{100}) ≈ 0 < 0.5`). Proof: bound the denominator via `Finset.sum_le_sum` + `exp`
    monotonicity, transport through the division with `div_le_div_iff₀`, and use `e^{lᵢ−c}·e^c = e^{lᵢ}`. -/
theorem softmax_ge_exp_div_card (s : Finset ι) (l : ι → ℝ) (i : ι) (c : ℝ)
    (hc : ∀ j ∈ s, l j ≤ c) :
    Real.exp (l i - c) / s.card ≤ softmax s l i := by
  rcases s.eq_empty_or_nonempty with he | hs
  · subst he; simp [softmax, softmaxDenom]
  · have hD : 0 < softmaxDenom s l := softmaxDenom_pos s l hs
    have hcard : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
    have hDle : softmaxDenom s l ≤ (s.card : ℝ) * Real.exp c := by
      rw [softmaxDenom]
      calc ∑ j ∈ s, Real.exp (l j)
          ≤ ∑ j ∈ s, Real.exp c :=
            Finset.sum_le_sum (fun j hj => Real.exp_le_exp.mpr (hc j hj))
        _ = (s.card : ℝ) * Real.exp c := by rw [Finset.sum_const, nsmul_eq_mul]
    have hkey : Real.exp (l i - c) * Real.exp c = Real.exp (l i) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [softmax, div_le_div_iff₀ hcard hD]
    calc Real.exp (l i - c) * softmaxDenom s l
        ≤ Real.exp (l i - c) * ((s.card : ℝ) * Real.exp c) :=
          mul_le_mul_of_nonneg_left hDle (Real.exp_pos _).le
      _ = (Real.exp (l i - c) * Real.exp c) * (s.card : ℝ) := by ring
      _ = Real.exp (l i) * (s.card : ℝ) := by rw [hkey]

/-- **The argmax (highest-logit) action always carries at least the uniform share `1/|s|`.** If action `i`
    has a maximal logit on `s` (`∀ j ∈ s, l j ≤ l i`), then `1 / s.card ≤ softmax s l i`. Immediate corollary
    of `softmax_ge_exp_div_card` specialized at `c = l i`: the floor `e^{lᵢ − lᵢ}/|s| = e^0/|s| = 1/|s|`. So the
    greedy/argmax action is never suppressed below uniform — a concrete guarantee for the exploration floor. -/
theorem softmax_argmax_ge_uniform (s : Finset ι) (l : ι → ℝ) (i : ι)
    (hi : ∀ j ∈ s, l j ≤ l i) :
    1 / s.card ≤ softmax s l i := by
  have h := softmax_ge_exp_div_card s l i (l i) hi
  rwa [sub_self, Real.exp_zero] at h

/-- **Log-probabilities are nonpositive.** `log softmax(l)ᵢ ≤ 0` — since a softmax probability lies in `(0,1]`,
    its log is `≤ 0` (`Real.log_nonpos`). The log-policy the actor optimizes is always `≤ 0`. -/
theorem log_softmax_nonpos (s : Finset ι) (l : ι → ℝ) (i : ι) (hs : s.Nonempty) (hi : i ∈ s) :
    Real.log (softmax s l i) ≤ 0 :=
  Real.log_nonpos (softmax_pos s l i hs).le (softmax_le_one s l i hs hi)

/-- **Softmax input-perturbation (uniform form).** `|softmax s a i − softmax s b i| ≤ e^{2ε} − 1`. -/
theorem softmax_input_perturb_le (s : Finset ι) (a b : ι → ℝ) (i : ι) (ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hε : 0 ≤ ε) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |softmax s a i - softmax s b i| ≤ Real.exp (2*ε) - 1 := by
  refine (softmax_input_perturb s a b i ε hs hi hab).trans ?_
  have h1 : (0:ℝ) ≤ Real.exp (2*ε) - 1 := by
    have := Real.one_le_exp (by linarith : (0:ℝ) ≤ 2*ε); linarith
  calc softmax s b i * (Real.exp (2*ε) - 1)
      ≤ 1 * (Real.exp (2*ε) - 1) := by gcongr; exact softmax_le_one s b i hs hi
    _ = Real.exp (2*ε) - 1 := one_mul _

/-! ### Composition: softmax Float rounding + input perturbation, on the runnable policy -/

/-- **Softmax-of-logits composition (abstract).** For a Float `logits` array and ANY ideal real logits `LR`
    within `ε` of `toReal ∘ logits`, `(Train.softmax logits)[i]!` is within `Bsm + (e^{2ε} − 1)` of
    `Net.softmax LR i` — the softmax Float-rounding budget `Bsm` (`train_softmax_error`) plus the softmax
    input-perturbation `e^{2ε} − 1`. Triangle inequality on `Net.softmax (toReal∘logits)`. -/
theorem softmax_logits_error (logits : Array Float) (LR : Nat → ℝ) (i : Nat) (ε Bsm : ℝ)
    (hi : i < logits.size) (hpos : 0 < logits.size)
    (hsm : |toReal ((Puffer.RL.Train.softmax logits)[i]!)
             - softmax (Finset.range logits.size) (fun j => toReal logits[j]!) i| ≤ Bsm)
    (hε : 0 ≤ ε)
    (hpert : ∀ k ∈ Finset.range logits.size, |toReal logits[k]! - LR k| ≤ ε) :
    |toReal ((Puffer.RL.Train.softmax logits)[i]!) - softmax (Finset.range logits.size) LR i|
      ≤ Bsm + (Real.exp (2*ε) - 1) := by
  have hpert' := softmax_input_perturb_le (Finset.range logits.size) (fun j => toReal logits[j]!) LR i ε
    (Finset.nonempty_range_iff.mpr (by omega)) (Finset.mem_range.mpr hi) hε hpert
  calc |toReal ((Puffer.RL.Train.softmax logits)[i]!) - softmax (Finset.range logits.size) LR i|
      ≤ |toReal ((Puffer.RL.Train.softmax logits)[i]!)
            - softmax (Finset.range logits.size) (fun j => toReal logits[j]!) i|
        + |softmax (Finset.range logits.size) (fun j => toReal logits[j]!) i
            - softmax (Finset.range logits.size) LR i| := abs_sub_le _ _ _
    _ ≤ Bsm + (Real.exp (2*ε) - 1) := add_le_add hsm hpert'

/-- The ideal REAL logits — layer-2 output with the exact real hidden `hRList` (`= toReal b2ₖ + dotRm W2ₖ hR`).
    The `Net.softmax` of these is the ideal action distribution. -/
noncomputable def idealLogitR (p : MLP) (size s : Nat) (k : Nat) : ℝ :=
  toReal p.b2[k]! + dotRm (p.W2[k]!).toList (hRList p (oneHot size s))

theorem forwardAll_logits_size (p : MLP) (x : Array Float) : (forwardAll p x).2.2.size = p.b2.size := by
  show ((Array.range p.b2.size).map _).size = _
  rw [Array.size_map, Array.size_range]

theorem policyProbs_eq (p : MLP) (size s : Nat) :
    policyProbs p size s = Puffer.RL.Train.softmax (forwardAll p (oneHot size s)).2.2 := rfl

/-- **Running-policy softmax-of-logits bound (capstone).** The trainer's actual `policyProbs p size s`
    output `[i]!` — the action probability it samples from — is within `Bsm + (e^{2ε} − 1)` of the ideal ℝ
    softmax of the ideal real logits `idealLogitR`. `Bsm` is the softmax Float-rounding budget (discharge via
    `train_softmax_error`); `ε` a uniform bound on the forward-pass logit errors, discharged concretely here
    from `forwardAll_logit_error`. Caps the forward-pass trifecta at the policy output on the code that runs. -/
theorem policyProbs_error (p : MLP) (size s : Nat) (i : Nat) (ε Bsm : ℝ)
    (hi : i < p.b2.size) (hpos : 0 < p.b2.size)
    (hsm : |toReal ((policyProbs p size s)[i]!)
             - softmax (Finset.range p.b2.size)
                 (fun j => toReal (forwardAll p (oneHot size s)).2.2[j]!) i| ≤ Bsm)
    (hε : 0 ≤ ε)
    (hεbnd : ∀ k, k < p.b2.size →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p (oneHot size s)).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p (oneHot size s)).2.1).toList (hRList p (oneHot size s))
        ≤ ε) :
    |toReal ((policyProbs p size s)[i]!)
        - softmax (Finset.range p.b2.size) (idealLogitR p size s) i|
      ≤ Bsm + (Real.exp (2*ε) - 1) := by
  set L := (forwardAll p (oneHot size s)).2.2 with hL
  have hsize : L.size = p.b2.size := forwardAll_logits_size p (oneHot size s)
  have hpert : ∀ k ∈ Finset.range L.size, |toReal L[k]! - idealLogitR p size s k| ≤ ε := by
    intro k hk
    rw [hsize] at hk
    have hk' : k < p.b2.size := Finset.mem_range.mp hk
    exact (forwardAll_logit_error p (oneHot size s) k hk').trans (hεbnd k hk')
  rw [policyProbs_eq]
  rw [policyProbs_eq] at hsm
  have := softmax_logits_error L (idealLogitR p size s) i ε Bsm (by rw [hsize]; exact hi)
    (by rw [hsize]; exact hpos) (by rw [hsize]; exact hsm) hε hpert
  rw [hsize] at this
  exact this

end Puffer.RL.PolicyBound
