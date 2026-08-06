/-
The PPO probability ratio and clipped surrogate under policy perturbation — the trust-region objective's
sensitivity to a logit shift, capping the PPO objective story.

PPO ascends `min(g·r, g·clip(r, 1−ε, 1+ε))` with the ratio `r = π(a|s)/π_old(a|s) = exp(log π − oldLogp)`
(`PPO.ppoObjective`/`PPO.ratio`). How far does this objective drift when the current policy's logits move
(e.g. from forward-pass rounding, `ε`)? The answer composes a72's softmax bound with one clean observation:
the ratio IS the softmax scaled — `ratio (log softmax(l)ᵢ) c = softmax(l)ᵢ · e^{−c}` (`ratio_eq_scaled`), so

  • `ratio_perturb` — `|ratio_a − ratio_b| ≤ ratio_b · (e^{2ε} − 1)` directly from `softmax_input_perturb`.
  • `clampR_lipschitz` — the clip `clampR` is 1-Lipschitz; `ppoObjective_perturb` — the clipped surrogate is
    `|g|`-Lipschitz in the ratio (`min`/`clip` 1-Lipschitz, both branches scaled by `g`).
  • `ppoObjective_policy_perturb` — composed: the clipped surrogate between the perturbed and ideal policy is
    within `|g| · ratio_ideal · (e^{2ε} − 1)`.
  • `ppoObjective_run_perturb` — on the actual forward logits: the PPO objective of the ℝ-softmax of the
    running net's logits (`Net.softmax (toReal ∘ forward-logits)`, the policy at the ℝ level — the softmax's
    OWN Float rounding, budget `Bsm`, is a72's separate term) is within `|g| · ratio_ideal · (e^{2ε} − 1)` of
    the ideal-logit objective, `ε` a uniform forward-pass logit-error bound (from `forwardAll_logit_error`).

All ℝ (Mathlib) — the perturbation lemmas are axiom-clean (no Float axioms); `ppoObjective_run_perturb`
inherits `ForwardExec`'s footprint via the logit error.
-/
import Puffer.RL.PolicyBound
import Puffer.RL.PPO

namespace Puffer.RL.PPOPerturb

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.PolicyBound

/-! ### The PPO ratio is the softmax scaled -/

variable {ι : Type*}

/-- The PPO ratio at policy `l` is the softmax scaled: `ratio (log softmax(l)ᵢ) c = softmax(l)ᵢ · e^{−c}`. -/
theorem ratio_eq_scaled (s : Finset ι) (l : ι → ℝ) (i : ι) (c : ℝ) (hs : s.Nonempty) :
    Puffer.RL.PPO.ratio (Real.log (softmax s l i)) c = softmax s l i * Real.exp (-c) := by
  rw [Puffer.RL.PPO.ratio,
    show Real.log (softmax s l i) - c = Real.log (softmax s l i) + (-c) from by ring,
    Real.exp_add, Real.exp_log (softmax_pos s l i hs)]

/-- **PPO ratio perturbation.** Under an `ε`-perturbation of the logits, the ratio moves within
    `ratio_ideal · (e^{2ε} − 1)` — the softmax bound scaled by `e^{−c}`. -/
theorem ratio_perturb (s : Finset ι) (a b : ι → ℝ) (i : ι) (c ε : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |Puffer.RL.PPO.ratio (Real.log (softmax s a i)) c - Puffer.RL.PPO.ratio (Real.log (softmax s b i)) c|
      ≤ Puffer.RL.PPO.ratio (Real.log (softmax s b i)) c * (Real.exp (2*ε) - 1) := by
  rw [ratio_eq_scaled s a i c hs, ratio_eq_scaled s b i c hs, ← sub_mul, abs_mul,
    abs_of_pos (Real.exp_pos (-c))]
  calc |softmax s a i - softmax s b i| * Real.exp (-c)
      ≤ (softmax s b i * (Real.exp (2*ε) - 1)) * Real.exp (-c) :=
        mul_le_mul_of_nonneg_right (softmax_input_perturb s a b i ε hs hi hab) (Real.exp_pos _).le
    _ = softmax s b i * Real.exp (-c) * (Real.exp (2*ε) - 1) := by ring

/-! ### The clipped surrogate is `|g|`-Lipschitz in the ratio -/

/-- `clampR` (clip to `[lo,hi]`) is 1-Lipschitz. -/
theorem clampR_lipschitz (x y lo hi : ℝ) :
    |Puffer.RL.PPO.clampR x lo hi - Puffer.RL.PPO.clampR y lo hi| ≤ |x - y| := by
  rw [Puffer.RL.PPO.clampR, Puffer.RL.PPO.clampR, max_comm lo (min x hi), max_comm lo (min y hi)]
  exact (abs_max_sub_max_le_abs (min x hi) (min y hi) lo).trans (abs_inf_sub_inf_le_abs x y hi)

/-- **PPO clipped-surrogate perturbation.** `ppoObjective` is `|g|`-Lipschitz in the ratio (`min`/`clip`
    1-Lipschitz, both branches scaled by `g`). -/
theorem ppoObjective_perturb (g ra rb ε : ℝ) :
    |Puffer.RL.PPO.ppoObjective g ra ε - Puffer.RL.PPO.ppoObjective g rb ε| ≤ |g| * |ra - rb| := by
  rw [Puffer.RL.PPO.ppoObjective, Puffer.RL.PPO.ppoObjective]
  refine (abs_min_sub_min_le_max _ _ _ _).trans ?_
  rw [max_le_iff]
  refine ⟨?_, ?_⟩
  · rw [← mul_sub, abs_mul]
  · rw [← mul_sub, abs_mul]
    exact mul_le_mul_of_nonneg_left (clampR_lipschitz ra rb (1-ε) (1+ε)) (abs_nonneg _)

/-- **PPO objective is `|g|`-Lipschitz in the clip width `ε`.** Holding the advantage-scale `g` and ratio
    `r` fixed, moving the PPO clip hyperparameter from `ε₁` to `ε₂` shifts the clipped surrogate
    `ppoObjective g r ε = min(g·r, g·clampR r (1−ε) (1+ε))` by at most `|g| · |ε₁ − ε₂|`. The clip window
    `[1−ε, 1+ε]` grows symmetrically at unit rate in `ε`, so `clampR r (1−ε) (1+ε)` is 1-Lipschitz in `ε`
    (both endpoints move at rate 1, and clamping only ever tracks one endpoint), and the surrounding
    `min`/scale-by-`g` inflate this to `|g|`. This is the objective's sensitivity to the clip hyperparameter
    itself — complementary to `ppoObjective_perturb` (sensitivity to the ratio) — and the `|g|` modulus is
    sharp (see `ppoObjective_clipwidth_lipschitz_tight`). -/
theorem ppoObjective_clipwidth_lipschitz (g r ε1 ε2 : ℝ) :
    |Puffer.RL.PPO.ppoObjective g r ε1 - Puffer.RL.PPO.ppoObjective g r ε2| ≤ |g| * |ε1 - ε2| := by
  have hclamp : |Puffer.RL.PPO.clampR r (1 - ε1) (1 + ε1) - Puffer.RL.PPO.clampR r (1 - ε2) (1 + ε2)|
      ≤ |ε1 - ε2| := by
    rw [Puffer.RL.PPO.clampR, Puffer.RL.PPO.clampR]
    refine (abs_max_sub_max_le_max _ _ _ _).trans ?_
    rw [max_le_iff]
    refine ⟨?_, ?_⟩
    · exact le_of_eq (by rw [show (1 - ε1) - (1 - ε2) = -(ε1 - ε2) from by ring, abs_neg])
    · refine (abs_min_sub_min_le_max _ _ _ _).trans ?_
      rw [sub_self, abs_zero, show (1 + ε1) - (1 + ε2) = ε1 - ε2 from by ring]
      exact max_le (abs_nonneg _) le_rfl
  rw [Puffer.RL.PPO.ppoObjective, Puffer.RL.PPO.ppoObjective]
  refine (abs_min_sub_min_le_max _ _ _ _).trans ?_
  rw [max_le_iff]
  refine ⟨?_, ?_⟩
  · rw [sub_self, abs_zero]; positivity
  · rw [← mul_sub, abs_mul]
    exact mul_le_mul_of_nonneg_left hclamp (abs_nonneg _)

/-- **Sharpness of the clip-width Lipschitz modulus.** The `|g|` constant in
    `ppoObjective_clipwidth_lipschitz` cannot be shrunk: at `g = 1, r = 100, ε₁ = 0, ε₂ = 1` the two
    objectives differ by exactly `|g|·|ε₁−ε₂| = 1` (with `r = 100` far above the clip window, the objective
    reads off the moving upper clip endpoint `g·(1+ε)`, so its ε-sensitivity is exactly `|g|`). -/
theorem ppoObjective_clipwidth_lipschitz_tight :
    |Puffer.RL.PPO.ppoObjective 1 100 0 - Puffer.RL.PPO.ppoObjective 1 100 1|
      = |(1 : ℝ)| * |(0 : ℝ) - 1| := by
  rw [Puffer.RL.PPO.ppoObjective, Puffer.RL.PPO.ppoObjective, Puffer.RL.PPO.clampR,
    Puffer.RL.PPO.clampR]
  norm_num

/-- **PPO clipped surrogate under policy perturbation.** The surrogate between the perturbed and ideal policy
    is within `|g| · ratio_ideal · (e^{2ε} − 1)`. -/
theorem ppoObjective_policy_perturb (s : Finset ι) (a b : ι → ℝ) (i : ι) (c ε g : ℝ) (hs : s.Nonempty)
    (hi : i ∈ s) (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |Puffer.RL.PPO.ppoObjective g (Puffer.RL.PPO.ratio (Real.log (softmax s a i)) c) ε
        - Puffer.RL.PPO.ppoObjective g (Puffer.RL.PPO.ratio (Real.log (softmax s b i)) c) ε|
      ≤ |g| * (Puffer.RL.PPO.ratio (Real.log (softmax s b i)) c * (Real.exp (2*ε) - 1)) :=
  (ppoObjective_perturb g _ _ ε).trans
    (mul_le_mul_of_nonneg_left (ratio_perturb s a b i c ε hs hi hab) (abs_nonneg _))

/-! ### Capstone: the running-policy PPO objective under forward-pass error -/

open Puffer.RL.NNTrain
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- **Running-net PPO objective perturbation.** The clipped surrogate of the ℝ-softmax of the ACTUAL forward
    logits (`Net.softmax (toReal ∘ forward-logits)` — the running net's policy at the ℝ level; the softmax's
    own Float rounding is a72's separate `Bsm`) is within `|g| · ratio_ideal · (e^{2ε} − 1)` of the ideal-logit
    objective, `ε` a uniform forward-pass logit-error bound (from `forwardAll_logit_error`). -/
theorem ppoObjective_run_perturb (p : MLP) (size s : Nat) (i : Nat) (c ε g : ℝ)
    (hpos : 0 < p.b2.size) (hi : i < p.b2.size)
    (hεbnd : ∀ k, k < p.b2.size →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p (oneHot size s)).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p (oneHot size s)).2.1).toList (hRList p (oneHot size s))
        ≤ ε) :
    |Puffer.RL.PPO.ppoObjective g (Puffer.RL.PPO.ratio (Real.log
            (softmax (Finset.range p.b2.size)
              (fun j => toReal (forwardAll p (oneHot size s)).2.2[j]!) i)) c) ε
        - Puffer.RL.PPO.ppoObjective g (Puffer.RL.PPO.ratio (Real.log
            (softmax (Finset.range p.b2.size) (idealLogitR p size s) i)) c) ε|
      ≤ |g| * (Puffer.RL.PPO.ratio (Real.log
          (softmax (Finset.range p.b2.size) (idealLogitR p size s) i)) c * (Real.exp (2*ε) - 1)) := by
  apply ppoObjective_policy_perturb (Finset.range p.b2.size)
    (fun j => toReal (forwardAll p (oneHot size s)).2.2[j]!) (idealLogitR p size s) i c ε g
    (Finset.nonempty_range_iff.mpr (by omega)) (Finset.mem_range.mpr hi)
  intro k hk
  have hk' : k < p.b2.size := Finset.mem_range.mp hk
  exact (forwardAll_logit_error p (oneHot size s) k hk').trans (hεbnd k hk')

end Puffer.RL.PPOPerturb
