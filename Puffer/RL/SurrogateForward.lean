/-
The PPO clipped SURROGATE on the ACTUAL forward net vs the ideal — the actor-objective term's runnable
accuracy bound, built on the ratio-forward story (a119). Alongside the value-loss (a106) and entropy (a120)
forward bounds, this is the last of the three PPO-objective *terms* to land on the concrete network (the
surrogate is the objective's leading term; the ratio a119 was the exp inside it).

The surrogate `min(g·r, g·clip(r,lo,hi))` depends on the ratio `r` and a FIXED advantage `g` (a stored buffer
scalar, not recomputed in the forward pass), so its only forward-varying input is the ratio. Three pieces:

  • `ppoObjLoHi_perturb` — the ℝ surrogate `ppoObjLoHi` is `|g|`-Lipschitz in the ratio (general clip bounds),
    the `ppoObjLoHi` counterpart of `ppoObjective_perturb` (a74): `min`/`clip` 1-Lipschitz, both branches ×`g`.
  • `ppoSurrF_ratio_error` — ABSTRACT: given the Float ratio `r` is within `Bratio` of an ideal ratio
    `rIdeal`, the running `ppoSurrF` is within `ppoSurrF_error`'s rounding budget `+ |g|·Bratio` of the ℝ
    surrogate at `rIdeal`. Triangles `ppoSurrF_error` (Float→ℝ rounding) with `ppoObjLoHi_perturb` (ratio
    drift). The surrogate analog of `ratioFullF_ideal_error` (a115) — abstract in the ideal ratio and its bound.
  • `ppoSurrF_forward_error` — CONCRETE: discharges `rIdeal`/`Bratio` via `ratioFullF_forward_ideal_error`
    (a119), landing the running surrogate over the forward-net ratio within `ppoSurrF_error` `+ |g|·(a119
    budget)` of the ℝ surrogate at the ideal ratio `exp(log softmax(idealLogit)ₐ − oldLogp)`. The surrogate
    counterpart of a119's concrete ratio bound.

Axiom-clean (a119 ⊕ the surrogate runtime/perturb machinery: Float base + exp/log/add/sub/mul models +
`toReal_min`/`toReal_max`).
-/
import Puffer.RL.RatioForward

namespace Puffer.RL.SurrogateForward

open Puffer.FloatR
open Puffer.Net (softmax)
open Puffer.RL.NNTrain (MLP forwardAll)
open Puffer.RL.ActorCriticBound (idealLogit)
open Puffer.RL.LogSumExpBound (logpF)
open Puffer.RL.PPORuntime (ppoSurrF ppoSurrF_error ratioFullF)
open Puffer.RL.RatioForward (logpFwdBnd ratioFullF_forward_ideal_error)
open Puffer.RL.PPOPerturb (clampR_lipschitz)

/-- **ℝ surrogate is `|g|`-Lipschitz in the ratio** (general clip bounds `lo`/`hi`). The `min`/`clip` are
    1-Lipschitz and both branches scale by `g` — the `ppoObjLoHi` counterpart of `ppoObjective_perturb`. -/
theorem ppoObjLoHi_perturb (g ra rb lo hi : ℝ) :
    |Puffer.RL.PPO.ppoObjLoHi g ra lo hi - Puffer.RL.PPO.ppoObjLoHi g rb lo hi| ≤ |g| * |ra - rb| := by
  rw [Puffer.RL.PPO.ppoObjLoHi, Puffer.RL.PPO.ppoObjLoHi]
  refine (abs_min_sub_min_le_max _ _ _ _).trans ?_
  rw [max_le_iff]
  refine ⟨?_, ?_⟩
  · rw [← mul_sub, abs_mul]
  · rw [← mul_sub, abs_mul]
    exact mul_le_mul_of_nonneg_left (clampR_lipschitz ra rb lo hi) (abs_nonneg _)

/-- **Runnable surrogate vs the ideal-ratio surrogate (abstract).** Given the Float ratio `r` is within
    `Bratio` of an ideal ratio `rIdeal`, the running `ppoSurrF g r lo hi` is within `ppoSurrF_error`'s rounding
    budget plus `|g|·Bratio` of the ℝ surrogate at the ideal ratio. Abstract in `rIdeal`/`Bratio` — discharge
    via `ratioFullF_forward_ideal_error` (a119). -/
theorem ppoSurrF_ratio_error (g r lo hi : Float) (rIdeal Bratio : ℝ)
    (h : |toReal r - rIdeal| ≤ Bratio) :
    |toReal (ppoSurrF g r lo hi)
        - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)|
      ≤ u64 * max |toReal g * toReal r|
            |toReal g * Puffer.RL.PPO.clampR (toReal r) (toReal lo) (toReal hi)|
        + |toReal g| * Bratio := by
  have h1 := ppoSurrF_error g r lo hi
  have h2 : |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
        - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)|
      ≤ |toReal g| * Bratio :=
    (ppoObjLoHi_perturb (toReal g) (toReal r) rIdeal (toReal lo) (toReal hi)).trans
      (mul_le_mul_of_nonneg_left h (abs_nonneg _))
  calc |toReal (ppoSurrF g r lo hi)
          - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)|
      ≤ |toReal (ppoSurrF g r lo hi)
            - Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)|
        + |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
            - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h1 h2

/-- **Runnable surrogate vs the ideal surrogate, with BOTH ratio and advantage error.** Generalizes
    `ppoSurrF_ratio_error`: there the advantage weight `g` is treated as exact (compared against
    `ppoObjLoHi (toReal g) …`), but `g` is a *stored buffer scalar* and may itself differ from an ideal
    `gIdeal` by `Bg`. Given the Float ratio `r` is within `Bratio` of an ideal ratio `rIdeal` AND the Float
    weight `g` is within `Bg` of an ideal weight `gIdeal`, the running `ppoSurrF g r lo hi` is within
    `ppoSurrF_error`'s rounding budget, plus `|g|·Bratio` for the ratio drift, plus
    `Bg·max(|rIdeal|, |clip(rIdeal)|)` for the advantage drift, of the ℝ surrogate `ppoObjLoHi gIdeal rIdeal`.
    The `Bg` term uses the surrogate's exact Lipschitz constant in the advantage weight,
    `max(|rIdeal|, |clampR rIdeal lo hi|)` — the missing `g`-side companion to `ppoObjLoHi_perturb`'s
    ratio-side constant. Recovers `ppoSurrF_ratio_error` at `gIdeal = toReal g`, `Bg = 0`. Both perturbation
    hypotheses are load-bearing: dropping `hr` leaves `rIdeal` unconstrained (`r=1, rIdeal=100, g=lo=1, hi=100`,
    all budgets `0`: LHS `= 99`, RHS `= 0`); dropping `hg` leaves `gIdeal` unconstrained (`g=1, gIdeal=100,
    r=rIdeal=1, lo=0, hi=2`, all budgets `0`: LHS `= 99`, RHS `= 0`). Triangles `ppoSurrF_error` (Float→ℝ
    rounding), `ppoObjLoHi_perturb` (ratio drift), and the inline `g`-perturbation (advantage drift, via
    `abs_min_sub_min_le_max`). -/
theorem ppoSurrF_ratio_adv_error (g r lo hi : Float) (gIdeal rIdeal Bg Bratio : ℝ)
    (hg : |toReal g - gIdeal| ≤ Bg)
    (hr : |toReal r - rIdeal| ≤ Bratio) :
    |toReal (ppoSurrF g r lo hi)
        - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)|
      ≤ u64 * max |toReal g * toReal r|
            |toReal g * Puffer.RL.PPO.clampR (toReal r) (toReal lo) (toReal hi)|
        + |toReal g| * Bratio
        + Bg * max |rIdeal| |Puffer.RL.PPO.clampR rIdeal (toReal lo) (toReal hi)| := by
  -- (1) Float → ℝ rounding of the surrogate.
  have h1 := ppoSurrF_error g r lo hi
  -- (2) ratio drift: ppoObjLoHi is |g|-Lipschitz in the ratio.
  have h2 : |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
        - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)|
      ≤ |toReal g| * Bratio :=
    (ppoObjLoHi_perturb (toReal g) (toReal r) rIdeal (toReal lo) (toReal hi)).trans
      (mul_le_mul_of_nonneg_left hr (abs_nonneg _))
  -- (3) advantage drift: ppoObjLoHi is `max(|r|,|clip r|)`-Lipschitz in the advantage weight.
  have hgp : ∀ (a b rr ll hh : ℝ),
      |Puffer.RL.PPO.ppoObjLoHi a rr ll hh - Puffer.RL.PPO.ppoObjLoHi b rr ll hh|
        ≤ |a - b| * max |rr| |Puffer.RL.PPO.clampR rr ll hh| := by
    intro a b rr ll hh
    rw [Puffer.RL.PPO.ppoObjLoHi, Puffer.RL.PPO.ppoObjLoHi]
    refine (abs_min_sub_min_le_max _ _ _ _).trans ?_
    rw [max_le_iff]
    refine ⟨?_, ?_⟩
    · rw [show a * rr - b * rr = (a - b) * rr from by ring, abs_mul]
      exact mul_le_mul_of_nonneg_left (le_max_left _ _) (abs_nonneg _)
    · rw [show a * Puffer.RL.PPO.clampR rr ll hh - b * Puffer.RL.PPO.clampR rr ll hh
            = (a - b) * Puffer.RL.PPO.clampR rr ll hh from by ring, abs_mul]
      exact mul_le_mul_of_nonneg_left (le_max_right _ _) (abs_nonneg _)
  have h3 : |Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)
        - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)|
      ≤ Bg * max |rIdeal| |Puffer.RL.PPO.clampR rIdeal (toReal lo) (toReal hi)| :=
    (hgp (toReal g) gIdeal rIdeal (toReal lo) (toReal hi)).trans
      (mul_le_mul_of_nonneg_right hg (le_trans (abs_nonneg _) (le_max_left _ _)))
  -- triangle over the three drifts.
  calc |toReal (ppoSurrF g r lo hi)
          - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)|
      ≤ |toReal (ppoSurrF g r lo hi)
            - Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)|
        + |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
            - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)| := abs_sub_le _ _ _
    _ ≤ (u64 * max |toReal g * toReal r|
              |toReal g * Puffer.RL.PPO.clampR (toReal r) (toReal lo) (toReal hi)|)
        + (|toReal g| * Bratio
            + Bg * max |rIdeal| |Puffer.RL.PPO.clampR rIdeal (toReal lo) (toReal hi)|) := by
          refine add_le_add h1 ?_
          calc |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
                  - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)|
              ≤ |Puffer.RL.PPO.ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
                    - Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)|
                + |Puffer.RL.PPO.ppoObjLoHi (toReal g) rIdeal (toReal lo) (toReal hi)
                    - Puffer.RL.PPO.ppoObjLoHi gIdeal rIdeal (toReal lo) (toReal hi)| := abs_sub_le _ _ _
            _ ≤ _ := add_le_add h2 h3
    _ = _ := by ring

/-- **Runnable PPO surrogate on the actual forward net vs the ideal.** The trainer's clipped surrogate over
    the ratio computed from the REAL forward logits — `ppoSurrF g (ratioFullF (logpF (forwardAll…) a n) oldLogp)
    lo hi` — is within `ppoSurrF_error`'s rounding budget plus `|g|·(a119 ratio budget)` of the ℝ surrogate at
    the ideal ratio `exp(log softmax(idealLogit)ₐ − oldLogp)`. Composes `ppoSurrF_ratio_error` with
    `ratioFullF_forward_ideal_error` (a119). -/
theorem ppoSurrF_forward_error (p : MLP) (obs : Array Float) (g oldLogp lo hi : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList
            (Puffer.RL.ForwardExec.hRList p obs) ≤ ε) :
    |toReal (ppoSurrF g (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp) lo hi)
        - Puffer.RL.PPO.ppoObjLoHi (toReal g)
            (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp))
            (toReal lo) (toReal hi)|
      ≤ u64 * max |toReal g * toReal (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp)|
            |toReal g * Puffer.RL.PPO.clampR
              (toReal (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp))
              (toReal lo) (toReal hi)|
        + |toReal g| *
            ((expEps * Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
                + Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
                    * (Real.exp (u64 * |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
                          - toReal oldLogp|) - 1))
              + Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
                  * (Real.exp (logpFwdBnd p obs a n d ε) - 1)) := by
  exact ppoSurrF_ratio_error g (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp) lo hi
    (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)) _
    (ratioFullF_forward_ideal_error p obs oldLogp a n ε d hd hden hRden hne ha hn hεbnd)

end Puffer.RL.SurrogateForward
