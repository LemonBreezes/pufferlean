/-
The RUNNABLE critic value loss on the forward value, bounded against its IDEAL — tying the Float ½·square
rounding (`valueSqLossF_error`, a105) to the forward-pass value error (`value_error`/`sq_loss_perturb`, a75).

The actor-critic trainer computes the critic loss `½(V − ret)²` on `V = (policyAndValue p obs).2` (the forward
net's value output). Two error sources separate the runnable Float loss from the ideal `½(V* − ret)²` (`V*`
the ideal real value): the Float arithmetic of the `½·square` (a105), and the forward-pass error of `V`
itself (a75). `valueSqLossF_forward_error` is their triangle:

  • a105 (`valueSqLossF_error`) — `|toReal(valueSqLossF V ret) − ½(toReal V − toReal ret)²|` (the sub/square/
    0.5-multiply rounding; `0.5·x² = x²/2` bridged by `ring`).
  • a75 (`sq_loss_perturb` ∘ `value_error`) — `|½(toReal V − toReal ret)² − ½(V* − toReal ret)²|` (the
    forward-value perturbation propagated through the half-square).

So the value loss the trainer actually feeds the critic gradient is within the sum of the two of its ideal.
Axiom-clean beyond the trusted Float base (a105's `mul/sub_model` + `ofScientific` + a75's forward footprint).
-/
import Puffer.RL.ActorCriticBound
import Puffer.RL.PPORuntime

namespace Puffer.RL.ValueLossForward

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.NNTrain
open Puffer.RL.ActorCriticBound (value_error sq_loss_perturb idealLogit)
open Puffer.RL.ForwardExec (hRList)
open Puffer.RL.PPORuntime (valueSqLossF valueSqLossF_error sqBnd)

/-- **Runnable value-loss relative-error sandwich.** The executable unclipped critic value loss
    `valueSqLossF val ret = 0.5·((val−ret)·(val−ret))` (the exact `½·(val−ret)²` term the AD tape uses) stays
    within a MULTIPLICATIVE `[(1−u)⁵, (1+u)⁵]` envelope of the ideal real half-squared error
    `(toReal val − toReal ret)²/2`, `u = u64` the unit roundoff. Five rounding slots compound: the inner
    subtraction's `(1+δ₁)` (entering the square TWICE, hence `(1+δ₁)²`), the squaring multiply `(1+δ₂)`, the
    outer `0.5·` multiply `(1+δ₃)`, and the `0.5` decimal-literal embedding (`toReal_ofScientific_close`) — the
    product of all five factors is pinned to `[(1−u)⁵, (1+u)⁵]`. Unlike the ADDITIVE `valueSqLossF_error`, this
    is a RELATIVE two-sided bound: the runnable loss never differs from the ideal by more than a `≈5u` factor,
    so (with `valueSqLossF_nonneg`) it can neither flip sign nor spuriously read near-zero when the true
    squared error is large — the lower envelope `(1−u)⁵·½(V−ret)² > 0` forbids it whenever
    `toReal val ≠ toReal ret`. -/
theorem valueSqLossF_relative_sandwich (val ret : Float) :
    (1 - u64) ^ 5 * ((toReal val - toReal ret) ^ 2 / 2)
        ≤ toReal (valueSqLossF val ret)
    ∧ toReal (valueSqLossF val ret)
        ≤ (1 + u64) ^ 5 * ((toReal val - toReal ret) ^ 2 / 2) := by
  obtain ⟨δ1, hδ1, he1⟩ := sub_model val ret
  obtain ⟨δ2, hδ2, he2⟩ := mul_model (val - ret) (val - ret)
  obtain ⟨δ3, hδ3, he3⟩ := mul_model (0.5 : Float) ((val - ret) * (val - ret))
  have h05 : |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| :=
    toReal_ofScientific_close 5 true 1
  have hu1 : u64 < 1 := u64_lt_one
  have hunn : (0:ℝ) ≤ 1 - u64 := by linarith
  have h1u : (0:ℝ) ≤ 1 + u64 := by have := u64_pos; linarith
  obtain ⟨hδ1lo, hδ1hi⟩ := abs_le.mp hδ1
  obtain ⟨hδ2lo, hδ2hi⟩ := abs_le.mp hδ2
  obtain ⟨hδ3lo, hδ3hi⟩ := abs_le.mp hδ3
  have h05' : |(0.5:ℝ)| = 0.5 := by norm_num
  rw [h05', abs_le] at h05
  obtain ⟨h05lo, h05hi⟩ := h05
  -- exact expansion of the runnable loss into the ideal times the five rounding factors
  have hval : toReal (valueSqLossF val ret)
      = ((toReal val - toReal ret) ^ 2 / 2)
        * (2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3)) := by
    unfold valueSqLossF
    rw [he3, he2, he1]; ring
  -- per-factor two-sided bounds
  have hf0lo : 1 - u64 ≤ 2 * toReal (0.5:Float) := by linarith
  have hf0hi : 2 * toReal (0.5:Float) ≤ 1 + u64 := by linarith
  have hf0nn : (0:ℝ) ≤ 2 * toReal (0.5:Float) := by linarith
  have hf1lo : 1 - u64 ≤ 1 + δ1 := by linarith
  have hf1hi : 1 + δ1 ≤ 1 + u64 := by linarith
  have hf1nn : (0:ℝ) ≤ 1 + δ1 := by linarith
  have hf2lo : 1 - u64 ≤ 1 + δ2 := by linarith
  have hf2hi : 1 + δ2 ≤ 1 + u64 := by linarith
  have hf2nn : (0:ℝ) ≤ 1 + δ2 := by linarith
  have hf3lo : 1 - u64 ≤ 1 + δ3 := by linarith
  have hf3hi : 1 + δ3 ≤ 1 + u64 := by linarith
  have hf3nn : (0:ℝ) ≤ 1 + δ3 := by linarith
  have hb_lo : (1 - u64)^2 ≤ (1 + δ1)^2 := pow_le_pow_left₀ hunn hf1lo 2
  have hb_hi : (1 + δ1)^2 ≤ (1 + u64)^2 := pow_le_pow_left₀ hf1nn hf1hi 2
  have hb_nn : (0:ℝ) ≤ (1 + δ1)^2 := by positivity
  have hD2 : (0:ℝ) ≤ (toReal val - toReal ret)^2 / 2 := by positivity
  -- the five-factor product is pinned to [(1-u)^5, (1+u)^5]
  have hKhi : 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3) ≤ (1+u64)^5 := by
    have u12 : 2 * toReal (0.5:Float) * (1+δ1)^2 ≤ (1+u64) * (1+u64)^2 :=
      mul_le_mul hf0hi hb_hi hb_nn h1u
    have u123 : 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) ≤ (1+u64) * (1+u64)^2 * (1+u64) :=
      mul_le_mul u12 hf2hi hf2nn (mul_nonneg h1u (by positivity))
    have u1234 : 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3)
        ≤ (1+u64) * (1+u64)^2 * (1+u64) * (1+u64) :=
      mul_le_mul u123 hf3hi hf3nn (mul_nonneg (mul_nonneg h1u (by positivity)) h1u)
    calc 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3)
        ≤ (1+u64) * (1+u64)^2 * (1+u64) * (1+u64) := u1234
      _ = (1+u64)^5 := by ring
  have hKlo : (1-u64)^5 ≤ 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3) := by
    have l12 : (1-u64) * (1-u64)^2 ≤ 2 * toReal (0.5:Float) * (1+δ1)^2 :=
      mul_le_mul hf0lo hb_lo (by positivity) hf0nn
    have l123 : (1-u64) * (1-u64)^2 * (1-u64)
        ≤ 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) :=
      mul_le_mul l12 hf2lo hunn (mul_nonneg hf0nn hb_nn)
    have l1234 : (1-u64) * (1-u64)^2 * (1-u64) * (1-u64)
        ≤ 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3) :=
      mul_le_mul l123 hf3lo hunn (mul_nonneg (mul_nonneg hf0nn hb_nn) hf2nn)
    calc (1-u64)^5 = (1-u64) * (1-u64)^2 * (1-u64) * (1-u64) := by ring
      _ ≤ 2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3) := l1234
  refine ⟨?_, ?_⟩
  · rw [hval]
    have hh := mul_le_mul_of_nonneg_left hKlo hD2
    calc (1 - u64)^5 * ((toReal val - toReal ret)^2/2)
        = ((toReal val - toReal ret)^2/2) * (1-u64)^5 := by ring
      _ ≤ ((toReal val - toReal ret)^2/2)
            * (2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3)) := hh
  · rw [hval]
    have hh := mul_le_mul_of_nonneg_left hKhi hD2
    calc ((toReal val - toReal ret)^2/2)
            * (2 * toReal (0.5:Float) * (1+δ1)^2 * (1+δ2) * (1+δ3))
        ≤ ((toReal val - toReal ret)^2/2) * (1+u64)^5 := hh
      _ = (1+u64)^5 * ((toReal val - toReal ret)^2/2) := by ring

/-- **Forward value loss vanishes exactly at the target (soundness + completeness).** The runnable critic
    value loss `valueSqLossF val ret = ½·(val−ret)²` (computed natively in `Float`, with sub/×/½ rounding) is
    `0` in ℝ *iff* the predicted value equals the return in ℝ. Both directions are load-bearing: completeness
    (`val=ret ⟹ 0`) shows rounding never manufactures a spurious positive loss at the optimum, and soundness
    (`0 ⟹ val=ret`) shows rounding never manufactures a spurious zero away from it — the Float loss reports the
    critic as perfectly fit exactly when it is. The factorization `toReal (valueSqLossF val ret) =
    (toReal ½ · (1+δ₁)²(1+δ₂)(1+δ₃))·(toReal val − toReal ret)²` isolates the squared real gap against a
    strictly-POSITIVE rounding coefficient (each `(1+δ) ≥ 1−u64 > 0` and `toReal 0.5 > 0`), so the product is
    `0` iff the gap is. -/
theorem valueSqLossF_eq_zero_iff (val ret : Float) :
    toReal (valueSqLossF val ret) = 0 ↔ toReal val = toReal ret := by
  obtain ⟨δ1, hδ1, he1⟩ := sub_model val ret
  obtain ⟨δ2, hδ2, he2⟩ := mul_model (val - ret) (val - ret)
  obtain ⟨δ3, hδ3, he3⟩ := mul_model (0.5 : Float) ((val - ret) * (val - ret))
  have h05 : (0:ℝ) < toReal (0.5 : Float) := by
    have h : |toReal (0.5:Float) - (0.5:ℝ)| ≤ u64 * |(0.5:ℝ)| := toReal_ofScientific_close 5 true 1
    have hu : (u64:ℝ) < 1 := by unfold u64; norm_num
    rw [abs_le] at h; norm_num at h ⊢; nlinarith [h.1, u64_pos]
  have hu1 : (u64:ℝ) < 1 := by unfold u64; norm_num
  have hf1 : (0:ℝ) < 1 + δ1 := by have := (abs_le.mp hδ1).1; linarith
  have hf2 : (0:ℝ) < 1 + δ2 := by have := (abs_le.mp hδ2).1; linarith
  have hf3 : (0:ℝ) < 1 + δ3 := by have := (abs_le.mp hδ3).1; linarith
  -- exact factorization: strictly-positive rounding coefficient times the squared real gap
  have hfac : toReal (valueSqLossF val ret)
      = (toReal (0.5 : Float) * ((1 + δ1)^2 * (1 + δ2) * (1 + δ3)))
          * (toReal val - toReal ret)^2 := by
    unfold valueSqLossF; rw [he3, he2, he1]; ring
  have hApos : (0:ℝ) < toReal (0.5 : Float) * ((1 + δ1)^2 * (1 + δ2) * (1 + δ3)) :=
    mul_pos h05 (mul_pos (mul_pos (pow_pos hf1 2) hf2) hf3)
  constructor
  · intro h
    rw [hfac] at h
    have hS : (toReal val - toReal ret)^2 = 0 :=
      (mul_eq_zero.mp h).resolve_left (ne_of_gt hApos)
    have hd : toReal val - toReal ret = 0 := (pow_eq_zero_iff (by norm_num : (2:ℕ) ≠ 0)).mp hS
    linarith
  · intro h
    rw [hfac, h]; ring

/-- **Runnable critic value loss vs ideal.** The value loss the trainer actually computes on the forward
    critic value — `valueSqLossF (policyAndValue p obs).2 ret` (unclipped `½(V−ret)²` in `Float`) — is within
    the Float-rounding budget (a105) + the forward-value perturbation (a75) of the IDEAL `½(V*−ret)²`, `V*` the
    ideal real value at the critic output index. Combines the ½·square COMPUTATION rounding with the
    forward-pass value error. -/
theorem valueSqLossF_forward_error (p : MLP) (obs : Array Float) (ret : Float) (hpos : 0 < p.b2.size) :
    |toReal (valueSqLossF (policyAndValue p obs).2 ret)
        - (idealLogit p obs (p.b2.size - 1) - toReal ret) ^ 2 / 2|
      ≤ (u64 * |toReal (0.5 : Float) * toReal (((policyAndValue p obs).2 - ret) * ((policyAndValue p obs).2 - ret))|
          + |toReal (0.5 : Float)| * sqBnd ((policyAndValue p obs).2 - ret)
              (toReal ((policyAndValue p obs).2) - toReal ret)
              (u64 * |toReal ((policyAndValue p obs).2) - toReal ret|)
          + |(toReal ((policyAndValue p obs).2) - toReal ret) ^ 2| * (u64 * |(0.5 : ℝ)|))
        + (z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
            + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
          * (|toReal ((policyAndValue p obs).2) - toReal ret|
              + |idealLogit p obs (p.b2.size - 1) - toReal ret|) / 2 := by
  set V := (policyAndValue p obs).2 with hV
  set Vstar := idealLogit p obs (p.b2.size - 1) with hVstar
  -- a105: the Float ½·square rounding, reference 0.5·(toReal V − toReal ret)²
  have h105 := valueSqLossF_error V ret
  -- 0.5·x² = x²/2
  have hhalf : (0.5 : ℝ) * (toReal V - toReal ret) ^ 2 = (toReal V - toReal ret) ^ 2 / 2 := by ring
  rw [hhalf] at h105
  -- a75: the forward-value perturbation through the ½·square
  have hpert := sq_loss_perturb (toReal V) Vstar (toReal ret)
    (z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
      + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
    (value_error p obs hpos)
  calc |toReal (valueSqLossF V ret) - (Vstar - toReal ret) ^ 2 / 2|
      ≤ |toReal (valueSqLossF V ret) - (toReal V - toReal ret) ^ 2 / 2|
        + |(toReal V - toReal ret) ^ 2 / 2 - (Vstar - toReal ret) ^ 2 / 2| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h105 hpert

end Puffer.RL.ValueLossForward
