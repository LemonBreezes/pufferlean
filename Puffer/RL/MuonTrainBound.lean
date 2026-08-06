/-
Training-step COMPOSITION bound: the weight operator norm stays bounded across arbitrarily many Muon
updates. One step is the affine contraction recurrence `‖W'‖ ≤ ρ·‖W‖ + C` (`MuonStepBound.stepMat_opNorm_le`,
`ρ = |toReal(1−lr·wd)|` the weight-decay contraction factor, `C` the per-step forcing `|lr·scale|·B +
√(r·c)·E`). Iterating it:

  • `affine_recur_le`      : `a(n+1) ≤ ρ·a(n) + C` (ρ ≥ 0) ⟹ `a(n) ≤ ρⁿ·a(0) + C·∑_{k<n} ρᵏ` (induction,
      geometric-sum shift via `Finset.sum_range_succ'`).
  • `affine_recur_uniform` : for `0 ≤ ρ < 1`, `C ≥ 0`, `a(0) ≥ 0` the sequence is UNIFORMLY bounded:
      `a(n) ≤ a(0) + C/(1−ρ)` (`geom_sum_mul` gives `∑_{k<n} ρᵏ = (1−ρⁿ)/(1−ρ) ≤ 1/(1−ρ)`).
  • `weights_bounded_over_training` : applied to `a(n) = ‖toMatrixF (Wₙ)‖` — if each training step satisfies
      the Muon per-step recurrence (from `stepMat_opNorm_le`, with `ρ = |toReal(1−lr·wd)| < 1` and uniform
      forcing `C`), the weight operator norm stays `≤ ‖toMatrixF (W₀)‖ + C/(1−ρ)` for EVERY step `n`.

Axiom-clean (pure ℝ analysis over the trusted-Float per-step bound). This is the training-stability statement:
under Muon with weight decay `lr·wd ∈ (0,2)` (so `ρ < 1`), the trained weights never blow up — their operator
norm is bounded for all time by the initial norm plus a fixed multiple of the (dimension-free O(1)) per-step
forcing.
-/
import Mathlib
import Puffer.RL.MuonStepBound

namespace Puffer.RL.MuonTrainBound

open scoped Matrix Matrix.Norms.L2Operator BigOperators
open Puffer.FloatR (toReal)
open Puffer.FloatR.Muon (Mat)
open Puffer.RL.MatrixEmbed (toMatrixF)

/-- **Affine recurrence unrolled.** `a(n+1) ≤ ρ·a(n) + C` (`ρ ≥ 0`) gives `a(n) ≤ ρⁿ·a(0) + C·∑_{k<n} ρᵏ`. -/
theorem affine_recur_le (a : Nat → ℝ) (ρ C : ℝ) (hρ : 0 ≤ ρ)
    (hrec : ∀ n, a (n + 1) ≤ ρ * a n + C) :
    ∀ n, a n ≤ ρ ^ n * a 0 + C * ∑ k ∈ Finset.range n, ρ ^ k := by
  intro n
  induction n with
  | zero => simp
  | succ m ih =>
    have hsum : ∑ k ∈ Finset.range (m + 1), ρ ^ k = ρ * (∑ k ∈ Finset.range m, ρ ^ k) + 1 := by
      rw [Finset.sum_range_succ', pow_zero]
      simp only [pow_succ]
      rw [← Finset.sum_mul]; ring
    calc a (m + 1) ≤ ρ * a m + C := hrec m
      _ ≤ ρ * (ρ ^ m * a 0 + C * ∑ k ∈ Finset.range m, ρ ^ k) + C := by nlinarith [ih, hρ]
      _ = ρ ^ (m + 1) * a 0 + C * (ρ * (∑ k ∈ Finset.range m, ρ ^ k) + 1) := by ring
      _ = ρ ^ (m + 1) * a 0 + C * ∑ k ∈ Finset.range (m + 1), ρ ^ k := by rw [hsum]

/-- **Uniform bound.** For `0 ≤ ρ < 1`, `C ≥ 0`, `a(0) ≥ 0`: the sequence stays `≤ a(0) + C/(1−ρ)` for all `n`. -/
theorem affine_recur_uniform (a : Nat → ℝ) (ρ C : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : 0 ≤ C)
    (ha0 : 0 ≤ a 0) (hrec : ∀ n, a (n + 1) ≤ ρ * a n + C) (n : Nat) :
    a n ≤ a 0 + C / (1 - ρ) := by
  have h1ρ : 0 < 1 - ρ := by linarith
  have hgeom : ∑ k ∈ Finset.range n, ρ ^ k ≤ 1 / (1 - ρ) := by
    have hmul : (∑ k ∈ Finset.range n, ρ ^ k) * (1 - ρ) = 1 - ρ ^ n := by
      nlinarith [geom_sum_mul ρ n]
    rw [le_div_iff₀ h1ρ, hmul]
    nlinarith [show (0:ℝ) ≤ ρ ^ n by positivity]
  have hpow : ρ ^ n ≤ 1 := pow_le_one₀ hρ0 hρ1.le
  have hle := affine_recur_le a ρ C hρ0 hrec n
  have h1 : ρ ^ n * a 0 ≤ a 0 := by nlinarith [hpow, ha0]
  have h2 : C * ∑ k ∈ Finset.range n, ρ ^ k ≤ C / (1 - ρ) := by
    rw [div_eq_mul_inv, ← one_div]; exact mul_le_mul_of_nonneg_left hgeom hC
  linarith [hle, h1, h2]

/-- **Geometric convergence to the attractor `C/(1−ρ)`.** For `0 ≤ ρ < 1` and the affine-contraction
    recurrence `a(n+1) ≤ ρ·a(n) + C`, the sequence satisfies the SHARP transient bound
    `a n ≤ C/(1−ρ) + ρⁿ·(a 0 − C/(1−ρ))`: the excess over the attractor `C/(1−ρ)` decays geometrically at
    rate `ρ`. This strictly sharpens `affine_recur_uniform` (whose bound `a 0 + C/(1−ρ)` is the `ρⁿ ≤ 1`,
    `a 0 ≥ 0` slackening of this one) and pins the exact `n`-dependence: as `n → ∞` the bound → `C/(1−ρ)`,
    the self-bounding region boundary. No sign hypotheses on `C` or `a 0` are needed — only `0 ≤ ρ < 1`.
    Proof reuses `affine_recur_le`'s induction, then supplies the exact geometric closed form
    `∑_{k<n} ρᵏ = (1−ρⁿ)/(1−ρ)` (`geom_sum_mul` + `eq_div_iff`) and rearranges (`field_simp`/`ring`). -/
theorem affine_recur_geom (a : Nat → ℝ) (ρ C : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hrec : ∀ n, a (n + 1) ≤ ρ * a n + C) (n : Nat) :
    a n ≤ C / (1 - ρ) + ρ ^ n * (a 0 - C / (1 - ρ)) := by
  have h1ρ : (0 : ℝ) < 1 - ρ := by linarith
  have hne : (1 : ℝ) - ρ ≠ 0 := h1ρ.ne'
  have hle := affine_recur_le a ρ C hρ0 hrec n
  have hmul : (∑ k ∈ Finset.range n, ρ ^ k) * (1 - ρ) = 1 - ρ ^ n := by
    nlinarith [geom_sum_mul ρ n]
  have hsum : ∑ k ∈ Finset.range n, ρ ^ k = (1 - ρ ^ n) / (1 - ρ) := (eq_div_iff hne).mpr hmul
  rw [hsum] at hle
  have heq : ρ ^ n * a 0 + C * ((1 - ρ ^ n) / (1 - ρ))
      = C / (1 - ρ) + ρ ^ n * (a 0 - C / (1 - ρ)) := by
    field_simp
    ring
  linarith [hle, heq.le]

/-! ### N-step trajectory-error accumulation (the whole-training-run interval)

The `affine_recur_*` family above solves a SCALAR recurrence. These three lift it to TWO TRAJECTORIES in a
space `α` with a triangle-obeying distance `d`: the *actual* (runnable `Float`) trajectory `θ` and the *ideal*
(exact-ℝ) trajectory `θ'` that follows the ideal step map `F` exactly. The reduction is the standard one — the
per-step divergence splits (triangle inequality) into a rounding term (`≤ B`, the SINGLE-STEP composition
bound) and an input-error term amplified by the step map's Lipschitz constant (`≤ L·e_k`), giving the affine
recurrence `e_{k+1} ≤ L·e_k + B` — after which the scalar closed forms apply. The `hstep` slot is exactly what
`ReverseNetUpdateBound.mlpStep_reverse_bounded_sup_error` supplies (with `d` the ℓ∞ distance on parameters and
`F` the ideal gradient-ascent map); `L`/`hlip` — the ideal step map's Lipschitz constant (a gradient-Lipschitz
/ Hessian-type bound) — is the remaining hypothesis to discharge for a fully-closed whole-run interval. -/

/-- **Per-step affine recurrence for two-trajectory divergence.** With the ideal trajectory `θ'` following `F`
    exactly, the triangle inequality splits `d (θ (k+1)) (θ' (k+1))` into the per-step rounding term `hstep`
    (`≤ B`) and the Lipschitz-amplified input-error term `hlip` (`≤ L · d (θ k) (θ' k)`), yielding
    `d (θ (k+1)) (θ' (k+1)) ≤ L · d (θ k) (θ' k) + B`. -/
theorem trajectory_step_recur {α : Type*} (d : α → α → ℝ) (F : α → α)
    (θ θ' : Nat → α) (L B : ℝ)
    (htri : ∀ x y z, d x z ≤ d x y + d y z)
    (hlip : ∀ x y, d (F x) (F y) ≤ L * d x y)
    (hstep : ∀ k, d (θ (k + 1)) (F (θ k)) ≤ B)
    (hideal : ∀ k, θ' (k + 1) = F (θ' k)) :
    ∀ k, d (θ (k + 1)) (θ' (k + 1)) ≤ L * d (θ k) (θ' k) + B := by
  intro k
  calc d (θ (k + 1)) (θ' (k + 1))
      = d (θ (k + 1)) (F (θ' k)) := by rw [hideal k]
    _ ≤ d (θ (k + 1)) (F (θ k)) + d (F (θ k)) (F (θ' k)) := htri _ _ _
    _ ≤ B + L * d (θ k) (θ' k) := add_le_add (hstep k) (hlip _ _)
    _ = L * d (θ k) (θ' k) + B := by ring

/-- **N-step trajectory-error accumulation (geometric-sum closed form).** For the actual (`Float`) trajectory
    `θ` and the ideal (exact-ℝ) trajectory `θ'` following the ideal step map `F` exactly, given a uniform
    PER-STEP bound `B` (how far each actual step strays from `F` applied at the actual state — the single-step
    composition bound) and an `L`-Lipschitz ideal step map (`0 ≤ L`), the accumulated divergence after `n` steps
    is `d (θ n) (θ' n) ≤ L^n · d (θ 0) (θ' 0) + B · Σ_{k<n} L^k`: the initial error amplified by `L^n` plus the
    per-step rounding summed with geometric weights. Holds for ALL `L ≥ 0` (no contraction needed) — for `L = 1`
    it is `d₀ + n·B` (linear drift), for `L > 1` it grows geometrically. Reduces to the scalar affine recurrence
    (`trajectory_step_recur`) and applies `affine_recur_le`. -/
theorem nstep_trajectory_error {α : Type*} (d : α → α → ℝ) (F : α → α)
    (θ θ' : Nat → α) (L B : ℝ) (hL : 0 ≤ L)
    (htri : ∀ x y z, d x z ≤ d x y + d y z)
    (hlip : ∀ x y, d (F x) (F y) ≤ L * d x y)
    (hstep : ∀ k, d (θ (k + 1)) (F (θ k)) ≤ B)
    (hideal : ∀ k, θ' (k + 1) = F (θ' k))
    (n : Nat) :
    d (θ n) (θ' n) ≤ L ^ n * d (θ 0) (θ' 0) + B * ∑ k ∈ Finset.range n, L ^ k :=
  affine_recur_le (fun k => d (θ k) (θ' k)) L B hL
    (trajectory_step_recur d F θ θ' L B htri hlip hstep hideal) n

/-- **N-step trajectory error stays HORIZON-FREE under a contractive ideal step.** If the ideal step map is a
    strict contraction (`0 ≤ L < 1`), the per-step bound `B ≥ 0`, and the initial divergence is nonnegative,
    then the actual-vs-ideal trajectory divergence is bounded by `d (θ 0) (θ' 0) + B/(1−L)` for EVERY `n` — no
    growth with the number of training steps. So under a contractive ideal step the runnable `Float` trajectory
    never drifts more than a fixed neighborhood `B/(1−L)` (the per-step rounding `B` geometrically damped) away
    from the exact-ℝ trajectory, however long training runs. This is the `L^n ≤ 1` slackening of
    `nstep_trajectory_error` (via `affine_recur_uniform`), and the strongest whole-run statement available once
    a contraction constant `L < 1` is supplied (e.g. from weight decay making the ideal ascent contractive). -/
theorem nstep_trajectory_error_uniform {α : Type*} (d : α → α → ℝ) (F : α → α)
    (θ θ' : Nat → α) (L B : ℝ) (hL0 : 0 ≤ L) (hL1 : L < 1) (hB : 0 ≤ B)
    (hd0 : 0 ≤ d (θ 0) (θ' 0))
    (htri : ∀ x y z, d x z ≤ d x y + d y z)
    (hlip : ∀ x y, d (F x) (F y) ≤ L * d x y)
    (hstep : ∀ k, d (θ (k + 1)) (F (θ k)) ≤ B)
    (hideal : ∀ k, θ' (k + 1) = F (θ' k))
    (n : Nat) :
    d (θ n) (θ' n) ≤ d (θ 0) (θ' 0) + B / (1 - L) :=
  affine_recur_uniform (fun k => d (θ k) (θ' k)) L B hL0 hL1 hB hd0
    (trajectory_step_recur d F θ θ' L B htri hlip hstep hideal) n

/-- **Ascent-map Lipschitz reduction: a gradient-Lipschitz `G` gives the ideal-step Lipschitz `L = 1 + |lr|·G`.**
    If the gradient map `g` is `G`-Lipschitz on a normed parameter space `E` (a smoothness / Hessian-type bound),
    then one exact-ℝ gradient-ascent step `F θ = θ + lr • g θ` is `(1 + |lr|·G)`-Lipschitz:
    `‖F x − F y‖ ≤ (1 + |lr|·G)·‖x − y‖`. This is the glue that discharges the `hlip` hypothesis of the N-step
    accumulation `nstep_trajectory_error`/`_uniform` — with the norm distance `d x y = ‖x − y‖` (which supplies
    the `htri` triangle inequality via `norm_sub_le`) and this map `F` — turning the gradient-Lipschitz `G` into
    the ideal-step Lipschitz constant `L = 1 + |lr|·G`. So the whole-run interval reduces to bounding `G`, the
    gradient's Lipschitz constant (see `Puffer.FloatR.ADR.derivR_const_of_affine` (`G = 0` on the affine
    fragment) and `derivR_mul_var_lipschitz` (`G = 2` on the bilinear atom) for the first fragments of that
    bound). Proof: `F x − F y = (x − y) + lr • (g x − g y)`; the triangle inequality and `‖lr • v‖ = |lr|·‖v‖`
    then bound it by `‖x − y‖ + |lr|·G·‖x − y‖`. -/
theorem ascent_map_lipschitz {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (g : E → E) (lr G : ℝ) (hg : ∀ x y, ‖g x - g y‖ ≤ G * ‖x - y‖) (x y : E) :
    ‖(x + lr • g x) - (y + lr • g y)‖ ≤ (1 + |lr| * G) * ‖x - y‖ := by
  have h1 : (x + lr • g x) - (y + lr • g y) = (x - y) + lr • (g x - g y) := by
    rw [smul_sub]; abel
  rw [h1]
  calc ‖(x - y) + lr • (g x - g y)‖
      ≤ ‖x - y‖ + ‖lr • (g x - g y)‖ := norm_add_le _ _
    _ = ‖x - y‖ + |lr| * ‖g x - g y‖ := by rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x - y‖ + |lr| * (G * ‖x - y‖) := by
        have h := mul_le_mul_of_nonneg_left (hg x y) (abs_nonneg lr); linarith
    _ = (1 + |lr| * G) * ‖x - y‖ := by ring

/-- **Weights stay bounded across training.** For a weight sequence `W : ℕ → Mat` whose operator norm
    satisfies the Muon per-step contraction recurrence `‖toMatrixF (W(n+1))‖ ≤ ρ·‖toMatrixF (W n)‖ + C`
    (from `stepMat_opNorm_le`: `ρ = |toReal(1−lr·wd)|`, `C` the per-step forcing `|toReal(lr·scale)|·B +
    √(r·c)·E`) with `0 ≤ ρ < 1` (weight decay `lr·wd ∈ (0,2)`) and `C ≥ 0`, the weight operator norm stays
    `≤ ‖toMatrixF (W 0)‖ + C/(1−ρ)` for EVERY training step `n`. The Muon training-stability bound. -/
theorem weights_bounded_over_training {r c : Nat} (W : Nat → Mat) (ρ C : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : 0 ≤ C)
    (hstep : ∀ n, ‖toMatrixF r c (W (n + 1))‖ ≤ ρ * ‖toMatrixF r c (W n)‖ + C) (n : Nat) :
    ‖toMatrixF r c (W n)‖ ≤ ‖toMatrixF r c (W 0)‖ + C / (1 - ρ) :=
  affine_recur_uniform (fun k => ‖toMatrixF r c (W k)‖) ρ C hρ0 hρ1 hC (norm_nonneg _) hstep n

/-- **Biases stay bounded across training.** For a bias sequence `b : ℕ → Array Float` whose `i`-th entry
    satisfies the Muon per-entry recurrence `|toReal (b(n+1)[i])| ≤ ρ·|toReal (b(n)[i])| + C` (from
    `stepVec_entry_le`: `ρ = (1+u64)²·|toReal(1−lr·wd)|`, `C = (1+u64)²·|toReal lr|·(uniform |upd[i]|)`) with
    `0 ≤ ρ < 1` and `C ≥ 0`, the entry stays `≤ |toReal (b 0 [i])| + C/(1−ρ)` for EVERY training step `n`. The
    1D-bias analogue of `weights_bounded_over_training`, reusing the same `affine_recur_uniform`. -/
theorem bias_bounded_over_training (b : Nat → Array Float) (i : Nat) (ρ C : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : 0 ≤ C)
    (hstep : ∀ n, |toReal ((b (n + 1))[i]!)| ≤ ρ * |toReal ((b n)[i]!)| + C) (n : Nat) :
    |toReal ((b n)[i]!)| ≤ |toReal ((b 0)[i]!)| + C / (1 - ρ) :=
  affine_recur_uniform (fun k => |toReal ((b k)[i]!)|) ρ C hρ0 hρ1 hC (abs_nonneg _) hstep n

/-- **Sum of affine-recurrent sequences stays bounded.** If each of `m` nonnegative sequences `a k` obeys
    `a k (n+1) ≤ ρ·a k n + C k` (common contraction `ρ`), their SUM `∑ₖ a k n` obeys the same recurrence with
    combined forcing `∑ₖ C k`, hence (for `0 ≤ ρ < 1`) stays `≤ ∑ₖ a k 0 + (∑ₖ C k)/(1−ρ)` for all `n`. -/
theorem network_norm_bounded {m : Nat} (a : Fin m → Nat → ℝ) (ρ : ℝ) (C : Fin m → ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hC : ∀ k, 0 ≤ C k) (ha0 : ∀ k, 0 ≤ a k 0)
    (hstep : ∀ k n, a k (n + 1) ≤ ρ * a k n + C k) (n : Nat) :
    (∑ k, a k n) ≤ (∑ k, a k 0) + (∑ k, C k) / (1 - ρ) := by
  have hSstep : ∀ N, (∑ k, a k (N + 1)) ≤ ρ * (∑ k, a k N) + ∑ k, C k := by
    intro N
    calc ∑ k, a k (N + 1) ≤ ∑ k, (ρ * a k N + C k) := Finset.sum_le_sum (fun k _ => hstep k N)
      _ = ρ * (∑ k, a k N) + ∑ k, C k := by rw [Finset.sum_add_distrib, Finset.mul_sum]
  exact affine_recur_uniform (fun N => ∑ k, a k N) ρ (∑ k, C k) hρ0 hρ1
    (Finset.sum_nonneg (fun k _ => hC k)) (Finset.sum_nonneg (fun k _ => ha0 k)) hSstep n

/-- **Whole-network parameter norm bounded across training.** Given the four MLP parameter-tensor norm
    sequences (`W1`, `W2` operator norms, `b1`, `b2` scalar measures — e.g. per-entry `|toReal|`) each obeying
    the Muon per-step contraction recurrence with common `ρ < 1`, the whole-network norm
    `‖W1‖ + ‖W2‖ + n₁ + n₂` stays `≤` its initial value plus `(C₁+C₂+C₃+C₄)/(1−ρ)` for EVERY training step.
    The `Fin 4` instance of `network_norm_bounded`; all four tensors bounded simultaneously by one bound. -/
theorem mlp_params_bounded_over_training
    (aW1 aW2 ab1 ab2 : Nat → ℝ) (ρ CW1 CW2 Cb1 Cb2 : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hCW1 : 0 ≤ CW1) (hCW2 : 0 ≤ CW2) (hCb1 : 0 ≤ Cb1) (hCb2 : 0 ≤ Cb2)
    (haW10 : 0 ≤ aW1 0) (haW20 : 0 ≤ aW2 0) (hab10 : 0 ≤ ab1 0) (hab20 : 0 ≤ ab2 0)
    (hW1 : ∀ n, aW1 (n + 1) ≤ ρ * aW1 n + CW1) (hW2 : ∀ n, aW2 (n + 1) ≤ ρ * aW2 n + CW2)
    (hb1 : ∀ n, ab1 (n + 1) ≤ ρ * ab1 n + Cb1) (hb2 : ∀ n, ab2 (n + 1) ≤ ρ * ab2 n + Cb2) (n : Nat) :
    aW1 n + aW2 n + ab1 n + ab2 n
      ≤ (aW1 0 + aW2 0 + ab1 0 + ab2 0) + (CW1 + CW2 + Cb1 + Cb2) / (1 - ρ) := by
  have h := network_norm_bounded ![aW1, aW2, ab1, ab2] ρ ![CW1, CW2, Cb1, Cb2] hρ0 hρ1
    (fun k => by fin_cases k <;> assumption) (fun k => by fin_cases k <;> assumption)
    (fun k n => by fin_cases k <;> apply_assumption) n
  simpa [Fin.sum_univ_four] using h

/-! ### The bounded region as a training-loop invariant

The "for all `n`" bounds above are consequences of a stronger structural fact: a radius `R` with `ρ·R + C ≤
R` defines a FORWARD-INVARIANT region `{a ≤ R}` — one Muon step maps it into itself. So `‖params‖ ≤ R` is a
LOOP INVARIANT: established at init, it is preserved by every update, hence holds for the whole run. Any
`R ≥ C/(1−ρ)` is self-bounding (`self_bounding_radius`); the tightest is `R = a 0 + C/(1−ρ)`. -/

/-- A radius `R ≥ C/(1−ρ)` (for `ρ < 1`) is self-bounding: `ρ·R + C ≤ R`. -/
theorem self_bounding_radius (ρ C R : ℝ) (hρ1 : ρ < 1) (hR : C / (1 - ρ) ≤ R) : ρ * R + C ≤ R := by
  have h1ρ : 0 < 1 - ρ := by linarith
  rw [div_le_iff₀ h1ρ] at hR
  nlinarith [hR]

/-- **One-step region invariance.** If `a n ≤ R` and the radius is self-bounding (`ρ·R + C ≤ R`, `ρ ≥ 0`),
    then `a (n+1) ≤ R` — the step keeps the sequence inside `{· ≤ R}`. -/
theorem region_invariant_step (a : Nat → ℝ) (ρ C R : ℝ) (hρ : 0 ≤ ρ)
    (hself : ρ * R + C ≤ R) (hrec : ∀ n, a (n + 1) ≤ ρ * a n + C) (n : Nat) (hn : a n ≤ R) :
    a (n + 1) ≤ R :=
  le_trans (hrec n) (le_trans (by nlinarith [hn, hρ]) hself)

/-- **The bounded region is a loop invariant.** Established at init (`a 0 ≤ R`) and preserved by every step
    (self-bounding `R`), so `a n ≤ R` for the WHOLE run. The invariant form of `affine_recur_uniform`. -/
theorem region_invariant (a : Nat → ℝ) (ρ C R : ℝ) (hρ : 0 ≤ ρ)
    (hself : ρ * R + C ≤ R) (h0 : a 0 ≤ R) (hrec : ∀ n, a (n + 1) ≤ ρ * a n + C) :
    ∀ n, a n ≤ R := by
  intro n
  induction n with
  | zero => exact h0
  | succ m ih => exact region_invariant_step a ρ C R hρ hself hrec m ih

/-- **Whole-network bounded-region loop invariant.** For the summed MLP parameter norm `S n = ∑ₖ a k n`
    (weights' operator norms + biases' scalar measures) with common contraction `ρ < 1` and combined forcing
    `∑ₖ C k`, once `S 0 ≤ R` for any self-bounding `R` (e.g. `R = ∑ₖ a k 0 + (∑ₖ C k)/(1−ρ)`), the Muon
    training loop MAINTAINS `S n ≤ R` at every step — the parameters never leave the bounded region. -/
theorem network_region_invariant {m : Nat} (a : Fin m → Nat → ℝ) (ρ R : ℝ) (C : Fin m → ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hR : (∑ k, C k) / (1 - ρ) ≤ R) (hS0 : (∑ k, a k 0) ≤ R)
    (hstep : ∀ k n, a k (n + 1) ≤ ρ * a k n + C k) :
    ∀ n, (∑ k, a k n) ≤ R := by
  have hSstep : ∀ N, (∑ k, a k (N + 1)) ≤ ρ * (∑ k, a k N) + ∑ k, C k := by
    intro N
    calc ∑ k, a k (N + 1) ≤ ∑ k, (ρ * a k N + C k) := Finset.sum_le_sum (fun k _ => hstep k N)
      _ = ρ * (∑ k, a k N) + ∑ k, C k := by rw [Finset.sum_add_distrib, Finset.mul_sum]
  exact region_invariant (fun N => ∑ k, a k N) ρ (∑ k, C k) R hρ0
    (self_bounding_radius ρ (∑ k, C k) R hρ1 hR) hS0 hSstep

end Puffer.RL.MuonTrainBound
