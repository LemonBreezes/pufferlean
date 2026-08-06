/-
# The k-fold Newton–Schulz map is operator-norm Lipschitz — composing C47 across the Muon iterations

C47 (`NewtonSchulzMatrixLipschitz`) proved ONE Newton–Schulz step `nsStarStep a b c` is
`L_ns = |a| + 3|b|M² + 5|c|M⁴`-Lipschitz on the operator-norm ball `‖X‖ ≤ M` (abstractly, in a normed ∗-ring the
real square matrices instantiate). One full Muon NS map is `k` iterations of `nsStarStep`; this module composes
C47's single-step bound across those `k` iterations on NESTED balls — advancing C42's open Muon step-Lipschitz `L`.

The one step maps the ball of radius `M` into the ball of radius `nsMagBound a b c M = |a|·M + |b|·M³ + |c|·M⁵`
(`nsStarStep_mag`), so the `i`-th iterate lives in the ball of radius `nsRadius … i` (`nsIter_mem_ball`), and the
`k`-fold map's Lipschitz constant is the PRODUCT of the per-ball single-step constants:

* `nsStarStep_mag` / `nsStarStep_mag_le` — the range bound (one step: radius `M ↦ nsMagBound a b c M`).
* `nsRadius` / `nsRadius_nonneg` / `nsIter_mem_ball` — the nested-ball radii and that the `k`-th iterate stays in
  the `k`-th ball.
* `nsIter_lipschitz` (product form) — `‖nsIter k x − nsIter k y‖ ≤ (∏_{i<k} L_ns(nsRadius i))·‖x−y‖` on `‖x‖,‖y‖ ≤ M`
  (induction: composition of Lipschitz maps, the `k`-th step at radius `nsRadius k`).
* `nsIter_lipschitz_uniform` — the clean `L_ns^k` form: if every per-ball constant is `≤ Lu`, then
  `‖nsIter k x − nsIter k y‖ ≤ Lu^k·‖x−y‖`. This is the most useful for C42.
* `nsIter_lipschitz_uniform_delta` — in C42's `hlip` shape (`‖x−y‖ ≤ δ ⟹ ≤ Lu^k·δ`), the `k`-iteration Muon NS map's
  per-step operator-norm Lipschitz that `MuonAscentBridge.muon_whole_run_opnorm_interval` consumes.

**Scope (honestly disclosed).** This composes C47's single-NS-step Lipschitz across the `k` iterations of ONE Muon
NS map, giving the `k`-fold operator-norm Lipschitz — the product of per-ball constants, or `Lu^k` under a uniform
per-ball bound `Lu`. It is still LOCAL (on `‖X‖ ≤ M`, over the nested balls). Remaining toward C42's full Muon
whole-run `L`: (i) the uniform bound `Lu` requires the Newton–Schulz iteration's stability — Muon's coefficients are
chosen so the iterate norm stays `≲ 1` (the NS map converges to an orthogonal factor); supplying `hLu` is that
known property (taken here as a hypothesis, `nsRadius_nonincr`-style, not established); (ii) composing this NS map
with the outer momentum/parameter update `θ ↦ θ − lr·nsIter(momentum)`. This delivers the `k`-fold composition C47
flagged as remaining item (ii); the abstract bound instantiates on the repo's matrices exactly as C47's does.
-/
import Puffer.RL.NewtonSchulzMatrixLipschitz
open Puffer.RL.NewtonSchulzMatrixLipschitz

namespace Puffer.RL.NewtonSchulzIterate

variable {R : Type*} [NormedRing R] [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R]

/-- The per-step range bound: one Newton–Schulz step maps radius `M` to `|a|·M + |b|·M³ + |c|·M⁵`. -/
def nsMagBound (a b c M : ℝ) : ℝ := |a| * M + |b| * M ^ 3 + |c| * M ^ 5

/-- The single-step Lipschitz constant `L_ns` as a function of the ball radius `r`
    (`|a| + 3|b|r² + 5|c|r⁴`, matching C47's `nsStarStep_lipschitz`). -/
def nsLconst (a b c r : ℝ) : ℝ := |a| + 3 * |b| * r ^ 2 + 5 * |c| * r ^ 4

/-- `nsLconst` is nonnegative (all three terms are). -/
theorem nsLconst_nonneg (a b c r : ℝ) : 0 ≤ nsLconst a b c r := by
  rw [nsLconst]; positivity

/-- **One Newton–Schulz step's range bound:** `‖nsStarStep a b c x‖ ≤ |a|·‖x‖ + |b|·‖x‖³ + |c|·‖x‖⁵`
    (triangle over the `add`/`smul` structure + submultiplicativity `norm_mul3/5_le` + `‖star x‖ = ‖x‖`). -/
theorem nsStarStep_mag (a b c : ℝ) (x : R) :
    ‖nsStarStep a b c x‖ ≤ |a| * ‖x‖ + |b| * ‖x‖ ^ 3 + |c| * ‖x‖ ^ 5 := by
  have h3 : ‖x * star x * x‖ ≤ ‖x‖ ^ 3 := by
    refine (norm_mul3_le x (star x) x).trans (le_of_eq ?_); rw [norm_star]; ring
  have h5 : ‖x * star x * x * star x * x‖ ≤ ‖x‖ ^ 5 := by
    refine (norm_mul5_le x (star x) x (star x) x).trans (le_of_eq ?_); simp only [norm_star]; ring
  calc ‖nsStarStep a b c x‖
      = ‖a • x + b • (x * star x * x) + c • (x * star x * x * star x * x)‖ := by rw [nsStarStep]
    _ ≤ ‖a • x‖ + ‖b • (x * star x * x)‖ + ‖c • (x * star x * x * star x * x)‖ := by
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl); exact norm_add_le _ _
    _ = |a| * ‖x‖ + |b| * ‖x * star x * x‖ + |c| * ‖x * star x * x * star x * x‖ := by
        rw [norm_smul, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, Real.norm_eq_abs]
    _ ≤ |a| * ‖x‖ + |b| * ‖x‖ ^ 3 + |c| * ‖x‖ ^ 5 :=
        add_le_add (add_le_add le_rfl (mul_le_mul_of_nonneg_left h3 (abs_nonneg b)))
          (mul_le_mul_of_nonneg_left h5 (abs_nonneg c))

/-- One Newton–Schulz step maps the ball `‖x‖ ≤ M` into the ball of radius `nsMagBound a b c M`. -/
theorem nsStarStep_mag_le (a b c M : ℝ) (x : R) (_hM : 0 ≤ M) (hx : ‖x‖ ≤ M) :
    ‖nsStarStep a b c x‖ ≤ nsMagBound a b c M := by
  refine (nsStarStep_mag a b c x).trans ?_
  simp only [nsMagBound]
  gcongr

/-- **The `k`-fold Newton–Schulz map** = `k` iterations of `nsStarStep`. -/
def nsIter (a b c : ℝ) (k : ℕ) : R → R := (nsStarStep a b c)^[k]

omit [NormedStarGroup R] in
@[simp] theorem nsIter_zero (a b c : ℝ) (x : R) : nsIter a b c 0 x = x := rfl

omit [NormedStarGroup R] in
theorem nsIter_succ (a b c : ℝ) (k : ℕ) (x : R) :
    nsIter a b c (k + 1) x = nsStarStep a b c (nsIter a b c k x) := by
  simp only [nsIter, Function.iterate_succ_apply']

/-- The nested-ball radii: `nsRadius … 0 = M`, `nsRadius … (i+1) = nsMagBound a b c (nsRadius … i)`. -/
def nsRadius (a b c M : ℝ) : ℕ → ℝ
  | 0 => M
  | (i + 1) => nsMagBound a b c (nsRadius a b c M i)

theorem nsRadius_nonneg (a b c M : ℝ) (hM : 0 ≤ M) (k : ℕ) : 0 ≤ nsRadius a b c M k := by
  induction k with
  | zero => exact hM
  | succ k ih =>
      simp only [nsRadius, nsMagBound]
      exact add_nonneg (add_nonneg (mul_nonneg (abs_nonneg _) ih)
        (mul_nonneg (abs_nonneg _) (pow_nonneg ih 3))) (mul_nonneg (abs_nonneg _) (pow_nonneg ih 5))

/-- The `k`-th Newton–Schulz iterate stays in the `k`-th nested ball: `‖x‖ ≤ M ⟹ ‖nsIter k x‖ ≤ nsRadius … k`. -/
theorem nsIter_mem_ball (a b c M : ℝ) (hM : 0 ≤ M) (x : R) (hx : ‖x‖ ≤ M) (k : ℕ) :
    ‖nsIter a b c k x‖ ≤ nsRadius a b c M k := by
  induction k with
  | zero => simpa only [nsIter_zero, nsRadius] using hx
  | succ k ih =>
      rw [nsIter_succ]
      simpa only [nsRadius] using
        nsStarStep_mag_le a b c (nsRadius a b c M k) (nsIter a b c k x) (nsRadius_nonneg a b c M hM k) ih

/-- **The `k`-fold Newton–Schulz Lipschitz bound (product form).** On `‖x‖,‖y‖ ≤ M`,
    `‖nsIter k x − nsIter k y‖ ≤ (∏_{i<k} L_ns(nsRadius i))·‖x−y‖` — the product of the per-ball single-step
    constants (induction: composition of Lipschitz maps, the `k`-th step `L_ns`-Lipschitz at radius `nsRadius k`,
    where both iterates lie by `nsIter_mem_ball`). -/
theorem nsIter_lipschitz (a b c M : ℝ) (hM : 0 ≤ M) (x y : R) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M) (k : ℕ) :
    ‖nsIter a b c k x - nsIter a b c k y‖
      ≤ (∏ i ∈ Finset.range k, nsLconst a b c (nsRadius a b c M i)) * ‖x - y‖ := by
  induction k with
  | zero =>
      rw [nsIter_zero, nsIter_zero, Finset.range_zero, Finset.prod_empty, one_mul]
  | succ k ih =>
      rw [nsIter_succ, nsIter_succ]
      have hxk : ‖nsIter a b c k x‖ ≤ nsRadius a b c M k := nsIter_mem_ball a b c M hM x hx k
      have hyk : ‖nsIter a b c k y‖ ≤ nsRadius a b c M k := nsIter_mem_ball a b c M hM y hy k
      have hstep : ‖nsStarStep a b c (nsIter a b c k x) - nsStarStep a b c (nsIter a b c k y)‖
          ≤ nsLconst a b c (nsRadius a b c M k) * ‖nsIter a b c k x - nsIter a b c k y‖ := by
        rw [nsLconst]
        exact nsStarStep_lipschitz a b c (nsRadius a b c M k) (nsIter a b c k x) (nsIter a b c k y)
          (nsRadius_nonneg a b c M hM k) hxk hyk
      refine hstep.trans ?_
      rw [Finset.prod_range_succ]
      calc nsLconst a b c (nsRadius a b c M k) * ‖nsIter a b c k x - nsIter a b c k y‖
          ≤ nsLconst a b c (nsRadius a b c M k)
              * ((∏ i ∈ Finset.range k, nsLconst a b c (nsRadius a b c M i)) * ‖x - y‖) :=
            mul_le_mul_of_nonneg_left ih (nsLconst_nonneg a b c _)
        _ = (∏ i ∈ Finset.range k, nsLconst a b c (nsRadius a b c M i))
              * nsLconst a b c (nsRadius a b c M k) * ‖x - y‖ := by ring

/-- A product of `k` factors, each in `[0, Lu]`, is `≤ Lu^k`. -/
theorem prod_range_le_pow (g : ℕ → ℝ) (Lu : ℝ) (hg : ∀ i, 0 ≤ g i) (hLu : ∀ i, g i ≤ Lu) (k : ℕ) :
    ∏ i ∈ Finset.range k, g i ≤ Lu ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Finset.prod_range_succ, pow_succ]
      exact mul_le_mul ih (hLu k) (hg k) (pow_nonneg (le_trans (hg 0) (hLu 0)) k)

/-- **The `k`-fold Newton–Schulz Lipschitz bound, uniform `Lu^k` form.** If every per-ball single-step constant is
    `≤ Lu`, then `‖nsIter k x − nsIter k y‖ ≤ Lu^k·‖x−y‖` on `‖x‖,‖y‖ ≤ M` — the clean geometric Lipschitz for the
    whole `k`-iteration Muon NS map. -/
theorem nsIter_lipschitz_uniform (a b c M Lu : ℝ) (hM : 0 ≤ M) (x y : R) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M) (k : ℕ)
    (hLu : ∀ i, nsLconst a b c (nsRadius a b c M i) ≤ Lu) :
    ‖nsIter a b c k x - nsIter a b c k y‖ ≤ Lu ^ k * ‖x - y‖ := by
  refine (nsIter_lipschitz a b c M hM x y hx hy k).trans ?_
  refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
  exact prod_range_le_pow _ Lu (fun i => nsLconst_nonneg a b c _) hLu k

/-- The uniform bound in C42's `hlip` shape: `‖x − y‖ ≤ δ ⟹ ‖nsIter k x − nsIter k y‖ ≤ Lu^k·δ` — the `k`-iteration
    Muon NS map's per-step operator-norm Lipschitz, feeding `MuonAscentBridge.muon_whole_run_opnorm_interval`. -/
theorem nsIter_lipschitz_uniform_delta (a b c M Lu δ : ℝ) (hM : 0 ≤ M) (x y : R) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M)
    (k : ℕ) (hLu : ∀ i, nsLconst a b c (nsRadius a b c M i) ≤ Lu) (hLu0 : 0 ≤ Lu) (hδ : ‖x - y‖ ≤ δ) :
    ‖nsIter a b c k x - nsIter a b c k y‖ ≤ Lu ^ k * δ := by
  refine (nsIter_lipschitz_uniform a b c M Lu hM x y hx hy k hLu).trans ?_
  exact mul_le_mul_of_nonneg_left hδ (pow_nonneg hLu0 k)

end Puffer.RL.NewtonSchulzIterate
