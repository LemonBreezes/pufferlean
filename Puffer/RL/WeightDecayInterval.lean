/-
# Horizon-free whole-run interval under WEIGHT DECAY (the contraction case)

C27 (`WholeRunInterval`) proved the whole-run error interval in the GEOMETRIC form `L^n·d0 + B·Σ_{j<n} L^j` and
noted that the stronger HORIZON-FREE form `d0 + B/(1−L)` — bounded for EVERY `n`, no growth with training length —
requires a CONTRACTION `L < 1`, which plain gradient ascent (expansive, `L = 1 + |lr|·G ≥ 1`) never supplies. This
module supplies it via WEIGHT DECAY.

The weight-decay ascent step is `wdAscentE e lr wd σ = fun k => (1 − wd)·σ k + lr·∇e(σ) k` — the ascent with a
decay factor `1 − wd` on the parameters. Its sup-metric Lipschitz constant is
`L = |1 − toReal wd| + |toReal lr|·G` (`wdAscentE_sup_lipschitz`): the decay shrinks the identity part from `1`
(plain ascent) to `|1 − wd|`, so under `0 ≤ wd ≤ 1` and `wd > |lr|·G` (weight decay dominating the gradient's
expansion), `L = 1 − wd + |lr|·G < 1` — a genuine CONTRACTION (`wd_L_lt_one`).

Then `wd_whole_run_uniform_interval` gives the horizon-free bound: for the runnable trajectory `θ` (within `B` per
step of weight-decay ascent) and the ideal `θ'` (following it exactly), `∀ n k, |θ n k − θ' n k| ≤ d0 + B/(1 − L)`
— the runnable `Float` trajectory NEVER drifts more than the fixed neighborhood `d0 + B/(1 − L)` from the ideal
exact-ℝ trajectory, however long training runs. This is the strongest whole-run statement, mirroring C27's
accumulation but with C2's contractive `affine_recur_uniform` (which needs `L < 1`, `B ≥ 0`, `d0 ≥ 0`) in place of
the geometric `affine_recur_le`.

**Scope (honestly disclosed):** the horizon-free interval holds only under WEIGHT DECAY strong enough
(`toReal wd > |toReal lr|·G`, with `G` C26's concrete `Gtot`) to make the step contractive — plain gradient ascent
(C27) is expansive and only gets the growing geometric bound. The gradient-Lipschitz `hGtraj` is a hypothesis
(discharged at trajectory points by C26, as in C28); `B` (the per-step interval, C1) and the exact-ℝ-ascent-vs-
Muon/Adam gap are as in C27/C28. Weight decay here is the standard `(1 − wd)` parameter shrinkage; matching the
exact optimizer's decay schedule is a further step.
-/
import Puffer.RL.WholeRunInterval
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunInterval (errBound)
open Puffer.RL.MuonTrainBound (affine_recur_uniform)

namespace Puffer.RL.WeightDecayInterval

/-- **Weight-decay gradient-ascent step**: `σ k ↦ (1 − wd)·σ k + lr·∂e/∂(param k)` — one exact-ℝ gradient-ascent
    step with a decay factor `1 − wd` shrinking the parameters (standard L2 weight decay). -/
noncomputable def wdAscentE (e : Expr) (lr wd : Float) (σ : Nat → ℝ) : Nat → ℝ :=
  fun k => (1 - toReal wd) * σ k + toReal lr * derivR e σ k

/-- The weight-decay step is `(|1 − wd| + |lr|·G)`-Lipschitz (sup-metric): the decay shrinks the identity part to
    `|1 − wd|`, so with `wd > |lr|·G` the step CONTRACTS (`L < 1`). Mirror of C27's `gradAscentE_sup_lipschitz`. -/
theorem wdAscentE_sup_lipschitz (e : Expr) (lr wd : Float) (G δ : ℝ) (σ σ' : Nat → ℝ)
    (hclose : ∀ i, |σ i - σ' i| ≤ δ)
    (hG : ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    ∀ k, |wdAscentE e lr wd σ k - wdAscentE e lr wd σ' k|
      ≤ (|1 - toReal wd| + |toReal lr| * G) * δ := by
  intro k
  simp only [wdAscentE]
  calc |(1 - toReal wd) * σ k + toReal lr * derivR e σ k
          - ((1 - toReal wd) * σ' k + toReal lr * derivR e σ' k)|
      = |(1 - toReal wd) * (σ k - σ' k) + toReal lr * (derivR e σ k - derivR e σ' k)| := by ring_nf
    _ ≤ |(1 - toReal wd) * (σ k - σ' k)| + |toReal lr * (derivR e σ k - derivR e σ' k)| := abs_add_le _ _
    _ = |1 - toReal wd| * |σ k - σ' k| + |toReal lr| * |derivR e σ k - derivR e σ' k| := by
        rw [abs_mul, abs_mul]
    _ ≤ |1 - toReal wd| * δ + |toReal lr| * (G * δ) :=
        add_le_add (mul_le_mul_of_nonneg_left (hclose k) (abs_nonneg _))
          (mul_le_mul_of_nonneg_left (hG k) (abs_nonneg _))
    _ = (|1 - toReal wd| + |toReal lr| * G) * δ := by ring

/-- **The contraction condition.** For `0 ≤ wd ≤ 1`, the weight-decay Lipschitz `L = |1 − wd| + |lr|·G < 1`
    exactly when `|lr|·G < wd` — weight decay must dominate `|lr|·G` (with `G` C26's concrete `Gtot`) to make the
    ascent contractive, unlocking the horizon-free interval. -/
theorem wd_L_lt_one (lr wd : Float) (G : ℝ) (_hwd0 : 0 ≤ toReal wd) (hwd1 : toReal wd ≤ 1)
    (hdom : |toReal lr| * G < toReal wd) :
    |1 - toReal wd| + |toReal lr| * G < 1 := by
  rw [abs_of_nonneg (by linarith)]; linarith

/-- **HORIZON-FREE whole-run interval under weight-decay contraction.** For the runnable trajectory `θ` (within
    `B` per step of weight-decay ascent) and the ideal `θ'` (following it exactly), under the contraction
    `L = |1 − wd| + |lr|·G < 1` (`wd_L_lt_one`), the trajectories stay within a FIXED neighborhood for EVERY `n`:
    `∀ n k, |θ n k − θ' n k| ≤ d0 + B/(1 − L)` — no growth with training length. C27's accumulation with C2's
    contractive `affine_recur_uniform` and the weight-decay step's Lipschitz. -/
theorem wd_whole_run_uniform_interval (e : Expr) (lr wd : Float) (G B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = |1 - toReal wd| + |toReal lr| * G)
    (hGtraj : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
      ∀ k, |derivR e (θ p) k - derivR e (θ' p) k| ≤ G * δ)
    (hidealθ' : ∀ n, θ' (n + 1) = wdAscentE e lr wd (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k - wdAscentE e lr wd (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (hL1 : L < 1) (hB : 0 ≤ B) (hd0n : 0 ≤ d0) (n k : Nat) :
    |θ n k - θ' n k| ≤ d0 + B / (1 - L) := by
  have hbound : ∀ m j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro j; exact hd0 j
    | succ p ih =>
        intro j
        have hasc : ∀ i, |wdAscentE e lr wd (θ p) i - wdAscentE e lr wd (θ' p) i|
            ≤ L * errBound L B d0 p :=
          hL.symm ▸ wdAscentE_sup_lipschitz e lr wd G (errBound L B d0 p) (θ p) (θ' p) ih
            (hGtraj p (errBound L B d0 p) ih)
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - wdAscentE e lr wd (θ' p) j| := by rw [hidealθ' p]
          _ ≤ |θ (p + 1) j - wdAscentE e lr wd (θ p) j|
              + |wdAscentE e lr wd (θ p) j - wdAscentE e lr wd (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p j) (hasc j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have h0 : errBound L B d0 0 = d0 := rfl
  have hkey := affine_recur_uniform (errBound L B d0) L B hL0 hL1 hB (by rw [h0]; exact hd0n) hrec n
  rw [h0] at hkey
  exact (hbound n k).trans hkey

end Puffer.RL.WeightDecayInterval
