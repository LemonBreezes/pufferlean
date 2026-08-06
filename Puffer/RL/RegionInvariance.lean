/-
# Region-invariance via PROJECTED gradient ascent: the trajectory stays in the region for all N

The whole-run interval (C27/C28) holds only while the trajectory stays in C26's valid REGION (params bounded by
`R`). C27/C28 left that as an unestablished invariance. This module establishes it — but honestly: plain
gradient ascent is EXPANSIVE (`L = 1 + |lr|·G ≥ 1`), so it can escape ANY fixed region `R`; a fixed-`R`
invariance is generally FALSE for plain ascent. The clean, provable route is PROJECTED gradient ascent
(weight-clamping each coordinate to `[−R, R]`), which keeps params in the region BY CONSTRUCTION and — because
clamping is 1-Lipschitz — preserves the whole-run Lipschitz bound.

* `projAscentE e lr R σ = clamp(σ + lr·∇e(σ), −R, R)` (each coordinate clamped to `[−R, R]`).
* `projAscentE_traj_bounded` (THE REGION-INVARIANCE): a trajectory following projected ascent from a bounded
  start stays in `[−R, R]` for ALL `n` — `∀ n k, |θ n k| ≤ R`. By induction: the clamp lands in `[−R, R]` every
  step (`projAscentE_mem`), so boundedness is preserved unconditionally.
* `projAscentE_sup_lipschitz`: the projected step is still `(1 + |lr|·G)`-Lipschitz (clamp is 1-Lipschitz —
  `abs_clamp_sub_le` — composed with the C27 ascent-Lipschitz), so the whole-run geometric bound is unaffected.
* `proj_whole_run_sup_interval`: the whole-run interval `L^n·d0 + B·Σ L^j` for the PROJECTED trajectory (runnable
  `θ` within `B`/step of projected ascent, ideal `θ'` following it exactly) — C27's accumulation with the
  projected step map. Combined with `projAscentE_traj_bounded` (the ideal `θ'` stays in `[−R, R]`), this
  discharges the PARAM-BOUNDEDNESS part of C26/C28's region condition FOR the projected algorithm.

**Scope (honestly disclosed):** this establishes region-invariance for PROJECTED gradient ascent (weight-clamping
to `[−R, R]`), a specific algorithm variant — NOT plain gradient ascent (which is expansive and escapes bounded
regions; a fixed-`R` invariance there needs a contraction, e.g. weight decay). Weight-clamping differs from the
gradient-clipping real PPO/Muon uses. And it discharges only the `|σ i| ≤ R` PARAM-BOUNDEDNESS part of C26's
applicability conditions — the clip-interior (`ratio ∈ (lo,hi)`) and entropy-budget conditions are VALUE-level
(they depend on the objective's value at the params, not on param boundedness alone) and are NOT established here.
So this closes the region (param-bound) sub-condition of the C28 invariance for projected ascent; the value-level
sub-conditions and the plain-ascent case remain.
-/
import Puffer.RL.WholeRunInterval
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunInterval (gradAscentE errBound)
open Puffer.RL.MuonTrainBound (affine_recur_le)

namespace Puffer.RL.RegionInvariance

/-- Clamp to `[−R, R]` (`max (−R) (min · R)`) is 1-Lipschitz: `|clamp x − clamp y| ≤ |x − y|` (min/max are each
    1-Lipschitz, via Mathlib's `abs_max_sub_max_le_abs` + `abs_min_sub_min_le_max`). -/
theorem abs_clamp_sub_le (x y R : ℝ) :
    |max (-R) (min x R) - max (-R) (min y R)| ≤ |x - y| := by
  rw [max_comm (-R) (min x R), max_comm (-R) (min y R)]
  calc |max (min x R) (-R) - max (min y R) (-R)|
      ≤ |min x R - min y R| := abs_max_sub_max_le_abs _ _ _
    _ ≤ |x - y| := by simpa using abs_min_sub_min_le_max x R y R

/-- The clamp lands in `[−R, R]`: `|max (−R) (min t R)| ≤ R` for `R ≥ 0`. -/
theorem abs_clamp_le (t R : ℝ) (hR : 0 ≤ R) : |max (-R) (min t R)| ≤ R := by
  rw [abs_le]
  refine ⟨le_max_left _ _, max_le (by linarith) (min_le_right _ _)⟩

/-- **Projected gradient-ascent step**: `σ k ↦ clamp(σ k + lr·∂e/∂k, −R, R)` — one exact-ℝ ascent step with each
    coordinate clamped back into `[−R, R]` (projected/weight-clamped ascent). -/
noncomputable def projAscentE (e : Expr) (lr : Float) (R : ℝ) (σ : Nat → ℝ) : Nat → ℝ :=
  fun k => max (-R) (min (gradAscentE e lr σ k) R)

/-- One projected step lands in `[−R, R]` by construction: `∀ k, |projAscentE e lr R σ k| ≤ R` (`R ≥ 0`). -/
theorem projAscentE_mem (e : Expr) (lr : Float) (R : ℝ) (σ : Nat → ℝ) (hR : 0 ≤ R) :
    ∀ k, |projAscentE e lr R σ k| ≤ R :=
  fun _ => abs_clamp_le _ R hR

/-- **THE REGION-INVARIANCE.** A trajectory following projected ascent from a bounded start stays in `[−R, R]`
    for ALL `n`: `∀ n k, |θ n k| ≤ R`. Proved by induction — the clamp keeps every step in `[−R, R]`
    unconditionally, so params never escape the region (the invariance C27/C28 needed, now established for the
    projected algorithm). -/
theorem projAscentE_traj_bounded (e : Expr) (lr : Float) (R : ℝ) (θ : Nat → (Nat → ℝ)) (hR : 0 ≤ R)
    (h0 : ∀ k, |θ 0 k| ≤ R) (hstep : ∀ n, θ (n + 1) = projAscentE e lr R (θ n)) :
    ∀ n k, |θ n k| ≤ R := by
  intro n
  induction n with
  | zero => exact h0
  | succ p _ => intro k; rw [hstep p]; exact projAscentE_mem e lr R (θ p) hR k

/-- The projected step preserves the ascent-Lipschitz `L = 1 + |lr|·G` (clamp is 1-Lipschitz composed with the
    C27 ascent-Lipschitz), so projection does NOT weaken the whole-run geometric bound. -/
theorem projAscentE_sup_lipschitz (e : Expr) (lr : Float) (R G δ : ℝ) (σ σ' : Nat → ℝ)
    (hclose : ∀ i, |σ i - σ' i| ≤ δ)
    (hG : ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    ∀ k, |projAscentE e lr R σ k - projAscentE e lr R σ' k| ≤ (1 + |toReal lr| * G) * δ := by
  intro k
  exact (abs_clamp_sub_le _ _ R).trans
    (Puffer.RL.WholeRunInterval.gradAscentE_sup_lipschitz e lr G δ σ σ' hclose hG k)

/-- **Projected whole-run interval.** For the runnable trajectory `θ` (within `B` per step of projected ascent)
    and the ideal `θ'` (following projected ascent exactly), the geometric interval `L^n·d0 + B·Σ_{j<n} L^j` holds
    — C27's accumulation with the projected step map (its Lipschitz from `projAscentE_sup_lipschitz`). Together with
    `projAscentE_traj_bounded` (the ideal stays in `[−R, R]`), the region (param-bound) condition is discharged for
    the projected algorithm. -/
theorem proj_whole_run_sup_interval (e : Expr) (lr : Float) (R G B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * G)
    (hGtraj : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
      ∀ k, |derivR e (θ p) k - derivR e (θ' p) k| ≤ G * δ)
    (hidealθ' : ∀ n, θ' (n + 1) = projAscentE e lr R (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k - projAscentE e lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hbound : ∀ m j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro j; exact hd0 j
    | succ p ih =>
        intro j
        have hasc : ∀ i, |projAscentE e lr R (θ p) i - projAscentE e lr R (θ' p) i|
            ≤ L * errBound L B d0 p :=
          hL.symm ▸ projAscentE_sup_lipschitz e lr R G (errBound L B d0 p) (θ p) (θ' p) ih
            (hGtraj p (errBound L B d0 p) ih)
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - projAscentE e lr R (θ' p) j| := by rw [hidealθ' p]
          _ ≤ |θ (p + 1) j - projAscentE e lr R (θ p) j|
              + |projAscentE e lr R (θ p) j - projAscentE e lr R (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p j) (hasc j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have hkey := affine_recur_le (errBound L B d0) L B hL0 hrec n
  have h0 : errBound L B d0 0 = d0 := rfl
  rw [h0] at hkey
  exact (hbound n k).trans hkey

end Puffer.RL.RegionInvariance
