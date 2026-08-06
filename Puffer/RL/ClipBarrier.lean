/-
# The clip-interior barrier: discharging the clip clause of C35's `hTrap` from a concrete per-step margin

C35 (`TrajReachability`) reduced the ideal trajectory's clip-interior membership to a LOCAL one-step trapping
premise `hTrap` (`∀ σ ∈ region, one projected step keeps the ratio in `(lo,hi)`). This module discharges the
CLIP-INTERIOR clause of that premise — honestly, as a per-step margin condition, because a fixed clip band is NOT
unconditionally trapping under bounded displacement (`[lo,hi]` maps to `[lo−d, hi+d]`; genuine set-invariance of a
fixed band needs a CONTRACTION toward the interior, which is data-dependent — the same expansiveness obstruction as
C29). What IS provable, in closed form, is the one-step barrier:

* `in_open_of_disp` — the arithmetic core: if `v` is `d`-strictly-inside `(lo,hi)` (`lo + d < v < hi − d`) and the
  next value moves by at most `d` (`|v' − v| ≤ d`), then `v' ∈ (lo,hi)`.

* `projAscentE_disp` — the per-coordinate displacement of ONE projected step is bounded by `|lr|·Gmag` (clamp is a
  1-Lipschitz fixed-point on the region, composed with the gradient-magnitude budget `Gmag`). Concrete, no
  contraction assumed.

* `ratioE_projStep_disp` — the PPO ratio's value change over one projected step, in CLOSED FORM:
  `≤ exp(Mlog − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/c)·(|lr|·Gmag)`. Composes C17/C18's log-softmax
  value-Lipschitz (`logSoftmaxE_value_lip`, over the partition floor `c`) with the exp local-Lipschitz
  (`exp_abs_sub_le`, over the log-prob magnitude `Mlog`) and `projAscentE_disp`. This is the concrete `d`.

* `clip_barrier` / `clip_barrier_concrete` — the barrier: if `ratio(σ)` is at least `d` inside the clip window
  (with `d` the concrete per-step ratio move), then `ratio(projAscentE … σ) ∈ (lo,hi)`. This discharges the
  clip-interior clause of `hTrap` AT `σ`.

**Scope (honestly disclosed).** This is a ONE-STEP, per-point sufficient condition — NOT an unconditional trapping
region. A fixed clip band is not invariant under bounded displacement (no contraction ⟹ the band can only grow), so
this does NOT make `hTrap` hold globally for free; it converts the clip-interior clause into the concrete numeric
margin `ratio(σ) ∈ (lo + d, hi − d)` with `d` computed from the network's budgets — exactly the per-step check the
`puffer verify` harness performs. `Gmag` (objective gradient magnitude), `Mlog` (log-prob magnitude), and `c`
(partition floor) are the network's `Smooth`/log-softmax budget constants over the `R`-region, supplied as inputs
(dischargeable by the C17/C18/C26 budget machinery, finite and checkable). The entropy clauses of `hTrap` (value +
derivative budgets) are a separate barrier, not covered here. Weight-clamping vs real gradient-clipping is as in C29.
-/
import Puffer.RL.RegionInvariance
import Puffer.RL.WholeRunFromC26
import Puffer.RL.LogSoftmaxValueBudgetExpr
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE projAscentE_mem abs_clamp_sub_le)
open Puffer.RL.WholeRunInterval (gradAscentE)
open Puffer.RL.WholeRunFromC26 (ppoObjE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.SurrogateExpr (ratioE evalR_ratioE)
open Puffer.RL.LogSoftmaxValueBudgetExpr (logSoftmaxE_value_lip)

namespace Puffer.RL.ClipBarrier

/-- **The arithmetic core of the barrier.** If `v` is `d`-strictly-inside the open interval `(lo, hi)`
    (`lo + d < v` and `v < hi − d`) and the next value differs by at most `d` (`|v' − v| ≤ d`), then `v'` is still
    in `(lo, hi)`. Pure real arithmetic: `v' ≥ v − d > lo` and `v' ≤ v + d < hi`. -/
theorem in_open_of_disp {lo hi d v v' : ℝ}
    (hlo : lo + d < v) (hhi : v < hi - d) (hdisp : |v' - v| ≤ d) :
    lo < v' ∧ v' < hi := by
  rw [abs_le] at hdisp
  exact ⟨by linarith [hdisp.1], by linarith [hdisp.2]⟩

/-- **One projected ascent step moves each coordinate by at most `|lr|·Gmag`.** On the region `|σ k| ≤ R` the clamp
    fixes `σ k` (`max (−R) (min (σ k) R) = σ k`), so `|projAscentE e lr R σ k − σ k| = |clamp(σ k + lr·∂) − clamp(σ
    k)| ≤ |lr·∂|` (clamp 1-Lipschitz, `abs_clamp_sub_le`), and `|∂| ≤ Gmag` (the gradient-magnitude budget). No
    contraction assumed — this is a plain per-step displacement bound. -/
theorem projAscentE_disp (e : Expr) (lr : Float) (R Gmag : ℝ) (σ : Nat → ℝ)
    (hσ : ∀ k, |σ k| ≤ R) (hGmag : ∀ k, |derivR e σ k| ≤ Gmag) :
    ∀ k, |projAscentE e lr R σ k - σ k| ≤ |toReal lr| * Gmag := by
  intro k
  obtain ⟨h1, h2⟩ := abs_le.mp (hσ k)
  have hfix : max (-R) (min (σ k) R) = σ k := by rw [min_eq_left h2, max_eq_right h1]
  have hbase := abs_clamp_sub_le (gradAscentE e lr σ k) (σ k) R
  rw [hfix] at hbase
  have heq : gradAscentE e lr σ k - σ k = toReal lr * derivR e σ k := by
    simp only [gradAscentE]; ring
  show |max (-R) (min (gradAscentE e lr σ k) R) - σ k| ≤ |toReal lr| * Gmag
  calc |max (-R) (min (gradAscentE e lr σ k) R) - σ k|
      ≤ |gradAscentE e lr σ k - σ k| := hbase
    _ = |toReal lr * derivR e σ k| := by rw [heq]
    _ = |toReal lr| * |derivR e σ k| := abs_mul _ _
    _ ≤ |toReal lr| * Gmag := mul_le_mul_of_nonneg_left (hGmag k) (abs_nonneg _)

/-- **The PPO ratio's value change over one projected step, in closed form.** With the log-partition floored by
    `c > 0` at both points and the log-prob magnitude bounded by `Mlog` at both points,
    `|ratio(projAscentE … σ) − ratio(σ)| ≤ exp(Mlog − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/c)·(|lr|·Gmag)`.
    Composes `logSoftmaxE_value_lip` (C18, the log-softmax value-Lipschitz over the floor) with `exp_abs_sub_le`
    (the exp local-Lipschitz over `Mlog`) and `projAscentE_disp` (the step displacement `|lr|·Gmag`). The ratio is
    `exp(logSoftmax − oldLogp)`, so its non-smoothness (log-partition) is handled exactly by the C18 budget. -/
theorem ratioE_projStep_disp (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Mlog c : ℝ) (σ : Nat → ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ k, |σ k| ≤ R) (hR : 0 ≤ R)
    (hGmag : ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k| ≤ Gmag)
    (hc : 0 < c)
    (hfloorσ : c ≤ evalR (expSumE (e :: es)) σ)
    (hfloorτ : c ≤ evalR (expSumE (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ))
    (hMσ : evalR (logSoftmaxE chosen (e :: es)) σ ≤ Mlog)
    (hMτ : evalR (logSoftmaxE chosen (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) ≤ Mlog) :
    |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ)
      - evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ|
      ≤ Real.exp (Mlog - toReal oldLogp)
          * (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag) := by
  set O := ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret with hO
  set τ := projAscentE O lr R σ with hτ
  have hlogsm : ∀ lp ∈ (e :: es), Smooth lp := fun lp hlp =>
    (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h)
  have hτreg : ∀ k, |τ k| ≤ R := projAscentE_mem O lr R σ hR
  have hdispβ : ∀ k, |τ k - σ k| ≤ |toReal lr| * Gmag := projAscentE_disp O lr R Gmag σ hσ hGmag
  -- log-softmax value-Lipschitz between σ and its projected image (C18, over the floor c):
  have hLlip : |evalR (logSoftmaxE chosen (e :: es)) τ - evalR (logSoftmaxE chosen (e :: es)) σ|
      ≤ (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag) :=
    logSoftmaxE_value_lip chosen (e :: es) hch hlogsm τ σ R (|toReal lr| * Gmag) c
      hτreg hσ hdispβ hR hc hfloorτ hfloorσ
  -- ratio = exp(logSoftmax − oldLogp); the exp local-Lipschitz over Mlog:
  simp only [evalR_ratioE, Puffer.RL.PPO.ratio]
  have hxy : (evalR (logSoftmaxE chosen (e :: es)) τ - toReal oldLogp)
      - (evalR (logSoftmaxE chosen (e :: es)) σ - toReal oldLogp)
      = evalR (logSoftmaxE chosen (e :: es)) τ - evalR (logSoftmaxE chosen (e :: es)) σ := by ring
  have hexp := exp_abs_sub_le (evalR (logSoftmaxE chosen (e :: es)) τ - toReal oldLogp)
    (evalR (logSoftmaxE chosen (e :: es)) σ - toReal oldLogp) (Mlog - toReal oldLogp)
    (by linarith [hMτ]) (by linarith [hMσ])
  rw [hxy] at hexp
  calc |Real.exp (evalR (logSoftmaxE chosen (e :: es)) τ - toReal oldLogp)
          - Real.exp (evalR (logSoftmaxE chosen (e :: es)) σ - toReal oldLogp)|
      ≤ Real.exp (Mlog - toReal oldLogp)
          * |evalR (logSoftmaxE chosen (e :: es)) τ - evalR (logSoftmaxE chosen (e :: es)) σ| := hexp
    _ ≤ Real.exp (Mlog - toReal oldLogp)
          * ((vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag)) := by
        gcongr
    _ = Real.exp (Mlog - toReal oldLogp)
          * (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag) := by ring

/-- **The clip-interior barrier (abstract `d`).** If the PPO ratio at `σ` is at least `d` inside the clip window
    (`lo + d < ratio(σ) < hi − d`) and one projected step moves the ratio by at most `d`, then the ratio after the
    step is still in `(lo, hi)`. Directly `in_open_of_disp`. This is the clip-interior clause of C35's `hTrap`,
    discharged AT `σ` from the per-step margin. -/
theorem clip_barrier (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R d : ℝ) (σ : Nat → ℝ)
    (hd : |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
            (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ)
          - evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ| ≤ d)
    (hlo : toReal lo + d < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi - d) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) < toReal hi :=
  in_open_of_disp hlo hhi hd

/-- **The clip-interior barrier (concrete `d`).** The full one-step discharge: with the network budgets (`Gmag`
    gradient magnitude, `Mlog` log-prob magnitude, `c` partition floor), if `ratio(σ)` is inside the clip window by
    at least the concrete per-step ratio move `d = exp(Mlog − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/c)·
    (|lr|·Gmag)`, then `ratio(projAscentE … σ) ∈ (lo, hi)`. Composes `ratioE_projStep_disp` (the concrete `d`) with
    `clip_barrier`. This is the clip-interior clause of `hTrap` reduced to the concrete numeric margin the `puffer
    verify` harness checks each step. -/
theorem clip_barrier_concrete (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Mlog c : ℝ) (σ : Nat → ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ k, |σ k| ≤ R) (hR : 0 ≤ R)
    (hGmag : ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k| ≤ Gmag)
    (hc : 0 < c)
    (hfloorσ : c ≤ evalR (expSumE (e :: es)) σ)
    (hfloorτ : c ≤ evalR (expSumE (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ))
    (hMσ : evalR (logSoftmaxE chosen (e :: es)) σ ≤ Mlog)
    (hMτ : evalR (logSoftmaxE chosen (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) ≤ Mlog)
    (hlo : toReal lo + Real.exp (Mlog - toReal oldLogp)
            * (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag)
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ
          < toReal hi - Real.exp (Mlog - toReal oldLogp)
            * (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag)) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) < toReal hi :=
  clip_barrier chosen e V es logps oldLogp g lo hi cv ce ret lr R _ σ
    (ratioE_projStep_disp chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag Mlog c σ
      hch he hes hσ hR hGmag hc hfloorσ hfloorτ hMσ hMτ)
    hlo hhi

end Puffer.RL.ClipBarrier
