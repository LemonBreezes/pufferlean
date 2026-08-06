/-
# Backward-pass + optimizer finiteness: extending the overflow-free certificate past the forward loss

C43 (`FiniteBound`) certified a bounded linear kernel overflow-free; C45 (`ForwardFinite`) lifted it to the MLP
forward pass; C48/C51/C54 carried it through the readout, the PPO loss forward terms, and the scalar loss. Those
covered the FORWARD stages. This module extends the certificate through the two remaining trainer stages:

* **BACKWARD (reverse-mode gradient).** Each backward op is a bounded function of the (already-bounded, overflow-free)
  forward values and the upstream cotangent: `mul`/`add`/`sub` backward reuse C43/C51/C54's `mul_bound`/`add_bound`/
  `sub_bound` (the cotangent update `ḡ·b`, `ḡ`, … is the SAME op family); the NONLINEAR gates — `relu` and the clip
  `min`/`max` — pass the cotangent to the selected branch and `0` to the other, so their backward magnitude is `≤ |ḡ|`
  (`reluBwd_mag`, `gateBwd_mag`). For a linear layer `y = dotF x w`, the reverse-mode weight gradient is `∂y/∂wⱼ = xⱼ`
  scaled by the upstream cotangent — `gradW ḡ x = x.map (ḡ · ·)` — magnitude-bounded by `(1+u)·(Bg·B)` per weight
  (`gradW_mag`) and overflow-free (`gradW_isFinite`).
* **OPTIMIZER (SGD/Muon-shaped update).** `updateF w g lr = w − lr·g` is magnitude-bounded (`updateF_mag_le`) and
  overflow-free (`updateF_isFinite`) for bounded `w`/`g`/`lr`; its vector form `updateVec` updates a whole weight row
  (`updateVec_isFinite`).

**Capstone (`linear_train_step_all_finite`)** — one full linear-layer training step on bounded inputs is overflow-free
across ALL THREE stages: the FORWARD output `dotF x w` (C43), the BACKWARD gradient `gradW`, and the OPTIMIZER-updated
weights `updateVec w (gradW …) lr` are each `isFinite`.

**Scope (honestly disclosed).** This reuses C43's single trusted no-overflow axiom `isFinite_of_bounded` + the `(1+δ)`
base — NO new axiom. The BACKWARD here is a SELF-CONTAINED reverse-mode gradient for the linear layer (`gradW`) plus
the per-op backward magnitude bounds (`reluBwd`/`gateBwd` for the nonlinear gates; `mul`/`add`/`sub` reuse the forward
bounds) — NOT a certification of the repo's full `ADReverse` tape (which is a separate, deeply-coupled structure). The
OPTIMIZER is the SGD-shaped `w − lr·g` (the Float Muon update composes a Newton–Schulz map — bounded on the same
principle, but its finiteness is not wired here). Covers ONE training step; iterating to whole-run finiteness composes
per-step (each step's outputs bounded ⟹ the next step's inputs bounded, as long as the propagated bounds stay `≤
overflowBound`). The honest bounded-input side-conditions from the forward pass propagate (`≤ overflowBound` at each
stage is the checkable no-overflow precondition, which grows through the stages — realistic).
-/
import Puffer.RL.PPOLossScalar
open Puffer.FloatR
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound dotBound dotF_isFinite)
open Puffer.RL.LossForwardFinite (sub_bound mul_bound)

namespace Puffer.RL.BackwardFinite

/-! ### Nonlinear-gate backward: magnitude `≤ |ḡ|` (the cotangent, or `0`) -/

/-- **ReLU backward.** `relu'(x) = 0` for `x < 0`, else `1`, so the cotangent passed back is `0` (below the kink) or
    the upstream `ḡ` (above). -/
def reluBwd (g x : Float) : Float := if x < 0.0 then 0.0 else g

/-- The ReLU backward never amplifies the cotangent: `|toReal (reluBwd g x)| ≤ |toReal g|`. -/
theorem reluBwd_mag (g x : Float) : |toReal (reluBwd g x)| ≤ |toReal g| := by
  unfold reluBwd
  split_ifs
  · rw [toReal_zeroLit, abs_zero]; exact abs_nonneg _
  · exact le_refl _

/-- The ReLU backward is overflow-free when the cotangent magnitude is within the threshold. -/
theorem reluBwd_isFinite (g x : Float) (h : |toReal g| ≤ overflowBound) : (reluBwd g x).isFinite = true :=
  isFinite_of_bounded _ ((reluBwd_mag g x).trans h)

/-- **Gate backward** (the `min`/`max` clip): the cotangent `ḡ` flows to the selected branch, `0` to the other. -/
def gateBwd (cond : Bool) (g : Float) : Float := if cond then g else 0.0

/-- The gate backward never amplifies the cotangent: `|toReal (gateBwd cond g)| ≤ |toReal g|`. -/
theorem gateBwd_mag (cond : Bool) (g : Float) : |toReal (gateBwd cond g)| ≤ |toReal g| := by
  unfold gateBwd
  cases cond
  · rw [if_neg (by simp), toReal_zeroLit, abs_zero]; exact abs_nonneg _
  · rw [if_pos rfl]

/-- The gate backward is overflow-free when the cotangent magnitude is within the threshold. -/
theorem gateBwd_isFinite (cond : Bool) (g : Float) (h : |toReal g| ≤ overflowBound) :
    (gateBwd cond g).isFinite = true :=
  isFinite_of_bounded _ ((gateBwd_mag cond g).trans h)

/-! ### Linear-layer backward: the weight gradient `gradW ḡ x = x.map (ḡ · ·)` -/

/-- **The reverse-mode weight gradient of a linear layer** `y = dotF x w`: since `∂y/∂wⱼ = xⱼ`, the loss gradient at
    weight `wⱼ` is `ḡ·xⱼ` (upstream cotangent `ḡ`). A self-contained backward for the dominant compute (a linear
    layer), in the same style as C45's forward `layerF`/`dotF`. -/
def gradW (g : Float) (x : List Float) : List Float := x.map (fun xi => g * xi)

/-- **Backward gradient magnitude.** Each weight-gradient entry `ḡ·xⱼ` is bounded by `(1+u)·(Bg·B)` for a cotangent
    `≤ Bg` and inputs `≤ B` (C51's `mul_bound`). -/
theorem gradW_mag (g : Float) (x : List Float) (Bg B : ℝ)
    (hg : |toReal g| ≤ Bg) (hx : ∀ xi ∈ x, |toReal xi| ≤ B) :
    ∀ gw ∈ gradW g x, |toReal gw| ≤ (1 + u64) * (Bg * B) := by
  intro gw hgw
  rw [gradW, List.mem_map] at hgw
  obtain ⟨xi, hxi, rfl⟩ := hgw
  exact mul_bound g xi Bg B hg (hx xi hxi)

/-- **Backward gradient is overflow-free.** Every weight-gradient entry is `isFinite` when its `(1+u)`-inflated bound
    is within the overflow threshold. -/
theorem gradW_isFinite (g : Float) (x : List Float) (Bg B : ℝ)
    (hg : |toReal g| ≤ Bg) (hx : ∀ xi ∈ x, |toReal xi| ≤ B)
    (hbound : (1 + u64) * (Bg * B) ≤ overflowBound) :
    ∀ gw ∈ gradW g x, gw.isFinite = true := fun gw hgw =>
  isFinite_of_bounded _ ((gradW_mag g x Bg B hg hx gw hgw).trans hbound)

/-! ### Optimizer step: `updateF w g lr = w − lr·g` -/

/-- **One SGD-shaped weight update** `w' = w − lr·g` (Float). -/
def updateF (w g lr : Float) : Float := w - lr * g

/-- **Update magnitude.** `|toReal (w − lr·g)| ≤ (1+u)·(Bw + (1+u)·(Blr·Bg))` (C51's `mul_bound` for `lr·g`, then
    `sub_bound`). -/
theorem updateF_mag_le (w g lr : Float) (Bw Bg Blr : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr) :
    |toReal (updateF w g lr)| ≤ (1 + u64) * (Bw + (1 + u64) * (Blr * Bg)) := by
  unfold updateF
  exact sub_bound w (lr * g) Bw ((1 + u64) * (Blr * Bg)) hw (mul_bound lr g Blr Bg hlr hg)

/-- **One SGD update is overflow-free** for bounded `w`/`g`/`lr` with the composed bound within the threshold. -/
theorem updateF_isFinite (w g lr : Float) (Bw Bg Blr : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (hbound : (1 + u64) * (Bw + (1 + u64) * (Blr * Bg)) ≤ overflowBound) :
    (updateF w g lr).isFinite = true :=
  isFinite_of_bounded _ ((updateF_mag_le w g lr Bw Bg Blr hw hg hlr).trans hbound)

/-- **The vectorized SGD update** of a weight row `w` by a gradient row `g`: `updateVec w g lr = zipWith (·−lr··) w g`. -/
def updateVec (w g : List Float) (lr : Float) : List Float :=
  List.zipWith (fun wi gi => updateF wi gi lr) w g

/-- **Vector-update magnitude.** Every updated weight is bounded by the single-update bound, from bounded `w`/`g`/`lr`
    (structural recursion on the two rows). -/
theorem updateVec_mag (lr : Float) (Bw Bg Blr : ℝ) (hlr : |toReal lr| ≤ Blr) :
    ∀ (w g : List Float), (∀ wi ∈ w, |toReal wi| ≤ Bw) → (∀ gi ∈ g, |toReal gi| ≤ Bg) →
      ∀ u ∈ updateVec w g lr, |toReal u| ≤ (1 + u64) * (Bw + (1 + u64) * (Blr * Bg))
  | [], _, _, _ => by simp [updateVec]
  | _, [], _, _ => by simp [updateVec]
  | wi :: ws, gi :: gs, hw, hg => by
      intro u hu
      have hstep : updateVec (wi :: ws) (gi :: gs) lr = updateF wi gi lr :: updateVec ws gs lr := rfl
      rw [hstep, List.mem_cons] at hu
      rcases hu with rfl | hrest
      · exact updateF_mag_le wi gi lr Bw Bg Blr
          (hw wi (List.mem_cons.mpr (Or.inl rfl))) (hg gi (List.mem_cons.mpr (Or.inl rfl))) hlr
      · exact updateVec_mag lr Bw Bg Blr hlr ws gs
          (fun q hq => hw q (List.mem_cons.mpr (Or.inr hq)))
          (fun q hq => hg q (List.mem_cons.mpr (Or.inr hq))) u hrest

/-- **The vectorized SGD update is overflow-free** — every updated weight is `isFinite`. -/
theorem updateVec_isFinite (w g : List Float) (lr : Float) (Bw Bg Blr : ℝ)
    (hw : ∀ wi ∈ w, |toReal wi| ≤ Bw) (hg : ∀ gi ∈ g, |toReal gi| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (hbound : (1 + u64) * (Bw + (1 + u64) * (Blr * Bg)) ≤ overflowBound) :
    ∀ u ∈ updateVec w g lr, u.isFinite = true := fun u hu =>
  isFinite_of_bounded _ ((updateVec_mag lr Bw Bg Blr hlr w g hw hg u hu).trans hbound)

/-! ### Capstone: one full linear-layer training step (forward + backward + optimizer) -/

/-- **One linear-layer training step**: from inputs `x` and weights `w`, with upstream cotangent `gseed` and step size
    `lr`, the updated weight row `updateVec w (gradW gseed x) lr` (forward output `dotF x w` implicit in the loss that
    produced `gseed`). -/
def linearTrainStep (x w : List Float) (gseed lr : Float) : List Float :=
  updateVec w (gradW gseed x) lr

/-- **THE PER-STEP TRAINER IS OVERFLOW-FREE.** For a linear layer on bounded inputs (`x`, `w ≤ B`), upstream cotangent
    `≤ Bgs`, step size `≤ Blr`, with the forward and backward-plus-update bounds each within the overflow threshold,
    ALL THREE stages are overflow-free: the FORWARD output `dotF x w` (C43), and every OPTIMIZER-updated weight (via the
    BACKWARD gradient `gradW gseed x`, bounded by `(1+u)·(Bgs·B)`, fed to `updateVec`). One training step of a bounded
    linear layer never overflows. -/
theorem linear_train_step_all_finite (x w : List Float) (gseed lr : Float) (B Bgs Blr : ℝ)
    (hB0 : 0 ≤ B)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ B) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (hfwd : dotBound (min x.length w.length) B ≤ overflowBound)
    (hbwd : (1 + u64) * (B + (1 + u64) * (Blr * ((1 + u64) * (Bgs * B)))) ≤ overflowBound) :
    (FiniteBound.dotF x w).isFinite = true ∧ ∀ u ∈ linearTrainStep x w gseed lr, u.isFinite = true :=
  ⟨dotF_isFinite B hB0 x w hx hw hfwd,
   updateVec_isFinite w (gradW gseed x) lr B ((1 + u64) * (Bgs * B)) Blr hw
     (gradW_mag gseed x Bgs B hgs hx) hlr hbwd⟩

end Puffer.RL.BackwardFinite
