/-
# Muon-update finiteness: the Float Newton–Schulz circuit in the overflow-free certificate chain

C57 (`BackwardFinite`) certified one linear-layer training step overflow-free with the SGD-shaped update
`w − lr·g`, and disclosed that the Float MUON update — which first passes the gradient through the Newton–Schulz
map — was not wired. This module wires it: the Muon-shaped update is `w ← w − lr·nsScalarF(g)`, where
`nsScalarF a b c σ = σ·(a + b·σ² + c·σ⁴)` (`Puffer/Float/Exec.lean`) is the runnable Float NS scalar circuit —
the per-singular-value model of the NS orthogonalization, the same representation whose ℝ-error is bounded by
`MuonRuntime.nsScalarF_error` and whose spectral story runs through `MuonScalarBound`/C44/C56.

* `nsScalarFBound` / `nsScalarF_mag_le` / `nsScalarF_isFinite` — the circuit's BUDGET-style magnitude bound
  (each of the 7 Float ops inflating by `(1+u64)`, mirroring the op tree exactly, in C43/C51/C54's style —
  unlike `MuonRuntime.nsScalarF_bound`, whose bound is data-dependent) and the overflow-free certificate.
* `nsIterF` / `nsIterFBound` / `nsIterF_mag_le` / `nsIterF_isFinite` — the `k`-fold iterated circuit (the `k`
  NS iterations of one Muon step, in the scalar per-singular-value model), with the CRUDE iterated bound: each
  iteration's budget feeds the next. The bound GROWS through iterations — honestly: C56's `[0,1]` invariant
  interval holds for EXACT arithmetic; the Float circuit adds `(1+δ)` inflation per op, and transporting the
  invariant interval through those errors is a separate Float-error analysis (`nsScalarF_error`), not done here.
* `muonUpdateF` / `muonUpdateVec` (+ `_mag`/`_isFinite`) — the Muon-shaped update `w − lr·nsScalarF(g)`, scalar
  and vector forms, by direct reuse of C57's `updateF`/`updateVec` at the NS-transformed gradient.
* `muon_train_step_all_finite` (capstone) — one linear-layer training step with the MUON-shaped update is
  overflow-free across all three stages: the FORWARD `dotF x w` (C43), the BACKWARD gradient `gradW` (C57,
  bounded), and every MUON-updated weight (the gradient passed through `nsScalarF`, then `w − lr·…`).

**Scope (honestly disclosed).** Certified here is the SCALAR NS circuit `nsScalarF` (the per-singular-value /
per-coordinate model of the Muon orthogonalization — the representation the repo's runtime error analysis
`MuonRuntime` uses) and its `k`-fold iterate, composed into the update. NOT covered: the full MATRIX Newton–Schulz
(the matrix products `X·XᵀX` per iteration — each row is a `dotF`, so C43's machinery applies per product, but the
full k-iteration matrix wiring is not assembled here); the repo's full `ADReverse` tape (as in C57 — the backward
here is C57's self-contained linear-layer `gradW`); and whole-run composition (per-step bounds compose while they
stay `≤ overflowBound`, as in C57). Reuses C43's single trusted no-overflow axiom `isFinite_of_bounded` + the
`(1+δ)` base — NO new axiom. The iterated bound is the crude one (grows with `k`); the honest bounded-input
side-conditions (`≤ overflowBound` at each stage) are the checkable no-overflow preconditions.
-/
import Puffer.RL.BackwardFinite
open Puffer.FloatR
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound dotBound dotF_isFinite)
open Puffer.RL.LossForwardFinite (mul_bound)
open Puffer.RL.PPOLossScalar (add_bound)
open Puffer.RL.BackwardFinite (updateF updateF_mag_le updateVec updateVec_mag updateVec_isFinite
  gradW gradW_mag)

namespace Puffer.RL.MuonUpdateFinite

/-! ### The Float NS scalar circuit: budget magnitude + finiteness -/

/-- **Budget magnitude bound for one Float NS scalar circuit** `nsScalarF a b c σ = σ·(a + b·(σ·σ) +
    c·((σ·σ)·(σ·σ)))`: one `(1+u64)` factor per Float op (4 muls inside the polynomial, 2 adds, 1 outer mul),
    mirroring the op tree exactly — coefficients bounded by `Ba`/`Bb`/`Bc`, input by `Bσ`. -/
noncomputable def nsScalarFBound (Ba Bb Bc Bσ : ℝ) : ℝ :=
  (1 + u64) * (Bσ * ((1 + u64) *
    ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
      + (1 + u64) * (Bc * ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))))))))

/-- **The Float NS scalar circuit is magnitude-bounded** by `nsScalarFBound` for bounded coefficients and input —
    composing C51's `mul_bound` and C54's `add_bound` through the circuit's 7 ops, bottom-up. -/
theorem nsScalarF_mag_le (a b c σ : Float) (Ba Bb Bc Bσ : ℝ)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) (hσ : |toReal σ| ≤ Bσ) :
    |toReal (nsScalarF a b c σ)| ≤ nsScalarFBound Ba Bb Bc Bσ := by
  have hS2 := mul_bound σ σ Bσ Bσ hσ hσ
  have hTb := mul_bound b (σ * σ) Bb ((1 + u64) * (Bσ * Bσ)) hb hS2
  have hTab := add_bound a (b * (σ * σ)) Ba ((1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ)))) ha hTb
  have hS4 := mul_bound (σ * σ) (σ * σ) ((1 + u64) * (Bσ * Bσ)) ((1 + u64) * (Bσ * Bσ)) hS2 hS2
  have hTc := mul_bound c ((σ * σ) * (σ * σ)) Bc
    ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))) hc hS4
  have hT := add_bound (a + b * (σ * σ)) (c * ((σ * σ) * (σ * σ))) _ _ hTab hTc
  have hfin := mul_bound σ (a + b * (σ * σ) + c * ((σ * σ) * (σ * σ))) Bσ _ hσ hT
  unfold nsScalarF nsScalarFBound
  exact hfin

/-- **The Float NS scalar circuit is overflow-free** when its budget bound is within the threshold. -/
theorem nsScalarF_isFinite (a b c σ : Float) (Ba Bb Bc Bσ : ℝ)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) (hσ : |toReal σ| ≤ Bσ)
    (hbound : nsScalarFBound Ba Bb Bc Bσ ≤ overflowBound) :
    (nsScalarF a b c σ).isFinite = true :=
  isFinite_of_bounded _ ((nsScalarF_mag_le a b c σ Ba Bb Bc Bσ ha hb hc hσ).trans hbound)

/-! ### The k-fold iterated circuit (the k NS iterations of one Muon step, scalar model) -/

/-- The `k`-fold Float NS scalar iterate: `k` applications of `nsScalarF a b c`. -/
def nsIterF (a b c : Float) : Nat → Float → Float
  | 0, σ => σ
  | k + 1, σ => nsScalarF a b c (nsIterF a b c k σ)

/-- The CRUDE iterated budget: each iteration's `nsScalarFBound` output feeds the next iteration's input budget.
    Grows with `k` (honest — the exact-arithmetic `[0,1]` invariance of C56 is not transported through the
    Float `(1+δ)` errors here). -/
noncomputable def nsIterFBound (Ba Bb Bc : ℝ) : Nat → ℝ → ℝ
  | 0, Bσ => Bσ
  | k + 1, Bσ => nsScalarFBound Ba Bb Bc (nsIterFBound Ba Bb Bc k Bσ)

/-- **The k-fold iterate is magnitude-bounded** by the iterated budget (induction over `k`, each step
    `nsScalarF_mag_le` at the previous iterate's budget). -/
theorem nsIterF_mag_le (a b c : Float) (Ba Bb Bc : ℝ)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ (k : Nat) (σ : Float) (Bσ : ℝ), |toReal σ| ≤ Bσ →
      |toReal (nsIterF a b c k σ)| ≤ nsIterFBound Ba Bb Bc k Bσ
  | 0, _, _, hσ => hσ
  | k + 1, σ, Bσ, hσ =>
      nsScalarF_mag_le a b c (nsIterF a b c k σ) Ba Bb Bc (nsIterFBound Ba Bb Bc k Bσ)
        ha hb hc (nsIterF_mag_le a b c Ba Bb Bc ha hb hc k σ Bσ hσ)

/-- **The k-fold iterate is overflow-free** when the (final) iterated budget is within the threshold. -/
theorem nsIterF_isFinite (a b c σ : Float) (k : Nat) (Ba Bb Bc Bσ : ℝ)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) (hσ : |toReal σ| ≤ Bσ)
    (hbound : nsIterFBound Ba Bb Bc k Bσ ≤ overflowBound) :
    (nsIterF a b c k σ).isFinite = true :=
  isFinite_of_bounded _ ((nsIterF_mag_le a b c Ba Bb Bc ha hb hc k σ Bσ hσ).trans hbound)

/-! ### The Muon-shaped update: `w − lr·nsScalarF(g)` -/

/-- The composed Muon-update budget: C57's update bound with the gradient budget replaced by the NS circuit's
    output budget `nsScalarFBound Ba Bb Bc Bg`. -/
noncomputable def muonUpdateBound (Bw Blr Ba Bb Bc Bg : ℝ) : ℝ :=
  (1 + u64) * (Bw + (1 + u64) * (Blr * nsScalarFBound Ba Bb Bc Bg))

/-- **One Muon-shaped weight update**: the gradient is first orthogonalized by the NS scalar circuit, then the
    SGD-shaped step applies — `w − lr·nsScalarF(g)` (= C57's `updateF` at the NS-transformed gradient). -/
def muonUpdateF (w g lr a b c : Float) : Float := updateF w (nsScalarF a b c g) lr

/-- **Muon-update magnitude**: C57's `updateF_mag_le` at the NS-transformed gradient's budget. -/
theorem muonUpdateF_mag_le (w g lr a b c : Float) (Bw Bg Blr Ba Bb Bc : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    |toReal (muonUpdateF w g lr a b c)| ≤ muonUpdateBound Bw Blr Ba Bb Bc Bg :=
  updateF_mag_le w (nsScalarF a b c g) lr Bw (nsScalarFBound Ba Bb Bc Bg) Blr hw
    (nsScalarF_mag_le a b c g Ba Bb Bc Bg ha hb hc hg) hlr

/-- **One Muon-shaped update is overflow-free** for bounded weight/gradient/step/coefficients. -/
theorem muonUpdateF_isFinite (w g lr a b c : Float) (Bw Bg Blr Ba Bb Bc : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hbound : muonUpdateBound Bw Blr Ba Bb Bc Bg ≤ overflowBound) :
    (muonUpdateF w g lr a b c).isFinite = true :=
  isFinite_of_bounded _
    ((muonUpdateF_mag_le w g lr a b c Bw Bg Blr Ba Bb Bc hw hg hlr ha hb hc).trans hbound)

/-- **The vectorized Muon-shaped update**: each gradient entry passes through the NS circuit, then the row updates
    — by construction C57's `updateVec` at the mapped gradient row. -/
def muonUpdateVec (w g : List Float) (lr a b c : Float) : List Float :=
  updateVec w (g.map (nsScalarF a b c)) lr

/-- Every NS-transformed gradient entry is bounded by the circuit budget. -/
theorem mapNS_mag (g : List Float) (a b c : Float) (Ba Bb Bc Bg : ℝ)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hg : ∀ gi ∈ g, |toReal gi| ≤ Bg) :
    ∀ gi' ∈ g.map (nsScalarF a b c), |toReal gi'| ≤ nsScalarFBound Ba Bb Bc Bg := by
  intro gi' hgi'
  rw [List.mem_map] at hgi'
  obtain ⟨gi, hgi, rfl⟩ := hgi'
  exact nsScalarF_mag_le a b c gi Ba Bb Bc Bg ha hb hc (hg gi hgi)

/-- **Vector Muon-update magnitude** — every updated weight bounded by the composed Muon-update budget. -/
theorem muonUpdateVec_mag (w g : List Float) (lr a b c : Float) (Bw Bg Blr Ba Bb Bc : ℝ)
    (hw : ∀ wi ∈ w, |toReal wi| ≤ Bw) (hg : ∀ gi ∈ g, |toReal gi| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ u ∈ muonUpdateVec w g lr a b c, |toReal u| ≤ muonUpdateBound Bw Blr Ba Bb Bc Bg :=
  fun u hu =>
    updateVec_mag lr Bw (nsScalarFBound Ba Bb Bc Bg) Blr hlr w (g.map (nsScalarF a b c)) hw
      (mapNS_mag g a b c Ba Bb Bc Bg ha hb hc hg) u hu

/-- **The vectorized Muon-shaped update is overflow-free** — every updated weight is `isFinite`. -/
theorem muonUpdateVec_isFinite (w g : List Float) (lr a b c : Float) (Bw Bg Blr Ba Bb Bc : ℝ)
    (hw : ∀ wi ∈ w, |toReal wi| ≤ Bw) (hg : ∀ gi ∈ g, |toReal gi| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hbound : muonUpdateBound Bw Blr Ba Bb Bc Bg ≤ overflowBound) :
    ∀ u ∈ muonUpdateVec w g lr a b c, u.isFinite = true := fun u hu =>
  isFinite_of_bounded _
    ((muonUpdateVec_mag w g lr a b c Bw Bg Blr Ba Bb Bc hw hg hlr ha hb hc u hu).trans hbound)

/-! ### Capstone: one full training step with the Muon-shaped update -/

/-- **One linear-layer training step with the Muon-shaped update**: backward gradient `gradW gseed x` (C57),
    each entry orthogonalized by the NS circuit, then the weight row updates. -/
def muonTrainStep (x w : List Float) (gseed lr a b c : Float) : List Float :=
  muonUpdateVec w (gradW gseed x) lr a b c

/-- **THE PER-STEP MUON TRAINER IS OVERFLOW-FREE.** For a linear layer on bounded inputs (`x`, `w ≤ B`), upstream
    cotangent `≤ Bgs`, step size `≤ Blr`, NS coefficients `≤ Ba`/`Bb`/`Bc`, with the forward and the
    backward-through-NS-update bounds each within the overflow threshold, all stages are overflow-free: the
    FORWARD `dotF x w` (C43) and every MUON-updated weight — the BACKWARD gradient `gradW gseed x` (bounded by
    `(1+u)·(Bgs·B)`, C57) passed through the NS circuit and the `w − lr·…` update. C57's capstone with the SGD
    update replaced by the Muon-shaped one. -/
theorem muon_train_step_all_finite (x w : List Float) (gseed lr a b c : Float)
    (B Bgs Blr Ba Bb Bc : ℝ) (hB0 : 0 ≤ B)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ B) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hfwd : dotBound (min x.length w.length) B ≤ overflowBound)
    (hbwd : muonUpdateBound B Blr Ba Bb Bc ((1 + u64) * (Bgs * B)) ≤ overflowBound) :
    (FiniteBound.dotF x w).isFinite = true
      ∧ ∀ u ∈ muonTrainStep x w gseed lr a b c, u.isFinite = true :=
  ⟨dotF_isFinite B hB0 x w hx hw hfwd,
   muonUpdateVec_isFinite w (gradW gseed x) lr a b c B ((1 + u64) * (Bgs * B)) Blr Ba Bb Bc
     hw (gradW_mag gseed x Bgs B hgs hx) hlr ha hb hc hbwd⟩

end Puffer.RL.MuonUpdateFinite
