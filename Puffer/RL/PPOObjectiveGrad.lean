/-
# PPO total-objective gradient-Lipschitz assembly

The gradient-side companion to C13's value assembly (`PPOObjectiveExpr.evalR_ppoTotalObjE_concrete`): through
the `add`/`sub`/`scale` linear structure of `ppoTotalObjE surr valSq ent cv ce = add (sub surr (scale cv valSq))
(scale ce ent)`, the per-term gradient-Lipschitz constants combine — by triangle inequality — into the total
objective's:

    |Δ∇(ppoTotalObjE)| ≤ Lsurr + |toReal cv|·LvalSq + |toReal ce|·Lent

where `Lsurr`, `LvalSq`, `Lent` bound the per-term gradient variations. The gradient is exactly linear in the
term gradients (`derivR_ppoTotalObjE`), so the assembly is an exact triangle bound with the `scale` coefficients
`|toReal cv|`, `|toReal ce|` weighting the value and entropy contributions.

* `ppoTotalObjE_gradient_lipschitz` — the general assembly (per-term bounds ↦ total bound).
* `ppoTotalObjE_gradient_lipschitz_value` — the value term DISCHARGED via C4: since `valueSqErrE V ret` is
  `Smooth` (C12), its gradient-Lipschitz is the CONCRETE `dLip R (valueSqErrE V ret)·δ` (no floor/active
  hypothesis), so the value contributes exactly `|toReal cv|·dLip R (valueSqErrE V ret)·δ`.

**Scope (honestly disclosed):** this is the STRUCTURAL assembly — it reduces the total objective's
gradient-Lipschitz to the three per-term bounds. The value term is fully discharged here (C4, unconditional).
The surrogate term's bound is C8's clip-interior reduction (`ppoSurrogateE_interior` ⟹ `|toReal g|·(ratio's
Lipschitz)` on the unclipped region — off the interior the clip kink is non-Lipschitz) and the entropy term's is
C5 (`exp`) + C6 (`log`, floor from C9) — both remain as the `Lsurr`/`Lent` hypotheses (their concrete discharge
is separate). So this cleanly isolates exactly what is proven (the linear assembly + value term) from what each
remaining term needs.
-/
import Puffer.RL.PPOObjectiveExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.PPOObjectiveExpr
open Puffer.RL.ValueEntropyExpr (valueSqErrE valueSqErrE_gradient_lipschitz)

namespace Puffer.RL.PPOObjectiveGrad

/-- Structural gradient of the assembled objective: linear in the term gradients (`add`/`sub`/`scale`). -/
theorem derivR_ppoTotalObjE (surr valSq ent : Expr) (cv ce : Float) (σ : Nat → ℝ) (k : Nat) :
    derivR (ppoTotalObjE surr valSq ent cv ce) σ k
      = derivR surr σ k - toReal cv * derivR valSq σ k + toReal ce * derivR ent σ k := by
  simp only [ppoTotalObjE, derivR]

/-- **Total-objective gradient-Lipschitz assembly.** Given per-term gradient-variation bounds `Lsurr`, `LvalSq`,
    `Lent` for the surrogate, value, and entropy sub-`Expr`s, the assembled objective's gradient variation is
    bounded by `Lsurr + |toReal cv|·LvalSq + |toReal ce|·Lent` — an exact triangle bound over its linear
    `add`/`sub`/`scale` structure, with the `scale` coefficients weighting the value/entropy terms. -/
theorem ppoTotalObjE_gradient_lipschitz (surr valSq ent : Expr) (cv ce : Float)
    (σ σ' : Nat → ℝ) (k : Nat) (Lsurr LvalSq Lent : ℝ)
    (hsurr : |derivR surr σ k - derivR surr σ' k| ≤ Lsurr)
    (hval : |derivR valSq σ k - derivR valSq σ' k| ≤ LvalSq)
    (hent : |derivR ent σ k - derivR ent σ' k| ≤ Lent) :
    |derivR (ppoTotalObjE surr valSq ent cv ce) σ k - derivR (ppoTotalObjE surr valSq ent cv ce) σ' k|
      ≤ Lsurr + |toReal cv| * LvalSq + |toReal ce| * Lent := by
  rw [derivR_ppoTotalObjE, derivR_ppoTotalObjE]
  have key : (derivR surr σ k - toReal cv * derivR valSq σ k + toReal ce * derivR ent σ k)
        - (derivR surr σ' k - toReal cv * derivR valSq σ' k + toReal ce * derivR ent σ' k)
      = (derivR surr σ k - derivR surr σ' k)
        + (-(toReal cv * (derivR valSq σ k - derivR valSq σ' k))
           + toReal ce * (derivR ent σ k - derivR ent σ' k)) := by ring
  rw [key]
  have hcv : |toReal cv| * |derivR valSq σ k - derivR valSq σ' k| ≤ |toReal cv| * LvalSq :=
    mul_le_mul_of_nonneg_left hval (abs_nonneg _)
  have hce : |toReal ce| * |derivR ent σ k - derivR ent σ' k| ≤ |toReal ce| * Lent :=
    mul_le_mul_of_nonneg_left hent (abs_nonneg _)
  calc |(derivR surr σ k - derivR surr σ' k)
          + (-(toReal cv * (derivR valSq σ k - derivR valSq σ' k))
             + toReal ce * (derivR ent σ k - derivR ent σ' k))|
      ≤ |derivR surr σ k - derivR surr σ' k|
          + |-(toReal cv * (derivR valSq σ k - derivR valSq σ' k))
             + toReal ce * (derivR ent σ k - derivR ent σ' k)| := abs_add_le _ _
    _ ≤ |derivR surr σ k - derivR surr σ' k|
          + (|toReal cv * (derivR valSq σ k - derivR valSq σ' k)|
             + |toReal ce * (derivR ent σ k - derivR ent σ' k)|) := by
        gcongr
        exact (abs_add_le _ _).trans_eq (by rw [abs_neg])
    _ = |derivR surr σ k - derivR surr σ' k|
          + (|toReal cv| * |derivR valSq σ k - derivR valSq σ' k|
             + |toReal ce| * |derivR ent σ k - derivR ent σ' k|) := by rw [abs_mul, abs_mul]
    _ ≤ Lsurr + |toReal cv| * LvalSq + |toReal ce| * Lent := by linarith [hsurr, hcv, hce]

/-- **Value term discharged via C4.** With the value head `V` `Smooth`, the value squared error `valueSqErrE V
    ret` has the CONCRETE gradient-Lipschitz `dLip R (valueSqErrE V ret)·δ` (C12/C4, no floor/active hypothesis),
    so the total-objective bound becomes `Lsurr + |toReal cv|·(dLip R (valueSqErrE V ret)·δ) + |toReal ce|·Lent`
    — the value term is fully concrete; only the surrogate (C8 interior) and entropy (C5+C6) bounds remain
    hypotheses. -/
theorem ppoTotalObjE_gradient_lipschitz_value (surr ent V : Expr) (cv ce ret : Float) (hV : Smooth V)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (Lsurr Lent : ℝ)
    (hsurr : |derivR surr σ k - derivR surr σ' k| ≤ Lsurr)
    (hent : |derivR ent σ k - derivR ent σ' k| ≤ Lent) :
    |derivR (ppoTotalObjE surr (valueSqErrE V ret) ent cv ce) σ k
        - derivR (ppoTotalObjE surr (valueSqErrE V ret) ent cv ce) σ' k|
      ≤ Lsurr + |toReal cv| * (dLip R (valueSqErrE V ret) * δ) + |toReal ce| * Lent :=
  ppoTotalObjE_gradient_lipschitz surr (valueSqErrE V ret) ent cv ce σ σ' k Lsurr _ Lent
    hsurr (valueSqErrE_gradient_lipschitz V ret hV σ σ' R δ k hσ hσ' hδ hR) hent

end Puffer.RL.PPOObjectiveGrad
