/-
# Concrete entropy gradient-Lipschitz from uniform per-log-prob budgets

C15 (`EntropyGradExpr`) proved the categorical entropy's gradient-Lipschitz in two pieces: the per-term bound
`expLogTerm_lip` (each `exp(lp)·lp` term, given `lp`'s value/derivative magnitude + Lipschitz budgets `M`/`Lv`/
`Dm`/`Dl`) and the sum assembly `entropyCatE_gradient_lipschitz` (a UNIFORM per-term bound `C` ⟹ `n·C`). This
module composes them: given UNIFORM per-log-prob budgets shared by every log-prob, the entropy
`entropyCatE logps = −Σᵢ exp(logpᵢ)·logpᵢ` is gradient-Lipschitz with the concrete constant

    |Δ∇(entropyCatE logps)|  ≤  (length logps) · exp(M)·((M+1)·Dl + (M+2)·Dm·Lv) · δ.

* `entropyCatE_gradient_lipschitz_uniform` — over a `List` of log-probs, each satisfying the same budgets.
* `entropyCatE_ofFn_gradient_lipschitz` — the softmax-over-`n`-classes shape: log-probs given as a vector
  `logp : Fin n → Expr`, the membership hypotheses reduced to clean per-index budgets (via `List.ofFn`).

**Scope (honestly disclosed):** the uniform budgets `M`/`Lv`/`Dm`/`Dl` are HYPOTHESES here — they hold uniformly
for a softmax's log-probs because all `logpᵢ = logSoftmaxE (logitᵢ) logits` share the SAME partition (only the
chosen logit varies), so C17 (`LogSoftmaxBudgetExpr`, the derivative budgets `Dm`/`Dl`) and C18
(`LogSoftmaxValueBudgetExpr`, the value budgets `M`/`Lv`) discharge them per log-prob; making them uniform ACROSS
the log-probs needs a common bound on the per-logit budgets (`vMag`/`vLip`/`dMag`/`dLip R (logitᵢ)`) — e.g. a max
over the `n` logits — which is a mechanical (but fiddly) instantiation left to the caller. The per-index form
here is exactly what that instantiation feeds. The uniform-`C` restriction comes from C15's
`entropyCatE_gradient_lipschitz`; a non-uniform (sum-of-per-term) variant would use C15's `crossTermE_grad_diff_le`
directly.
-/
import Puffer.RL.EntropyGradExpr

open Puffer.FloatR.ADR
open Puffer.RL.EntropyGradExpr
open Puffer.RL.ValueEntropyExpr (entropyCatE)

namespace Puffer.RL.EntropyConcreteExpr

/-- **Concrete entropy gradient-Lipschitz (uniform budgets over a list).** If every log-prob `lp ∈ logps` shares
    the budgets `M` (value cap), `Lv·δ` (value Lipschitz), `Dm` (derivative magnitude), `Dl·δ` (derivative
    Lipschitz), then `|derivR (entropyCatE logps) σ k − derivR (entropyCatE logps) σ' k| ≤ (length logps)·exp(M)·
    ((M+1)·Dl + (M+2)·Dm·Lv)·δ`. Composes C15's per-term `expLogTerm_lip` (with the uniform per-term constant `C =
    exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ`) through C15's sum assembly `entropyCatE_gradient_lipschitz`. -/
theorem entropyCatE_gradient_lipschitz_uniform (logps : List Expr) (σ σ' : Nat → ℝ) (k : Nat)
    (M Lv Dm Dl δ : ℝ)
    (hMσ : ∀ lp ∈ logps, |evalR lp σ| ≤ M) (hMσ' : ∀ lp ∈ logps, |evalR lp σ'| ≤ M)
    (hLv : ∀ lp ∈ logps, |evalR lp σ - evalR lp σ'| ≤ Lv * δ)
    (hDmσ : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dm) (hDmσ' : ∀ lp ∈ logps, |derivR lp σ' k| ≤ Dm)
    (hDl : ∀ lp ∈ logps, |derivR lp σ k - derivR lp σ' k| ≤ Dl * δ) (hDmn : 0 ≤ Dm) :
    |derivR (entropyCatE logps) σ k - derivR (entropyCatE logps) σ' k|
      ≤ (logps.length : ℝ) * (Real.exp M * ((M + 1) * Dl + (M + 2) * Dm * Lv) * δ) := by
  apply entropyCatE_gradient_lipschitz
  intro lp hlp
  exact expLogTerm_lip lp σ σ' k M Lv Dm Dl δ (hMσ lp hlp) (hMσ' lp hlp) (hLv lp hlp)
    (hDmσ lp hlp) (hDmσ' lp hlp) (hDl lp hlp) hDmn

/-- **Softmax-over-`n`-classes form.** The log-probs given as a vector `logp : Fin n → Expr` (the natural softmax
    shape), with the budgets stated per index `i`. The entropy over the `n` log-probs is gradient-Lipschitz with
    `n · exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ`. A direct specialization of `entropyCatE_gradient_lipschitz_uniform`
    to `List.ofFn logp` (membership hypotheses reduced via `List.forall_mem_ofFn_iff`, length via
    `List.length_ofFn`). -/
theorem entropyCatE_ofFn_gradient_lipschitz {n : Nat} (logp : Fin n → Expr) (σ σ' : Nat → ℝ) (k : Nat)
    (M Lv Dm Dl δ : ℝ)
    (hMσ : ∀ i, |evalR (logp i) σ| ≤ M) (hMσ' : ∀ i, |evalR (logp i) σ'| ≤ M)
    (hLv : ∀ i, |evalR (logp i) σ - evalR (logp i) σ'| ≤ Lv * δ)
    (hDmσ : ∀ i, |derivR (logp i) σ k| ≤ Dm) (hDmσ' : ∀ i, |derivR (logp i) σ' k| ≤ Dm)
    (hDl : ∀ i, |derivR (logp i) σ k - derivR (logp i) σ' k| ≤ Dl * δ) (hDmn : 0 ≤ Dm) :
    |derivR (entropyCatE (List.ofFn logp)) σ k - derivR (entropyCatE (List.ofFn logp)) σ' k|
      ≤ (n : ℝ) * (Real.exp M * ((M + 1) * Dl + (M + 2) * Dm * Lv) * δ) := by
  have h := entropyCatE_gradient_lipschitz_uniform (List.ofFn logp) σ σ' k M Lv Dm Dl δ
    (List.forall_mem_ofFn_iff.mpr hMσ) (List.forall_mem_ofFn_iff.mpr hMσ')
    (List.forall_mem_ofFn_iff.mpr hLv) (List.forall_mem_ofFn_iff.mpr hDmσ)
    (List.forall_mem_ofFn_iff.mpr hDmσ') (List.forall_mem_ofFn_iff.mpr hDl) hDmn
  rwa [List.length_ofFn] at h

end Puffer.RL.EntropyConcreteExpr
