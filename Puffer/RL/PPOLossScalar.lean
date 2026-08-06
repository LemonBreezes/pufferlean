/-
# The scalar PPO loss: assembling C51's forward loss terms into one overflow-free value

C51 (`LossForwardFinite`) certified the individual PPO loss FORWARD terms overflow-free — the clipped surrogate
`ppoSurrF`, the value squared error `valSqF`, the ratio, the softmax/log-partition — but explicitly left "assemble
the terms into the single scalar loss value" as remaining. This module performs that assembly: mirroring
`ppoTotalObjE surr valSq ent cv ce = add (sub surr (scale cv valSq)) (scale ce ent)`, it composes the (bounded,
finite) Float loss-term values into the scalar total loss `ppoLossF = (surr − cv·valSq) + ce·ent` and certifies the
assembled scalar overflow-free, reusing C43's `isFinite_of_bounded` + the `(1+δ)` base — **NO new axiom**.

* `add_bound` — the addition magnitude-bound helper (sibling of C51's `sub_bound`/`mul_bound`), `(1+u)` growth.
* `ppoLossF` — the scalar PPO total loss `(surr − cv·valSq) + ce·ent` (Float ops, mirroring `ppoTotalObjE`).
* `lossBound` / `ppoLossF_mag_le` — the assembled scalar magnitude bound, composing `mul_bound` (the `cv`/`ce`
  weightings), `sub_bound`, and `add_bound` over the `add`/`sub`/`mul` structure (each op inflating by `(1+u)`).
* `ppoLossF_isFinite` — the scalar PPO loss is overflow-free (`ppoLossF_mag_le` + `isFinite_of_bounded`).
* `ppoLossF_isFinite_of_terms` — the end-to-end certificate: the surrogate and value terms are the concrete C51
  `ppoSurrF`/`valSqF` discharged from their own inputs (via `ppoSurrF_mag_le`/`valSqF_mag_le`), the entropy term a
  bounded value (`|ent| ≤ Be`, the network's entropy budget) — so the assembled scalar loss is overflow-free directly
  from the network/rollout budgets.

**Scope (honestly disclosed).** This assembles C51's PPO loss forward terms into the scalar total loss and certifies
it overflow-free, reusing C43's `isFinite_of_bounded` + the `(1+δ)` base — NO new axiom. It covers the FORWARD loss
scalar (`surrogate − cv·value + ce·entropy`). The entropy term's value bound `Be` is supplied (dischargeable from the
network's entropy magnitude budget, as in C24/C31/C37). The honest side-conditions from C51 propagate (logits bounded
above for the ratio/partition, the partition floored for the log, every operand magnitude-bounded). It does NOT cover
the BACKWARD / AD pass or the OPTIMIZER — a full-trainer finiteness composes those next. The assembled magnitude bound
grows by a `(1+u)` factor per `add`/`sub`/`mul` op (realistic); `lossBound … ≤ overflowBound` is the checkable
no-overflow condition.
-/
import Puffer.RL.LossForwardFinite
open Puffer.FloatR
open Puffer.RL.FiniteBound
open Puffer.RL.LossFinite
open Puffer.RL.LossForwardFinite

namespace Puffer.RL.PPOLossScalar

/-- `0 ≤ 1 + u64` (roundoff factor positive). -/
private theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith

/-- **Addition magnitude bound** to explicit operand caps: `|toReal (a + b)| ≤ (1+u)·(Ba + Bb)` — the sibling of
    C51's `sub_bound`/`mul_bound`, from C43's `add_mag_le`. -/
theorem add_bound (a b : Float) (Ba Bb : ℝ) (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) :
    |toReal (a + b)| ≤ (1 + u64) * (Ba + Bb) :=
  (add_mag_le a b).trans (mul_le_mul_of_nonneg_left (add_le_add ha hb) one_add_u64_nonneg)

/-- **The scalar PPO total loss** `(surr − cv·valSq) + ce·ent` (Float ops, mirroring `ppoTotalObjE surr valSq ent
    cv ce = add (sub surr (scale cv valSq)) (scale ce ent)`). -/
def ppoLossF (surr valSq ent cv ce : Float) : Float := (surr - cv * valSq) + ce * ent

/-- The assembled scalar-loss magnitude bound: `sub` (`surr − cv·valSq`) then `add` (`+ ce·ent`), each `(1+u)`-inflated,
    over the two `(1+u)`-inflated products `cv·valSq` and `ce·ent`. -/
noncomputable def lossBound (Bs Bv Be Bcv Bce : ℝ) : ℝ :=
  (1 + u64) * ((1 + u64) * (Bs + (1 + u64) * (Bcv * Bv)) + (1 + u64) * (Bce * Be))

/-- **Assembled scalar-loss magnitude bound.** From term caps `|surr| ≤ Bs`, `|valSq| ≤ Bv`, `|ent| ≤ Be`, `|cv| ≤
    Bcv`, `|ce| ≤ Bce`, the scalar loss `|toReal (ppoLossF …)| ≤ lossBound Bs Bv Be Bcv Bce`, composing `mul_bound`
    (the `cv`/`ce` weightings), `sub_bound`, and `add_bound`. -/
theorem ppoLossF_mag_le (surr valSq ent cv ce : Float) (Bs Bv Be Bcv Bce : ℝ)
    (hs : |toReal surr| ≤ Bs) (hv : |toReal valSq| ≤ Bv) (he : |toReal ent| ≤ Be)
    (hcv : |toReal cv| ≤ Bcv) (hce : |toReal ce| ≤ Bce) :
    |toReal (ppoLossF surr valSq ent cv ce)| ≤ lossBound Bs Bv Be Bcv Bce := by
  have hcvv : |toReal (cv * valSq)| ≤ (1 + u64) * (Bcv * Bv) := mul_bound cv valSq Bcv Bv hcv hv
  have hcee : |toReal (ce * ent)| ≤ (1 + u64) * (Bce * Be) := mul_bound ce ent Bce Be hce he
  have hsub : |toReal (surr - cv * valSq)| ≤ (1 + u64) * (Bs + (1 + u64) * (Bcv * Bv)) :=
    sub_bound surr (cv * valSq) Bs ((1 + u64) * (Bcv * Bv)) hs hcvv
  show |toReal ((surr - cv * valSq) + ce * ent)| ≤ lossBound Bs Bv Be Bcv Bce
  exact add_bound (surr - cv * valSq) (ce * ent) _ _ hsub hcee

/-- **The scalar PPO loss is overflow-free** when its assembled bound is within threshold — `ppoLossF_mag_le` +
    C43's `isFinite_of_bounded`. -/
theorem ppoLossF_isFinite (surr valSq ent cv ce : Float) (Bs Bv Be Bcv Bce : ℝ)
    (hs : |toReal surr| ≤ Bs) (hv : |toReal valSq| ≤ Bv) (he : |toReal ent| ≤ Be)
    (hcv : |toReal cv| ≤ Bcv) (hce : |toReal ce| ≤ Bce)
    (hbound : lossBound Bs Bv Be Bcv Bce ≤ overflowBound) :
    (ppoLossF surr valSq ent cv ce).isFinite = true :=
  isFinite_of_bounded _
    ((ppoLossF_mag_le surr valSq ent cv ce Bs Bv Be Bcv Bce hs hv he hcv hce).trans hbound)

/-- **END-TO-END scalar loss finiteness from the network/rollout budgets.** The surrogate and value terms are the
    concrete C51 `ppoSurrF`/`valSqF`, discharged from their own inputs (`ppoSurrF_mag_le`/`valSqF_mag_le`); the
    entropy term is a bounded value (`|ent| ≤ Be`, the network's entropy budget). Given the assembled bound within
    threshold, the scalar PPO loss `ppoLossF (ppoSurrF …) (valSqF …) ent cv ce` is overflow-free — the PPO FORWARD
    loss is now certified overflow-free directly from the logit/activation/return budgets. -/
theorem ppoLossF_isFinite_of_terms (g r lo hi V ret ent cv ce : Float)
    (Bg Br Bl Bh BV Bret Be Bcv Bce : ℝ)
    (hg : |toReal g| ≤ Bg) (hr : |toReal r| ≤ Br) (hl : |toReal lo| ≤ Bl) (hh : |toReal hi| ≤ Bh)
    (hV : |toReal V| ≤ BV) (hret : |toReal ret| ≤ Bret) (he : |toReal ent| ≤ Be)
    (hcv : |toReal cv| ≤ Bcv) (hce : |toReal ce| ≤ Bce)
    (hbound : lossBound
        (max ((1 + u64) * (Bg * Br)) ((1 + u64) * (Bg * (max (max Br Bl) Bh))))
        ((1 + u64) * (((1 + u64) * (BV + Bret)) * ((1 + u64) * (BV + Bret))))
        Be Bcv Bce ≤ overflowBound) :
    (ppoLossF (ppoSurrF g r lo hi) (valSqF V ret) ent cv ce).isFinite = true :=
  ppoLossF_isFinite (ppoSurrF g r lo hi) (valSqF V ret) ent cv ce _ _ Be Bcv Bce
    (ppoSurrF_mag_le g r lo hi Bg Br Bl Bh hg hr hl hh)
    (valSqF_mag_le V ret BV Bret hV hret)
    he hcv hce hbound

end Puffer.RL.PPOLossScalar
