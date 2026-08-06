/-
# The quantitative Float-side partition floor: discharging C51/C72's floor hypotheses

C51 (`LossForwardFinite.logPart_isFinite`) certified the log-partition `log(Σ exp(logit))` overflow-free
GIVEN the partition floored `c ≤ toReal (sumExpF logits)` with `c > 0` — the checkable side-condition the
`log` domain needs, left as a hypothesis (`sumExpF_pos` showed positivity but not a QUANTITATIVE floor).
C72 (`LogTapeFinite.LogFloored`) likewise takes a per-input floor on every `log` argument's value. Both
disclosed the "ℝ→Float floor transfer" as remaining glue. This module supplies it: a CONCRETE, positive,
dischargeable floor on the Float partition from the logits' LOWER bounds alone.

**The math.** `sumExpF` is a right fold — `sumExpF (l :: ls) = Float.exp l + sumExpF ls` — so the HEAD's
exp term participates in exactly ONE Float addition. By the pre-existing `exp_model`,
`toReal (Float.exp l) ≥ Real.exp (toReal l)·(1−expEps) ≥ Real.exp (−M)·(1−expEps)` once `toReal l ≥ −M`;
and one rounded addition of the (nonnegative, `sumExpF_nonneg`) tail shrinks that contribution by at most
`(1−u64)` (`add_model`, nonneg summands). Hence the LENGTH-INDEPENDENT floor

    `floorC M := Real.exp (−M) · (1 − expEps) · (1 − u64)  ≤  toReal (sumExpF (l :: ls))`,

positive for every `M` (`floorC_pos`). Only the HEAD logit's lower bound is load-bearing (the tail only
helps); the convenience forms take the uniform two-sided logit bounds — the network-budget interface.

* `expF_floor` / `add_floor` — the two `(1+δ)`-model floor steps (exp lower bound; a nonneg rounded add
  preserves the first summand's contribution up to `(1−u64)`).
* `sumExpF_floor` / `sumExpF_floor_of_bounds` — THE concrete partition floor (head-only / all-logits forms).
* `logPart_isFinite_concrete` — C51's certificate with the floor DISCHARGED: the log-partition finiteness
  now needs ONLY two-sided logit bounds (`−M ≤ toReal l ≤ M`) and the budget check — fully in the network
  budgets, no free floor hypothesis.
* `evalF_partition_floor` — the C72 connector: an `Expr`-level partition-SHAPED log argument
  `.add (.exp a) b` (the compiled softmax-partition shape) has `evalF` floored at `floorC M`, directly
  discharging `LogFloored`'s floor conjunct at `c := floorC M` for such arguments.

**Scope (honestly disclosed).** The floor is CONSERVATIVE — it keeps only the head term's contribution
(the true partition is at least the whole sum); this is what makes it length-independent and simple. The
logits-bounded-BELOW requirement is honest and irreducible: a logit `→ −∞` genuinely collapses its exp to
`0`, and if ALL logits escape below, the partition genuinely underflows toward `0` — the floor must come
from somewhere, and one bounded logit suffices. The connector covers partition-shaped (`add (exp ·) ·`)
log arguments; other log shapes keep C72's per-input floor hypothesis. NO new axiom — everything from the
pre-existing `exp_model`/`add_model` and C51's positivity lemmas.
-/
import Puffer.RL.LossForwardFinite
import Puffer.Float.Expr
open Puffer.FloatR
open Puffer.RL.FiniteBound
open Puffer.RL.LossForwardFinite

namespace Puffer.RL.FloatFloor

/-- `1 − expEps > 0` (`expEps = 2⁻⁵²`). -/
theorem one_sub_expEps_pos : 0 < 1 - expEps := by unfold expEps; norm_num

/-- `1 − u64 > 0` (`u64 = 2⁻⁵³`, from the pre-existing `u64_lt_one`). -/
theorem one_sub_u64_pos : 0 < 1 - u64 := by have := u64_lt_one; linarith

/-- **The concrete Float-side partition floor value**: the head exp term's guaranteed contribution —
    the ideal `exp(−M)` shrunk by the exp rounding `(1−expEps)` and ONE addition rounding `(1−u64)`.
    Length-independent (the head participates in exactly one add of `sumExpF`'s right fold). -/
noncomputable def floorC (M : ℝ) : ℝ := Real.exp (-M) * (1 - expEps) * (1 - u64)

/-- The floor is strictly positive for every `M` — the `0 < c` that `log_model`'s domain needs. -/
theorem floorC_pos (M : ℝ) : 0 < floorC M :=
  mul_pos (mul_pos (Real.exp_pos _) one_sub_expEps_pos) one_sub_u64_pos

/-- **Float exp lower bound.** From `exp_model` (`toReal (Float.exp l) = Real.exp (toReal l)·(1+δ)`,
    `|δ| ≤ expEps`): with the logit bounded BELOW (`−M ≤ toReal l`), the Float exp is at least the
    ideal `exp(−M)` shrunk by `(1−expEps)`. -/
theorem expF_floor (l : Float) (M : ℝ) (hl : -M ≤ toReal l) :
    Real.exp (-M) * (1 - expEps) ≤ toReal (Float.exp l) := by
  obtain ⟨δ, hδ, he⟩ := exp_model l
  rw [he]
  have h1 : 1 - expEps ≤ 1 + δ := by have := abs_le.mp hδ; linarith [this.1]
  have h2 : Real.exp (-M) ≤ Real.exp (toReal l) := Real.exp_le_exp.mpr hl
  calc Real.exp (-M) * (1 - expEps)
      ≤ Real.exp (toReal l) * (1 - expEps) :=
        mul_le_mul_of_nonneg_right h2 one_sub_expEps_pos.le
    _ ≤ Real.exp (toReal l) * (1 + δ) :=
        mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le

/-- **Nonneg-summand addition floor.** One rounded add of a NONNEGATIVE second summand preserves the
    first summand's contribution up to `(1−u64)`: any lower bound `x` on `toReal a` yields
    `x·(1−u64) ≤ toReal (a + b)` (from `add_model`, `toReal a + toReal b ≥ toReal a` and `1+δ ≥ 1−u64`). -/
theorem add_floor (a b : Float) (ha : 0 ≤ toReal a) (hb : 0 ≤ toReal b)
    (x : ℝ) (hx : x ≤ toReal a) :
    x * (1 - u64) ≤ toReal (a + b) := by
  obtain ⟨δ, hδ, he⟩ := add_model a b
  rw [he]
  have h1 : 1 - u64 ≤ 1 + δ := by have := abs_le.mp hδ; linarith [this.1]
  have hab : toReal a ≤ toReal a + toReal b := by linarith
  calc x * (1 - u64)
      ≤ toReal a * (1 - u64) := mul_le_mul_of_nonneg_right hx one_sub_u64_pos.le
    _ ≤ (toReal a + toReal b) * (1 - u64) := mul_le_mul_of_nonneg_right hab one_sub_u64_pos.le
    _ ≤ (toReal a + toReal b) * (1 + δ) := mul_le_mul_of_nonneg_left h1 (by linarith)

/-- **THE CONCRETE PARTITION FLOOR** (head-only form — the tight statement). For a nonempty logit list,
    `floorC M ≤ toReal (sumExpF (l :: ls))` needs ONLY the head logit bounded below: `sumExpF`'s right
    fold puts the head's exp through exactly one add of the nonnegative tail (`sumExpF_nonneg`), so its
    `expF_floor` contribution survives with a single `(1−u64)` shrink. -/
theorem sumExpF_floor (l : Float) (ls : List Float) (M : ℝ) (hl : -M ≤ toReal l) :
    floorC M ≤ toReal (sumExpF (l :: ls)) := by
  show floorC M ≤ toReal (Float.exp l + sumExpF ls)
  unfold floorC
  exact add_floor _ _ (expF_pos l).le (sumExpF_nonneg ls) _ (expF_floor l M hl)

/-- The all-logits convenience form (the network-budget interface): uniform lower bounds on the logits
    give the floor (only the head's bound is used). -/
theorem sumExpF_floor_of_bounds (l : Float) (ls : List Float) (M : ℝ)
    (hbelow : ∀ a ∈ l :: ls, -M ≤ toReal a) :
    floorC M ≤ toReal (sumExpF (l :: ls)) :=
  sumExpF_floor l ls M (hbelow l (List.mem_cons.mpr (Or.inl rfl)))

/-- **THE LOG-PARTITION CERTIFICATE, FLOOR DISCHARGED** (capstone). C51's `logPart_isFinite` with the
    floor hypothesis ELIMINATED: for a nonempty logit list with TWO-SIDED bounds `−M ≤ toReal a ≤ M`
    (the network logit budgets) and the single budget check, `log(Σ exp(logit))` is overflow-free —
    `c := floorC M` supplied by `sumExpF_floor_of_bounds`, its positivity by `floorC_pos`. The
    log-partition finiteness is now fully in the network budgets: no free floor hypothesis remains. -/
theorem logPart_isFinite_concrete (l : Float) (ls : List Float) (M : ℝ)
    (habove : ∀ a ∈ l :: ls, toReal a ≤ M)
    (hbelow : ∀ a ∈ l :: ls, -M ≤ toReal a)
    (hbound : max |Real.log (floorC M)|
          |Real.log (sumBound (l :: ls).length (Real.exp M * (1 + expEps)))| * (1 + logEps)
        ≤ overflowBound) :
    (Float.log (sumExpF (l :: ls))).isFinite = true :=
  logPart_isFinite (l :: ls) M (floorC M) habove (floorC_pos M)
    (sumExpF_floor_of_bounds l ls M hbelow) hbound

/-- **The C72 connector.** An `Expr`-level PARTITION-SHAPED log argument `.add (.exp a) b` — the compiled
    softmax-partition shape `exp(logit) + rest` — has its `evalF` floored at `floorC M`, given the inner
    logit's value bounded below and the rest nonnegative. This is exactly the floor conjunct
    `c ≤ toReal (evalF arg σ)` that C72's `LogFloored σ (floorC M) (.log (.add (.exp a) b))` requires,
    discharged from the value bounds (the same one-add head argument, at the `Expr` level). -/
theorem evalF_partition_floor (σ : Nat → Float) (a b : Puffer.FloatR.ADR.Expr) (M : ℝ)
    (ha : -M ≤ toReal (Puffer.FloatR.ADR.evalF a σ))
    (hb : 0 ≤ toReal (Puffer.FloatR.ADR.evalF b σ)) :
    floorC M ≤ toReal (Puffer.FloatR.ADR.evalF (.add (.exp a) b) σ) := by
  show floorC M ≤ toReal (Float.exp (Puffer.FloatR.ADR.evalF a σ) + Puffer.FloatR.ADR.evalF b σ)
  unfold floorC
  exact add_floor _ _ (expF_pos _).le hb _ (expF_floor _ M ha)

end Puffer.RL.FloatFloor
