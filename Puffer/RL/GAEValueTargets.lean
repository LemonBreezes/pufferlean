/-
The SECOND output of `NNTrain.computeGAE` — the value-function regression target (the "returns to go" the
critic is trained against) — closed-form and bounded, completing the FULL `computeGAE` on top of a85's
advantage half.

`computeGAE` returns `(adv, returns)` where `returns[t] = adv[t] + Vₜ` (the GAE advantage plus the value
baseline = the TD(λ) return estimate). a85 (`GAEInvariant`) closed the advantage array `.1`; this file does
the returns array `.2`, which is a forward `map` over the SAME final advantage array inside the one `Id.run`.

  • `computeGAE_eq` — reduces the WHOLE `computeGAE` pair (not just `.1`) to `(advFold, map (·+Vₜ) advFold)`,
    the a83 `MProd`/`Id.run_pure` recipe kept over both components.
  • `computeGAE_snd_getElem!` — the structural link `returns[t] = adv[t] + Vₜ` (`rangeMapGetElem!`).
  • `computeGAE_valueTarget_getElem!` — the closed form `returns[t] = gadvList (gaeSuffix t) + Vₜ`
    (composing the structural link with a85's `computeGAE_getElem!`). Pure structural — NO Float axioms.
  • `computeGAE_valueTarget_error` — capstone: the runnable value-target slot `t` is within
    `u64·|Aₜ + Vₜ| + gadvErrBnd (gaeSuffix t)` of the ℝ GAE return (`gadvListR` fold `+` `toReal Vₜ`) —
    a85's `computeGAE_error` plus the single Float add `Aₜ + Vₜ`.

Axiom-clean beyond the trusted Float (1+δ) base (`add_model`/`mul_model`/`toReal`/`toReal_zeroLit`, inherited
via a85); the closed forms use only logic axioms.
-/
import Puffer.RL.GAEInvariant
import Puffer.RL.ForwardExec

namespace Puffer.RL.GAEValueTargets

open Puffer.RL.NNTrain
open Puffer.RL.BackwardLoopReduction
open Puffer.RL.GAEInvariant
open Puffer.FloatR

theorem computeGAE_eq (traj : Array Transition) (gamma lam : Float) :
    computeGAE traj gamma lam
      = (((List.range traj.size).foldl (gaeStep traj gamma lam traj.size)
            ⟨Array.replicate traj.size 0.0, 0.0⟩).1,
         (Array.range traj.size).map (fun t =>
            (((List.range traj.size).foldl (gaeStep traj gamma lam traj.size)
              ⟨Array.replicate traj.size 0.0, 0.0⟩).1)[t]! + traj[t]!.value)) := by
  simp only [computeGAE, Array.set!_eq_setIfInBounds, bind_pure_comp, map_pure,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one,
    Nat.div_one, List.forIn_pure_yield_eq_foldl, List.range_eq_range', Id.run_pure]
  rfl

/-- **`computeGAE`'s value-target array has one entry per transition.** The second output of `computeGAE` (the
    critic regression targets `Rₜ = Aₜ + Vₜ`) has size exactly `traj.size` — the structural companion of
    `computeGAE_fst_size` (which sizes the advantage array `.1`). It is a forward `map` over `Array.range traj.size`
    inside the one `Id.run`, so its length matches the trajectory regardless of `gamma`/`lam`. Consumers (`updateAC`,
    `ppoGrad`, `updatePPO`) index `returns[t]!` for every `t < traj.size`, so this length invariant is what keeps
    those lookups in range. -/
theorem computeGAE_snd_size (traj : Array Transition) (gamma lam : Float) :
    (computeGAE traj gamma lam).2.size = traj.size := by
  rw [computeGAE_eq]
  dsimp only
  rw [Array.size_map, Array.size_range]

open Puffer.RL.ForwardExec (rangeMapGetElem!)

/-- The value-target (critic regression target) slot `t` of `computeGAE` is `Aₜ + Vₜ`. -/
theorem computeGAE_snd_getElem! (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) :
    (computeGAE traj gamma lam).2[t]! = (computeGAE traj gamma lam).1[t]! + traj[t]!.value := by
  rw [computeGAE_eq]
  dsimp only
  rw [rangeMapGetElem! _ _ t ht]

/-- **`computeGAE` value-target closed form.** Slot `t` of the returns output holds `Aₜ + Vₜ = gadvList
    (gaeSuffix t) + traj[t]!.value` — the GAE advantage plus the value baseline. -/
theorem computeGAE_valueTarget_getElem! (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) :
    (computeGAE traj gamma lam).2[t]!
      = gadvList (gaeSuffix traj gamma lam traj.size t) + traj[t]!.value := by
  rw [computeGAE_snd_getElem! traj gamma lam t ht, computeGAE_getElem! traj gamma lam t ht]

/-- **`computeGAE` value target is causal (depends only on current-and-future transitions).**
    If two trajectories have the same length and agree on every transition from index `t` onward,
    then their critic regression targets `returns[t] = Aₜ + Vₜ` computed by `computeGAE` coincide.
    The value target at `t` is a function of the suffix `traj[t..n)` alone — changing any earlier
    transition leaves `returns[t]` unchanged. This is the frame/causality companion to
    `computeGAE_valueTarget_getElem!` (closed form) and `computeGAE_fst_recurrence` (advantage
    recurrence): the backward GAE loop never lets past data leak into a slot's target. The suffix
    hypothesis is load-bearing — the per-position TD error `deltaAt` at `s` reads `traj[s]!` and
    `traj[s+1]!`, and the value baseline reads `traj[t]!`, so agreement must extend over the whole
    suffix `[t, n)`. -/
theorem computeGAE_valueTarget_local
    (traj₁ traj₂ : Array Transition) (gamma lam : Float) (t : Nat)
    (hsize : traj₁.size = traj₂.size) (ht : t < traj₁.size)
    (hagree : ∀ j, t ≤ j → j < traj₁.size → traj₁[j]! = traj₂[j]!) :
    (computeGAE traj₁ gamma lam).2[t]! = (computeGAE traj₂ gamma lam).2[t]! := by
  -- per-position TD error agrees on the shared suffix
  have hdelta : ∀ s, t ≤ s → s < traj₁.size →
      deltaAt traj₁ gamma traj₁.size s = deltaAt traj₂ gamma traj₁.size s := by
    intro s hts hs
    unfold deltaAt
    have es := hagree s hts hs
    by_cases hnext : s + 1 < traj₁.size
    · have es1 := hagree (s + 1) (by omega) hnext
      rw [es, es1]
    · rw [if_neg hnext, if_neg hnext, es]
  -- per-position recursion weight agrees on the shared suffix
  have hw : ∀ s, t ≤ s → s < traj₁.size →
      wAt traj₁ gamma lam s = wAt traj₂ gamma lam s := by
    intro s hts hs
    unfold wAt
    rw [hagree s hts hs]
  -- the whole (δ,w) suffix agrees, hence the GAE fold does
  have hsuf : gaeSuffix traj₁ gamma lam traj₁.size t = gaeSuffix traj₂ gamma lam traj₂.size t := by
    rw [← hsize]
    unfold gaeSuffix
    apply List.map_congr_left
    intro k hk
    rw [List.mem_range] at hk
    have hs : t + k < traj₁.size := by omega
    rw [hdelta (t + k) (by omega) hs, hw (t + k) (by omega) hs]
  -- and the value baseline Vₜ agrees
  have hval : traj₁[t]!.value = traj₂[t]!.value := by rw [hagree t le_rfl ht]
  rw [computeGAE_valueTarget_getElem! traj₁ gamma lam t ht,
      computeGAE_valueTarget_getElem! traj₂ gamma lam t (hsize ▸ ht),
      hsuf, hval]

/-- **`computeGAE` advantage array is causal (depends only on current-and-future transitions).** If two
    trajectories have the same length and agree on every transition from index `t` onward, then the GAE
    advantages `Aₜ = (computeGAE …).1[t]!` computed by the imperative backward loop coincide at `t`: the
    advantage at `t` is a function of the suffix `traj[t..n)` alone, so changing any strictly-earlier
    transition leaves `Aₜ` unchanged. This is the advantage-array (`.1`) companion of
    `computeGAE_valueTarget_local` (which covers the value-target array `.2`), and it is genuinely INDEPENDENT
    of it: since Float addition is not cancellative, `Aₜ = Rₜ − Vₜ` does NOT follow from `Rₜ = Aₜ + Vₜ` with
    equal `Rₜ, Vₜ`, so advantage causality cannot be recovered from value-target causality — and the advantage
    array is the more primitive object anyway (consumed directly by the actor-critic gradient/update). The
    suffix hypothesis is load-bearing at BOTH ends: the per-step TD error `deltaAt` at position `s` reads
    `traj[s]!` and (through the `·nnt` value product Float never simplifies away) `traj[s+1]!.value`, so
    agreement must cover the whole suffix `[t, n)` — dropping any interior index breaks it; `hsize` is also
    load-bearing (`gaeSuffix … n t` bakes `n` into both the `t+1 < n` bootstrap gate and the suffix length
    `n − t`). Proved from the backward-loop closed form `computeGAE_getElem!` plus congruence of the `(δ,w)`
    suffix (identical `hdelta`/`hw`/`hsuf` machinery to the value-target version). -/
theorem computeGAE_fst_local
    (traj₁ traj₂ : Array Transition) (gamma lam : Float) (t : Nat)
    (hsize : traj₁.size = traj₂.size) (ht : t < traj₁.size)
    (hagree : ∀ j, t ≤ j → j < traj₁.size → traj₁[j]! = traj₂[j]!) :
    (computeGAE traj₁ gamma lam).1[t]! = (computeGAE traj₂ gamma lam).1[t]! := by
  -- per-position TD error agrees on the shared suffix
  have hdelta : ∀ s, t ≤ s → s < traj₁.size →
      deltaAt traj₁ gamma traj₁.size s = deltaAt traj₂ gamma traj₁.size s := by
    intro s hts hs
    unfold deltaAt
    have es := hagree s hts hs
    by_cases hnext : s + 1 < traj₁.size
    · have es1 := hagree (s + 1) (by omega) hnext
      rw [es, es1]
    · rw [if_neg hnext, if_neg hnext, es]
  -- per-position recursion weight agrees on the shared suffix
  have hw : ∀ s, t ≤ s → s < traj₁.size →
      wAt traj₁ gamma lam s = wAt traj₂ gamma lam s := by
    intro s hts hs
    unfold wAt
    rw [hagree s hts hs]
  -- the whole (δ,w) suffix agrees, hence the GAE fold does
  have hsuf : gaeSuffix traj₁ gamma lam traj₁.size t = gaeSuffix traj₂ gamma lam traj₂.size t := by
    rw [← hsize]
    unfold gaeSuffix
    apply List.map_congr_left
    intro k hk
    rw [List.mem_range] at hk
    have hs : t + k < traj₁.size := by omega
    rw [hdelta (t + k) (by omega) hs, hw (t + k) (by omega) hs]
  rw [computeGAE_getElem! traj₁ gamma lam t ht,
      computeGAE_getElem! traj₂ gamma lam t (hsize ▸ ht),
      hsuf]

/-- **`computeGAE` value-target accuracy.** The runnable returns slot `t` is within `gadvErrBnd + one add`
    of the ℝ GAE return (advantage fold `+` value baseline). Composes a85's `computeGAE_error` with the
    single Float add `Aₜ + Vₜ`. -/
theorem computeGAE_valueTarget_error (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) :
    |toReal ((computeGAE traj gamma lam).2[t]!)
        - (gadvListR ((gaeSuffix traj gamma lam traj.size t).map (fun p => (toReal p.1, toReal p.2)))
            + toReal (traj[t]!.value))|
      ≤ u64 * |toReal ((computeGAE traj gamma lam).1[t]!) + toReal (traj[t]!.value)|
        + gadvErrBnd (gaeSuffix traj gamma lam traj.size t) := by
  rw [computeGAE_snd_getElem! traj gamma lam t ht]
  set A := (computeGAE traj gamma lam).1[t]! with hA
  set GR := gadvListR ((gaeSuffix traj gamma lam traj.size t).map (fun p => (toReal p.1, toReal p.2)))
  have hadd : |toReal (A + traj[t]!.value) - (toReal A + toReal (traj[t]!.value))|
      ≤ u64 * |toReal A + toReal (traj[t]!.value)| := add_error A (traj[t]!.value)
  have hgae : |toReal A - GR| ≤ gadvErrBnd (gaeSuffix traj gamma lam traj.size t) := by
    rw [hA]; exact computeGAE_error traj gamma lam t ht
  calc |toReal (A + traj[t]!.value) - (GR + toReal (traj[t]!.value))|
      = |(toReal (A + traj[t]!.value) - (toReal A + toReal (traj[t]!.value)))
          + (toReal A - GR)| := by congr 1; ring
    _ ≤ |toReal (A + traj[t]!.value) - (toReal A + toReal (traj[t]!.value))| + |toReal A - GR| :=
        abs_add_le _ _
    _ ≤ u64 * |toReal A + toReal (traj[t]!.value)|
          + gadvErrBnd (gaeSuffix traj gamma lam traj.size t) := add_le_add hadd hgae

/-- **Runtime value-target geometric bound.** The critic regression target `returns[t] = Aₜ + Vₜ` computed by
    `computeGAE` is magnitude-bounded: `≤ (D/(1−W) + gadvErrBnd) + |Vₜ| + rounding`, given the TD errors `≤ D`,
    weights `≤ W ∈ [0,1)` (bounding the advantage via a132's `computeGAE_bounded`) and the value baseline `Vₜ`.
    So bounded TD errors and a bounded value baseline yield a bounded critic target — the fold's stability
    carried through the `Aₜ + Vₜ` add. Completes the GAE-stability story (return = advantage + value). -/
theorem computeGAE_valueTarget_bounded (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) (D W : ℝ) (hD : 0 ≤ D) (hW0 : 0 ≤ W) (hW1 : W < 1)
    (hδ : ∀ p ∈ gaeSuffix traj gamma lam traj.size t, |toReal p.1| ≤ D)
    (hw : ∀ p ∈ gaeSuffix traj gamma lam traj.size t, |toReal p.2| ≤ W) :
    |toReal ((computeGAE traj gamma lam).2[t]!)|
      ≤ (D / (1 - W) + gadvErrBnd (gaeSuffix traj gamma lam traj.size t))
        + |toReal (traj[t]!.value)|
        + u64 * |toReal ((computeGAE traj gamma lam).1[t]!) + toReal (traj[t]!.value)| := by
  have hA := computeGAE_bounded traj gamma lam t ht D W hD hW0 hW1 hδ hw
  rw [computeGAE_snd_getElem! traj gamma lam t ht]
  set A := (computeGAE traj gamma lam).1[t]! with hAeq
  have hadd := add_error A (traj[t]!.value)
  have htri := abs_sub_abs_le_abs_sub (toReal (A + traj[t]!.value)) (toReal A + toReal (traj[t]!.value))
  have habs := abs_add_le (toReal A) (toReal (traj[t]!.value))
  linarith

end Puffer.RL.GAEValueTargets
