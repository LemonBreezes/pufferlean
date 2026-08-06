/-
Cracking the backward-recursion `Id.run` loops → `List.foldl` — the structural bridge for the discounted
returns and GAE advantages (M3), long blocked on a do-notation reduction subtlety.

`Train.discountedReturns` (`Gₜ = Σ_{k≥t} γ^{k−t} rₖ`) and `NNTrain.computeGAE` (the GAE advantages) are the
one part of the pipeline still computed by an imperative BACKWARD loop building an array via `set!` — unlike
the forward `map`s, these mutate a running scalar and an array in lock-step. Connecting them to a functional
`foldl` (the prerequisite for a closed-form / accuracy bound) was blocked because the reduction leaves a goal
that resists `rfl`: the do-notation packs the two `mut` variables into an `MProd` (monadic product), NOT a
`Prod`, so a `Prod`-typed step function is never definitionally equal to the loop's `MProd`-typed one. Using
`MProd` (and `Id.run_pure` to strip the residual `pure`) closes it.

  • `drStep` / `discountedReturns_eq_foldl` — `discountedReturns` IS the `foldl` of the backward step
    `g ← rₜ + γ·g`, `returns[t] ← g` (`t = n−1−i`) over `List.range n`, `MProd`-stated. `discountedReturns_size`.
  • `gaeStep` / `computeGAE_fst_eq_foldl` — likewise the GAE advantage array is the `foldl` of the masked
    step `lastA ← δₜ + γλ·nntₜ·lastA` (`δ` the TD error, `nnt` the terminal mask). `computeGAE_fst_size`.

The reductions are pure structural equalities — NO float axioms. With the imperative loops now `foldl`s, the
loop invariant → closed-form (`Gₜ = gaeHead`-style geometric sum) / accuracy bound (`gaeHeadF_error`) is the
unblocked next step.
-/
import Puffer.RL.Train
import Puffer.RL.NNTrain

namespace Puffer.RL.BackwardLoopReduction

open Puffer.RL.Train
open Puffer.RL.NNTrain

/-! ### Discounted returns (`Train.discountedReturns`) -/

/-- One backward step of `discountedReturns`: `MProd` state `⟨g, returns⟩`, `g ← rₜ + γ·g`, `returns[t] ← g`
    (`t = n−1−i`; the do-notation packs the two `mut`s into `MProd`). -/
def drStep (traj : Array (Nat × Nat × Float)) (gamma : Float) (n : Nat) :
    MProd Float (Array Float) → Nat → MProd Float (Array Float) :=
  fun r i => ⟨traj[n - 1 - i]!.2.2 + gamma * r.1,
    r.2.setIfInBounds (n - 1 - i) (traj[n - 1 - i]!.2.2 + gamma * r.1)⟩

/-- **The runnable `discountedReturns` IS the `foldl` of `drStep`** over `List.range n`. Pure structural
    reduction (`Std.Legacy.Range.forIn → List.foldl`, `MProd` state, `Id.run_pure`). -/
theorem discountedReturns_eq_foldl (traj : Array (Nat × Nat × Float)) (gamma : Float) :
    discountedReturns traj gamma
      = ((List.range traj.size).foldl (drStep traj gamma traj.size)
          ⟨0.0, Array.replicate traj.size 0.0⟩).2 := by
  simp only [discountedReturns, Array.set!_eq_setIfInBounds, bind_pure_comp, map_pure,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one,
    Nat.div_one, List.forIn_pure_yield_eq_foldl, List.range_eq_range', Id.run_pure]
  rfl

/-- `discountedReturns` preserves the trajectory length (`set!` preserves size). -/
theorem discountedReturns_size (traj : Array (Nat × Nat × Float)) (gamma : Float) :
    (discountedReturns traj gamma).size = traj.size := by
  rw [discountedReturns_eq_foldl]
  have hstep : ∀ (l : List Nat) (st : MProd Float (Array Float)),
      (l.foldl (drStep traj gamma traj.size) st).2.size = st.2.size := by
    intro l
    induction l with
    | nil => intro st; rfl
    | cons a as ih => intro st; rw [List.foldl_cons, ih]; simp [drStep]
  rw [hstep]; simp

/-! ### GAE advantages (`NNTrain.computeGAE`) -/

/-- One backward step of `computeGAE`: `MProd` state `⟨adv, lastA⟩`, `lastA ← δₜ + γλ·nntₜ·lastA`,
    `adv[t] ← lastA` (`δ` the TD error, `nnt` the terminal mask, `t = n−1−i`). -/
def gaeStep (traj : Array Transition) (gamma lam : Float) (n : Nat) :
    MProd (Array Float) Float → Nat → MProd (Array Float) Float :=
  fun st i =>
    let t := n - 1 - i
    let nnt := if traj[t]!.terminal then 0.0 else 1.0
    let vNext := if t + 1 < n then traj[t+1]!.value else 0.0
    let delta := traj[t]!.reward + gamma * vNext * nnt - traj[t]!.value
    let lastA := delta + gamma * lam * nnt * st.2
    ⟨st.1.setIfInBounds t lastA, lastA⟩

/-- **The runnable `computeGAE` advantage array IS the `foldl` of `gaeStep`** over `List.range n`. -/
theorem computeGAE_fst_eq_foldl (traj : Array Transition) (gamma lam : Float) :
    (computeGAE traj gamma lam).1
      = ((List.range traj.size).foldl (gaeStep traj gamma lam traj.size)
          ⟨Array.replicate traj.size 0.0, 0.0⟩).1 := by
  simp only [computeGAE, Array.set!_eq_setIfInBounds, bind_pure_comp, map_pure,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one,
    Nat.div_one, List.forIn_pure_yield_eq_foldl, List.range_eq_range', Id.run_pure]
  rfl

/-- `computeGAE`'s advantage array has the trajectory length. -/
theorem computeGAE_fst_size (traj : Array Transition) (gamma lam : Float) :
    (computeGAE traj gamma lam).1.size = traj.size := by
  rw [computeGAE_fst_eq_foldl]
  have hstep : ∀ (l : List Nat) (st : MProd (Array Float) Float),
      (l.foldl (gaeStep traj gamma lam traj.size) st).1.size = st.1.size := by
    intro l
    induction l with
    | nil => intro st; rfl
    | cons a as ih => intro st; rw [List.foldl_cons, ih]; simp [gaeStep]
  rw [hstep]; simp

/-- **The GAE advantage recurrence, read off the executable advantage array.** For every interior index `t`
    (`t+1 < traj.size`) the advantage array produced by the imperative backward loop `computeGAE` satisfies the
    defining GAE fixpoint `Aₜ = δₜ + γλ·nntₜ·Aₜ₊₁`, with the TD error `δₜ = rₜ + γ·V₍ₜ₊₁₎·nntₜ − Vₜ` and the
    terminal mask `nntₜ = if terminalₜ then 0 else 1`. Recovered DIRECTLY from the `set!`-mutated array the loop
    builds — not from a separate spec: the proof carries a size invariant, `setIfInBounds` read-back lemmas, a
    suffix-preservation invariant (entries already written are never touched again), a last-written-entry =
    running-scalar invariant, and `List.range'` prefix splits so that both `A[t]` and `A[t+1]` are read off the
    SAME final array. The executable-advantage companion of `discountedReturns_recurrence` (returns array). -/
theorem computeGAE_fst_recurrence (traj : Array Transition) (gamma lam : Float)
    (t : Nat) (ht : t + 1 < traj.size) :
    (computeGAE traj gamma lam).1[t]!
      = (traj[t]!.reward
            + gamma * traj[t+1]!.value * (if traj[t]!.terminal then 0.0 else 1.0)
            - traj[t]!.value)
        + gamma * lam * (if traj[t]!.terminal then 0.0 else 1.0)
            * (computeGAE traj gamma lam).1[t+1]! := by
  rw [computeGAE_fst_eq_foldl]
  simp only [List.range_eq_range']
  revert ht
  generalize hn : traj.size = n
  intro ht
  generalize hg : gaeStep traj gamma lam n = g
  generalize hi : (⟨Array.replicate n 0.0, 0.0⟩ : MProd (Array Float) Float) = init
  have hsize : ∀ (l : List Nat) (st : MProd (Array Float) Float),
      (l.foldl g st).1.size = st.1.size := by
    intro l
    induction l with
    | nil => intro st; rfl
    | cons a as ih =>
      intro st; rw [List.foldl_cons, ih, ← hg]; simp [gaeStep, Array.size_setIfInBounds]
  have hinitsize : init.1.size = n := by rw [← hi]; exact Array.size_replicate
  have hself_read : ∀ (a0 : Array Float) (i : Nat) (v : Float),
      i < a0.size → (a0.setIfInBounds i v)[i]! = v := by
    intro a0 i v hi
    rw [Array.getElem!_eq_getD, Array.getD]
    simp [hi]
  have hne_read : ∀ (a0 : Array Float) (i j : Nat) (v : Float),
      i ≠ j → (a0.setIfInBounds i v)[j]! = a0[j]! := by
    intro a0 i j v hij
    rw [Array.getElem!_eq_getD, Array.getD, Array.getElem!_eq_getD, Array.getD]
    by_cases hj : j < a0.size
    · simp [hj, hij]
    · simp [hj]
  have hgfst : ∀ (st : MProd (Array Float) Float) (i : Nat),
      (g st i).1 = st.1.setIfInBounds (n - 1 - i) (g st i).2 := by
    intro st i; rw [← hg]; rfl
  have hpres : ∀ (l : List Nat) (st : MProd (Array Float) Float) (t' : Nat),
      (∀ j ∈ l, n - 1 - j ≠ t') → (l.foldl g st).1[t']! = st.1[t']! := by
    intro l
    induction l with
    | nil => intro st t' _; rfl
    | cons a as ih =>
      intro st t' hne
      rw [List.foldl_cons, ih (g st a) t' (fun j hj => hne j (List.mem_cons_of_mem a hj))]
      rw [hgfst st a]
      exact hne_read st.1 (n - 1 - a) t' (g st a).2 (hne a List.mem_cons_self)
  have hlast : ∀ k, 1 ≤ k → k ≤ n →
      ((List.range' 0 k).foldl g init).1[n - k]! = ((List.range' 0 k).foldl g init).2 := by
    intro k hk1 hkn
    have hconcat : List.range' 0 k = List.range' 0 (k - 1) ++ [k - 1] := by
      have h := @List.range'_concat 1 0 (k - 1)
      rw [show (k - 1) + 1 = k by omega, show (0 + 1 * (k - 1)) = k - 1 by omega] at h
      exact h
    rw [hconcat, List.foldl_append, List.foldl_cons, List.foldl_nil]
    rw [hgfst ((List.range' 0 (k - 1)).foldl g init) (k - 1)]
    rw [show n - 1 - (k - 1) = n - k by omega]
    apply hself_read
    rw [hsize, hinitsize]; omega
  have esplit_t : List.range' 0 n = List.range' 0 (n - t) ++ List.range' (n - t) t := by
    have h := @List.range'_append 0 (n - t) t 1
    rw [show (0 + 1 * (n - t)) = n - t by omega, show (n - t) + t = n by omega] at h
    exact h.symm
  have esplit_t1 : List.range' 0 n
      = List.range' 0 (n - t - 1) ++ List.range' (n - t - 1) (t + 1) := by
    have h := @List.range'_append 0 (n - t - 1) (t + 1) 1
    rw [show (0 + 1 * (n - t - 1)) = n - t - 1 by omega,
        show (n - t - 1) + (t + 1) = n by omega] at h
    exact h.symm
  have hA : ((List.range' 0 n).foldl g init).1[t]!
      = ((List.range' 0 (n - t)).foldl g init).2 := by
    rw [esplit_t, List.foldl_append]
    rw [hpres (List.range' (n - t) t) ((List.range' 0 (n - t)).foldl g init) t ?_]
    · have hl := hlast (n - t) (by omega) (by omega)
      rw [show n - (n - t) = t by omega] at hl
      exact hl
    · intro j hj
      rw [List.mem_range'] at hj
      obtain ⟨i, hi, rfl⟩ := hj
      omega
  have hB : ((List.range' 0 n).foldl g init).1[t + 1]!
      = ((List.range' 0 (n - t - 1)).foldl g init).2 := by
    rw [esplit_t1, List.foldl_append]
    rw [hpres (List.range' (n - t - 1) (t + 1)) ((List.range' 0 (n - t - 1)).foldl g init) (t + 1) ?_]
    · have hl := hlast (n - t - 1) (by omega) (by omega)
      rw [show n - (n - t - 1) = t + 1 by omega] at hl
      exact hl
    · intro j hj
      rw [List.mem_range'] at hj
      obtain ⟨i, hi, rfl⟩ := hj
      omega
  have hPm : (List.range' 0 (n - t)).foldl g init
      = g ((List.range' 0 (n - t - 1)).foldl g init) (n - t - 1) := by
    have hc : List.range' 0 (n - t) = List.range' 0 (n - t - 1) ++ [n - t - 1] := by
      have h := @List.range'_concat 1 0 (n - t - 1)
      rw [show (n - t - 1) + 1 = n - t by omega, show (0 + 1 * (n - t - 1)) = n - t - 1 by omega] at h
      exact h
    rw [hc, List.foldl_append, List.foldl_cons, List.foldl_nil]
  have hg2 : ∀ st : MProd (Array Float) Float,
      (g st (n - t - 1)).2
        = (traj[t]!.reward
              + gamma * traj[t+1]!.value * (if traj[t]!.terminal then 0.0 else 1.0)
              - traj[t]!.value)
          + gamma * lam * (if traj[t]!.terminal then 0.0 else 1.0) * st.2 := by
    intro st
    rw [← hg]
    simp only [gaeStep]
    rw [show n - 1 - (n - t - 1) = t by omega, if_pos ht]
  rw [hA, hB, hPm]
  exact hg2 ((List.range' 0 (n - t - 1)).foldl g init)

end Puffer.RL.BackwardLoopReduction
