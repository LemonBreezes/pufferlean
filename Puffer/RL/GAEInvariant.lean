/-
The backward-loop invariant for `NNTrain.computeGAE` → its closed form (the masked GAE advantage recursion
`Aₜ = δₜ + γλ·(1−doneₜ)·Aₜ₊₁`) and a certified Float↔ℝ accuracy bound. This closes the ADVANTAGES half of
the M3 trifecta, the companion to the RETURNS half (`ReturnsInvariant`, a84) atop a83's `foldl` reduction.

Unlike the discounted returns (constant factor `γ`, raw rewards), GAE carries a PER-STEP weight
`wₜ = γλ·(1−doneₜ)` (the terminal mask resets the recursion at episode boundaries) and a COMPUTED TD-error
`δₜ = rₜ + γ·V_{t+1}·(1−doneₜ) − Vₜ`. So the closed form folds over a `(δ, w)` suffix rather than a plain
reward list. We prove — by induction on the number of backward steps `m` — the fold-state invariant: after
`m` steps (touching `t ∈ {n−m,…,n−1}`), the running scalar `.2` holds `A_{n−m}` and every settled advantage
slot `.1[t]!` (`t ≥ n−m`) holds `Aₜ`. Instantiated at `m = n`, this reads the whole advantage array off.

  • `deltaAt` / `wAt` / `gaeSuffix` / `gaeSuffix_succ` — the per-position TD error and recursion weight
    (matching the loop's `delta` and `gamma*lam*nnt` char-for-char), the `(δ,w)` suffix, and its head-peel.
  • `gadvList` — the ℝ-free Float GAE fold `Aₜ = δₜ + wₜ·Aₜ₊₁` (base `0.0`, matching the loop's init).
  • `gae_invariant` — the fold-state invariant, by induction on `m` with `setIfInBounds` read-back.
  • `computeGAE_getElem!` — the closed form: `(computeGAE …).1[t]! = gadvList (gaeSuffix t)` (`m = n`).
    Pure structural — NO Float axioms.
  • `gadvList_error` / `gadvErrBnd` — `gadvList` tracks the ℝ recursion `gadvListR` (over the `toReal`
    deltas/weights) within the computable accumulated-rounding bound `gadvErrBnd` (mirror of `dret_error`).
  • `computeGAE_error` — capstone: the runnable advantage slot `t` is within `gadvErrBnd` of the ℝ GAE fold.

Scope: the ℝ reference `gadvListR` folds over the `toReal` of the ACTUAL Float deltas/weights the loop
computes — so the bound covers the fold's own add/mul rounding. The rounding INSIDE each `δₜ`/`wₜ`
(reward/value arithmetic, the `γ·V·mask` products) is a separate budget that composes on top (as the raw
rewards were exact in a84); it is not modelled here.

Axiom-clean beyond the trusted Float (1+δ) base (`add_model`/`mul_model`/`toReal`/`toReal_zeroLit`); the
closed-form `computeGAE_getElem!` uses none of them.
-/
import Puffer.RL.BackwardLoopReduction
import Puffer.Float.ADReverse
import Puffer.RL.GAE

namespace Puffer.RL.GAEInvariant

open Puffer.RL.NNTrain
open Puffer.RL.BackwardLoopReduction
open Puffer.FloatR.ADReverse (set!_getElem!_self set!_getElem!_ne_gen)

/-- The TD error at position `t`: `δₜ = rₜ + γ·V_{t+1}·(1−doneₜ) − Vₜ` (exactly the loop's `delta`). -/
def deltaAt (traj : Array Transition) (gamma : Float) (n t : Nat) : Float :=
  traj[t]!.reward
    + gamma * (if t + 1 < n then traj[t+1]!.value else 0.0) * (if traj[t]!.terminal then 0.0 else 1.0)
    - traj[t]!.value

/-- The GAE recursion weight at `t`: `wₜ = γλ·(1−doneₜ)` (exactly the loop's `gamma*lam*nnt`). -/
def wAt (traj : Array Transition) (gamma lam : Float) (t : Nat) : Float :=
  gamma * lam * (if traj[t]!.terminal then 0.0 else 1.0)

/-- The per-step `(δ, w)` suffix `[(δₜ,wₜ),…,(δₙ₋₁,wₙ₋₁)]`. -/
def gaeSuffix (traj : Array Transition) (gamma lam : Float) (n t : Nat) : List (Float × Float) :=
  (List.range (n - t)).map (fun k => (deltaAt traj gamma n (t + k), wAt traj gamma lam (t + k)))

/-- The GAE advantage of a `(δ, w)` suffix: `Aₜ = δₜ + wₜ·Aₜ₊₁` (base `0.0`, matching `computeGAE`). -/
def gadvList : List (Float × Float) → Float
  | [] => 0.0
  | (δ, w) :: rest => δ + w * gadvList rest

/-- Peel the head of the suffix. -/
theorem gaeSuffix_succ (traj : Array Transition) (gamma lam : Float) (n t : Nat) (h : t < n) :
    gaeSuffix traj gamma lam n t
      = (deltaAt traj gamma n t, wAt traj gamma lam t) :: gaeSuffix traj gamma lam n (t + 1) := by
  rw [gaeSuffix, gaeSuffix, show n - t = (n - (t + 1)) + 1 by omega, List.range_succ_eq_map,
    List.map_cons, List.map_map]
  simp only [Nat.add_zero]
  congr 1
  apply List.map_congr_left
  intro k _
  simp only [Function.comp, Nat.succ_eq_add_one]
  rw [show t + (k + 1) = t + 1 + k by omega]

/-- The fold-state invariant for `computeGAE`: after `m` backward steps, `.2` holds `A_{n−m}` and every
    settled slot `.1[t]!` (`t ≥ n−m`) holds `Aₜ = gadvList (gaeSuffix t)`. -/
theorem gae_invariant (traj : Array Transition) (gamma lam : Float) :
    ∀ m, m ≤ traj.size →
      let S := (List.range' 0 m).foldl (gaeStep traj gamma lam traj.size)
                 ⟨Array.replicate traj.size 0.0, 0.0⟩
      S.1.size = traj.size
      ∧ S.2 = gadvList (gaeSuffix traj gamma lam traj.size (traj.size - m))
      ∧ ∀ t, traj.size - m ≤ t → t < traj.size →
          S.1[t]! = gadvList (gaeSuffix traj gamma lam traj.size t) := by
  intro m
  induction m with
  | zero =>
    intro _
    refine ⟨by simp, ?_, ?_⟩
    · simp only [Nat.sub_zero]
      rw [gaeSuffix, Nat.sub_self]
      rfl
    · intro t ht ht'
      simp only [Nat.sub_zero] at ht
      omega
  | succ k ih =>
    intro hm
    have hk : k ≤ traj.size := by omega
    obtain ⟨ihsize, ihg, iharr⟩ := ih hk
    have hrange : List.range' 0 (k + 1) = List.range' 0 k ++ [k] := by
      rw [List.range'_concat]; norm_num
    set S := (List.range' 0 k).foldl (gaeStep traj gamma lam traj.size)
               ⟨Array.replicate traj.size 0.0, 0.0⟩ with hS
    have hfold : (List.range' 0 (k + 1)).foldl (gaeStep traj gamma lam traj.size)
                   ⟨Array.replicate traj.size 0.0, 0.0⟩
                 = gaeStep traj gamma lam traj.size S k := by
      rw [hrange, List.foldl_append, List.foldl_cons, List.foldl_nil]
    set t₀ := traj.size - 1 - k with ht₀
    have ht₀eq : t₀ = traj.size - (k + 1) := by omega
    have ht₀lt : t₀ < traj.size := by omega
    have hgnew : deltaAt traj gamma traj.size t₀ + wAt traj gamma lam t₀ * S.2
        = gadvList (gaeSuffix traj gamma lam traj.size t₀) := by
      rw [ihg, show traj.size - k = t₀ + 1 by omega, gaeSuffix_succ traj gamma lam traj.size t₀ ht₀lt,
        gadvList]
    rw [hfold]
    have hlast : (gaeStep traj gamma lam traj.size S k).2
        = deltaAt traj gamma traj.size t₀ + wAt traj gamma lam t₀ * S.2 := by
      simp only [gaeStep, deltaAt, wAt]; rw [← ht₀]
    have harr : (gaeStep traj gamma lam traj.size S k).1
        = S.1.setIfInBounds t₀ (deltaAt traj gamma traj.size t₀ + wAt traj gamma lam t₀ * S.2) := by
      simp only [gaeStep, deltaAt, wAt]; rw [← ht₀]
    refine ⟨?_, ?_, ?_⟩
    · rw [harr, Array.size_setIfInBounds, ihsize]
    · rw [hlast, hgnew, ht₀eq]
    · intro t ht ht'
      rw [harr, ← Array.set!_eq_setIfInBounds]
      by_cases htt : t = t₀
      · subst htt
        rw [set!_getElem!_self S.1 t₀ _ (by rw [ihsize]; exact ht₀lt), hgnew]
      · rw [set!_getElem!_ne_gen S.1 t₀ _ t (Ne.symm htt)]
        exact iharr t (by omega) ht'

open Puffer.RL.BackwardLoopReduction (computeGAE_fst_eq_foldl)

/-- **`computeGAE` closed form.** Advantage slot `t` holds `Aₜ = gadvList (gaeSuffix t)` — the masked GAE
    recursion over the TD-error suffix, read off the backward loop's `foldl` invariant at `m = n`. -/
theorem computeGAE_getElem! (traj : Array Transition) (gamma lam : Float) (t : Nat) (ht : t < traj.size) :
    (computeGAE traj gamma lam).1[t]! = gadvList (gaeSuffix traj gamma lam traj.size t) := by
  rw [computeGAE_fst_eq_foldl, List.range_eq_range']
  have h := (gae_invariant traj gamma lam traj.size le_rfl).2.2 t (by omega) ht
  simpa using h

open Puffer.FloatR (u64 toReal add_error mul_error toReal_zeroLit)

/-- The ℝ GAE advantage of a `(δ, w)` suffix: `Aₜ = δₜ + wₜ·Aₜ₊₁` (base `0`). -/
noncomputable def gadvListR : List (ℝ × ℝ) → ℝ
  | [] => 0
  | (δ, w) :: rest => δ + w * gadvListR rest

/-- Prefix product of the recursion weights: `∏ wⱼ` over a `(δ,w)` list. -/
noncomputable def wProdR : List (ℝ × ℝ) → ℝ
  | [] => 1
  | p :: rest => p.2 * wProdR rest

@[simp] theorem wProdR_nil : wProdR [] = 1 := rfl
@[simp] theorem wProdR_cons (p : ℝ × ℝ) (rest : List (ℝ × ℝ)) :
    wProdR (p :: rest) = p.2 * wProdR rest := rfl

/-- **GAE temporal-composition law (per-step weights).** The per-step-weighted GAE advantage over a
    concatenated `(δ,w)` suffix splits into the first segment plus the second segment discounted by the
    accumulated weight product `wProdR xs = ∏ⱼ wⱼ`:
    `gadvListR (xs ++ ys) = gadvListR xs + wProdR xs · gadvListR ys`.
    This is the per-position generalization of `GAE.gaeHead_append` (constant weight `w`, discount factor
    `w^|xs|`) to the actual masked weights `wₜ = γλ(1−doneₜ)` that `computeGAE` folds. It is the semigroup
    law of the masked GAE recursion: the second segment's contribution is scaled by the cumulative weight
    seen crossing the first segment. -/
theorem gadvListR_append (xs ys : List (ℝ × ℝ)) :
    gadvListR (xs ++ ys) = gadvListR xs + wProdR xs * gadvListR ys := by
  induction xs with
  | nil => simp [gadvListR]
  | cons p rest ih =>
      obtain ⟨δ, w⟩ := p
      simp only [List.cons_append, gadvListR, wProdR_cons, ih]
      ring

/-- **Episode-boundary severance.** Whenever the accumulated weight product over a prefix is zero
    (`wProdR xs = 0` — some weight `wⱼ = 0`, i.e. a terminal/`done` step masked `γλ(1−doneⱼ)` to `0`),
    the GAE advantage over `xs ++ ys` does NOT depend on the continuation `ys`: `gadvListR (xs ++ ys) =
    gadvListR xs`. The terminal mask severs the advantage's dependence on the next episode — the formal
    statement of GAE's "reset at episode ends". The hypothesis is load-bearing: with a nonzero product the
    tail `ys` genuinely contributes `wProdR xs · gadvListR ys`. Immediate from `gadvListR_append`. -/
theorem gadvListR_truncate_of_wProd_zero (xs ys : List (ℝ × ℝ)) (h : wProdR xs = 0) :
    gadvListR (xs ++ ys) = gadvListR xs := by
  rw [gadvListR_append, h, zero_mul, add_zero]

/-- **GAE advantage closed form (per-step weights).** The masked GAE recurrence `Aₜ = δₜ + wₜ·Aₜ₊₁` equals the
    prefix-product-weighted sum of TD errors `Σ_{i<n} (∏_{j<i} wⱼ)·δᵢ` — the exact discounted-sum-of-TD-errors
    form of GAE, generalizing `GAE.gaeHead_eq_geoSum` (`Σ wᵏδ`, constant weight) to the per-position weights
    `wₜ = γλ(1−doneₜ)`. At an episode boundary some `wⱼ = 0`, so every prefix product past it vanishes and future
    TD errors drop out of the sum — the discounting correctly truncates at episode ends. The ℝ-closed-form
    companion of `computeGAE_fst_recurrence` (the same recurrence on the executable array). -/
theorem gadvListR_eq_prefixSum (L : List (ℝ × ℝ)) :
    gadvListR L = ∑ i ∈ Finset.range L.length, wProdR (L.take i) * (L.getD i (0, 0)).1 := by
  induction L with
  | nil => simp [gadvListR]
  | cons p rest ih =>
    obtain ⟨δ, w⟩ := p
    show δ + w * gadvListR rest = _
    rw [List.length_cons, Finset.sum_range_succ']
    simp only [List.take_zero, wProdR_nil, one_mul, List.getD_cons_zero,
      List.take_succ_cons, wProdR_cons, List.getD_cons_succ]
    rw [ih, Finset.mul_sum, add_comm]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    ring

/-- The computable accumulated-rounding bound for `gadvList` (per-step weight `w`). -/
noncomputable def gadvErrBnd : List (Float × Float) → ℝ
  | [] => 0
  | (δ, w) :: rest =>
      u64 * |toReal δ + toReal (w * gadvList rest)|
      + u64 * |toReal w * toReal (gadvList rest)|
      + |toReal w| * gadvErrBnd rest

/-- **`gadvList` accuracy.** The Float GAE fold tracks its ℝ counterpart (over the `toReal` deltas/weights)
    within `gadvErrBnd` — mirror of `dret_error`, per-step weight `w`, base via `toReal_zeroLit`. -/
theorem gadvList_error (dws : List (Float × Float)) :
    |toReal (gadvList dws) - gadvListR (dws.map (fun p => (toReal p.1, toReal p.2)))|
      ≤ gadvErrBnd dws := by
  induction dws with
  | nil => simp [gadvList, gadvListR, gadvErrBnd, toReal_zeroLit]
  | cons p rest ih =>
      obtain ⟨δ, w⟩ := p
      simp only [gadvList, List.map_cons, gadvListR, gadvErrBnd]
      have split :
          toReal (δ + w * gadvList rest)
              - (toReal δ + toReal w * gadvListR (rest.map (fun p => (toReal p.1, toReal p.2))))
            = (toReal (δ + w * gadvList rest) - (toReal δ + toReal (w * gadvList rest)))
              + (toReal (w * gadvList rest) - toReal w * toReal (gadvList rest))
              + toReal w * (toReal (gadvList rest)
                  - gadvListR (rest.map (fun p => (toReal p.1, toReal p.2)))) := by
        ring
      rw [split]
      calc |(toReal (δ + w * gadvList rest) - (toReal δ + toReal (w * gadvList rest)))
              + (toReal (w * gadvList rest) - toReal w * toReal (gadvList rest))
              + toReal w * (toReal (gadvList rest)
                  - gadvListR (rest.map (fun p => (toReal p.1, toReal p.2))))|
          ≤ (|toReal (δ + w * gadvList rest) - (toReal δ + toReal (w * gadvList rest))|
              + |toReal (w * gadvList rest) - toReal w * toReal (gadvList rest)|)
              + |toReal w * (toReal (gadvList rest)
                  - gadvListR (rest.map (fun p => (toReal p.1, toReal p.2))))| :=
            (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
        _ ≤ (u64 * |toReal δ + toReal (w * gadvList rest)|
              + u64 * |toReal w * toReal (gadvList rest)|)
              + |toReal w| * gadvErrBnd rest := by
            refine add_le_add (add_le_add (add_error δ (w * gadvList rest))
              (mul_error w (gadvList rest))) ?_
            rw [abs_mul]
            exact mul_le_mul_of_nonneg_left ih (abs_nonneg _)

/-- **Capstone (GAE trifecta).** The runnable `computeGAE` advantage slot `t` is within `gadvErrBnd` of the
    ℝ GAE fold over the `toReal` TD-error suffix — the loop invariant (`computeGAE_getElem!`) composed with
    the Float↔ℝ accuracy bound (`gadvList_error`). -/
theorem computeGAE_error (traj : Array Transition) (gamma lam : Float) (t : Nat) (ht : t < traj.size) :
    |toReal ((computeGAE traj gamma lam).1[t]!)
        - gadvListR ((gaeSuffix traj gamma lam traj.size t).map (fun p => (toReal p.1, toReal p.2)))|
      ≤ gadvErrBnd (gaeSuffix traj gamma lam traj.size t) := by
  rw [computeGAE_getElem! traj gamma lam t ht]
  exact gadvList_error (gaeSuffix traj gamma lam traj.size t)

/-! ### Geometric boundedness: bounded TD errors give a bounded advantage -/

/-- **GAE advantage geometric bound.** For the per-position-weighted recurrence `gadvListR`, if every TD error
    has magnitude `≤ D` and every recursion weight `≤ W ∈ [0,1)` (the discount·λ·(1−done) factor), then the
    advantage is bounded by `D/(1−W)` — regardless of horizon. The advantage counterpart of the return bound
    `GAE.gaeHead_bounded` (a130), with per-step weights `wₜ = γλ(1−doneₜ)` instead of a constant discount. -/
theorem gadvListR_bounded (D W : ℝ) (hD : 0 ≤ D) (hW0 : 0 ≤ W) (hW1 : W < 1) :
    ∀ (L : List (ℝ × ℝ)), (∀ p ∈ L, |p.1| ≤ D) → (∀ p ∈ L, |p.2| ≤ W) →
      |gadvListR L| ≤ D / (1 - W) := by
  have hden : 0 < 1 - W := by linarith
  have hne : (1 - W) ≠ 0 := ne_of_gt hden
  intro L
  induction L with
  | nil => intro _ _; simp only [gadvListR, abs_zero]; exact div_nonneg hD hden.le
  | cons p rest ih =>
    intro hδ hw
    obtain ⟨δ, w⟩ := p
    have hδ1 : |δ| ≤ D := hδ (δ, w) (List.mem_cons_self ..)
    have hw1 : |w| ≤ W := hw (δ, w) (List.mem_cons_self ..)
    have hrest : |gadvListR rest| ≤ D / (1 - W) :=
      ih (fun q hq => hδ q (List.mem_cons_of_mem _ hq)) (fun q hq => hw q (List.mem_cons_of_mem _ hq))
    simp only [gadvListR]
    calc |δ + w * gadvListR rest| ≤ |δ| + |w * gadvListR rest| := abs_add_le _ _
      _ = |δ| + |w| * |gadvListR rest| := by rw [abs_mul]
      _ ≤ D + W * (D / (1 - W)) := add_le_add hδ1 (mul_le_mul hw1 hrest (abs_nonneg _) hW0)
      _ = D / (1 - W) := by field_simp; ring

/-- **Runtime GAE advantage geometric bound.** The runnable `computeGAE` advantage at slot `t` is bounded by
    `D/(1−W) + gadvErrBnd` given every (Float) TD error has `|toReal δ| ≤ D` and weight `|toReal w| ≤ W ∈ [0,1)`.
    Composes the ℝ advantage bound `gadvListR_bounded` with the Float↔ℝ accuracy `computeGAE_error` — so
    bounded TD errors yield a bounded advantage on the code that runs, feeding stable PPO updates. -/
theorem computeGAE_bounded (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) (D W : ℝ) (hD : 0 ≤ D) (hW0 : 0 ≤ W) (hW1 : W < 1)
    (hδ : ∀ p ∈ gaeSuffix traj gamma lam traj.size t, |toReal p.1| ≤ D)
    (hw : ∀ p ∈ gaeSuffix traj gamma lam traj.size t, |toReal p.2| ≤ W) :
    |toReal ((computeGAE traj gamma lam).1[t]!)|
      ≤ D / (1 - W) + gadvErrBnd (gaeSuffix traj gamma lam traj.size t) := by
  set L := (gaeSuffix traj gamma lam traj.size t).map (fun p => (toReal p.1, toReal p.2)) with hL
  have hbnd : |gadvListR L| ≤ D / (1 - W) := by
    apply gadvListR_bounded D W hD hW0 hW1 L
    · intro q hq; rw [hL, List.mem_map] at hq; obtain ⟨p, hp, rfl⟩ := hq; exact hδ p hp
    · intro q hq; rw [hL, List.mem_map] at hq; obtain ⟨p, hp, rfl⟩ := hq; exact hw p hp
  have herr := computeGAE_error traj gamma lam t ht
  rw [← hL] at herr
  have htri := abs_sub_abs_le_abs_sub (toReal ((computeGAE traj gamma lam).1[t]!)) (gadvListR L)
  linarith

end Puffer.RL.GAEInvariant
