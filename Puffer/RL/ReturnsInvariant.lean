/-
The backward-loop invariant for `Train.discountedReturns` → its closed form (`Gₜ = Σ_{k≥t} γ^{k−t} rₖ`)
and a certified Float↔ℝ accuracy bound. This closes the RETURNS half of the M3 trifecta, atop the
`foldl` structural reduction (`BackwardLoopReduction`, a83) that finally cracked the do-notation `MProd` loop.

With the imperative backward loop now a `List.foldl` of `drStep`, we prove — by induction on the number of
steps taken — the fold-state invariant: after `m` steps (over `List.range' 0 m`, touching indices
`t ∈ {n−m,…,n−1}`), the running scalar `.1` holds `G_{n−m}` and every settled array slot `.2[t]!`
(`t ≥ n−m`) holds `Gₜ`. Instantiated at `m = n`, this reads the whole returns array off in closed form.

  • `rewSuffix` / `rewSuffix_succ` — the reward suffix `[rₜ,…,rₙ₋₁]` (`rₖ = traj[k]!.2.2`) and its head-peel.
  • `dret` — the ℝ-free Float discounted return `Gₜ = rₜ + γ·Gₜ₊₁` (base `0.0`, matching the loop's init;
    note `(0.0 : Float)` is NOT defeq to `(0 : Float)`, so `dret` uses `0.0`, not `gaeHeadF`'s `0`).
  • `dr_invariant` — the fold-state invariant (size preserved, `.1 = G_{n−m}`, settled slots `= Gₜ`), by
    induction on `m` with `setIfInBounds` read-back bookkeeping (`set!_getElem!_self`/`_ne_gen`).
  • `discountedReturns_getElem!` — the closed form: `discountedReturns[t]! = dret γ (rewSuffix t)`
    (`m = n`). Pure structural — NO Float axioms.
  • `dret_error` / `dretErrBnd` — `dret` tracks the ℝ recurrence `gaeHead (toReal γ) (map toReal ds)` within
    the computable accumulated-rounding bound `dretErrBnd` (mirror of `gaeHeadF_error`, base via
    `toReal_zeroLit`).
  • `discountedReturns_error` — capstone: the runnable returns slot `t` is within `dretErrBnd` of the ℝ
    discounted return of the reward suffix from `t` (invariant ∘ accuracy).
  • `discountedReturns_geoSum_error` — the explicit textbook form: within `dretErrBnd` of the discounted sum
    `Σᵢ γⁱ·rₜ₊ᵢ` (`gaeHead_eq_geoSum` unfolds the recurrence; `rewSuffix_map_getD` makes the summand explicit).

Axiom-clean beyond the trusted Float (1+δ) base (`add_model`/`mul_model`/`toReal`/`toReal_zeroLit`); the
closed-form `discountedReturns_getElem!` uses none of them.
-/
import Puffer.RL.BackwardLoopReduction
import Puffer.Float.ADReverse
import Puffer.RL.GAE

namespace Puffer.RL.ReturnsInvariant

open Puffer.RL.Train
open Puffer.RL.BackwardLoopReduction

/-- The reward suffix `[rₜ, rₜ₊₁, …, rₙ₋₁]` (`rₖ = traj[k]!.2.2`). -/
def rewSuffix (traj : Array (Nat × Nat × Float)) (n t : Nat) : List Float :=
  (List.range (n - t)).map (fun k => (traj[t + k]!).2.2)

/-- Peel the head: `rewSuffix t = rₜ :: rewSuffix (t+1)` for `t < n`. -/
theorem rewSuffix_succ (traj : Array (Nat × Nat × Float)) (n t : Nat) (h : t < n) :
    rewSuffix traj n t = (traj[t]!).2.2 :: rewSuffix traj n (t + 1) := by
  rw [rewSuffix, rewSuffix, show n - t = (n - (t + 1)) + 1 by omega, List.range_succ_eq_map,
    List.map_cons, List.map_map]
  simp only [Nat.add_zero]
  congr 1
  apply List.map_congr_left
  intro k _
  simp only [Function.comp, Nat.succ_eq_add_one]
  rw [show t + (k + 1) = t + 1 + k by omega]

/-- The discounted return of a reward list `Gₜ = rₜ + γ·Gₜ₊₁` (base `0.0`, matching `discountedReturns`). -/
def dret (gamma : Float) : List Float → Float
  | [] => 0.0
  | r :: rest => r + gamma * dret gamma rest

open Puffer.FloatR.ADReverse (set!_getElem!_self set!_getElem!_ne_gen)

/-- The fold-state invariant for `discountedReturns`: after processing the first `m` backward steps
    (`List.range' 0 m`, touching `t ∈ {n−m,…,n−1}`), the running scalar `.1` holds `G_{n−m}` and every
    settled slot `.2[t]!` (`t ≥ n−m`) holds `Gₜ = dret γ (rewSuffix t)`. -/
theorem dr_invariant (traj : Array (Nat × Nat × Float)) (gamma : Float) :
    ∀ m, m ≤ traj.size →
      let S := (List.range' 0 m).foldl (drStep traj gamma traj.size)
                 ⟨0.0, Array.replicate traj.size 0.0⟩
      S.2.size = traj.size
      ∧ S.1 = dret gamma (rewSuffix traj traj.size (traj.size - m))
      ∧ ∀ t, traj.size - m ≤ t → t < traj.size →
          S.2[t]! = dret gamma (rewSuffix traj traj.size t) := by
  intro m
  induction m with
  | zero =>
    intro _
    refine ⟨by simp, ?_, ?_⟩
    · simp only [Nat.sub_zero]
      rw [rewSuffix, Nat.sub_self]
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
    set S := (List.range' 0 k).foldl (drStep traj gamma traj.size)
               ⟨0.0, Array.replicate traj.size 0.0⟩ with hS
    have hfold : (List.range' 0 (k + 1)).foldl (drStep traj gamma traj.size)
                   ⟨0.0, Array.replicate traj.size 0.0⟩
                 = drStep traj gamma traj.size S k := by
      rw [hrange, List.foldl_append, List.foldl_cons, List.foldl_nil]
    set t₀ := traj.size - 1 - k with ht₀
    have ht₀eq : t₀ = traj.size - (k + 1) := by omega
    have ht₀lt : t₀ < traj.size := by omega
    -- the new g-value equals `dret` of the suffix at `t₀`
    have hgnew : traj[t₀]!.2.2 + gamma * S.1 = dret gamma (rewSuffix traj traj.size t₀) := by
      rw [ihg, show traj.size - k = t₀ + 1 by omega, rewSuffix_succ traj traj.size t₀ ht₀lt, dret]
    rw [hfold]
    have hval : (drStep traj gamma traj.size S k).1 = traj[t₀]!.2.2 + gamma * S.1 := by
      simp only [drStep]; rw [← ht₀]
    have harr : (drStep traj gamma traj.size S k).2
        = S.2.setIfInBounds t₀ (traj[t₀]!.2.2 + gamma * S.1) := by
      simp only [drStep]; rw [← ht₀]
    refine ⟨?_, ?_, ?_⟩
    · rw [harr, Array.size_setIfInBounds, ihsize]
    · rw [hval, hgnew, ht₀eq]
    · intro t ht ht'
      rw [harr, ← Array.set!_eq_setIfInBounds]
      by_cases htt : t = t₀
      · subst htt
        rw [set!_getElem!_self S.2 t₀ _ (by rw [ihsize]; exact ht₀lt), hgnew]
      · rw [set!_getElem!_ne_gen S.2 t₀ _ t (Ne.symm htt)]
        exact iharr t (by omega) ht'

/-! ### Closed form and accuracy -/

open Puffer.RL.BackwardLoopReduction (discountedReturns_eq_foldl)
open Puffer.FloatR (u64 toReal add_error mul_error toReal_zeroLit add_model mul_model u64_lt_one)
open Puffer.RL.GAE (gaeHead gaeHead_cons gaeHead_eq_geoSum)

/-- **`discountedReturns` closed form.** Slot `t` holds `Gₜ = dret γ (reward suffix from t)` — the exact
    discounted return of the tail rewards, read off the backward loop's `foldl` invariant at `m = n`. -/
theorem discountedReturns_getElem! (traj : Array (Nat × Nat × Float)) (gamma : Float)
    (t : Nat) (ht : t < traj.size) :
    (discountedReturns traj gamma)[t]! = dret gamma (rewSuffix traj traj.size t) := by
  rw [discountedReturns_eq_foldl, List.range_eq_range']
  have h := (dr_invariant traj gamma traj.size le_rfl).2.2 t (by omega) ht
  simpa using h

/-- **Float discounted return of nonnegative rewards is nonnegative.** If `γ ≥ 0` (as a real) and every
    reward in `ds` embeds to a nonnegative real, then the *executable* Float discounted return `dret gamma ds`
    embeds to a nonnegative real. This is the sign-preservation law for the running scalar of the backward
    loop, proved directly against the trusted `(1+δ)` rounding model — the key fact being `1 + δ > 0` (since
    `|δ| ≤ u64 < 1`), so multiplication and addition of nonnegatives stay nonnegative even after rounding.
    (Order is NOT preserved by the model, but sign is.) -/
theorem dret_nonneg (gamma : Float) (hγ : 0 ≤ toReal gamma) :
    ∀ ds : List Float, (∀ r ∈ ds, 0 ≤ toReal r) → 0 ≤ toReal (dret gamma ds) := by
  intro ds
  induction ds with
  | nil =>
    intro _
    show (0 : ℝ) ≤ toReal (0.0 : Float)
    rw [toReal_zeroLit]
  | cons r rest ih =>
    intro hrew
    have hrest : 0 ≤ toReal (dret gamma rest) :=
      ih (fun x hx => hrew x (List.mem_cons.mpr (Or.inr hx)))
    have hr : 0 ≤ toReal r := hrew r (List.mem_cons.mpr (Or.inl rfl))
    obtain ⟨δ1, hδ1, hmul⟩ := mul_model gamma (dret gamma rest)
    obtain ⟨δ0, hδ0, hadd⟩ := add_model r (gamma * dret gamma rest)
    have h1pos1 : (0 : ℝ) < 1 + δ1 := by
      have := (abs_lt.mp (lt_of_le_of_lt hδ1 u64_lt_one)).1; linarith
    have h1pos0 : (0 : ℝ) < 1 + δ0 := by
      have := (abs_lt.mp (lt_of_le_of_lt hδ0 u64_lt_one)).1; linarith
    have hprod : 0 ≤ toReal (gamma * dret gamma rest) := by
      rw [hmul]; exact mul_nonneg (mul_nonneg hγ hrest) h1pos1.le
    show 0 ≤ toReal (r + gamma * dret gamma rest)
    rw [hadd]
    exact mul_nonneg (add_nonneg hr hprod) h1pos0.le

/-- **Nonnegative rewards ⇒ nonnegative `discountedReturns` slot.** The runnable value-regression target the
    critic is trained toward — slot `t` of the executable `discountedReturns` array — embeds to a nonnegative
    real whenever `γ ≥ 0` and every reward in the suffix `[rₜ,…,rₙ₋₁]` is nonnegative. The array counterpart of
    `dret_nonneg`, via the closed form `discountedReturns_getElem!`; a Float-level sign-conservation law
    surviving IEEE rounding (mirror of the ℝ-level `GAE.gaeHead_nonneg`). Both hypotheses are load-bearing: a
    single negative reward, or a negative `γ` on an all-nonnegative stream, can drive the slot negative. -/
theorem discountedReturns_nonneg (traj : Array (Nat × Nat × Float)) (gamma : Float)
    (t : Nat) (ht : t < traj.size) (hγ : 0 ≤ toReal gamma)
    (hrew : ∀ r ∈ rewSuffix traj traj.size t, 0 ≤ toReal r) :
    0 ≤ toReal ((discountedReturns traj gamma)[t]!) := by
  rw [discountedReturns_getElem! traj gamma t ht]
  exact dret_nonneg gamma hγ (rewSuffix traj traj.size t) hrew

/-- **Strictly positive immediate reward ⇒ strictly positive discounted-return target.** If the discount `γ ≥ 0`
    (as a real), the immediate reward at step `t` embeds to a *strictly* positive real, and every later reward in
    the suffix `[rₜ₊₁, …, rₙ₋₁]` embeds to a nonnegative real, then slot `t` of the executable `discountedReturns`
    array — the value-regression target the critic is trained toward — embeds to a *strictly* positive real. This
    sharpens the weak sign law `discountedReturns_nonneg` (`0 ≤`) to a STRICT one (`0 <`): IEEE rounding cannot
    collapse a positive immediate reward to zero, because the `(1+δ)` rounding factors are strictly positive
    (`|δ| ≤ u64 < 1`, so `1 + δ > 0`). All four hypotheses are load-bearing: with a merely-nonnegative head reward
    the return can be `0` (an all-zero stream); a single strongly-negative later reward, or a negative `γ` on a
    positive tail, can drive the discounted `γ·Gₜ₊₁` term below `−rₜ` and push the slot to `0`/negative; and an
    out-of-range `t` reads the default `0.0`. Proof: the closed form gives slot `t` `= (rₜ ⊕ (γ ⊗ Gₜ₊₁))` where the
    tail `Gₜ₊₁ ≥ 0` (`dret_nonneg`) so `γ ⊗ Gₜ₊₁ ≥ 0` (both `(1+δ)` factors positive), hence `rₜ + (γ·Gₜ₊₁) > 0`,
    and the outer rounding `(1+δ₀) > 0` preserves the strict sign (`mul_pos`). -/
theorem discountedReturns_pos (traj : Array (Nat × Nat × Float)) (gamma : Float)
    (t : Nat) (ht : t < traj.size) (hγ : 0 ≤ toReal gamma)
    (hr : 0 < toReal (traj[t]!.2.2))
    (hrest : ∀ r ∈ rewSuffix traj traj.size (t + 1), 0 ≤ toReal r) :
    0 < toReal ((discountedReturns traj gamma)[t]!) := by
  rw [discountedReturns_getElem! traj gamma t ht, rewSuffix_succ traj traj.size t ht, dret]
  set G := dret gamma (rewSuffix traj traj.size (t + 1)) with hG
  have hGnn : 0 ≤ toReal G := dret_nonneg gamma hγ _ hrest
  obtain ⟨δ1, hδ1, hmul⟩ := mul_model gamma G
  obtain ⟨δ0, hδ0, hadd⟩ := add_model (traj[t]!.2.2) (gamma * G)
  have h1pos1 : (0 : ℝ) < 1 + δ1 := by
    have := (abs_lt.mp (lt_of_le_of_lt hδ1 u64_lt_one)).1; linarith
  have h1pos0 : (0 : ℝ) < 1 + δ0 := by
    have := (abs_lt.mp (lt_of_le_of_lt hδ0 u64_lt_one)).1; linarith
  have hprod : 0 ≤ toReal (gamma * G) := by
    rw [hmul]; exact mul_nonneg (mul_nonneg hγ hGnn) h1pos1.le
  rw [hadd]
  exact mul_pos (by linarith) h1pos0

/-- **`discountedReturns` satisfies the backward Bellman recurrence.** For `t + 1 < traj.size`, adjacent slots of
    the runnable returns array obey `Gₜ = rₜ + γ·Gₜ₊₁` (`rₜ = traj[t]!.2.2`) — the defining semantics of the
    value-regression target the critic is trained toward, read directly off the executable array. It is exactly the
    recurrence the backward loop implements; here it is recovered from the closed form (`discountedReturns_getElem!`
    twice, peeling the reward suffix by `rewSuffix_succ` and unfolding one `dret` step). -/
theorem discountedReturns_recurrence (traj : Array (Nat × Nat × Float)) (gamma : Float)
    (t : Nat) (ht : t + 1 < traj.size) :
    (discountedReturns traj gamma)[t]!
      = (traj[t]!).2.2 + gamma * (discountedReturns traj gamma)[t + 1]! := by
  have ht0 : t < traj.size := by omega
  rw [discountedReturns_getElem! traj gamma t ht0,
      rewSuffix_succ traj traj.size t ht0, dret,
      ← discountedReturns_getElem! traj gamma (t + 1) ht]

/-- The computable accumulated-rounding bound for `dret` (base `0.0`) — same shape as `gaeErrBnd`. -/
noncomputable def dretErrBnd (gamma : Float) : List Float → ℝ
  | [] => 0
  | δ :: rest =>
      u64 * |toReal δ + toReal (gamma * dret gamma rest)|
      + u64 * |toReal gamma * toReal (dret gamma rest)|
      + |toReal gamma| * dretErrBnd gamma rest

/-- **`dret` accuracy.** The Float discounted return `dret γ ds` tracks the ℝ recurrence
    `gaeHead (toReal γ) (map toReal ds)` within `dretErrBnd γ ds` — mirror of `gaeHeadF_error`,
    base case via `toReal_zeroLit`. -/
theorem dret_error (gamma : Float) (ds : List Float) :
    |toReal (dret gamma ds) - gaeHead (toReal gamma) (ds.map toReal)| ≤ dretErrBnd gamma ds := by
  induction ds with
  | nil => simp [dret, dretErrBnd, toReal_zeroLit]
  | cons δ rest ih =>
      simp only [dret, List.map_cons, gaeHead_cons, dretErrBnd]
      have split :
          toReal (δ + gamma * dret gamma rest)
              - (toReal δ + toReal gamma * gaeHead (toReal gamma) (rest.map toReal))
            = (toReal (δ + gamma * dret gamma rest) - (toReal δ + toReal (gamma * dret gamma rest)))
              + (toReal (gamma * dret gamma rest) - toReal gamma * toReal (dret gamma rest))
              + toReal gamma * (toReal (dret gamma rest) - gaeHead (toReal gamma) (rest.map toReal)) := by
        ring
      rw [split]
      calc |(toReal (δ + gamma * dret gamma rest) - (toReal δ + toReal (gamma * dret gamma rest)))
              + (toReal (gamma * dret gamma rest) - toReal gamma * toReal (dret gamma rest))
              + toReal gamma * (toReal (dret gamma rest) - gaeHead (toReal gamma) (rest.map toReal))|
          ≤ (|toReal (δ + gamma * dret gamma rest) - (toReal δ + toReal (gamma * dret gamma rest))|
              + |toReal (gamma * dret gamma rest) - toReal gamma * toReal (dret gamma rest)|)
              + |toReal gamma * (toReal (dret gamma rest) - gaeHead (toReal gamma) (rest.map toReal))| :=
            (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
        _ ≤ (u64 * |toReal δ + toReal (gamma * dret gamma rest)|
              + u64 * |toReal gamma * toReal (dret gamma rest)|)
              + |toReal gamma| * dretErrBnd gamma rest := by
            refine add_le_add (add_le_add (add_error δ (gamma * dret gamma rest))
              (mul_error gamma (dret gamma rest))) ?_
            rw [abs_mul]
            exact mul_le_mul_of_nonneg_left ih (abs_nonneg _)

/-- **Runtime discounted-return geometric bound.** The runnable Float `dret γ ds` is bounded by the geometric
    estimate `R/(1−γ)` plus the arithmetic-rounding budget `dretErrBnd`, given every reward has magnitude `≤ R`
    and `γ ∈ [0,1)`. Composes the ℝ geometric bound `GAE.gaeHead_bounded` (a130) with the Float↔ℝ accuracy
    `dret_error` — so a uniformly-bounded reward stream yields a bounded return even on the executable code. -/
theorem dret_bounded (gamma : Float) (ds : List Float) (R : ℝ)
    (hγ0 : 0 ≤ toReal gamma) (hγ1 : toReal gamma < 1) (hR : 0 ≤ R)
    (hrew : ∀ r ∈ ds, |toReal r| ≤ R) :
    |toReal (dret gamma ds)| ≤ R / (1 - toReal gamma) + dretErrBnd gamma ds := by
  have hbnd : |Puffer.RL.GAE.gaeHead (toReal gamma) (ds.map toReal)| ≤ R / (1 - toReal gamma) := by
    apply Puffer.RL.GAE.gaeHead_bounded (toReal gamma) R hγ0 hγ1 hR
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨r, hr, rfl⟩ := hx
    exact hrew r hr
  have herr := dret_error gamma ds
  have htri := abs_sub_abs_le_abs_sub (toReal (dret gamma ds))
    (Puffer.RL.GAE.gaeHead (toReal gamma) (ds.map toReal))
  linarith

/-- **Capstone (returns trifecta).** The runnable `discountedReturns` slot `t` is within `dretErrBnd` of
    the ℝ discounted return of the reward suffix from `t` — the loop invariant (`discountedReturns_getElem!`)
    composed with the Float↔ℝ accuracy bound (`dret_error`). -/
theorem discountedReturns_error (traj : Array (Nat × Nat × Float)) (gamma : Float)
    (t : Nat) (ht : t < traj.size) :
    |toReal ((discountedReturns traj gamma)[t]!)
        - gaeHead (toReal gamma) ((rewSuffix traj traj.size t).map toReal)|
      ≤ dretErrBnd gamma (rewSuffix traj traj.size t) := by
  rw [discountedReturns_getElem! traj gamma t ht]
  exact dret_error gamma (rewSuffix traj traj.size t)

/-- **Runtime discounted-returns geometric bound (array slot).** The runnable `discountedReturns` array — what
    the trainer actually computes — is bounded at slot `t` by `R/(1−γ) + dretErrBnd` given every reward in the
    suffix has magnitude `≤ R` and `γ ∈ [0,1)`. Composes the ℝ geometric bound `GAE.gaeHead_bounded` (a130) with
    the array's Float↔ℝ accuracy `discountedReturns_error`. The array counterpart of `dret_bounded`. -/
theorem discountedReturns_bounded (traj : Array (Nat × Nat × Float)) (gamma : Float) (t : Nat)
    (ht : t < traj.size) (R : ℝ) (hγ0 : 0 ≤ toReal gamma) (hγ1 : toReal gamma < 1) (hR : 0 ≤ R)
    (hrew : ∀ r ∈ rewSuffix traj traj.size t, |toReal r| ≤ R) :
    |toReal ((discountedReturns traj gamma)[t]!)|
      ≤ R / (1 - toReal gamma) + dretErrBnd gamma (rewSuffix traj traj.size t) := by
  have hbnd : |Puffer.RL.GAE.gaeHead (toReal gamma) ((rewSuffix traj traj.size t).map toReal)|
      ≤ R / (1 - toReal gamma) := by
    apply Puffer.RL.GAE.gaeHead_bounded (toReal gamma) R hγ0 hγ1 hR
    intro x hx
    rw [List.mem_map] at hx
    obtain ⟨r, hr, rfl⟩ := hx
    exact hrew r hr
  have herr := discountedReturns_error traj gamma t ht
  have htri := abs_sub_abs_le_abs_sub (toReal ((discountedReturns traj gamma)[t]!))
    (Puffer.RL.GAE.gaeHead (toReal gamma) ((rewSuffix traj traj.size t).map toReal))
  linarith

/-- `getD` through the `toReal`-mapped reward suffix (in range): slot `i` is `rₜ₊ᵢ = traj[t+i]!.2.2`. -/
theorem rewSuffix_map_getD (traj : Array (Nat × Nat × Float)) (n t i : Nat) (hi : i < n - t) :
    ((rewSuffix traj n t).map toReal).getD i 0 = toReal (traj[t+i]!.2.2) := by
  rw [rewSuffix, List.map_map, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (by omega)]
  simp [Function.comp]

/-- **Discounted-returns explicit geometric sum + accuracy.** The runnable `discountedReturns` slot `t` is
    within `dretErrBnd` of the TEXTBOOK discounted sum `Σᵢ γⁱ·rₜ₊ᵢ` — `gaeHead_eq_geoSum` unfolds the
    recurrence to the closed form and `rewSuffix_map_getD` makes each summand the explicit reward
    `rₜ₊ᵢ = traj[t+i]!.2.2`. The honest textbook statement: `Gₜ ≈ Σᵢ γⁱ·rₜ₊ᵢ`. -/
theorem discountedReturns_geoSum_error (traj : Array (Nat × Nat × Float)) (gamma : Float) (t : Nat)
    (ht : t < traj.size) :
    |toReal ((discountedReturns traj gamma)[t]!)
        - ∑ i ∈ Finset.range (traj.size - t), (toReal gamma) ^ i * toReal (traj[t+i]!.2.2)|
      ≤ dretErrBnd gamma (rewSuffix traj traj.size t) := by
  have h := discountedReturns_error traj gamma t ht
  rw [gaeHead_eq_geoSum] at h
  have hlen : ((rewSuffix traj traj.size t).map toReal).length = traj.size - t := by
    simp [rewSuffix]
  have hsum : ∑ i ∈ Finset.range (traj.size - t), (toReal gamma) ^ i * toReal (traj[t+i]!.2.2)
      = ∑ i ∈ Finset.range (((rewSuffix traj traj.size t).map toReal).length),
          (toReal gamma) ^ i * ((rewSuffix traj traj.size t).map toReal).getD i 0 := by
    rw [hlen]
    exact Finset.sum_congr rfl (fun i hi => by
      rw [rewSuffix_map_getD traj traj.size t i (Finset.mem_range.mp hi)])
  rw [hsum]
  exact h

end Puffer.RL.ReturnsInvariant
