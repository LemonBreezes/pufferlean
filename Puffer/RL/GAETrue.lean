/-
CAPSTONE of the GAE accuracy story: the runnable `computeGAE` advantage vs the TRUE-arithmetic GAE — closing
the δ/w internal-rounding gap a85 held atomic, by composing the three pieces built for it.

a85 (`GAEInvariant.computeGAE_error`) bounds the runnable Float advantage against `gadvListR` over the
`toReal` of the ACTUAL computed Float deltas/weights — the δ/w rounding was out of scope. a89
(`gadvListR_perturb`) is the fold's input-sensitivity; a90 (`deltaAt_error`/`wAt_error`) is the per-position
δ/w rounding. This file threads them together:

  • `gaePerturbUB` / `gadvPerturbBnd_le` — reduce a89's exact perturbation budget `gadvPerturbBnd` to a
    computable one, given per-position `|Δδᵢ| ≤ eδ i` and `|Δwᵢ| ≤ ew i` (monotone in the δ/w slots; the
    `|wa|`/`|gadvListR rb|` coefficients are shared). Pure ℝ.
  • `trueGaeSuffix` — the exact-arithmetic `(δR, wR)` suffix (a90's true references).
  • `computeGAE_trueGAE_error` — the capstone: `|toReal(computeGAE.1[t]!) − gadvListR (trueGaeSuffix t)| ≤
    gadvErrBnd (gaeSuffix t) + gaePerturbUB …`. The triangle `a85 ⊕ (a89 ▸ gadvPerturbBnd_le with a90's
    per-position budgets)`. The first term is the fold's own add/mul rounding; the second is the per-position
    δ/w rounding propagated through the fold — together the WHOLE Float↔ℝ error of the runnable GAE.

Axiom-clean beyond the trusted Float (1+δ) base (`add/sub/mul_model` + `toReal` + `toReal_zeroLit`/
`toReal_oneLit`). The GAE accuracy story is now complete: advantages (a85) + value-targets (a86) + this
true-arithmetic upgrade; the returns side (a84/a88) was already exact (raw rewards).
-/
import Puffer.RL.GAEInputPerturb
import Puffer.RL.GAEDeltaError

namespace Puffer.RL.GAETrue

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.GAEInvariant (gadvListR gaeSuffix computeGAE_error gadvErrBnd)
open Puffer.RL.GAEInputPerturb (gadvPerturbBnd gadvPerturbBnd_nonneg gadvListR_perturb)
open Puffer.RL.GAEDeltaError (deltaR wR deltaAt_error wAt_error deltaErrBnd wErrBnd)

/-- Computable upper bound for `gadvPerturbBnd` when per-position δ/w perturbations are bounded by `eδ`/`ew`
    (index-shifted functions). -/
noncomputable def gaePerturbUB : List (ℝ × ℝ) → List (ℝ × ℝ) → (Nat → ℝ) → (Nat → ℝ) → ℝ
  | [], _, _, _ => 0
  | _ :: _, [], _, _ => 0
  | (_, wa) :: ra, (_, _) :: rb, eδ, ew =>
      eδ 0 + |wa| * gaePerturbUB ra rb (fun i => eδ (i+1)) (fun i => ew (i+1)) + ew 0 * |gadvListR rb|

/-- **Horizon-independent bound on the propagated δ/w rounding error `gaePerturbUB`.** If every recursion
    weight of the runnable suffix `aR` is bounded by `W ∈ [0,1)`, every true-arithmetic TD error of `bR` by `D`
    and every true weight of `bR` by the same `W`, and the per-position δ/w error budgets are uniformly bounded
    by `Eδ`/`Ew`, then the total propagated perturbation `gaePerturbUB aR bR eδ ew` (the second term of
    `gaeSlotBnd` — the per-position δ/w internal rounding pushed through the whole GAE fold) is at most the
    geometric constant `(Eδ + Ew·D/(1−W)) / (1−W)`, INDEPENDENT of the trajectory length. This is the
    perturbation-side analogue of `GAEInvariant.gadvListR_bounded` (which bounds the advantage itself): it
    certifies that the δ/w internal-rounding contribution to the true-GAE error stays uniformly controlled no
    matter how long the episode, via the same `1/(1−W)` geometric damping the masked weights provide. Proof: an
    induction on `aR` (case-split on `bR`), closing on the geometric fixpoint identity `Eδ + W·M + Ew·G = M`
    where `M = (Eδ + Ew·G)/(1−W)`, `G = D/(1−W)`; the tail-magnitude `|gadvListR rb|` term is controlled by
    `gadvListR_bounded`. All nine hypotheses are load-bearing (each has a concrete falsifying counterexample);
    `0 ≤ W` is derived locally in the cons/cons branch from `|wa| ≤ W`, and length mismatch is absorbed by the
    cons/nil branch returning `0 ≤ M`, so neither is a free hypothesis. -/
theorem gaePerturbUB_bounded (W Eδ Ew D : ℝ) (hW1 : W < 1)
    (hEδ : 0 ≤ Eδ) (hEw : 0 ≤ Ew) (hD : 0 ≤ D) :
    ∀ (aR bR : List (ℝ × ℝ)) (eδ ew : Nat → ℝ),
      (∀ p ∈ aR, |p.2| ≤ W) → (∀ p ∈ bR, |p.1| ≤ D) → (∀ p ∈ bR, |p.2| ≤ W) →
      (∀ i, i < aR.length → eδ i ≤ Eδ) → (∀ i, i < aR.length → ew i ≤ Ew) →
      gaePerturbUB aR bR eδ ew ≤ (Eδ + Ew * (D / (1 - W))) / (1 - W) := by
  have hden : (0:ℝ) < 1 - W := by linarith
  have hne : (1 - W) ≠ 0 := ne_of_gt hden
  set G : ℝ := D / (1 - W) with hGdef
  have hG_nonneg : 0 ≤ G := by rw [hGdef]; exact div_nonneg hD hden.le
  set M : ℝ := (Eδ + Ew * G) / (1 - W) with hMdef
  have hM_nonneg : 0 ≤ M := by
    rw [hMdef]; exact div_nonneg (add_nonneg hEδ (mul_nonneg hEw hG_nonneg)) hden.le
  have hMeq : Eδ + W * M + Ew * G = M := by
    rw [hMdef]; field_simp; ring
  intro aR
  induction aR with
  | nil => intro bR eδ ew _ _ _ _ _; simp only [gaePerturbUB]; exact hM_nonneg
  | cons pa ra ih =>
    intro bR eδ ew haw hbd hbw heδ hew
    obtain ⟨δa, wa⟩ := pa
    cases bR with
    | nil => simp only [gaePerturbUB]; exact hM_nonneg
    | cons pb rb =>
      obtain ⟨δb, wb⟩ := pb
      simp only [gaePerturbUB]
      have hwa : |wa| ≤ W := haw (δa, wa) (List.mem_cons_self ..)
      have hW0 : 0 ≤ W := (abs_nonneg wa).trans hwa
      have haw' : ∀ p ∈ ra, |p.2| ≤ W := fun p hp => haw p (List.mem_cons_of_mem _ hp)
      have hbd' : ∀ p ∈ rb, |p.1| ≤ D := fun p hp => hbd p (List.mem_cons_of_mem _ hp)
      have hbw' : ∀ p ∈ rb, |p.2| ≤ W := fun p hp => hbw p (List.mem_cons_of_mem _ hp)
      have heδ' : ∀ i, i < ra.length → eδ (i + 1) ≤ Eδ := by
        intro i hi; exact heδ (i + 1) (by simp only [List.length_cons]; omega)
      have hew' : ∀ i, i < ra.length → ew (i + 1) ≤ Ew := by
        intro i hi; exact hew (i + 1) (by simp only [List.length_cons]; omega)
      have htail : gaePerturbUB ra rb (fun i => eδ (i + 1)) (fun i => ew (i + 1)) ≤ M :=
        ih rb (fun i => eδ (i + 1)) (fun i => ew (i + 1)) haw' hbd' hbw' heδ' hew'
      have hgadv : |gadvListR rb| ≤ G := by
        rw [hGdef]; exact Puffer.RL.GAEInvariant.gadvListR_bounded D W hD hW0 hW1 rb hbd' hbw'
      have heδ0 : eδ 0 ≤ Eδ := heδ 0 (by simp)
      have hew0 : ew 0 ≤ Ew := hew 0 (by simp)
      have hterm1 : |wa| * gaePerturbUB ra rb (fun i => eδ (i + 1)) (fun i => ew (i + 1)) ≤ W * M := by
        calc |wa| * gaePerturbUB ra rb (fun i => eδ (i + 1)) (fun i => ew (i + 1))
            ≤ |wa| * M := mul_le_mul_of_nonneg_left htail (abs_nonneg wa)
          _ ≤ W * M := mul_le_mul_of_nonneg_right hwa hM_nonneg
      have hterm2 : ew 0 * |gadvListR rb| ≤ Ew * G := by
        calc ew 0 * |gadvListR rb|
            ≤ Ew * |gadvListR rb| := mul_le_mul_of_nonneg_right hew0 (abs_nonneg _)
          _ ≤ Ew * G := mul_le_mul_of_nonneg_left hgadv hEw
      calc eδ 0 + |wa| * gaePerturbUB ra rb (fun i => eδ (i + 1)) (fun i => ew (i + 1))
              + ew 0 * |gadvListR rb|
          ≤ Eδ + W * M + Ew * G := add_le_add (add_le_add heδ0 hterm1) hterm2
        _ = M := hMeq

/-- **`gadvPerturbBnd` reduction.** With per-position `|δaᵢ−δbᵢ| ≤ eδ i` and `|waᵢ−wbᵢ| ≤ ew i`, the exact
    perturbation budget `gadvPerturbBnd` is at most the computable `gaePerturbUB`. -/
theorem gadvPerturbBnd_le : ∀ (aR bR : List (ℝ × ℝ)) (eδ ew : Nat → ℝ), aR.length = bR.length →
    (∀ i, i < aR.length → |aR[i]!.1 - bR[i]!.1| ≤ eδ i) →
    (∀ i, i < aR.length → |aR[i]!.2 - bR[i]!.2| ≤ ew i) →
    gadvPerturbBnd aR bR ≤ gaePerturbUB aR bR eδ ew := by
  intro aR
  induction aR with
  | nil => intro bR eδ ew hlen _ _; simp [gadvPerturbBnd, gaePerturbUB]
  | cons pa ra ih =>
    intro bR eδ ew hlen hδ hw
    obtain ⟨δa, wa⟩ := pa
    cases bR with
    | nil => simp at hlen
    | cons pb rb =>
      obtain ⟨δb, wb⟩ := pb
      have hlen' : ra.length = rb.length := by simpa using hlen
      have hδ0 : |δa - δb| ≤ eδ 0 := by simpa using hδ 0 (by simp)
      have hw0 : |wa - wb| ≤ ew 0 := by simpa using hw 0 (by simp)
      have hδtail : ∀ i, i < ra.length → |ra[i]!.1 - rb[i]!.1| ≤ (fun i => eδ (i+1)) i := by
        intro i hi
        have := hδ (i+1) (by simpa using hi)
        simpa using this
      have hwtail : ∀ i, i < ra.length → |ra[i]!.2 - rb[i]!.2| ≤ (fun i => ew (i+1)) i := by
        intro i hi
        have := hw (i+1) (by simpa using hi)
        simpa using this
      have ihr := ih rb (fun i => eδ (i+1)) (fun i => ew (i+1)) hlen' hδtail hwtail
      simp only [gadvPerturbBnd, gaePerturbUB]
      have hstep : |wa| * gadvPerturbBnd ra rb ≤ |wa| * gaePerturbUB ra rb (fun i => eδ (i+1)) (fun i => ew (i+1)) :=
        mul_le_mul_of_nonneg_left ihr (abs_nonneg _)
      have hwmag : |wa - wb| * |gadvListR rb| ≤ ew 0 * |gadvListR rb| :=
        mul_le_mul_of_nonneg_right hw0 (abs_nonneg _)
      exact add_le_add (add_le_add hδ0 hstep) hwmag

/-- `getElem!` through a `List.range`-map (in range). -/
theorem listRangeMap_getElem! {β} [Inhabited β] (m : Nat) (g : Nat → β) (i : Nat) (hi : i < m) :
    ((List.range m).map g)[i]! = g i := by
  rw [List.getElem!_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hi]; rfl

/-- The TRUE real `(δ,w)` suffix from `t`: exact-arithmetic TD errors and weights. -/
noncomputable def trueGaeSuffix (traj : Array Transition) (gamma lam : Float) (n t : Nat) :
    List (ℝ × ℝ) :=
  (List.range (n - t)).map (fun k => (deltaR traj gamma n (t + k), wR traj gamma lam (t + k)))

open Puffer.RL.NNTrain (computeGAE)

/-- **Capstone: runnable GAE vs TRUE-arithmetic GAE.** The runnable advantage slot `t` is within
    `gadvErrBnd (gaeSuffix t)` (the fold's add/mul rounding, a85) `+ gaePerturbUB` (the per-position δ/w
    rounding a89⊕a90 propagated through the fold) of the exact-arithmetic GAE `gadvListR (trueGaeSuffix t)`.
    Closes the δ/w internal-rounding gap a85 held atomic. -/
theorem computeGAE_trueGAE_error (traj : Array Transition) (gamma lam : Float) (t : Nat)
    (ht : t < traj.size) :
    |toReal ((computeGAE traj gamma lam).1[t]!)
        - gadvListR (trueGaeSuffix traj gamma lam traj.size t)|
      ≤ gadvErrBnd (gaeSuffix traj gamma lam traj.size t)
        + gaePerturbUB ((gaeSuffix traj gamma lam traj.size t).map (fun p => (toReal p.1, toReal p.2)))
            (trueGaeSuffix traj gamma lam traj.size t)
            (fun i => deltaErrBnd traj gamma traj.size (t + i))
            (fun i => wErrBnd traj gamma lam (t + i)) := by
  set n := traj.size
  set aR := (gaeSuffix traj gamma lam n t).map (fun p => (toReal p.1, toReal p.2)) with haR
  set bR := trueGaeSuffix traj gamma lam n t with hbR
  -- aR indexes to the toReal Float δ/w; bR to the true reals
  have haRget : ∀ i, i < n - t → aR[i]! = (toReal (Puffer.RL.GAEInvariant.deltaAt traj gamma n (t + i)),
      toReal (Puffer.RL.GAEInvariant.wAt traj gamma lam (t + i))) := by
    intro i hi
    rw [haR, gaeSuffix, List.map_map, listRangeMap_getElem! _ _ i hi]
    rfl
  have hbRget : ∀ i, i < n - t → bR[i]! = (deltaR traj gamma n (t + i), wR traj gamma lam (t + i)) := by
    intro i hi
    rw [hbR, trueGaeSuffix, listRangeMap_getElem! _ _ i hi]
  have haRlen : aR.length = n - t := by rw [haR, gaeSuffix]; simp
  have hbRlen : bR.length = n - t := by rw [hbR, trueGaeSuffix]; simp
  have hlen : aR.length = bR.length := by rw [haRlen, hbRlen]
  have hδ : ∀ i, i < aR.length → |aR[i]!.1 - bR[i]!.1| ≤ deltaErrBnd traj gamma n (t + i) := by
    intro i hi
    rw [haRlen] at hi
    rw [haRget i hi, hbRget i hi]
    exact deltaAt_error traj gamma n (t + i)
  have hw : ∀ i, i < aR.length → |aR[i]!.2 - bR[i]!.2| ≤ wErrBnd traj gamma lam (t + i) := by
    intro i hi
    rw [haRlen] at hi
    rw [haRget i hi, hbRget i hi]
    exact wAt_error traj gamma lam (t + i)
  have hperturb : |gadvListR aR - gadvListR bR|
      ≤ gaePerturbUB aR bR (fun i => deltaErrBnd traj gamma n (t + i)) (fun i => wErrBnd traj gamma lam (t + i)) :=
    (gadvListR_perturb aR bR hlen).trans
      (gadvPerturbBnd_le aR bR (fun i => deltaErrBnd traj gamma n (t + i))
        (fun i => wErrBnd traj gamma lam (t + i)) hlen hδ hw)
  have ha85 := computeGAE_error traj gamma lam t ht
  calc |toReal ((computeGAE traj gamma lam).1[t]!) - gadvListR bR|
      ≤ |toReal ((computeGAE traj gamma lam).1[t]!) - gadvListR aR| + |gadvListR aR - gadvListR bR| := by
        have := abs_sub_le (toReal ((computeGAE traj gamma lam).1[t]!)) (gadvListR aR) (gadvListR bR)
        exact this
    _ ≤ gadvErrBnd (gaeSuffix traj gamma lam n t)
        + gaePerturbUB aR bR (fun i => deltaErrBnd traj gamma n (t + i)) (fun i => wErrBnd traj gamma lam (t + i)) :=
        add_le_add ha85 hperturb

/-! ### Aggregate over the trajectory -/

/-- The per-slot true-GAE error budget (the RHS of `computeGAE_trueGAE_error`). -/
noncomputable def gaeSlotBnd (traj : Array Transition) (gamma lam : Float) (n t : Nat) : ℝ :=
  gadvErrBnd (gaeSuffix traj gamma lam n t)
    + gaePerturbUB ((gaeSuffix traj gamma lam n t).map (fun p => (toReal p.1, toReal p.2)))
        (trueGaeSuffix traj gamma lam n t)
        (fun i => deltaErrBnd traj gamma n (t + i)) (fun i => wErrBnd traj gamma lam (t + i))

/-- **Aggregate true-GAE error over a trajectory.** The total of the runnable advantages tracks the total of
    the true-arithmetic GAE advantages within the sum of the per-slot budgets — the `Finset.sum` aggregate of
    `computeGAE_trueGAE_error` across the trajectory (`|Σ a − Σ b| ≤ Σ|a−b| ≤ Σ budget`). -/
theorem computeGAE_totalAdv_error (traj : Array Transition) (gamma lam : Float) :
    |∑ t ∈ Finset.range traj.size, toReal ((computeGAE traj gamma lam).1[t]!)
        - ∑ t ∈ Finset.range traj.size, gadvListR (trueGaeSuffix traj gamma lam traj.size t)|
      ≤ ∑ t ∈ Finset.range traj.size, gaeSlotBnd traj gamma lam traj.size t := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine Finset.sum_le_sum (fun t ht => ?_)
  exact computeGAE_trueGAE_error traj gamma lam t (Finset.mem_range.mp ht)

end Puffer.RL.GAETrue
