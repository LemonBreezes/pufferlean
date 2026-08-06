/-
`epReturns`: the last uncharacterized `segmentRollout` output — Float episode
sums, with (1+δ) error bounds against the abstract trace.

Every other output of the collection phase was characterized EXACTLY (a55–a58);
`epReturns` is different in kind: the runtime accumulates each episode's return
with Float additions (`epRet := epRet + r`), so exact equality with the ℝ spec
is false and the right statement is an error bound. This module:

* STRUCTURE (exact, no Float axioms): `segmentRollout_epReturns_eq` — the
  episode-returns output is precisely the per-episode LEFT-FOLD Float sums of
  the recorded buffer, split at terminal flags (`epSplit`/`chunksAux`; the
  splitting is a natural transformation, `chunksAux_map`).
* ERROR (trusted (1+δ) base): `foldlAdd_error` — the left-fold Float sum of a
  list deviates from the exact ℝ sum by at most `((1+u)ⁿ − 1)·(Σ|xᵢ|)`-style
  bounds (per-add `add_error`, accumulated; the S-coefficient computation shows
  the slack is exactly `P·u·Σ|xᵢ|` per step, so the geometric form is tight for
  this analysis).
* CAPSTONE `segmentRollout_epReturns_error`: pairwise (`List.Forall₂`), every
  recorded episode return is within `((1+u64)^horizon − 1)·horizon·Rmax` of the
  corresponding EXACT episode return of the abstract shared-stream trace
  (`episodeReturnsR`), for any env with ℝ-rewards bounded by `Rmax`. For
  realistic parameters (horizon 128, Rmax 1) the bound is ≈ `1.4e-14` — the
  console diagnostic is faithful to the spec to ~14 digits.

The chunk correspondence rides a56's exact refinement: the buffer's
(reward, terminal) pairs project to the abstract trace's `(r, done)` pairs, so
BOTH sides split into the same episodes; only the per-episode summation
arithmetic differs — and that is exactly what the fold bound prices.
-/
import Puffer.RL.LoopVecRollout

namespace Puffer.RL.Loop

open Puffer.RL.NNTrain (MLP policyAndValue)
open Puffer.FloatR

/-! ### Episode splitting (structural, polymorphic) -/

/-- Split a `(value, terminal)` stream into completed-episode chunks: values
    accumulate into the current chunk; a terminal closes it. The trailing
    incomplete chunk is DISCARDED — exactly as `segmentRollout` never pushes a
    partial `epRet`. -/
def chunksAux {α : Type*} : List (α × Bool) → List α → List (List α)
  | [], _ => []
  | (r, d) :: rest, cur =>
    if d then (cur ++ [r]) :: chunksAux rest [] else chunksAux rest (cur ++ [r])

/-- **Chunking is natural**: splitting commutes with mapping the values —
    the episode BOUNDARIES depend only on the Bool stream. This is what lets
    the Float buffer and the ℝ trace split into corresponding episodes. -/
theorem chunksAux_map {α β : Type*} (f : α → β) :
    ∀ (ps : List (α × Bool)) (cur : List α),
      chunksAux (ps.map (fun q => (f q.1, q.2))) (cur.map f)
        = (chunksAux ps cur).map (List.map f) := by
  intro ps
  induction ps with
  | nil => intro cur; rfl
  | cons q rest ih =>
    intro cur
    obtain ⟨r, d⟩ := q
    by_cases hd : d = true
    · simp [chunksAux, hd, ← ih]
    · have hd' : d = false := Bool.eq_false_iff.mpr hd
      simp [chunksAux, hd', ← ih]

/-- Nil-accumulator form of naturality (the shape the capstone uses). -/
theorem chunksAux_map_nil {α β : Type*} (f : α → β) (ps : List (α × Bool)) :
    chunksAux (ps.map (fun q => (f q.1, q.2))) []
      = (chunksAux ps []).map (List.map f) := by
  simpa using chunksAux_map f ps []

/-- Chunk lengths are bounded by the input length (plus the open chunk). -/
theorem chunksAux_length_le {α : Type*} :
    ∀ (ps : List (α × Bool)) (cur : List α),
      ∀ c ∈ chunksAux ps cur, c.length ≤ cur.length + ps.length := by
  intro ps
  induction ps with
  | nil => intro cur c hc; simp [chunksAux] at hc
  | cons q rest ih =>
    intro cur c hc
    obtain ⟨r, d⟩ := q
    by_cases hd : d = true
    · simp only [chunksAux, hd, if_true, List.mem_cons] at hc
      rcases hc with hc | hc
      · subst hc
        simp only [List.length_append, List.length_cons, List.length_nil]
        omega
      · have h := ih [] c hc
        simp only [List.length_nil, List.length_cons] at h ⊢
        omega
    · have hd' : d = false := Bool.eq_false_iff.mpr hd
      simp only [chunksAux, hd', Bool.false_eq_true, if_false] at hc
      have h := ih (cur ++ [r]) c hc
      simp only [List.length_append, List.length_cons, List.length_nil] at h ⊢
      omega

/-- Chunk members come from the input stream (or the open chunk). -/
theorem chunksAux_mem {α : Type*} :
    ∀ (ps : List (α × Bool)) (cur : List α),
      ∀ c ∈ chunksAux ps cur, ∀ x ∈ c, x ∈ cur ∨ ∃ d, (x, d) ∈ ps := by
  intro ps
  induction ps with
  | nil => intro cur c hc; simp [chunksAux] at hc
  | cons q rest ih =>
    intro cur c hc x hx
    obtain ⟨r, d⟩ := q
    by_cases hd : d = true
    · simp only [chunksAux, hd, if_true, List.mem_cons] at hc
      rcases hc with hc | hc
      · subst hc
        rcases List.mem_append.mp hx with hx | hx
        · exact Or.inl hx
        · exact Or.inr ⟨d, by simp at hx; simp [hx]⟩
      · rcases ih [] c hc x hx with h | ⟨d', hd'⟩
        · simp at h
        · exact Or.inr ⟨d', List.mem_cons_of_mem _ hd'⟩
    · have hd' : d = false := Bool.eq_false_iff.mpr hd
      simp only [chunksAux, hd', Bool.false_eq_true, if_false] at hc
      rcases ih (cur ++ [r]) c hc x hx with h | ⟨d'', hd''⟩
      · rcases List.mem_append.mp h with h | h
        · exact Or.inl h
        · exact Or.inr ⟨d, by simp at h; simp [h]⟩
      · exact Or.inr ⟨d'', List.mem_cons_of_mem _ hd''⟩

/-- The runtime's accumulator-style splitter (matches `segStep`'s `epRet`
    bookkeeping literally): returns `(completed episode sums, open sum)`. -/
def epSplit : List (Float × Bool) → Float → List Float × Float
  | [], acc => ([], acc)
  | (r, d) :: rest, acc =>
    if d then ((acc + r) :: (epSplit rest 0.0).1, (epSplit rest 0.0).2)
    else epSplit rest (acc + r)

/-- The accumulator splitter computes chunkwise LEFT-FOLD Float sums: `epSplit`
    with `acc = ` the fold of the open chunk equals folding each completed
    chunk. Exact Float-level equality (same adds, same order). -/
theorem epSplit_fst_eq_chunks :
    ∀ (ps : List (Float × Bool)) (cur : List Float),
      (epSplit ps (cur.foldl (· + ·) 0.0)).1
        = (chunksAux ps cur).map (fun c => c.foldl (· + ·) 0.0) := by
  intro ps
  induction ps with
  | nil => intro cur; rfl
  | cons q rest ih =>
    intro cur
    obtain ⟨r, d⟩ := q
    have hfold : cur.foldl (· + ·) 0.0 + r = (cur ++ [r]).foldl (· + ·) 0.0 := by
      simp [List.foldl_append]
    by_cases hd : d = true
    · simp only [epSplit, chunksAux, hd, if_true, List.map_cons]
      refine congrArg₂ List.cons hfold ?_
      simpa using ih []
    · have hd' : d = false := Bool.eq_false_iff.mpr hd
      simp only [epSplit, chunksAux, hd', Bool.false_eq_true, if_false]
      rw [hfold, ih (cur ++ [r])]

/-- Nil-accumulator corollary: `epSplit ps 0.0` folds each completed chunk. -/
theorem epSplit_fst_eq_chunks_nil (ps : List (Float × Bool)) :
    (epSplit ps 0.0).1 = (chunksAux ps []).map (fun c => c.foldl (· + ·) 0.0) := by
  simpa using epSplit_fst_eq_chunks ps []

/-! ### The runtime's epReturns, characterized -/

section Refinement

variable {S : Type}

/-- The action the policy draws at `(st, rng)` (names the recurring
    `sampleCat` expression). -/
def actAt (env : Puffer.RL.Env S) (p : MLP) (st : S) (rng : UInt64) : Nat :=
  Puffer.RL.Train.sampleCat (policyAndValue p (env.observe st)).1
    (Puffer.RL.Train.uniform01 (Puffer.RL.Train.rngNext rng).1)

/-- The transition `segStep` records at `(st, rng)`. -/
def recAt (env : Puffer.RL.Env S) (p : MLP) (st : S) (rng : UInt64) :
    Puffer.RL.NNTrain.Transition :=
  { obs := env.observe st, action := actAt env p st rng,
    reward := (env.step st (actAt env p st rng)).2.1,
    value := (policyAndValue p (env.observe st)).2,
    oldLogp := (policyAndValue p (env.observe st)).1[actAt env p st rng]!.log,
    terminal := (env.step st (actAt env p st rng)).2.2 }

@[simp] theorem recAt_reward (env : Puffer.RL.Env S) (p : MLP) (st : S) (rng : UInt64) :
    (recAt env p st rng).reward = (env.step st (actAt env p st rng)).2.1 := rfl

@[simp] theorem recAt_terminal (env : Puffer.RL.Env S) (p : MLP) (st : S) (rng : UInt64) :
    (recAt env p st rng).terminal = (env.step st (actAt env p st rng)).2.2 := rfl

/-- `segStep_of_terminal`, repackaged through `actAt`/`recAt` (defeq). -/
theorem segStep_terminal' (env : Puffer.RL.Env S) (p : MLP)
    {er : Float} {eps : Array Float} {rng : UInt64} {st : S}
    {traj : Array Puffer.RL.NNTrain.Transition}
    (h : (env.step st (actAt env p st rng)).2.2 = true) :
    segStep env p ⟨er, eps, rng, st, traj⟩
      = ⟨0.0, eps.push (er + (recAt env p st rng).reward),
         (env.reset (Puffer.RL.Train.rngNext rng).2).2,
         (env.reset (Puffer.RL.Train.rngNext rng).2).1,
         traj.push (recAt env p st rng)⟩ :=
  segStep_of_terminal env p h

/-- `segStep_of_not_terminal`, repackaged through `actAt`/`recAt` (defeq). -/
theorem segStep_not_terminal' (env : Puffer.RL.Env S) (p : MLP)
    {er : Float} {eps : Array Float} {rng : UInt64} {st : S}
    {traj : Array Puffer.RL.NNTrain.Transition}
    (h : ¬ (env.step st (actAt env p st rng)).2.2 = true) :
    segStep env p ⟨er, eps, rng, st, traj⟩
      = ⟨er + (recAt env p st rng).reward, eps, (Puffer.RL.Train.rngNext rng).2,
         (env.step st (actAt env p st rng)).1,
         traj.push (recAt env p st rng)⟩ :=
  segStep_of_not_terminal env p h

/-- **Structural invariant** for the `epReturns`/`epRet` slots of the iterated
    loop: the recorded trajectory grows by some `new` suffix, and the episode
    bookkeeping is exactly `epSplit` over `new`'s `(reward, terminal)` pairs
    from the incoming open sum. Exact — no Float axioms. -/
theorem segStep_iterate_ep (env : Puffer.RL.Env S) (p : MLP) :
    ∀ (n : Nat) (er : Float) (eps : Array Float) (rng : UInt64) (st : S)
      (traj : Array Puffer.RL.NNTrain.Transition),
      ∃ new : List Puffer.RL.NNTrain.Transition,
        ((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.snd.snd.snd.toList
            = traj.toList ++ new
        ∧ ((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.fst.toList
            = eps.toList
              ++ (epSplit (new.map (fun tr => (tr.reward, tr.terminal))) er).1
        ∧ ((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).fst
            = (epSplit (new.map (fun tr => (tr.reward, tr.terminal))) er).2 := by
  intro n
  induction n with
  | zero =>
    intro er eps rng st traj
    exact ⟨[], by simp, by simp [epSplit], by simp [epSplit]⟩
  | succ n ih =>
    intro er eps rng st traj
    rw [Function.iterate_succ_apply]
    by_cases h : (env.step st (actAt env p st rng)).2.2 = true
    · rw [segStep_terminal' env p h]
      obtain ⟨new', h1, h2, h3⟩ := ih 0.0
        (eps.push (er + (recAt env p st rng).reward))
        ((env.reset (Puffer.RL.Train.rngNext rng).2).2)
        ((env.reset (Puffer.RL.Train.rngNext rng).2).1)
        (traj.push (recAt env p st rng))
      refine ⟨recAt env p st rng :: new', ?_, ?_, ?_⟩
      · rw [h1, Array.toList_push]
        simp [List.append_assoc]
      · rw [h2, Array.toList_push]
        simp only [List.map_cons, epSplit, recAt_terminal, h, if_true]
        simp [List.append_assoc]
      · rw [h3]
        simp only [List.map_cons, epSplit, recAt_terminal, h, if_true]
    · rw [segStep_not_terminal' env p h]
      obtain ⟨new', h1, h2, h3⟩ := ih
        (er + (recAt env p st rng).reward) eps (Puffer.RL.Train.rngNext rng).2
        ((env.step st (actAt env p st rng)).1)
        (traj.push (recAt env p st rng))
      have hflag : (env.step st (actAt env p st rng)).2.2 = false :=
        Bool.eq_false_iff.mpr h
      refine ⟨recAt env p st rng :: new', ?_, ?_, ?_⟩
      · rw [h1, Array.toList_push]
        simp [List.append_assoc]
      · rw [h2]
        simp only [List.map_cons, epSplit, recAt_terminal, hflag,
          Bool.false_eq_true, if_false, recAt_reward]
      · rw [h3]
        simp only [List.map_cons, epSplit, recAt_terminal, hflag,
          Bool.false_eq_true, if_false, recAt_reward]

/-- **The `epReturns` output is the chunkwise Float fold of the buffer**: the
    episode returns `segmentRollout` reports are precisely the left-fold sums
    of the recorded rewards, split at the recorded terminals. Exact. -/
theorem segmentRollout_epReturns_eq (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.2.2.toList
      = (chunksAux ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList.map
          (fun tr => (tr.reward, tr.terminal))) []).map
            (fun c => c.foldl (· + ·) 0.0) := by
  obtain ⟨new, h1, h2, _⟩ := segStep_iterate_ep env p horizon 0.0 #[] rng0 s0 #[]
  have hbuf : (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList = new := by
    rw [segmentRollout_eq_iterate]
    simpa [segOut] using h1
  have heps : (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.2.2.toList
      = (epSplit (new.map (fun tr => (tr.reward, tr.terminal))) 0.0).1 := by
    rw [segmentRollout_eq_iterate]
    simpa [segOut] using h2
  rw [heps, hbuf, epSplit_fst_eq_chunks_nil]

/-! ### The Float left-fold error bound -/

/-- **Left-fold summation error, closed form**: the Float running sum deviates
    from the exact ℝ sum by at most `((1+u)ⁿ − 1)` times the magnitude budget
    (accumulator + terms). Per-step slack is exactly `(1+u)ⁿ·u·Σ|xᵢ|`. -/
theorem foldlAdd_error :
    ∀ (xs : List Float) (acc : Float),
      |toReal (xs.foldl (· + ·) acc) - (toReal acc + (xs.map toReal).sum)|
        ≤ ((1 + u64) ^ xs.length - 1)
            * (|toReal acc| + (xs.map (fun x => |toReal x|)).sum) := by
  intro xs
  induction xs with
  | nil =>
    intro acc
    simp
  | cons x xs ih =>
    intro acc
    have hu : (0:ℝ) ≤ u64 := u64_pos.le
    have hP0 : (0:ℝ) ≤ (1 + u64) ^ xs.length := by positivity
    have hP : (1:ℝ) ≤ (1 + u64) ^ xs.length := one_le_pow₀ (by linarith)
    obtain ⟨δ, hδ, hval⟩ := add_model acc x
    have hadd : |toReal (acc + x) - (toReal acc + toReal x)|
        ≤ u64 * |toReal acc + toReal x| := by
      rw [hval]
      have : (toReal acc + toReal x) * (1 + δ) - (toReal acc + toReal x)
          = (toReal acc + toReal x) * δ := by ring
      rw [this, abs_mul]
      exact mul_le_mul_of_nonneg_left hδ (abs_nonneg _) |>.trans
        (by rw [mul_comm])
    have habs : |toReal acc + toReal x| ≤ |toReal acc| + |toReal x| := abs_add_le _ _
    have hmag : |toReal (acc + x)| ≤ (1 + u64) * (|toReal acc| + |toReal x|) := by
      have h1 : |toReal (acc + x)|
          ≤ |toReal acc + toReal x| + u64 * |toReal acc + toReal x| := by
        have hsplit : toReal (acc + x)
            = (toReal acc + toReal x) + (toReal (acc + x) - (toReal acc + toReal x)) := by
          ring
        calc |toReal (acc + x)|
            = |(toReal acc + toReal x) + (toReal (acc + x) - (toReal acc + toReal x))| := by
              rw [← hsplit]
          _ ≤ |toReal acc + toReal x| + |toReal (acc + x) - (toReal acc + toReal x)| :=
              abs_add_le _ _
          _ ≤ _ := add_le_add le_rfl hadd
      nlinarith [mul_le_mul_of_nonneg_left habs hu]
    have hstep := ih (acc + x)
    have hS : (0:ℝ) ≤ (xs.map (fun x => |toReal x|)).sum :=
      List.sum_nonneg (by
        intro y hy
        obtain ⟨z, _, rfl⟩ := List.mem_map.mp hy
        exact abs_nonneg _)
    have hM : (0:ℝ) ≤ |toReal acc| + |toReal x| := by positivity
    have hsplit : toReal ((x :: xs).foldl (· + ·) acc)
          - (toReal acc + ((x :: xs).map toReal).sum)
        = (toReal (xs.foldl (· + ·) (acc + x)) - (toReal (acc + x) + (xs.map toReal).sum))
          + (toReal (acc + x) - (toReal acc + toReal x)) := by
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      ring
    rw [hsplit]
    calc |(toReal (xs.foldl (· + ·) (acc + x)) - (toReal (acc + x) + (xs.map toReal).sum))
          + (toReal (acc + x) - (toReal acc + toReal x))|
        ≤ |toReal (xs.foldl (· + ·) (acc + x)) - (toReal (acc + x) + (xs.map toReal).sum)|
          + |toReal (acc + x) - (toReal acc + toReal x)| := abs_add_le _ _
      _ ≤ ((1 + u64) ^ xs.length - 1)
            * (|toReal (acc + x)| + (xs.map (fun x => |toReal x|)).sum)
          + u64 * (|toReal acc| + |toReal x|) :=
          add_le_add hstep (hadd.trans (mul_le_mul_of_nonneg_left habs hu))
      _ ≤ ((1 + u64) ^ xs.length - 1)
            * ((1 + u64) * (|toReal acc| + |toReal x|)
                + (xs.map (fun x => |toReal x|)).sum)
          + u64 * (|toReal acc| + |toReal x|) := by
          have := mul_le_mul_of_nonneg_left
            (add_le_add hmag (le_refl ((xs.map (fun x => |toReal x|)).sum)))
            (by linarith : (0:ℝ) ≤ (1 + u64) ^ xs.length - 1)
          linarith
      _ ≤ ((1 + u64) ^ (x :: xs).length - 1)
            * (|toReal acc| + ((x :: xs).map (fun x => |toReal x|)).sum) := by
          simp only [List.length_cons, List.map_cons, List.sum_cons, pow_succ]
          nlinarith [mul_nonneg (mul_nonneg hP0 hu) hS]

/-- Per-chunk corollary with uniform hypotheses: a chunk of length `≤ H` with
    rewards bounded by `Rmax` has fold error `≤ ((1+u64)^H − 1)·H·Rmax`. -/
theorem chunk_fold_error {Rmax : ℝ} (hR0 : 0 ≤ Rmax) (H : Nat) (c : List Float)
    (hlen : c.length ≤ H) (hmem : ∀ x ∈ c, |toReal x| ≤ Rmax) :
    |toReal (c.foldl (· + ·) 0.0) - (c.map toReal).sum|
      ≤ ((1 + u64) ^ H - 1) * (H * Rmax) := by
  have hu : (0:ℝ) ≤ u64 := u64_pos.le
  have h := foldlAdd_error c 0.0
  rw [toReal_zeroLit] at h
  simp only [abs_zero, zero_add] at h
  have hsum : (c.map (fun x => |toReal x|)).sum ≤ c.length * Rmax := by
    have := List.sum_le_card_nsmul (c.map (fun x => |toReal x|)) Rmax (by
      intro y hy
      obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hy
      exact hmem z hz)
    simpa [nsmul_eq_mul] using this
  have hpow : (1 + u64) ^ c.length - 1 ≤ (1 + u64) ^ H - 1 := by
    have := pow_le_pow_right₀ (by linarith : (1:ℝ) ≤ 1 + u64) hlen
    linarith
  have hlenR : (c.length : ℝ) * Rmax ≤ (H : ℝ) * Rmax :=
    mul_le_mul_of_nonneg_right (by exact_mod_cast hlen) hR0
  have hb0 : (0:ℝ) ≤ (1 + u64) ^ c.length - 1 := by
    have : (1:ℝ) ≤ (1 + u64) ^ c.length := one_le_pow₀ (by linarith)
    linarith
  calc |toReal (c.foldl (· + ·) 0.0) - (c.map toReal).sum|
      ≤ ((1 + u64) ^ c.length - 1) * (c.map (fun x => |toReal x|)).sum := h
    _ ≤ ((1 + u64) ^ c.length - 1) * (c.length * Rmax) := by
        exact mul_le_mul_of_nonneg_left hsum hb0
    _ ≤ ((1 + u64) ^ H - 1) * (H * Rmax) := by
        refine mul_le_mul hpow hlenR ?_ ?_
        · positivity
        · linarith

/-! ### The abstract episode returns, and the capstone -/

/-- Pointwise facts over one list lift to `Forall₂` over its two images. -/
theorem forall₂_map_map {α β γ : Type*} {P : β → γ → Prop} (f : α → β) (g : α → γ) :
    ∀ (l : List α), (∀ c ∈ l, P (f c) (g c)) → List.Forall₂ P (l.map f) (l.map g) := by
  intro l
  induction l with
  | nil => intro _; exact List.Forall₂.nil
  | cons c l ih =>
    intro h
    exact List.Forall₂.cons (h c (List.mem_cons_self ..))
      (ih fun c' hc' => h c' (List.mem_cons_of_mem _ hc'))

/-- The EXACT episode returns of an abstract trace: split the `(r, done)`
    stream at terminals, sum each completed episode in ℝ. -/
noncomputable def episodeReturnsR {S O A : Type*} (τ : List (Transition S O A)) : List ℝ :=
  (chunksAux (τ.map (fun tr => (tr.r, tr.done))) []).map (fun c => c.sum)

/-- The buffer's ℝ-embedded `(reward, terminal)` pairs are the abstract trace's
    `(r, done)` pairs (projection of a56's exact refinement). -/
theorem segmentRollout_pairs (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList.map
        (fun tr => (toReal tr.reward, tr.terminal))
      = (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) horizon
          (s0, (), rng0)).map (fun tr => (tr.r, tr.done)) := by
  have h := congrArg (List.map (fun x : Array Float × Nat × ℝ × Bool => (x.2.2.1, x.2.2.2)))
    (segmentRollout_refinesR env p horizon s0 rng0)
  simpa [List.map_map, Function.comp_def, projRT, projL] using h

/-- Rewards along the induced shared-stream trace are bounded whenever the
    runtime env's (ℝ-embedded) step rewards are. -/
theorem rolloutR_reward_bound (env : Puffer.RL.Env S) (p : MLP) {Rmax : ℝ}
    (hR : ∀ (s : S) (a : Nat), |toReal (env.step s a).2.1| ≤ Rmax) {n : Nat}
    {x : S × Unit × UInt64} :
    ∀ tr ∈ rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) n x,
      |tr.r| ≤ Rmax := by
  intro tr htr
  obtain ⟨y, rfl⟩ := mem_rolloutR _ _ htr
  exact hR y.1 _

/-- **CAPSTONE: every reported episode return is within
    `((1+u64)^horizon − 1)·horizon·Rmax` of the exact episode return of the
    abstract trace.** The episode STRUCTURE matches exactly (a56 refinement +
    naturality of chunking); only the summation arithmetic differs, priced by
    the left-fold bound. For horizon 128, `Rmax = 1`: `≈ 1.4·10⁻¹⁴`. -/
theorem segmentRollout_epReturns_error (env : Puffer.RL.Env S) (p : MLP)
    {Rmax : ℝ} (hR0 : 0 ≤ Rmax)
    (hR : ∀ (s : S) (a : Nat), |toReal (env.step s a).2.1| ≤ Rmax)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    List.Forall₂ (fun (f : Float) (x : ℝ) =>
        |toReal f - x| ≤ ((1 + u64) ^ horizon - 1) * (horizon * Rmax))
      (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.2.2.toList
      (episodeReturnsR (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p)
        horizon (s0, (), rng0))) := by
  rw [segmentRollout_epReturns_eq]
  unfold episodeReturnsR
  rw [← segmentRollout_pairs env p horizon s0 rng0]
  have hmapmap : (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList.map
        (fun tr => (toReal tr.reward, tr.terminal))
      = ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList.map
          (fun tr => (tr.reward, tr.terminal))).map (fun q => (toReal q.1, q.2)) := by
    simp [List.map_map]
  rw [hmapmap, chunksAux_map_nil, List.map_map]
  set ps := (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1.toList.map
    (fun tr => (tr.reward, tr.terminal)) with hps
  have hpslen : ps.length = horizon := by
    simp [hps, segmentRollout_sizeR]
  -- pointwise over the SAME chunk list
  have hpoint : ∀ c ∈ chunksAux ps [],
      |toReal (c.foldl (· + ·) 0.0) - ((List.sum ∘ List.map toReal) c)|
        ≤ ((1 + u64) ^ horizon - 1) * (horizon * Rmax) := by
    intro c hc
    refine chunk_fold_error hR0 horizon c ?_ ?_
    · have := chunksAux_length_le ps [] c hc
      simpa [hpslen] using this
    · intro x hx
      rcases chunksAux_mem ps [] c hc x hx with h | ⟨d, hd⟩
      · simp at h
      · -- (x, d) ∈ ps ⇒ (toReal x, d) is a trace (r, done) pair ⇒ bounded
        have hmemR : (toReal x, d) ∈ (rolloutR ((toLoopEnvR env).withAutoReset)
            (toLoopPolicyR p) horizon (s0, (), rng0)).map (fun tr => (tr.r, tr.done)) := by
          rw [← segmentRollout_pairs env p horizon s0 rng0]
          rw [hmapmap]
          exact List.mem_map_of_mem hd
        obtain ⟨ltr, hltr, heq⟩ := List.mem_map.mp hmemR
        have : ltr.r = toReal x := congrArg Prod.fst heq
        rw [← this]
        exact rolloutR_reward_bound env p hR ltr hltr
  -- assemble the Forall₂ over the two maps of the same list
  exact forall₂_map_map _ _ _ hpoint

end Refinement

end Puffer.RL.Loop
