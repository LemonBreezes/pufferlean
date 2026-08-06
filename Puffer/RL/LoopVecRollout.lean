/-
The `vecRollout` refinement: sequential multi-env collection over ONE stream.

`NNTrain.vecRollout` rolls `numEnvs` env instances one segment each — but NOT
as a product system: it threads a SINGLE splitmix64 stream sequentially, env
`k+1`'s segment starting from the stream position env `k`'s segment left off.
So (as the a53 analysis predicted) it does not factor through `Env.pi`; its
honest spec is "N successive shared-stream rollouts with the stream threaded
through", expressed here as `vecSpec` over the a56 `EnvR`/`PolicyR` machinery:
segment `k` is `rolloutR` from `(states[k], (), σₖ)` where `σ₀ = rng0` and
`σₖ₊₁` is the `R`-component of the `k`-th segment's orbit endpoint
(`stateAtR … horizon`).

New per-segment facts (strengthening a56, which only related the TRACE): the
runtime segment's final state, final rng, and bootstrap value are the abstract
orbit endpoint's data (`segStep_iterate_joint` → `segmentRollout_final_state`/
`_final_rng`/`_bootV`). Then the vec-level do-loop reduces to a `List.foldl`
(`vecRollout_eq_foldl`) and one induction (`vecFold_spec`) delivers the
capstone `vecRollout_refines` plus `vecRollout_final_states`,
`vecRollout_final_rng`, `vecRollout_bootVals` — every training-relevant output
of `vecRollout` characterized abstractly, still EXACTLY (opaque `toReal` only).

Not covered (deliberately): the `epReturns` output — Float SUMS of rewards,
used only for console diagnostics, never for updates; its
Float↔ℝ story is (1+δ) error-bound territory, not exact refinement. Also open:
the interleaving-equivalence suggested in `vecRollout`'s docstring ("sequential
= interleaved up to RNG order") is a distributional statement about permuted
rng draws, not a trace equality — a different kind of theorem.
-/
import Puffer.RL.LoopR

namespace Puffer.RL.Loop

open Puffer.RL.Train (rngNext uniform01 sampleCat)
open Puffer.RL.NNTrain (MLP policyAndValue)

/-! ### Sequential multi-env spec, over the general shared-stream machines -/

variable {R S O A H : Type*}

/-- The spec of sequential multi-env collection: one shared-stream segment per
    start state, each segment's rollout recorded and the stream threaded from
    each orbit endpoint into the next segment. Returns
    `(traces, final states, final stream)`. The policy's private state is
    re-seeded with `h0` each segment, as the runtime does (its MLP policy is
    stateless: `H = Unit`). -/
def vecSpec (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H) (horizon : Nat) :
    List S → R → List (List (Transition S O A)) × List S × R
  | [], r => ([], [], r)
  | s :: rest, r =>
    ((rolloutR E P horizon (s, h0, r)) ::
        (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).1,
     (stateAtR E P (s, h0, r) horizon).1 ::
        (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).2.1,
     (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).2.2)

@[simp] theorem vecSpec_nil (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) (r : R) : vecSpec E P h0 horizon [] r = ([], [], r) := rfl

@[simp] theorem vecSpec_cons (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) (s : S) (rest : List S) (r : R) :
    vecSpec E P h0 horizon (s :: rest) r
      = ((rolloutR E P horizon (s, h0, r)) ::
            (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).1,
         (stateAtR E P (s, h0, r) horizon).1 ::
            (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).2.1,
         (vecSpec E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2).2.2) := rfl

/-- One trace per start state. -/
theorem vecSpec_num_traces (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) : ∀ (l : List S) (r : R),
    (vecSpec E P h0 horizon l r).1.length = l.length := by
  intro l
  induction l with
  | nil => intro r; rfl
  | cons s rest ih => intro r; simp [ih]

/-- Every trace has exactly `horizon` transitions. -/
theorem vecSpec_trace_length (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) : ∀ (l : List S) (r : R),
    ∀ t ∈ (vecSpec E P h0 horizon l r).1, t.length = horizon := by
  intro l
  induction l with
  | nil => intro r t ht; simp at ht
  | cons s rest ih =>
    intro r t ht
    rw [vecSpec_cons, List.mem_cons] at ht
    rcases ht with ht | ht
    · rw [ht]; exact rolloutR_length _ _ horizon _
    · exact ih _ t ht

section Refinement

variable {S : Type}

/-! ### Per-segment endpoint facts (strengthening a56's trace-only refinement) -/

/-- `segStep` on a terminal step, as an equation (the a56 inline `hstep`,
    exported). -/
theorem segStep_of_terminal (env : Puffer.RL.Env S) (p : MLP)
    {er : Float} {eps : Array Float} {rng : UInt64} {st : S}
    {traj : Array Puffer.RL.NNTrain.Transition}
    (h : (env.step st
        (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = true) :
    segStep env p ⟨er, eps, rng, st, traj⟩
      = ⟨0.0, eps.push (er + (env.step st
          (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1),
         (env.reset (rngNext rng).2).2, (env.reset (rngNext rng).2).1,
         traj.push
           { obs := env.observe st,
             action := sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1),
             reward := (env.step st
               (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1,
             value := (policyAndValue p (env.observe st)).2,
             oldLogp := (policyAndValue p (env.observe st)).1[
               sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1)]!.log,
             terminal := (env.step st
               (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 }⟩ := by
  simp only [segStep, if_pos h]

/-- `segStep` on a non-terminal step, as an equation. -/
theorem segStep_of_not_terminal (env : Puffer.RL.Env S) (p : MLP)
    {er : Float} {eps : Array Float} {rng : UInt64} {st : S}
    {traj : Array Puffer.RL.NNTrain.Transition}
    (h : ¬ (env.step st
        (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = true) :
    segStep env p ⟨er, eps, rng, st, traj⟩
      = ⟨er + (env.step st
          (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1,
         eps, (rngNext rng).2,
         (env.step st
           (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).1,
         traj.push
           { obs := env.observe st,
             action := sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1),
             reward := (env.step st
               (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1,
             value := (policyAndValue p (env.observe st)).2,
             oldLogp := (policyAndValue p (env.observe st)).1[
               sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1)]!.log,
             terminal := (env.step st
               (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 }⟩ := by
  simp only [segStep, if_neg h]

/-- **The runtime loop's `(state, rng)` is the abstract orbit.** The iterated
    `segStep`'s state and stream slots are the components of `stateAtR` — the
    joint-state counterpart of the a56 trace invariant. -/
theorem segStep_iterate_joint (env : Puffer.RL.Env S) (p : MLP) :
    ∀ (n : Nat) (er : Float) (eps : Array Float) (rng : UInt64) (st : S)
      (traj : Array Puffer.RL.NNTrain.Transition),
      ((((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.snd.snd.fst : S), ((),
        (((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.snd.fst : UInt64)))
        = stateAtR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) (st, (), rng) n := by
  intro n
  induction n with
  | zero => intro er eps rng st traj; rfl
  | succ n ih =>
    intro er eps rng st traj
    rw [Function.iterate_succ_apply, stateAtR_shift]
    by_cases h : (env.step st
        (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = true
    · rw [segStep_of_terminal env p h, ih, tickR_toLoop]
      simp [h]
    · rw [segStep_of_not_terminal env p h, ih, tickR_toLoop]
      have hflag : (env.step st
          (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = false :=
        Bool.eq_false_iff.mpr h
      simp [hflag]

/-- The runtime segment's persistent final state is the abstract orbit
    endpoint's state. -/
theorem segmentRollout_final_state (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.1
      = (stateAtR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) (s0, (), rng0) horizon).1 := by
  rw [segmentRollout_eq_iterate]
  exact congrArg Prod.fst (segStep_iterate_joint env p horizon 0.0 #[] rng0 s0 #[])

/-- The runtime segment's returned rng is the abstract orbit endpoint's stream
    position — the value the NEXT env's segment starts from. -/
theorem segmentRollout_final_rng (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.2.1
      = (stateAtR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) (s0, (), rng0) horizon).2.2 := by
  rw [segmentRollout_eq_iterate]
  exact congrArg (fun x : S × Unit × UInt64 => x.2.2)
    (segStep_iterate_joint env p horizon 0.0 #[] rng0 s0 #[])

/-- The bootstrap value is the value head at the final state's observation. -/
theorem segmentRollout_bootV (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.1
      = (policyAndValue p (env.observe
          (Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).2.2.1)).2 := by
  rw [segmentRollout_eq_iterate]
  rfl

/-! ### The vec loop as a fold -/

/-- The `MProd` accumulator of `vecRollout`'s desugared loop:
    `⟨bootVals, epReturns, newStates, rng, trajs⟩`. -/
abbrev VecState (S : Type) :=
  MProd (Array Float) (MProd (Array Float)
    (MProd (Array S) (MProd UInt64 (Array (Array Puffer.RL.NNTrain.Transition)))))

/-- One iteration of `vecRollout`'s loop body: run a segment from the current
    stream position, record its outputs, thread its final rng onward. -/
def vecStep (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (r : VecState S) (st : S) : VecState S :=
  ⟨r.fst.push (Puffer.RL.NNTrain.segmentRollout env p horizon st r.snd.snd.snd.fst).2.1,
   r.snd.fst ++ (Puffer.RL.NNTrain.segmentRollout env p horizon st r.snd.snd.snd.fst).2.2.2.2,
   r.snd.snd.fst.push (Puffer.RL.NNTrain.segmentRollout env p horizon st r.snd.snd.snd.fst).2.2.1,
   (Puffer.RL.NNTrain.segmentRollout env p horizon st r.snd.snd.snd.fst).2.2.2.1,
   r.snd.snd.snd.snd.push (Puffer.RL.NNTrain.segmentRollout env p horizon st r.snd.snd.snd.fst).1⟩

/-- The final read-out: `(trajs, bootVals, newStates, rng, epReturns)`. -/
def vecOut (a : VecState S) :
    Array (Array Puffer.RL.NNTrain.Transition) × Array Float × Array S × UInt64 × Array Float :=
  (a.snd.snd.snd.snd, a.fst, a.snd.snd.fst, a.snd.snd.snd.fst, a.snd.fst)

/-- **The vec do-loop is a fold** over the start states. -/
theorem vecRollout_eq_foldl (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Puffer.RL.NNTrain.vecRollout env p horizon states rng0
      = vecOut (states.toList.foldl (vecStep env p horizon) ⟨#[], #[], #[], rng0, #[]⟩) := by
  simp only [Puffer.RL.NNTrain.vecRollout, bind_pure_comp, map_pure,
    ← Array.forIn_toList, List.forIn_pure_yield_eq_foldl, Id.run_pure]
  rfl

/-! ### The induction and the capstone -/

/-- **Vec-fold invariant**: from any accumulator, folding the remaining start
    states appends, per env, the projected shared-stream trace, the orbit
    endpoint state, and the value head at it — with the stream threaded exactly
    as `vecSpec` says. Four coupled facts, one induction. -/
theorem vecFold_spec (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat) :
    ∀ (l : List S) (bootVals epRets : Array Float) (newStates : Array S)
      (rng : UInt64) (trajs : Array (Array Puffer.RL.NNTrain.Transition)),
      ((l.foldl (vecStep env p horizon)
          ⟨bootVals, epRets, newStates, rng, trajs⟩).snd.snd.snd.snd.toList.map
            (fun t => t.toList.map projRT)
        = trajs.toList.map (fun t => t.toList.map projRT)
          ++ (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon l rng).1.map
              (List.map projL))
      ∧ ((l.foldl (vecStep env p horizon)
          ⟨bootVals, epRets, newStates, rng, trajs⟩).snd.snd.fst.toList
        = newStates.toList
          ++ (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon l rng).2.1)
      ∧ ((l.foldl (vecStep env p horizon)
          ⟨bootVals, epRets, newStates, rng, trajs⟩).snd.snd.snd.fst
        = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon l rng).2.2)
      ∧ ((l.foldl (vecStep env p horizon)
          ⟨bootVals, epRets, newStates, rng, trajs⟩).fst.toList
        = bootVals.toList
          ++ (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon l rng).2.1.map
              (fun s => (policyAndValue p (env.observe s)).2)) := by
  intro l
  induction l with
  | nil =>
    intro bootVals epRets newStates rng trajs
    exact ⟨by simp, by simp, rfl, by simp⟩
  | cons s rest ih =>
    intro bootVals epRets newStates rng trajs
    rw [List.foldl_cons]
    have hv : vecStep env p horizon ⟨bootVals, epRets, newStates, rng, trajs⟩ s
        = ⟨bootVals.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.1,
           epRets ++ (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.2.2,
           newStates.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.1,
           (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.2.1,
           trajs.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).1⟩ := rfl
    rw [hv]
    obtain ⟨ih1, ih2, ih3, ih4⟩ := ih
      (bootVals.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.1)
      (epRets ++ (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.2.2)
      (newStates.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.1)
      ((Puffer.RL.NNTrain.segmentRollout env p horizon s rng).2.2.2.1)
      (trajs.push (Puffer.RL.NNTrain.segmentRollout env p horizon s rng).1)
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [ih1, segmentRollout_final_rng env p horizon s rng, Array.toList_push,
        List.map_append, vecSpec_cons]
      simp [segmentRollout_refinesR, List.append_assoc]
    · rw [ih2, segmentRollout_final_rng env p horizon s rng, Array.toList_push,
        vecSpec_cons]
      simp [segmentRollout_final_state, List.append_assoc]
    · rw [ih3, segmentRollout_final_rng env p horizon s rng, vecSpec_cons]
    · rw [ih4, segmentRollout_final_rng env p horizon s rng, Array.toList_push,
        vecSpec_cons]
      simp [segmentRollout_bootV, segmentRollout_final_state, List.append_assoc]

/-- **CAPSTONE: `vecRollout` refines the sequential shared-stream spec.** Every
    per-env buffer the multi-env collector records is, exactly, the projected
    `rolloutR` from its start state at its threaded stream position. -/
theorem vecRollout_refines (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1).toList.map
        (fun t => t.toList.map projRT)
      = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon
          states.toList rng0).1.map (List.map projL) := by
  rw [vecRollout_eq_foldl]
  simpa using (vecFold_spec env p horizon states.toList #[] #[] #[] rng0 #[]).1

/-- The persistent states handed to the next update are the abstract orbit
    endpoints. -/
theorem vecRollout_final_states (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).2.2.1).toList
      = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon
          states.toList rng0).2.1 := by
  rw [vecRollout_eq_foldl]
  simpa using (vecFold_spec env p horizon states.toList #[] #[] #[] rng0 #[]).2.1

/-- The rng handed to the next update is the spec's final stream position. -/
theorem vecRollout_final_rng (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    (Puffer.RL.NNTrain.vecRollout env p horizon states rng0).2.2.2.1
      = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon
          states.toList rng0).2.2 := by
  rw [vecRollout_eq_foldl]
  exact (vecFold_spec env p horizon states.toList #[] #[] #[] rng0 #[]).2.2.1

/-- The bootstrap values are the value head at the abstract endpoint states. -/
theorem vecRollout_bootVals (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).2.1).toList
      = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon
          states.toList rng0).2.1.map (fun s => (policyAndValue p (env.observe s)).2) := by
  rw [vecRollout_eq_foldl]
  simpa using (vecFold_spec env p horizon states.toList #[] #[] #[] rng0 #[]).2.2.2

/-- One buffer per env instance. -/
theorem vecRollout_num_segments (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1).size = states.size := by
  have h := congrArg List.length (vecRollout_refines env p horizon states rng0)
  simpa [vecSpec_num_traces] using h

/-- Every per-env buffer holds exactly `horizon` transitions — so the flattened
    experience buffer `buildBatch` consumes has exactly `numEnvs × horizon`
    entries. -/
theorem vecRollout_segment_size (env : Puffer.RL.Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    ∀ t ∈ (Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1, t.size = horizon := by
  intro t ht
  have hmem : (t.toList.map projRT)
      ∈ ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1).toList.map
          (fun u => u.toList.map projRT) :=
    List.mem_map_of_mem (by simpa using ht)
  rw [vecRollout_refines env p horizon states rng0] at hmem
  obtain ⟨tr, htr, heq⟩ := List.mem_map.mp hmem
  have hlen := congrArg List.length heq
  have htrlen : tr.length = horizon :=
    vecSpec_trace_length _ _ _ horizon states.toList rng0 tr htr
  simpa [htrlen] using hlen.symm

end Refinement

end Puffer.RL.Loop
