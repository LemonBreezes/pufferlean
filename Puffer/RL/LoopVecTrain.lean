/-
The Float↔ℝ refinement connecting the RUNNABLE trainer's rollout to the
abstract env/policy loop.

`NNTrain.segmentRollout` (`Puffer/RL/VecTrain.lean`) is the executable
data-collection phase: one env instance stepped `horizon` times under the MLP
policy, auto-resetting on terminal, recording Float `Transition`s into the
experience buffer. `Loop.rollout` (`Puffer/RL/Loop.lean`) is the ℝ-level spec:
two Mealy machines wired head-to-tail, the trace as the orbit log.

This module proves they are the SAME object. The key observation: the rollout
phase RECORDS numbers, it does not compute with them — so the refinement is
EXACT (list equality after projecting to the common fields, rewards embedded by
the trusted `FloatR.toReal`), with no (1+δ) error terms anywhere. The Float gap
only opens downstream, where GAE/PPO *arithmetic* consumes the buffer — and
those kernels already carry their own verified error bounds (`GAERuntime`,
`GAEValueTargets`, …).

Scope note (why deterministic reset): `segmentRollout` threads ONE rng stream
through both the policy's action sampling and the env's auto-resets. An env
whose `reset` consumes rng therefore breaks the env/policy state split that
`Loop` models — the rng ping-pongs between the two machines. For envs with
DETERMINISTIC reset (`reset r = (sInit, r)` — squared and every fixed-start
env), the stream belongs entirely to the policy and the split is
exact. That hypothesis (`hreset`) is the honest boundary of this refinement;
rng-consuming resets need a joint-state generalization of `Loop.Env` (future).

Refinement chain:
  `segmentRollout`  =  `segOut ∘ (segStep)^[horizon]`     (`segmentRollout_eq_iterate`,
                                                           do-loop → iterated map)
  `(segStep)^[n]` trace  =  `Loop.rollout` trace           (`segStep_iterate_spec`,
                                                           induction along the orbit)
  capstone `segmentRollout_refines` (generic over the env `Model`), plus
  transfers: buffer size
  (`segmentRollout_size`), reward-stream identification (`segmentRollout_rewards`)
  and the horizon-independent return bound (`segmentRollout_return_bounded`).
-/
import Puffer.RL.Loop
import Puffer.RL.VecTrain
import Puffer.Float.Basic

namespace Puffer.RL.Loop

open Puffer.RL.Train (rngNext uniform01 sampleCat)
open Puffer.RL.NNTrain (MLP policyAndValue)

variable {S : Type}

/-! ### The abstract instances the runtime refines -/

/-- The runtime env (`Puffer.RL.Env`, Float rewards) as a `Loop.Env`: same
    state, same observation function, reward embedded into ℝ by the trusted
    `toReal`. `sInit` is the (deterministic) reset state. -/
noncomputable def toLoopEnv (env : Puffer.RL.Env S) (sInit : S) : Env S (Array Float) Nat where
  init := sInit
  obs := env.observe
  step s a :=
    ((env.step s a).1, Puffer.FloatR.toReal (env.step s a).2.1, (env.step s a).2.2)

/-- The runtime MLP + categorical sampler as a `Loop.Policy`: the internal
    state `H` is exactly the rng word (`splitmix64` stream), advanced once per
    action — precisely what `segmentRollout` does. -/
def toLoopPolicy (p : MLP) (seed : UInt64) : Policy (Array Float) Nat UInt64 where
  init := seed
  act h o :=
    (sampleCat (policyAndValue p o).1 (uniform01 (rngNext h).1), (rngNext h).2)

/-- One tick of the abstract composite, computed: observe, sample, step,
    auto-reset — definitionally. -/
theorem tick_toLoop (env : Puffer.RL.Env S) (p : MLP) (seed : UInt64) (sInit : S)
    (st : S) (rng : UInt64) :
    tick ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p seed) (st, rng)
      = (⟨st, env.observe st,
          sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1),
          Puffer.FloatR.toReal (env.step st
            (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1,
          (env.step st
            (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2⟩,
         ((if (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2
           then sInit
           else (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).1),
          (rngNext rng).2)) := rfl

/-! ### The runtime loop as an iterated map -/

/-- The `MProd` accumulator of `segmentRollout`'s desugared loop, in elaboration
    order: `⟨epRet, epReturns, rng, st, traj⟩`. -/
abbrev SegState (S : Type) :=
  MProd Float (MProd (Array Float) (MProd UInt64 (MProd S (Array Puffer.RL.NNTrain.Transition))))

/-- One iteration of `segmentRollout`'s loop body, as a pure map on the
    accumulator — a transcription of the elaborated `do`-block. -/
def segStep (env : Puffer.RL.Env S) (p : MLP) (r : SegState S) : SegState S :=
  let st := r.snd.snd.snd.fst
  let rng := r.snd.snd.fst
  let obs := env.observe st
  let a := sampleCat (policyAndValue p obs).1 (uniform01 (rngNext rng).1)
  let out := env.step st a
  let traj' := r.snd.snd.snd.snd.push
    { obs := obs, action := a, reward := out.2.1,
      value := (policyAndValue p obs).2,
      oldLogp := (policyAndValue p obs).1[a]!.log,
      terminal := out.2.2 }
  if out.2.2 = true then
    ⟨0.0, r.snd.fst.push (r.fst + out.2.1),
     (env.reset (rngNext rng).2).2, (env.reset (rngNext rng).2).1, traj'⟩
  else
    ⟨r.fst + out.2.1, r.snd.fst, (rngNext rng).2, out.1, traj'⟩

/-- The final read-out of the loop accumulator: `(traj, bootValue, state, rng,
    episodeReturns)`, with the bootstrap value `V(observe finalState)`. -/
def segOut (env : Puffer.RL.Env S) (p : MLP) (a : SegState S) :
    Array Puffer.RL.NNTrain.Transition × Float × S × UInt64 × Array Float :=
  (a.snd.snd.snd.snd, (policyAndValue p (env.observe a.snd.snd.snd.fst)).2,
   a.snd.snd.snd.fst, a.snd.snd.fst, a.snd.fst)

/-- A `forIn` whose body ignores the index and always yields (an
    if-then-else of yields) is the iterate of the branch map. The general
    reduction for `segmentRollout`-shaped loops, where the branching prevents
    `List.forIn_pure_yield_eq_foldl` from applying directly. -/
theorem forIn_ite_yield_iterate {β : Type} (c : β → Prop) [DecidablePred c]
    (f₁ f₂ : β → β) :
    ∀ (l : List Nat) (init : β),
      (forIn (m := Id) l init (fun _ r =>
          if c r then pure (ForInStep.yield (f₁ r)) else pure (ForInStep.yield (f₂ r))))
        = (fun r => if c r then f₁ r else f₂ r)^[l.length] init := by
  intro l
  induction l with
  | nil => intro init; rfl
  | cons x xs ih =>
    intro init
    rw [List.forIn_cons]
    by_cases h : c init
    · simp only [if_pos h, pure_bind, List.length_cons, Function.iterate_succ_apply]
      exact ih (f₁ init)
    · simp only [if_neg h, pure_bind, List.length_cons, Function.iterate_succ_apply]
      exact ih (f₂ init)

/-- **The do-loop is the iterated map.** `segmentRollout` equals `segOut` of
    `horizon` applications of `segStep` — the executable loop, freed from its
    monadic packaging. -/
theorem segmentRollout_eq_iterate (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0
      = segOut env p ((segStep env p)^[horizon] ⟨0.0, #[], rng0, s0, #[]⟩) := by
  simp only [Puffer.RL.NNTrain.segmentRollout, bind_pure_comp, map_pure,
    Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_one_sub_one, Nat.div_one]
  rw [forIn_ite_yield_iterate, List.length_range']
  rfl

/-! ### The refinement -/

/-- Observable content of a recorded runtime transition: `(obs, action,
    ℝ-reward, terminal)` — the fields the buffer shares with the spec.
    (`value`/`oldLogp` are functions of `obs`/`action` and the fixed policy.) -/
noncomputable def projRT (tr : Puffer.RL.NNTrain.Transition) : Array Float × Nat × ℝ × Bool :=
  (tr.obs, tr.action, Puffer.FloatR.toReal tr.reward, tr.terminal)

/-- Observable content of an abstract transition. -/
def projL (tr : Transition S (Array Float) Nat) : Array Float × Nat × ℝ × Bool :=
  (tr.o, tr.a, tr.r, tr.done)

/-- **Loop invariant: the iterated runtime map logs the abstract orbit.** After
    `n` iterations from any accumulator, the recorded trajectory is the initial
    trajectory plus (the projection of) the abstract `rollout` from the same
    `(state, rng)` — for any env with deterministic reset. -/
theorem segStep_iterate_spec (env : Puffer.RL.Env S) (p : MLP) (seed : UInt64)
    (sInit : S) (hreset : ∀ r, env.reset r = (sInit, r)) :
    ∀ (n : Nat) (er : Float) (eps : Array Float) (rng : UInt64) (st : S)
      (traj : Array Puffer.RL.NNTrain.Transition),
      ((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.snd.snd.snd.toList.map projRT
        = traj.toList.map projRT
          ++ (rollout ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p seed) n
              (st, rng)).map projL := by
  intro n
  induction n with
  | zero =>
    intro er eps rng st traj
    simp
  | succ n ih =>
    intro er eps rng st traj
    rw [Function.iterate_succ_apply, rollout_succ, List.map_cons, tick_toLoop]
    by_cases h : (env.step st
        (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = true
    · have hstep : segStep env p ⟨er, eps, rng, st, traj⟩
          = ⟨0.0, eps.push (er + (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1),
             (rngNext rng).2, sInit,
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
        simp only [segStep, if_pos h, hreset]
      rw [hstep, ih, Array.toList_push, List.map_append, h]
      simp [projRT, projL, List.append_assoc]
    · have hstep : segStep env p ⟨er, eps, rng, st, traj⟩
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
      rw [hstep, ih, Array.toList_push, List.map_append]
      have hflag : (env.step st
          (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = false :=
        Bool.eq_false_iff.mpr h
      rw [hflag]
      simp [projRT, projL, List.append_assoc]

/-- **CAPSTONE: the runnable rollout refines the abstract loop, exactly.** For
    any env with deterministic reset, the buffer `segmentRollout` records is —
    field for field, in order — the abstract `Loop.rollout` of the induced
    env/policy pair (rewards embedded in ℝ). No error terms: collection is
    exact; only downstream arithmetic pays Float rent. -/
theorem segmentRollout_refines (env : Puffer.RL.Env S) (p : MLP) (sInit : S)
    (hreset : ∀ r, env.reset r = (sInit, r)) (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map projRT
      = (rollout ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p rng0) horizon
          (s0, rng0)).map projL := by
  rw [segmentRollout_eq_iterate]
  simpa using segStep_iterate_spec env p rng0 sInit hreset horizon 0.0 #[] rng0 s0 #[]

/-! ### Transfers: Loop theorems, now about the executable buffer -/

/-- The buffer holds exactly `horizon` transitions (`rollout_length`,
    transported). -/
theorem segmentRollout_size (env : Puffer.RL.Env S) (p : MLP) (sInit : S)
    (hreset : ∀ r, env.reset r = (sInit, r)) (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).size = horizon := by
  have h := congrArg List.length (segmentRollout_refines env p sInit hreset horizon s0 rng0)
  simpa using h

/-- The ℝ-embedded reward stream of the executable buffer IS the abstract
    trace's reward stream. -/
theorem segmentRollout_rewards (env : Puffer.RL.Env S) (p : MLP) (sInit : S)
    (hreset : ∀ r, env.reset r = (sInit, r)) (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map
        (fun tr => Puffer.FloatR.toReal tr.reward)
      = rewards (rollout ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p rng0) horizon
          (s0, rng0)) := by
  have h := congrArg (List.map (fun x : Array Float × Nat × ℝ × Bool => x.2.2.1))
    (segmentRollout_refines env p sInit hreset horizon s0 rng0)
  simpa [List.map_map, projRT, projL, rewards, Function.comp] using h

/-- **The executable buffer's discounted return obeys the spec-level geometric
    bound**: rewards uniformly bounded by `R` (in ℝ) and `w ∈ [0,1)` give
    `|return| ≤ R/(1−w)` for every horizon, policy, and seed —
    `traceReturn_bounded`, now a statement about `segmentRollout`'s output. -/
theorem segmentRollout_return_bounded (env : Puffer.RL.Env S) (p : MLP) (sInit : S)
    (hreset : ∀ r, env.reset r = (sInit, r)) {w R : ℝ}
    (hw0 : 0 ≤ w) (hw1 : w < 1) (hR : 0 ≤ R)
    (hstep : ∀ (s : S) (a : Nat), |Puffer.FloatR.toReal (env.step s a).2.1| ≤ R)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    |GAE.discountedReturn w
        (((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map
          (fun tr => Puffer.FloatR.toReal tr.reward))|
      ≤ R / (1 - w) := by
  rw [segmentRollout_rewards env p sInit hreset horizon s0 rng0]
  exact traceReturn_bounded ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p rng0)
    hw0 hw1 hR (fun s a => hstep s a) horizon (s0, rng0)

end Puffer.RL.Loop
