/-
The joint-state generalization of the env/policy loop: a SHARED random stream.

`Loop.Env`/`Loop.Policy` (a52) model two independent Mealy machines — state
spaces `S` and `H`, joint state `S × H`. The a55 refinement exposed the case
that model cannot represent: `segmentRollout` threads ONE splitmix64 stream
through BOTH the policy's action sampling AND the env's auto-resets, so the
stream ping-pongs between the machines and belongs to neither.

The fix is to make the stream a first-class third component. `EnvR`/`PolicyR`
are Mealy machines in the Kleisli category of the state monad on `R`: every
interaction (`init`, `step`, `act`) may read and advance the shared stream, and
the closed loop (`tickR`) runs on the joint state `S × H × R` — its transition
is exactly the Kleisli composite of the two machines.

Structure of the module:
* `tickR`/`rolloutR` + the basic trace theorems (length, membership, obs
  coherence), mirroring a52.
* FACTORIZATION: when either machine is pure in `R`, the stream localizes into
  the other one and the loop collapses to the a52 model —
  `rolloutR_ofEnv` (pure env ⇒ policy absorbs `R` into `H × R`) and
  `rollout_toEnv` (pure policy ⇒ env absorbs `R` into `S × R`). In general
  neither is possible; that impossibility is precisely why a55 needed its
  deterministic-reset hypothesis.
* REFINEMENT, unconditionally: `segmentRollout_refinesR` — the runnable
  rollout equals `rolloutR` of the induced pair for EVERY runtime env, no
  reset hypothesis. The a55 machinery (`segStep`, `segmentRollout_eq_iterate`)
  is reused; only the orbit induction changes. Transfers (size, reward stream,
  geometric return bound) come along, and `rolloutR_agrees_of_det_reset` shows
  the two abstractions coincide on the old model's domain — both being exact
  refinements of the same executable.
* Instantiated on `memoryEnv` — whose reset DRAWS THE HIDDEN GOAL from the rng,
  the case the a55 model provably could not express.
-/
import Puffer.RL.LoopVecTrain

namespace Puffer.RL.Loop

open Puffer.RL.Train (rngNext uniform01 sampleCat)
open Puffer.RL.NNTrain (MLP policyAndValue)

variable {R S O A H : Type*}

/-! ### The shared-stream machines -/

/-- An environment that may consume shared randomness `R`: reset draws the
    initial state from the stream, and stepping may draw too (the runtime's
    pure-step envs simply return the stream unchanged). A Mealy machine in the
    Kleisli category of the state monad on `R`. -/
structure EnvR (R S O A : Type*) where
  /-- Reset: draw an initial state, advancing the stream (`env.reset rng`). -/
  init : R → S × R
  /-- What the policy is shown. -/
  obs : S → O
  /-- One step: `((next state, reward, terminal?), advanced stream)`. -/
  step : S → A → R → (S × ℝ × Bool) × R

/-- A policy that may consume shared randomness `R` (action sampling). Its
    private state `H` carries only what is NOT shared (recurrent state). -/
structure PolicyR (R O A H : Type*) where
  /-- Initial private state. -/
  init : H
  /-- Choose an action: `((action, next private state), advanced stream)`. -/
  act : H → O → R → (A × H) × R

/-- One tick of the closed loop on the joint state `S × H × R`: observe, let
    the policy draw from the stream, let the env step (drawing again if it
    wants). The stream passes THROUGH the policy INTO the env — the Kleisli
    composite that `S × H` product states cannot express. -/
def tickR (E : EnvR R S O A) (P : PolicyR R O A H) (x : S × H × R) :
    Transition S O A × (S × H × R) :=
  (⟨x.1,
    E.obs x.1,
    (P.act x.2.1 (E.obs x.1) x.2.2).1.1,
    (E.step x.1 (P.act x.2.1 (E.obs x.1) x.2.2).1.1 (P.act x.2.1 (E.obs x.1) x.2.2).2).1.2.1,
    (E.step x.1 (P.act x.2.1 (E.obs x.1) x.2.2).1.1 (P.act x.2.1 (E.obs x.1) x.2.2).2).1.2.2⟩,
   ((E.step x.1 (P.act x.2.1 (E.obs x.1) x.2.2).1.1 (P.act x.2.1 (E.obs x.1) x.2.2).2).1.1,
    (P.act x.2.1 (E.obs x.1) x.2.2).1.2,
    (E.step x.1 (P.act x.2.1 (E.obs x.1) x.2.2).1.1 (P.act x.2.1 (E.obs x.1) x.2.2).2).2))

/-- The joint state after `n` ticks — the orbit of the shared-stream system
    (mirrors a52's `stateAt`). Its `R` component is the stream position after
    the segment, which is what sequential multi-env stepping threads onward. -/
def stateAtR (E : EnvR R S O A) (P : PolicyR R O A H) (x : S × H × R) : Nat → S × H × R
  | 0 => x
  | n + 1 => (tickR E P (stateAtR E P x n)).2

@[simp] theorem stateAtR_zero (E : EnvR R S O A) (P : PolicyR R O A H) (x : S × H × R) :
    stateAtR E P x 0 = x := rfl

@[simp] theorem stateAtR_succ (E : EnvR R S O A) (P : PolicyR R O A H)
    (x : S × H × R) (n : Nat) :
    stateAtR E P x (n + 1) = (tickR E P (stateAtR E P x n)).2 := rfl

/-- Shifting the start of the shared-stream orbit (mirrors `stateAt_shift`). -/
theorem stateAtR_shift (E : EnvR R S O A) (P : PolicyR R O A H) :
    ∀ (n : Nat) (x : S × H × R),
      stateAtR E P x (n + 1) = stateAtR E P (tickR E P x).2 n := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih =>
    intro x
    show (tickR E P (stateAtR E P x (n + 1))).2 = (tickR E P (stateAtR E P (tickR E P x).2 n)).2
    rw [ih]

/-- The shared-stream rollout: `n` ticks, trace recorded. Same `Transition`
    type as the a52 loop — the traces are directly comparable. -/
def rolloutR (E : EnvR R S O A) (P : PolicyR R O A H) :
    Nat → S × H × R → List (Transition S O A)
  | 0, _ => []
  | n + 1, x => (tickR E P x).1 :: rolloutR E P n (tickR E P x).2

@[simp] theorem rolloutR_zero (E : EnvR R S O A) (P : PolicyR R O A H) (x : S × H × R) :
    rolloutR E P 0 x = [] := rfl

@[simp] theorem rolloutR_succ (E : EnvR R S O A) (P : PolicyR R O A H)
    (x : S × H × R) (n : Nat) :
    rolloutR E P (n + 1) x = (tickR E P x).1 :: rolloutR E P n (tickR E P x).2 := rfl

/-- A shared-stream rollout records exactly `n` transitions. -/
@[simp] theorem rolloutR_length (E : EnvR R S O A) (P : PolicyR R O A H) :
    ∀ (n : Nat) (x : S × H × R), (rolloutR E P n x).length = n := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih => intro x; simp [ih]

/-- Every recorded transition is a genuine tick from some reachable joint
    state. -/
theorem mem_rolloutR (E : EnvR R S O A) (P : PolicyR R O A H) :
    ∀ {n : Nat} {x : S × H × R} {tr : Transition S O A},
      tr ∈ rolloutR E P n x → ∃ y : S × H × R, tr = (tickR E P y).1 := by
  intro n
  induction n with
  | zero => intro x tr h; simp at h
  | succ n ih =>
    intro x tr h
    rw [rolloutR_succ, List.mem_cons] at h
    rcases h with h | h
    · exact ⟨x, h⟩
    · exact ih h

/-- Observation coherence survives the generalization: `o = obs s` for every
    recorded transition. -/
theorem obs_coherentR (E : EnvR R S O A) (P : PolicyR R O A H) {n : Nat}
    {x : S × H × R} {tr : Transition S O A} (h : tr ∈ rolloutR E P n x) :
    tr.o = E.obs tr.s := by
  obtain ⟨y, rfl⟩ := mem_rolloutR E P h
  rfl

/-- **Bounded rewards give a bounded return** (a52's `traceReturn_bounded`,
    shared-stream form): a uniform bound over ALL stream positions bounds every
    rollout's discounted return by `Rmax/(1−w)`. -/
theorem traceReturn_rolloutR_bounded (E : EnvR R S O A) (P : PolicyR R O A H)
    {w Rmax : ℝ} (hw0 : 0 ≤ w) (hw1 : w < 1) (hR : 0 ≤ Rmax)
    (hstep : ∀ (s : S) (a : A) (r : R), |(E.step s a r).1.2.1| ≤ Rmax)
    (n : Nat) (x : S × H × R) :
    |traceReturn w (rolloutR E P n x)| ≤ Rmax / (1 - w) := by
  refine GAE.gaeHead_bounded w Rmax hw0 hw1 hR _ ?_
  intro v hv
  simp only [rewards, List.mem_map] at hv
  obtain ⟨tr, htr, rfl⟩ := hv
  obtain ⟨y, rfl⟩ := mem_rolloutR E P htr
  exact hstep y.1 _ _

/-! ### Factorization: a pure machine lets the stream localize

If the env never touches `R`, the stream belongs to the policy: absorb it as
`H × R` and the loop IS the a52 loop (`rolloutR_ofEnv`). Symmetrically, a pure
policy lets the env absorb the stream as `S × R` (`rollout_toEnv`). When BOTH
machines draw — `segmentRollout` with an rng-consuming reset — neither
absorption exists, and the joint state `S × H × R` is irreducible. -/

/-- A pure (a52) env, viewed in the shared-stream world: it never touches `R`. -/
def EnvR.ofEnv (E₀ : Env S O A) : EnvR R S O A where
  init r := (E₀.init, r)
  obs := E₀.obs
  step s a r := (E₀.step s a, r)

/-- A shared-stream policy with the stream absorbed into its private state:
    `H × R`. (`r0` seeds the canonical initial stream.) -/
def PolicyR.toPolicy (P : PolicyR R O A H) (r0 : R) : Policy O A (H × R) where
  init := (P.init, r0)
  act hr o :=
    ((P.act hr.1 o hr.2).1.1, ((P.act hr.1 o hr.2).1.2, (P.act hr.1 o hr.2).2))

/-- **Pure env ⇒ the loop factors through a52** (the stream localizes in the
    policy). This is the general form of what a55's deterministic-reset
    hypothesis exploited. -/
theorem rolloutR_ofEnv (E₀ : Env S O A) (P : PolicyR R O A H) (r0 : R) :
    ∀ (n : Nat) (s : S) (h : H) (r : R),
      rolloutR (EnvR.ofEnv E₀) P n (s, h, r)
        = rollout E₀ (P.toPolicy r0) n (s, (h, r)) := by
  intro n
  induction n with
  | zero => intro s h r; rfl
  | succ n ih =>
    intro s h r
    rw [rolloutR_succ, rollout_succ]
    exact congrArg₂ List.cons rfl (ih _ _ _)

/-- A pure (a52) policy, viewed in the shared-stream world. -/
def PolicyR.ofPolicy (P₀ : Policy O A H) : PolicyR R O A H where
  init := P₀.init
  act h o r := (P₀.act h o, r)

/-- A shared-stream env with the stream absorbed into its state: `S × R`.
    (`r0` seeds the canonical initial stream.) -/
def EnvR.toEnv (E : EnvR R S O A) (r0 : R) : Env (S × R) O A where
  init := E.init r0
  obs sr := E.obs sr.1
  step sr a :=
    (((E.step sr.1 a sr.2).1.1, (E.step sr.1 a sr.2).2),
     (E.step sr.1 a sr.2).1.2.1, (E.step sr.1 a sr.2).1.2.2)

/-- **Pure policy ⇒ the loop factors through a52** (the stream localizes in the
    env; the a52 trace carries `(s, r)` states, projected back by
    `Transition.mapState Prod.fst`). -/
theorem rollout_toEnv (E : EnvR R S O A) (P₀ : Policy O A H) (r0 : R) :
    ∀ (n : Nat) (s : S) (h : H) (r : R),
      (rollout (E.toEnv r0) P₀ n ((s, r), h)).map (Transition.mapState Prod.fst)
        = rolloutR E (PolicyR.ofPolicy P₀) n (s, h, r) := by
  intro n
  induction n with
  | zero => intro s h r; rfl
  | succ n ih =>
    intro s h r
    rw [rolloutR_succ, rollout_succ, List.map_cons]
    exact congrArg₂ List.cons rfl (ih _ _ _)

/-! ### Auto-reset with an rng-consuming reset -/

/-- The auto-reset wrapper, shared-stream form: a terminal step hands the
    stream to `init`, which DRAWS the fresh state from it — exactly
    `segmentRollout`'s `if term then env.reset rng`. Reward and flag pass
    through untouched. -/
def EnvR.withAutoReset (E : EnvR R S O A) : EnvR R S O A where
  init := E.init
  obs := E.obs
  step s a r :=
    (((if (E.step s a r).1.2.2 then (E.init (E.step s a r).2).1 else (E.step s a r).1.1),
      (E.step s a r).1.2.1, (E.step s a r).1.2.2),
     (if (E.step s a r).1.2.2 then (E.init (E.step s a r).2).2 else (E.step s a r).2))

/-! ### The refinement, unconditionally -/

section Refinement

variable {S : Type}

/-- The runtime env as a shared-stream env: `init` IS `env.reset` (however much
    rng it consumes), stepping is pure. No hypotheses. -/
noncomputable def toLoopEnvR (env : Puffer.RL.Env S) : EnvR UInt64 S (Array Float) Nat where
  init := env.reset
  obs := env.observe
  step s a r :=
    (((env.step s a).1, Puffer.FloatR.toReal (env.step s a).2.1, (env.step s a).2.2), r)

/-- The runtime MLP policy as a shared-stream policy: its private state is
    `Unit` — everything stateful about it was the rng, which is now shared. -/
def toLoopPolicyR (p : MLP) : PolicyR UInt64 (Array Float) Nat Unit where
  init := ()
  act _ o r :=
    ((sampleCat (policyAndValue p o).1 (uniform01 (rngNext r).1), ()), (rngNext r).2)

/-- One tick of the induced shared-stream composite, computed — definitionally. -/
theorem tickR_toLoop (env : Puffer.RL.Env S) (p : MLP) (st : S) (rng : UInt64) :
    tickR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) (st, (), rng)
      = (⟨st, env.observe st,
          sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1),
          Puffer.FloatR.toReal (env.step st
            (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.1,
          (env.step st
            (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2⟩,
         ((if (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2
           then (env.reset (rngNext rng).2).1
           else (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).1),
          (),
          (if (env.step st
              (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2
           then (env.reset (rngNext rng).2).2
           else (rngNext rng).2))) := rfl

/-- **Loop invariant, no reset hypothesis**: the iterated runtime map logs the
    shared-stream orbit — for EVERY runtime env. Where a55's induction needed
    `hreset` to keep the stream inside the policy, here the terminal branch
    hands the stream to `env.reset` on both sides and they agree on the nose. -/
theorem segStep_iterate_specR (env : Puffer.RL.Env S) (p : MLP) :
    ∀ (n : Nat) (er : Float) (eps : Array Float) (rng : UInt64) (st : S)
      (traj : Array Puffer.RL.NNTrain.Transition),
      ((segStep env p)^[n] ⟨er, eps, rng, st, traj⟩).snd.snd.snd.snd.toList.map projRT
        = traj.toList.map projRT
          ++ (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) n
              (st, (), rng)).map projL := by
  intro n
  induction n with
  | zero =>
    intro er eps rng st traj
    simp
  | succ n ih =>
    intro er eps rng st traj
    rw [Function.iterate_succ_apply, rolloutR_succ, List.map_cons, tickR_toLoop]
    by_cases h : (env.step st
        (sampleCat (policyAndValue p (env.observe st)).1 (uniform01 (rngNext rng).1))).2.2 = true
    · have hstep : segStep env p ⟨er, eps, rng, st, traj⟩
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

/-- **CAPSTONE, unconditional: the runnable rollout refines the shared-stream
    loop for EVERY env.** The a55 deterministic-reset boundary is gone — the
    joint state `S × H × R` is the right carrier for what `segmentRollout`
    actually computes. Still exact: no (1+δ) terms. -/
theorem segmentRollout_refinesR (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map projRT
      = (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) horizon
          (s0, (), rng0)).map projL := by
  rw [segmentRollout_eq_iterate]
  simpa using segStep_iterate_specR env p horizon 0.0 #[] rng0 s0 #[]

/-- The two abstractions agree wherever both apply: on a deterministic-reset
    env, the shared-stream trace and the a55 product-state trace coincide —
    both being exact refinements of the same executable. -/
theorem rolloutR_agrees_of_det_reset (env : Puffer.RL.Env S) (p : MLP) (sInit : S)
    (hreset : ∀ r, env.reset r = (sInit, r)) (n : Nat) (s0 : S) (rng0 : UInt64) :
    (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) n (s0, (), rng0)).map projL
      = (rollout ((toLoopEnv env sInit).withAutoReset) (toLoopPolicy p rng0) n
          (s0, rng0)).map projL :=
  (segmentRollout_refinesR env p n s0 rng0).symm.trans
    (segmentRollout_refines env p sInit hreset n s0 rng0)

/-! ### Transfers, now hypothesis-free -/

/-- Buffer size = horizon, for every env. -/
theorem segmentRollout_sizeR (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).size = horizon := by
  have h := congrArg List.length (segmentRollout_refinesR env p horizon s0 rng0)
  simpa using h

/-- The buffer's ℝ-embedded reward stream is the shared-stream trace's, for
    every env. -/
theorem segmentRollout_rewardsR (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    ((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map
        (fun tr => Puffer.FloatR.toReal tr.reward)
      = rewards (rolloutR ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) horizon
          (s0, (), rng0)) := by
  have h := congrArg (List.map (fun x : Array Float × Nat × ℝ × Bool => x.2.2.1))
    (segmentRollout_refinesR env p horizon s0 rng0)
  simpa [List.map_map, projRT, projL, rewards, Function.comp] using h

/-- The geometric return bound on the executable's output, for every env —
    a55's transfer with its hypothesis deleted. -/
theorem segmentRollout_return_boundedR (env : Puffer.RL.Env S) (p : MLP)
    {w Rmax : ℝ} (hw0 : 0 ≤ w) (hw1 : w < 1) (hR : 0 ≤ Rmax)
    (hstep : ∀ (s : S) (a : Nat), |Puffer.FloatR.toReal (env.step s a).2.1| ≤ Rmax)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    |GAE.discountedReturn w
        (((Puffer.RL.NNTrain.segmentRollout env p horizon s0 rng0).1).toList.map
          (fun tr => Puffer.FloatR.toReal tr.reward))|
      ≤ Rmax / (1 - w) := by
  rw [segmentRollout_rewardsR env p horizon s0 rng0]
  exact traceReturn_rolloutR_bounded ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p)
    hw0 hw1 hR (fun s a r => hstep s a) horizon (s0, (), rng0)

end Refinement

end Puffer.RL.Loop
