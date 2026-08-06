/-
The abstract env/policy interaction loop, over ℝ.

This is the mathematical object at the center of PufferLib: an environment
(a Mealy machine / coalgebra `step : S → A → S × ℝ × Bool` with an observation
map `obs : S → O`) wired head-to-tail with a policy (a Mealy machine
`act : H → O → A × H`; any sampling randomness lives inside `H` as an explicit
rng word, exactly as in the C runtime where `rng` is an env/trainer struct
field). One `tick` of the closed system advances `S × H` and emits a recorded
`Transition`; a `rollout` is the length-`n` trace of ticks — precisely the
data-collection phase of `torch_pufferl.py::rollouts` (the `for t in
range(horizon)` loop) and `src/pufferlib.cu`, with auto-reset available as the
`Env.withAutoReset` wrapper (what `src/vecenv.h` does when a sub-env terminates).

Headline theorems:
* `rollout_getElem?` — the trace is exactly the orbit of the composite
  dynamical system: entry `t` is the tick taken at the `t`-th iterate of the
  closed-loop transition map (`stateAt_eq_iterate`).
* `traceReturn_eq_geoSum` / `traceReturn_rollout_succ` — the discounted return
  of a rollout is the geometric sum `Σ_t wᵗ·r_t` and satisfies the Bellman-style
  recurrence, connecting the loop to the verified GAE module.
* `rollout_hom` — env morphisms (coalgebra morphisms) preserve traces: behavior
  is invariant under state relabeling, so the trainer only ever depends on the
  env's observable behavior, never its state representation.
* `traceReturn_prod` — running two envs as one product system (vectorization in
  miniature) yields the componentwise rollouts and the summed return.
-/
import Mathlib
import Puffer.RL.GAE

namespace Puffer.RL.Loop

variable {S S' O A H : Type*}

/-- An ENVIRONMENT over state `S`, observations `O`, actions `A` — the abstract
    contract behind `ocean/*/…​.h` (`c_reset`/`c_step` + `compute_observations`).
    Deterministic: the C envs' randomness is an explicit `rng` word inside the
    state struct, so `step` is a pure function. As a coalgebra this is a Mealy
    machine `S × A → S × (O × ℝ × Bool)` with output factored through `obs`. -/
structure Env (S O A : Type*) where
  /-- Initial state (`c_reset` from a fixed seed). -/
  init : S
  /-- What the policy is shown (`compute_observations`). -/
  obs : S → O
  /-- One env tick: `(next state, reward, terminal?)` (`c_step`). -/
  step : S → A → S × ℝ × Bool

/-- A POLICY over observations `O`, actions `A`, internal state `H` — the
    abstract contract behind `policy.forward_eval` + `sample_logits` in
    `torch_pufferl.py`. `H` carries everything the policy threads forward:
    recurrent (LSTM) state AND the sampler's rng word, so `act` is pure. -/
structure Policy (O A H : Type*) where
  /-- Initial internal state (zeroed LSTM state + rng seed). -/
  init : H
  /-- Choose an action from what is seen: `(action, next internal state)`. -/
  act : H → O → A × H

/-- One recorded step of the interaction — the tuple the rollout buffer stores
    (`observations[t], actions[t], rewards[t], terminals[t]` in
    `torch_pufferl.py::rollouts`), plus the pre-step env state `s` for spec
    purposes (the C buffers store only its image `o = obs s`). -/
structure Transition (S O A : Type*) where
  /-- Env state the action was taken in. -/
  s : S
  /-- Observation shown to the policy (provably `= obs s`: `obs_coherent`). -/
  o : O
  /-- Action the policy chose. -/
  a : A
  /-- Reward emitted by the env step. -/
  r : ℝ
  /-- Terminal flag emitted by the env step. -/
  done : Bool

/-- One TICK of the closed loop: show `obs`, let the policy act, step the env.
    Returns the recorded `Transition` and the advanced joint state — the loop
    body of `torch_pufferl.py::rollouts`. The composite of the two open Mealy
    machines is a CLOSED dynamical system on `S × H`; `tick` is its transition
    map paired with its output. (Projection style throughout so that the
    coherence theorems below hold by `rfl`.) -/
def tick (E : Env S O A) (P : Policy O A H) (sh : S × H) : Transition S O A × (S × H) :=
  (⟨sh.1,
    E.obs sh.1,
    (P.act sh.2 (E.obs sh.1)).1,
    (E.step sh.1 (P.act sh.2 (E.obs sh.1)).1).2.1,
    (E.step sh.1 (P.act sh.2 (E.obs sh.1)).1).2.2⟩,
   ((E.step sh.1 (P.act sh.2 (E.obs sh.1)).1).1,
    (P.act sh.2 (E.obs sh.1)).2))

/-- The joint state after `n` ticks — the orbit of the closed-loop dynamical
    system. -/
def stateAt (E : Env S O A) (P : Policy O A H) (sh : S × H) : Nat → S × H
  | 0 => sh
  | n + 1 => (tick E P (stateAt E P sh n)).2

@[simp] theorem stateAt_zero (E : Env S O A) (P : Policy O A H) (sh : S × H) :
    stateAt E P sh 0 = sh := rfl

@[simp] theorem stateAt_succ (E : Env S O A) (P : Policy O A H) (sh : S × H) (n : Nat) :
    stateAt E P sh (n + 1) = (tick E P (stateAt E P sh n)).2 := rfl

/-- **The closed loop is literally an iterated map.** `stateAt` is the `n`-fold
    iterate of the composite transition `sh ↦ (tick sh).2` — the env/policy pair,
    once wired, is nothing but a discrete dynamical system on `S × H`. -/
theorem stateAt_eq_iterate (E : Env S O A) (P : Policy O A H) (sh : S × H) :
    ∀ n : Nat, stateAt E P sh n = (fun x => (tick E P x).2)^[n] sh := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [stateAt_succ, ih, Function.iterate_succ_apply']

/-- A ROLLOUT: run the loop `n` ticks from joint state `sh` and record the
    trace — the data-collection phase (`horizon` steps) of the trainer. -/
def rollout (E : Env S O A) (P : Policy O A H) : Nat → S × H → List (Transition S O A)
  | 0, _ => []
  | n + 1, sh => (tick E P sh).1 :: rollout E P n (tick E P sh).2

@[simp] theorem rollout_zero (E : Env S O A) (P : Policy O A H) (sh : S × H) :
    rollout E P 0 sh = [] := rfl

@[simp] theorem rollout_succ (E : Env S O A) (P : Policy O A H) (sh : S × H) (n : Nat) :
    rollout E P (n + 1) sh = (tick E P sh).1 :: rollout E P n (tick E P sh).2 := rfl

/-- The canonical trace: roll out from the initial env state and initial policy
    state (`vec.reset()` + zeroed recurrent state in `create_pufferl`). -/
def run (E : Env S O A) (P : Policy O A H) (n : Nat) : List (Transition S O A) :=
  rollout E P n (E.init, P.init)

/-- **A rollout records exactly `n` transitions** — the buffer shape
    `[horizon, agents]` is full, no more, no fewer. -/
@[simp] theorem rollout_length (E : Env S O A) (P : Policy O A H) :
    ∀ (n : Nat) (sh : S × H), (rollout E P n sh).length = n := by
  intro n
  induction n with
  | zero => intro sh; rfl
  | succ n ih => intro sh; simp [ih]

/-- Shifting the start of the orbit: the `(n+1)`-st state from `sh` is the
    `n`-th state from the once-ticked start. -/
theorem stateAt_shift (E : Env S O A) (P : Policy O A H) :
    ∀ (n : Nat) (sh : S × H), stateAt E P sh (n + 1) = stateAt E P (tick E P sh).2 n := by
  intro n
  induction n with
  | zero => intro sh; rfl
  | succ n ih =>
    intro sh
    show (tick E P (stateAt E P sh (n + 1))).2 = (tick E P (stateAt E P (tick E P sh).2 n)).2
    rw [ih]

/-- **The trace is the orbit.** Entry `t` of a rollout is precisely the tick
    taken at the `t`-th iterate of the closed-loop map: the recorded buffer is a
    faithful log of the dynamical system, with no gaps, reordering, or
    fabrication. This is the spec that makes the rollout buffer trustworthy as
    training data. -/
theorem rollout_getElem? (E : Env S O A) (P : Policy O A H) :
    ∀ (n t : Nat) (sh : S × H), t < n →
      (rollout E P n sh)[t]? = some (tick E P (stateAt E P sh t)).1 := by
  intro n
  induction n with
  | zero => intro t sh ht; exact absurd ht (Nat.not_lt_zero t)
  | succ n ih =>
    intro t sh ht
    cases t with
    | zero => simp
    | succ t =>
      rw [rollout_succ, List.getElem?_cons_succ, stateAt_shift]
      exact ih t (tick E P sh).2 (Nat.lt_of_succ_lt_succ ht)

/-- Every recorded transition is a genuine tick from SOME reachable joint
    state. -/
theorem mem_rollout (E : Env S O A) (P : Policy O A H) :
    ∀ {n : Nat} {sh : S × H} {tr : Transition S O A},
      tr ∈ rollout E P n sh → ∃ sh' : S × H, tr = (tick E P sh').1 := by
  intro n
  induction n with
  | zero => intro sh tr h; simp at h
  | succ n ih =>
    intro sh tr h
    rw [rollout_succ, List.mem_cons] at h
    rcases h with h | h
    · exact ⟨sh, h⟩
    · exact ih h

/-- **Observation coherence.** Every recorded observation is the env's
    observation of the recorded state: `o = obs s`. (The C runtime never stores
    `s`; this is the invariant that lets it get away with storing only `o`.) -/
theorem obs_coherent (E : Env S O A) (P : Policy O A H) {n : Nat} {sh : S × H}
    {tr : Transition S O A} (h : tr ∈ rollout E P n sh) : tr.o = E.obs tr.s := by
  obtain ⟨sh', rfl⟩ := mem_rollout E P h
  rfl

/-- **Reward coherence.** Every recorded reward is the env's reward for the
    recorded state/action pair — rewards in the buffer are never stale or
    misaligned with their transition. -/
theorem reward_coherent (E : Env S O A) (P : Policy O A H) {n : Nat} {sh : S × H}
    {tr : Transition S O A} (h : tr ∈ rollout E P n sh) :
    tr.r = (E.step tr.s tr.a).2.1 := by
  obtain ⟨sh', rfl⟩ := mem_rollout E P h
  rfl

/-- **Terminal-flag coherence.** Same for the recorded `done` flag. -/
theorem done_coherent (E : Env S O A) (P : Policy O A H) {n : Nat} {sh : S × H}
    {tr : Transition S O A} (h : tr ∈ rollout E P n sh) :
    tr.done = (E.step tr.s tr.a).2.2 := by
  obtain ⟨sh', rfl⟩ := mem_rollout E P h
  rfl

/-- **Successive transitions chain through `step`.** Consecutive rollout entries
    satisfy `s_{t+1} = (step s_t a_t).1`: the trace is a genuine trajectory of
    the env, not an arbitrary list of locally-valid transitions. -/
theorem rollout_chain (E : Env S O A) (P : Policy O A H) {n t : Nat} {sh : S × H}
    {tr tr' : Transition S O A} (h : t + 1 < n)
    (h1 : (rollout E P n sh)[t]? = some tr)
    (h2 : (rollout E P n sh)[t + 1]? = some tr') :
    tr'.s = (E.step tr.s tr.a).1 := by
  rw [rollout_getElem? E P n t sh (Nat.lt_of_succ_lt h)] at h1
  rw [rollout_getElem? E P n (t + 1) sh h] at h2
  cases h1
  cases h2
  rfl

/-! ### Returns of a trace — the bridge to the verified GAE module -/

/-- The reward stream of a trace (`rewards[t]` in the buffer). -/
def rewards (τ : List (Transition S O A)) : List ℝ := τ.map Transition.r

/-- Discounted return of a trace, via the verified recurrence
    `GAE.discountedReturn` (= `gaeHead`, the backward recursion the CUDA kernel
    runs). -/
def traceReturn (w : ℝ) (τ : List (Transition S O A)) : ℝ :=
  GAE.discountedReturn w (rewards τ)

@[simp] theorem rewards_nil : rewards ([] : List (Transition S O A)) = [] := rfl

@[simp] theorem rewards_cons (tr : Transition S O A) (τ : List (Transition S O A)) :
    rewards (tr :: τ) = tr.r :: rewards τ := rfl

/-- **The Bellman-style recurrence at the loop level.** The discounted return of
    an `(n+1)`-tick rollout is the first tick's reward plus `w` times the return
    of the remaining `n`-tick rollout from the advanced joint state:
    `G(sh) = r₀ + w·G(sh')`. -/
theorem traceReturn_rollout_succ (E : Env S O A) (P : Policy O A H) (w : ℝ)
    (n : Nat) (sh : S × H) :
    traceReturn w (rollout E P (n + 1) sh)
      = (tick E P sh).1.r + w * traceReturn w (rollout E P n (tick E P sh).2) := by
  simp [traceReturn, GAE.discountedReturn]

/-- **Closed form: the return of a rollout is the geometric sum along the
    orbit**, `Σ_{t<n} wᵗ·r(stateAt t)` — the textbook discounted-return formula,
    derived for the actual loop from the verified `gaeHead` closed form. -/
theorem traceReturn_eq_geoSum (E : Env S O A) (P : Policy O A H) (w : ℝ)
    (n : Nat) (sh : S × H) :
    traceReturn w (rollout E P n sh)
      = ∑ t ∈ Finset.range n, w ^ t * (tick E P (stateAt E P sh t)).1.r := by
  unfold traceReturn
  rw [GAE.discountedReturn_eq_geoSum]
  have hlen : (rewards (rollout E P n sh)).length = n := by
    simp [rewards]
  rw [hlen]
  refine Finset.sum_congr rfl fun t ht => ?_
  have ht' := Finset.mem_range.mp ht
  congr 1
  rw [List.getD_eq_getElem?_getD, rewards, List.getElem?_map,
    rollout_getElem? E P n t sh ht']
  rfl

/-- **Bounded rewards give a bounded return, for every rollout of every policy.**
    If the env's per-step reward is uniformly bounded by `R` and `w ∈ [0,1)`,
    every rollout's discounted return has magnitude `≤ R/(1−w)`, independent of
    horizon — the loop-level form of the fundamental RL stability estimate. -/
theorem traceReturn_bounded (E : Env S O A) (P : Policy O A H) {w R : ℝ}
    (hw0 : 0 ≤ w) (hw1 : w < 1) (hR : 0 ≤ R)
    (hstep : ∀ (s : S) (a : A), |(E.step s a).2.1| ≤ R) (n : Nat) (sh : S × H) :
    |traceReturn w (rollout E P n sh)| ≤ R / (1 - w) := by
  refine GAE.gaeHead_bounded w R hw0 hw1 hR _ ?_
  intro x hx
  simp only [rewards, List.mem_map] at hx
  obtain ⟨tr, htr, rfl⟩ := hx
  obtain ⟨sh', rfl⟩ := mem_rollout E P htr
  exact hstep sh'.1 _

/-! ### Env morphisms: behavior is independent of state representation

An env morphism is a coalgebra morphism: a state map commuting with `obs` and
`step` (rewards and terminals on the nose). The trace of a rollout — hence
everything the trainer ever sees — is invariant under such a map. This is the
formal license for every state-refactoring: two envs related by a morphism are
indistinguishable to PufferLib. -/

/-- A morphism of envs over the same `O` and `A`: a state map that commutes
    with observation and stepping (preserving reward and terminal). -/
structure Hom (E : Env S O A) (E' : Env S' O A) where
  /-- The state map. -/
  toFun : S → S'
  /-- Initial states correspond. -/
  init_eq : toFun E.init = E'.init
  /-- Observations agree through the map. -/
  obs_eq : ∀ s, E'.obs (toFun s) = E.obs s
  /-- Stepping commutes with the map; reward and terminal are preserved. -/
  step_eq : ∀ s a, E'.step (toFun s) a = (toFun (E.step s a).1, (E.step s a).2)

/-- Relabel the recorded state of a transition. -/
def Transition.mapState (f : S → S') (tr : Transition S O A) : Transition S' O A :=
  ⟨f tr.s, tr.o, tr.a, tr.r, tr.done⟩

/-- One tick through a morphism: tick in `E'` at the mapped state is the mapped
    tick in `E`. -/
theorem tick_hom {E : Env S O A} {E' : Env S' O A} (φ : Hom E E')
    (P : Policy O A H) (sh : S × H) :
    tick E' P (φ.toFun sh.1, sh.2)
      = ((tick E P sh).1.mapState φ.toFun,
         (φ.toFun (tick E P sh).2.1, (tick E P sh).2.2)) := by
  unfold tick Transition.mapState
  simp only [φ.obs_eq, φ.step_eq]

/-- **Env morphisms preserve traces.** Rolling out in `E'` from a mapped state
    is the state-relabeled rollout in `E`: observations, actions, rewards, and
    terminals are IDENTICAL. The trainer's view of an env depends only on its
    observable behavior, never on its internal state representation. -/
theorem rollout_hom {E : Env S O A} {E' : Env S' O A} (φ : Hom E E')
    (P : Policy O A H) :
    ∀ (n : Nat) (sh : S × H),
      rollout E' P n (φ.toFun sh.1, sh.2)
        = (rollout E P n sh).map (Transition.mapState φ.toFun) := by
  intro n
  induction n with
  | zero => intro sh; rfl
  | succ n ih =>
    intro sh
    rw [rollout_succ, rollout_succ, List.map_cons, tick_hom φ P sh]
    exact congrArg _ (ih (tick E P sh).2)

/-- Relabeling states does not change the reward stream. -/
theorem rewards_mapState (f : S → S') (τ : List (Transition S O A)) :
    rewards (τ.map (Transition.mapState f)) = rewards τ := by
  simp only [rewards, List.map_map]
  rfl

/-- **Env morphisms preserve returns.** Immediate corollary: the discounted
    return — the trainer's objective — is invariant under env morphisms. -/
theorem traceReturn_hom {E : Env S O A} {E' : Env S' O A} (φ : Hom E E')
    (P : Policy O A H) (w : ℝ) (n : Nat) (sh : S × H) :
    traceReturn w (rollout E' P n (φ.toFun sh.1, sh.2))
      = traceReturn w (rollout E P n sh) := by
  unfold traceReturn
  rw [rollout_hom φ P n sh, rewards_mapState]

/-- **Canonical runs through a morphism.** Starting both envs from their own
    initial states (`init_eq` aligns them), the canonical traces agree up to
    state relabeling. -/
theorem run_hom {E : Env S O A} {E' : Env S' O A} (φ : Hom E E')
    (P : Policy O A H) (n : Nat) :
    run E' P n = (run E P n).map (Transition.mapState φ.toFun) := by
  unfold run
  rw [← φ.init_eq]
  exact rollout_hom φ P n (E.init, P.init)

/-- **Canonical returns through a morphism.** The canonical run's return is the
    same in both envs. -/
theorem traceReturn_run_hom {E : Env S O A} {E' : Env S' O A} (φ : Hom E E')
    (P : Policy O A H) (w : ℝ) (n : Nat) :
    traceReturn w (run E' P n) = traceReturn w (run E P n) := by
  unfold traceReturn
  rw [run_hom φ P n, rewards_mapState]

/-! ### Products: vectorization in miniature

PufferLib's vec layer (`src/vecenv.h`) runs `N` env copies as one big system
whose state/obs/action spaces are the `N`-fold products, stepped componentwise.
The two-copy product below is the generating case (iterate for `N`): the
product system's rollout is the pair of component rollouts (`stateAt_prod`,
`rewards_rollout_prod`) and its return is the sum of component returns
(`traceReturn_prod`) — vectorizing envs neither creates nor destroys behavior. -/

variable {S₁ S₂ O₁ O₂ A₁ A₂ H₁ H₂ : Type*}

/-- The product env: componentwise step, summed reward, either-terminates.
    (The vec layer keeps per-agent rewards; the sum is the scalar shadow that
    lets the product stay an `Env`. `rewards_rollout_prod` recovers the
    per-component streams exactly.) -/
def Env.prod (E₁ : Env S₁ O₁ A₁) (E₂ : Env S₂ O₂ A₂) :
    Env (S₁ × S₂) (O₁ × O₂) (A₁ × A₂) where
  init := (E₁.init, E₂.init)
  obs s := (E₁.obs s.1, E₂.obs s.2)
  step s a :=
    (((E₁.step s.1 a.1).1, (E₂.step s.2 a.2).1),
     (E₁.step s.1 a.1).2.1 + (E₂.step s.2 a.2).2.1,
     ((E₁.step s.1 a.1).2.2 || (E₂.step s.2 a.2).2.2))

/-- The product policy: componentwise action choice, componentwise state. -/
def Policy.prod (P₁ : Policy O₁ A₁ H₁) (P₂ : Policy O₂ A₂ H₂) :
    Policy (O₁ × O₂) (A₁ × A₂) (H₁ × H₂) where
  init := (P₁.init, P₂.init)
  act h o :=
    (((P₁.act h.1 o.1).1, (P₂.act h.2 o.2).1),
     ((P₁.act h.1 o.1).2, (P₂.act h.2 o.2).2))

/-- One tick of the product system is the pair of component ticks (with summed
    reward and or-ed terminal). Definitionally true — the product wiring adds
    nothing. -/
theorem tick_prod (E₁ : Env S₁ O₁ A₁) (E₂ : Env S₂ O₂ A₂)
    (P₁ : Policy O₁ A₁ H₁) (P₂ : Policy O₂ A₂ H₂)
    (s₁ : S₁) (s₂ : S₂) (h₁ : H₁) (h₂ : H₂) :
    tick (E₁.prod E₂) (P₁.prod P₂) ((s₁, s₂), (h₁, h₂))
      = (⟨(s₁, s₂),
          ((tick E₁ P₁ (s₁, h₁)).1.o, (tick E₂ P₂ (s₂, h₂)).1.o),
          ((tick E₁ P₁ (s₁, h₁)).1.a, (tick E₂ P₂ (s₂, h₂)).1.a),
          (tick E₁ P₁ (s₁, h₁)).1.r + (tick E₂ P₂ (s₂, h₂)).1.r,
          ((tick E₁ P₁ (s₁, h₁)).1.done || (tick E₂ P₂ (s₂, h₂)).1.done)⟩,
         (((tick E₁ P₁ (s₁, h₁)).2.1, (tick E₂ P₂ (s₂, h₂)).2.1),
          ((tick E₁ P₁ (s₁, h₁)).2.2, (tick E₂ P₂ (s₂, h₂)).2.2))) := rfl

/-- **The product orbit is the pair of component orbits.** Running two envs in
    one vectorized system traverses exactly the states each would traverse
    alone: components never interfere. -/
theorem stateAt_prod (E₁ : Env S₁ O₁ A₁) (E₂ : Env S₂ O₂ A₂)
    (P₁ : Policy O₁ A₁ H₁) (P₂ : Policy O₂ A₂ H₂) :
    ∀ (n : Nat) (s₁ : S₁) (s₂ : S₂) (h₁ : H₁) (h₂ : H₂),
      stateAt (E₁.prod E₂) (P₁.prod P₂) ((s₁, s₂), (h₁, h₂)) n
        = (((stateAt E₁ P₁ (s₁, h₁) n).1, (stateAt E₂ P₂ (s₂, h₂) n).1),
           ((stateAt E₁ P₁ (s₁, h₁) n).2, (stateAt E₂ P₂ (s₂, h₂) n).2)) := by
  intro n
  induction n with
  | zero => intro s₁ s₂ h₁ h₂; rfl
  | succ n ih =>
    intro s₁ s₂ h₁ h₂
    rw [stateAt_succ, ih, stateAt_succ, stateAt_succ]
    rw [tick_prod]

/-- **The product reward stream is the pointwise sum of component streams.**
    The vec layer's per-agent rewards are recovered exactly: the product rollout
    carries `r₁ₜ + r₂ₜ` at each `t`, where the component rollouts carry `r₁ₜ`
    and `r₂ₜ`. -/
theorem rewards_rollout_prod (E₁ : Env S₁ O₁ A₁) (E₂ : Env S₂ O₂ A₂)
    (P₁ : Policy O₁ A₁ H₁) (P₂ : Policy O₂ A₂ H₂) :
    ∀ (n : Nat) (s₁ : S₁) (s₂ : S₂) (h₁ : H₁) (h₂ : H₂),
      rewards (rollout (E₁.prod E₂) (P₁.prod P₂) n ((s₁, s₂), (h₁, h₂)))
        = List.zipWith (· + ·)
            (rewards (rollout E₁ P₁ n (s₁, h₁)))
            (rewards (rollout E₂ P₂ n (s₂, h₂))) := by
  intro n
  induction n with
  | zero => intro s₁ s₂ h₁ h₂; rfl
  | succ n ih =>
    intro s₁ s₂ h₁ h₂
    rw [rollout_succ, rollout_succ, rollout_succ, rewards_cons, rewards_cons,
      rewards_cons, List.zipWith_cons_cons, tick_prod]
    exact congrArg _ (ih _ _ _ _)

/-- Additivity of the discounted-return recurrence over equal-length pointwise
    sums: `gaeHead w (xs ⊞ ys) = gaeHead w xs + gaeHead w ys`. (The additive
    sibling of `GAE.gaeHead_sub_zipWith`.) -/
theorem gaeHead_add_zipWith (w : ℝ) :
    ∀ (xs ys : List ℝ), xs.length = ys.length →
      GAE.gaeHead w (List.zipWith (· + ·) xs ys)
        = GAE.gaeHead w xs + GAE.gaeHead w ys := by
  intro xs
  induction xs with
  | nil =>
    intro ys h
    cases ys with
    | nil => simp
    | cons y ys => simp at h
  | cons x xs ih =>
    intro ys h
    cases ys with
    | nil => simp at h
    | cons y ys =>
      rw [List.zipWith_cons_cons, GAE.gaeHead_cons, GAE.gaeHead_cons,
        GAE.gaeHead_cons, ih ys (by simpa using h)]
      ring

/-- **Vectorization soundness (two-copy case).** The discounted return of the
    product system is the SUM of the component returns: batching envs together
    changes how the computation is laid out, not what is computed. This is the
    `N = 2` generating case of `src/vecenv.h`'s claim that one vectorized
    system of `N` envs is the same object as `N` independent envs. -/
theorem traceReturn_prod (E₁ : Env S₁ O₁ A₁) (E₂ : Env S₂ O₂ A₂)
    (P₁ : Policy O₁ A₁ H₁) (P₂ : Policy O₂ A₂ H₂) (w : ℝ) (n : Nat)
    (s₁ : S₁) (s₂ : S₂) (h₁ : H₁) (h₂ : H₂) :
    traceReturn w (rollout (E₁.prod E₂) (P₁.prod P₂) n ((s₁, s₂), (h₁, h₂)))
      = traceReturn w (rollout E₁ P₁ n (s₁, h₁))
        + traceReturn w (rollout E₂ P₂ n (s₂, h₂)) := by
  unfold traceReturn GAE.discountedReturn
  rw [rewards_rollout_prod, gaeHead_add_zipWith]
  simp [rewards]

/-! ### N-fold vectorization: the general product

The binary product above generalizes to a dependent product over ANY finite
index family — `N` (possibly different) envs stepped as one system, which is
exactly `src/vecenv.h`: `total_agents` sub-envs, one flat joint state, stepped
componentwise, rewards delivered per-agent. The capstone `traceReturn_pi` rides
on the orbit machinery (`traceReturn_eq_geoSum`), not on list surgery: the
vectorized orbit is the tuple of component orbits (`stateAt_pi`), so the
vectorized return is the sum of component returns by exchanging two finite
sums. -/

section Pi

variable {ι : Type*} [Fintype ι] {Sv Ov Av Hv : ι → Type*}

/-- The `ι`-indexed product env: componentwise observation and step, summed
    reward, any-component-terminates. The scalar reward sum is the shadow that
    keeps the product an `Env`; per-component rewards are recovered exactly by
    `traceReturn_pi`. -/
def Env.pi (E : ∀ i, Env (Sv i) (Ov i) (Av i)) :
    Env (∀ i, Sv i) (∀ i, Ov i) (∀ i, Av i) where
  init := fun i => (E i).init
  obs s := fun i => (E i).obs (s i)
  step s a :=
    ((fun i => ((E i).step (s i) (a i)).1),
     ∑ i, ((E i).step (s i) (a i)).2.1,
     decide (∃ i, ((E i).step (s i) (a i)).2.2 = true))

/-- The `ι`-indexed product policy: componentwise action choice and state. -/
def Policy.pi (P : ∀ i, Policy (Ov i) (Av i) (Hv i)) :
    Policy (∀ i, Ov i) (∀ i, Av i) (∀ i, Hv i) where
  init := fun i => (P i).init
  act h o :=
    ((fun i => ((P i).act (h i) (o i)).1),
     (fun i => ((P i).act (h i) (o i)).2))

/-- One tick of the vectorized system is the tuple of component ticks (with
    summed reward and any-terminated flag). Definitionally true — vectorized
    wiring adds nothing. -/
theorem tick_pi (E : ∀ i, Env (Sv i) (Ov i) (Av i))
    (P : ∀ i, Policy (Ov i) (Av i) (Hv i)) (sh : (∀ i, Sv i) × (∀ i, Hv i)) :
    tick (Env.pi E) (Policy.pi P) sh
      = (⟨sh.1,
          fun i => (tick (E i) (P i) (sh.1 i, sh.2 i)).1.o,
          fun i => (tick (E i) (P i) (sh.1 i, sh.2 i)).1.a,
          ∑ i, (tick (E i) (P i) (sh.1 i, sh.2 i)).1.r,
          decide (∃ i, (tick (E i) (P i) (sh.1 i, sh.2 i)).1.done = true)⟩,
         ((fun i => (tick (E i) (P i) (sh.1 i, sh.2 i)).2.1),
          (fun i => (tick (E i) (P i) (sh.1 i, sh.2 i)).2.2))) := rfl

/-- **The vectorized orbit is the tuple of component orbits.** Batching `N`
    envs into one system traverses, at every index, exactly the states that env
    would traverse alone — sub-envs never interfere (`Fin N → S` form of
    `stateAt_prod`). -/
theorem stateAt_pi (E : ∀ i, Env (Sv i) (Ov i) (Av i))
    (P : ∀ i, Policy (Ov i) (Av i) (Hv i)) :
    ∀ (n : Nat) (sh : (∀ i, Sv i) × (∀ i, Hv i)),
      stateAt (Env.pi E) (Policy.pi P) sh n
        = ((fun i => (stateAt (E i) (P i) (sh.1 i, sh.2 i) n).1),
           (fun i => (stateAt (E i) (P i) (sh.1 i, sh.2 i) n).2)) := by
  intro n
  induction n with
  | zero => intro sh; rfl
  | succ n ih =>
    intro sh
    simp only [stateAt_succ]
    rw [ih sh, tick_pi]

/-- **Vectorization soundness, `N`-fold.** The discounted return of the
    vectorized system is the sum over sub-envs of their independent returns:
    `src/vecenv.h`'s batching changes how the computation is laid out, not what
    is computed, for any number of (possibly different) envs at once. -/
theorem traceReturn_pi (E : ∀ i, Env (Sv i) (Ov i) (Av i))
    (P : ∀ i, Policy (Ov i) (Av i) (Hv i)) (w : ℝ) (n : Nat)
    (sh : (∀ i, Sv i) × (∀ i, Hv i)) :
    traceReturn w (rollout (Env.pi E) (Policy.pi P) n sh)
      = ∑ i, traceReturn w (rollout (E i) (P i) n (sh.1 i, sh.2 i)) := by
  simp only [traceReturn_eq_geoSum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  have hr : (tick (Env.pi E) (Policy.pi P)
        (stateAt (Env.pi E) (Policy.pi P) sh t)).1.r
      = ∑ i, (tick (E i) (P i) (stateAt (E i) (P i) (sh.1 i, sh.2 i) t)).1.r := by
    rw [stateAt_pi E P t sh]
    rfl
  rw [hr, Finset.mul_sum]

end Pi

/-- **Identically-seeded copies are perfectly redundant.** `N` copies of the
    same env/policy started from the same joint state produce `N` times the
    single-copy return — no new information. This is precisely why the real vec
    layer seeds every sub-env differently (`rng` per env struct): the value of
    vectorization comes from decorrelated starts, which the general
    `traceReturn_pi` (arbitrary per-index states `sh.1 i`) accounts for. -/
theorem traceReturn_pi_const (E : Env S O A) (P : Policy O A H) (N : Nat)
    (w : ℝ) (n : Nat) (s : S) (h : H) :
    traceReturn w (rollout (Env.pi fun _ : Fin N => E)
        (Policy.pi fun _ : Fin N => P) n ((fun _ => s), (fun _ => h)))
      = N * traceReturn w (rollout E P n (s, h)) := by
  rw [traceReturn_pi]
  simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-! ### Auto-reset: episodes inside one infinite stream

The vec layer never stops: a sub-env whose step reports `done` is immediately
reset (`src/vecenv.h`), so the trainer sees one unbroken stream with terminal
flags marking episode boundaries. That wrapper is itself just another `Env`. -/

/-- The auto-reset wrapper: identical to `E` except that a terminal step lands
    in `E.init` instead of the terminal state. Reward and terminal flag are
    reported unchanged (`withAutoReset_out`), exactly as the vec layer delivers
    them to the buffer. -/
def Env.withAutoReset (E : Env S O A) : Env S O A where
  init := E.init
  obs := E.obs
  step s a :=
    ((if (E.step s a).2.2 then E.init else (E.step s a).1),
     (E.step s a).2.1, (E.step s a).2.2)

/-- Auto-reset changes only the successor state: reward and terminal flag pass
    through untouched. -/
@[simp] theorem withAutoReset_out (E : Env S O A) (s : S) (a : A) :
    (E.withAutoReset.step s a).2 = (E.step s a).2 := rfl

/-- On non-terminal steps the auto-reset wrapper is invisible. -/
theorem withAutoReset_step_of_not_done (E : Env S O A) (s : S) (a : A)
    (h : (E.step s a).2.2 = false) :
    (E.withAutoReset.step s a).1 = (E.step s a).1 := by
  simp [Env.withAutoReset, h]

/-- On terminal steps the auto-reset wrapper lands in the initial state. -/
theorem withAutoReset_step_of_done (E : Env S O A) (s : S) (a : A)
    (h : (E.step s a).2.2 = true) :
    (E.withAutoReset.step s a).1 = E.init := by
  simp [Env.withAutoReset, h]

end Puffer.RL.Loop
