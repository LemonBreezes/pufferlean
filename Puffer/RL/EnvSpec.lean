/-
Generic environment interface for the trainer — Mathlib-free.

A first-class record (not a typeclass) so an env's runtime config (size, target, …)
is captured in the closures. Any `Puffer.Env.*` model is adapted to `Env S` in
`Puffer/RL/Envs.lean`; the PPO/GAE trainer works over `Env S` for any state type `S`.
-/
namespace Puffer.RL

/-- A discrete-action RL environment over state type `S`, as seen by the trainer. -/
structure Env (S : Type) where
  /-- Number of discrete actions. -/
  numActions : Nat
  /-- Dimension of the observation feature vector. -/
  obsDim : Nat
  /-- Hard cap on episode length (steps). -/
  maxSteps : Nat
  /-- Reset to an initial state (may consume RNG); returns the advanced RNG. -/
  reset : UInt64 → S × UInt64
  /-- Step: `(next state, reward, terminal?)`. -/
  step : S → Nat → S × Float × Bool
  /-- Encode a state as the network's input features. -/
  observe : S → Array Float
  /-- Env-specific scalar metrics (`name → value`) read from a state, surfaced on the
      dashboard/wandb under `environment/<name>` (PufferLib's per-env `info` metrics).
      Default: none. -/
  metrics : S → Array (String × Float) := fun _ => #[]

/-- A CONTINUOUS-action RL environment over state type `S` (Tier 2). The action is a
    real-valued vector of length `actionDim`; the env clips/uses it as it sees fit
    (e.g. `squared_continuous` clamps each dim to `[-1,1]`). The Gaussian-policy
    trainer (`Puffer/RL/ContVecTrain.lean`) works over `ContEnv S` for any `S`. -/
structure ContEnv (S : Type) where
  /-- Number of continuous action dimensions. -/
  actionDim : Nat
  /-- Dimension of the observation feature vector. -/
  obsDim : Nat
  /-- Hard cap on episode length (steps). -/
  maxSteps : Nat
  /-- Reset to an initial state (may consume RNG); returns the advanced RNG. -/
  reset : UInt64 → S × UInt64
  /-- Step with a continuous action vector: `(next state, reward, terminal?)`. -/
  step : S → Array Float → S × Float × Bool
  /-- Encode a state as the network's input features. -/
  observe : S → Array Float
  /-- Env-specific scalar metrics (`name → value`), surfaced under `environment/<name>`.
      Default: none. -/
  metrics : S → Array (String × Float) := fun _ => #[]

/-- A MULTI-AGENT RL environment over shared state `S` (Tier 2). `numAgents` agents
    share one env instance and one (parameter-shared) policy; the env is stepped with
    ALL agents' actions at once and returns a per-agent reward vector and a GLOBAL
    terminal (the common cooperative case: agents' episodes end together). Each agent
    contributes its own transition stream to the buffer. The multi-agent trainer
    (`Puffer/RL/MultiVecTrain.lean`) works over `MultiEnv S` for any `S`. -/
structure MultiEnv (S : Type) where
  /-- Number of agents sharing this env. -/
  numAgents : Nat
  /-- Number of discrete actions (per agent; the policy is shared). -/
  numActions : Nat
  /-- Observation feature length (per agent). -/
  obsDim : Nat
  /-- Hard cap on episode length (steps). -/
  maxSteps : Nat
  /-- Reset to an initial state (may consume RNG); returns the advanced RNG. -/
  reset : UInt64 → S × UInt64
  /-- Step with all agents' actions: `(next state, per-agent rewards, global terminal?)`. -/
  step : S → Array Nat → S × Array Float × Bool
  /-- Per-agent observations (`numAgents` feature vectors). -/
  observe : S → Array (Array Float)
  /-- Env-specific scalar metrics (`name → value`), surfaced under `environment/<name>`.
      Default: none. -/
  metrics : S → Array (String × Float) := fun _ => #[]

end Puffer.RL
