/-
# Multi-agent PPO with parameter sharing (Tier 2)

Multi-agent Ocean envs (overcooked, trash_pickup, …) have `numAgents` agents sharing
ONE env instance. PufferLib trains them with a single PARAMETER-SHARED policy: each
agent samples its own action from the shared net given its own observation, the env is
stepped with all actions at once, and each agent contributes its OWN transition stream
to the experience buffer.

That makes the multi-agent case a thin layer over the discrete trainer: a rollout
produces `numAgents` streams per env instance (instead of 1), and everything downstream
— GAE per stream (`buildBatch`/`computeGAEBoot`), batch-level advantage normalization,
shuffled-minibatch clipped-surrogate PPO on the shared MLP (`updatePPOIdx`) — is reused
verbatim. Only the rollout is new.

Global-terminal model (the common cooperative case): the env returns a per-agent reward
vector and one shared terminal; on terminal the whole env resets and all agents' streams
break together. Continuing envs (overcooked) simply never terminate and bootstrap at the
horizon boundary.

Mathlib-free (extends `VecTrain`); the binary links no Mathlib.
-/
import Puffer.RL.VecTrain

namespace Puffer.RL.NNTrain

open Puffer.RL (Env MultiEnv)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 sampleCat)

/-- Roll ONE multi-agent env instance for `horizon` steps under the shared policy `p`,
    producing one transition stream PER agent (all stepped together, reset together on
    terminal). Returns `(streams, bootVals, finalState, rng, teamEpReturns)` — one stream
    and one bootstrap value per agent; `teamEpReturns` are the summed team rewards of
    episodes that completed. -/
def multiSegmentRollout {S : Type} (env : MultiEnv S) (p : MLP) (horizon : Nat)
    (s0 : S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × S × UInt64 × Array Float := Id.run do
  let N := env.numAgents
  let mut st := s0
  let mut rng := rng0
  let mut streams : Array (Array Transition) := Array.replicate N #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obsAll := env.observe st
    let mut actions : Array Nat := #[]
    let mut vals : Array Float := #[]
    let mut logps : Array Float := #[]
    for a in [0:N] do
      let (probs, v) := policyAndValue p (obsAll[a]!)
      let (word, rng') := rngNext rng
      rng := rng'
      let act := sampleCat probs (uniform01 word)
      actions := actions.push act
      vals := vals.push v
      logps := logps.push (Float.log probs[act]!)
    let (st', rewards, term) := env.step st actions
    for a in [0:N] do
      streams := streams.set! a ((streams[a]!).push
        { obs := obsAll[a]!, action := actions[a]!, reward := rewards[a]!, value := vals[a]!,
          oldLogp := logps[a]!, terminal := term })
    epRet := epRet + rewards.foldl (· + ·) 0.0
    st := st'
    if term then
      epReturns := epReturns.push epRet
      epRet := 0.0
      let (sReset, rng'') := env.reset rng
      rng := rng''
      st := sReset
  let obsF := env.observe st
  let mut bootVals : Array Float := #[]
  for a in [0:N] do
    let (_, v) := policyAndValue p (obsF[a]!)
    bootVals := bootVals.push v
  return (streams, bootVals, st, rng, epReturns)

/-- Roll `numEnvs` multi-agent instances; flatten to `numEnvs·numAgents` streams. -/
def multiVecRollout {S : Type} (env : MultiEnv S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let mut rng := rng0
  let mut streams : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (strs, boots, st', rng', epRets) := multiSegmentRollout env p horizon st rng
    rng := rng'
    streams := streams ++ strs
    bootVals := bootVals ++ boots
    newStates := newStates.push st'
    epReturns := epReturns ++ epRets
  return (streams, bootVals, newStates, rng, epReturns)

end Puffer.RL.NNTrain
