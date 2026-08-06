/-
Seed RL trainer — a complete, RUNNABLE REINFORCE loop on a small tabular MDP, in
native `Float`, Mathlib-free (so the `puffer` binary stays lean).

No autodiff needed yet: a tabular softmax policy has an analytic policy gradient
`∇_logit log π(a|s) = 1[·=a] − π(·|s)`. This proves the whole pipeline — env
rollout → discounted returns → policy-gradient update — runs and *learns* (episode
return climbs toward the optimum). It grows into the full trainer (NN policy via
`Puffer/Float/*`, GAE, PPO, Muon) per PLAN.md M2–M5.
-/

namespace Puffer.RL.Train


/-! ### Deterministic PRNG (splitmix64) -/

/-- One splitmix64 step: returns `(random word, next state)`. -/
def rngNext (s : UInt64) : UInt64 × UInt64 :=
  let s' := s + (0x9E3779B97F4A7C15 : UInt64)
  let z := s'
  let z := (z ^^^ (z >>> 30)) * (0xBF58476D1CE4E5B9 : UInt64)
  let z := (z ^^^ (z >>> 27)) * (0x94D049BB133111EB : UInt64)
  let z := z ^^^ (z >>> 31)
  (z, s')

/-- Uniform `Float` in `[0,1)` from a 64-bit word (top 53 bits). -/
def uniform01 (r : UInt64) : Float :=
  Float.ofNat (r >>> 11).toNat / 9007199254740992.0

/-! ### Softmax policy over discrete actions -/

/-- Numerically-stable softmax. -/
def softmax (logits : Array Float) : Array Float := Id.run do
  let m := logits.foldl (fun a x => if x > a then x else a) (-1.0e30)
  let exps := logits.map (fun x => Float.exp (x - m))
  let z := exps.foldl (· + ·) 0.0
  return exps.map (fun e => e / z)

/-- LayerNorm `((xᵢ − μ)/√(σ² + ε))·wᵢ + bᵢ` over a feature vector (`μ` = mean, `σ²` = variance).
    A forward-pass kernel for the NN policy (PLAN M2); runnable native `Float`, Mathlib-free. -/
def layerNorm (x w b : Array Float) (eps : Float) : Array Float := Id.run do
  let c := Float.ofNat x.size
  let mu := x.foldl (· + ·) 0.0 / c
  let var := (x.map (fun xi => (xi - mu) * (xi - mu))).foldl (· + ·) 0.0 / c
  let denom := Float.sqrt (var + eps)
  return (Array.range x.size).map (fun i => (x[i]! - mu) / denom * w[i]! + b[i]!)

/-- Sample a categorical index from `probs` given a uniform `u ∈ [0,1)`. -/
def sampleCat (probs : Array Float) (u : Float) : Nat := Id.run do
  let mut acc := 0.0
  for i in [0:probs.size] do
    acc := acc + probs[i]!
    if u < acc then return i
  return probs.size - 1

/-- **The sampled action index is always in bounds.** For a nonempty probability array,
    `sampleCat probs u < probs.size` — the early-return path returns a loop index `i ∈ [0, probs.size)`, and the
    fall-through path returns `probs.size - 1`. This is the runtime safety guarantee that every categorical action
    drawn during `rollout` is a legal index, so the ensuing `step`/`theta` lookups it feeds never go out of range.
    Proved by a loop invariant (`state.fst = some a → a < probs.size`) carried through the desugared `forIn`. -/
theorem sampleCat_lt_size (probs : Array Float) (u : Float) (h : 0 < probs.size) :
    sampleCat probs u < probs.size := by
  have key : ∀ (l : List Nat) (st : MProd (Option Nat) Float),
      (∀ a, st.fst = some a → a < probs.size) → (∀ i ∈ l, i < probs.size) →
      ∀ a, (forIn (m := Id) l st (fun (i : Nat) (r : MProd (Option Nat) Float) =>
          if u < r.snd + probs[i]!
          then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
          else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst = some a → a < probs.size := by
    intro l
    induction l with
    | nil =>
      intro st hst _ a ha
      simp only [List.forIn_nil] at ha
      exact hst a ha
    | cons x xs ih =>
      intro st hst hmem a ha
      by_cases hb : u < st.snd + probs[x]!
      · simp only [List.forIn_cons, hb, if_true, pure_bind] at ha
        have hax : (some x : Option Nat) = some a := ha
        exact (Option.some.inj hax) ▸ hmem x List.mem_cons_self
      · simp only [List.forIn_cons, hb, if_false, pure_bind] at ha
        exact ih ⟨none, st.snd + probs[x]!⟩ (by intro a ha'; simp at ha')
          (fun i hi => hmem i (List.mem_cons_of_mem x hi)) a ha
  have hmem : ∀ i ∈ List.range' 0 probs.size, i < probs.size := by
    intro i hi
    have := List.mem_range'.mp hi
    omega
  have hkey := key (List.range' 0 probs.size) ⟨none, 0.0⟩ (by intro a ha; simp at ha) hmem
  unfold sampleCat
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
    Nat.add_one_sub_one, Nat.div_one]
  show (match (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst with
        | none => probs.size - 1
        | some a => a) < probs.size
  cases hv : (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst with
  | none => exact Nat.sub_lt h Nat.one_pos
  | some a => exact hkey a hv

/-! ### Rollout, returns, and the REINFORCE update -/

/-- Discounted returns `G_t = Σ_{k≥t} γ^{k-t} r_k`. -/
def discountedReturns (traj : Array (Nat × Nat × Float)) (gamma : Float) : Array Float := Id.run do
  let n := traj.size
  let mut returns := Array.replicate n 0.0
  let mut g := 0.0
  for i in [0:n] do
    let t := n - 1 - i
    let (_, _, r) := traj[t]!
    g := r + gamma * g
    returns := returns.set! t g
  return returns

/-- One REINFORCE update: `θ[s][k] += lr·(G_t − b)·(1[k=a] − π(k|s))`, accumulated
    over the trajectory (probabilities taken at the rollout policy). `b` is a
    variance-reducing baseline. -/
def updatePolicy (theta : Array (Array Float)) (traj : Array (Nat × Nat × Float))
    (returns : Array Float) (lr baseline : Float) : Array (Array Float) := Id.run do
  let mut th := theta
  for t in [0:traj.size] do
    let (s, a, _) := traj[t]!
    let adv := returns[t]! - baseline
    let probs := softmax (theta.getD s #[0.0, 0.0])
    let row := th.getD s #[0.0, 0.0]
    let newRow := (Array.range row.size).map (fun k =>
      let indic := if k == a then 1.0 else 0.0
      row[k]! + lr * adv * (indic - probs[k]!))
    th := th.set! s newRow
  return th

end Puffer.RL.Train
