/-
`vecSpec` ↔ `Env.pi`: sequential collection = parallel product, under
independent streams.

The a57 spec `vecSpec` runs sub-envs SEQUENTIALLY, threading one stream through
all segments — the Lean trainer's implementation. PufferLib's C vec layer
(`src/vecenv.h`) instead steps all sub-envs in LOCKSTEP, each with its own
`rng` field — the parallel product semantics that a53 formalized (for pure
envs) as `Env.pi`. This module proves the two agree.

The bridge has three parts:
* `EnvR.pi`/`PolicyR.pi` — the dependent product of SHARED-STREAM machines
  where the joint stream is the product `∀ i, R i` and each component advances
  only its own coordinate. That is precisely "independent per-env streams" —
  the C vec layer's per-env `rng` structs. The a53 product theorems replay:
  `tickR_pi` is `rfl`, `stateAtR_pi` gives non-interference, and
  `traceReturn_rolloutR_pi` gives return additivity (via the orbit machinery
  `rolloutR_getElem?` / `traceReturn_rolloutR_eq_geoSum`, mirrored from a52).
* DECOUPLING (`vecSpec_traces_eq_zipWith`): sequential threading is nothing
  but a particular CHOICE of per-segment seeds — the `vecStreams` sequence
  `σ₀ = r`, `σₖ₊₁ = ` segment `k`'s orbit-endpoint stream. Each trace depends
  only on its own `(state, seed)`; the segments never otherwise interact.
* CAPSTONE (`vecSpec_return_eq_pi`): the sum of the sequential spec's returns
  IS the return of the lockstep-parallel product system seeded with
  `vecStreams` — and, chained through a57's refinement,
  `vecRollout_return_eq_pi`: the total ℝ-discounted return of the buffers the
  EXECUTABLE collects equals the parallel pi-system's return. The sequential
  Lean trainer and the parallel vec-layer semantics compute the same training
  signal, exactly, when the seeds correspond.

What is NOT claimed: with a SINGLE threaded stream vs genuinely independent
per-env seeds, the drawn actions differ — the agreement above is at
corresponding seeds (`vecStreams`). "Same distribution over fresh seeds" is a
probabilistic statement outside this exact-refinement layer.
-/
import Puffer.RL.LoopVecRollout

namespace Puffer.RL.Loop

open Puffer.RL.NNTrain (MLP policyAndValue)

variable {R S O A H : Type*}

/-! ### Orbit machinery for the shared-stream rollout (mirrors a52) -/

/-- **The shared-stream trace is the orbit**: entry `t` of `rolloutR` is the
    tick at the `t`-th iterate of the closed-loop map. -/
theorem rolloutR_getElem? (E : EnvR R S O A) (P : PolicyR R O A H) :
    ∀ (n t : Nat) (x : S × H × R), t < n →
      (rolloutR E P n x)[t]? = some (tickR E P (stateAtR E P x t)).1 := by
  intro n
  induction n with
  | zero => intro t x ht; exact absurd ht (Nat.not_lt_zero t)
  | succ n ih =>
    intro t x ht
    cases t with
    | zero => simp
    | succ t =>
      rw [rolloutR_succ, List.getElem?_cons_succ, stateAtR_shift]
      exact ih t (tickR E P x).2 (Nat.lt_of_succ_lt_succ ht)

/-- The discounted return of a shared-stream rollout is the geometric sum along
    the orbit (mirrors `traceReturn_eq_geoSum`). -/
theorem traceReturn_rolloutR_eq_geoSum (E : EnvR R S O A) (P : PolicyR R O A H)
    (w : ℝ) (n : Nat) (x : S × H × R) :
    traceReturn w (rolloutR E P n x)
      = ∑ t ∈ Finset.range n, w ^ t * (tickR E P (stateAtR E P x t)).1.r := by
  unfold traceReturn
  rw [GAE.discountedReturn_eq_geoSum]
  have hlen : (rewards (rolloutR E P n x)).length = n := by
    simp [rewards]
  rw [hlen]
  refine Finset.sum_congr rfl fun t ht => ?_
  have ht' := Finset.mem_range.mp ht
  congr 1
  rw [List.getD_eq_getElem?_getD, rewards, List.getElem?_map,
    rolloutR_getElem? E P n t x ht']
  rfl

/-! ### The product of shared-stream machines: independent per-component streams -/

section Pi

variable {ι : Type*} [Fintype ι] {Rv Sv Ov Av Hv : ι → Type*}

/-- The `ι`-indexed product of shared-stream envs, with INDEPENDENT streams:
    the joint stream is `∀ i, Rv i`, and each component's `init`/`step`
    advances only its own coordinate — the C vec layer's per-env `rng` field,
    formalized. -/
def EnvR.pi (E : ∀ i, EnvR (Rv i) (Sv i) (Ov i) (Av i)) :
    EnvR (∀ i, Rv i) (∀ i, Sv i) (∀ i, Ov i) (∀ i, Av i) where
  init r := ((fun i => ((E i).init (r i)).1), (fun i => ((E i).init (r i)).2))
  obs s := fun i => (E i).obs (s i)
  step s a r :=
    (((fun i => ((E i).step (s i) (a i) (r i)).1.1),
      ∑ i, ((E i).step (s i) (a i) (r i)).1.2.1,
      decide (∃ i, ((E i).step (s i) (a i) (r i)).1.2.2 = true)),
     (fun i => ((E i).step (s i) (a i) (r i)).2))

/-- The `ι`-indexed product of shared-stream policies, each sampling from its
    own stream coordinate. -/
def PolicyR.pi (P : ∀ i, PolicyR (Rv i) (Ov i) (Av i) (Hv i)) :
    PolicyR (∀ i, Rv i) (∀ i, Ov i) (∀ i, Av i) (∀ i, Hv i) where
  init := fun i => (P i).init
  act h o r :=
    (((fun i => ((P i).act (h i) (o i) (r i)).1.1),
      (fun i => ((P i).act (h i) (o i) (r i)).1.2)),
     (fun i => ((P i).act (h i) (o i) (r i)).2))

/-- One tick of the product is the tuple of component ticks (summed reward,
    any-terminated flag, componentwise streams). Definitional. -/
theorem tickR_pi (E : ∀ i, EnvR (Rv i) (Sv i) (Ov i) (Av i))
    (P : ∀ i, PolicyR (Rv i) (Ov i) (Av i) (Hv i))
    (x : (∀ i, Sv i) × (∀ i, Hv i) × (∀ i, Rv i)) :
    tickR (EnvR.pi E) (PolicyR.pi P) x
      = (⟨x.1,
          fun i => (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).1.o,
          fun i => (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).1.a,
          ∑ i, (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).1.r,
          decide (∃ i, (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).1.done = true)⟩,
         ((fun i => (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).2.1),
          (fun i => (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).2.2.1),
          (fun i => (tickR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i)).2.2.2))) := rfl

/-- **Independent streams do not interact**: the product orbit is the tuple of
    component orbits (shared-stream form of `stateAt_pi`). -/
theorem stateAtR_pi (E : ∀ i, EnvR (Rv i) (Sv i) (Ov i) (Av i))
    (P : ∀ i, PolicyR (Rv i) (Ov i) (Av i) (Hv i)) :
    ∀ (n : Nat) (x : (∀ i, Sv i) × (∀ i, Hv i) × (∀ i, Rv i)),
      stateAtR (EnvR.pi E) (PolicyR.pi P) x n
        = ((fun i => (stateAtR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i) n).1),
           (fun i => (stateAtR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i) n).2.1),
           (fun i => (stateAtR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i) n).2.2)) := by
  intro n
  induction n with
  | zero => intro x; rfl
  | succ n ih =>
    intro x
    simp only [stateAtR_succ]
    rw [ih x, tickR_pi]

/-- **Return additivity under independent streams** (shared-stream form of
    `traceReturn_pi`): the product system's discounted return is the sum of
    the components' independent returns. -/
theorem traceReturn_rolloutR_pi (E : ∀ i, EnvR (Rv i) (Sv i) (Ov i) (Av i))
    (P : ∀ i, PolicyR (Rv i) (Ov i) (Av i) (Hv i)) (w : ℝ) (n : Nat)
    (x : (∀ i, Sv i) × (∀ i, Hv i) × (∀ i, Rv i)) :
    traceReturn w (rolloutR (EnvR.pi E) (PolicyR.pi P) n x)
      = ∑ i, traceReturn w (rolloutR (E i) (P i) n (x.1 i, x.2.1 i, x.2.2 i)) := by
  simp only [traceReturn_rolloutR_eq_geoSum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun t _ => ?_
  have hr : (tickR (EnvR.pi E) (PolicyR.pi P)
        (stateAtR (EnvR.pi E) (PolicyR.pi P) x t)).1.r
      = ∑ i, (tickR (E i) (P i)
          (stateAtR (E i) (P i) (x.1 i, x.2.1 i, x.2.2 i) t)).1.r := by
    rw [stateAtR_pi E P t x]
    rfl
  rw [hr, Finset.mul_sum]

end Pi

/-! ### Decoupling: sequential threading is a choice of seeds -/

/-- The per-segment seed sequence induced by sequential threading:
    `σ₀ = r`, `σₖ₊₁ = ` segment `k`'s orbit-endpoint stream. -/
def vecStreams (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H) (horizon : Nat) :
    List S → R → List R
  | [], _ => []
  | s :: rest, r =>
    r :: vecStreams E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2

@[simp] theorem vecStreams_nil (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) (r : R) : vecStreams E P h0 horizon [] r = [] := rfl

@[simp] theorem vecStreams_cons (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) (s : S) (rest : List S) (r : R) :
    vecStreams E P h0 horizon (s :: rest) r
      = r :: vecStreams E P h0 horizon rest (stateAtR E P (s, h0, r) horizon).2.2 := rfl

@[simp] theorem vecStreams_length (E : EnvR R S O A) (P : PolicyR R O A H) (h0 : H)
    (horizon : Nat) : ∀ (l : List S) (r : R),
    (vecStreams E P h0 horizon l r).length = l.length := by
  intro l
  induction l with
  | nil => intro r; rfl
  | cons s rest ih => intro r; simp [ih]

/-- **Decoupling**: the sequential spec's traces are exactly the independent
    per-seed rollouts, zipped with the `vecStreams` seeds — each segment
    depends only on its own `(state, seed)`; the threading never couples them
    beyond seed choice. -/
theorem vecSpec_traces_eq_zipWith (E : EnvR R S O A) (P : PolicyR R O A H)
    (h0 : H) (horizon : Nat) : ∀ (l : List S) (r : R),
    (vecSpec E P h0 horizon l r).1
      = List.zipWith (fun s ρ => rolloutR E P horizon (s, h0, ρ)) l
          (vecStreams E P h0 horizon l r) := by
  intro l
  induction l with
  | nil => intro r; rfl
  | cons s rest ih =>
    intro r
    rw [vecSpec_cons, vecStreams_cons, List.zipWith_cons_cons, ih]

/-! ### The capstone: sequential sum = parallel product return -/

/-- **Independent-seed sum = product return, zipped form.** For any seed list
    of the right length, the summed returns of the per-seed rollouts equal the
    return of the lockstep product system over `Fin l.length` with those
    per-component seeds. -/
theorem sum_zipWith_return_eq_pi (E : EnvR R S O A) (P : PolicyR R O A H)
    (h0 : H) (w : ℝ) (horizon : Nat) (l : List S) (ρs : List R)
    (hlen : ρs.length = l.length) :
    ((List.zipWith (fun s ρ => rolloutR E P horizon (s, h0, ρ)) l ρs).map
        (traceReturn w)).sum
      = traceReturn w (rolloutR (EnvR.pi fun _ : Fin l.length => E)
          (PolicyR.pi fun _ : Fin l.length => P) horizon
          ((fun i => l[i]), (fun _ => h0),
           (fun i => ρs[(i : Nat)]'(by rw [hlen]; exact i.isLt)))) := by
  rw [traceReturn_rolloutR_pi]
  have hz : List.zipWith (fun s ρ => rolloutR E P horizon (s, h0, ρ)) l ρs
      = List.ofFn (fun i : Fin l.length =>
          rolloutR E P horizon (l[i], h0, ρs[(i : Nat)]'(by rw [hlen]; exact i.isLt))) := by
    refine List.ext_getElem (by simp [hlen]) ?_
    intro i h1 h2
    simp
  rw [hz, List.map_ofFn, List.sum_ofFn]
  exact Finset.sum_congr rfl fun i _ => rfl

/-- **CAPSTONE: the sequential spec computes the parallel product's return.**
    The sum of `vecSpec`'s per-segment returns equals the return of the
    lockstep-parallel product system (`EnvR.pi`, independent per-component
    streams) seeded with the `vecStreams` sequence — the C vec layer's
    parallel semantics and the sequential collection spec agree on the
    training signal, exactly, at corresponding seeds. -/
theorem vecSpec_return_eq_pi (E : EnvR R S O A) (P : PolicyR R O A H)
    (h0 : H) (w : ℝ) (horizon : Nat) (l : List S) (r : R) :
    ((vecSpec E P h0 horizon l r).1.map (traceReturn w)).sum
      = traceReturn w (rolloutR (EnvR.pi fun _ : Fin l.length => E)
          (PolicyR.pi fun _ : Fin l.length => P) horizon
          ((fun i => l[i]), (fun _ => h0),
           (fun i => (vecStreams E P h0 horizon l r)[(i : Nat)]'(by
              rw [vecStreams_length]; exact i.isLt)))) := by
  rw [vecSpec_traces_eq_zipWith]
  exact sum_zipWith_return_eq_pi E P h0 w horizon l _ (vecStreams_length E P h0 horizon l r)

section Executable

variable {S : Type}

/-- **The executable, all the way up**: the total ℝ-discounted return of the
    buffers `vecRollout` ACTUALLY collects equals the return of the parallel
    product system with independent per-env streams at the corresponding
    (`vecStreams`) seeds. Sequential Lean collector ≡ parallel vec-layer
    semantics, on the training signal, exactly. -/
theorem vecRollout_return_eq_pi (env : Puffer.RL.Env S) (p : MLP)
    (horizon : Nat) (states : Array S) (rng0 : UInt64) (w : ℝ) :
    (((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1).toList.map
        (fun t => GAE.discountedReturn w
          (t.toList.map (fun tr => Puffer.FloatR.toReal tr.reward)))).sum
      = traceReturn w (rolloutR
          (EnvR.pi fun _ : Fin states.toList.length => (toLoopEnvR env).withAutoReset)
          (PolicyR.pi fun _ : Fin states.toList.length => toLoopPolicyR p) horizon
          ((fun i => states.toList[i]), (fun _ => ()),
           (fun i => (vecStreams ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) ()
              horizon states.toList rng0)[(i : Nat)]'(by
                rw [vecStreams_length]; exact i.isLt)))) := by
  rw [← vecSpec_return_eq_pi]
  have hlist : ((Puffer.RL.NNTrain.vecRollout env p horizon states rng0).1).toList.map
        (fun t => GAE.discountedReturn w
          (t.toList.map (fun tr => Puffer.FloatR.toReal tr.reward)))
      = (vecSpec ((toLoopEnvR env).withAutoReset) (toLoopPolicyR p) () horizon
          states.toList rng0).1.map (traceReturn w) := by
    have h := congrArg (List.map (fun L : List (Array Float × Nat × ℝ × Bool) =>
        GAE.discountedReturn w (L.map (fun x => x.2.2.1))))
      (vecRollout_refines env p horizon states rng0)
    simpa [List.map_map, Function.comp_def, projRT, projL, traceReturn, rewards,
      GAE.discountedReturn] using h
  rw [hlist]

end Executable

end Puffer.RL.Loop
