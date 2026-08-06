/-
# Native FFI kernels (the M7 performance path)

Hot kernels implemented in native C (`ffi/pufferffi.c`, compiled + linked by Lake via the
`pufferffiObj` custom `target` + the `puffer` exe's `moreLinkObjs` in `lakefile.lean`) and
called through Lean's `@[extern]`. The verified Lean implementations
stay as the ORACLE: each C kernel is validated bit-for-bit against its Lean twin
(`Puffer.FloatR.dotF`) and benchmarked (`puffer bench-ffi`, `puffer verify-ffi`).

`dotFFI` sums a `FloatArray` dot product in the SAME right-nested order as `dotF`
(`x₀·w₀ + (x₁·w₁ + ⋯)`, accumulated from the last element backward), so it is
bit-for-bit identical to `dotF` — the native kernel is a drop-in, oracle-validated
replacement for the reduction that dominates the forward pass.

Mathlib-free; 0 imports. `FloatArray` (unboxed contiguous f64) is the FFI-friendly
array type — its C representation is a `lean_sarray` of doubles (`lean_float_array_cptr`).
-/
namespace Puffer.Float.FFI

/-- Smoke test: `2x + 1` in native C (validates the FFI linkage). -/
@[extern "lean_ffi_test"]
opaque testFFI (x : Float) : Float

/-- Native dot product of two `FloatArray`s, bit-for-bit matching `Puffer.FloatR.dotF`
    (right-nested summation, accumulated from the last element backward). -/
@[extern "lean_ffi_dot"]
opaque dotFFI (x w : FloatArray) : Float

/-- Reference dot in pure Lean, in `dotFFI`'s (and `dotF`'s) right-fold order —
    the oracle the native kernel is validated against. -/
def dotRef (x w : FloatArray) : Float := Id.run do
  let n := min x.size w.size
  let mut acc := 0.0
  for i in [0:n] do
    let j := n - 1 - i          -- last element first ⇒ right-nested association
    acc := x[j]! * w[j]! + acc
  return acc

/-! ### MLP + PPO gradient (the training hot path)

`params` is one flat `FloatArray`: `W1[H·D]`, `b1[H]`, `W2[O·H]`, `b2[O]` (`O = A+1`),
row-major. These are the native twins of `Puffer.RL.NNTrain.mlpGradPPO`'s forward +
objective + backward, FD-validated and cross-checked against that Lean oracle. -/

/-- Single-transition PPO objective PRIMAL in native C (for FD gradient checks). -/
@[extern "lean_ffi_mlp_ppo_obj1"]
opaque mlpPPOObj1FFI (params obs : FloatArray) (H D A a : USize)
  (adv ret oldLogp vfCoef entCoef clipEps : Float) : Float

/-- Batched MLP+PPO gradient in native C: sums `dObj/dparams` over `N` transitions
    (`obsB` is `N·D` row-major; `acts`/`advs`/`rets`/`oldlps` are length-`N`) into a
    flat `FloatArray[P]` (same layout as `params`). -/
@[extern "lean_ffi_mlp_ppo_grad_batch"]
opaque mlpPPOGradBatchFFI (params obsB acts advs rets oldlps : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps : Float) : FloatArray

/-- Batched rollout action sampler in native C: over the `N×O` logit batch `Yb` (row-major, `O=A+1`:
    `A` policy logits then the value), samples a categorical action per env, and returns
    `[actions(N); logps(N); values(N)]` (size `3·N`, `actions` as f64 integers). `rng` is the starting
    splitmix64 STATE; env `n` uses word `hash(rng + (n+1)·G)` — exactly the per-env `rngNext` stream — so
    the result is BIT-IDENTICAL to `softmax`→`sampleCat`→`log`, and the caller advances `rng` by `N·G`
    (O(1)). Replaces the per-env Lean softmax/sample/logp glue (4 small Array allocs/env) with one call. -/
@[extern "lean_ffi_sample_actions_batch"]
opaque sampleActionsBatchFFI (Yb : FloatArray) (N A O : USize) (rng : UInt64) : FloatArray

/-- Fill the flat env-major obs column for one rollout timestep `s`: `obsCol[(e·T+s)·D+j] = xb[e·D+j]`,
    where `xb` is the `N·D` batch already built for the forward. In place (the caller threads `obsCol`
    linearly). Lets the SoA rollout keep obs UNBOXED so the minibatch gather is a C copy, not a boxed
    `Array Float` walk. -/
@[extern "lean_ffi_scatter_obs"]
opaque scatterObsFFI (obsCol xb : FloatArray) (N D T s : USize) : FloatArray

/-- `arr[off : off+len]` as one C memcpy — the C-speed replacement for the interpreted
    `mk ((Array.range len).map (fun i => arr[off+i]!))` column extraction in the resident-rollout glue.
    Owned convention; calling it several times on the same `arr` is fine (Lean incs per extra use). -/
@[extern "lean_ffi_slice"]
opaque sliceFFI (arr : FloatArray) (off len : USize) : FloatArray

/-- **GAE (truncated, no bootstrap) in native C** — the `buildBatchSoA` computation bit-for-bit, over the
    SoA `values`/`rewards`/`terminals` columns (`N` segments of length `T`, row `e·T+t`). Backward per
    segment: `nnt = term?0:1`, `r = clamp(rew,-1,1)`, `δ = r + γ·V[t+1]·nnt − V[t]`, `A = δ + γλ·A·nnt`;
    `adv[T-1]=0`; `returns[t] = adv[t] + V[t]`. O(N·T) native replaces the boxed-Lean scan (the largest
    slice of rollout+GAE after the native rollout driver). Returns `[adv(N·T); returns(N·T)]`. -/
@[extern "lean_ffi_gae_soa"]
opaque gaeSoaFFI (vals rews terms : FloatArray) (N T : USize) (gamma lam : Float) : FloatArray

/-- GAE with a bootstrap value + advantage batch-normalization in native C — the LSTM plugin trainer's
    `computeGAEBoot` + adv-normalize over TIME-MAJOR rollout columns (row = t·N+n). Returns
    `[adv(N·T); ret(N·T)]` time-major (advantages already normalized). Replaces the boxed-`Array Float`
    Lean GAE loop (per-element boxing was ~50ms/update). -/
@[extern "lean_ffi_lstm_gae_boot_norm"]
opaque lstmGaeBootNormFFI (vals rews terms bootV : FloatArray) (N T : USize) (gamma lam : Float) : FloatArray

/-- Flat per-epoch shuffle in C — the bit-exact twin of `epochs ×` `shuffleIdx` (Fisher–Yates + splitmix64
    `rngNext`), replacing the interpreted-Lean `permFlat` loop (`epochs·NT` `Array.push`, the biggest
    host-side per-update cost). Returns `perm[epochs·NT]` (f64-encoded indices; epoch `e` = `perm[e·NT …]`).
    The caller advances its own rng by `epochs·NT·G` (rngNext's state is `+G`/call ⇒ that is exactly the
    post-shuffle rng). Pure: deterministic in `(NT, epochs, rng)`. -/
@[extern "lean_ffi_shuffle_perm"]
opaque shufflePermFFI (NT epochs : USize) (rng : UInt64) : FloatArray

/-- Gather the shuffled minibatch rows `idxs` (f64-encoded) out of the flat SoA trajectory columns into
    the contiguous buffers the step kernels take: returns `(mbObs[Nmb·D], mbAct, mbAdv, mbRet, mbOlp)`.
    Replaces the per-index Lean push loops with one C pass (obs is a straight row copy). -/
@[extern "lean_ffi_gather_minibatch"]
opaque gatherMinibatchFFI (obsCol actions advs rets olps idxs : FloatArray) (Nmb D : USize) :
  FloatArray × FloatArray × FloatArray × FloatArray × FloatArray

/-- Prioritized-replay weights + sampling in one C pass: computes `prioW[e] = exp(α·log(Σ_t|adv|+ε))` over
    all `N` segments, normalizes, then draws `mbSegs` segments (splitmix64, `s_i = rng + i·GOLD`) via a
    cumulative lower_bound, and returns `(sampledIdx (as Float), mbPrio)`. Replaces the two interpreted-Lean
    O(N·T) / O(mbSegs·N) host loops. Bit-identical; the caller advances `rng` by `mbSegs·GOLD` afterward. -/
@[extern "lean_ffi_prio_sample"]
opaque prioSampleFFI (advFlat : FloatArray) (N T mbSegs : USize) (prioAlpha annealBeta : Float) (rng : UInt64) :
  FloatArray × FloatArray

/-- MinGRU minibatch gather of the 6 scalar buffers the BPTT kernel takes (`pact, padv, pret, pold, pterm,
    pov`), for the `Bmb` sampled segments. Obs is NO LONGER gathered here — it is device-resident (the rollout
    scatters it to `g_dMGObsTraj`, the BPTT gathers it on-device). `advMean/advStd` (the sampled minibatch's
    advantage mean/std) are computed inside this pass; `mbPrio` weights `padv`. Bit-identical to the Lean
    loops (same reads, same op order). -/
@[extern "lean_ffi_gather_seq_minibatch"]
opaque gatherSeqMinibatchFFI (obsCol actCol logpCol termCol advFlat valueBuf segIdx mbPrio : FloatArray)
  (T Bmb D N : USize) :
  FloatArray   -- ONE packed [6·T·Bmb] buffer = [act | adv | ret | old | term | ov] (was 6 arrays)

/-- Batched MLP+PPO gradient WITH PufferLib value-loss clipping (`torch_pufferl.py`): extra
    `oldvals` (V at collection = PufferLib's `mb_values`) + `vfClip`; `v_loss` uses the clipped
    `max((V−R)², (v_clipped−R)²)` form. `vfClip ≤ 0` reduces to `mlpPPOGradBatchFFI`. -/
@[extern "lean_ffi_mlp_ppo_grad_batch_vclip"]
opaque mlpPPOGradBatchVclipFFI (params obsB acts advs rets oldlps oldvals : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps vfClip : Float) : FloatArray

/-- Native MLP forward (the rollout hot path): `out = b2 + W2·relu(b1 + W1·obs)`, the
    `O` outputs, bit-for-bit matching `forwardAll` (right-folded dots). -/
@[extern "lean_ffi_mlp_forward"]
opaque mlpForwardFFI (params obs : FloatArray) (H D O : USize) : FloatArray

/-- Single-transition Gaussian-PPO objective PRIMAL (continuous head; for FD checks).
    `act` is the `A`-dim continuous action. -/
@[extern "lean_ffi_gauss_ppo_obj1"]
opaque gaussPPOObj1FFI (params obs act : FloatArray) (H D A : USize)
  (adv ret oldLogp vfCoef entCoef clipEps : Float) : Float

/-- Batched Gaussian (continuous) MLP+PPO gradient (`dout` width `O = 2A+1`: means,
    log-stds, value); `actsB` is `N·A` row-major. Native twin of `mlpGradPPOCont`. -/
@[extern "lean_ffi_gauss_ppo_grad_batch"]
opaque gaussPPOGradBatchFFI (params obsB actsB advs rets oldlps : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps : Float) : FloatArray

/-- Gaussian-head gradient with value-loss clipping (see `mlpPPOGradBatchVclipFFI`). -/
@[extern "lean_ffi_gauss_ppo_grad_batch_vclip"]
opaque gaussPPOGradBatchVclipFFI (params obsB actsB advs rets oldlps oldvals : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps vfClip : Float) : FloatArray

/-! ### CNN-encoder head (spatial obs; the largest AD tape)

`params` is one flat `FloatArray`: `convW[nF·(C·k·k)]`, `convB[nF]`, `W1[hidden·flatDim]`,
`b1[hidden]`, `W2[O·hidden]`, `b2[O]` (`O = A+1`, `flatDim = nF·outH·outW`,
`outH = (inH-k)/s+1`, `outW = (inW-k)/s+1`), row-major. Native twins of
`Puffer.RL.NNTrain.cnnGradPPO` (conv+dense forward, PPO objective, reverse-mode), the
biggest FFI win since the conv tape is enormous. FD-validated + oracle-checked. -/

/-- Single-transition CNN-PPO objective PRIMAL in native C (for FD gradient checks).
    `obs` is the flat `C·inH·inW` observation; `a` the chosen action. -/
@[extern "lean_ffi_cnn_ppo_obj1"]
opaque cnnPPOObj1FFI (params obs : FloatArray) (C inH inW nF k s hidden A a : USize)
  (adv ret oldLogp vfCoef entCoef clipEps : Float) : Float

/-- Batched CNN+PPO gradient in native C: sums `dObj/dparams` over `N` transitions
    (`obsB` is `N·(C·inH·inW)` row-major) into a flat `FloatArray[P]` (params layout). -/
@[extern "lean_ffi_cnn_ppo_grad_batch"]
opaque cnnPPOGradBatchFFI (params obsB acts advs rets oldlps : FloatArray)
  (N C inH inW nF k s hidden A : USize) (vfCoef entCoef clipEps : Float) : FloatArray

/-- CNN head gradient with value-loss clipping (see `mlpPPOGradBatchVclipFFI`). -/
@[extern "lean_ffi_cnn_ppo_grad_batch_vclip"]
opaque cnnPPOGradBatchVclipFFI (params obsB acts advs rets oldlps oldvals : FloatArray)
  (N C inH inW nF k s hidden A : USize) (vfCoef entCoef clipEps vfClip : Float) : FloatArray

/-! ### LSTM head — truncated-BPTT gradient (recurrence in C over a whole sequence)

`params` is one flat `FloatArray`: `Wx[4H·D]`, `Wh[4H·H]`, `bih[4H]`, `Wo[dout·H]`,
`bo[dout]` (`dout = A+1`), gate rows stacked `i|f|g|o`. Operates on ONE env-sequence
(length `T`): `obsSeq` is `T·D`; `acts`/`advs`/`rets`/`oldlps`/`terms` are length-`T`
(`terms[t]≠0` marks a terminal, so the next step starts from a detached zeroed state
and the gradient does not flow across the boundary); `h0`/`c0` are the detached
BPTT-initial state (length `H`). Native twin of `Puffer.RL.NNTrain.recPPOGradSeq` — the
forward unroll on one AD tape + reverse-mode IS BPTT. FD-validated + oracle-checked. -/

/-- Summed per-step PPO objective over one LSTM sequence (primal, for FD gradient checks). -/
@[extern "lean_ffi_lstm_ppo_obj_seq"]
opaque lstmPPOObjSeqFFI (params obsSeq acts advs rets oldlps terms h0 c0 : FloatArray)
  (T H D A : USize) (vfCoef entCoef clipEps : Float) : Float

/-- Truncated-BPTT gradient of the summed PPO objective over one LSTM sequence, returned
    as a flat `FloatArray[P]` (params layout). Native twin of `recPPOGradSeq`. -/
@[extern "lean_ffi_lstm_ppo_grad_seq"]
opaque lstmPPOGradSeqFFI (params obsSeq acts advs rets oldlps terms h0 c0 : FloatArray)
  (T H D A : USize) (vfCoef entCoef clipEps : Float) : FloatArray

/-- LSTM BPTT gradient with value-loss clipping (extra `oldvals` + `vfClip`; see
    `mlpPPOGradBatchVclipFFI`). -/
@[extern "lean_ffi_lstm_ppo_grad_seq_vclip"]
opaque lstmPPOGradSeqVclipFFI (params obsSeq acts advs rets oldlps terms h0 c0 oldvals : FloatArray)
  (T H D A : USize) (vfCoef entCoef clipEps vfClip : Float) : FloatArray

/-- **Batched LSTM forward step for the rollout** — the native-C twin of `lstmCellF`, batched over
    the whole `N`-row batch in one call (replaces the per-env Lean Array glue that became the
    trainer's bottleneck once `lstmPPOGradSeqFFI` took over the BPTT step). `params` is the
    `flattenRec` layout; `obs`/`h`/`c` are `N·D`/`N·H`/`N·H` row-major. Returns
    `[hN(N·H); cN(N·H); out(N·O, O=A+1: logits then value)]` — `out` feeds directly into
    `sampleActionsBatchFFI`. Same left-fold order as `dotL` ⇒ bit-exact vs `lstmCellF`. -/
@[extern "lean_ffi_lstm_fwd_step_batch"]
opaque lstmFwdStepBatchFFI (params obs h c : FloatArray) (N D H A : USize) : FloatArray

/-- Native MinGRU BPTT PPO gradient (PufferLib's default net: encoder → `numLayers` MinGRU →
    decoder), with value-loss clipping. Params/gradient in the `flattenMG` layout; one
    zero-initial-state env-sequence, state reset at terminals. The C twin of `mingruGradSeq`. -/
@[extern "lean_ffi_mingru_ppo_grad_seq"]
opaque mingruPPOGradSeqFFI (params obsSeq acts advs rets oldlps terms oldvals : FloatArray)
  (T H obsSize numLayers A : USize) (vfCoef entCoef clipEps vfClip : Float) : FloatArray

end Puffer.Float.FFI
