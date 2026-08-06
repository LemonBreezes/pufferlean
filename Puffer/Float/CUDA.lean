/-!
# Native CUDA kernels (`@[extern]` twins of `ffi/puffercuda.cu`)

The nvcc-compiled GPU training-step layer. Each kernel here is verified against the machine-checked
Lean oracle at the stated precision (`verify-*-gpu` modes) — the GPU counterpart of `Puffer.Float.BLAS`.
These run in f64 to stay tight against the f64 oracle (V-Trace scan is bit-exact; Muon's Newton–Schulz
via f64 cuBLAS matches to ~1e-13). See `docs/gpu-training-scope.md`.
-/
namespace Puffer.Float.CUDA

/-- M0 build-integration self-test: `Y[i] = 2·i + 1` computed on the GPU (CPU fallback if no
    device). Proves the nvcc target compiles, links through Lean's toolchain, and launches. -/
@[extern "lean_cuda_selftest"]
opaque cudaSelftestFFI (n : USize) : FloatArray

/-- M1: V-Trace advantage on the GPU. `rewards/values/terminals/importance` are `B·T` row-major
    (each row a segment, Lean `Transition` convention). Returns advantages `B·T`. One thread per
    segment, f64 sequential scan with the vec delta `ρ·(r+γV′·nnt−V)` — bit-exact vs
    `computePuffAdvantageV` (same op order, `--fmad=false`). CPU fallback if no device. -/
@[extern "lean_cuda_vtrace"]
opaque cudaVtraceFFI (rewards values terminals importance : FloatArray) (B T : USize)
  (gamma lam rhoClip cClip : Float) : FloatArray

/-- M1b: V-Trace advantage for the MinGRU trainer (GPU). One thread per segment, f64 backward scan
    with the MinGRU delta `ρ·r + γV′·nnt − V` (ρ on the reward only) and the last step bootstrapped
    by `bootv[row]` (V(s_T)) — matching the trainer's original per-segment closure / `vtraceMinGRUFlat`
    op-for-op (`--fmad=false`) ⇒ bit-exact. `rewards/values/terms/imps` are B·T row-major; `bootv` is B.
    Returns advantages[B·T]. CPU fallback if no device. -/
@[extern "lean_cuda_vtrace_mingru"]
opaque cudaVtraceMinGRUFFI (rewards values terms imps bootv : FloatArray) (B T : USize)
  (gamma lam rhoClip cClip : Float) : FloatArray

/-- M2: Muon step for one 2D weight matrix on the GPU — Nesterov → Newton–Schulz (5 iters,
    `muonCoeffs`) → decoupled weight decay, ascent form, matching `Puffer.Float.Muon.stepMat`.
    `W/grad/mom` are `rows·cols` row-major (f64). Returns `[newW; newMom]` (size `2·rows·cols`).
    The NS matmuls are naive f64 kernels in Lean's summation order (`--fmad=false`) ⇒ bit-exact. -/
@[extern "lean_cuda_muon_stepmat"]
opaque cudaMuonStepMatFFI (W grad mom : FloatArray) (rows cols : USize)
  (lr wd mu eps : Float) : FloatArray

/-- M3: batched MLP PPO gradient on the GPU — same contract as `mlpPPOGradBatchBlasFFI` (forward →
    per-row PPO objective backward → backward GEMMs → summed flat gradient `g[P]`). `bf16 = 1` is the
    PufferLib-precision default (bf16 operands / f32 accumulate, verify ~1e-2 round-then-compare);
    `bf16 = 0` is the f32 tight cross-check (~1e-5 vs the f64 oracle). Needs a device. -/
@[extern "lean_cuda_mlp_ppo_grad"]
opaque cudaMlpPpoGradFFI (params obsB acts advs rets oldlps : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps : Float) (bf16 : UInt8) : FloatArray

/-- M4: advantage normalize on the GPU — `advN[i] = (adv[i]-mean)/(std+1e-8)`, matching `normalizeAdv`.
    f64 sequential folds ⇒ bit-exact. -/
@[extern "lean_cuda_adv_normalize"]
opaque cudaAdvNormalizeFFI (adv : FloatArray) (n : USize) : FloatArray

/-- M5: one full PPO+Muon training step ON THE GPU, intermediates resident — normalize (M4) →
    gradient (M3, then ÷N to the mean gradient, PPO/PufferLib `.mean()` convention) → Muon (M2, matrices
    via Newton–Schulz + biases via stepVec). `pm` is the COMBINED `[params(P); mom(P)]` buffer (flat
    `[W1|b1|W2|b2]` then the matching Muon momentum); returns the new `[params; mom]` (size `2·P`), so the
    trainer threads one buffer with no per-minibatch split/recombine. `bf16=1` = bf16 tensor-core gradient
    (PufferLib default); `bf16=0` = f32 tight. Muon runs f64 on the widened gradient + f64 momentum. Device
    buffers persist across calls (`ts_buf` cache) so a training loop pays the mallocs once. Drives `train-ppo-gpu`. -/
@[extern "lean_cuda_train_step"]
opaque cudaTrainStepFFI (pm obsB acts advRaw rets oldlps : FloatArray) (N H D A : USize)
  (lr wd mu eps vfCoef entCoef clipEps : Float) (bf16 : UInt8) : FloatArray

/-- **Whole-update RESIDENT step**: one call runs ALL `epochs × numMB` minibatches on the device. The SoA
    columns (`obs` `NT·D`, `acts/adv/ret/olp` `NT`), the per-epoch shuffles `perm` (`epochs·NT` f64-encoded
    indices, minibatch `m` of epoch `e` = rows `perm[e·NT + m·mbSize …]`), and `pm=[params;mom]` upload
    ONCE; the device loops (gather → normalize → gradient → Muon, `pm` resident in f64, updated in place)
    and `pm` downloads ONCE. Collapses the per-minibatch host↔device transfers to O(1)/update — the win in
    the large-batch regime (see `bench-train-step`). Same result as looping `cudaTrainStepFFI` over the same
    `perm`; returns the new `[params; mom]` (`2·P`). `bf16=1` = bf16 tensor cores (PufferLib default). -/
@[extern "lean_cuda_train_update"]
opaque cudaTrainUpdateFFI (pm obs acts adv ret olp perm : FloatArray)
  (NT D H A epochs numMB : USize)
  (lr wd mu eps vfCoef entCoef clipEps : Float) (bf16 : UInt8) : FloatArray

/-- R2: device categorical sampler — the GPU twin of `Puffer.Float.FFI.sampleActionsBatchFFI`. Over the
    `N×O` logit batch (`O=A+1`), samples one action per env and returns `[actions(N); logps(N); values(N)]`.
    env `n` draws splitmix64 word `hash(rng + (n+1)·G)` — the same per-env stream as the CPU sampler — so
    given the same logits+rng the result matches to transcendental ULP (device `exp/log` vs libm). -/
@[extern "lean_cuda_sample_actions"]
opaque cudaSampleActionsFFI (Yb : FloatArray) (N A O : USize) (rng : UInt64) : FloatArray

/-- **Multi-discrete sampler** — `K` categorical heads (sizes `headSizes`, `O = Σsizes+1`). Over the `N×O`
    logit batch, samples each head and returns `[actions(N×K, col-major: head h of env n at h·N+n);
    jointLogp(N); value(N)]` (size `(K+2)·N`). Distinct per-(env,head) rng; `K=1` is a single categorical. -/
@[extern "lean_cuda_sample_actions_md"]
opaque cudaSampleActionsMDFFI (Yb headSizes : FloatArray) (N K O : USize) (rng : UInt64) : FloatArray

/-- **Multi-discrete minibatch PPO gradient** — the md twin of `cudaMlpPpoGradFFI`. Joint log-prob over the
    `K` heads, one PPO clip on the joint ratio, gradient decomposed per head. `acts` is `N·K` (row-major),
    `headSizes` the `K` per-head action counts (`O = Σsizes+1`). Returns the summed flat gradient `g[P]`. -/
@[extern "lean_cuda_mlp_ppo_grad_md"]
opaque cudaMlpPpoGradMDFFI (params obsB acts advs rets oldlps headSizes : FloatArray) (N H D K : USize)
  (vfCoef entCoef clipEps : Float) (bf16 : UInt8) : FloatArray

/-- **Continuous (diagonal-Gaussian) sampler** — `d` real action dims (`O = 2·d+1`: means, raw logstds,
    value). Over the `N×O` batch, samples `aᵢ = μᵢ + σᵢ·zᵢ` (`z~N(0,1)`, Box–Muller, clamped logstd) and
    returns `[actions(N×d, col-major: dim i of env n at i·N+n); logp(N); value(N)]` (size `(d+2)·N`). -/
@[extern "lean_cuda_sample_actions_cont"]
opaque cudaSampleActionsContFFI (Yb : FloatArray) (N d O : USize) (rng : UInt64) : FloatArray

/-- **Continuous minibatch PPO gradient** — the Gaussian twin of `cudaMlpPpoGradFFI` (see
    `ContVecTrain.lean`). `acts` is `N·d` real actions (row-major), `O = 2·d+1`; one PPO clip on the joint
    ratio, per-dim mean/logstd gradients (clamp-gated logstd) + entropy bonus. Returns flat gradient `g[P]`. -/
@[extern "lean_cuda_mlp_ppo_grad_cont"]
opaque cudaMlpPpoGradContFFI (params obsB acts advs rets oldlps : FloatArray) (N H D d : USize)
  (vfCoef entCoef clipEps : Float) (bf16 : UInt8) : FloatArray

/-- **Batched MinGRU forward step** (PufferLib's default recurrent policy on the GPU). One step over `N`
    envs: Linear encoder (obs→H) → `L` MinGRU layers → linear action+value heads, carrying the `N·L·H`
    recurrent state. `params` is the `flattenMG` layout `[wEnc|bEnc|layers|wDec|bDec|wVal|bVal]`, `obs` is
    `N·D`, `state` is `N·L·H`. Returns `[out(N·O, O=A+1: logits then value); newState(N·L·H)]`. GEMMs use the
    f32/bf16 `gemm32` path (PufferLib policy precision); the cell math matches `Net.MinGRU.stepForward`.
    Replaces the O(N·T·L) Lean rollout forward with one batched device call per timestep. -/
@[extern "lean_cuda_mingru_step"]
opaque cudaMinGRUStepFFI (params obs state : FloatArray) (N D H L A : USize) (bf16 : UInt8) : FloatArray

/-- **Batched MinGRU BPTT PPO gradient** (GPU twin of the C `mingruPPOGradSeqFFI`). `B` sequences of length
    `T`; inputs laid out `[T][B][…]` (`obs` `T·B·D`, the rest `T·B`). Forward stores activations, backward
    runs the PPO objective BPTT — cuBLAS GEMMs (f32) + custom cell/PPO kernels — summing the gradient over
    the batch. Returns the flat gradient `g[P]` (`flattenMG` layout). Replaces the single-threaded C BPTT.
    `obs` may be empty when relying on the device-resident obs trajectory: pass the sampled `segIdx` (segment
    indices as Floats) and the BPTT gathers obs on-device from the rollout's `g_dMGObsTraj` (no host obs). An
    empty `segIdx` falls back to uploading `obs`. -/
@[extern "lean_cuda_mingru_ppo_grad"]
opaque cudaMinGRUPpoGradFFI (params obs scal : FloatArray)
  (B T H D L A K : USize) (hs : FloatArray)
  (vfCoef entCoef clipEps vfClip : Float) (segIdx mbPrio : FloatArray) : IO FloatArray
  -- MULTI-DISCRETE: `A` is the LOGITS WIDTH (`W = Σ headSizes`) — every tensor of the BPTT follows from
  -- it, so the recurrent core is unchanged. `K` is the number of categorical heads (1 = single-discrete,
  -- the untouched path) and `hs` the `K` head sizes (ignored, may be empty, when `K = 1`). With `K > 1`
  -- the action column is `K` per row and the head gradient is `k_mg_ppo_b_md` (joint ratio, per-head
  -- softmax/entropy) instead of `k_mg_ppo_b`.
  -- `scal` = packed [(K+5)·T·B] host scalars ([act(K-wide)|adv|ret|old|term|ov]), OR empty + nonempty
  -- `mbPrio` ⇒ DEVICE-column mode: gather the scalars on-GPU from the resident columns (mgPrep) and
  -- iterate value/ratio on-GPU.
  -- **The RETURNED SIZE is the authoritative gradient MODE** (the caller must branch on it, never guess):
  --   `0`         ⇒ RESIDENT — the summed gradient stayed on-GPU; call `cudaMinGRUMuonResidentFFI` with an
  --                 EMPTY `gClip` (it gradclips + consumes it there).
  --   `P+2·T·B`   ⇒ HOST — `[0,P)` is the real summed gradient (then new_logp, new_value); the caller must
  --                 gradclip it and pass it as a non-empty `gClip`. Device-column mode falls back to this
  --                 form when the resident gradient buffer cannot be allocated.
  -- When NEITHER form can be produced (BPTT activation storage unavailable, or an empty `obs`/`scal` whose
  -- device-resident source is gone) this **throws** rather than returning zeros: a silent zero gradient
  -- used to let a whole run finish with rc=0 having learned nothing.
  -- IO: in device-column mode the call is pure side effect on device state — it must never be CSE'd
  -- or dead-code-eliminated.

/-- H2D the rollout's scalar columns to device globals ONCE per update (init valueBuf:=valCol, ratioBuf:=1);
    the device-resident minibatch prep (V-Trace/gather/iterate) then reads them. `K` = action components
    per row (1 = single-discrete; `K` heads for multi-discrete, `act` then being `N·T·K`, row `e·T+s`
    holding its `K` head actions contiguously). -/
@[extern "lean_cuda_mg_prep"]
opaque cudaMgPrepFFI (rew term act logp val boot : FloatArray) (N T K : USize) : IO Unit

/-- V-Trace on the device columns → g_dcAdv; returns Σ_t|adv| per segment (N doubles) for host sampling.
    IO (mutates/reads device state; must not be CSE'd across minibatches). -/
@[extern "lean_cuda_mg_vtrace"]
opaque cudaMgVtraceFFI (N T : USize) (gamma lam rhoClip cClip : Float) : IO FloatArray

/-- **Dashboard loss surfacing** — enable (`on=1`) / disable per-minibatch loss reduction inside the
    MinGRU BPTT grad (`lean_cuda_mingru_ppo_grad`). Off by default (zero cost); the `--log` dashboard
    turns it on. Read-only D2H reduction, no training buffer written ⇒ determinism-safe. -/
@[extern "lean_cuda_mg_loss_enable"]
opaque cudaMgLossEnableFFI (on : UInt8) : IO Unit

/-- Read the 7 most-recent dashboard losses computed by the grad: `[policy, value, entropy, total,
    old_kl, kl, clipfrac]`. Zeros until the first grad call with loss surfacing enabled. -/
@[extern "lean_cuda_mg_read_losses"]
opaque cudaMgReadLossesFFI : IO FloatArray

/-- R3: device MLP forward `relu(obs·W1ᵀ+b1)·W2ᵀ+b2` via the f32/bf16 `gemm32` path. `params` flat
    `[W1|b1|W2|b2]`, `obs` `N·D` (f64); returns logits `N·O`. The per-timestep forward the rollout driver
    uses, exposed host-in/host-out as the reference component for `verify-rollout-gpu`. -/
@[extern "lean_cuda_mlp_forward"]
opaque cudaMlpForwardFFI (params obs : FloatArray) (N D H O : USize) (bf16 : UInt8) : FloatArray

/-- **Native per-update MLP rollout** (single-discrete plugin trainer): runs the whole `T`-horizon
    rollout in one FFI call — policy weights resident (uploaded once), per step obs H2D → resident forward
    → device sample → CPU plugin env-step (`h` is the `Puffer.Plugin` env handle) → host column scatter.
    Replaces the Lean per-timestep loop (per-step weight re-upload + malloc storm + logits round-trip +
    boxed-array scatter). IO — it steps (mutates) the env. Returns the SoA experience columns
    `[obsCol(NT·D); actCol; logpCol; valCol; rewCol; termCol; finalObs(N·D)]`, row `e·T+s`; `finalObs`
    threads the persistent env state to the next update. Bit-identical to the old loop (same forward,
    same `k_sample` rng = `rolloutRng + s·N·G`, same env stepping). -/
@[extern "lean_cuda_plugin_rollout"]
opaque cudaPluginRolloutFFI (h policyH : USize) (obs0 : FloatArray) (N D H A T : USize)
  (bf16 : UInt8) (rolloutRng : UInt64) : IO FloatArray

/-- **Enable the buffered + graph-replayed rollout path** for `cudaPluginRolloutFFI` (an N-way
    concurrent-stream buffer split + per-buffer CUDA graph replay of the per-step forward — already
    built, already measured breakout MLP@4096 4.73M→7.25M SPS, but dormant since nothing in the CLI
    ever set its `PUFFER_ROLL_BUFFERS`/`PUFFER_ROLL_GRAPH` env vars). `nbuf ≤ 1` reverts to the
    single-buffer path. Call once before the rollout loop starts (process-global, cached). -/
@[extern "lean_cuda_set_roll_buffers"]
opaque cudaSetRollBuffersFFI (nbuf : USize) (graphOn : UInt8) : IO Unit

/-- **Device-resident policy weights.** Upload the initial `[params(P); mom(P)]` (f64) to a persistent
    device buffer ONCE and get an opaque handle; the rollout reads its params and `cudaTrainUpdateResidentFFI`
    updates params+mom in place — no per-update PCIe round-trip of the weights (PufferLib keeps the policy
    on-device the whole run). `policyDownloadFFI` reads `[params;mom]` back on demand; `policyFreeFFI` frees it. -/
@[extern "lean_cuda_policy_load"]
opaque policyLoadFFI (pm : FloatArray) (P : USize) : IO USize
@[extern "lean_cuda_policy_download"]
opaque policyDownloadFFI (handle P : USize) : IO FloatArray
@[extern "lean_cuda_policy_free"]
opaque policyFreeFFI (handle : USize) : IO Unit

/-- Resident-weights twin of `cudaTrainUpdateFFI`: params+mom live in the `handle` device buffer, updated
    IN PLACE by Muon (no H2D of params / D2H of the result). Only the per-update columns/perm upload.
    Returns a 1-element array `[params[0]]` for the divergence guard. Bit-identical to the host-threaded path.

    `oldVal` is the rollout's value column (PufferLib's `mb_values`) and `vfClip`/`maxGradNorm` are
    `train.vf_clip_coef` / `train.max_grad_norm`: with them the value loss is PufferLib's CLIPPED
    `½·max((v−ret)², (v_clipped−ret)²)` and the minibatch gradient gets their global
    `clip_grad_norm_`. An EMPTY `oldVal` (or `vfClip ≤ 0` / `maxGradNorm ≤ 0`) selects the older
    unclipped/unnormalized arithmetic exactly. -/
@[extern "lean_cuda_train_update_resident"]
opaque cudaTrainUpdateResidentFFI (handle : USize) (obs acts adv ret olp oldVal perm : FloatArray)
  (NT D H A epochs numMB : USize)
  (lr wd mu eps vfCoef entCoef clipEps vfClip maxGradNorm : Float) (bf16 : UInt8) : IO FloatArray

/-- **Wide-action whole-update RESIDENT step** — the MD (`mode=1`) / Cont (`mode=2`) twin of
    `cudaTrainUpdateResidentFFI`. Uploads the SoA columns (`acts` is `NT·W`, `W`=K heads / d dims) + the
    `epochs·NT` shuffle ONCE, then loops `epochs×numMB` entirely on the GPU: gather → per-minibatch
    adv-normalize → forward → `k_ppo_dout_md`(headSizes)/`k_ppo_dout_cont` backward → in-place Muon over the
    resident `[params;mom]` handle. Moves the per-minibatch gather + adv-norm off interpreted Lean. `O` =
    `Σheads+1` (MD) / `2·d+1` (Cont); `headSizes` used only for `mode=1`. Returns `[params[0]]` (guard).
    Skips the obs H2D+f32-convert entirely when the rollout left it device-resident (`g_dObsTraj_valid`,
    same flag/buffer `cudaPluginRolloutFFI`'s sibling uses — ported here too, was previously always doing
    the ~7.7M-cast host round-trip every update). Bit-identical to the old per-minibatch host-gather path
    either way. IO (mutates the resident buffer).

    `oldVal`/`vfClip`/`maxGradNorm` carry PufferLib's clipped value loss + `clip_grad_norm_`, exactly as
    on the single-discrete sibling (empty `oldVal` / non-positive coefficients ⇒ the older arithmetic). -/
@[extern "lean_cuda_train_update_wide_resident"]
opaque cudaTrainUpdateWideResidentFFI (handle : USize) (obs acts adv ret olp oldVal perm headSizes : FloatArray)
  (NT D H O W : USize) (mode : UInt32) (epochs numMB : USize)
  (lr wd mu eps vfCoef entCoef clipEps vfClip maxGradNorm : Float) (bf16 : UInt8) : IO FloatArray

/-- **Native per-update rollout for MULTI-DISCRETE (`mode=1`) / CONTINUOUS-Gaussian (`mode=2`)** — the
    W-wide-action twin of `cudaPluginRolloutFFI` (W = K heads / d dims). Same resident-weights device
    forward, but the sampler is `k_sample_md`/`k_sample_cont` and the action column is `NT·W` row-major.
    `headSizes` is used only for `mode=1` (`mode=2` ignores it). `O` = net head width (`Σheads+1` / `2·d+1`).
    Returns `[obsCol(NT·D); actCol(NT·W); logpCol(NT); valCol(NT); rewCol(NT); termCol(NT); finalObs(N·D)]`,
    row `e·T+s`. Also scatters the f32 obs into the device-resident trajectory buffer (`g_dObsTraj`, shared
    with `cudaPluginRolloutFFI`'s single-discrete sibling), so `cudaTrainUpdateWideResidentFFI` can skip its
    own obs re-upload. Also has the SAME buffered + CUDA-graph-replayed rollout path as the single-discrete
    sibling (`cudaSetRollBuffersFFI`-gated, W-wide via new `k_sample_md_seg_f32`/`k_sample_cont_seg_f32`
    kernels + a dedicated `bufpool_wide_t`/`buf_worker_wide` thread pool), including the same PINNED host
    staging when buffered (`hb_buf` slots 4-7, so `cudaMemcpyAsync` genuinely overlaps instead of silently
    falling back to a sync-equivalent copy through pageable memory). Bit-identical to the old per-step FFI
    loop in the non-buffered case (same forward, same sampler rng); buffered is tolerance-close (bf16
    tiling + f32 sampling), same trade as the MLP sibling's buffered rollout. IO. -/
@[extern "lean_cuda_plugin_rollout_multi"]
opaque cudaPluginRolloutMultiFFI (h policyH : USize) (obs0 headSizes : FloatArray) (N D H O W T : USize)
  (mode : UInt32) (bf16 : UInt8) (rolloutRng : UInt64) : IO FloatArray

/-- Resident in-place Muon over an MLP policy handle (`[params;mom]`, layout `[W1|b1|W2|b2]`) — the resident
    twin of `muonStepMlpBlasFFI` for the MD/Cont plugin trainers. `gRaw` is the raw summed minibatch gradient;
    `gscale` (=1/N) mean-scales it. Orthogonalizes `W1`/`W2` (`muon_mat_dev` == `muon_mat_cpu`) and Nesterov-
    steps `b1`/`b2` (`k_stepvec` == `stepvec_cpu`) IN PLACE — uploading only `gRaw`. Returns the new `params(P)`
    for the (unchanged, host-param) MD/Cont grad. `O` = `Σheads+1` (MD) / `2·d+1` (Cont). Bit-identical. IO. -/
@[extern "lean_cuda_muon_step_mlp_resident"]
opaque cudaMuonStepMlpResidentFFI (handle : USize) (gRaw : FloatArray) (H D O : USize)
  (gscale lr wd mu eps : Float) : IO FloatArray

/-- **Native per-update MinGRU rollout** (recurrent single-discrete): the MinGRU twin of
    `cudaPluginRolloutFFI`. Weights upload+convert to f32 ONCE (the old loop re-uploaded ~1.7MB every
    timestep), recurrent state (`N·L·H`) stays f32 RESIDENT on device (threaded + reset on terminals),
    forward = `mingru_fwd_dev` (encoder→L layers→heads), then device sample → CPU env-step. Returns
    `[obsCol(NT·D); actCol; logpCol; valCol; rewCol; termCol; finalObs(N·D); finalState(N·L·H); bootVals(N)]`
    — `finalState` threads the persistent recurrent state, `bootVals` is V(s_T) for the last segment's
    advantage bootstrap (folded in, was a separate forward). Bit-identical to the old loop. IO. -/
@[extern "lean_cuda_plugin_rollout_mingru"]
opaque cudaPluginRolloutMinGRUFFI (h policyH : USize) (obs0 state0 : FloatArray) (N D H L A T : USize)
  (wantLog : UInt8) (rolloutRng : UInt64) : IO FloatArray

/-- **Native per-update MinGRU rollout, MULTI-DISCRETE** (`K` categorical heads, sizes `headSizes`,
    logits width `W = Σ headSizes`). The K-head twin of `cudaPluginRolloutMinGRUFFI`, and since the
    fused-MD migration it runs that driver's FAST path: concurrent stream-buffers, ONE fused
    `k_mg_fused_step_w_md` launch per (step, buffer) — obs-traj scatter + encoder + gate layers (folded
    terminal reset) + heads + per-head sampling + resident-column writes — pinned zero-copy obs, the
    per-(t,buf) CUDA-graph table, and resident obs/state chaining. The K-wide pieces the old comment
    called impossible are parameterised: the sampler writes `K` row-major actions per row into a pinned
    plane the envs read directly (`act[e·K+h]`), the action D2H is `8·nb·K` bytes, and the K-wide action
    column is stamped straight into the resident device column.

    The LEGACY non-fused arm (whole-batch forward → `k_sample_md` → threaded env-step → host column
    scatter) remains as the fallback and is taken automatically when the fused kernel cannot express the
    shape (`H % 16 ≠ 0`, `H > 128`, `W+1 > H`, or the shared-memory budget), when `PUFFER_MG_WMMA=0`,
    when a needed pinned/device buffer cannot be allocated — or on demand via `PUFFER_MG_MD_FUSED=0`.
    Which arm ran is printed once to stderr. The single-discrete arm is bit-for-bit untouched (separate
    C function, separate kernel).

    Obs are NOT returned: the device-resident obs trajectory (`g_dMGObsTraj`, which the BPTT gathers
    from) is REQUIRED and the call aborts if it cannot be allocated. Returns
    `[actCol(NT·K, row e·T+s); logpCol(NT); valCol(NT); rewCol(NT); termCol(NT); finalObs(N·D);
    finalState(N·L·H); bootVals(N)]`; `logp` is the JOINT log-prob `Σ_h log p_h(a_h)`, the convention
    `k_mg_ppo_b_md` differentiates. Under the FUSED arm the act/logp/val columns of that return are left
    UNWRITTEN — they live in the resident device columns instead, so callers must check
    `cudaMgColsReadyFFI` before reading them (exactly as on the single-discrete path). Passing EMPTY
    `obs0`/`state0` requests the CHAINED form and returns just `[rewCol(NT); termCol(NT)]`. IO (steps
    the env). -/
@[extern "lean_cuda_plugin_rollout_mingru_md"]
opaque cudaPluginRolloutMinGRUMDFFI (h policyH : USize) (obs0 state0 headSizes : FloatArray)
  (N D H L W K T : USize) (rolloutRng : UInt64) : IO FloatArray

/-- **Resident chaining (Conn-lite)**: after a chain-capable rollout (buffered + device-direct columns +
    f32obs staging), obs live on in the pinned ping-pong and the recurrent state in the device buffer —
    a same-shape follow-up call may pass EMPTY `obs0`/`state0` and gets back only `[rewCol(NT);
    termCol(NT)]` when `wantLog=1`, else an EMPTY array — the ~NT·(D+5)-sized full return and the f64
    finalObs/finalState round trips vanish. BIT-IDENTICAL (the removed round trips were exact
    widen/narrow identities). Query this after the first rollout; `PUFFER_MG_CHAIN=0` disables. -/
@[extern "lean_cuda_mg_chain_ready"]
opaque cudaMgChainReadyFFI (N D LH : USize) : IO Bool

/-- Resident in-place Muon over the MinGRU policy handle (`[weights(P); mom(P)]`, shared with the rollout):
    orthogonalizes each `flattenMG` matrix and Nesterov-steps each bias (`k_stepvec` == the host formula in
    `muonStepFlatMG`), all in place. Two gradient modes: `gClip` NON-empty ⇒ upload the host-clipped
    gradient (the original contract); `gClip` EMPTY ⇒ consume the RESIDENT gradient the device-column BPTT
    left on-GPU, gradclipping there with Lean's exact formula (`cc = (if maxGradNorm>0 ∧ gnorm>maxGradNorm
    then maxGradNorm/gnorm else 1)·gscale`, `gnorm = √Σ(g·gscale)²` — tree-order sum, ~1 ulp vs the host
    fold). Returns the new `wFlat(P)` (EMPTY under the resident-gradient contract). IO (mutates the
    resident buffer).

    Which form to pass is decided by `cudaMinGRUPpoGradFFI`'s RETURN SIZE, not independently: empty return
    ⇒ empty `gClip`. Passing an empty `gClip` with no resident gradient present **throws** — it used to
    warn once, skip the step entirely and hand back a zero `wFlat`, so the trainer kept reporting healthy
    updates/SPS and exited 0 while learning nothing. -/
@[extern "lean_cuda_mingru_muon_resident"]
opaque cudaMinGRUMuonResidentFFI (handle : USize) (gClip : FloatArray) (H D L A : USize)
  (lr wd mu eps maxGradNorm gscale : Float) : IO FloatArray

/-- Did THIS update's rollout stamp the device-resident scalar columns directly (device-direct mode)?
    Consumes the stamp: true at most once per rollout, so a stale stamp can never suppress a needed
    `cudaMgPrepFFI`. When true, the six column slices + prep upload are skipped entirely. -/
@[extern "lean_cuda_mg_cols_ready"]
opaque cudaMgColsReadyFFI (N T : USize) : IO Bool

/-- Device V-Trace + prioritized sampling in one call: runs the resident-column V-Trace, then the
    prioritized-replay sampler ON DEVICE (verbatim splitmix64 stream and formulas; device exp/log differ
    from glibc at ulps ⇒ tolerance-class vs the host sampler), leaving segIdx/mbPrio RESIDENT for the
    BPTT (which is then called with BOTH arrays empty). Returns false (gated off / not ready) ⇒ fall
    back to `cudaMgVtraceFFI` + `prioSampleFFI`. The caller advances its rng by mbSegs·GOLD either way. -/
@[extern "lean_cuda_mg_vtrace_prio"]
opaque cudaMgVtracePrioFFI (N T : USize) (gamma lam rhoClip cClip : Float)
  (mbSegs : USize) (prioAlpha annealBeta : Float) (rng : UInt64) : IO Bool

/-- R6: **resident CNN encoder forward** (the "CNN path") — im2col → conv → relu → pixel→filter transpose
    → dense → logits, on the device (f32/bf16 `gemm32`). `params` flat `[convW|convB|W1|b1|W2|b2]`, `obs`
    `N·(C·inH·inW)`; returns logits `N·O`. The GPU twin of the CPU `cnnForward`; the conv-encoder any CNN
    env's rollout would use (breakout's device physics is the separate, larger, env-specific piece). -/
@[extern "lean_cuda_cnn_forward"]
opaque cudaCnnForwardFFI (params obs : FloatArray) (N C inH inW nF k s hidden O : USize) (bf16 : UInt8) : FloatArray

/-- R6 (breakout physics foundation): device `sinf`/`cosf` — bit-exact port of `Puffer.Numeric.SinCosF`
    (glibc sincosf polynomial, exact IEEE constants + correctly-rounded FMA). `isCos=0` → sin, `1` → cos.
    The precision crux for breakout's fire/paddle angles (and all `SinCosF` physics envs). -/
@[extern "lean_cuda_sincosf"]
opaque cudaSinCosFFI (angles : FloatArray) (N : USize) (isCos : UInt8) : FloatArray

end Puffer.Float.CUDA
