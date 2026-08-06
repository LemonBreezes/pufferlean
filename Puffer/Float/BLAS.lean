/-
# BLAS + cuBLAS accelerated kernels (the M7 GPU/BLAS path)

Native `@[extern]` twins of the GEMM-dominant hot path — a batched dense layer with
ReLU, `Y[N×H] = relu(X[N×D] · W[H×D]ᵀ + b[H])` (row-major) — with three backends:

* `denseForwardRefFFI`   — scalar C loops (left-folded; the in-C reference).
* `denseForwardBlasFFI`  — OpenBLAS `cblas_dgemm` (CPU, multithreaded, AVX-512).
* `denseForwardCublasFFI`— cuBLAS `cublasDgemm` (GPU; falls back to the CPU BLAS path
  if no device is usable, so the result is always correct).

**Amended oracle discipline.** The scalar/right-fold kernels in `Puffer.Float.FFI`
(`pufferffi.c`) are BIT-IDENTICAL to the verified Lean `dotF`/AD oracle. BLAS and cuBLAS
use blocked / parallel reductions, so they are NOT bit-identical — they match the oracle
only to floating-point tolerance (~1e-10..1e-13 relative). That is the deliberate trade:
the bit-exact scalar path stays the oracle-faithful default; this is the FAST,
tolerance-validated path (`puffer verify-blas` checks the tolerance and the 3-way
agreement; `puffer bench-blas` reports the scaling).

Compiled from `ffi/pufferblas.c` by the `pufferblasObj` Lake target and linked into the
`puffer` exe together with OpenBLAS + cuBLAS (see `lakefile.lean`). Mathlib-free.
-/
namespace Puffer.Float.BLAS

/-- Scalar reference: `Y[N×H] = relu(X[N×D]·W[H×D]ᵀ + b[H])`, row-major, left-folded. -/
@[extern "lean_ffi_dense_forward_ref"]
opaque denseForwardRefFFI (X W b : FloatArray) (N D H : USize) : FloatArray

/-- OpenBLAS `cblas_dgemm` batched dense forward + ReLU. Matches the reference to
    floating-point tolerance (blocked reduction ⇒ not bit-exact). -/
@[extern "lean_ffi_dense_forward_blas"]
opaque denseForwardBlasFFI (X W b : FloatArray) (N D H : USize) : FloatArray

/-- cuBLAS `cublasDgemm` batched dense forward + ReLU (GPU; CPU-BLAS fallback if no
    usable device). Matches the reference to floating-point tolerance. -/
@[extern "lean_ffi_dense_forward_cublas"]
opaque denseForwardCublasFFI (X W b : FloatArray) (N D H : USize) : FloatArray

/-- cuBLASLt **bf16 tensor-core** batched dense forward + ReLU (bf16 inputs, f32 accumulate —
    PufferLib's CUDA precision). The first GPU kernel toward matching PufferLib's throughput.
    Inputs are rounded to bf16 (8 mantissa bits) before the GEMM, so this matches the f64
    oracle only to ~1e-1 absolute / bf16 relative — the deliberate precision/speed trade
    PufferLib makes. Bias+ReLU are applied on the host in f64. CPU-BLAS f64 fallback if no
    usable device. -/
@[extern "lean_ffi_dense_forward_cublaslt_bf16"]
opaque denseForwardCublasLtBf16FFI (X W b : FloatArray) (N D H : USize) : FloatArray

/-- Device-RESIDENT bf16 GEMM throughput in GF/s: `reps` `cublasLtMatmul` iterations on buffers
    that never leave the GPU (no host↔device transfer, no f64→bf16 conversion). The tensor-core
    ceiling of our path — the gap to `denseForwardCublasLtBf16FFI` is the per-call transfer tax we
    still pay while activations live in f64 on the host. Returns `-1` if no usable device. -/
@[extern "lean_ffi_bench_cublaslt_bf16_resident"]
opaque benchCublasLtBf16ResidentFFI (N D H reps : USize) : Float

/-! ### Batched 2-layer MLP forward (the vectorized-rollout hot path)

`params` is the flat MLP (`W1[H·D], b1[H], W2[O·H], b2[O]`, `O=A+1`, as `flattenMLP`);
`Xb` is a batch of `N` observations, row-major `N·D`. Returns `Yb[N·O]` row-major,
`Yb = relu(Xb·W1ᵀ + b1)·W2ᵀ + b2`. `Ref` is scalar right-fold (per row bit-exact vs
`mlpForwardFFI`/`forwardAll`); `Blas` uses two `cblas_dgemm`s (tolerance, not bit-exact).
Used to batch the policy forward across the `N` parallel env instances at each rollout
timestep (`vecRolloutBatched`). -/

/-- Scalar batched MLP forward (right-fold; per-row bit-exact oracle). -/
@[extern "lean_ffi_mlp_forward_batch_ref"]
opaque mlpForwardBatchRefFFI (params Xb : FloatArray) (N D H O : USize) : FloatArray

/-- OpenBLAS batched MLP forward (two `cblas_dgemm`s; matches the reference to tolerance). -/
@[extern "lean_ffi_mlp_forward_batch_blas"]
opaque mlpForwardBatchBlasFFI (params Xb : FloatArray) (N D H O : USize) : FloatArray

/-- bf16 tensor-core batched 2-layer MLP forward with the intermediate activation kept
    **resident on the GPU** — GEMM1's bf16 output feeds GEMM2's bf16 input directly (bias+ReLU
    fused via a cuBLASLt epilogue), so only params+Xb upload and Yb downloads, with no host
    round-trip between layers. bf16 precision ⇒ matches the oracle to ~1e-1. CPU-f64 fallback. -/
@[extern "lean_ffi_mlp_forward_batch_cublaslt_bf16"]
opaque mlpForwardBatchCublasLtBf16FFI (params Xb : FloatArray) (N D H O : USize) : FloatArray

/-- Device-RESIDENT 2-layer MLP forward throughput in GF/s: weights, input and activations all
    stay on the GPU across `reps` forwards (no per-call transfer). The end-to-end ceiling of the
    resident forward — the rollout pattern once observations live on-device. `-1` if no device. -/
@[extern "lean_ffi_bench_mlp2_bf16_resident"]
opaque benchMlp2Bf16ResidentFFI (N D H O reps : USize) : Float

/-- Throughput (GF/s) of the persistent-policy pattern: weights uploaded ONCE, then per forward
    only the fresh obs uploads and the output downloads (H1 stays on-GPU). The C-level measurement
    of what the `mlpPolicy*` API delivers (accurate cudaEvent timing, no Lean-side per-call cost). -/
@[extern "lean_ffi_bench_mlp2_bf16_wres"]
opaque benchMlp2Bf16WeightsResidentFFI (N D H O reps : USize) : Float

/-! ### Persistent resident policy (the rollout pattern)

Upload the constant policy weights to the GPU ONCE (`mlpPolicyLoadFFI` → an opaque handle), then
run many forwards streaming only the fresh observations (`mlpPolicyForwardFFI`) — the weights and
the hidden activation H1 stay resident, so each timestep pays only an obs upload + logits/value
download. `mlpPolicyFreeFFI` releases the handle. This turns the weights-resident throughput into a
real, callable API a rollout can drive. Params/obs layout matches `mlpForwardBatchBlasFFI`. -/

/-- Load a 2-layer MLP policy onto the GPU (weights uploaded once). Returns an opaque handle, or
    `0` if no usable device (caller then falls back to the CPU/OpenBLAS forward). -/
@[extern "lean_ffi_mlp_policy_load"]
opaque mlpPolicyLoadFFI (params : FloatArray) (D H O : USize) : USize

/-- Re-upload weights into an existing handle (dims unchanged), reusing the device buffers — call
    after each optimizer step so ONE resident policy serves the whole training run. Returns `1`. -/
@[extern "lean_ffi_mlp_policy_update"]
opaque mlpPolicyUpdateFFI (handle : USize) (params : FloatArray) : UInt8

/-- Forward `N` observations through a resident policy handle: only Xb uploads and Yb downloads;
    the weights + hidden activation stay on the GPU. bf16 tensor cores ⇒ ~1e-1 vs the f64 oracle. -/
@[extern "lean_ffi_mlp_policy_forward"]
opaque mlpPolicyForwardFFI (handle : USize) (X : FloatArray) (N : USize) : FloatArray

/-- Release a resident policy handle (frees its device buffers). Returns `1` on success. -/
@[extern "lean_ffi_mlp_policy_free"]
opaque mlpPolicyFreeFFI (handle : USize) : UInt8

/-- Batched MLP+PPO gradient via BLAS (the training hot path): same contract as
    `Puffer.Float.FFI.mlpPPOGradBatchFFI` — sums `dObj/dparams` over the `N`-transition
    minibatch into a flat `FloatArray[P]` — but the forward and backward matmuls are
    `cblas_dgemm`s (only the per-row PPO objective→dOut stays scalar). Matches the scalar
    kernel/Lean oracle to tolerance (~1e-11), not bit-exactly. -/
@[extern "lean_ffi_mlp_ppo_grad_batch_blas"]
opaque mlpPPOGradBatchBlasFFI (params obsB acts advs rets oldlps : FloatArray) (N H D A : USize)
  (vfCoef entCoef clipEps : Float) : FloatArray

/-- Whole-MLP Muon step on the CPU (native C): Nesterov → 5-iter Newton–Schulz → decoupled weight
    decay for `W1`/`W2`, Nesterov+wd for the biases — a bit-exact port of `Puffer.FloatR.Muon.stepMat`/
    `stepVec` (and the GPU `cudaMuonStepMatFFI`), ~100× the pure-Lean `applyMuon`. `pm` is the COMBINED
    `[params(P); mom(P)]` buffer (flat `[W1|b1|W2|b2]` then the matching momentum); `grad` is the RAW
    summed minibatch gradient (`mlpPPOGradBatchBlasFFI` output) and `gscale` (= 1/N) makes it the mean.
    Returns the new `[params; mom]` (size `2·P`) — one buffer threaded, no split. Drives the CPU
    `train-ppo-cpu` step so it is gradient-bound, not Newton–Schulz-bound. -/
@[extern "lean_ffi_muon_step_mlp"]
opaque muonStepMlpBlasFFI (pm grad : FloatArray) (H D O : USize)
  (gscale lr wd mu eps : Float) : FloatArray

/-- Batched CNN+PPO gradient via BLAS (im2col + GEMMs): same contract as
    `Puffer.Float.FFI.cnnPPOGradBatchFFI` — sums `dObj/dparams` over the `N`-transition
    minibatch into a flat `FloatArray[P]` (convW, convB, W1, b1, W2, b2 layout). The conv
    is im2col + `cblas_dgemm`, the dense layers + backward are GEMMs; only the per-row PPO
    objective→dOut stays scalar. Matches the scalar kernel/Lean oracle to tolerance. -/
@[extern "lean_ffi_cnn_ppo_grad_batch_blas"]
opaque cnnPPOGradBatchBlasFFI (params obsB acts advs rets oldlps : FloatArray)
  (N C inH inW nF k s hidden A : USize) (vfCoef entCoef clipEps : Float) (nScalar : USize := 0) : FloatArray

/-- Batched LSTM+PPO truncated-BPTT gradient via BLAS, batched over the `B` env-sequences
    (each length `T`). Returns the SUM over the sequences of the per-sequence gradient — it
    equals `Σ_b` (`Puffer.Float.FFI.lstmPPOGradSeqFFI` on sequence `b`). Inputs are
    TIME-major (`obsB[(t·B+b)·D+d]`, `acts/…/terms[t·B+b]`); `h0s`/`c0s` are `B·H`. Only the
    per-row PPO objective→dOut stays scalar; the gate/output/backward are `cblas_dgemm`s.
    Matches the scalar kernel/Lean oracle to tolerance, not bit-exactly. -/
@[extern "lean_ffi_lstm_ppo_grad_batch_blas"]
opaque lstmPPOGradBatchBlasFFI (params obsB acts advs rets oldlps terms h0s c0s : FloatArray)
  (B T H D A : USize) (vfCoef entCoef clipEps : Float) : FloatArray

/-- **f32 tier** of `lstmPPOGradBatchBlasFFI` — same algorithm, all GEMMs `cblas_sgemm`, weights/
    activations staged to `float` once and the gradient accumulates in float (widened to `f64` only
    for the returned array). A further tolerance step past the f64-BLAS kernel (verified against it by
    `verify-lstm-grad-f32`), matching this project's usual f32-tier bar (~1e-5..1e-6 relative). Opt-in
    (`PUFFER_LSTM_F32=1` gates `trainPluginEnvRec`'s choice of this vs the f64-BLAS kernel). -/
@[extern "lean_ffi_lstm_ppo_grad_batch_blas_f32"]
opaque lstmPPOGradBatchBlasF32FFI (params obsB acts advs rets oldlps terms h0s c0s : FloatArray)
  (B T H D A : USize) (vfCoef entCoef clipEps : Float) : FloatArray

/-- **bf16 tensor-core tier** of `lstmPPOGradBatchBlasFFI` — same algorithm as the f32 tier, but the GPU
    BPTT GEMMs run on bf16 tensor cores (float buffers rounded to bf16 for the MAC, f32 accumulate via
    `CUBLAS_COMPUTE_32F_FAST_16BF`; the gate/PPO elementwise kernels stay f32). Another tolerance step
    past the f32 tier (bf16 has an 8-bit mantissa; verified vs the f64 path by `verify-lstm-grad-bf16`).
    No CPU bf16 form — the no-device fallback is the shared f32 CPU path. Opt-in (`PUFFER_LSTM_BF16=1`
    gates `trainPluginEnvRec`'s choice of this vs the f32 default). -/
@[extern "lean_ffi_lstm_ppo_grad_batch_blas_bf16"]
opaque lstmPPOGradBatchBlasBf16FFI (params obsB acts advs rets oldlps terms h0s c0s : FloatArray)
  (B T H D A : USize) (vfCoef entCoef clipEps : Float) : FloatArray

/-- **Batched LSTM forward step via BLAS** — the BLAS twin of `lstmFwdStepBatchFFI`
    (`Puffer.Float.FFI`, naive scalar loops, bit-exact vs `lstmCellF`). Same math, gate/head
    pre-activations via `cblas_dgemm` instead of scalar dot products — faster (SIMD-vectorized,
    blocked) but only tolerance-close, not bit-exact, matching `lstmPPOGradBatchBlasFFI`'s own
    trade. Returns `[hN(N·H); cN(N·H); out(N·O, O=A+1: logits then value)]`. -/
@[extern "lean_ffi_lstm_fwd_step_batch_blas"]
opaque lstmFwdStepBatchBlasFFI (params obs h c : FloatArray) (N D H A : USize) : FloatArray

/-- **f32 tier** of `lstmFwdStepBatchBlasFFI` — same algorithm via `cblas_sgemm`, staged to `float`
    throughout, widened to `f64` only for the output. Verified against the f64-BLAS twin by
    `verify-lstm-fwd-f32`. Opt-in (`PUFFER_LSTM_F32=1`). -/
@[extern "lean_ffi_lstm_fwd_step_batch_blas_f32"]
opaque lstmFwdStepBatchBlasF32FFI (params obs h c : FloatArray) (N D H A : USize) : FloatArray

/-- `1` if a CUDA device is usable, else `0`. -/
@[extern "lean_ffi_cuda_available"]
opaque cudaAvailableFFI (u : Unit) : UInt8

/-- Cap OpenBLAS's own thread pool (process-global). At the LSTM plugin trainer's problem shape (many
    small, sequential per-timestep GEMMs), the default all-cores threading measured WORSE than a small
    fixed count -- see `trainPluginEnvRec`, which calls this once at startup. Safe to call once per
    process: a `puffer train` invocation only ever runs one trainer to completion. -/
@[extern "lean_ffi_blas_set_threads"]
opaque blasSetThreadsFFI (n : USize) : IO Unit

end Puffer.Float.BLAS
