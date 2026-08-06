# Scope: an nvcc CUDA kernel layer for the GPU training step

**Question:** what would it take to move the RL *training step* (V-Trace advantage → PPO
backward/gradient → Muon → advantage-normalize → prioritized replay → action sampling) onto the
GPU, so the whole loop is GPU-resident like PufferLib — without abandoning this project's
verification discipline? (Envs stay Lean/CPU; obs upload is accepted, as established earlier.)

**Answer:** a **staged GO**. The FLOP-heavy work is already cuBLAS; the custom-kernel surface is
small and mostly verbatim ports of PufferLib's CUDA. The real cost is not lines of code — it's the
per-kernel **verification contract**, the **RNG-determinism** decision, and a **residency refactor**
that only prioritized replay forces. Recommended first slice: build-MVP + V-Trace + Muon.

---

## The key insight: it's mostly cuBLAS

For the MLP training step, cuBLAS/cuBLASLt already covers **100% of the matmul-shaped work**:

- the 2 forward GEMMs run resident today (the `mlpPolicy*` path),
- all **5 PPO-backward GEMMs** (dW2, dH1, dW1, …) are plain `cublasLtMatmul`,
- all **15 Newton–Schulz GEMMs** in Muon are `cublasGemmEx`.

So the genuinely custom `__global__` surface is small. Total inventory ≈ **17 kernels, but ~12 are
< 30 LOC** elementwise/scan/reduction, and most port near-verbatim from PufferLib's `src/pufferlib.cu`
and `src/muon.cu`.

| Group | Custom kernels | Notes |
|---|---|---|
| **PPO grad** | `ppo_dout` (~40 LOC, the one irreducible kernel), relu-mask, bias column-sum | relu-mask → a cuBLASLt DRELU epilogue = 0 custom; bias-sum → ones-vector GEMM = 0 custom |
| **Muon** | `muon_nesterov`, `muon_weight_update`, `muon_norm_{partials,reduce,apply}` (+ optional grad-clip) | already written in `muon.cu`; strip the bf16 wrappers → plain f32. The 15 NS GEMMs are cuBLAS |
| **Advantage** | `puff_advantage` (V-Trace scan, ~20 LOC), `ppo_var_mean` (~38 LOC tree reduction), adv-normalize | V-Trace = the exact vec-delta we already verified |
| **Prio replay** *(defer)* | prio-reduction, normalize, build_cdf, imp-weights, multinomial, index/select-copy | + a rollout-buffer residency refactor — the real hidden cost |
| **Sampling** *(optional)* | `sample_logits` discrete branch (~60 LOC) + device splitmix64 | needs the RNG decision |

**The `ppo_dout` kernel is the only genuinely irreducible one for an MLP MVP:** one thread per row,
no cross-thread reduction — softmax/LSE over the action logits, logp, `ratio = exp(logp − oldLogp)`,
the PPO clip branch, entropy gradient, value-clip gradient. It mirrors the per-row scalar block
already in `ffi/pufferblas.c` (`lean_ffi_mlp_ppo_grad_batch_blas`); structural reference is
PufferLib's `ppo_loss_compute`.

---

## Build integration — prototyped, works

Adding an nvcc-compiled object to the Lake build is **~6 lines and was proven end-to-end during
scoping** (a kernel compiled, linked through Lean's `lld`, and ran on the GPU):

```lean
-- reuse Lake's buildO with the compiler overridden to nvcc; NO Makefile, NO device linking
target puffercudaObj pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "puffercuda.o"
  let src ← inputTextFile <| pkg.dir / "ffi" / "puffercuda.cu"
  buildO oFile src #["-I", (← getLeanIncludeDir).toString, "-I", "/opt/cuda/include"]
    #["-arch=sm_120", "-allow-unsupported-compiler", "-std=c++17",
      "-Xcompiler=-fPIC", "-O2", "--threads", "0"] (compiler := "/opt/cuda/bin/nvcc")
-- then add puffercudaObj to the exe's moreLinkObjs (cudart is already linked by absolute path)
```

- **One `.cu` translation unit**, separate compilation only — no `-dc`/`-dlink`/`cudadevrt`. The
  nvcc `.o` self-registers its fatbin via a global ctor; `leanc` already pulls `-lc++/-lc++abi` and
  `cudart` resolves launch/registration.
- **Biggest build risk (found the hard way):** CUDA 13's `host_config.h` hard-`#error`s on gcc > 15,
  and this box has **gcc 16.1.1**, so bare nvcc *fails*. Works today with `-allow-unsupported-compiler`;
  the durable fix is pinning `-ccbin gcc-15`.
- **Second risk:** `leanc` links `-Wl,--gc-sections`, so every kernel's `extern "C"` launcher must
  stay reachable from an `@[extern]` or its fatbin registration is stripped. Keep launchers
  `extern "C"` (unmangled, matches `@[extern]`).

---

## Milestones

| # | Deliverable | Effort | Verification | Dep |
|---|---|---|---|---|
| **M0** | nvcc Lake target + one trivial kernel callable via `@[extern]` | **S** | `lake build puffer` green; kernel returns expected value end-to-end (proven) | — |
| **M1** | **V-Trace** `puff_advantage` (per-row scan, vec delta `ρ·(r+γV′·nnt−V)`) | **S** | `verify-vtrace-gpu` vs `computePuffAdvantageV`; reuse `vtrace_ref.py` / `vtrace_binary_parity.py`; f32 bit-exact or bf16-round-then-bit-compare | M0 |
| **M2** | **Muon** (6 kernels from `muon.cu`, NS via cuBLAS f32, resident momentum) | **M** | `verify-muon-gpu` vs `applyMuon` ~1e-4 **and** the proven NS op-norm bound; `coeffListOk` gates `ns_coeffs` | M0 |
| **M3** | **PPO gradient** (`ppo_dout` + relu-mask + bias-sum; 5 backward GEMMs on cuBLAS) | **M** | `verify-ppo-grad-gpu` 3-way (Lean `mlpGradPPO` / CPU-BLAS / CUDA) + eps=1e-6 finite-diff | M0 |
| **M4** | **Advantage normalize** (`ppo_var_mean` `/n` `+1e-8`, elementwise) | **S** | `verify-advnorm-gpu` vs `normalizeAdv` ~1e-5 | M3 |
| **M5** | **Resident MLP step + `--shadow`** (chain M1–M4, weights/grads/adv resident) | **M** | shadow mode diffs every GPU tensor against the parallel CPU Lean-oracle path per step; multi-seed learning-curve envelope | M1–M4 |
| **M6** | **GPU sampling** (device splitmix64 + inverse-CDF) *(optional)* | **L** | `verify-sample-gpu` fixed-`u` bit-exact vs `sampleCat` + χ² GOF | RNG decision |
| **M7** | **Prioritized replay + buffer residency refactor** *(deferred)* | **L** | `build_cdf`/multinomial bit-exact given fixed-`u`; weights at tolerance | M6 + refactor |

**M0–M5 (the verifiable, fully-GPU MLP training step): ~3–5 focused weeks.** M6+M7: +2–3 weeks.

> **Status — M0–M5 done: a full GPU training step, verified.** `verify-train-step-gpu` runs one
> PPO+Muon update entirely on the GPU (M4 normalize → M3 gradient → M2 Muon: matrices via Newton–Schulz,
> biases via stepVec), intermediates resident (one upload, one download, no host round-trip between
> phases), and diffs newParams+newMom against the CPU/Lean **shadow** oracle (`normalizeAdv →
> mlpPPOGradBatchBlas → applyMuon`): **f32 max|Δ| = 0 (bit-exact whole step)**, bf16 = 1e-4/7e-4.
> Notably bf16 is *far* tighter end-to-end than the raw gradient (0.08) because Muon's NS normalizes the
> update direction, suppressing the bf16 magnitude error — bf16 is fine through the optimizer. M4
> (`verify-advnorm-gpu`) is bit-exact.
>
> **Wired into a converging trainer (`train-ppo-gpu <env>`).** The step's device buffers now persist
> across updates (a 33-slot `ts_buf` cache grown on demand, freed at process exit — the ~33 mallocs are
> paid once for a whole run, not per step; `verify-train-step-gpu` stays f32-bit-exact through the cache).
> `trainPPOGpu` (rollout on CPU → `cudaTrainStepFFI` per minibatch, threading flat params + Muon momentum,
> cosine-annealed lr) converges to the optimum on the toy envs at the **bf16 default** (PufferLib
> precision): `squared` → greedy 200/200, len 2 (optimal); `chain_mdp` → greedy 12.0, 200/200.
>
> **Mean vs summed gradient (the one subtlety that mattered).** The gradient kernels/oracle *sum* over the
> N-sample minibatch; PPO/PufferLib *average* the loss (`.mean()`). Muon matrices are scale-invariant
> (frobNorm cancels the 1/N, so the orthogonalized direction — hence the weight update — is identical
> either way), which is why the single-step verify with zero momentum never caught it. But the **biases**
> take a raw (un-orthogonalized) Nesterov update, so a summed gradient is N× too large and, amplified ~20×
> by μ=0.95 momentum, blows the biases up → huge logits → `exp(ratio)` overflow → divergence within ~20
> updates (params fine, but greedy argmax collapses to NOOP; sampled return looked deceptively perfect
> because `softmax(NaN)`→`sampleCat`→last action happened to be optimal on `squared`). Fix: divide the
> widened gradient by N (`k_scale_const`) before Muon — matches the mean-loss convention; the oracle
> divides too, so `verify-train-step-gpu` stays bit-exact.
>
> **Earlier — M0–M3.** M3 (PPO gradient) is live: `verify-ppo-grad-gpu` → f32-tight `max|Δ|≈0`
> (GEMM layouts + the `ppo_dout` kernel are correct) and bf16-default `max|Δ|=0.0065` (PufferLib
> precision, round-then-compare tier), vs the CPU-BLAS f64 oracle. The 5 GEMMs now run on **cuBLAS
> tensor cores** (`cublasGemmEx`, `COMPUTE_32F_FAST_16BF` for the bf16 default, `COMPUTE_32F` for the
> f32-tight tier) — the naive-matmul first cut was swapped out. The layouts were derived as the
> row-major→col-major mappings of the CPU-BLAS oracle's `cblas_dgemm`s and confirmed by the f32-tight
> `max|Δ|≈0`. Honest speed note: `bench-ppo-grad-gpu` shows tensor cores engaged (bf16 < f32) but only
> ~1.06× end-to-end — the standalone gradient is malloc/transfer-bound (19 `cudaMalloc`s + the f64→f32
> obs upload per call swamp the GEMM), exactly like the forward before residency; the real speedup
> arrives with **M5** (resident buffers). `ppo_dout`'s per-row softmax buffer is capped at A≤64. Next:
> M4 (advantage normalize) → M5 (resident step + `--shadow`).
>
> **Earlier — M0 + M1 + M2 beat the plan.** The nvcc target is live
> (`ffi/puffercuda.cu`, `Puffer/Float/CUDA.lean`, `verify-cuda`/`verify-vtrace-gpu`/`verify-muon-gpu`).
> M1 **V-Trace** and M2 **Muon** both landed at **bit-exact** (`max|Δ| = 0`), the *strongest* tier —
> not the tolerance the scope anticipated — because the kernels run in f64 in Lean's exact op order
> with `--fmad=false` (no mul-add fusion), and the NS matmuls are naive f64 kernels that sum in Lean's
> order (sidestepping the cuBLAS row/col-major hazard). Two calibration notes for later kernels: (1)
> `--fmad=false` is what buys bit-exactness for any f64 sequential kernel; (2) reductions must match
> Lean's *fold structure* exactly — `frobNorm` is a two-level fold (per-row sum, then sum-of-row-sums),
> and a flat sequential sum was ~1e-7 off until matched. Build gotcha confirmed: `-ccbin gcc-15`
> (CUDA 13 hard-errors on the box's gcc 16.1.1).

---

## Verification contract (the actual cost driver)

Extend the existing two-tier discipline — **one `verify-*-gpu` mode per kernel, each diffed against
an existing pure-Lean oracle** (no new oracle needed). The crisp rule:

> **bit-exact ⇔ (sequential-fold OR elementwise OR integer) AND f32/f64 AND no fast-math intrinsic.**
> Everything else (tree/warp reductions, bf16 tensor-core GEMMs, `__expf`/`__logf`) is **tolerance-only**,
> exactly like the current `verify-blas` tier (f64 cuBLAS is bit-exact; bf16 ≈ 1e-2).

- **Bit-exact survives** for: the V-Trace scan, `build_cdf` prefix sum, multinomial binary search,
  `muon_nesterov`, `muon_weight_update`, and fixed-`u` inverse-CDF sampling.
- **Traps to state up front:** `__expf`/`__logf` make even "sequential" heads non-bit-exact (a draw
  can flip near a CDF boundary) → f32 tolerance, or an IEEE-math verify build. And **`ppo_var_mean`
  must use `/n` (not `/(n−1)`) + `+1e-8`.**
- **Buffer-precision policy (decide before M3):** keep the verify *reference* path f32/f64 and use
  bf16-round-then-bit-compare — never an absolute tolerance on bf16 (or eps=1e-6 finite-diff becomes
  meaningless). This one call sets bit-exact-vs-1e-2 for every downstream kernel.
- **RNG decision — recommend porting `splitmix64` (`rngNext`/`uniform01`) into the kernels** (trivial
  add/xor/shr/mul on `uint64`, counter-based seed `= f(seed, t, env)`). Gives bit-exact fixed-`u`
  sampling verification *and* stronger full-run reproducibility than PufferLib's curand-Philox.
  Fallback: keep sampling on the CPU (logits already download; it's O(numEnvs·numActions)).

New verify code ≈ **450–550 LOC** across ~6 `verify-*-gpu` modes — this, not the kernels, is where
the weeks go.

---

## Precision policy — the call (unblocks M3)

**Decision: bf16 is the default, matching PufferLib.** The GPU training step stores the FLOP-heavy
tensors (activations, gradients, GEMM operands) in **bf16** and accumulates in **f32** — precision_t =
bf16 with `CUBLAS_COMPUTE_32F`, exactly PufferLib's own recipe — with the Muon momentum kept in **f32**
(also as PufferLib does). The north star is parity with PufferLib's CUDA, and PufferLib is bf16; bf16 is
where the tensor-core throughput is.

Verification adapts (it is *not* dropped):
- **Tight cross-check at f32.** Every training-step kernel is ALSO runnable in an f32 build and checked
  against the f64 Lean oracle at ~1e-5 (bit-exact where no fast-math) — this is what proves the *logic*
  (GEMM layouts, `ppo_dout` branches) is correct, independent of bf16 rounding.
- **bf16 default verified by round-then-bit-compare** against the f64 oracle (the technique that pinned
  the V-Trace vec delta in `tools/vtrace_binary_parity.py`): bf16-round the oracle's inputs/outputs and
  compare — rigorous *in bf16*, not a loose tolerance. Plus the end-to-end multi-seed convergence backstop.
- **f32-tight is the finite-diff home:** the eps=1e-6 cross-check runs on the f32 build, where it stays
  meaningful; on the bf16 default it would be ~1e-2 and is skipped.

So each kernel ships with two verify modes — `verify-*-gpu` (bf16 default, round-then-compare ~1e-2 vs
f64) and an f32 tight variant (~1e-5) — and the same code path toggles the operand precision. This keeps
the project's *verified-parity* thesis (the f32 tight check + bf16 round-then-compare are both rigorous)
while defaulting to PufferLib's bf16 for speed. (M1/M2 remain f64/bit-exact as the reference oracles they
already are; the *production* training step is bf16.)

## Top risks

1. **Fast-math breaks bit-exactness** (`__expf`/`__logf`/`__powf`) — a draw flips near a boundary.
   → f32 tolerance ~1e-5, or IEEE-math verify build.
2. **bf16 buffer precision** — resident bf16 grad/adv buffers push tolerance to ~1e-2 and kill the
   finite-diff cross-check. → keep the reference f32/f64; bf16-round-then-compare.
3. **Prio-replay residency refactor** — `select_copy`/`index_copy`/per-mb advantage recompute assume
   GPU-resident rollout+advantage structs; ours are CPU. **The largest hidden cost, disproportionate
   to kernel LOC.** → defer to M7.
4. **Muon has no executable check today** (pure Lean + proved bounds); GPU Muon is the first. → dual
   contract: point-diff vs `applyMuon` **and** interval vs the proven op-norm bound.
5. **Silent semantic mismatches** if ported verbatim: `var /(n−1)` vs `/n`, reward-clamp location
   (Lean folds it into the advantage; PufferLib clamps at rollout-write), Muon grad **sign**
   (ascent vs descent), eps add-vs-max, and adv-normalize/entropy/prio terms the current BLAS path
   omits. → reconcile per kernel, unit-test heads in isolation.
6. **Row-major→col-major transpose slips** in the 3 backward + 15 NS GEMMs are silent. → per-stage diffs.

---

## Alternatives

- **Reuse PufferLib's compiled `_C` as a *reference harness*** (not a runtime link — it's a pybind11
  `.so` needing nccl/curand/cudnn). `tools/vtrace_binary_parity.py` already does exactly this: it
  bf16-rounds our recursion and bit-matches the real `_C.puff_advantage` (that's how the vec delta was
  pinned, 128/128 exact). Extend it per ported kernel for a stronger-than-tolerance check.
- **cuBLAS-max** — cuBLASLt DRELU/BIAS epilogues absorb relu-mask + bias-add (already used in
  `lt_layer`); a ones-vector GEMM absorbs bias column-sums → the custom surface shrinks toward *just*
  `ppo_dout` + the Muon/advantage kernels.
- **Keep sampling + prio-replay on the CPU (pragmatic v1)** — logits already download every timestep;
  sampling is trivially cheap and not the bottleneck. Sacrifices "GPU-everything" purity but
  eliminates the RNG-determinism and residency-refactor risks entirely.

**Where bit-exactness is honestly lost:** every tree/warp reduction, every bf16 tensor-core GEMM,
every fast-math intrinsic — tolerance-only by construction, consistent with the existing `verify-blas`
tier.

---

## Recommendation

**Staged GO. First slice = M0 (build MVP) + M1 (V-Trace) + M2 (Muon)** — highest reuse, lowest risk,
immediately verifiable against oracles that already exist, and neither touches RNG or residency. Then
M3–M5 for the actual training-step payoff (PPO grad + normalize + resident step with `--shadow`).
**Decide the buffer-precision policy (bf16 vs f32) before M3.** **Defer** M6 (sampling/RNG) and M7
(prio-replay/residency) — they carry the disproportionate structural cost and the weakest
verifiability. **Do not attempt a big-bang GPU-everything loop:** the LOC is small, but the
verification contract and the residency refactor are where the real weeks are.

*Scoped via a 6-agent research workflow grounded in both codebases; the build MVP was prototyped
end-to-end (nvcc kernel compiled, linked through Lean's lld, ran on the GPU).*
