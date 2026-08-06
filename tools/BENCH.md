# Benchmark: `train-ppo-gpu` vs the CPU trainers

How the GPU PPO+Muon trainer (`train-ppo-gpu`, whole step on the device via `cudaTrainStepFFI`)
compares against the CPU. Two views: an **isolated step** micro-benchmark (the thing that actually
differs — rollout is identical CPU code in both), and a **matched end-to-end** trainer run.

**Hardware:** RTX 5090 (GB202, Blackwell sm_120, 32 GB) + Ryzen 9 9950X (16C/32T), OpenBLAS multi-threaded.

Reproduce:
```
lake build puffer
./.lake/build/bin/puffer bench-train-step          # isolated step sweep (below)
./.lake/build/bin/puffer train-ppo-gpu squared     # GPU trainer, self-reports SPS decomposition
./.lake/build/bin/puffer train-ppo-cpu squared     # CPU trainer, IDENTICAL config/algorithm
./.lake/build/bin/puffer verify-muon-cpu           # native-C Muon == Lean applyMuon (bit-exact)
```

## 1. Isolated step — `bench-train-step`

One PPO+Muon update on identical synthetic minibatches, warmup + 20 reps. GPU = `cudaTrainStepFFI`
(resident whole step: normalize + gradient + Muon, device buffers cached across calls). CPU =
`mlpPPOGradBatchBlasFFI` (multi-threaded OpenBLAS gradient). The GPU number does **more** than the
CPU number (it also normalizes and runs Muon), so the CPU/GPU ratio is a **conservative lower bound**
on the GPU step's advantage.

| N | D | H | A | CPU-grad (OpenBLAS) | GPU-step bf16 | GPU-step f32 | CPU-grad / GPU-bf16 |
|------|-----|-----|----|--------------------:|--------------:|-------------:|:--------------------|
| 256  | 25  | 32  | 5  | 0.03 ms | 0.30 ms | 0.31 ms | **0.11× — CPU wins 10×** |
| 1024 | 128 | 128 | 6  | 0.7–1.8 ms | 1.59 ms | 1.61 ms | ~1× (crossover) |
| 2048 | 128 | 256 | 17 | 6.3–9.7 ms | 3.12 ms | 3.14 ms | **2–3× — GPU wins** |
| 4096 | 256 | 256 | 17 | 23–30 ms | 5.95 ms | 6.1 ms | **4–5×** |
| 8192 | 256 | 512 | 17 | 56–86 ms | 12.4 ms | 12.7 ms | **4–7×** |

- **Crossover at N·H ≈ 500K.** Below it (toy RL: `squared` runs N=256, H=32) the GPU step is
  **launch/transfer-bound** — ~0.3 ms of fixed per-call overhead swamps a trivial GEMM, so the CPU
  is ~10× faster. Above it the GPU tensor cores win, scaling to ~5–7×.
- **bf16 ≈ f32 on the whole step** (0.30 vs 0.31; 5.95 vs 6.1). The resident step is dominated by the
  f64 Muon Newton–Schulz + host↔device transfers, *not* the bf16 gradient GEMM — the tensor cores only
  accelerate one sub-component. (The GEMM itself is huge: device-resident bf16 hits 14–153 TFLOPS in
  `bench-blas`, but transfers throttle small problems to ~0.1–1 TFLOPS.)
- CPU-grad times are noisy (OpenBLAS thread contention); GPU times are deterministic. The crossover
  ordering is stable across runs.

## 2. Matched end-to-end — `train-ppo-gpu` vs `train-ppo-cpu`

Same trainer, same config, same env, **only the step backend differs**: GPU `cudaTrainStepFFI` vs
CPU (OpenBLAS gradient ÷N + native-C Muon `muonStepMlpBlasFFI`, per-minibatch normalize, no clip —
matching the GPU step). Each trainer self-reports a `rollout / step / glue` decomposition (SPS excludes
the greedy-eval diagnostics). Both reach the identical optimum.

| env (hidden 32) | GPU SPS | CPU SPS | winner | GPU step % | CPU step % |
|-----------------|--------:|--------:|:------:|:----------:|:----------:|
| `squared`  (409,600 env-steps) | 152.1k | **176.8k** | CPU 1.2× | 62% | 57% |
| `chain_mdp`(230,400 env-steps) | 169.7k | **303k** | CPU 1.8× | 72% | 48% |

(Rollout: OpenBLAS forward + native-C batched sampler; trajectory SoA + native-C gather; params+mom one
buffer; GPU step is the whole update resident on-device — see below.)

### The CPU Muon was the bottleneck — now fixed

Originally the CPU step ran the **pure-Lean** Muon (`applyMuon`, boxed `Array (Array Float)` Newton–
Schulz): ~4.4 ms/step at H=32, **91–95% of the CPU step**, giving only 13.9k / 37.6k SPS. Replacing it
with a **native-C whole-MLP Muon** (`lean_ffi_muon_step_mlp`, naive matmuls in the Lean op order,
`-ffp-contract=off` ⇒ **bit-exact** with the oracle — `verify-muon-cpu` max|Δ|=0) cut the step ~18× and
lifted the CPU trainer **10.4× / 6.9×** (to 143.9k / 260.3k SPS). The CPU step is now gradient-bound
(45–54%), not NS-bound.

## The crossover, resolved

With the CPU Muon fixed, the end-to-end result (§2) now **agrees with the isolated step (§1)**:

- **Toy RL (small batch × hidden):** the GPU step is launch/transfer-bound (~0.3 ms fixed overhead per
  call × thousands of minibatches), so the **CPU wins** — 1.2× on `squared`, 1.9× on `chain_mdp`. The
  GPU's step is 61–72% of its time doing almost nothing but paying overhead.
- **PufferLib-scale (N·H ≳ 500K):** the GPU step is 3–7× the CPU **gradient** and far more vs a full CPU
  step (§1) — this is where `train-ppo-gpu` pays off; toy RL envs never reach it, large batch × hidden
  training does.

Earlier this doc reported the GPU winning end-to-end on toy envs — that was entirely the unoptimized
Lean Muon, now removed. The honest picture: **use the CPU trainer for toy/small models, the GPU trainer
for large batch × hidden.**

## 3. Rollout forward — OpenBLAS vs scalar Ref

`vecRolloutBatched` does one batched policy forward per timestep (batch = numEnvs). Both trainers now
run it through OpenBLAS (`mlpForwardBatchBlasFFI`) instead of the scalar Ref. Measured on `train-ppo-cpu
squared` (mean of 3, hidden 32 / 32 envs / D 25):

| rollout forward | rollout+GAE time | trainer SPS |
|-----------------|-----------------:|------------:|
| scalar Ref      | ~1013 ms | ~141k |
| **OpenBLAS**    | **~920 ms** | **~149k** |

~4–9% off the rollout, ~3% overall. **This is smaller than `bench-rollout` suggests** — that bench (256
envs, breakout obs 118) shows BLAS *losing* 0.56× at hidden 128 and only winning (1.4–3.8×) at hidden
512–2048. The difference is OpenBLAS's own threading heuristic: at the toy trainer's tiny 32×32 GEMM it
stays **single-threaded** (no thread-spawn cost) and its tuned kernel beats scalar; the `bench-rollout`
"loss" is the *medium* zone where OpenBLAS multithreads a not-yet-amortized GEMM. So: BLAS wins at tiny
(toy trainers) and large (hidden ≥512), loses only in the medium multithreaded-unamortized zone (which
the toy trainers stay below).

### Native-C batched action sampler

The rest of the rollout was **Lean glue** — the per-env `softmax → sampleCat → log(prob) → value`, which
allocated 4 small `Array`s *per env per timestep* (N × horizon × updates times). Moved into one C call
per timestep (`sampleActionsBatchFFI`, `ffi/pufferffi.c`), returning `[actions; logps; values]` for the
whole batch. It is **bit-exact** with the Lean path (`verify-sample-batch`: 0 action mismatches, logp/
value Δ = 0): env `n` draws the splitmix64 word `hash(rng + (n+1)·G)` — exactly the per-env `rngNext`
stream — so the caller advances `rng` by `N·G` in O(1). (Behavior-preserving for deterministic-reset
envs, which the toy benchmark envs are; both trainers still converge 200/200.)

Measured on `train-ppo-cpu squared` (mean of 3): rollout+GAE ~920 ms → **~800 ms** (~15% off the
rollout), trainer ~149k → **~158k SPS** (~6% overall). More on `squared` (A=5, more softmax work) than
`chain_mdp` (A=2).

### SoA trajectory buffer

The trajectory was AoS — `Array (Array Transition)`, a heap `Transition` record per env per timestep,
concatenated by `buildBatch`. Replaced with `TrajCols`: the scalar fields (`actions / values /
oldLogps / rewards / terms`) are flat `FloatArray` columns and GAE (`buildBatchSoA`) runs on them
directly (no `++`). `obs` stays a per-transition **reference** array — copying it into a flat column
just double-copies what the minibatch packer already gathers, so it's kept by reference (an earlier
flat-obs-column version *regressed* `squared` for exactly this reason). Env-major layout (`row = e·T+s`)
keeps the shuffle/minibatches bit-identical, so training is unchanged: `verify-rollout-soa` diffs the SoA
rollout+GAE against the AoS path field-by-field — **all Δ = 0** (obs/action/value/oldLogp/reward/
terminal/adv/return), and both trainers still converge 200/200.

Effect: **CPU-glue −~30%** (the minibatch packer reads flat columns instead of dereferencing
`Transition` records, and the AoS concatenation is gone), rollout −4–6%, and overall SPS `squared` 158k
→ **162k**, `chain_mdp` 264k → **290k** (~3–10%).

### Native-C minibatch gather

The packer still copied the minibatch out of the columns in Lean — and `obs` was a boxed `Array Float`,
so `for x in tc.obs[t]` unboxed D floats per index. Made `obs` a flat **unboxed** column filled per
timestep by a C scatter (`scatterObsFFI`, reusing the forward's `xb`), then replaced the per-index Lean
push loops with one C pass (`gatherMinibatchFFI`) that row-copies the shuffled minibatch out of the SoA
columns into the contiguous `mb*` buffers the step kernels take. (An earlier flat-obs attempt with a Lean
`set!` fill *regressed* — the fill has to be C, and the flat obs must not add a second copy.)
`verify-rollout-soa` now also diffs the C gather against the AoS buffer — **Δ = 0**.

Effect: **CPU-glue −~24%** (`squared` 233 → 177 ms; the boxed obs walk is gone), rollout +~4% (the scatter
cost), net overall SPS `squared` 162k → **170k**, `chain_mdp` 290k → **304k** (~3–5%).

### Combined params+mom buffer

The step kernels (`cudaTrainStepFFI`, `muonStepMlpBlasFFI`) returned `[newParams; newMom]` (2P), and the
trainer split it back into two `FloatArray`s with `mk ((range P).map …)` × 2 — ~26M boxed ops per update,
and (because the split sat inside the timed `s0…s1` window) it was charged to the *step*, not the glue.
Changed both kernels to take the **combined** `[params; mom]` buffer as input too (`mom` is just the second
half), so the trainer threads one `pm` and the returned value *is* the next `pm` — no split, no recombine.
`unflat pm` already reads only `pm[0:P]`, so the rollout/eval are unchanged, and `mlpPPOGradBatchBlasFFI`
reads `pm[0:P]` for its forward. Both verifies stay bit-exact (`verify-train-step-gpu` f32 Δ=0,
`verify-muon-cpu` Δ=0).

Effect (shows up in the *step* time, where the split was charged): CPU-step `squared` 1427 → 1313 ms,
GPU-step 2095 → 2017 ms; net SPS `squared` 170k → **177k** GPU 132k → **137k**, `chain_mdp` GPU 142k →
**147k** (~2–4%).

### Whole-update resident GPU step (`cudaTrainUpdateFFI`)

The GPU trainer called `cudaTrainStepFFI` **once per minibatch** — `epochs·numMB` = 16 FFI calls/update,
each re-uploading obs+params, re-widening params to f32 on the host, `malloc`/`free`-ing staging, and
doing its own `cudaDeviceSynchronize`. `bench-train-step` had already shown the GPU step is
launch/transfer-bound below N·H ≈ 500K — this per-call overhead *is* that bound. Replaced with a
**whole-update resident** FFI: one call uploads the SoA columns + the `epochs` shuffles + `[params;mom]`
**once**, loops all `epochs × numMB` minibatches on-device (gather → normalize → gradient → Muon, params
resident in f64, updated in place), and downloads `[params;mom]` **once**. `verify-train-update-resident`
diffs it against looping `cudaTrainStepFFI` over the same perms — **f32 Δ = 0**; both trainers converge
200/200 (`train-ppo-gpu <env> [noresident]` toggles it).

Effect — **GPU-step `squared` 2017 → 1683 ms, `chain_mdp` ~1150 → 974 ms; SPS `squared` 137k → 152k,
`chain_mdp` 147k → 170k (~11–15%)** — from collapsing 16 overhead-bound calls into one. `bench-train-update`
sweeps the isolated step: neutral at large *compute* (overhead already amortized — 8192×256 is ~1.0×) but
**1.25× at large *obs*** (16384×128, D=2048: the per-minibatch obs re-upload is what residency removes).

**The large-batch ceiling is now the rollout, not the step.** `squared_big` (hidden 256, 512 envs × 32 →
N=4096×H=256 minibatches) runs the GPU step in ~1.4 s but the **CPU rollout is ~80% of the update** — at
512 parallel envs the Lean `env.observe`/`env.step` dominates, so the resident-step win is a small slice
end-to-end. Matching PufferLib in this regime needs the *rollout* on the GPU too (resident policy + device
envs), which is the next structural frontier — a much larger undertaking than the training step.

## Takeaways

- **Toy RL envs:** CPU wins — the GPU step is launch/transfer-bound and the CPU Muon is now native-C
  (bit-exact, ~100× the Lean path).
- **PufferLib-scale (N·H ≳ 500K):** the GPU step is 3–7× the CPU gradient — the regime `train-ppo-gpu`
  is for.
- **Precision:** bf16 (the PufferLib default) barely beats f32 on the *whole* GPU step because the f64
  Muon + transfers dominate; tensor cores matter for the gradient GEMM, a minority of the step.
- **Where it stands.** Muon, the rollout matmul, the per-env softmax/sample glue, the AoS→SoA trajectory,
  the C minibatch gather, the combined params+mom buffer, and the whole-update resident GPU step are all
  done. The CPU trainer is gradient-bound (~57% step); the GPU trainer's step is now one resident call/
  update. **Both trainers are step/rollout-bound with the glue squeezed out.**
- **The remaining frontier is the rollout on the GPU.** Both trainers rollout on the CPU (Lean
  `env.observe` / `env.step` + OpenBLAS forward). At large batch that's ~80% of the update (`squared_big`),
  so it caps the large-batch GPU regime the resident step unlocks. A GPU-resident rollout (device policy
  forward + device envs, PufferLib-style) is the next structural step — and the biggest remaining one.
  Below that, returns are sub-single-digit.
