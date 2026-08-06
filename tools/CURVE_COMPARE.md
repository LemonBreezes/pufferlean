# Empirical learning-curve comparison: Lean trainer vs a PPO reference

The project's philosophy is that **per-step numerics are rigorously proven**, but **a full
RL run is chaotic**, so full-run agreement is validated **empirically** — the learning curves
should track. This directory holds that comparison for the Lean `puffer train-ppo` trainer.

## What was run

Two PPO trainers on the SAME env + seed + hyperparameters, curves overlaid:

* **Lean side** — the actual compiled binary:
  ```
  .lake/build/bin/puffer train-ppo-curve chain_mdp 20
  .lake/build/bin/puffer train-ppo-curve squared   40
  ```
  `train-ppo-curve <env> <win>` is numerically identical to `train-ppo` (same
  `trainPPO` loop, same `updatePPO`); it only reports the moving-average return every
  `win` episodes instead of every 200, so the early *rising* part of the curve is
  resolved. (The stock `puffer train-ppo <env>` output — reporting every 200 episodes —
  is also captured verbatim in `curve_data/lean_chain_mdp.txt` and
  `curve_data/lean_squared.txt`.)

* **Python reference** — `tools/ppo_ref.py`, a small **faithful NumPy port** of the Lean
  trainer. It is NOT PufferLib's C/vectorized PPO (that trainer differs in weight init,
  optimizer bookkeeping, vectorization and reward logging, so it is not an
  episode-for-episode comparable baseline for this toy trainer). Instead it is a
  line-by-line reimplementation of the exact Lean code paths, so the comparison is
  *fair by construction*:

  | component        | ported from |
  |------------------|-------------|
  | splitmix64 PRNG, `uniform01`, `softmax`, `sampleCat` | `Puffer/RL/Train.lean` |
  | `initMLP`/`randMat`, `forwardAll`, `rolloutEnv`, `computeGAE`, `normalizeAdv`, `mlpGradPPO`, `updatePPO`, `trainPPO` | `Puffer/RL/NNTrain.lean` |
  | `chainMDPEnv`, `squaredEnv` | `Puffer/RL/Envs.lean` |
  | env dynamics | `Puffer/Env/ChainMDP/Model.lean`, `Puffer/Env/Squared/Model.lean` |
  | hyperparameters (lr 0.03, γ 0.99, λ 0.95, vf 0.5, ent 0.01, clip 0.2, 4 epochs, seed 0x1234) | `Exe/Puffer.lean` `runPPO` |

  The PPO gradient in `ppo_ref.py` is the **analytical** gradient of the SAME objective
  the Lean autodiff builds in `mlpGradPPO`
  (`min(ρ·A, clip(ρ)·A) − vf·½(V−R)² + ent·H`), with the same subgradient conventions
  (clip-grad 0 outside `[1−ε,1+ε]`; `min` picks the smaller branch; `relu' = 1[z>0]`).

Driver: `tools/curve_compare.py` runs both, writes per-env curve JSON + a `summary.json`
and the overlay PNG under `tools/curve_data/`.

```
tools/curve_compare.py                 # both envs, plot + quantitative summary
tools/curve_compare.py chain_mdp 20    # one env at a chosen window
```

## Result

Overlay plot: **`tools/curve_data/curve_compare.png`** (Lean solid blue, reference dashed red).

| env       | windows | Lean final | ref final | Lean AUC | ref AUC | max \|Δ\| | corr |
|-----------|---------|-----------|-----------|----------|---------|-----------|------|
| chain_mdp | 150 (win 20) | 11.950 | 11.950 | 11.9373 | 11.9373 | **0.0** | 1.0 |
| squared   | 100 (win 40) | 1.000  | 1.000  | 0.9825   | 0.9825  | **0.0** | 1.0 |

Both curves show the expected learning: `chain_mdp` climbs 7.75 → ~11.99 (the optimum:
walk to the terminal-reward cell and stay, over the 14-step episode); `squared` climbs
−0.45 → 1.0 (reach the fixed target). The Lean and reference curves are **bit-for-bit
identical** at every reported window.

## Honest caveats

* **Determinism.** Both trainers are fully deterministic (fixed seed `0x1234`,
  splitmix64 PRNG). Rerunning the Lean binary gives an identical byte stream (verified:
  same `md5sum` across runs), and the Python port reproduces it exactly. There is no
  run-to-run variance to average over.

* **Why the agreement is *exact*, not just "tracking".** These envs are deterministic and
  the policy converges within ~100–200 episodes. The only expected source of divergence
  between Lean's proven scalar `dotF` and NumPy's vectorized dot is a few low-order bits
  in the logits — enough to flip a `sampleCat` action only when two action probabilities
  are extremely close. On chain_mdp / squared that essentially never happens along these
  trajectories, so the two independent implementations trace the identical run. This is a
  *stronger* outcome than the philosophy requires (which only asks that trends/asymptotes
  track), and it is honest: it reflects the low arithmetic sensitivity of these particular
  toy envs, not a rigged comparison.

* **The comparison is not trivially tautological.** As a sensitivity check, running the
  reference with *different* PRNG seeds (same algorithm/env/hyperparameters, a different
  stochastic realization) gives curves that *differ* early but *track* to the same
  asymptote — e.g. first-200-episode average 11.30 (seed 0x1234) vs 9.50 (0xABCDEF) vs
  9.59 (0x999), all converging to ~11.9–12.0. So the metric does distinguish runs; the
  seed-matched faithful port matching bit-for-bit is a real result, not an artifact of an
  insensitive metric.

* **Scope.** This validates the *runnable* Lean PPO trainer (chain_mdp, squared) against a
  matched reference. It does not compare against PufferLib's own C PPO trainer, and does
  not cover the Muon-optimizer path (`train-muon`) — the same driver would extend to it.

## Files

* `tools/ppo_ref.py`            — faithful NumPy PPO reference (the baseline).
* `tools/curve_compare.py`      — runs both sides, emits data + summary + PNG.
* `tools/curve_data/curve_compare.png`   — overlay plot.
* `tools/curve_data/summary.json`        — quantitative agreement table.
* `tools/curve_data/{lean,ref}_<env>_win<N>.json` — the raw curves.
* `tools/curve_data/lean_{chain_mdp,squared}.txt` — stock `puffer train-ppo` output (every 200 eps).

---

## Full PufferLib-algorithm comparison (all trainers)

Once the trainers reached full parity with **current** PufferLib (Muon + V-Trace + prioritized
replay + iterated value/ratio + value/reward clipping + cosine LR + the MinGRU network), the
comparison was redone against the *current* algorithm, not the old SGD path.

**`tools/puffer_ref.py`** — a from-scratch NumPy implementation of `torch_pufferl.py`'s MLP vec
loop (Newton–Schulz Muon, `compute_puff_advantage`, prioritized-replay minibatching with iterated
value/ratio, clipped value loss, cosine LR), run on the shared toy envs at the SAME scale as
`puffer train-ffi` (16 envs × 32 steps, hidden 16):

```
python3 tools/puffer_ref.py chain_mdp --updates 200
python3 tools/puffer_ref.py squared   --updates 200
```

Both the Lean trainer and this independent reference **converge to the same optimum** (chain_mdp
≈ 9–12, squared → 1.0). Whole-run curves are NOT identical and cannot be: the Newton–Schulz
orthogonalization uses a different matmul summation order in NumPy, and prioritized replay samples
with a different RNG (`torch.multinomial` vs Lean's `rngNext`), so trajectories diverge chaotically
after a few steps — exactly the "chaotic full run" this project validates empirically.

**Why not run PufferLib itself?** The installed `_C` is a bf16 build (`precision_bytes = 2`) and
`torch_pufferl.py` requires float32; and PufferLib trains its own Ocean C envs with a 128×4 MinGRU
over 4096 agents — a different task/scale from these toy runs. So the live curve check is against
the NumPy reference, and the *rigorous* parity is the per-component bit-exact evidence:

| Component | vs | Δ |
| --- | --- | --- |
| Muon optimizer | `muon.py` (60-digit decimal) | ≈ 3e-16 |
| V-Trace advantage | `bindings_cpu.cpp` (`vtrace_ref.py`) | **0** |
| MinGRU forward_eval | `models.py` (`mingru_ref.py`) | **0** |
| MinGRU forward_train (Heinsen scan) | numpy Heinsen; ≡ forward_eval | **0** / 5.5e-17 |
| MinGRU BPTT gradient (native C) | Lean AD oracle (`verify-mingru-kernel`) | **0** |
| MLP/CNN/gauss/LSTM grad kernels | Lean AD oracle | ≤ 1e-15 |

All six trainer families (MLP, continuous, CNN, LSTM, multi-agent, MinGRU) were run and converge;
the consolidated curves + this table are rendered in the published comparison artifact.
