# Full-run validation: the Lean trainer vs faithful references

The project's philosophy: **per-step numerics are rigorously proven** — every kernel is checked
bit-for-bit against a Lean f64 autodiff oracle and against the real PufferLib code — while **a full
RL run is chaotic**, so full-run agreement is validated **empirically**: the trainer must converge to
the same optimum as an independent reference.

Everything below is run by one command — the parity + convergence suite:

```
python3 tools/compare.py          # 22/22: per-component parity (A–C) + convergence (D) + perf (E)
```

## The rigorous part — per-component bit-exact parity

Because the full run is chaotic, the *strong* guarantee is per-component: each kernel matches its
reference to machine precision. `compare.py` (sections A–C) checks all of these every run:

| Component | vs | Δ |
| --- | --- | --- |
| Muon optimizer | real `pufferlib.muon.Muon` (`muon.py`) | ≈ 3e-16 |
| V-Trace advantage | `bindings_cpu.cpp` (`tools/vtrace_ref.py`) | **0** |
| MinGRU forward | real `pufferlib.models` (`tools/mingru_ref.py`) | **0** |
| MinGRU BPTT gradient (native C / CUDA) | Lean AD oracle | **0** |
| MLP / CNN / Gaussian / LSTM grad kernels | Lean AD oracle | ≤ 1e-15 |

The GPU tiers add their own tolerances (f64 bit-exact; f32 ~1e-6; bf16 the PufferLib default ~1e-2) —
see `compare.py`'s section C.

## The empirical part — convergence to the known optimum

Full runs are validated by outcome, not step-for-step. `compare.py` (section D) trains the **actual
Lean binary** on toy envs whose optimum is known and confirms it reaches it:

| env                  | network  | reaches                    |
|----------------------|----------|----------------------------|
| `squared`            | MinGRU   | return → 1.000  (opt ≈ 1.0) |
| `squared_continuous` | Gaussian | return → ~0.96  (opt ≈ 0.9) |

The trainer is deterministic at a fixed seed — rerunning gives an identical byte stream (same
`md5sum`), so there is no run-to-run variance to average over.

## The faithful references

`tools/puffer_ref.py` is a from-scratch **NumPy** reimplementation of `torch_pufferl.py`'s vec loop
(Newton–Schulz Muon, `compute_puff_advantage`, prioritized-replay minibatching with iterated
value/ratio, clipped value loss, cosine LR). Run standalone it converges to the same optimum on the
shared toy envs. Whole-run curves are deliberately **not** bit-identical and cannot be: Newton–Schulz
uses a different matmul summation order in NumPy and prioritized replay draws with a different RNG
(`torch.multinomial` vs Lean's `rngNext`), so trajectories diverge chaotically after a few steps —
exactly the "chaotic full run" this project validates by asymptote, not step-for-step.
`tools/ppo_ref.py` is the analogous line-by-line reference for the simpler PPO path (documenting the
exact objective `min(ρ·A, clip(ρ)·A) − vf·½(V−R)² + ent·H` the Lean autodiff builds).

## Why not diff against PufferLib's live trainer here?

PufferLib trains its own Ocean C envs with a 128×4 MinGRU over 4096 agents — a different task/scale
from these toy runs, and episode-for-episode curve overlays aren't comparable (different weight init,
optimizer bookkeeping, vectorization, RNG). So the *live* full-run check is the convergence test above
against the NumPy reference; the **rigorous** cross-library parity is the per-component bit-exact
evidence in the table (and `compare.py`'s section A drives the real compiled `_C` / torch modules
directly). Head-to-head *throughput* vs PufferLib lives separately in `docs/benchmark.html`.
