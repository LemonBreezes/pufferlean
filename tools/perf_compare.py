#!/usr/bin/env python3
"""perf_compare.py — time PufferLib's real kernels/ops so pufferlean's device kernels can be
compared head-to-head at matched shapes. Runs INSIDE PufferLib's venv (needs torch + pufferlib + _C).

    cd ~/src/PufferLib && unset LD_LIBRARY_PATH && .venv/bin/python <lean>/tools/perf_compare.py

Emits one JSON line: {"step":[...], "mingru":[...], "vtrace":{...}, "muon":{...}, "misc":{...}}.
Each timing is the median of `reps` after warmup, in milliseconds. The Lean side times the twin
device kernels (bench-train-step, bench-blas, verify-*-gpu reps) and compare.py tabulates.
"""
import json, sys, statistics, time
import torch

DEV = torch.device("cuda")

def timed(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

# The exact shapes `puffer bench-train-step` uses, so the full-step rows line up 1:1.
STEP_SHAPES = [
    dict(N=1024, D=128, H=128, A=6),
    dict(N=2048, D=128, H=256, A=17),
    dict(N=4096, D=256, H=256, A=17),
    dict(N=8192, D=256, H=512, A=17),
]

def bench_mlp_step():
    """A full PPO+Muon MLP training step in torch (bf16 autocast, PufferLib's precision): forward
    (Linear→ReLU→Linear heads) → PPO clipped surrogate + value loss → backward → Muon-style update.
    The twin of our fused resident `cudaTrainStepFFI`."""
    from pufferlib.muon import Muon
    out = []
    for s in STEP_SHAPES:
        N, D, H, A = s["N"], s["D"], s["H"], s["A"]
        O = A + 1
        obs = torch.randn(N, D, device=DEV)
        acts = torch.randint(0, A, (N,), device=DEV)
        adv = torch.randn(N, device=DEV)
        ret = torch.randn(N, device=DEV)
        oldlp = torch.randn(N, device=DEV)
        W1 = torch.nn.Linear(D, H, device=DEV); W2 = torch.nn.Linear(H, O, device=DEV)
        opt = Muon(list(W1.parameters()) + list(W2.parameters()), lr=0.01, momentum=0.95, weight_decay=0.0)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = torch.relu(W1(obs)); y = W2(h)
                logits, val = y[:, :A], y[:, A]
                logp = torch.log_softmax(logits.float(), -1).gather(1, acts[:, None]).squeeze(1)
                ratio = torch.exp(logp - oldlp)
                a = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a).mean()
                vl = 0.5 * (val.float() - ret).pow(2).mean()
                loss = pg + 0.5 * vl
            loss.backward()
            opt.step()
        out.append(dict(**s, ms=timed(step)))
    return out

def bench_mingru():
    """MinGRU (PufferLib's default net, hidden=128, num_layers=4) forward_train (Heinsen parallel
    scan) + backward, and per-step forward_eval — over B sequences of length T. The twin of our
    cudaMinGRUStepFFI (per-step) and cudaMinGRUPpoGradFFI (BPTT)."""
    from pufferlib.models import MinGRU
    H, L = 128, 4
    out = []
    for B, T in [(256, 16), (512, 32), (1024, 32)]:
        gru = MinGRU(H, L).to(DEV)
        x = torch.randn(B, T, H, device=DEV, requires_grad=True)      # [B, T, H] batch-first

        def fwd_train():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gru.forward_train(x)

        def fwd_bwd_train():
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = gru.forward_train(x)
            y.float().sum().backward()

        def step_eval():                                  # per-step rollout recurrence (f32; lerp needs f32)
            with torch.no_grad():
                state = (gru.initial_state(B, 'cuda')[0],)
                hh = torch.randn(B, H, device=DEV)
                for t in range(T):
                    hh, state = gru.forward_eval(hh, state)

        row = dict(H=H, L=L, B=B, T=T)
        try: row["fwd_train_ms"] = timed(fwd_train, reps=30)
        except Exception as e: row["fwd_train_err"] = f"{type(e).__name__}: {e}"
        try: row["fwd_bwd_train_ms"] = timed(fwd_bwd_train, reps=30)
        except Exception as e: row["fwd_bwd_train_err"] = f"{type(e).__name__}: {e}"
        try: row["step_eval_ms"] = timed(step_eval, reps=20)
        except Exception as e: row["step_eval_err"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return out

def bench_vtrace():
    """PufferLib's compiled _C.puff_advantage (bf16 CUDA) over B×T segments."""
    from pufferlib import _C
    rows = []
    for B, T in [(1024, 128), (4096, 128), (8192, 128)]:
        v = torch.randn(B, T, dtype=torch.bfloat16, device=DEV)
        r = torch.randn(B, T, dtype=torch.bfloat16, device=DEV)
        d = (torch.rand(B, T, device=DEV) < 0.1).to(torch.bfloat16)
        i = torch.rand(B, T, dtype=torch.bfloat16, device=DEV)
        adv = torch.zeros(B, T, dtype=torch.bfloat16, device=DEV)
        def vt(): _C.puff_advantage(v.data_ptr(), r.data_ptr(), d.data_ptr(), i.data_ptr(),
                                    adv.data_ptr(), B, T, 0.99, 0.95, 1.0, 1.0)
        rows.append(dict(B=B, T=T, ms=timed(vt, reps=100)))
    return rows

def bench_muon():
    from pufferlib.muon import Muon
    rows = []
    for r, c in [(128, 128), (256, 256), (512, 512), (768, 128)]:
        p = torch.nn.Parameter(torch.randn(r, c, device=DEV))
        opt = Muon([p], lr=0.01, momentum=0.95, weight_decay=0.0)
        p.grad = torch.randn(r, c, device=DEV)
        rows.append(dict(rows=r, cols=c, ms=timed(opt.step, reps=100)))
    return rows

def bench_misc():
    rows = {}
    N, O = 8192, 18
    logits = torch.randn(N, O, device=DEV)
    def samp():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.multinomial(torch.softmax(logits, -1), 1)
    rows["categorical_sample_ms"] = timed(samp, reps=100)
    adv = torch.randn(N, device=DEV)
    def norm(): (adv - adv.mean()) / (adv.std() + 1e-8)
    rows["adv_normalize_ms"] = timed(norm, reps=100)
    return rows

def main():
    out = {}
    for name, fn in [("step", bench_mlp_step), ("mingru", bench_mingru), ("vtrace", bench_vtrace),
                     ("muon", bench_muon), ("misc", bench_misc)]:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
