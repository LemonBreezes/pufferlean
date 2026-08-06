#!/usr/bin/env python3
"""Direct output-parity checks of pufferlib-lean's MATH against the ACTUAL PufferLib library.

Runs INSIDE the PufferLib venv (needs torch + pufferlib + a compiled `_C`). Each check drives the
REAL PufferLib code — the compiled CUDA V-Trace kernel, the real `pufferlib.muon.Muon` optimizer,
and the real `pufferlib.models` MinGRU network — and compares it to the same reference algorithm
pufferlib-lean's FFI kernels are verified against (`tools/vtrace_ref.py`, `tools/mingru_ref.py`,
`tools/puffer_ref.py`). Emits one JSON line per check.

    cd ~/src/PufferLib && unset LD_LIBRARY_PATH && .venv/bin/python <this> --tools <lean>/tools

The lean side is the verified spec: Lean FFI kernel == Lean AD/numpy ref (the `puffer verify-*`
modes), and THIS script closes the loop numpy-ref == actual PufferLib. Transitively: our FFI == PufferLib.
"""
import argparse, json, math, os, sys

import numpy as np

NS_COEFS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]


def _summ(name, got, ref, note=""):
    g = np.asarray(got, float).ravel(); r = np.asarray(ref, float).ravel()
    d = np.abs(g - r)
    max_abs = float(d.max()) if d.size else 0.0
    max_rel = float((d / (np.abs(r) + 1e-300)).max()) if d.size else 0.0
    return {"name": name, "n": int(g.size), "max_abs": max_abs, "max_rel": max_rel, "note": note}


# ---- 1. V-Trace: actual compiled _C.puff_advantage (bf16 CUDA) vs the vec-kernel recursion -------
def check_vtrace():
    import torch
    from pufferlib import _C
    B, T = 24, 32
    gamma, lam, rho_clip, c_clip = 0.972, 0.949, 2.10, 1.08
    gg = torch.Generator().manual_seed(3)
    vals = (torch.randn(B, T, generator=gg) * 2.0)
    rews = (torch.rand(B, T, generator=gg) * 2.0 - 1.0)
    dones = (torch.rand(B, T, generator=gg) < 0.12).float()
    imp = (torch.rand(B, T, generator=gg) * 1.8 + 0.1)
    vb, rb = vals.to(torch.bfloat16).cuda(), rews.to(torch.bfloat16).cuda()
    db, ib = dones.to(torch.bfloat16).cuda(), imp.to(torch.bfloat16).cuda()
    adv = torch.zeros(B, T, dtype=torch.bfloat16, device='cuda')
    _C.puff_advantage(vb.data_ptr(), rb.data_ptr(), db.data_ptr(), ib.data_ptr(),
                      adv.data_ptr(), B, T, gamma, lam, rho_clip, c_clip)
    torch.cuda.synchronize()
    kernel = adv.float().cpu().numpy()
    V = vb.float().cpu().numpy(); R = rb.float().cpu().numpy()
    D = db.float().cpu().numpy(); I = ib.float().cpu().numpy()
    gm, lm = np.float32(gamma), np.float32(lam)
    ref = np.zeros((B, T), np.float32)
    for b in range(B):
        last = np.float32(0.0)
        for t in range(T - 2, -1, -1):
            nnt = np.float32(1.0) - D[b, t + 1]
            rho = np.float32(min(I[b, t], rho_clip)); c = np.float32(min(I[b, t], c_clip))
            r = np.float32(max(-1.0, min(1.0, R[b, t + 1])))
            delta = rho * (r + gm * V[b, t + 1] * nnt - V[b, t])
            last = np.float32(delta + gm * lm * c * last * nnt)
            ref[b, t] = last
    import torch as _t
    ref_bf16 = _t.tensor(ref).to(_t.bfloat16).float().numpy()
    exact = int((kernel == ref_bf16).sum())
    s = _summ("V-Trace advantage", kernel, ref_bf16,
              f"compiled _C.puff_advantage (bf16 CUDA), bit-exact {exact}/{B*T}")
    s["bit_exact"] = (exact == B * T)
    return s


# ---- 2. Muon: actual pufferlib.muon.Muon.step() vs a faithful numpy reproduction ----------------
def _zeropower_np(G, eps=1e-7):
    X = G.astype(np.float64).copy()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / max(np.linalg.norm(G), eps)      # muon.py: clamp(G.norm, min=eps) == max(norm, eps)
    n = X.shape[0]
    for a, b, c in NS_COEFS:
        s = X @ X.T
        y = c * s + b * np.eye(n)
        y = y @ s + a * np.eye(n)
        X = y @ X
    return X.T if tall else X


def _muon_np(W, g, buf, lr, mu, wd):
    buf = mu * buf + g
    upd = g + mu * buf
    if upd.ndim >= 2:
        upd = _zeropower_np(upd) * max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5
    return W * (1.0 - lr * wd) - lr * upd, buf


def check_muon():
    import torch
    from pufferlib.muon import Muon
    torch.set_default_dtype(torch.float64)
    lr, mu, wd = 0.1, 0.95, 0.0
    gg = torch.Generator().manual_seed(11)
    worst = None
    for shape in [(8, 5), (16, 16), (5, 12)]:          # tall, square, wide matrices
        W = torch.randn(*shape, generator=gg, dtype=torch.float64)
        g = torch.randn(*shape, generator=gg, dtype=torch.float64)
        p = torch.nn.Parameter(W.clone())
        p.grad = g.clone()
        opt = Muon([p], lr=lr, momentum=mu, weight_decay=wd)
        opt.step()
        ref, _ = _muon_np(W.numpy(), g.numpy(), np.zeros(shape), lr, mu, wd)
        s = _summ("Muon step", p.detach().numpy(), ref)
        if worst is None or s["max_abs"] > worst["max_abs"]:
            worst = s
    torch.set_default_dtype(torch.float32)
    worst["name"] = "Muon optimizer step"
    worst["note"] = "real pufferlib.muon.Muon.step() vs numpy NS+nesterov (worst of tall/square/wide)"
    return worst


# ---- 3/4. MinGRU: actual pufferlib.models network vs the sequential + Heinsen references --------
def _build_mingru(H, num_layers, A, obs_size, seed):
    import torch
    from pufferlib.models import MinGRU, DefaultEncoder, DefaultDecoder
    torch.manual_seed(seed)
    enc = DefaultEncoder(obs_size, H).double()
    gru = MinGRU(H, num_layers).double()
    dec = DefaultDecoder([A], H).double()          # discrete head, nvec=[A]
    w = {
        "wEnc": enc.encoder.weight.detach().tolist(), "bEnc": enc.encoder.bias.detach().tolist(),
        "layers": [l.weight.detach().tolist() for l in gru.layers],
        "wDec": dec.decoder.weight.detach().tolist(), "bDec": dec.decoder.bias.detach().tolist(),
        "wVal": dec.value_function.weight.detach().tolist(), "bVal": dec.value_function.bias.detach().tolist(),
    }
    return enc, gru, dec, w


def check_mingru_eval(mingru_ref):
    import torch
    H, L, A, obs_size, T = 8, 4, 3, 6, 10
    enc, gru, dec, w = _build_mingru(H, L, A, obs_size, seed=7)
    torch.manual_seed(21)
    obs = torch.randn(T, obs_size, dtype=torch.float64)
    state = (gru.initial_state(1, 'cpu')[0].double(),)   # initial_state builds f32 zeros; lerp needs f64
    logits_t, values_t = [], []
    with torch.no_grad():
        for t in range(T):
            # DefaultEncoder.forward hardcodes .float(); apply the same Linear in f64 to keep the
            # MinGRU recurrence at full precision (the encoder is a plain Linear, checked elsewhere).
            h = torch.nn.functional.linear(obs[t].view(1, -1), enc.encoder.weight, enc.encoder.bias)
            h, state = gru.forward_eval(h, state)
            lg, vv = dec(h)
            logits_t.append(lg[0].tolist()); values_t.append(float(vv[0, 0]))
    logits_ref, values_ref = mingru_ref.seq_forward(w, H, L, obs.tolist())
    a = _summ("MinGRU forward_eval", [x for r in logits_t for x in r],
              [x for r in logits_ref for x in r], "real pufferlib.models MinGRU.forward_eval (f64) vs sequential ref")
    b = _summ("MinGRU value_eval", values_t, values_ref, "")
    a["max_abs"] = max(a["max_abs"], b["max_abs"]); a["max_rel"] = max(a["max_rel"], b["max_rel"])
    return a


def check_mingru_train(mingru_ref):
    import torch
    H, L, A, obs_size, T = 8, 4, 3, 6, 10
    enc, gru, dec, w = _build_mingru(H, L, A, obs_size, seed=7)
    torch.manual_seed(21)
    obs = torch.randn(T, obs_size, dtype=torch.float64)
    with torch.no_grad():
        h = torch.nn.functional.linear(obs, enc.encoder.weight, enc.encoder.bias).unsqueeze(0)  # [1,T,H] f64
        h = gru.forward_train(h)                   # Heinsen parallel scan
        lg, vv = dec(h)
    logits_t = lg[0].tolist(); values_t = vv[0, :, 0].tolist()
    logits_ref, values_ref = mingru_ref.seq_forward_train(w, H, L, obs.tolist())
    a = _summ("MinGRU forward_train (Heinsen)", [x for r in logits_t for x in r],
              [x for r in logits_ref for x in r], "real pufferlib.models MinGRU.forward_train (Heinsen scan, f64) vs ref")
    b = _summ("MinGRU value_train", values_t, values_ref, "")
    a["max_abs"] = max(a["max_abs"], b["max_abs"]); a["max_rel"] = max(a["max_rel"], b["max_rel"])
    return a


# ---- performance: dense-forward GEMM on the SAME GPU at PufferLib's bf16 vs our f64 -------------
def bench_gemm(sizes=(256, 512, 1024, 2048)):
    """relu(X·Wᵀ+b) throughput (GF/s) via torch on CUDA, at f64 (our cuBLAS precision) and bf16
    (PufferLib's tensor-core precision). Square N=D=H, FLOPs=2·N³. Pairs with `puffer bench-blas`."""
    import time
    import torch
    if not torch.cuda.is_available():
        return {"skip": "no CUDA for torch"}
    out = []
    for dt, name in [(torch.float64, "f64"), (torch.bfloat16, "bf16")]:
        row = {}
        for N in sizes:
            X = torch.randn(N, N, device="cuda", dtype=dt)
            W = torch.randn(N, N, device="cuda", dtype=dt)
            b = torch.randn(N, device="cuda", dtype=dt)
            for _ in range(3):
                (torch.relu(X @ W.t() + b))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = 20
            for _ in range(reps):
                Y = torch.relu(X @ W.t() + b)
            torch.cuda.synchronize()
            dt_s = (time.perf_counter() - t0) / reps
            row[N] = 2.0 * N ** 3 / (dt_s * 1e9)       # GF/s
        out.append({"prec": name, "gflops": row})
    return {"torch_gemm": out}


# ---- performance: the actual native bf16 PufferLib trainer throughput (SPS) on breakout ---------
def perf_breakout(steps):
    import pufferlib.pufferl as P
    from pufferlib import _C
    compiled = getattr(_C, "env_name", None)
    if compiled not in (None, "breakout"):
        return {"name": "PufferLib native trainer", "skip": f"_C compiled for '{compiled}', not breakout"}
    saved = sys.argv
    sys.argv = [saved[0], "--train.total-timesteps", str(int(steps))]
    try:
        args = P.load_config("breakout")
        pufferl = _C.create_pufferl(args)
        last = {}
        while pufferl.global_step < args["train"]["total_timesteps"]:
            _C.rollouts(pufferl); _C.train(pufferl)
            last = dict(P.unroll_nested_dict(_C.log(pufferl)))
        _C.close(pufferl)
    finally:
        sys.argv = saved
    def g(*names, d=0.0):
        for n in names:
            for k in last:
                if k == n or k.endswith("/" + n):
                    return last[k]
        return d
    return {"name": "PufferLib native trainer (breakout, bf16 GPU)", "sps": float(g("SPS")),
            "steps": int(g("agent_steps")), "episode_return": float(g("episode_return", "score")),
            "note": "actual pufferlib._C native C++/CUDA trainer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="path to the pufferlib-lean tools/ dir (for mingru_ref)")
    ap.add_argument("--perf-steps", type=int, default=0,
                    help="if >0, also run the native PufferLib breakout trainer for this many steps and report SPS")
    args = ap.parse_args()
    sys.path.insert(0, args.tools)
    import mingru_ref

    results = []
    def run(fn, *a):
        try:
            results.append(fn(*a))
        except Exception as e:
            import traceback
            results.append({"name": fn.__name__, "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc().splitlines()[-3:]})
    run(check_vtrace)
    run(check_muon)
    run(check_mingru_eval, mingru_ref)
    run(check_mingru_train, mingru_ref)
    perf = []
    if args.perf_steps > 0:
        try:
            perf.append(perf_breakout(args.perf_steps))
        except Exception as e:
            perf.append({"name": "PufferLib native trainer", "error": f"{type(e).__name__}: {e}"})
    gemm = {}
    if args.perf_steps > 0:
        try:
            gemm = bench_gemm()
        except Exception as e:
            gemm = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps({"results": results, "perf": perf, "gemm": gemm}))


if __name__ == "__main__":
    main()
