#!/usr/bin/env python3
"""Direct parity of our V-Trace algorithm against the ACTUAL compiled PufferLib kernel
`pufferlib._C.puff_advantage` (native bf16 CUDA), not just a port of the source.

Feeds random (values, rewards in [-1,1], dones, importance) as bf16 CUDA buffers, calls the
real kernel, and reproduces it with the same recursion our Lean/numpy V-Trace uses (float32
carry, bf16-rounded output — exactly what puff_advantage_row_vec does). Reports bit match.

This is what pinned the delta form: for horizon % 8 == 0 (every real GPU `puffer train`) the
launcher dispatches `puff_advantage_row_vec`, whose delta is  rho*(r + gamma*v' - v)  (rho on the
WHOLE TD error), NOT the CPU/scalar  rho*r + gamma*v' - v. Our Lean trainers match the vec form
bit-for-bit; see the "vec-form 128/128, scalar-form 19/128" split this script prints.

Run from the PufferLib checkout with its venv + a bf16 `_C` build (breakout is fine):
    cd ~/src/PufferLib && unset LD_LIBRARY_PATH && .venv/bin/python <this>
"""
import numpy as np
import torch
from pufferlib import _C

def run(B, T, gamma, lam, rho_clip, c_clip, seed):
    g = torch.Generator().manual_seed(seed)
    vals  = (torch.randn(B, T, generator=g) * 2.0)
    rews  = (torch.rand(B, T, generator=g) * 2.0 - 1.0)          # already clamped range
    dones = (torch.rand(B, T, generator=g) < 0.12).float()
    imp   = (torch.rand(B, T, generator=g) * 1.8 + 0.1)          # importance ratios

    # store as bf16 (PufferLib's rollout-buffer precision), move to CUDA
    vb, rb = vals.to(torch.bfloat16).cuda(), rews.to(torch.bfloat16).cuda()
    db, ib = dones.to(torch.bfloat16).cuda(), imp.to(torch.bfloat16).cuda()
    adv = torch.zeros(B, T, dtype=torch.bfloat16, device='cuda')

    _C.puff_advantage(vb.data_ptr(), rb.data_ptr(), db.data_ptr(), ib.data_ptr(),
                      adv.data_ptr(), B, T, gamma, lam, rho_clip, c_clip)
    torch.cuda.synchronize()
    kernel = adv.float().cpu().numpy()

    # reproduce with OUR recursion on the same bf16-rounded inputs
    V = vb.float().cpu().numpy().astype(np.float32); R = rb.float().cpu().numpy().astype(np.float32)
    D = db.float().cpu().numpy().astype(np.float32); I = ib.float().cpu().numpy().astype(np.float32)
    gm, lm = np.float32(gamma), np.float32(lam)
    def recur(vec_form):
        ref = np.zeros((B, T), np.float32)
        for b in range(B):
            last = np.float32(0.0)
            for t in range(T - 2, -1, -1):
                nnt = np.float32(1.0) - D[b, t + 1]
                rho = np.float32(min(I[b, t], rho_clip)); c = np.float32(min(I[b, t], c_clip))
                if vec_form:                          # rho scales whole TD error (GPU vec kernel)
                    delta = rho * (R[b, t + 1] + gm * V[b, t + 1] * nnt - V[b, t])
                else:                                  # rho scales only reward (CPU/scalar kernel)
                    delta = rho * R[b, t + 1] + gm * V[b, t + 1] * nnt - V[b, t]
                last = np.float32(delta + gm * lm * c * last * nnt)
                ref[b, t] = last
        return torch.tensor(ref).to(torch.bfloat16).float().numpy()   # match bf16 storage
    res = {}
    for name, vf in (("vec", True), ("scalar", False)):
        rb16 = recur(vf)
        res[name] = (int((kernel == rb16).sum()), B * T, float(np.abs(kernel - rb16).max()))
    return res

if __name__ == "__main__":
    print("Direct parity: OUR V-Trace vs actual compiled pufferlib._C.puff_advantage (bf16 CUDA)\n")
    cfgs = [
        dict(B=8,  T=16, gamma=0.99, lam=0.95, rho_clip=1.0, c_clip=1.0, seed=0),
        dict(B=16, T=32, gamma=0.972,lam=0.949,rho_clip=2.10,c_clip=1.08,seed=1),  # breakout.ini hypers
        dict(B=32, T=64, gamma=0.90, lam=0.80, rho_clip=1.5, c_clip=1.0, seed=7),
    ]
    allok = True
    for c in cfgs:
        res = run(**c)
        (ve, vt, vm), (se, st, sm) = res["vec"], res["scalar"]
        allok &= (ve == vt)
        print(f"  B={c['B']:2d} T={c['T']:2d} rho={c['rho_clip']} c={c['c_clip']}:  "
              f"vec-form {ve:4d}/{vt} (max|Δ|={vm:.2e})   "
              f"scalar-form {se:4d}/{st} (max|Δ|={sm:.2e})")
    print("\n" + ("=> actual GPU kernel uses the VEC delta:  delta = rho*(r + gamma*v' - v)"
                  if allok else "=> neither form matches; investigate further."))
