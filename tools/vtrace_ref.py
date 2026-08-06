#!/usr/bin/env python3
"""Cross-check Lean's `computePuffAdvantage` against PufferLib's advantage kernel.

`puffer verify-vtrace` emits, per fixed segment, the (rewards, values, terminals, importance)
in Lean's `Transition` convention plus the advantages/returns Lean computed. This script
reconstructs PufferLib's RAW buffers (its `rewards`/`terminals` are stored one step ahead of
Lean's — reward[j]=r(a_{j-1}), done[j]=terminal(o_j)) and runs the VERBATIM GPU kernel
`puff_advantage_row_vec` from `~/src/PufferLib/src/pufferlib.cu`, asserting Lean's output matches.

That GPU kernel — the path `puffer train` runs whenever the horizon is a multiple of 8 (i.e. all
real training) — uses  δ = ρ·(r + γ·V'·nnt − V)  (ρ scales the WHOLE TD error). PufferLib's
CPU/scalar kernels instead use  δ = ρ·r + γV'·nnt − V  (ρ on the reward only); the two agree iff
ρ = 1. We match the GPU path that actually trains, and confirmed it bit-for-bit against the ACTUAL
compiled `pufferlib._C.puff_advantage` (see tools/README / the vtrace_binary_parity check).

This validates BOTH the V-Trace/GAE recurrence AND the index mapping (Lean's `traj[t].reward` /
`traj[t+1].value` / `traj[t].terminal` ≡ PufferLib's `reward[t+1]` / `value[t+1]` / `done[t+1]`).
PufferLib clamps rewards to [-1,1] at rollout-write (`clamp_precision_kernel`), before the kernel
reads them; Lean folds that clamp into `computePuffAdvantage`, so we clamp here to match.

The real `_C` kernel runs in float32 (or bf16); we port it in Python float (f64) to match Lean's
f64 — the ALGORITHM is identical, only the working precision differs (Lean's f64 is strictly more
precise). So we compare at floating-point-roundoff scale, exactly like the other verify-* refs.

    tools/vtrace_ref.py                    # build+run `puffer verify-vtrace`, check it
    puffer verify-vtrace | tools/vtrace_ref.py -
"""
import json
import struct
import subprocess
import sys

# roundoff threshold (abs OR rel); observed ~1e-16, far below this
TOL = 1e-12


def f64(bitpattern):
    return struct.unpack("<d", struct.pack("<Q", int(bitpattern)))[0]


def arr(bits):
    return [f64(b) for b in bits]


def puff_advantage_cpu(values, rewards, dones, importance, gamma, lam, rho_clip, c_clip):
    """Verbatim port of the GPU kernel src/pufferlib.cu `puff_advantage_row_vec` (one row), in f64.

    A_{T-1} stays 0; for t = T-2 .. 0:
        rho = min(imp_t, rho_clip); c = min(imp_t, c_clip); nnt = 1 - dones[t+1]
        r   = clamp(rewards[t+1], -1, 1)                       # rollout clamps before the kernel
        delta = rho*(r + gamma*values[t+1]*nnt - values[t])    # rho scales the WHOLE TD error
        A_t   = delta + gamma*lam*c*A_{t+1}*nnt
    """
    T = len(values)
    adv = [0.0] * T
    last = 0.0
    for t in range(T - 2, -1, -1):
        t_next = t + 1
        nnt = 1.0 - dones[t_next]
        imp = importance[t]
        rho = imp if imp < rho_clip else rho_clip
        c = imp if imp < c_clip else c_clip
        r = max(-1.0, min(1.0, rewards[t_next]))
        delta = rho * (r + gamma * values[t_next] * nnt - values[t])
        last = delta + gamma * lam * c * last * nnt
        adv[t] = last
    return adv


def check(name, got, ref):
    n = len(got)
    max_abs = max_rel = 0.0
    for a, b in zip(got, ref):
        d = abs(a - b)
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, d / (abs(b) + 1e-300))
    ok = max_abs <= TOL or max_rel <= TOL
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:10s} n={n:2d}  max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        lines = sys.stdin.read().splitlines()
    else:
        out = subprocess.run(["lake", "env", ".lake/build/bin/puffer", "verify-vtrace"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr)
            return 1
        lines = out.stdout.splitlines()

    rows = [json.loads(l) for l in lines if l.strip().startswith("{")]
    if not rows:
        print("no vtrace rows emitted", file=sys.stderr)
        return 1

    print(f"compute_puff_advantage cross-check  ({len(rows)} segments)\n")
    all_ok = True
    for i, r in enumerate(rows):
        gamma, lam = f64(r["gamma"]), f64(r["lam"])
        rho_clip, c_clip = f64(r["rhoClip"]), f64(r["cClip"])
        rewards_L = arr(r["rewards"])       # Lean convention: reward[t] = r(a_t)
        values_L = arr(r["values"])         # value[t] = V(o_t)
        terminals_L = [float(x) for x in r["terminals"]]   # terminal[t] = terminal(o_{t+1})
        importance = arr(r["importance"])
        adv_L, ret_L = arr(r["adv"]), arr(r["ret"])
        T = len(values_L)

        # Reconstruct PufferLib's raw buffers (its reward/terminal are one step ahead):
        c_values = values_L
        c_rewards = [0.0] + rewards_L[:T - 1]
        c_dones = [0.0] + terminals_L[:T - 1]

        adv_ref = puff_advantage_cpu(c_values, c_rewards, c_dones, importance,
                                     gamma, lam, rho_clip, c_clip)
        ret_ref = [adv_ref[t] + values_L[t] for t in range(T)]

        print(f"segment {i}  (T={T})")
        all_ok &= check("adv", adv_L, adv_ref)
        all_ok &= check("returns", ret_L, ret_ref)

    print()
    if all_ok:
        print(f"{2*len(rows)}/{2*len(rows)} checks match the verbatim puff_advantage_cpu kernel at roundoff scale")
        return 0
    print("MISMATCH — Lean computePuffAdvantage diverges from the C kernel")
    return 1


if __name__ == "__main__":
    sys.exit(main())
