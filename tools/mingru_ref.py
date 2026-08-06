#!/usr/bin/env python3
"""Cross-check Lean's MinGRU forward (Puffer/Net/MinGRU.lean) against a faithful port of
PufferLib's default policy network `DefaultEncoder → MinGRU → DefaultDecoder`
(`~/src/PufferLib/pufferlib/models.py`).

The Lean self-test emits fixed weights + an observation sequence + its forward outputs as f64
bit patterns; this recomputes the SAME forward in Python (f64) using the SAME sequential dot
order Lean uses, and asserts the logits/values match at floating-point-roundoff scale.

`models.py` forward_eval recurrence, per layer (Linear(H,3H,bias=False) → chunk 3):
    z   = sigmoid(gate);  g = x+0.5 if x>=0 else sigmoid(x)
    out = (1-z)*state + z*g                      # torch.lerp(state, g, z)
    h   = sigmoid(proj)*out + (1-sigmoid(proj))*h  # highway
Encoder: Linear(obs,H) with bias.  Decoder: Linear(H,A) logits + Linear(H,1) value.
No LayerNorm in this path. `forward_train` is the Heinsen-scan equivalent (same math).

torch's real forward uses BLAS (a different summation order) → a documented ~1e-13 roundoff
difference from this sequential reference; the ALGORITHM is identical. We compare the sequential
reference to Lean's sequential forward, which agree to bit / roundoff scale.

    tools/mingru_ref.py                # build+run the Lean self-test, then check
    lean --run Puffer/Net/MinGRU.lean | tools/mingru_ref.py -
"""
import json
import math
import struct
import subprocess
import sys

TOL = 1e-12   # abs OR rel; observed at machine-eps scale


def f64(bp):
    return struct.unpack("<d", struct.pack("<Q", int(bp)))[0]


def vec(bits):
    return [f64(x) for x in bits]


def mat(bits):
    return [vec(row) for row in bits]


def dot(w, x):
    acc = 0.0
    for i in range(len(w)):
        acc += w[i] * x[i]     # same left-to-right order as Lean's `dot`
    return acc


def linear(W, b, x):
    return [dot(W[i], x) + b[i] for i in range(len(W))]


def matvec(W, x):
    return [dot(row, x) for row in W]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def g_act(x):
    return x + 0.5 if x >= 0.0 else sigmoid(x)


def layer_step(W, h, prev, H):
    y = matvec(W, h)          # [3H]
    out, hnew = [], []
    for j in range(H):
        hid, gate, proj = y[j], y[H + j], y[2 * H + j]
        z = sigmoid(gate)
        o = (1.0 - z) * prev[j] + z * g_act(hid)
        hg = sigmoid(proj)
        hnew.append(hg * o + (1.0 - hg) * h[j])
        out.append(o)
    return hnew, out


def step_forward(w, H, obs, state):
    e = linear(w["wEnc"], w["bEnc"], obs)
    h = e
    new_state = []
    for i in range(len(w["layers"])):
        h, out = layer_step(w["layers"][i], h, state[i], H)
        new_state.append(out)
    logits = linear(w["wDec"], w["bDec"], h)
    value = linear(w["wVal"], w["bVal"], h)[0]
    return logits, value, new_state


def seq_forward(w, H, num_layers, obs_seq):
    state = [[0.0] * H for _ in range(num_layers)]
    logits_seq, value_seq = [], []
    for obs in obs_seq:
        logits, value, state = step_forward(w, H, obs, state)
        logits_seq.append(logits)
        value_seq.append(value)
    return logits_seq, value_seq


# ---- forward_train: the Heinsen log-space parallel scan (models.py:144) -----------

def softplus(x):
    return x + math.log(1.0 + math.exp(-x)) if x > 0.0 else math.log(1.0 + math.exp(x))


def log_g(x):
    return math.log(x + 0.5) if x >= 0.0 else -softplus(-x)


def heinsen_scan(log_coeffs, log_values):
    """out_t = (1-z_t)*out_{t-1} + z_t*g_t via exp(a_star + logcumsumexp(log_values - a_star));
    same online logcumsumexp (running max m, sum s) as Lean's heinsenScan → bit-exact."""
    n = len(log_coeffs)
    out = [0.0] * n
    a_star = 0.0
    m = -1e308
    s = 0.0
    for t in range(n):
        a_star += log_coeffs[t]
        x = log_values[t] - a_star
        if x > m:
            s = s * math.exp(m - x) + 1.0
            m = x
        else:
            s = s + math.exp(x - m)
        out[t] = math.exp(a_star + m + math.log(s))
    return out


def seq_forward_train(w, H, num_layers, obs_seq):
    T = len(obs_seq)
    h = [linear(w["wEnc"], w["bEnc"], obs) for obs in obs_seq]   # [T][H]
    for l in range(num_layers):
        W = w["layers"][l]
        ys = [matvec(W, ht) for ht in h]                        # [T][3H]
        out_cols = []
        for jj in range(H):
            log_coeffs = [-softplus(ys[t][H + jj]) for t in range(T)]
            log_values = [-softplus(-ys[t][H + jj]) + log_g(ys[t][jj]) for t in range(T)]
            out_cols.append(heinsen_scan(log_coeffs, log_values))
        new_h = []
        for t in range(T):
            row = []
            for jj in range(H):
                hg = sigmoid(ys[t][2 * H + jj])
                row.append(hg * out_cols[jj][t] + (1.0 - hg) * h[t][jj])
            new_h.append(row)
        h = new_h
    logits_seq = [linear(w["wDec"], w["bDec"], h[t]) for t in range(T)]
    value_seq = [linear(w["wVal"], w["bVal"], h[t])[0] for t in range(T)]
    return logits_seq, value_seq


def check(name, got, ref):
    max_abs = max_rel = 0.0
    n = 0
    for gr, rr in zip(got, ref):
        gr = gr if isinstance(gr, list) else [gr]
        rr = rr if isinstance(rr, list) else [rr]
        for a, bb in zip(gr, rr):
            n += 1
            d = abs(a - bb)
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, d / (abs(bb) + 1e-300))
    ok = max_abs <= TOL or max_rel <= TOL
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:8s} n={n:3d}  max|Δ|={max_abs:.3e}  max rel={max_rel:.3e}")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        text = sys.stdin.read()
    else:
        out = subprocess.run(["lake", "env", "lean", "--run", "Puffer/Net/MinGRUSelfTest.lean"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr)
            return 1
        text = out.stdout
    line = next((l for l in text.splitlines() if l.strip().startswith("{")), None)
    if not line:
        print("no MinGRU self-test row emitted", file=sys.stderr)
        return 1
    j = json.loads(line)
    H, num_layers = j["H"], j["numLayers"]
    w = {
        "wEnc": mat(j["wEnc"]), "bEnc": vec(j["bEnc"]),
        "layers": [mat(l) for l in j["layers"]],
        "wDec": mat(j["wDec"]), "bDec": vec(j["bDec"]),
        "wVal": mat(j["wVal"]), "bVal": vec(j["bVal"]),
    }
    obs_seq = mat(j["obsSeq"])
    logits_L, values_L = mat(j["logits"]), vec(j["values"])

    logits_ref, values_ref = seq_forward(w, H, num_layers, obs_seq)

    print(f"MinGRU forward cross-check  (H={H}, layers={num_layers}, T={j['T']}, A={j['A']})\n")
    ok = True
    print("  forward_eval (sequential recurrence):")
    ok &= check("logits", logits_L, logits_ref)
    ok &= check("values", values_L, values_ref)

    if "logitsTrain" in j:
        logitsT_L, valuesT_L = mat(j["logitsTrain"]), vec(j["valuesTrain"])
        logitsT_ref, valuesT_ref = seq_forward_train(w, H, num_layers, obs_seq)
        print("  forward_train (Heinsen parallel scan) vs numpy Heinsen:")
        ok &= check("logitsT", logitsT_L, logitsT_ref)
        ok &= check("valuesT", valuesT_L, valuesT_ref)
        print("  forward_train == forward_eval (same recurrence, up to float order):")
        ok &= check("train=eval", logitsT_L, logits_L)
        ok &= check("valT=valE", valuesT_L, values_L)

    print()
    if ok:
        print("Lean MinGRU forward_eval + Heinsen forward_train match the models.py recurrence at roundoff scale")
        return 0
    print("MISMATCH — Lean MinGRU diverges from the reference")
    return 1


if __name__ == "__main__":
    sys.exit(main())
