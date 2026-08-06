#!/usr/bin/env python3
"""NumPy reference of the CURRENT PufferLib training algorithm (torch_pufferl.py) for the MLP
vec trainer — Muon + V-Trace + prioritized replay + iterated value/ratio + value/reward clipping
+ cosine LR — run on the shared toy envs at the SAME scale as the Lean `train-ffi` trainer.

This is the END-TO-END cross-check that sits on top of the per-component BIT-EXACT parity checks
(V-Trace vs the C kernel via vtrace_ref.py, MinGRU vs models.py
via mingru_ref.py, every native gradient kernel vs a Lean AD oracle). Bit-exact WHOLE-RUN
agreement is impossible here — the Newton-Schulz matmul float order and the prioritized-sampling
RNG (torch.multinomial / Lean rngNext / NumPy) all differ, so the runs diverge chaotically after a
few steps (exactly the "chaotic full run" the project validates EMPIRICALLY). The check is that an
INDEPENDENT implementation of the same algorithm converges to the same optimum with a similar
learning curve.

    tools/puffer_ref.py chain_mdp --updates 200 > ref_chain.csv    # (update, mean_ep_return)
"""
import argparse
import numpy as np

import ppo_ref as R   # forward_all / policy_and_value / envs / rng_next / sample_cat / init_mlp / CONFIGS

# ---- Muon (pufferlib/muon.py) -----------------------------------------------------
NS_COEFS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]


def zeropower(G, eps=1e-7):
    X = G.astype(np.float64).copy()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / (np.linalg.norm(X) + eps)
    for a, b, c in NS_COEFS:
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X.T if tall else X


def muon_step(W, g, mom, lr, mu=0.95):
    """Ascent form (matches Lean applyMuon): Nesterov momentum → orthogonalize → W += lr·scale·o."""
    mom = mu * mom + g
    upd = g + mu * mom
    if upd.ndim >= 2:
        o = zeropower(upd) * max(1.0, upd.shape[0] / upd.shape[1]) ** 0.5
        return W + lr * o, mom
    return W + lr * upd, mom


def cosine_lr(lr, u, updates, min_ratio=0.0):
    if u == 0:
        return lr
    lo = lr * min_ratio
    return lo + 0.5 * (lr - lo) * (1.0 + np.cos(np.pi * u / max(updates, 1)))


# ---- V-Trace advantage (compute_puff_advantage), iterated value + ratio -----------
def puff_advantage(seg, values, importance, gamma, lam, rho_clip=1.0, c_clip=1.0):
    n = len(seg)
    adv = np.zeros(n)
    lastA = 0.0
    for t in range(n - 2, -1, -1):
        nnt = 0.0 if seg[t]["terminal"] else 1.0
        imp = importance[t]
        rho = min(imp, rho_clip)
        c = min(imp, c_clip)
        r = max(-1.0, min(1.0, seg[t]["reward"]))          # reward clip [-1,1] (rollout-write clamp)
        delta = rho * (r + gamma * values[t + 1] * nnt - values[t])   # GPU vec kernel: rho scales whole TD error
        lastA = delta + gamma * lam * c * lastA * nnt
        adv[t] = lastA
    return adv


# ---- value-clipped PPO transition gradient (ppo_ref's + PufferLib value clipping) --
def grad_transition(p, x, a, advN, ret, oldval, oldLogp, vfCoef, entCoef, clipEps, vfClip):
    z1, h, out = R.forward_all(p, x)
    A = len(p.b2) - 1
    logits = out[:A]
    m = np.max(logits); ex = np.exp(logits - m); Z = np.sum(ex)
    pi = ex / Z; lse = m + np.log(Z); logp = logits - lse
    rho = np.exp(logp[a] - oldLogp)
    dlogp_a = -pi.copy(); dlogp_a[a] += 1.0
    surr1 = rho * advN
    ratioC = min(max(rho, 1.0 - clipEps), 1.0 + clipEps)
    surr2 = ratioC * advN
    if surr1 <= surr2:
        dpol = advN * rho * dlogp_a
    else:
        dpol = (advN * rho * dlogp_a) if (1.0 - clipEps) < rho < (1.0 + clipEps) else np.zeros(A)
    sum_pl = np.sum(pi * logp)
    dent = entCoef * (-(pi * logp - pi * sum_pl))
    dout = np.zeros(A + 1)
    dout[:A] = dpol + dent
    # PufferLib clipped value loss: 0.5*max((V-R)^2, (Vclip-R)^2)
    V = out[A]
    if vfClip > 0.0:
        d = V - oldval
        vclip = oldval + max(-vfClip, min(vfClip, d))
        du = (V - ret) ** 2; dc = (vclip - ret) ** 2
        dvloss = (V - ret) if (du >= dc or (-vfClip < d < vfClip)) else 0.0
    else:
        dvloss = V - ret
    dout[A] = -vfCoef * dvloss
    gb2 = dout; gW2 = np.outer(dout, h)
    dz1 = (p.W2.T @ dout) * (z1 > 0.0)
    return np.outer(dz1, x), dz1, gW2, gb2


# ---- vec rollout (16 envs × horizon), auto-reset on terminal ----------------------
def vec_rollout(env, p, states, horizon, rng):
    segs, new_states, ep_returns = [], [], []
    for st in states:
        s, traj, ep = st, [], 0.0
        for _ in range(horizon):
            obs = env.observe(s)
            probs, v = R.policy_and_value(p, obs)
            word, rng = R.rng_next(rng)
            a = R.sample_cat(probs, R.uniform01(word))
            s2, r, term = env.step(s, a)
            traj.append({"obs": obs, "action": a, "reward": r, "value": v,
                         "oldLogp": float(np.log(probs[a])), "terminal": term})
            ep += r
            if term:
                ep_returns.append(ep); ep = 0.0; s = env.reset()
            else:
                s = s2
        segs.append(traj); new_states.append(s)
    return segs, new_states, rng, ep_returns


def train_puffer(name, updates, num_envs=16, horizon=32, epochs=4, num_mb=4, max_grad_norm=0.5):
    cfg = R.CONFIGS[name]
    env = cfg["env"]()
    p, rng = R.init_mlp(env.obsDim, cfg["hidden"], env.numActions + 1, cfg["seed"])
    mW1, mb1, mW2, mb2 = (np.zeros_like(p.W1), np.zeros_like(p.b1),
                          np.zeros_like(p.W2), np.zeros_like(p.b2))
    states = [env.reset() for _ in range(num_envs)]
    g, gam, lam = cfg, cfg["gamma"], cfg["lam"]
    vf, ent, clip = cfg["vfCoef"], cfg["entCoef"], cfg["clipEps"]
    prio_a, prio_b0, vfClip = 0.8, 0.2, 0.2
    curve = []
    for u in range(updates):
        segs, states, rng, ep_returns = vec_rollout(env, p, states, horizon, rng)
        lr = cosine_lr(cfg["lr"], u, updates)
        nseg = len(segs)
        valueBuf = [[tr["value"] for tr in seg] for seg in segs]
        ratioBuf = [[1.0] * len(seg) for seg in segs]
        mb_segs = max(1, nseg // num_mb)
        anneal_b = prio_b0 + (1 - prio_b0) * prio_a * u / max(updates, 1)
        for _mb in range(max(1, epochs * num_mb)):
            advPerSeg = [puff_advantage(segs[e], valueBuf[e], ratioBuf[e], gam, lam) for e in range(nseg)]
            prioW = np.array([np.sum(np.abs(a)) ** prio_a for a in advPerSeg])
            probs = (prioW + 1e-6) / (prioW.sum() + 1e-6)
            cum = np.cumsum(probs); total = cum[-1]
            sampled = []
            for _ in range(mb_segs):
                word, rng = R.rng_next(rng)
                idx = int(np.searchsorted(cum, R.uniform01(word) * total))
                sampled.append(min(idx, nseg - 1))
            mbPrio = [(nseg * probs[i]) ** (-anneal_b) for i in sampled]
            advVals = np.array([x for i in sampled for x in advPerSeg[i]])
            am, astd = advVals.mean(), advVals.std()
            gW1 = np.zeros_like(p.W1); gb1 = np.zeros_like(p.b1)
            gW2 = np.zeros_like(p.W2); gb2 = np.zeros_like(p.b2)
            nTr = 0
            for jj, i in enumerate(sampled):
                seg, adv, vbuf, w = segs[i], advPerSeg[i], valueBuf[i], mbPrio[jj]
                for t, tr in enumerate(seg):
                    advN = w * (adv[t] - am) / (astd + 1e-8)
                    dW1, db1, dW2, db2 = grad_transition(
                        p, tr["obs"], tr["action"], advN, adv[t] + vbuf[t], vbuf[t],
                        tr["oldLogp"], vf, ent, clip, vfClip)
                    gW1 += dW1; gb1 += db1; gW2 += dW2; gb2 += db2
                    nTr += 1
            gnorm = np.sqrt(sum(np.sum(x * x) for x in (gW1, gb1, gW2, gb2))) / max(nTr, 1)
            cc = min(1.0, max_grad_norm / gnorm) if (max_grad_norm > 0 and gnorm > max_grad_norm) else 1.0
            sc = cc / max(nTr, 1)
            p.W1, mW1 = muon_step(p.W1, sc * gW1, mW1, lr)
            p.b1, mb1 = muon_step(p.b1, sc * gb1, mb1, lr)
            p.W2, mW2 = muon_step(p.W2, sc * gW2, mW2, lr)
            p.b2, mb2 = muon_step(p.b2, sc * gb2, mb2, lr)
            for i in set(sampled):
                seg = segs[i]
                nr, nv = [], []
                for tr in seg:
                    pr, v = R.policy_and_value(p, tr["obs"])
                    nr.append(float(np.exp(np.log(pr[tr["action"]]) - tr["oldLogp"])))
                    nv.append(v)
                ratioBuf[i], valueBuf[i] = nr, nv
        curve.append((u + 1, float(np.mean(ep_returns)) if ep_returns else 0.0))
    return curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env", nargs="?", default="chain_mdp")
    ap.add_argument("--updates", type=int, default=200)
    a = ap.parse_args()
    for upd, ret in train_puffer(a.env, a.updates):
        print(f"{upd},{ret}")
