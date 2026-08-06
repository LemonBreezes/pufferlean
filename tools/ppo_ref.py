#!/usr/bin/env python3
"""A small, FAITHFUL NumPy reimplementation of the Lean `puffer train-ppo` trainer.

This is the reference the Lean RL trainer is compared against (see tools/CURVE_COMPARE.md).
It is a line-by-line port of the Lean code paths so the two curves are a FAIR comparison:
they share the same PRNG, the same weight init, the same env dynamics, the same PPO
objective, the same GAE/normalization, and the same SGD-ascent update with the same
hyperparameters. The ONLY intended differences are:

  * float arithmetic order inside dot products / softmax (NumPy vectorized vs Lean's
    proven scalar `dotF`), which perturbs the low bits and hence the sampled actions
    after a few steps -> the trajectories diverge stochastically, exactly the "chaotic
    full run" the project says to validate EMPIRICALLY, not bit-for-bit.

Ported Lean sources:
  Puffer/RL/Train.lean      : rngNext (splitmix64), uniform01, softmax, sampleCat
  Puffer/RL/NNTrain.lean    : initMLP/randMat, forwardAll, policyAndValue, rolloutEnv,
                              computeGAE, normalizeAdv, mlpGradPPO, updatePPO, trainPPO
  Puffer/RL/Envs.lean       : chainMDPEnv, squaredEnv
  Puffer/Env/ChainMDP/Model : reset/step/rewardAt/movePos
  Puffer/Env/Squared/Model  : init/move/step
  Exe/Puffer.lean runPPO    : the exact hyperparameters per env

The PPO objective gradient is the analytical gradient of the SAME objective the Lean
autodiff builds in `mlpGradPPO` (min(rho*A, clip(rho)*A) - vf*0.5*(V-R)^2 + ent*H), with
the same subgradient conventions (clip gradient 0 outside [1-eps,1+eps]; min picks the
smaller branch; relu' = 1[z>0]).
"""
import numpy as np

MASK64 = (1 << 64) - 1

# ---- splitmix64 PRNG (Puffer/RL/Train.lean rngNext) -------------------------------

def rng_next(s):
    s = (s + 0x9E3779B97F4A7C15) & MASK64
    z = s
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = (z ^ (z >> 31)) & MASK64
    return z, s

def uniform01(r):
    # Float.ofNat (r >>> 11) / 2^53
    return float(r >> 11) / 9007199254740992.0

# ---- softmax / categorical sampling (Puffer/RL/Train.lean) ------------------------

def softmax(logits):
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / np.sum(e)

def sample_cat(probs, u):
    acc = 0.0
    for i in range(len(probs)):
        acc += probs[i]
        if u < acc:
            return i
    return len(probs) - 1

# ---- MLP (Puffer/RL/NNTrain.lean) -------------------------------------------------

class MLP:
    __slots__ = ("W1", "b1", "W2", "b2")
    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

def rand_mat(rows, cols, scale, rng):
    m = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            word, rng = rng_next(rng)
            m[i, j] = uniform01(word) * (2.0 * scale) - scale
    return m, rng

def init_mlp(din, H, dout, seed):
    W1, rng = rand_mat(H, din, 0.3, seed)
    W2, rng = rand_mat(dout, H, 0.3, rng)
    return MLP(W1, np.zeros(H), W2, np.zeros(dout)), rng

def forward_all(p, x):
    z1 = p.b1 + p.W1 @ x
    h = np.maximum(z1, 0.0)
    logits = p.b2 + p.W2 @ h
    return z1, h, logits

def policy_and_value(p, obs):
    _, _, out = forward_all(p, obs)
    A = len(p.b2) - 1
    return softmax(out[:A]), out[A]

# ---- Envs (Puffer/RL/Envs.lean + Env/*/Model.lean) --------------------------------

class ChainMDP:
    numActions = 2
    def __init__(self, size):
        self.size = size
        self.obsDim = size
        self.maxSteps = size + 9
    def reset(self):
        return {"pos": 1, "tick": 0}
    def move_pos(self, pos, action):
        v = pos + (2 * action - 1)
        v = min(max(v, 0), self.size - 1)
        return v
    def reward_at(self, pos):
        if pos == 0: return 0.001
        if pos == self.size - 1: return 1.0
        return 0.0
    def step(self, s, a):
        newpos = self.move_pos(s["pos"], a)
        r = self.reward_at(newpos)
        if s["tick"] + 1 == self.size + 9:
            ns = {"pos": 1, "tick": 0}; term = True
        else:
            ns = {"pos": newpos, "tick": s["tick"] + 1}; term = False
        return ns, r, term
    def observe(self, s):
        o = np.zeros(self.size); o[s["pos"]] = 1.0; return o

class Squared:
    numActions = 5
    def __init__(self, size, targetIdx):
        self.size = size
        self.obsDim = size * size
        self.maxSteps = 3 * size + 2
        self.targetR = targetIdx // size
        self.targetC = targetIdx % size
    def init(self):
        return {"r": self.size // 2, "c": self.size // 2, "tick": 0}
    def move(self, r, c, a):
        if a == 1: return r + 1, c
        if a == 2: return r - 1, c
        if a == 3: return r, c - 1
        if a == 4: return r, c + 1
        return r, c
    def reset(self):
        return self.init()
    def step(self, s, a):
        nr, nc = self.move(s["r"], s["c"], a)
        if s["tick"] + 1 > 3 * self.size or nr < 0 or nc < 0 or nr >= self.size or nc >= self.size:
            return self.init(), -1.0, True
        if nr == self.targetR and nc == self.targetC:
            return self.init(), 1.0, True
        return {"r": nr, "c": nc, "tick": s["tick"] + 1}, 0.0, False
    def observe(self, s):
        o = np.zeros(self.size * self.size); o[s["r"] * self.size + s["c"]] = 1.0; return o

# ---- Rollout / GAE / normalize (Puffer/RL/NNTrain.lean) ---------------------------

def rollout_env(env, p, rng):
    s = env.reset()
    traj = []
    ret = 0.0
    for _ in range(env.maxSteps):
        obs = env.observe(s)
        probs, v = policy_and_value(p, obs)
        word, rng = rng_next(rng)
        a = sample_cat(probs, uniform01(word))
        s2, r, term = env.step(s, a)
        oldlogp = float(np.log(probs[a]))
        traj.append({"obs": obs, "action": a, "reward": r, "value": v,
                     "oldLogp": oldlogp, "terminal": term})
        ret += r
        s = s2
        if term:
            break
    return traj, ret, rng

def compute_gae(traj, gamma, lam):
    n = len(traj)
    adv = np.zeros(n)
    lastA = 0.0
    for i in range(n):
        t = n - 1 - i
        nnt = 0.0 if traj[t]["terminal"] else 1.0
        vNext = traj[t + 1]["value"] if t + 1 < n else 0.0
        delta = traj[t]["reward"] + gamma * vNext * nnt - traj[t]["value"]
        lastA = delta + gamma * lam * nnt * lastA
        adv[t] = lastA
    returns = np.array([adv[t] + traj[t]["value"] for t in range(n)])
    return adv, returns

def normalize_adv(adv):
    n = float(len(adv))
    mean = np.sum(adv) / n
    var = np.sum((adv - mean) ** 2) / n
    std = np.sqrt(var)
    return (adv - mean) / (std + 1e-8)

# ---- PPO objective gradient (analytical port of mlpGradPPO) -----------------------
# Objective (to ASCEND): min(rho*A, clip(rho)*A) - vf*0.5*(V-R)^2 + ent*H
#   rho   = exp(logp_a - oldLogp),  logp_k = out_k - logsumexp(out[:A])
#   H     = -sum_k pi_k * log pi_k  (pi = softmax(out[:A]))
# We differentiate wrt the logits `out` (length A+1: first A are action logits, last is V).

def ppo_grad_transition(p, x, a, adv, ret, oldLogp, vfCoef, entCoef, clipEps):
    z1, h, out = forward_all(p, x)
    A = len(p.b2) - 1
    logits = out[:A]
    m = np.max(logits)
    ex = np.exp(logits - m)
    Z = np.sum(ex)
    pi = ex / Z
    lse = m + np.log(Z)
    logp = logits - lse                      # log pi_k
    # --- policy surrogate gradient wrt logits[:A] ---
    rho = np.exp(logp[a] - oldLogp)
    # d logp_a / d logits_k = 1[k=a] - pi_k
    dlogp_a = -pi.copy(); dlogp_a[a] += 1.0
    # min(rho*A, clip(rho)*A): pick branch with smaller VALUE (Lean minV: a<=b -> a)
    surr1 = rho * adv
    ratioC = min(max(rho, 1.0 - clipEps), 1.0 + clipEps)
    surr2 = ratioC * adv
    if surr1 <= surr2:
        # gradient flows through surr1 = rho*adv; d rho/d logits = rho * dlogp_a
        dpol = adv * rho * dlogp_a
    else:
        # surr2 = clip(rho)*adv; clip grad is 1 strictly inside (1-eps,1+eps) else 0
        inside = (1.0 - clipEps) < rho < (1.0 + clipEps)
        dpol = (adv * rho * dlogp_a) if inside else np.zeros(A)
    # --- entropy gradient H = -sum pi_k logp_k, wrt logits ---
    # dH/d logit_j = -pi_j * (logp_j + H_scalar_removed...) ; standard result:
    #   dH/d logit_j = pi_j * ( sum_k pi_k logp_k - logp_j )  ... derive carefully:
    # H = -sum_k pi_k logp_k. logp_k = logit_k - lse. pi = softmax(logit).
    # d pi_k / d logit_j = pi_k (1[k=j] - pi_j)
    # d logp_k / d logit_j = 1[k=j] - pi_j
    # dH/d logit_j = -sum_k [ (d pi_k) logp_k + pi_k (d logp_k) ]
    #   term1 = -sum_k pi_k(1[k=j]-pi_j) logp_k = -(pi_j logp_j - pi_j sum_k pi_k logp_k)
    #   term2 = -sum_k pi_k (1[k=j]-pi_j) = -(pi_j - pi_j) = 0   (since sum pi_k =1)
    sum_pl = np.sum(pi * logp)
    dH = -(pi * logp - pi * sum_pl)          # length A
    dent = entCoef * dH
    # --- combined gradient wrt out[:A] ---
    dout = np.zeros(A + 1)
    dout[:A] = dpol + dent
    # --- value gradient: -vf*0.5*(V-R)^2 -> wrt V = out[A]: -vf*(V-R) ---
    V = out[A]
    dout[A] = -vfCoef * (V - ret)
    # --- backprop through layer 2, relu, layer 1 ---
    gb2 = dout
    gW2 = np.outer(dout, h)
    dh = p.W2.T @ dout
    dz1 = dh * (z1 > 0.0)
    gb1 = dz1
    gW1 = np.outer(dz1, x)
    return gW1, gb1, gW2, gb2

def ppo_grad(p, traj, adv, returns, vfCoef, entCoef, clipEps):
    advN = normalize_adv(adv)
    gW1 = np.zeros_like(p.W1); gb1 = np.zeros_like(p.b1)
    gW2 = np.zeros_like(p.W2); gb2 = np.zeros_like(p.b2)
    for t, tr in enumerate(traj):
        dW1, db1, dW2, db2 = ppo_grad_transition(
            p, tr["obs"], tr["action"], advN[t], returns[t], tr["oldLogp"],
            vfCoef, entCoef, clipEps)
        gW1 += dW1; gb1 += db1; gW2 += dW2; gb2 += db2
    return gW1, gb1, gW2, gb2

def update_ppo(p, traj, adv, returns, lr, vfCoef, entCoef, clipEps):
    gW1, gb1, gW2, gb2 = ppo_grad(p, traj, adv, returns, vfCoef, entCoef, clipEps)
    return MLP(p.W1 + lr * gW1, p.b1 + lr * gb1, p.W2 + lr * gW2, p.b2 + lr * gb2)

# ---- trainPPO (Exe/Puffer.lean runPPO hyperparameters) ----------------------------

CONFIGS = {
    "chain_mdp": dict(env=lambda: ChainMDP(5), hidden=16, episodes=3000, epochs=4,
                      lr=0.03, gamma=0.99, lam=0.95, vfCoef=0.5, entCoef=0.01,
                      clipEps=0.2, seed=0x1234),
    "squared":   dict(env=lambda: Squared(5, 14), hidden=24, episodes=4000, epochs=4,
                      lr=0.03, gamma=0.99, lam=0.95, vfCoef=0.5, entCoef=0.01,
                      clipEps=0.2, seed=0x1234),
}

def train_ppo(name, print_every=200):
    cfg = CONFIGS[name]
    env = cfg["env"]()
    p, rng = init_mlp(env.obsDim, cfg["hidden"], env.numActions + 1, cfg["seed"])
    curve = []          # (episode, avg_return_last_window)
    retSum = 0.0
    window = 0
    for ep in range(cfg["episodes"]):
        traj, ret, rng = rollout_env(env, p, rng)
        adv, returns = compute_gae(traj, cfg["gamma"], cfg["lam"])
        for _ in range(cfg["epochs"]):
            p = update_ppo(p, traj, adv, returns, cfg["lr"],
                           cfg["vfCoef"], cfg["entCoef"], cfg["clipEps"])
        retSum += ret
        window += 1
        if window == print_every:
            curve.append((ep + 1, retSum / print_every))
            retSum = 0.0
            window = 0
    return curve

if __name__ == "__main__":
    import sys, json
    name = sys.argv[1] if len(sys.argv) > 1 else "chain_mdp"
    curve = train_ppo(name)
    for ep, r in curve:
        print(f"  ep {ep}: avg return (last 200) = {r:.6f}")
    print(json.dumps(curve))
