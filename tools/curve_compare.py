#!/usr/bin/env python3
"""Empirical learning-curve comparison: Lean `puffer train-ppo` vs the NumPy PPO reference.

Per the project philosophy (per-step numerics are proven; a full RL run is chaotic and is
validated EMPIRICALLY), this runs BOTH trainers on the SAME env + seed + hyperparameters and
overlays their reward-vs-episode learning curves, with a quantitative agreement summary.

  * Lean side  : the actual `puffer` binary, subcommand `train-ppo-curve <env> <win>`
                 (identical numerics to `train-ppo`; only the reporting cadence differs).
  * Python side: tools/ppo_ref.py — a faithful line-by-line port of the Lean trainer
                 (same splitmix64 PRNG, weight init, env, GAE, PPO objective, SGD update).

Both are deterministic (fixed seed 0x1234); rerunning gives identical curves.

Usage:
  tools/curve_compare.py                 # chain_mdp + squared, win=20/40, plot + summary
  tools/curve_compare.py chain_mdp 20    # a single env at a chosen window
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, ".lake", "build", "bin", "puffer")
DATA = os.path.join(ROOT, "tools", "curve_data")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ppo_ref as R


def lean_curve(env, win):
    """Run the real Lean binary and parse '<episode> <avg_return>' lines."""
    out = subprocess.run([BIN, "train-ppo-curve", env, str(win)],
                         capture_output=True, text=True, check=True).stdout
    curve = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ep, r = line.split()
        curve.append((int(ep), float(r)))
    return curve


def summary(name, lean, ref):
    """Quantitative agreement between two [(episode, return)] curves on shared episodes."""
    import statistics
    ld = dict(lean); rd = dict(ref)
    eps = sorted(set(ld) & set(rd))
    lv = [ld[e] for e in eps]
    rv = [rd[e] for e in eps]
    diffs = [abs(a - b) for a, b in zip(lv, rv)]
    n = len(eps)
    auc_l = sum(lv) / n
    auc_r = sum(rv) / n
    # Pearson correlation (guard against zero variance when both curves are flat/identical)
    try:
        corr = statistics.correlation(lv, rv) if n > 1 and statistics.pstdev(lv) > 0 and statistics.pstdev(rv) > 0 else 1.0
    except statistics.StatisticsError:
        corr = 1.0
    return {
        "env": name, "windows": n,
        "lean_final": lv[-1], "ref_final": rv[-1],
        "lean_auc": auc_l, "ref_auc": auc_r,
        "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / n,
        "correlation": corr,
        "identical": all(d == 0.0 for d in diffs),
    }


def main():
    os.makedirs(DATA, exist_ok=True)
    if len(sys.argv) > 1:
        jobs = [(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)]
    else:
        jobs = [("chain_mdp", 20), ("squared", 40)]

    results = {}
    curves = {}
    for env, win in jobs:
        print(f"== {env} (window={win}) ==")
        lean = lean_curve(env, win)
        ref = R.train_ppo(env, print_every=win)
        with open(os.path.join(DATA, f"lean_{env}_win{win}.json"), "w") as f:
            json.dump(lean, f)
        with open(os.path.join(DATA, f"ref_{env}_win{win}.json"), "w") as f:
            json.dump(ref, f)
        s = summary(env, lean, ref)
        results[env] = s
        curves[env] = (win, lean, ref)
        print(json.dumps(s, indent=2))

    with open(os.path.join(DATA, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- overlay plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(curves)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.5), squeeze=False)
        for ax, (env, (win, lean, ref)) in zip(axes[0], curves.items()):
            le, lv = zip(*lean); re, rv = zip(*ref)
            ax.plot(le, lv, label="Lean puffer train-ppo", lw=2, color="#1f77b4")
            ax.plot(re, rv, label="NumPy PPO reference", lw=1.4, ls="--", color="#d62728")
            s = results[env]
            ax.set_title(f"{env}  (seed 0x1234, win={win})\n"
                         f"final Lean={s['lean_final']:.3f} / ref={s['ref_final']:.3f}, "
                         f"corr={s['correlation']:.4f}, maxΔ={s['max_abs_diff']:.3g}")
            ax.set_xlabel("episode")
            ax.set_ylabel(f"avg return (last {win} eps)")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        png = os.path.join(DATA, "curve_compare.png")
        fig.savefig(png, dpi=110)
        print(f"\nplot saved -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
