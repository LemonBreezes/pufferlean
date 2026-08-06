#!/usr/bin/env python3
"""Upload puffer-lean training metrics (JSONL) to Weights & Biases or Neptune.

The Lean trainer (`puffer train <env> --track wandb --log run.jsonl`) writes one JSON
object per log interval using PufferLib's exact metric keys:

    losses/policy_loss   losses/value_loss   losses/entropy   losses/approx_kl
    losses/old_approx_kl losses/clipfrac     losses/explained_variance
    performance/uptime   performance/agent_steps   performance/epoch   performance/sps
    environment/episode_return   environment/episode_length

This sidecar replays those rows into `wandb.log(...)` / a Neptune run — the actual cloud
upload. Lean has no native HTTP client and wandb/neptune are Python SDKs, so the real
integration lives here; run it with YOUR account (`wandb login` / NEPTUNE_API_TOKEN).

    # 1. train, emitting the metric dict
    puffer train squared --track wandb --log run.jsonl
    # 2. upload it (your account)
    python tools/puffer_track.py --backend wandb run.jsonl
    # or preview the exact log calls without uploading / needing the SDK:
    python tools/puffer_track.py --backend wandb run.jsonl --dry-run

Step is PufferLib's `performance/agent_steps`, matching how pufferl.py logs to wandb.
"""
import argparse
import json
import sys


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_keys(row):
    """The namespaced metric keys wandb/neptune receive (drop bookkeeping like 'backend')."""
    return {k: v for k, v in row.items() if "/" in k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="metrics JSONL written by `puffer train ... --log <path>`")
    ap.add_argument("--backend", choices=["wandb", "neptune"], default="wandb")
    ap.add_argument("--project", default="puffer-lean")
    ap.add_argument("--name", default=None, help="run name (default: auto)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the log calls that would be made instead of uploading (no SDK/creds needed)")
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    if not rows:
        print(f"no metric rows in {args.jsonl}", file=sys.stderr)
        return 1

    if args.dry_run:
        for row in rows:
            step = int(row.get("performance/agent_steps", 0))
            print(f"{args.backend}.log(step={step}, {metric_keys(row)})")
        print(f"[dry-run] {len(rows)} rows -> {args.backend} (project={args.project!r})")
        return 0

    try:
        if args.backend == "wandb":
            import wandb
            run = wandb.init(project=args.project, name=args.name)
            for row in rows:
                step = int(row.get("performance/agent_steps", 0))
                wandb.log(metric_keys(row), step=step)
            run.finish()
        else:  # neptune
            import neptune
            run = neptune.init_run(project=args.project, name=args.name)
            for row in rows:
                for k, v in metric_keys(row).items():
                    run[k].append(v)
            run.stop()
    except ImportError:
        print(f"error: `{args.backend}` is not installed. `pip install {args.backend}` and "
              f"authenticate (wandb login / NEPTUNE_API_TOKEN), or preview with --dry-run.",
              file=sys.stderr)
        return 1

    print(f"uploaded {len(rows)} rows to {args.backend} (project={args.project!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
