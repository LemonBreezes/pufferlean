#!/usr/bin/env python3
"""Bridge puffer-lean training metrics to Weights & Biases — PufferLib's only tracker.

The Lean trainer has no HTTP client and wandb is a Python SDK, so `puffer train <env> --wandb`
spawns THIS script as a live daemon and streams it one JSON metric row per ~0.6 s dashboard
tick. The daemon replays those rows into `wandb.log(...)` exactly as PufferLib's `pufferl.py`
does — same init, same native `_C` metric keys, same `step=agent_steps`, same end-of-run
checkpoint→Artifact upload. From the wandb side the run is indistinguishable from a native
PufferLib run.

Two modes:

  # live daemon (how `puffer train ... --wandb` uses it; reads rows on stdin):
  #   line 1        : {"<config-key>": "<val>", ...}   (becomes wandb.init(config=...))
  #   lines 2..N-1  : {"SPS":.., "agent_steps":.., "loss/policy":.., "env/score":.., ...}
  #   line N        : {"__finish__": {"checkpoint": "<path or null>"}}
  python tools/puffer_track.py --daemon --project puffer4 --group debug [--tag foo]
  python tools/puffer_track.py --daemon --dry-run ...   # preview log calls, no SDK/creds needed

  # offline: replay a saved JSONL (one row per line) after the fact
  python tools/puffer_track.py run.jsonl --project puffer4

Run it under YOUR account: `wandb login` (or WANDB_API_KEY=...); WANDB_MODE=offline works too.
PufferLib logs to wandb only once `env/score` exists (the sweep target key) — the daemon
mirrors that gate. Native `_C` keys (matching src/bindings.cu, NOT the --slowly torch names):

    SPS  agent_steps  uptime  epoch
    loss/policy  loss/value  loss/entropy  loss/total  loss/old_kl  loss/kl  loss/clipfrac
    perf/rollout  perf/train  perf/eval_gpu  perf/eval_env  perf/train_misc  perf/train_forward
    util/gpu_percent  util/gpu_mem  util/vram_used_gb  util/vram_total_gb  util/cpu_mem_gb
    env/score  env/perf  env/episode_return  env/episode_length  (env's own PufferLib Log)
"""
import argparse
import json
import sys


def metric_keys(row):
    """The namespaced metric keys wandb receives (drop control/bookkeeping fields)."""
    return {k: v for k, v in row.items() if "/" in k or k in ("SPS", "agent_steps", "uptime", "epoch")}


def run_daemon(args):
    """Hold a wandb run open, logging rows streamed on stdin until a __finish__ / EOF.

    Mirrors pufferl.py::_train: init with id=generate_id() + the whole config dict, log the
    cumulative flat dict each tick with step=agent_steps once env/score exists, and at the end
    upload the final checkpoint as Artifact(run_id, type='model') then run.finish()."""
    # Line 1 is the config dict for wandb.init(config=...). Tolerate a missing/blank first line.
    first = sys.stdin.readline()
    try:
        config = json.loads(first) if first.strip() else {}
    except json.JSONDecodeError:
        config = {}

    wb = None
    run_id = None
    if not args.dry_run:
        try:
            import wandb
        except ImportError:
            print("puffer_track: `wandb` not installed — `pip install wandb` and `wandb login`, "
                  "or use --dry-run. Draining stdin so training is unaffected.", file=sys.stderr)
            # Drain until the trainer's __finish__ (or EOF); breaking on __finish__ lets the Lean side's
            # `finish` (which writes that line then waits for us) return instead of deadlocking.
            for line in sys.stdin:
                if "__finish__" in line:
                    break
            return 0
        run_id = wandb.util.generate_id()
        wb = wandb.init(
            id=run_id, config=config,
            project=args.project, group=args.group,
            tags=[args.tag] if args.tag else [],
            settings=wandb.Settings(console="off"),
        )
    else:
        run_id = "dryrun"
        print(f"[dry-run] wandb.init(project={args.project!r}, group={args.group!r}, "
              f"tags={[args.tag] if args.tag else []}, config={config})", file=sys.stderr)

    seen_target = False          # PufferLib gates wandb.log on the sweep target key (env/score)
    n_logged = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "__finish__" in row:
            ckpt = (row["__finish__"] or {}).get("checkpoint")
            if not args.dry_run:
                if ckpt:                                 # end-of-run model → wandb Artifact
                    import wandb
                    art = wandb.Artifact(run_id, type="model")
                    art.add_file(ckpt)
                    wb.log_artifact(art)
                wb.finish()
            else:
                print(f"[dry-run] finish(checkpoint={ckpt!r}) after {n_logged} rows", file=sys.stderr)
            break
        if not seen_target:
            if "env/score" not in row:
                continue
            seen_target = True
        step = int(row.get("agent_steps", 0))
        keys = metric_keys(row)
        if args.dry_run:
            print(f"[dry-run] wandb.log(step={step}, {keys})", file=sys.stderr)
        else:
            wb.log(keys, step=step)
        n_logged += 1
    else:
        # stdin closed without __finish__ (trainer crashed / killed) — still close the run.
        if not args.dry_run and wb is not None:
            wb.finish()
    return 0


def run_replay(args):
    """Offline: replay a saved JSONL file into a fresh wandb run."""
    with open(args.jsonl) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    if not rows:
        print(f"no metric rows in {args.jsonl}", file=sys.stderr)
        return 1
    if args.dry_run:
        for row in rows:
            print(f"wandb.log(step={int(row.get('agent_steps', 0))}, {metric_keys(row)})")
        print(f"[dry-run] {len(rows)} rows -> wandb (project={args.project!r})")
        return 0
    try:
        import wandb
    except ImportError:
        print("error: `wandb` not installed. `pip install wandb` and `wandb login`, or --dry-run.",
              file=sys.stderr)
        return 1
    run = wandb.init(project=args.project, group=args.group,
                     tags=[args.tag] if args.tag else [])
    for row in rows:
        run.log(metric_keys(row), step=int(row.get("agent_steps", 0)))
    run.finish()
    print(f"uploaded {len(rows)} rows to wandb (project={args.project!r})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="offline: a metrics JSONL to replay (omit for --daemon)")
    ap.add_argument("--daemon", action="store_true", help="live mode: read rows from stdin, hold the run open")
    ap.add_argument("--project", default="puffer4")     # PufferLib's default wandb project
    ap.add_argument("--group", default="debug")         # PufferLib's default wandb group
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the wandb calls instead of making them (no SDK/creds needed)")
    args = ap.parse_args()

    if args.daemon:
        return run_daemon(args)
    if not args.jsonl:
        ap.error("give a JSONL file to replay, or pass --daemon for live streaming")
    return run_replay(args)


if __name__ == "__main__":
    sys.exit(main())
