#!/bin/bash
# All-env training sweep: runs every built ocean plugin and reports SPS + reward trend.
#
#   ./tools/env_sweep.sh                       # defaults: 1024 agents, 20M steps
#   STEPS=4000000 AGENTS=4096 ./tools/env_sweep.sh
#   ./tools/env_sweep.sh breakout squared      # subset
#
# Reads the batch-reward trend, NOT the episode return: many ocean envs never set a terminal
# flag (upstream reports their episodes through a per-env `Log` channel our plugin ABI does not
# carry), so a terminal-normalised mean reads a permanent 0.0 for them and hides real learning.
#
# GPU runs are strictly sequential on purpose — concurrent runs corrupt each other's timings.
set -u
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH
# `puffer train` renders PufferLib's live dashboard by DEFAULT now; force the parseable lines we grep.
export PUFFER_PLAIN_LOG=1
# This sweep needs gawk (asort() in the band classifier) and bc; guard so a mawk/BSD-awk box fails loud.
command -v bc >/dev/null 2>&1 || { echo "env_sweep.sh needs bc" >&2; exit 1; }
echo | awk 'BEGIN{x[1]=1; asort(x)}' >/dev/null 2>&1 || { echo "env_sweep.sh needs gawk (asort() missing from $(command -v awk))" >&2; exit 1; }
STEPS=${STEPS:-20000000}
AGENTS=${AGENTS:-1024}
OUT=${OUT:-/tmp/puffer_env_sweep}
TIMEOUT=${TIMEOUT:-300}
mkdir -p "$OUT"
: > "$OUT/summary.tsv"

if [ $# -gt 0 ]; then envs="$*"; else
  envs=$(ls ocean/*/libenv_*.so 2>/dev/null | sed 's|.*/libenv_||;s|\.so||')
fi
[ -z "$envs" ] && { echo "no built env plugins — run ./ocean/build.sh first"; exit 1; }

printf '%-20s %-12s %10s %6s %-9s %s\n' env verdict SPS upd metric "episode_return trend"
printf -- '-%.0s' {1..92}; echo
for env in $envs; do
  log="$OUT/$env.log"
  timeout "$TIMEOUT" .lake/build/bin/puffer train "$env" \
      --total-agents "$AGENTS" --train.total-timesteps "$STEPS" > "$log" 2>&1
  rc=$?
  sps=$(grep -oE '⇒ [0-9]+ SPS' "$log" | tail -1 | grep -oE '[0-9]+')
  upd=$(grep -oE '[0-9]+ updates' "$log" | head -1 | grep -oE '[0-9]+')
  # Prefer the env's OWN log (PufferLib's `Log.episode_return`, via the plugin log channel) — it is
  # the number comparable with upstream, and the only one defined for the 14 envs that never raise a
  # terminal flag. Fall back to the batch-reward trend when an env exports no log channel.
  # Per-env metric override. `episode_return` is the right field for most envs, but three envs make
  # it meaningless and each for a different, verified reason:
  #   convert     — the env NEVER assigns log.episode_return (declared at convert.h:15, written
  #                 nowhere; upstream's my_log exports it anyway, so PufferLib prints 0.0 too). Its
  #                 score/n and perf/n are pinned to exactly 1.0 by construction, so batch reward is
  #                 the only monotone signal.
  #   robocode    — 1v1 selfplay with exactly antisymmetric rewards (robocode.h:380-386, :584-591),
  #                 so the summed episode_return is identically 0. `score` = mean damage dealt per
  #                 agent, which does rise with skill. NB slot_0_score is a win rate that converges
  #                 to 0.5 by symmetry under selfplay — a sanity check, not a progress metric.
  #   minimal     — its Log has no episode_return field at all (perf/score/n only).
  # slimevolley legitimately reads ~0 (a random policy cannot score off the built-in bot), so it
  # keeps the default and a 0 there is a true statement, not a bug.
  case "$env" in
    convert)           metric=""      ;;   # force the batch-reward fallback
    robocode|cartpole) metric="score" ;;   # these Logs have no episode_return field
    # `n` (events per update, reported UNDIVIDED) is the honest progress signal for these three;
    # it equals the summed batch reward. Their episode_return/score are constants or drifting
    # stopwatches — see the notes below and docs/KNOWN_GAPS.md.
    target|rware)      metric="n" ;;
    # minimal has no episode_return either, and `score` is actively misleading: it grows more
    # negative with episode LENGTH, and a random policy already scores -872, so our "-26 -> -900"
    # was a length readout, not a collapse. `perf` is the normalised progress field
    # (random floor 0.166, ours holds 0.21-0.25).
    # minimal: `score` is a wall-clock STOPWATCH, not a score — it is -(mean steps between catches)
    # and drifts -28 -> -1272 under a FIXED random policy, because the env never resets so the
    # inter-catch mean grows with elapsed time regardless of skill. `perf` is also contaminated for
    # the first ~10 updates (it starts near 0.85 only because no interval can be long yet). `n` =
    # catches per update = summed batch reward, and is the honest signal. Measured: the trained
    # policy reads -900 at update 183 where a RANDOM policy reads -1597, i.e. 1.8x better — this env
    # was never declining.
    minimal)  metric="n" ;;
    *)                 metric="episode_return" ;;
  esac
  sums=""; src="envlog"
  # NB the leading-space anchor is required: a bare `n=` also matches the "n=" inside
  # "retur[n=]1.000000", which silently interleaved bogus 1.0 samples into target/rware.
  [ -n "$metric" ] && sums=$(grep -oE "(^| )$metric=[-+0-9.]+" "$log" | sed "s/.*$metric=//")
  [ -n "$metric" ] && src="$metric"
  # Batch-reward fallback must cover BOTH trainer log formats: the MLP/MD/Cont trainers print
  # "batch reward Σ=...", while the recurrent trainer prints "(mean/step ...)" and no Σ at all.
  # Routing the MD envs onto the recurrent core silently emptied this for convert until it read both.
  if [ -z "$sums" ]; then sums=$(grep -oE 'Σ=[-+0-9.]+' "$log" | sed 's/Σ=//'); src="batchrew"; fi
  if [ -z "$sums" ]; then sums=$(grep -oE 'mean/step [-+0-9.e]+' "$log" | sed 's|mean/step ||'); src="mean/step"; fi
  first=$(echo "$sums" | head -1); last=$(echo "$sums" | tail -1); n=$(echo "$sums" | grep -c .)
  # Classify on the LAST-THIRD mean vs the FIRST-THIRD mean, not first-vs-last: a single noisy final
  # sample used to flip a healthy env to "REWARD-DOWN" (double_pendulum sat at 6.37 -> 6.32, converged
  # and flat, and got reported as declining). CONVERGED is its own verdict — a plateau is not a
  # regression, and conflating them sent me chasing three non-bugs.
  # MEDIAN of each band, not the mean. These series are heavy-tailed — target reads
  # 404 1830 1780 2350 1010 2520: clearly rising, but one 1010 dip drags the trailing MEAN below the
  # leading one and the env was reported as DECLINES. A median band ignores the outlier and reports
  # the trend the numbers actually show. (Verdict thresholds are unchanged.)
  bands=$(echo "$sums" | awk '{v[NR]=$1} END{ if(NR<2){print "NA NA"; exit}
      k=int(NR/3); if(k<1)k=1;
      for(i=1;i<=k;i++) A[i]=v[i];
      for(i=1;i<=k;i++) B[i]=v[NR-k+i];
      n=asort(A); m=asort(B);
      am=(n%2)?A[(n+1)/2]:(A[n/2]+A[n/2+1])/2;
      bm=(m%2)?B[(m+1)/2]:(B[m/2]+B[m/2+1])/2;
      printf "%.6f %.6f", am, bm }')
  fb=$(echo "$bands" | cut -d' ' -f1); lb=$(echo "$bands" | cut -d' ' -f2)
  # Reference values MEASURED from PufferLib's own native _C trainer on this box at 12M steps
  # (see docs/KNOWN_GAPS.md). An env whose final band lands at or above the reference is at parity,
  # however its curve moved inside the run — maze and drmario both decline within a run yet finish
  # exactly where PufferLib finishes, and flagging them as regressions sent me chasing two non-bugs.
  case "$env" in
    maze)      ref=0.126  ;;
    drmario)   ref=-2.531 ;;
    lightsout) ref=-1.752 ;;
    laser_puzzle) ref=1.454 ;;
    double_pendulum) ref=4.328 ;;   # both sides sit below the random floor on this env; ours leads
    *)         ref="" ;;
  esac
  if [ "$rc" -ne 0 ]; then verdict="CRASH(rc=$rc)"
  elif [ "$n" -eq 0 ]; then verdict="NO-OUTPUT"
  elif [ "$(echo "$first == 0 && $last == 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then verdict="ZERO-REWARD"
  elif [ "$n" -lt 2 ]; then verdict="ONE-SAMPLE"
  elif [ "$fb" = "NA" ]; then verdict="ONE-SAMPLE"
  else
    # relative change against the magnitude of the first band; |delta| < 5% counts as CONVERGED
    rel=$(echo "scale=6; d=($lb)-($fb); m=($fb); if(m<0) m=-m; if(m<0.000001) m=0.000001; d/m" | bc -l 2>/dev/null || echo 0)
    up=$(echo "$rel > 0.05"  | bc -l 2>/dev/null || echo 0)
    dn=$(echo "$rel < -0.05" | bc -l 2>/dev/null || echo 0)
    if   [ "$up" = "1" ]; then verdict="LEARNS"
    elif [ "$dn" = "1" ]; then verdict="DECLINES"
    else verdict="CONVERGED"; fi
    # a measured reference overrides the within-run trend: matching upstream is not a regression
    # 10% tolerance, correct on both signs: for a NEGATIVE reference, "within 10%" means allowing a
    # slightly MORE negative value (ref*1.1), not less — ref*0.9 tightens the bar and wrongly failed
    # drmario (ref -2.531, ours -2.55).
    if [ -n "$ref" ]; then
      if [ "$(echo "$ref < 0" | bc -l)" = "1" ]; then tol=$(echo "$ref * 1.1" | bc -l)
      else tol=$(echo "$ref * 0.9" | bc -l); fi
      [ "$(echo "$lb >= $tol" | bc -l 2>/dev/null || echo 0)" = "1" ] && verdict="PARITY(ref)"
    fi
  fi
  trend=$(echo "$sums" | awk '{printf "%.3g ", $1}' | cut -c1-46)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$env" "$verdict" "${sps:-0}" "${upd:-0}" "${first:-NA}" "${last:-NA}" >> "$OUT/summary.tsv"
  printf '%-20s %-12s %9.2fM %6s %-9s %s\n' "$env" "$verdict" "$(echo "${sps:-0}/1000000" | bc -l)" "${upd:-0}" "$src" "$trend"
done
echo
echo "logs + summary.tsv in $OUT"
echo "NOTE: single-seed REWARD-DOWN is usually seed noise — re-check with --train.seed before concluding."
