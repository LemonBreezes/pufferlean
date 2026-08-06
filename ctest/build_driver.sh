#!/usr/bin/env bash
# Compile a headless C trace driver for an Ocean env.
#   ctest/build_driver.sh <env>
# Links the real vendored raylib but never calls render fns, so it runs with no
# display. Output: ctest/bin/<env>_trace
set -euo pipefail
ENV="${1:?usage: build_driver.sh <env>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PUF="${PUFFERLIB_DIR:-$HOME/src/PufferLib}"
RL="$PUF/raylib-5.5_linux_amd64"

mkdir -p "$HERE/bin"
gcc -O2 \
    -I"$RL/include" -I"$PUF/src" -I"$PUF/vendor" -I"$PUF/ocean/$ENV" \
    "$HERE/drivers/${ENV}_trace.c" "$RL/lib/libraylib.a" -lm \
    -o "$HERE/bin/${ENV}_trace"
echo "built ctest/bin/${ENV}_trace"
