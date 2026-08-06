#!/bin/sh
# Build ocean env plugins: each ocean/<name>/adapter.c wraps the real ocean env into libenv_<name>.so.
# Links the bundled raylib (rpath) so the env's c_render compiles/links, though we never call it (headless).
#   ./ocean/build.sh            # build every ocean/<name>/adapter.c
#   ./ocean/build.sh breakout   # one
# build continues past per-env failures
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"; RAY="${RAYLIB_HOME:-$PWD/_raylib}"
# raylib 5.5 is vendored in ocean/_raylib (git-ignored). If absent, fetch the prebuilt release for this
# platform; on anything but x86_64-Linux/macOS, build raylib 5.5 yourself and point RAYLIB_HOME at it.
if [ ! -e "$RAY/include/raylib.h" ] || [ ! -e "$RAY/lib/libraylib.so" ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) RL=raylib-5.5_linux_amd64 ;;
    Darwin-*)     RL=raylib-5.5_macos ;;
    *) echo "raylib not found at $RAY and no prebuilt for $(uname -sm) — build raylib 5.5 and set RAYLIB_HOME" >&2; exit 1 ;;
  esac
  command -v curl >/dev/null 2>&1 || { echo "need curl to fetch raylib (or set RAYLIB_HOME to an existing raylib 5.5)" >&2; exit 1; }
  echo "raylib not found — fetching $RL (raylib 5.5)..."
  { curl -fL "https://github.com/raysan5/raylib/releases/download/5.5/$RL.tar.gz" | tar xz && rm -rf _raylib && mv "$RL" _raylib; } \
    || { echo "raylib fetch failed; build raylib 5.5 manually and set RAYLIB_HOME" >&2; exit 1; }
  RAY="$PWD/_raylib"
fi
build_one(){ name="$1"; [ -f "$name/adapter.c" ] || { echo "  (no ocean/$name/adapter.c)"; return 0; }
  # Optional per-env overrides: ocean/<name>/build.flags may set ENVCC (compiler) and EXTRA (extra cflags),
  # e.g. envs whose C uses clang vector builtins (__builtin_elementwise_*) set `ENVCC=clang EXTRA=-march=native`.
  ENVCC="${CC:-gcc}"; EXTRA=""
  [ -f "$name/build.flags" ] && . "./$name/build.flags"
  command -v "$ENVCC" >/dev/null 2>&1 || { echo "  (skipping $name: compiler '$ENVCC' not found — install it or set CC)"; return 0; }
  echo "  building libenv_$name.so"
  $ENVCC -shared -fPIC -O2 $EXTRA -I "$name" -I "$RAY/include" -I "$PWD/_stubs" -I "$ROOT/ffi" \
     "$name/adapter.c" "$RAY/lib/libraylib.so" -lm -Wl,-rpath,"$RAY/lib" \
     -o "$name/libenv_$name.so"; }
if [ $# -gt 0 ]; then for n in "$@"; do build_one "$n"; done
else for d in */; do build_one "${d%/}"; done; fi
echo done.
