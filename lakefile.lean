import Lake
open Lake DSL System

/-!
# Lake build configuration for pufferlean

Migrated from `lakefile.toml` to `lakefile.lean` so the native FFI kernels
(`ffi/pufferffi.c`) can be compiled and linked by Lake itself via a custom
`target` + `moreLinkObjs`, retiring the old manual `bash ffi/build.sh` step and
the relative-path `moreLinkArgs = ["ffi/pufferffi.o"]` wart. `lake build puffer`
is now self-contained: Lake compiles the C oracle-kernels and links them into the
trainer exe. The `trace_*` difftest binaries do not depend on the FFI object.
-/

package «pufferlean»

-- Build-portability splices: each resolves an absolute path when the lakefile ELABORATES (reading env
-- with a fallback = this dev box), so both the IO compile targets and the pure `moreLinkArgs` can use
-- them. A change to $CUDA_HOME etc. needs `lake build puffer -R` (they are baked at configure time).
open Lean Elab Term in
/-- CUDA toolkit prefix: `$CUDA_HOME`, then `$CUDA_PATH`, then first existing of /opt/cuda,
    /usr/local/cuda (default /opt/cuda, this box). -/
elab "cudaHome%" : term => do
  let mut h := (← IO.getEnv "CUDA_HOME").getD ((← IO.getEnv "CUDA_PATH").getD "")
  if h == "" then
    for p in ["/opt/cuda", "/usr/local/cuda"] do
      if h == "" && (← (System.FilePath.mk p).pathExists) then h := p
  return .lit (.strVal (if h == "" then "/opt/cuda" else h))

open Lean Elab Term in
/-- OpenBLAS shared library (`$OPENBLAS_LIB`, default /usr/lib64/libopenblas.so). Linked by absolute
    path — see the `moreLinkArgs` note on why a broad `-L` is avoided. -/
elab "openblasLib%" : term => do
  return .lit (.strVal ((← IO.getEnv "OPENBLAS_LIB").getD "/usr/lib64/libopenblas.so"))

open Lean Elab Term in
/-- OpenBLAS include dir (`$OPENBLAS_INC`, default /usr/include/openblas). -/
elab "openblasInc%" : term => do
  return .lit (.strVal ((← IO.getEnv "OPENBLAS_INC").getD "/usr/include/openblas"))

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.28.0"

@[default_target]
lean_lib Puffer

@[test_driver]
lean_lib PufferTests

/-- Compile the native FFI kernels (`ffi/pufferffi.c`) to an object file. These are
    the hot-path C twins of the verified Lean AD gradients, invoked via `@[extern]` in
    `Puffer.Float.FFI` and validated bit-for-bit against the Lean oracle
    (`puffer verify-*-ffi`). `-I<lean include>` is system-dependent, so it is passed as
    a weak (untraced) arg; `-O2 -fPIC` are traced (a change rebuilds the object). -/
target pufferffiObj pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "pufferffi.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "pufferffi.c"
  buildO oFile srcJob #["-I", (← getLeanIncludeDir).toString] #["-O2", "-fPIC"]

/-- Compile the BLAS + cuBLAS accelerated kernels (`ffi/pufferblas.c`, the M7 GPU/BLAS
    path — a batched dense-forward GEMM with OpenBLAS and cuBLAS backends, `@[extern]` in
    `Puffer.Float.BLAS`). System include dirs (`openblas`, CUDA) are untraced weak-args. -/
target pufferblasObj pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "pufferblas.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "pufferblas.c"
  buildO oFile srcJob
    #["-I", (← getLeanIncludeDir).toString, "-I", openblasInc%, "-I", cudaHome% ++ "/include"]
    -- -ffp-contract=off: no mul-add (FMA) fusion, so the naive-C Muon step matches the Lean/GPU
    -- oracle's f64 op order bit-for-bit (same rationale as the nvcc `--fmad=false`).
    #["-O2", "-fPIC", "-ffp-contract=off"]

/-- Compile the runtime env-plugin loader (`ffi/puffer_loader.c`) — dlopen's `libenv_<name>.so` at
    runtime and drives it through the `puffer_env.h` ABI. The ONLY env-facing code in `puffer`; it sees
    no specific env at compile time. `dlopen`/`dlsym` live in glibc (≥2.34), so no extra `-ldl` needed. -/
target pufferLoaderObj pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "puffer_loader.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "puffer_loader.c"
  buildO oFile srcJob #["-I", (← getLeanIncludeDir).toString] #["-O2", "-fPIC"]

/-- Compile the native CUDA kernels (`ffi/puffercuda.cu`, the nvcc GPU training-step layer — V-Trace,
    Muon, …) to an object linked into the exe alongside the plain-C FFI objects. Reuses Lake's `buildO`
    with the compiler overridden to `nvcc` — separate compilation only (no `-dc`/`-dlink`); the object
    self-registers its fatbin and `cudart` (already linked) resolves launch. The host compiler
    (`-ccbin`) defaults to `gcc-15` because CUDA 13's host_config hard-errors on gcc > 15; the GPU arch
    defaults to `sm_120` (this box's RTX 5090). BUILDING ELSEWHERE: set `PUFFER_CUDA_ARCH=sm_XX` for your
    GPU (sm_86 30xx / sm_89 40xx / sm_90 Hopper, or `native`), `PUFFER_NVCC_CCBIN=<gcc≤15 or clang>` (or
    empty to omit), `CUDACXX`/`CUDA_HOME` for nvcc. f64 IEEE ops are arch-independent, so `--fmad=false`
    and the `verify-*-gpu` bit-exact checks hold on any GPU. `extern "C"` keeps launchers `--gc-sections`-reachable. -/
target puffercudaObj pkg : FilePath := do
  let oFile := pkg.buildDir / "ffi" / "puffercuda.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "puffercuda.cu"
  let arch := (← IO.getEnv "PUFFER_CUDA_ARCH").getD "sm_120"
  let ccbin := match (← IO.getEnv "PUFFER_NVCC_CCBIN") with
    | some "" => #[]                                   -- box whose default gcc is already ≤ 15
    | some cc => #["-ccbin", cc]
    | none    => #["-ccbin", "gcc-15"]
  let nvccCandExists ← (System.FilePath.mk (cudaHome% ++ "/bin/nvcc")).pathExists
  let nvcc := (← IO.getEnv "CUDACXX").getD (if nvccCandExists then cudaHome% ++ "/bin/nvcc" else "nvcc")
  buildO oFile srcJob
    #["-I", (← getLeanIncludeDir).toString, "-I", cudaHome% ++ "/include"]
    -- --fmad=false: no mul-add fusion, so f64 ops match the CPU/Lean oracle bit-for-bit
    -- (V-Trace scan is then bit-exact vs computePuffAdvantageV; Muon stays f64-tight).
    (#[s!"-arch={arch}"] ++ ccbin ++ #["-std=c++17", "-Xcompiler=-fPIC", "-O2", "--fmad=false", "--threads", "0"])
    (compiler := nvcc)


/-- The trainer/CLI. Links the native FFI objects via `moreLinkObjs` and the OpenBLAS +
    cuBLAS libraries for the GPU/BLAS path via `moreLinkArgs` — both scoped to this exe
    only, so the `trace_*` binaries stay dependency-free. The `-Wl,-rpath` lets the
    binary find the CUDA libs at runtime without `LD_LIBRARY_PATH`. -/
lean_exe puffer where
  root := `Exe.Puffer
  moreLinkObjs := #[pufferffiObj, pufferblasObj, puffercudaObj, pufferLoaderObj]
  -- Link the external libs by ABSOLUTE PATH rather than `-L<dir> -l<name>`: the Lean
  -- toolchain links with `lld` under its own `--sysroot` + bundled glibc, and adding
  -- `-L/usr/lib64` to the search path shadows that glibc (the system's glibc ≥2.34
  -- dropped `__libc_csu_init/_fini`, which the toolchain's `Scrt1.o` still needs).
  -- Absolute paths pull in exactly these libraries without perturbing libc resolution.
  -- The `-rpath` lets the binary find the CUDA runtime libs without `LD_LIBRARY_PATH`.
  -- Paths come from `cudaHomeStr`/`openblasLibStr` (env-overridable via CUDA_HOME/OPENBLAS_LIB); the
  -- absolute-path linking is kept (we discover the DIR, not add a broad -L) to preserve the glibc note.
  moreLinkArgs := #[openblasLib%,
                    cudaHome% ++ "/lib64/libcudart.so", cudaHome% ++ "/lib64/libcublas.so",
                    cudaHome% ++ "/lib64/libcublasLt.so",
                    "-Wl,-rpath," ++ cudaHome% ++ "/lib64"]
