/* Native BLAS + cuBLAS accelerated kernels for the puffer trainer (the M7 GPU/BLAS
   path). Each has a Lean `@[extern]` twin in `Puffer/Float/BLAS.lean`.

   ORACLE DISCIPLINE, amended: the scalar/right-fold kernels in `pufferffi.c` are
   BIT-IDENTICAL to the verified Lean `dotF`/AD oracle. BLAS and cuBLAS use blocked /
   multithreaded / parallel reductions, so their results are NOT bit-identical — they
   match the oracle only to floating-point tolerance (~1e-10..1e-13 relative). This is
   the deliberate trade: the bit-exact scalar path stays the oracle-faithful default;
   this file is the FAST, tolerance-validated path. `puffer verify-blas` checks the
   tolerance; `puffer bench-blas` reports the scaling of all three backends.

   The demonstrated op is a batched dense layer with ReLU (the GEMM-dominant hot path
   of the encoder/rollout):  Y[N×H] = relu( X[N×D] · W[H×D]^T + b[H] ),  row-major.

   DIMENSION LIMIT: the (reference/OpenBLAS) CBLAS API takes `int` extents, so every
   size_t dimension is cast to int at the dgemm boundary. All matrix extents — including
   the batch-scaled conv rows R = N·oHoW — must therefore fit in INT_MAX (2^31-1). This
   holds for every realistic RL minibatch (thousands of transitions, R ~ millions); it
   would only overflow at an absurd single-minibatch size (R > 2.1e9, tens of GB of
   buffers). The scalar kernels in pufferffi.c use size_t throughout and have no such cap;
   exceeding INT_MAX is the one case where the BLAS path is not a drop-in for them. */
#include <lean/lean.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

/* Shared `--log` dashboard-loss channel, DEFINED (non-static) in ffi/puffercuda.cu (always linked into
   the puffer exe — see lakefile.lean moreLinkObjs). The LSTM BPTT grad below reduces the same 7 losses
   into g_mgLoss under the same toggle, so lean_cuda_mg_read_losses (also in puffercuda.cu) serves the
   LSTM trainer too. Off by default ⇒ the reduction below is skipped entirely (zero cost). */
extern int g_mgLossOn;
extern double g_mgLoss[7];

/* GPU-resident LSTM BPTT (ffi/puffercuda.cu). Fills gOut[P] and returns 1 on success; returns 0 (no
   usable device / unsupported shape) so the three BLAS BPTT kernels below run their CPU fallback. On --log
   render frames `outHostOrNull` receives the T·B·O logits for the shared dashboard-loss reducer. `tier`
   picks the GEMM precision: 0 = cublasDgemm (f64 default), 1 = cublasSgemm (f32 tier), 2 = bf16 tensor
   cores (float buffers, CUBLAS_COMPUTE_32F_FAST_16BF, f32 accumulate). */
extern int cuda_lstm_ppo_grad_batch(
    const double* pp, const double* obsB, const double* actA, const double* advA,
    const double* retA, const double* oldA, const double* termA,
    const double* h0s, const double* c0s, size_t B, size_t T, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double* gOut, double* outHostOrNull, int tier);
/* PUFFER_LSTM_GPU=0 forces the CPU BLAS path (A/B + the CPU-oracle cross-check); default uses the GPU. */
static int lstm_gpu_off(void){ static int v=-1; if(v<0){ const char* e=getenv("PUFFER_LSTM_GPU"); v=(e&&e[0]=='0'); } return v; }

/* --- helpers ------------------------------------------------------------------ */
static inline void bias_relu(double* Y, const double* b, size_t N, size_t H) {
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < H; j++) {
      double v = Y[i*H + j] + b[j];
      Y[i*H + j] = v > 0.0 ? v : 0.0;
    }
}

/* --- scalar reference (in this file, for an apples-to-apples 3-way benchmark) ---
   Left-folded so it equals the Lean naive `denseForwardRef` primal. */
LEAN_EXPORT lean_obj_res lean_ffi_dense_forward_ref(
    lean_obj_arg Xa, lean_obj_arg Wa, lean_obj_arg ba, size_t N, size_t D, size_t H) {
  const double* X = lean_float_array_cptr(Xa);
  const double* W = lean_float_array_cptr(Wa);
  const double* b = lean_float_array_cptr(ba);
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*H, N*H);
  double* Y = lean_float_array_cptr(Yo);
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < H; j++) {
      double acc = 0.0; const double* xr = X + i*D; const double* wr = W + j*D;
      for (size_t d = 0; d < D; d++) acc += xr[d] * wr[d];
      double v = acc + b[j];
      Y[i*H + j] = v > 0.0 ? v : 0.0;
    }
  lean_dec(Xa); lean_dec(Wa); lean_dec(ba);
  return Yo;
}

/* --- OpenBLAS (CPU, multithreaded, AVX-512) ----------------------------------- */
LEAN_EXPORT lean_obj_res lean_ffi_dense_forward_blas(
    lean_obj_arg Xa, lean_obj_arg Wa, lean_obj_arg ba, size_t N, size_t D, size_t H) {
  const double* X = lean_float_array_cptr(Xa);
  const double* W = lean_float_array_cptr(Wa);
  const double* b = lean_float_array_cptr(ba);
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*H, N*H);
  double* Y = lean_float_array_cptr(Yo);
  /* Y[N×H] = X[N×D] · W[H×D]^T  (row-major): A=X (NoTrans), B=W (Trans). */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (int)N, (int)H, (int)D, 1.0, X, (int)D, W, (int)D, 0.0, Y, (int)H);
  bias_relu(Y, b, N, H);
  lean_dec(Xa); lean_dec(Wa); lean_dec(ba);
  return Yo;
}

/* --- cuBLAS (GPU) ------------------------------------------------------------- */
/* Cached handle (lazy). Lean FFI is single-threaded here, so a static is safe. */
static cublasHandle_t g_cublas = NULL;
static int g_cublas_ok = 0;
static cublasHandle_t get_cublas(void) {
  if (!g_cublas_ok) {
    if (cublasCreate(&g_cublas) == CUBLAS_STATUS_SUCCESS) g_cublas_ok = 1;
    else g_cublas_ok = -1;
  }
  return g_cublas_ok == 1 ? g_cublas : NULL;
}

LEAN_EXPORT lean_obj_res lean_ffi_dense_forward_cublas(
    lean_obj_arg Xa, lean_obj_arg Wa, lean_obj_arg ba, size_t N, size_t D, size_t H) {
  const double* X = lean_float_array_cptr(Xa);
  const double* W = lean_float_array_cptr(Wa);
  const double* b = lean_float_array_cptr(ba);
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*H, N*H);
  double* Y = lean_float_array_cptr(Yo);
  cublasHandle_t h = get_cublas();
  double *dX=NULL,*dW=NULL,*dY=NULL;
  if (!h ||
      cudaMalloc((void**)&dX, sizeof(double)*N*D) ||
      cudaMalloc((void**)&dW, sizeof(double)*H*D) ||
      cudaMalloc((void**)&dY, sizeof(double)*N*H)) {
    /* GPU unavailable: fall back to the CPU BLAS path so the result is still correct. */
    if (dX) cudaFree(dX); if (dW) cudaFree(dW); if (dY) cudaFree(dY);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)N,(int)H,(int)D, 1.0, X,(int)D, W,(int)D, 0.0, Y,(int)H);
    bias_relu(Y, b, N, H);
    lean_dec(Xa); lean_dec(Wa); lean_dec(ba);
    return Yo;
  }
  cudaMemcpy(dX, X, sizeof(double)*N*D, cudaMemcpyHostToDevice);
  cudaMemcpy(dW, W, sizeof(double)*H*D, cudaMemcpyHostToDevice);
  double al = 1.0, be = 0.0;
  /* Row-major Y[N×H]=X·W^T. In column-major (cuBLAS): the row-major arrays are their
     transposes, so compute Ycm[H×N] = op_T(Wcm[D×H]) · op_N(Xcm[D×N]).
     dW is Wrm[H×D] = Wcm[D×H] (lda=D); dX is Xrm[N×D] = Xcm[D×N] (ldb=D); dY is
     Yrm[N×H] = Ycm[H×N] (ldc=H). */
  cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)H, (int)N, (int)D,
              &al, dW, (int)D, dX, (int)D, &be, dY, (int)H);
  cudaDeviceSynchronize();
  cudaMemcpy(Y, dY, sizeof(double)*N*H, cudaMemcpyDeviceToHost);
  cudaFree(dX); cudaFree(dW); cudaFree(dY);
  bias_relu(Y, b, N, H);
  lean_dec(Xa); lean_dec(Wa); lean_dec(ba);
  return Yo;
}

/* --- cuBLASLt bf16 (GPU tensor cores) ----------------------------------------
   The FIRST step toward matching PufferLib's CUDA: the dense forward run in bf16 on
   tensor cores (bf16 inputs, f32 accumulate) instead of f64 DGEMM. This is PufferLib's
   precision. NOT bit-exact with the f64 oracle — inputs are rounded to bf16 (8 mantissa
   bits) before the GEMM, so it matches only to ~1e-1 absolute / bf16 relative, exactly
   like PufferLib's own bf16 path. `verify-blas` reports the gap; `bench-blas` the speedup.

   Layout mirrors the f64 cuBLAS above: row-major Y[N×H]=X·Wᵀ ⇒ column-major
   Ycm[H×N] = op_T(Wcm[D×H]) · op_N(Xcm[D×N]); A=W (ld=D, OP_T), B=X (ld=D, OP_N),
   C=Y f32 (ld=H). Bias+ReLU are applied on the host in f64 (cheap, O(N·H)). */
static inline uint16_t f32_to_bf16(float f) {
  uint32_t x; memcpy(&x, &f, sizeof x);
  uint32_t is_nan = ((x & 0x7fffffffu) > 0x7f800000u);
  uint32_t rbias = ((x >> 16) & 1u) + 0x7fffu;      /* round to nearest even */
  uint16_t r = (uint16_t)((x + rbias) >> 16);
  if (is_nan) r |= 0x0040u;
  return r;
}

static cublasLtHandle_t g_lt = NULL;
static int g_lt_ok = 0;
static cublasLtHandle_t get_cublaslt(void) {
  if (!g_lt_ok) g_lt_ok = (cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS) ? 1 : -1;
  return g_lt_ok == 1 ? g_lt : NULL;
}

/* Persistent device buffers, grown on demand and reused across calls (a cudaMalloc per
   call costs ~100µs–1ms and swamped the GEMM). Freed at process exit. Single-threaded FFI.
   Slots 0–3: single-layer bf16 forward.  Slots 4–12: resident 2-layer MLP forward. */
#define NDBUF 16
static void* g_db[NDBUF];
static size_t g_dbsz[NDBUF];
static void* dev_buf(int i, size_t bytes) {
  if (g_dbsz[i] < bytes) {
    if (g_db[i]) cudaFree(g_db[i]);
    if (cudaMalloc(&g_db[i], bytes) != cudaSuccess) { g_db[i] = NULL; g_dbsz[i] = 0; return NULL; }
    g_dbsz[i] = bytes;
  }
  return g_db[i];
}

LEAN_EXPORT lean_obj_res lean_ffi_dense_forward_cublaslt_bf16(
    lean_obj_arg Xa, lean_obj_arg Wa, lean_obj_arg ba, size_t N, size_t D, size_t H) {
  const double* X = lean_float_array_cptr(Xa);
  const double* W = lean_float_array_cptr(Wa);
  const double* b = lean_float_array_cptr(ba);
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*H, N*H);
  double* Y = lean_float_array_cptr(Yo);
  cublasLtHandle_t lt = get_cublaslt();

  uint16_t *hX = (uint16_t*)malloc(sizeof(uint16_t)*N*D);
  uint16_t *hW = (uint16_t*)malloc(sizeof(uint16_t)*H*D);
  float    *hY = (float*)malloc(sizeof(float)*N*H);
  size_t wsSize = 32u<<20;
  void *dX = dev_buf(0, sizeof(uint16_t)*N*D), *dW = dev_buf(1, sizeof(uint16_t)*H*D);
  float *dY = (float*)dev_buf(2, sizeof(float)*N*H);
  void *ws = dev_buf(3, wsSize);
  int fail = (!lt || !hX || !hW || !hY || !dX || !dW || !dY || !ws);
  if (!fail) {
    for (size_t i = 0; i < N*D; i++) hX[i] = f32_to_bf16((float)X[i]);
    for (size_t i = 0; i < H*D; i++) hW[i] = f32_to_bf16((float)W[i]);
    cudaMemcpy(dX, hX, sizeof(uint16_t)*N*D, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW, sizeof(uint16_t)*H*D, cudaMemcpyHostToDevice);
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t Ad=NULL, Bd=NULL, Cd=NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, D, H, D);   /* W stored col-major [D,H] */
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, D, N, D);   /* X stored col-major [D,N] */
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F,  H, N, H);   /* Y col-major [H,N]=Yrm[N,H] */
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &wsSize, sizeof(wsSize));
    cublasLtMatmulHeuristicResult_t heur; int nAlgo = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &nAlgo);
    float al = 1.0f, be = 0.0f;
    cublasStatus_t st = cublasLtMatmul(lt, op, &al, dW, Ad, dX, Bd, &be, dY, Cd, dY, Cd,
                                       nAlgo > 0 ? &heur.algo : NULL, ws, wsSize, 0);
    cudaDeviceSynchronize();
    if (Ad) cublasLtMatrixLayoutDestroy(Ad); if (Bd) cublasLtMatrixLayoutDestroy(Bd);
    if (Cd) cublasLtMatrixLayoutDestroy(Cd); if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (op) cublasLtMatmulDescDestroy(op);
    if (st != CUBLAS_STATUS_SUCCESS) fail = 1;
    else {
      cudaMemcpy(hY, dY, sizeof(float)*N*H, cudaMemcpyDeviceToHost);
      for (size_t i = 0; i < N*H; i++) Y[i] = (double)hY[i];
    }
  }
  /* device buffers (dX/dW/dY/ws) are cached in g_db[] — not freed here */
  if (fail) {   /* GPU/bf16 unavailable → correct f64 CPU result so callers never break */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)N,(int)H,(int)D, 1.0, X,(int)D, W,(int)D, 0.0, Y,(int)H);
  }
  free(hX); free(hW); free(hY);
  bias_relu(Y, b, N, H);
  lean_dec(Xa); lean_dec(Wa); lean_dec(ba);
  return Yo;
}

/* Device-RESIDENT bf16 GEMM throughput (GF/s): times `reps` cublasLtMatmul iterations on
   buffers that never leave the GPU — no host↔device transfer, no f64→bf16 conversion. This is
   the tensor-core CEILING of our path (comparable to torch's bf16 GEMM); the gap to the
   end-to-end `..._cublaslt_bf16` forward above is exactly the per-call transfer tax we still pay
   because Lean holds activations in f64 on the host. Returns -1 if no usable device. */
LEAN_EXPORT double lean_ffi_bench_cublaslt_bf16_resident(size_t N, size_t D, size_t H, size_t reps) {
  cublasLtHandle_t lt = get_cublaslt();
  if (!lt) return -1.0;
  void *dX=NULL,*dW=NULL,*ws=NULL; float *dY=NULL; size_t wsSize = 32u<<20;
  if (cudaMalloc(&dX,sizeof(uint16_t)*N*D) || cudaMalloc(&dW,sizeof(uint16_t)*H*D)
      || cudaMalloc((void**)&dY,sizeof(float)*N*H) || cudaMalloc(&ws,wsSize)) {
    if(dX)cudaFree(dX); if(dW)cudaFree(dW); if(dY)cudaFree(dY); if(ws)cudaFree(ws); return -1.0;
  }
  cudaMemset(dX,0,sizeof(uint16_t)*N*D); cudaMemset(dW,0,sizeof(uint16_t)*H*D);
  cublasLtMatmulDesc_t op=NULL; cublasLtMatrixLayout_t Ad=NULL,Bd=NULL,Cd=NULL; cublasLtMatmulPreference_t pref=NULL;
  cublasOperation_t ta=CUBLAS_OP_T, tb=CUBLAS_OP_N;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
  cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, D, H, D);
  cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, D, N, D);
  cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, H, N, H);
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof wsSize);
  cublasLtMatmulHeuristicResult_t heur; int nAlgo=0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &nAlgo);
  const cublasLtMatmulAlgo_t* algo = nAlgo>0 ? &heur.algo : NULL;
  float al=1.0f, be=0.0f;
  for (int i=0;i<5;i++) cublasLtMatmul(lt,op,&al,dW,Ad,dX,Bd,&be,dY,Cd,dY,Cd,algo,ws,wsSize,0);
  cudaDeviceSynchronize();
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0,0);
  for (size_t i=0;i<reps;i++) cublasLtMatmul(lt,op,&al,dW,Ad,dX,Bd,&be,dY,Cd,dY,Cd,algo,ws,wsSize,0);
  cudaEventRecord(e1,0); cudaEventSynchronize(e1);
  float ms=0.0f; cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatmulDescDestroy(op);
  cudaFree(dX); cudaFree(dW); cudaFree(dY); cudaFree(ws);
  double sec = ((double)ms/1000.0) / (double)reps;
  return 2.0*(double)N*(double)D*(double)H / (sec*1e9);
}

/* --- On-device bf16 residency: a whole 2-layer MLP forward on the GPU ----------
   One tensor-core "layer": Dout[m,n] = epi(op_T(A[k,m]) · op_N(B[k,n]) + bias[m]), with the
   bias+ReLU fused into the GEMM via a cuBLASLt epilogue — so the layer's OUTPUT stays on the GPU
   and feeds the next GEMM directly, with NO host round-trip between layers. Storage is f32 and
   the compute type is CUBLAS_COMPUTE_32F_FAST_16BF: inputs are rounded to bf16 for the tensor-core
   multiply (PufferLib's precision, ~1e-1 vs the f64 oracle) with f32 accumulate. f32 storage is
   what makes the fused bias+ReLU epilogue supported (bf16 output + epilogue returns NOT_SUPPORTED).
   Returns the matmul status. */
static cublasStatus_t lt_layer(cublasLtHandle_t lt, size_t m, size_t n, size_t k,
    const float* A, const float* B, float* Dout,
    const float* bias, int relu, void* ws, size_t wsSize) {
  cublasLtMatmulDesc_t op = NULL;
  cublasLtMatrixLayout_t Ad=NULL, Bd=NULL, Cd=NULL;
  cublasLtMatmulPreference_t pref = NULL;
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
  if (bias) {
    cublasLtEpilogue_t epi = relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
  }
  cublasLtMatrixLayoutCreate(&Ad, CUDA_R_32F, k, m, k);
  cublasLtMatrixLayoutCreate(&Bd, CUDA_R_32F, k, n, k);
  cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, n, m);
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof wsSize);
  cublasLtMatmulHeuristicResult_t heur; int nAlgo = 0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &nAlgo);
  float al = 1.0f, be = 0.0f;
  cublasStatus_t st = cublasLtMatmul(lt, op, &al, A, Ad, B, Bd, &be, Dout, Cd, Dout, Cd,
                                     nAlgo > 0 ? &heur.algo : NULL, ws, wsSize, 0);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatmulDescDestroy(op);
  return st;
}

/* Like lt_layer but with bf16 (CUDA_R_16BF) INPUTS A,B and f32 output — used for the resident
   policy's layer 1 so the obs (and W1) travel to the GPU as bf16, HALVING the per-timestep obs
   upload bytes (2·N·D vs 4·N·D). cuBLASLt requires A,B share the input type, so W1 is bf16 too
   (uploaded once per optimizer step). bf16-in + f32-out + fused RELU_BIAS epilogue IS supported
   (unlike bf16-OUTPUT + epilogue). Compute is CUBLAS_COMPUTE_32F (f32 accumulate). */
static cublasStatus_t lt_layer_bf16in(cublasLtHandle_t lt, size_t m, size_t n, size_t k,
    const void* A, const void* B, float* Dout,
    const float* bias, int relu, void* ws, size_t wsSize) {
  cublasLtMatmulDesc_t op = NULL;
  cublasLtMatrixLayout_t Ad=NULL, Bd=NULL, Cd=NULL;
  cublasLtMatmulPreference_t pref = NULL;
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
  if (bias) {
    cublasLtEpilogue_t epi = relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias);
  }
  cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16BF, k, m, k);
  cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16BF, k, n, k);
  cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F,  m, n, m);
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof wsSize);
  cublasLtMatmulHeuristicResult_t heur; int nAlgo = 0;
  cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &nAlgo);
  float al = 1.0f, be = 0.0f;
  cublasStatus_t st = cublasLtMatmul(lt, op, &al, A, Ad, B, Bd, &be, Dout, Cd, Dout, Cd,
                                     nAlgo > 0 ? &heur.algo : NULL, ws, wsSize, 0);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatmulDescDestroy(op);
  return st;
}

/* Batched 2-layer MLP forward, bf16 tensor cores (FAST_16BF), intermediate RESIDENT on the GPU.
   Same contract as lean_ffi_mlp_forward_batch_blas: params W1[H·D],b1[H],W2[O·H],b2[O] (O=A+1),
   Xb[N·D] → Yb[N·O] = (relu(Xb·W1ᵀ+b1))·W2ᵀ+b2, row-major. The hidden activation H1 never leaves
   the device (GEMM1's f32 output feeds GEMM2 directly; bias+ReLU fused into GEMM1's epilogue). Only
   ONE upload (params+Xb) and ONE download (Yb) per call, vs a host round-trip per layer. bf16
   precision ⇒ matches the f64 oracle to ~1e-1; CPU-BLAS f64 fallback if no device. */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_forward_batch_cublaslt_bf16(
    lean_obj_arg pa, lean_obj_arg Xa, size_t N, size_t D, size_t H, size_t O) {
  const double* pp = lean_float_array_cptr(pa);
  const double* X  = lean_float_array_cptr(Xa);
  const double* W1 = pp; const double* b1 = W1 + H*D; const double* W2 = b1 + H; const double* b2 = W2 + O*H;
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*O, N*O);
  double* Y = lean_float_array_cptr(Yo);
  cublasLtHandle_t lt = get_cublaslt();
  size_t wsSize = 32u<<20;
  float *hW1=(float*)malloc(4*H*D), *hX=(float*)malloc(4*N*D), *hW2=(float*)malloc(4*O*H);
  float *hb1=(float*)malloc(4*H), *hb2=(float*)malloc(4*O), *hY=(float*)malloc(4*N*O);
  float *dW1=(float*)dev_buf(4,4*H*D), *dX=(float*)dev_buf(5,4*N*D), *dW2=(float*)dev_buf(6,4*O*H);
  float *db1=(float*)dev_buf(7,4*H), *db2=(float*)dev_buf(8,4*O), *dH1=(float*)dev_buf(9,4*H*N);
  float *dOut=(float*)dev_buf(10,4*O*N); void *ws=dev_buf(12,wsSize);
  int fail = (!lt || !hW1||!hX||!hW2||!hb1||!hb2||!hY || !dW1||!dX||!dW2||!db1||!db2||!dH1||!dOut||!ws);
  if (!fail) {
    for (size_t i=0;i<H*D;i++) hW1[i]=(float)W1[i];
    for (size_t i=0;i<N*D;i++) hX[i] =(float)X[i];
    for (size_t i=0;i<O*H;i++) hW2[i]=(float)W2[i];
    for (size_t i=0;i<H;i++) hb1[i]=(float)b1[i];
    for (size_t i=0;i<O;i++) hb2[i]=(float)b2[i];
    cudaMemcpy(dW1,hW1,4*H*D,cudaMemcpyHostToDevice); cudaMemcpy(dX,hX,4*N*D,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,4*O*H,cudaMemcpyHostToDevice);
    cudaMemcpy(db1,hb1,4*H,cudaMemcpyHostToDevice);   cudaMemcpy(db2,hb2,4*O,cudaMemcpyHostToDevice);
    /* layer1: H1[H,N]=relu(W1·Xᵀ + b1) — f32 output, stays resident */
    cublasStatus_t s1 = lt_layer(lt, H, N, D, dW1, dX, dH1, db1, 1, ws, wsSize);
    /* layer2: Out[O,N]=W2·H1ᵀ + b2 — f32 output, downloaded to the host in f64 */
    cublasStatus_t s2 = lt_layer(lt, O, N, H, dW2, dH1, dOut, db2, 0, ws, wsSize);
    cudaDeviceSynchronize();
    if (s1 != CUBLAS_STATUS_SUCCESS || s2 != CUBLAS_STATUS_SUCCESS) fail = 1;
    else { cudaMemcpy(hY,dOut,4*N*O,cudaMemcpyDeviceToHost); for (size_t i=0;i<N*O;i++) Y[i]=(double)hY[i]; }
  }
  if (fail) {   /* correct f64 CPU result so callers never break */
    double* Z=(double*)malloc(sizeof(double)*N*H);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)N,(int)H,(int)D,1.0,X,(int)D,W1,(int)D,0.0,Z,(int)H);
    for (size_t n=0;n<N;n++) for (size_t j=0;j<H;j++){double v=Z[n*H+j]+b1[j];Z[n*H+j]=v>0.0?v:0.0;}
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,(int)N,(int)O,(int)H,1.0,Z,(int)H,W2,(int)H,0.0,Y,(int)O);
    for (size_t n=0;n<N;n++) for (size_t kk=0;kk<O;kk++) Y[n*O+kk]+=b2[kk];
    free(Z);
  }
  free(hW1);free(hX);free(hW2);free(hb1);free(hb2);free(hY);
  lean_dec(pa); lean_dec(Xa);
  return Yo;
}

/* Device-RESIDENT 2-layer MLP forward throughput (GF/s): weights, input and activations all stay
   on the GPU across `reps` forwards — no per-call transfer. The end-to-end ceiling of the resident
   forward (the rollout pattern once obs live on-device). Returns -1 if no device or a matmul fails. */
LEAN_EXPORT double lean_ffi_bench_mlp2_bf16_resident(size_t N, size_t D, size_t H, size_t O, size_t reps) {
  cublasLtHandle_t lt = get_cublaslt();
  if (!lt) return -1.0;
  size_t wsSize = 32u<<20;
  float *dW1=NULL,*dX=NULL,*dW2=NULL,*db1=NULL,*db2=NULL,*dH1=NULL,*dOut=NULL; void* ws=NULL;
  if (cudaMalloc((void**)&dW1,4*H*D)||cudaMalloc((void**)&dX,4*N*D)||cudaMalloc((void**)&dW2,4*O*H)||cudaMalloc((void**)&db1,4*H)
      ||cudaMalloc((void**)&db2,4*O)||cudaMalloc((void**)&dH1,4*H*N)||cudaMalloc((void**)&dOut,4*O*N)||cudaMalloc(&ws,wsSize)) {
    if(dW1)cudaFree(dW1);if(dX)cudaFree(dX);if(dW2)cudaFree(dW2);if(db1)cudaFree(db1);
    if(db2)cudaFree(db2);if(dH1)cudaFree(dH1);if(dOut)cudaFree(dOut);if(ws)cudaFree(ws); return -1.0;
  }
  cudaMemset(dW1,0,4*H*D);cudaMemset(dX,0,4*N*D);cudaMemset(dW2,0,4*O*H);cudaMemset(db1,0,4*H);cudaMemset(db2,0,4*O);
  int bad = 0;
  for (int i=0;i<5;i++) {
    bad |= (lt_layer(lt,H,N,D,dW1,dX,dH1,db1,1,ws,wsSize) != CUBLAS_STATUS_SUCCESS);
    bad |= (lt_layer(lt,O,N,H,dW2,dH1,dOut,db2,0,ws,wsSize) != CUBLAS_STATUS_SUCCESS);
  }
  double gfps = -1.0;
  if (!bad) {
    cudaDeviceSynchronize();
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0,0);
    for (size_t i=0;i<reps;i++) {
      lt_layer(lt,H,N,D,dW1,dX,dH1,db1,1,ws,wsSize);
      lt_layer(lt,O,N,H,dW2,dH1,dOut,db2,0,ws,wsSize);
    }
    cudaEventRecord(e1,0); cudaEventSynchronize(e1);
    float ms=0.0f; cudaEventElapsedTime(&ms,e0,e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
    double sec = ((double)ms/1000.0)/(double)reps;
    gfps = (2.0*(double)N*(double)D*(double)H + 2.0*(double)N*(double)H*(double)O)/(sec*1e9);
  }
  cudaFree(dW1);cudaFree(dX);cudaFree(dW2);cudaFree(db1);cudaFree(db2);cudaFree(dH1);cudaFree(dOut);cudaFree(ws);
  return gfps;
}

/* --- Persistent resident policy: weights uploaded ONCE, forwarded many times ----
   A rollout runs the SAME policy for many timesteps, so uploading the (constant) weights every
   forward is pure waste. `_load` uploads W1,b1,W2,b2 to the GPU once and returns an opaque handle;
   `_forward` then streams only the fresh obs Xb up and Yb down (H1 stays resident), at the
   weights-resident throughput; `_free` releases it. This is the persistent-policy architecture
   that turns the weights-resident microbenchmark into real, callable rollout throughput. */
typedef struct {
  cublasLtHandle_t lt;
  size_t D, H, O, capN;
  uint16_t *dW1b;                    /* bf16 W1 (device; uploaded once, updated in place) */
  float *db1, *dW2, *db2;            /* f32 device (b1, W2, b2)                            */
  uint16_t *dXb;                     /* bf16 obs (device), grown to capN rows             */
  float *dH1, *dOut;                 /* f32 activation buffers (device)                   */
  uint16_t *hXb;                     /* bf16 obs staging (PINNED host), capN rows          */
  float *hY;                         /* f32 output staging (PINNED host), capN rows        */
  float *hW1, *hb1, *hW2, *hb2;      /* host f32 weight copies (for the CPU fallback)      */
  void *ws; size_t wsSize;
} MlpPolicy;

/* Upload weights once; returns an opaque handle (as size_t), or 0 if no usable device. */
LEAN_EXPORT size_t lean_ffi_mlp_policy_load(lean_obj_arg pa, size_t D, size_t H, size_t O) {
  cublasLtHandle_t lt = get_cublaslt();
  const double* pp = lean_float_array_cptr(pa);
  const double *W1=pp,*b1=W1+H*D,*W2=b1+H,*b2=W2+O*H;
  MlpPolicy* p = (MlpPolicy*)calloc(1, sizeof(MlpPolicy));
  size_t ret = 0;
  if (lt && p) {
    p->lt=lt; p->D=D; p->H=H; p->O=O; p->wsSize=32u<<20; p->capN=0;
    p->hW1=(float*)malloc(4*H*D); p->hb1=(float*)malloc(4*H);
    p->hW2=(float*)malloc(4*O*H); p->hb2=(float*)malloc(4*O);
    uint16_t* tW1 = (uint16_t*)malloc(2*H*D);   /* bf16 W1 staging */
    if (p->hW1&&p->hb1&&p->hW2&&p->hb2&&tW1) {
      for (size_t i=0;i<H*D;i++) { p->hW1[i]=(float)W1[i]; tW1[i]=f32_to_bf16(p->hW1[i]); }
      for (size_t i=0;i<H;i++)   p->hb1[i]=(float)b1[i];
      for (size_t i=0;i<O*H;i++) p->hW2[i]=(float)W2[i];
      for (size_t i=0;i<O;i++)   p->hb2[i]=(float)b2[i];
      if (!cudaMalloc((void**)&p->dW1b,2*H*D) && !cudaMalloc((void**)&p->db1,4*H)
          && !cudaMalloc((void**)&p->dW2,4*O*H) && !cudaMalloc((void**)&p->db2,4*O)
          && !cudaMalloc(&p->ws,p->wsSize)) {
        cudaMemcpy(p->dW1b,tW1,2*H*D,cudaMemcpyHostToDevice);
        cudaMemcpy(p->db1,p->hb1,4*H,cudaMemcpyHostToDevice);
        cudaMemcpy(p->dW2,p->hW2,4*O*H,cudaMemcpyHostToDevice);
        cudaMemcpy(p->db2,p->hb2,4*O,cudaMemcpyHostToDevice);
        ret = (size_t)p;
      }
    }
    free(tW1);
  }
  if (!ret && p) {
    if(p->dW1b)cudaFree(p->dW1b);if(p->db1)cudaFree(p->db1);if(p->dW2)cudaFree(p->dW2);if(p->db2)cudaFree(p->db2);if(p->ws)cudaFree(p->ws);
    free(p->hW1);free(p->hb1);free(p->hW2);free(p->hb2); free(p);
  }
  lean_dec(pa);
  return ret;
}

/* Re-upload weights into an existing handle (dims unchanged), reusing the device buffers — called
   after each optimizer step so one resident policy serves the whole training run. Returns 1/0. */
LEAN_EXPORT uint8_t lean_ffi_mlp_policy_update(size_t handle, lean_obj_arg pa) {
  MlpPolicy* p = (MlpPolicy*)handle;
  if (!p) { lean_dec(pa); return 0; }
  const double* pp = lean_float_array_cptr(pa);
  size_t D=p->D, H=p->H, O=p->O;
  const double *W1=pp,*b1=W1+H*D,*W2=b1+H,*b2=W2+O*H;
  uint16_t* tW1 = (uint16_t*)malloc(2*H*D);
  if (!tW1) { lean_dec(pa); return 0; }
  for (size_t i=0;i<H*D;i++) { p->hW1[i]=(float)W1[i]; tW1[i]=f32_to_bf16(p->hW1[i]); }  /* f32 for fallback, bf16 for GPU */
  for (size_t i=0;i<H;i++)   p->hb1[i]=(float)b1[i];
  for (size_t i=0;i<O*H;i++) p->hW2[i]=(float)W2[i];
  for (size_t i=0;i<O;i++)   p->hb2[i]=(float)b2[i];
  cudaMemcpy(p->dW1b,tW1,2*H*D,cudaMemcpyHostToDevice);
  cudaMemcpy(p->db1,p->hb1,4*H,cudaMemcpyHostToDevice);
  cudaMemcpy(p->dW2,p->hW2,4*O*H,cudaMemcpyHostToDevice);
  cudaMemcpy(p->db2,p->hb2,4*O,cudaMemcpyHostToDevice);
  free(tW1);
  lean_dec(pa);
  return 1;
}

static int policy_ensure(MlpPolicy* p, size_t N) {   /* grow the device + PINNED-host buffers to N rows */
  if (p->capN >= N) return 0;
  if (p->dXb) cudaFree(p->dXb); if (p->dH1) cudaFree(p->dH1); if (p->dOut) cudaFree(p->dOut);
  if (p->hXb) cudaFreeHost(p->hXb); if (p->hY) cudaFreeHost(p->hY);
  p->dXb=NULL; p->dH1=p->dOut=NULL; p->hXb=NULL; p->hY=NULL; p->capN=0;
  if (cudaMalloc((void**)&p->dXb,2*N*p->D)||cudaMalloc((void**)&p->dH1,4*p->H*N)||cudaMalloc((void**)&p->dOut,4*p->O*N))
    return -1;
  /* pinned (cacheable, NOT write-combined) staging so H2D/D2H run at full PCIe bandwidth */
  if (cudaHostAlloc((void**)&p->hXb,2*N*p->D,cudaHostAllocDefault)||cudaHostAlloc((void**)&p->hY,4*N*p->O,cudaHostAllocDefault))
    return -1;
  p->capN=N; return 0;
}

/* Forward N observations through the resident policy: only Xb uploads and Yb downloads; the
   weights and the hidden activation H1 stay on the GPU. Returns Yb[N·O] (f64). */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_policy_forward(size_t handle, lean_obj_arg Xa, size_t N) {
  MlpPolicy* p = (MlpPolicy*)handle;
  const double* X = lean_float_array_cptr(Xa);
  size_t D = p?p->D:1, H = p?p->H:1, O = p?p->O:1;
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*O, N*O);
  double* Y = lean_float_array_cptr(Yo);
  int fail = (!p || policy_ensure(p, N) != 0);
  if (!fail) {
    /* obs → bf16 straight into the pinned staging buffer; upload HALF the bytes (2·N·D) */
    for (size_t i=0;i<N*D;i++) p->hXb[i]=f32_to_bf16((float)X[i]);
    cudaMemcpy(p->dXb,p->hXb,2*N*D,cudaMemcpyHostToDevice);
    cublasStatus_t s1 = lt_layer_bf16in(p->lt,H,N,D,p->dW1b,p->dXb,p->dH1,p->db1,1,p->ws,p->wsSize);  /* bf16 in */
    cublasStatus_t s2 = lt_layer(p->lt,O,N,H,p->dW2,p->dH1,p->dOut,p->db2,0,p->ws,p->wsSize);           /* f32 H1 */
    cudaDeviceSynchronize();
    if (s1!=CUBLAS_STATUS_SUCCESS || s2!=CUBLAS_STATUS_SUCCESS) fail = 1;
    else { cudaMemcpy(p->hY,p->dOut,4*N*O,cudaMemcpyDeviceToHost); for (size_t i=0;i<N*O;i++) Y[i]=(double)p->hY[i]; }
  }
  if (fail) {   /* CPU fallback from the host weight copies (or zeros if the handle is invalid) */
    if (p && p->hW1) {
      double* Z=(double*)malloc(sizeof(double)*N*H);
      for (size_t n=0;n<N;n++) for (size_t j=0;j<H;j++) {
        double acc=p->hb1[j]; const double* xr=X+n*D; const float* wr=p->hW1+j*D;
        for (size_t d=0;d<D;d++) acc+=xr[d]*wr[d]; Z[n*H+j]=acc>0.0?acc:0.0;
      }
      for (size_t n=0;n<N;n++) for (size_t k=0;k<O;k++) {
        double acc=p->hb2[k]; const double* zr=Z+n*H; const float* wr=p->hW2+k*H;
        for (size_t j=0;j<H;j++) acc+=zr[j]*wr[j]; Y[n*O+k]=acc;
      }
      free(Z);
    } else for (size_t i=0;i<N*O;i++) Y[i]=0.0;
  }
  lean_dec(Xa);
  return Yo;
}

/* Release the resident policy. Returns 1 on success, 0 if the handle was already invalid. */
LEAN_EXPORT uint8_t lean_ffi_mlp_policy_free(size_t handle) {
  MlpPolicy* p = (MlpPolicy*)handle;
  if (!p) return 0;
  if(p->dW1b)cudaFree(p->dW1b);if(p->db1)cudaFree(p->db1);if(p->dW2)cudaFree(p->dW2);if(p->db2)cudaFree(p->db2);
  if(p->dXb)cudaFree(p->dXb);if(p->dH1)cudaFree(p->dH1);if(p->dOut)cudaFree(p->dOut);if(p->ws)cudaFree(p->ws);
  if(p->hXb)cudaFreeHost(p->hXb);if(p->hY)cudaFreeHost(p->hY);
  free(p->hW1);free(p->hb1);free(p->hW2);free(p->hb2); free(p);
  return 1;
}

/* Throughput (GF/s) of the persistent-policy pattern the API above implements: weights uploaded
   ONCE, then per forward only the fresh obs uploads and the output downloads (H1 stays on-GPU).
   The C-level measurement (accurate cudaEvent timing, no Lean-side per-call cost); the load/forward
   API delivers the same pattern, correctness-checked in `verify-blas-fwd`. -1 on device failure. */
LEAN_EXPORT double lean_ffi_bench_mlp2_bf16_wres(size_t N, size_t D, size_t H, size_t O, size_t reps) {
  size_t hnd = 0;
  {
    /* build a zero "params" on the host and load it as a policy */
    size_t P = H*D + H + O*H + O;
    lean_object* pa = lean_alloc_sarray(sizeof(double), P, P);
    double* pp = lean_float_array_cptr(pa);
    for (size_t i=0;i<P;i++) pp[i]=0.0;
    hnd = lean_ffi_mlp_policy_load(pa, D, H, O);   /* consumes pa */
  }
  if (!hnd) return -1.0;
  MlpPolicy* p = (MlpPolicy*)hnd;
  if (policy_ensure(p, N) != 0) { lean_ffi_mlp_policy_free(hnd); return -1.0; }
  memset(p->hXb, 0, 2*N*D);                        /* bf16 obs staging (pinned, from policy_ensure) */
  int bad = 0;
  for (int i=0;i<5;i++) {                          /* warmup */
    cudaMemcpy(p->dXb,p->hXb,2*N*D,cudaMemcpyHostToDevice);
    bad |= (lt_layer_bf16in(p->lt,H,N,D,p->dW1b,p->dXb,p->dH1,p->db1,1,p->ws,p->wsSize) != CUBLAS_STATUS_SUCCESS);
    bad |= (lt_layer(p->lt,O,N,H,p->dW2,p->dH1,p->dOut,p->db2,0,p->ws,p->wsSize) != CUBLAS_STATUS_SUCCESS);
    cudaMemcpy(p->hY,p->dOut,4*N*O,cudaMemcpyDeviceToHost);
  }
  double gfps = -1.0;
  if (!bad) {
    cudaDeviceSynchronize();
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0,0);
    for (size_t i=0;i<reps;i++) {                  /* per iter: only bf16 obs up + out down */
      cudaMemcpy(p->dXb,p->hXb,2*N*D,cudaMemcpyHostToDevice);
      lt_layer_bf16in(p->lt,H,N,D,p->dW1b,p->dXb,p->dH1,p->db1,1,p->ws,p->wsSize);
      lt_layer(p->lt,O,N,H,p->dW2,p->dH1,p->dOut,p->db2,0,p->ws,p->wsSize);
      cudaMemcpy(p->hY,p->dOut,4*N*O,cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1,0); cudaEventSynchronize(e1);
    float ms=0.0f; cudaEventElapsedTime(&ms,e0,e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
    double sec = ((double)ms/1000.0)/(double)reps;
    gfps = (2.0*(double)N*(double)D*(double)H + 2.0*(double)N*(double)H*(double)O)/(sec*1e9);
  }
  lean_ffi_mlp_policy_free(hnd);
  return gfps;
}

/* --- Batched 2-layer MLP forward (the vectorized-rollout hot path) ------------
   params flat layout W1[H·D], b1[H], W2[O·H], b2[O] (O = A+1), matching `flattenMLP`.
   Xb is the batch of `N` observations, row-major [N·D]. Returns Yb[N·O] row-major:
   Yb = (relu(Xb·W1ᵀ + b1)) · W2ᵀ + b2 — the whole minibatch/timestep-batch forward.
   `_ref` is the scalar right-fold twin (per row == `lean_ffi_mlp_forward`, bit-exact vs
   `forwardAll`); `_blas` uses two `cblas_dgemm`s (tolerance, not bit-exact). */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_forward_batch_ref(
    lean_obj_arg pa, lean_obj_arg Xa, size_t N, size_t D, size_t H, size_t O) {
  const double* pp = lean_float_array_cptr(pa);
  const double* X = lean_float_array_cptr(Xa);
  const double* W1 = pp; const double* b1 = W1 + H*D; const double* W2 = b1 + H; const double* b2 = W2 + O*H;
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*O, N*O);
  double* Y = lean_float_array_cptr(Yo);
  double* h = (double*)malloc(sizeof(double)*H);
  for (size_t n = 0; n < N; n++) {
    const double* x = X + n*D;
    for (size_t j = 0; j < H; j++) {
      double acc = 0.0; const double* w = W1 + j*D;
      for (size_t d = D; d-- > 0;) acc = w[d]*x[d] + acc;   /* right-fold (== mlp_forward) */
      double z = b1[j] + acc; h[j] = z > 0.0 ? z : 0.0;
    }
    double* y = Y + n*O;
    for (size_t k = 0; k < O; k++) {
      double acc = 0.0; const double* w = W2 + k*H;
      for (size_t j = H; j-- > 0;) acc = w[j]*h[j] + acc;
      y[k] = b2[k] + acc;
    }
  }
  free(h); lean_dec(pa); lean_dec(Xa);
  return Yo;
}

LEAN_EXPORT lean_obj_res lean_ffi_mlp_forward_batch_blas(
    lean_obj_arg pa, lean_obj_arg Xa, size_t N, size_t D, size_t H, size_t O) {
  const double* pp = lean_float_array_cptr(pa);
  const double* X = lean_float_array_cptr(Xa);
  const double* W1 = pp; const double* b1 = W1 + H*D; const double* W2 = b1 + H; const double* b2 = W2 + O*H;
  lean_object* Yo = lean_alloc_sarray(sizeof(double), N*O, N*O);
  double* Y = lean_float_array_cptr(Yo);
  double* Z = (double*)malloc(sizeof(double)*N*H);
  /* Z[N×H] = Xb[N×D]·W1[H×D]ᵀ */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (int)N, (int)H, (int)D, 1.0, X, (int)D, W1, (int)D, 0.0, Z, (int)H);
  for (size_t n = 0; n < N; n++)
    for (size_t j = 0; j < H; j++) { double v = Z[n*H+j] + b1[j]; Z[n*H+j] = v > 0.0 ? v : 0.0; }
  /* Y[N×O] = H1[N×H]·W2[O×H]ᵀ */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (int)N, (int)O, (int)H, 1.0, Z, (int)H, W2, (int)H, 0.0, Y, (int)O);
  for (size_t n = 0; n < N; n++)
    for (size_t k = 0; k < O; k++) Y[n*O+k] += b2[k];
  free(Z); lean_dec(pa); lean_dec(Xa);
  return Yo;
}

/* --- Batched MLP+PPO gradient via BLAS (the training hot path) ----------------
   Same contract as `lean_ffi_mlp_ppo_grad_batch` (pufferffi.c): sums dObj/dparams over
   the `N`-transition minibatch into a flat FloatArray[P] (W1[H·D],b1[H],W2[O·H],b2[O],
   O=A+1). The forward (Z=Xb·W1ᵀ, Out=H1·W2ᵀ) and the backward (dW2=dOutᵀ·H1, dH1=dOut·W2,
   dW1=dZ1ᵀ·Xb) are all `cblas_dgemm`s; only the per-row PPO objective backward → dOut is
   scalar (softmax/value/entropy, identical to the scalar kernel). Blocked reductions ⇒
   NOT bit-identical to the Lean oracle, matches to tolerance (~1e-11). */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_ppo_grad_batch_blas(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa,
    size_t N, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1;
  size_t P = H*D + H + O*H + O;
  const double* pp = lean_float_array_cptr(pa);
  const double* W1 = pp; const double* b1 = W1 + H*D; const double* W2 = b1 + H; const double* b2 = W2 + O*H;
  const double* Xb = lean_float_array_cptr(obsBa);
  const double* actA = lean_float_array_cptr(actsa); const double* advA = lean_float_array_cptr(advsa);
  const double* retA = lean_float_array_cptr(retsa); const double* oldA = lean_float_array_cptr(oldlpsa);
  lean_object* go = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(go);
  for (size_t t = 0; t < P; t++) g[t] = 0.0;
  double* gW1 = g; double* gb1 = gW1 + H*D; double* gW2 = gb1 + H; double* gb2 = gW2 + O*H;
  double* H1  = (double*)malloc(sizeof(double)*N*H);
  double* Out = (double*)malloc(sizeof(double)*N*O);
  double* dOut= (double*)malloc(sizeof(double)*N*O);
  double* dH1 = (double*)malloc(sizeof(double)*N*H);
  double* pk  = (double*)malloc(sizeof(double)*A);
  /* forward: H1 = relu(Xb·W1ᵀ + b1) */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)H,(int)D, 1.0, Xb,(int)D, W1,(int)D, 0.0, H1,(int)H);
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < H; j++) { double v = H1[n*H+j]+b1[j]; H1[n*H+j] = v>0.0?v:0.0; }
  /* forward: Out = H1·W2ᵀ + b2 */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)O,(int)H, 1.0, H1,(int)H, W2,(int)H, 0.0, Out,(int)O);
  for (size_t n = 0; n < N; n++) for (size_t k = 0; k < O; k++) Out[n*O+k] += b2[k];
  /* per-row PPO objective backward → dOut (scalar; identical to the fused kernel) */
  for (size_t n = 0; n < N; n++) {
    const double* out = Out + n*O; double* dout = dOut + n*O;
    size_t a = (size_t)actA[n]; double adv = advA[n], ret = retA[n], oldLogp = oldA[n];
    double sumexp = 0.0; for (size_t k = 0; k < A; k++) sumexp += exp(out[k]); double lse = log(sumexp);
    double pout = 0.0; for (size_t k = 0; k < A; k++) { pk[k] = exp(out[k]-lse); pout += pk[k]*out[k]; }
    double logpA = out[a]-lse; double ratio = exp(logpA-oldLogp); double lo = 1.0-clipEps, hi = 1.0+clipEps;
    double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio); double surr1 = adv*ratio, surr2 = adv*ratioC;
    double dPol; if (surr1 <= surr2) dPol = adv*ratio; else { double cg = (lo<ratio && ratio<hi)?1.0:0.0; dPol = adv*cg*ratio; }
    for (size_t k = 0; k < A; k++) { double dp = dPol*(((k==a)?1.0:0.0)-pk[k]); double de = entCoef*pk[k]*(pout-out[k]); dout[k] = dp+de; }
    dout[A] = -vfCoef*(out[A]-ret);
  }
  /* backward: db2, dW2 = dOutᵀ·H1, dH1 = dOut·W2, dZ1 = dH1⊙relu', db1, dW1 = dZ1ᵀ·Xb */
  for (size_t n = 0; n < N; n++) for (size_t k = 0; k < O; k++) gb2[k] += dOut[n*O+k];
  cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (int)O,(int)H,(int)N, 1.0, dOut,(int)O, H1,(int)H, 0.0, gW2,(int)H);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N,(int)H,(int)O, 1.0, dOut,(int)O, W2,(int)H, 0.0, dH1,(int)H);
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < H; j++) if (!(H1[n*H+j] > 0.0)) dH1[n*H+j] = 0.0;
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < H; j++) gb1[j] += dH1[n*H+j];
  cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (int)H,(int)D,(int)N, 1.0, dH1,(int)H, Xb,(int)D, 0.0, gW1,(int)D);
  free(H1); free(Out); free(dOut); free(dH1); free(pk);
  lean_dec(pa); lean_dec(obsBa); lean_dec(actsa); lean_dec(advsa); lean_dec(retsa); lean_dec(oldlpsa);
  return go;
}

/* --- Batched CNN+PPO gradient via BLAS (im2col + GEMMs) -----------------------
   Same contract as `lean_ffi_cnn_ppo_grad_batch` (pufferffi.c): sums dObj/dparams over
   the N-transition minibatch into a flat FloatArray[P] with layout convW[nF·Ckk],
   convB[nF], W1[hidden·flatDim], b1[hidden], W2[O·hidden], b2[O] (Ckk=C·k·k,
   flatDim=nF·oHoW, oHoW=oH·oW, O=A+1). The conv is done as im2col + GEMM; the dense
   layers + backward are GEMMs; only the per-row PPO objective→dOut stays scalar.

   LAYOUT NOTE: im2col/conv are PIXEL-major per sample (row r=(n,oy,ox), col f), but the
   dense `feat` is FILTER-major (fi = f·oHoW + p), matching the scalar kernel. So there is
   a per-sample transpose between featPix[(N·oHoW)×nF] and Feat[N×flatDim] (and back for
   dFeat). Blocked reductions ⇒ matches the scalar/Lean oracle to tolerance, not bit-exact. */
static inline size_t blas_patch_obs(size_t idx, size_t C, size_t inH, size_t inW,
                                    size_t k, size_t s, size_t oy, size_t ox) {
  size_t c = idx/(k*k); size_t rem = idx%(k*k); size_t ky = rem/k; size_t kx = rem%k;
  return (c*inH + (oy*s+ky))*inW + (ox*s+kx);
}

LEAN_EXPORT lean_obj_res lean_ffi_cnn_ppo_grad_batch_blas(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa,
    lean_obj_arg advsa, lean_obj_arg retsa, lean_obj_arg oldlpsa,
    size_t N, size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t A,
    double vfCoef, double entCoef, double clipEps, size_t nScalar) {
  size_t O = A + 1;
  size_t oH = (inH-k)/s + 1, oW = (inW-k)/s + 1, oHoW = oH*oW;
  /* obs row = [nScalar passthrough | C·inH·inW image]; conv reads the image, scalars append to Feat. */
  size_t Ckk = C*k*k, flatDim = nF*oHoW + nScalar, R = N*oHoW, imgSz = C*inH*inW, inSz = nScalar + imgSz;
  size_t P = nF*Ckk + nF + hidden*flatDim + hidden + O*hidden + O;
  const double* pp = lean_float_array_cptr(pa);
  const double* convW = pp; const double* convB = convW + nF*Ckk;
  const double* W1 = convB + nF; const double* b1 = W1 + hidden*flatDim;
  const double* W2 = b1 + hidden; const double* b2 = W2 + O*hidden;
  const double* obs = lean_float_array_cptr(obsBa);
  const double* actA = lean_float_array_cptr(actsa); const double* advA = lean_float_array_cptr(advsa);
  const double* retA = lean_float_array_cptr(retsa); const double* oldA = lean_float_array_cptr(oldlpsa);
  lean_object* go = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(go);
  for (size_t t = 0; t < P; t++) g[t] = 0.0;
  double* gConvW = g; double* gConvB = gConvW + nF*Ckk; double* gW1 = gConvB + nF;
  double* gb1 = gW1 + hidden*flatDim; double* gW2 = gb1 + hidden; double* gb2 = gW2 + O*hidden;
  double* Xcol   = (double*)malloc(sizeof(double)*R*Ckk);   /* im2col patches [(N·oHoW)×Ckk] */
  double* featPix= (double*)malloc(sizeof(double)*R*nF);    /* relu(conv) pixel-major [R×nF]  */
  double* Feat   = (double*)malloc(sizeof(double)*N*flatDim);/* filter-major [N×flatDim]      */
  double* Hh     = (double*)malloc(sizeof(double)*N*hidden);
  double* Out    = (double*)malloc(sizeof(double)*N*O);
  double* dOut   = (double*)malloc(sizeof(double)*N*O);
  double* dH     = (double*)malloc(sizeof(double)*N*hidden);
  double* dFeat  = (double*)malloc(sizeof(double)*N*flatDim);
  double* dPre   = (double*)malloc(sizeof(double)*R*nF);
  double* pk     = (double*)malloc(sizeof(double)*A);
  /* im2col */
  for (size_t n = 0; n < N; n++) { const double* on = obs + n*inSz;
    for (size_t oy = 0; oy < oH; oy++) for (size_t ox = 0; ox < oW; ox++) {
      size_t r = n*oHoW + oy*oW + ox; double* xr = Xcol + r*Ckk;
      for (size_t idx = 0; idx < Ckk; idx++) xr[idx] = on[nScalar + blas_patch_obs(idx,C,inH,inW,k,s,oy,ox)];
    }
  }
  /* conv: featPix = relu(Xcol·convWᵀ + convB) [R×nF] */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)R,(int)nF,(int)Ckk, 1.0, Xcol,(int)Ckk, convW,(int)Ckk, 0.0, featPix,(int)nF);
  for (size_t r = 0; r < R; r++) for (size_t f = 0; f < nF; f++) { double v = featPix[r*nF+f]+convB[f]; featPix[r*nF+f] = v>0.0?v:0.0; }
  /* transpose pixel-major → filter-major: Feat[n][f·oHoW+p] = featPix[(n·oHoW+p)][f] */
  for (size_t n = 0; n < N; n++) for (size_t f = 0; f < nF; f++) for (size_t p = 0; p < oHoW; p++)
    Feat[n*flatDim + f*oHoW + p] = featPix[(n*oHoW + p)*nF + f];
  /* passthrough scalars: append obs[n][0:nScalar] after the conv features */
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < nScalar; j++)
    Feat[n*flatDim + nF*oHoW + j] = obs[n*inSz + j];
  /* dense: Hh = relu(Feat·W1ᵀ + b1) [N×hidden] */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)hidden,(int)flatDim, 1.0, Feat,(int)flatDim, W1,(int)flatDim, 0.0, Hh,(int)hidden);
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < hidden; j++) { double v = Hh[n*hidden+j]+b1[j]; Hh[n*hidden+j] = v>0.0?v:0.0; }
  /* dense: Out = Hh·W2ᵀ + b2 [N×O] */
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)O,(int)hidden, 1.0, Hh,(int)hidden, W2,(int)hidden, 0.0, Out,(int)O);
  for (size_t n = 0; n < N; n++) for (size_t kk = 0; kk < O; kk++) Out[n*O+kk] += b2[kk];
  /* per-row PPO objective backward → dOut */
  for (size_t n = 0; n < N; n++) {
    const double* out = Out + n*O; double* dout = dOut + n*O;
    size_t a = (size_t)actA[n]; double adv = advA[n], ret = retA[n], oldLogp = oldA[n];
    double sumexp = 0.0; for (size_t kk = 0; kk < A; kk++) sumexp += exp(out[kk]); double lse = log(sumexp);
    double pout = 0.0; for (size_t kk = 0; kk < A; kk++) { pk[kk] = exp(out[kk]-lse); pout += pk[kk]*out[kk]; }
    double logpA = out[a]-lse; double ratio = exp(logpA-oldLogp); double lo = 1.0-clipEps, hi = 1.0+clipEps;
    double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio); double surr1 = adv*ratio, surr2 = adv*ratioC;
    double dPol; if (surr1 <= surr2) dPol = adv*ratio; else { double cg = (lo<ratio && ratio<hi)?1.0:0.0; dPol = adv*cg*ratio; }
    for (size_t kk = 0; kk < A; kk++) { double dp = dPol*(((kk==a)?1.0:0.0)-pk[kk]); double de = entCoef*pk[kk]*(pout-out[kk]); dout[kk] = dp+de; }
    dout[A] = -vfCoef*(out[A]-ret);
  }
  /* dense backward */
  for (size_t n = 0; n < N; n++) for (size_t kk = 0; kk < O; kk++) gb2[kk] += dOut[n*O+kk];
  cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (int)O,(int)hidden,(int)N, 1.0, dOut,(int)O, Hh,(int)hidden, 0.0, gW2,(int)hidden);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N,(int)hidden,(int)O, 1.0, dOut,(int)O, W2,(int)hidden, 0.0, dH,(int)hidden);
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < hidden; j++) if (!(Hh[n*hidden+j] > 0.0)) dH[n*hidden+j] = 0.0;
  for (size_t n = 0; n < N; n++) for (size_t j = 0; j < hidden; j++) gb1[j] += dH[n*hidden+j];
  cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, (int)hidden,(int)flatDim,(int)N, 1.0, dH,(int)hidden, Feat,(int)flatDim, 0.0, gW1,(int)flatDim);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N,(int)flatDim,(int)hidden, 1.0, dH,(int)hidden, W1,(int)flatDim, 0.0, dFeat,(int)flatDim);
  /* conv backward: transpose dFeat (filter-major) → pixel-major, apply relu mask */
  for (size_t n = 0; n < N; n++) for (size_t f = 0; f < nF; f++) for (size_t p = 0; p < oHoW; p++) {
    size_t rr = n*oHoW + p; double d = dFeat[n*flatDim + f*oHoW + p];
    dPre[rr*nF + f] = (featPix[rr*nF + f] > 0.0) ? d : 0.0;
  }
  for (size_t r = 0; r < R; r++) for (size_t f = 0; f < nF; f++) gConvB[f] += dPre[r*nF + f];
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)nF,(int)Ckk,(int)R, 1.0, dPre,(int)nF, Xcol,(int)Ckk, 0.0, gConvW,(int)Ckk);
  free(Xcol); free(featPix); free(Feat); free(Hh); free(Out); free(dOut); free(dH); free(dFeat); free(dPre); free(pk);
  lean_dec(pa); lean_dec(obsBa); lean_dec(actsa); lean_dec(advsa); lean_dec(retsa); lean_dec(oldlpsa);
  return go;
}

/* --- Batched LSTM+PPO truncated-BPTT gradient via BLAS (batched over sequences) ---
   The LSTM recurrence is sequential in TIME, so a single sequence is matrix-VECTOR (no
   GEMM win). The batching axis is the B parallel env-sequences: at each timestep the gate
   and output computations over all B become GEMMs, while the recurrence still steps t=0..T.

   Returns the SUM over the B sequences of the per-sequence gradient — i.e. it equals
   Σ_b (scalar lstmPPOGradSeqFFI on sequence b). Params flat: Wx[4H·D], Wh[4H·H], bih[4H],
   Wo[O·H], bo[O] (O=A+1). Inputs are TIME-MAJOR so each timestep's B rows are contiguous:
   obsB[(t·B+b)·D+d], acts/advs/rets/oldlps/terms[t·B+b]; h0s/c0s[b·H+j] are the detached
   BPTT-initial states. terms[t·B+b]≠0 ⇒ sequence b resets (detached zero state) at t+1, so
   the gradient does not flow across that boundary (nor past t=0), per-sequence.
   Blocked reductions ⇒ matches the scalar/Lean oracle to tolerance, not bit-exactly. */
static inline double blas_sig(double x) { return 1.0/(1.0+exp(-x)); }

/* Read-only reduction of the 7 PufferLib dashboard losses (shared g_mgLoss channel) from a batched
   LSTM BPTT grad's stored logits `OUT`. Single categorical head (A cats), UNCLIPPED value loss (matches
   the head's dout[A]=-vfCoef·(v−ret)). advA is the batch-normalized advantage the grad optimized. Reads
   only host buffers (OUT + the objective inputs) — nothing the grad writes — so it cannot perturb the
   update; called only when g_mgLossOn (render frames). Layout: OUT[(t·B+b)·O + k], scalars at t·B+b.
   Same loss composition as the MLP/MinGRU paths (value loss carries the 0.5; total = pg+vf·vl−ec·ent). */
static void lstm_surface_losses(const double* OUT, const double* actA, const double* advA,
    const double* retA, const double* oldA, size_t T, size_t B, size_t A, size_t O,
    double vfCoef, double entCoef, double clipEps){
  size_t n=T*B; if(n==0) return;
  double lo=1.0-clipEps, hi=1.0+clipEps;
  double sPg=0.0,sV=0.0,sEnt=0.0,sKL=0.0,sOldKL=0.0; long nclip=0;
  for(size_t t=0;t<T;t++) for(size_t b=0;b<B;b++){
    const double* out=OUT+(t*B+b)*O; size_t idx=t*B+b;
    size_t a=(size_t)actA[idx]; double adv=advA[idx], ret=retA[idx], oldlp=oldA[idx];
    double mx=out[0]; for(size_t k=1;k<A;k++) if(out[k]>mx) mx=out[k];
    double se=0.0; for(size_t k=0;k<A;k++) se+=exp(out[k]-mx); double lse=mx+log(se);
    double ent=0.0; for(size_t k=0;k<A;k++){ double lp=out[k]-lse; ent -= exp(lp)*lp; }
    double newlp=out[a]-lse, newval=out[A];
    double lgr=newlp-oldlp, ratio=exp(lgr), ratioC=ratio<lo?lo:(ratio>hi?hi:ratio);
    double aa=-adv*ratio, bb=-adv*ratioC; sPg += (aa>bb?aa:bb);
    if(ratio<lo||ratio>hi) nclip++;
    sKL += (ratio-1.0)-lgr; sOldKL += -lgr;
    sEnt += ent; sV += 0.5*(newval-ret)*(newval-ret);
  }
  double dn=(double)n, pg=sPg/dn, vl=sV/dn, en=sEnt/dn;
  g_mgLoss[0]=pg; g_mgLoss[1]=vl; g_mgLoss[2]=en;
  g_mgLoss[3]=pg + vfCoef*vl - entCoef*en;
  g_mgLoss[4]=sOldKL/dn; g_mgLoss[5]=sKL/dn; g_mgLoss[6]=(double)nclip/dn;
}

LEAN_EXPORT lean_obj_res lean_ffi_lstm_ppo_grad_batch_blas(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, lean_obj_arg termsa,
    lean_obj_arg h0sa, lean_obj_arg c0sa,
    size_t B, size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1, H4 = 4*H;
  size_t P = H4*D + H4*H + H4 + O*H + O;
  const double* pp = lean_float_array_cptr(pa);
  const double* Wx = pp; const double* Wh = Wx + H4*D; const double* bih = Wh + H4*H;
  const double* Wo = bih + H4; const double* bo = Wo + O*H;
  const double* obsB = lean_float_array_cptr(obsBa);
  const double* actA = lean_float_array_cptr(actsa); const double* advA = lean_float_array_cptr(advsa);
  const double* retA = lean_float_array_cptr(retsa); const double* oldA = lean_float_array_cptr(oldlpsa);
  const double* termA = lean_float_array_cptr(termsa);
  const double* h0s = lean_float_array_cptr(h0sa); const double* c0s = lean_float_array_cptr(c0sa);
  lean_object* go = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(go);
  /* GPU-resident BPTT (device twin of this function); falls through to the CPU BLAS code below if no
     usable device / unsupported shape. Same f64 tolerance vs the scalar oracle (verify-lstm-blas). */
  if (!lstm_gpu_off()) {
    double* outH = g_mgLossOn ? (double*)malloc(sizeof(double)*T*O*B) : NULL;
    if (cuda_lstm_ppo_grad_batch(pp, obsB, actA, advA, retA, oldA, termA, h0s, c0s,
                                 B, T, H, D, A, vfCoef, entCoef, clipEps, g, outH, 0)) {
      if (outH) { lstm_surface_losses(outH, actA, advA, retA, oldA, T, B, A, O, vfCoef, entCoef, clipEps); free(outH); }
      lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
      return go;
    }
    if (outH) free(outH);
  }
  for (size_t t = 0; t < P; t++) g[t] = 0.0;
  double* gWx = g; double* gWh = gWx + H4*D; double* gbih = gWh + H4*H; double* gWo = gbih + H4; double* gbo = gWo + O*H;
  size_t BH = B*H, BH4 = B*H4, BO = B*O;
  /* per-timestep stored activations */
  double* II = (double*)malloc(sizeof(double)*T*BH); double* FF = (double*)malloc(sizeof(double)*T*BH);
  double* GG = (double*)malloc(sizeof(double)*T*BH); double* OO = (double*)malloc(sizeof(double)*T*BH);
  double* TC = (double*)malloc(sizeof(double)*T*BH); double* HP = (double*)malloc(sizeof(double)*T*BH);
  double* CP = (double*)malloc(sizeof(double)*T*BH); double* HT = (double*)malloc(sizeof(double)*T*BH);
  double* OUT= (double*)malloc(sizeof(double)*T*BO);
  double* G   = (double*)malloc(sizeof(double)*BH4);
  double* Hprev = (double*)malloc(sizeof(double)*BH); double* Cprev = (double*)malloc(sizeof(double)*BH);
  double* dOut = (double*)malloc(sizeof(double)*BO); double* dHt = (double*)malloc(sizeof(double)*BH);
  double* dG = (double*)malloc(sizeof(double)*BH4); double* dHprev = (double*)malloc(sizeof(double)*BH);
  double* dCprev = (double*)malloc(sizeof(double)*BH); double* dHnext = (double*)malloc(sizeof(double)*BH);
  double* dCnext = (double*)malloc(sizeof(double)*BH); double* pk = (double*)malloc(sizeof(double)*A);
  for (size_t i = 0; i < BH; i++) { Hprev[i] = h0s[i]; Cprev[i] = c0s[i]; }
  /* FORWARD (time-major, batched over B) */
  for (size_t t = 0; t < T; t++) {
    if (t > 0) for (size_t b = 0; b < B; b++) if (termA[(t-1)*B + b] != 0.0)
      for (size_t j = 0; j < H; j++) { Hprev[b*H+j] = 0.0; Cprev[b*H+j] = 0.0; }
    double* HPt = HP + t*BH; double* CPt = CP + t*BH;
    for (size_t i = 0; i < BH; i++) { HPt[i] = Hprev[i]; CPt[i] = Cprev[i]; }
    const double* Xt = obsB + t*B*D;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)H4,(int)D, 1.0, Xt,(int)D, Wx,(int)D, 0.0, G,(int)H4);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)H4,(int)H, 1.0, Hprev,(int)H, Wh,(int)H, 1.0, G,(int)H4);
    double* Ht = HT + t*BH;
    for (size_t b = 0; b < B; b++) for (size_t j = 0; j < H; j++) {
      double gi = G[b*H4 + j] + bih[j],       gf = G[b*H4 + H+j]   + bih[H+j];
      double gg = G[b*H4 + 2*H+j] + bih[2*H+j], gob = G[b*H4 + 3*H+j] + bih[3*H+j];
      double iv = blas_sig(gi), fv = blas_sig(gf), gv = tanh(gg), ov = blas_sig(gob);
      double cv = fv*Cprev[b*H+j] + iv*gv; double tcv = tanh(cv); double hv = ov*tcv;
      size_t o = t*BH + b*H + j;
      II[o]=iv; FF[o]=fv; GG[o]=gv; OO[o]=ov; TC[o]=tcv; Ht[b*H+j]=hv;
      Cprev[b*H+j]=cv; Hprev[b*H+j]=hv;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)O,(int)H, 1.0, Ht,(int)H, Wo,(int)H, 0.0, OUT+t*BO,(int)O);
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < O; kk++) OUT[t*BO + b*O + kk] += bo[kk];
  }
  /* BACKWARD (reverse time) */
  for (size_t i = 0; i < BH; i++) { dHnext[i] = 0.0; dCnext[i] = 0.0; }
  for (size_t tt = T; tt-- > 0; ) {
    size_t t = tt; const double* Ot = OUT + t*BO;
    for (size_t b = 0; b < B; b++) {
      const double* out = Ot + b*O; double* dout = dOut + b*O;
      size_t idx = t*B + b; size_t a = (size_t)actA[idx];
      double adv = advA[idx], ret = retA[idx], oldLogp = oldA[idx];
      double sumexp = 0.0; for (size_t kk = 0; kk < A; kk++) sumexp += exp(out[kk]); double lse = log(sumexp);
      double pout = 0.0; for (size_t kk = 0; kk < A; kk++) { pk[kk] = exp(out[kk]-lse); pout += pk[kk]*out[kk]; }
      double logpA = out[a]-lse; double ratio = exp(logpA-oldLogp); double lo = 1.0-clipEps, hi = 1.0+clipEps;
      double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio); double surr1 = adv*ratio, surr2 = adv*ratioC;
      double dPol; if (surr1 <= surr2) dPol = adv*ratio; else { double cg = (lo<ratio && ratio<hi)?1.0:0.0; dPol = adv*cg*ratio; }
      for (size_t kk = 0; kk < A; kk++) { double dp = dPol*(((kk==a)?1.0:0.0)-pk[kk]); double de = entCoef*pk[kk]*(pout-out[kk]); dout[kk] = dp+de; }
      dout[A] = -vfCoef*(out[A]-ret);
    }
    double* Ht = HT + t*BH;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)O,(int)H,(int)B, 1.0, dOut,(int)O, Ht,(int)H, 1.0, gWo,(int)H);
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < O; kk++) gbo[kk] += dOut[b*O+kk];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)B,(int)H,(int)O, 1.0, dOut,(int)O, Wo,(int)H, 0.0, dHt,(int)H);
    for (size_t i = 0; i < BH; i++) dHt[i] += dHnext[i];
    double* CPt = CP + t*BH;
    for (size_t b = 0; b < B; b++) for (size_t j = 0; j < H; j++) {
      size_t o = t*BH + b*H + j;
      double iv=II[o], fv=FF[o], gv=GG[o], ov=OO[o], tcv=TC[o], cprev=CPt[b*H+j];
      double dh = dHt[b*H+j];
      double dc = dCnext[b*H+j] + dh*ov*(1.0-tcv*tcv);
      double do_ = dh*tcv; double df = dc*cprev, di = dc*gv, dg_ = dc*iv;
      dCprev[b*H+j] = dc*fv;
      dG[b*H4 + j]     = di*iv*(1.0-iv);
      dG[b*H4 + H+j]   = df*fv*(1.0-fv);
      dG[b*H4 + 2*H+j] = dg_*(1.0-gv*gv);
      dG[b*H4 + 3*H+j] = do_*ov*(1.0-ov);
    }
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < H4; kk++) gbih[kk] += dG[b*H4+kk];
    const double* Xt = obsB + t*B*D; double* HPt = HP + t*BH;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)H4,(int)D,(int)B, 1.0, dG,(int)H4, Xt,(int)D, 1.0, gWx,(int)D);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)H4,(int)H,(int)B, 1.0, dG,(int)H4, HPt,(int)H, 1.0, gWh,(int)H);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)B,(int)H,(int)H4, 1.0, dG,(int)H4, Wh,(int)H, 0.0, dHprev,(int)H);
    for (size_t b = 0; b < B; b++) {
      int flow = (t > 0 && termA[(t-1)*B + b] == 0.0);
      for (size_t j = 0; j < H; j++) {
        if (flow) { dHnext[b*H+j] = dHprev[b*H+j]; dCnext[b*H+j] = dCprev[b*H+j]; }
        else      { dHnext[b*H+j] = 0.0;           dCnext[b*H+j] = 0.0; }
      }
    }
  }
  /* dashboard losses (shared g_mgLoss channel), read-only over the stored logits (render-frame cadence) */
  if(g_mgLossOn) lstm_surface_losses(OUT, actA, advA, retA, oldA, T, B, A, O, vfCoef, entCoef, clipEps);
  free(II);free(FF);free(GG);free(OO);free(TC);free(HP);free(CP);free(HT);free(OUT);free(G);
  free(Hprev);free(Cprev);free(dOut);free(dHt);free(dG);free(dHprev);free(dCprev);free(dHnext);free(dCnext);free(pk);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
  return go;
}

/* CPU f32 BPTT-grad fallback, shared by the f32 and bf16 GPU tiers (defined just below the f32 wrapper). */
static void lstm_grad_batch_cpu_f32(
    const double* ppd, const double* obsBd, const double* actA, const double* advA,
    const double* retA, const double* oldA, const double* termA,
    const double* h0sd, const double* c0sd,
    size_t B, size_t T, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double* g);

/* ---- Batched LSTM+PPO BPTT gradient via BLAS, f32 TIER (a further precision step past the f64-BLAS
   tier above) ------------------------------------------------------------------------------------
   Same algorithm as `lean_ffi_lstm_ppo_grad_batch_blas`, but weights/obs/all per-timestep activations
   are staged to float ONCE and every GEMM is `cblas_sgemm` -- half the memory traffic and (on
   AVX2/AVX-512) twice the SIMD lanes/cycle vs double, at the cost of another tolerance step past the
   f64-BLAS kernel (verified against it by `verify-lstm-grad-f32`, ~1e-5..1e-6 relative -- this
   project's usual f32-tier bar, e.g. `cudaMlpPpoGradFFI`'s bf16=0 f32 cross-check). The gradient
   ACCUMULATES in float (matching every GEMM-tier kernel elsewhere in this file: reduced precision
   during the hot loop, widened to f64 only for the Lean-facing output) -- only the small PPO-objective
   scalars (adv/ret/oldLogp, T·B of them, not T·B·H) stay double on read, cast to float per use, since
   staging them costs nothing and every actual FLOP happens in float regardless. Opt-in
   (PUFFER_LSTM_F32=1 gates `trainPluginEnvRec`'s choice of this vs the f64-BLAS kernel above) --
   default stays f64-BLAS. Same I/O contract as `lean_ffi_lstm_ppo_grad_batch_blas`. */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_ppo_grad_batch_blas_f32(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, lean_obj_arg termsa,
    lean_obj_arg h0sa, lean_obj_arg c0sa,
    size_t B, size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1, H4 = 4*H;
  size_t P = H4*D + H4*H + H4 + O*H + O;
  const double* ppd = lean_float_array_cptr(pa);
  const double* obsBd = lean_float_array_cptr(obsBa);
  const double* actA = lean_float_array_cptr(actsa); const double* advA = lean_float_array_cptr(advsa);
  const double* retA = lean_float_array_cptr(retsa); const double* oldA = lean_float_array_cptr(oldlpsa);
  const double* termA = lean_float_array_cptr(termsa);
  const double* h0sd = lean_float_array_cptr(h0sa); const double* c0sd = lean_float_array_cptr(c0sa);
  lean_object* go = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(go);
  /* GPU-resident BPTT, f32 tier (cublasSgemm); falls through to the CPU cblas_sgemm code below if no
     usable device / unsupported shape. Verified vs the f64 path by verify-lstm-grad-f32. */
  if (!lstm_gpu_off()) {
    double* outH = g_mgLossOn ? (double*)malloc(sizeof(double)*T*O*B) : NULL;
    if (cuda_lstm_ppo_grad_batch(ppd, obsBd, actA, advA, retA, oldA, termA, h0sd, c0sd,
                                 B, T, H, D, A, vfCoef, entCoef, clipEps, g, outH, 1)) {
      if (outH) { lstm_surface_losses(outH, actA, advA, retA, oldA, T, B, A, O, vfCoef, entCoef, clipEps); free(outH); }
      lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
      return go;
    }
    if (outH) free(outH);
  }
  lstm_grad_batch_cpu_f32(ppd, obsBd, actA, advA, retA, oldA, termA, h0sd, c0sd,
                          B, T, H, D, A, vfCoef, entCoef, clipEps, g);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
  return go;
}

/* CPU f32 BPTT-grad fallback (no usable device): the staged-to-float cblas_sgemm twin of the f64-BLAS
   body, extracted verbatim so the f32 and bf16 GPU tiers share ONE CPU path (bf16 has no CPU form — f32
   is its closest fallback). Writes the flat gradient g[P]; the caller owns the lean_obj lifetimes. */
static void lstm_grad_batch_cpu_f32(
    const double* ppd, const double* obsBd, const double* actA, const double* advA,
    const double* retA, const double* oldA, const double* termA,
    const double* h0sd, const double* c0sd,
    size_t B, size_t T, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double* g) {
  size_t O = A + 1, H4 = 4*H;
  double* ogWx = g; double* ogWh = ogWx + H4*D; double* ogbih = ogWh + H4*H; double* ogWo = ogbih + H4; double* ogbo = ogWo + O*H;

  float* Wx = (float*)malloc(sizeof(float)*H4*D); float* Wh = (float*)malloc(sizeof(float)*H4*H);
  float* bih = (float*)malloc(sizeof(float)*H4); float* Wo = (float*)malloc(sizeof(float)*O*H); float* bo = (float*)malloc(sizeof(float)*O);
  for (size_t i = 0; i < H4*D; i++) Wx[i] = (float)ppd[i];
  for (size_t i = 0; i < H4*H; i++) Wh[i] = (float)ppd[H4*D+i];
  for (size_t i = 0; i < H4; i++) bih[i] = (float)ppd[H4*D+H4*H+i];
  for (size_t i = 0; i < O*H; i++) Wo[i] = (float)ppd[H4*D+H4*H+H4+i];
  for (size_t i = 0; i < O; i++) bo[i] = (float)ppd[H4*D+H4*H+H4+O*H+i];
  float* obsB = (float*)malloc(sizeof(float)*T*B*D);
  for (size_t i = 0; i < T*B*D; i++) obsB[i] = (float)obsBd[i];

  size_t BH = B*H, BH4 = B*H4, BO = B*O;
  float* gWx = (float*)calloc(H4*D, sizeof(float)); float* gWh = (float*)calloc(H4*H, sizeof(float));
  float* gbih = (float*)calloc(H4, sizeof(float)); float* gWo = (float*)calloc(O*H, sizeof(float)); float* gbo = (float*)calloc(O, sizeof(float));
  float* II = (float*)malloc(sizeof(float)*T*BH); float* FF = (float*)malloc(sizeof(float)*T*BH);
  float* GG = (float*)malloc(sizeof(float)*T*BH); float* OO = (float*)malloc(sizeof(float)*T*BH);
  float* TC = (float*)malloc(sizeof(float)*T*BH); float* HP = (float*)malloc(sizeof(float)*T*BH);
  float* CP = (float*)malloc(sizeof(float)*T*BH); float* HT = (float*)malloc(sizeof(float)*T*BH);
  float* OUT = (float*)malloc(sizeof(float)*T*BO);
  float* G = (float*)malloc(sizeof(float)*BH4);
  float* Hprev = (float*)malloc(sizeof(float)*BH); float* Cprev = (float*)malloc(sizeof(float)*BH);
  float* dOut = (float*)malloc(sizeof(float)*BO); float* dHt = (float*)malloc(sizeof(float)*BH);
  float* dG = (float*)malloc(sizeof(float)*BH4); float* dHprev = (float*)malloc(sizeof(float)*BH);
  float* dCprev = (float*)malloc(sizeof(float)*BH); float* dHnext = (float*)malloc(sizeof(float)*BH);
  float* dCnext = (float*)malloc(sizeof(float)*BH); float* pk = (float*)malloc(sizeof(float)*A);
  for (size_t i = 0; i < BH; i++) { Hprev[i] = (float)h0sd[i]; Cprev[i] = (float)c0sd[i]; }
  /* FORWARD (time-major, batched over B) */
  for (size_t t = 0; t < T; t++) {
    if (t > 0) for (size_t b = 0; b < B; b++) if (termA[(t-1)*B + b] != 0.0)
      for (size_t j = 0; j < H; j++) { Hprev[b*H+j] = 0.0f; Cprev[b*H+j] = 0.0f; }
    float* HPt = HP + t*BH; float* CPt = CP + t*BH;
    for (size_t i = 0; i < BH; i++) { HPt[i] = Hprev[i]; CPt[i] = Cprev[i]; }
    const float* Xt = obsB + t*B*D;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)H4,(int)D, 1.0f, Xt,(int)D, Wx,(int)D, 0.0f, G,(int)H4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)H4,(int)H, 1.0f, Hprev,(int)H, Wh,(int)H, 1.0f, G,(int)H4);
    float* Ht = HT + t*BH;
    for (size_t b = 0; b < B; b++) for (size_t j = 0; j < H; j++) {
      float gi = G[b*H4 + j] + bih[j],       gf = G[b*H4 + H+j]   + bih[H+j];
      float gg = G[b*H4 + 2*H+j] + bih[2*H+j], gob = G[b*H4 + 3*H+j] + bih[3*H+j];
      float iv = 1.0f/(1.0f+expf(-gi)), fv = 1.0f/(1.0f+expf(-gf)), gv = tanhf(gg), ov = 1.0f/(1.0f+expf(-gob));
      float cv = fv*Cprev[b*H+j] + iv*gv; float tcv = tanhf(cv); float hv = ov*tcv;
      size_t o = t*BH + b*H + j;
      II[o]=iv; FF[o]=fv; GG[o]=gv; OO[o]=ov; TC[o]=tcv; Ht[b*H+j]=hv;
      Cprev[b*H+j]=cv; Hprev[b*H+j]=hv;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)B,(int)O,(int)H, 1.0f, Ht,(int)H, Wo,(int)H, 0.0f, OUT+t*BO,(int)O);
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < O; kk++) OUT[t*BO + b*O + kk] += bo[kk];
  }
  /* BACKWARD (reverse time) */
  for (size_t i = 0; i < BH; i++) { dHnext[i] = 0.0f; dCnext[i] = 0.0f; }
  for (size_t tt = T; tt-- > 0; ) {
    size_t t = tt; const float* Ot = OUT + t*BO;
    for (size_t b = 0; b < B; b++) {
      const float* out = Ot + b*O; float* dout = dOut + b*O;
      size_t idx = t*B + b; size_t a = (size_t)actA[idx];
      float adv = (float)advA[idx], ret = (float)retA[idx], oldLogp = (float)oldA[idx];
      float sumexp = 0.0f; for (size_t kk = 0; kk < A; kk++) sumexp += expf(out[kk]); float lse = logf(sumexp);
      float pout = 0.0f; for (size_t kk = 0; kk < A; kk++) { pk[kk] = expf(out[kk]-lse); pout += pk[kk]*out[kk]; }
      float logpA = out[a]-lse; float ratio = expf(logpA-oldLogp); float lo = 1.0f-(float)clipEps, hi = 1.0f+(float)clipEps;
      float ratioC = ratio<lo?lo:(ratio>hi?hi:ratio); float surr1 = adv*ratio, surr2 = adv*ratioC;
      float dPol; if (surr1 <= surr2) dPol = adv*ratio; else { float cg = (lo<ratio && ratio<hi)?1.0f:0.0f; dPol = adv*cg*ratio; }
      for (size_t kk = 0; kk < A; kk++) { float dp = dPol*(((kk==a)?1.0f:0.0f)-pk[kk]); float de = (float)entCoef*pk[kk]*(pout-out[kk]); dout[kk] = dp+de; }
      dout[A] = -(float)vfCoef*(out[A]-ret);
    }
    float* Ht = HT + t*BH;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)O,(int)H,(int)B, 1.0f, dOut,(int)O, Ht,(int)H, 1.0f, gWo,(int)H);
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < O; kk++) gbo[kk] += dOut[b*O+kk];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)B,(int)H,(int)O, 1.0f, dOut,(int)O, Wo,(int)H, 0.0f, dHt,(int)H);
    for (size_t i = 0; i < BH; i++) dHt[i] += dHnext[i];
    float* CPt = CP + t*BH;
    for (size_t b = 0; b < B; b++) for (size_t j = 0; j < H; j++) {
      size_t o = t*BH + b*H + j;
      float iv=II[o], fv=FF[o], gv=GG[o], ov=OO[o], tcv=TC[o], cprev=CPt[b*H+j];
      float dh = dHt[b*H+j];
      float dc = dCnext[b*H+j] + dh*ov*(1.0f-tcv*tcv);
      float do_ = dh*tcv; float df = dc*cprev, di = dc*gv, dg_ = dc*iv;
      dCprev[b*H+j] = dc*fv;
      dG[b*H4 + j]     = di*iv*(1.0f-iv);
      dG[b*H4 + H+j]   = df*fv*(1.0f-fv);
      dG[b*H4 + 2*H+j] = dg_*(1.0f-gv*gv);
      dG[b*H4 + 3*H+j] = do_*ov*(1.0f-ov);
    }
    for (size_t b = 0; b < B; b++) for (size_t kk = 0; kk < H4; kk++) gbih[kk] += dG[b*H4+kk];
    const float* Xt = obsB + t*B*D; float* HPt = HP + t*BH;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)H4,(int)D,(int)B, 1.0f, dG,(int)H4, Xt,(int)D, 1.0f, gWx,(int)D);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)H4,(int)H,(int)B, 1.0f, dG,(int)H4, HPt,(int)H, 1.0f, gWh,(int)H);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)B,(int)H,(int)H4, 1.0f, dG,(int)H4, Wh,(int)H, 0.0f, dHprev,(int)H);
    for (size_t b = 0; b < B; b++) {
      int flow = (t > 0 && termA[(t-1)*B + b] == 0.0);
      for (size_t j = 0; j < H; j++) {
        if (flow) { dHnext[b*H+j] = dHprev[b*H+j]; dCnext[b*H+j] = dCprev[b*H+j]; }
        else      { dHnext[b*H+j] = 0.0f;           dCnext[b*H+j] = 0.0f; }
      }
    }
  }
  for (size_t i = 0; i < H4*D; i++) ogWx[i] = (double)gWx[i];
  for (size_t i = 0; i < H4*H; i++) ogWh[i] = (double)gWh[i];
  for (size_t i = 0; i < H4; i++) ogbih[i] = (double)gbih[i];
  for (size_t i = 0; i < O*H; i++) ogWo[i] = (double)gWo[i];
  for (size_t i = 0; i < O; i++) ogbo[i] = (double)gbo[i];
  free(Wx);free(Wh);free(bih);free(Wo);free(bo);free(obsB);
  free(gWx);free(gWh);free(gbih);free(gWo);free(gbo);
  /* dashboard losses (shared g_mgLoss channel): widen the f32 logits to a scratch double buffer (render
     frames only) so the single reducer serves both tiers; read-only, cannot perturb the update. */
  if(g_mgLossOn){ double* od=(double*)malloc(sizeof(double)*T*BO);
    if(od){ for(size_t i=0;i<T*BO;i++) od[i]=(double)OUT[i];
      lstm_surface_losses(od, actA, advA, retA, oldA, T, B, A, O, vfCoef, entCoef, clipEps); free(od); } }
  free(II);free(FF);free(GG);free(OO);free(TC);free(HP);free(CP);free(HT);free(OUT);free(G);
  free(Hprev);free(Cprev);free(dOut);free(dHt);free(dG);free(dHprev);free(dCprev);free(dHnext);free(dCnext);free(pk);
}

/* ---- Batched LSTM+PPO BPTT gradient via BLAS, bf16 TENSOR-CORE TIER (a further precision step past the
   f32 tier) ------------------------------------------------------------------------------------------
   Identical I/O contract and algorithm to `lean_ffi_lstm_ppo_grad_batch_blas_f32`, but the GPU BPTT runs
   the GEMMs on bf16 tensor cores (float buffers rounded to bf16 for the MAC, f32 accumulate — cuBLAS
   CUBLAS_COMPUTE_32F_FAST_16BF; the gate/PPO elementwise kernels stay f32). Another tolerance step past
   the f32 tier (bf16 has an 8-bit mantissa; verified vs the f64 path by `verify-lstm-grad-bf16`). Opt-in
   (`PUFFER_LSTM_BF16=1` gates `trainPluginEnvRec`'s choice). No CPU bf16 form — the no-device fallback is
   the shared f32 CPU path (`lstm_grad_batch_cpu_f32`). */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_ppo_grad_batch_blas_bf16(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, lean_obj_arg termsa,
    lean_obj_arg h0sa, lean_obj_arg c0sa,
    size_t B, size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1, H4 = 4*H;
  size_t P = H4*D + H4*H + H4 + O*H + O;
  const double* ppd = lean_float_array_cptr(pa);
  const double* obsBd = lean_float_array_cptr(obsBa);
  const double* actA = lean_float_array_cptr(actsa); const double* advA = lean_float_array_cptr(advsa);
  const double* retA = lean_float_array_cptr(retsa); const double* oldA = lean_float_array_cptr(oldlpsa);
  const double* termA = lean_float_array_cptr(termsa);
  const double* h0sd = lean_float_array_cptr(h0sa); const double* c0sd = lean_float_array_cptr(c0sa);
  lean_object* go = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(go);
  /* GPU-resident BPTT, bf16 tier (tensor cores); falls through to the shared f32 CPU path if no usable
     device / unsupported shape. Verified vs the f64 path by verify-lstm-grad-bf16. */
  if (!lstm_gpu_off()) {
    double* outH = g_mgLossOn ? (double*)malloc(sizeof(double)*T*O*B) : NULL;
    if (cuda_lstm_ppo_grad_batch(ppd, obsBd, actA, advA, retA, oldA, termA, h0sd, c0sd,
                                 B, T, H, D, A, vfCoef, entCoef, clipEps, g, outH, 2)) {
      if (outH) { lstm_surface_losses(outH, actA, advA, retA, oldA, T, B, A, O, vfCoef, entCoef, clipEps); free(outH); }
      lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
      return go;
    }
    if (outH) free(outH);
  }
  lstm_grad_batch_cpu_f32(ppd, obsBd, actA, advA, retA, oldA, termA, h0sd, c0sd,
                          B, T, H, D, A, vfCoef, entCoef, clipEps, g);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(termsa);lean_dec(h0sa);lean_dec(c0sa);
  return go;
}

/* ---- Batched LSTM forward step via BLAS (rollout) --------------------------------------------
   BLAS twin of `lean_ffi_lstm_fwd_step_batch` (pufferffi.c: naive scalar per-row dot products,
   bit-exact vs `lstmCellF`). Same math -- the gate/head pre-activations go through cblas_dgemm
   (blocked, SIMD-vectorized) instead of scalar loops, which is where the naive kernel's rollout time
   went (measured: transposing rollout output into the BPTT kernel's SoA layout is only ~4-7% of
   "rollout" time; the forward math dominates the rest). Trades bit-exactness for speed, same as
   `lean_ffi_lstm_ppo_grad_batch_blas` vs `lean_ffi_lstm_ppo_grad_seq` -- and the trainer already
   accepted that precision class for the BPTT step, so this doesn't add a new one, just extends the
   existing tolerance zone to the rollout (verified against the scalar twin by `verify-lstm-fwd-blas`).
   params flat: Wx[4H·D],Wh[4H·H],bih[4H],Wo[O·H],bo[O] (O=A+1). obs/h/c are N·D/N·H/N·H row-major.
   Returns [hN(N·H); cN(N·H); out(N·O)] -- same layout as the scalar twin. */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_fwd_step_batch_blas(
    lean_obj_arg pa, lean_obj_arg obsa, lean_obj_arg ha, lean_obj_arg ca,
    size_t N, size_t D, size_t H, size_t A) {
  size_t O = A + 1, H4 = 4*H;
  const double* pp = lean_float_array_cptr(pa);
  const double* Wx = pp; const double* Wh = Wx + H4*D; const double* bih = Wh + H4*H;
  const double* Wo = bih + H4; const double* bo = Wo + O*H;
  const double* obs = lean_float_array_cptr(obsa);
  const double* hin = lean_float_array_cptr(ha); const double* cin = lean_float_array_cptr(ca);
  size_t outSz = N*H + N*H + N*O;
  lean_object* go = lean_alloc_sarray(sizeof(double), outSz, outSz);
  double* out = lean_float_array_cptr(go);
  double* hN = out; double* cN = out + N*H; double* outO = out + 2*N*H;
  double* G = (double*)malloc(sizeof(double)*N*H4);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)H4,(int)D, 1.0, obs,(int)D, Wx,(int)D, 0.0, G,(int)H4);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)H4,(int)H, 1.0, hin,(int)H, Wh,(int)H, 1.0, G,(int)H4);
  for (size_t b = 0; b < N; b++) for (size_t j = 0; j < H; j++) {
    double gi = G[b*H4+j] + bih[j], gf = G[b*H4+H+j] + bih[H+j];
    double gg = G[b*H4+2*H+j] + bih[2*H+j], gob = G[b*H4+3*H+j] + bih[3*H+j];
    double iv = blas_sig(gi), fv = blas_sig(gf), gv = tanh(gg), ov = blas_sig(gob);
    double cprev = cin[b*H+j]; double cj = fv*cprev + iv*gv;
    cN[b*H+j] = cj; hN[b*H+j] = ov*tanh(cj);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)O,(int)H, 1.0, hN,(int)H, Wo,(int)H, 0.0, outO,(int)O);
  for (size_t b = 0; b < N; b++) for (size_t m = 0; m < O; m++) outO[b*O+m] += bo[m];
  free(G);
  lean_dec(pa); lean_dec(obsa); lean_dec(ha); lean_dec(ca);
  return go;
}

/* f32 twin of `lean_ffi_lstm_fwd_step_batch_blas` -- weights/obs/h/c staged to float once, all GEMMs
   `cblas_sgemm`, elementwise gate math in float, widened to f64 only for the Lean-facing output.
   Verified against the f64-BLAS twin by `verify-lstm-fwd-f32`. Opt-in (PUFFER_LSTM_F32=1). */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_fwd_step_batch_blas_f32(
    lean_obj_arg pa, lean_obj_arg obsa, lean_obj_arg ha, lean_obj_arg ca,
    size_t N, size_t D, size_t H, size_t A) {
  size_t O = A + 1, H4 = 4*H;
  const double* ppd = lean_float_array_cptr(pa);
  const double* obsd = lean_float_array_cptr(obsa);
  const double* hind = lean_float_array_cptr(ha); const double* cind = lean_float_array_cptr(ca);
  size_t outSz = N*H + N*H + N*O;
  lean_object* go = lean_alloc_sarray(sizeof(double), outSz, outSz);
  double* out = lean_float_array_cptr(go);
  float* Wx = (float*)malloc(sizeof(float)*H4*D); float* Wh = (float*)malloc(sizeof(float)*H4*H);
  float* bih = (float*)malloc(sizeof(float)*H4); float* Wo = (float*)malloc(sizeof(float)*O*H); float* bo = (float*)malloc(sizeof(float)*O);
  for (size_t i = 0; i < H4*D; i++) Wx[i] = (float)ppd[i];
  for (size_t i = 0; i < H4*H; i++) Wh[i] = (float)ppd[H4*D+i];
  for (size_t i = 0; i < H4; i++) bih[i] = (float)ppd[H4*D+H4*H+i];
  for (size_t i = 0; i < O*H; i++) Wo[i] = (float)ppd[H4*D+H4*H+H4+i];
  for (size_t i = 0; i < O; i++) bo[i] = (float)ppd[H4*D+H4*H+H4+O*H+i];
  float* obs = (float*)malloc(sizeof(float)*N*D); float* hin = (float*)malloc(sizeof(float)*N*H); float* cin = (float*)malloc(sizeof(float)*N*H);
  for (size_t i = 0; i < N*D; i++) obs[i] = (float)obsd[i];
  for (size_t i = 0; i < N*H; i++) { hin[i] = (float)hind[i]; cin[i] = (float)cind[i]; }
  float* hN = (float*)malloc(sizeof(float)*N*H); float* cN = (float*)malloc(sizeof(float)*N*H); float* outO = (float*)malloc(sizeof(float)*N*O);
  float* G = (float*)malloc(sizeof(float)*N*H4);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)H4,(int)D, 1.0f, obs,(int)D, Wx,(int)D, 0.0f, G,(int)H4);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)H4,(int)H, 1.0f, hin,(int)H, Wh,(int)H, 1.0f, G,(int)H4);
  for (size_t b = 0; b < N; b++) for (size_t j = 0; j < H; j++) {
    float gi = G[b*H4+j] + bih[j], gf = G[b*H4+H+j] + bih[H+j];
    float gg = G[b*H4+2*H+j] + bih[2*H+j], gob = G[b*H4+3*H+j] + bih[3*H+j];
    float iv = 1.0f/(1.0f+expf(-gi)), fv = 1.0f/(1.0f+expf(-gf)), gv = tanhf(gg), ov = 1.0f/(1.0f+expf(-gob));
    float cprev = cin[b*H+j]; float cj = fv*cprev + iv*gv;
    cN[b*H+j] = cj; hN[b*H+j] = ov*tanhf(cj);
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)N,(int)O,(int)H, 1.0f, hN,(int)H, Wo,(int)H, 0.0f, outO,(int)O);
  for (size_t b = 0; b < N; b++) for (size_t m = 0; m < O; m++) outO[b*O+m] += bo[m];
  double* houtD = out; double* coutD = out + N*H; double* ooutD = out + 2*N*H;
  for (size_t i = 0; i < N*H; i++) { houtD[i] = (double)hN[i]; coutD[i] = (double)cN[i]; }
  for (size_t i = 0; i < N*O; i++) ooutD[i] = (double)outO[i];
  free(Wx);free(Wh);free(bih);free(Wo);free(bo);free(obs);free(hin);free(cin);free(hN);free(cN);free(outO);free(G);
  lean_dec(pa); lean_dec(obsa); lean_dec(ha); lean_dec(ca);
  return go;
}

/* Cap OpenBLAS's own thread pool (process-global). At the LSTM plugin trainer's problem shape (many
   small, SEQUENTIAL per-timestep GEMMs -- can't parallelize across time, only within one call) the
   default all-cores threading measured WORSE than a small fixed count: at a 256-env/H64/T64 config,
   BPTT went 1.2s (8 threads) -> 2.7s (32 threads, the untouched default on this 32-core box) -- the
   thread-team wake/sync overhead per tiny sequential GEMM call dominates the parallelism gained. A
   fixed 4-8 was close to optimal at both that config and a larger 1024-env/H128/T64 one (where
   thread=1 left real parallelism on the table: 11.1K vs 13.3K SPS). Called ONCE from
   `trainPluginEnvRec` at trainer startup; safe because a `puffer train` process only ever runs one
   trainer to completion -- no other BLAS-heavy code shares the process to have its preference
   overridden. */
extern void openblas_set_num_threads(int num_threads);
LEAN_EXPORT lean_obj_res lean_ffi_blas_set_threads(size_t n, lean_obj_arg w) {
  (void)w;
  openblas_set_num_threads((int)n);
  /* this OpenBLAS build dispatches its GEMMs via OpenMP (libgomp, pulled in transitively by
     libopenblas.so -- confirmed via `nm -D`), and `openblas_set_num_threads` alone did NOT change
     measured behavior (a known OpenBLAS quirk with USE_OPENMP builds: the real knob is
     `omp_set_num_threads`/`OMP_NUM_THREADS`). libgomp isn't a direct link input of this object file
     (only reachable transitively at runtime), so resolve it dynamically instead of adding a link-time
     dependency. */
  void (*omp_set)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "omp_set_num_threads");
  if (omp_set) omp_set((int)n);
  return lean_io_result_mk_ok(lean_box(0));
}

/* Reports 1 if a CUDA device is usable, else 0 (so the CLI can label the bench).
   Takes the Lean `Unit` value (ignored) so it is a genuine function, not a CAF. */
LEAN_EXPORT uint8_t lean_ffi_cuda_available(lean_obj_arg unit) {
  lean_dec(unit);
  int n = 0;
  return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}

/* ---- CPU Muon step for the whole MLP -------------------------------------------------------
   A native-C port of the GPU `muon_mat_dev`/`k_stepvec` (ffi/puffercuda.cu) and the pure-Lean
   `Puffer.FloatR.Muon.stepMat`/`stepVec` — the SAME f64 op order (naive matmuls, i/j/l left-to-right;
   two-level frobNorm fold), so compiled with `-ffp-contract=off` (no FMA) it is BIT-EXACT with the
   Lean oracle, at ~100× the speed of the boxed `Array (Array Float)` Lean path. This is the CPU
   counterpart the codebase lacked: it takes the CPU PPO+Muon step from
   NS-bound (91–95% pure-Lean Muon) to gradient-bound. Naive matmuls (no threading) are the right
   call at the small hidden sizes a CPU trainer uses; the Gram matrices are min(rows,cols)². */
static const double MUON_C[5][3] = {
  {4.0848,-6.8946,2.9270},{3.9505,-6.3029,2.6377},{3.7418,-5.5913,2.3037},
  {2.8769,-3.1427,1.2046},{2.8366,-3.0525,1.2012}};

/* One 2D weight (rows·cols, row-major): Nesterov (m←μm+g, upd=g+μm) → 5-iter Newton–Schulz over
   `MUON_C` → decoupled wd, ascent form `(1−lr·wd)·W + lr·scale·ortho`. `gscale` pre-scales the
   gradient (the mean ÷N; matches `unflattenMLPGrad`). Writes nW (rows·cols) and nM (rows·cols). */
static void muon_mat_cpu(int rows, int cols, const double* W, const double* G, const double* M,
                         double* nW, double* nM, double gscale, double lr, double wd, double mu, double eps) {
  long n = (long)rows*cols; int mn = rows<cols?rows:cols;
  double* u  = (double*)malloc(sizeof(double)*n);
  double* X  = (double*)malloc(sizeof(double)*n);
  double* Xt = (double*)malloc(sizeof(double)*n);
  double* A  = (double*)malloc(sizeof(double)*mn*mn);
  double* P  = (double*)malloc(sizeof(double)*n);
  double* Q  = (double*)malloc(sizeof(double)*n);
  for(long i=0;i<n;i++){ double gi=G[i]*gscale; nM[i]=mu*M[i]+gi; u[i]=gi+mu*nM[i]; }   /* Nesterov */
  double s=0.0;                                                                          /* ‖upd‖ (two-level) */
  for(int i=0;i<rows;i++){ double rs=0.0; for(int j=0;j<cols;j++){ double v=u[(long)i*cols+j]; rs=rs+v*v; } s=s+rs; }
  double inv=1.0/(sqrt(s)+eps);
  for(long i=0;i<n;i++) X[i]=inv*u[i];
  for(int it=0;it<5;it++){
    double a=MUON_C[it][0], b=MUON_C[it][1], c=MUON_C[it][2];
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) Xt[(long)j*rows+i]=X[(long)i*cols+j];   /* Xᵀ */
    if(rows<=cols){                                                                          /* A=X·Xᵀ; P=A·X; Q=A·P */
      for(int i=0;i<rows;i++) for(int j=0;j<rows;j++){ double t=0.0; for(int l=0;l<cols;l++) t=t+X[(long)i*cols+l]*Xt[(long)l*rows+j]; A[(long)i*rows+j]=t; }
      for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){ double t=0.0; for(int l=0;l<rows;l++) t=t+A[(long)i*rows+l]*X[(long)l*cols+j]; P[(long)i*cols+j]=t; }
      for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){ double t=0.0; for(int l=0;l<rows;l++) t=t+A[(long)i*rows+l]*P[(long)l*cols+j]; Q[(long)i*cols+j]=t; }
    } else {                                                                                /* A=Xᵀ·X; P=X·A; Q=P·A */
      for(int i=0;i<cols;i++) for(int j=0;j<cols;j++){ double t=0.0; for(int l=0;l<rows;l++) t=t+Xt[(long)i*rows+l]*X[(long)l*cols+j]; A[(long)i*cols+j]=t; }
      for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){ double t=0.0; for(int l=0;l<cols;l++) t=t+X[(long)i*cols+l]*A[(long)l*cols+j]; P[(long)i*cols+j]=t; }
      for(int i=0;i<rows;i++) for(int j=0;j<cols;j++){ double t=0.0; for(int l=0;l<cols;l++) t=t+P[(long)i*cols+l]*A[(long)l*cols+j]; Q[(long)i*cols+j]=t; }
    }
    for(long i=0;i<n;i++) X[i]=a*X[i]+b*P[i]+c*Q[i];
  }
  double scale=sqrt(fmax(1.0,(double)rows/(double)cols)); double c1=1.0-lr*wd, c2=lr*scale;
  for(long i=0;i<n;i++) nW[i]=c1*W[i]+c2*X[i];
  free(u);free(X);free(Xt);free(A);free(P);free(Q);
}

/* One 1D param (bias): Nesterov + decoupled wd, no orthogonalization. */
static void stepvec_cpu(int n, const double* b, const double* g, const double* m,
                        double* nb, double* nm, double gscale, double lr, double wd, double mu){
  for(int i=0;i<n;i++){ double gi=g[i]*gscale; double newm=mu*m[i]+gi; double upd=gi+mu*newm; nb[i]=b[i]*(1.0-lr*wd)+lr*upd; nm[i]=newm; }
}

/* Whole-MLP Muon step. params/grad/mom flat [W1(H·D)|b1(H)|W2(O·H)|b2(O)] (f64). `grad` is the RAW
   SUMMED minibatch gradient (as `mlpPPOGradBatchBlasFFI` returns); `gscale` (= 1/N) makes it the mean.
   Returns [newParams(P); newMom(P)] (size 2·P). Bit-exact with `normalizeAdv→mlpPPOGradBatchBlas→applyMuon`. */
LEAN_EXPORT lean_obj_res lean_ffi_muon_step_mlp(
    lean_obj_arg pa, lean_obj_arg ga,
    size_t H, size_t D, size_t O, double gscale, double lr, double wd, double mu, double eps){
  size_t P = H*D + H + O*H + O;
  size_t oW1=0, ob1=H*D, oW2=H*D+H, ob2=H*D+H+O*H;
  /* pa is the combined [params(P); mom(P)] buffer; mom is the second half (no Lean split/recombine) */
  const double* pp = lean_float_array_cptr(pa);
  const double* gg = lean_float_array_cptr(ga);
  const double* mm = pp + P;
  lean_object* Oo = lean_alloc_sarray(sizeof(double), 2*P, 2*P);
  double* out = lean_float_array_cptr(Oo); double* nmo = out + P;
  muon_mat_cpu((int)H,(int)D, pp+oW1, gg+oW1, mm+oW1, out+oW1, nmo+oW1, gscale, lr, wd, mu, eps);
  stepvec_cpu((int)H,        pp+ob1, gg+ob1, mm+ob1, out+ob1, nmo+ob1, gscale, lr, wd, mu);
  muon_mat_cpu((int)O,(int)H, pp+oW2, gg+oW2, mm+oW2, out+oW2, nmo+oW2, gscale, lr, wd, mu, eps);
  stepvec_cpu((int)O,        pp+ob2, gg+ob2, mm+ob2, out+ob2, nmo+ob2, gscale, lr, wd, mu);
  lean_dec(pa); lean_dec(ga);
  return Oo;
}
