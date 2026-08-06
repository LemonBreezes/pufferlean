/* Native CUDA kernels for the puffer trainer's GPU training step (the nvcc-compiled layer).
   Compiled by nvcc via the `puffercudaObj` Lake target and linked into the `puffer` exe alongside
   the plain-C FFI objects. Each `extern "C"` launcher has a Lean `@[extern]` twin in
   `Puffer/Float/CUDA.lean`; each is verified against the machine-checked Lean oracle (`verify-*-gpu`).

   These run in DOUBLE precision to stay tight against the f64 Lean oracle (V-Trace is a sequential
   scan → bit-exact; Muon's Newton–Schulz is f64 cuBLAS GEMMs → ~1e-13 tolerance). Production speed
   (f32/bf16 tensor cores) is a later precision-policy choice; correctness comes first. */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>   /* cp.async staging pipeline in the fused rollout forward */
#include <mma.h>                         /* tf32 tensor-core layer stages in the fused rollout forward */
#include <lean/lean.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <pthread.h>          /* persistent-thread env-step pool in the native rollout driver */
#include <unistd.h>           /* sysconf: core count for the adaptive rollout buffer count */
#include <dlfcn.h>            /* NVML query for the dashboard + the dlopen'd env-plugin ABI */
#include "puffer_handle.h"   /* the dlopen'd env plugin ABI, for the native rollout driver */
/* env-gated (PUFFER_TS_PROFILE) wall-clock stamp for the train-step phase breakdown. */
static double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (double)ts.tv_sec*1e3 + (double)ts.tv_nsec*1e-6; }
/* Device schedule policy, set BEFORE the context exists. On this virtualized GPU the default (Auto)
   sync waits cost ~123us/step in the wide rollout (native timers) against 24us of GPU work — the
   interrupt/yield wake path dominates. SPIN keeps the polling on-core (a few worker threads on a
   32-core box). PUFFER_CUDA_SYNC = s(pin, default) | y(ield) | b(locking) | a(uto). */
__attribute__((constructor)) static void mg_sched_flags(void){
  const char* e=getenv("PUFFER_CUDA_SYNC");
  unsigned f=cudaDeviceScheduleSpin;
  if(e){ if(e[0]=='y') f=cudaDeviceScheduleYield;
    else if(e[0]=='b') f=cudaDeviceScheduleBlockingSync;
    else if(e[0]=='a') f=cudaDeviceScheduleAuto; }
  cudaSetDeviceFlags(f);
}

/* --- M0: build-integration self-test ---------------------------------------------------------
   Proves an nvcc-compiled __global__ kernel compiles, links through Lean's lld, and runs on the
   GPU: returns Y[i] = 2·i + 1 computed on the device (CPU fallback if no device). */
__global__ void k_selftest(double* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 2.0 * (double)i + 1.0;
}

extern "C" LEAN_EXPORT lean_obj_res lean_cuda_selftest(size_t n) {
  lean_object* Yo = lean_alloc_sarray(sizeof(double), n, n);
  double* Y = lean_float_array_cptr(Yo);
  double* d = NULL;
  if (n > 0 && cudaMalloc((void**)&d, sizeof(double)*n) == cudaSuccess) {
    int block = 256, grid = (int)((n + block - 1) / block);
    k_selftest<<<grid, block>>>(d, (long)n);
    cudaDeviceSynchronize();
    cudaMemcpy(Y, d, sizeof(double)*n, cudaMemcpyDeviceToHost);
    cudaFree(d);
  } else {
    for (size_t i = 0; i < n; i++) Y[i] = 2.0*(double)i + 1.0;
  }
  return Yo;
}

/* --- M1: V-Trace advantage (GPU) -------------------------------------------------------------
   One thread per segment (row); sequential backward scan along the horizon, in f64 with the SAME
   op order as `VecTrain.computePuffAdvantageV` and no FMA (--fmad=false) → BIT-EXACT vs the oracle.
   Layout: B rows × T horizon, row-major. Per row, adv[T-1]=0 and for t=T-2..0:
     nnt = terminal[t]?0:1;  rho=min(imp[t],rhoClip);  c=min(imp[t],cClip);  r=clamp(reward[t],-1,1)
     delta = rho·(r + gamma·values[t+1]·nnt − values[t]);  last = delta + gamma·lam·c·last·nnt
   (the vec-kernel delta — ρ scales the whole TD error — matching what GPU PufferLib actually runs). */
__global__ void k_vtrace(const double* rewards, const double* values, const double* terms,
                         const double* imps, double* adv, int B, int T,
                         double gamma, double lam, double rhoClip, double cClip) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= B) return;
  const double* rw = rewards + (long)row*T; const double* vv = values + (long)row*T;
  const double* tm = terms   + (long)row*T; const double* im = imps   + (long)row*T;
  double* ad = adv + (long)row*T;
  ad[T-1] = 0.0;
  double last = 0.0;
  for (int t = T-2; t >= 0; t--) {
    double nnt = (tm[t] != 0.0) ? 0.0 : 1.0;
    double imp = im[t];
    double rho = imp < rhoClip ? imp : rhoClip;
    double c   = imp < cClip   ? imp : cClip;
    double rr = rw[t];
    double r  = rr > 1.0 ? 1.0 : (rr < -1.0 ? -1.0 : rr);
    double delta = rho * (r + gamma * vv[t+1] * nnt - vv[t]);
    last = delta + gamma * lam * c * last * nnt;
    ad[t] = last;
  }
}

/* rewards/values/terminals/importance are B·T row-major (Lean Transition convention: reward[t],
   terminal[t] at index t, external values[t]). Returns advantages[B·T]. CPU-fallback if no device. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_vtrace(
    lean_obj_arg rew, lean_obj_arg val, lean_obj_arg term, lean_obj_arg imp,
    size_t B, size_t T, double gamma, double lam, double rhoClip, double cClip) {
  const double* hR = lean_float_array_cptr(rew);
  const double* hV = lean_float_array_cptr(val);
  const double* hT = lean_float_array_cptr(term);
  const double* hI = lean_float_array_cptr(imp);
  size_t n = B*T;
  lean_object* Ao = lean_alloc_sarray(sizeof(double), n, n);
  double* hA = lean_float_array_cptr(Ao);
  double *dR=NULL,*dV=NULL,*dT=NULL,*dI=NULL,*dA=NULL;
  int ok = (B>0 && T>0
    && cudaMalloc((void**)&dR,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dV,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dT,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dI,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dA,sizeof(double)*n)==cudaSuccess);
  if (ok) {
    cudaMemcpy(dR,hR,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,hV,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dT,hT,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,hI,sizeof(double)*n,cudaMemcpyHostToDevice);
    int block=128, grid=(int)((B+block-1)/block);
    k_vtrace<<<grid,block>>>(dR,dV,dT,dI,dA,(int)B,(int)T,gamma,lam,rhoClip,cClip);
    cudaDeviceSynchronize();
    cudaMemcpy(hA,dA,sizeof(double)*n,cudaMemcpyDeviceToHost);
  } else {   /* CPU fallback: identical scan */
    for (size_t row=0; row<B; row++) {
      const double* rw=hR+row*T; const double* vv=hV+row*T; const double* tm=hT+row*T; const double* im=hI+row*T;
      double* ad=hA+row*T; if (T>0) ad[T-1]=0.0; double last=0.0;
      for (long t=(long)T-2; t>=0; t--) {
        double nnt=(tm[t]!=0.0)?0.0:1.0; double imv=im[t];
        double rho=imv<rhoClip?imv:rhoClip; double c=imv<cClip?imv:cClip;
        double rr=rw[t]; double r=rr>1.0?1.0:(rr<-1.0?-1.0:rr);
        double delta=rho*(r+gamma*vv[t+1]*nnt-vv[t]);
        last=delta+gamma*lam*c*last*nnt; ad[t]=last;
      }
    }
  }
  if(dR)cudaFree(dR);if(dV)cudaFree(dV);if(dT)cudaFree(dT);if(dI)cudaFree(dI);if(dA)cudaFree(dA);
  lean_dec(rew); lean_dec(val); lean_dec(term); lean_dec(imp);
  return Ao;
}

/* --- M1b: V-Trace advantage for the MinGRU trainer (GPU) --------------------------------------
   The MinGRU trainer uses a DIFFERENT V-Trace variant than k_vtrace above (verified against its own
   oracle `vtraceMinGRUFlat`, itself a copy of the trainer's original per-segment scan):
     - delta = rho·r + gamma·V'·nnt − V   (rho scales ONLY the reward, not the whole TD error)
     - the LAST step is bootstrapped with bootVals[row] (V(s_T)), so the scan runs t = T-1 .. 0
       (k_vtrace instead forces adv[T-1]=0 and scans T-2..0).
   One thread per segment; f64 sequential backward scan in the SAME op order as the Lean closure, no
   FMA (--fmad=false) ⇒ BIT-EXACT. Layout B rows × T horizon, row-major; bootVals is B. */
__global__ void k_vtrace_mingru(const double* rewards, const double* values, const double* terms,
                                const double* imps, const double* bootv, double* adv, int B, int T,
                                double gamma, double lam, double rhoClip, double cClip) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= B) return;
  const double* rw = rewards + (long)row*T; const double* vv = values + (long)row*T;
  const double* tm = terms   + (long)row*T; const double* im = imps   + (long)row*T;
  double* ad = adv + (long)row*T;
  double last = 0.0;
  for (int t = T-1; t >= 0; t--) {
    double nnt   = (tm[t] > 0.5) ? 0.0 : 1.0;                 /* terminal := termCol > 0.5 */
    double vNext = (t+1 < T) ? vv[t+1] : bootv[row];          /* last step bootstraps V(s_T) */
    double imp = im[t];
    double rho = imp < rhoClip ? imp : rhoClip;
    double c   = imp < cClip   ? imp : cClip;
    double rr = rw[t];
    double r  = rr > 1.0 ? 1.0 : (rr < -1.0 ? -1.0 : rr);
    double delta = rho * r + gamma*vNext*nnt - vv[t];         /* rho on the reward only */
    last = delta + gamma*lam*c*last*nnt;
    ad[t] = last;
  }
}

/* rewards/values/terminals/importance are B·T row-major; bootv is B (V at the segment's final obs).
   Returns advantages[B·T]. CPU-fallback if no device (identical scan). Bit-exact vs vtraceMinGRUFlat. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_vtrace_mingru(
    lean_obj_arg rew, lean_obj_arg val, lean_obj_arg term, lean_obj_arg imp, lean_obj_arg bv,
    size_t B, size_t T, double gamma, double lam, double rhoClip, double cClip) {
  const double* hR = lean_float_array_cptr(rew);
  const double* hV = lean_float_array_cptr(val);
  const double* hT = lean_float_array_cptr(term);
  const double* hI = lean_float_array_cptr(imp);
  const double* hB = lean_float_array_cptr(bv);
  size_t n = B*T;
  lean_object* Ao = lean_alloc_sarray(sizeof(double), n, n);
  double* hA = lean_float_array_cptr(Ao);
  double *dR=NULL,*dV=NULL,*dT=NULL,*dI=NULL,*dB=NULL,*dA=NULL;
  int ok = (B>0 && T>0
    && cudaMalloc((void**)&dR,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dV,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dT,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dI,sizeof(double)*n)==cudaSuccess
    && cudaMalloc((void**)&dB,sizeof(double)*B)==cudaSuccess
    && cudaMalloc((void**)&dA,sizeof(double)*n)==cudaSuccess);
  if (ok) {
    cudaMemcpy(dR,hR,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,hV,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dT,hT,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,hI,sizeof(double)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeof(double)*B,cudaMemcpyHostToDevice);
    int block=128, grid=(int)((B+block-1)/block);
    k_vtrace_mingru<<<grid,block>>>(dR,dV,dT,dI,dB,dA,(int)B,(int)T,gamma,lam,rhoClip,cClip);
    cudaDeviceSynchronize();
    cudaMemcpy(hA,dA,sizeof(double)*n,cudaMemcpyDeviceToHost);
  } else {   /* CPU fallback: identical scan */
    for (size_t row=0; row<B; row++) {
      const double* rw=hR+row*T; const double* vv=hV+row*T; const double* tm=hT+row*T; const double* im=hI+row*T;
      double* ad=hA+row*T; double last=0.0;
      for (long t=(long)T-1; t>=0; t--) {
        double nnt=(tm[t]>0.5)?0.0:1.0; double vNext=(t+1<(long)T)?vv[t+1]:hB[row];
        double imv=im[t]; double rho=imv<rhoClip?imv:rhoClip; double c=imv<cClip?imv:cClip;
        double rr=rw[t]; double r=rr>1.0?1.0:(rr<-1.0?-1.0:rr);
        double delta=rho*r+gamma*vNext*nnt-vv[t];
        last=delta+gamma*lam*c*last*nnt; ad[t]=last;
      }
    }
  }
  if(dR)cudaFree(dR);if(dV)cudaFree(dV);if(dT)cudaFree(dT);if(dI)cudaFree(dI);if(dB)cudaFree(dB);if(dA)cudaFree(dA);
  lean_dec(rew); lean_dec(val); lean_dec(term); lean_dec(imp); lean_dec(bv);
  return Ao;
}

/* ================= Device-resident minibatch prep (PufferLib's all-on-GPU train step) =================
   The rollout's scalar columns are H2D'd to device globals ONCE per update; then V-Trace, the scalar
   gather, and the value/ratio iterate all run on-device — no per-minibatch host round-trips (the old host
   path re-H2D'd the columns for V-Trace, gathered on the CPU, and iterated in interpreted Lean). Only tiny
   things cross the bus: Σ|adv| per segment (for host sampling), segIdx/mbPrio, and the gradient. All
   reductions are single-thread/sequential so results are BIT-IDENTICAL to the host path. */
static double *g_dcRew,*g_dcTerm,*g_dcAct,*g_dcLogp,*g_dcVal0,*g_dcBoot,*g_dcValue,*g_dcRatio,*g_dcAdv,*g_dcL;
static size_t g_dcSz=0; static long g_dcN=0,g_dcT=0; static int g_dc_valid=0;
/* MULTI-DISCRETE: the action column is the ONLY K-wide one (N·T·K); every other column stays N·T. It
   therefore gets its own size guard/allocation (g_dcActSz) while the nine scalar columns keep the shared
   one. g_dcK records the width the columns currently hold — consumers (the BPTT's device-column gather)
   refuse a mismatch rather than reading a K=1 column as if it were K-wide. K=1 is byte-for-byte the old
   behaviour (same ten buffers, same sizes; only the malloc grouping differs). */
static size_t g_dcActSz=0; static long g_dcK=1;
static int dc_alloc_k(size_t NT, size_t K){ size_t b=NT*sizeof(double);   /* all scalar slots N·T (boot/L use first N) */
  if(K<1) K=1;
  double** ps[9]={&g_dcRew,&g_dcTerm,&g_dcLogp,&g_dcVal0,&g_dcBoot,&g_dcValue,&g_dcRatio,&g_dcAdv,&g_dcL};
  if(g_dcSz<b){ for(int i=0;i<9;i++){ if(*ps[i]) cudaFree(*ps[i]); if(cudaMalloc((void**)ps[i],b)!=cudaSuccess){ *ps[i]=NULL; return 0; } } g_dcSz=b; }
  { size_t ab=NT*K*sizeof(double);
    if(g_dcActSz<ab){ if(g_dcAct) cudaFree(g_dcAct);
      if(cudaMalloc((void**)&g_dcAct,ab)!=cudaSuccess){ g_dcAct=NULL; g_dcActSz=0; return 0; } g_dcActSz=ab; } }
  g_dcK=(long)K; return 1; }
static int dc_alloc(size_t NT){ return dc_alloc_k(NT,1); }
__global__ void k_fill1(double* d,long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=1.0; }
/* Device-DIRECT column fill from the rollout (retires the prep H2D): the rollout already has act/logp/val
   on-device (the sampler's compact dO) and rew/term in pinned staging — scatter them straight into the
   resident columns per step instead of D2H → host col scatter → 10.5MB pageable re-upload per update.
   Same doubles ⇒ BIT-IDENTICAL to the prep path. g_dc_fromroll: consumed by lean_cuda_mg_cols_ready so
   Lean skips its slice-building + prep call only when THIS update's rollout actually stamped the columns. */
static int g_dc_fromroll=0;
/* Counts the BPTT/PPO minibatches since the last rollout stamped the device columns, plus the
   rollout index; reset per rollout and bumped per minibatch (rollout bookkeeping). */
static long g_mbSinceRoll=0, g_rollIdx=-1;
/* ---- LOSS SURFACING for the `--log` dashboard --------------------------------------------------
   Off by default (zero cost). When the trainer passes `--log` it calls lean_cuda_mg_loss_enable(1),
   and the BPTT grad then reduces the 7 dashboard losses (policy/value/entropy/total/old_kl/kl/
   clipfrac) into g_mgLoss each minibatch — a pure read-only D2H reduction that never writes any
   training buffer, so it cannot perturb determinism. lean_cuda_mg_read_losses copies g_mgLoss out.
   NON-static (external linkage): this is the ONE shared dashboard-loss channel — the feed-forward
   resident MLP/MD/Cont whole-update steps (ff_surface_losses, below) and the batched LSTM BPTT grad
   in pufferblas.c (which declares these `extern`) reduce into the same g_mgLoss via the same toggle,
   so lean_cuda_mg_read_losses serves every trainer. Only one trainer runs per process, so there is
   no cross-writer race. */
int g_mgLossOn=0;
double g_mgLoss[7]={0,0,0,0,0,0,0};
/* ---- CONVENTION (default ON; PUFFER_MG_KEEP_ROLL_STATE=1 restores the old threading) -----------
   Zero the carried rollout state at the START of every rollout — i.e. PufferLib's own convention
   (torch_pufferl.py rollouts(): `self.state = tuple(torch.zeros_like(s) ...)`), under which the
   rollout's h and forward_train's h both begin each horizon-length segment at 0 and the PPO ratio
   at epoch-0/mb-0 is therefore EXACTLY 1. We used to thread h across updates forever (moba never
   writes terminals, so k_mg_reset never fired), which made oldLogp and newLogp come from two
   different state conventions and caused moba's peak-then-collapse (see the ROOT CAUSE commit).
   DEFAULT ON so every MinGRU env matches PufferLib; set PUFFER_MG_KEEP_ROLL_STATE=1 to restore the
   pre-fix threading (e.g. to regenerate the old single-discrete bit-identity anchors). */
static inline int mg_zero_roll_state(void){
  static int v=-1; if(v<0){ const char* e=getenv("PUFFER_MG_KEEP_ROLL_STATE");
    v = (e&&e[0]&&e[0]!='0') ? 0 : 1; } return v; }
/* ---- resident chaining (Conn-lite): the rollout keeps obs (pinned ping-pong, by parity) and the
   recurrent state (dSa, device) RESIDENT across updates, so the per-update Lean/FFI boundary sheds its
   fat crossings: the ~NT·(D+5)+… full return (~270MB at squared@4096) shrinks to [rewCol;termCol]
   (4MB) on logging updates and EMPTY otherwise, and the f64 finalObs/finalState→obs0/state0 round
   trips vanish. BIT-IDENTICAL by construction: the removed round trips were exact widen/narrow
   identities (f32→f64→f32, u8→f64→u8). Protocol: update 0 passes real obs0/state0 (full return, C
   stamps chain-capable); Lean queries lean_cuda_mg_chain_ready once, then passes EMPTY obs0/state0.
   Gated PUFFER_MG_CHAIN (default ON; =0 restores the legacy per-update round trip). */
/* `md` tags WHICH driver left the stamp (0 single-discrete / 1 multi-discrete). N/D/LH alone do not
   distinguish them, and the two arms hold different things resident (the MD one also owns the K-wide
   device columns), so the MD driver refuses a stamp that is not its own rather than rolling out from
   another arm's residency. */
typedef struct { int valid; int md; int pc; long N,D,LH; const void* hA; const void* hB; const void* dSa; } mgchain_t;
static mgchain_t g_mgchain;
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_chain_ready(size_t N, size_t D, size_t LH, lean_obj_arg w){
  (void)w;
  int r=(g_mgchain.valid && g_mgchain.N==(long)N && g_mgchain.D==(long)D && g_mgchain.LH==(long)LH)?1:0;
  return lean_io_result_mk_ok(lean_box(r));
}
__global__ void k_mg_cols_alv(const double* dO, double* act, double* logp, double* val0, double* value,
    long rb, long nb, long s, long T){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nb) return;
  long row=(rb+i)*T+s; double v=dO[2*nb+i];
  act[row]=dO[i]; logp[row]=dO[nb+i]; val0[row]=v; value[row]=v; }
/* whole-update rew/term column scatter: ONE launch reading the pinned [T·N] planes zero-copy (coalesced
   full-word reads) — replaces T·nbuf per-step H2Ds + scatter kernels that clogged every cycle's stream. */
__global__ void k_mg_cols_rt_all(const double* rewP, const double* termP, double* dRew, double* dTerm,
    long N, long T){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=N*T) return;
  long s=i/N, e=i%N, row=e*T+s; dRew[row]=rewP[i]; dTerm[row]=termP[i]; }
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_cols_ready(size_t N, size_t T, lean_obj_arg w){
  (void)w;
  int r=(g_dc_fromroll && g_dc_valid && g_dcN==(long)N && g_dcT==(long)T);
  g_dc_fromroll=0;                                   /* consume: one rollout per prep-skip */
  return lean_io_result_mk_ok(lean_box(r?1:0)); }
/* Σ_t|adv[e,t]| per segment (one thread/segment, sequential t) — the host prioSample's inner sum, on-device. */
__global__ void k_adv_l1(const double* adv,double* out,int N,int T){ int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=N) return;
  const double* a=adv+(long)e*T; double s=0.0; for(int t=0;t<T;t++) s+=fabs(a[t]); out[e]=s; }

/* `K` = action components per row (1 single-discrete / K multi-discrete heads); `acta` is N·T·K, row
   `e·T+s` holding its K head actions contiguously. K=1 is the original call, unchanged. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_prep(
    lean_obj_arg rewa, lean_obj_arg terma, lean_obj_arg acta, lean_obj_arg logpa, lean_obj_arg vala, lean_obj_arg boota,
    size_t N, size_t T, size_t K){
  size_t NT=N*T; g_dc_valid=0; g_dcN=(long)N; g_dcT=(long)T; if(K<1) K=1;
  g_mbSinceRoll=0; g_rollIdx++;                        /* ratio-debug bookkeeping (no effect otherwise) */
  if(dc_alloc_k(NT,K)){
    cudaMemcpy(g_dcRew, lean_float_array_cptr(rewa), NT*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcTerm,lean_float_array_cptr(terma),NT*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcAct, lean_float_array_cptr(acta), NT*K*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcLogp,lean_float_array_cptr(logpa),NT*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcVal0,lean_float_array_cptr(vala), NT*8, cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcBoot,lean_float_array_cptr(boota),N*8,  cudaMemcpyHostToDevice);
    cudaMemcpy(g_dcValue,g_dcVal0,NT*8,cudaMemcpyDeviceToDevice);     /* valueBuf := valCol */
    k_fill1<<<(int)((NT+255)/256),256>>>(g_dcRatio,(long)NT);         /* ratioBuf := 1 */
    cudaDeviceSynchronize(); g_dc_valid=1;
  }
  lean_dec(rewa);lean_dec(terma);lean_dec(acta);lean_dec(logpa);lean_dec(vala);lean_dec(boota);
  return lean_io_result_mk_ok(lean_box(0));
}
/* Enable/disable per-minibatch loss surfacing for the `--log` dashboard (see g_mgLoss above). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_loss_enable(uint8_t on, lean_obj_arg w){
  (void)w; g_mgLossOn = on ? 1 : 0; return lean_io_result_mk_ok(lean_box(0));
}
/* Copy the 7 most-recent dashboard losses out (policy,value,entropy,total,old_kl,kl,clipfrac). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_read_losses(lean_obj_arg w){
  (void)w; lean_object* o=lean_alloc_sarray(sizeof(double),7,7);
  double* p=lean_float_array_cptr(o); for(int i=0;i<7;i++) p[i]=g_mgLoss[i];
  return lean_io_result_mk_ok(o);
}

/* ---- NVML via dlopen: microsecond GPU%/VRAM for the dashboard header, replacing the ~25ms
   nvidia-smi subprocess (which sat on the critical path, GPU idle during it — ~1.5% wall even
   cached). dlopen rather than link-time to avoid a hard build dependency on the driver lib and the
   Lean toolchain's lld/libc-stub quirks; this is what PufferLib does via pynvml. The two struct
   layouts read below (nvmlUtilization_t {uint gpu, memory}; nvmlMemory_t {ull total, free, used})
   are ABI-stable across NVML versions. Returns [gpu%, vram_used_GB, vram_total_GB]; zeros if NVML is
   unavailable, so the dashboard degrades to blank cells rather than failing. */
static int g_nvml=0;                 /* 0 untried, 1 ok, -1 unavailable */
static void* g_nvmlDev=NULL;
static int (*p_nvmlUtil)(void*, void*)=NULL;
static int (*p_nvmlMem)(void*, void*)=NULL;
static void nvml_try_init(void){
  void* h=dlopen("libnvidia-ml.so.1", RTLD_NOW|RTLD_GLOBAL);
  if(!h) h=dlopen("libnvidia-ml.so", RTLD_NOW|RTLD_GLOBAL);
  if(!h){ g_nvml=-1; return; }
  int (*p_init)(void)=(int(*)(void))dlsym(h,"nvmlInit_v2");
  int (*p_hb)(unsigned,void**)=(int(*)(unsigned,void**))dlsym(h,"nvmlDeviceGetHandleByIndex_v2");
  p_nvmlUtil=(int(*)(void*,void*))dlsym(h,"nvmlDeviceGetUtilizationRates");
  p_nvmlMem =(int(*)(void*,void*))dlsym(h,"nvmlDeviceGetMemoryInfo");
  g_nvml = (p_init && p_hb && p_nvmlUtil && p_nvmlMem && p_init()==0 && p_hb(0,&g_nvmlDev)==0) ? 1 : -1;
}
extern "C" LEAN_EXPORT lean_obj_res lean_nvml_stats(lean_obj_arg w){
  (void)w; double g=0.0,vu=0.0,vt=0.0;
  if(g_nvml==0) nvml_try_init();
  if(g_nvml==1){
    unsigned int util[2]={0,0};                 /* nvmlUtilization_t {gpu, memory} */
    if(p_nvmlUtil(g_nvmlDev,util)==0) g=(double)util[0];
    unsigned long long mem[3]={0,0,0};          /* nvmlMemory_t {total, free, used} */
    if(p_nvmlMem(g_nvmlDev,mem)==0){ vt=(double)mem[0]/1073741824.0; vu=(double)mem[2]/1073741824.0; }
  }
  lean_object* o=lean_alloc_sarray(sizeof(double),3,3);
  double* p=lean_float_array_cptr(o); p[0]=g; p[1]=vu; p[2]=vt;
  return lean_io_result_mk_ok(o);
}
/* V-Trace on the device columns → g_dcAdv, and returns Σ_t|adv| per segment (N doubles) for host sampling. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_vtrace(
    size_t N, size_t T, double gamma, double lam, double rhoClip, double cClip){
  lean_object* Lo=lean_alloc_sarray(sizeof(double),N,N); double* hL=lean_float_array_cptr(Lo);
  if(g_dc_valid){
    int block=128,grid=(int)((N+block-1)/block);
    k_vtrace_mingru<<<grid,block>>>(g_dcRew,g_dcValue,g_dcTerm,g_dcRatio,g_dcBoot,g_dcAdv,(int)N,(int)T,gamma,lam,rhoClip,cClip);
    k_adv_l1<<<grid,block>>>(g_dcAdv,g_dcL,(int)N,(int)T);
    cudaMemcpy(hL,g_dcL,N*8,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<N;i++) hL[i]=0.0;
  return lean_io_result_mk_ok(Lo);
}
/* ===== DEVICE prioritized sampling (the last per-minibatch host round-trip) =======================
   Replicates lean_ffi_prio_sample's math op-for-op ON DEVICE: prioW=exp(a·log(L1+1e-12)),
   probs=(w+1e-6)/(Σw+1e-6), deterministic chunked cumsum, splitmix64 draws (verbatim constants),
   lower_bound search, IS weights — writing segIdx/mbPrio into RESIDENT buffers the BPTT consumes with
   no H2D. TOLERANCE-class vs the host path (device exp/log differ from glibc at ulps ⇒ boundary draws
   can pick a neighboring segment) — deterministic run-to-run; PUFFER_MG_DEVPRIO=0 restores host
   sampling. The Σ|adv| D2H and the segIdx/mbPrio H2D both disappear. */
static int g_prio_fresh=0; static long g_prioN=0, g_prioB=0;
static void* bg(int i, size_t bytes);            /* persistent device cache (defined with the BPTT) */
static inline int ceildiv(long a, int b);
__global__ void k_prio_w(const double* L1, double* w, double a, long N){
  long e=(long)blockIdx.x*blockDim.x+threadIdx.x; if(e<N) w[e]=exp(a*log(L1[e]+1e-12)); }
__global__ void k_prio_sum(const double* w, double* s, long N){   /* deterministic single-block tree */
  __shared__ double sh[256]; int t=threadIdx.x;
  double p=0.0; for(long i=t;i<N;i+=blockDim.x) p+=w[i]; sh[t]=p; __syncthreads();
  for(int st=128;st>0;st>>=1){ if(t<st) sh[t]+=sh[t+st]; __syncthreads(); }
  if(t==0) s[0]=sh[0]; }
#define PRIO_C 64
__global__ void k_prio_scan_part(const double* w, const double* s, double* pp, double* cum, double* ctot, long N){
  int c=blockIdx.x; if(threadIdx.x) return;
  long R=(N+PRIO_C-1)/PRIO_C, r0=(long)c*R, r1=(r0+R>N)?N:(r0+R);
  double denom=s[0]+1e-6, acc=0.0;
  for(long e=r0;e<r1;e++){ double p=(w[e]+1e-6)/denom; pp[e]=p; acc+=p; cum[e]=acc; }
  ctot[c]=acc; }
__global__ void k_prio_scan_comb(double* ctot){
  if(threadIdx.x||blockIdx.x) return;
  double acc=0.0; for(int c=0;c<PRIO_C;c++){ double t=ctot[c]; ctot[c]=acc; acc+=t; } ctot[PRIO_C]=acc; }
__global__ void k_prio_scan_add(double* cum, const double* ctot, long N){
  long e=(long)blockIdx.x*blockDim.x+threadIdx.x; if(e>=N) return;
  long R=(N+PRIO_C-1)/PRIO_C; cum[e]+=ctot[e/R]; }
__global__ void k_prio_draw(const double* cum, const double* pp, const double* ctot, double* segR, double* mpR,
    long N, long B, double annealBeta, unsigned long long rng){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=B) return;
  double total=ctot[PRIO_C];
  unsigned long long z=rng+(unsigned long long)(i+1)*0x9E3779B97F4A7C15ULL;
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; z=z^(z>>31);
  double u=(double)(z>>11)/9007199254740992.0*total;
  long L=0,R=N; while(L<R){ long mid=(L+R)>>1; if(cum[mid]>=u) R=mid; else L=mid+1; }
  long idx=(L<N)?L:(N-1);
  segR[i]=(double)idx;
  mpR[i]=exp((-annealBeta)*log((double)N*pp[idx]+1e-12)); }
/* IO Bool: run device V-Trace + prioritized sampling, stamping resident segIdx/mbPrio for the BPTT.
   Returns 0 (Lean falls back to the host two-call path) when gated off or buffers unavailable. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mg_vtrace_prio(
    size_t N, size_t T, double gamma, double lam, double rhoClip, double cClip,
    size_t mbSegs, double prioAlpha, double annealBeta, uint64_t rng, lean_obj_arg w){
  (void)w;
  static int gate=-1; if(gate<0){ const char* e=getenv("PUFFER_MG_DEVPRIO"); gate=(e==NULL||e[0]!='0'); }
  g_prio_fresh=0;
  double *dPw=(double*)bg(71,8*(size_t)N), *dPp=(double*)bg(72,8*(size_t)N), *dCum=(double*)bg(73,8*(size_t)N);
  double *dCt=(double*)bg(74,8*(PRIO_C+1)), *dSegR=(double*)bg(75,8*mbSegs), *dMpR=(double*)bg(76,8*mbSegs);
  double *dS=(double*)bg(77,8);
  int ok=(gate && g_dc_valid && g_dcN==(long)N && g_dcT==(long)T && dPw&&dPp&&dCum&&dCt&&dSegR&&dMpR&&dS);
  if(ok){
    int block=128,grid=(int)((N+block-1)/block), Bk=256;
    k_vtrace_mingru<<<grid,block>>>(g_dcRew,g_dcValue,g_dcTerm,g_dcRatio,g_dcBoot,g_dcAdv,(int)N,(int)T,gamma,lam,rhoClip,cClip);
    k_adv_l1<<<grid,block>>>(g_dcAdv,g_dcL,(int)N,(int)T);
    k_prio_w<<<ceildiv((long)N,Bk),Bk>>>(g_dcL,dPw,prioAlpha,(long)N);
    k_prio_sum<<<1,256>>>(dPw,dS,(long)N);
    k_prio_scan_part<<<PRIO_C,1>>>(dPw,dS,dPp,dCum,dCt,(long)N);
    k_prio_scan_comb<<<1,1>>>(dCt);
    k_prio_scan_add<<<ceildiv((long)N,Bk),Bk>>>(dCum,dCt,(long)N);
    k_prio_draw<<<ceildiv((long)mbSegs,Bk),Bk>>>(dCum,dPp,dCt,dSegR,dMpR,(long)N,(long)mbSegs,annealBeta,(unsigned long long)rng);
    g_prio_fresh=1; g_prioN=(long)N; g_prioB=(long)mbSegs;
  }
  return lean_io_result_mk_ok(lean_box(ok?1:0));
}

/* mean/std of the sampled minibatch's adv — PARALLEL two-pass atomic reduction (fast; reduction order
   differs from the host so this is bit-TOLERANT, matching the iterate's device-exp). ms=[mean,std,sum,sq]. */
/* adv mean/std over the sampled minibatch — DETERMINISTIC block-tree reduction (was per-element f64
   atomicAdd: 81µs/launch of serialized atomics AND scheduler-order-dependent rounding — it passed the
   determinism checks only by luck). Fixed chunking + fixed tree + fixed final fold ⇒ reproducible.
   sq=0: Σ adv → ms[0]=mean; sq=1: Σ (adv−mean)² → ms[1]=std. */
__global__ void k_mg_advpart(const double* adv,const double* seg,int B,int T,const double* ms,int sq,double* part){
  __shared__ double sh[256];
  long n=(long)B*T, i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  double v=0.0;
  if(i<n){ int t=(int)(i/B),bi=(int)(i%B); double a=adv[(long)seg[bi]*T+t];
    if(sq){ double d=a-ms[0]; v=d*d; } else v=a; }
  sh[threadIdx.x]=v; __syncthreads();
  for(int s=128;s>0;s>>=1){ if((int)threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x==0) part[blockIdx.x]=sh[0]; }
__global__ void k_mg_ms_fin(double* ms,const double* part,int nBlk,int nb,int sq){
  if(threadIdx.x||blockIdx.x) return;
  double s=0.0; for(int b=0;b<nBlk;b++) s+=part[b];
  double d=(double)(nb>0?nb:1);
  if(sq) ms[1]=sqrt(s/d); else ms[0]=s/d; }
/* gather the 6 device scalars for the sampled minibatch into timestep-major dst[t·B+bi] (matches the host gather). */
/* `K` = action components per row: the act column is K-wide (row·K+h), everything else is scalar. K=1
   collapses the loop to the original single store. */
__global__ void k_mg_gather_scal(const double* adv,const double* value,const double* act,const double* logp,const double* term,
    const double* seg,const double* mbPrio,const double* ms,
    double* dAct,double* dAdv,double* dRet,double* dOld,double* dTrm,double* dOv, int B,int T,int K){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*T) return;
  int t=(int)(idx/B), bi=(int)(idx%B); long e=(long)seg[bi], si=e*T+t, os=(long)t*B+bi;
  double a=adv[si], v=value[si], stde=ms[1]+1e-8;
  for(int hh=0;hh<K;hh++) dAct[os*(long)K+hh]=act[si*(long)K+hh];
  dAdv[os]=mbPrio[bi]*(a-ms[0])/stde; dRet[os]=a+v; dOld[os]=logp[si]; dOv[os]=v; dTrm[os]=term[si]>0.5?1.0:0.0; }
/* iterate value/ratio from the BPTT's new_logp/new_value. The serial form (T threads looping bi in order)
   was 930µs/call — 13% of train. Parallel two-pass version, BIT-IDENTICAL: the in-order loop's final write
   for a segment comes from the LARGEST bi that sampled it, so pass 1 records that bi per segment (atomicMax
   — a max is order-independent, unlike a float sum) and pass 2 lets only that writer store. */
__global__ void k_mg_lastw(int* lastw, const double* seg, int B){
  int bi=blockIdx.x*blockDim.x+threadIdx.x; if(bi>=B) return;
  atomicMax(&lastw[(long)seg[bi]], bi); }
__global__ void k_mg_iterate(double* value,double* ratio,const double* logp,const float* newlp,const float* newval,const double* seg,const int* lastw,int B,int T){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*T) return;
  int t=(int)(idx/B), bi=(int)(idx%B); long e=(long)seg[bi];
  if(lastw[e]!=bi) return;
  long si=e*T+t, os=(long)t*B+bi;
  ratio[si]=exp((double)newlp[os]-logp[si]); value[si]=(double)newval[os]; }

/* --- M2: Muon step for a 2D weight matrix (GPU) ----------------------------------------------
   Nesterov momentum → Newton–Schulz orthogonalization (5 quintic iters, muonCoeffs) → decoupled
   weight decay, ascent form — matching `Puffer.Float.Muon.stepMat` op-for-op. The Newton–Schulz
   matmuls use a NAIVE f64 kernel that accumulates over the shared index in the SAME order as Lean's
   `matmul` (`s += A[i][l]·B[l][j]`); with --fmad=false that makes the whole step BIT-EXACT vs the
   oracle — stronger than tolerance, and it sidesteps the row-major↔col-major cuBLAS layout hazard.
   (Production speed would swap in cuBLAS at a tolerance; correctness first.) */
static const double MUON_COEFFS[5][3] = {
  {4.0848,-6.8946,2.9270},{3.9505,-6.3029,2.6377},{3.7418,-5.5913,2.3037},
  {2.8769,-3.1427,1.2046},{2.8366,-3.0525,1.2012}};
__constant__ double D_MUON_COEFFS[5][3] = {
  {4.0848,-6.8946,2.9270},{3.9505,-6.3029,2.6377},{3.7418,-5.5913,2.3037},
  {2.8769,-3.1427,1.2046},{2.8366,-3.0525,1.2012}};   /* device copy: __constant__ (host statics are invisible to device code) */

__global__ void k_nesterov1(double* nm, const double* m, const double* g, double mu, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) nm[i]=mu*m[i]+g[i]; }       /* m←μ·m+g   */
__global__ void k_nesterov2(double* u, const double* g, const double* nm, double mu, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) u[i]=g[i]+mu*nm[i]; }        /* g+μ·m     */
/* 1/(‖x‖+eps) via a single-block PARALLEL reduction (launch <<<1,256>>>). Was <<<1,1>>> two-level fold
   (one thread, ~14% of GPU-train). Flat tree-order sum ≈ the fold to ~n·ε (numerically equivalent; the
   Frobenius norm is a scalar normalizer, order carries no meaning). */
__global__ void k_frobnorm_inv(double* out, const double* x, double eps, int rows, int cols){
  __shared__ double sh[256];
  long n=(long)rows*cols; int t=threadIdx.x, B=blockDim.x;
  double p=0.0; for(long i=t;i<n;i+=B){ double v=x[i]; p+=v*v; } sh[t]=p; __syncthreads();
  for(int s=B/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  if(t==0) out[0]=1.0/(sqrt(sh[0])+eps); }
__global__ void k_scale_by(double* dst, const double* src, const double* inv, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=inv[0]*src[i]; }
__global__ void k_matmul(double* C, const double* A, const double* B, int M, int K, int N){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*N) return;
  int i=(int)(idx/N), j=(int)(idx%N); double s=0.0;
  for(int l=0;l<K;l++) s=s+A[(long)i*K+l]*B[(long)l*N+j];      /* Lean matmul order, no FMA */
  C[idx]=s; }
__global__ void k_transpose(double* Xt, const double* X, int rows, int cols){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)rows*cols) return;
  int i=(int)(idx/cols), j=(int)(idx%cols); Xt[(long)j*rows+i]=X[(long)i*cols+j]; }
__global__ void k_lincomb3(double* X, double a, const double* Xs, double b, const double* Y,
                           double c, const double* Z, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) X[i]=a*Xs[i]+b*Y[i]+c*Z[i]; }
__global__ void k_finalize(double* nW, const double* W, const double* o, double c1, double c2, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) nW[i]=c1*W[i]+c2*o[i]; }

static inline int ceildiv(long a, int b){ return (int)((a + b - 1) / b); }

/* W/grad/mom are rows·cols row-major (f64). Returns [newW (rows·cols); newMom (rows·cols)] (size 2·n). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_muon_stepmat(
    lean_obj_arg Wa, lean_obj_arg Ga, lean_obj_arg Ma, size_t rows, size_t cols,
    double lr, double wd, double mu, double eps) {
  const double* hW=lean_float_array_cptr(Wa); const double* hG=lean_float_array_cptr(Ga);
  const double* hM=lean_float_array_cptr(Ma);
  size_t n = rows*cols;
  lean_object* Oo = lean_alloc_sarray(sizeof(double), 2*n, 2*n);
  double* hO = lean_float_array_cptr(Oo);
  size_t szA = (rows*rows > cols*cols ? rows*rows : cols*cols);
  double *dW=NULL,*dG=NULL,*dM=NULL,*dNM=NULL,*dU=NULL,*dX=NULL,*dXt=NULL,*dA=NULL,*dP=NULL,*dQ=NULL,*dNW=NULL,*dInv=NULL;
  int ok = (n>0
    && cudaMalloc((void**)&dW,8*n)==cudaSuccess && cudaMalloc((void**)&dG,8*n)==cudaSuccess
    && cudaMalloc((void**)&dM,8*n)==cudaSuccess && cudaMalloc((void**)&dNM,8*n)==cudaSuccess
    && cudaMalloc((void**)&dU,8*n)==cudaSuccess && cudaMalloc((void**)&dX,8*n)==cudaSuccess
    && cudaMalloc((void**)&dXt,8*n)==cudaSuccess && cudaMalloc((void**)&dA,8*szA)==cudaSuccess
    && cudaMalloc((void**)&dP,8*n)==cudaSuccess && cudaMalloc((void**)&dQ,8*n)==cudaSuccess
    && cudaMalloc((void**)&dNW,8*n)==cudaSuccess && cudaMalloc((void**)&dInv,8)==cudaSuccess);
  if (ok) {
    cudaMemcpy(dW,hW,8*n,cudaMemcpyHostToDevice); cudaMemcpy(dG,hG,8*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dM,hM,8*n,cudaMemcpyHostToDevice);
    int B=256; int gN=ceildiv((long)n,B);
    k_nesterov1<<<gN,B>>>(dNM,dM,dG,mu,(long)n);          /* newMom = μ·mom + grad     */
    k_nesterov2<<<gN,B>>>(dU,dG,dNM,mu,(long)n);          /* update = grad + μ·newMom  */
    k_frobnorm_inv<<<1,256>>>(dInv,dU,eps,(int)rows,(int)cols); /* inv = 1/(‖update‖+eps) */
    k_scale_by<<<gN,B>>>(dX,dU,dInv,(long)n);             /* X = inv · update          */
    int R=(int)rows, C=(int)cols;
    for (int it=0; it<5; it++) {
      double a=MUON_COEFFS[it][0], b=MUON_COEFFS[it][1], c=MUON_COEFFS[it][2];
      if (rows <= cols) {                                  /* A=X·Xᵀ[R,R]; AX; AAX      */
        k_transpose<<<ceildiv((long)n,B),B>>>(dXt,dX,R,C);
        k_matmul<<<ceildiv((long)R*R,B),B>>>(dA,dX,dXt,R,C,R);
        k_matmul<<<gN,B>>>(dP,dA,dX,R,R,C);
        k_matmul<<<gN,B>>>(dQ,dA,dP,R,R,C);
      } else {                                             /* A=Xᵀ·X[C,C]; XA; XAA      */
        k_transpose<<<ceildiv((long)n,B),B>>>(dXt,dX,R,C);
        k_matmul<<<ceildiv((long)C*C,B),B>>>(dA,dXt,dX,C,R,C);
        k_matmul<<<gN,B>>>(dP,dX,dA,R,C,C);
        k_matmul<<<gN,B>>>(dQ,dP,dA,R,C,C);
      }
      k_lincomb3<<<gN,B>>>(dX,a,dX,b,dP,c,dQ,(long)n);     /* X = a·X + b·(A?)  + c·(A?) */
    }
    double scale = sqrt(fmax(1.0, (double)rows/(double)cols));
    double c1 = 1.0 - lr*wd, c2 = lr*scale;
    k_finalize<<<gN,B>>>(dNW,dW,dX,c1,c2,(long)n);         /* newW = (1-lr·wd)·W + lr·scale·ortho */
    cudaDeviceSynchronize();
    cudaMemcpy(hO,      dNW,8*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(hO + n,  dNM,8*n,cudaMemcpyDeviceToHost);
  } else {                                                 /* CPU fallback: identical ops */
    double* nm=(double*)malloc(8*n); double* u=(double*)malloc(8*n); double* X=(double*)malloc(8*n);
    double* Xt=(double*)malloc(8*n); double* A=(double*)malloc(8*szA); double* P=(double*)malloc(8*n); double* Q=(double*)malloc(8*n);
    for(size_t i=0;i<n;i++){ nm[i]=mu*hM[i]+hG[i]; u[i]=hG[i]+mu*nm[i]; }
    double s=0.0; for(size_t i=0;i<rows;i++){double rs=0.0;for(size_t j=0;j<cols;j++){double v=u[i*cols+j];rs=rs+v*v;}s=s+rs;}
    double inv=1.0/(sqrt(s)+eps);
    for(size_t i=0;i<n;i++) X[i]=inv*u[i];
    int R=(int)rows,Cc=(int)cols;
    for(int it=0;it<5;it++){ double a=MUON_COEFFS[it][0],b=MUON_COEFFS[it][1],c=MUON_COEFFS[it][2];
      for(int i=0;i<R;i++)for(int j=0;j<Cc;j++) Xt[(long)j*R+i]=X[(long)i*Cc+j];
      if(rows<=cols){ for(int i=0;i<R;i++)for(int j=0;j<R;j++){double t=0;for(int l=0;l<Cc;l++)t=t+X[(long)i*Cc+l]*Xt[(long)l*R+j];A[(long)i*R+j]=t;}
        for(int i=0;i<R;i++)for(int j=0;j<Cc;j++){double t=0;for(int l=0;l<R;l++)t=t+A[(long)i*R+l]*X[(long)l*Cc+j];P[(long)i*Cc+j]=t;}
        for(int i=0;i<R;i++)for(int j=0;j<Cc;j++){double t=0;for(int l=0;l<R;l++)t=t+A[(long)i*R+l]*P[(long)l*Cc+j];Q[(long)i*Cc+j]=t;} }
      else { for(int i=0;i<Cc;i++)for(int j=0;j<Cc;j++){double t=0;for(int l=0;l<R;l++)t=t+Xt[(long)i*R+l]*X[(long)l*Cc+j];A[(long)i*Cc+j]=t;}
        for(int i=0;i<R;i++)for(int j=0;j<Cc;j++){double t=0;for(int l=0;l<Cc;l++)t=t+X[(long)i*Cc+l]*A[(long)l*Cc+j];P[(long)i*Cc+j]=t;}
        for(int i=0;i<R;i++)for(int j=0;j<Cc;j++){double t=0;for(int l=0;l<Cc;l++)t=t+P[(long)i*Cc+l]*A[(long)l*Cc+j];Q[(long)i*Cc+j]=t;} }
      for(size_t i=0;i<n;i++) X[i]=a*X[i]+b*P[i]+c*Q[i]; }
    double scale=sqrt(fmax(1.0,(double)rows/(double)cols)); double c1=1.0-lr*wd,c2=lr*scale;
    for(size_t i=0;i<n;i++){ hO[i]=c1*hW[i]+c2*X[i]; hO[n+i]=nm[i]; }
    free(nm);free(u);free(X);free(Xt);free(A);free(P);free(Q);
  }
  if(dW)cudaFree(dW);if(dG)cudaFree(dG);if(dM)cudaFree(dM);if(dNM)cudaFree(dNM);if(dU)cudaFree(dU);
  if(dX)cudaFree(dX);if(dXt)cudaFree(dXt);if(dA)cudaFree(dA);if(dP)cudaFree(dP);if(dQ)cudaFree(dQ);if(dNW)cudaFree(dNW);if(dInv)cudaFree(dInv);
  lean_dec(Wa); lean_dec(Ga); lean_dec(Ma);
  return Oo;
}

/* --- M3: MLP PPO gradient (GPU) --------------------------------------------------------------
   GPU port of `lean_ffi_mlp_ppo_grad_batch_blas` (ffi/pufferblas.c): forward (2 GEMMs) → the per-row
   PPO objective backward `ppo_dout` (softmax/logp/clip/entropy/value; the one irreducible kernel) →
   backward (3 GEMMs) + relu-mask + bias column-sums → the flat summed gradient g[P] (W1[H·D],b1[H],
   W2[O·H],b2[O]). Matmuls are naive row-major kernels (Lean/BLAS order) with an OPTIONAL bf16 round of
   the operands (`bf16` flag): bf16=1 is the PufferLib-precision DEFAULT (bf16 storage, f32 accumulate,
   verify ~1e-2 round-then-compare); bf16=0 is the f32 tight cross-check (~1e-5 vs the f64 oracle). */
/* Cached cuBLAS handle (lazy; single-threaded FFI). */
static cublasHandle_t g_cu = NULL; static int g_cu_ok = 0;
static int g_cu_ws_ok = 0;   /* main handle got its 32MB workspace — precondition for the cloned side/pool
                                handles (algo selection is workspace-gated; a clone must match EXACTLY) */
static cublasHandle_t cu_handle(void){
  if(!g_cu_ok){
    if(cublasCreate(&g_cu)==CUBLAS_STATUS_SUCCESS){
      /* persistent workspace + tensor-op math: avoid per-call workspace alloc / heuristic churn that
         dominates the many tiny serial GEMMs of the Muon Newton–Schulz (torch caches these plans). */
      cublasSetMathMode(g_cu, CUBLAS_DEFAULT_MATH);
      void* ws=NULL; if(cudaMalloc(&ws, 32*1024*1024)==cudaSuccess){ cublasSetWorkspace(g_cu, ws, 32*1024*1024); g_cu_ws_ok=1; }
      else cudaGetLastError();
      g_cu_ok=1;
    } else g_cu_ok=-1;
  }
  return g_cu_ok==1?g_cu:NULL; }

/* f32-buffer GEMM on tensor cores: bf ? bf16 multiply (COMPUTE_32F_FAST_16BF — PufferLib precision,
   the fast default) : full f32 (COMPUTE_32F — the tight cross-check). f32 A/B/C, f32 accumulate; the
   FAST_16BF path rounds operands to bf16 for the tensor-core MAC without any manual bf16 buffers. The
   (opA,opB,m,n,k,lda,ldb,ldc) are the row-major→col-major mappings of the CPU-BLAS oracle's cblas_dgemms. */
static cublasStatus_t gemm32(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc, int bf){
  float al=1.0f, be=0.0f;
  cublasComputeType_t ct = bf ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
  return cublasGemmEx(h, opA, opB, m, n, k, &al, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb,
                      &be, C, CUDA_R_32F, ldc, ct, CUBLAS_GEMM_DEFAULT);
}

__global__ void k_relu_bias(float* H, const float* pre, const float* b, int N, int H_){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*H_) return;
  float v=pre[idx]+b[idx%H_]; H[idx]=v>0.0f?v:0.0f; }
__global__ void k_add_bias(float* Y, const float* pre, const float* b, int N, int O){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*O) return; Y[idx]=pre[idx]+b[idx%O]; }
__global__ void k_dz1_mask(float* dZ, const float* dH, const float* H, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dZ[i]=(H[i]>0.0f)?dH[i]:0.0f; }
/* gb[j]=Σ_n Mx[n·J+j] — one BLOCK per column, block-reduce over the N rows (launch <<<J,256>>>). Was
   thread-per-column (only J threads, each a sequential N-row sum; ~12% of GPU-train when J is small, N big). */
__global__ void k_colsum(float* gb, const float* Mx, int N, int J){
  __shared__ float sh[256];
  int j=blockIdx.x; if(j>=J) return; int t=threadIdx.x, B=blockDim.x;
  float p=0.0f; for(int n=t;n<N;n+=B) p+=Mx[(long)n*J+j]; sh[t]=p; __syncthreads();
  for(int s=B/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  if(t==0) gb[j]=sh[0]; }

/* Value-head gradient with PufferLib's CLIPPED value loss (torch_pufferl.py:329-332):
     v_clipped = v_old + clamp(v_new − v_old, −vfClip, +vfClip)
     v_loss    = ½·max((v_new − ret)², (v_clipped − ret)²)
   d(v_loss)/dv_new is (v_new − ret) when the unclipped branch wins OR the clamp is inactive, else 0
   (the clamp kills the gradient through v_clipped). `ovA == NULL` or `vfClip <= 0` ⇒ the plain
   unclipped ½(v_new − ret)², i.e. exactly the pre-fix behaviour — so the verified legacy call sites
   that pass (NULL, 0) stay bit-identical. Identical formula to `k_mg_ppo_b`'s (the MinGRU path). */
__device__ __forceinline__ float d_vloss_grad(float vnew, float ret, const double* ovA, long i, float vfClip){
  if(vfClip<=0.0f || !ovA) return vnew-ret;
  float vold=(float)ovA[i]; float dd=vnew-vold;
  float vc=vold+(dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd));
  float du=(vnew-ret)*(vnew-ret), cc=(vc-ret)*(vc-ret);
  if(du>=cc) return vnew-ret;
  if(dd>-vfClip && dd<vfClip) return vnew-ret;
  return 0.0f;
}

/* MULTI-DISCRETE per-row PPO backward: K categorical heads (sizes headSizes[K], O=Σsizes+1). The joint
   log-prob is Σ_h log p_h(a_h); one PPO clip on the joint ratio; the gradient decomposes per head
   (each head's logits get dPol·(onehot(a_h)−p_h) + entropy). K=1 reduces exactly to k_ppo_dout. actA is
   N×K (row-major: agent n's head h at actA[n·K+h]); oldA is the joint old log-prob. Value head = out[O-1]. */
__global__ void k_ppo_dout_md(const float* Out, const double* actA, const double* advA, const double* retA,
                              const double* oldA, const double* ovA, float* dOut, int N, int K,
                              const int* headSizes, int O,
                              float vfCoef, float entCoef, float clipEps, float vfClip){
  int n=(int)((long)blockIdx.x*blockDim.x+threadIdx.x); if(n>=N) return;
  const float* out=Out+(long)n*O; float* dout=dOut+(long)n*O;
  float adv=(float)advA[n], ret=(float)retA[n], oldLogp=(float)oldA[n];
  float jointLogp=0.0f; int off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh]; int a=(int)actA[(long)n*K+hh];
    float se=0.0f; for(int k=0;k<sz;k++) se+=expf(out[off+k]); jointLogp += out[off+a]-logf(se); off+=sz; }
  float ratio=expf(jointLogp-oldLogp); float lo=1.0f-clipEps, hi=1.0f+clipEps;
  float ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); float surr1=adv*ratio, surr2=adv*ratioC;
  float dPol; if(surr1<=surr2) dPol=adv*ratio; else { float cg=(lo<ratio&&ratio<hi)?1.0f:0.0f; dPol=adv*cg*ratio; }
  off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh]; int a=(int)actA[(long)n*K+hh];
    float se=0.0f; for(int k=0;k<sz;k++) se+=expf(out[off+k]); float lse=logf(se);
    float pout=0.0f; float pk[64];
    for(int k=0;k<sz;k++){ pk[k]=expf(out[off+k]-lse); pout+=pk[k]*out[off+k]; }
    for(int k=0;k<sz;k++){ float dp=dPol*(((k==a)?1.0f:0.0f)-pk[k]); float de=entCoef*pk[k]*(pout-out[off+k]); dout[off+k]=dp+de; }
    off+=sz; }
  dout[O-1]=-vfCoef*d_vloss_grad(out[O-1],ret,ovA,n,vfClip);
}

/* one thread per row: dOut[n,·] from Out (matches pufferblas.c ppo-objective backward, f32 exp/log). */
__global__ void k_ppo_dout(const float* Out, const double* actA, const double* advA, const double* retA,
                           const double* oldA, const double* ovA, float* dOut, int N, int A, int O,
                           float vfCoef, float entCoef, float clipEps, float vfClip){
  int n=(int)((long)blockIdx.x*blockDim.x+threadIdx.x); if(n>=N) return;
  const float* out=Out+(long)n*O; float* dout=dOut+(long)n*O;
  int a=(int)actA[n]; float adv=(float)advA[n], ret=(float)retA[n], oldLogp=(float)oldA[n];
  float sumexp=0.0f; for(int k=0;k<A;k++) sumexp=sumexp+expf(out[k]); float lse=logf(sumexp);
  float pout=0.0f; float pk[64];
  for(int k=0;k<A;k++){ pk[k]=expf(out[k]-lse); pout=pout+pk[k]*out[k]; }
  float logpA=out[a]-lse; float ratio=expf(logpA-oldLogp); float lo=1.0f-clipEps, hi=1.0f+clipEps;
  float ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); float surr1=adv*ratio, surr2=adv*ratioC;
  float dPol; if(surr1<=surr2) dPol=adv*ratio; else { float cg=(lo<ratio&&ratio<hi)?1.0f:0.0f; dPol=adv*cg*ratio; }
  for(int k=0;k<A;k++){ float dp=dPol*(((k==a)?1.0f:0.0f)-pk[k]); float de=entCoef*pk[k]*(pout-out[k]); dout[k]=dp+de; }
  dout[A]=-vfCoef*d_vloss_grad(out[A],ret,ovA,n,vfClip);
}

/* params flat (f64): W1[H·D],b1[H],W2[O·H],b2[O] (O=A+1); obsB[N·D],acts/advs/rets/oldlps[N] (f64).
   bf16Flag: 1 = bf16 default (PufferLib precision), 0 = f32 tight cross-check. Returns g[P] (f64). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mlp_ppo_grad(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, size_t N, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, uint8_t bf16Flag) {
  size_t O=A+1, P=H*D+H+O*H+O;
  const double* pp=lean_float_array_cptr(pa); const double* Xd=lean_float_array_cptr(obsBa);
  const double* actd=lean_float_array_cptr(actsa); const double* advd=lean_float_array_cptr(advsa);
  const double* retd=lean_float_array_cptr(retsa); const double* oldd=lean_float_array_cptr(oldlpsa);
  const double* W1d=pp; const double* b1d=W1d+H*D; const double* W2d=b1d+H; const double* b2d=W2d+O*H;
  lean_object* go=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(go);
  int bf=(int)bf16Flag;
  cublasHandle_t h = cu_handle();
  /* host f32 stagings */
  float *hX=(float*)malloc(4*N*D),*hW1=(float*)malloc(4*H*D),*hW2=(float*)malloc(4*O*H),*hb1=(float*)malloc(4*H),*hb2=(float*)malloc(4*O);
  float *hgW1=(float*)malloc(4*H*D),*hgb1=(float*)malloc(4*H),*hgW2=(float*)malloc(4*O*H),*hgb2=(float*)malloc(4*O);
  double *hAc=(double*)malloc(8*N),*hAd=(double*)malloc(8*N),*hRe=(double*)malloc(8*N),*hOl=(double*)malloc(8*N);
  float *dX=NULL,*dW1=NULL,*dW2=NULL,*db1=NULL,*db2=NULL,*dH1=NULL,*dOut=NULL,*dPre=NULL,*ddOut=NULL,*ddH1=NULL,*ddZ=NULL,*dgW1=NULL,*dgb1=NULL,*dgW2=NULL,*dgb2=NULL;
  double *dac=NULL,*dad=NULL,*dre=NULL,*dol=NULL;
  int ok = (N>0 && h != NULL &&
    !cudaMalloc((void**)&dX,4*N*D) && !cudaMalloc((void**)&dW1,4*H*D) && !cudaMalloc((void**)&dW2,4*O*H) &&
    !cudaMalloc((void**)&db1,4*H) && !cudaMalloc((void**)&db2,4*O) && !cudaMalloc((void**)&dH1,4*N*H) &&
    !cudaMalloc((void**)&dOut,4*N*O) && !cudaMalloc((void**)&dPre,4*N*(H>O?H:O)) && !cudaMalloc((void**)&ddOut,4*N*O) &&
    !cudaMalloc((void**)&ddH1,4*N*H) && !cudaMalloc((void**)&ddZ,4*N*H) && !cudaMalloc((void**)&dgW1,4*H*D) &&
    !cudaMalloc((void**)&dgb1,4*H) && !cudaMalloc((void**)&dgW2,4*O*H) && !cudaMalloc((void**)&dgb2,4*O) &&
    !cudaMalloc((void**)&dac,8*N) && !cudaMalloc((void**)&dad,8*N) && !cudaMalloc((void**)&dre,8*N) && !cudaMalloc((void**)&dol,8*N));
  if (ok) {
    for(size_t i=0;i<N*D;i++) hX[i]=(float)Xd[i];
    for(size_t i=0;i<H*D;i++) hW1[i]=(float)W1d[i];
    for(size_t i=0;i<O*H;i++) hW2[i]=(float)W2d[i];
    for(size_t i=0;i<H;i++) hb1[i]=(float)b1d[i];
    for(size_t i=0;i<O;i++) hb2[i]=(float)b2d[i];
    for(size_t i=0;i<N;i++){ hAc[i]=actd[i]; hAd[i]=advd[i]; hRe[i]=retd[i]; hOl[i]=oldd[i]; }
    cudaMemcpy(dX,hX,4*N*D,cudaMemcpyHostToDevice); cudaMemcpy(dW1,hW1,4*H*D,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,4*O*H,cudaMemcpyHostToDevice); cudaMemcpy(db1,hb1,4*H,cudaMemcpyHostToDevice);
    cudaMemcpy(db2,hb2,4*O,cudaMemcpyHostToDevice);
    cudaMemcpy(dac,hAc,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dad,hAd,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dre,hRe,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dol,hOl,8*N,cudaMemcpyHostToDevice);
    int B=256;
    #define GRID(x) ceildiv((long)(x),B)
    /* forward: row-major Zpre[N,H]=Xb·W1ᵀ ⇒ cublas(OP_T,OP_N, H,N,D, W1,D, Xb,D, → dPre,H) */
    gemm32(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)H,(int)N,(int)D, dW1,(int)D, dX,(int)D, dPre,(int)H, bf);
    k_relu_bias<<<GRID(N*H),B>>>(dH1,dPre,db1,(int)N,(int)H);           /* H1 = relu(Zpre+b1) */
    gemm32(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)O,(int)N,(int)H, dW2,(int)H, dH1,(int)H, dPre,(int)O, bf); /* Outpre[N,O]=H1·W2ᵀ */
    k_add_bias<<<GRID(N*O),B>>>(dOut,dPre,db2,(int)N,(int)O);           /* Out = Outpre+b2  */
    k_ppo_dout<<<GRID(N),B>>>(dOut,dac,dad,dre,dol,NULL,ddOut,(int)N,(int)A,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,0.0f);
    k_colsum<<<(int)O,256>>>(dgb2,ddOut,(int)N,(int)O);                  /* gb2 = Σ_n dOut   */
    gemm32(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)H,(int)O,(int)N, dH1,(int)H, ddOut,(int)O, dgW2,(int)H, bf); /* gW2[O,H]=dOutᵀ·H1 */
    gemm32(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)H,(int)N,(int)O, dW2,(int)H, ddOut,(int)O, ddH1,(int)H, bf); /* dH1[N,H]=dOut·W2  */
    k_dz1_mask<<<GRID(N*H),B>>>(ddZ,ddH1,dH1,(long)N*H);                /* dZ1 = dH1⊙relu'  */
    k_colsum<<<(int)H,256>>>(dgb1,ddZ,(int)N,(int)H);                    /* gb1 = Σ_n dZ1    */
    gemm32(h, CUBLAS_OP_N, CUBLAS_OP_T, (int)D,(int)H,(int)N, dX,(int)D, ddZ,(int)H, dgW1,(int)D, bf);   /* gW1[H,D]=dZ1ᵀ·Xb  */
    #undef GRID
    cudaDeviceSynchronize();
    cudaMemcpy(hgW1,dgW1,4*H*D,cudaMemcpyDeviceToHost); cudaMemcpy(hgb1,dgb1,4*H,cudaMemcpyDeviceToHost);
    cudaMemcpy(hgW2,dgW2,4*O*H,cudaMemcpyDeviceToHost); cudaMemcpy(hgb2,dgb2,4*O,cudaMemcpyDeviceToHost);
    size_t o=0;
    for(size_t i=0;i<H*D;i++) g[o++]=(double)hgW1[i];
    for(size_t i=0;i<H;i++)   g[o++]=(double)hgb1[i];
    for(size_t i=0;i<O*H;i++) g[o++]=(double)hgW2[i];
    for(size_t i=0;i<O;i++)   g[o++]=(double)hgb2[i];
  } else { for(size_t i=0;i<P;i++) g[i]=0.0; }   /* (verify requires a device) */
  if(dX)cudaFree(dX);if(dW1)cudaFree(dW1);if(dW2)cudaFree(dW2);if(db1)cudaFree(db1);if(db2)cudaFree(db2);
  if(dH1)cudaFree(dH1);if(dOut)cudaFree(dOut);if(dPre)cudaFree(dPre);if(ddOut)cudaFree(ddOut);if(ddH1)cudaFree(ddH1);
  if(ddZ)cudaFree(ddZ);if(dgW1)cudaFree(dgW1);if(dgb1)cudaFree(dgb1);if(dgW2)cudaFree(dgW2);if(dgb2)cudaFree(dgb2);
  if(dac)cudaFree(dac);if(dad)cudaFree(dad);if(dre)cudaFree(dre);if(dol)cudaFree(dol);
  free(hX);free(hW1);free(hW2);free(hb1);free(hb2);free(hgW1);free(hgb1);free(hgW2);free(hgb2);free(hAc);free(hAd);free(hRe);free(hOl);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);
  return go;
}

/* --- M4: advantage normalize (GPU) -----------------------------------------------------------
   advN[i] = (adv[i] - mean) / (std + 1e-8), matching `normalizeAdv` (mean = Σadv/n, var = Σ(x-mean)²/n
   via sequential left-folds, std = √var). f64 + --fmad=false ⇒ BIT-EXACT. md[0]=mean, md[1]=std+1e-8. */
/* mean + std of adv[n] via a single-block PARALLEL reduction (launch <<<1,256>>>). Was <<<1,1>>> (one
   thread, memory-latency-bound, ~58% of GPU-train). The tree-order sum differs from a sequential fold by
   ~n·ε (numerically equivalent, not bit-exact — adv-norm is a normalization, order is not meaningful). */
__global__ void k_var_mean(const double* adv, double* md, int n){
  __shared__ double sh[256];
  int t=threadIdx.x, B=blockDim.x;
  double p=0.0; for(int i=t;i<n;i+=B) p+=adv[i]; sh[t]=p; __syncthreads();
  for(int s=B/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  double mean=sh[0]/(double)n; __syncthreads();
  double q=0.0; for(int i=t;i<n;i+=B){ double d=adv[i]-mean; q+=d*d; } sh[t]=q; __syncthreads();
  for(int s=B/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }
  if(t==0){ md[0]=mean; md[1]=sqrt(sh[0]/(double)n)+1e-8; } }
__global__ void k_normalize(double* advN, const double* adv, const double* md, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) advN[i]=(adv[i]-md[0])/md[1]; }

extern "C" LEAN_EXPORT lean_obj_res lean_cuda_adv_normalize(lean_obj_arg adva, size_t n){
  const double* hA=lean_float_array_cptr(adva);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),n,n); double* hO=lean_float_array_cptr(Oo);
  double *dA=NULL,*dO=NULL,*dMd=NULL;
  int ok=(n>0 && !cudaMalloc((void**)&dA,8*n) && !cudaMalloc((void**)&dO,8*n) && !cudaMalloc((void**)&dMd,16));
  if(ok){
    cudaMemcpy(dA,hA,8*n,cudaMemcpyHostToDevice);
    k_var_mean<<<1,256>>>(dA,dMd,(int)n);
    k_normalize<<<ceildiv((long)n,256),256>>>(dO,dA,dMd,(long)n);
    cudaDeviceSynchronize();
    cudaMemcpy(hO,dO,8*n,cudaMemcpyDeviceToHost);
  } else {   /* CPU fallback: identical folds */
    double s=0.0; for(size_t i=0;i<n;i++) s=s+hA[i]; double mean=s/(double)n;
    double v=0.0; for(size_t i=0;i<n;i++){double d=hA[i]-mean;v=v+d*d;} double denom=sqrt(v/(double)n)+1e-8;
    for(size_t i=0;i<n;i++) hO[i]=(hA[i]-mean)/denom;
  }
  if(dA)cudaFree(dA);if(dO)cudaFree(dO);if(dMd)cudaFree(dMd);
  lean_dec(adva);
  return Oo;
}

/* --- M5: one resident PPO+Muon training step on the GPU --------------------------------------
   Chains M4 (normalize) → M3 (gradient) → M2 (Muon) with the intermediates RESIDENT on the device —
   one upload (params/minibatch/momentum), one download (newParams/newMomentum), no host round-trip
   between phases. Verified against the CPU/Lean oracle (normalizeAdv → mlpPPOGradBatchBlas → applyMuon)
   by the --shadow check. Gradient GEMMs are bf16 tensor cores (bf16Flag) / f32; Muon runs f64 on the
   (f32→f64-widened) gradient, matching the f64 momentum + oracle. */
__global__ void k_f32_to_f64(double* dst, const float* src, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(double)src[i]; }
__global__ void k_f64_to_f32(float* dst, const double* src, long n){    /* widen resident f64 params → f32 */
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(float)src[i]; }
__global__ void k_scale_const(double* x, double c, long n){    /* x ← c·x (in place)  */
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=c*x[i]; }
/* on-device minibatch gather from resident columns: dst[k]=src[idx[k]] (rows of `cols` for the f32 obs). */
__global__ void k_gather_f32(float* dst, const float* src, const double* idx, int Nmb, int cols){
  long t=(long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(long)Nmb*cols) return;
  int k=(int)(t/cols), j=(int)(t%cols); dst[t]=src[(long)((int)idx[k])*cols+j]; }
__global__ void k_gather_f64(double* dst, const double* src, const double* idx, int Nmb){
  int k=(int)((long)blockIdx.x*blockDim.x+threadIdx.x); if(k>=Nmb) return; dst[k]=src[(int)idx[k]]; }
/* cols-wide f64 gather (the W-wide action column for the MD/Cont whole-update): dst[k·cols+j]=src[idx[k]·cols+j]. */
__global__ void k_gather_f64_wide(double* dst, const double* src, const double* idx, int Nmb, int cols){
  long t=(long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(long)Nmb*cols) return;
  int k=(int)(t/cols), j=(int)(t%cols); dst[t]=src[(long)((int)idx[k])*cols+j]; }
/* bias Muon (stepVec): newMom=μ·mom+g; upd=g+μ·newMom; newB=b·(1-lr·wd)+lr·upd */
__global__ void k_stepvec(double* nb, double* nm, const double* b, const double* g, const double* m,
    double lr, double wd, double mu, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){
    double newm=mu*m[i]+g[i]; double upd=g[i]+mu*newm; nb[i]=b[i]*(1.0-lr*wd)+lr*upd; nm[i]=newm; } }
/* graph-friendly twin: lr read from coef[0] (device) so a captured muon graph survives the per-update
   lr anneal — same double, bit-identical to k_stepvec. */
__global__ void k_stepvec_g(double* nb, double* nm, const double* b, const double* g, const double* m,
    const double* coef, double wd, double mu, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ double lr=coef[0];
    double newm=mu*m[i]+g[i]; double upd=g[i]+mu*newm; nb[i]=b[i]*(1.0-lr*wd)+lr*upd; nm[i]=newm; } }

/* Muon step for one f64 matrix on-device (launches the M2 kernels); scratch buffers supplied. */
static void muon_mat_dev(int rows, int cols, const double* dW, const double* dG, const double* dM,
    double* dNewW, double* dNewM, double lr, double wd, double mu, double eps,
    double* dU, double* dX, double* dXt, double* dA, double* dP, double* dQ, double* dInv){
  long n=(long)rows*cols; int B=256; int gN=ceildiv(n,B);
  k_nesterov1<<<gN,B>>>(dNewM,dM,dG,mu,n);
  k_nesterov2<<<gN,B>>>(dU,dG,dNewM,mu,n);
  k_frobnorm_inv<<<1,256>>>(dInv,dU,eps,rows,cols);
  k_scale_by<<<gN,B>>>(dX,dU,dInv,n);
  for(int it=0;it<5;it++){ double a=MUON_COEFFS[it][0],b=MUON_COEFFS[it][1],c=MUON_COEFFS[it][2];
    if(rows<=cols){ k_transpose<<<gN,B>>>(dXt,dX,rows,cols);
      k_matmul<<<ceildiv((long)rows*rows,B),B>>>(dA,dX,dXt,rows,cols,rows);
      k_matmul<<<gN,B>>>(dP,dA,dX,rows,rows,cols);
      k_matmul<<<gN,B>>>(dQ,dA,dP,rows,rows,cols);
    } else { k_transpose<<<gN,B>>>(dXt,dX,rows,cols);
      k_matmul<<<ceildiv((long)cols*cols,B),B>>>(dA,dXt,dX,cols,rows,cols);
      k_matmul<<<gN,B>>>(dP,dX,dA,rows,cols,cols);
      k_matmul<<<gN,B>>>(dQ,dP,dA,rows,cols,cols); }
    k_lincomb3<<<gN,B>>>(dX,a,dX,b,dP,c,dQ,n); }
  double scale=sqrt(fmax(1.0,(double)rows/(double)cols)); double c1=1.0-lr*wd,c2=lr*scale;
  k_finalize<<<gN,B>>>(dNewW,dW,dX,c1,c2,n);
}

/* --- bf16/f32 Newton–Schulz Muon (production path, bf16Flag==1): identical structure to
   muon_mat_dev, but the NS matmuls run through cuBLAS (gemm32, bf16 tensor cores) instead of the
   naive f64 k_matmul — 74% of the train step at N=8192 was that f64 NS. The f64 path above stays as
   the bit-exact oracle (verify-train-step-gpu f32 tier); this matches PufferLib's bf16 Muon (verify
   bf16 tier, tol 0.2). Nesterov/frobnorm/finalize stay f64 for the momentum + param state; only the
   orthogonalization iterates in f32/bf16 on scratch. -------------------------------------------- */
__global__ void k_scale_by_f32(float* dst, const double* src, const double* inv, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(float)(inv[0]*src[i]); }
__global__ void k_transpose_f32(float* Xt, const float* X, int rows, int cols){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)rows*cols) return;
  int i=(int)(idx/cols), j=(int)(idx%cols); Xt[(long)j*rows+i]=X[(long)i*cols+j]; }
__global__ void k_lincomb3_f32(float* X, float a, const float* Xs, float b, const float* Y,
                               float c, const float* Z, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) X[i]=a*Xs[i]+b*Y[i]+c*Z[i]; }
__global__ void k_finalize_f32(double* nW, const double* W, const float* o, double c1, double c2, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) nW[i]=c1*W[i]+c2*(double)o[i]; }
/* graph-friendly twin: c1/c2 read from a device coef block ([1]=c1, [2+ci]=c2 of matrix-segment ci) so a
   captured muon graph survives the per-update lr anneal — same doubles, bit-identical to k_finalize_f32. */
__global__ void k_finalize_f32_g(double* nW, const double* W, const float* o, const double* coef, int ci, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) nW[i]=coef[1]*W[i]+coef[2+ci]*(double)o[i]; }
/* row-major C[M×N] = A[M×K]·B[K×N] via cublasSgemm (col-major): swap A/B and M/N, both OP_N. The
   legacy Sgemm has far lower per-call host overhead than cublasGemmEx's algo heuristic — decisive for
   the Muon NS, a serial chain of ~30 tiny GEMMs where that overhead, not compute, dominates. */
static inline void ns_matmul_rm(cublasHandle_t h, float* C, const float* A, const float* B,
                                int M, int K, int N, int bf){
  float al=1.0f, be=0.0f; (void)bf;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &al, B, N, A, K, &be, C, N);
}
static void muon_mat_dev_bf(cublasHandle_t h, int rows, int cols, const double* dW, const double* dG,
    const double* dM, double* dNewW, double* dNewM, double lr, double wd, double mu, double eps,
    double* dUf, double* dInv, float* dX, float* dXt, float* dA, float* dP, float* dQ, int bf,
    cudaStream_t st=0,                           /* stream param: CUDA-graph capture needs a non-legacy stream */
    const double* coefDev=NULL, int coefIdx=0){  /* device coef block ([1]=c1, [2+i]=c2ᵢ): lets a captured
                                                    graph survive the per-update lr anneal (k_finalize_f32_g) */
  long n=(long)rows*cols; int B=256; int gN=ceildiv(n,B);
  k_nesterov1<<<gN,B,0,st>>>(dNewM,dM,dG,mu,n);
  k_nesterov2<<<gN,B,0,st>>>(dUf,dG,dNewM,mu,n);
  k_frobnorm_inv<<<1,256,0,st>>>(dInv,dUf,eps,rows,cols);
  k_scale_by_f32<<<gN,B,0,st>>>(dX,dUf,dInv,n);
  for(int it=0;it<5;it++){ float a=(float)MUON_COEFFS[it][0],b=(float)MUON_COEFFS[it][1],c=(float)MUON_COEFFS[it][2];
    if(rows<=cols){ k_transpose_f32<<<gN,B,0,st>>>(dXt,dX,rows,cols);
      ns_matmul_rm(h,dA,dX,dXt,rows,cols,rows,bf);     /* A = X·Xᵀ  */
      ns_matmul_rm(h,dP,dA,dX,rows,rows,cols,bf);      /* P = A·X   */
      ns_matmul_rm(h,dQ,dA,dP,rows,rows,cols,bf);      /* Q = A·P   */
    } else { k_transpose_f32<<<gN,B,0,st>>>(dXt,dX,rows,cols);
      ns_matmul_rm(h,dA,dXt,dX,cols,rows,cols,bf);     /* A = Xᵀ·X  */
      ns_matmul_rm(h,dP,dX,dA,rows,cols,cols,bf);      /* P = X·A   */
      ns_matmul_rm(h,dQ,dP,dA,rows,cols,cols,bf); }    /* Q = P·A   */
    k_lincomb3_f32<<<gN,B,0,st>>>(dX,a,dX,b,dP,c,dQ,n); }
  if(coefDev) k_finalize_f32_g<<<gN,B,0,st>>>(dNewW,dW,dX,coefDev,coefIdx,n);
  else { double scale=sqrt(fmax(1.0,(double)rows/(double)cols)); double c1=1.0-lr*wd,c2=lr*scale;
    k_finalize_f32<<<gN,B,0,st>>>(dNewW,dW,dX,c1,c2,n); }
}

/* Persistent train-step device-buffer cache: 33 slots grown on demand and reused across UPDATES, so
   the ~33 cudaMallocs are paid once for a whole training run (freed at process exit), not per step. */
#define NTS 44   /* +2: old-value column + its minibatch gather (value-loss clip); +1 gradclip scalar */
static void* g_ts[NTS]; static size_t g_tssz[NTS];
static void* ts_buf(int i, size_t bytes){
  if(g_tssz[i] < bytes){
    if(g_ts[i]) cudaFree(g_ts[i]);
    if(cudaMalloc(&g_ts[i], bytes) != cudaSuccess){ g_ts[i]=NULL; g_tssz[i]=0; return NULL; }
    g_tssz[i]=bytes;
  }
  return g_ts[i];
}
/* Device-resident obs trajectory shared between the buffered rollout (writer) and the resident trainer
   (reader) — see the definition of obstraj_buf / k_scatter_obs_traj below. Declared here so the trainer
   (above the rollout in this file) can read it. */
static float* g_dObsTraj; static size_t g_dObsTrajSz; static int g_dObsTraj_valid;

/* params/mom flat (f64): W1[H·D],b1[H],W2[O·H],b2[O] (O=A+1). obsB[N·D],acts/advRaw/rets/oldlps[N].
   Returns [newParams[P]; newMom[P]] (2·P). Whole step resident on-device; device buffers are cached
   across calls (ts_buf) so a training loop pays the mallocs once. Needs cuBLAS + a device. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_train_step(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advra,
    lean_obj_arg retsa, lean_obj_arg oldlpsa,
    size_t N, size_t H, size_t D, size_t A, double lr, double wd, double mu, double eps,
    double vfCoef, double entCoef, double clipEps, uint8_t bf16Flag) {
  size_t O=A+1, P=H*D+H+O*H+O; int bf=(int)bf16Flag;
  size_t oW1=0, ob1=H*D, oW2=H*D+H, ob2=H*D+H+O*H;
  /* pa is the combined [params(P); mom(P)] buffer; mom is the second half (avoids a Lean split/recombine) */
  const double* par=lean_float_array_cptr(pa); const double* Xd=lean_float_array_cptr(obsBa);
  const double* actd=lean_float_array_cptr(actsa); const double* advd=lean_float_array_cptr(advra);
  const double* retd=lean_float_array_cptr(retsa); const double* oldd=lean_float_array_cptr(oldlpsa);
  const double* momd=par+P;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),2*P,2*P); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  size_t SZ = (H*D>O*H?H*D:O*H); size_t md=(H>D?H:D); if(O>md) md=O; size_t SZA=md*md;
  float *hXf=(float*)malloc(4*N*D),*hW1f=(float*)malloc(4*H*D),*hW2f=(float*)malloc(4*O*H),*hb1f=(float*)malloc(4*H),*hb2f=(float*)malloc(4*O);
  double *hac=(double*)malloc(8*N),*har=(double*)malloc(8*N),*hre=(double*)malloc(8*N),*hol=(double*)malloc(8*N);
  /* cached device buffers (ts_buf), grown on demand, reused across updates — mallocs paid once */
  float *dXf=(float*)ts_buf(0,4*N*D),*dW1f=(float*)ts_buf(1,4*H*D),*dW2f=(float*)ts_buf(2,4*O*H),*db1f=(float*)ts_buf(3,4*H),*db2f=(float*)ts_buf(4,4*O);
  float *dH1=(float*)ts_buf(5,4*N*H),*dOut=(float*)ts_buf(6,4*N*O),*dPre=(float*)ts_buf(7,4*N*(H>O?H:O)),*ddOut=(float*)ts_buf(8,4*N*O),*ddH1=(float*)ts_buf(9,4*N*H),*ddZ=(float*)ts_buf(10,4*N*H);
  float *dgW1f=(float*)ts_buf(11,4*H*D),*dgb1f=(float*)ts_buf(12,4*H),*dgW2f=(float*)ts_buf(13,4*O*H),*dgb2f=(float*)ts_buf(14,4*O);
  double *dac=(double*)ts_buf(15,8*N),*dar=(double*)ts_buf(16,8*N),*dan=(double*)ts_buf(17,8*N),*dre=(double*)ts_buf(18,8*N),*dol=(double*)ts_buf(19,8*N),*dMd=(double*)ts_buf(20,16);
  double *dPar=(double*)ts_buf(21,8*P),*dMom=(double*)ts_buf(22,8*P),*dG=(double*)ts_buf(23,8*P),*dNP=(double*)ts_buf(24,8*P),*dNM=(double*)ts_buf(25,8*P);
  double *dU=(double*)ts_buf(26,8*SZ),*dXn=(double*)ts_buf(27,8*SZ),*dXt=(double*)ts_buf(28,8*SZ),*dAm=(double*)ts_buf(29,8*SZA),*dPn=(double*)ts_buf(30,8*SZ),*dQn=(double*)ts_buf(31,8*SZ),*dInv=(double*)ts_buf(32,8);
  /* f32 NS scratch for the bf16 Muon path (bf16Flag==1) */
  float *mX=(float*)ts_buf(33,4*SZ),*mXt=(float*)ts_buf(34,4*SZ),*mA=(float*)ts_buf(35,4*SZA),*mP=(float*)ts_buf(36,4*SZ),*mQ=(float*)ts_buf(37,4*SZ);
  int ok = (N>0 && h!=NULL && dXf&&dW1f&&dW2f&&db1f&&db2f&&dH1&&dOut&&dPre&&ddOut&&ddH1&&ddZ&&dgW1f&&dgb1f&&dgW2f&&dgb2f&&
    dac&&dar&&dan&&dre&&dol&&dMd&&dPar&&dMom&&dG&&dNP&&dNM&&dU&&dXn&&dXt&&dAm&&dPn&&dQn&&dInv&&mX&&mXt&&mA&&mP&&mQ);
  static int prof=-1; if(prof<0) prof=(getenv("PUFFER_TS_PROFILE")!=NULL);   /* cached: no per-step getenv */
  double t0=0,t1=0,t2=0,t3=0,t4=0; if(prof) t0=now_ms();
  if (ok) {
    for(size_t i=0;i<N*D;i++) hXf[i]=(float)Xd[i];
    for(size_t i=0;i<H*D;i++) hW1f[i]=(float)par[oW1+i];
    for(size_t i=0;i<O*H;i++) hW2f[i]=(float)par[oW2+i];
    for(size_t i=0;i<H;i++) hb1f[i]=(float)par[ob1+i];
    for(size_t i=0;i<O;i++) hb2f[i]=(float)par[ob2+i];
    for(size_t i=0;i<N;i++){ hac[i]=actd[i]; har[i]=advd[i]; hre[i]=retd[i]; hol[i]=oldd[i]; }
    cudaMemcpy(dXf,hXf,4*N*D,cudaMemcpyHostToDevice); cudaMemcpy(dW1f,hW1f,4*H*D,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2f,hW2f,4*O*H,cudaMemcpyHostToDevice); cudaMemcpy(db1f,hb1f,4*H,cudaMemcpyHostToDevice); cudaMemcpy(db2f,hb2f,4*O,cudaMemcpyHostToDevice);
    cudaMemcpy(dac,hac,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dar,har,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dre,hre,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dol,hol,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dPar,par,8*P,cudaMemcpyHostToDevice); cudaMemcpy(dMom,momd,8*P,cudaMemcpyHostToDevice);
    if(prof){ cudaDeviceSynchronize(); t1=now_ms(); }
    int B=256;
    #define GR(x) ceildiv((long)(x),B)
    k_var_mean<<<1,256>>>(dar,dMd,(int)N); k_normalize<<<GR(N),B>>>(dan,dar,dMd,(long)N);   /* advN (resident) */
    /* gradient (resident): advA = dan */
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)N,(int)D, dW1f,(int)D, dXf,(int)D, dPre,(int)H, bf);
    k_relu_bias<<<GR(N*H),B>>>(dH1,dPre,db1f,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,(int)N,(int)H, dW2f,(int)H, dH1,(int)H, dPre,(int)O, bf);
    k_add_bias<<<GR(N*O),B>>>(dOut,dPre,db2f,(int)N,(int)O);
    k_ppo_dout<<<GR(N),B>>>(dOut,dac,dan,dre,dol,NULL,ddOut,(int)N,(int)A,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,0.0f);
    k_colsum<<<(int)O,256>>>(dgb2f,ddOut,(int)N,(int)O);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,(int)N, dH1,(int)H, ddOut,(int)O, dgW2f,(int)H, bf);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,(int)N,(int)O, dW2f,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
    k_dz1_mask<<<GR(N*H),B>>>(ddZ,ddH1,dH1,(long)N*H);
    k_colsum<<<(int)H,256>>>(dgb1f,ddZ,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,(int)N, dXf,(int)D, ddZ,(int)H, dgW1f,(int)D, bf);
    /* widen gradient f32 → f64 into dG (resident) */
    k_f32_to_f64<<<GR(H*D),B>>>(dG+oW1,dgW1f,(long)H*D); k_f32_to_f64<<<GR(H),B>>>(dG+ob1,dgb1f,(long)H);
    k_f32_to_f64<<<GR(O*H),B>>>(dG+oW2,dgW2f,(long)O*H); k_f32_to_f64<<<GR(O),B>>>(dG+ob2,dgb2f,(long)O);
    /* MEAN gradient: the kernels SUM over the N-sample minibatch (matching the summed oracle); PPO/PufferLib
       average the loss, so divide by N. Muon matrices are scale-invariant (frobNorm cancels 1/N), so weight
       updates are unchanged; this is what keeps the raw (un-orthogonalized) bias update from exploding. */
    k_scale_const<<<GR(P),B>>>(dG, 1.0/(double)N, (long)P);
    if(prof){ cudaDeviceSynchronize(); t2=now_ms(); }
    /* Muon (resident): matrices W1,W2 via NS (bf=1 → bf16 cuBLAS, PufferLib speed; bf=0 → f64 naive,
       bit-exact oracle); biases via stepVec (f64, tiny — no matmul) */
    if(bf){
      muon_mat_dev_bf(h,(int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dNP+oW1,dNM+oW1, lr,wd,mu,eps, dU,dInv, mX,mXt,mA,mP,mQ, bf);
      k_stepvec<<<GR(H),B>>>(dNP+ob1,dNM+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
      muon_mat_dev_bf(h,(int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dNP+oW2,dNM+oW2, lr,wd,mu,eps, dU,dInv, mX,mXt,mA,mP,mQ, bf);
      k_stepvec<<<GR(O),B>>>(dNP+ob2,dNM+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
    } else {
      muon_mat_dev((int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dNP+oW1,dNM+oW1, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(H),B>>>(dNP+ob1,dNM+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
      muon_mat_dev((int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dNP+oW2,dNM+oW2, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(O),B>>>(dNP+ob2,dNM+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
    }
    #undef GR
    cudaDeviceSynchronize();
    if(prof) t3=now_ms();
    cudaMemcpy(out,   dNP,8*P,cudaMemcpyDeviceToHost);
    cudaMemcpy(out+P, dNM,8*P,cudaMemcpyDeviceToHost);
    if(prof){ t4=now_ms(); fprintf(stderr,
      "[ts N=%zu H=%zu D=%zu A=%zu bf=%d] upload=%.3f compute=%.3f muon=%.3f download=%.3f total=%.3f ms\n",
      N,H,D,A,bf, t1-t0,t2-t1,t3-t2,t4-t3,t4-t0); }
  } else { for(size_t i=0;i<2*P;i++) out[i]=0.0; }
  /* device buffers are cached in g_ts[] — NOT freed here (persist across the training run) */
  free(hXf);free(hW1f);free(hW2f);free(hb1f);free(hb2f);free(hac);free(har);free(hre);free(hol);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advra);lean_dec(retsa);lean_dec(oldlpsa);
  return Oo;
}

/* on-device gradclip kernels (defined with the MinGRU trainer below) — the resident MLP/wide steps
   below now use the same pair for PufferLib's `clip_grad_norm_`. */
__global__ void k_gclip_norm(double* sumsq, const double* g, double sc, long P);
__global__ void k_gclip_scale(double* out, const double* g, const double* sumsq, double maxNorm, double sc, long P);

/* --- Whole-update RESIDENT PPO+Muon: one FFI call runs ALL epochs×minibatches on the device ------------
   Uploads the SoA columns (obs NT·D, acts/adv/ret/olp NT), the per-epoch permutations (epochs·NT
   f64-encoded indices), and pm=[params;mom] ONCE; loops epochs×numMB on-device (gather minibatch →
   normalize → gradient → Muon, params/mom kept resident f64, updated IN PLACE), then downloads pm ONCE.
   Collapses the per-minibatch host↔device transfers to O(1) per update — the win in the large-batch
   regime (bench-train-step crossover). Same math as looping lean_cuda_train_step over the same perms;
   returns the new [params; mom] (2·P). Requires cuBLAS + a device (writes zeros otherwise). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_train_update(
    lean_obj_arg pmA, lean_obj_arg obsA, lean_obj_arg actsA, lean_obj_arg advA,
    lean_obj_arg retA, lean_obj_arg olpA, lean_obj_arg permA,
    size_t NT, size_t D, size_t H, size_t A, size_t epochs, size_t numMB,
    double lr, double wd, double mu, double eps, double vfCoef, double entCoef, double clipEps, uint8_t bf16Flag){
  size_t O=A+1, P=H*D+H+O*H+O; int bf=(int)bf16Flag;
  size_t oW1=0, ob1=H*D, oW2=H*D+H, ob2=H*D+H+O*H;
  const double* pmd=lean_float_array_cptr(pmA); const double* obsd=lean_float_array_cptr(obsA);
  const double* actd=lean_float_array_cptr(actsA); const double* advd=lean_float_array_cptr(advA);
  const double* retd=lean_float_array_cptr(retA); const double* oldd=lean_float_array_cptr(olpA);
  const double* permd=lean_float_array_cptr(permA);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),2*P,2*P); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  size_t SZ=(H*D>O*H?H*D:O*H); size_t md=(H>D?H:D); if(O>md) md=O; size_t SZA=md*md;
  /* per-minibatch working buffers (slots 0-23), sized for the MAX minibatch (≤ NT) */
  float *dXf=(float*)ts_buf(0,4*NT*D),*dW1f=(float*)ts_buf(1,4*H*D),*dW2f=(float*)ts_buf(2,4*O*H),*db1f=(float*)ts_buf(3,4*H),*db2f=(float*)ts_buf(4,4*O);
  float *dH1=(float*)ts_buf(5,4*NT*H),*dOut=(float*)ts_buf(6,4*NT*O),*dPre=(float*)ts_buf(7,4*NT*(H>O?H:O)),*ddOut=(float*)ts_buf(8,4*NT*O),*ddH1=(float*)ts_buf(9,4*NT*H),*ddZ=(float*)ts_buf(10,4*NT*H);
  float *dgW1f=(float*)ts_buf(11,4*H*D),*dgb1f=(float*)ts_buf(12,4*H),*dgW2f=(float*)ts_buf(13,4*O*H),*dgb2f=(float*)ts_buf(14,4*O);
  double *dac=(double*)ts_buf(15,8*NT),*dar=(double*)ts_buf(16,8*NT),*dan=(double*)ts_buf(17,8*NT),*dre=(double*)ts_buf(18,8*NT),*dol=(double*)ts_buf(19,8*NT),*dMd=(double*)ts_buf(20,16);
  double *dPar=(double*)ts_buf(21,8*P),*dMom=(double*)ts_buf(22,8*P),*dG=(double*)ts_buf(23,8*P);
  double *dU=(double*)ts_buf(26,8*SZ),*dXn=(double*)ts_buf(27,8*SZ),*dXt=(double*)ts_buf(28,8*SZ),*dAm=(double*)ts_buf(29,8*SZA),*dPn=(double*)ts_buf(30,8*SZ),*dQn=(double*)ts_buf(31,8*SZ),*dInv=(double*)ts_buf(32,8);
  /* resident columns + perm (slots 33-38) */
  float *dObsCol=(float*)ts_buf(33,4*NT*D);
  double *dActCol=(double*)ts_buf(34,8*NT),*dAdvCol=(double*)ts_buf(35,8*NT),*dRetCol=(double*)ts_buf(36,8*NT),*dOlpCol=(double*)ts_buf(37,8*NT),*dPerm=(double*)ts_buf(38,8*epochs*NT);
  double *dObsD=(double*)ts_buf(40,8*NT*D);              /* f64 obs staging → GPU-convert (no host f64→f32 loop) */
  int ok=(NT>0 && h!=NULL && dObsD && dXf&&dW1f&&dW2f&&db1f&&db2f&&dH1&&dOut&&dPre&&ddOut&&ddH1&&ddZ&&dgW1f&&dgb1f&&dgW2f&&dgb2f&&
    dac&&dar&&dan&&dre&&dol&&dMd&&dPar&&dMom&&dG&&dU&&dXn&&dXt&&dAm&&dPn&&dQn&&dInv&&dObsCol&&dActCol&&dAdvCol&&dRetCol&&dOlpCol&&dPerm);
  if(ok){
    cudaMemcpy(dObsD,obsd,8*NT*D,cudaMemcpyHostToDevice);  /* upload f64 obs, convert on GPU (was a 7.7M CPU cast/update) */
    k_f64_to_f32<<<ceildiv((long)NT*D,256),256>>>(dObsCol,dObsD,(long)NT*D);
    cudaMemcpy(dActCol,actd,8*NT,cudaMemcpyHostToDevice); cudaMemcpy(dAdvCol,advd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dRetCol,retd,8*NT,cudaMemcpyHostToDevice); cudaMemcpy(dOlpCol,oldd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dPerm,permd,8*epochs*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dPar,pmd,8*P,cudaMemcpyHostToDevice); cudaMemcpy(dMom,pmd+P,8*P,cudaMemcpyHostToDevice);
    int B=256;
    #define GR(x) ceildiv((long)(x),B)
    size_t mbSize=NT/numMB; if(mbSize<1) mbSize=1;
    for(size_t e=0;e<epochs;e++) for(size_t m=0;m<numMB;m++){
      size_t lo=m*mbSize; size_t hi=(m+1==numMB)?NT:(m+1)*mbSize; if(hi>NT) hi=NT;
      if(hi<=lo) continue;
      int Nmb=(int)(hi-lo); const double* idx=dPerm + e*NT + lo;
      k_f64_to_f32<<<GR(H*D),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<GR(H),B>>>(db1f,dPar+ob1,(long)H);
      k_f64_to_f32<<<GR(O*H),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<GR(O),B>>>(db2f,dPar+ob2,(long)O);
      k_gather_f32<<<GR((long)Nmb*D),B>>>(dXf,dObsCol,idx,Nmb,(int)D);
      k_gather_f64<<<GR(Nmb),B>>>(dac,dActCol,idx,Nmb); k_gather_f64<<<GR(Nmb),B>>>(dar,dAdvCol,idx,Nmb);
      k_gather_f64<<<GR(Nmb),B>>>(dre,dRetCol,idx,Nmb); k_gather_f64<<<GR(Nmb),B>>>(dol,dOlpCol,idx,Nmb);
      k_var_mean<<<1,256>>>(dar,dMd,Nmb); k_normalize<<<GR(Nmb),B>>>(dan,dar,dMd,(long)Nmb);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,Nmb,(int)D, dW1f,(int)D, dXf,(int)D, dPre,(int)H, bf);
      k_relu_bias<<<GR((long)Nmb*H),B>>>(dH1,dPre,db1f,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,Nmb,(int)H, dW2f,(int)H, dH1,(int)H, dPre,(int)O, bf);
      k_add_bias<<<GR((long)Nmb*O),B>>>(dOut,dPre,db2f,Nmb,(int)O);
      k_ppo_dout<<<GR(Nmb),B>>>(dOut,dac,dan,dre,dol,NULL,ddOut,Nmb,(int)A,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,0.0f);
      k_colsum<<<(int)O,256>>>(dgb2f,ddOut,Nmb,(int)O);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,Nmb, dH1,(int)H, ddOut,(int)O, dgW2f,(int)H, bf);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,Nmb,(int)O, dW2f,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
      k_dz1_mask<<<GR((long)Nmb*H),B>>>(ddZ,ddH1,dH1,(long)Nmb*H);
      k_colsum<<<(int)H,256>>>(dgb1f,ddZ,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,Nmb, dXf,(int)D, ddZ,(int)H, dgW1f,(int)D, bf);
      k_f32_to_f64<<<GR(H*D),B>>>(dG+oW1,dgW1f,(long)H*D); k_f32_to_f64<<<GR(H),B>>>(dG+ob1,dgb1f,(long)H);
      k_f32_to_f64<<<GR(O*H),B>>>(dG+oW2,dgW2f,(long)O*H); k_f32_to_f64<<<GR(O),B>>>(dG+ob2,dgb2f,(long)O);
      k_scale_const<<<GR(P),B>>>(dG,1.0/(double)Nmb,(long)P);
      /* Muon updates params/mom IN PLACE (dNewW=dW, dNewM=dM — each thread touches its own index) */
      muon_mat_dev((int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dPar+oW1,dMom+oW1, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(H),B>>>(dPar+ob1,dMom+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
      muon_mat_dev((int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dPar+oW2,dMom+oW2, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(O),B>>>(dPar+ob2,dMom+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
    }
    #undef GR
    cudaDeviceSynchronize();
    cudaMemcpy(out,dPar,8*P,cudaMemcpyDeviceToHost); cudaMemcpy(out+P,dMom,8*P,cudaMemcpyDeviceToHost);
  } else { for(size_t i=0;i<2*P;i++) out[i]=0.0; }
  lean_dec(pmA);lean_dec(obsA);lean_dec(actsA);lean_dec(advA);lean_dec(retA);lean_dec(olpA);lean_dec(permA);
  return Oo;
}

/* Read-only reduction of the 7 dashboard losses into the shared g_mgLoss channel for the feed-forward
   whole-update resident trainers. Called on the LAST minibatch of the LAST epoch when g_mgLossOn, it
   reads back only EXISTING device buffers (logits dOut, actions dac, normalized-adv dan, returns dre,
   old-logp dol, old-value dov) — no kernel launch, no training buffer written — so it cannot perturb
   training determinism. Same PufferLib loss decomposition as the MinGRU block. mode: 0=single-discrete
   (A cats), 1=multi-discrete (K heads, sizes hsHost), 2=continuous Gaussian (d dims). Value-loss clip
   applies only when dov!=NULL && vfClip>0 (matching d_vloss_grad's NULL contract). Defined after the
   CONT_* macros / k_ppo_dout_cont, below. */
static void ff_surface_losses(const float* dOut, const double* dac, const double* dan,
    const double* dre, const double* dol, const double* dov, long Nmb, int O,
    int mode, int A, int K, const int* hsHost, int dcont,
    double vfCoef, double entCoef, double clipEps, double vfClip);

/* Resident-weights twin of lean_cuda_train_update: params+mom live in the device-resident policy handle
   (policyH → [params(P); mom(P)]), updated IN PLACE by Muon — NO per-update H2D of params nor D2H of the
   result (the rollout reads the same resident params next update). Only the per-update columns/perm still
   upload, like the original. Returns a 1-element FloatArray [params[0]] for the caller's divergence guard
   (a lossless 8-byte peek). Bit-identical to lean_cuda_train_update threaded through the host. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_train_update_resident(
    size_t policyH, lean_obj_arg obsA, lean_obj_arg actsA, lean_obj_arg advA,
    lean_obj_arg retA, lean_obj_arg olpA, lean_obj_arg ovA, lean_obj_arg permA,
    size_t NT, size_t D, size_t H, size_t A, size_t epochs, size_t numMB,
    double lr, double wd, double mu, double eps, double vfCoef, double entCoef, double clipEps,
    double vfClip, double maxGradNorm,
    uint8_t bf16Flag, lean_obj_arg w){
  (void)w;                                                /* IO: this mutates the resident policy buffer */
  size_t O=A+1, P=H*D+H+O*H+O; int bf=(int)bf16Flag;
  size_t oW1=0, ob1=H*D, oW2=H*D+H, ob2=H*D+H+O*H;
  const double* obsd=lean_float_array_cptr(obsA);
  const double* actd=lean_float_array_cptr(actsA); const double* advd=lean_float_array_cptr(advA);
  const double* retd=lean_float_array_cptr(retA); const double* oldd=lean_float_array_cptr(olpA);
  const double* ovd=lean_float_array_cptr(ovA); size_t ovN=lean_sarray_size(ovA);
  const double* permd=lean_float_array_cptr(permA);
  /* value-clip needs the rollout's OLD values; an empty `ovA` (or vfClip<=0) turns the clip off. */
  int useVClip=(vfClip>0.0 && ovN>=NT);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),1,1); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  size_t SZ=(H*D>O*H?H*D:O*H); size_t md=(H>D?H:D); if(O>md) md=O; size_t SZA=md*md;
  float *dXf=(float*)ts_buf(0,4*NT*D),*dW1f=(float*)ts_buf(1,4*H*D),*dW2f=(float*)ts_buf(2,4*O*H),*db1f=(float*)ts_buf(3,4*H),*db2f=(float*)ts_buf(4,4*O);
  float *dH1=(float*)ts_buf(5,4*NT*H),*dOut=(float*)ts_buf(6,4*NT*O),*dPre=(float*)ts_buf(7,4*NT*(H>O?H:O)),*ddOut=(float*)ts_buf(8,4*NT*O),*ddH1=(float*)ts_buf(9,4*NT*H),*ddZ=(float*)ts_buf(10,4*NT*H);
  float *dgW1f=(float*)ts_buf(11,4*H*D),*dgb1f=(float*)ts_buf(12,4*H),*dgW2f=(float*)ts_buf(13,4*O*H),*dgb2f=(float*)ts_buf(14,4*O);
  double *dac=(double*)ts_buf(15,8*NT),*dar=(double*)ts_buf(16,8*NT),*dan=(double*)ts_buf(17,8*NT),*dre=(double*)ts_buf(18,8*NT),*dol=(double*)ts_buf(19,8*NT),*dMd=(double*)ts_buf(20,16);
  double *dPar=(double*)policyH,*dMom=dPar+P,*dG=(double*)ts_buf(23,8*P);   /* dPar/dMom RESIDENT (policy handle); dG scratch */
  double *dU=(double*)ts_buf(26,8*SZ),*dXn=(double*)ts_buf(27,8*SZ),*dXt=(double*)ts_buf(28,8*SZ),*dAm=(double*)ts_buf(29,8*SZA),*dPn=(double*)ts_buf(30,8*SZ),*dQn=(double*)ts_buf(31,8*SZ),*dInv=(double*)ts_buf(32,8);
  /* f32 NS scratch for the bf16 Muon path (bf16Flag==1) — free ts_buf slots 21,22,24,25,39 */
  float *mX=(float*)ts_buf(21,4*SZ),*mXt=(float*)ts_buf(22,4*SZ),*mA=(float*)ts_buf(24,4*SZA),*mP=(float*)ts_buf(25,4*SZ),*mQ=(float*)ts_buf(39,4*SZ);
  /* obs: use the rollout's DEVICE-resident f32 trajectory directly when valid (no obs H2D, no f64→f32) —
     it is exactly (float)obsCol. Else H2D obsA + convert (single-buffer / fallback). */
  int obsResident = (g_dObsTraj_valid && g_dObsTraj && g_dObsTrajSz>=(size_t)4*NT*D);
  float *dObsCol = obsResident ? g_dObsTraj : (float*)ts_buf(33,4*NT*D);
  double *dActCol=(double*)ts_buf(34,8*NT),*dAdvCol=(double*)ts_buf(35,8*NT),*dRetCol=(double*)ts_buf(36,8*NT),*dOlpCol=(double*)ts_buf(37,8*NT),*dPerm=(double*)ts_buf(38,8*epochs*NT);
  double *dObsD=(double*)ts_buf(40,8*NT*D);              /* f64 obs staging → GPU-convert (no host f64→f32 loop) */
  /* old-value column + its per-minibatch gather (value-loss clip), and the gradclip scratch scalar */
  double *dOvCol=useVClip?(double*)ts_buf(41,8*NT):NULL, *dov=useVClip?(double*)ts_buf(42,8*NT):NULL;
  double *dSq=(double*)ts_buf(43,8);
  if(useVClip && (!dOvCol||!dov)) useVClip=0;
  int ok=(NT>0 && policyH && h!=NULL && dObsD && dSq && dXf&&dW1f&&dW2f&&db1f&&db2f&&dH1&&dOut&&dPre&&ddOut&&ddH1&&ddZ&&dgW1f&&dgb1f&&dgW2f&&dgb2f&&
    dac&&dar&&dan&&dre&&dol&&dMd&&dG&&dU&&dXn&&dXt&&dAm&&dPn&&dQn&&dInv&&mX&&mXt&&mA&&mP&&mQ&&dObsCol&&dActCol&&dAdvCol&&dRetCol&&dOlpCol&&dPerm);
  if(!useVClip) dov=NULL;                                 /* kernel takes NULL ⇒ unclipped value loss */
  if(ok){
    if(!obsResident){ cudaMemcpy(dObsD,obsd,8*NT*D,cudaMemcpyHostToDevice);  /* upload f64 obs, convert on GPU */
      k_f64_to_f32<<<ceildiv((long)NT*D,256),256>>>(dObsCol,dObsD,(long)NT*D); }
    cudaMemcpy(dActCol,actd,8*NT,cudaMemcpyHostToDevice); cudaMemcpy(dAdvCol,advd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dRetCol,retd,8*NT,cudaMemcpyHostToDevice); cudaMemcpy(dOlpCol,oldd,8*NT,cudaMemcpyHostToDevice);
    if(useVClip) cudaMemcpy(dOvCol,ovd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dPerm,permd,8*epochs*NT,cudaMemcpyHostToDevice);
    int B=256;
    #define GR(x) ceildiv((long)(x),B)
    size_t mbSize=NT/numMB; if(mbSize<1) mbSize=1;
    for(size_t e=0;e<epochs;e++) for(size_t m=0;m<numMB;m++){
      size_t lo=m*mbSize; size_t hi=(m+1==numMB)?NT:(m+1)*mbSize; if(hi>NT) hi=NT;
      if(hi<=lo) continue;
      int Nmb=(int)(hi-lo); const double* idx=dPerm + e*NT + lo;
      k_f64_to_f32<<<GR(H*D),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<GR(H),B>>>(db1f,dPar+ob1,(long)H);
      k_f64_to_f32<<<GR(O*H),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<GR(O),B>>>(db2f,dPar+ob2,(long)O);
      k_gather_f32<<<GR((long)Nmb*D),B>>>(dXf,dObsCol,idx,Nmb,(int)D);
      k_gather_f64<<<GR(Nmb),B>>>(dac,dActCol,idx,Nmb); k_gather_f64<<<GR(Nmb),B>>>(dar,dAdvCol,idx,Nmb);
      k_gather_f64<<<GR(Nmb),B>>>(dre,dRetCol,idx,Nmb); k_gather_f64<<<GR(Nmb),B>>>(dol,dOlpCol,idx,Nmb);
      if(useVClip) k_gather_f64<<<GR(Nmb),B>>>(dov,dOvCol,idx,Nmb);
      k_var_mean<<<1,256>>>(dar,dMd,Nmb); k_normalize<<<GR(Nmb),B>>>(dan,dar,dMd,(long)Nmb);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,Nmb,(int)D, dW1f,(int)D, dXf,(int)D, dPre,(int)H, bf);
      k_relu_bias<<<GR((long)Nmb*H),B>>>(dH1,dPre,db1f,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,Nmb,(int)H, dW2f,(int)H, dH1,(int)H, dPre,(int)O, bf);
      k_add_bias<<<GR((long)Nmb*O),B>>>(dOut,dPre,db2f,Nmb,(int)O);
      k_ppo_dout<<<GR(Nmb),B>>>(dOut,dac,dan,dre,dol,dov,ddOut,Nmb,(int)A,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,(float)vfClip);
      /* dashboard losses on the LAST minibatch of the LAST epoch only (read-only D2H, render-frame cadence) */
      if(g_mgLossOn && e+1==epochs && m+1==numMB)
        ff_surface_losses(dOut,dac,dan,dre,dol,dov,(long)Nmb,(int)O,0,(int)A,1,NULL,0,vfCoef,entCoef,clipEps,vfClip);
      k_colsum<<<(int)O,256>>>(dgb2f,ddOut,Nmb,(int)O);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,Nmb, dH1,(int)H, ddOut,(int)O, dgW2f,(int)H, bf);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,Nmb,(int)O, dW2f,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
      k_dz1_mask<<<GR((long)Nmb*H),B>>>(ddZ,ddH1,dH1,(long)Nmb*H);
      k_colsum<<<(int)H,256>>>(dgb1f,ddZ,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,Nmb, dXf,(int)D, ddZ,(int)H, dgW1f,(int)D, bf);
      k_f32_to_f64<<<GR(H*D),B>>>(dG+oW1,dgW1f,(long)H*D); k_f32_to_f64<<<GR(H),B>>>(dG+ob1,dgb1f,(long)H);
      k_f32_to_f64<<<GR(O*H),B>>>(dG+oW2,dgW2f,(long)O*H); k_f32_to_f64<<<GR(O),B>>>(dG+ob2,dgb2f,(long)O);
      /* mean over the minibatch, then PufferLib's global grad-norm clip (torch_pufferl.py:347,
         `clip_grad_norm_(policy.parameters(), max_grad_norm)`) — one norm over the WHOLE parameter
         vector, exactly the fused scale+clip the MinGRU trainer already uses. maxGradNorm<=0 ⇒
         the bare mean scale, i.e. the pre-fix path. */
      if(maxGradNorm>0.0){
        k_gclip_norm<<<1,256>>>(dSq,dG,1.0/(double)Nmb,(long)P);
        k_gclip_scale<<<GR(P),B>>>(dG,dG,dSq,maxGradNorm,1.0/(double)Nmb,(long)P);
      } else k_scale_const<<<GR(P),B>>>(dG,1.0/(double)Nmb,(long)P);
      /* Muon updates params/mom IN PLACE inside the resident policy handle. bf16Flag==1 (PufferLib
         precision) → NS orthogonalization via bf16 cuBLAS tensor cores (was the f64 naive k_matmul, ~40%
         of GPU time); bf16Flag==0 → f64 naive (the bit-exact oracle for verify-train-update-resident). */
      if(bf){
        muon_mat_dev_bf(h,(int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dPar+oW1,dMom+oW1, lr,wd,mu,eps, dU,dInv, mX,mXt,mA,mP,mQ, bf);
        k_stepvec<<<GR(H),B>>>(dPar+ob1,dMom+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
        muon_mat_dev_bf(h,(int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dPar+oW2,dMom+oW2, lr,wd,mu,eps, dU,dInv, mX,mXt,mA,mP,mQ, bf);
        k_stepvec<<<GR(O),B>>>(dPar+ob2,dMom+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
      } else {
        muon_mat_dev((int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dPar+oW1,dMom+oW1, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
        k_stepvec<<<GR(H),B>>>(dPar+ob1,dMom+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
        muon_mat_dev((int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dPar+oW2,dMom+oW2, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
        k_stepvec<<<GR(O),B>>>(dPar+ob2,dMom+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
      }
    }
    #undef GR
    cudaDeviceSynchronize();
    cudaMemcpy(out,dPar,8,cudaMemcpyDeviceToHost);         /* params[0] for the divergence guard (no full D2H) */
  } else { out[0]=0.0; }
  lean_dec(obsA);lean_dec(actsA);lean_dec(advA);lean_dec(retA);lean_dec(olpA);lean_dec(ovA);lean_dec(permA);
  return lean_io_result_mk_ok(Oo);
}

__global__ void k_ppo_dout_cont(const float*, const double*, const double*, const double*, const double*,
                                const double*, float*, int, int, int, float, float, float, float);   /* defined below */

/* Whole-update RESIDENT step for the WIDE-action plugin trainers — MULTI-DISCRETE (mode 1) and
   CONTINUOUS/Gaussian (mode 2). The W-wide-action twin of lean_cuda_train_update_resident: the SoA
   columns (obs NT·D, acts NT·W, adv/ret/olp NT) + the epochs·NT shuffle upload ONCE; the device loops
   epochs×numMB doing gather → per-minibatch adv-normalize → forward → PPO objective backward (k_ppo_dout_md
   with headSizes / k_ppo_dout_cont) → backward GEMMs → in-place Muon over the resident [params;mom] handle.
   Moves the per-minibatch gather + adv-norm that ran in interpreted Lean onto the GPU. Bit-identical to the
   old host-gather path: same gather order, same adv-norm (k_var_mean/k_normalize == the host fold), same
   grad kernels, same Muon. Returns params[0] for the divergence guard. O = Σheads+1 (MD) / 2·d+1 (Cont). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_train_update_wide_resident(
    size_t policyH, lean_obj_arg obsA, lean_obj_arg actsA, lean_obj_arg advA,
    lean_obj_arg retA, lean_obj_arg olpA, lean_obj_arg ovA, lean_obj_arg permA, lean_obj_arg hsA,
    size_t NT, size_t D, size_t H, size_t O, size_t Wdim, uint32_t mode, size_t epochs, size_t numMB,
    double lr, double wd, double mu, double eps, double vfCoef, double entCoef, double clipEps,
    double vfClip, double maxGradNorm,
    uint8_t bf16Flag, lean_obj_arg w){
  (void)w;
  size_t P=H*D+H+O*H+O; int bf=(int)bf16Flag; int W=(int)Wdim;
  size_t oW1=0, ob1=H*D, oW2=H*D+H, ob2=H*D+H+O*H;
  const double* obsd=lean_float_array_cptr(obsA);
  const double* actd=lean_float_array_cptr(actsA); const double* advd=lean_float_array_cptr(advA);
  const double* retd=lean_float_array_cptr(retA); const double* oldd=lean_float_array_cptr(olpA);
  const double* ovd=lean_float_array_cptr(ovA); size_t ovN=lean_sarray_size(ovA);
  const double* permd=lean_float_array_cptr(permA); const double* hs=lean_float_array_cptr(hsA);
  int useVClip=(vfClip>0.0 && ovN>=NT);   /* empty ovA / vfClip<=0 ⇒ unclipped value loss */
  lean_object* Oo=lean_alloc_sarray(sizeof(double),1,1); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  size_t SZ=(H*D>O*H?H*D:O*H); size_t md=(H>D?H:D); if(O>md) md=O; size_t SZA=md*md;
  float *dXf=(float*)ts_buf(0,4*NT*D),*dW1f=(float*)ts_buf(1,4*H*D),*dW2f=(float*)ts_buf(2,4*O*H),*db1f=(float*)ts_buf(3,4*H),*db2f=(float*)ts_buf(4,4*O);
  float *dH1=(float*)ts_buf(5,4*NT*H),*dOut=(float*)ts_buf(6,4*NT*O),*dPre=(float*)ts_buf(7,4*NT*(H>O?H:O)),*ddOut=(float*)ts_buf(8,4*NT*O),*ddH1=(float*)ts_buf(9,4*NT*H),*ddZ=(float*)ts_buf(10,4*NT*H);
  float *dgW1f=(float*)ts_buf(11,4*H*D),*dgb1f=(float*)ts_buf(12,4*H),*dgW2f=(float*)ts_buf(13,4*O*H),*dgb2f=(float*)ts_buf(14,4*O);
  double *dac=(double*)ts_buf(15,8*(size_t)NT*W),*dar=(double*)ts_buf(16,8*NT),*dan=(double*)ts_buf(17,8*NT),*dre=(double*)ts_buf(18,8*NT),*dol=(double*)ts_buf(19,8*NT),*dMd=(double*)ts_buf(20,16);
  double *dPar=(double*)policyH,*dMom=dPar+P,*dG=(double*)ts_buf(23,8*P);
  double *dU=(double*)ts_buf(26,8*SZ),*dXn=(double*)ts_buf(27,8*SZ),*dXt=(double*)ts_buf(28,8*SZ),*dAm=(double*)ts_buf(29,8*SZA),*dPn=(double*)ts_buf(30,8*SZ),*dQn=(double*)ts_buf(31,8*SZ),*dInv=(double*)ts_buf(32,8);
  /* obs: use the rollout's DEVICE-resident f32 trajectory directly when valid (no obs H2D, no f64→f32) —
     same gate lean_cuda_train_update_resident already has for the MLP sibling, now ported here too. */
  int obsResident = (g_dObsTraj_valid && g_dObsTraj && g_dObsTrajSz>=(size_t)4*NT*D);
  float *dObsCol = obsResident ? g_dObsTraj : (float*)ts_buf(33,4*NT*D);
  double *dActCol=(double*)ts_buf(34,8*(size_t)NT*W),*dAdvCol=(double*)ts_buf(35,8*NT),*dRetCol=(double*)ts_buf(36,8*NT),*dOlpCol=(double*)ts_buf(37,8*NT),*dPerm=(double*)ts_buf(38,8*epochs*NT);
  int* dHs=(mode==1)?(int*)ts_buf(39,sizeof(int)*(size_t)W):(int*)1;   /* head sizes (MD only) */
  double *dObsD=(double*)ts_buf(40,8*NT*D);              /* f64 obs staging → GPU-convert (no host f64→f32 loop) */
  double *dOvCol=useVClip?(double*)ts_buf(41,8*NT):NULL, *dov=useVClip?(double*)ts_buf(42,8*NT):NULL;
  double *dSq=(double*)ts_buf(43,8);
  if(useVClip && (!dOvCol||!dov)) useVClip=0;
  int ok=(NT>0 && policyH && W>0 && h!=NULL && dObsD && dSq && dXf&&dW1f&&dW2f&&db1f&&db2f&&dH1&&dOut&&dPre&&ddOut&&ddH1&&ddZ&&dgW1f&&dgb1f&&dgW2f&&dgb2f&&
    dac&&dar&&dan&&dre&&dol&&dMd&&dG&&dU&&dXn&&dXt&&dAm&&dPn&&dQn&&dInv&&dObsCol&&dActCol&&dAdvCol&&dRetCol&&dOlpCol&&dPerm&&dHs);
  if(!useVClip) dov=NULL;                                 /* kernel takes NULL ⇒ unclipped value loss */
  if(ok){
    if(!obsResident){ cudaMemcpy(dObsD,obsd,8*NT*D,cudaMemcpyHostToDevice);  /* upload f64 obs, convert on GPU (was a 7.7M CPU cast/update) */
      k_f64_to_f32<<<ceildiv((long)NT*D,256),256>>>(dObsCol,dObsD,(long)NT*D); }
    cudaMemcpy(dActCol,actd,8*(size_t)NT*W,cudaMemcpyHostToDevice); cudaMemcpy(dAdvCol,advd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dRetCol,retd,8*NT,cudaMemcpyHostToDevice); cudaMemcpy(dOlpCol,oldd,8*NT,cudaMemcpyHostToDevice);
    if(useVClip) cudaMemcpy(dOvCol,ovd,8*NT,cudaMemcpyHostToDevice);
    cudaMemcpy(dPerm,permd,8*epochs*NT,cudaMemcpyHostToDevice);
    int* hHs=NULL;   /* host head sizes, kept alive past the upload for the dashboard loss reduction (MD) */
    if(mode==1){ hHs=(int*)malloc(sizeof(int)*(size_t)W); for(int i=0;i<W;i++) hHs[i]=(int)hs[i];
      cudaMemcpy(dHs,hHs,sizeof(int)*(size_t)W,cudaMemcpyHostToDevice); }
    int B=256;
    #define GR(x) ceildiv((long)(x),B)
    size_t mbSize=NT/numMB; if(mbSize<1) mbSize=1;
    for(size_t e=0;e<epochs;e++) for(size_t m=0;m<numMB;m++){
      size_t lo=m*mbSize; size_t hi=(m+1==numMB)?NT:(m+1)*mbSize; if(hi>NT) hi=NT;
      if(hi<=lo) continue;
      int Nmb=(int)(hi-lo); const double* idx=dPerm + e*NT + lo;
      k_f64_to_f32<<<GR(H*D),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<GR(H),B>>>(db1f,dPar+ob1,(long)H);
      k_f64_to_f32<<<GR(O*H),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<GR(O),B>>>(db2f,dPar+ob2,(long)O);
      k_gather_f32<<<GR((long)Nmb*D),B>>>(dXf,dObsCol,idx,Nmb,(int)D);
      k_gather_f64_wide<<<GR((long)Nmb*W),B>>>(dac,dActCol,idx,Nmb,W);
      k_gather_f64<<<GR(Nmb),B>>>(dar,dAdvCol,idx,Nmb);
      k_gather_f64<<<GR(Nmb),B>>>(dre,dRetCol,idx,Nmb); k_gather_f64<<<GR(Nmb),B>>>(dol,dOlpCol,idx,Nmb);
      if(useVClip) k_gather_f64<<<GR(Nmb),B>>>(dov,dOvCol,idx,Nmb);
      k_var_mean<<<1,256>>>(dar,dMd,Nmb); k_normalize<<<GR(Nmb),B>>>(dan,dar,dMd,(long)Nmb);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,Nmb,(int)D, dW1f,(int)D, dXf,(int)D, dPre,(int)H, bf);
      k_relu_bias<<<GR((long)Nmb*H),B>>>(dH1,dPre,db1f,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,Nmb,(int)H, dW2f,(int)H, dH1,(int)H, dPre,(int)O, bf);
      k_add_bias<<<GR((long)Nmb*O),B>>>(dOut,dPre,db2f,Nmb,(int)O);
      if(mode==1) k_ppo_dout_md<<<GR(Nmb),B>>>(dOut,dac,dan,dre,dol,dov,ddOut,Nmb,W,dHs,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,(float)vfClip);
      else        k_ppo_dout_cont<<<GR(Nmb),B>>>(dOut,dac,dan,dre,dol,dov,ddOut,Nmb,W,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,(float)vfClip);
      /* dashboard losses on the LAST minibatch of the LAST epoch only (read-only D2H, render-frame cadence) */
      if(g_mgLossOn && e+1==epochs && m+1==numMB){
        if(mode==1) ff_surface_losses(dOut,dac,dan,dre,dol,dov,(long)Nmb,(int)O,1,0,W,hHs,0,vfCoef,entCoef,clipEps,vfClip);
        else        ff_surface_losses(dOut,dac,dan,dre,dol,dov,(long)Nmb,(int)O,2,0,0,NULL,W,vfCoef,entCoef,clipEps,vfClip);
      }
      k_colsum<<<(int)O,256>>>(dgb2f,ddOut,Nmb,(int)O);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,Nmb, dH1,(int)H, ddOut,(int)O, dgW2f,(int)H, bf);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,Nmb,(int)O, dW2f,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
      k_dz1_mask<<<GR((long)Nmb*H),B>>>(ddZ,ddH1,dH1,(long)Nmb*H);
      k_colsum<<<(int)H,256>>>(dgb1f,ddZ,Nmb,(int)H);
      gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,Nmb, dXf,(int)D, ddZ,(int)H, dgW1f,(int)D, bf);
      k_f32_to_f64<<<GR(H*D),B>>>(dG+oW1,dgW1f,(long)H*D); k_f32_to_f64<<<GR(H),B>>>(dG+ob1,dgb1f,(long)H);
      k_f32_to_f64<<<GR(O*H),B>>>(dG+oW2,dgW2f,(long)O*H); k_f32_to_f64<<<GR(O),B>>>(dG+ob2,dgb2f,(long)O);
      /* mean over the minibatch, then PufferLib's global grad-norm clip (torch_pufferl.py:347,
         `clip_grad_norm_(policy.parameters(), max_grad_norm)`) — one norm over the WHOLE parameter
         vector, exactly the fused scale+clip the MinGRU trainer already uses. maxGradNorm<=0 ⇒
         the bare mean scale, i.e. the pre-fix path. */
      if(maxGradNorm>0.0){
        k_gclip_norm<<<1,256>>>(dSq,dG,1.0/(double)Nmb,(long)P);
        k_gclip_scale<<<GR(P),B>>>(dG,dG,dSq,maxGradNorm,1.0/(double)Nmb,(long)P);
      } else k_scale_const<<<GR(P),B>>>(dG,1.0/(double)Nmb,(long)P);
      muon_mat_dev((int)H,(int)D, dPar+oW1,dG+oW1,dMom+oW1, dPar+oW1,dMom+oW1, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(H),B>>>(dPar+ob1,dMom+ob1, dPar+ob1,dG+ob1,dMom+ob1, lr,wd,mu,(long)H);
      muon_mat_dev((int)O,(int)H, dPar+oW2,dG+oW2,dMom+oW2, dPar+oW2,dMom+oW2, lr,wd,mu,eps, dU,dXn,dXt,dAm,dPn,dQn,dInv);
      k_stepvec<<<GR(O),B>>>(dPar+ob2,dMom+ob2, dPar+ob2,dG+ob2,dMom+ob2, lr,wd,mu,(long)O);
    }
    #undef GR
    cudaDeviceSynchronize();
    cudaMemcpy(out,dPar,8,cudaMemcpyDeviceToHost);
    if(hHs) free(hHs);
  } else { out[0]=0.0; }
  lean_dec(obsA);lean_dec(actsA);lean_dec(advA);lean_dec(retA);lean_dec(olpA);lean_dec(ovA);lean_dec(permA);lean_dec(hsA);
  return lean_io_result_mk_ok(Oo);
}


/* === R2: device categorical action sampler (GPU-resident rollout) =================================
   The device twin of `lean_ffi_sample_actions_batch` (ffi/pufferffi.c): over the N×O logit batch (O=A+1:
   A policy logits then the value), sample one action/env, and return [actions(N); logps(N); values(N)].
   env n draws splitmix64 word hash(rng + (n+1)·G) — the SAME per-env stream as the CPU sampler and the
   rollout's rngNext. Sampling matches softmax+sampleCat op-for-op, so actions + values match the CPU path
   EXACTLY and logp to transcendental ULP (device exp/log vs libm). The one caveat (device≠host `exp`): at a
   measure-zero cumulative-probability boundary — u within a ULP of a category edge — the sampled action can
   differ from the CPU sampler. That is a valid categorical draw either way, the same tolerance the bf16
   rollout forward already accepts (docs/gpu-rollout-scope.md), not a defect. One thread per env. R3 reads
   resident logits and writes resident columns; this test FFI round-trips. */
__device__ __forceinline__ unsigned long long d_sm64(unsigned long long s){
  unsigned long long z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}
__global__ void k_sample(const double* Yb, double* out, int N, int A, int O, unsigned long long rng){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  const double* row = Yb + (long)n*O;
  double m = row[0];
  for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  double z=0.0; for(int k=0;k<A;k++) z += exp(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)n + 1ULL)*0x9E3779B97F4A7C15ULL);  /* n+1 in 64-bit (no int overflow at N≥2³¹) */
  double u = (double)(word >> 11) / 9007199254740992.0;       /* uniform01: top 53 bits / 2^53 */
  double acc=0.0; int a=A-1;
  for(int k=0;k<A;k++){ acc += exp(row[k]-m)/z; if(u < acc){ a=k; break; } }
  out[n]     = (double)a;
  out[N+n]   = log(exp(row[a]-m)/z);                           /* log(probs[a]) */
  out[2*N+n] = row[A];                                         /* value head */
}
/* f32 twin of k_sample: reads the f32 forward logits DIRECTLY (no f32→f64 upcast) and does the softmax in
   f32 (expf/logf) — the forward is already bf16-precision, so f64 here was doubly wasted (it was the #1
   GPU kernel, ~22%). The rng draw `u` stays the identical f64 word, so only the softmax rounding differs
   (a bf16-tolerance numerics change, like our other bf16 paths). Used by the rollout; the f64 k_sample
   stays for the bit-exact sampler verify. */
__global__ void k_sample_f32(const float* Yb, double* out, int N, int A, int O, unsigned long long rng){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  const float* row = Yb + (long)n*O;
  float m = row[0];
  for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  float z=0.0f; for(int k=0;k<A;k++) z += expf(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)n + 1ULL)*0x9E3779B97F4A7C15ULL);
  double u = (double)(word >> 11) / 9007199254740992.0;
  float acc=0.0f; int a=A-1;
  for(int k=0;k<A;k++){ acc += expf(row[k]-m)/z; if(u < (double)acc){ a=k; break; } }
  out[n]     = (double)a;
  out[N+n]   = (double)logf(expf(row[a]-m)/z);
  out[2*N+n] = (double)row[A];
}
/* f32 segmented twins (buffered rollout): compact per-segment [act;logp;val] output, global-row rng. */
__global__ void k_sample_seg_f32(const float* Yb, double* out, int count, int rowBase, int A, int O, unsigned long long rng){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  const float* row = Yb + (long)j*O;
  float m = row[0]; for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  float z=0.0f; for(int k=0;k<A;k++) z += expf(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)(rowBase+j) + 1ULL)*0x9E3779B97F4A7C15ULL);
  double u = (double)(word >> 11) / 9007199254740992.0;
  float acc=0.0f; int a=A-1;
  for(int k=0;k<A;k++){ acc += expf(row[k]-m)/z; if(u < (double)acc){ a=k; break; } }
  out[j]=(double)a; out[count+j]=(double)logf(expf(row[a]-m)/z); out[2*count+j]=(double)row[A];
}
__global__ void k_sample_seg_g_f32(const float* Yb, double* out, int count, int rowBase, int A, int O, const unsigned long long* rngp){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return; unsigned long long rng=*rngp;
  const float* row = Yb + (long)j*O;
  float m = row[0]; for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  float z=0.0f; for(int k=0;k<A;k++) z += expf(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)(rowBase+j) + 1ULL)*0x9E3779B97F4A7C15ULL);
  double u = (double)(word >> 11) / 9007199254740992.0;
  float acc=0.0f; int a=A-1;
  for(int k=0;k<A;k++){ acc += expf(row[k]-m)/z; if(u < (double)acc){ a=k; break; } }
  out[j]=(double)a; out[count+j]=(double)logf(expf(row[a]-m)/z); out[2*count+j]=(double)row[A];
}
/* Segmented single-discrete sampler for the double-buffer rollout: sample `count` rows whose logits are
   the compact block `Yb` (count rows), writing a COMPACT per-segment output [act(count); logp(count);
   val(count)] to `out`. The rng draw uses the GLOBAL row index (rowBase+j), so it is bit-identical to the
   whole-batch k_sample for the same rows. */
__global__ void k_sample_seg(const double* Yb, double* out, int count, int rowBase, int A, int O, unsigned long long rng){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  const double* row = Yb + (long)j*O;
  double m = row[0];
  for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  double z=0.0; for(int k=0;k<A;k++) z += exp(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)(rowBase+j) + 1ULL)*0x9E3779B97F4A7C15ULL);
  double u = (double)(word >> 11) / 9007199254740992.0;
  double acc=0.0; int a=A-1;
  for(int k=0;k<A;k++){ acc += exp(row[k]-m)/z; if(u < acc){ a=k; break; } }
  out[j]         = (double)a;
  out[count+j]   = log(exp(row[a]-m)/z);                       /* log(probs[a]) */
  out[2*count+j] = row[A];                                     /* value head */
}
/* Graph-friendly twin of k_sample_seg: rng read from device memory (*rngp) instead of a kernel arg, so the
   per-timestep rng change doesn't require a graph rebuild — bit-identical to k_sample_seg for *rngp==rng. */
__global__ void k_sample_seg_g(const double* Yb, double* out, int count, int rowBase, int A, int O, const unsigned long long* rngp){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  unsigned long long rng=*rngp;
  const double* row = Yb + (long)j*O;
  double m = row[0];
  for(int k=1;k<A;k++) if(row[k]>m) m=row[k];
  double z=0.0; for(int k=0;k<A;k++) z += exp(row[k]-m);
  unsigned long long word = d_sm64(rng + ((unsigned long long)(rowBase+j) + 1ULL)*0x9E3779B97F4A7C15ULL);
  double u = (double)(word >> 11) / 9007199254740992.0;
  double acc=0.0; int a=A-1;
  for(int k=0;k<A;k++){ acc += exp(row[k]-m)/z; if(u < acc){ a=k; break; } }
  out[j]         = (double)a;
  out[count+j]   = log(exp(row[a]-m)/z);
  out[2*count+j] = row[A];
}
__global__ void k_set_u64(unsigned long long* p, unsigned long long v){ *p=v; }
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_sample_actions(lean_obj_arg Yba,
    size_t N, size_t A, size_t O, uint64_t rng){
  const double* Yb=lean_float_array_cptr(Yba);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),3*N,3*N); double* out=lean_float_array_cptr(Oo);
  double *dY=NULL,*dO=NULL;
  if(N>0 && cudaMalloc((void**)&dY,8*N*O)==cudaSuccess && cudaMalloc((void**)&dO,8*3*N)==cudaSuccess){
    cudaMemcpy(dY,Yb,8*N*O,cudaMemcpyHostToDevice);
    k_sample<<<ceildiv((long)N,256),256>>>(dY,dO,(int)N,(int)A,(int)O,(unsigned long long)rng);
    cudaMemcpy(out,dO,8*3*N,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<3*N;i++) out[i]=0.0;
  if(dY)cudaFree(dY); if(dO)cudaFree(dO);
  lean_dec(Yba); return Oo;
}

/* === Multi-discrete action support (K categorical heads; O = Σheadsizes + 1) ================== */
/* Sample each of the K heads from the N×O logit batch; return [actions(N×K, col-major head h at h·N+n);
   jointLogp(N); value(N)] (size (K+2)·N). Distinct rng per (row,head). K=1 = a single categorical head. */
__global__ void k_sample_md(const double* Yb, double* out, int N, int K, const int* headSizes, int O, unsigned long long rng){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  const double* row=Yb+(long)n*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  double jointLogp=0.0; int off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh];
    double m=row[off]; for(int k=1;k<sz;k++) if(row[off+k]>m) m=row[off+k];
    double z=0.0; for(int k=0;k<sz;k++) z+=exp(row[off+k]-m);
    unsigned long long word=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)hh+1ULL)*G2);
    double u=(double)(word>>11)/9007199254740992.0;
    double acc=0.0; int a=sz-1;
    for(int k=0;k<sz;k++){ acc+=exp(row[off+k]-m)/z; if(u<acc){ a=k; break; } }
    out[(long)hh*N+n]=(double)a; jointLogp += log(exp(row[off+a]-m)/z); off+=sz; }
  out[(long)K*N+n]=jointLogp; out[(long)(K+1)*N+n]=row[O-1];
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_sample_actions_md(lean_obj_arg Yba, lean_obj_arg hsA,
    size_t N, size_t K, size_t O, uint64_t rng){
  const double* Yb=lean_float_array_cptr(Yba); const double* hs=lean_float_array_cptr(hsA);
  int* headSizes=(int*)malloc(sizeof(int)*K); for(size_t i=0;i<K;i++) headSizes[i]=(int)hs[i];
  size_t outLen=(K+2)*N;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),outLen,outLen); double* out=lean_float_array_cptr(Oo);
  double *dY=NULL,*dO=NULL; int* dHs=NULL;
  if(N>0 && cudaMalloc((void**)&dY,8*N*O)==cudaSuccess && cudaMalloc((void**)&dO,8*outLen)==cudaSuccess
     && cudaMalloc((void**)&dHs,sizeof(int)*K)==cudaSuccess){
    cudaMemcpy(dY,Yb,8*N*O,cudaMemcpyHostToDevice); cudaMemcpy(dHs,headSizes,sizeof(int)*K,cudaMemcpyHostToDevice);
    k_sample_md<<<ceildiv((long)N,256),256>>>(dY,dO,(int)N,(int)K,dHs,(int)O,(unsigned long long)rng);
    cudaMemcpy(out,dO,8*outLen,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<outLen;i++) out[i]=0.0;
  if(dY)cudaFree(dY); if(dO)cudaFree(dO); if(dHs)cudaFree(dHs); free(headSizes);
  lean_dec(Yba); lean_dec(hsA); return Oo;
}
/* Multi-discrete minibatch PPO gradient — the md twin of lean_cuda_mlp_ppo_grad. acts N×K, O=Σheadsizes+1. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mlp_ppo_grad_md(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, lean_obj_arg hsA, size_t N, size_t H, size_t D, size_t K,
    double vfCoef, double entCoef, double clipEps, uint8_t bf16Flag){
  const double* hs=lean_float_array_cptr(hsA);
  int* headSizes=(int*)malloc(sizeof(int)*K); size_t A=0; for(size_t i=0;i<K;i++){ headSizes[i]=(int)hs[i]; A+=(size_t)headSizes[i]; }
  size_t O=A+1, P=H*D+H+O*H+O;
  const double* pp=lean_float_array_cptr(pa); const double* Xd=lean_float_array_cptr(obsBa);
  const double* actd=lean_float_array_cptr(actsa); const double* advd=lean_float_array_cptr(advsa);
  const double* retd=lean_float_array_cptr(retsa); const double* oldd=lean_float_array_cptr(oldlpsa);
  const double* W1d=pp; const double* b1d=W1d+H*D; const double* W2d=b1d+H; const double* b2d=W2d+O*H;
  lean_object* go=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(go);
  int bf=(int)bf16Flag; cublasHandle_t h=cu_handle();
  float *hX=(float*)malloc(4*N*D),*hW1=(float*)malloc(4*H*D),*hW2=(float*)malloc(4*O*H),*hb1=(float*)malloc(4*H),*hb2=(float*)malloc(4*O);
  float *hgW1=(float*)malloc(4*H*D),*hgb1=(float*)malloc(4*H),*hgW2=(float*)malloc(4*O*H),*hgb2=(float*)malloc(4*O);
  float *dX=NULL,*dW1=NULL,*dW2=NULL,*db1=NULL,*db2=NULL,*dH1=NULL,*dOut=NULL,*dPre=NULL,*ddOut=NULL,*ddH1=NULL,*ddZ=NULL,*dgW1=NULL,*dgb1=NULL,*dgW2=NULL,*dgb2=NULL;
  double *dac=NULL,*dad=NULL,*dre=NULL,*dol=NULL; int* dHs=NULL;
  int ok=(N>0 && h!=NULL &&
    !cudaMalloc((void**)&dX,4*N*D) && !cudaMalloc((void**)&dW1,4*H*D) && !cudaMalloc((void**)&dW2,4*O*H) &&
    !cudaMalloc((void**)&db1,4*H) && !cudaMalloc((void**)&db2,4*O) && !cudaMalloc((void**)&dH1,4*N*H) &&
    !cudaMalloc((void**)&dOut,4*N*O) && !cudaMalloc((void**)&dPre,4*N*(H>O?H:O)) && !cudaMalloc((void**)&ddOut,4*N*O) &&
    !cudaMalloc((void**)&ddH1,4*N*H) && !cudaMalloc((void**)&ddZ,4*N*H) && !cudaMalloc((void**)&dgW1,4*H*D) &&
    !cudaMalloc((void**)&dgb1,4*H) && !cudaMalloc((void**)&dgW2,4*O*H) && !cudaMalloc((void**)&dgb2,4*O) &&
    !cudaMalloc((void**)&dac,8*N*K) && !cudaMalloc((void**)&dad,8*N) && !cudaMalloc((void**)&dre,8*N) && !cudaMalloc((void**)&dol,8*N) &&
    !cudaMalloc((void**)&dHs,sizeof(int)*K));
  if(ok){
    for(size_t i=0;i<N*D;i++) hX[i]=(float)Xd[i];
    for(size_t i=0;i<H*D;i++) hW1[i]=(float)W1d[i];
    for(size_t i=0;i<O*H;i++) hW2[i]=(float)W2d[i];
    for(size_t i=0;i<H;i++) hb1[i]=(float)b1d[i];
    for(size_t i=0;i<O;i++) hb2[i]=(float)b2d[i];
    cudaMemcpy(dX,hX,4*N*D,cudaMemcpyHostToDevice); cudaMemcpy(dW1,hW1,4*H*D,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,4*O*H,cudaMemcpyHostToDevice); cudaMemcpy(db1,hb1,4*H,cudaMemcpyHostToDevice);
    cudaMemcpy(db2,hb2,4*O,cudaMemcpyHostToDevice);
    cudaMemcpy(dac,actd,8*N*K,cudaMemcpyHostToDevice); cudaMemcpy(dad,advd,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dre,retd,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dol,oldd,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dHs,headSizes,sizeof(int)*K,cudaMemcpyHostToDevice);
    int B=256;
    #define GRM(x) ceildiv((long)(x),B)
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)N,(int)D, dW1,(int)D, dX,(int)D, dPre,(int)H, bf);
    k_relu_bias<<<GRM(N*H),B>>>(dH1,dPre,db1,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,(int)N,(int)H, dW2,(int)H, dH1,(int)H, dPre,(int)O, bf);
    k_add_bias<<<GRM(N*O),B>>>(dOut,dPre,db2,(int)N,(int)O);
    k_ppo_dout_md<<<GRM(N),B>>>(dOut,dac,dad,dre,dol,NULL,ddOut,(int)N,(int)K,dHs,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,0.0f);
    k_colsum<<<(int)O,256>>>(dgb2,ddOut,(int)N,(int)O);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,(int)N, dH1,(int)H, ddOut,(int)O, dgW2,(int)H, bf);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,(int)N,(int)O, dW2,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
    k_dz1_mask<<<GRM(N*H),B>>>(ddZ,ddH1,dH1,(long)N*H);
    k_colsum<<<(int)H,256>>>(dgb1,ddZ,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,(int)N, dX,(int)D, ddZ,(int)H, dgW1,(int)D, bf);
    #undef GRM
    cudaDeviceSynchronize();
    cudaMemcpy(hgW1,dgW1,4*H*D,cudaMemcpyDeviceToHost); cudaMemcpy(hgb1,dgb1,4*H,cudaMemcpyDeviceToHost);
    cudaMemcpy(hgW2,dgW2,4*O*H,cudaMemcpyDeviceToHost); cudaMemcpy(hgb2,dgb2,4*O,cudaMemcpyDeviceToHost);
    size_t o=0;
    for(size_t i=0;i<H*D;i++) g[o++]=(double)hgW1[i];
    for(size_t i=0;i<H;i++)   g[o++]=(double)hgb1[i];
    for(size_t i=0;i<O*H;i++) g[o++]=(double)hgW2[i];
    for(size_t i=0;i<O;i++)   g[o++]=(double)hgb2[i];
  } else for(size_t i=0;i<P;i++) g[i]=0.0;
  if(dX)cudaFree(dX);if(dW1)cudaFree(dW1);if(dW2)cudaFree(dW2);if(db1)cudaFree(db1);if(db2)cudaFree(db2);
  if(dH1)cudaFree(dH1);if(dOut)cudaFree(dOut);if(dPre)cudaFree(dPre);if(ddOut)cudaFree(ddOut);if(ddH1)cudaFree(ddH1);
  if(ddZ)cudaFree(ddZ);if(dgW1)cudaFree(dgW1);if(dgb1)cudaFree(dgb1);if(dgW2)cudaFree(dgW2);if(dgb2)cudaFree(dgb2);
  if(dac)cudaFree(dac);if(dad)cudaFree(dad);if(dre)cudaFree(dre);if(dol)cudaFree(dol);if(dHs)cudaFree(dHs);
  free(hX);free(hW1);free(hW2);free(hb1);free(hb2);free(hgW1);free(hgb1);free(hgW2);free(hgb2);free(headSizes);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);lean_dec(hsA);
  return go;
}

/* === Continuous (diagonal-Gaussian) action support (O = 2·d + 1) ================================
   Policy head: out[0..d)=means μ, out[d..2d)=raw logstds (clamped to [LSLO,LSHI]), out[2d]=value.
   σ=exp(logstd); sample aᵢ=μᵢ+σᵢ·zᵢ, zᵢ~N(0,1) (Box–Muller); logp=Σ(−½zᵢ²−logstdᵢ−½log2π). This is
   the CONTINUOUS twin of k_sample_md/k_ppo_dout_md, verified vs Puffer/RL/ContVecTrain.lean (AD). */
/* logstd read-clamp. PufferLib clamps to [-20, 2] (safe_continuous_logstd, src/pufferlib.cu:449).
   Our floor was -5, and because our raw-logstd gradient is ALSO gated by the clamp mask (see
   k_ppo_dout_cont), a saturated logstd was ABSORBING: once σ hit e^-5 the gradient became 0 and the
   dim could never recover. Widened to their bound so the reachable σ range matches (2e-9 .. 7.4). */
#define CONT_LSLO (-20.0)
#define CONT_LSHI ( 2.0)
#define CONT_HALFLOG2PI 0.9189385332046727   /* ½·log(2π) */

/* Sample all d dims of the N×O Gaussian batch; out=[actions(d×N, col-major dim i at i·N+n); logp(N);
   value(N)] (size (d+2)·N). Two rng words per (row,dim) for Box–Muller; clamped logstd matches the grad. */
__global__ void k_sample_cont(const double* Yb, double* out, int N, int d, int O, unsigned long long rng){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  const double* row=Yb+(long)n*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  double logp=0.0;
  for(int i=0;i<d;i++){
    double lsr=row[d+i]; double ls = lsr<CONT_LSLO?CONT_LSLO:(lsr>CONT_LSHI?CONT_LSHI:lsr);
    unsigned long long w1=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+1))*G2);
    unsigned long long w2=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+2))*G2);
    double u1=(double)(w1>>11)/9007199254740992.0; if(u1<1.0e-7) u1=1.0e-7;
    double u2=(double)(w2>>11)/9007199254740992.0;
    double z=sqrt(-2.0*log(u1))*cos(6.283185307179586*u2);
    double a=row[i]+exp(ls)*z;
    out[(long)i*N+n]=a; logp += -0.5*z*z - ls - CONT_HALFLOG2PI;
  }
  out[(long)d*N+n]=logp; out[(long)(d+1)*N+n]=row[O-1];
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_sample_actions_cont(lean_obj_arg Yba,
    size_t N, size_t d, size_t O, uint64_t rng){
  const double* Yb=lean_float_array_cptr(Yba);
  size_t outLen=(d+2)*N;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),outLen,outLen); double* out=lean_float_array_cptr(Oo);
  double *dY=NULL,*dO=NULL;
  if(N>0 && cudaMalloc((void**)&dY,8*N*O)==cudaSuccess && cudaMalloc((void**)&dO,8*outLen)==cudaSuccess){
    cudaMemcpy(dY,Yb,8*N*O,cudaMemcpyHostToDevice);
    k_sample_cont<<<ceildiv((long)N,256),256>>>(dY,dO,(int)N,(int)d,(int)O,(unsigned long long)rng);
    cudaMemcpy(out,dO,8*outLen,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<outLen;i++) out[i]=0.0;
  if(dY)cudaFree(dY); if(dO)cudaFree(dO);
  lean_dec(Yba); return Oo;
}

/* ---- f32 segmented MD/Cont samplers (buffered rollout for the wide/multi-discrete-continuous trainer)
   These are the W-wide twins of k_sample_seg_f32/k_sample_seg_g_f32: sample straight from the compact
   per-buffer FLOAT forward output `Yb` (no f64 widening first, matching the single-discrete buffered
   path's own precision trade), global-row rng (rowBase+j) so splitting into buffers reproduces the
   same per-row draw as the whole-batch k_sample_md/k_sample_cont, and a compact [act(K/d × count col);
   logp(count); val(count)] output so each buffer's chunk can sit back-to-back in the shared dO array
   exactly like the single-discrete case's [act;logp;val] triples do. */
__global__ void k_sample_md_seg_f32(const float* Yb, double* out, int count, int rowBase, int K, const int* headSizes, int O, unsigned long long rng){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  const float* row=Yb+(long)j*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  long n=rowBase+j;
  float jointLogp=0.0f; int off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh];
    float m=row[off]; for(int k=1;k<sz;k++) if(row[off+k]>m) m=row[off+k];
    float z=0.0f; for(int k=0;k<sz;k++) z+=expf(row[off+k]-m);
    unsigned long long word=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)hh+1ULL)*G2);
    double u=(double)(word>>11)/9007199254740992.0;
    float acc=0.0f; int a=sz-1;
    for(int k=0;k<sz;k++){ acc+=expf(row[off+k]-m)/z; if(u<(double)acc){ a=k; break; } }
    out[(long)hh*count+j]=(double)a; jointLogp += logf(expf(row[off+a]-m)/z); off+=sz; }
  out[(long)K*count+j]=(double)jointLogp; out[(long)(K+1)*count+j]=(double)row[O-1];
}
__global__ void k_sample_md_seg_g_f32(const float* Yb, double* out, int count, int rowBase, int K, const int* headSizes, int O, const unsigned long long* rngp){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  unsigned long long rng=*rngp;
  const float* row=Yb+(long)j*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  long n=rowBase+j;
  float jointLogp=0.0f; int off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh];
    float m=row[off]; for(int k=1;k<sz;k++) if(row[off+k]>m) m=row[off+k];
    float z=0.0f; for(int k=0;k<sz;k++) z+=expf(row[off+k]-m);
    unsigned long long word=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)hh+1ULL)*G2);
    double u=(double)(word>>11)/9007199254740992.0;
    float acc=0.0f; int a=sz-1;
    for(int k=0;k<sz;k++){ acc+=expf(row[off+k]-m)/z; if(u<(double)acc){ a=k; break; } }
    out[(long)hh*count+j]=(double)a; jointLogp += logf(expf(row[off+a]-m)/z); off+=sz; }
  out[(long)K*count+j]=(double)jointLogp; out[(long)(K+1)*count+j]=(double)row[O-1];
}
__global__ void k_sample_cont_seg_f32(const float* Yb, double* out, int count, int rowBase, int d, int O, unsigned long long rng){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  const float* row=Yb+(long)j*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  long n=rowBase+j;
  float logp=0.0f;
  for(int i=0;i<d;i++){
    float lsr=row[d+i]; float ls = lsr<CONT_LSLO?CONT_LSLO:(lsr>CONT_LSHI?CONT_LSHI:lsr);
    unsigned long long w1=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+1))*G2);
    unsigned long long w2=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+2))*G2);
    double u1=(double)(w1>>11)/9007199254740992.0; if(u1<1.0e-7) u1=1.0e-7;
    double u2=(double)(w2>>11)/9007199254740992.0;
    float z=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2));
    float a=row[i]+expf(ls)*z;
    out[(long)i*count+j]=(double)a; logp += -0.5f*z*z - ls - (float)CONT_HALFLOG2PI;
  }
  out[(long)d*count+j]=(double)logp; out[(long)(d+1)*count+j]=(double)row[O-1];
}
__global__ void k_sample_cont_seg_g_f32(const float* Yb, double* out, int count, int rowBase, int d, int O, const unsigned long long* rngp){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=count) return;
  unsigned long long rng=*rngp;
  const float* row=Yb+(long)j*O;
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  long n=rowBase+j;
  float logp=0.0f;
  for(int i=0;i<d;i++){
    float lsr=row[d+i]; float ls = lsr<CONT_LSLO?CONT_LSLO:(lsr>CONT_LSHI?CONT_LSHI:lsr);
    unsigned long long w1=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+1))*G2);
    unsigned long long w2=d_sm64(rng + ((unsigned long long)n+1ULL)*G + ((unsigned long long)(2*i+2))*G2);
    double u1=(double)(w1>>11)/9007199254740992.0; if(u1<1.0e-7) u1=1.0e-7;
    double u2=(double)(w2>>11)/9007199254740992.0;
    float z=(float)(sqrt(-2.0*log(u1))*cos(6.283185307179586*u2));
    float a=row[i]+expf(ls)*z;
    out[(long)i*count+j]=(double)a; logp += -0.5f*z*z - ls - (float)CONT_HALFLOG2PI;
  }
  out[(long)d*count+j]=(double)logp; out[(long)(d+1)*count+j]=(double)row[O-1];
}

/* Continuous per-row PPO backward. out[0..d)=μ, out[d..2d)=raw logstd, out[2d]=value; actA is N×d
   (row-major, aᵢ at actA[n·d+i]); oldA is the joint old log-prob. Gradients (see ContVecTrain.lean):
     ∂logp/∂μᵢ = zᵢ·e^{−lsᵢ},  ∂logp/∂lsᵢ = zᵢ²−1,  ∂H/∂lsᵢ = 1,  with zᵢ=(aᵢ−μᵢ)e^{−lsᵢ}, lsᵢ clamped.
   The raw-logstd grad is gated by the clamp mask (0 outside [LSLO,LSHI]). One PPO clip on the joint ratio. */
__global__ void k_ppo_dout_cont(const float* Out, const double* actA, const double* advA, const double* retA,
                                const double* oldA, const double* ovA, float* dOut, int N, int d, int O,
                                float vfCoef, float entCoef, float clipEps, float vfClip){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  const float* out=Out+(long)n*O; float* dout=dOut+(long)n*O;
  float adv=(float)advA[n], ret=(float)retA[n], oldLogp=(float)oldA[n];
  float lsLo=(float)CONT_LSLO, lsHi=(float)CONT_LSHI, hlog2pi=(float)CONT_HALFLOG2PI;
  float logp=0.0f;
  for(int i=0;i<d;i++){
    float lsr=out[d+i]; float ls=lsr<lsLo?lsLo:(lsr>lsHi?lsHi:lsr); float invStd=expf(-ls);
    float z=((float)actA[(long)n*d+i]-out[i])*invStd; logp += -0.5f*z*z - ls - hlog2pi;
  }
  float ratio=expf(logp-oldLogp); float lo=1.0f-clipEps, hi=1.0f+clipEps;
  float ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); float surr1=adv*ratio, surr2=adv*ratioC;
  float dPol; if(surr1<=surr2) dPol=adv*ratio; else { float cg=(lo<ratio&&ratio<hi)?1.0f:0.0f; dPol=adv*cg*ratio; }
  for(int i=0;i<d;i++){
    float lsr=out[d+i]; float ls=lsr<lsLo?lsLo:(lsr>lsHi?lsHi:lsr); float invStd=expf(-ls);
    float cmask=(lsLo<lsr && lsr<lsHi)?1.0f:0.0f;
    float z=((float)actA[(long)n*d+i]-out[i])*invStd;
    dout[i]   = dPol*z*invStd;                          /* ∂/∂μᵢ  */
    dout[d+i] = (dPol*(z*z-1.0f) + entCoef)*cmask;       /* ∂/∂(raw logstdᵢ), gated by clamp */
  }
  dout[O-1]=-vfCoef*d_vloss_grad(out[O-1],ret,ovA,n,vfClip);
}

/* Definition of ff_surface_losses (forward-declared before lean_cuda_train_update_resident). Copies the
   already-computed device buffers of ONE minibatch back to host and reduces the 7 PufferLib dashboard
   losses into g_mgLoss — read-only, no kernel launch, no training buffer touched, so it cannot perturb
   the update's determinism. new_logp/new_value/entropy are recomputed from the logits exactly as the
   corresponding head kernel (k_ppo_dout / k_ppo_dout_md / k_ppo_dout_cont) does; the loss composition
   matches the MinGRU block (g_mgLoss[3]=pg+vf·vl−ent·ec, value loss carries the 0.5). */
static void ff_surface_losses(const float* dOut, const double* dac, const double* dan,
    const double* dre, const double* dol, const double* dov, long Nmb, int O,
    int mode, int A, int K, const int* hsHost, int dcont,
    double vfCoef, double entCoef, double clipEps, double vfClip){
  if(Nmb<=0) return;
  long n=Nmb;
  int wact = (mode==1)?K:((mode==2)?dcont:1);   /* action-column stride per row (K heads / d dims / 1) */
  float*  hOut=(float*) malloc(4*(size_t)n*(size_t)O);
  double* hac =(double*)malloc(8*(size_t)n*(size_t)wact);
  double* han =(double*)malloc(8*(size_t)n); double* hre=(double*)malloc(8*(size_t)n);
  double* hol =(double*)malloc(8*(size_t)n); double* hov=dov?(double*)malloc(8*(size_t)n):NULL;
  if(hOut&&hac&&han&&hre&&hol&&(hov||!dov)){
    cudaDeviceSynchronize();
    cudaMemcpy(hOut,dOut,4*(size_t)n*(size_t)O,cudaMemcpyDeviceToHost);
    cudaMemcpy(hac ,dac ,8*(size_t)n*(size_t)wact,cudaMemcpyDeviceToHost);
    cudaMemcpy(han ,dan ,8*(size_t)n,cudaMemcpyDeviceToHost);
    cudaMemcpy(hre ,dre ,8*(size_t)n,cudaMemcpyDeviceToHost);
    cudaMemcpy(hol ,dol ,8*(size_t)n,cudaMemcpyDeviceToHost);
    if(dov) cudaMemcpy(hov,dov,8*(size_t)n,cudaMemcpyDeviceToHost);
    int clip=(dov!=NULL && vfClip>0.0);
    double lo=1.0-clipEps, hi=1.0+clipEps;
    double halfLog2pi=CONT_HALFLOG2PI, halfLog2pieE=0.5+CONT_HALFLOG2PI;  /* ½(1+log2π) = ½ + ½·log2π */
    double sPg=0.0,sV=0.0,sEnt=0.0,sKL=0.0,sOldKL=0.0; long nclip=0;
    for(long r=0;r<n;r++){
      const float* out=hOut+(size_t)r*(size_t)O;
      double newlp=0.0, ent=0.0, newval=0.0;
      if(mode==2){                                     /* continuous diagonal Gaussian (O=2d+1) */
        for(int i=0;i<dcont;i++){
          double lsr=out[dcont+i]; double ls=lsr<CONT_LSLO?CONT_LSLO:(lsr>CONT_LSHI?CONT_LSHI:lsr);
          double z=(hac[(size_t)r*dcont+i]-out[i])*exp(-ls);
          newlp += -0.5*z*z - ls - halfLog2pi;
          ent   += ls + halfLog2pieE;                  /* differential entropy Σ(logstd + ½(1+log2π)) */
        }
        newval=out[O-1];
      } else if(mode==1){                              /* multi-discrete: K categorical heads */
        int off=0;
        for(int hh=0;hh<K;hh++){ int sz=hsHost[hh]; int a=(int)hac[(size_t)r*K+hh];
          double mx=out[off]; for(int k=1;k<sz;k++) if(out[off+k]>mx) mx=out[off+k];
          double se=0.0; for(int k=0;k<sz;k++) se+=exp((double)out[off+k]-mx);
          double lse=mx+log(se);
          newlp += (double)out[off+a]-lse;
          for(int k=0;k<sz;k++){ double lp=(double)out[off+k]-lse; ent -= exp(lp)*lp; }
          off+=sz; }
        newval=out[O-1];
      } else {                                         /* single categorical head (O=A+1) */
        int a=(int)hac[r];
        double mx=out[0]; for(int k=1;k<A;k++) if(out[k]>mx) mx=out[k];
        double se=0.0; for(int k=0;k<A;k++) se+=exp((double)out[k]-mx);
        double lse=mx+log(se);
        newlp=(double)out[a]-lse;
        for(int k=0;k<A;k++){ double lp=(double)out[k]-lse; ent -= exp(lp)*lp; }
        newval=out[A];
      }
      sEnt += ent;                                                  /* accumulate the row's entropy (was dropped ⇒ dashboard showed 0) */
      double adv=han[r], ret=hre[r], oldlp=hol[r];
      double lgr=newlp-oldlp, ratio=exp(lgr), ratioC=ratio<lo?lo:(ratio>hi?hi:ratio);
      double aa=-adv*ratio, bb=-adv*ratioC; sPg += (aa>bb?aa:bb);   /* pg_loss = mean(max(...)) */
      if(ratio<lo||ratio>hi) nclip++;
      sKL += (ratio-1.0)-lgr; sOldKL += -lgr;                        /* approx_kl (k3) / old_approx_kl */
      double du=(newval-ret)*(newval-ret), vloss;
      if(clip){ double vold=hov[r], dd=newval-vold;
        double vc=vold+(dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd)); double cc=(vc-ret)*(vc-ret);
        vloss=0.5*(cc>du?cc:du);
      } else vloss=0.5*du;
      sV += vloss;
    }
    double dn=(double)n, pg=sPg/dn, vlF=sV/dn, entF=sEnt/dn;
    g_mgLoss[0]=pg; g_mgLoss[1]=vlF; g_mgLoss[2]=entF;
    g_mgLoss[3]=pg + vfCoef*vlF - entCoef*entF;                      /* total composite loss */
    g_mgLoss[4]=sOldKL/dn; g_mgLoss[5]=sKL/dn; g_mgLoss[6]=(double)nclip/dn;
  }
  free(hOut);free(hac);free(han);free(hre);free(hol);if(hov)free(hov);
}

/* Continuous minibatch PPO gradient — the Gaussian twin of lean_cuda_mlp_ppo_grad. acts N×d, O=2·d+1. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mlp_ppo_grad_cont(
    lean_obj_arg pa, lean_obj_arg obsBa, lean_obj_arg actsa, lean_obj_arg advsa,
    lean_obj_arg retsa, lean_obj_arg oldlpsa, size_t N, size_t H, size_t D, size_t d,
    double vfCoef, double entCoef, double clipEps, uint8_t bf16Flag){
  size_t O=2*d+1, P=H*D+H+O*H+O;
  const double* pp=lean_float_array_cptr(pa); const double* Xd=lean_float_array_cptr(obsBa);
  const double* actd=lean_float_array_cptr(actsa); const double* advd=lean_float_array_cptr(advsa);
  const double* retd=lean_float_array_cptr(retsa); const double* oldd=lean_float_array_cptr(oldlpsa);
  const double* W1d=pp; const double* b1d=W1d+H*D; const double* W2d=b1d+H; const double* b2d=W2d+O*H;
  lean_object* go=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(go);
  int bf=(int)bf16Flag; cublasHandle_t h=cu_handle();
  float *hX=(float*)malloc(4*N*D),*hW1=(float*)malloc(4*H*D),*hW2=(float*)malloc(4*O*H),*hb1=(float*)malloc(4*H),*hb2=(float*)malloc(4*O);
  float *hgW1=(float*)malloc(4*H*D),*hgb1=(float*)malloc(4*H),*hgW2=(float*)malloc(4*O*H),*hgb2=(float*)malloc(4*O);
  float *dX=NULL,*dW1=NULL,*dW2=NULL,*db1=NULL,*db2=NULL,*dH1=NULL,*dOut=NULL,*dPre=NULL,*ddOut=NULL,*ddH1=NULL,*ddZ=NULL,*dgW1=NULL,*dgb1=NULL,*dgW2=NULL,*dgb2=NULL;
  double *dac=NULL,*dad=NULL,*dre=NULL,*dol=NULL;
  int ok=(N>0 && h!=NULL &&
    !cudaMalloc((void**)&dX,4*N*D) && !cudaMalloc((void**)&dW1,4*H*D) && !cudaMalloc((void**)&dW2,4*O*H) &&
    !cudaMalloc((void**)&db1,4*H) && !cudaMalloc((void**)&db2,4*O) && !cudaMalloc((void**)&dH1,4*N*H) &&
    !cudaMalloc((void**)&dOut,4*N*O) && !cudaMalloc((void**)&dPre,4*N*(H>O?H:O)) && !cudaMalloc((void**)&ddOut,4*N*O) &&
    !cudaMalloc((void**)&ddH1,4*N*H) && !cudaMalloc((void**)&ddZ,4*N*H) && !cudaMalloc((void**)&dgW1,4*H*D) &&
    !cudaMalloc((void**)&dgb1,4*H) && !cudaMalloc((void**)&dgW2,4*O*H) && !cudaMalloc((void**)&dgb2,4*O) &&
    !cudaMalloc((void**)&dac,8*N*d) && !cudaMalloc((void**)&dad,8*N) && !cudaMalloc((void**)&dre,8*N) && !cudaMalloc((void**)&dol,8*N));
  if(ok){
    for(size_t i=0;i<N*D;i++) hX[i]=(float)Xd[i];
    for(size_t i=0;i<H*D;i++) hW1[i]=(float)W1d[i];
    for(size_t i=0;i<O*H;i++) hW2[i]=(float)W2d[i];
    for(size_t i=0;i<H;i++) hb1[i]=(float)b1d[i];
    for(size_t i=0;i<O;i++) hb2[i]=(float)b2d[i];
    cudaMemcpy(dX,hX,4*N*D,cudaMemcpyHostToDevice); cudaMemcpy(dW1,hW1,4*H*D,cudaMemcpyHostToDevice);
    cudaMemcpy(dW2,hW2,4*O*H,cudaMemcpyHostToDevice); cudaMemcpy(db1,hb1,4*H,cudaMemcpyHostToDevice);
    cudaMemcpy(db2,hb2,4*O,cudaMemcpyHostToDevice);
    cudaMemcpy(dac,actd,8*N*d,cudaMemcpyHostToDevice); cudaMemcpy(dad,advd,8*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dre,retd,8*N,cudaMemcpyHostToDevice); cudaMemcpy(dol,oldd,8*N,cudaMemcpyHostToDevice);
    int B=256;
    #define GRC(x) ceildiv((long)(x),B)
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)N,(int)D, dW1,(int)D, dX,(int)D, dPre,(int)H, bf);
    k_relu_bias<<<GRC(N*H),B>>>(dH1,dPre,db1,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,(int)N,(int)H, dW2,(int)H, dH1,(int)H, dPre,(int)O, bf);
    k_add_bias<<<GRC(N*O),B>>>(dOut,dPre,db2,(int)N,(int)O);
    k_ppo_dout_cont<<<GRC(N),B>>>(dOut,dac,dad,dre,dol,NULL,ddOut,(int)N,(int)d,(int)O,(float)vfCoef,(float)entCoef,(float)clipEps,0.0f);
    k_colsum<<<(int)O,256>>>(dgb2,ddOut,(int)N,(int)O);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)H,(int)O,(int)N, dH1,(int)H, ddOut,(int)O, dgW2,(int)H, bf);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)H,(int)N,(int)O, dW2,(int)H, ddOut,(int)O, ddH1,(int)H, bf);
    k_dz1_mask<<<GRC(N*H),B>>>(ddZ,ddH1,dH1,(long)N*H);
    k_colsum<<<(int)H,256>>>(dgb1,ddZ,(int)N,(int)H);
    gemm32(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)D,(int)H,(int)N, dX,(int)D, ddZ,(int)H, dgW1,(int)D, bf);
    #undef GRC
    cudaDeviceSynchronize();
    cudaMemcpy(hgW1,dgW1,4*H*D,cudaMemcpyDeviceToHost); cudaMemcpy(hgb1,dgb1,4*H,cudaMemcpyDeviceToHost);
    cudaMemcpy(hgW2,dgW2,4*O*H,cudaMemcpyDeviceToHost); cudaMemcpy(hgb2,dgb2,4*O,cudaMemcpyDeviceToHost);
    size_t o=0;
    for(size_t i=0;i<H*D;i++) g[o++]=(double)hgW1[i];
    for(size_t i=0;i<H;i++)   g[o++]=(double)hgb1[i];
    for(size_t i=0;i<O*H;i++) g[o++]=(double)hgW2[i];
    for(size_t i=0;i<O;i++)   g[o++]=(double)hgb2[i];
  } else for(size_t i=0;i<P;i++) g[i]=0.0;
  if(dX)cudaFree(dX);if(dW1)cudaFree(dW1);if(dW2)cudaFree(dW2);if(db1)cudaFree(db1);if(db2)cudaFree(db2);
  if(dH1)cudaFree(dH1);if(dOut)cudaFree(dOut);if(dPre)cudaFree(dPre);if(ddOut)cudaFree(ddOut);if(ddH1)cudaFree(ddH1);
  if(ddZ)cudaFree(ddZ);if(dgW1)cudaFree(dgW1);if(dgb1)cudaFree(dgb1);if(dgW2)cudaFree(dgW2);if(dgb2)cudaFree(dgb2);
  if(dac)cudaFree(dac);if(dad)cudaFree(dad);if(dre)cudaFree(dre);if(dol)cudaFree(dol);
  free(hX);free(hW1);free(hW2);free(hb1);free(hb2);free(hgW1);free(hgb1);free(hgW2);free(hgb2);
  lean_dec(pa);lean_dec(obsBa);lean_dec(actsa);lean_dec(advsa);lean_dec(retsa);lean_dec(oldlpsa);
  return go;
}

/* === MinGRU recurrent policy (PufferLib's default net): batched single-step forward for the rollout ===
   Policy = Linear encoder (obs→H) → `L` MinGRU layers (each Wl is 3H×H: hid/gate/proj) → linear action +
   value heads. This is the GPU twin of Puffer/Net/MinGRU.stepForward, matching the C reference
   (lean_ffi_mingru_ppo_grad_seq) op-for-op; the GEMMs use the f32/bf16 gemm32 path (PufferLib policy
   precision), the recurrence math is the exact minGRU cell. One batched step over N envs, carrying the
   N·L·H recurrent state on the device, replacing the O(N·T·L) Lean rollout forward. */
__device__ __forceinline__ float d_sigf(float x){ return 1.0f/(1.0f+expf(-x)); }
__device__ __forceinline__ float d_gactf(float x){ return x>=0.0f ? x+0.5f : d_sigf(x); }

/* per (n,j): minGRU cell for layer `l`. yb is N×3H (hid|gate|proj), hin the layer INPUT N×H (residual);
   stIn/stOut the FULL N×L×H state (read/write slice [n,l,·]). hnew[n,j]=hg·o+(1-hg)·hin; o=(1-z)prev+z·g. */
__global__ void k_mingru_gate(const float* yb, const float* hin, const float* stIn,
                              float* hnew, float* stOut, int N, int L, int H, int l){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*H) return;
  int n=(int)(idx/H), j=(int)(idx%H);
  const float* y=yb+(long)n*3*H;
  float hid=y[j], gate=y[H+j], proj=y[2*H+j];
  float z=d_sigf(gate), gg=d_gactf(hid), prev=stIn[(long)n*L*H + (long)l*H + j];
  float o=(1.0f-z)*prev+z*gg, hg=d_sigf(proj);
  hnew[(long)n*H+j]=hg*o+(1.0f-hg)*hin[(long)n*H+j];
  stOut[(long)n*L*H + (long)l*H + j]=o;
}
/* assemble out[n·O]=[logits(A); value] into a device double buffer. */
__global__ void k_mingru_asm(const float* lg, const float* val, double* out, int N, int A, int O){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*O) return;
  int n=(int)(idx/O), k=(int)(idx%O);
  out[idx]= (k<A) ? (double)lg[(long)n*A+k] : (double)val[n];
}
/* f32 twin (wide-graph rollout arm): assemble straight into a float buffer for k_sample_seg_*_f32 —
   no f64 widening of logits that came out of the f32 forward anyway. */
__global__ void k_mingru_asm_f32(const float* lg, const float* val, float* out, int N, int A, int O){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*O) return;
  int n=(int)(idx/O), k=(int)(idx%O);
  out[idx]= (k<A) ? lg[(long)n*A+k] : val[n];
}

/* ===== bf16-STORAGE wide rollout arm (PufferLib's exact rollout precision/shape, source-diffed
   2026-08-03) =====================================================================================
   Their squared trainer: 2 buffers x 2048 rows, bf16 weights+activations end-to-end, 3 plain
   cublasGemmEx(CUDA_R_16BF, COMPUTE_32F) GEMMs + elementwise per step, one PRE-CAPTURED graph per
   (t,buf), u8 obs DMA'd then cast in-graph, f32-math sampler over bf16 logits. This arm replicates
   that config inside our stack while preserving the architecture invariants: the f64 resident master
   weights stay authoritative (bf16 working copy cast once per update), the sampler keeps OUR
   splitmix64 per-global-row stream (seed via a device scalar updated once per update, per-step t
   BAKED into each pre-captured graph), and the BPTT still trains on exactly what the policy saw (the
   obs traj scatter widens the SAME bf16 values the forward consumed; for u8 envs bf16(u8) is exact).
   TOLERANCE-CLASS tier (bf16 activations round every stage) — gated by PUFFER_MG_WIDEBF (1 force /
   0 off / unset = auto at H>=128, the shape regime where the fused kernel measured behind PufferLib;
   pong h32 / breakout h64 and every other trainer are untouched by the auto gate). */
__global__ void k_d2bf(__nv_bfloat16* dst, const double* src, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16((float)src[i]); }
__global__ void k_u82bf(__nv_bfloat16* dst, const unsigned char* src, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16((float)src[i]); }   /* u8→bf16 exact (≤8 sig bits) */
__global__ void k_f2bf_n(__nv_bfloat16* dst, const float* src, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16(src[i]); }
__global__ void k_bf2f_n(float* dst, const __nv_bfloat16* src, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__bfloat162float(src[i]); }
__global__ void k_add_bias_bfw(__nv_bfloat16* Y, const __nv_bfloat16* b, long N, int Hc){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=N*Hc) return;
  Y[i]=__float2bfloat16(__bfloat162float(Y[i])+__bfloat162float(b[i%Hc])); }
/* pack the (A+1)xH combined head [wDec rows; wVal] + (A+1) bias [bDec; bVal] from the flat bf16 copy */
__global__ void k_mg_pack_head_bf(__nv_bfloat16* wH, __nv_bfloat16* bH, const __nv_bfloat16* wDec,
    const __nv_bfloat16* bDec, const __nv_bfloat16* wVal, const __nv_bfloat16* bVal, int A, int H){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; long tot=(long)(A+1)*H; if(i>=tot) return;
  int r=(int)(i/H), cc=(int)(i%H);
  wH[i]=(r<A)? wDec[(long)r*H+cc] : wVal[cc];
  if(i<(long)A+1) bH[i]=(i<A)? bDec[i] : bVal[0];
}
/* bf16 MinGRU gate (f32 internal math, mirrors k_mingru_gate) with the terminal reset FOLDED into the
   state read (terms = previous step's terminals, NULL at s==0) — same fold the fused kernel uses. */
__global__ void k_mingru_gate_bfw(const __nv_bfloat16* yb, const __nv_bfloat16* hin, __nv_bfloat16* st,
    __nv_bfloat16* hnew, const double* terms, int N, int L, int H, int l){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*H) return;
  int n=(int)(idx/H), j=(int)(idx%H);
  const __nv_bfloat16* y=yb+(long)n*3*H;
  float hid=__bfloat162float(y[j]), gate=__bfloat162float(y[H+j]), proj=__bfloat162float(y[2*H+j]);
  float prev=(terms && terms[n]!=0.0)? 0.0f : __bfloat162float(st[(long)n*L*H+(long)l*H+j]);
  float z=d_sigf(gate), gg=d_gactf(hid);
  float o=__fmaf_rn(z,gg,__fmaf_rn(-z,prev,prev)), hg=d_sigf(proj);   /* explicit FMA: tolerance tier */
  float hi2=__bfloat162float(hin[(long)n*H+j]);
  hnew[(long)n*H+j]=__float2bfloat16(__fmaf_rn(hg,o,__fmaf_rn(-hg,hi2,hi2)));
  st[(long)n*L*H+(long)l*H+j]=__float2bfloat16(o);
}
__global__ void k_mg_reset_bfw(__nv_bfloat16* st, const double* terms, int N, int LH){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)N*LH) return; int n=(int)(idx/LH);
  if(terms[n]!=0.0) st[idx]=__float2bfloat16(0.0f); }
/* obs traj scatter, bf16 source → f32 traj (the BPTT trains on exactly what this forward consumed) */
__global__ void k_scatter_mg_obs_bfw(float* traj, const __nv_bfloat16* xb, long rb, long nb, long s, long T, long D){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=nb*D) return;
  long le=idx/D, j=idx%D; traj[((rb+le)*T+s)*D+j]=__bfloat162float(xb[le*D+j]); }
/* f32-math sampler over bf16 logits: OUR splitmix64 per-global-row stream; the per-update rolloutRng
   comes from a device scalar (updated once per update) and the per-step offset t·N·G is BAKED into
   each pre-captured graph. Writes actions compact (dOact, D2H) + the device-direct resident columns. */
__global__ void k_sample_wbf(const __nv_bfloat16* Lg, const __nv_bfloat16* bH, double* dOact, int nb, long rowBase, int A, int O,
    const unsigned long long* rngp, unsigned long long tOff,
    double* cAct, double* cLogp, double* cVal0, double* cValue, long trajRow0, long T_traj, long sStep){
  int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=nb) return;
  const __nv_bfloat16* row=Lg+(long)j*O;
  /* head bias folded in here (f32 add, no bf16 re-round). (Inline per-thread head DOTS were tried to
     kill the head-GEMM node too and REVERTED: 768 serial MACs/thread stretched the graph 30->92us.) */
  float lg[64];                                   /* A<=63 for this arm (auto-gate shapes are small-A) */
  float m=__bfloat162float(row[0])+__bfloat162float(bH[0]); lg[0]=m;
  for(int k=1;k<A;k++){ lg[k]=__bfloat162float(row[k])+__bfloat162float(bH[k]); if(lg[k]>m) m=lg[k]; }
  float z=0.0f; for(int k=0;k<A;k++) z+=expf(lg[k]-m);
  unsigned long long seed=(*rngp)+tOff;
  unsigned long long word=d_sm64(seed+((unsigned long long)(rowBase+j)+1ULL)*0x9E3779B97F4A7C15ULL);
  double u=(double)(word>>11)/9007199254740992.0;
  float accf=0.0f; int a=A-1;
  for(int k=0;k<A;k++){ accf+=expf(lg[k]-m)/z; if(u<(double)accf){ a=k; break; } }
  double lp=(double)logf(expf(lg[a]-m)/z), v=(double)(__bfloat162float(row[A])+__bfloat162float(bH[A]));
  dOact[j]=(double)a;
  if(cAct){ long r=((long)trajRow0+j)*T_traj+sStep;              /* global row = trajRow0(=rb) + local j */
    cAct[r]=(double)a; cLogp[r]=lp; cVal0[r]=v; cValue[r]=v; }
}
static inline cublasStatus_t gemm_bfx(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc){
  float al=1.0f, be=0.0f;                          /* their exact call: R_16BF storage, f32 compute */
  return cublasGemmEx(h,opA,opB,m,n,k,&al,A,CUDA_R_16BF,lda,B,CUDA_R_16BF,ldb,&be,C,CUDA_R_16BF,ldc,
                      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
__global__ void k_d2f(float* dst, const double* src, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(float)src[i]; }
__global__ void k_f2d(double* dst, const float* src, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(double)src[i]; }
__global__ void k_u82f(float* dst, const unsigned char* src, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=(float)src[i]; }
/* zero-copy host->device staging as a COMPUTE kernel: on this vGPU every copy-engine H2D on the
   rollout stream pays its own submission/handoff latency (~20-25us); a kernel reading pinned host
   over UVA does the same bytes coalesced with no engine round. */
__global__ void k_h2d_f64(double* dst, const double* __restrict__ srcHost, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=srcHost[i]; }
/* wide-arm PROLOGUE mega-node: u8->bf16 obs cast + f32 traj scatter + encoder-bias broadcast into
   dH1 (the enc GEMM then runs beta=1) + terms zero-copy staging — four former nodes in ONE. The
   per-NODE overhead in this process measured ~6us unprofiled (9-node graph = 74us for ~24us of
   kernel time), so node-count is the currency here. */
__global__ void k_wbf_prologue(__nv_bfloat16* dX, const unsigned char* dU8, float* traj,
    long rb, long nb, long s, long T, long D,
    __nv_bfloat16* dH1, const __nv_bfloat16* bEnc, int H,
    double* dTm, const double* __restrict__ termsHost){
  long nObs=nb*D, nH=nb*(long)H;
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nObs){
    float v=(float)dU8[i];
    dX[i]=__float2bfloat16(v);
    if(traj){ long le=i/D, j=i%D; traj[((rb+le)*T+s)*D+j]=v; }
  } else if(i<nObs+nH){
    long j=i-nObs; dH1[j]=bEnc[j%H];
  } else if(i<nObs+nH+nb){
    long k=i-nObs-nH; if(termsHost) dTm[k]=termsHost[k];
  }
}
static inline cublasStatus_t gemm_bfx_b1(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb,
    __nv_bfloat16* C, int ldc){
  float al=1.0f, be=1.0f;                          /* beta=1: C prefilled with the bias broadcast */
  return cublasGemmEx(h,opA,opB,m,n,k,&al,A,CUDA_R_16BF,lda,B,CUDA_R_16BF,ldb,&be,C,CUDA_R_16BF,ldc,
                      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
/* completion stamp: LAST node of the wide per-step graph writes a per-step value to pinned host; the
   worker spins on it instead of cudaStreamSynchronize (measured 123us/step wait for 28.5us of graph —
   the runtime's sync path, not GPU latency: an empty launch+sync round trips in 4.2us on this box).
   Same-device PCIe posted writes arrive in order, so the stamp lands after the sampler's actPin. */
__global__ void k_stamp32(volatile unsigned* p, unsigned v){ if(threadIdx.x==0) *p=v; }
/* GPU keep-warm: the rollout's ~20%% duty cycle (24us graphs every ~125us) lets DVFS drop the SM
   clock to 232-772MHz between launches (measured mid-run), so every graph executes on a ramping
   clock — the true identity of the "sync wall" (nsys pins clocks, which is why profiled runs were
   mysteriously 1.5x faster). One 32-thread block spinning on %%globaltimer on a LOWEST-priority
   stream holds the clock domain at boost for `ns` wall-nanoseconds; ~zero SM displacement. */
__global__ void k_warm(unsigned long long ns){
  unsigned long long t0,t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t0)); t=t0;
  while(t-t0<ns){ asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); }
}
static void mg_warm_kick(double ms){
  static cudaStream_t ws=NULL; static int en=-1; static cudaGraphExec_t wg=NULL;
  if(en<0){ const char* e=getenv("PUFFER_GPU_WARM"); en=(e!=NULL&&e[0]=='1')?1:0; }   /* opt-in:
    a MONOLITHIC 25ms spin kernel STARVED the rollout (20.7 -> 9.4M): kernel execution is
    effectively serialized across channels here, so the hog blocked everything. This version slices
    the warm time into 20us chunk kernels inside ONE graph — every node boundary is an interleave
    point for real work; one graph launch covers ~`ms` of clock-pinning. */
  if(!en) return;
  if(!ws){ int lo=0,hi=0; cudaDeviceGetStreamPriorityRange(&lo,&hi);
    if(cudaStreamCreateWithPriority(&ws,cudaStreamNonBlocking,lo)!=cudaSuccess){ ws=NULL; en=0; return; } }
  if(!wg){
    int nchunk=(int)(ms*1e3/20.0); if(nchunk<1) nchunk=1; if(nchunk>4000) nchunk=4000;
    cudaGraph_t g=NULL; cudaStreamBeginCapture(ws,cudaStreamCaptureModeThreadLocal);
    for(int i=0;i<nchunk;i++) k_warm<<<1,32,0,ws>>>(20000ULL);
    if(cudaStreamEndCapture(ws,&g)!=cudaSuccess || !g || cudaGraphInstantiate(&wg,g,0)!=cudaSuccess){
      if(g) cudaGraphDestroy(g); en=0; cudaGetLastError(); return; }
    cudaGraphDestroy(g);
  }
  if(cudaStreamQuery(ws)==cudaSuccess) cudaGraphLaunch(wg,ws);   /* re-arm only when drained */
  cudaGetLastError();
}

/* Device-resident MinGRU obs trajectory (f32, [N·T·D], segment-major row e·T+s) — mirrors the MLP's
   g_dObsTraj. The rollout scatters the per-step f32 obs here (dObsF is already (float)obs, so this is the
   SAME value the BPTT's k_d2f would produce ⇒ bit-identical); the BPTT then gathers the sampled segments'
   obs on-device (device→device, ~free) instead of the host gather (7.7MB/minibatch copy) + f64 obs H2D.
   Persists between the rollout FFI and the per-minibatch BPTT FFI calls; g_dMGObsTraj_valid is set by the
   rollout, g_dMGObsTraj_N records the segment count for the BPTT's size guard. */
static float* g_dMGObsTraj=NULL; static size_t g_dMGObsTrajSz=0; static int g_dMGObsTraj_valid=0; static long g_dMGObsTraj_N=0;
static float* mgobstraj_buf(size_t elems){ size_t bytes=elems*4;
  if(g_dMGObsTrajSz<bytes){ if(g_dMGObsTraj)cudaFree(g_dMGObsTraj);
    if(cudaMalloc((void**)&g_dMGObsTraj,bytes)!=cudaSuccess){ g_dMGObsTraj=NULL; g_dMGObsTrajSz=0; return NULL; }
    g_dMGObsTrajSz=bytes; }
  return g_dMGObsTraj; }
/* scatter step-s obs (xf=[nb·D] f32, a buffer slice) into the trajectory at global row (rb+le)·T+s. */
__global__ void k_scatter_mg_obs(float* traj, const float* xf, long rb, long nb, long s, long T, long D){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=nb*D) return;
  long le=idx/D, j=idx%D; traj[((rb+le)*T+s)*D+j]=xf[le*D+j]; }
/* gather the sampled minibatch into the BPTT's timestep-major dObsF[(t·Bmb+bi)·D+j] = traj[(seg[bi]·T+t)·D+j]. */
__global__ void k_gather_mg_obs(float* dObsF, const float* traj, const double* segIdx, long Bmb, long T, long D){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=Bmb*T*D) return;
  long bt=idx/D, j=idx%D; long t=bt/Bmb, bi=bt%Bmb; long e=(long)segIdx[bi];
  dObsF[bt*D+j]=traj[(e*T+t)*D+j]; }

/* fused rollout forward (defined below, used by both the buffered rollout and — for verify coverage —
   the step FFI when PUFFER_MG_FUSED is on) */
#define MG_FR 8
__global__ void k_mg_fused_step(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A, const unsigned long long* rngDev, unsigned long long tOff);
static int mg_fused(void);
static size_t mg_fused_shbytes(int D,int H,int L,int A);
static size_t mg_fused_shbytes_fr(int D,int H,int L,int A,int FR);
__global__ void k_mg_fused_step_p(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A, const unsigned long long* rngDev, unsigned long long tOff);
static int mg_pipe(void);
static size_t mg_pipe_shbytes(int D,int H,int L,int A,int FR);
static int mg_pipe_optin(size_t shb);
__global__ void k_mg_fused_step_w(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A, const unsigned long long* rngDev, unsigned long long tOff);
#define MG_WFR_DECL 16
static int mg_wmma(void);
static size_t mg_wmma_shbytes(int D,int H,int L,int A);
static int mg_wmma_optin(size_t shb);
static void mg_pub_wencpad(const float* dP, int H, int D, int L, int A);
static void mg_pub_wlbf(const float* dP, int H, int D, int L);
/* stream-ordered completion stamp: runs AFTER the preceding D2H in the stream, so the host spinning on
   the pinned flag knows the samples have landed — WITHOUT a contended cudaStreamSynchronize call and
   WITHOUT fence_system in the hot kernel (the in-kernel zero-copy variant measured 4x kernel slowdown). */
__global__ void k_mg_stamp(int* flagH, int stamp){ __threadfence_system(); *flagH=stamp; }
static int mg_fused_optin(size_t shb);
__global__ void k_cpyf(float* d, const float* s, long n);
/* params(f64): wEnc[H·D],bEnc[H],layers[L·3H·H],wDec[A·H],bDec[A],wVal[H],bVal[1]. obs f64 N·D, state
   f64 N·L·H. Returns [out(N·O=A+1: logits then value); newState(N·L·H)] (f64), size N·O + N·L·H. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mingru_step(
    lean_obj_arg pa, lean_obj_arg obsa, lean_obj_arg statea,
    size_t N, size_t D, size_t H, size_t L, size_t A, uint8_t bf16){
  size_t O=A+1, wEncSz=H*D, layerSz=3*H*H, P=wEncSz+H+L*layerSz+A*H+A+H+1;
  size_t outLen=N*O + N*L*H;
  const double* pp=lean_float_array_cptr(pa); const double* obsd=lean_float_array_cptr(obsa); const double* std_=lean_float_array_cptr(statea);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),outLen,outLen); double* out=lean_float_array_cptr(Oo);
  int bf=(int)bf16; cublasHandle_t hbl=cu_handle();
  float* hP=(float*)malloc(4*P); for(size_t i=0;i<P;i++) hP[i]=(float)pp[i];
  float *dP=NULL,*dHb=NULL,*dHn=NULL,*dYb=NULL,*dSt=NULL,*dStInF=NULL,*dLg=NULL,*dVal=NULL,*dObsF=NULL; double *dObs=NULL,*dStIn=NULL,*dOutO=NULL;
  int ok = (N>0 && hbl!=NULL &&
    !cudaMalloc((void**)&dP,4*P) && !cudaMalloc((void**)&dHb,4*N*H) && !cudaMalloc((void**)&dHn,4*N*H) &&
    !cudaMalloc((void**)&dYb,4*N*3*H) && !cudaMalloc((void**)&dSt,4*N*L*H) && !cudaMalloc((void**)&dStInF,4*N*L*H) &&
    !cudaMalloc((void**)&dLg,4*N*A) && !cudaMalloc((void**)&dVal,4*N) && !cudaMalloc((void**)&dObsF,4*N*D) &&
    !cudaMalloc((void**)&dObs,8*N*D) && !cudaMalloc((void**)&dStIn,8*N*L*H) && !cudaMalloc((void**)&dOutO,8*outLen));
  if(ok){
    cudaMemcpy(dP,hP,4*P,cudaMemcpyHostToDevice);
    cudaMemcpy(dObs,obsd,8*N*D,cudaMemcpyHostToDevice); cudaMemcpy(dStIn,std_,8*N*L*H,cudaMemcpyHostToDevice);
    const float* dWEnc=dP; const float* dBEnc=dP+wEncSz; const float* dLayers=dP+wEncSz+H;
    const float* dWDec=dLayers+L*layerSz; const float* dBDec=dWDec+A*H; const float* dWVal=dBDec+A; const float* dBVal=dWVal+H;
    int B=256;
    #define GG(x) ceildiv((long)(x),B)
    k_d2f<<<GG(N*D),B>>>(dObsF,dObs,(long)N*D);
    k_d2f<<<GG(N*L*H),B>>>(dStInF,dStIn,(long)N*L*H);
    size_t shb=mg_fused()? mg_fused_shbytes((int)D,(int)H,(int)L,(int)A) : 0;
    size_t shbWv=(mg_fused() && mg_wmma())? mg_wmma_shbytes((int)D,(int)H,(int)L,(int)A) : 0;
    if(shbWv){ mg_pub_wencpad(dP,(int)H,(int)D,(int)L,(int)A);   /* pad from THIS call's params — the
       verify entry must exercise (and never inherit) the big-H WMMA-encoder state */
      mg_pub_wlbf(dP,(int)H,(int)D,(int)L); }
    /* mirror the rollout worker's selection: the WMMA variant can fit where the plain layout cannot
       (big H) — the verify entry must exercise the SAME kernel the rollout runs */
    if((shb && mg_fused_optin(shb)) || (shbWv && mg_wmma_optin(shbWv))){
      /* fused forward (in-place state: seed dSt with the input state, kernel reads prev + writes o) */
      k_cpyf<<<GG(N*L*H),B>>>(dSt,dStInF,(long)N*L*H);
      { size_t shbW2=shbWv;
        size_t shbP2=(shb && 8*H<=1024 && mg_pipe())? mg_pipe_shbytes((int)D,(int)H,(int)L,(int)A,MG_FR) : 0;
        if(shbW2 && mg_wmma_optin(shbW2))
          k_mg_fused_step_w<<<(int)((N+15)/16),512,shbW2>>>(dP, dObsF, 0, 16, NULL,0,0,0, dSt,(long)L*H, NULL,
            NULL,0,0,0ULL, NULL,NULL,NULL,NULL, dOutO,(int)N,(int)D,(int)H,(int)L,(int)A, NULL, 0ULL);
        else if(shbP2 && mg_pipe_optin(shbP2))
          k_mg_fused_step_p<<<(int)((N+MG_FR-1)/MG_FR),(int)(8*H),shbP2>>>(dP, dObsF, 0, MG_FR, NULL,0,0,0, dSt,(long)L*H, NULL,
            NULL,0,0,0ULL, NULL,NULL,NULL,NULL, dOutO,(int)N,(int)D,(int)H,(int)L,(int)A, NULL, 0ULL);
        else
          k_mg_fused_step<<<(int)((N+MG_FR-1)/MG_FR),(int)((8*H<=1024)?8*H:4*H),shb>>>(dP, dObsF, 0, MG_FR, NULL,0,0,0, dSt,(long)L*H, NULL,
            NULL,0,0,0ULL, NULL,NULL,NULL,NULL, dOutO,(int)N,(int)D,(int)H,(int)L,(int)A, NULL, 0ULL); }
    } else {
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)N,(int)D, dWEnc,(int)D, dObsF,(int)D, dHb,(int)H, bf);
    k_add_bias<<<GG(N*H),B>>>(dHb,dHb,dBEnc,(int)N,(int)H);
    for(size_t l=0;l<L;l++){
      const float* Wl=dLayers+l*layerSz;
      gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)(3*H),(int)N,(int)H, Wl,(int)H, dHb,(int)H, dYb,(int)(3*H), bf);
      k_mingru_gate<<<GG(N*H),B>>>(dYb,dHb,dStInF,dHn,dSt,(int)N,(int)L,(int)H,(int)l);
      float* tmp=dHb; dHb=dHn; dHn=tmp;   /* hb = hnew */
    }
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)A,(int)N,(int)H, dWDec,(int)H, dHb,(int)H, dLg,(int)A, bf);
    k_add_bias<<<GG(N*A),B>>>(dLg,dLg,dBDec,(int)N,(int)A);
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,1,(int)N,(int)H, dWVal,(int)H, dHb,(int)H, dVal,1, bf);
    k_add_bias<<<GG(N),B>>>(dVal,dVal,dBVal,(int)N,1);
    k_mingru_asm<<<GG(N*O),B>>>(dLg,dVal,dOutO,(int)N,(int)A,(int)O);
    }
    k_f2d<<<GG(N*L*H),B>>>(dOutO+N*O,dSt,(long)N*L*H);
    cudaMemcpy(out,dOutO,8*outLen,cudaMemcpyDeviceToHost);
    #undef GG
  } else for(size_t i=0;i<outLen;i++) out[i]=0.0;
  if(dP)cudaFree(dP);if(dHb)cudaFree(dHb);if(dHn)cudaFree(dHn);if(dYb)cudaFree(dYb);if(dSt)cudaFree(dSt);
  if(dStInF)cudaFree(dStInF);if(dLg)cudaFree(dLg);if(dVal)cudaFree(dVal);if(dObsF)cudaFree(dObsF);
  if(dObs)cudaFree(dObs);if(dStIn)cudaFree(dStIn);if(dOutO)cudaFree(dOutO);
  free(hP); lean_dec(pa);lean_dec(obsa);lean_dec(statea);
  return Oo;
}

/* === MinGRU BPTT PPO gradient (batched over B sequences of length T) — GPU twin of the C reference
   lean_ffi_mingru_ppo_grad_seq. Forward stores activations, backward runs the PPO objective BPTT. All
   matmuls via cublasSgemm (f32; beta=1 accumulates a gradient across timesteps); custom kernels for the
   minGRU cell fwd/bwd + PPO. Inputs laid out [T][B][…] (per timestep contiguous over the batch). ===== */
/* row-major C[M×N] = opA(A)·opB(B) via the col-major swap trick: pass B then A. beta accumulates. */
/* bf16 tensor cores for the MinGRU BPTT GEMMs — DEFAULT ON (set PUFFER_MG_BF16=0 to force f32). bf16 uses
   the tensor cores for the compute-bound recurrent backward (breakout MinGRU BPTT ~−7%), matches what
   PufferLib itself trains in, and makes the gradient's forward consistent with our already-bf16 rollout
   forward (old_logp and new_logp both bf16). Trades bf16-precision gradients (converge a touch slower).
   verify-mingru-grad-gpu forces f32 (PUFFER_MG_BF16=0) to keep its tight 1e-4 kernel-LOGIC check — bf16 is
   that same logic with tensor-core GEMMs. Resolved once. */
static int mg_bf16(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_BF16"); f=(e==NULL||e[0]!='0'); } return f; }
static cublasStatus_t sgemm_rm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc, float beta){
  float al=1.0f;
  /* col-major C[N×M] = opB(B[N×K])·opA(A[K×M]); ld's are the row-major col-counts. */
  if(mg_bf16()) return cublasGemmEx(h, opB, opA, N, M, K, &al, B, CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
  return cublasSgemm(h, opB, opA, N, M, K, &al, B, ldb, A, lda, &beta, C, ldc);
}
/* DETERMINISTIC parallel column-sum for the BPTT bias grads — the old serial k_colsum_acc was 675µs/launch
   (J threads walking all T·B rows; J=1 for the value bias) = 28% of train. Two fixed-shape stages: block
   (c,j) tree-reduces a fixed row-chunk of column j → part[c][j]; a tiny second kernel folds the C partials
   in fixed order. No atomics on floats ⇒ run-to-run deterministic (order ≠ the serial loop's ⇒ f32-tolerance
   change, like the earlier parallelized reductions — covered by verify-mingru-grad-gpu). */
#define CSUM_C 64
__global__ void k_colsum_part(float* part, const float* M, long N, int J){
  int j=blockIdx.y, c=blockIdx.x;
  long R=(N+gridDim.x-1)/gridDim.x, r0=(long)c*R, r1=(r0+R>N)?N:(r0+R);
  __shared__ float sh[256];
  float s=0.0f;
  for(long r=r0+threadIdx.x; r<r1; r+=blockDim.x) s+=M[r*J+j];
  sh[threadIdx.x]=s; __syncthreads();
  for(int st=128;st>0;st>>=1){ if((int)threadIdx.x<st) sh[threadIdx.x]+=sh[threadIdx.x+st]; __syncthreads(); }
  if(threadIdx.x==0) part[(long)c*J+j]=sh[0]; }
__global__ void k_colsum_fin(float* g, const float* part, int C, int J){
  int j=(int)((long)blockIdx.x*blockDim.x+threadIdx.x); if(j>=J) return;
  float s=0.0f; for(int c=0;c<C;c++) s+=part[(long)c*J+j]; g[j]+=s; }
static void colsum_acc_dev(float* g, const float* M, long N, int J, float* part, cudaStream_t st=0){
  dim3 grid(CSUM_C, J); k_colsum_part<<<grid,256,0,st>>>(part, M, N, J);
  k_colsum_fin<<<(J+255)/256,256,0,st>>>(g, part, CSUM_C, J); }
__global__ void k_cpyf(float* d, const float* s, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=s[i]; }
/* ===== bf16-STORAGE path for the MinGRU BPTT (PUFFER_MG_PREC=bf16) ==============================
   The batched BPTT GEMMs [3H×(T·B)×H] have K=H=64 ⇒ arithmetic intensity ~24 FLOP/byte << the ~66
   balance point ⇒ MEMORY-bound. mg_bf16() (f32 storage + bf16 tensor cores) leaves the memory traffic
   at f32 (4B) so it can't help; bf16 STORAGE (2B) halves the traffic → ~2× on these GEMMs, matching
   PufferLib's measured f32→bf16 train speedup. Same operation graph as the f32 spec — a coarser
   precision INSTANTIATION, verified vs the Lean f64 oracle at bf16 tolerance. GRADIENTS ACCUMULATE IN
   F32 (bf16's 8-bit mantissa can't sum thousands of rows): the grad GEMMs output f32 (gemm_bf2f). The
   small heads/PPO stay f32 (negligible bytes) with two boundary casts (aHfin, dDhfAll). Default OFF. */
typedef __nv_bfloat16 bf16;
static int mg_bf16store(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_PREC"); f=(e!=NULL && (e[0]=='b'||e[0]=='B')); } return f; }
__global__ void k_f2bf(bf16* d, const float* s, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2bfloat16(s[i]); }
__global__ void k_bf2f(float* d, const bf16* s, long n){ long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__bfloat162float(s[i]); }
/* row-major bf16 GEMMs via the col-major swap trick (pass B then A). f32 accumulate (COMPUTE_32F).
   NB: cuBLAS bf16 GEMMs require EVEN leading dimensions — callers must keep bf16 lds ∈ {H,3H}, H even
   (the encoder GEMMs, whose ld is the arbitrary env obs dim D, stay f32). */
static cublasStatus_t gemm_bf(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int M,int N,int K, const bf16* A,int lda, const bf16* B,int ldb, bf16* C,int ldc, float beta){
  float al=1.0f;
  return cublasGemmEx(h,opB,opA,N,M,K,&al, B,CUDA_R_16BF,ldb, A,CUDA_R_16BF,lda, &beta, C,CUDA_R_16BF,ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
static cublasStatus_t gemm_bf2f(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int M,int N,int K, const bf16* A,int lda, const bf16* B,int ldb, float* C,int ldc, float beta){
  float al=1.0f;
  return cublasGemmEx(h,opB,opA,N,M,K,&al, B,CUDA_R_16BF,ldb, A,CUDA_R_16BF,lda, &beta, C,CUDA_R_32F,ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
/* cast+TRANSPOSE the f32 layer weights → bf16 WlT[l][H×3H] (row-major). bf16 GEMMs validate lds strictly
   (f32 tolerates the fwd-layer OP_T pattern's ldb; bf16 rejects it) — with Wl stored transposed, both
   weight-using GEMMs become by-the-book patterns: dIn·Wlᵀ = dIn·WlT (OP_N/OP_N) and dy·Wl = dy·WlTᵀ
   (OP_N/OP_T), same products. */
__global__ void k_f2bf_lT(bf16* WT, const float* W, int L, int H){
  long lay=(long)3*H*H, i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)L*lay) return;
  long l=i/lay, r=(i%lay)/H, c=(i%lay)%H;                      /* W[l][r<3H][c<H] */
  WT[l*lay + c*3*H + r]=__float2bfloat16(W[i]);
}
__global__ void k_mg_reset(float* st, const double* terms, int B, int LH, int t){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*LH) return; int b=(int)(idx/LH);
  if(terms[(long)t*B+b]!=0.0) st[idx]=0.0f; }

/* --- Native per-update MinGRU rollout (recurrent single-discrete). Same win as the MLP rollout driver
   but for the PufferLib-default recurrent net: weights upload+convert to f32 ONCE (they were re-uploaded
   ~1.7MB EVERY timestep by cudaMinGRUStepFFI), and the recurrent state stays f32 RESIDENT on device
   (threaded in place + reset on terminals via k_mg_reset), instead of round-tripping N·L·H f64 per step.
   Per step: obs H2D → resident MinGRU forward (encoder→L layers→heads) → assemble logits+value →
   device k_sample → sample D2H → CPU plugin env-step → device state-reset on terminals → host column
   scatter. Bit-identical to the old loop (same mingru_fwd_dev kernels, same k_sample rng, same reset).
   Threads obs AND recurrent state across updates; returns
   [obsCol(NT·D); actCol; logpCol; valCol; rewCol; termCol; finalObs(N·D); finalState(N·L·H); bootVals(N)].
   bootVals = value at (finalObs, finalState) — the bootstrap the old code computed with an extra step. */
/* st = the stream all kernels launch on (caller must cublasSetStream(hbl,st)); lets the concurrent-buffer
   workers run each buffer's forward on its own stream. dStCur==dStNxt allowed (in-place state, race-free). */
static void mingru_fwd_dev(cublasHandle_t hbl, const float* dP, const float* dObsF, const float* dStCur,
    float* dStNxt, float* dHb, float* dHn, float* dYb, float* dLg, float* dVal, int N, int D, int H, int L, int A, int bf, cudaStream_t st){
  int Bk=256; size_t wEncSz=(size_t)H*D, layerSz=(size_t)3*H*H;
  const float* dWEnc=dP; const float* dBEnc=dP+wEncSz; const float* dLayers=dP+wEncSz+H;
  const float* dWDec=dLayers+(size_t)L*layerSz; const float* dBDec=dWDec+(size_t)A*H; const float* dWVal=dBDec+A; const float* dBVal=dWVal+H;
  #define MF(x) ceildiv((long)(x),Bk)
  gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,H,N,D, dWEnc,D, dObsF,D, dHb,H, bf);
  k_add_bias<<<MF((long)N*H),Bk,0,st>>>(dHb,dHb,dBEnc,N,H);
  for(int l=0;l<L;l++){ const float* Wl=dLayers+(size_t)l*layerSz;
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,3*H,N,H, Wl,H, dHb,H, dYb,3*H, bf);
    k_mingru_gate<<<MF((long)N*H),Bk,0,st>>>(dYb,dHb,dStCur,dHn,dStNxt,N,L,H,l);
    float* tmp=dHb; dHb=dHn; dHn=tmp; }
  gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,A,N,H, dWDec,H, dHb,H, dLg,A, bf);
  k_add_bias<<<MF((long)N*A),Bk,0,st>>>(dLg,dLg,dBDec,N,A);
  gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,1,N,H, dWVal,H, dHb,H, dVal,1, bf);
  k_add_bias<<<MF((long)N),Bk,0,st>>>(dVal,dVal,dBVal,N,1);
  #undef MF
}
/* ===== FUSED per-step rollout forward =========================================================
   The per-step forward was 5 GEMMs + 8 kernels + 3 transfers per (step, buffer) — with 64 steps × 4
   buffers that's ~4K GPU ops/update of 1-10µs kernels: the rollout pays kernel FLOORS and launch
   overhead, not compute (~57K FLOP/row). This kernel does the WHOLE step in one launch: obs load
   (+ trajectory scatter), encoder, L gate layers, decoder/value heads, and the categorical sample.
   One block per row (env), blockDim=H: thread j computes exactly the 3 combined-projection dots its
   gate consumes (rows j, H+j, 2H+j of Wl) — no inter-thread exchange of y needed; the hidden vector
   lives in shared. Weights stream from L2 (~250KB resident). Sequential-FMA dots ⇒ closer to the f64
   oracle than the FAST_16BF tensor-core GEMMs they replace (verified vs Lean via
   verify-mingru-step-gpu); the sampler replicates k_sample_seg's f64 math + splitmix64 draw verbatim
   (same rng stream by global row). NOT bit-identical to the cuBLAS path (dot order differs ⇒ near-tie
   actions can flip) — tolerance-class change, run-to-run deterministic. Gate: PUFFER_MG_FUSED=0 to
   fall back; auto-fallback when H>1024 or A>H (thread-per-output assumptions). */
/* f32 sampler tier for the fused rollout kernels (DEFAULT ON; PUFFER_MG_SAMPF32=0 for the f64 tier).
   The f64 sampler block was named by the occupancy analysis (native-rollout-design.md §13) as one of
   the two binding scalar costs of the fused kernel — and it ran in DOUBLE precision, whose ALU
   throughput on this consumer GPU (GB202) is 1/64 of f32: each row paid ~2A f64 exp + 1 f64 log (long
   f64-FMA software sequences) per step. This tier keeps the IDENTICAL splitmix64 word and f64 uniform
   draw `u` (same rng stream, full 53-bit draw precision — the compare promotes the f32 cumsum to f64,
   costing nothing) but does the softmax max/exp/sum/cumsum in f32. Default-ON rationale: the logits it
   consumes are ALREADY f32 (the whole forward is f32/tf32/bf16), so f64 softmax over them added
   precision only to the cumsum arithmetic, not the information content — and PufferLib itself samples
   from bf16 logits, strictly less precise than this. Measured: squared +12%, pong +26%, breakout +5.5%
   (ratios vs PufferLib native 0.59→0.69 / 0.77→0.97 / 0.80→0.86). Verified: run-to-run deterministic;
   3-seed learning-health on squared tracks the f64 tier within seed variance; pong solves (saturates
   normalized 1.0). Near-tie actions can flip vs the f64 tier — a sampled-action precision-tier change,
   the sensitive kind §13.4 flagged, which is why it got the full multi-seed treatment before landing.
   Published once per rollout call (no CUDA graphs in this path — per-step rng is a kernel arg — so a
   __constant__ flip is safe). */
__constant__ int c_mgSampF32;
/* u8 obs tile loader for ANY D (the per-row uchar4 fast path needs D%4==0 so every row starts
   4-aligned; envs like squared have D=121 and previously fell all the way back to the F32 WIRE —
   4x the zero-copy PCIe traffic, measured as ~85% of the remaining squared gap vs PufferLib, whose
   own trainer ships u8). This loads the block's obs tile [row0·D, row0·D+nr·D) as ALIGNED uchar4
   transactions regardless of row alignment: scalar head to 4-align the pointer, packed body, scalar
   tail — at most 3+3 scalar bytes per tile. Values are exact widenings ((float)u8), so switching the
   wire from f32 to u8 is bit-identical end to end. */
__device__ __forceinline__ void d_mg_obs_u8_tile(const unsigned char* ob, float* obsSh, int Ds, float* traj,
    long trajRow0, long row0, long sStep, long T_traj, int nr, int D, int tid, int bdim){
  long tb0=row0*(long)D, tbN=(long)nr*D;   /* Ds = obsSh row stride (the WMMA variant pads it past D) */
  uintptr_t a0=(uintptr_t)(ob+tb0);
  int head=(int)((4-(a0&3))&3); if((long)head>tbN) head=(int)tbN;
  long body=(tbN-head)&~3L;
  for(long i=tid;i<head;i+=bdim){ int rr=(int)(i/D), k=(int)(i-(long)rr*D);
    float v=(float)ob[tb0+i]; obsSh[(long)rr*Ds+k]=v; if(traj) traj[((trajRow0+row0+rr)*T_traj+sStep)*D+k]=v; }
  const uchar4* ob4=(const uchar4*)(ob+tb0+head);
  for(long i4=tid;i4<(body>>2);i4+=bdim){ uchar4 w=ob4[i4]; long o=head+(i4<<2);
    unsigned char bs[4]={w.x,w.y,w.z,w.w};
    #pragma unroll
    for(int j2=0;j2<4;j2++){ long li=o+j2; int rr=(int)(li/D), k=(int)(li-(long)rr*D);
      float v=(float)bs[j2]; obsSh[(long)rr*Ds+k]=v; if(traj) traj[((trajRow0+row0+rr)*T_traj+sStep)*D+k]=v; } }
  for(long i=head+body+tid;i<tbN;i+=bdim){ int rr=(int)(i/D), k=(int)(i-(long)rr*D);
    float v=(float)ob[tb0+i]; obsSh[(long)rr*Ds+k]=v; if(traj) traj[((trajRow0+row0+rr)*T_traj+sStep)*D+k]=v; }
}
__device__ __forceinline__ void d_mg_sample_row(const float* lg, int A, long n, long rowBase,
    unsigned long long rng, double* sampOut, long sampCount,
    double* cAct, double* cLogp, double* cVal0, double* cValue, long trajRow0, long T_traj, long sStep){
  unsigned long long word=d_sm64(rng+((unsigned long long)(rowBase+n)+1ULL)*0x9E3779B97F4A7C15ULL);
  double u=(double)(word>>11)/9007199254740992.0;
  double lp, v=(double)lg[A]; int a=A-1;
  if(c_mgSampF32){
    float m=lg[0]; for(int k=1;k<A;k++) if(lg[k]>m) m=lg[k];
    float z=0.0f; for(int k=0;k<A;k++) z+=expf(lg[k]-m);
    float accf=0.0f;
    for(int k=0;k<A;k++){ accf+=expf(lg[k]-m)/z; if(u<(double)accf){ a=k; break; } }
    lp=(double)logf(expf(lg[a]-m)/z);
  } else {
    double m=(double)lg[0]; for(int k=1;k<A;k++){ double vv=(double)lg[k]; if(vv>m)m=vv; }
    double z=0.0; for(int k=0;k<A;k++) z+=exp((double)lg[k]-m);
    double acc=0.0;
    for(int k=0;k<A;k++){ acc+=exp((double)lg[k]-m)/z; if(u<acc){ a=k; break; } }
    lp=log(exp((double)lg[a]-m)/z);
  }
  sampOut[n]=(double)a; sampOut[sampCount+n]=lp; sampOut[2*sampCount+n]=v;
  if(cAct){ long r=(trajRow0+n)*T_traj+sStep;
    cAct[r]=(double)a; cLogp[r]=lp; cVal0[r]=v; cValue[r]=v; }
}
/* MULTI-DISCRETE row sampler — the in-kernel twin of k_sample_md, op-for-op: per-head max-subtracted
   f64 softmax over the head's slice of the W=Σheadsizes logits, one draw per head from
   d_sm64(rng + (globalRow+1)·G + (head+1)·G2), joint logp = Σ_h log p_h(a_h), value = lg[W].
   Matching k_sample_md EXACTLY is load-bearing: it produced the `oldLogp` that k_mg_ppo_b_md's ratio
   divides by, and the same (row,head) rng stream means the fused arm draws what the non-fused arm
   would have drawn for the same logits. Deliberately NOT following the c_mgSampF32 f32 tier (that
   tier exists only for the single-discrete d_mg_sample_row, whose f64 anchor is k_sample_seg).
   OUTPUT LAYOUT (`sampOut`, one contiguous per-buffer block of sampCount·(K+2) doubles):
     [act(sampCount×K, ROW-major: n·K+h) | jointLogp(sampCount) | value(sampCount)]
   — row-major actions so one D2H of 8·nb·K bytes lands exactly as step_range_*'s act[e·K+h] wants
   them, no host repack. The resident columns take the K-wide act column (row·K+h, the layout
   mg_prep/k_mg_gather_scal/k_mg_ppo_b_md expect) and the scalar logp/val columns. */
__device__ __forceinline__ void d_mg_sample_row_md(const float* lg, int W, int K, const int* headSizes,
    long n, long rowBase, unsigned long long rng, double* sampOut, long sampCount,
    double* cAct, double* cLogp, double* cVal0, double* cValue, long trajRow0, long T_traj, long sStep){
  const unsigned long long G=0x9E3779B97F4A7C15ULL, G2=0xD1B54A32D192ED03ULL;
  long row=rowBase+n;                      /* GLOBAL row — same seed as the whole-batch k_sample_md */
  double v=(double)lg[W], jointLogp=0.0; int off=0;
  long r=(trajRow0+n)*T_traj+sStep;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh];
    if(sz<=0) continue;   /* a zero-width head would leave a=sz-1=-1: an OOB lg[off-1] read AND a -1
       stamped into the action column, which k_mg_ppo_b_md then uses to index a logits row (OOB read
       and write). k_sample_md carries the same latent flaw, but there the actions crossed back
       through a host column first; here they go straight into the resident device column. */
    double m=(double)lg[off]; for(int k=1;k<sz;k++){ double x=(double)lg[off+k]; if(x>m) m=x; }
    double z=0.0; for(int k=0;k<sz;k++) z+=exp((double)lg[off+k]-m);
    unsigned long long word=d_sm64(rng + ((unsigned long long)row+1ULL)*G + ((unsigned long long)hh+1ULL)*G2);
    double u=(double)(word>>11)/9007199254740992.0;
    double acc=0.0; int a=sz-1;
    for(int k=0;k<sz;k++){ acc+=exp((double)lg[off+k]-m)/z; if(u<acc){ a=k; break; } }
    sampOut[n*(long)K+hh]=(double)a;
    if(cAct) cAct[r*(long)K+hh]=(double)a;
    jointLogp += log(exp((double)lg[off+a]-m)/z); off+=sz; }
  sampOut[sampCount*(long)K+n]=jointLogp; sampOut[sampCount*(long)K+sampCount+n]=v;
  if(cAct){ cLogp[r]=jointLogp; cVal0[r]=v; cValue[r]=v; }
}
/* Tiling: FR rows per block (a v1 with one row/block re-read ALL ~127KB of weights per block — 130MB
   of L2 traffic, 145µs/launch, SLOWER than cuBLAS; weight reuse across rows is what makes GEMMs fast).
   Each block stages the current stage's weight matrix in SHARED (padded stride +1 ⇒ bank-conflict-free),
   loads it ONCE, and pushes FR rows through it — weight traffic ÷FR. blockDim=4·H: threads (r0=tid/H,
   j=tid%H) sweep rows rr=r0,r0+4,… so a warp shares rr (hSh reads broadcast) and spans j (wbuf reads
   conflict-free via the pad). Decoder+value handled as one (A+1)×H matrix. */
__global__ void k_mg_fused_step(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A,
    const unsigned long long* rngDev, unsigned long long tOff){
  /* graph-friendly rng: when rngDev is set, the effective per-step seed is *rngDev + tOff (the
     per-update rolloutRng lives in the device scalar, the per-step offset t·N·G is BAKED into the
     capturing graph) — the same u64 the eager launch passes as `rng`, so replay is bit-identical. */
  if(rngDev) rng=*rngDev+tOff;
  extern __shared__ float sh[];                 /* [wbuf(max stage, padded) | obs(FR·D) | h(FR·H) | lg(FR·(A+1)) | term(FR)] */
  int wbufSz=(3*H)*(H+1); { int e=H*(D+1); if(e>wbufSz) wbufSz=e; int d=(A+1)*(H+1); if(d>wbufSz) wbufSz=d; }
  float* wbuf=sh; float* obsSh=sh+wbufSz; float* hSh=obsSh+FR*D; float* lgSh=hSh+FR*H;
  float* termSh=lgSh+FR*(A+1);               /* per-row terminal gate, staged ONCE (terms may be pinned-host) */
  int tid=threadIdx.x, r0=tid/H, j=tid%H, rpp=blockDim.x/H;   /* rows per pass (launch: 8H threads when H≤128) */
  long row0=(long)blockIdx.x*FR;             /* first (local) row of this block's tile */
  int nr=(int)((nRows-row0)<FR?(nRows-row0):FR); if(nr<=0) return;
  if(tid<nr) termSh[tid]=(terms && terms[row0+tid]!=0.0)?1.0f:0.0f;
  size_t wEncSz=(size_t)H*D, layerSz=(size_t)3*H*H;
  const float* wEnc=dP; const float* bEnc=dP+wEncSz; const float* layers=dP+wEncSz+H;
  const float* wDec=layers+(size_t)L*layerSz; const float* bDec=wDec+(size_t)A*H;
  const float* wVal=bDec+A; const float* bVal=wVal+H;
  /* obs tile → shared (coalesced), and the trajectory scatter (replaces k_scatter_mg_obs). obsKind picks
     the TRANSPORT element: 0 f32; 1 u8 (byte-native envs — u8→f32 widen is EXACT ⇒ bit-identical);
     2 bf16 (RNE-rounded — tolerance-class, PufferLib's own obs precision). The traj stays f32 holding
     exactly the values this forward consumed, so the BPTT trains on what the policy saw. */
  /* sub-word transports MUST use packed loads: per-thread 1-2 byte zero-copy PCIe reads measured a ~4x
     kernel slowdown (each warp transaction half/quarter-filled). u8 loads uchar4 (D%4==0), bf16 loads
     uint pairs (D%2==0); odd-D falls back to scalar. */
  if(obsKind==1 && (D&3)==0){
    const uchar4* ob=(const uchar4*)obsF; int D4=D>>2;
    for(int i4=tid;i4<nr*D4;i4+=blockDim.x){ int rr=i4/D4, k4=i4%D4;
      uchar4 w=ob[(row0+rr)*(long)D4+k4]; int k=4*k4;
      float v0=(float)w.x,v1=(float)w.y,v2=(float)w.z,v3=(float)w.w;
      float* os=obsSh+rr*D+k; os[0]=v0; os[1]=v1; os[2]=v2; os[3]=v3;
      if(traj){ float* tj=traj+((trajRow0+row0+rr)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; tj[2]=v2; tj[3]=v3; } }
  } else if(obsKind==1){                       /* u8, ANY D: tile-linear aligned loads (see d_mg_obs_u8_tile) */
    d_mg_obs_u8_tile((const unsigned char*)obsF, obsSh, D, traj, trajRow0, row0, sStep, T_traj, nr, D, tid, blockDim.x);
  } else if(obsKind==2 && (D&1)==0){
    const unsigned int* ob=(const unsigned int*)obsF; int D2=D>>1;
    for(int i2=tid;i2<nr*D2;i2+=blockDim.x){ int rr=i2/D2, k2=i2%D2;
      unsigned int w=ob[(row0+rr)*(long)D2+k2]; int k=2*k2;
      __nv_bfloat16 lo=*(__nv_bfloat16*)&w, hi=*(((__nv_bfloat16*)&w)+1);
      float v0=__bfloat162float(lo), v1=__bfloat162float(hi);
      float* os=obsSh+rr*D+k; os[0]=v0; os[1]=v1;
      if(traj){ float* tj=traj+((trajRow0+row0+rr)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; } }
  } else
  for(int i=tid;i<nr*D;i+=blockDim.x){ int rr=i/D, k=i%D; long oi=(row0+rr)*(long)D+k;
    float v = (obsKind==1)? (float)((const unsigned char*)obsF)[oi]
            : (obsKind==2)? __bfloat162float(((const __nv_bfloat16*)obsF)[oi])
            : ((const float*)obsF)[oi];
    obsSh[rr*D+k]=v; if(traj) traj[((trajRow0+row0+rr)*T_traj+sStep)*D+k]=v; }
  /* encoder weights → shared, padded stride D+1 */
  for(int i=tid;i<H*D;i+=blockDim.x) wbuf[(i/D)*(D+1)+(i%D)]=wEnc[i];
  __syncthreads();
  for(int rr=r0;rr<nr;rr+=rpp){ const float* w=wbuf+(size_t)j*(D+1); float acc=bEnc[j];
    for(int k=0;k<D;k++) acc+=w[k]*obsSh[rr*D+k]; hSh[rr*H+j]=acc; }
  __syncthreads();
  /* L gate layers: stage Wl in shared once, thread (rr,j) dots rows j, H+j, 2H+j (its own gate).
     The rr sweep runs a UNIFORM ceil(MG_FR/rpp) iterations (body guarded by act) — __syncthreads()
     inside a thread-divergent loop would deadlock on partial tiles. rpp=8 (H≤128) ⇒ ONE pass. */
  for(int l=0;l<L;l++){
    if((H&3)==0){                                /* float4 staging: 4 consecutive elements share a row (H%4==0) */
      const float4* src=(const float4*)(layers+(size_t)l*layerSz);
      for(int i4=tid;i4<3*H*H/4;i4+=blockDim.x){ float4 v=src[i4]; int i=4*i4;
        float* w=wbuf+(i/H)*(H+1)+(i%H); w[0]=v.x; w[1]=v.y; w[2]=v.z; w[3]=v.w; }
    } else
      for(int i=tid;i<3*H*H;i+=blockDim.x) wbuf[(i/H)*(H+1)+(i%H)]=layers[(size_t)l*layerSz+i];
    __syncthreads();
    for(int it=0;it<(FR+rpp-1)/rpp;it++){
      int rr=r0+rpp*it; int act=(rr<nr);
      float hnew=0.0f,o=0.0f; long sidx=0;
      if(act){
        const float *w0=wbuf+(size_t)j*(H+1), *w1=wbuf+(size_t)(H+j)*(H+1), *w2=wbuf+(size_t)(2*H+j)*(H+1);
        float hid=0.0f,gate=0.0f,proj=0.0f;
        for(int k=0;k<H;k++){ float hv=hSh[rr*H+k]; hid+=w0[k]*hv; gate+=w1[k]*hv; proj+=w2[k]*hv; }
        float z=d_sigf(gate), gg=d_gactf(hid);
        sidx=(row0+rr)*LHs+(long)l*H+j;
        /* folded terminal reset: gate the state read on the PREVIOUS step's terms (replaces the
           per-step k_mg_reset launch; identical values — 0 gated == 0 stored) */
        float prev=(termSh[rr]!=0.0f)? 0.0f : st[sidx];
        o=(1.0f-z)*prev+z*gg; float hg=d_sigf(proj);
        hnew=hg*o+(1.0f-hg)*hSh[rr*H+j];
      }
      __syncthreads();                          /* all dots read hSh before the in-place update */
      if(act){ hSh[rr*H+j]=hnew; st[sidx]=o; }
      __syncthreads();
    }
  }
  /* heads as ONE (A+1)×H matrix (row A = value), then sample (k_sample_seg's f64 math, verbatim) */
  for(int i=tid;i<(A+1)*H;i+=blockDim.x){ int r=i/H,c=i%H; wbuf[r*(H+1)+c]= (r<A)? wDec[i] : wVal[c]; }
  __syncthreads();
  for(int rr=r0;rr<nr;rr+=rpp) if(j<A+1){ const float* w=wbuf+(size_t)j*(H+1); float acc=(j<A)?bDec[j]:bVal[0];
    for(int k=0;k<H;k++) acc+=w[k]*hSh[rr*H+k]; lgSh[rr*(A+1)+j]=acc; }
  __syncthreads();
  if(outAsm) for(int i=tid;i<nr*(A+1);i+=blockDim.x) outAsm[(row0)*(A+1)+i]=(double)lgSh[i];
  if(sampOut && j==0) for(int rr=r0;rr<nr;rr+=rpp)
    d_mg_sample_row(lgSh+rr*(A+1), A, row0+rr, rowBase, rng, sampOut, sampCount,
                    cAct, cLogp, cVal0, cValue, trajRow0, T_traj, sStep);
}

/* ===== cp.async PIPELINED fused step ============================================================
   Same math, same per-element accumulation ORDER (k ascending across column-halves ⇒ BIT-IDENTICAL
   to k_mg_fused_step) — but each stage's weight matrix streams into TWO half-buffers via cp.async,
   overlapping the NEXT half/stage load with the CURRENT dots (the plain kernel serializes
   load→sync→dots per stage). Requires the single-row-pass regime (blockDim==FR·H, i.e. FR==8, H≤128).
   Choreography per stage: wait(1)/sync → dots(half0) → wait(0)/sync → issue next.h0 → dots(half1)+
   gate→regs → sync → write hSh/state → sync → issue next.h1 — at every wait at most 2 groups are in
   flight, and a buffer is only overwritten after the sync that proves every thread is done reading it. */
__device__ static inline void mg_issue_half(int stage, int half, float* dst, int tid, int bdim,
    const float* wEnc, const float* layers, const float* wDec, const float* wVal,
    int D, int H, int L, int A, size_t layerSz){
  int R,C; const float* base=NULL; int isHeads=0;
  if(stage==0){ R=H; C=D; base=wEnc; }
  else if(stage<=L){ R=3*H; C=H; base=layers+(size_t)(stage-1)*layerSz; }
  else { R=A+1; C=H; isHeads=1; }
  int c0=(half==0)?0:C/2; int W=(half==0)?(C/2):(C-C/2);
  /* float4-clean stages (layers/heads: C=H, W=H/2, all %4) use ALIGNED stride W + 16B chunks — the
     bank conflicts this would cause in the dots are eliminated by the ROTATED dot order (each thread
     starts its k-loop at its own offset), not by odd-stride padding. Non-clean stages (encoder, odd D)
     keep 4B chunks + odd stride + straight dot order. */
  if(!isHeads && (C&3)==0 && (c0&3)==0 && (W&3)==0){   /* heads: wVal sits at +A·H+A floats — not 16B-aligned */
    int W4=W>>2;
    for(int i=tid;i<R*W4;i+=bdim){ int r=i/W4, c4=i%W4;
      const float* srcp = isHeads? ((r<A)? wDec+(size_t)r*C+c0+4*c4 : wVal+c0+4*c4) : (base+(size_t)r*C+c0+4*c4);
      __pipeline_memcpy_async(&dst[(size_t)r*W+4*c4], srcp, 16);
    }
  } else {
    int S=(W&1)?W:W+1;
    for(int i=tid;i<R*W;i+=bdim){ int r=i/W, c=i%W;
      const float* srcp = isHeads? ((r<A)? wDec+(size_t)r*C+c0+c : wVal+c0+c) : (base+(size_t)r*C+c0+c);
      __pipeline_memcpy_async(&dst[(size_t)r*S+c], srcp, 4);
    }
  }
}
__global__ void k_mg_fused_step_p(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A,
    const unsigned long long* rngDev, unsigned long long tOff){
  /* graph-friendly rng: when rngDev is set, the effective per-step seed is *rngDev + tOff (the
     per-update rolloutRng lives in the device scalar, the per-step offset t·N·G is BAKED into the
     capturing graph) — the same u64 the eager launch passes as `rng`, so replay is bit-identical. */
  if(rngDev) rng=*rngDev+tOff;
  extern __shared__ float sh[];
  /* [HB0(maxHalf) | HB1(maxHalf) | obs(FR·D) | h(FR·H) | lg(FR·(A+1)) | term(FR)] */
  int maxHalf; { int e0=(D/2&1)?(D/2):(D/2+1), e1=((D-D/2)&1)?(D-D/2):(D-D/2+1);
    maxHalf=H*(e0>e1?e0:e1); int lh=3*H*(H/2); if(lh>maxHalf) maxHalf=lh;
    int hsr=((H/2)&1)?(H/2):(H/2+1);
    int hh=(A+1)*hsr; if(hh>maxHalf) maxHalf=hh; maxHalf=(maxHalf+3)&~3; }
  float* HB0=sh; float* HB1=sh+maxHalf; float* obsSh=HB1+maxHalf;
  float* hSh=obsSh+FR*D; float* lgSh=hSh+FR*H; float* termSh=lgSh+FR*(A+1);
  int tid=threadIdx.x, r0=tid/H, j=tid%H, bdim=blockDim.x;
  long row0=(long)blockIdx.x*FR;
  int nr=(int)((nRows-row0)<FR?(nRows-row0):FR); if(nr<=0) return;
  int act=(r0<nr); int rr=r0;
  if(tid<nr) termSh[tid]=(terms && terms[row0+tid]!=0.0)?1.0f:0.0f;
  size_t wEncSz=(size_t)H*D, layerSz=(size_t)3*H*H;
  const float* wEnc=dP; const float* bEnc=dP+wEncSz; const float* layers=dP+wEncSz+H;
  const float* wDec=layers+(size_t)L*layerSz; const float* bDec=wDec+(size_t)A*H;
  const float* wVal=bDec+A; const float* bVal=wVal+H;
  /* prologue: stream the encoder halves while the obs tile stages normally */
  mg_issue_half(0,0,HB0,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
  mg_issue_half(0,1,HB1,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
  if(obsKind==1 && (D&3)==0){
    const uchar4* ob=(const uchar4*)obsF; int D4=D>>2;
    for(int i4=tid;i4<nr*D4;i4+=bdim){ int r=i4/D4, k4=i4%D4;
      uchar4 w=ob[(row0+r)*(long)D4+k4]; int k=4*k4;
      float v0=(float)w.x,v1=(float)w.y,v2=(float)w.z,v3=(float)w.w;
      float* os=obsSh+r*D+k; os[0]=v0; os[1]=v1; os[2]=v2; os[3]=v3;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; tj[2]=v2; tj[3]=v3; } }
  } else if(obsKind==1){                       /* u8, ANY D: tile-linear aligned loads (see d_mg_obs_u8_tile) */
    d_mg_obs_u8_tile((const unsigned char*)obsF, obsSh, D, traj, trajRow0, row0, sStep, T_traj, nr, D, tid, bdim);
  } else if(obsKind==2 && (D&1)==0){
    const unsigned int* ob=(const unsigned int*)obsF; int D2=D>>1;
    for(int i2=tid;i2<nr*D2;i2+=bdim){ int r=i2/D2, k2=i2%D2;
      unsigned int w=ob[(row0+r)*(long)D2+k2]; int k=2*k2;
      __nv_bfloat16 lo=*(__nv_bfloat16*)&w, hi=*(((__nv_bfloat16*)&w)+1);
      float v0=__bfloat162float(lo), v1=__bfloat162float(hi);
      float* os=obsSh+r*D+k; os[0]=v0; os[1]=v1;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; } }
  } else
  for(int i=tid;i<nr*D;i+=bdim){ int r=i/D, k=i%D; long oi=(row0+r)*(long)D+k;
    float v = (obsKind==1)? (float)((const unsigned char*)obsF)[oi]
            : (obsKind==2)? __bfloat162float(((const __nv_bfloat16*)obsF)[oi])
            : ((const float*)obsF)[oi];
    obsSh[r*D+k]=v; if(traj) traj[((trajRow0+row0+r)*T_traj+sStep)*D+k]=v; }
  /* encoder */
  { int C0=D/2, W1=D-C0, S0=(C0&1)?C0:C0+1, S1=(W1&1)?W1:W1+1;
    float acc=0.0f;
    __pipeline_wait_prior(1); __syncthreads();
    if(act){ acc=bEnc[j]; const float* w=HB0+(size_t)j*S0; const float* ob=obsSh+rr*D;
      for(int k=0;k<C0;k++) acc+=w[k]*ob[k]; }
    __pipeline_wait_prior(0); __syncthreads();
    mg_issue_half(1,0,HB0,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
    if(act){ const float* w=HB1+(size_t)j*S1; const float* ob=obsSh+rr*D+C0;
      for(int k=0;k<W1;k++) acc+=w[k]*ob[k]; hSh[rr*H+j]=acc; }
    __syncthreads();
    mg_issue_half(1,1,HB1,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
  }
  /* gate layers (stage l+1 streams while layer l computes). ALIGNED stride W + ROTATED dot order:
     thread j starts at k2=j%W and wraps — reads hit 32 distinct banks across a warp (conflict-free
     without padding), enabling the 16B copies. Reassociates the layer sums (tolerance-class; the
     oracle verify covers it) — deterministic, same terms, rotated order. */
  for(int l=0;l<L;l++){
    int W=H/2;
    float hid=0.0f,gate=0.0f,proj=0.0f,hnew=0.0f,o=0.0f; long sidx=0;
    __pipeline_wait_prior(1); __syncthreads();
    if(act){ const float *w0=HB0+(size_t)j*W, *w1=HB0+(size_t)(H+j)*W, *w2=HB0+(size_t)(2*H+j)*W;
      const float* hv=hSh+rr*H;
      int k2=j%W;
      for(int k=0;k<W;k++){ float h=hv[k2]; hid+=w0[k2]*h; gate+=w1[k2]*h; proj+=w2[k2]*h;
        k2=(k2+1==W)?0:k2+1; } }
    __pipeline_wait_prior(0); __syncthreads();
    mg_issue_half(l+2,0,HB0,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
    if(act){ const float *w0=HB1+(size_t)j*W, *w1=HB1+(size_t)(H+j)*W, *w2=HB1+(size_t)(2*H+j)*W;
      const float* hv=hSh+rr*H;
      int k2=j%W;
      for(int k=0;k<W;k++){ float h=hv[W+k2]; hid+=w0[k2]*h; gate+=w1[k2]*h; proj+=w2[k2]*h;
        k2=(k2+1==W)?0:k2+1; }
      float z=d_sigf(gate), gg=d_gactf(hid);
      sidx=(row0+rr)*LHs+(long)l*H+j;
      float prev=(termSh[rr]!=0.0f)? 0.0f : st[sidx];
      o=(1.0f-z)*prev+z*gg; float hg=d_sigf(proj);
      hnew=hg*o+(1.0f-hg)*hSh[rr*H+j]; }
    __syncthreads();
    if(act){ hSh[rr*H+j]=hnew; st[sidx]=o; }
    __syncthreads();
    mg_issue_half(l+2,1,HB1,tid,bdim,wEnc,layers,wDec,wVal,D,H,L,A,layerSz); __pipeline_commit();
  }
  /* heads ((A+1)×H, row A = value) — 4B/odd-stride staging (misaligned wVal), straight dot order */
  { int W=H/2, S=(W&1)?W:W+1;
    float acc=0.0f;
    __pipeline_wait_prior(1); __syncthreads();
    if(act && j<A+1){ acc=(j<A)?bDec[j]:bVal[0]; const float* w=HB0+(size_t)j*S; const float* hv=hSh+rr*H;
      for(int k=0;k<W;k++) acc+=w[k]*hv[k]; }
    __pipeline_wait_prior(0); __syncthreads();
    if(act && j<A+1){ const float* w=HB1+(size_t)j*S; const float* hv=hSh+rr*H;
      for(int k=0;k<H-W;k++) acc+=w[k]*hv[W+k]; lgSh[rr*(A+1)+j]=acc; }
    __syncthreads();
  }
  if(outAsm) for(int i=tid;i<nr*(A+1);i+=bdim) outAsm[(row0)*(A+1)+i]=(double)lgSh[i];
  if(sampOut && j==0 && act)
    d_mg_sample_row(lgSh+rr*(A+1), A, row0+rr, rowBase, rng, sampOut, sampCount,
                    cAct, cLogp, cVal0, cValue, trajRow0, T_traj, sStep);
}
static size_t mg_pipe_shbytes(int D,int H,int L,int A,int FR){
  (void)L;
  int e0=(D/2&1)?(D/2):(D/2+1), e1=((D-D/2)&1)?(D-D/2):(D-D/2+1);
  long maxHalf=(long)H*(e0>e1?e0:e1); long lh=(long)3*H*(H/2); if(lh>maxHalf) maxHalf=lh;
  long hsr=((H/2)&1)?(H/2):(H/2+1);
  long hh=(long)(A+1)*hsr; if(hh>maxHalf) maxHalf=hh;
  maxHalf=(maxHalf+3)&~3L;                       /* 16B-align the HB1 base */
  long tot=4L*(2*maxHalf + (long)FR*D + (long)FR*H + (long)FR*(A+1) + FR);
  if(H<8 || 8*H>1024 || A+1>H || (H&7) || tot>90*1024) return 0;   /* blockDim=8H, FR=8, W=H/2 %4 (16B rows) */
  return (size_t)tot;
}
/* default OFF: measured a WASH at 4B chunks (36.7us vs 34.4us kernel; SPS -0.5%) — the per-element
   cp.async issue overhead + 2 extra syncs/stage cancel the latency hiding. 16B chunks conflict with the
   odd-stride bank layout; an XOR-swizzle would reconcile them (future work). PUFFER_MG_PIPE=1 opts in. */
static int mg_pipe(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_PIPE"); f=(e!=NULL&&e[0]=='1'); } return f; }
static int mg_pipe_optin(size_t shb){ static int state=0;
  if(shb<=48*1024) return 1;
  if(state==0) state=(cudaFuncSetAttribute(k_mg_fused_step_p, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(96*1024))==cudaSuccess)?1:-1;
  return state==1; }

/* ===== tf32 TENSOR-CORE fused step ==============================================================
   The ledger shows the fused forward is bound by the shared/LSU pipe (scalar dots do ~4 shared loads
   per 3 FMAs; cp.async overlap bought nothing because staging and dots contend for the same pipe).
   WMMA's lever here is ldmatrix: warp-cooperative fragment loads cut the dot-side shared traffic ~4x.
   The two LAYER stages run as m16n16k8 tf32 MMA tiles with f32 accumulation (inputs rounded to tf32:
   ~1e-3 forward tolerance — TIGHTER than the FAST_16BF cuBLAS default this trainer shipped pre-fusion);
   encoder, heads, gate, sampler stay scalar-exact. FR=16 (fragment-native rows), blockDim=512
   (16 warps; 3H/16 n-tiles per layer). Gated PUFFER_MG_WMMA=1; needs H%16==0. */
using namespace nvcuda;
/* big-H encoder WMMA support: the encoder weight matrix lives at ld=D (arbitrary env obs dim), generally
   fragment-illegal; a padded copy (ld=Dp=(D+7)&~7, pad cols zeroed) is prepared ONCE per rollout call and
   published via __constant__. NULL ⇒ the kernel keeps the scalar encoder — safe by default; only the
   big-H (encStaged==0) branch consumes it. */
__constant__ float* c_wEncPad;
__constant__ int c_encDp;
__constant__ int c_encH;
/* bf16 tier, ENCODER stage: a bf16 pack of the PADDED encoder weights (stride Dp). Only consumed when
   the layer tier is also live (c_wlBf) AND Dp%16==0 AND Dp<=H (the obs bf16 tile reuses the layer
   tier's hShB shared reserve — no layout change). u8 obs are exactly representable in bf16. */
__constant__ __nv_bfloat16* c_wEncBf;
/* FMA gate for the fused kernels' scalar sections: this TU compiles --fmad=false so the f32 tier
   stays bit-exact vs the Lean oracle; when the (default) bf16 tolerance tier is active, explicit
   __fmaf_rn variants reclaim the contraction the flag forfeits (~2x on the scalar dot chains). */
__constant__ int c_mgFma;
/* bf16 forward tier (PUFFER_MG_WPREC=bf16, default OFF): the WMMA layer stages run m16n16k16 bf16 —
   PufferLib's own forward precision — halving the mma/fragment count vs tf32's k=8. A bf16-packed copy
   of the layer weights is prepared per call and published here; NULL keeps the tf32 tier. */
__constant__ __nv_bfloat16* c_wlBf;
__constant__ int c_wlBfH;
__constant__ int c_wlBfL;
__global__ void k_pack_wlbf(__nv_bfloat16* dst, const float* layers, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16(layers[i]); }
static int mg_wprec_bf(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_WPREC");
  f=(e && (e[0]=='f'||e[0]=='F'||e[0]=='0'))?0:1; } return f; }   /* DEFAULT ON (PufferLib's own forward
  precision; layer+encoder verified via multi-seed learning health, +2-3% SPS); PUFFER_MG_WPREC=f32 opts out */   /* the H the pad was built for — an equal-Dp different-H consumer must NOT
                              pass the guard (stale weights / OOB reads past the 4·H·Dp allocation) */
__global__ void k_pad_wenc(float* dst, const float* wEnc, int H, int D, int Dp){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)H*Dp) return;
  int r=(int)(i/Dp), k=(int)(i%Dp); dst[i]=(k<D)? wEnc[(long)r*D+k] : 0.0f; }
static void* bg(int i, size_t bytes);
static inline int ceildiv(long n, int b);
/* prepare + publish the padded encoder weights for the big-H WMMA encoder. Called per rollout/step call
   (re-pads from the CALLER's dP — content can never be stale); publishes NULL on allocation failure so
   the kernel falls back to the scalar encoder instead of reading freed VRAM (review-mandated). */
static void mg_pub_wlbf(const float* dP, int H, int D, int L){
  if(!mg_wprec_bf() || (H%16)) return;
  size_t layerSz=(size_t)3*H*H; const float* layers=dP+(size_t)H*D+H;
  __nv_bfloat16* dst=(__nv_bfloat16*)bg(89,2*(size_t)L*layerSz);
  static __nv_bfloat16* pub=NULL; static int pubH=0,pubL=0;
  if(!dst){ if(pub){ __nv_bfloat16* nul=NULL; int z=0;
      cudaMemcpyToSymbol(c_wlBf,&nul,sizeof(void*)); cudaMemcpyToSymbol(c_wlBfH,&z,sizeof(int));
      cudaMemcpyToSymbol(c_wlBfL,&z,sizeof(int)); pub=NULL; pubH=0; pubL=0; } return; }
  k_pack_wlbf<<<ceildiv((long)L*layerSz,256),256>>>(dst,layers,(long)L*layerSz);
  if(pub!=dst || pubH!=H || pubL!=L){
    cudaMemcpyToSymbol(c_wlBf,&dst,sizeof(void*));
    cudaMemcpyToSymbol(c_wlBfH,&H,sizeof(int)); cudaMemcpyToSymbol(c_wlBfL,&L,sizeof(int));
    pub=dst; pubH=H; pubL=L; }
}
static void mg_pub_wencpad(const float* dP, int H, int D, int L, int A){
  (void)L;
  long wbufFull=(long)H*(D+1); { long d2=(long)(A+1)*(H+1); if(d2>wbufFull) wbufFull=d2; }
  long fullTot=4L*(wbufFull + 16L*3*H + 16L*D + 16L*H + 16L*(A+1) + 16);
  if(fullTot<=90*1024 || mg_wmma_shbytes(D,H,L,A)==0) return;   /* not a big-H WMMA config */
  int Dp=(int)(((long)D+7)&~7L);
  float* dPad=(float*)bg(88,4L*(long)H*Dp);
  static float* pubPad=NULL; static int pubDp=0, pubH=0;
  static __nv_bfloat16* pubBf=NULL;
  if(!dPad){ if(pubPad){ float* nul=NULL; int z=0; cudaMemcpyToSymbol(c_wEncPad,&nul,sizeof(float*));
      cudaMemcpyToSymbol(c_encDp,&z,sizeof(int)); cudaMemcpyToSymbol(c_encH,&z,sizeof(int));
      pubPad=NULL; pubDp=0; pubH=0; }
    if(pubBf){ __nv_bfloat16* nul=NULL; cudaMemcpyToSymbol(c_wEncBf,&nul,sizeof(void*)); pubBf=NULL; }
    return; }
  k_pad_wenc<<<ceildiv((long)H*Dp,256),256>>>(dPad,dP,H,D,Dp);
  if(pubPad!=dPad || pubDp!=Dp || pubH!=H){
    cudaMemcpyToSymbol(c_wEncPad,&dPad,sizeof(float*));
    cudaMemcpyToSymbol(c_encDp,&Dp,sizeof(int)); cudaMemcpyToSymbol(c_encH,&H,sizeof(int));
    pubPad=dPad; pubDp=Dp; pubH=H; }
  /* bf16 tier: pack the padded copy for the k16 encoder. Re-packed from THIS call's dPad every call
     (never stale); shape gates mirror the kernel's. Cleared when ineligible so a shape change can
     never leave a stale pointer live (kernel also re-checks c_encDp/c_encH/alignment). */
  if(mg_wprec_bf() && (Dp%16)==0 && Dp<=H){
    __nv_bfloat16* dBf=(__nv_bfloat16*)bg(87,2L*(long)H*Dp);
    if(dBf){ k_pack_wlbf<<<ceildiv((long)H*Dp,256),256>>>(dBf,dPad,(long)H*Dp);
      if(pubBf!=dBf){ cudaMemcpyToSymbol(c_wEncBf,&dBf,sizeof(void*)); pubBf=dBf; } }
    else if(pubBf){ __nv_bfloat16* nul=NULL; cudaMemcpyToSymbol(c_wEncBf,&nul,sizeof(void*)); pubBf=NULL; }
  } else if(pubBf){ __nv_bfloat16* nul=NULL; cudaMemcpyToSymbol(c_wEncBf,&nul,sizeof(void*)); pubBf=NULL; }
}
__global__ void k_mg_fused_step_w(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A,
    const unsigned long long* rngDev, unsigned long long tOff){
  /* graph-friendly rng: when rngDev is set, the effective per-step seed is *rngDev + tOff (the
     per-update rolloutRng lives in the device scalar, the per-step offset t·N·G is BAKED into the
     capturing graph) — the same u64 the eager launch passes as `rng`, so replay is bit-identical. */
  if(rngDev) rng=*rngDev+tOff;
  extern __shared__ float sh[];
  /* big-H mode: when the FULL layout (encoder weights staged) would blow the 90KB budget, skip the
     encoder staging and read wEnc global-direct in the encoder dot (a pure copy elision — same values,
     same order, bit-identical; rows are L2-hot across the rpp passes). wbuf then only holds the heads.
     The host's mg_wmma_shbytes computes the SAME predicate, so layout and allocation always agree. */
  int wbufSz=H*(D+1); { int d=(A+1)*(H+1); if(d>wbufSz) wbufSz=d; }
  int encStaged=1;
  { long tot=4L*((long)wbufSz + (long)FR*3*H + (long)FR*D + (long)FR*H + (long)FR*(A+1) + FR)
             +2L*FR*H+32;   /* + the bf16-tile RESERVE — matches the host's gate-independent predicate */
    if(tot>90*1024){ encStaged=0; wbufSz=(int)((((A+1)*(H+1))+7)&~7); } }   /* pad: ySh/hSh 256-bit-aligned */
  int Ds = encStaged? D : ((D+7)&~7);   /* big-H: fragment-legal obs stride (pad cols zeroed below) */
  float* wbuf=sh; float* ySh=sh+wbufSz; float* obsSh=ySh+(size_t)FR*3*H;
  float* hSh=obsSh+FR*Ds; float* lgSh=hSh+FR*H; float* termSh=lgSh+FR*(A+1);
  int tid=threadIdx.x, r0=tid/H, j=tid%H, rpp=blockDim.x/H, warp=tid>>5;
  long row0=(long)blockIdx.x*FR;
  int nr=(int)((nRows-row0)<FR?(nRows-row0):FR); if(nr<=0) return;
  if(tid<nr) termSh[tid]=(terms && terms[row0+tid]!=0.0)?1.0f:0.0f;
  for(int i=tid;i<FR*H;i+=blockDim.x) if(i>=nr*H) hSh[i]=0.0f;   /* zero ALL tail rows the fragments read
     (the old single-shot `if(tid...)` covered only blockDim elements — latent for nRows%FR!=0) */
  size_t wEncSz=(size_t)H*D, layerSz=(size_t)3*H*H;
  const float* wEnc=dP; const float* bEnc=dP+wEncSz; const float* layers=dP+wEncSz+H;
  const float* wDec=layers+(size_t)L*layerSz; const float* bDec=wDec+(size_t)A*H;
  const float* wVal=bDec+A; const float* bVal=wVal+H;
  /* obs tile (+ traj scatter) — as the plain kernel */
  if(obsKind==1 && (D&3)==0){
    const uchar4* ob=(const uchar4*)obsF; int D4=D>>2;
    for(int i4=tid;i4<nr*D4;i4+=blockDim.x){ int r=i4/D4, k4=i4%D4;
      uchar4 w=ob[(row0+r)*(long)D4+k4]; int k=4*k4;
      float v0=(float)w.x,v1=(float)w.y,v2=(float)w.z,v3=(float)w.w;
      float* os=obsSh+r*Ds+k; os[0]=v0; os[1]=v1; os[2]=v2; os[3]=v3;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; tj[2]=v2; tj[3]=v3; } }
  } else if(obsKind==1){                       /* u8, ANY D: tile-linear aligned loads (see d_mg_obs_u8_tile) */
    d_mg_obs_u8_tile((const unsigned char*)obsF, obsSh, Ds, traj, trajRow0, row0, sStep, T_traj, nr, D, tid, blockDim.x);
  } else if(obsKind==2 && (D&1)==0){
    const unsigned int* ob=(const unsigned int*)obsF; int D2=D>>1;
    for(int i2=tid;i2<nr*D2;i2+=blockDim.x){ int r=i2/D2, k2=i2%D2;
      unsigned int w=ob[(row0+r)*(long)D2+k2]; int k=2*k2;
      __nv_bfloat16 lo=*(__nv_bfloat16*)&w, hi=*(((__nv_bfloat16*)&w)+1);
      float v0=__bfloat162float(lo), v1=__bfloat162float(hi);
      float* os=obsSh+r*Ds+k; os[0]=v0; os[1]=v1;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; } }
  } else
  for(int i=tid;i<nr*D;i+=blockDim.x){ int r=i/D, k=i%D; long oi=(row0+r)*(long)D+k;
    float v = (obsKind==1)? (float)((const unsigned char*)obsF)[oi]
            : (obsKind==2)? __bfloat162float(((const __nv_bfloat16*)obsF)[oi])
            : ((const float*)obsF)[oi];
    obsSh[r*Ds+k]=v; if(traj) traj[((trajRow0+row0+r)*T_traj+sStep)*D+k]=v; }
  /* encoder — scalar-exact when staged (h≤64 envs: byte-lineage preserved). Big-H: tf32 WMMA over the
     padded obs tile × the padded weight copy (same tier as the layers), scalar global-direct fallback
     when the padded weights weren't prepared (c_wEncPad NULL — e.g. an uninstrumented caller). */
  if(encStaged) for(int i=tid;i<H*D;i+=blockDim.x) wbuf[(i/D)*(D+1)+(i%D)]=wEnc[i];
  if(!encStaged){   /* zero the pad columns + tail rows the encoder fragments read */
    for(int i=tid;i<FR*Ds;i+=blockDim.x){ int r=i/Ds, k=i%Ds; if(k>=D || r>=nr) obsSh[i]=0.0f; } }
  __syncthreads();
  __nv_bfloat16* hShB=(__nv_bfloat16*)(((size_t)(termSh+FR)+31)&~(size_t)31);   /* bf16 tier only
     (32B-aligned — wmma mptr documents a 256-bit requirement; the host's 32B reserve covers the pad) */
  int useBf=(c_wlBf!=NULL && c_wlBfH==H && c_wlBfL>=L);
  if(!encStaged && useBf && c_wEncBf && c_encDp==Ds && c_encH==H && (Ds&15)==0 && Ds<=H){
    /* bf16 m16n16k16 encoder — the layer tier extended to the encoder GEMM. The obs tile converts
       through hShB (fits: Ds<=H gate, and hSh isn't live until the accumulator stores below); u8 obs
       values are EXACT in bf16 (integers <=256), weights get the same tier rounding as the layers. */
    for(int i=tid;i<FR*Ds;i+=blockDim.x) hShB[i]=__float2bfloat16(obsSh[i]);
    __syncthreads();
    { int nT=H/16;
      for(int nt=warp; nt<nT; nt+=(int)(blockDim.x>>5)){
        wmma::fragment<wmma::accumulator,16,16,16,float> cfr; wmma::fill_fragment(cfr,0.0f);
        for(int kt=0;kt<Ds/16;kt++){
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfr;
          wmma::load_matrix_sync(afr, hShB+kt*16, Ds);
          wmma::load_matrix_sync(bfr, c_wEncBf+(size_t)nt*16*Ds+kt*16, Ds);
          wmma::mma_sync(cfr,afr,bfr,cfr);
        }
        wmma::store_matrix_sync(hSh+nt*16, cfr, H, wmma::mem_row_major);
      } }
    __syncthreads();
    for(int i=tid;i<FR*H;i+=blockDim.x) hSh[i]+=bEnc[i%H];
  } else if(!encStaged && c_wEncPad && c_encDp==Ds && c_encH==H){
    const float* wp=c_wEncPad;
    { int nT=H/16;
      for(int nt=warp; nt<nT; nt+=(int)(blockDim.x>>5)){
        wmma::fragment<wmma::accumulator,16,16,8,float> cfr; wmma::fill_fragment(cfr,0.0f);
        for(int kt=0;kt<Ds/8;kt++){
          wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bfr;
          wmma::load_matrix_sync(afr, obsSh+kt*8, Ds);
          wmma::load_matrix_sync(bfr, wp+(size_t)nt*16*Ds+kt*8, Ds);
          for(int t2=0;t2<afr.num_elements;t2++) afr.x[t2]=wmma::__float_to_tf32(afr.x[t2]);
          for(int t2=0;t2<bfr.num_elements;t2++) bfr.x[t2]=wmma::__float_to_tf32(bfr.x[t2]);
          wmma::mma_sync(cfr,afr,bfr,cfr);
        }
        wmma::store_matrix_sync(hSh+nt*16, cfr, H, wmma::mem_row_major);
      } }
    __syncthreads();
    for(int i=tid;i<FR*H;i+=blockDim.x) hSh[i]+=bEnc[i%H];   /* bias (tail rows get bEnc — finite, unread) */
  } else {
    const float* w = encStaged? (wbuf+(size_t)j*(D+1)) : (wEnc+(size_t)j*D);
    for(int rr=r0;rr<nr;rr+=rpp){ float acc=bEnc[j];
      if(c_mgFma) for(int k=0;k<D;k++) acc=__fmaf_rn(w[k],obsSh[rr*Ds+k],acc);
      else        for(int k=0;k<D;k++) acc+=w[k]*obsSh[rr*Ds+k];
      hSh[rr*H+j]=acc; }
  }
  __syncthreads();
  /* layers — tf32 WMMA: y[FR x 3H] = hSh[FR x H] · Wl^T, tiles m16n16k8, f32 accumulate.
     B-fragments load DIRECTLY FROM GLOBAL (L2-hot weights, ld=H aligned): each weight element feeds
     exactly one fragment per block, so the shared staging round-trip (write+sync+read) the scalar
     design needed is pure overhead here — the per-step restaging the persistent-kernel idea chased,
     removed without any persistence. */
  for(int l=0;l<L;l++){
    const float* Wl=layers+(size_t)l*layerSz;
    if(useBf){
      /* bf16 m16n16k16: half the k-steps and fragment loads of the tf32 path — PufferLib's forward tier */
      for(int i=tid;i<FR*H;i+=blockDim.x) hShB[i]=__float2bfloat16(hSh[i]);
      __syncthreads();
      int nTiles=3*H/16, nw=(int)(blockDim.x>>5);
      for(int nt0=warp; nt0<nTiles; nt0+=2*nw){                    /* dual-tile: see the tf32 loop's note */
        int nt1=nt0+nw; int two=(nt1<nTiles);
        wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
        wmma::fill_fragment(c0,0.0f); if(two) wmma::fill_fragment(c1,0.0f);
        for(int kt=0;kt<H/16;kt++){
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b0,b1;
          wmma::load_matrix_sync(afr, hShB+kt*16, H);
          wmma::load_matrix_sync(b0, c_wlBf+(size_t)l*layerSz+(size_t)nt0*16*H+kt*16, H);
          if(two) wmma::load_matrix_sync(b1, c_wlBf+(size_t)l*layerSz+(size_t)nt1*16*H+kt*16, H);
          wmma::mma_sync(c0,afr,b0,c0);
          if(two) wmma::mma_sync(c1,afr,b1,c1);
        }
        wmma::store_matrix_sync(ySh+nt0*16, c0, 3*H, wmma::mem_row_major);
        if(two) wmma::store_matrix_sync(ySh+nt1*16, c1, 3*H, wmma::mem_row_major);
      }
    } else {
    { int nTiles=3*H/16, nw=(int)(blockDim.x>>5);
      /* DUAL-TILE per warp: at h128 nTiles(24) > warps(16), so half the warps used to run TWO serial
         16-step k-loops, each step latency-chained on an un-prefetched L2 B-fragment load — the node-
         level nsys profile put this kernel at 83% of ALL GPU time (~42us/launch) with exactly that
         critical path. Pairing tiles (nt, nt+nw) in ONE k-loop halves the longest warp's chain and
         dual-issues the two independent B-loads (L2 latency overlap), reusing one A-fragment for both.
         BIT-IDENTICAL: each tile's fragment/convert/mma sequence is unchanged — only the warp->tile
         mapping changed, and tiles are independent accumulators. Shapes where no warp owns two tiles
         (e.g. h64: 12 tiles / 16 warps) degrade to exactly the old behavior. */
      for(int nt0=warp; nt0<nTiles; nt0+=2*nw){
        int nt1=nt0+nw; int two=(nt1<nTiles);
        wmma::fragment<wmma::accumulator,16,16,8,float> c0,c1;
        wmma::fill_fragment(c0,0.0f); if(two) wmma::fill_fragment(c1,0.0f);
        for(int kt=0;kt<H/8;kt++){
          wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b0,b1;
          wmma::load_matrix_sync(afr, hSh+kt*8, H);
          wmma::load_matrix_sync(b0, Wl+(size_t)nt0*16*H+kt*8, H);
          if(two) wmma::load_matrix_sync(b1, Wl+(size_t)nt1*16*H+kt*8, H);
          for(int t=0;t<afr.num_elements;t++) afr.x[t]=wmma::__float_to_tf32(afr.x[t]);
          for(int t=0;t<b0.num_elements;t++) b0.x[t]=wmma::__float_to_tf32(b0.x[t]);
          if(two) for(int t=0;t<b1.num_elements;t++) b1.x[t]=wmma::__float_to_tf32(b1.x[t]);
          wmma::mma_sync(c0,afr,b0,c0);
          if(two) wmma::mma_sync(c1,afr,b1,c1);
        }
        wmma::store_matrix_sync(ySh+nt0*16, c0, 3*H, wmma::mem_row_major);
        if(two) wmma::store_matrix_sync(ySh+nt1*16, c1, 3*H, wmma::mem_row_major);
      } }
    }
    __syncthreads();
    /* gate — elementwise over nr·H, y from ySh */
    { float hn[4], oo[4]; long si[4]; int cnt=0;
      for(int idx=tid; idx<nr*H && cnt<4; idx+=blockDim.x){
        int rr=idx/H, jj=idx%H;
        const float* y=ySh+(size_t)rr*3*H;
        float hid=y[jj], gt=y[H+jj], pj=y[2*H+jj];
        float z=d_sigf(gt), gg=d_gactf(hid);
        si[cnt]=(row0+rr)*LHs+(long)l*H+jj;
        float prev=(termSh[rr]!=0.0f)? 0.0f : st[si[cnt]];
        float hg=d_sigf(pj), hi2=hSh[rr*H+jj];
        if(c_mgFma){ oo[cnt]=__fmaf_rn(z,gg,__fmaf_rn(-z,prev,prev));
          hn[cnt]=__fmaf_rn(hg,oo[cnt],__fmaf_rn(-hg,hi2,hi2)); }
        else { oo[cnt]=(1.0f-z)*prev+z*gg;
          hn[cnt]=hg*oo[cnt]+(1.0f-hg)*hi2; }
        cnt++;
      }
      __syncthreads();
      { int c2=0; for(int idx=tid; idx<nr*H && c2<4; idx+=blockDim.x){
          int rr=idx/H, jj=idx%H; hSh[rr*H+jj]=hn[c2]; st[si[c2]]=oo[c2]; c2++; } }
      __syncthreads();
    }
  }
  /* heads — scalar-exact */
  for(int i=tid;i<(A+1)*H;i+=blockDim.x){ int r=i/H,c=i%H; wbuf[r*(H+1)+c]= (r<A)? wDec[i] : wVal[c]; }
  __syncthreads();
  for(int rr=r0;rr<nr;rr+=rpp) if(j<A+1){ const float* w=wbuf+(size_t)j*(H+1); float acc=(j<A)?bDec[j]:bVal[0];
    if(c_mgFma) for(int k=0;k<H;k++) acc=__fmaf_rn(w[k],hSh[rr*H+k],acc);
    else        for(int k=0;k<H;k++) acc+=w[k]*hSh[rr*H+k];
    lgSh[rr*(A+1)+j]=acc; }
  __syncthreads();
  if(outAsm) for(int i=tid;i<nr*(A+1);i+=blockDim.x) outAsm[(row0)*(A+1)+i]=(double)lgSh[i];
  if(sampOut && j==0) for(int rr=r0;rr<nr;rr+=rpp)
    d_mg_sample_row(lgSh+rr*(A+1), A, row0+rr, rowBase, rng, sampOut, sampCount,
                    cAct, cLogp, cVal0, cValue, trajRow0, T_traj, sStep);
}
/* ===== MULTI-DISCRETE fused step (K categorical heads) ==========================================
   A VERBATIM copy of k_mg_fused_step_w above — same obs staging, same encoder tiers, same WMMA layer
   loop, same head GEMM, same shared-memory layout with A := W = Σ headSizes — differing ONLY in the
   sampling tail (d_mg_sample_row_md instead of d_mg_sample_row) and the two extra args it needs.
   It is a COPY rather than a shared parameterised body on purpose: the single-discrete kernel's
   generated code must stay bit-identical (the trainer's squared/breakout/pong traces are compared
   byte-for-byte), and this file's other fused variants (_p, _w) are copies for the same reason.
   Shared-memory fit is decided by the SAME mg_wmma_shbytes(D,H,L,W) the host calls, so layout and
   allocation always agree; a shape that returns 0 there never reaches this kernel (the driver falls
   back to the non-fused MD rollout). */
__global__ void k_mg_fused_step_w_md(const float* __restrict__ dP, const void* __restrict__ obsF, int obsKind, int FR,
    float* traj, long trajRow0, long sStep, long T_traj, float* st, long LHs, const double* terms,
    double* sampOut, long sampCount, long rowBase, unsigned long long rng,
    double* cAct, double* cLogp, double* cVal0, double* cValue,
    double* outAsm, int nRows, int D, int H, int L, int A, int K, const int* __restrict__ headSizes,
    const unsigned long long* rngDev, unsigned long long tOff){
  /* graph-friendly rng: when rngDev is set, the effective per-step seed is *rngDev + tOff (the
     per-update rolloutRng lives in the device scalar, the per-step offset t·N·G is BAKED into the
     capturing graph) — the same u64 the eager launch passes as `rng`, so replay is bit-identical. */
  if(rngDev) rng=*rngDev+tOff;
  extern __shared__ float sh[];
  /* big-H mode: when the FULL layout (encoder weights staged) would blow the 90KB budget, skip the
     encoder staging and read wEnc global-direct in the encoder dot (a pure copy elision — same values,
     same order, bit-identical; rows are L2-hot across the rpp passes). wbuf then only holds the heads.
     The host's mg_wmma_shbytes computes the SAME predicate, so layout and allocation always agree. */
  int wbufSz=H*(D+1); { int d=(A+1)*(H+1); if(d>wbufSz) wbufSz=d; }
  int encStaged=1;
  { long tot=4L*((long)wbufSz + (long)FR*3*H + (long)FR*D + (long)FR*H + (long)FR*(A+1) + FR)
             +2L*FR*H+32;   /* + the bf16-tile RESERVE — matches the host's gate-independent predicate */
    if(tot>90*1024){ encStaged=0; wbufSz=(int)((((A+1)*(H+1))+7)&~7); } }   /* pad: ySh/hSh 256-bit-aligned */
  int Ds = encStaged? D : ((D+7)&~7);   /* big-H: fragment-legal obs stride (pad cols zeroed below) */
  float* wbuf=sh; float* ySh=sh+wbufSz; float* obsSh=ySh+(size_t)FR*3*H;
  float* hSh=obsSh+FR*Ds; float* lgSh=hSh+FR*H; float* termSh=lgSh+FR*(A+1);
  int tid=threadIdx.x, r0=tid/H, j=tid%H, rpp=blockDim.x/H, warp=tid>>5;
  long row0=(long)blockIdx.x*FR;
  int nr=(int)((nRows-row0)<FR?(nRows-row0):FR); if(nr<=0) return;
  if(tid<nr) termSh[tid]=(terms && terms[row0+tid]!=0.0)?1.0f:0.0f;
  for(int i=tid;i<FR*H;i+=blockDim.x) if(i>=nr*H) hSh[i]=0.0f;   /* zero ALL tail rows the fragments read
     (the old single-shot `if(tid...)` covered only blockDim elements — latent for nRows%FR!=0) */
  size_t wEncSz=(size_t)H*D, layerSz=(size_t)3*H*H;
  const float* wEnc=dP; const float* bEnc=dP+wEncSz; const float* layers=dP+wEncSz+H;
  const float* wDec=layers+(size_t)L*layerSz; const float* bDec=wDec+(size_t)A*H;
  const float* wVal=bDec+A; const float* bVal=wVal+H;
  /* obs tile (+ traj scatter) — as the plain kernel */
  if(obsKind==1 && (D&3)==0){
    const uchar4* ob=(const uchar4*)obsF; int D4=D>>2;
    for(int i4=tid;i4<nr*D4;i4+=blockDim.x){ int r=i4/D4, k4=i4%D4;
      uchar4 w=ob[(row0+r)*(long)D4+k4]; int k=4*k4;
      float v0=(float)w.x,v1=(float)w.y,v2=(float)w.z,v3=(float)w.w;
      float* os=obsSh+r*Ds+k; os[0]=v0; os[1]=v1; os[2]=v2; os[3]=v3;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; tj[2]=v2; tj[3]=v3; } }
  } else if(obsKind==1){                       /* u8, ANY D: tile-linear aligned loads (see d_mg_obs_u8_tile) */
    d_mg_obs_u8_tile((const unsigned char*)obsF, obsSh, Ds, traj, trajRow0, row0, sStep, T_traj, nr, D, tid, blockDim.x);
  } else if(obsKind==2 && (D&1)==0){
    const unsigned int* ob=(const unsigned int*)obsF; int D2=D>>1;
    for(int i2=tid;i2<nr*D2;i2+=blockDim.x){ int r=i2/D2, k2=i2%D2;
      unsigned int w=ob[(row0+r)*(long)D2+k2]; int k=2*k2;
      __nv_bfloat16 lo=*(__nv_bfloat16*)&w, hi=*(((__nv_bfloat16*)&w)+1);
      float v0=__bfloat162float(lo), v1=__bfloat162float(hi);
      float* os=obsSh+r*Ds+k; os[0]=v0; os[1]=v1;
      if(traj){ float* tj=traj+((trajRow0+row0+r)*T_traj+sStep)*D+k; tj[0]=v0; tj[1]=v1; } }
  } else
  for(int i=tid;i<nr*D;i+=blockDim.x){ int r=i/D, k=i%D; long oi=(row0+r)*(long)D+k;
    float v = (obsKind==1)? (float)((const unsigned char*)obsF)[oi]
            : (obsKind==2)? __bfloat162float(((const __nv_bfloat16*)obsF)[oi])
            : ((const float*)obsF)[oi];
    obsSh[r*Ds+k]=v; if(traj) traj[((trajRow0+row0+r)*T_traj+sStep)*D+k]=v; }
  /* encoder — scalar-exact when staged (h≤64 envs: byte-lineage preserved). Big-H: tf32 WMMA over the
     padded obs tile × the padded weight copy (same tier as the layers), scalar global-direct fallback
     when the padded weights weren't prepared (c_wEncPad NULL — e.g. an uninstrumented caller). */
  if(encStaged) for(int i=tid;i<H*D;i+=blockDim.x) wbuf[(i/D)*(D+1)+(i%D)]=wEnc[i];
  if(!encStaged){   /* zero the pad columns + tail rows the encoder fragments read */
    for(int i=tid;i<FR*Ds;i+=blockDim.x){ int r=i/Ds, k=i%Ds; if(k>=D || r>=nr) obsSh[i]=0.0f; } }
  __syncthreads();
  __nv_bfloat16* hShB=(__nv_bfloat16*)(((size_t)(termSh+FR)+31)&~(size_t)31);   /* bf16 tier only
     (32B-aligned — wmma mptr documents a 256-bit requirement; the host's 32B reserve covers the pad) */
  int useBf=(c_wlBf!=NULL && c_wlBfH==H && c_wlBfL>=L);
  if(!encStaged && useBf && c_wEncBf && c_encDp==Ds && c_encH==H && (Ds&15)==0 && Ds<=H){
    /* bf16 m16n16k16 encoder — the layer tier extended to the encoder GEMM. The obs tile converts
       through hShB (fits: Ds<=H gate, and hSh isn't live until the accumulator stores below); u8 obs
       values are EXACT in bf16 (integers <=256), weights get the same tier rounding as the layers. */
    for(int i=tid;i<FR*Ds;i+=blockDim.x) hShB[i]=__float2bfloat16(obsSh[i]);
    __syncthreads();
    { int nT=H/16;
      for(int nt=warp; nt<nT; nt+=(int)(blockDim.x>>5)){
        wmma::fragment<wmma::accumulator,16,16,16,float> cfr; wmma::fill_fragment(cfr,0.0f);
        for(int kt=0;kt<Ds/16;kt++){
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfr;
          wmma::load_matrix_sync(afr, hShB+kt*16, Ds);
          wmma::load_matrix_sync(bfr, c_wEncBf+(size_t)nt*16*Ds+kt*16, Ds);
          wmma::mma_sync(cfr,afr,bfr,cfr);
        }
        wmma::store_matrix_sync(hSh+nt*16, cfr, H, wmma::mem_row_major);
      } }
    __syncthreads();
    for(int i=tid;i<FR*H;i+=blockDim.x) hSh[i]+=bEnc[i%H];
  } else if(!encStaged && c_wEncPad && c_encDp==Ds && c_encH==H){
    const float* wp=c_wEncPad;
    { int nT=H/16;
      for(int nt=warp; nt<nT; nt+=(int)(blockDim.x>>5)){
        wmma::fragment<wmma::accumulator,16,16,8,float> cfr; wmma::fill_fragment(cfr,0.0f);
        for(int kt=0;kt<Ds/8;kt++){
          wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> bfr;
          wmma::load_matrix_sync(afr, obsSh+kt*8, Ds);
          wmma::load_matrix_sync(bfr, wp+(size_t)nt*16*Ds+kt*8, Ds);
          for(int t2=0;t2<afr.num_elements;t2++) afr.x[t2]=wmma::__float_to_tf32(afr.x[t2]);
          for(int t2=0;t2<bfr.num_elements;t2++) bfr.x[t2]=wmma::__float_to_tf32(bfr.x[t2]);
          wmma::mma_sync(cfr,afr,bfr,cfr);
        }
        wmma::store_matrix_sync(hSh+nt*16, cfr, H, wmma::mem_row_major);
      } }
    __syncthreads();
    for(int i=tid;i<FR*H;i+=blockDim.x) hSh[i]+=bEnc[i%H];   /* bias (tail rows get bEnc — finite, unread) */
  } else {
    const float* w = encStaged? (wbuf+(size_t)j*(D+1)) : (wEnc+(size_t)j*D);
    for(int rr=r0;rr<nr;rr+=rpp){ float acc=bEnc[j];
      if(c_mgFma) for(int k=0;k<D;k++) acc=__fmaf_rn(w[k],obsSh[rr*Ds+k],acc);
      else        for(int k=0;k<D;k++) acc+=w[k]*obsSh[rr*Ds+k];
      hSh[rr*H+j]=acc; }
  }
  __syncthreads();
  /* layers — tf32 WMMA: y[FR x 3H] = hSh[FR x H] · Wl^T, tiles m16n16k8, f32 accumulate.
     B-fragments load DIRECTLY FROM GLOBAL (L2-hot weights, ld=H aligned): each weight element feeds
     exactly one fragment per block, so the shared staging round-trip (write+sync+read) the scalar
     design needed is pure overhead here — the per-step restaging the persistent-kernel idea chased,
     removed without any persistence. */
  for(int l=0;l<L;l++){
    const float* Wl=layers+(size_t)l*layerSz;
    if(useBf){
      /* bf16 m16n16k16: half the k-steps and fragment loads of the tf32 path — PufferLib's forward tier */
      for(int i=tid;i<FR*H;i+=blockDim.x) hShB[i]=__float2bfloat16(hSh[i]);
      __syncthreads();
      int nTiles=3*H/16, nw=(int)(blockDim.x>>5);
      for(int nt0=warp; nt0<nTiles; nt0+=2*nw){                    /* dual-tile: see the tf32 loop's note */
        int nt1=nt0+nw; int two=(nt1<nTiles);
        wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
        wmma::fill_fragment(c0,0.0f); if(two) wmma::fill_fragment(c1,0.0f);
        for(int kt=0;kt<H/16;kt++){
          wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b0,b1;
          wmma::load_matrix_sync(afr, hShB+kt*16, H);
          wmma::load_matrix_sync(b0, c_wlBf+(size_t)l*layerSz+(size_t)nt0*16*H+kt*16, H);
          if(two) wmma::load_matrix_sync(b1, c_wlBf+(size_t)l*layerSz+(size_t)nt1*16*H+kt*16, H);
          wmma::mma_sync(c0,afr,b0,c0);
          if(two) wmma::mma_sync(c1,afr,b1,c1);
        }
        wmma::store_matrix_sync(ySh+nt0*16, c0, 3*H, wmma::mem_row_major);
        if(two) wmma::store_matrix_sync(ySh+nt1*16, c1, 3*H, wmma::mem_row_major);
      }
    } else {
    { int nTiles=3*H/16, nw=(int)(blockDim.x>>5);
      /* DUAL-TILE per warp: at h128 nTiles(24) > warps(16), so half the warps used to run TWO serial
         16-step k-loops, each step latency-chained on an un-prefetched L2 B-fragment load — the node-
         level nsys profile put this kernel at 83% of ALL GPU time (~42us/launch) with exactly that
         critical path. Pairing tiles (nt, nt+nw) in ONE k-loop halves the longest warp's chain and
         dual-issues the two independent B-loads (L2 latency overlap), reusing one A-fragment for both.
         BIT-IDENTICAL: each tile's fragment/convert/mma sequence is unchanged — only the warp->tile
         mapping changed, and tiles are independent accumulators. Shapes where no warp owns two tiles
         (e.g. h64: 12 tiles / 16 warps) degrade to exactly the old behavior. */
      for(int nt0=warp; nt0<nTiles; nt0+=2*nw){
        int nt1=nt0+nw; int two=(nt1<nTiles);
        wmma::fragment<wmma::accumulator,16,16,8,float> c0,c1;
        wmma::fill_fragment(c0,0.0f); if(two) wmma::fill_fragment(c1,0.0f);
        for(int kt=0;kt<H/8;kt++){
          wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> afr;
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b0,b1;
          wmma::load_matrix_sync(afr, hSh+kt*8, H);
          wmma::load_matrix_sync(b0, Wl+(size_t)nt0*16*H+kt*8, H);
          if(two) wmma::load_matrix_sync(b1, Wl+(size_t)nt1*16*H+kt*8, H);
          for(int t=0;t<afr.num_elements;t++) afr.x[t]=wmma::__float_to_tf32(afr.x[t]);
          for(int t=0;t<b0.num_elements;t++) b0.x[t]=wmma::__float_to_tf32(b0.x[t]);
          if(two) for(int t=0;t<b1.num_elements;t++) b1.x[t]=wmma::__float_to_tf32(b1.x[t]);
          wmma::mma_sync(c0,afr,b0,c0);
          if(two) wmma::mma_sync(c1,afr,b1,c1);
        }
        wmma::store_matrix_sync(ySh+nt0*16, c0, 3*H, wmma::mem_row_major);
        if(two) wmma::store_matrix_sync(ySh+nt1*16, c1, 3*H, wmma::mem_row_major);
      } }
    }
    __syncthreads();
    /* gate — elementwise over nr·H, y from ySh */
    { float hn[4], oo[4]; long si[4]; int cnt=0;
      for(int idx=tid; idx<nr*H && cnt<4; idx+=blockDim.x){
        int rr=idx/H, jj=idx%H;
        const float* y=ySh+(size_t)rr*3*H;
        float hid=y[jj], gt=y[H+jj], pj=y[2*H+jj];
        float z=d_sigf(gt), gg=d_gactf(hid);
        si[cnt]=(row0+rr)*LHs+(long)l*H+jj;
        float prev=(termSh[rr]!=0.0f)? 0.0f : st[si[cnt]];
        float hg=d_sigf(pj), hi2=hSh[rr*H+jj];
        if(c_mgFma){ oo[cnt]=__fmaf_rn(z,gg,__fmaf_rn(-z,prev,prev));
          hn[cnt]=__fmaf_rn(hg,oo[cnt],__fmaf_rn(-hg,hi2,hi2)); }
        else { oo[cnt]=(1.0f-z)*prev+z*gg;
          hn[cnt]=hg*oo[cnt]+(1.0f-hg)*hi2; }
        cnt++;
      }
      __syncthreads();
      { int c2=0; for(int idx=tid; idx<nr*H && c2<4; idx+=blockDim.x){
          int rr=idx/H, jj=idx%H; hSh[rr*H+jj]=hn[c2]; st[si[c2]]=oo[c2]; c2++; } }
      __syncthreads();
    }
  }
  /* heads — scalar-exact */
  for(int i=tid;i<(A+1)*H;i+=blockDim.x){ int r=i/H,c=i%H; wbuf[r*(H+1)+c]= (r<A)? wDec[i] : wVal[c]; }
  __syncthreads();
  for(int rr=r0;rr<nr;rr+=rpp) if(j<A+1){ const float* w=wbuf+(size_t)j*(H+1); float acc=(j<A)?bDec[j]:bVal[0];
    if(c_mgFma) for(int k=0;k<H;k++) acc=__fmaf_rn(w[k],hSh[rr*H+k],acc);
    else        for(int k=0;k<H;k++) acc+=w[k]*hSh[rr*H+k];
    lgSh[rr*(A+1)+j]=acc; }
  __syncthreads();
  if(outAsm) for(int i=tid;i<nr*(A+1);i+=blockDim.x) outAsm[(row0)*(A+1)+i]=(double)lgSh[i];
  if(sampOut && j==0) for(int rr=r0;rr<nr;rr+=rpp)
    d_mg_sample_row_md(lgSh+rr*(A+1), A, K, headSizes, row0+rr, rowBase, rng, sampOut, sampCount,
                    cAct, cLogp, cVal0, cValue, trajRow0, T_traj, sStep);
}
#define MG_WFR 16
static size_t mg_wmma_shbytes(int D,int H,int L,int A){
  (void)L;
  long wbuf=(long)H*(D+1); { long d=(long)(A+1)*(H+1); if(d>wbuf) wbuf=d; }   /* layers load global-direct */
  long bfTile=mg_wprec_bf()? (2L*MG_WFR*H+32) : 0;   /* appended bf16 hidden tile (16B-align slack) */
  long bfRes=2L*MG_WFR*H+32;   /* PREDICATE reserve: layout choice must be gate-independent so the kernel
                                  (which cannot see the env gate) always derives the same encStaged */
  long rest=4L*((long)MG_WFR*3*H + (long)MG_WFR*D + (long)MG_WFR*H + (long)MG_WFR*(A+1) + MG_WFR)+bfTile;
  if(H<16 || (H%16) || A+1>H) return 0;
  if(H>128) return 0;   /* gate stage caches 4 elems/thread: FR·H must be ≤ 4·blockDim(512) — H≥144 would
                           silently drop gate elements (review-caught; latent pre-existing, capped now) */
  long totP=4L*wbuf+4L*((long)MG_WFR*3*H + (long)MG_WFR*D + (long)MG_WFR*H + (long)MG_WFR*(A+1) + MG_WFR)+bfRes;
  long tot=4L*wbuf+rest;
  if(totP<=90*1024) return (size_t)tot;   /* decision by the RESERVED predicate; size by the actual layout */
  /* big-H (e.g. H=128): the kernel skips the encoder weight staging (global-direct reads — bit-identical
     copy elision); wbuf shrinks to the heads' (A+1)(H+1), padded to 8 floats so ySh/hSh keep the 256-bit
     alignment wmma mptr requires. Kernel derives the SAME predicate + padding. */
  long hb=(((long)(A+1)*(H+1))+7)&~7L;
  long Dp=((long)D+7)&~7L;
  long rest2=4L*((long)MG_WFR*3*H + (long)MG_WFR*Dp + (long)MG_WFR*H + (long)MG_WFR*(A+1) + MG_WFR)+bfTile;
  long tot2=4L*hb+rest2;
  return (tot2<=90*1024)? (size_t)tot2 : 0;
}
/* default ON: +3.4% measured; tf32 forward ~1e-3 vs the f64 oracle — TIGHTER than the FAST_16BF-compute
   default this trainer shipped pre-fusion. PUFFER_MG_WMMA=0 restores the exact-f32 fused kernel. */
static int mg_wmma(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_WMMA"); f=(e==NULL||e[0]!='0'); } return f; }
static int mg_wmma_optin(size_t shb){ static int state=0;
  if(state==0) state=(cudaFuncSetAttribute(k_mg_fused_step_w, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(96*1024))==cudaSuccess)?1:-1;
  return (shb<=48*1024) || state==1; }
/* same one-shot 96KB opt-in for the MULTI-DISCRETE twin (the attribute is per-FUNCTION, so the
   single-discrete call above does NOT cover it — an uncovered >48KB launch fails silently). */
static int mg_wmma_optin_md(size_t shb){ static int state=0;
  if(state==0) state=(cudaFuncSetAttribute(k_mg_fused_step_w_md, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(96*1024))==cudaSuccess)?1:-1;
  return (shb<=48*1024) || state==1; }
/* shared bytes + launch guard for the fused kernel; returns 0 if the config doesn't fit (caller falls back) */
static size_t mg_fused_shbytes_fr(int D,int H,int L,int A,int FR){
  (void)L; long wbuf=(long)(3*H)*(H+1); { long e=(long)H*(D+1); if(e>wbuf) wbuf=e; long d=(long)(A+1)*(H+1); if(d>wbuf) wbuf=d; }
  long tot=4L*(wbuf + (long)FR*D + (long)FR*H + (long)FR*(A+1) + FR);
  if(H<1 || 4*H>1024 || A+1>H || tot>90*1024) return 0;
  return (size_t)tot;
}
static size_t mg_fused_shbytes(int D,int H,int L,int A){ return mg_fused_shbytes_fr(D,H,L,A,MG_FR); }
static int mg_fused_optin(size_t shb){           /* raise the 96KB shared limit ONCE (harmless for small
                                                    sizes — review finding: returning early for <=48KB left
                                                    a LATER larger-FR launch uncovered, silently failing
                                                    every kernel in the resident loop); returns 0 (caller
                                                    falls back) if the GPU refuses AND shb needs it */
  static int state=0;                            /* 0=untried 1=ok -1=refused */
  if(state==0) state = (cudaFuncSetAttribute(k_mg_fused_step, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(96*1024))==cudaSuccess)? 1 : -1;
  return (shb<=48*1024) || state==1;
}
static int mg_fused(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_FUSED"); f=(e==NULL||e[0]!='0'); } return f; }
static void* g_rb2[24]; static size_t g_rb2sz[24];
static void* rb2(int i, size_t bytes){
  if(g_rb2sz[i] < bytes){ if(g_rb2[i]) cudaFree(g_rb2[i]);
    if(cudaMalloc(&g_rb2[i], bytes)!=cudaSuccess){ g_rb2[i]=NULL; g_rb2sz[i]=0; return NULL; }
    g_rb2sz[i]=bytes; }
  return g_rb2[i];
}
/* Persistent PINNED host staging for the MinGRU rollout — the per-step obs H2D is PCIe-bandwidth-bound at
   scale (f64 obs > forward compute); page-locked memory ~2×'s the transfer. Grows as needed, freed never. */
static void* g_mghb[12]; static size_t g_mghbsz[12]; static int g_mghb_pin[12];
static void* mg_hb(int i, size_t bytes){
  if(g_mghbsz[i] < bytes){ if(g_mghb[i]){ if(g_mghb_pin[i]) cudaFreeHost(g_mghb[i]); else free(g_mghb[i]); }
    g_mghb_pin[i]=(cudaHostAlloc(&g_mghb[i], bytes, cudaHostAllocDefault)==cudaSuccess);
    if(!g_mghb_pin[i]) g_mghb[i]=malloc(bytes);
    g_mghbsz[i]= g_mghb[i]? bytes : 0; }
  return g_mghb[i];
}
static int mg_hb_pinned(int i){ return g_mghb_pin[i]; }
/* zero-copy obs: the fused kernel reads the PINNED host obs directly over PCIe (UVA) — same bytes as the
   H2D it replaces, but overlapped with the kernel's own execution instead of a separate stream hop.
   PUFFER_MG_ZCOBS=0 disables (falls back to the explicit H2D). */
static int mg_zcobs(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_ZCOBS"); f=(e==NULL||e[0]!='0'); } return f; }
/* ---- persistent-thread env-step pool ------------------------------------------------------------
   The rollout's per-step env-step + column-scatter is embarrassingly parallel across envs (each env
   owns a disjoint block of rows), but the GPU forward/sample is serial. A persistent pthread pool lets
   P workers split the env-step+scatter while the main thread drives the GPU — matching PufferLib's
   native persistent-thread vecenv. Barrier-synchronized: main runs the GPU phase, both sides rendezvous
   so workers step their env partition, then rendezvous again before the next step. Deterministic (each
   env writes disjoint columns ⇒ bit-identical to the serial path). Env-agnostic: workers call the generic
   eh->step_range. Falls back to serial when step_range is NULL (a plugin .so built before it existed).
   Shared by all three native rollout drivers (MLP / multi / MinGRU). */
typedef struct {
  Handle* eh;
  const double *hSamp, *cur, *actRM; double *nxt, *hRT;   /* hSamp=[act(W×Nstride col); logp; val]; actRM/cur/nxt/hRT are GLOBAL (row-indexed) */
  double *obsCol,*actCol,*logpCol,*valCol,*rewCol,*termCol;
  long N, D, T, Nstride, rowBase; int nAgents, W, envLo, envHi, skipObs; size_t s;  /* skipObs: obs is device-resident ⇒ don't scatter obsCol */
  /* W = action components/row (1 discrete / K multi-discrete / d continuous). Workers partition [envLo,envHi).
     hSamp is a per-segment compact block: local row = globalRow−rowBase, stride Nstride. Whole-batch
     (single-buffer): envLo=0,envHi=numEnvs,rowBase=0,Nstride=N. Double-buffer: called once per env-half. */
  int nthreads, alive; pthread_barrier_t bar;
} rollpool_t;
static rollpool_t g_rp; static pthread_t g_rp_th[64]; static int g_rp_init=0;

/* One worker's slice: step its env range, then scatter those envs' rows into the SoA columns. W-wide
   actions: the sampler writes col-major (act[w·Nstride+localRow]); actCol is row-major (row·W+w). */
static void rp_run(rollpool_t* p, int tid){
  int nt=p->nthreads, A=p->nAgents, W=p->W; long D=p->D, T=p->T, N=p->N, Ns=p->Nstride, rb=p->rowBase; size_t s=p->s;
  int span=p->envHi-p->envLo;
  int eLo=p->envLo+(int)((long)tid*span/nt), eHi=p->envLo+(int)((long)(tid+1)*span/nt);
  if(eLo>=eHi) return;
  p->eh->step_range(p->eh->env, p->actRM, p->nxt, p->hRT, p->hRT+N, eLo, eHi-eLo);
  int skipObs=p->skipObs;
  for(long e=(long)eLo*A; e<(long)eHi*A; e++){ long row=e*T+(long)s, le=e-rb;   /* le = local row within the segment */
    if(!skipObs) for(long j=0;j<D;j++) p->obsCol[row*D+j]=p->cur[e*D+j];   /* obs BEFORE the step (skipped when device-resident) */
    for(int wc=0;wc<W;wc++) p->actCol[row*W+wc]=p->hSamp[(long)wc*Ns+le];
    p->logpCol[row]=p->hSamp[(long)W*Ns+le]; p->valCol[row]=p->hSamp[(long)(W+1)*Ns+le];
    p->rewCol[row]=p->hRT[e];   p->termCol[row]=p->hRT[N+e]; }
}
static void* rp_worker(void* arg){
  long tid=(long)arg;
  for(;;){ pthread_barrier_wait(&g_rp.bar); if(!g_rp.alive) return NULL; rp_run(&g_rp,(int)tid); pthread_barrier_wait(&g_rp.bar); }
}
static int rp_threads(void){                                       /* PUFFER_ROLL_THREADS, default 8 */
  const char* e=getenv("PUFFER_ROLL_THREADS"); int n=0;            /* manual parse: atoi→__isoc23_strtol,
     a C23 glibc symbol the Lean toolchain's bundled glibc lacks, so it would break the link. */
  if(e){ while(*e>='0'&&*e<='9'){ n=n*10+(*e-'0'); e++; } }
  if(n<1) n=8; if(n>63) n=63; return n;   /* 8 halves env-step+scatter vs 4 on this 16C/32T box; 16 oversubscribes the barrier */
}

/* ---- concurrent stream-buffers for the MinGRU rollout (PUFFER_MG_ROLL_BUFFERS, default 1 = off) --------
   The online rollout is a strict per-agent chain (obs[t]←env-step[t-1]), so the only parallelism is ACROSS
   agent slices: nbuf worker threads, each its own CUDA stream + cuBLAS handle, run the full T-horizon over a
   disjoint agent slice with its own IN-PLACE recurrent state slice. Buffer A's CPU env-step then overlaps
   buffer B's GPU forward. Bit-identical to the single-buffer path (same k_sample_seg rng on the GLOBAL row,
   same in-place state, same reset); only the per-buffer GEMMs are smaller (the tradeoff to weigh vs the
   overlap — cf. the MLP double-buffer which was a net loss). Obs stays device-resident (each buffer scatters
   its slice into g_dMGObsTraj via k_scatter_obs_traj). */
#define MAXBUF_MG 16
/* PUFFER_MG_ROLL_BUFFERS overrides; else default 4 when N≥512 (measured sweet spot: +13–26% at 1024–4096
   envs), 1 below that (at N≤128, 4 buffers over-split the per-step GEMMs and it's a ~20% LOSS). */
static int mg_roll_buffers(long N){ static int f=-2;
  if(f==-2){ const char* e=getenv("PUFFER_MG_ROLL_BUFFERS"); int n=0; int set=(e!=NULL);
    if(e){ while(*e>='0'&&*e<='9'){ n=n*10+(*e-'0'); e++; } }
    if(set){ if(n<1) n=1; if(n>MAXBUF_MG) n=MAXBUF_MG; f=n; } else f=-1; }        /* -1 = adaptive default */
  if(f>=0) return f;
  /* adaptive default: the fused-kernel rollout made the GPU share cheap, so the per-buffer cycle is
     dominated by env-CPU + stream-sync — BOTH scale down with buffer count (each buffer = one worker
     thread stepping its env slice). Measured @4096 on 32 cores: 4→9.2M, 8→10.9M, 16→11.2M, 32→11.1M
     SPS. Cap by cores (leave headroom for the env pool/main) and keep slices ≥128 rows so the fused
     kernel retains enough blocks. Buffer count is BIT-NEUTRAL (global-row rng, disjoint slices). */
  if(N<512) return 1;
  long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<4) ncpu=4;
  long nb=N/128; if(nb>16) nb=16; if(nb>ncpu/2) nb=ncpu/2; if(nb<4) nb=4;
  return (int)nb; }
typedef struct {
  Handle* eh; const double* obs0; const float* dP; float* dMGTraj;
  double *dObs,*dY,*dO,*dTerms; float *dObsF,*dSa,*dHb,*dHn,*dYb,*dLg,*dVal;
  double *hSamp,*hA,*hB,*hRT,*actRM,*actPin; float *hObsF32,*hObsF32B; int f32obs;
  int dcOK;                  /* device-direct resident columns enable flag */
  double *rewPlane,*termPlane;   /* pinned [T·N] planes: workers memcpy slices; ONE per-update scatter */
  int* flagH; int spin;   /* stream-ordered stamp-spin completion (64B-strided pinned flags; default OFF) */
  int obsKind;            /* obs TRANSPORT element: 0 f32, 1 u8 (byte envs, exact), 2 bf16 (tolerance) */
  double prof[64][8];        /* per-buffer cycle profile (PUFFER_ROLL_PROFILE=1): cast/launch/sync/act/env/scat/tail */
  double *actCol,*logpCol,*valCol,*rewCol,*termCol;
  long N,D,T,LH; int H,L,A,O,bf,nAg; uint64_t rolloutRng;
  int rowBase[64], rowN[64], envLo[64], envHi[64];
  int nbuf;   /* M1: needed by workers to derive E = max(1, PUFFER_MG_ENVTHREADS/nbuf) */
  int wideBf; /* bf16-storage wide arm (mg_wide_bf_body) — set by the driver when gated in */
  int startPc, skipPre; /* resident chaining: obs ping-pong parity carried across updates; skip the
                           obs0 pre-loop cast when the staging already holds the live obs */
  /* MULTI-DISCRETE arm (mg_buf_worker_md): md=1 selects it, K = action heads per row, dHs = the
     device head-size vector, A above holds W = Σ headSizes (the logits width). md=0 for every
     single-discrete call — the worker dispatch is one `if(p->md)` line, exactly like wideBf. */
  int md, K; const int* dHs;
} mgbufpool_t;
static mgbufpool_t g_mgbp;
static cudaStream_t g_mgbufst[64]; static cublasHandle_t g_mgbufh[64]; static int g_mgbuf_ready=0;
static void mgbuf_init(int nbuf){
  for(int b=g_mgbuf_ready;b<nbuf;b++){ cudaStreamCreate(&g_mgbufst[b]); cublasCreate(&g_mgbufh[b]);
    void* ws=NULL; if(cudaMalloc(&ws,4*1024*1024)==cudaSuccess) cublasSetWorkspace(g_mgbufh[b],ws,4*1024*1024); }
  if(nbuf>g_mgbuf_ready) g_mgbuf_ready=nbuf;
}
/* ---- Wide-graph rollout arm (PUFFER_MG_WIDEG=1, composes with PUFFER_MG_FUSED=0 + a small
   PUFFER_MG_ROLL_BUFFERS): CUDA-graph capture of the non-fused (GEMM) per-step chain — the piece the
   M6 wide-GEMM refutation predates. M6's wide arm lost partly because each step paid ~12 real launches
   (5 GEMMs + gate/bias/asm/sample kernels); capturing the step-invariant chain (forward + assemble +
   sample, rng via a per-buffer device scalar exactly like the MLP path's g_dRbase) collapses that to
   one graph launch. Step-VARYING launches (obs traj scatter, resident-column fold, terminal reset —
   all take `s`) stay outside the graph, same split the MLP buffered path uses. The sampler inside the
   graph follows the PUFFER_MG_SAMPF32 tier (default f32 — k_mingru_asm_f32 + k_sample_seg_g_f32,
   aliasing the f64 dY scratch as float; =0 keeps the f64 k_mingru_asm + k_sample_seg_g). Re-captures
   if the (nb,D,H,L,A) shape changes; capture failure falls back to eager launches permanently. */
static int mg_wideg(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_WIDEG"); f=(e!=NULL&&e[0]=='1')?1:0; } return f; }
static int mg_sampf32_host(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_SAMPF32"); f=(e==NULL||e[0]!='0')?1:0; } return f; }
static cudaGraphExec_t g_mgg[MAXBUF_MG]; static int g_mgg_ok[MAXBUF_MG];
static unsigned long long* g_mgg_rng[MAXBUF_MG]; static long g_mgg_key[MAXBUF_MG];
static void* mg_buf_worker(void* arg);
/* M4a: PERSISTENT buffer-worker threads (mirrors the proven g_rp barrier pattern). Always sized to
   MAXBUF_MG (16) — nbuf varies call-to-call (adaptive on N), but the barrier's party count must be
   fixed once threads exist, so EVERY created thread is released each call and self-gates on b<nbuf
   (indices ≥nbuf sit out via mg_buf_worker's own nb<=0 early-return, reached with rowN[b] left at 0
   for those slots — see the caller). Kills the pthread_create+join pair (~10-30us × nbuf) every
   update; zero data-flow change, so bit-identity is structural (same mg_buf_worker body, same call). */
static pthread_t g_mgbufw_th[MAXBUF_MG]; static pthread_barrier_t g_mgbufw_bar;
static int g_mgbufw_n=0, g_mgbufw_alive=0;
static void* mgbufw_thread(void* arg){
  long b=(long)arg;
  for(;;){ pthread_barrier_wait(&g_mgbufw_bar); if(!g_mgbufw_alive) return NULL;
    mg_buf_worker((void*)b); pthread_barrier_wait(&g_mgbufw_bar); }
}
static int mgbufw_init(void){
  static int fail=0; if(fail) return 0;
  if(g_mgbufw_n==MAXBUF_MG) return 1;
  if(g_mgbufw_n!=0) return 0;   /* defensive: g_mgbufw_n only ever transitions 0→MAXBUF_MG (see below),
                                   so this is unreachable in practice — kept as a guard regardless */
  if(pthread_barrier_init(&g_mgbufw_bar,NULL,MAXBUF_MG+1)!=0){ fail=1; return 0; }
  g_mgbufw_alive=1;
  for(long b=0;b<MAXBUF_MG;b++){
    if(pthread_create(&g_mgbufw_th[b],NULL,mgbufw_thread,(void*)b)!=0){
      /* REVIEW-CAUGHT DEADLOCK (fixed): the original recovery path called ONE more barrier_wait from
         here to "wake" the b already-started threads — but the barrier needs MAXBUF_MG+1 TOTAL
         arrivals to release ANYONE, and at most b+1 (<MAXBUF_MG+1, always) can ever arrive once the
         remaining creates have failed. That call — and every already-started thread's first wait —
         would block forever: the calling (rollout) thread hangs inside mgbufw_init, never reaching
         the fallback. Fix: do NOT touch the barrier again. The b already-started threads are
         abandoned, parked forever in their first wait — a harmless leak (this file's other
         persistent pools — g_rp, the muon pool, the bs side-stream pool — also have no teardown
         path; parking-forever-on-failure is consistent with, not worse than, standing practice).
         Joining or destroying here would itself block/UB on threads stuck in the barrier. */
      g_mgbufw_alive=0; fail=1; return 0;
    }
  }
  g_mgbufw_n=MAXBUF_MG; return 1;
}
/* ---- M1: per-buffer env sub-pool (nested inside mg_buf_worker) ---------------------------------
   Decouples env-step FAN-OUT from GPU forward WIDTH. Historically nbuf served both roles: more
   buffers meant both more launch parallelism (narrower forwards) AND more env-stepping threads
   (one per buffer, serial with its own GPU cycle). PUFFER_MG_ENVTHREADS sets the TOTAL env-stepping
   thread count across all buffers; E=max(1,total/nbuf) persistent sub-workers per buffer split that
   buffer's [envLo,envHi) further. E=1 (env var unset, or total<=nbuf) takes the ORIGINAL code path
   verbatim — this is the default, so M1 is a no-op until the env var raises E above nbuf. All writes
   are to disjoint absolute (env,step) rows regardless of partition depth ⇒ bit-identical to the
   serial path at any E. No live resize: if a buffer's pool already exists at a DIFFERENT E than
   requested, the call falls back to E=1 for that buffer rather than tearing down live threads. */
static int mg_env_threads_total(void){ static int v=-1; if(v<0){ const char* e=getenv("PUFFER_MG_ENVTHREADS");
  v=0; if(e){ while(*e>='0'&&*e<='9'){ v=v*10+(*e-'0'); e++; } } if(v<0) v=0; if(v>256) v=256; } return v; }
/* NOTE (measured 2026-08-04): converting this pool's pthread barrier to a sense-reversal SPIN barrier
   (PufferLib's OMP threads spin) bought ~nothing where it engages (2buf +0.8M, 4buf flat) and created
   a real footgun — 32 spinning env threads oversubscribed the 32-core box to 688K SPS. The futex
   wakeup latency was NOT a meaningful cost here; reverted to the sleeping barrier. */
typedef struct {
  mgbufpool_t* p; int b;                  /* shared context + which buffer owns this sub-pool */
  const double* actRM; void* pfOut; double* nxt;   /* refreshed by the owning thread each step */
  size_t s; long nb; int elo, ehi;
  int nthreads, alive, E; pthread_barrier_t bar;
} mgsub_t;
static mgsub_t g_mgsub[MAXBUF_MG]; static pthread_t g_mgsub_th[MAXBUF_MG][256];
static void mgsub_run(mgsub_t* q, int tid){
  int nt=q->nthreads, span=q->ehi-q->elo;
  int eLo=q->elo+(int)((long)tid*span/nt), eHi=q->elo+(int)((long)(tid+1)*span/nt);
  if(eLo>=eHi) return;
  mgbufpool_t* p=q->p; long D=p->D,T=p->T,N=p->N,rb=p->rowBase[q->b],nb=q->nb; int nAg=p->nAg;
  size_t s=q->s; int okind=p->obsKind, f32o=p->f32obs;
  if(okind==1)      p->eh->step_range_u8  (p->eh->env, q->actRM, (unsigned char*) q->pfOut, p->hRT, p->hRT+N, eLo, eHi-eLo);
  else if(okind==2) p->eh->step_range_bf16(p->eh->env, q->actRM, (unsigned short*)q->pfOut, p->hRT, p->hRT+N, eLo, eHi-eLo);
  else if(f32o)     p->eh->step_range_f32 (p->eh->env, q->actRM, (float*)         q->pfOut, p->hRT, p->hRT+N, eLo, eHi-eLo);
  else              p->eh->step_range     (p->eh->env, q->actRM, q->nxt,                    p->hRT, p->hRT+N, eLo, eHi-eLo);
  (void)D;
  if(p->dcOK){
    if(p->rewCol) for(long e=(long)eLo*nAg; e<(long)eHi*nAg; e++){ long row=e*T+(long)s;
      p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
  } else {
    for(long e=(long)eLo*nAg; e<(long)eHi*nAg; e++){ long row=e*T+(long)s, le=e-rb;
      p->actCol[row]=p->hSamp[3*rb+le]; p->logpCol[row]=p->hSamp[3*rb+nb+le]; p->valCol[row]=p->hSamp[3*rb+2*nb+le];
      p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
  }
}
typedef struct { mgsub_t* q; int tid; } mgsub_arg_t;
static mgsub_arg_t g_mgsub_arg[MAXBUF_MG][256];
static void* mgsub_worker2(void* arg){
  mgsub_arg_t* a=(mgsub_arg_t*)arg; mgsub_t* q=a->q; int tid=a->tid;
  for(;;){ pthread_barrier_wait(&q->bar); if(!q->alive) return NULL; mgsub_run(q,tid); pthread_barrier_wait(&q->bar); }
}
/* returns the sub-pool for buffer b sized to E, or NULL if E<=1 or a live pool exists at a different E
   (safe fallback — caller takes the original unnested path). */
static mgsub_t* mgsub_for(int b, int E){
  if(E<=1) return NULL;
  mgsub_t* q=&g_mgsub[b];
  if(q->nthreads==E) return q;
  if(q->nthreads!=0) return NULL;   /* live pool at a different E — fall back rather than resize */
  q->nthreads=E; q->alive=1; pthread_barrier_init(&q->bar,NULL,E+1);
  for(int t=0;t<E;t++){ g_mgsub_arg[b][t].q=q; g_mgsub_arg[b][t].tid=t;
    pthread_create(&g_mgsub_th[b][t],NULL,mgsub_worker2,(void*)&g_mgsub_arg[b][t]); }
  return q;
}
/* ---- bf16-storage wide worker (see the kernel block's header comment near k_d2bf) ---------------
   Requires dcOK (device-direct columns + pinned term plane: the gate fold and the resident logp/val
   writes depend on them). Per step: pinned-DMA obs H2D → ONE pre-captured graph (cast → enc GEMM →
   L gate layers → head GEMM → sampler; per-t offsets BAKED, per-update rng via device scalar) → 8·nb
   action D2H → sync → env sub-pool step → pinned rew/term plane memcpys. */
typedef struct { __nv_bfloat16 *dP,*wHead,*bHead,*dX,*dH1,*dHn,*dY3,*dLg,*dSt; unsigned char* dU8;
  unsigned long long* rng; double* dTm; volatile unsigned* flag; } wbf_bufs_t;   /* flag: pinned
  per-buffer completion stamps (64B apart), device-written via UVA */   /* dTm: DEVICE terminals plane slice — the gate
  used to read termPlane (pinned HOST) straight from the kernel: 2048 zero-copy PCIe doubles per launch
  made k_mingru_gate_bfw 8.1us vs PufferLib's 1.9us equivalent (nsys, 2026-08-03). A 16KB async H2D
  inside the captured graph replaces the host reads. */
static wbf_bufs_t g_wbf;
static cudaGraphExec_t* g_wbfg=NULL; static signed char* g_wbfg_ok=NULL; static long g_wbfg_cap=0;
static unsigned long long g_wbfg_key=0;
/* ---- FUSED-kernel per-(t,buf) graph table (the synthesis the fused path never had: its per-step rng
   ARG blocked graph capture; the device-scalar + baked-tOff pattern removes that, so each (t,buf)'s
   single fused launch + action D2H captures ONCE and replays for the rest of the run — killing the
   per-step launch/enqueue latency chain that profiling (spin-flag neutrality, sync-dominated buffer
   cycles) identified as the residual rollout wall. BIT-IDENTICAL: *rngDev + tOff == the eager seed. */
static cudaGraphExec_t* g_fgg=NULL; static signed char* g_fgg_ok=NULL; static long g_fgg_cap=0;
static unsigned long long g_fgg_key=0; static unsigned long long* g_fgrng=NULL;
/* per-(s,b) GPU chain for the wide-bf16 arm -- ONE function shared by the worker's eager/lazy
   path and the driver's setup-time PRE-CAPTURE, so a pre-captured graph is identical to what the
   worker would have recorded lazily. inclObs: the u8 obs H2D rides INSIDE the graph -- the source
   pf[(startPc+s)&1] is parity-stable because the chain stamp advances startPc by T per update and
   even T keeps it constant (gated on T%2==0). actPin: the sampler writes actions ZERO-COPY into a
   pinned flat array the envs read directly -- no D2H node, no host repack loop. */
/* ONE-CHANNEL mode (default ON): both wide buffers share stream 0. This GPU timeslices ACTIVE
   channels (measured: dual-stream graph rounds 63.7us each vs 46us avg on one shared stream; in-app
   the queue delay was ~95us/launch with 2 buffer streams + pool channels alive) — the classic
   overlap intuition INVERTS when the channel scheduler round-robins. PUFFER_MG_ONESTREAM=0 restores
   per-buffer streams. */
static int wbf_onestream(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_ONESTREAM");
  f=(e!=NULL&&e[0]=='1')?1:0; } return f; }   /* default OFF: in-app the shared channel made the
  buffers wait behind each other's copies (queue 95->119us/step) despite the microbenchmark win */
static inline cudaStream_t wbf_stream(int b){ return g_mgbufst[wbf_onestream()?0:b]; }
static void wbf_issue_chain(mgbufpool_t* p, int b, long s, int inclObs){
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  long D=p->D,T=p->T,N=p->N,LH=p->LH; int H=p->H,L=p->L,A=p->A,B=256;
  long rb=p->rowBase[b], nb=p->rowN[b];
  cudaStream_t st=wbf_stream(b); cublasHandle_t h=g_mgbufh[b]; cublasSetStream(h,st);

  int okind=p->obsKind, O=A+1, Hc=H;
  size_t wEncSz=(size_t)H*D;
  __nv_bfloat16 *wEnc=g_wbf.dP, *bEnc=g_wbf.dP+wEncSz, *layers=g_wbf.dP+wEncSz+H;
  #define WMG(x) ceildiv((long)(x),B)
  if(g_wbf.flag) k_stamp32<<<1,32,0,st>>>(g_wbf.flag+16*b+8, (unsigned)(s+1));   /* HEAD stamp (queue-
     delay vs exec-span attribution under PUFFER_ROLL_PROFILE) */
  /* 5-NODE graph (was 9): [prologue mega] -> [enc GEMM beta=1] -> [layer GEMM] -> [gate] ->
     [sampler w/ inline heads]. Per-node overhead in this process is ~6us unprofiled — node count is
     the currency. The obs H2D stays on the copy engine (bulk; a zc kernel read measured +50us/step),
     ISSUED by the worker post-env-step so its engine handoff runs off the launch->complete path. */
  if(inclObs){
    k_wbf_prologue<<<WMG(nb*D+nb*H+nb),B,0,st>>>(g_wbf.dX+rb*D, g_wbf.dU8+rb*D,
      p->dMGTraj, rb, nb, (long)s, T, D,
      g_wbf.dH1+rb*H, bEnc, Hc,
      g_wbf.dTm+rb, (s>0)? p->termPlane+((long)s-1)*N+rb : NULL);
    gemm_bfx_b1(h,CUBLAS_OP_T,CUBLAS_OP_N,Hc,(int)nb,(int)D, wEnc,(int)D, g_wbf.dX+rb*D,(int)D, g_wbf.dH1+rb*H,Hc);
  } else {
    if(okind==1) k_u82bf<<<WMG(nb*D),B,0,st>>>(g_wbf.dX+rb*D, g_wbf.dU8+rb*D, nb*D);
    else         k_f2bf_n<<<WMG(nb*D),B,0,st>>>(g_wbf.dX+rb*D, p->dObsF+rb*D, nb*D);
    if(p->dMGTraj) k_scatter_mg_obs_bfw<<<WMG(nb*D),B,0,st>>>(p->dMGTraj, g_wbf.dX+rb*D, rb, nb, (long)s, T, D);
    gemm_bfx(h,CUBLAS_OP_T,CUBLAS_OP_N,Hc,(int)nb,(int)D, wEnc,(int)D, g_wbf.dX+rb*D,(int)D, g_wbf.dH1+rb*H,Hc);
    k_add_bias_bfw<<<WMG(nb*H),B,0,st>>>(g_wbf.dH1+rb*H, bEnc, nb, Hc);
  }
  {
    __nv_bfloat16 *hb=g_wbf.dH1+rb*H, *hn=g_wbf.dHn+rb*H;
    /* terminals come from DEVICE (dTm): the gate used to zero-copy-read the pinned host plane per
       row -- 8.1us vs 1.9us for PufferLib's device-read equivalent. The 16KB H2D is issued OUTSIDE
       the graph, right after the env step while the stream is idle (an in-graph copy node was tried
       first and serialized a PCIe round trip ahead of the whole chain). */
    const double* tg=(s==0)? NULL : g_wbf.dTm+rb;
    for(int l=0;l<L;l++){
      __nv_bfloat16* Wl=layers+(size_t)l*3*H*H;
      gemm_bfx(h,CUBLAS_OP_T,CUBLAS_OP_N,3*Hc,(int)nb,Hc, Wl,Hc, hb,Hc, g_wbf.dY3+rb*3*H,3*Hc);
      k_mingru_gate_bfw<<<WMG(nb*H),B,0,st>>>(g_wbf.dY3+rb*3*H, hb, g_wbf.dSt+rb*LH, hn, tg, (int)nb, L, Hc, l);
      __nv_bfloat16* t2=hb; hb=hn; hn=t2;
    }
    gemm_bfx(h,CUBLAS_OP_T,CUBLAS_OP_N,O,(int)nb,Hc, g_wbf.wHead,Hc, hb,Hc, g_wbf.dLg+rb*O,O);
    unsigned long long tOff=(unsigned long long)((uint64_t)((long)s*N)*G);
    k_sample_wbf<<<WMG(nb),B,0,st>>>(g_wbf.dLg+rb*O, g_wbf.bHead,
      p->actPin? p->actPin+rb : p->dO+3*rb, (int)nb, rb, A, O, g_wbf.rng, tOff,
      g_dcAct, g_dcLogp, g_dcVal0, g_dcValue, rb, T, (long)s);
  }
  if(!p->actPin) cudaMemcpyAsync(p->hSamp+3*rb, p->dO+3*rb, 8*(size_t)nb, cudaMemcpyDeviceToHost, st);
  if(g_wbf.flag) k_stamp32<<<1,32,0,st>>>(g_wbf.flag+16*b, (unsigned)(s+1));
  #undef WMG
}
static void mg_wide_bf_body(int b){
  mgbufpool_t* p=&g_mgbp;
  long D=p->D,T=p->T,N=p->N,LH=p->LH; int L=p->L; (void)L;
  long rb=p->rowBase[b], nb=p->rowN[b]; int elo=p->envLo[b], ehi=p->envHi[b];
  int E=1; { int tot=mg_env_threads_total(); if(tot<=0) tot=16;   /* wide buffers NEED env fan-out */
    if(tot>p->nbuf) E=tot/p->nbuf; if(E<1) E=1; }
  cudaStream_t st=wbf_stream(b);
  int okind=p->obsKind, f32o=p->f32obs;
  float* pf[2]={p->hObsF32,p->hObsF32B}; int pc=p->startPc;
  const double* cur=p->obs0; double* nxt=p->hA;
  int inclObs=(okind==1 && (T%2)==0);            /* obs H2D rides inside the graph (parity-stable) */
  double* acts = p->actPin? p->actPin : p->actRM;
  int rprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL); double rp0=0; double* PR=p->prof[b];
  #define WRPT(k) do{ if(rprof){ double t=now_ms(); PR[k]+=t-rp0; rp0=t; } }while(0)
  if(rprof) rp0=now_ms();
  if(f32o && !p->skipPre){                         /* s=0 staging (mirrors mg_buf_worker's pre-loop) */
    const double* cs=p->obs0+rb*D; long ne=(long)nb*D;
    if(okind==1){ unsigned char* od=(unsigned char*)pf[pc]+(size_t)rb*D; for(long i=0;i<ne;i++) od[i]=(unsigned char)cs[i]; }
    else { float* of=pf[pc]+rb*D; for(long i=0;i<ne;i++) of[i]=(float)cs[i]; }
  }
  #define WMG(x) ceildiv((long)(x),256)
  for(size_t s=0;s<(size_t)T;s++){
    if(okind==1){ if(!inclObs || s==0) cudaMemcpyAsync(g_wbf.dU8+rb*D, ((const unsigned char*)pf[pc])+(size_t)rb*D, (size_t)nb*D, cudaMemcpyHostToDevice, st); }
    else {
      if(!f32o){ float* of=p->hObsF32+rb*D; const double* cs=cur+rb*D; long ne=(long)nb*D; for(long i=0;i<ne;i++) of[i]=(float)cs[i]; }
      cudaMemcpyAsync(p->dObsF+rb*D, (f32o?pf[pc]:p->hObsF32)+rb*D, 4*(size_t)nb*D, cudaMemcpyHostToDevice, st);
    }
    WRPT(0);                                     /* obs stage/issue (empty when in-graph) */
    long gi=(long)s*p->nbuf+b;
    if(g_wbfg && g_wbfg_ok && gi<g_wbfg_cap && g_wbfg_ok[gi]>=0){
      if(g_wbfg_ok[gi]==1) cudaGraphLaunch(g_wbfg[gi],st);
      else {
        static pthread_mutex_t capMu=PTHREAD_MUTEX_INITIALIZER;   /* shared-stream mode: two lazy
          captures on ONE stream would collide; serialize them (cold path — setup pre-captures) */
        pthread_mutex_lock(&capMu);
        cudaGraph_t gr=NULL; cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal);
        wbf_issue_chain(p,b,(long)s,inclObs);
        cudaError_t ce=cudaStreamEndCapture(st,&gr);
        if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_wbfg[gi],gr,0)==cudaSuccess){
          g_wbfg_ok[gi]=1; cudaGraphDestroy(gr); cudaGraphLaunch(g_wbfg[gi],st); }
        else { g_wbfg_ok[gi]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError(); wbf_issue_chain(p,b,(long)s,inclObs); }
        pthread_mutex_unlock(&capMu);
      }
    } else wbf_issue_chain(p,b,(long)s,inclObs);
    WRPT(1);                                     /* launch/enqueue */
    if(g_wbf.flag){ volatile unsigned* fl=g_wbf.flag+16*b;
      if(rprof){ volatile unsigned* fh=g_wbf.flag+16*b+8;
        while(*fh!=(unsigned)(s+1)) __builtin_ia32_pause();
        { double t=now_ms(); PR[7]+=t-rp0; }   /* queue delay: launch -> head stamp */
      }
      while(*fl!=(unsigned)(s+1)) __builtin_ia32_pause(); }
    else cudaStreamSynchronize(st);
    WRPT(2);                                     /* completion wait (flag spin / sync fallback) */
    if(!p->actPin) for(long i=0;i<nb;i++) p->actRM[rb+i]=p->hSamp[3*rb+i];
    WRPT(3);                                     /* act repack (empty with actPin) */
    mgsub_t* sub=(E>1)? mgsub_for(b,E) : NULL;
    if(sub){
      sub->p=p; sub->b=b; sub->actRM=acts; sub->s=s; sub->nb=nb; sub->elo=elo; sub->ehi=ehi;
      sub->pfOut=(void*)pf[1-pc]; sub->nxt=nxt;
      pthread_barrier_wait(&sub->bar); pthread_barrier_wait(&sub->bar);
    } else {
      int nAg=p->nAg;
      if(okind==1)   p->eh->step_range_u8 (p->eh->env, acts, (unsigned char*)pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
      else if(f32o)  p->eh->step_range_f32(p->eh->env, acts, pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
      else           p->eh->step_range    (p->eh->env, acts, nxt, p->hRT, p->hRT+N, elo, ehi-elo);
      if(p->rewCol) for(long e=(long)elo*nAg; e<(long)ehi*nAg; e++){ long row=e*T+(long)s;
        p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
    }
    WRPT(4);                                     /* env step */
    memcpy(p->rewPlane +(long)s*N+rb, p->hRT+rb,   8*(size_t)nb);
    memcpy(p->termPlane+(long)s*N+rb, p->hRT+N+rb, 8*(size_t)nb);
    /* next step's obs H2D issued NOW (post-env): the copy engine works during host bookkeeping and
       the next launch API, off the graph's launch->complete path. Terms ride the next graph as the
       zc kernel; the non-inclObs fallback keeps the explicit post-env terms H2D. */
    if(inclObs && s+1<(size_t)T) cudaMemcpyAsync(g_wbf.dU8+rb*D, ((const unsigned char*)pf[1-pc])+(size_t)rb*D, (size_t)nb*D, cudaMemcpyHostToDevice, st);
    if(!inclObs) cudaMemcpyAsync(g_wbf.dTm+rb, p->termPlane+(long)s*N+rb, 8*(size_t)nb, cudaMemcpyHostToDevice, st);
    if(f32o) pc^=1; else { cur=nxt; nxt=(nxt==p->hA)?p->hB:p->hA; }
    WRPT(5);                                     /* rew/term planes (+fallback terms H2D issue) */
  }
  if(inclObs) k_h2d_f64<<<WMG(nb),256,0,st>>>(g_wbf.dTm+rb, p->termPlane+((long)T-1)*N+rb, nb);
  k_mg_reset_bfw<<<WMG(nb*LH),256,0,st>>>(g_wbf.dSt+rb*LH, g_wbf.dTm+rb, (int)nb, (int)LH);
  WRPT(6);                                       /* tail */
  #undef WRPT
  #undef WMG
}
/* ---- MULTI-DISCRETE buffer worker: the K-head twin of mg_buf_worker's FUSED arm ------------------
   Same cycle, same machinery: pinned zero-copy obs staging → ONE k_mg_fused_step_w_md launch (obs
   traj scatter + encoder + L gate layers with the folded terminal reset + heads + per-head sample +
   the folded resident-column writes) → ONE K-wide action D2H → threaded CPU env-step (optionally
   fanned out through the per-buffer mgsub pool) → pinned rew/term planes; and the per-(t,buf)
   CUDA-graph table (g_fgg) captures kernel+D2H once and replays them for the rest of the run, with
   the per-update rng entering through the g_fgrng device scalar and the per-step offset baked.
   DELIBERATE RESTRICTIONS (the driver enforces them and falls back to the non-fused MD rollout
   otherwise, so nothing here can silently corrupt memory):
     · WMMA variant only. The plain/cp.async fused layouts stage the whole 3H×H layer matrix in
       shared (4·3·128·129 B ≈ 198KB at H=128) and do NOT fit at the H every MD env trains at;
       PUFFER_MG_WMMA=0 therefore drops MD back to the non-fused arm rather than to a layout that
       cannot exist.
     · dcOK required — act/logp/val land directly in the resident device columns, so ONLY the actions
       cross back, and they land row-major in PINNED actPin exactly as step_range_*'s act[e·K+h].
   Buffer count / slicing / rng are all row-global, so like the single-discrete arm the result does
   not depend on nbuf. */
static void mg_buf_worker_md(int b){
  mgbufpool_t* p=&g_mgbp;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  long D=p->D,T=p->T,N=p->N,LH=p->LH; int H=p->H,L=p->L,W=p->A,K=p->K,nAg=p->nAg,B=256;
  long rb=p->rowBase[b], nb=p->rowN[b]; int elo=p->envLo[b], ehi=p->envHi[b];
  if(nb<=0) return;
  int E=1; { int tot=mg_env_threads_total(); if(tot>p->nbuf) E=tot/p->nbuf; if(E<1) E=1; }
  cudaStream_t st=g_mgbufst[b];
  float* dSt=p->dSa + rb*LH;                                     /* this buffer's IN-PLACE state slice */
  const double* cur=p->obs0; double* nxt=p->hA;
  float* pf[2]={p->hObsF32, p->hObsF32B}; int pc=p->startPc, f32o=p->f32obs;
  int okind=p->obsKind; size_t oesz=(okind==1)?1:(okind==2)?2:4;
  if(f32o && !p->skipPre){                                       /* s=0 staging (as the SD worker) */
    const double* cs=p->obs0+rb*D; long ne=(long)nb*D;
    if(okind==1){ unsigned char* od=(unsigned char*)pf[pc]+(size_t)rb*D; for(long i=0;i<ne;i++) od[i]=(unsigned char)cs[i]; }
    else if(okind==2){ __nv_bfloat16* od=(__nv_bfloat16*)pf[pc]+(size_t)rb*D; for(long i=0;i<ne;i++) od[i]=__float2bfloat16((float)cs[i]); }
    else { float* of=pf[pc]+rb*D; for(long i=0;i<ne;i++) of[i]=(float)cs[i]; }
  }
  size_t shbW=mg_wmma_shbytes((int)D,H,L,W);                     /* nonzero + opted-in (driver checked) */
  int zc=(mg_zcobs() && mg_hb_pinned(4) && (!f32o || mg_hb_pinned(5)));
  double* actH=p->actPin;                                        /* pinned N·K row-major action plane */
  int rprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL); double rp0=0; double* PR=p->prof[b];
  #define RPT(k) do{ if(rprof){ double t=now_ms(); PR[k]+=t-rp0; rp0=t; } }while(0)
  #define MG(x) ceildiv((long)(x),B)
  for(size_t s=0;s<(size_t)T;s++){
    unsigned long long seed=(unsigned long long)(p->rolloutRng+(uint64_t)((long)s*N)*G);
    if(rprof) rp0=now_ms();
    if(!f32o){ float* of=p->hObsF32+rb*D; const double* cs=cur+rb*D; long ne=(long)nb*D;
      for(long i=0;i<ne;i++) of[i]=(float)cs[i]; }
    RPT(0);                                                      /* cast */
    const void* obsHost=(const void*)(((const char*)(f32o?pf[pc]:p->hObsF32))+(size_t)rb*D*oesz);
    if(!zc) cudaMemcpyAsync(p->dObsF+rb*D, obsHost, 4*(size_t)nb*D, cudaMemcpyHostToDevice, st);
    const double* tg=(s==0)? NULL : (p->termPlane+((long)s-1)*N+rb);   /* zero-copy pinned term plane */
    double* sampOut=p->dO+(long)rb*(K+2);   /* per-buffer block: [act(nb·K row-major)|logp(nb)|val(nb)] */
    const unsigned long long* fgDev=NULL; unsigned long long fgOff=0;
    int fgLaunched=0;
    auto fusedL=[&](void){
      k_mg_fused_step_w_md<<<(int)((nb+15)/16),512,shbW,st>>>(p->dP, zc? obsHost : (const void*)(p->dObsF+rb*D), okind, 16,
        p->dMGTraj, rb, (long)s, T, dSt, (long)LH, tg,
        sampOut, (long)nb, rb, seed,
        g_dcAct, g_dcLogp, g_dcVal0, g_dcValue,
        NULL, (int)nb, (int)D, H, L, W, K, p->dHs, fgDev, fgOff);
    };
    long fgi=(long)s*p->nbuf+b;
    int useFG=(g_fgg && g_fgg_ok && g_fgrng && fgi<g_fgg_cap && g_fgg_ok[fgi]>=0);
    if(useFG){
      if(g_fgg_ok[fgi]==1){ cudaGraphLaunch(g_fgg[fgi],st); fgLaunched=1; }
      else {
        fgDev=g_fgrng; fgOff=(unsigned long long)((uint64_t)((long)s*N)*G);
        cudaGraph_t gr=NULL; cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal);
        fusedL();
        cudaMemcpyAsync(actH+rb*K, sampOut, 8*(size_t)nb*K, cudaMemcpyDeviceToHost, st);
        cudaError_t ce=cudaStreamEndCapture(st,&gr);
        if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_fgg[fgi],gr,0)==cudaSuccess){
          g_fgg_ok[fgi]=1; cudaGraphDestroy(gr); cudaGraphLaunch(g_fgg[fgi],st); fgLaunched=1; }
        else { g_fgg_ok[fgi]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError();
          fgDev=NULL; fgOff=0; fusedL(); }
      }
    } else fusedL();
    if(!fgLaunched) cudaMemcpyAsync(actH+rb*K, sampOut, 8*(size_t)nb*K, cudaMemcpyDeviceToHost, st);
    RPT(1);                                                      /* launch+enqueue */
    cudaStreamSynchronize(st);                                   /* wait for THIS buffer only */
    RPT(2); RPT(3);                                              /* sync (no host act repack: D2H IS row-major) */
    mgsub_t* sub=(E>1)? mgsub_for(b,E) : NULL;
    if(sub){
      sub->p=p; sub->b=b; sub->actRM=actH; sub->s=s; sub->nb=nb; sub->elo=elo; sub->ehi=ehi;
      sub->pfOut=(void*)pf[1-pc]; sub->nxt=nxt;
      pthread_barrier_wait(&sub->bar); pthread_barrier_wait(&sub->bar);
      RPT(4); RPT(5);
    } else {
      if(okind==1)      p->eh->step_range_u8  (p->eh->env, actH, (unsigned char*) pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
      else if(okind==2) p->eh->step_range_bf16(p->eh->env, actH, (unsigned short*)pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
      else if(f32o)     p->eh->step_range_f32 (p->eh->env, actH, pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
      else              p->eh->step_range     (p->eh->env, actH, nxt, p->hRT, p->hRT+N, elo, ehi-elo);
      RPT(4);
      if(p->rewCol) for(long e=(long)elo*nAg; e<(long)ehi*nAg; e++){ long row=e*T+(long)s;
        p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
      RPT(5);
    }
    memcpy(p->rewPlane +(long)s*N+rb, p->hRT+rb,   8*(size_t)nb);
    memcpy(p->termPlane+(long)s*N+rb, p->hRT+N+rb, 8*(size_t)nb);
    if(f32o) pc^=1; else { cur=nxt; nxt=(nxt==p->hA)?p->hB:p->hA; }
    RPT(6);                                                      /* tail */
  }
  /* fused mode folded the per-step reset into the state READ — apply the LAST step's terms once so the
     state BUFFER matches the old per-step semantics for the bootstrap/finalState. */
  k_mg_reset<<<MG(nb*LH),B,0,st>>>(dSt, p->termPlane+((long)T-1)*N+rb, (int)nb, (int)LH, 0);
  #undef RPT
  #undef MG
}
static void* mg_buf_worker(void* arg){
  int b=(int)(long)arg; mgbufpool_t* p=&g_mgbp;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  long D=p->D,T=p->T,N=p->N,LH=p->LH; int H=p->H,L=p->L,A=p->A,O=p->O,bf=p->bf,nAg=p->nAg,B=256;
  long rb=p->rowBase[b], nb=p->rowN[b]; int elo=p->envLo[b], ehi=p->envHi[b];
  if(nb<=0) return NULL;
  if(p->md){ mg_buf_worker_md(b); return NULL; }                  /* K-head arm (md=0 single-discrete) */
  if(p->wideBf){ mg_wide_bf_body(b); return NULL; }               /* bf16-storage wide arm (gated) */
  int E=1; { int tot=mg_env_threads_total(); if(tot>p->nbuf) E=tot/p->nbuf; if(E<1) E=1; }
  cudaStream_t st=g_mgbufst[b]; cublasHandle_t h=g_mgbufh[b]; cublasSetStream(h,st);
  float* dSt=p->dSa + rb*LH;                                     /* this buffer's IN-PLACE recurrent state slice */
  const double* cur=p->obs0; double* nxt=p->hA;
  /* f32-obs mode (plugin exports step_range_f32): the env writes f32 obs DIRECTLY into pinned ping-pong
     staging — the f64 intermediate + per-step cast pass vanish from the critical path. BIT-IDENTICAL:
     envs compute obs in float/uint8, so (float)native == (float)(double)native. Step 0 still casts the
     f64 obs0 from Lean (once). pf[pc] = current step's obs; env writes pf[1-pc]. */
  float* pf[2]={p->hObsF32, p->hObsF32B}; int pc=p->startPc, f32o=p->f32obs;
  int okind=p->obsKind; size_t oesz=(okind==1)?1:(okind==2)?2:4;   /* transport element size */
  if(f32o && !p->skipPre){
    const double* cs=p->obs0+rb*D; long ne=(long)nb*D;
    if(okind==1){ unsigned char* od=(unsigned char*)pf[pc]+(size_t)rb*D; for(long i=0;i<ne;i++) od[i]=(unsigned char)cs[i]; }
    else if(okind==2){ __nv_bfloat16* od=(__nv_bfloat16*)pf[pc]+(size_t)rb*D; for(long i=0;i<ne;i++) od[i]=__float2bfloat16((float)cs[i]); }
    else { float* of=pf[pc]+rb*D; for(long i=0;i<ne;i++) of[i]=(float)cs[i]; }
  }
  size_t shbF=mg_fused()? mg_fused_shbytes((int)D,H,L,A) : 0;
  int plainM=(shbF!=0 && mg_fused_optin(shbF));
  int fusedM=plainM;   /* extended below: the WMMA variant can fit where the plain layout cannot (big H) */
  int bd=(8*H<=1024)?8*H:4*H;                    /* 8 rows/pass (one sync-pass per layer) when H≤128 */
  size_t shbP=(fusedM && mg_pipe() && bd==8*H)? mg_pipe_shbytes((int)D,H,L,A,MG_FR) : 0;
  int pipeM=(shbP!=0 && mg_pipe_optin(shbP));    /* cp.async pipelined variant (bit-identical) */
  size_t shbW=(mg_fused() && mg_wmma())? mg_wmma_shbytes((int)D,H,L,A) : 0;
  int wmmaM=(shbW!=0 && mg_wmma_optin(shbW));    /* tf32 tensor-core variant (tolerance-class) */
  fusedM=(plainM || wmmaM);   /* the non-wmma launch arm is reached only when plainM (wmmaM checked first) */
  /* zero-copy obs (fused mode): kernel reads the pinned host staging directly — no separate H2D hop */
  int zc=(fusedM && mg_zcobs() && mg_hb_pinned(4) && (!f32o || mg_hb_pinned(5)));
  int rprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL); double rp0=0; double* PR=p->prof[b];
  #define RPT(k) do{ if(rprof){ double t=now_ms(); PR[k]+=t-rp0; rp0=t; } }while(0)
  #define MG(x) ceildiv((long)(x),B)
  for(size_t s=0;s<(size_t)T;s++){
    unsigned long long seed=(unsigned long long)(p->rolloutRng+(uint64_t)((long)s*N)*G);
    if(rprof) rp0=now_ms();
    if(!f32o){
      /* fallback: cast this buffer's obs f64→f32 on the worker thread (overlaps other buffers' GPU) then
         H2D the f32 — halves the PCIe traffic vs f64. Same (float)cur value as the f32-obs mode. */
      float* of=p->hObsF32+rb*D; const double* cs=cur+rb*D; long ne=(long)nb*D; for(long i=0;i<ne;i++) of[i]=(float)cs[i];
    }
    RPT(0);                                                     /* cast */
    const void* obsHost=(const void*)(((const char*)(f32o?pf[pc]:p->hObsF32))+(size_t)rb*D*oesz);
    if(!zc) cudaMemcpyAsync(p->dObsF+rb*D, obsHost, 4*(size_t)nb*D, cudaMemcpyHostToDevice, st);   /* okind==0 only (kinds require zc) */
    int fgLaunched=0;
    if(fusedM){
      /* ONE launch: traj scatter + encoder + gates(+folded terminal reset) + heads + sample
         (+folded resident-column writes). The terms gate reads the packed [rew;term] staging in dcOK
         mode: the pinned term PLANE, zero-copy — same doubles. (An explicit pre-kernel H2D of the
         terms was tried 2026-08-03 and REVERTED: it serialized a PCIe round trip ahead of every
         launch, -7%, while the in-kernel zero-copy read is overlapped by the other warps' staging.)
         Per-(t,buf) graph replay when the table exists (see g_fgg): the eager path passes
         fgDev=NULL and the seed as an arg. */
      const double* tg = (s==0)? NULL : (p->dcOK? p->termPlane+((long)s-1)*N+rb : p->dTerms+rb);
      const unsigned long long* fgDev=NULL; unsigned long long fgOff=0;
      auto fusedL=[&](void){
      if(wmmaM)
        k_mg_fused_step_w<<<(int)((nb+15)/16),512,shbW,st>>>(p->dP, zc? obsHost : (const void*)(p->dObsF+rb*D), okind, 16,
          p->dMGTraj, rb, (long)s, T, dSt, (long)LH, tg,
          p->dO+3*rb, (long)nb, rb, seed,
          p->dcOK?g_dcAct:NULL, p->dcOK?g_dcLogp:NULL, p->dcOK?g_dcVal0:NULL, p->dcOK?g_dcValue:NULL,
          NULL, (int)nb, (int)D, H, L, A, fgDev, fgOff);
      else if(pipeM)
        k_mg_fused_step_p<<<(int)((nb+MG_FR-1)/MG_FR),bd,shbP,st>>>(p->dP, zc? obsHost : (const void*)(p->dObsF+rb*D), okind, MG_FR,
          p->dMGTraj, rb, (long)s, T, dSt, (long)LH, tg,
          p->dO+3*rb, (long)nb, rb, seed,
          p->dcOK?g_dcAct:NULL, p->dcOK?g_dcLogp:NULL, p->dcOK?g_dcVal0:NULL, p->dcOK?g_dcValue:NULL,
          NULL, (int)nb, (int)D, H, L, A, fgDev, fgOff);
      else
      k_mg_fused_step<<<(int)((nb+MG_FR-1)/MG_FR),bd,shbF,st>>>(p->dP, zc? obsHost : (const void*)(p->dObsF+rb*D), okind, MG_FR,
        p->dMGTraj, rb, (long)s, T, dSt, (long)LH, tg,
        p->dO+3*rb, (long)nb, rb, seed,
        p->dcOK?g_dcAct:NULL, p->dcOK?g_dcLogp:NULL, p->dcOK?g_dcVal0:NULL, p->dcOK?g_dcValue:NULL,
        NULL, (int)nb, (int)D, H, L, A, fgDev, fgOff);
      };
      long fgi=(long)s*p->nbuf+b;
      int useFG=(p->dcOK && !p->spin && g_fgg && g_fgg_ok && g_fgrng && fgi<g_fgg_cap && g_fgg_ok[fgi]>=0);
      if(useFG){
        if(g_fgg_ok[fgi]==1){ cudaGraphLaunch(g_fgg[fgi],st); fgLaunched=1; }
        else {
          fgDev=g_fgrng; fgOff=(unsigned long long)((uint64_t)((long)s*N)*G);
          cudaGraph_t gr=NULL; cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal);
          fusedL();
          cudaMemcpyAsync(p->hSamp+3*rb, p->dO+3*rb, 8*(size_t)(p->dcOK?1:3)*nb, cudaMemcpyDeviceToHost, st);
          cudaError_t ce=cudaStreamEndCapture(st,&gr);
          if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_fgg[fgi],gr,0)==cudaSuccess){
            g_fgg_ok[fgi]=1; cudaGraphDestroy(gr); cudaGraphLaunch(g_fgg[fgi],st); fgLaunched=1; }
          else { g_fgg_ok[fgi]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError();
            fgDev=NULL; fgOff=0; fusedL(); }
        }
      } else fusedL();
    } else {
      if(p->dMGTraj) k_scatter_mg_obs<<<MG(nb*D),B,0,st>>>(p->dMGTraj, p->dObsF+rb*D, rb, nb, (long)s, T, D);
      int sf32=mg_sampf32_host();
      auto gemmChain=[&](int devRng, unsigned long long seedArg){
        mingru_fwd_dev(h, p->dP, p->dObsF+rb*D, dSt, dSt, p->dHb+rb*H, p->dHn+rb*H, p->dYb+rb*3*H, p->dLg+rb*A, p->dVal+rb, (int)nb, (int)D, H, L, A, bf, st);
        if(sf32){                                                  /* f32 sampler tier (PUFFER_MG_SAMPF32, default) —
                                                                      aliases the f64 dY scratch as float (write-then-read, same stream) */
          float* yf=(float*)(p->dY)+rb*O;
          k_mingru_asm_f32<<<MG(nb*O),B,0,st>>>(p->dLg+rb*A, p->dVal+rb, yf, (int)nb, A, O);
          if(devRng) k_sample_seg_g_f32<<<MG(nb),B,0,st>>>(yf, p->dO+3*rb, (int)nb, (int)rb, A, O, g_mgg_rng[b]);
          else       k_sample_seg_f32<<<MG(nb),B,0,st>>>(yf, p->dO+3*rb, (int)nb, (int)rb, A, O, seedArg);
        } else {
          k_mingru_asm<<<MG(nb*O),B,0,st>>>(p->dLg+rb*A, p->dVal+rb, p->dY+rb*O, (int)nb, A, O);
          if(devRng) k_sample_seg_g<<<MG(nb),B,0,st>>>(p->dY+rb*O, p->dO+3*rb, (int)nb, (int)rb, A, O, g_mgg_rng[b]);
          else       k_sample_seg<<<MG(nb),B,0,st>>>(p->dY+rb*O, p->dO+3*rb, (int)nb, (int)rb, A, O, seedArg);
        }
      };
      long gkey=((long)nb<<32)^((long)D<<20)^((long)H<<12)^((long)L<<8)^(long)A^((long)sf32<<40)^((long)bf<<41);
      int wg=mg_wideg();
      if(wg && g_mgg_key[b]!=gkey && g_mgg_ok[b]==1){                /* shape/tier changed → invalidate, recapture */
        cudaGraphExecDestroy(g_mgg[b]); g_mgg_ok[b]=0; }
      if(wg && g_mgg_ok[b]>=0){
        if(!g_mgg_rng[b] && cudaMalloc((void**)&g_mgg_rng[b],sizeof(unsigned long long))!=cudaSuccess){ g_mgg_ok[b]=-1; gemmChain(0,seed); }
        else {
          k_set_u64<<<1,1,0,st>>>(g_mgg_rng[b], seed);               /* per-step rng into the device scalar */
          if(g_mgg_ok[b]==1){ cudaGraphLaunch(g_mgg[b], st); }
          else {                                                     /* capture once, then replay */
            cudaGraph_t gr=NULL; cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            gemmChain(1, 0);
            cudaError_t ce=cudaStreamEndCapture(st,&gr);
            if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_mgg[b],gr,0)==cudaSuccess){
              g_mgg_ok[b]=1; g_mgg_key[b]=gkey; cudaGraphDestroy(gr); cudaGraphLaunch(g_mgg[b], st); }
            else { g_mgg_ok[b]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError(); gemmChain(0, seed); }
          }
        }
      } else gemmChain(0, seed);
    }
    if(p->dcOK && !fusedM) k_mg_cols_alv<<<MG(nb),B,0,st>>>(p->dO+3*rb, g_dcAct,g_dcLogp,g_dcVal0,g_dcValue, rb, nb, (long)s, T);
    /* device-direct mode: only the ACTIONS cross back (8KB) — logp/val live in the resident columns and
       their host copies have no reader (Lean skips the slices when colsReady). The fused-graph path
       already contains this D2H as a captured node. */
    if(!fgLaunched) cudaMemcpyAsync(p->hSamp+3*rb, p->dO+3*rb, 8*(size_t)(p->dcOK?1:3)*nb, cudaMemcpyDeviceToHost, st);
    if(p->spin){
      /* stream-ordered stamp AFTER the D2H, then spin on the pinned flag — replaces the contended
         cudaStreamSynchronize driver call. Bounded: fall back to a real sync if never signalled. */
      k_mg_stamp<<<1,1,0,st>>>(p->flagH+b*16,(int)(s+1));
      RPT(1);                                                   /* launch+enqueue */
      volatile int* fl=(volatile int*)(p->flagH+b*16); long spins=0;
      while(*fl!=(int)(s+1)){
        __builtin_ia32_pause();
        if(((++spins)&0xFFFFF)==0 && spins>(1L<<28)){ cudaStreamSynchronize(st);
          fprintf(stderr,"[puffer] spin-flag timeout (buf %d step %zu) — synced\n",b,s); break; }
      }
      RPT(2);                                                   /* spin-wait */
    } else {
      RPT(1);                                                   /* launch+enqueue */
      cudaStreamSynchronize(st);                                /* wait for THIS buffer only (others run) */
      RPT(2);                                                   /* sync */
    }
    for(long i=0;i<nb;i++) p->actRM[rb+i]=p->hSamp[3*rb+i];      /* act → global row-major for step_range */
    RPT(3);                                                     /* act copy */
    mgsub_t* sub = (E>1)? mgsub_for(b,E) : NULL;
    if(sub){
      /* M1: env-step + host-tail scatter fan out to E persistent sub-workers over [elo,ehi) — same
         step_range_* call, same per-row scatter expressions as below, just partitioned finer. Disjoint
         absolute rows ⇒ bit-identical regardless of E. */
      sub->p=p; sub->b=b; sub->actRM=p->actRM; sub->s=s; sub->nb=nb; sub->elo=elo; sub->ehi=ehi;
      sub->pfOut = f32o? (void*)pf[1-pc] : (void*)pf[1-pc];   /* okind!=0 or f32o both write pf ping-pong */
      sub->nxt = nxt;
      pthread_barrier_wait(&sub->bar);                         /* release sub-workers */
      pthread_barrier_wait(&sub->bar);                         /* join */
      RPT(4); RPT(5);   /* env + scatter fused inside the sub-workers */
    } else {
    if(okind==1)      p->eh->step_range_u8  (p->eh->env, p->actRM, (unsigned char*) pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
    else if(okind==2) p->eh->step_range_bf16(p->eh->env, p->actRM, (unsigned short*)pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
    else if(f32o)     p->eh->step_range_f32 (p->eh->env, p->actRM, pf[1-pc], p->hRT, p->hRT+N, elo, ehi-elo);
    else              p->eh->step_range     (p->eh->env, p->actRM, nxt, p->hRT, p->hRT+N, elo, ehi-elo);
    RPT(4);                                                     /* env */
    if(p->dcOK){                                                /* host rew/term cols only (ep-return logging;
                                                                   NULL when the chained caller skipped logging) */
      if(p->rewCol) for(long e=(long)elo*nAg; e<(long)ehi*nAg; e++){ long row=e*T+(long)s;
        p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
    } else {
      for(long e=(long)elo*nAg; e<(long)ehi*nAg; e++){ long row=e*T+(long)s, le=e-rb;
        p->actCol[row]=p->hSamp[3*rb+le]; p->logpCol[row]=p->hSamp[3*rb+nb+le]; p->valCol[row]=p->hSamp[3*rb+2*nb+le];
        p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
    }
    RPT(5);                                                     /* host col scatter */
    }
    if(p->dcOK){
      /* rew/term → PINNED PLANES [T·N] (host memcpy only — NO per-step H2D, NO per-step scatter kernel:
         ONE per-update kernel scatters both columns after the workers join; the next step's folded reset
         gate reads the term plane zero-copy, staged once into shared). */
      memcpy(p->rewPlane +(long)s*N+rb, p->hRT+rb,   8*(size_t)nb);
      memcpy(p->termPlane+(long)s*N+rb, p->hRT+N+rb, 8*(size_t)nb);
      if(!fusedM) k_mg_reset<<<MG(nb*LH),B,0,st>>>(dSt, p->termPlane+(long)s*N+rb, (int)nb, (int)LH, 0);
    } else {
      cudaMemcpyAsync(p->dTerms+rb, p->hRT+N+rb, 8*(size_t)nb, cudaMemcpyHostToDevice, st);   /* feeds the next step's folded reset */
      if(!fusedM) k_mg_reset<<<MG(nb*LH),B,0,st>>>(dSt, p->dTerms+rb, (int)nb, (int)LH, 0);
    }
    if(f32o) pc^=1; else { cur=nxt; nxt=(nxt==p->hA)?p->hB:p->hA; }
    RPT(6);                                                     /* tail (rew/term H2D + col launches) */
  }
  #undef RPT
  /* fused mode folded the per-step reset into the state READ — the state BUFFER still holds unreset rows,
     so apply the LAST step's terms once for the bootstrap/finalState (matches the old per-step semantics). */
  if(fusedM) k_mg_reset<<<MG(nb*LH),B,0,st>>>(dSt, (p->dcOK? p->termPlane+((long)T-1)*N+rb : p->dTerms+rb), (int)nb, (int)LH, 0);
  #undef MG
  return NULL;
}

static lean_obj_res mg_contract_fail(const char* what, const char* detail, size_t P);   /* defined with the BPTT */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_plugin_rollout_mingru(
    size_t hh, size_t policyH, lean_obj_arg obs0a, lean_obj_arg state0a,
    size_t N, size_t D, size_t H, size_t L, size_t A, size_t T, uint8_t wantLog, uint64_t rolloutRng, lean_obj_arg w){
  double wcall0=0; int wcprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL); if(wcprof) wcall0=now_ms();
  { static double warmMs=-1; if(warmMs<0){ const char* e=getenv("PUFFER_GPU_WARM_MS"); warmMs=e?atof(e):25.0; }
    mg_warm_kick(warmMs); }
  #define WCSTAMP(var) double var=0; if(wcprof) var=now_ms();

  (void)w;
  Handle* eh=(Handle*)hh;
  size_t O=A+1, wEncSz=H*D, layerSz=3*H*H, P=wEncSz+H+L*layerSz+A*H+A+H+1;
  const uint64_t G=0x9E3779B97F4A7C15ULL; int bf=1;
  const double* obs0=lean_float_array_cptr(obs0a); const double* st0=lean_float_array_cptr(state0a);
  long NT=(long)N*T, LH=(long)L*H, cols=NT*((long)D+5)+(long)N*D+(long)N*LH+(long)N;
  /* resident chaining: EMPTY obs0+state0 ⇒ use the stamped resident obs/state; the return shrinks to
     [rewCol(NT);termCol(NT)] when wantLog else EMPTY. Requires a valid same-shape stamp (a prior
     chain-capable call this run) — a stale/absent stamp ABORTS (see below), never silent garbage. */
  int residentIn=(lean_sarray_size(obs0a)==0 && lean_sarray_size(state0a)==0);
  int chainStampOK=(g_mgchain.valid && g_mgchain.N==(long)N && g_mgchain.D==(long)D && g_mgchain.LH==LH);
  if(residentIn && !chainStampOK){
    /* Was a warn-once + EMPTY return, which is indistinguishable from a legitimate chained non-logging
       return: the caller then sliced an empty array (lean_ffi_slice memcpy's without bounds checks),
       fed the resulting garbage into mg_prep, and trained on it — reporting healthy updates/SPS and
       exiting 0. The stamp goes stale exactly when a rollout allocation fails mid-run (Lean latches
       `chained` once), i.e. under the same VRAM pressure as the zero-gradient bug. Abort instead. */
    lean_dec(obs0a); lean_dec(state0a);
    return mg_contract_fail("rollout",
      "resident-chain call (empty obs0/state0) without a valid same-shape chain stamp — a mid-run "
      "rollout allocation failure dropped the resident obs/state chain", P);
  }
  long colsAlloc = residentIn? (wantLog? 2*NT : 0) : cols;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),colsAlloc,colsAlloc); double* out=lean_float_array_cptr(Oo);
  double *obsCol,*actCol,*logpCol,*valCol,*rewCol,*termCol,*finalObs,*finalState,*bootVals;
  if(residentIn){
    obsCol=actCol=logpCol=valCol=finalObs=finalState=bootVals=NULL;
    rewCol = wantLog? out : NULL; termCol = wantLog? out+NT : NULL;
  } else {
    obsCol=out; actCol=out+NT*D; logpCol=actCol+NT; valCol=logpCol+NT; rewCol=valCol+NT; termCol=rewCol+NT;
    finalObs=termCol+NT; finalState=finalObs+(long)N*D; bootVals=finalState+(long)N*LH;
  }
  cublasHandle_t hbl=cu_handle();
  float *dP=(float*)rb2(0,4*P), *dSa=(float*)rb2(1,4*N*LH), *dSb=(float*)rb2(2,4*N*LH), *dObsF=(float*)rb2(3,4*N*D);
  float *dHb=(float*)rb2(4,4*N*H), *dHn=(float*)rb2(5,4*N*H), *dYb=(float*)rb2(6,4*N*3*H), *dLg=(float*)rb2(7,4*N*A), *dVal=(float*)rb2(8,4*N);
  double *dObs=(double*)rb2(9,8*N*D), *dY=(double*)rb2(10,8*N*O), *dO=(double*)rb2(11,8*3*N), *dTerms=(double*)rb2(12,8*N);
  float *hStF=(float*)malloc(4*N*LH);
  /* PINNED obs/sample/rew staging (the obs H2D is PCIe-bound at scale) */
  double *hSamp=(double*)mg_hb(0,8*3*N), *hA=(double*)mg_hb(1,8*N*D), *hB=(double*)mg_hb(2,8*N*D), *hRT=(double*)mg_hb(3,8*2*N), *hVal=(double*)malloc(8*N);
  int ok=(N>0 && hbl!=NULL && eh && policyH && dP&&dSa&&dSb&&dObsF&&dHb&&dHn&&dYb&&dLg&&dVal&&dObs&&dY&&dO&&dTerms&&hStF&&hSamp&&hA&&hB&&hRT&&hVal);
  if(ok){
    cublasSetStream(hbl,0); int B=256;
    #define GR(x) ceildiv((long)(x),B)
    k_d2f<<<GR(P),B>>>(dP,(const double*)policyH,(long)P);            /* weights RESIDENT (handle f64 → f32 dP), no upload */
    if(mg_wmma()){ mg_pub_wencpad(dP,(int)H,(int)D,(int)L,(int)A);   /* big-H WMMA encoder pad (helper
       re-pads from THIS call's dP and publishes; NULL-publishes on alloc failure) */
      mg_pub_wlbf(dP,(int)H,(int)D,(int)L); }                          /* bf16 forward tier (gated) */
    { static int pubSamp=-1;                                           /* f32 sampler tier (see d_mg_sample_row), default ON */
      static int want=-1; if(want<0){ const char* e=getenv("PUFFER_MG_SAMPF32"); want=(e==NULL||e[0]!='0')?1:0; }
      if(pubSamp!=want){ cudaMemcpyToSymbol(c_mgSampF32,&want,sizeof(int)); pubSamp=want; } }
    { static int pubFma=-1; int wantF=mg_wprec_bf()?1:0;                 /* FMA scalar sections: tolerance tier only */
      if(pubFma!=wantF){ cudaMemcpyToSymbol(c_mgFma,&wantF,sizeof(int)); pubFma=wantF; } }
    if(!residentIn){ for(long i=0;i<N*LH;i++) hStF[i]=(float)st0[i];
      cudaMemcpy(dSa,hStF,4*N*LH,cudaMemcpyHostToDevice); }   /* state resident; chained calls reuse dSa in place */
    if(mg_zero_roll_state()) cudaMemset(dSa,0,4*(size_t)N*LH);   /* PufferLib convention: restart each rollout at h=0
      (default ON; covers all single-discrete arms since every one reads state from dSa via dStCur below) */
    /* Persistent-thread env-step: workers split env-step + column-scatter while main drives the GPU
       (single-discrete ⇒ W=1). Serial fallback when the plugin predates puffer_env_step_range. */
    int threaded = (eh->step_range != NULL);
    int nAg = eh->numAgents>0 ? eh->numAgents : 1;
    if(threaded && !g_rp_init){
      g_rp.nthreads=rp_threads(); g_rp.alive=1;
      pthread_barrier_init(&g_rp.bar,NULL,g_rp.nthreads+1);
      for(long t=0;t<g_rp.nthreads;t++) pthread_create(&g_rp_th[t],NULL,rp_worker,(void*)t);
      g_rp_init=1;
    }
    float* dMGTraj=mgobstraj_buf((size_t)N*T*D); g_dMGObsTraj_valid=0;   /* device-resident obs (BPTT gathers it) */
    if(threaded){                                                          /* shape-invariant fields (per call) */
      g_rp.eh=eh; g_rp.obsCol=obsCol; g_rp.actCol=actCol; g_rp.logpCol=logpCol; g_rp.valCol=valCol;
      g_rp.rewCol=rewCol; g_rp.termCol=termCol; g_rp.N=(long)N; g_rp.D=(long)D; g_rp.T=(long)T;
      g_rp.nAgents=nAg; g_rp.envLo=0; g_rp.envHi=eh->N; g_rp.rowBase=0; g_rp.Nstride=(long)N; g_rp.W=1;
      g_rp.skipObs=(dMGTraj!=NULL);                                        /* obs device-resident ⇒ skip host obsCol scatter */
    }
    static int mgrprof=-1; if(mgrprof<0) mgrprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL);   /* phase attribution */
    static double MR_h2d=0,MR_gpu=0,MR_d2h=0,MR_env=0,MR_rst=0; double mrt0=0;
    float *dStCur=dSa, *dStNxt=dSb;
    const double* cur=obs0; double* nxt=hA; float* fObs32=NULL;   /* fObs32: final obs when in f32-obs mode */
    /* concurrent stream-buffers: nbuf workers run disjoint agent slices, env-step of one overlapping the GPU
       forward of another (in-place state in dSa; bootstrap below reads dSa via the unswapped dStCur). */
    /* bf16-storage wide arm pre-gate (shape + env var only; dcOK availability checked below).
       PUFFER_MG_WIDEBF: 1 force / 0 off / unset = auto at H>=128 (squared-class shapes — the regime
       where the fused kernel measured behind PufferLib's wide bf16 GEMM rollout; pong h32 / breakout
       h64 are untouched). Forces 2 wide buffers, PufferLib's own split. */
    int wideBf0=0, wbfFresh=0;
    { const char* e=getenv("PUFFER_MG_WIDEBF");
      wideBf0=(e? e[0]=='1' : (H>=128 && mg_wprec_bf())) && A<64 && threaded; }   /* AUTO at H>=128
        under the (default) bf16 tolerance tier ONLY — the wide arm is inherently bf16 storage, so
        PUFFER_MG_WPREC=f32 keeps the fused arm and its bit-exact anchor. (squared-class):
        after its 2026-08-04 rehabilitation (device terms, 6-node graph, setup pre-capture, zero-copy
        pinned actions, tier FMA) the wide arm measures ABOVE the fused arm at h128 (25.5M vs 24.6M
        inclusive @100M steps; steady ~26.4M vs their 28.7M). h<=64 keeps the fused arm (its 16-way
        co-residency wins there — PufferLib's own per-env tuning agrees: 8 buffers for breakout,
        2x2048 for squared). PUFFER_MG_WIDEBF=0/1 forces. */
    /* buffer count: the SAME adaptive rule for both arms (16 at N=4096). Measured @100M on squared,
       2 reps: nbuf 2=24.3 3=24.3 4=25.6 6=24.2 8=26.1 12=26.0 16=26.2-26.7M. Narrow-and-many wins
       even for the wide arm — its per-graph fixed cost (~6us) is small enough that 16-way
       env/GPU overlap dominates, so PufferLib's 2x2048 geometry is NOT the optimum in our stack. */
    int nbuf= mg_roll_buffers((long)N);
    /* nbuf==1 through the BUFFERED machinery (one whole-batch fused launch per step + env sub-pool
       fan-out) is allowed when explicitly requested — the adaptive default never returns 1 at N>=512,
       and small-N runs keep the legacy single-buffer path (different sampler tier) untouched. */
    int useBuf=((nbuf>1 || (nbuf==1 && getenv("PUFFER_MG_ROLL_BUFFERS")!=NULL)) && threaded && dMGTraj!=NULL);
    if(useBuf){
      /* spin-flag completion: the fused kernel writes actions to PINNED actRM (zero-copy) and stamps a
         pinned flag; workers spin instead of streamSync+D2H (2 fewer contended driver calls per step).
         The counter memset MUST complete before any worker kernel increments it (different streams). */
      int spin=0;
      { const char* e=getenv("PUFFER_MG_SPIN"); int want=(e!=NULL&&e[0]=='1');   /* default OFF: measured net-negative here */
        size_t shbM=mg_fused()? mg_fused_shbytes((int)D,(int)H,(int)L,(int)A) : 0;
        if(want && shbM && mg_fused_optin(shbM)){
          g_mgbp.flagH=(int*)mg_hb(6,64*MAXBUF_MG);
          if(g_mgbp.flagH && mg_hb_pinned(6)){ spin=1; memset(g_mgbp.flagH,0,64*MAXBUF_MG); }
        }
      }
      double* actRM=(double*)malloc(8*N);
      if(!actRM) useBuf=0;
      else {
        g_mgbp.spin=spin;
        mgbuf_init(nbuf);
        /* device-direct resident columns: allocate + init here so the workers scatter act/logp/val/rew/term
           straight into them (retires the Lean-side slice build + prep's 10.5MB pageable H2D). */
        g_mgbp.dcOK=0;
        g_mgbp.rewPlane =(double*)mg_hb(8,8*(size_t)N*T);   /* pinned [T·N] planes (worker memcpy targets) */
        g_mgbp.termPlane=(double*)mg_hb(9,8*(size_t)N*T);
        if(g_mgbp.rewPlane && g_mgbp.termPlane && mg_hb_pinned(8) && mg_hb_pinned(9) && dc_alloc((size_t)N*T)){
          g_dcN=(long)N; g_dcT=(long)T; g_dc_valid=0; g_dc_fromroll=0;
          k_fill1<<<GR((long)N*T),B>>>(g_dcRatio,(long)N*T);          /* ratioBuf := 1 (prep semantics) */
          g_mgbp.dcOK=1;
        }
        g_mgbp.eh=eh; g_mgbp.obs0=obs0; g_mgbp.dP=dP; g_mgbp.dMGTraj=dMGTraj;
        g_mgbp.dObs=dObs; g_mgbp.dY=dY; g_mgbp.dO=dO; g_mgbp.dTerms=dTerms;
        g_mgbp.dObsF=dObsF; g_mgbp.dSa=dSa; g_mgbp.dHb=dHb; g_mgbp.dHn=dHn; g_mgbp.dYb=dYb; g_mgbp.dLg=dLg; g_mgbp.dVal=dVal;
        g_mgbp.hSamp=hSamp; g_mgbp.hA=hA; g_mgbp.hB=hB; g_mgbp.hRT=hRT; g_mgbp.actRM=actRM;
        g_mgbp.hObsF32=(float*)mg_hb(4,4*N*D);   /* pinned f32 obs staging (workers cast into it → f32 H2D) */
        g_mgbp.hObsF32B=(float*)mg_hb(5,4*N*D);  /* ping-pong twin for the direct f32-obs env mode */
        g_mgbp.actPin=NULL;                      /* wide arm: pinned flat action array the sampler writes
          ZERO-COPY and the envs read directly (no D2H node, no host repack loop) */
        { double* ap=(double*)mg_hb(7,8*(size_t)N); if(ap && mg_hb_pinned(7)) g_mgbp.actPin=ap; }
        g_wbf.flag=NULL;
        { volatile unsigned* fp=(volatile unsigned*)mg_hb(11,64*MAXBUF_MG);
          if(fp && mg_hb_pinned(11)){ g_wbf.flag=fp;
            for(int b2=0;b2<MAXBUF_MG;b2++) fp[16*b2]=0; } }   /* stamps run 1..T each update (value
          baked per graph); zeroed here so update boundaries can never alias */
        { const char* e=getenv("PUFFER_MG_F32OBS");   /* gate: PUFFER_MG_F32OBS=0 forces the f64 obs path */
          g_mgbp.f32obs=(eh->step_range_f32!=NULL && g_mgbp.hObsF32 && g_mgbp.hObsF32B && !(e&&e[0]=='0')); }
        /* obs TRANSPORT kind: u8 for byte-native envs (EXACT widen ⇒ bit-identical, default ON; 4x fewer
           bytes), bf16 opt-in via PUFFER_MG_OBSPREC=bf16 (RNE-rounded, tolerance-class — PufferLib's own
           obs precision; 2x). PUFFER_MG_OBSPREC=f32 forces full precision. Kinds need the fused zero-copy
           pipeline (the kernel does the widen) on top of the f32obs ping-pong staging. */
        g_mgbp.obsKind=0;
        { const char* op=getenv("PUFFER_MG_OBSPREC");
          int forceF32=(op && op[0]=='f');
          int wantBf=(op && (op[0]=='b'||op[0]=='B'));
          size_t shbM=mg_fused()? mg_fused_shbytes((int)D,(int)H,(int)L,(int)A) : 0;
          size_t shbMW=(mg_fused() && mg_wmma())? mg_wmma_shbytes((int)D,(int)H,(int)L,(int)A) : 0;
          int zcM=(((shbM!=0 && mg_fused_optin(shbM)) || (shbMW!=0 && mg_wmma_optin(shbMW)))
                   && mg_zcobs() && mg_hb_pinned(4) && mg_hb_pinned(5));
          if(g_mgbp.f32obs && zcM && !forceF32){
            if(eh->obsKind==1 && eh->step_range_u8) g_mgbp.obsKind=1;   /* ANY D: the tile-linear aligned
              loader (d_mg_obs_u8_tile) killed the old D%4 restriction — the per-byte scalar zero-copy
              path that measured a large regression is no longer reachable. Squared (D=121) was falling
              all the way to the F32 WIRE on this gate: 4x the PCIe traffic, ~85% of its residual gap
              vs PufferLib (whose trainer ships u8 obs — source-level diff, 2026-08-03). */
            else if(wantBf && eh->step_range_bf16) g_mgbp.obsKind=2;
          } }
        /* bf16-storage wide arm setup: needs dcOK (gate fold + resident columns). u8 wire is forced on
           for u8 envs regardless of the fused zc gate (this arm H2Ds from pinned staging, no zero-copy). */
        g_mgbp.wideBf=0;
        if(wideBf0 && g_mgbp.dcOK){
          if(g_mgbp.f32obs && eh->obsKind==1 && eh->step_range_u8) g_mgbp.obsKind=1;
          g_wbf.dP=(__nv_bfloat16*)bg(90,2*P);
          __nv_bfloat16* hw=(__nv_bfloat16*)bg(91,2*((size_t)(A+1)*H+(size_t)(A+1)+8));
          g_wbf.wHead=hw; g_wbf.bHead=hw? hw+(size_t)(A+1)*H : NULL;
          g_wbf.dX =(__nv_bfloat16*)bg(92,2*(size_t)N*D);
          g_wbf.dH1=(__nv_bfloat16*)bg(93,2*(size_t)N*H);
          g_wbf.dHn=(__nv_bfloat16*)bg(94,2*(size_t)N*H);
          g_wbf.dY3=(__nv_bfloat16*)bg(95,2*(size_t)N*3*H);
          g_wbf.dLg=(__nv_bfloat16*)bg(96,2*(size_t)N*O);
          g_wbf.dSt=(__nv_bfloat16*)bg(97,2*(size_t)N*LH);
          g_wbf.dU8=(unsigned char*)bg(98,(size_t)N*D);
          g_wbf.rng=(unsigned long long*)bg(99,8);
          g_wbf.dTm=(double*)bg(102,8*(size_t)N);
          if(g_wbf.dP&&g_wbf.wHead&&g_wbf.dX&&g_wbf.dH1&&g_wbf.dHn&&g_wbf.dY3&&g_wbf.dLg&&g_wbf.dSt&&g_wbf.dU8&&g_wbf.rng&&g_wbf.dTm){
            size_t wEncSz2=(size_t)H*D, layerSz2=(size_t)3*H*H;
            __nv_bfloat16 *wDecB=g_wbf.dP+wEncSz2+H+(size_t)L*layerSz2, *bDecB=wDecB+(size_t)A*H,
                          *wValB=bDecB+A, *bValB=wValB+H;
            k_d2bf<<<GR(P),B>>>(g_wbf.dP,(const double*)policyH,(long)P);
            k_mg_pack_head_bf<<<GR((long)(A+1)*H),B>>>(g_wbf.wHead,g_wbf.bHead,wDecB,bDecB,wValB,bValB,(int)A,(int)H);
            k_f2bf_n<<<GR((long)N*LH),B>>>(g_wbf.dSt,dSa,(long)N*LH);
            /* rng upload ASYNC from pinned staging + device-side event ordering (the old sync
               cudaMemcpy blocked on the legacy default stream every update) */
            { unsigned long long* hs=(unsigned long long*)mg_hb(10,16);
              static cudaEvent_t rngEv=NULL;
              if(hs && mg_hb_pinned(10) && (rngEv || cudaEventCreateWithFlags(&rngEv,cudaEventDisableTiming)==cudaSuccess)){
                hs[0]=rolloutRng;
                cudaMemcpyAsync(g_wbf.rng,hs,8,cudaMemcpyHostToDevice,g_mgbufst[0]);
                cudaEventRecord(rngEv,g_mgbufst[0]);
                for(int b2=1;b2<nbuf;b2++) cudaStreamWaitEvent(g_mgbufst[b2],rngEv,0);
              } else cudaMemcpy(g_wbf.rng,&rolloutRng,8,cudaMemcpyHostToDevice); }
            /* per-(t,buf) graph table, keyed on shapes + every device/pinned address BAKED into the
               graphs (bg/rb2/mg_hb reallocs would silently invalidate captured pointers) */
            unsigned long long key=1469598103934665603ULL;
            #define KH(v) do{ key^=(unsigned long long)(v); key*=1099511628211ULL; }while(0)
            KH(N);KH(D);KH(H);KH(L);KH(A);KH(T);KH(nbuf);KH(g_mgbp.obsKind);KH(g_mgbp.f32obs);
            KH((uintptr_t)g_wbf.dP);KH((uintptr_t)g_wbf.wHead);KH((uintptr_t)g_wbf.dX);
            KH((uintptr_t)g_wbf.dH1);KH((uintptr_t)g_wbf.dY3);KH((uintptr_t)g_wbf.dLg);
            KH((uintptr_t)g_wbf.dSt);KH((uintptr_t)g_wbf.dU8);KH((uintptr_t)g_wbf.rng);KH((uintptr_t)g_wbf.dTm);
            KH((uintptr_t)g_mgbp.hObsF32);KH((uintptr_t)g_mgbp.hObsF32B);KH((uintptr_t)g_mgbp.actPin);
            KH((uintptr_t)g_wbf.flag);KH(wbf_onestream());
            KH((uintptr_t)dMGTraj);KH((uintptr_t)g_dcAct);KH((uintptr_t)dObsF);KH((uintptr_t)dO);
            KH((uintptr_t)g_mgbp.termPlane);
            #undef KH
            long need=(long)T*nbuf;
            if(key!=g_wbfg_key || need>g_wbfg_cap || !g_wbfg){
              if(g_wbfg){ for(long i=0;i<g_wbfg_cap;i++) if(g_wbfg_ok[i]==1) cudaGraphExecDestroy(g_wbfg[i]);
                free(g_wbfg); free(g_wbfg_ok); g_wbfg=NULL; g_wbfg_ok=NULL; g_wbfg_cap=0; }
              g_wbfg=(cudaGraphExec_t*)calloc((size_t)need,sizeof(cudaGraphExec_t));
              g_wbfg_ok=(signed char*)calloc((size_t)need,1);
              if(g_wbfg && g_wbfg_ok){ g_wbfg_cap=need; g_wbfg_key=key; wbfFresh=1; }
              else { free(g_wbfg); free(g_wbfg_ok); g_wbfg=NULL; g_wbfg_ok=NULL; g_wbfg_cap=0; }
            }
            g_mgbp.wideBf=1;
          }
        }
        /* fused-kernel graph table (per (t,buf)) + the per-update rollout-rng device scalar. Keyed on
           shapes + every pointer the captured launches bake (a cache realloc silently invalidates
           captured graphs). PUFFER_MG_FGRAPH=0 disables. */
        { int fgGate; { const char* e=getenv("PUFFER_MG_FGRAPH"); fgGate=(e==NULL||e[0]!='0'); }
          if(fgGate && g_mgbp.dcOK && !g_mgbp.wideBf){
            if(!g_fgrng) g_fgrng=(unsigned long long*)bg(100,8);
            if(g_fgrng){
              { unsigned long long* hs=(unsigned long long*)mg_hb(10,16);
                static cudaEvent_t fgEv=NULL;
                if(hs && mg_hb_pinned(10) && (fgEv || cudaEventCreateWithFlags(&fgEv,cudaEventDisableTiming)==cudaSuccess)){
                  hs[1]=rolloutRng;
                  cudaMemcpyAsync(g_fgrng,hs+1,8,cudaMemcpyHostToDevice,g_mgbufst[0]);
                  cudaEventRecord(fgEv,g_mgbufst[0]);
                  for(int b2=1;b2<nbuf;b2++) cudaStreamWaitEvent(g_mgbufst[b2],fgEv,0);
                } else cudaMemcpy(g_fgrng,&rolloutRng,8,cudaMemcpyHostToDevice); }
              unsigned long long key=1469598103934665603ULL;
              #define KH(v) do{ key^=(unsigned long long)(v); key*=1099511628211ULL; }while(0)
              KH(N);KH(D);KH(H);KH(L);KH(A);KH(T);KH(nbuf);KH(g_mgbp.obsKind);KH(g_mgbp.f32obs);
              KH((uintptr_t)dP);KH((uintptr_t)dObsF);KH((uintptr_t)g_mgbp.hObsF32);KH((uintptr_t)g_mgbp.hObsF32B);
              KH((uintptr_t)dO);KH((uintptr_t)hSamp);KH((uintptr_t)g_mgbp.termPlane);KH((uintptr_t)dMGTraj);
              KH((uintptr_t)g_dcAct);KH((uintptr_t)dSa);KH((uintptr_t)dTerms);
              #undef KH
              long need=(long)T*nbuf;
              if(key!=g_fgg_key || need>g_fgg_cap || !g_fgg){
                if(g_fgg){ for(long i=0;i<g_fgg_cap;i++) if(g_fgg_ok[i]==1) cudaGraphExecDestroy(g_fgg[i]);
                  free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
                g_fgg=(cudaGraphExec_t*)calloc((size_t)need,sizeof(cudaGraphExec_t));
                g_fgg_ok=(signed char*)calloc((size_t)need,1);
                if(g_fgg && g_fgg_ok){ g_fgg_cap=need; g_fgg_key=key; }
                else { free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
              }
            }
          } else if(!fgGate && g_fgg){ /* gate turned off mid-process: drop the table (workers go eager) */
            for(long i=0;i<g_fgg_cap;i++) if(g_fgg_ok[i]==1) cudaGraphExecDestroy(g_fgg[i]);
            free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
        }
        g_mgbp.actCol=actCol; g_mgbp.logpCol=logpCol; g_mgbp.valCol=valCol; g_mgbp.rewCol=rewCol; g_mgbp.termCol=termCol;
        /* resident chaining: continue from the stamped ping-pong parity, skip the obs0 pre-cast */
        g_mgbp.startPc = residentIn? g_mgchain.pc : 0;
        g_mgbp.skipPre = residentIn? 1 : 0;
        g_mgbp.N=(long)N; g_mgbp.D=(long)D; g_mgbp.T=(long)T; g_mgbp.LH=LH; g_mgbp.H=(int)H; g_mgbp.L=(int)L;
        g_mgbp.A=(int)A; g_mgbp.O=(int)O; g_mgbp.bf=bf; g_mgbp.nAg=nAg; g_mgbp.rolloutRng=rolloutRng;
        g_mgbp.nbuf=nbuf;
        g_mgbp.md=0; g_mgbp.K=1; g_mgbp.dHs=NULL;   /* single-discrete: never the K-head worker arm */
        int nenv=eh->N;
        for(int b=0;b<nbuf;b++){ int elo=(int)((long)b*nenv/nbuf), ehi=(int)((long)(b+1)*nenv/nbuf);
          g_mgbp.envLo[b]=elo; g_mgbp.envHi[b]=ehi; g_mgbp.rowBase[b]=(int)((long)elo*nAg); g_mgbp.rowN[b]=(int)((long)(ehi-elo)*nAg); }
        {
        memset(g_mgbp.prof,0,sizeof(g_mgbp.prof));
        for(int b=nbuf;b<MAXBUF_MG;b++) g_mgbp.rowN[b]=0;   /* idle slots: mg_buf_worker's nb<=0 early-return */
        /* PRE-CAPTURE the wide graph table (fresh table only): captures record API-only (no GPU
           execution), so all T*nbuf graphs cost ~14ms ONCE here instead of stalling update 0 by
           50-64ms (measured via nsys inter-launch gap analysis). Uses the SAME wbf_issue_chain the
           worker replays. MUST sit after rowBase/rowN are filled (a zero-size grid from unset rowN
           invalidated every capture with cudaErrorInvalidValue — found the hard way) and after
           startPc, whose parity is baked into the in-graph obs source. */
        if(g_mgbp.wideBf && g_wbfg && wbfFresh){
          int inclObs=(g_mgbp.obsKind==1 && (T%2)==0);
          for(int b2=0;b2<nbuf;b2++) for(long s2=0;s2<(long)T;s2++){
            long gi=s2*(long)nbuf+b2; if(g_wbfg_ok[gi]!=0) continue;
            cudaGraph_t gr=NULL;
            cudaStreamBeginCapture(wbf_stream(b2),cudaStreamCaptureModeThreadLocal);
            wbf_issue_chain(&g_mgbp,b2,s2,inclObs);
            cudaError_t ce=cudaStreamEndCapture(wbf_stream(b2),&gr);
            if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_wbfg[gi],gr,0)==cudaSuccess){
              g_wbfg_ok[gi]=1; cudaGraphDestroy(gr); }
            else { g_wbfg_ok[gi]=0; if(gr) cudaGraphDestroy(gr); cudaGetLastError(); }   /* leave 0:
              the worker's lazy capture gets its own try (a setup-time failure must not poison the
              table into eager-forever) */
          }
        }
        WCSTAMP(wc_setup)
        if(mgbufw_init()){
          pthread_barrier_wait(&g_mgbufw_bar);               /* release all MAXBUF_MG workers */
          pthread_barrier_wait(&g_mgbufw_bar);                /* join */
        } else {
          pthread_t bth[MAXBUF_MG];                           /* pool init failed once — permanent fallback */
          for(int b=0;b<nbuf;b++) pthread_create(&bth[b],NULL,mg_buf_worker,(void*)(long)b);
          for(int b=0;b<nbuf;b++) pthread_join(bth[b],NULL);
        }
        WCSTAMP(wc_work)
        if(wcprof) fprintf(stderr,"[mg-call] setup=%.2f workers=%.2f ",wc_setup-wcall0,wc_work-wc_setup);
        if(g_mgbp.wideBf) k_bf2f_n<<<GR((long)N*LH),B>>>(dSa,g_wbf.dSt,(long)N*LH);   /* widen bf16 state
          back into dSa so the (unchanged, f32) bootstrap/finalState tail reads it transparently */
        cudaDeviceSynchronize();
        if(getenv("PUFFER_ROLL_PROFILE")){
          static const char* nm[8]={"cast","launch","sync","act","env","scat","tail","queue"};
          double agg[8]={0}; for(int b=0;b<nbuf;b++) for(int k=0;k<8;k++) agg[k]+=g_mgbp.prof[b][k];
          fprintf(stderr,"[mg-cyc]"); for(int k=0;k<8;k++) fprintf(stderr," %s=%.1f",nm[k],agg[k]/nbuf);
          fprintf(stderr," ms/buffer (cumulative this update, %d buffers)\n",nbuf);
        }
        if(g_mgbp.f32obs) fObs32=((g_mgbp.startPc+T)%2)? g_mgbp.hObsF32B : g_mgbp.hObsF32;   /* final obs (ping-pong, parity carried across chained updates) */
        else cur=(T%2==1)?hA:hB;                   /* final obs buffer (workers ran T swaps from obs0→hA→hB→…) */
        }
        free(actRM);
      }
    }
    if(!useBuf) for(size_t s=0;s<T;s++){
      if(mgrprof) mrt0=now_ms();
      cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice);
      if(mgrprof){ MR_h2d+=now_ms()-mrt0; mrt0=now_ms(); }
      k_d2f<<<GR(N*D),B>>>(dObsF,dObs,(long)N*D);
      if(dMGTraj) k_scatter_mg_obs<<<GR(N*D),B>>>(dMGTraj,dObsF,0,(long)N,(long)s,(long)T,(long)D);   /* obs → device traj */
      mingru_fwd_dev(hbl,dP,dObsF,dStCur,dStNxt,dHb,dHn,dYb,dLg,dVal,(int)N,(int)D,(int)H,(int)L,(int)A,bf,0);
      k_mingru_asm<<<GR(N*O),B>>>(dLg,dVal,dY,(int)N,(int)A,(int)O);        /* [logits(A); value] per row */
      k_sample<<<GR(N),B>>>(dY,dO,(int)N,(int)A,(int)O,(unsigned long long)(rolloutRng+(uint64_t)(s*N)*G));
      cudaDeviceSynchronize();
      if(mgrprof){ MR_gpu+=now_ms()-mrt0; mrt0=now_ms(); }
      cudaMemcpy(hSamp,dO,8*3*N,cudaMemcpyDeviceToHost);
      if(mgrprof){ MR_d2h+=now_ms()-mrt0; mrt0=now_ms(); }
      if(threaded){
        g_rp.s=s; g_rp.cur=cur; g_rp.nxt=nxt; g_rp.hSamp=hSamp; g_rp.actRM=hSamp; g_rp.hRT=hRT;
        pthread_barrier_wait(&g_rp.bar);                                   /* release workers: env-step + scatter */
        pthread_barrier_wait(&g_rp.bar);                                   /* wait until all partitions done */
      } else {
        eh->step(eh->env, hSamp, nxt, hRT, hRT+N);
        for(size_t e=0;e<N;e++){ long row=(long)e*T+(long)s;
          if(!dMGTraj) for(size_t j=0;j<D;j++) obsCol[row*D+j]=cur[e*D+j];
          actCol[row]=hSamp[e]; logpCol[row]=hSamp[N+e]; valCol[row]=hSamp[2*N+e];
          rewCol[row]=hRT[e]; termCol[row]=hRT[N+e]; }
      }
      if(mgrprof){ MR_env+=now_ms()-mrt0; mrt0=now_ms(); }
      cudaMemcpy(dTerms,hRT+N,8*N,cudaMemcpyHostToDevice);                 /* reset new state on terminals (needs hRT from the step) */
      k_mg_reset<<<GR((long)N*LH),B>>>(dStNxt,dTerms,(int)N,(int)LH,0);
      if(mgrprof){ cudaDeviceSynchronize(); MR_rst+=now_ms()-mrt0; }
      float* t2=dStCur; dStCur=dStNxt; dStNxt=t2;                          /* advance state (ping-pong) */
      cur=nxt; nxt=(nxt==hA)?hB:hA;
    }
    if(mgrprof) fprintf(stderr,"[mg-roll] h2d=%.0f gpu(fwd+sample)=%.0f d2h=%.0f env+scatter=%.0f reset=%.0f ms (cumulative)\n",MR_h2d,MR_gpu,MR_d2h,MR_env,MR_rst);
    if(dMGTraj){ g_dMGObsTraj_valid=1; g_dMGObsTraj_N=(long)N; }   /* obs trajectory ready for the BPTT's on-device gather */
    /* bootstrap V(s_T) at (finalObs, finalState=dStCur); dStNxt is free scratch. f32-obs mode: the final
       obs are already f32 (pinned) — H2D straight to dObsF; widening for finalObs is exact (f32⊂f64). */
    if(fObs32){
      /* final obs live in the transport kind's element — H2D raw (dObs as byte scratch) + device widen */
      if(g_mgbp.obsKind==1){ cudaMemcpy(dObs,fObs32,(size_t)N*D,cudaMemcpyHostToDevice);
        k_u82f<<<GR(N*D),B>>>(dObsF,(const unsigned char*)dObs,(long)N*D); }
      else if(g_mgbp.obsKind==2){ cudaMemcpy(dObs,fObs32,2*(size_t)N*D,cudaMemcpyHostToDevice);
        k_bf2f<<<GR(N*D),B>>>(dObsF,(const __nv_bfloat16*)dObs,(long)N*D); }
      else cudaMemcpy(dObsF,fObs32,4*N*D,cudaMemcpyHostToDevice);
    }
    else { cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice); k_d2f<<<GR(N*D),B>>>(dObsF,dObs,(long)N*D); }
    mingru_fwd_dev(hbl,dP,dObsF,dStCur,dStNxt,dHb,dHn,dYb,dLg,dVal,(int)N,(int)D,(int)H,(int)L,(int)A,bf,0);
    if(useBuf && g_mgbp.dcOK){
      k_f2d<<<GR(N),B>>>(g_dcBoot,dVal,(long)N);                 /* boot column device-direct */
      /* whole-update rew/term column scatter from the pinned planes (replaces T·nbuf per-step scatters) */
      k_mg_cols_rt_all<<<GR((long)N*T),B>>>(g_mgbp.rewPlane,g_mgbp.termPlane,g_dcRew,g_dcTerm,(long)N,(long)T);
    }
    if(finalState || bootVals) cudaDeviceSynchronize();
    /* resident chaining: nothing below reads device results on the host (finalObs/finalState/bootVals
       are all skipped), and every downstream consumer (vtrace, grad, muon — all legacy-stream or
       stream-ordered) already serializes behind these enqueues — so the device-wide sync is pure
       drain-wait on the critical path. Skipping it lets the tail's bootstrap forward + column scatter
       overlap Lean's next-phase enqueues. */
    if(useBuf && g_mgbp.dcOK){ g_dc_valid=1; g_dc_fromroll=1; g_mbSinceRoll=0; g_rollIdx++; }   /* columns stamped — Lean skips prep */
    if(wcprof) fprintf(stderr,"tail-gpu=%.2f ",now_ms()-wcall0);
    if(bootVals){ cudaMemcpy(hVal,dVal,4*N,cudaMemcpyDeviceToHost);         /* dVal is f32 — read as f32 then widen */
      float* hValF=(float*)hVal; for(long i=N-1;i>=0;i--) bootVals[i]=(double)hValF[i]; }
    if(finalObs){
    if(fObs32){
      if(g_mgbp.obsKind==1){ const unsigned char* q=(const unsigned char*)fObs32;
        for(size_t i=0;i<N*D;i++) finalObs[i]=(double)q[i]; }
      else if(g_mgbp.obsKind==2){ const unsigned short* q=(const unsigned short*)fObs32;
        for(size_t i=0;i<N*D;i++){ union{float f;unsigned u;} v; v.u=((unsigned)q[i])<<16; finalObs[i]=(double)v.f; } }
      else for(size_t i=0;i<N*D;i++) finalObs[i]=(double)fObs32[i];
    }
    else       for(size_t i=0;i<N*D;i++) finalObs[i]=cur[i];
    }
    if(finalState){ cudaMemcpy(hStF,dStCur,4*N*LH,cudaMemcpyDeviceToHost); for(long i=0;i<N*LH;i++) finalState[i]=(double)hStF[i]; }
    /* resident-chain stamp: obs live in the pinned ping-pong at parity (startPc+T)%2, state lives in
       dSa — a same-shape chained call may now skip both round trips. Only chain-capable configs stamp
       (buffered + device-direct columns + f32obs staging; conditions are deterministic per run). */
    { int chainGate; { const char* e=getenv("PUFFER_MG_CHAIN"); chainGate=(e==NULL||e[0]!='0'); }
      if(chainGate && useBuf && g_mgbp.dcOK && g_mgbp.f32obs){
        g_mgchain.valid=1; g_mgchain.md=0; g_mgchain.pc=(int)((g_mgbp.startPc+T)%2);
        g_mgchain.N=(long)N; g_mgchain.D=(long)D; g_mgchain.LH=LH;
        g_mgchain.hA=g_mgbp.hObsF32; g_mgchain.hB=g_mgbp.hObsF32B; g_mgchain.dSa=dSa;
      } else g_mgchain.valid=0; }
    #undef GR
  }
  free(hStF); free(hVal);   /* hSamp/hA/hB/hRT are the persistent pinned cache (mg_hb) — not freed */
  if(wcprof) fprintf(stderr,"return=%.2f ms\n",now_ms()-wcall0);
  lean_dec(obs0a); lean_dec(state0a);
  /* !ok used to zero-fill the whole return: no env was stepped, yet the trainer happily ran V-Trace, the
     BPTT and the Muon over an all-zero rollout, printed episode returns and SPS, and exited 0. Same rule
     as the gradient path — an unusable rollout is fatal, not a quiet bag of zeros. */
  if(!ok){ lean_dec(Oo); return mg_contract_fail("rollout",
      "device/pinned rollout buffers unavailable — no environment step could be driven", P); }
  return lean_io_result_mk_ok(Oo);
}
/* ===== MULTI-DISCRETE MinGRU rollout ==============================================================
   TWO ARMS, chosen per call and both K-head correct:

   [A] FUSED (default when the shape fits): the K-head port of lean_cuda_plugin_rollout_mingru's fast
       path — concurrent stream-buffers (mg_buf_worker_md), ONE k_mg_fused_step_w_md launch per (step,
       buffer) doing traj-scatter + encoder + L gate layers (folded terminal reset) + heads + per-head
       sampling + resident-column writes, pinned zero-copy obs staging, a device-resident obs
       trajectory, per-(t,buf) CUDA graphs (g_fgg) and — after the first update — resident obs/state
       chaining. What made this "one action per row" before is now parameterised: the sampler writes K
       row-major actions per row into a pinned plane the envs read directly (act[e·K+h], the step_range
       ABI's own layout), the action D2H is 8·nb·K bytes, and the K-wide act column goes straight into
       the resident device column (g_dcAct, row·K+h) so mg_prep is skipped entirely.
   [B] LEGACY non-fused (fallback, unchanged): whole-batch mingru_fwd_dev → k_mingru_asm → k_sample_md
       → threaded CPU env-step → host column scatter. Taken when the fused kernel cannot express the
       shape (mg_wmma_shbytes==0: H%16, H>128, W+1>H, or the shared-memory budget), when PUFFER_MG_WMMA
       is off, when the plugin predates step_range, or when any pinned/device buffer the fused arm needs
       could not be allocated — and forced by PUFFER_MG_MD_FUSED=0. Correctness never depends on which
       arm ran; only throughput does.

   The single-discrete driver and kernels are untouched: mg_buf_worker dispatches to the K-head worker
   on p->md (0 for every single-discrete call), and k_mg_fused_step_w_md is a separate kernel.

   Contract:
     · obs are NOT returned — the device-resident trajectory (g_dMGObsTraj) is REQUIRED, since the BPTT
       always gathers from it; if that buffer cannot be allocated we abort rather than train on nothing.
     · the action column is N·T·K (row e·T+s, K heads contiguous) — the layout mg_prep/k_mg_gather_scal/
       k_mg_ppo_b_md expect.
     · logp is the JOINT log-prob Σ_h log p_h(a_h) (k_sample_md's convention, reproduced op-for-op by
       d_mg_sample_row_md), matching the gradient's.
     · FUSED arm + device columns: act/logp/val/rew/term/boot are stamped device-DIRECT, so Lean skips
       its slice build and mg_prep (lean_cuda_mg_cols_ready). Those three columns of the return are then
       left unwritten — exactly as on the single-discrete path — and Lean must not read them without
       checking cols_ready.
   Returns [actCol(NT·K); logpCol(NT); valCol(NT); rewCol(NT); termCol(NT); finalObs(N·D);
   finalState(N·L·H); bootVals(N)], or the CHAINED compact form [rewCol(NT); termCol(NT)] when called
   with EMPTY obs0/state0. W = Σ head sizes = the logits width the policy emits. */
static int mg_md_fused(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_MG_MD_FUSED");
  f=(e==NULL||e[0]!='0'); } return f; }   /* PUFFER_MG_MD_FUSED=0 ⇒ always the legacy arm */
/* ---- BEHAVIOURAL KNOB (PUFFER_MG_H0_VALUE) ------------------------------------------------------
   ISOLATE THE VALUE CHANNEL. The rollout's value column is V(o_t | h carried across segments AND
   across updates); the value head that is TRAINED is evaluated by the BPTT forward, which restarts
   every segment at h=0 (k_mg_scan_fwd2: `float state=0.0f;`). That value column is the V-Trace
   bootstrap AND — via k_mg_gather_scal's `dRet=a+v` / `dOv=v` — the regression target and the
   value-clip centre. Zeroing the rollout state (the default; PUFFER_MG_KEEP_ROLL_STATE=1 restores
   the old threading) removes that gap but ALSO makes the PPO ratio convention consistent, so it
   cannot say which channel carries the collapse.
   This gate re-derives the value column (and the bootstrap) with an h=0-at-segment-start forward over
   the SAME resident obs trajectory and the SAME weights the rollout just used, and overwrites
   g_dcValue / g_dcBoot with it — leaving g_dcLogp (oldLogp), the sampled actions, the carried rollout
   state and the whole policy path bit-for-bit untouched. g_dcVal0 is deliberately NOT overwritten (it
   preserves the rollout value the drift is measured against).
   DEFAULT OFF — it changes training. */
static inline int mg_h0_value(void){
  static int v=-1; if(v<0){ const char* e=getenv("PUFFER_MG_H0_VALUE");
    v = (e&&e[0]&&e[0]!='0') ? 1 : 0; } return v; }
/* obs[:,t,:] out of the [N,T,D] resident trajectory into a dense [N,D] f32 step slice */
__global__ void k_mg_gather_t(float* dst, const float* traj, long N, long T, long D, long t){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N*D) return;
  long n=idx/D, j=idx%D; dst[idx]=traj[(n*T+t)*D+j]; }
/* scatter the step's value into the [N,T] row-major value column at (n·T+t) */
__global__ void k_mg_scatter_valcol(double* col, const float* v, long N, long T, long t){
  long n=(long)blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
  col[n*T+t]=(double)v[n]; }
static void mg_h0_value_refresh(cublasHandle_t hbl, const float* dP, const float* traj,
    const float* dFinalObsF, const double* termPlane, long N, long T, long D,
    int H, int L, int A, long LH, int bf){
  if(!mg_h0_value()) return;
  if(!(hbl && dP && traj && dFinalObsF && g_dcValue && g_dcBoot && N>0 && T>0)){
    static int warned=0; if(!warned){ warned=1;
      fprintf(stderr,"[h0val] PUFFER_MG_H0_VALUE=1 requested but a required buffer is missing — OFF\n"); }
    return; }
  float* dS0=(float*)bg(104,4*(size_t)N*LH);  float* dS1=(float*)bg(105,4*(size_t)N*LH);
  float* dX =(float*)bg(106,4*(size_t)N*D);   float* dHb=(float*)bg(107,4*(size_t)N*H);
  float* dHn=(float*)bg(108,4*(size_t)N*H);   float* dYb=(float*)bg(109,4*(size_t)N*3*H);
  float* dLg=(float*)bg(111,4*(size_t)N*A);   float* dV =(float*)bg(101,4*(size_t)N);
  if(!(dS0&&dS1&&dX&&dHb&&dHn&&dYb&&dLg&&dV)){
    static int warned=0; if(!warned){ warned=1;
      fprintf(stderr,"[h0val] scratch allocation failed — gate OFF (training UNCHANGED)\n"); }
    return; }
  int Bk=256;
  cudaMemset(dS0,0,4*(size_t)N*LH);                      /* PufferLib's convention: h=0 at t=0 */
  for(long t=0;t<T;t++){
    k_mg_gather_t<<<ceildiv(N*D,Bk),Bk>>>(dX,traj,N,T,D,t);
    mingru_fwd_dev(hbl,dP,dX,dS0,dS1,dHb,dHn,dYb,dLg,dV,(int)N,(int)D,H,L,A,bf,0);
    k_mg_scatter_valcol<<<ceildiv(N,Bk),Bk>>>(g_dcValue,dV,N,T,t);
    if(termPlane)                                        /* same in-segment reset the BPTT scan applies */
      k_mg_reset<<<ceildiv(N*LH,Bk),Bk>>>(dS1,termPlane+t*N,(int)N,(int)LH,0);
    { float* tmp=dS0; dS0=dS1; dS1=tmp; }
  }
  /* bootstrap V(o_T) continued from the SAME h=0-rooted state (the rollout's own boot used the state
     carried in from previous updates) */
  mingru_fwd_dev(hbl,dP,dFinalObsF,dS0,dS1,dHb,dHn,dYb,dLg,dV,(int)N,(int)D,H,L,A,bf,0);
  k_f2d<<<ceildiv(N,Bk),Bk>>>(g_dcBoot,dV,N);
  cudaDeviceSynchronize();
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_plugin_rollout_mingru_md(
    size_t hh, size_t policyH, lean_obj_arg obs0a, lean_obj_arg state0a, lean_obj_arg hsA,
    size_t N, size_t D, size_t H, size_t L, size_t W, size_t K, size_t T,
    uint64_t rolloutRng, lean_obj_arg w){
  (void)w;
  Handle* eh=(Handle*)hh;
  size_t O=W+1, wEncSz=H*D, layerSz=3*H*H, P=wEncSz+H+L*layerSz+W*H+W+H+1;
  const uint64_t G=0x9E3779B97F4A7C15ULL; int bf=1;
  const double* hs=lean_float_array_cptr(hsA);
  long NT=(long)N*T, LH=(long)L*H;
  /* resident chaining: EMPTY obs0+state0 ⇒ obs live in the pinned ping-pong (by parity) and the state
     in dSa. A stale/absent stamp ABORTS (same rule as the single-discrete driver: an empty return is
     indistinguishable from a legitimate chained one, and Lean would slice garbage and train on it). */
  int residentIn=(lean_sarray_size(obs0a)==0 && lean_sarray_size(state0a)==0);
  int chainStampOK=(g_mgchain.valid && g_mgchain.md==1 &&   /* md: only THIS driver's own stamp counts */
                    g_mgchain.N==(long)N && g_mgchain.D==(long)D && g_mgchain.LH==LH);
  if(residentIn && !chainStampOK){
    lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA);
    return mg_contract_fail("multi-discrete rollout",
      "resident-chain call (empty obs0/state0) without a valid same-shape chain stamp — a mid-run "
      "rollout allocation failure dropped the resident obs/state chain", P);
  }
  const double* obs0=residentIn? NULL : lean_float_array_cptr(obs0a);
  const double* st0 =residentIn? NULL : lean_float_array_cptr(state0a);
  long cols=NT*((long)K+4)+(long)N*D+(long)N*LH+(long)N;
  long colsAlloc=residentIn? 2*NT : cols;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),colsAlloc,colsAlloc); double* out=lean_float_array_cptr(Oo);
  double *actCol,*logpCol,*valCol,*rewCol,*termCol,*finalObs,*finalState,*bootVals;
  if(residentIn){
    actCol=logpCol=valCol=finalObs=finalState=bootVals=NULL; rewCol=out; termCol=out+NT;
  } else {
    actCol=out; logpCol=actCol+NT*(long)K; valCol=logpCol+NT; rewCol=valCol+NT; termCol=rewCol+NT;
    finalObs=termCol+NT; finalState=finalObs+(long)N*D; bootVals=finalState+(long)N*LH;
  }
  cublasHandle_t hbl=cu_handle();
  float *dP=(float*)rb2(0,4*P), *dSa=(float*)rb2(1,4*N*LH), *dSb=(float*)rb2(2,4*N*LH), *dObsF=(float*)rb2(3,4*N*D);
  float *dHb=(float*)rb2(4,4*N*H), *dHn=(float*)rb2(5,4*N*H), *dYb=(float*)rb2(6,4*N*3*H);
  float *dLg=(float*)rb2(7,4*N*W), *dVal=(float*)rb2(8,4*N);
  double *dObs=(double*)rb2(9,8*N*D), *dY=(double*)rb2(10,8*N*O), *dO=(double*)rb2(11,8*(K+2)*N), *dTerms=(double*)rb2(12,8*N);
  int* dHs=(int*)rb2(23,sizeof(int)*K);
  float* dMGTraj=mgobstraj_buf((size_t)N*T*D); g_dMGObsTraj_valid=0;
  /* the f32 state staging is ONLY the up/down conversion buffer for the non-chained contract — a chained
     call never touches the host state at all, so it must not pay a 4·N·L·H malloc per update either */
  float *hStF=residentIn? NULL : (float*)malloc(4*N*LH);
  double *hVal=(double*)malloc(8*N);
  int threaded=(eh && eh->step_range!=NULL);
  int nAg=(eh && eh->numAgents>0)?eh->numAgents:1;
  /* ---- arm selection (explicit, and every gate is a REASON the fused arm cannot run) --------------
     The kernel and this predicate call the SAME mg_wmma_shbytes(D,H,L,W), so a shape it rejects can
     never reach a launch with a mismatched shared-memory layout. */
  size_t shbMD=(mg_md_fused() && mg_fused() && mg_wmma())? mg_wmma_shbytes((int)D,(int)H,(int)L,(int)W) : 0;
  int fusedArm=(N>0 && K>0 && W>0 && shbMD!=0 && mg_wmma_optin_md(shbMD) && threaded && dMGTraj!=NULL
                && hbl!=NULL && eh && policyH && dP&&dSa&&dSb&&dObsF&&dHb&&dHn&&dYb&&dLg&&dVal&&dO&&dObs&&dHs&&hVal
                && (residentIn || hStF));   /* dObs: the bootstrap's raw-transport H2D scratch */
  int nbuf=fusedArm? mg_roll_buffers((long)N) : 0;
  /* pinned staging the fused arm needs (shared mg_hb cache with the single-discrete driver) */
  double *hSamp=NULL,*hA=NULL,*hB=NULL,*hRT=NULL,*actRM=NULL;
  if(fusedArm){
    hRT=(double*)mg_hb(3,8*2*N);
    g_mgbp.hObsF32 =(float*)mg_hb(4,4*(size_t)N*D);
    g_mgbp.hObsF32B=(float*)mg_hb(5,4*(size_t)N*D);
    g_mgbp.actPin  =(double*)mg_hb(7,8*(size_t)N*K);        /* K-wide row-major pinned action plane */
    g_mgbp.rewPlane=(double*)mg_hb(8,8*(size_t)N*T);
    g_mgbp.termPlane=(double*)mg_hb(9,8*(size_t)N*T);
    hA=(double*)mg_hb(1,8*(size_t)N*D); hB=(double*)mg_hb(2,8*(size_t)N*D);   /* f64 ping-pong (non-f32obs plugins) */
    if(!(hRT && g_mgbp.hObsF32 && g_mgbp.hObsF32B && g_mgbp.actPin && g_mgbp.rewPlane && g_mgbp.termPlane
         && hA && hB && mg_hb_pinned(7) && mg_hb_pinned(8) && mg_hb_pinned(9))) fusedArm=0;
    /* device-direct resident columns (K-wide act) — REQUIRED: the fused kernel writes act/logp/val
       there and nowhere else, so without them the host columns would be left empty. */
    if(fusedArm){
      if(dc_alloc_k((size_t)NT,K)){ g_dcN=(long)N; g_dcT=(long)T; g_dc_valid=0; g_dc_fromroll=0; }
      else fusedArm=0;
    }
  }
  if(residentIn && !fusedArm){
    lean_dec(Oo); lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA); free(hStF); free(hVal);
    return mg_contract_fail("multi-discrete rollout",
      "a chained (empty obs0/state0) call arrived but the FUSED arm is unavailable this call — the "
      "resident obs/state only exist inside it, so there is nothing to roll out from", P);
  }
  /* announce the arm — and RE-announce whenever it changes (an arm flip mid-run means an allocation
     started failing, which is exactly when a silent 10x slowdown would otherwise look like bad luck) */
  { static int lastArm=-1;
    if(lastArm!=fusedArm){ lastArm=fusedArm;
      if(fusedArm) fprintf(stderr,"[puffer] MD MinGRU rollout: FUSED arm (k_mg_fused_step_w_md, %d buffers, %zu B shared)\n",nbuf,shbMD);
      else fprintf(stderr,"[puffer] MD MinGRU rollout: legacy non-fused arm (fused unavailable: "
        "shbytes=%zu wmma=%d fused=%d gate=%d threaded=%d traj=%d) — correct but slow\n",
        shbMD,mg_wmma(),mg_fused(),mg_md_fused(),threaded,dMGTraj!=NULL); } }
  if(!fusedArm){
    /* ---- [B] legacy non-fused arm (unchanged) --------------------------------------------------- */
    hSamp=(double*)malloc(8*(size_t)(K+2)*N); hA=(double*)malloc(8*N*D); hB=(double*)malloc(8*N*D);
    hRT=(double*)malloc(8*2*N); actRM=(double*)malloc(8*(size_t)K*N);
    int ok=(N>0 && K>0 && W>0 && hbl!=NULL && eh && policyH && dMGTraj
            && dP&&dSa&&dSb&&dObsF&&dHb&&dHn&&dYb&&dLg&&dVal&&dObs&&dY&&dO&&dTerms&&dHs
            && hStF&&hSamp&&hA&&hB&&hRT&&hVal&&actRM
            && lean_sarray_size(hsA)>=K && lean_sarray_size(obs0a)>=N*D && lean_sarray_size(state0a)>=(size_t)N*LH);
    if(ok){
      cublasSetStream(hbl,0); int B=256;
      #define GR(x) ceildiv((long)(x),B)
      k_d2f<<<GR(P),B>>>(dP,(const double*)policyH,(long)P);         /* weights RESIDENT (f64 handle → f32 dP) */
      { int* hHs=(int*)malloc(sizeof(int)*K); for(size_t i=0;i<K;i++) hHs[i]=(int)hs[i];
        cudaMemcpy(dHs,hHs,sizeof(int)*K,cudaMemcpyHostToDevice); free(hHs); }
      for(long i=0;i<N*LH;i++) hStF[i]=(float)st0[i];
      cudaMemcpy(dSa,hStF,4*N*LH,cudaMemcpyHostToDevice);            /* recurrent state (threaded across updates) */
      if(mg_zero_roll_state()) cudaMemset(dSa,0,4*(size_t)N*LH);     /* PufferLib convention (default ON; PUFFER_MG_KEEP_ROLL_STATE=1 to thread) */
      if(threaded && !g_rp_init){
        g_rp.nthreads=rp_threads(); g_rp.alive=1;
        pthread_barrier_init(&g_rp.bar,NULL,g_rp.nthreads+1);
        for(long t=0;t<g_rp.nthreads;t++) pthread_create(&g_rp_th[t],NULL,rp_worker,(void*)t);
        g_rp_init=1;
      }
      if(threaded){                                                  /* shape-invariant fields (per call) */
        g_rp.eh=eh; g_rp.obsCol=NULL; g_rp.actCol=actCol; g_rp.logpCol=logpCol; g_rp.valCol=valCol;
        g_rp.rewCol=rewCol; g_rp.termCol=termCol; g_rp.N=(long)N; g_rp.D=(long)D; g_rp.T=(long)T;
        g_rp.nAgents=nAg; g_rp.envLo=0; g_rp.envHi=eh->N; g_rp.rowBase=0; g_rp.Nstride=(long)N; g_rp.W=(int)K;
        g_rp.skipObs=1;                                              /* obs are device-resident (dMGTraj) */
      }
      float *dStCur=dSa, *dStNxt=dSb;
      const double* cur=obs0; double* nxt=hA;
      for(size_t s=0;s<T;s++){
        cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice);
        k_d2f<<<GR(N*D),B>>>(dObsF,dObs,(long)N*D);
        k_scatter_mg_obs<<<GR(N*D),B>>>(dMGTraj,dObsF,0,(long)N,(long)s,(long)T,(long)D);   /* obs → device traj */
        mingru_fwd_dev(hbl,dP,dObsF,dStCur,dStNxt,dHb,dHn,dYb,dLg,dVal,(int)N,(int)D,(int)H,(int)L,(int)W,bf,0);
        k_mingru_asm<<<GR(N*O),B>>>(dLg,dVal,dY,(int)N,(int)W,(int)O);   /* [logits(W); value] per row */
        k_sample_md<<<GR(N),B>>>(dY,dO,(int)N,(int)K,dHs,(int)O,(unsigned long long)(rolloutRng+(uint64_t)(s*N)*G));
        cudaDeviceSynchronize();
        cudaMemcpy(hSamp,dO,8*(size_t)(K+2)*N,cudaMemcpyDeviceToHost);   /* [act(K×N col); jointLogp(N); val(N)] */
        for(size_t e=0;e<N;e++) for(size_t k2=0;k2<K;k2++) actRM[e*K+k2]=hSamp[(long)k2*(long)N+e];  /* col→row for env-step */
        if(threaded){
          g_rp.s=s; g_rp.cur=cur; g_rp.nxt=nxt; g_rp.hSamp=hSamp; g_rp.actRM=actRM; g_rp.hRT=hRT;
          pthread_barrier_wait(&g_rp.bar);                            /* release workers: env-step + scatter */
          pthread_barrier_wait(&g_rp.bar);                            /* wait until all partitions done */
        } else {
          eh->step(eh->env, actRM, nxt, hRT, hRT+N);
          for(size_t e=0;e<N;e++){ long row=(long)e*T+(long)s;
            for(size_t k2=0;k2<K;k2++) actCol[row*(long)K+k2]=hSamp[(long)k2*(long)N+e];
            logpCol[row]=hSamp[(long)K*(long)N+e]; valCol[row]=hSamp[(long)(K+1)*(long)N+e];
            rewCol[row]=hRT[e]; termCol[row]=hRT[N+e]; }
        }
        cudaMemcpy(dTerms,hRT+N,8*N,cudaMemcpyHostToDevice);          /* reset new state on terminals */
        k_mg_reset<<<GR((long)N*LH),B>>>(dStNxt,dTerms,(int)N,(int)LH,0);
        float* t2=dStCur; dStCur=dStNxt; dStNxt=t2;                   /* advance state (ping-pong) */
        cur=nxt; nxt=(nxt==hA)?hB:hA;
      }
      g_dMGObsTraj_valid=1; g_dMGObsTraj_N=(long)N;                   /* obs trajectory ready for the BPTT gather */
      /* bootstrap V(s_T) at (finalObs, finalState=dStCur); dStNxt is free scratch */
      cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice); k_d2f<<<GR(N*D),B>>>(dObsF,dObs,(long)N*D);
      mingru_fwd_dev(hbl,dP,dObsF,dStCur,dStNxt,dHb,dHn,dYb,dLg,dVal,(int)N,(int)D,(int)H,(int)L,(int)W,bf,0);
      cudaDeviceSynchronize();
      cudaMemcpy(hVal,dVal,4*N,cudaMemcpyDeviceToHost);               /* dVal is f32 — read as f32 then widen */
      { float* hValF=(float*)hVal; for(long i=N-1;i>=0;i--) bootVals[i]=(double)hValF[i]; }
      for(size_t i=0;i<N*D;i++) finalObs[i]=cur[i];
      cudaMemcpy(hStF,dStCur,4*N*LH,cudaMemcpyDeviceToHost);
      for(long i=0;i<N*LH;i++) finalState[i]=(double)hStF[i];
      g_mgchain.valid=0;                                              /* the legacy arm never chains */
      g_dc_fromroll=0;                                                /* nor stamps the device columns */
      #undef GR
    }
    free(hStF); free(hSamp); free(hA); free(hB); free(hRT); free(hVal); free(actRM);
    lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA);
    if(!ok){ lean_dec(Oo); return mg_contract_fail("multi-discrete rollout",
        "device/pinned rollout buffers (incl. the REQUIRED device obs trajectory) unavailable, or the "
        "obs0/state0/headSizes arrays were short — no environment step could be driven", P); }
    return lean_io_result_mk_ok(Oo);
  }
  /* ---- [A] FUSED arm ------------------------------------------------------------------------- */
  {
    int okShapes=(lean_sarray_size(hsA)>=K
                  && (residentIn || (lean_sarray_size(obs0a)>=N*D && lean_sarray_size(state0a)>=(size_t)N*LH)));
    if(!okShapes){
      lean_dec(Oo); lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA); free(hStF); free(hVal);
      return mg_contract_fail("multi-discrete rollout",
        "the obs0/state0/headSizes arrays were shorter than the declared shapes", P);
    }
    cublasSetStream(hbl,0); int B=256;
    #define GR(x) ceildiv((long)(x),B)
    k_d2f<<<GR(P),B>>>(dP,(const double*)policyH,(long)P);            /* weights RESIDENT (f64 handle → f32 dP) */
    mg_pub_wencpad(dP,(int)H,(int)D,(int)L,(int)W);                   /* big-H WMMA encoder pad (A := W) */
    mg_pub_wlbf(dP,(int)H,(int)D,(int)L);                             /* bf16 forward tier (gated) */
    { static int pubFma=-1; int wantF=mg_wprec_bf()?1:0;              /* FMA scalar sections: tolerance tier only */
      if(pubFma!=wantF){ cudaMemcpyToSymbol(c_mgFma,&wantF,sizeof(int)); pubFma=wantF; } }
    { int* hHs=(int*)malloc(sizeof(int)*K); if(hHs){ for(size_t i=0;i<K;i++) hHs[i]=(int)hs[i];
        cudaMemcpy(dHs,hHs,sizeof(int)*K,cudaMemcpyHostToDevice); free(hHs); } }
    if(!residentIn){ for(long i=0;i<N*LH;i++) hStF[i]=(float)st0[i];
      cudaMemcpy(dSa,hStF,4*N*LH,cudaMemcpyHostToDevice); }           /* chained calls reuse dSa in place */
    if(mg_zero_roll_state()) cudaMemset(dSa,0,4*(size_t)N*LH);        /* PufferLib convention (default ON; PUFFER_MG_KEEP_ROLL_STATE=1 to thread) */
    if(!g_rp_init && threaded){                                       /* (unused by this arm, kept warm for the fallback) */
      g_rp.nthreads=rp_threads(); g_rp.alive=1;
      pthread_barrier_init(&g_rp.bar,NULL,g_rp.nthreads+1);
      for(long t=0;t<g_rp.nthreads;t++) pthread_create(&g_rp_th[t],NULL,rp_worker,(void*)t);
      g_rp_init=1;
    }
    mgbuf_init(nbuf);
    k_fill1<<<GR(NT),B>>>(g_dcRatio,NT);                              /* ratioBuf := 1 (prep semantics) */
    g_mgbp.dcOK=1; g_mgbp.spin=0; g_mgbp.wideBf=0; g_mgbp.flagH=NULL; g_wbf.flag=NULL;
    g_mgbp.md=1; g_mgbp.K=(int)K; g_mgbp.dHs=dHs;
    g_mgbp.eh=eh; g_mgbp.obs0=obs0; g_mgbp.dP=dP; g_mgbp.dMGTraj=dMGTraj;
    g_mgbp.dObs=dObs; g_mgbp.dY=dY; g_mgbp.dO=dO; g_mgbp.dTerms=dTerms;
    g_mgbp.dObsF=dObsF; g_mgbp.dSa=dSa; g_mgbp.dHb=dHb; g_mgbp.dHn=dHn; g_mgbp.dYb=dYb; g_mgbp.dLg=dLg; g_mgbp.dVal=dVal;
    g_mgbp.hSamp=NULL; g_mgbp.hA=hA; g_mgbp.hB=hB; g_mgbp.hRT=hRT; g_mgbp.actRM=g_mgbp.actPin;
    { const char* e=getenv("PUFFER_MG_F32OBS");                       /* env writes f32 obs straight into pinned staging */
      g_mgbp.f32obs=(eh->step_range_f32!=NULL && !(e&&e[0]=='0')); }
    /* CHECKED INVARIANT (the workers rely on it non-locally): only the f32-obs ping-pong can hold obs
       resident, so a chained call MUST be in that mode — otherwise the worker would read p->obs0,
       which a chained call does not have. The chain stamp below is gated on f32obs, so this is
       unreachable unless the mode flipped under a live stamp (PUFFER_MG_F32OBS set mid-run). */
    if(residentIn && !g_mgbp.f32obs){
      lean_dec(Oo); lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA); free(hStF); free(hVal);
      return mg_contract_fail("multi-discrete rollout",
        "a chained (empty obs0/state0) call arrived while the f32-obs staging is disabled — the "
        "resident obs only exist in that ping-pong, so there is nothing to roll out from", P);
    }
    g_mgbp.obsKind=0;
    { const char* op=getenv("PUFFER_MG_OBSPREC");
      int forceF32=(op && op[0]=='f'), wantBf=(op && (op[0]=='b'||op[0]=='B'));
      int zcM=(mg_zcobs() && mg_hb_pinned(4) && mg_hb_pinned(5));
      if(g_mgbp.f32obs && zcM && !forceF32){
        if(eh->obsKind==1 && eh->step_range_u8) g_mgbp.obsKind=1;      /* u8 wire: EXACT widen, 4x fewer bytes */
        else if(wantBf && eh->step_range_bf16) g_mgbp.obsKind=2;
      } }
    g_mgbp.actCol=NULL; g_mgbp.logpCol=NULL; g_mgbp.valCol=NULL;      /* device-direct: no host act/logp/val */
    g_mgbp.rewCol=rewCol; g_mgbp.termCol=termCol;
    g_mgbp.startPc = residentIn? g_mgchain.pc : 0;
    g_mgbp.skipPre = residentIn? 1 : 0;
    g_mgbp.N=(long)N; g_mgbp.D=(long)D; g_mgbp.T=(long)T; g_mgbp.LH=LH; g_mgbp.H=(int)H; g_mgbp.L=(int)L;
    g_mgbp.A=(int)W; g_mgbp.O=(int)O; g_mgbp.bf=bf; g_mgbp.nAg=nAg; g_mgbp.rolloutRng=rolloutRng;
    g_mgbp.nbuf=nbuf;
    { int nenv=eh->N;
      for(int b=0;b<nbuf;b++){ int elo=(int)((long)b*nenv/nbuf), ehi=(int)((long)(b+1)*nenv/nbuf);
        g_mgbp.envLo[b]=elo; g_mgbp.envHi[b]=ehi; g_mgbp.rowBase[b]=(int)((long)elo*nAg); g_mgbp.rowN[b]=(int)((long)(ehi-elo)*nAg); }
      for(int b=nbuf;b<MAXBUF_MG;b++) g_mgbp.rowN[b]=0; }              /* idle slots: nb<=0 early-return */
    /* per-(t,buf) fused-graph table + the per-update rng device scalar. Keyed on shapes (INCLUDING K and
       W, so a single-discrete table can never be replayed here) and on every pointer the captured
       launches bake — a cache realloc silently invalidates captured graphs. PUFFER_MG_FGRAPH=0 disables. */
    { int fgGate; { const char* e=getenv("PUFFER_MG_FGRAPH"); fgGate=(e==NULL||e[0]!='0'); }
      if(fgGate){
        if(!g_fgrng) g_fgrng=(unsigned long long*)bg(100,8);
        if(g_fgrng){
          { unsigned long long* hs2=(unsigned long long*)mg_hb(10,16);
            static cudaEvent_t fgEv=NULL;
            if(hs2 && mg_hb_pinned(10) && (fgEv || cudaEventCreateWithFlags(&fgEv,cudaEventDisableTiming)==cudaSuccess)){
              hs2[1]=rolloutRng;
              cudaMemcpyAsync(g_fgrng,hs2+1,8,cudaMemcpyHostToDevice,g_mgbufst[0]);
              cudaEventRecord(fgEv,g_mgbufst[0]);
              for(int b2=1;b2<nbuf;b2++) cudaStreamWaitEvent(g_mgbufst[b2],fgEv,0);
            } else cudaMemcpy(g_fgrng,&rolloutRng,8,cudaMemcpyHostToDevice); }
          unsigned long long key=1469598103934665603ULL;
          #define KH(v) do{ key^=(unsigned long long)(v); key*=1099511628211ULL; }while(0)
          KH(N);KH(D);KH(H);KH(L);KH(W);KH(K);KH(T);KH(nbuf);KH(g_mgbp.obsKind);KH(g_mgbp.f32obs);KH(0x4D44ULL);   /* "MD" tag: never collide with the single-discrete table */
          KH((uintptr_t)dP);KH((uintptr_t)dObsF);KH((uintptr_t)g_mgbp.hObsF32);KH((uintptr_t)g_mgbp.hObsF32B);
          KH((uintptr_t)dO);KH((uintptr_t)g_mgbp.actPin);KH((uintptr_t)g_mgbp.termPlane);KH((uintptr_t)dMGTraj);
          KH((uintptr_t)g_dcAct);KH((uintptr_t)g_dcLogp);KH((uintptr_t)g_dcVal0);KH((uintptr_t)g_dcValue);
          KH((uintptr_t)dSa);KH((uintptr_t)dHs);
          #undef KH
          long need=(long)T*nbuf;
          if(key!=g_fgg_key || need>g_fgg_cap || !g_fgg){
            if(g_fgg){ for(long i=0;i<g_fgg_cap;i++) if(g_fgg_ok[i]==1) cudaGraphExecDestroy(g_fgg[i]);
              free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
            g_fgg=(cudaGraphExec_t*)calloc((size_t)need,sizeof(cudaGraphExec_t));
            g_fgg_ok=(signed char*)calloc((size_t)need,1);
            if(g_fgg && g_fgg_ok){ g_fgg_cap=need; g_fgg_key=key; }
            else { free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
          }
        }
      } else if(g_fgg){
        for(long i=0;i<g_fgg_cap;i++) if(g_fgg_ok[i]==1) cudaGraphExecDestroy(g_fgg[i]);
        free(g_fgg); free(g_fgg_ok); g_fgg=NULL; g_fgg_ok=NULL; g_fgg_cap=0; }
    }
    memset(g_mgbp.prof,0,sizeof(g_mgbp.prof));
    if(mgbufw_init()){
      pthread_barrier_wait(&g_mgbufw_bar);                            /* release all MAXBUF_MG workers */
      pthread_barrier_wait(&g_mgbufw_bar);                            /* join */
    } else {
      pthread_t bth[MAXBUF_MG];                                       /* pool init failed once — permanent fallback */
      for(int b=0;b<nbuf;b++) pthread_create(&bth[b],NULL,mg_buf_worker,(void*)(long)b);
      for(int b=0;b<nbuf;b++) pthread_join(bth[b],NULL);
    }
    cudaDeviceSynchronize();
    if(getenv("PUFFER_ROLL_PROFILE")){
      static const char* nm[8]={"cast","launch","sync","act","env","scat","tail","queue"};
      double agg[8]={0}; for(int b=0;b<nbuf;b++) for(int k=0;k<8;k++) agg[k]+=g_mgbp.prof[b][k];
      fprintf(stderr,"[mg-cyc-md]"); for(int k=0;k<8;k++) fprintf(stderr," %s=%.1f",nm[k],agg[k]/nbuf);
      fprintf(stderr," ms/buffer (cumulative this update, %d buffers)\n",nbuf);
    }
    g_dMGObsTraj_valid=1; g_dMGObsTraj_N=(long)N;                     /* obs trajectory ready for the BPTT gather */
    /* final obs for the bootstrap: pinned ping-pong at parity (carried across chained updates) */
    float* fObs32=NULL; const double* cur=NULL;
    if(g_mgbp.f32obs) fObs32=((g_mgbp.startPc+T)%2)? g_mgbp.hObsF32B : g_mgbp.hObsF32;
    else cur=(T%2==1)?hA:hB;                                          /* workers ran T swaps obs0→hA→hB→… */
    if(fObs32){
      if(g_mgbp.obsKind==1){ cudaMemcpy(dObs,fObs32,(size_t)N*D,cudaMemcpyHostToDevice);
        k_u82f<<<GR(N*D),B>>>(dObsF,(const unsigned char*)dObs,(long)N*D); }
      else if(g_mgbp.obsKind==2){ cudaMemcpy(dObs,fObs32,2*(size_t)N*D,cudaMemcpyHostToDevice);
        k_bf2f<<<GR(N*D),B>>>(dObsF,(const __nv_bfloat16*)dObs,(long)N*D); }
      else cudaMemcpy(dObsF,fObs32,4*(size_t)N*D,cudaMemcpyHostToDevice);
    } else { cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice); k_d2f<<<GR(N*D),B>>>(dObsF,dObs,(long)N*D); }
    mingru_fwd_dev(hbl,dP,dObsF,dSa,dSb,dHb,dHn,dYb,dLg,dVal,(int)N,(int)D,(int)H,(int)L,(int)W,bf,0);
    k_f2d<<<GR(N),B>>>(g_dcBoot,dVal,(long)N);                        /* boot column device-direct */
    /* whole-update rew/term column scatter from the pinned planes (replaces T·nbuf per-step scatters) */
    k_mg_cols_rt_all<<<GR(NT),B>>>(g_mgbp.rewPlane,g_mgbp.termPlane,g_dcRew,g_dcTerm,(long)N,(long)T);
    g_dc_valid=1; g_dc_fromroll=1; g_mbSinceRoll=0; g_rollIdx++;      /* columns stamped — Lean skips prep */
    if(finalState || bootVals) cudaDeviceSynchronize();
    if(bootVals){ cudaMemcpy(hVal,dVal,4*N,cudaMemcpyDeviceToHost);   /* dVal is f32 — read as f32 then widen */
      float* hValF=(float*)hVal; for(long i=N-1;i>=0;i--) bootVals[i]=(double)hValF[i]; }
    if(finalObs){
      if(fObs32){
        if(g_mgbp.obsKind==1){ const unsigned char* q=(const unsigned char*)fObs32;
          for(size_t i=0;i<N*D;i++) finalObs[i]=(double)q[i]; }
        else if(g_mgbp.obsKind==2){ const unsigned short* q=(const unsigned short*)fObs32;
          for(size_t i=0;i<N*D;i++){ union{float f;unsigned u;} v; v.u=((unsigned)q[i])<<16; finalObs[i]=(double)v.f; } }
        else for(size_t i=0;i<N*D;i++) finalObs[i]=(double)fObs32[i];
      } else for(size_t i=0;i<N*D;i++) finalObs[i]=cur[i];
    }
    if(finalState){ cudaMemcpy(hStF,dSa,4*N*LH,cudaMemcpyDeviceToHost);
      for(long i=0;i<N*LH;i++) finalState[i]=(double)hStF[i]; }
    /* PUFFER_MG_H0_VALUE (gate, DEFAULT OFF): re-derive g_dcValue/g_dcBoot at h=0-per-segment. Placed
       AFTER every host read of dVal/dSa/dObsF above, so with the gate off nothing here executes and
       with it on nothing the rollout returns to Lean is disturbed — only the two value buffers the
       V-Trace/target path reads. */
    mg_h0_value_refresh(hbl,dP,dMGTraj,dObsF,g_mgbp.termPlane,(long)N,(long)T,(long)D,
                        (int)H,(int)L,(int)W,LH,bf);
    /* resident-chain stamp: obs live in the pinned ping-pong at parity (startPc+T)%2, state in dSa.
       GATED ON EVEN T: chaining advances startPc by T per update, so an ODD T would flip every step's
       ping-pong parity between updates — and each (t,buf) graph BAKES the obs pointer it captured
       (zero-copy pinned staging), so the replay would read the stale half. Even T keeps startPc
       constant, which is what makes the captured pointer valid for the whole run. (The wide-bf16 arm
       carries the same T%2 gate for the same reason.) */
    { int chainGate; { const char* e=getenv("PUFFER_MG_CHAIN"); chainGate=(e==NULL||e[0]!='0'); }
      if(chainGate && g_mgbp.f32obs && (T%2)==0){
        g_mgchain.valid=1; g_mgchain.md=1; g_mgchain.pc=(int)((g_mgbp.startPc+T)%2);
        g_mgchain.N=(long)N; g_mgchain.D=(long)D; g_mgchain.LH=LH;
        g_mgchain.hA=g_mgbp.hObsF32; g_mgchain.hB=g_mgbp.hObsF32B; g_mgchain.dSa=dSa;
      } else g_mgchain.valid=0; }
    #undef GR
  }
  free(hStF); free(hVal);   /* the pinned staging is the persistent mg_hb cache — not freed */
  lean_dec(obs0a); lean_dec(state0a); lean_dec(hsA);
  return lean_io_result_mk_ok(Oo);
}

/* ---- Batched (PufferLib fused-scan-style) MinGRU BPTT kernels -------------------------------------
   PufferLib's fused MinGRU trains the WHOLE [B,T,H] sequence with ONE big projection GEMM per layer
   ([3H×(B·T)×H]) feeding a parallel scan, instead of T tiny per-timestep GEMMs ([3H×B×H]). We adopt the
   same batched-GEMM structure (which is the dominant BPTT cost — the per-step [3H×B×H] GEMMs are far too
   small to saturate the tensor cores) while keeping our VERIFIED per-step gate cell for the recurrence
   itself (which threads a per-layer [B·H] state and handles episode resets exactly as the sequential
   path did). Same gate math ⇒ per-element bit-identical projections; only the weight-grad reductions sum
   over B·T rows at once instead of T×B accumulations (a reassociation, ≤1e-4 like the parallel-fold).
   Layer-major activation layout: slc=(l·T+t)·B·H, so each layer's T·B·H block is GEMM-contiguous. */

/* PPO head grad over ALL T·B rows at once (row r=t·B+b). Writes dl[r·A], dvalue[r], newlp/newval[r]. */
__global__ void k_mg_ppo_b(const float* lgAll, const float* valAll, const double* acts, const double* advs,
   const double* rets, const double* olds, const double* ovs, float* dl, float* dvalue,
   float* newlpAll, float* newvalAll, int B, int T, int A, float vfCoef, float entCoef, float clipEps, float vfClip){
  long r=(long)blockIdx.x*blockDim.x+threadIdx.x; if(r>=(long)T*B) return;
  const float* logits=lgAll+r*A;
  float mx=logits[0]; for(int k=1;k<A;k++) if(logits[k]>mx) mx=logits[k];
  float se=0.0f; for(int k=0;k<A;k++) se+=expf(logits[k]-mx); float lse=mx+logf(se);
  float pout=0.0f; float pk[32]; for(int k=0;k<A;k++){ pk[k]=expf(logits[k]-lse); pout+=pk[k]*logits[k]; }
  int a=(int)acts[r]; float adv=(float)advs[r], ret=(float)rets[r];
  float oldLogp=(float)olds[r], vold=(float)ovs[r];
  float logpA=logits[a]-lse; float ratio=expf(logpA-oldLogp); float lo=1.0f-clipEps,hi=1.0f+clipEps;
  newlpAll[r]=logpA; newvalAll[r]=valAll[r];
  float ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); float surr1=adv*ratio,surr2=adv*ratioC;
  float dPol; if(surr1<=surr2) dPol=adv*ratio; else { float cg=(lo<ratio&&ratio<hi)?1.0f:0.0f; dPol=adv*cg*ratio; }
  float vnew=valAll[r], dvl;
  if(vfClip>0.0f){ float dd=vnew-vold; float vc=vold+(dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd));
    float du=(vnew-ret)*(vnew-ret),cc=(vc-ret)*(vc-ret);
    if(du>=cc) dvl=vnew-ret; else if(dd>-vfClip&&dd<vfClip) dvl=vnew-ret; else dvl=0.0f; } else dvl=vnew-ret;
  dvalue[r]=-vfCoef*dvl;
  for(int k=0;k<A;k++) dl[r*A+k]=dPol*(((k==a)?1.0f:0.0f)-pk[k])+entCoef*pk[k]*(pout-logits[k]);
}
/* MULTI-DISCRETE twin of k_mg_ppo_b (K categorical heads, sizes headSizes[K], logits width W=Σsizes).
   Exactly the decomposition k_ppo_dout_md uses on the MLP path: the joint log-prob is Σ_h log p_h(a_h),
   ONE PPO clip on the joint ratio, and the gradient factorises per head (dPol·(onehot(a_h)−p_h) plus the
   per-head entropy term). Differences from k_ppo_dout_md are only those forced by the recurrent path it
   lives in: the per-head softmax subtracts the head max (as k_mg_ppo_b does — k_ppo_dout_md does not),
   the value head is a SEPARATE tensor (valAll/dvalue) rather than logits column O−1, and it carries the
   value-loss CLIP (vfClip) and the new_logp/new_value writeback the replay iterate needs. The joint
   log-prob convention (max-subtracted per head, summed) matches k_sample_md, which produced oldLogp.
   NOT called when K==1 — the single-discrete path keeps k_mg_ppo_b bit-for-bit. */
__global__ void k_mg_ppo_b_md(const float* lgAll, const float* valAll, const double* acts, const double* advs,
   const double* rets, const double* olds, const double* ovs, float* dl, float* dvalue,
   float* newlpAll, float* newvalAll, int B, int T, int W, int K, const int* headSizes,
   float vfCoef, float entCoef, float clipEps, float vfClip){
  long r=(long)blockIdx.x*blockDim.x+threadIdx.x; if(r>=(long)T*B) return;
  const float* logits=lgAll+r*(long)W;
  float adv=(float)advs[r], ret=(float)rets[r], oldLogp=(float)olds[r], vold=(float)ovs[r];
  /* Joint log-prob in DOUBLE, in the same algebraic form the sampler uses. k_sample_md — which
     produced the `oldLogp` this ratio differentiates — accumulates in double as log(exp(x-m)/z);
     this accumulated K per-head terms in FLOAT as x-(m+log z). Mathematically identical, numerically
     not, and the error compounds across heads BEFORE the ratio is formed: the MD BPTT check read
     max rel 0.325 on the bf16 tier where single-discrete is orders better. The logits are still f32
     here vs f64 in the sampler, so this cannot be exact — it removes the asymmetry, not the tier. */
  double jointLogp=0.0; int off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh]; int a=(int)acts[r*(long)K+hh];
    float mx=logits[off]; for(int k=1;k<sz;k++) if(logits[off+k]>mx) mx=logits[off+k];
    float se=0.0f; for(int k=0;k<sz;k++) se+=expf(logits[off+k]-mx);
    jointLogp += log(exp((double)logits[off+a]-(double)mx)/(double)se); off+=sz; }
  float ratio=(float)exp(jointLogp-(double)oldLogp); float lo=1.0f-clipEps,hi=1.0f+clipEps;
  newlpAll[r]=jointLogp; newvalAll[r]=valAll[r];
  float ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); float surr1=adv*ratio,surr2=adv*ratioC;
  float dPol; if(surr1<=surr2) dPol=adv*ratio; else { float cg=(lo<ratio&&ratio<hi)?1.0f:0.0f; dPol=adv*cg*ratio; }
  float vnew=valAll[r], dvl;
  if(vfClip>0.0f){ float dd=vnew-vold; float vc=vold+(dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd));
    float du=(vnew-ret)*(vnew-ret),cc=(vc-ret)*(vc-ret);
    if(du>=cc) dvl=vnew-ret; else if(dd>-vfClip&&dd<vfClip) dvl=vnew-ret; else dvl=0.0f; } else dvl=vnew-ret;
  dvalue[r]=-vfCoef*dvl;
  off=0;
  for(int hh=0;hh<K;hh++){ int sz=headSizes[hh]; int a=(int)acts[r*(long)K+hh];
    float mx=logits[off]; for(int k=1;k<sz;k++) if(logits[off+k]>mx) mx=logits[off+k];
    float se=0.0f; for(int k=0;k<sz;k++) se+=expf(logits[off+k]-mx); float lse=mx+logf(se);
    /* no pk[] scratch: heads are unbounded here (k_ppo_dout_md caps at 64), so p is recomputed */
    float pout=0.0f; for(int k=0;k<sz;k++) pout+=expf(logits[off+k]-lse)*logits[off+k];
    for(int k=0;k<sz;k++){ float p=expf(logits[off+k]-lse);
      dl[r*(long)W+off+k]=dPol*(((k==a)?1.0f:0.0f)-p)+entCoef*p*(pout-logits[off+k]); }
    off+=sz; }
}
/* FUSED whole-T scan cells: the recurrence is sequential in t but elementwise over (b,j), so ONE launch of
   B·H threads walking t internally replaces the 2·T-launch loop (gate+reset / gate+donext). Per-element op
   order is IDENTICAL to the loop ⇒ bit-identical; the recurrent state/state-grad live in registers (the
   stateL/dOnextL/dprevL globals disappear). ~4·T fewer launches per layer per call. */
/* DEDUP'd scan pair (f32 path). The original fwd scan stored SIX activation planes per element-step, but
   five were duplicates of data already resident elsewhere: hid/gate/proj are byte-copies of the layer GEMM
   output (kept per-layer in aCmb now that the GEMM writes there directly), hin is the layer's own input
   (kept as the aHinC chain: each scan writes `out` straight into the NEXT layer's input slice, so nothing
   is ever overwritten and no copy is needed), and prev is oS[t-1] gated by terms[t-1] (recomputed in bwd
   from the oS it already reads). fwd stores 7→2 per element-step (44B→24B of traffic on a bandwidth-floor
   kernel) and the final-hidden k_cpyf dies because chain slice L IS the head input. Same values, same
   per-element op order ⇒ bit-identical to the old pair. */
__global__ void k_mg_scan_fwd2(const float* cmb, const float* in, float* out, float* oS,
   const double* terms, int B,int H,int T, const float* h0=NULL, int lay=0, int LH=0){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H), j=(int)(idx%H);
  /* h0==NULL ⇒ the shipped convention (segment restarts at 0); callers always pass NULL. The h0
     branch (rollout-start state for this segment/layer) is retained as a safe no-op. */
  float state=h0? h0[(long)b*LH+(long)lay*H+j] : 0.0f;
  for(int t=0;t<T;t++){
    long sofs=(long)t*B*H+idx;
    const float* y=cmb+(long)t*B*3*H+(long)b*3*H;
    float hid=y[j], gate=y[H+j], proj=y[2*H+j];
    float z=d_sigf(gate), gg=d_gactf(hid), prev=state;
    /* explicit FMA under the tolerance tier only (c_mgFma; --fmad=false anchors the f32 tier's
       bit-exactness) — contraction halves the FLOP latency and IMPROVES accuracy */
    float o, hg=d_sigf(proj), hinj=in[sofs];
    if(c_mgFma){ o=__fmaf_rn(z,gg,__fmaf_rn(-z,prev,prev));
      out[sofs]=__fmaf_rn(hg,o,__fmaf_rn(-hg,hinj,hinj)); }
    else { o=(1.0f-z)*prev+z*gg;
      out[sofs]=hg*o+(1.0f-hg)*hinj; }
    oS[sofs]=o;
    state=o;
    if(terms[(long)t*B+b]!=0.0) state=0.0f;
  }
}
__global__ void k_mg_scan_bwd2(const float* cmb, const float* oS, const float* in,
   const float* dhnIn, float* dy, float* dhnOut, const double* terms, int B,int H,int T,
   const float* h0=NULL, int lay=0, int LH=0){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H || T<=0) return;
  int b=(int)(idx/H), j=(int)(idx%H);
  float dOnext=0.0f;
  /* 2-stage software pipeline: the walk is BACKWARD in t, which defeats the forward-biased hardware
     prefetchers and left every load's latency exposed on the serial dOnext chain (measured ~60% of the
     bandwidth floor vs ~95% for the forward scan). Each iteration issues the NEXT (t-1) iteration's
     loads before computing its own, and o(t-1) is carried in a register — it is exactly this
     iteration's prev-candidate (oPrev), so the oS re-read disappears. Same FP ops in the same order
     per element ⇒ bit-identical; only load scheduling changes. */
  long sofs=(long)(T-1)*B*H+idx;
  long dyb=(long)(T-1)*B*3*H+(long)b*3*H;
  float hid=cmb[dyb+j], gate=cmb[dyb+H+j], proj=cmb[dyb+2*H+j];
  float o=oS[sofs], hin=in[sofs], dhnj=dhnIn[sofs];
  float oPrev=(T>1)? oS[sofs-(long)B*H] : 0.0f;
  for(int t=T-1;t>=0;t--){
    float nhid=0.0f,ngate=0.0f,nproj=0.0f,nhin=0.0f,ndhn=0.0f,noPrev=0.0f;
    long nsofs=sofs-(long)B*H, ndyb=dyb-(long)B*3*H;
    if(t>0){
      nhid=cmb[ndyb+j]; ngate=cmb[ndyb+H+j]; nproj=cmb[ndyb+2*H+j];
      nhin=in[nsofs]; ndhn=dhnIn[nsofs];
      noPrev=(t>1)? oS[nsofs-(long)B*H] : 0.0f;
    }
    /* t==0's predecessor is the SEGMENT's initial state: 0 under the shipped convention (h0 always
       NULL; the h0 branch is retained as a safe no-op, dprev at t=0 is dropped either way). */
    float prev=(t>0)? ((terms[(long)(t-1)*B+b]==0.0)? oPrev : 0.0f)
                    : (h0? h0[(long)b*LH+(long)lay*H+j] : 0.0f);
    float z=d_sigf(gate), hg=d_sigf(proj), g=(hid>=0.0f)?(hid+0.5f):d_sigf(hid);
    float dhg=dhnj*(o-hin);
    float do_=dhnj*hg + dOnext;
    dhnOut[sofs]=dhnj*(1.0f-hg);
    float dz=do_*(g-prev);
    float dprev=do_*(1.0f-z);
    float dg=do_*z, dgate=dz*z*(1.0f-z), dproj=dhg*hg*(1.0f-hg);
    float sg=(hid>=0.0f)?1.0f:(d_sigf(hid)*(1.0f-d_sigf(hid)));
    dy[dyb+j]=dg*sg; dy[dyb+H+j]=dgate; dy[dyb+2*H+j]=dproj;
    dOnext=(t>0 && terms[(long)(t-1)*B+b]==0.0)?dprev:0.0f;
    hid=nhid; gate=ngate; proj=nproj; hin=nhin; dhnj=ndhn;
    o=oPrev; oPrev=noPrev;
    sofs=nsofs; dyb=ndyb;
  }
}
/* bf16-storage twins of the fused scans (same math, activations in bf16, recurrent state/grad in f32 regs) */
__global__ void k_mg_scan_fwd_bf(const bf16* cmb, const bf16* in, bf16* out,
   bf16* hidS,bf16* gateS,bf16* projS,bf16* oS,bf16* hinS,bf16* prevS,
   const double* terms, int B,int H,int T){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H), j=(int)(idx%H);
  float state=0.0f;
  for(int t=0;t<T;t++){
    long sofs=(long)t*B*H+idx;
    const bf16* y=cmb+(long)t*B*3*H+(long)b*3*H;
    float hid=__bfloat162float(y[j]), gate=__bfloat162float(y[H+j]), proj=__bfloat162float(y[2*H+j]);
    float z=d_sigf(gate), gg=d_gactf(hid), prev=state;
    float o=(1.0f-z)*prev+z*gg, hg=d_sigf(proj), hinj=__bfloat162float(in[sofs]);
    out[sofs]=__float2bfloat16(hg*o+(1.0f-hg)*hinj);
    hidS[sofs]=__float2bfloat16(hid); gateS[sofs]=__float2bfloat16(gate); projS[sofs]=__float2bfloat16(proj);
    oS[sofs]=__float2bfloat16(o); hinS[sofs]=__float2bfloat16(hinj); prevS[sofs]=__float2bfloat16(prev);
    state=o;
    if(terms[(long)t*B+b]!=0.0) state=0.0f;
  }
}
__global__ void k_mg_scan_bwd_bf(const bf16* hidS,const bf16* gateS,const bf16* projS,const bf16* oS,
   const bf16* prevS,const bf16* hinS,const bf16* dhnIn, bf16* dy, bf16* dhnOut,
   const double* terms, int B,int H,int T){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H), j=(int)(idx%H);
  float dOnext=0.0f;
  for(int t=T-1;t>=0;t--){
    long sofs=(long)t*B*H+idx;
    float hid=__bfloat162float(hidS[sofs]), gate=__bfloat162float(gateS[sofs]), proj=__bfloat162float(projS[sofs]);
    float o=__bfloat162float(oS[sofs]), prev=__bfloat162float(prevS[sofs]), hin=__bfloat162float(hinS[sofs]);
    float z=d_sigf(gate), hg=d_sigf(proj), g=(hid>=0.0f)?(hid+0.5f):d_sigf(hid);
    float dhnj=__bfloat162float(dhnIn[sofs]);
    float dhg=dhnj*(o-hin);
    float do_=dhnj*hg + dOnext;
    dhnOut[sofs]=__float2bfloat16(dhnj*(1.0f-hg));
    float dz=do_*(g-prev);
    float dprev=do_*(1.0f-z);
    float dg=do_*z, dgate=dz*z*(1.0f-z), dproj=dhg*hg*(1.0f-hg);
    float sg=(hid>=0.0f)?1.0f:(d_sigf(hid)*(1.0f-d_sigf(hid)));
    long dyb=(long)t*B*3*H+(long)b*3*H;
    dy[dyb+j]=__float2bfloat16(dg*sg); dy[dyb+H+j]=__float2bfloat16(dgate); dy[dyb+2*H+j]=__float2bfloat16(dproj);
    dOnext=(t>0 && terms[(long)(t-1)*B+b]==0.0)?dprev:0.0f;
  }
}

/* Persistent device-buffer cache for the MinGRU BPTT gradient — the ~41 scratch/activation/accumulator
   buffers were cudaMalloc/cudaFree'd EVERY minibatch call (incl. ~25MB of activation tensors). Sizes are
   constant within a run (fixed T,B,H,D,L,A), so alloc once and reuse. Stable addresses across calls are
   also the prerequisite for capturing the forward/backward loop as a CUDA graph. */
static void* g_bg[112]; static size_t g_bgsz[112]; /* 0–54 f32 BPTT; 55–68 bf16-storage twins;
   69 advpart partials; 70 resident grad; 71–77 device V-Trace/prio sampler; 78–79 dedup'd scan
   activations (aCmb/aHinC — f32 path only); 80 per-matrix Muon NS scratch slabs; 84-89 fused-muon
   segment tables + wenc pads; 90-99 the bf16-storage wide rollout arm (weights/head/activations/state/
   u8 staging/rng scalar); 103 multi-discrete head sizes (BPTT, K>1 only) */
/* resident-weight handoff (muon → BPTT): muon casts its fresh f64 weights into dP (bg 0) on-device and
   stamps these; the grad skips its host cast+H2D when the params it was passed provably match (w0). */
static int g_mgw_fresh=0; static size_t g_mgw_P=0; static double g_mgw_w0=0;
static int g_mgw_empty=0;   /* empty-pa handoff form: muon returned an EMPTY wFlat (no D2H) — the grad
                               trusts the stamp alone (there are no host params left to value-guard) */
/* resident-GRADIENT handoff (BPTT → muon): in device-column mode the grad packs its f32 accumulators
   into a resident f64 buffer (bg 70) and returns NO host copy (the old path D2H'd g[P] + 1MB of unused
   newlp/newval per minibatch); the muon then gradclips ON-DEVICE and consumes it directly. */
static int g_mgg_fresh=0; static size_t g_mgg_P=0;
/* on-device gradclip: single-block deterministic tree Σ(g·sc)² (fixed order ⇒ run-to-run reproducible;
   reassociates vs the Lean host fold ⇒ cc differs by ~1 ulp — tolerance-class), then scale into dOut. */
__global__ void k_gclip_norm(double* sumsq, const double* g, double sc, long P){
  __shared__ double sh[256]; int t=threadIdx.x;
  double s=0.0; for(long i=t;i<P;i+=blockDim.x){ double x=g[i]*sc; s+=x*x; } sh[t]=s; __syncthreads();
  for(int st=128;st>0;st>>=1){ if(t<st) sh[t]+=sh[t+st]; __syncthreads(); }
  if(t==0) sumsq[0]=sh[0]; }
__global__ void k_gclip_scale(double* out, const double* g, const double* sumsq, double maxNorm, double sc, long P){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=P) return;
  double gnorm=sqrt(sumsq[0]);
  double cc=((maxNorm>0.0 && gnorm>maxNorm)? maxNorm/gnorm : 1.0)*sc;
  out[i]=g[i]*cc; }
static void* bg(int i, size_t bytes){
  if(g_bgsz[i] < bytes){ if(g_bg[i]) cudaFree(g_bg[i]);
    if(cudaMalloc(&g_bg[i], bytes)!=cudaSuccess){ g_bg[i]=NULL; g_bgsz[i]=0; return NULL; }
    g_bgsz[i]=bytes; }
  return g_bg[i];
}

/* BPTT side stream: weight-grad GEMMs + column-sums only feed the final gradient pack — they are OFF
   the scan→dy·Wl→scan critical chain. NON-blocking stream (a blocking one would make every subsequent
   legacy launch implicitly wait, serializing the fork); ALL ordering is explicit events. The handle
   clones cu_handle()'s config (bit-identical GEMM algorithm selection — same rule as the muon pool). */
static cudaStream_t g_bs_st=NULL; static cublasHandle_t g_bs_h=NULL; static void* g_bs_ws=NULL; static int g_bs_bad=0;
static cudaEvent_t g_bs_ev[12]; static int g_bs_evN=0;   /* [0] fork, [1] join, [2+l] gWl WAR guards */
static void bs_teardown(void){   /* all-or-nothing (mu_pool rule): a partial init must not strand VRAM */
  if(g_bs_h){ cublasDestroy(g_bs_h); g_bs_h=NULL; }
  if(g_bs_ws){ cudaFree(g_bs_ws); g_bs_ws=NULL; }
  if(g_bs_st){ cudaStreamDestroy(g_bs_st); g_bs_st=NULL; }
  while(g_bs_evN>0){ g_bs_evN--; cudaEventDestroy(g_bs_ev[g_bs_evN]); }
}
static int bs_init(void){
  if(g_bs_bad) return 0;
  if(!g_cu_ws_ok){ g_bs_bad=1; return 0; }   /* main handle runs on the DEFAULT workspace (its 32MB alloc
    failed): a 32MB-workspace clone could select different GEMM algorithms than the sequential path —
    refuse the fork rather than silently break the side-on == side-off bit-identity invariant */
  if(!g_bs_st && cudaStreamCreateWithFlags(&g_bs_st,cudaStreamNonBlocking)!=cudaSuccess){ g_bs_st=NULL; g_bs_bad=1; return 0; }
  if(!g_bs_h){
    if(cublasCreate(&g_bs_h)!=CUBLAS_STATUS_SUCCESS){ g_bs_h=NULL; g_bs_bad=1; bs_teardown(); return 0; }
    cublasSetMathMode(g_bs_h,CUBLAS_DEFAULT_MATH);
    if(cudaMalloc(&g_bs_ws,32*1024*1024)!=cudaSuccess){ cudaGetLastError(); g_bs_ws=NULL;
      g_bs_bad=1; bs_teardown(); return 0; }
    cublasSetWorkspace(g_bs_h,g_bs_ws,32*1024*1024);
    cublasSetStream(g_bs_h,g_bs_st); }
  while(g_bs_evN<12){ if(cudaEventCreateWithFlags(&g_bs_ev[g_bs_evN],cudaEventDisableTiming)!=cudaSuccess){
      g_bs_bad=1; bs_teardown(); return 0; } g_bs_evN++; }
  return 1;
}
/* ===== resident-contract failure: ABORT, never a well-formed zero =====================================
   The MinGRU trainer's per-minibatch chain (BPTT grad → resident Muon) is a CONTRACT: Lean passes an
   EMPTY `obs`/`scal` (those live in the device-resident trajectory/columns) and an EMPTY `gClip` (the
   gradient stays on-device in bg 70). Every link of it is allocation-dependent, and pre-fix a failed
   allocation degraded SILENTLY: the grad returned an empty array (which Lean read as "resident"), the
   Muon found no resident gradient, warned once, took NO step, and returned a zero wFlat. The run kept
   printing updates, episode returns and SPS, finished, and EXITED 0 — while learning absolutely nothing.
   Reproduced on tetris/g2048 (N·T=524288, hidden 512, P≈1.7M) whenever the card is nearly full.

   Where a real gradient can still be produced we now degrade GRACEFULLY (the grad packs a HOST gradient
   and Lean clips it host-side — see the mode note in lean_cuda_mingru_ppo_grad). Where it cannot — no
   BPTT activation storage, or the resident obs/columns the caller no longer keeps a host copy of — there
   is no recovery short of re-sizing the whole update, so we raise a Lean IO ERROR. `main : IO Unit`
   propagates it to a stderr message and exit code 1: a crash is strictly better than a trainer that
   reports healthy numbers, exits 0, and learned nothing. */
static lean_obj_res mg_contract_fail(const char* what, const char* detail, size_t P){
  size_t freeB=0,totB=0; cudaMemGetInfo(&freeB,&totB);
  char buf[1200];
  snprintf(buf,sizeof(buf),
    "[puffer] *** MinGRU %s FAILED: %s\n"
    "[puffer] ***   P=%zu params, VRAM free %.2f / %.2f GB.\n"
    "[puffer] *** Aborting: continuing would have applied a ZERO gradient, i.e. printed healthy-looking\n"
    "[puffer] *** updates/SPS (and exited 0) while learning nothing.\n"
    "[puffer] *** Retry with fewer envs (--num-envs), a shorter --train.horizon, a smaller\n"
    "[puffer] *** --train.minibatch-size, PUFFER_MG_PREC=bf16, or free VRAM held by other processes.",
    what, detail, P, (double)freeB/1073741824.0, (double)totB/1073741824.0);
  fprintf(stderr,"%s\n",buf); fflush(stderr);
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
}
/* params(f64) flattenMG layout. obs [T·B·D], acts/advs/rets/oldlps/terms/oldvals [T·B] (f64). Returns the
   summed gradient g[P] (f64), P = H·D+H+L·3H·H+A·H+A+H+1. */
/* obsa may be empty when the caller relies on the device-resident obs trajectory (g_dMGObsTraj, set by the
   rollout): if segIdxa is non-empty and the trajectory is valid/large-enough, the sampled obs are gathered
   ON-DEVICE (no host gather, no f64 obs H2D). Otherwise obsa (host f64 [T·B·D]) is uploaded as before. */
/* MULTI-DISCRETE: `A` is the LOGITS WIDTH (W = Σ head sizes) — the whole BPTT is width-generic in it, so
   the encoder/layer/decoder/value GEMMs, the activation store and the Muon segment table all follow from
   A alone. `K` (>1) and `hsA` (the K head sizes) only redirect the PPO head kernel to k_mg_ppo_b_md and
   widen the action column to K per row. K==1 takes literally the same code path as before. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mingru_ppo_grad(
    lean_obj_arg pa, lean_obj_arg obsa, lean_obj_arg scala,
    size_t B, size_t T, size_t H, size_t D, size_t L, size_t A, size_t K, lean_obj_arg hsA,
    double vfCoef, double entCoef, double clipEps, double vfClip, lean_obj_arg segIdxa, lean_obj_arg mbPrioa,
    lean_obj_arg wio){
  (void)wio;
  size_t wEncSz=H*D, layerSz=3*H*H, O=A+1, P=wEncSz+H+L*layerSz+A*H+A+H+1;
  size_t Kh=(K<1)?1:K;
  const double* pp=lean_float_array_cptr(pa);
  const double* obsd=lean_float_array_cptr(obsa);
  /* scal = packed [(K+5)·T·B] = [act(K-wide)|adv|ret|old|term|ov]; slice the six (one Lean buffer
     instead of six). K=1 ⇒ the classic [6·T·B] offsets, unchanged. */
  const double* scald=lean_float_array_cptr(scala); size_t nbs=T*B;
  const double* actd=scald; const double* advd=scald+Kh*nbs; const double* retd=scald+(Kh+1)*nbs;
  const double* oldd=scald+(Kh+2)*nbs; const double* termd=scald+(Kh+3)*nbs; const double* ovd=scald+(Kh+4)*nbs;
  size_t segN=lean_sarray_size(segIdxa);
  const double* segIdxd = segN>0 ? lean_float_array_cptr(segIdxa) : NULL;
  /* device-column mode: gather the 6 scalars on-device from the resident columns (V-Trace/gather/iterate all
     on GPU); `scala` is empty then. Else H2D the packed host `scala`. */
  /* resident-prio mode: BOTH segIdx and mbPrio empty + a fresh device-sampler stamp (consumed here) —
     the sampled indices/weights already live on-device (bg 75/76), no H2D at all. */
  int prioRes = (segN==0 && lean_sarray_size(mbPrioa)==0 && g_prio_fresh && g_prioB==(long)B && g_prioN==g_dcN);
  g_prio_fresh=0;
  /* g_dcK guard: a K-wide action column must not be gathered as if it were scalar (or vice versa) —
     mg_prep records the width it uploaded, and a mismatch drops out of device-column mode rather than
     silently reading every K-th action. */
  int devCol = ((segN>0 && lean_sarray_size(mbPrioa)>0) || prioRes) && g_dc_valid && g_dcT==(long)T
               && g_dcK==(long)Kh;
  const double* mbPriod = (devCol && !prioRes) ? lean_float_array_cptr(mbPrioa) : NULL;
  /* on-device obs gather when the rollout left a valid, big-enough trajectory (else H2D the host obsa) */
  int useTraj = (segN>0 || prioRes) && g_dMGObsTraj_valid && g_dMGObsTraj && g_dMGObsTrajSz>=(size_t)4*(size_t)g_dMGObsTraj_N*T*D;
  /* ---- MODE: this FFI is AUTHORITATIVE, and it decides LATE ----------------------------------------
     The RETURNED SIZE *is* the mode, and Lean reads it (no independent guess on the Lean side, so the
     two can never disagree):
       size 0        ⇒ RESIDENT: the summed gradient stayed on-device in bg(70); call the Muon with an
                       EMPTY gClip (it gradclips + consumes it there). The fast path — bit-identical.
       size P+2·T·B  ⇒ HOST: the resident buffer was unavailable; g[0..P) is the real summed gradient
                       (plus new_logp/new_value), and Lean must gradclip it and pass it to the Muon.
     The old code fixed OUTP=0 from `devCol` alone, HERE, before a single device buffer had been
     allocated — so any later allocation failure returned an empty array that Lean still read as
     "resident", and the Muon then stepped with nothing. The decision now happens next to the bg(70)
     attempt, after the whole BPTT has actually run, and nothing that can fail sits between it and the
     return. When no gradient could be produced at all we raise an IO error instead (mg_contract_fail). */
  lean_object* go=NULL; double* g=NULL;
  /* The two EMPTY-INPUT contracts, checked rather than assumed: an empty `scal` only means anything if
     the resident scalar columns are live (devCol), and an empty `obs` only if the resident obs
     trajectory is (useTraj). Pre-fix both were read regardless — a 6·T·B / T·B·D out-of-bounds read of a
     zero-length Lean array — because the caller can no longer supply a host copy of either. */
  const char* fail=NULL;
  if(lean_sarray_size(scala)==0 && !devCol)
    fail="empty `scal` but the device-resident scalar columns are gone (g_dc_valid=0 or T mismatch): the "
         "rollout could not stamp them and mg_prep's ~10*N*T*8B column allocation failed";
  else if(lean_sarray_size(obsa)==0 && !useTraj)
    fail="empty `obs` but the device-resident obs trajectory is gone: the rollout's g_dMGObsTraj "
         "(4*N*T*D bytes) allocation failed or is too small";
  cublasHandle_t hbl=cu_handle();
  /* resident-weight fast path: the muon step already cast its fresh f64 weights into dP (bg 0) on-device;
     skip the host cast + 250KB H2D when the passed params provably match (w0 guard). */
  /* two handoff forms: value-guarded (non-empty pa: pp[0] must equal the w0 the muon recorded) and the
     EMPTY-pa contract (the resident muon returns an empty wFlat instead of a 256KB D2H — valid only
     while its freshness stamp holds; anything else refuses loudly rather than training on stale weights). */
  int paEmpty=(lean_sarray_size(pa)==0);
  int wFresh=(g_mgw_fresh && g_mgw_P==P &&
              (paEmpty ? g_mgw_empty : (lean_sarray_size(pa)>=P && pp[0]==g_mgw_w0)));
  int paBad=(paEmpty && !wFresh);
  if(paBad) fprintf(stderr,"[puffer] mingru grad: empty params without a resident-muon stamp — refusing\n");
  float* hP=NULL;
  if(!wFresh && !paBad){ hP=(float*)malloc(4*P); for(size_t i=0;i<P;i++) hP[i]=(float)pp[i]; }
  /* device params (f32) + inputs */
  size_t TLBH=T*L*B*H;
  /* PERSISTENT device buffers (bg cache) — allocated once, reused across BPTT calls (was ~41 cudaMalloc +
     cudaFree per minibatch, incl. ~25MB activations). Accumulators are memset to 0 each call and inputs
     re-uploaded, so reuse is bit-identical. */
  float *dP=(float*)bg(0,4*P);
  /* dObs is the HOST-obs staging buffer, read ONLY when !useTraj. In the trainer useTraj is always
     true (the rollout leaves the obs trajectory device-resident), so allocating it unconditionally
     burned 8*T*B*D bytes for nothing -- 122MB on tetris, 31MB on breakout -- and that is exactly the
     VRAM pressure that pushes the BPTT's own buffers into the allocation-failure path this function
     now has to abort on. Allocate it only when it will actually be read. */
  double *dObs=useTraj? NULL : (double*)bg(1,8*T*B*D),*dAct=(double*)bg(3,8*T*B*Kh),*dAdv=(double*)bg(4,8*T*B),*dRet=(double*)bg(5,8*T*B),*dOld=(double*)bg(6,8*T*B),*dTrm=(double*)bg(7,8*T*B),*dOv=(double*)bg(8,8*T*B);
  /* multi-discrete head sizes (device); K==1 never allocates or reads it */
  int* dHs=NULL; int hsHost[256]; hsHost[0]=(int)A;   /* host copy, read only by the entropy diagnostic */
  if(Kh>1){ dHs=(Kh<=256)? (int*)bg(103,sizeof(int)*Kh) : NULL;
    if(dHs && lean_sarray_size(hsA)>=Kh){ int hh4[256];
      const double* hsd=lean_float_array_cptr(hsA);
      for(size_t i=0;i<Kh;i++) hh4[i]=(int)hsd[i];
      memcpy(hsHost,hh4,sizeof(int)*Kh);
      cudaMemcpy(dHs,hh4,sizeof(int)*Kh,cudaMemcpyHostToDevice); }
    else dHs=NULL; }
  float *dObsF=(float*)bg(2,4*T*B*D);   /* the batched path (below) supersedes the old per-timestep scratch (bg 9–22) */
  float *aO=(float*)bg(26,4*TLBH),*aLg=(float*)bg(30,4*T*B*A),*aVal=(float*)bg(31,4*T*B);
  float *aHfin=NULL;   /* bf16 path only (f32 boundary cast of the final hidden); f32 path reads the chain */
  /* dedup'd f32 activation storage (see k_mg_scan_fwd2): per-layer GEMM outputs written in place (aCmb)
     + the layer-input chain (aHinC: slice 0 = encoder output, slice l+1 = layer l's output, slice L = the
     final hidden the heads read). Replaces aHid/aGate/aProj/aHin/aPrev/aHfin/dCmb (bg 23-25/27-29/43) —
     5 of the 6 stored planes were duplicates; net −133MB VRAM at the flagship config. f32 path only, so
     allocated AFTER bf is finalized below (the bf16 branch keeps its own 6-plane twins). */
  float *aCmb=NULL,*aHinC=NULL;
  float *gWEnc=(float*)bg(32,4*wEncSz),*gBEnc=(float*)bg(33,4*H),*gLay=(float*)bg(34,4*L*layerSz),*gWDec=(float*)bg(35,4*A*H),*gBDec=(float*)bg(36,4*A),*gWVal=(float*)bg(37,4*H),*gBVal=(float*)bg(38,4);
  float *dNewlp=(float*)bg(39,4*T*B),*dNewval=(float*)bg(40,4*T*B);
  /* batched (fused-scan-style) buffers: whole [T·B,·] tensors so each GEMM covers all T at once. */
  float *dIn=(float*)bg(41,4*T*B*H),*dOut=(float*)bg(42,4*T*B*H);   /* bf16-path f32 boundary scratch */
  float *dDlAll=(float*)bg(44,4*T*B*A),*dDvalAll=(float*)bg(45,4*T*B),*dDhfAll=(float*)bg(46,4*T*B*H);
  float *dDyAll=(float*)bg(47,4*T*B*3*H),*dDhnOut=(float*)bg(48,4*T*B*H);
  float *dCsPart=(float*)bg(49,4*CSUM_C*(H>A?H:A));   /* column-sum partials (the fused scans freed 49–51) */
  double *dSeg=(double*)bg(52,8*B);   /* sampled segment indices (device) for the on-device obs gather */
  int bf = mg_bf16store() && (H%2==0);   /* bf16 GEMM lds are H/3H — cuBLAS needs them even */
  if(mg_bf16store() && !bf){ static int warned=0; if(!warned){ warned=1;
    fprintf(stderr,"[puffer] PUFFER_MG_PREC=bf16 requested but disabled (H=%zu odd) — running f32\n",H); } }
  /* bf16-storage twins (allocated only when PUFFER_MG_PREC=bf16): weights + the activations that flow through
     the memory-bound LAYER-stack GEMMs. Encoder/heads/PPO stay f32 (boundary casts dInb, aHfin, dDhfAllb). */
  bf16 *dWlTb=NULL,*dInb=NULL,*dOutb=NULL,*dCmbb=NULL,*dDhfAllb=NULL,*dDyAllb=NULL,*dDhnOutb=NULL;
  bf16 *aHidb=NULL,*aGateb=NULL,*aProjb=NULL,*aOb=NULL,*aHinb=NULL,*aPrevb=NULL;
  if(bf){
    dWlTb=(bf16*)bg(55,2*L*layerSz); dInb=(bf16*)bg(57,2*T*B*H); dOutb=(bf16*)bg(58,2*T*B*H);
    dCmbb=(bf16*)bg(59,2*T*B*3*H); dDhfAllb=(bf16*)bg(60,2*T*B*H); dDyAllb=(bf16*)bg(61,2*T*B*3*H); dDhnOutb=(bf16*)bg(62,2*T*B*H);
    aHidb=(bf16*)bg(63,2*TLBH); aGateb=(bf16*)bg(64,2*TLBH); aProjb=(bf16*)bg(65,2*TLBH); aOb=(bf16*)bg(66,2*TLBH); aHinb=(bf16*)bg(67,2*TLBH); aPrevb=(bf16*)bg(68,2*TLBH);
    aHfin=(float*)bg(29,4*T*B*H);
    if(!(dWlTb&&dInb&&dOutb&&dCmbb&&dDhfAllb&&dDyAllb&&dDhnOutb&&aHidb&&aGateb&&aProjb&&aOb&&aHinb&&aPrevb&&aHfin)){
      bf=0; static int warned=0; if(!warned){ warned=1;
        fprintf(stderr,"[puffer] PUFFER_MG_PREC=bf16 requested but bf16 buffer alloc failed — running f32\n"); } }
  }
  if(!bf){ aCmb=(float*)bg(78,4*L*T*B*3*H); aHinC=(float*)bg(79,4*(L+1)*T*B*H); }
  int ok = (B>0 && !paBad && hbl!=NULL && dP&&(useTraj||dObs)&&dObsF&&dAct&&dAdv&&dRet&&dOld&&dTrm&&dOv
    &&aO&&aLg&&aVal&&(bf||(aCmb&&aHinC))
    &&gWEnc&&gBEnc&&gLay&&gWDec&&gBDec&&gWVal&&gBVal&&dNewlp&&dNewval
    &&dIn&&dOut&&dDlAll&&dDvalAll&&dDhfAll&&dDyAll&&dDhnOut&&dCsPart&&dSeg
    &&(Kh==1 || dHs));
  /* !ok = the BPTT itself cannot run: there is no gradient to fall back to (host or device), so this is
     the hard-abort case. Name the culprit group — the activation store (aCmb+aHinC, ~4·L·T·B·3H +
     4·(L+1)·T·B·H bytes, hundreds of MB at the flagship config) is by far the usual one. */
  if(!fail && !ok){
    static char why[512];
    snprintf(why,sizeof(why),
      "BPTT setup failed at B=%zu T=%zu H=%zu D=%zu L=%zu A=%zu — cublas=%s params=%s "
      "activation-store=%s grad-accum=%s bwd-scratch=%s io-scalars=%s%s",
      B,T,H,D,L,A, hbl?"ok":"FAILED", dP?"ok":"FAILED",
      ((bf||(aCmb&&aHinC))&&aO&&aLg&&aVal)?"ok":"FAILED",
      (gWEnc&&gBEnc&&gLay&&gWDec&&gBDec&&gWVal&&gBVal)?"ok":"FAILED",
      (dIn&&dOut&&dDlAll&&dDvalAll&&dDhfAll&&dDyAll&&dDhnOut&&dCsPart&&dSeg&&dNewlp&&dNewval)?"ok":"FAILED",
      ((useTraj||dObs)&&dObsF&&dAct&&dAdv&&dRet&&dOld&&dTrm&&dOv)?"ok":"FAILED",
      paBad?" params=STALE(empty pa without a resident-muon stamp)"
           :((Kh>1&&!dHs)?" head-sizes=FAILED(K>256, empty hs, or bg(103) alloc)":(B?"":" B=0")));
    fail=why;
  }
  if(fail) ok=0;
  if(ok){
    if(!wFresh) cudaMemcpy(dP,hP,4*P,cudaMemcpyHostToDevice);   /* else: muon already cast them into dP on-device */
    if(!useTraj) cudaMemcpy(dObs,obsd,8*T*B*D,cudaMemcpyHostToDevice);   /* host obs only when not device-resident */
    if(!devCol){                                                        /* host scalars: H2D the 6 packed slices */
      cudaMemcpy(dAct,actd,8*T*B*Kh,cudaMemcpyHostToDevice);
      cudaMemcpy(dAdv,advd,8*T*B,cudaMemcpyHostToDevice); cudaMemcpy(dRet,retd,8*T*B,cudaMemcpyHostToDevice);
      cudaMemcpy(dOld,oldd,8*T*B,cudaMemcpyHostToDevice); cudaMemcpy(dTrm,termd,8*T*B,cudaMemcpyHostToDevice);
      cudaMemcpy(dOv,ovd,8*T*B,cudaMemcpyHostToDevice);
    }
    const float* dWEnc=dP; const float* dBEnc=dP+wEncSz; const float* dLayers=dP+wEncSz+H;
    const float* dWDec=dLayers+L*layerSz; const float* dBDec=dWDec+A*H; const float* dWVal=dBDec+A; const float* dBVal=dWVal+H;
    int Bk=256;
    #define GG(x) ceildiv((long)(x),Bk)
    long TB=(long)T*(long)B;
    if(prioRes) dSeg=(double*)bg(75,8*B);                              /* resident sampler output — no H2D */
    else if(useTraj||devCol) cudaMemcpy(dSeg,segIdxd,8*B,cudaMemcpyHostToDevice);   /* sampled seg indices (obs + scalar gather) */
    float* dH0=NULL;
    if(useTraj) k_gather_mg_obs<<<GG(T*B*D),Bk>>>(dObsF,g_dMGObsTraj,dSeg,(long)B,(long)T,(long)D);
    else k_d2f<<<GG(T*B*D),Bk>>>(dObsF,dObs,(long)T*B*D);
    if(devCol){   /* gather the 6 scalars on-device from the resident columns (advMean/advStd parallel reduction) */
      double* dMbPrio = prioRes ? (double*)bg(76,8*B) : (double*)bg(53,8*B);
      double* dMs=(double*)bg(54,32);
      if(!prioRes) cudaMemcpy(dMbPrio,mbPriod,8*B,cudaMemcpyHostToDevice);
      cudaMemset(dMs,0,32);
      { int nBlk=GG(TB); double* dPart=(double*)bg(69,8*(size_t)nBlk);
        k_mg_advpart<<<nBlk,Bk>>>(g_dcAdv,dSeg,(int)B,(int)T,dMs,0,dPart);
        k_mg_ms_fin<<<1,1>>>(dMs,dPart,nBlk,(int)TB,0);
        k_mg_advpart<<<nBlk,Bk>>>(g_dcAdv,dSeg,(int)B,(int)T,dMs,1,dPart);
        k_mg_ms_fin<<<1,1>>>(dMs,dPart,nBlk,(int)TB,1); }
      k_mg_gather_scal<<<GG(TB),Bk>>>(g_dcAdv,g_dcValue,g_dcAct,g_dcLogp,g_dcTerm,dSeg,dMbPrio,dMs,
        dAct,dAdv,dRet,dOld,dTrm,dOv,(int)B,(int)T,(int)Kh);
    }
    cudaMemset(gWEnc,0,4*wEncSz); cudaMemset(gBEnc,0,4*H); cudaMemset(gLay,0,4*L*layerSz);
    cudaMemset(gWDec,0,4*A*H); cudaMemset(gBDec,0,4*A); cudaMemset(gWVal,0,4*H); cudaMemset(gBVal,0,4);
    /* ---- forward (batched over T: ONE big GEMM per stage, cheap sequential gate cell) ---- */
    /* bf16-storage path (PUFFER_MG_PREC=bf16): convert weights to bf16 once; the LAYER-stack GEMMs +
       activations run in bf16 (memory-bound → ~2×). The ENCODER GEMMs stay f32: their ld is D (arbitrary
       env obs dim — cuBLAS bf16 GEMMs need even lds) and they're a small share of the traffic; boundary
       casts dIn→dInb (fwd) and dhnInb→f32 (bwd). Heads/PPO stay f32. Grads accumulate in f32. */
    if(bf) k_f2bf_lT<<<GG(L*layerSz),Bk>>>(dWlTb,dLayers,(int)L,(int)H);   /* bf16 weights, TRANSPOSED (see k_f2bf_lT) */
    /* encoder: layer-0 input [T·B·H] = obs[T·B·D]·WEncᵀ + bEnc (all T at once) — ALWAYS f32. f32 path
       writes it straight into chain slice 0; bf16 path keeps dIn as the f32 side of its cast boundary. */
    float* enc0 = bf ? dIn : aHinC;
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)TB,(int)D, dWEnc,(int)D, dObsF,(int)D, enc0,(int)H, mg_bf16());
    k_add_bias<<<GG(TB*H),Bk>>>(enc0,enc0,dBEnc,(int)TB,(int)H);
    if(bf) k_f2bf<<<GG(TB*H),Bk>>>(dInb,dIn,TB*H);   /* boundary: encoder output → bf16 layer-0 input */
    for(size_t l=0;l<L;l++){
      const float* Wl=dLayers+l*layerSz;
      /* combined = in·Wlᵀ → [T·B·3H] (ONE big GEMM replaces T tiny [3H×B×H] GEMMs), then sequential gate
         cell over t. f32: the GEMM writes the layer's aCmb slice IN PLACE (bwd re-reads it — no re-store)
         and the scan writes its output into chain slice l+1 (the next layer's input — no ping-pong, no copy). */
      size_t ls=l*T*B*H;
      if(bf){
        const bf16* WlTb=dWlTb+l*layerSz;
        /* combined = dIn·WlT (WlT pre-transposed ⇒ plain OP_N/OP_N — the strict-legal bf16 pattern) */
        gemm_bf(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)TB,(int)(3*H),(int)H, dInb,(int)H, WlTb,(int)(3*H), dCmbb,(int)(3*H), 0.0f);
        k_mg_scan_fwd_bf<<<GG(B*H),Bk>>>(dCmbb, dInb, dOutb,
          aHidb+ls,aGateb+ls,aProjb+ls,aOb+ls,aHinb+ls,aPrevb+ls, dTrm,(int)B,(int)H,(int)T);
        bf16* tb=dInb; dInb=dOutb; dOutb=tb;
      } else {
        float* cmbL=aCmb+l*T*B*3*H;
        gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)(3*H),(int)TB,(int)H, Wl,(int)H, aHinC+ls,(int)H, cmbL,(int)(3*H), mg_bf16());
        k_mg_scan_fwd2<<<GG(B*H),Bk>>>(cmbL, aHinC+ls, aHinC+ls+T*B*H, aO+ls, dTrm,(int)B,(int)H,(int)T,
          dH0,(int)l,(int)(L*H));
      }
    }
    if(bf) k_bf2f<<<GG(TB*H),Bk>>>(aHfin,dInb,TB*H);   /* final hidden → f32 (bf16 boundary only) */
    float* aHf = bf ? aHfin : aHinC+L*T*B*H;   /* f32: chain slice L IS the final hidden — the copy dies */
    /* decoder + value heads (batched, ALWAYS f32 — tiny, read aHf) */
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)A,(int)TB,(int)H, dWDec,(int)H, aHf,(int)H, aLg,(int)A, mg_bf16());
    k_add_bias<<<GG(TB*A),Bk>>>(aLg,aLg,dBDec,(int)TB,(int)A);
    gemm32(hbl,CUBLAS_OP_T,CUBLAS_OP_N,1,(int)TB,(int)H, dWVal,(int)H, aHf,(int)H, aVal,1, mg_bf16());
    k_add_bias<<<GG(TB),Bk>>>(aVal,aVal,dBVal,(int)TB,1);
    /* ---- backward (batched over T) ---- */
    /* PPO head grad for ALL T·B rows; dhf[T·B·H] = dl·WDec + dval·WVal (f32) */
    if(Kh>1)
      k_mg_ppo_b_md<<<GG(TB),Bk>>>(aLg,aVal,dAct,dAdv,dRet,dOld,dOv,dDlAll,dDvalAll,dNewlp,dNewval,
        (int)B,(int)T,(int)A,(int)Kh,dHs,(float)vfCoef,(float)entCoef,(float)clipEps,(float)vfClip);
    else
      k_mg_ppo_b<<<GG(TB),Bk>>>(aLg,aVal,dAct,dAdv,dRet,dOld,dOv,dDlAll,dDvalAll,dNewlp,dNewval,
        (int)B,(int)T,(int)A,(float)vfCoef,(float)entCoef,(float)clipEps,(float)vfClip);
    /* ---- LOSS SURFACING for the `--log` dashboard (g_mgLossOn; off by default ⇒ zero cost). Reduces
       the 7 dashboard losses into g_mgLoss from the buffers the head kernel just wrote — a pure
       read-only D2H reduction (no kernel, no training buffer touched), so it cannot perturb training
       determinism. Runs every minibatch when on; g_mgLoss then holds the most-recent minibatch. */
    if(g_mgLossOn){
      long n=TB; size_t W=(size_t)A;
      float* hlg=(float*)malloc(4*(size_t)n*W); float* hnl=(float*)malloc(4*(size_t)n);
      float* hnv=(float*)malloc(4*(size_t)n);   double* hol=(double*)malloc(8*(size_t)n);
      double* had=(double*)malloc(8*(size_t)n); double* hre=(double*)malloc(8*(size_t)n);
      double* hov=(double*)malloc(8*(size_t)n);
      if(hlg&&hnl&&hnv&&hol&&had&&hre&&hov&&n>0){
        cudaDeviceSynchronize();
        cudaMemcpy(hlg,aLg,4*(size_t)n*W,cudaMemcpyDeviceToHost);
        cudaMemcpy(hnl,dNewlp,4*(size_t)n,cudaMemcpyDeviceToHost);
        cudaMemcpy(hnv,dNewval,4*(size_t)n,cudaMemcpyDeviceToHost);
        cudaMemcpy(hol,dOld,8*(size_t)n,cudaMemcpyDeviceToHost);
        cudaMemcpy(had,dAdv,8*(size_t)n,cudaMemcpyDeviceToHost);
        cudaMemcpy(hre,dRet,8*(size_t)n,cudaMemcpyDeviceToHost);
        cudaMemcpy(hov,dOv,8*(size_t)n,cudaMemcpyDeviceToHost);
        double lo=1.0-clipEps, hi=1.0+clipEps;
        double sPg=0.0,sV=0.0,sEnt=0.0,sKL=0.0,sOldKL=0.0; long nclip=0;
        for(long r=0;r<n;r++){
          double lgr=(double)hnl[r]-hol[r];                 /* logratio = newLogp − oldLogp */
          double ratio=exp(lgr), ratioC=ratio<lo?lo:(ratio>hi?hi:ratio);
          double adv=had[r], ret=hre[r];
          double a=-adv*ratio, b=-adv*ratioC; sPg += (a>b?a:b);   /* PufferLib pg_loss=mean(max(...)) */
          if(ratio<lo||ratio>hi) nclip++;
          sKL += (ratio-1.0) - lgr;                         /* approx_kl (Schulman k3) */
          sOldKL += -lgr;                                   /* old_approx_kl = mean(−logratio) */
          double vnew=(double)hnv[r], vold=hov[r], dd=vnew-vold;
          double vc=vold + (dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd));
          double du=(vnew-ret)*(vnew-ret), cc=(vc-ret)*(vc-ret);
          sV += 0.5*((vfClip>0.0 && cc>du)? cc : du);       /* clipped value loss (incl. 0.5) */
          const float* lg=hlg+(size_t)r*W; int off=0;
          for(size_t hh=0;hh<Kh;hh++){                       /* per-head joint entropy */
            int sz=(Kh>1)?hsHost[hh]:(int)A;
            double mx=lg[off]; for(int k=1;k<sz;k++) if(lg[off+k]>mx) mx=lg[off+k];
            double se=0.0; for(int k=0;k<sz;k++) se+=exp((double)lg[off+k]-mx);
            double lse=mx+log(se);
            for(int k=0;k<sz;k++){ double lp=(double)lg[off+k]-lse; sEnt -= exp(lp)*lp; }
            off+=sz; }
        }
        double dn=(double)n, pg=sPg/dn, vl=sV/dn, ent=sEnt/dn;
        g_mgLoss[0]=pg; g_mgLoss[1]=vl; g_mgLoss[2]=ent;
        g_mgLoss[3]=pg + vfCoef*vl - entCoef*ent;            /* total (PufferLib composite loss) */
        g_mgLoss[4]=sOldKL/dn; g_mgLoss[5]=sKL/dn; g_mgLoss[6]=(double)nclip/dn;
      }
      free(hlg);free(hnl);free(hnv);free(hol);free(had);free(hre);free(hov);
    }
    g_mbSinceRoll++;
    /* side-stream fork (PUFFER_MG_BPTT_SIDE=0 disables): the weight-grad GEMMs/colsums below and the
       per-layer gWl in the bwd loop feed ONLY the final pack — run them on g_bs_st while the legacy
       chain continues. dDyAll ping-pongs (bg 82) so the side gWl of layer l reads one buffer while the
       chain's next scan writes the other; for L>2 an event guard makes the WAR timing-independent.
       Side colsums get their own partials (bg 83) — dCsPart stays with the chain's encoder colsum. */
    float* dDyAllB=NULL; float* dCsPartS=NULL; int side=0;
    { static int sg=-1; if(sg<0){ const char* e=getenv("PUFFER_MG_BPTT_SIDE"); sg=(e&&e[0]=='0')?0:1; }
      if(sg && !bf && L<=9){ dDyAllB=(float*)bg(82,4*T*B*3*H); dCsPartS=(float*)bg(83,4*CSUM_C*(H>A?H:A));
        side=(dDyAllB && dCsPartS && bs_init()); } }
    if(side) cudaEventRecord(g_bs_ev[0],0);   /* dDlAll/dDvalAll ready (k_mg_ppo_b above) */
    sgemm_rm(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)TB,(int)H,(int)A, dDlAll,(int)A, dWDec,(int)H, dDhfAll,(int)H, 0.0f);
    sgemm_rm(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)TB,(int)H,1, dDvalAll,1, dWVal,(int)H, dDhfAll,(int)H, 1.0f);
    /* decoder/value weight grads (sum over all T·B rows) */
    if(side){
      cudaStreamWaitEvent(g_bs_st,g_bs_ev[0],0);
      sgemm_rm(g_bs_h,CUBLAS_OP_T,CUBLAS_OP_N,(int)A,(int)H,(int)TB, dDlAll,(int)A, aHf,(int)H, gWDec,(int)H, 0.0f);
      colsum_acc_dev(gBDec,dDlAll,TB,(int)A,dCsPartS,g_bs_st);
      sgemm_rm(g_bs_h,CUBLAS_OP_T,CUBLAS_OP_N,1,(int)H,(int)TB, dDvalAll,1, aHf,(int)H, gWVal,(int)H, 0.0f);
      colsum_acc_dev(gBVal,dDvalAll,TB,1,dCsPartS,g_bs_st);
    } else {
      sgemm_rm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)A,(int)H,(int)TB, dDlAll,(int)A, aHf,(int)H, gWDec,(int)H, 0.0f);
      colsum_acc_dev(gBDec,dDlAll,TB,(int)A,dCsPart);
      sgemm_rm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,1,(int)H,(int)TB, dDvalAll,1, aHf,(int)H, gWVal,(int)H, 0.0f);
      colsum_acc_dev(gBVal,dDvalAll,TB,1,dCsPart);
    }
    /* recurrent backward: layers OUTER (L-1..0), timesteps INNER (T-1..0, sequential for the state grad
       dOnextL). dhnIn = grad entering this layer's output; dDhnOut = grad exiting to the layer below. */
    if(bf){
      k_f2bf<<<GG(TB*H),Bk>>>(dDhfAllb,dDhfAll,TB*H);   /* boundary: f32 head grad → bf16 for the recurrent bwd */
      bf16* dhnInb=dDhfAllb;
      for(size_t ll=L;ll-->0;){ size_t l=ll;
        const bf16* WlTb=dWlTb+l*layerSz; float* Wg=gLay+l*layerSz; size_t ls=l*T*B*H;
        k_mg_scan_bwd_bf<<<GG(B*H),Bk>>>(aHidb+ls,aGateb+ls,aProjb+ls,aOb+ls,aPrevb+ls,aHinb+ls,
          dhnInb, dDyAllb, dDhnOutb, dTrm,(int)B,(int)H,(int)T);
        gemm_bf2f(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)(3*H),(int)H,(int)TB, dDyAllb,(int)(3*H), aHinb+ls,(int)H, Wg,(int)H, 0.0f);
        /* dhnOut += dy·Wl = dy·WlTᵀ (OP_N/OP_T on the pre-transposed weights — strict-legal) */
        gemm_bf(hbl,CUBLAS_OP_N,CUBLAS_OP_T,(int)TB,(int)H,(int)(3*H), dDyAllb,(int)(3*H), WlTb,(int)(3*H), dDhnOutb,(int)H, 1.0f);
        bf16* tb=dhnInb; dhnInb=dDhnOutb; dDhnOutb=tb;
      }
      /* boundary: bf16 encoder-output grad → f32, then f32 encoder grads (ld=D may be odd) */
      k_bf2f<<<GG(TB*H),Bk>>>(dOut,dhnInb,TB*H);   /* dOut is free f32 scratch in the bf branch */
      colsum_acc_dev(gBEnc,dOut,TB,(int)H,dCsPart);
      sgemm_rm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)D,(int)TB, dOut,(int)H, dObsF,(int)D, gWEnc,(int)D, 0.0f);
    } else {
      float* dhnIn=dDhfAll;
      for(size_t ll=L;ll-->0;){ size_t l=ll;
        const float* Wl=dLayers+l*layerSz; float* Wg=gLay+l*layerSz; size_t ls=l*T*B*H;
        float* dyB=(side && (((L-1-l)&1)==1))? dDyAllB : dDyAll;   /* ping-pong from the top layer down */
        if(side && l+2<L) cudaStreamWaitEvent(0,g_bs_ev[2+l+2],0);   /* WAR: gWl(l+2) read this buffer */
        k_mg_scan_bwd2<<<GG(B*H),Bk>>>(aCmb+l*T*B*3*H, aO+ls, aHinC+ls,
          dhnIn, dyB, dDhnOut, dTrm,(int)B,(int)H,(int)T, dH0,(int)l,(int)(L*H));
        /* gWl = dyᵀ·hin (all T·B) — pack-only output, forked to the side stream; dhnOut += dy·Wl stays
           on the chain (the next scan needs it) */
        if(side){
          cudaEventRecord(g_bs_ev[0],0);
          cudaStreamWaitEvent(g_bs_st,g_bs_ev[0],0);
          sgemm_rm(g_bs_h,CUBLAS_OP_T,CUBLAS_OP_N,(int)(3*H),(int)H,(int)TB, dyB,(int)(3*H), aHinC+ls,(int)H, Wg,(int)H, 0.0f);
          cudaEventRecord(g_bs_ev[2+l],g_bs_st);
        } else
          sgemm_rm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)(3*H),(int)H,(int)TB, dyB,(int)(3*H), aHinC+ls,(int)H, Wg,(int)H, 0.0f);
        sgemm_rm(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)TB,(int)H,(int)(3*H), dyB,(int)(3*H), Wl,(int)H, dDhnOut,(int)H, 1.0f);
        float* tmp=dhnIn; dhnIn=dDhnOut; dDhnOut=tmp;   /* exit grad → next (lower) layer's incoming grad */
      }
      /* dhnIn now = grad w.r.t. encoder output (all T). encoder grads: */
      colsum_acc_dev(gBEnc,dhnIn,TB,(int)H,dCsPart);
      sgemm_rm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H,(int)D,(int)TB, dhnIn,(int)H, dObsF,(int)D, gWEnc,(int)D, 0.0f);
      if(side){ cudaEventRecord(g_bs_ev[1],g_bs_st); cudaStreamWaitEvent(0,g_bs_ev[1],0); }   /* join: the
        pack/D2H below reads gWDec/gBDec/gWVal/gBVal/gLay — all side outputs; non-blocking stream ⇒ the
        join must be explicit and enqueued BEFORE any consumer */
    }
    int* dLastw=(devCol)? (int*)bg(50,4*(size_t)g_dcN) : NULL;
    if(devCol && dLastw){                              /* value/ratio on-device (parallel last-writer form) */
      cudaMemset(dLastw,0xFF,4*(size_t)g_dcN);         /* -1 = no writer */
      k_mg_lastw<<<GG(B),Bk>>>(dLastw,dSeg,(int)B);
      k_mg_iterate<<<GG(T*B),Bk>>>(g_dcValue,g_dcRatio,g_dcLogp,dNewlp,dNewval,dSeg,dLastw,(int)B,(int)T);
    }
    double* dGfull=(devCol)? (double*)bg(70,8*P) : NULL;
    /* >>> THE MODE DECISION <<< (see the note at the head of the function). Taken here, once the BPTT
       has run and bg(70) has actually been attempted: resident iff we truly hold the device buffer,
       otherwise pack a HOST gradient — which is a fully correct step, just with a PCIe round trip and a
       host-side gradclip. bg(70) is only 8·P (13.6MB at tetris' P=1.7M), so this fallback is cheap
       insurance rather than the common path; the common OOM victim is the activation store, and that
       one cannot be recovered from at all (handled by the !ok abort above). */
    size_t OUTP = dGfull? 0 : P+2*T*B;
    go=lean_alloc_sarray(sizeof(double),OUTP,OUTP); g=OUTP? lean_float_array_cptr(go):NULL;
    for(size_t i=0;i<OUTP;i++) g[i]=0.0;
    if(dGfull){
      /* device-column mode: pack the f32 accumulators → RESIDENT f64 grad (flattenMG order) and return NO
         host copy — the muon gradclips + consumes it on-device. (The old path D2H'd g[P] plus 1MB/mb of
         newlp/newval that device-column Lean never reads.) */
      #define GG(x) ceildiv((long)(x),Bk)
      size_t o=0;
      k_f2d<<<GG(wEncSz),Bk>>>(dGfull+o,gWEnc,(long)wEncSz); o+=wEncSz;
      k_f2d<<<GG(H),Bk>>>(dGfull+o,gBEnc,(long)H); o+=H;
      k_f2d<<<GG(L*layerSz),Bk>>>(dGfull+o,gLay,(long)(L*layerSz)); o+=L*layerSz;
      k_f2d<<<GG(A*H),Bk>>>(dGfull+o,gWDec,(long)(A*H)); o+=A*H;
      k_f2d<<<GG(A),Bk>>>(dGfull+o,gBDec,(long)A); o+=A;
      k_f2d<<<GG(H),Bk>>>(dGfull+o,gWVal,(long)H); o+=H;
      k_f2d<<<GG(1),Bk>>>(dGfull+o,gBVal,1); o+=1;
      #undef GG
      g_mgg_fresh=1; g_mgg_P=P;
      /* NO sync: everything is stream-ordered on the legacy stream and nothing host-visible was produced —
         the resident muon enqueues right behind, and the host runs ahead (the rollout's one sync drains). */
    } else {
      /* HOST mode — reached either the classic way (`scal`/`obs` were host arrays, e.g. verify-mingru-
         grad-gpu) or as the devCol FALLBACK when bg(70) could not be allocated. In the fallback the
         device-side minibatch state (k_mg_iterate's value/ratio update above) has already happened
         exactly as in resident mode; only the gradient hand-off changes, so the step stays correct. */
      cudaDeviceSynchronize();
      if(devCol){ static int warned=0; if(!warned){ warned=1;
        fprintf(stderr,"[puffer] mingru grad: resident-grad buffer (bg 70, %zu bytes) unavailable — "
                       "falling back to the HOST gradient path (correct, just slower)\n", 8*P); } }
      /* host mode: pack gradient accumulators (f32) → g[P] (f64) in flattenMG order */
      float* hg=(float*)malloc(4*P); size_t o=0;
      cudaMemcpy(hg+o,gWEnc,4*wEncSz,cudaMemcpyDeviceToHost); o+=wEncSz;
      cudaMemcpy(hg+o,gBEnc,4*H,cudaMemcpyDeviceToHost); o+=H;
      cudaMemcpy(hg+o,gLay,4*L*layerSz,cudaMemcpyDeviceToHost); o+=L*layerSz;
      cudaMemcpy(hg+o,gWDec,4*A*H,cudaMemcpyDeviceToHost); o+=A*H;
      cudaMemcpy(hg+o,gBDec,4*A,cudaMemcpyDeviceToHost); o+=A;
      cudaMemcpy(hg+o,gWVal,4*H,cudaMemcpyDeviceToHost); o+=H;
      cudaMemcpy(hg+o,gBVal,4,cudaMemcpyDeviceToHost); o+=1;
      for(size_t i=0;i<P;i++) g[i]=(double)hg[i];
      free(hg);
      /* new_logp[T·B] ++ new_value[T·B] */
      float* hx=(float*)malloc(4*T*B);
      cudaMemcpy(hx,dNewlp,4*T*B,cudaMemcpyDeviceToHost); for(size_t i=0;i<T*B;i++) g[P+i]=(double)hx[i];
      cudaMemcpy(hx,dNewval,4*T*B,cudaMemcpyDeviceToHost); for(size_t i=0;i<T*B;i++) g[P+T*B+i]=(double)hx[i];
      free(hx);
    }
    (void)O;
  }
  /* device buffers are the persistent bg cache — not freed here */
  free(hP);
  lean_dec(pa);lean_dec(obsa);lean_dec(scala);lean_dec(hsA);lean_dec(segIdxa);lean_dec(mbPrioa);
  /* `fail` ⇒ no gradient exists in ANY form. Pre-fix this branch zero-filled g[0..P) (host mode) or did
     nothing at all (devCol mode, where g was already NULL) and returned "successfully", which is exactly
     how a whole training run could complete with rc=0 having applied only zero gradients. */
  if(fail) return mg_contract_fail("BPTT gradient", fail, P);
  return lean_io_result_mk_ok(go);   /* IO: devCol mode is pure device side effect — never CSE/DCE'd */
}

/* Concurrent-NS pool: one blocking stream + one cuBLAS handle per Muon matrix chain. The per-matrix
   Newton–Schulz chains are data-disjoint (each reads/writes its own dW/dG/dM slice and its own scratch
   slab), so running them on K blocking streams overlaps ~155 tiny serial launches that left a 170-SM
   card almost idle. Blocking streams give the barriers for free: the legacy-stream gclip before the
   chains and the legacy-stream k_d2f handoff after both serialize against all of them. Handles clone
   cu_handle()'s exact configuration (CUBLAS_DEFAULT_MATH + 32MB workspace) so GEMM algorithm selection
   — and therefore every bit — matches the sequential path. */
/* FUSED whole-Muon Newton–Schulz for SMALL models: ONE kernel launch steps every matrix and bias
   (one block per flattenMG segment) — the streamed path enqueues ~155 tiny launches per call, pure
   host-enqueue cost when the matrices are tiny (pong 1L×h32: muon was 28% of the training wall).
   Replicates the muon_mat_dev_bf sequence op-for-op: f64 nesterov (in place), the EXACT 256-lane
   tree frobnorm, f32 NS iterates, f64 finalize. Only the NS matmuls deviate (in-block k-ascending
   dot vs cuBLAS Sgemm) — the same tolerance class that already separates the bf path from the
   PUFFER_MUON_EXACT oracle path. Biases run the k_stepvec math elementwise (cols==0 marks a bias).
   Gated to segments whose scratch fits in shared (~<=88KB); big models keep the streamed pool. */
__global__ void k_muon_ns_fused(double* dW, double* dM, const double* dG,
    const long* segOfs, const int* segRows, const int* segCols, int nSeg,
    double lr, double wd, double mu, double eps){
  int seg=blockIdx.x; if(seg>=nSeg) return;
  long ofs=segOfs[seg]; int rows=segRows[seg], cols=segCols[seg];
  double* W=dW+ofs; double* M=dM+ofs; const double* G=dG+ofs;
  int t=threadIdx.x, B=blockDim.x;
  if(cols==0){                                   /* bias: k_stepvec elementwise */
    for(long i=t;i<rows;i+=B){ double newm=mu*M[i]+G[i]; double upd=G[i]+mu*newm;
      W[i]=W[i]*(1.0-lr*wd)+lr*upd; M[i]=newm; }
    return;
  }
  long n=(long)rows*cols; int sq=(rows<=cols)?rows:cols;
  extern __shared__ float shf[];
  float* X=shf; float* Xt=X+n; float* A=Xt+n; float* P=A+(long)sq*sq; float* Q=P+n;
  double* U=(double*)(((size_t)(Q+n)+7)&~(size_t)7);
  __shared__ double shd[256]; __shared__ double inv;
  for(long i=t;i<n;i+=B){ double newm=mu*M[i]+G[i]; M[i]=newm; U[i]=G[i]+mu*newm; }
  __syncthreads();
  { double p=0.0; for(long i=t;i<n;i+=B){ double v=U[i]; p+=v*v; } shd[t]=p; __syncthreads();
    for(int s2=B/2;s2>0;s2>>=1){ if(t<s2) shd[t]+=shd[t+s2]; __syncthreads(); }
    if(t==0) inv=1.0/(sqrt(shd[0])+eps); }
  __syncthreads();
  for(long i=t;i<n;i+=B) X[i]=(float)(inv*U[i]);
  __syncthreads();
  for(int it=0;it<5;it++){
    float a=(float)D_MUON_COEFFS[it][0], b=(float)D_MUON_COEFFS[it][1], c=(float)D_MUON_COEFFS[it][2];
    for(long i=t;i<n;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); Xt[(long)cc*rows+r]=X[i]; }
    __syncthreads();
    if(rows<=cols){
      for(long i=t;i<(long)rows*rows;i+=B){ int r=(int)(i/rows), cc=(int)(i%rows); float sacc=0.0f;
        for(int l=0;l<cols;l++) sacc+=X[(long)r*cols+l]*Xt[(long)l*rows+cc]; A[i]=sacc; }
      __syncthreads();
      for(long i=t;i<n;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); float sacc=0.0f;
        for(int l=0;l<rows;l++) sacc+=A[(long)r*rows+l]*X[(long)l*cols+cc]; P[i]=sacc; }
      __syncthreads();
      for(long i=t;i<n;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); float sacc=0.0f;
        for(int l=0;l<rows;l++) sacc+=A[(long)r*rows+l]*P[(long)l*cols+cc]; Q[i]=sacc; }
    } else {
      for(long i=t;i<(long)cols*cols;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); float sacc=0.0f;
        for(int l=0;l<rows;l++) sacc+=Xt[(long)r*rows+l]*X[(long)l*cols+cc]; A[i]=sacc; }
      __syncthreads();
      for(long i=t;i<n;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); float sacc=0.0f;
        for(int l=0;l<cols;l++) sacc+=X[(long)r*cols+l]*A[(long)l*cols+cc]; P[i]=sacc; }
      __syncthreads();
      for(long i=t;i<n;i+=B){ int r=(int)(i/cols), cc=(int)(i%cols); float sacc=0.0f;
        for(int l=0;l<cols;l++) sacc+=P[(long)r*cols+l]*A[(long)l*cols+cc]; Q[i]=sacc; }
    }
    __syncthreads();
    for(long i=t;i<n;i+=B) X[i]=a*X[i]+b*P[i]+c*Q[i];
    __syncthreads();
  }
  { double scale=sqrt(fmax(1.0,(double)rows/(double)cols)); double c1=1.0-lr*wd, c2=lr*scale;
    for(long i=t;i<n;i+=B) W[i]=c1*W[i]+c2*(double)X[i]; }
}
static int mu_fused_optin(size_t shb){ static int state=0;
  if(state==0) state=(cudaFuncSetAttribute(k_muon_ns_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(96*1024))==cudaSuccess)?1:-1;
  return state==1 && shb<=90*1024; }
#define MU_POOL 5
static cudaStream_t g_mu_st[MU_POOL]; static cublasHandle_t g_mu_h[MU_POOL]; static void* g_mu_ws[MU_POOL];
static int g_mu_n=0, g_mu_bad=0;
static void mu_pool_teardown(void){   /* all-or-nothing: a partial pool must not strand VRAM on the shared card */
  while(g_mu_n>0){ g_mu_n--;
    cublasDestroy(g_mu_h[g_mu_n]); cudaStreamDestroy(g_mu_st[g_mu_n]);
    if(g_mu_ws[g_mu_n]){ cudaFree(g_mu_ws[g_mu_n]); g_mu_ws[g_mu_n]=NULL; } }
}
static int mu_pool(int n){
  if(g_mu_bad) return 0; if(n>MU_POOL) n=MU_POOL;
  while(g_mu_n<n){
    cudaStream_t st=NULL; cublasHandle_t hh=NULL; void* ws=NULL;
    if(cudaStreamCreate(&st)!=cudaSuccess){ g_mu_bad=1; mu_pool_teardown(); return 0; }
    if(cublasCreate(&hh)!=CUBLAS_STATUS_SUCCESS){ cudaStreamDestroy(st); g_mu_bad=1; mu_pool_teardown(); return 0; }
    cublasSetMathMode(hh, CUBLAS_DEFAULT_MATH);
    /* the 32MB workspace is BIT-LOAD-BEARING (cuBLAS algorithm selection is workspace-gated): a failed
       alloc must hard-fail to the sequential path, never run with a differently-configured handle */
    if(cudaMalloc(&ws,32*1024*1024)!=cudaSuccess){ cudaGetLastError();   /* clear the sticky error */
      cublasDestroy(hh); cudaStreamDestroy(st); g_mu_bad=1; mu_pool_teardown(); return 0; }
    cublasSetWorkspace(hh,ws,32*1024*1024);
    cublasSetStream(hh,st);
    g_mu_ws[g_mu_n]=ws; g_mu_st[g_mu_n]=st; g_mu_h[g_mu_n]=hh; g_mu_n++; }
  return 1;
}

/* Resident in-place Muon step over the device-resident MinGRU policy handle (policyH → [weights(P);mom(P)]).
   The flat-array twin of muonStepFlatMG done ON the resident buffer: each flattenMG matrix is orthogonalized
   with muon_mat_dev (bit-for-bit == cudaMuonStepMatFFI), each bias takes the k_stepvec Nesterov (== the host
   Nesterov muonStepFlatMG uses). Uploads only the clipped gradient; weights+mom stay resident. Returns the
   new wFlat (P) so the (unchanged, host-param) gradient kernel reads the fresh weights next minibatch. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mingru_muon_resident(
    size_t policyH, lean_obj_arg gClipA, size_t H, size_t D, size_t L, size_t A,
    double lr, double wd, double mu, double eps, double maxGradNorm, double gscale, lean_obj_arg w){
  (void)w;
  size_t wEncSz=H*D, layerSz=3*H*H, P=wEncSz+H+L*layerSz+A*H+A+H+1;
  const double* gc=lean_float_array_cptr(gClipA);
  size_t gcN=lean_sarray_size(gClipA);
  lean_object* Oo=NULL; double* out=NULL;   /* allocated below once the return form (full vs empty) is known */
  double* dW=(double*)policyH; double* dM=dW+P;                  /* resident [weights(P); mom(P)] */
  size_t maxN=wEncSz; if(3*H*H>maxN) maxN=3*H*H; if(A*H>maxN) maxN=A*H;   /* largest matrix (1·H ≤ these) */
  size_t maxSZA=9*H*H; { size_t s; s=(H>D?H*H:D*D); if(s>maxSZA)maxSZA=s; s=(A>H?A*A:H*H); if(s>maxSZA)maxSZA=s; }
  double *dG=(double*)rb2(13,8*P), *dU=(double*)rb2(14,8*maxN), *dX=(double*)rb2(15,8*maxN), *dXt=(double*)rb2(16,8*maxN);
  double *dAa=(double*)rb2(17,8*maxSZA), *dPp=(double*)rb2(18,8*maxN), *dQq=(double*)rb2(19,8*maxN), *dInv=(double*)rb2(20,8);
  /* resident-grad mode: gClip EMPTY ⇒ the grad FFI left the raw summed gradient on-device (bg 70);
     gradclip runs here on-GPU (k_gclip_norm/scale — Lean's exact formula, tree-order sum) into dG.
     The stamp is CONSUMED (one grad per muon step) so a failed grad can never leave a stale gradient
     to be silently re-consumed. */
  int gres=(gcN==0 && g_mgg_fresh && g_mgg_P==P);
  g_mgg_fresh=0;
  double *dGfull=NULL,*dSq=NULL;
  if(gres){ dGfull=(double*)bg(70,8*P); dSq=(double*)rb2(21,8); if(!dGfull||!dSq) gres=0; }
  /* Empty gClip means "consume the resident gradient" — and there is none. Pre-fix this warned once and
     then FELL THROUGH: `ok` went false, no step was taken at all, and a zero-filled wFlat was returned
     (which Lean then uploaded as the BPTT's f32 weights on the next minibatch, because the w0 guard is
     NaN-poisoned). The run kept going and exited 0 having learned nothing.
     The gradient FFI is now AUTHORITATIVE about the mode — it returns a HOST gradient whenever it could
     not leave a resident one, and hard-errors when it could produce neither — so reaching this point
     means the two sides genuinely disagreed (e.g. bg(70)/rb2(21) failed between the two calls, or the
     grad FFI was never called for this minibatch). That is not recoverable here: this function has no
     gradient of any kind. Die loudly. */
  if(gcN==0 && !gres)
    { lean_dec(gClipA); return mg_contract_fail("resident Muon step",
        "empty gClip but no resident gradient is present — the BPTT left no bg(70) stamp, or its "
        "8*P device buffer / the gradclip scratch could not be allocated", P); }
  int ok=(P>0 && policyH && dG&&dU&&dX&&dXt&&dAa&&dPp&&dQq&&dInv && (gres || gcN>=P));
  /* Same rule for the host-gradient form: a short gClip or a failed Muon scratch allocation used to
     zero-fill the returned wFlat and silently skip the step. */
  if(!ok)
    { lean_dec(gClipA); return mg_contract_fail("resident Muon step",
        (gcN>0 && gcN<P) ? "host gClip shorter than the parameter vector"
        : (!policyH ? "no resident policy handle"
                    : "Muon device scratch (rb2 13-20) allocation failed"), P); }
  /* CUDA-graph replay of the whole resident-mode step (gclip + 5 MMAT chains + 3 biases ≈ 155 serial tiny
     launches): all pointers are STABLE across calls (resident dW/dM, rb2/bg scratch). REWORKED from the
     earlier default-OFF version, which lost for two reasons now fixed: (1) it keyed the graph on lr, so
     the per-update anneal forced a re-capture per update — lr (and the per-segment finalize c1/c2 it
     derives) now lives in a DEVICE coef block (rb2 22) read by k_finalize_f32_g/k_stepvec_g, so the graph
     captures ONCE per run and a 200-byte async copy replaces every re-capture; (2) it paid a
     sync + full-P D2H per call while the plain resident path returns EMPTY and stays pure-enqueue — the
     graph path now uses the same empty-return g_mgw stamp handoff. Engages only where the fused
     whole-muon is NOT eligible (big-H models, e.g. h128's 384×128 layer overflows shared — small models
     already have the better 1-launch k_muon_ns_fused). BIT-IDENTICAL to the plain streamed walk: same
     kernels, same order, per-segment chains are data-independent (disjoint weight slices), and the coef
     doubles are the same host-computed values the immediate-arg kernels received. PUFFER_MUON_GRAPH=0
     disables. */
  static cudaStream_t g_must=NULL; static cudaGraphExec_t g_mgex=NULL; static int g_mgbad=-1;
  static double g_mgmx=0, g_mggs=0; static size_t g_mgP=0; static int g_mgex_valid=0;
  if(g_mgbad<0){ const char* e=getenv("PUFFER_MUON_GRAPH"); g_mgbad=(e!=NULL&&e[0]=='0')?1:0; }   /* default ON (see rework note) */
  int mexPre; { const char* e=getenv("PUFFER_MUON_EXACT"); mexPre=(e!=NULL&&e[0]=='1'); }
  /* fused whole-muon eligibility (shape math only — mirrors the plain path's check below): when the
     fused kernel can take the whole step in ONE launch, the graph has nothing to win. */
  int fusedElig=0;
  { static int fg=-1; if(fg<0){ const char* e=getenv("PUFFER_MUON_FUSED"); fg=(e&&e[0]=='0')?0:1; }
    if(fg && !mexPre && L+6<=16){ long mx=0;
      long rr[16]; long cc2[16]; int nSeg=0;
      rr[nSeg]=(long)H; cc2[nSeg]=(long)D; nSeg++;
      for(size_t l=0;l<L;l++){ rr[nSeg]=3L*H; cc2[nSeg]=(long)H; nSeg++; }
      rr[nSeg]=(long)A; cc2[nSeg]=(long)H; nSeg++; rr[nSeg]=1; cc2[nSeg]=(long)H; nSeg++;
      for(int i2=0;i2<nSeg;i2++){ long nn=rr[i2]*cc2[i2]; long sq=(rr[i2]<=cc2[i2])?rr[i2]:cc2[i2];
        long bytes=4L*(4*nn+sq*sq)+8L*nn+64; if(bytes>mx) mx=bytes; }
      fusedElig=mu_fused_optin((size_t)mx); } }
  double* dCoef=(double*)rb2(22,8*32);                            /* [0]=lr, [1]=c1, [2+i]=c2ᵢ (≤19 matrices) */
  int useGraph=(ok && gres && !mexPre && !g_mgbad && !fusedElig && dCoef!=NULL);
  /* empty-wFlat contract: under the resident-grad path the returned weights' only consumed byte was the
     w0 guard — return EMPTY (no sync, no 256KB D2H; the grad trusts the g_mgw stamp instead) and the
     whole vtrace→grad→muon minibatch chain becomes pure enqueue. Both the plain resident path AND the
     graph path use it now; only the host fallback keeps the full D2H + value guard. */
  int emptyRet=gres;
  Oo=lean_alloc_sarray(sizeof(double), emptyRet?0:P, emptyRet?0:P);
  out=emptyRet? NULL : lean_float_array_cptr(Oo);
  if(useGraph && (!g_mgex_valid || g_mgP!=P || g_mgmx!=maxGradNorm || g_mggs!=gscale)){
    if(!g_must) cudaStreamCreate(&g_must);
    cublasHandle_t mh0=cu_handle(); cublasSetStream(mh0,g_must);
    /* CONCURRENT capture: the plain resident path runs the matrix chains concurrently on the mu_pool
       streams — a sequential single-stream capture serializes them on the GPU and measured a wash (the
       enqueue saving reappeared as rollout-phase wait for the longer serial muon). Capture the SAME
       fork/join topology instead: gclip on g_must → fork event → each chain on its mu_pool stream with
       its OWN scratch slab (the shared rb2 scratch would race across branches — same rule as the plain
       conc path) → join events back into g_must. Falls back to the sequential capture when the pool or
       slabs are unavailable. */
    int nMatG=(int)L+3;
    static cudaEvent_t evFork=NULL, evJoin[MU_POOL];
    if(!evFork){ cudaEventCreateWithFlags(&evFork,cudaEventDisableTiming);
      for(int i2=0;i2<MU_POOL;i2++) cudaEventCreateWithFlags(&evJoin[i2],cudaEventDisableTiming); }
    int nstG=(nMatG<MU_POOL)?nMatG:MU_POOL;
    size_t aX=((8*maxN+255)&~(size_t)255), aXt=aX+((4*maxN+255)&~(size_t)255),
           aA=aXt+((4*maxN+255)&~(size_t)255), aP2=aA+((4*maxSZA+255)&~(size_t)255),
           aQ2=aP2+((4*maxN+255)&~(size_t)255), aI=aQ2+((4*maxN+255)&~(size_t)255), slab=aI+256;
    char* slabsG=NULL; int concG=mu_pool(nstG)?1:0;
    if(concG){ slabsG=(char*)bg(80,(size_t)nMatG*slab); if(!slabsG) concG=0; }
    int capOK=(cudaStreamBeginCapture(g_must,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    if(capOK){
      k_gclip_norm<<<1,256,0,g_must>>>(dSq,dGfull,gscale,(long)P);
      k_gclip_scale<<<ceildiv((long)P,256),256,0,g_must>>>(dG,dGfull,dSq,maxGradNorm,gscale,(long)P);
      if(concG){ cudaEventRecord(evFork,g_must);
        for(int si=0;si<nstG;si++) cudaStreamWaitEvent(g_mu_st[si],evFork,0); }   /* pull pool streams into the capture, after gclip */
      { int B=256; size_t off=0; int mi=0;
        #define MMATG(r,c) do{ \
            if(concG){ int si=mi%nstG; char* sb=slabsG+(size_t)mi*slab; cublasSetStream(g_mu_h[si],g_mu_st[si]); \
              muon_mat_dev_bf(g_mu_h[si],(int)(r),(int)(c), dW+off,dG+off,dM+off, dW+off,dM+off, lr,wd,mu,eps, \
                (double*)sb,(double*)(sb+aI),(float*)(sb+aX),(float*)(sb+aXt),(float*)(sb+aA), \
                (float*)(sb+aP2),(float*)(sb+aQ2), 0, g_mu_st[si], dCoef, mi); } \
            else muon_mat_dev_bf(mh0,(int)(r),(int)(c), dW+off,dG+off,dM+off, dW+off,dM+off, lr,wd,mu,eps, \
                dU,dInv,(float*)dX,(float*)dXt,(float*)dAa,(float*)dPp,(float*)dQq, 0, g_must, dCoef, mi); \
            mi++; off+=(size_t)(r)*(size_t)(c); }while(0)
        #define MBIAG(nn)  do{ k_stepvec_g<<<ceildiv((long)(nn),B),B,0,g_must>>>(dW+off,dM+off, dW+off,dG+off,dM+off, dCoef,wd,mu,(long)(nn)); off+=(size_t)(nn); }while(0)
        MMATG(H,D); MBIAG(H);
        for(size_t l=0;l<L;l++) MMATG(3*H,H);
        MMATG(A,H); MBIAG(A); MMATG(1,H); MBIAG(1);
        #undef MMATG
        #undef MBIAG
        if(concG) for(int si=0;si<nstG;si++){ cudaEventRecord(evJoin[si],g_mu_st[si]); cudaStreamWaitEvent(g_must,evJoin[si],0); }
      }
      cudaGraph_t gr=NULL;
      if(cudaStreamEndCapture(g_must,&gr)==cudaSuccess && gr){
        int fresh=0;
        if(g_mgex_valid){ cudaGraphExecUpdateResultInfo ri;
          if(cudaGraphExecUpdate(g_mgex,gr,&ri)!=cudaSuccess){ cudaGraphExecDestroy(g_mgex); g_mgex_valid=0; } }
        if(!g_mgex_valid){ fresh=(cudaGraphInstantiate(&g_mgex,gr,0)==cudaSuccess); g_mgex_valid=fresh; }
        cudaGraphDestroy(gr);
        if(g_mgex_valid){ g_mgP=P; g_mgmx=maxGradNorm; g_mggs=gscale; }
        else g_mgbad=1;
        (void)fresh;
      } else { cudaStreamEndCapture(g_must,NULL); g_mgbad=1; }
    } else g_mgbad=1;
    cublasSetStream(mh0,0); cudaGetLastError();                  /* restore + clear any capture-mode error */
    if(g_mgbad){ static int warned=0; if(!warned){ warned=1;
      fprintf(stderr,"[puffer] muon graph capture failed — plain launch path\n"); } }
  }
  if(useGraph && !g_mgbad && g_mgex_valid){
    /* per-call coef refresh (48-200 bytes, async on the graph's own stream ⇒ ordered before the launch).
       hCoef is static: an async H2D source must outlive the call. Same doubles the immediate-arg kernels
       would have received ⇒ bit-identical. */
    static double hCoef[32];
    { int mi=0; hCoef[0]=lr; hCoef[1]=1.0-lr*wd;
      #define CSC(r,c) do{ hCoef[2+mi]=lr*sqrt(fmax(1.0,(double)(r)/(double)(c))); mi++; }while(0)
      CSC(H,D); for(size_t l=0;l<L;l++) CSC(3*H,H); CSC(A,H); CSC(1,H);
      #undef CSC
      cudaMemcpyAsync(dCoef,hCoef,8*(size_t)(2+mi),cudaMemcpyHostToDevice,g_must); }
    cudaGraphLaunch(g_mgex,g_must);
    /* resident-weight handoff — same empty-return stamp as the plain path (no sync, no D2H); d2f on the
       graph's stream so it orders after the muon. */
    { float* dPg=(float*)bg(0,4*P); int B=256;
      if(dPg){ k_d2f<<<ceildiv((long)P,B),B,0,g_must>>>(dPg,dW,(long)P);
        g_mgw_fresh=1; g_mgw_P=P; g_mgw_empty=1; g_mgw_w0=nan(""); }
      else g_mgw_fresh=0; }
    lean_dec(gClipA);
    return lean_io_result_mk_ok(Oo);
  }
  if(ok){
    if(gres){
      k_gclip_norm<<<1,256>>>(dSq,dGfull,gscale,(long)P);
      k_gclip_scale<<<ceildiv((long)P,256),256>>>(dG,dGfull,dSq,maxGradNorm,gscale,(long)P);
    } else
      cudaMemcpy(dG,gc,8*P,cudaMemcpyHostToDevice);              /* host-clipped gradient upload (fallback) */
    int B=256; size_t off=0;
    /* Newton–Schulz backend: default = muon_mat_dev_bf (f32 cuBLAS NS, f64 momentum/finalize — the same
       verified production path the MLP trainers use; was 3ms/update of naive-f64 k_matmul here).
       PUFFER_MUON_EXACT=1 keeps the bit-exact f64 kernels (the Lean-oracle-exact path). The f64 scratch
       (rb2, sized 8·maxN) is reinterpreted as f32 for the NS iterates — half the bytes, always fits. */
    int mex; { const char* e=getenv("PUFFER_MUON_EXACT"); mex=(e!=NULL&&e[0]=='1'); }
    /* fused whole-Muon step (small models): ONE launch replaces the ~155-launch streamed walk — at tiny
       shapes the walk is pure host-enqueue cost (pong: muon was 28% of the wall). Segment table uploaded
       once (keyed on P); every matrix segment must fit the fused kernel's shared budget or we fall back.
       PUFFER_MUON_FUSED=0 restores the streamed path. */
    int fused=0;
    { static int fg=-1; if(fg<0){ const char* e=getenv("PUFFER_MUON_FUSED"); fg=(e&&e[0]=='0')?0:1; }
      if(fg && !mex && L+6<=16){   /* nSeg=L+6 slots — bound BEFORE filling (L>=11 would smash the
                                        stack arrays below; such models fall back to the streamed path) */
        long so[16]; int sr[16], sc2[16]; int nSeg=0; long ofs2=0, mx=0;
        #define SEGM(r,c) do{ so[nSeg]=ofs2; sr[nSeg]=(int)(r); sc2[nSeg]=(int)(c); nSeg++; ofs2+=(long)(r)*(long)(c); }while(0)
        #define SEGB(nn)  do{ so[nSeg]=ofs2; sr[nSeg]=(int)(nn); sc2[nSeg]=0; nSeg++; ofs2+=(long)(nn); }while(0)
        SEGM(H,D); SEGB(H); for(size_t l=0;l<L;l++) SEGM(3*H,H); SEGM(A,H); SEGB(A); SEGM(1,H); SEGB(1);
        #undef SEGM
        #undef SEGB
        for(int i2=0;i2<nSeg;i2++){ if(sc2[i2]==0) continue;
          long nn=(long)sr[i2]*sc2[i2]; long sq=(sr[i2]<=sc2[i2])?sr[i2]:sc2[i2];
          long bytes=4L*(4*nn+sq*sq)+8L*nn+64; if(bytes>mx) mx=bytes; }
        long* dSo=(long*)bg(84,8*16); int* dSr=(int*)bg(85,4*16); int* dSc=(int*)bg(86,4*16);
        if(dSo&&dSr&&dSc&&mu_fused_optin((size_t)mx)){
          static size_t kH=0,kD=0,kL=0,kA=0;   /* full shape tuple: equal-P collisions across (H,D,L,A)
                                                  exist — a stale table would silently mis-slice */
          if(kH!=H||kD!=D||kL!=L||kA!=A){ cudaMemcpy(dSo,so,8*(size_t)nSeg,cudaMemcpyHostToDevice);
            cudaMemcpy(dSr,sr,4*(size_t)nSeg,cudaMemcpyHostToDevice);
            cudaMemcpy(dSc,sc2,4*(size_t)nSeg,cudaMemcpyHostToDevice); kH=H;kD=D;kL=L;kA=A; }
          k_muon_ns_fused<<<nSeg,256,(size_t)mx>>>(dW,dM,dG,dSo,dSr,dSc,nSeg,lr,wd,mu,eps);
          fused=1;
        } } }
    if(!fused){
    cublasHandle_t mh=cu_handle(); cublasSetStream(mh,0);
    /* concurrent per-matrix chains (PUFFER_MUON_CONC=0 restores sequential; exact path stays sequential).
       Per-matrix scratch slabs live in bg(80) — the shared rb2 scratch would race across streams. */
    int nMat=(int)L+3, mIdx=0;
    int conc; { const char* e=getenv("PUFFER_MUON_CONC"); conc=(e==NULL||e[0]!='0') && !mex; }
    size_t aX=((8*maxN+255)&~(size_t)255), aXt=aX+((4*maxN+255)&~(size_t)255),
           aA=aXt+((4*maxN+255)&~(size_t)255), aP2=aA+((4*maxSZA+255)&~(size_t)255),
           aQ2=aP2+((4*maxN+255)&~(size_t)255), aI=aQ2+((4*maxN+255)&~(size_t)255), slab=aI+256;
    char* slabs=NULL;
    if(conc){ if(!mu_pool(nMat)) conc=0;   /* pool FIRST — a pool failure must not strand the slab */
      else { slabs=(char*)bg(80,(size_t)nMat*slab); if(!slabs) conc=0; } }
    #define MMAT(r,c) do{ if(mex) muon_mat_dev((int)(r),(int)(c), dW+off,dG+off,dM+off, dW+off,dM+off, lr,wd,mu,eps, \
                            dU,dX,dXt,dAa,dPp,dQq,dInv); \
                          else if(conc){ char* sb=slabs+(size_t)mIdx*slab; int si=mIdx%MU_POOL; \
                            muon_mat_dev_bf(g_mu_h[si],(int)(r),(int)(c), dW+off,dG+off,dM+off, dW+off,dM+off, lr,wd,mu,eps, \
                              (double*)sb,(double*)(sb+aI),(float*)(sb+aX),(float*)(sb+aXt),(float*)(sb+aA), \
                              (float*)(sb+aP2),(float*)(sb+aQ2), 0, g_mu_st[si]); } \
                          else muon_mat_dev_bf(mh,(int)(r),(int)(c), dW+off,dG+off,dM+off, dW+off,dM+off, lr,wd,mu,eps, \
                            dU,dInv,(float*)dX,(float*)dXt,(float*)dAa,(float*)dPp,(float*)dQq, 0); \
                          mIdx++; off+=(size_t)(r)*(size_t)(c); }while(0)
    #define MBIA(nn)  do{ cudaStream_t bs=conc? g_mu_st[mIdx%MU_POOL] : (cudaStream_t)0; \
                          k_stepvec<<<ceildiv((long)(nn),B),B,0,bs>>>(dW+off,dM+off, dW+off,dG+off,dM+off, lr,wd,mu,(long)(nn)); \
                          off+=(size_t)(nn); }while(0)   /* biases have no slab: mIdx counts MATRICES only
                             (it indexes the nMat-slab scratch allocation — a bias increment would push
                             later matrices past the end) */
    MMAT(H,D);                                    /* wEnc */
    MBIA(H);                                      /* bEnc */
    for(size_t l=0;l<L;l++) MMAT(3*H,H);          /* MinGRU layers */
    MMAT(A,H);                                    /* wDec */
    MBIA(A);                                      /* bDec */
    MMAT(1,H);                                    /* wVal */
    MBIA(1);                                      /* bVal */
    #undef MMAT
    #undef MBIA
    }
    /* resident-weight handoff: cast the fresh f64 weights into the BPTT's f32 dP (bg 0) ON-DEVICE, so the
       next grad call skips its per-minibatch host cast + 250KB H2D. Empty contract (emptyRet): no sync,
       no D2H — the grad trusts g_mgw_fresh+g_mgw_empty. Full form keeps the D2H + w0 value guard. */
    { float* dPg=(float*)bg(0,4*P);
      if(dPg){ k_d2f<<<ceildiv((long)P,B),B>>>(dPg,dW,(long)P);
        g_mgw_fresh=1; g_mgw_P=P; g_mgw_empty=emptyRet?1:0;
        if(emptyRet) g_mgw_w0=nan("");   /* poison the value guard: NaN==x is always false, so a non-empty
                                            pa (e.g. a fallback's zero-filled wFlat) can never spuriously
                                            pass — it falls through to the honest host cast+upload */
        if(!emptyRet){ cudaDeviceSynchronize(); cudaMemcpy(out,dW,8*P,cudaMemcpyDeviceToHost); g_mgw_w0=out[0]; } }
      else { g_mgw_fresh=0;
        if(!emptyRet){ cudaDeviceSynchronize(); cudaMemcpy(out,dW,8*P,cudaMemcpyDeviceToHost); } } }
  }   /* no `else`: !ok returned an IO error above — a step is never silently skipped */
  lean_dec(gClipA);
  return lean_io_result_mk_ok(Oo);
}

/* Resident in-place Muon over a device-resident MLP policy handle (policyH → [params(P);mom(P)],
   layout [W1(H·D)|b1(H)|W2(O·H)|b2(O)]) — the resident twin of muonStepMlpBlasFFI (muon_mat_cpu +
   stepvec_cpu) used by the MD/Cont plugin trainers. `gRaw` is the RAW summed minibatch gradient;
   gscale (=1/N) mean-scales it (k_scale_const, then muon_mat_dev/k_stepvec == muon_mat_cpu/stepvec_cpu
   which fold gscale into the Nesterov the same way, and k_frobnorm_inv is the same two-level sum ⇒
   bit-identical). Uploads only gRaw; params+mom stay resident. Returns the new params(P) for the
   (unchanged, host-param) MD/Cont gradient kernel next minibatch. Works for any O (MD: Σheads+1, Cont: 2d+1). */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_muon_step_mlp_resident(
    size_t policyH, lean_obj_arg gRawA, size_t H, size_t D, size_t O,
    double gscale, double lr, double wd, double mu, double eps, lean_obj_arg w){
  (void)w;
  size_t P=H*D+H+O*H+O; size_t oW1=0,ob1=H*D,oW2=H*D+H,ob2=H*D+H+O*H;
  const double* gr=lean_float_array_cptr(gRawA);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),P,P); double* out=lean_float_array_cptr(Oo);
  double* dW=(double*)policyH; double* dM=dW+P;                  /* resident [params(P); mom(P)] */
  size_t SZ=(H*D>O*H?H*D:O*H); size_t md=(H>D?H:D); if(O>md) md=O; size_t SZA=md*md;
  double *dG=(double*)rb2(13,8*P), *dU=(double*)rb2(14,8*SZ), *dX=(double*)rb2(15,8*SZ), *dXt=(double*)rb2(16,8*SZ);
  double *dAa=(double*)rb2(17,8*SZA), *dPp=(double*)rb2(18,8*SZ), *dQq=(double*)rb2(19,8*SZ), *dInv=(double*)rb2(20,8);
  int ok=(P>0 && policyH && dG&&dU&&dX&&dXt&&dAa&&dPp&&dQq&&dInv);
  if(ok){
    cudaMemcpy(dG,gr,8*P,cudaMemcpyHostToDevice);               /* the ONLY per-minibatch weight-side upload */
    int B=256;
    k_scale_const<<<ceildiv((long)P,B),B>>>(dG,gscale,(long)P); /* mean-scale (matches muon_mat_cpu/stepvec_cpu gi=G·gscale) */
    muon_mat_dev((int)H,(int)D, dW+oW1,dG+oW1,dM+oW1, dW+oW1,dM+oW1, lr,wd,mu,eps, dU,dX,dXt,dAa,dPp,dQq,dInv);
    k_stepvec<<<ceildiv((long)H,B),B>>>(dW+ob1,dM+ob1, dW+ob1,dG+ob1,dM+ob1, lr,wd,mu,(long)H);
    muon_mat_dev((int)O,(int)H, dW+oW2,dG+oW2,dM+oW2, dW+oW2,dM+oW2, lr,wd,mu,eps, dU,dX,dXt,dAa,dPp,dQq,dInv);
    k_stepvec<<<ceildiv((long)O,B),B>>>(dW+ob2,dM+ob2, dW+ob2,dG+ob2,dM+ob2, lr,wd,mu,(long)O);
    cudaDeviceSynchronize();
    cudaMemcpy(out,dW,8*P,cudaMemcpyDeviceToHost);             /* new params for the host-param grad */
  } else for(size_t i=0;i<P;i++) out[i]=0.0;
  lean_dec(gRawA);
  return lean_io_result_mk_ok(Oo);
}

/* === Device MLP forward + SoA scatter helpers ====================================================
   The per-timestep GPU forward (`lean_cuda_mlp_forward` / `cudaMlpForwardFFI`) the generic plugin
   trainer uses, plus the env-major column scatter kernels. All env-agnostic — the env itself is a C
   plugin (ocean/<name>/adapter.c) stepped on the host; only the policy runs here. */

/* device MLP forward on a given stream: f32 weights + f32 obs → f32 logits. All work (GEMMs + bias
   kernels) issued on `st`, so two halves on two streams overlap on the device (double-buffer path). */
static void mlp_forward_dev_s(cublasHandle_t h, const float* dW1f, const float* db1f, const float* dW2f,
    const float* db2f, const float* dXf, int N, int D, int H, int O, int bf,
    float* dPre, float* dH1, float* dOut, cudaStream_t st){
  int B=256;
  cublasSetStream(h, st);
  gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,H,N,D, dW1f,D, dXf,D, dPre,H, bf);     /* Zpre[N,H]=Xb·W1ᵀ */
  k_relu_bias<<<ceildiv((long)N*H,B),B,0,st>>>(dH1,dPre,db1f,N,H);
  gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,O,N,H, dW2f,H, dH1,H, dPre,O, bf);     /* Outpre[N,O]=H1·W2ᵀ */
  k_add_bias<<<ceildiv((long)N*O,B),B,0,st>>>(dOut,dPre,db2f,N,O);
}
/* device MLP forward on the default stream (the single-buffer forward FFI + rollout path). */
static void mlp_forward_dev(cublasHandle_t h, const float* dW1f, const float* db1f, const float* dW2f,
    const float* db2f, const float* dXf, int N, int D, int H, int O, int bf,
    float* dPre, float* dH1, float* dOut){
  mlp_forward_dev_s(h,dW1f,db1f,dW2f,db2f,dXf,N,D,H,O,bf,dPre,dH1,dOut,(cudaStream_t)0);
}

/* device forward, host-in/host-out (the reference component for verify-rollout-gpu): params/obs f64 →
   logits f64 via the f32/bf16 gemm32 path. Same math as the driver's per-timestep forward. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_mlp_forward(lean_obj_arg pa, lean_obj_arg obsa,
    size_t N, size_t D, size_t H, size_t O, uint8_t bf16){
  size_t P=H*D+H+O*H+O; int bf=(int)bf16; size_t oW1=0,ob1=H*D,oW2=H*D+H,ob2=H*D+H+O*H;
  const double* par=lean_float_array_cptr(pa); const double* obs=lean_float_array_cptr(obsa);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),N*O,N*O); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  double *dPar=NULL,*dObs=NULL,*dOutF=NULL; float *dW1f=NULL,*db1f=NULL,*dW2f=NULL,*db2f=NULL,*dXf=NULL,*dPre=NULL,*dH1=NULL,*dOut=NULL;
  int ok=(N>0 && h!=NULL
    && !cudaMalloc((void**)&dPar,8*P) && !cudaMalloc((void**)&dObs,8*N*D) && !cudaMalloc((void**)&dOutF,8*N*O)
    && !cudaMalloc((void**)&dW1f,4*H*D) && !cudaMalloc((void**)&db1f,4*H) && !cudaMalloc((void**)&dW2f,4*O*H) && !cudaMalloc((void**)&db2f,4*O)
    && !cudaMalloc((void**)&dXf,4*N*D) && !cudaMalloc((void**)&dPre,4*N*(H>O?H:O)) && !cudaMalloc((void**)&dH1,4*N*H) && !cudaMalloc((void**)&dOut,4*N*O));
  if(ok){
    cudaMemcpy(dPar,par,8*P,cudaMemcpyHostToDevice); cudaMemcpy(dObs,obs,8*N*D,cudaMemcpyHostToDevice);
    int B=256;
    k_f64_to_f32<<<ceildiv((long)H*D,B),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<ceildiv((long)H,B),B>>>(db1f,dPar+ob1,(long)H);
    k_f64_to_f32<<<ceildiv((long)O*H,B),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<ceildiv((long)O,B),B>>>(db2f,dPar+ob2,(long)O);
    k_f64_to_f32<<<ceildiv((long)N*D,B),B>>>(dXf,dObs,(long)N*D);
    mlp_forward_dev(h,dW1f,db1f,dW2f,db2f,dXf,(int)N,(int)D,(int)H,(int)O,bf,dPre,dH1,dOut);
    k_f32_to_f64<<<ceildiv((long)N*O,B),B>>>(dOutF,dOut,(long)N*O);
    cudaDeviceSynchronize(); cudaMemcpy(out,dOutF,8*N*O,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<N*O;i++) out[i]=0.0;
  if(dPar)cudaFree(dPar);if(dObs)cudaFree(dObs);if(dOutF)cudaFree(dOutF);if(dW1f)cudaFree(dW1f);if(db1f)cudaFree(db1f);
  if(dW2f)cudaFree(dW2f);if(db2f)cudaFree(db2f);if(dXf)cudaFree(dXf);if(dPre)cudaFree(dPre);if(dH1)cudaFree(dH1);if(dOut)cudaFree(dOut);
  lean_dec(pa); lean_dec(obsa); return Oo;
}

/* --- Native per-update MLP rollout (single-discrete): the whole T-horizon rollout in ONE FFI call.
   The old path ran the T-loop in Lean, re-uploading the policy weights every timestep, mallocing/freeing
   ~22 device buffers per step, round-tripping logits D2H→H2D between forward and sample, and scattering
   the experience columns through a boxed-array Lean interpreter loop. Here the weights upload + convert to
   f32 ONCE (resident across the horizon); each step does obs H2D → resident forward → device sample →
   sample D2H → CPU plugin env-step (`h->step`) → host column scatter in C. obs H2D (N·D) + actions D2H
   (3N) per step stay — the env is a host CPU plugin (irreducible). Bit-identical to the old loop: same
   `mlp_forward_dev`, same `k_sample` with rng = rolloutRng + s·N·G, same env stepping.
   Returns [obsCol(NT·D); actCol(NT); logpCol(NT); valCol(NT); rewCol(NT); termCol(NT); finalObs(N·D)]
   (row = e·T+s), where finalObs threads the persistent env state to the next update. IO (mutates env). */
static void* g_rb[16]; static size_t g_rbsz[16];
static void* rb_buf(int i, size_t bytes){
  if(g_rbsz[i] < bytes){ if(g_rb[i]) cudaFree(g_rb[i]);
    if(cudaMalloc(&g_rb[i], bytes)!=cudaSuccess){ g_rb[i]=NULL; g_rbsz[i]=0; return NULL; }
    g_rbsz[i]=bytes; }
  return g_rb[i];
}
/* Persistent PINNED host-buffer cache (page-locked, allocated once and reused across updates). Pinned
   memory lets cudaMemcpyAsync overlap with kernels (needed by the double-buffer) and speeds even the
   single-buffer H2D/D2H; allocating once avoids cudaHostAlloc's per-call page-pinning cost (which, if
   done per update, more than eats the transfer win). Falls back to pageable malloc if pinning fails. */
static void* g_hb[8]; static size_t g_hbsz[8];
static void* hb_buf(int i, size_t bytes){
  if(g_hbsz[i] < bytes){ if(g_hb[i]) cudaFreeHost(g_hb[i]);
    if(cudaHostAlloc(&g_hb[i], bytes, cudaHostAllocDefault)!=cudaSuccess){ g_hb[i]=(void*)malloc(bytes); }
    g_hbsz[i]= g_hb[i]? bytes : 0; }
  return g_hb[i];
}


/* --- device-resident policy weights ([params(P); mom(P)] f64, one persistent cudaMalloc) ------------
   PufferLib keeps the policy on-device the whole run; so do we. `policy_load` uploads the initial
   [params;mom] ONCE and returns the device pointer as an opaque USize handle. The rollout reads its
   params, the optimizer (train_update_resident) updates params+mom IN PLACE — neither re-uploads nor
   downloads per update. f64 H2D/D2H is lossless, so this is bit-identical to threading pm through the host.
   `policy_download` reads [params;mom] back on demand (checkpoint/log); `policy_free` releases it. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_policy_load(lean_obj_arg pmA, size_t P, lean_obj_arg w){
  (void)w;
  const double* pm=lean_float_array_cptr(pmA); double* d=NULL; size_t hb=0;
  if(cudaMalloc((void**)&d,8*2*P)==cudaSuccess){ cudaMemcpy(d,pm,8*2*P,cudaMemcpyHostToDevice); hb=(size_t)d; }
  lean_dec(pmA);
  return lean_io_result_mk_ok(lean_box_usize(hb));
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_policy_download(size_t hb, size_t P, lean_obj_arg w){
  (void)w;
  lean_object* Oo=lean_alloc_sarray(sizeof(double),2*P,2*P); double* out=lean_float_array_cptr(Oo);
  if(hb) cudaMemcpy(out,(void*)hb,8*2*P,cudaMemcpyDeviceToHost); else for(size_t i=0;i<2*P;i++) out[i]=0.0;
  return lean_io_result_mk_ok(Oo);
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_policy_free(size_t hb, lean_obj_arg w){
  (void)w; if(hb) cudaFree((void*)hb);
  return lean_io_result_mk_ok(lean_box(0));
}

/* Two CUDA streams + cuBLAS handles + completion events for the double-buffer rollout (lazy, once). The
   double-buffer splits the batch into two env-halves and pipelines them across two streams: while the GPU
   runs half B's forward on stream B, the CPU env-steps half A (whose sample already landed), overlapping
   GPU forward with CPU env-step. Needs pinned host buffers (async copies) — wired via hb_buf when enabled.

   MEASURED: default OFF because it is a NET LOSS on this box (set PUFFER_ROLL_DBUF=1 to force it on). A
   clean, non-portage-contaminated profile of the single-buffer rollout is GPU-BOUND, not env-bound
   (breakout 2M: gpu 6040ms ≫ env+scatter 906ms; the old "env≈gpu" reading was a background portage
   compile starving the env-step worker threads). Two reasons the overlap can't win here: (1) env-step is
   only ~10% of the rollout, so there is little to hide; (2) the GPU is SHARED with a co-resident vLLM
   server that saturates it, so the two streams cannot run concurrently — splitting the batch just doubles
   the GPU wall-time (two half-batch passes) instead of overlapping. Interleaved A/B (breakout 2M):
   single-buffer ~8.6s rollout / 113K SPS; double-buffer 14.0s pinned (18.9s pageable). Kept behind the
   flag for a future re-test on a DEDICATED GPU (no vLLM), where two streams could actually overlap. */
static cudaStream_t g_db_st[2]; static cublasHandle_t g_db_h[2]; static cudaEvent_t g_db_ev[2]; static int g_db_init=0;
static void dbuf_init(void){
  if(g_db_init) return;
  for(int k=0;k<2;k++){ cudaStreamCreate(&g_db_st[k]); cublasCreate(&g_db_h[k]); cudaEventCreate(&g_db_ev[k]); }
  g_db_init=1;
}
static int dbuf_enabled(void){ static int f=-1; if(f<0){ const char* e=getenv("PUFFER_ROLL_DBUF"); f=(e!=NULL && e[0]!='0'); } return f; }

/* ---- Concurrent stream-buffers rollout (PufferLib's overlap design) --------------------------------
   The single-buffer rollout serializes GPU forward and CPU env-step (a sync between them). Split the N
   agents into `nbuf` buffers; each buffer = its own pthread + CUDA stream + cuBLAS handle, running the
   FULL T-horizon independently over its agent slice. The GPU is the one shared resource, so it serializes
   the buffers' forwards — while buffer A env-steps on the CPU, buffer B's forward runs on the GPU. That
   overlaps our two biggest rollout phases (GPU-forward ∥ CPU-env-step) instead of stalling between them.
   Global device scratch + host staging are sliced per buffer (disjoint rows ⇒ no races). Only the sampler
   uses the GLOBAL row index (k_sample_seg), so the per-agent rng matches the single-buffer path (the bf16
   forward's tiling differs by batch, so logits differ within bf16 tolerance — a numerics change, not a
   bug, like our other bf16 paths). Gated by PUFFER_ROLL_BUFFERS (default 1 = single-buffer). */
#define MAXBUF 16
static cudaStream_t g_bufst[MAXBUF]; static cublasHandle_t g_bufh[MAXBUF]; static int g_buf_ready=0;
/* Per-buffer CUDA graph of the per-timestep forward+sample compute (capture once, replay each step → no
   per-kernel launch overhead, PufferLib's last rollout trick). The per-step rng is fed via a device scalar
   (g_dRbase) so it need not rebuild the graph; H2D obs / D2H sample stay OUTSIDE the graph. */
static cudaGraphExec_t g_bufgraph[MAXBUF]; static int g_bufgraph_ok[MAXBUF]; static unsigned long long* g_dRbase[MAXBUF];
/* Lean-settable override for both knobs below, checked before the (still-supported, for manual
   debugging) PUFFER_ROLL_BUFFERS/PUFFER_ROLL_GRAPH env vars. Both defaulted to inert (1 buffer, no
   graph) since launch because NOTHING in the Lean CLI ever set the env vars -- this whole buffered+
   graph-replayed rollout path (already measured breakout MLP@4096 4.73M->7.25M SPS, 8 buffers) sat
   dormant. `lean_cuda_set_roll_buffers` lets `trainPluginEnv` opt in with a size-aware count
   (`min(8, numEnvs)`) instead of relying on an ambient env var nothing sets. */
static int g_roll_buffers_ovr = -1, g_roll_graph_ovr = -1;
static int roll_buffers(void){ if(g_roll_buffers_ovr>=0) return g_roll_buffers_ovr;
    static int f=-1; if(f<0){ const char* e=getenv("PUFFER_ROLL_BUFFERS"); int n=0;
    if(e){ while(*e>='0'&&*e<='9'){ n=n*10+(*e-'0'); e++; } } if(n<1) n=1; if(n>MAXBUF) n=MAXBUF; f=n; } return f; }
/* CUDA graph replay of the per-step forward — DEFAULT ON with the buffered path (PUFFER_ROLL_GRAPH=0 off).
   +14% at launch-bound configs (small per-buffer forward, e.g. 512 envs/8 buf); ~neutral at the GPU-saturated
   peak (4096/8, where genuine GPU execution — not launch overhead — is the bottleneck). Never negative;
   per-buffer eager fallback if capture fails. */
static int roll_graph(void){ if(g_roll_graph_ovr>=0) return g_roll_graph_ovr;
    static int f=-1; if(f<0){ const char* e=getenv("PUFFER_ROLL_GRAPH"); f=(e==NULL||e[0]!='0'); } return f; }
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_set_roll_buffers(size_t nbuf, uint8_t graphOn, lean_obj_arg w){
  (void)w;
  int n=(int)nbuf; if(n<1) n=1; if(n>MAXBUF) n=MAXBUF;
  g_roll_buffers_ovr = n; g_roll_graph_ovr = graphOn ? 1 : 0;
  return lean_io_result_mk_ok(lean_box(0));
}
static void* buf_worker(void* arg);
/* M4a, ported: persistent MLP buffer-worker threads (kills the per-update pthread_create+join pair
   mg_buf_worker's twin already proved worth 1.4-4.8% for MinGRU). Same design, same fix: sized to
   MAXBUF always so the barrier's party count is fixed once threads exist; every thread is released
   every call and self-gates via buf_worker's own (newly added) nb<=0 early-return. On a partial
   pthread_create failure, abandon the already-started threads (parked forever in their first wait —
   harmless, matches every persistent pool in this file having no teardown path) rather than trying
   to wake them: an earlier version of this exact pattern deadlocked doing that (adversarially caught
   for the MinGRU twin) — the barrier needs MAXBUF+1 total arrivals and at most b+1 can ever come once
   the remaining creates fail, so any wake attempt blocks forever instead of falling back. */
static pthread_t g_bufw_th[MAXBUF]; static pthread_barrier_t g_bufw_bar;
static int g_bufw_n=0, g_bufw_alive=0;
static void* bufw_thread(void* arg){
  long b=(long)arg;
  for(;;){ pthread_barrier_wait(&g_bufw_bar); if(!g_bufw_alive) return NULL;
    buf_worker((void*)b); pthread_barrier_wait(&g_bufw_bar); }
}
static int bufw_init(void){
  static int fail=0; if(fail) return 0;
  if(g_bufw_n==MAXBUF) return 1;
  if(g_bufw_n!=0) return 0;
  if(pthread_barrier_init(&g_bufw_bar,NULL,MAXBUF+1)!=0){ fail=1; return 0; }
  g_bufw_alive=1;
  for(long b=0;b<MAXBUF;b++){
    if(pthread_create(&g_bufw_th[b],NULL,bufw_thread,(void*)b)!=0){
      g_bufw_alive=0; fail=1; return 0;   /* abandon — see comment above; do NOT touch the barrier again */
    }
  }
  g_bufw_n=MAXBUF; return 1;
}
static void buf_init(int nbuf){
  if(g_buf_ready>=nbuf) return;
  for(int b=g_buf_ready;b<nbuf;b++){ cudaStreamCreate(&g_bufst[b]); cublasCreate(&g_bufh[b]);
    void* ws=NULL; if(cudaMalloc(&ws,4*1024*1024)==cudaSuccess) cublasSetWorkspace(g_bufh[b],ws,4*1024*1024);
    cudaMalloc((void**)&g_dRbase[b],sizeof(unsigned long long)); g_bufgraph_ok[b]=0; }
  g_buf_ready=nbuf;
}
/* Device-resident obs trajectory (f32, NT·D row-major, row e·T+s) — the rollout's biggest per-step cost was
   the CPU scatter of obs into the host obsCol column (~half the peak rollout). Instead, scatter the f32 obs
   (already on-device as dXf) into this device buffer with a kernel, and have the training read it directly:
   no CPU obs copy, no obs H2D, no host f64→f32 in the trainer. Bit-identical — both are (float)cur. Persists
   between the rollout and train FFI calls; g_dObsTraj_valid is set by the buffered rollout and consumed by
   lean_cuda_train_update_resident (single-buffer leaves it 0 → trainer H2D's obsA). Globals declared up by ts_buf. */
static float* obstraj_buf(size_t elems){ size_t bytes=elems*4;
  if(g_dObsTrajSz<bytes){ if(g_dObsTraj)cudaFree(g_dObsTraj);
    if(cudaMalloc((void**)&g_dObsTraj,bytes)!=cudaSuccess){ g_dObsTraj=NULL; g_dObsTrajSz=0; return NULL; }
    g_dObsTrajSz=bytes; }
  return g_dObsTraj; }
__global__ void k_scatter_obs_traj(float* traj, const float* xf, long nb, long rb, long s, long T, long D){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=nb*D) return;
  long le=idx/D, j=idx%D; long row=(rb+le)*T+s;           /* xf = dXf+rb*D (buffer slice); traj row e·T+s */
  traj[row*D+j]=xf[le*D+j]; }
typedef struct {
  Handle* eh; const double* obs0;
  const float *dW1f,*db1f,*dW2f,*db2f;
  double *dObs,*dY,*dO; float *dXf,*dPre,*dH1,*dOut;   /* global device scratch, sliced by row */
  double *hSamp,*hA,*hB,*hRT,*actRM;                   /* global host staging, sliced by row */
  double *obsCol,*actCol,*logpCol,*valCol,*rewCol,*termCol;
  long N,D,T; int H,O,A,bf,nAg; uint64_t rolloutRng;
  int rowBase[MAXBUF], rowN[MAXBUF], envLo[MAXBUF], envHi[MAXBUF];
} bufpool_t;
static bufpool_t g_bp;
static void* buf_worker(void* arg){
  int b=(int)(long)arg; bufpool_t* p=&g_bp;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  long D=p->D, T=p->T, N=p->N; int H=p->H,O=p->O,A=p->A,bf=p->bf,nAg=p->nAg,B=256;
  long rb=p->rowBase[b], nb=p->rowN[b]; int elo=p->envLo[b], enk=p->envHi[b]-p->envLo[b];
  if(nb<=0) return NULL;   /* idle slot (persistent pool sized to MAXBUF > this call's nbuf) */
  long maxHO=(H>O?H:O);
  cudaStream_t st=g_bufst[b]; cublasHandle_t h=g_bufh[b]; cublasSetStream(h,st);
  int useGraph=roll_graph();
  /* the per-timestep GPU compute (obs f32-cast → MLP forward → f64 logits → sample); devRng reads rng from
     the device scalar g_dRbase[b] (so it is capturable into a graph), else takes rng as an arg. */
  auto compute=[&](int devRng, unsigned long long rbaseArg){
    k_f64_to_f32<<<ceildiv(nb*D,B),B,0,st>>>(p->dXf+rb*D, p->dObs+rb*D, nb*D);
    mlp_forward_dev_s(h,p->dW1f,p->db1f,p->dW2f,p->db2f,p->dXf+rb*D,(int)nb,(int)D,H,O,bf,
                      p->dPre+rb*maxHO,p->dH1+rb*H,p->dOut+rb*O, st);
    if(devRng) k_sample_seg_g_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, p->dO+3*rb, (int)nb, (int)rb, A, O, g_dRbase[b]);
    else       k_sample_seg_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, p->dO+3*rb, (int)nb, (int)rb, A, O, rbaseArg);
  };
  int prof=(b==0)&&(getenv("PUFFER_ROLL_PROFILE")!=NULL);   /* buffer 0's per-step critical-path breakdown */
  static double BR_gpu=0,BR_env=0,BR_scat=0; double bt0=0;
  const double* cur=p->obs0; double* nxt=p->hA;
  for(size_t s=0;s<(size_t)T;s++){
    unsigned long long rbase=(unsigned long long)(p->rolloutRng+(uint64_t)((long)s*N)*G);
    if(prof) bt0=now_ms();
    cudaMemcpyAsync(p->dObs+rb*D, cur+rb*D, 8*(size_t)nb*D, cudaMemcpyHostToDevice, st);   /* H2D outside graph */
    if(useGraph && g_bufgraph_ok[b]>=0){
      k_set_u64<<<1,1,0,st>>>(g_dRbase[b], rbase);                    /* per-step rng into the device scalar */
      if(g_bufgraph_ok[b]==1){ cudaGraphLaunch(g_bufgraph[b], st); }
      else {                                                          /* capture once, then replay */
        cudaGraph_t gr=NULL; cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        compute(1, 0);
        cudaError_t ce=cudaStreamEndCapture(st,&gr);
        if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_bufgraph[b],gr,0)==cudaSuccess){
          g_bufgraph_ok[b]=1; cudaGraphDestroy(gr); cudaGraphLaunch(g_bufgraph[b], st); }
        else { g_bufgraph_ok[b]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError(); compute(0, rbase); }
      }
    } else compute(0, rbase);
    if(g_dObsTraj) k_scatter_obs_traj<<<ceildiv(nb*D,B),B,0,st>>>(g_dObsTraj, p->dXf+rb*D, nb, rb, (long)s, T, D);   /* device obs scatter (no CPU copy) */
    cudaMemcpyAsync(p->hSamp+3*rb, p->dO+3*rb, 8*3*(size_t)nb, cudaMemcpyDeviceToHost, st);   /* D2H outside graph */
    cudaStreamSynchronize(st);                         /* wait for THIS buffer's GPU (others run meanwhile) */
    if(prof){ BR_gpu+=now_ms()-bt0; bt0=now_ms(); }
    for(long i=0;i<nb;i++) p->actRM[rb+i]=p->hSamp[3*rb+i];   /* compact act → global row-major */
    p->eh->step_range(p->eh->env, p->actRM, nxt, p->hRT, p->hRT+N, elo, enk);   /* CPU env-step this buffer */
    if(prof){ BR_env+=now_ms()-bt0; bt0=now_ms(); }
    for(long e=(long)elo*nAg; e<(long)(elo+enk)*nAg; e++){ long row=e*T+(long)s, le=e-rb;
      if(!g_dObsTraj) memcpy(&p->obsCol[row*D], &cur[e*D], (size_t)D*sizeof(double));   /* obs on host only if not device-resident */
      p->actCol[row]=p->hSamp[3*rb+le]; p->logpCol[row]=p->hSamp[3*rb+nb+le]; p->valCol[row]=p->hSamp[3*rb+2*nb+le];
      p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
    if(prof) BR_scat+=now_ms()-bt0;
    cur=nxt; nxt=(nxt==p->hA)?p->hB:p->hA;
  }
  if(prof) fprintf(stderr,"[buf0] gpu(h2d+fwd+sample+sync)=%.0f  env-step=%.0f  scatter=%.0f  ms (cumulative, 1 buffer)\n",BR_gpu,BR_env,BR_scat);
  return NULL;
}

extern "C" LEAN_EXPORT lean_obj_res lean_cuda_plugin_rollout(
    size_t hh, size_t policyH, lean_obj_arg obs0a,
    size_t N, size_t D, size_t H, size_t A, size_t T, uint8_t bf16, uint64_t rolloutRng, lean_obj_arg w){
  (void)w;
  Handle* eh=(Handle*)hh;
  size_t O=A+1; int bf=(int)bf16;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  const double* obs0=lean_float_array_cptr(obs0a);
  long NT=(long)N*T, cols=NT*((long)D+5)+(long)N*D;   /* + finalObs(N·D) */
  lean_object* Oo=lean_alloc_sarray(sizeof(double),cols,cols); double* out=lean_float_array_cptr(Oo);
  double *obsCol=out, *actCol=out+NT*D, *logpCol=actCol+NT, *valCol=logpCol+NT, *rewCol=valCol+NT, *termCol=rewCol+NT;
  double *finalObs=termCol+NT;
  cublasHandle_t h=cu_handle();
  size_t oW1=0,ob1=H*D,oW2=H*D+H,ob2=H*D+H+O*H;
  double *dPar=(double*)policyH, *dObs=(double*)rb_buf(1,8*N*D), *dY=(double*)rb_buf(2,8*N*O), *dO=(double*)rb_buf(3,8*3*N);
  float *dW1f=(float*)rb_buf(4,4*H*D),*db1f=(float*)rb_buf(5,4*H),*dW2f=(float*)rb_buf(6,4*O*H),*db2f=(float*)rb_buf(7,4*O);
  float *dXf=(float*)rb_buf(8,4*N*D),*dPre=(float*)rb_buf(9,4*N*(H>O?H:O)),*dH1=(float*)rb_buf(10,4*N*H),*dOut=(float*)rb_buf(11,4*N*O);
  /* Persistent PINNED host buffers (allocated once, reused) so cudaMemcpyAsync is truly async for the
     double-buffer and the single-buffer H2D/D2H are faster — without paying cudaHostAlloc per update. */
  /* Host staging buffers. Pinned (page-locked) ONLY when the double-buffer is enabled — it needs
     cudaMemcpyAsync to overlap the two streams. The default single-buffer path is ~10% FASTER with plain
     pageable malloc on this box (measured, interleaved A/B), so pin only on demand: persistent pinned
     cache (allocated once) when dbuf is on, plain malloc (freed below) otherwise. */
  int wantPin = dbuf_enabled() || roll_buffers()>1;    /* pinned host staging for async overlap */
  double *hSamp,*hA,*hB,*hRT;
  if(wantPin){ hSamp=(double*)hb_buf(0,8*3*N); hA=(double*)hb_buf(1,8*N*D); hB=(double*)hb_buf(2,8*N*D); hRT=(double*)hb_buf(3,8*2*N); }
  else       { hSamp=(double*)malloc(8*3*N);   hA=(double*)malloc(8*N*D);   hB=(double*)malloc(8*N*D);   hRT=(double*)malloc(8*2*N);   }
  double *actRMf=NULL;                                              /* double-buffer only: global row-major actions */
  int ok=(N>0 && h!=NULL && eh && dPar&&dObs&&dY&&dO&&dW1f&&db1f&&dW2f&&db2f&&dXf&&dPre&&dH1&&dOut&&hSamp&&hA&&hB&&hRT);
  if(ok){
    cublasSetStream(h,0);
    int B=256;
    /* weights already device-resident (dPar = policy handle); convert to f32 ONCE (no PCIe upload) */
    k_f64_to_f32<<<ceildiv((long)H*D,B),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<ceildiv((long)H,B),B>>>(db1f,dPar+ob1,(long)H);
    k_f64_to_f32<<<ceildiv((long)O*H,B),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<ceildiv((long)O,B),B>>>(db2f,dPar+ob2,(long)O);
    /* Persistent-thread env-step: workers split env-step + column-scatter while main drives the GPU.
       Serial fallback when the plugin predates puffer_env_step_range. */
    int threaded = (eh->step_range != NULL);
    int nAg = eh->numAgents>0 ? eh->numAgents : 1;
    int nbuf = roll_buffers();
    g_dObsTraj_valid = 0;                                           /* single-buffer path → trainer H2D's obsA */
    int useBuffered = (nbuf>1 && threaded && (int)eh->N>=nbuf);
    double* bufActRM = useBuffered ? (double*)malloc(8*N) : NULL;   /* NULL (alloc fail) → graceful fallback to single-buffer */
    if(useBuffered && bufActRM){
      /* CONCURRENT STREAM-BUFFERS: nbuf independent (thread+stream) rollout pipelines; the GPU serializes
         their forwards so each buffer's CPU env-step overlaps the next buffer's GPU forward. */
      buf_init(nbuf);
      obstraj_buf((size_t)NT*D); g_dObsTraj_valid = (g_dObsTraj!=NULL);   /* device-resident obs trajectory */
      cudaDeviceSynchronize();                                      /* f32 weights ready before workers read them */
      g_bp.eh=eh; g_bp.obs0=obs0; g_bp.dW1f=dW1f; g_bp.db1f=db1f; g_bp.dW2f=dW2f; g_bp.db2f=db2f;
      g_bp.dObs=dObs; g_bp.dY=dY; g_bp.dO=dO; g_bp.dXf=dXf; g_bp.dPre=dPre; g_bp.dH1=dH1; g_bp.dOut=dOut;
      g_bp.hSamp=hSamp; g_bp.hA=hA; g_bp.hB=hB; g_bp.hRT=hRT; g_bp.actRM=bufActRM;
      g_bp.obsCol=obsCol; g_bp.actCol=actCol; g_bp.logpCol=logpCol; g_bp.valCol=valCol; g_bp.rewCol=rewCol; g_bp.termCol=termCol;
      g_bp.N=(long)N; g_bp.D=(long)D; g_bp.T=(long)T; g_bp.H=(int)H; g_bp.O=(int)O; g_bp.A=(int)A; g_bp.bf=bf; g_bp.nAg=nAg; g_bp.rolloutRng=rolloutRng;
      int totalEnv=eh->N;
      for(int b=0;b<nbuf;b++){ int elo=(int)((long)b*totalEnv/nbuf), ehi=(int)((long)(b+1)*totalEnv/nbuf);
        g_bp.envLo[b]=elo; g_bp.envHi[b]=ehi; g_bp.rowBase[b]=(int)((long)elo*nAg); g_bp.rowN[b]=(int)((long)(ehi-elo)*nAg); }
      for(int b=nbuf;b<MAXBUF;b++) g_bp.rowN[b]=0;   /* idle slots: buf_worker's nb<=0 early-return */
      if(bufw_init()){
        pthread_barrier_wait(&g_bufw_bar);            /* release all MAXBUF workers */
        pthread_barrier_wait(&g_bufw_bar);             /* join */
      } else {
        pthread_t bth[MAXBUF];                         /* pool init failed once — permanent fallback */
        for(int b=0;b<nbuf;b++) pthread_create(&bth[b],NULL,buf_worker,(void*)(long)b);
        for(int b=0;b<nbuf;b++) pthread_join(bth[b],NULL);
      }
      const double* fin=((T%2)==1)?hA:hB;                           /* after T ping-pong steps */
      for(size_t i=0;i<N*D;i++) finalObs[i]=fin[i];
      free(bufActRM);
    } else {
    int useDbuf = dbuf_enabled() && threaded && eh->N>=2;           /* double-buffer needs ≥2 envs to split */
    if(useDbuf){ dbuf_init(); actRMf=(double*)malloc(8*N); if(!actRMf) useDbuf=0; cudaDeviceSynchronize(); } /* weights f32-converted before the streams read them */
    if(threaded && !g_rp_init){
      g_rp.nthreads=rp_threads(); g_rp.alive=1;
      pthread_barrier_init(&g_rp.bar,NULL,g_rp.nthreads+1);
      for(long t=0;t<g_rp.nthreads;t++) pthread_create(&g_rp_th[t],NULL,rp_worker,(void*)t);
      g_rp_init=1;
    }
    if(threaded){                                                   /* shape-invariant fields (per call) */
      g_rp.eh=eh; g_rp.obsCol=obsCol; g_rp.actCol=actCol; g_rp.logpCol=logpCol; g_rp.valCol=valCol;
      g_rp.rewCol=rewCol; g_rp.termCol=termCol; g_rp.N=(long)N; g_rp.D=(long)D; g_rp.T=(long)T;
      g_rp.nAgents=nAg; g_rp.envLo=0; g_rp.envHi=eh->N; g_rp.rowBase=0; g_rp.Nstride=(long)N; g_rp.W=1;               /* single discrete action per row */
    }
    static int rprof=-1; if(rprof<0) rprof=(getenv("PUFFER_ROLL_PROFILE")!=NULL);  /* phase attribution */
    static double R_h2d=0,R_gpu=0,R_d2h=0,R_env=0; double rt0=0;
    int EA=eh->N/2; long RA=(long)EA*nAg;                           /* double-buffer env/row split */
    int Elo[2]={0,EA}, Enk[2]={EA, eh->N-EA}; long Rlo[2]={0,RA}, Rn[2]={RA,(long)N-RA};
    const double* cur=obs0; double* nxt=hA;                         /* ping-pong obs; obs0 read-only */
    for(size_t s=0;s<T;s++){
     if(useDbuf){
      /* Two env-halves pipelined on two streams: queue both halves' forward+sample, then env-step half k
         as soon as ITS sample lands — half 0's env-step (on the worker pool) overlaps half 1's GPU forward. */
      unsigned long long rbase=(unsigned long long)(rolloutRng+(uint64_t)(s*N)*G);
      for(int k=0;k<2;k++){                                         /* queue both halves on their own stream */
        long ro=Rlo[k], nk=Rn[k]; cudaStream_t st=g_db_st[k]; cublasHandle_t bh=g_db_h[k];
        cudaMemcpyAsync(dObs+ro*D, cur+ro*D, 8*(size_t)nk*D, cudaMemcpyHostToDevice, st);
        k_f64_to_f32<<<ceildiv(nk*(long)D,B),B,0,st>>>(dXf+ro*D, dObs+ro*D, nk*(long)D);
        mlp_forward_dev_s(bh,dW1f,db1f,dW2f,db2f,dXf+ro*D,(int)nk,(int)D,(int)H,(int)O,bf,
                          dPre+ro*(long)(H>O?H:O),dH1+ro*(long)H,dOut+ro*(long)O, st);
        k_f32_to_f64<<<ceildiv(nk*(long)O,B),B,0,st>>>(dY+ro*O, dOut+ro*O, nk*(long)O);
        k_sample_seg<<<ceildiv(nk,B),B,0,st>>>(dY+ro*O, dO+3*ro, (int)nk, (int)ro, (int)A, (int)O, rbase);
        cudaMemcpyAsync(hSamp+3*ro, dO+3*ro, 8*3*(size_t)nk, cudaMemcpyDeviceToHost, st);   /* compact [act;logp;val] */
        cudaEventRecord(g_db_ev[k], st);
      }
      for(int k=0;k<2;k++){                                         /* half k: env-step+scatter once its sample lands */
        cudaEventSynchronize(g_db_ev[k]);
        for(long e=Rlo[k]; e<Rlo[k]+Rn[k]; e++) actRMf[e]=hSamp[3*Rlo[k] + (e-Rlo[k])];    /* actions → global row-major */
        g_rp.s=s; g_rp.cur=cur; g_rp.nxt=nxt; g_rp.hSamp=hSamp+3*Rlo[k]; g_rp.actRM=actRMf; g_rp.hRT=hRT;
        g_rp.envLo=Elo[k]; g_rp.envHi=Elo[k]+Enk[k]; g_rp.rowBase=Rlo[k]; g_rp.Nstride=Rn[k];
        pthread_barrier_wait(&g_rp.bar);                            /* release workers: env-step + scatter */
        pthread_barrier_wait(&g_rp.bar);                            /* wait until this half's partitions done */
      }
     } else {
      if(rprof) rt0=now_ms();
      cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice);
      if(rprof){ R_h2d+=now_ms()-rt0; rt0=now_ms(); }
      k_f64_to_f32<<<ceildiv((long)N*D,B),B>>>(dXf,dObs,(long)N*D);
      mlp_forward_dev(h,dW1f,db1f,dW2f,db2f,dXf,(int)N,(int)D,(int)H,(int)O,bf,dPre,dH1,dOut);
      k_sample_f32<<<ceildiv((long)N,B),B>>>(dOut,dO,(int)N,(int)A,(int)O,(unsigned long long)(rolloutRng+(uint64_t)(s*N)*G));   /* f32 sampler reads the f32 logits directly (no f64 upcast) */
      cudaDeviceSynchronize();
      if(rprof){ R_gpu+=now_ms()-rt0; rt0=now_ms(); }
      cudaMemcpy(hSamp,dO,8*3*N,cudaMemcpyDeviceToHost);             /* [acts(N); logps(N); vals(N)] */
      if(rprof){ R_d2h+=now_ms()-rt0; rt0=now_ms(); }
      if(threaded){
        g_rp.s=s; g_rp.cur=cur; g_rp.nxt=nxt; g_rp.hSamp=hSamp; g_rp.actRM=hSamp; g_rp.hRT=hRT;
        pthread_barrier_wait(&g_rp.bar);                            /* release workers: env-step + scatter */
        pthread_barrier_wait(&g_rp.bar);                            /* wait until all partitions done */
      } else {
        eh->step(eh->env, hSamp, nxt, hRT, hRT+N);                  /* CPU env step: obs'→nxt, rew→hRT, term→hRT+N */
        for(size_t e=0;e<N;e++){ long row=(long)e*T+(long)s;
          for(size_t j=0;j<D;j++) obsCol[row*D+j]=cur[e*D+j];       /* obs BEFORE the step */
          actCol[row]=hSamp[e]; logpCol[row]=hSamp[N+e]; valCol[row]=hSamp[2*N+e];
          rewCol[row]=hRT[e]; termCol[row]=hRT[N+e]; }
      }
      if(rprof) R_env+=now_ms()-rt0;
     }
     cur=nxt; nxt=(nxt==hA)?hB:hA;                                  /* advance (no aliasing) */
    }
    if(rprof) fprintf(stderr,"[roll] h2d=%.0f gpu=%.0f d2h=%.0f env+scatter=%.0f ms (cumulative)\n",R_h2d,R_gpu,R_d2h,R_env);
    for(size_t i=0;i<N*D;i++) finalObs[i]=cur[i];                   /* persistent obs for the next update */
    }
  } else for(long i=0;i<cols;i++) out[i]=0.0;
  if(!wantPin){ free(hSamp); free(hA); free(hB); free(hRT); }     /* pinned buffers are the persistent cache — not freed here */
  free(actRMf);
  lean_dec(obs0a);
  return lean_io_result_mk_ok(Oo);
}

/* ---- Buffered + graph-replayed rollout for the wide (MD/Cont) plugin trainers ------------------
   The W-wide twin of bufpool_t/buf_worker: same concurrent-stream-buffer design (each buffer owns a
   disjoint row range, its own CUDA stream/cuBLAS handle, its own graph slot — all REUSING the MLP
   sibling's g_bufst/g_bufh/g_dRbase/g_bufgraph (plus roll_buffers()/roll_graph()) globals, safe because
   a `puffer train` process only ever runs one trainer to completion), generalized for W-wide actions
   (K heads / d dims) via the new k_sample_md_seg_f32/k_sample_cont_seg_f32 kernels above. Per-buffer
   compact device output `dO` is (W+2)-wide per row (not 3-wide like the single-discrete case):
   [act(W×nb col-major); logp(nb); val(nb)], chunks laid back-to-back at offset (W+2)·rowBase[b] —
   same "per-buffer contiguous chunk" layout bufpool_t's dO already uses, just W+2 wide instead of 3. */
typedef struct {
  Handle* eh; const double* obs0;
  const float *dW1f,*db1f,*dW2f,*db2f;
  double *dObs,*dO; float *dXf,*dPre,*dH1,*dOut;
  double *hSamp,*hA,*hB,*hRT,*actRM;
  double *obsCol,*actCol,*logpCol,*valCol,*rewCol,*termCol;
  long N,D,T; int H,O,W,mode,bf,nAg; uint64_t rolloutRng;
  int* dHs;                                            /* device head sizes (mode 1 only), shared (not per-buffer) */
  int rowBase[MAXBUF], rowN[MAXBUF], envLo[MAXBUF], envHi[MAXBUF];
} bufpool_wide_t;
static bufpool_wide_t g_bpw;
static void* buf_worker_wide(void* arg){
  int b=(int)(long)arg; bufpool_wide_t* p=&g_bpw;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  long D=p->D, T=p->T, N=p->N; int H=p->H,O=p->O,W=p->W,mode=p->mode,bf=p->bf,nAg=p->nAg,B=256;
  long rb=p->rowBase[b], nb=p->rowN[b]; int elo=p->envLo[b], enk=p->envHi[b]-p->envLo[b];
  if(nb<=0) return NULL;
  long maxHO=(H>O?H:O);
  cudaStream_t st=g_bufst[b]; cublasHandle_t h=g_bufh[b]; cublasSetStream(h,st);
  int useGraph=roll_graph();
  auto compute=[&](int devRng, unsigned long long rbaseArg){
    k_f64_to_f32<<<ceildiv(nb*D,B),B,0,st>>>(p->dXf+rb*D, p->dObs+rb*D, nb*D);
    mlp_forward_dev_s(h,p->dW1f,p->db1f,p->dW2f,p->db2f,p->dXf+rb*D,(int)nb,(int)D,H,O,bf,
                      p->dPre+rb*maxHO,p->dH1+rb*H,p->dOut+rb*O, st);
    double* outp=p->dO+(long)(W+2)*rb;
    if(mode==1){
      if(devRng) k_sample_md_seg_g_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, outp, (int)nb, (int)rb, W, p->dHs, O, g_dRbase[b]);
      else       k_sample_md_seg_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, outp, (int)nb, (int)rb, W, p->dHs, O, rbaseArg);
    } else {
      if(devRng) k_sample_cont_seg_g_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, outp, (int)nb, (int)rb, W, O, g_dRbase[b]);
      else       k_sample_cont_seg_f32<<<ceildiv(nb,B),B,0,st>>>(p->dOut+rb*O, outp, (int)nb, (int)rb, W, O, rbaseArg);
    }
  };
  int prof=(b==0)&&(getenv("PUFFER_ROLL_PROFILE")!=NULL);   /* buffer 0's per-step critical-path breakdown */
  static double WR_gpu=0,WR_env=0,WR_scat=0; double bt0=0;
  const double* cur=p->obs0; double* nxt=p->hA;
  for(size_t s=0;s<(size_t)T;s++){
    unsigned long long rbase=(unsigned long long)(p->rolloutRng+(uint64_t)((long)s*N)*G);
    if(prof) bt0=now_ms();
    cudaMemcpyAsync(p->dObs+rb*D, cur+rb*D, 8*(size_t)nb*D, cudaMemcpyHostToDevice, st);
    if(useGraph && g_bufgraph_ok[b]>=0){
      k_set_u64<<<1,1,0,st>>>(g_dRbase[b], rbase);
      if(g_bufgraph_ok[b]==1){ cudaGraphLaunch(g_bufgraph[b], st); }
      else {
        cudaGraph_t gr=NULL; cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        compute(1, 0);
        cudaError_t ce=cudaStreamEndCapture(st,&gr);
        if(ce==cudaSuccess && gr && cudaGraphInstantiate(&g_bufgraph[b],gr,0)==cudaSuccess){
          g_bufgraph_ok[b]=1; cudaGraphDestroy(gr); cudaGraphLaunch(g_bufgraph[b], st); }
        else { g_bufgraph_ok[b]=-1; if(gr) cudaGraphDestroy(gr); cudaGetLastError(); compute(0, rbase); }
      }
    } else compute(0, rbase);
    if(g_dObsTraj) k_scatter_obs_traj<<<ceildiv(nb*D,B),B,0,st>>>(g_dObsTraj, p->dXf+rb*D, nb, rb, (long)s, T, D);
    cudaMemcpyAsync(p->hSamp+(size_t)(W+2)*rb, p->dO+(size_t)(W+2)*rb, 8*(size_t)(W+2)*(size_t)nb, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    if(prof){ WR_gpu+=now_ms()-bt0; bt0=now_ms(); }
    for(long i=0;i<nb;i++) for(int wc=0;wc<W;wc++)
      p->actRM[(rb+i)*W+wc] = p->hSamp[(size_t)(W+2)*rb + (size_t)wc*nb + i];
    p->eh->step_range(p->eh->env, p->actRM, nxt, p->hRT, p->hRT+N, elo, enk);
    if(prof){ WR_env+=now_ms()-bt0; bt0=now_ms(); }
    for(long e=(long)elo*nAg; e<(long)(elo+enk)*nAg; e++){ long row=e*T+(long)s, le=e-rb;
      if(!g_dObsTraj) memcpy(&p->obsCol[row*D], &cur[e*D], (size_t)D*sizeof(double));
      for(int wc=0;wc<W;wc++) p->actCol[row*W+wc] = p->hSamp[(size_t)(W+2)*rb + (size_t)wc*nb + le];
      p->logpCol[row] = p->hSamp[(size_t)(W+2)*rb + (size_t)W*nb + le];
      p->valCol[row]  = p->hSamp[(size_t)(W+2)*rb + (size_t)(W+1)*nb + le];
      p->rewCol[row]=p->hRT[e]; p->termCol[row]=p->hRT[N+e]; }
    if(prof) WR_scat+=now_ms()-bt0;
    cur=nxt; nxt=(nxt==p->hA)?p->hB:p->hA;
  }
  if(prof) fprintf(stderr,"[bufW0] gpu(h2d+fwd+sample+sync)=%.0f  env-step=%.0f  scatter=%.0f  ms (cumulative, 1 buffer)\n",WR_gpu,WR_env,WR_scat);
  return NULL;
}
static pthread_t g_bufw_wide_th[MAXBUF]; static pthread_barrier_t g_bufw_wide_bar;
static int g_bufw_wide_n=0, g_bufw_wide_alive=0;
static void* bufw_wide_thread(void* arg){
  long b=(long)arg;
  for(;;){ pthread_barrier_wait(&g_bufw_wide_bar); if(!g_bufw_wide_alive) return NULL;
    buf_worker_wide((void*)b); pthread_barrier_wait(&g_bufw_wide_bar); }
}
static int bufw_wide_init(void){
  static int fail=0; if(fail) return 0;
  if(g_bufw_wide_n==MAXBUF) return 1;
  if(g_bufw_wide_n!=0) return 0;
  if(pthread_barrier_init(&g_bufw_wide_bar,NULL,MAXBUF+1)!=0){ fail=1; return 0; }
  g_bufw_wide_alive=1;
  for(long b=0;b<MAXBUF;b++){
    if(pthread_create(&g_bufw_wide_th[b],NULL,bufw_wide_thread,(void*)b)!=0){
      g_bufw_wide_alive=0; fail=1; return 0;   /* abandon, do NOT touch the barrier again — see bufw_init's comment */
    }
  }
  g_bufw_wide_n=MAXBUF; return 1;
}

/* Native per-update rollout driver for the MULTI-DISCRETE (mode 1) and CONTINUOUS/Gaussian (mode 2)
   plugin trainers — the W-wide-action twin of lean_cuda_plugin_rollout. Same resident-weights /
   device-forward / ping-pong-obs / persistent-thread env-step structure; the only per-mode differences
   are the sampler kernel (k_sample_md vs k_sample_cont) and the action width W (K heads / d dims). Both
   samplers write hSamp = [act(W×N col-major); logp(N); val(N)]; the driver transposes to row-major actRM
   for the env-step. hsA = headSizes (mode 1 only; ignored for mode 2). O passed directly (Σheads+1 /
   2·d+1). Output SoA columns [obs(NT·D); act(NT·W, row-major); logp(NT); val(NT); rew(NT); term(NT);
   finalObs(N·D)]. Bit-identical to the old per-step FFI loop (same forward, same sampler rng) in the
   non-buffered case; buffered (roll_buffers()>1) is tolerance-close (bf16 tiling + f32 sampling differs
   from the whole-batch path), same trade as the MLP sibling's buffered rollout. */
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_plugin_rollout_multi(
    size_t hh, size_t policyH, lean_obj_arg obs0a, lean_obj_arg hsA,
    size_t N, size_t D, size_t H, size_t O, size_t Wdim, size_t T, uint32_t mode, uint8_t bf16,
    uint64_t rolloutRng, lean_obj_arg w){
  (void)w;
  Handle* eh=(Handle*)hh;
  int bf=(int)bf16; int W=(int)Wdim;
  const uint64_t G=0x9E3779B97F4A7C15ULL;
  const double* obs0=lean_float_array_cptr(obs0a);
  const double* hs=lean_float_array_cptr(hsA);
  long NT=(long)N*T, cols=NT*((long)D+W+4)+(long)N*D;              /* obs+act(W)+logp+val+rew+term + finalObs */
  lean_object* Oo=lean_alloc_sarray(sizeof(double),cols,cols); double* out=lean_float_array_cptr(Oo);
  double *obsCol=out, *actCol=out+NT*D, *logpCol=actCol+NT*W, *valCol=logpCol+NT, *rewCol=valCol+NT, *termCol=rewCol+NT;
  double *finalObs=termCol+NT;
  cublasHandle_t h=cu_handle();
  size_t oW1=0,ob1=H*D,oW2=H*D+H,ob2=H*D+H+O*H;
  double *dPar=(double*)policyH, *dObs=(double*)rb_buf(1,8*N*D), *dY=(double*)rb_buf(2,8*N*O), *dO=(double*)rb_buf(3,8*(size_t)(W+2)*N);
  float *dW1f=(float*)rb_buf(4,4*H*D),*db1f=(float*)rb_buf(5,4*H),*dW2f=(float*)rb_buf(6,4*O*H),*db2f=(float*)rb_buf(7,4*O);
  float *dXf=(float*)rb_buf(8,4*N*D),*dPre=(float*)rb_buf(9,4*N*(H>O?H:O)),*dH1=(float*)rb_buf(10,4*N*H),*dOut=(float*)rb_buf(11,4*N*O);
  int* dHs=(mode==1)?(int*)rb_buf(12,sizeof(int)*(size_t)W):(int*)1;   /* mode 2 (cont) has no head sizes */
  /* Persistent PINNED host staging when buffered (needed for cudaMemcpyAsync to actually overlap, same
     rationale as the MLP sibling's wantPin) -- hb_buf slots 4-7 (0-3 are the MLP sibling's; the cache is
     process-global but grow-only per slot, so distinct slots avoid any cross-trainer size churn even
     though sharing would also be safe -- one trainer per process). actRM is plain malloc either way (it's
     read synchronously by the CPU env-step, never cudaMemcpyAsync'd). */
  int wantPin = roll_buffers()>1;
  double *hSamp,*hA,*hB,*hRT;
  if(wantPin){ hSamp=(double*)hb_buf(4,8*(size_t)(W+2)*N); hA=(double*)hb_buf(5,8*N*D); hB=(double*)hb_buf(6,8*N*D); hRT=(double*)hb_buf(7,8*2*N); }
  else       { hSamp=(double*)malloc(8*(size_t)(W+2)*N);   hA=(double*)malloc(8*N*D);   hB=(double*)malloc(8*N*D);   hRT=(double*)malloc(8*2*N);   }
  double *actRM=(double*)malloc(8*(size_t)W*N);
  int ok=(N>0 && W>0 && h!=NULL && eh && dPar&&dObs&&dY&&dO&&dW1f&&db1f&&dW2f&&db2f&&dXf&&dPre&&dH1&&dOut&&dHs&&hSamp&&actRM&&hA&&hB&&hRT);
  if(ok){
    cublasSetStream(h,0);
    int B=256;
    /* weights already device-resident (dPar = policy handle); convert to f32 ONCE (no PCIe upload) */
    k_f64_to_f32<<<ceildiv((long)H*D,B),B>>>(dW1f,dPar+oW1,(long)H*D); k_f64_to_f32<<<ceildiv((long)H,B),B>>>(db1f,dPar+ob1,(long)H);
    k_f64_to_f32<<<ceildiv((long)O*H,B),B>>>(dW2f,dPar+oW2,(long)O*H); k_f64_to_f32<<<ceildiv((long)O,B),B>>>(db2f,dPar+ob2,(long)O);
    if(mode==1){ int* hHs=(int*)malloc(sizeof(int)*(size_t)W); for(int i=0;i<W;i++) hHs[i]=(int)hs[i];
      cudaMemcpy(dHs,hHs,sizeof(int)*(size_t)W,cudaMemcpyHostToDevice); free(hHs); }
    int threaded = (eh->step_range != NULL);
    int nAg = eh->numAgents>0 ? eh->numAgents : 1;
    int nbuf = roll_buffers();
    g_dObsTraj_valid = 0;                                           /* single-buffer path → set below if resident */
    int useBuffered = (nbuf>1 && threaded && (int)eh->N>=nbuf);
    if(useBuffered){
      /* CONCURRENT STREAM-BUFFERS + device-resident obs (ported from lean_cuda_plugin_rollout, same
         globals, same design — see bufpool_wide_t's comment above). */
      buf_init(nbuf);
      obstraj_buf((size_t)NT*D); g_dObsTraj_valid = (g_dObsTraj!=NULL);
      cudaDeviceSynchronize();                                      /* f32 weights ready before workers read them */
      g_bpw.eh=eh; g_bpw.obs0=obs0; g_bpw.dW1f=dW1f; g_bpw.db1f=db1f; g_bpw.dW2f=dW2f; g_bpw.db2f=db2f;
      g_bpw.dObs=dObs; g_bpw.dO=dO; g_bpw.dXf=dXf; g_bpw.dPre=dPre; g_bpw.dH1=dH1; g_bpw.dOut=dOut;
      g_bpw.hSamp=hSamp; g_bpw.hA=hA; g_bpw.hB=hB; g_bpw.hRT=hRT; g_bpw.actRM=actRM;
      g_bpw.obsCol=obsCol; g_bpw.actCol=actCol; g_bpw.logpCol=logpCol; g_bpw.valCol=valCol; g_bpw.rewCol=rewCol; g_bpw.termCol=termCol;
      g_bpw.N=(long)N; g_bpw.D=(long)D; g_bpw.T=(long)T; g_bpw.H=(int)H; g_bpw.O=(int)O; g_bpw.W=W;
      g_bpw.mode=(int)mode; g_bpw.bf=bf; g_bpw.nAg=nAg; g_bpw.rolloutRng=rolloutRng; g_bpw.dHs=dHs;
      int totalEnv=eh->N;
      for(int b=0;b<nbuf;b++){ int elo=(int)((long)b*totalEnv/nbuf), ehi=(int)((long)(b+1)*totalEnv/nbuf);
        g_bpw.envLo[b]=elo; g_bpw.envHi[b]=ehi; g_bpw.rowBase[b]=(int)((long)elo*nAg); g_bpw.rowN[b]=(int)((long)(ehi-elo)*nAg); }
      for(int b=nbuf;b<MAXBUF;b++) g_bpw.rowN[b]=0;                 /* idle slots: buf_worker_wide's nb<=0 early-return */
      if(bufw_wide_init()){
        pthread_barrier_wait(&g_bufw_wide_bar);                     /* release all MAXBUF workers */
        pthread_barrier_wait(&g_bufw_wide_bar);                      /* join */
      } else {
        pthread_t bth[MAXBUF];                                      /* pool init failed once — permanent fallback */
        for(int b=0;b<nbuf;b++) pthread_create(&bth[b],NULL,buf_worker_wide,(void*)(long)b);
        for(int b=0;b<nbuf;b++) pthread_join(bth[b],NULL);
      }
      const double* fin=((T%2)==1)?hA:hB;                           /* after T ping-pong steps */
      for(size_t i=0;i<N*D;i++) finalObs[i]=fin[i];
    } else {
    if(threaded && !g_rp_init){
      g_rp.nthreads=rp_threads(); g_rp.alive=1;
      pthread_barrier_init(&g_rp.bar,NULL,g_rp.nthreads+1);
      for(long t=0;t<g_rp.nthreads;t++) pthread_create(&g_rp_th[t],NULL,rp_worker,(void*)t);
      g_rp_init=1;
    }
    if(threaded){                                                   /* shape-invariant fields (per call) */
      g_rp.eh=eh; g_rp.obsCol=obsCol; g_rp.actCol=actCol; g_rp.logpCol=logpCol; g_rp.valCol=valCol;
      g_rp.rewCol=rewCol; g_rp.termCol=termCol; g_rp.N=(long)N; g_rp.D=(long)D; g_rp.T=(long)T;
      g_rp.nAgents=nAg; g_rp.envLo=0; g_rp.envHi=eh->N; g_rp.rowBase=0; g_rp.Nstride=(long)N; g_rp.W=W;
      g_rp.skipObs=0;
    }
    const double* cur=obs0; double* nxt=hA;                         /* ping-pong obs; obs0 read-only */
    for(size_t s=0;s<T;s++){
      cudaMemcpy(dObs,cur,8*N*D,cudaMemcpyHostToDevice);
      k_f64_to_f32<<<ceildiv((long)N*D,B),B>>>(dXf,dObs,(long)N*D);
      mlp_forward_dev(h,dW1f,db1f,dW2f,db2f,dXf,(int)N,(int)D,(int)H,(int)O,bf,dPre,dH1,dOut);
      k_f32_to_f64<<<ceildiv((long)N*O,B),B>>>(dY,dOut,(long)N*O);   /* logits stay on device */
      unsigned long long rs=(unsigned long long)(rolloutRng+(uint64_t)(s*N)*G);
      if(mode==1) k_sample_md<<<ceildiv((long)N,B),B>>>(dY,dO,(int)N,W,dHs,(int)O,rs);
      else        k_sample_cont<<<ceildiv((long)N,B),B>>>(dY,dO,(int)N,W,(int)O,rs);
      cudaDeviceSynchronize();
      cudaMemcpy(hSamp,dO,8*(size_t)(W+2)*N,cudaMemcpyDeviceToHost);  /* [act(W×N col); logp(N); val(N)] */
      for(size_t e=0;e<N;e++) for(int wc=0;wc<W;wc++) actRM[e*W+wc]=hSamp[(long)wc*N+e];  /* col→row for env-step */
      if(threaded){
        g_rp.s=s; g_rp.cur=cur; g_rp.nxt=nxt; g_rp.hSamp=hSamp; g_rp.actRM=actRM; g_rp.hRT=hRT;
        pthread_barrier_wait(&g_rp.bar);                            /* release workers: env-step + scatter */
        pthread_barrier_wait(&g_rp.bar);                            /* wait until all partitions done */
      } else {
        eh->step(eh->env, actRM, nxt, hRT, hRT+N);                  /* CPU env step: obs'→nxt, rew→hRT, term→hRT+N */
        for(size_t e=0;e<N;e++){ long row=(long)e*T+(long)s;
          for(size_t j=0;j<D;j++) obsCol[row*D+j]=cur[e*D+j];       /* obs BEFORE the step */
          for(int wc=0;wc<W;wc++) actCol[row*W+wc]=hSamp[(long)wc*N+e];
          logpCol[row]=hSamp[(long)W*N+e]; valCol[row]=hSamp[(long)(W+1)*N+e];
          rewCol[row]=hRT[e]; termCol[row]=hRT[N+e]; }
      }
      cur=nxt; nxt=(nxt==hA)?hB:hA;                                 /* advance (no aliasing) */
    }
    for(size_t i=0;i<N*D;i++) finalObs[i]=cur[i];                   /* persistent obs for the next update */
    }
  } else for(long i=0;i<cols;i++) out[i]=0.0;
  if(!wantPin){ free(hSamp); free(hA); free(hB); free(hRT); }     /* pinned buffers are the persistent cache — not freed here */
  free(actRM);
  lean_dec(obs0a); lean_dec(hsA);
  return lean_io_result_mk_ok(Oo);
}

/* the driver. Returns the env-major SoA columns [obs(N·T·D); act(N·T); val(N·T); logp(N·T); rew(N·T);
   term(N·T)] (size N·T·(D+5)), D=size². Deterministic-reset ⇒ timestep s uses rng + s·N·G. */

/* === R6: resident CNN encoder forward (the "CNN path") ===========================================
   GPU port of the CPU-BLAS CNN forward (ffi/pufferblas.c cnn_ppo_grad_batch_blas, lines 776-794):
   im2col → conv GEMM (Xcol·convWᵀ) → relu+convB → pixel→filter transpose → dense (Feat·W1ᵀ → relu →
   Hh·W2ᵀ) → logits. Params flat [convW(nF·Ckk)|convB(nF)|W1(hidden·flatDim)|b1(hidden)|W2(O·hidden)|
   b2(O)], Ckk=C·k·k, oH=(inH-k)/s+1, flatDim=nF·oH·oW. f32/bf16 (gemm32); tolerance vs the f64 CPU
   `cnnForward` (verify-cnn-forward-gpu). The reusable conv-encoder any CNN env's rollout needs. */
__global__ void k_im2col(const float* obs, float* xcol, int N, int C, int inH, int inW, int k, int s, int oH, int oW){
  int Ckk=C*k*k, oHoW=oH*oW, inSz=C*inH*inW; long R=(long)N*oHoW;
  long t=(long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=R*Ckk) return;
  int r=(int)(t/Ckk), idx=(int)(t%Ckk);
  int n=r/oHoW, p=r%oHoW, oy=p/oW, ox=p%oW;
  int c=idx/(k*k), rem=idx%(k*k), ky=rem/k, kx=rem%k;
  int src=(c*inH + (oy*s+ky))*inW + (ox*s+kx);
  xcol[t]=obs[(long)n*inSz + src];
}
/* pixel-major featPix[(n·oHoW+p)·nF+f] → filter-major Feat[n·flatDim + f·oHoW + p]. */
__global__ void k_pix2filt(float* feat, const float* featPix, int N, int nF, int oHoW){
  int flatDim=nF*oHoW; long total=(long)N*flatDim;
  long t=(long)blockIdx.x*blockDim.x+threadIdx.x; if(t>=total) return;
  int n=(int)(t/flatDim), rem=(int)(t%flatDim), f=rem/oHoW, p=rem%oHoW;
  feat[t]=featPix[((long)n*oHoW+p)*nF + f];
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_cnn_forward(lean_obj_arg pa, lean_obj_arg obsa,
    size_t N, size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t O, uint8_t bf16){
  int bf=(int)bf16;
  size_t Ckk=C*k*k, oH=(inH-k)/s+1, oW=(inW-k)/s+1, oHoW=oH*oW, flatDim=nF*oHoW, R=N*oHoW, inSz=C*inH*inW;
  size_t P=nF*Ckk+nF+hidden*flatDim+hidden+O*hidden+O;
  size_t ocW=0, ocB=nF*Ckk, oW1=nF*Ckk+nF, ob1=oW1+hidden*flatDim, oW2=ob1+hidden, ob2=oW2+O*hidden;
  const double* par=lean_float_array_cptr(pa); const double* obs=lean_float_array_cptr(obsa);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),N*O,N*O); double* out=lean_float_array_cptr(Oo);
  cublasHandle_t h=cu_handle();
  double *dPar=NULL,*dObs=NULL,*dLog=NULL;
  float *dCW=NULL,*dCB=NULL,*dW1=NULL,*db1=NULL,*dW2=NULL,*db2=NULL,*dObsF=NULL,*dXcol=NULL,*dFeatPix=NULL,*dFeat=NULL,*dHh=NULL,*dOut=NULL;
  int ok=(N>0 && h!=NULL
    && !cudaMalloc((void**)&dPar,8*P) && !cudaMalloc((void**)&dObs,8*N*inSz) && !cudaMalloc((void**)&dLog,8*N*O)
    && !cudaMalloc((void**)&dCW,4*nF*Ckk) && !cudaMalloc((void**)&dCB,4*nF) && !cudaMalloc((void**)&dW1,4*hidden*flatDim)
    && !cudaMalloc((void**)&db1,4*hidden) && !cudaMalloc((void**)&dW2,4*O*hidden) && !cudaMalloc((void**)&db2,4*O)
    && !cudaMalloc((void**)&dObsF,4*N*inSz) && !cudaMalloc((void**)&dXcol,4*R*Ckk) && !cudaMalloc((void**)&dFeatPix,4*R*nF)
    && !cudaMalloc((void**)&dFeat,4*N*flatDim) && !cudaMalloc((void**)&dHh,4*N*hidden) && !cudaMalloc((void**)&dOut,4*N*O));
  if(ok){
    cudaMemcpy(dPar,par,8*P,cudaMemcpyHostToDevice); cudaMemcpy(dObs,obs,8*N*inSz,cudaMemcpyHostToDevice);
    int B=256;
    k_f64_to_f32<<<ceildiv((long)nF*Ckk,B),B>>>(dCW,dPar+ocW,(long)nF*Ckk); k_f64_to_f32<<<ceildiv((long)nF,B),B>>>(dCB,dPar+ocB,(long)nF);
    k_f64_to_f32<<<ceildiv((long)hidden*flatDim,B),B>>>(dW1,dPar+oW1,(long)hidden*flatDim); k_f64_to_f32<<<ceildiv((long)hidden,B),B>>>(db1,dPar+ob1,(long)hidden);
    k_f64_to_f32<<<ceildiv((long)O*hidden,B),B>>>(dW2,dPar+oW2,(long)O*hidden); k_f64_to_f32<<<ceildiv((long)O,B),B>>>(db2,dPar+ob2,(long)O);
    k_f64_to_f32<<<ceildiv((long)N*inSz,B),B>>>(dObsF,dObs,(long)N*inSz);
    k_im2col<<<ceildiv((long)R*Ckk,B),B>>>(dObsF,dXcol,(int)N,(int)C,(int)inH,(int)inW,(int)k,(int)s,(int)oH,(int)oW);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)nF,(int)R,(int)Ckk, dCW,(int)Ckk, dXcol,(int)Ckk, dFeatPix,(int)nF, bf);  /* featPix[R×nF]=Xcol·convWᵀ */
    k_relu_bias<<<ceildiv((long)R*nF,B),B>>>(dFeatPix,dFeatPix,dCB,(int)R,(int)nF);                                 /* +convB, relu (in place) */
    k_pix2filt<<<ceildiv((long)N*flatDim,B),B>>>(dFeat,dFeatPix,(int)N,(int)nF,(int)oHoW);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)hidden,(int)N,(int)flatDim, dW1,(int)flatDim, dFeat,(int)flatDim, dHh,(int)hidden, bf);
    k_relu_bias<<<ceildiv((long)N*hidden,B),B>>>(dHh,dHh,db1,(int)N,(int)hidden);
    gemm32(h,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,(int)N,(int)hidden, dW2,(int)hidden, dHh,(int)hidden, dOut,(int)O, bf);
    k_add_bias<<<ceildiv((long)N*O,B),B>>>(dOut,dOut,db2,(int)N,(int)O);
    k_f32_to_f64<<<ceildiv((long)N*O,B),B>>>(dLog,dOut,(long)N*O);
    cudaDeviceSynchronize(); cudaMemcpy(out,dLog,8*N*O,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<N*O;i++) out[i]=0.0;
  if(dPar)cudaFree(dPar);if(dObs)cudaFree(dObs);if(dLog)cudaFree(dLog);if(dCW)cudaFree(dCW);if(dCB)cudaFree(dCB);
  if(dW1)cudaFree(dW1);if(db1)cudaFree(db1);if(dW2)cudaFree(dW2);if(db2)cudaFree(db2);if(dObsF)cudaFree(dObsF);
  if(dXcol)cudaFree(dXcol);if(dFeatPix)cudaFree(dFeatPix);if(dFeat)cudaFree(dFeat);if(dHh)cudaFree(dHh);if(dOut)cudaFree(dOut);
  lean_dec(pa); lean_dec(obsa); return Oo;
}

/* === R6 (breakout physics foundation): bit-exact device sinf/cosf ================================
   Port of Puffer.Numeric.SinCosF (glibc 2.43 sincosf): polynomial with exact IEEE-754 f64 constants +
   correctly-rounded FMA, rounded to f32 (r32). Puffer.Float.fma = round(a·b+c) = CUDA fma(double), the
   constants are the same bit patterns, so this is BIT-EXACT vs the Lean model for |y| ≤ 120 (breakout's
   fire ≈ π/3.25 and paddle angles ∈ [−π/4, π/4] are well inside). |y| ≥ 120 falls back to the device
   libm (not bit-guaranteed — breakout never reaches it). Precision crux for breakout + all SinCosF envs. */
__device__ __forceinline__ double d_r32(double x){ return (double)(float)x; }
__device__ __forceinline__ unsigned int d_abstop12(double y){ return (__float_as_uint((float)y) >> 20) & 0x7ffu; }
__device__ __forceinline__ double d_polySin(double xs, double x2){
  const double cS1=__longlong_as_double(0xbfc555545995a603ULL), cS2=__longlong_as_double(0x3f81107605230bc4ULL), cS3=__longlong_as_double(0xbf2994eb3774cf24ULL);
  double x3=xs*x2, s1v=fma(x2,cS3,cS2), x7=x3*x2, s=fma(x3,cS1,xs);
  return d_r32(fma(x7,s1v,s));
}
__device__ __forceinline__ double d_polyCos(double x2, int ent){
  const double e0c0=__longlong_as_double(0x3ff0000000000000ULL),e0c1=__longlong_as_double(0xbfdffffffd0c621cULL),e0c2=__longlong_as_double(0x3fa55553e1068f19ULL),e0c3=__longlong_as_double(0xbf56c087e89a359dULL),e0c4=__longlong_as_double(0x3ef99343027bf8c3ULL);
  const double e1c0=__longlong_as_double(0xbff0000000000000ULL),e1c1=__longlong_as_double(0x3fdffffffd0c621cULL),e1c2=__longlong_as_double(0xbfa55553e1068f19ULL),e1c3=__longlong_as_double(0x3f56c087e89a359dULL),e1c4=__longlong_as_double(0xbef99343027bf8c3ULL);
  double c0=ent?e1c0:e0c0, c1=ent?e1c1:e0c1, c2=ent?e1c2:e0c2, c3=ent?e1c3:e0c3, c4=ent?e1c4:e0c4;
  double x4=x2*x2, hi=fma(x2,c4,c3), c1v=fma(x2,c1,c0), x6=x4*x2, c=fma(x4,c2,c1v);
  return d_r32(fma(x6,hi,c));
}
__device__ double d_sincos_core(int isCos, double y){
  const double hpiInv=__longlong_as_double(0x41645f306dc9c883ULL), hpi=__longlong_as_double(0x3ff921fb54442d18ULL);
  unsigned int at12=d_abstop12(y);
  if(at12 < 1012u){
    if(at12 < 920u) return isCos ? 1.0 : y;
    double x2=y*y; return isCos ? d_polyCos(x2,0) : d_polySin(y,x2);
  } else if(at12 < 1071u){
    double r=y*hpiInv; long long t=(long long)r; long long n=(t + 0x800000LL) >> 24;   /* reduce_fast */
    double xr=fma(-(double)n, hpi, y);
    int q=(int)(((n % 4) + 4) % 4);
    int ent = (q>=2) ? 1 : 0;
    double sgn = (q==1 || q==2) ? -1.0 : 1.0;
    double xs=xr*sgn, x2=xr*xr;
    int branch = isCos ? (1 - (q%2)) : (q%2);
    return (branch==0) ? d_polySin(xs,x2) : d_polyCos(x2,ent);
  } else return d_r32(isCos ? cos(y) : sin(y));
}
__device__ __forceinline__ double d_sinf(double y){ return d_sincos_core(0,y); }
__device__ __forceinline__ double d_cosf(double y){ return d_sincos_core(1,y); }
__global__ void k_sincos(const double* ang, double* out, int N, int isCos){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return; out[n]=d_sincos_core(isCos, ang[n]);
}
extern "C" LEAN_EXPORT lean_obj_res lean_cuda_sincosf(lean_obj_arg anga, size_t N, uint8_t isCos){
  const double* ang=lean_float_array_cptr(anga);
  lean_object* Oo=lean_alloc_sarray(sizeof(double),N,N); double* out=lean_float_array_cptr(Oo);
  double *dA=NULL,*dO=NULL;
  if(N>0 && cudaMalloc((void**)&dA,8*N)==cudaSuccess && cudaMalloc((void**)&dO,8*N)==cudaSuccess){
    cudaMemcpy(dA,ang,8*N,cudaMemcpyHostToDevice);
    k_sincos<<<ceildiv((long)N,256),256>>>(dA,dO,(int)N,(int)isCos);
    cudaMemcpy(out,dO,8*N,cudaMemcpyDeviceToHost);
  } else for(size_t i=0;i<N;i++) out[i]=0.0;
  if(dA)cudaFree(dA); if(dO)cudaFree(dO);
  lean_dec(anga); return Oo;
}

/* ===== GPU-resident batched LSTM+PPO truncated-BPTT gradient ====================================
   Device twin of pufferblas.c's `lean_ffi_lstm_ppo_grad_batch_blas` (f64, cblas) and its f32 twin.
   Same algorithm, same math — every cblas_dgemm/sgemm moves to cublasDgemm/Sgemm and the gate
   forward/backward + PPO objective become elementwise CUDA kernels; obs + all per-timestep
   activations (II/FF/GG/OO/TC/HP/CP/HT/OUT/dG) stay DEVICE-RESIDENT across the T-step recurrence, so
   only the inputs H2D once and the P-length gradient (plus, on --log render frames, the logits) D2H.
   Reduction order differs from cblas (blocked cuBLAS GEMMs + tree colsums) ⇒ matches the scalar/Lean
   oracle to f64 TOLERANCE (~1e-11), not bit-exactly — the same trade the CPU-BLAS path already makes
   (verify-lstm-blas). Deterministic: cuBLAS GEMMs pick a fixed algorithm per shape and the kernels use
   no atomics / fixed-order block reductions ⇒ run-to-run identical.

   The BATCHING vs the CPU per-timestep loop: the input projection X·Wxᵀ, the output head HT·Woᵀ, the
   output-head backward dOut·Wo, and all four weight gradients (gWx/gWh/gWo + the two bias colsums) do
   NOT depend on the recurrence, so each is ONE big GEMM/colsum over the whole [T·B] batch instead of T
   small ones. Only the recurrent projection Hprev·Whᵀ (fwd) and dG·Wh (bwd) stay per-timestep — the
   sequential critical path. Same operation set as the oracle, just re-associated reductions (tolerance,
   not bit-exact — as documented above). Called from pufferblas.c (extern "C"); returns 1 on success
   (gOut[P] filled), 0 to signal "no usable device / unsupported shape" so the caller runs its CPU BLAS
   path. `outHostOrNull` (non-NULL only on --log render frames) receives the T·B·O logits for the shared
   dashboard-loss reducer. `useF32`: 0 = cublasDgemm f64 (the default, ~1e-11 vs the oracle); 1 = the
   cublasSgemm f32 tier (PUFFER_LSTM_F32, ~1e-6 vs the f64 path, verify-lstm-grad-f32). ================ */
__device__ __forceinline__ float  l_exp(float x){ return expf(x); }
__device__ __forceinline__ double l_exp(double x){ return exp(x); }
__device__ __forceinline__ float  l_tanh(float x){ return tanhf(x); }
__device__ __forceinline__ double l_tanh(double x){ return tanh(x); }
__device__ __forceinline__ float  l_log(float x){ return logf(x); }
__device__ __forceinline__ double l_log(double x){ return log(x); }
template<typename R> __device__ __forceinline__ R l_sig(R x){ return (R)1/((R)1+l_exp(-x)); }

/* reset the recurrent (h,c) state at step t for sequences whose PREVIOUS step terminated (t>0 only) */
template<typename R>
__global__ void k_lstm_reset(R* Hst, R* Cst, const double* term, int B, int H, int t){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H);
  if(term[(long)(t-1)*B+b]!=0.0){ Hst[idx]=(R)0; Cst[idx]=(R)0; }
}
/* forward gate cell: gates from Gin (precomputed X·Wxᵀ, slice t) + Grec (Hprev·Whᵀ) + bih; store the
   OLD (h,c) into HP/CP, the activations, and update the running (Hst,Cst). Mirrors pufferblas.c 927-935. */
template<typename R>
__global__ void k_lstm_gate_fwd(const R* Gin, const R* Grec, const R* bih,
    R* Hst, R* Cst, R* II, R* FF, R* GG, R* OO, R* TC, R* HT, R* HP, R* CP, int B, int H){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H), j=(int)(idx%H); int H4=4*H; long g0=(long)b*H4;
  R oldH=Hst[idx], oldC=Cst[idx]; HP[idx]=oldH; CP[idx]=oldC;
  R gi=Gin[g0+j]     +Grec[g0+j]     +bih[j];
  R gf=Gin[g0+H+j]   +Grec[g0+H+j]   +bih[H+j];
  R gg=Gin[g0+2*H+j] +Grec[g0+2*H+j] +bih[2*H+j];
  R go=Gin[g0+3*H+j] +Grec[g0+3*H+j] +bih[3*H+j];
  R iv=l_sig(gi), fv=l_sig(gf), gv=l_tanh(gg), ov=l_sig(go);
  R cv=fv*oldC+iv*gv; R tcv=l_tanh(cv); R hv=ov*tcv;
  II[idx]=iv; FF[idx]=fv; GG[idx]=gv; OO[idx]=ov; TC[idx]=tcv; HT[idx]=hv;
  Cst[idx]=cv; Hst[idx]=hv;
}
template<typename R>
__global__ void k_lstm_add_bias(R* Y, const R* b, long rows, int cols){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=rows*(long)cols) return;
  Y[idx]+=b[idx%cols];
}
/* per-row single-categorical PPO backward over all T·B rows at once (LSTM head: UNCLIPPED value loss).
   Mirrors pufferblas.c 943-954 verbatim (f64) / its f32 twin (expf/logf), no max-subtraction in the lse. */
template<typename R>
__global__ void k_lstm_ppo_dout(const R* OUT, const double* actA, const double* advA,
    const double* retA, const double* oldA, R* dOut, long TB, int A, int O,
    R vfCoef, R entCoef, R clipEps){
  long n=(long)blockIdx.x*blockDim.x+threadIdx.x; if(n>=TB) return;
  const R* out=OUT+n*O; R* dout=dOut+n*O;
  int a=(int)actA[n]; R adv=(R)advA[n], ret=(R)retA[n], oldLogp=(R)oldA[n];
  R sumexp=(R)0; for(int k=0;k<A;k++) sumexp+=l_exp(out[k]); R lse=l_log(sumexp);
  R pout=(R)0; R pk[64];
  for(int k=0;k<A;k++){ pk[k]=l_exp(out[k]-lse); pout+=pk[k]*out[k]; }
  R logpA=out[a]-lse; R ratio=l_exp(logpA-oldLogp); R lo=(R)1-clipEps, hi=(R)1+clipEps;
  R ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); R surr1=adv*ratio, surr2=adv*ratioC;
  R dPol; if(surr1<=surr2) dPol=adv*ratio; else { R cg=(lo<ratio&&ratio<hi)?(R)1:(R)0; dPol=adv*cg*ratio; }
  for(int k=0;k<A;k++){ R dp=dPol*(((k==a)?(R)1:(R)0)-pk[k]); R de=entCoef*pk[k]*(pout-out[k]); dout[k]=dp+de; }
  dout[A]=-vfCoef*(out[A]-ret);
}
/* backward gate cell at step t: dHt = dHfromOut[t] + dHnext (folded in); produces dG[t] (B×H4) and the
   cell-state carry dCprevG. Mirrors pufferblas.c 961-972. */
template<typename R>
__global__ void k_lstm_gate_bwd(const R* II, const R* FF, const R* GG, const R* OO, const R* TC,
    const R* CP, const R* dHfromOut, const R* dHnext, const R* dCnext, R* dCprevG, R* dG, int B, int H){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H), j=(int)(idx%H); int H4=4*H; long g0=(long)b*H4;
  R iv=II[idx], fv=FF[idx], gv=GG[idx], ov=OO[idx], tcv=TC[idx], cprev=CP[idx];
  R dh=dHfromOut[idx]+dHnext[idx];
  R dc=dCnext[idx]+dh*ov*((R)1-tcv*tcv);
  R do_=dh*tcv; R df=dc*cprev, di=dc*gv, dg_=dc*iv;
  dCprevG[idx]=dc*fv;
  dG[g0+j]     =di*iv*((R)1-iv);
  dG[g0+H+j]   =df*fv*((R)1-fv);
  dG[g0+2*H+j] =dg_*((R)1-gv*gv);
  dG[g0+3*H+j] =do_*ov*((R)1-ov);
}
/* propagate the (h,c) grad to the previous step unless a terminal broke the recurrence. pufferblas 978-984. */
template<typename R>
__global__ void k_lstm_carry(R* dHnext, R* dCnext, const R* dHprevG, const R* dCprevG,
    const double* term, int B, int H, int t){
  long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)B*H) return;
  int b=(int)(idx/H);
  int flow=(t>0 && term[(long)(t-1)*B+b]==0.0);
  if(flow){ dHnext[idx]=dHprevG[idx]; dCnext[idx]=dCprevG[idx]; }
  else    { dHnext[idx]=(R)0;         dCnext[idx]=(R)0; }
}
/* deterministic bias gradient: one block per column tree-reduces all `rows` rows in a fixed order
   (no atomics ⇒ run-to-run identical; order differs from the serial CPU sum ⇒ f64 tolerance). Single
   shot over the whole [T·B] batch, so it OVERWRITES (the CPU accumulates per-timestep to the same total). */
template<typename R>
__global__ void k_lstm_colsum(R* gb, const R* M, long rows, int J){
  __shared__ R sh[256];
  int j=blockIdx.x; if(j>=J) return; int tid=threadIdx.x, nb=blockDim.x;
  R p=(R)0; for(long r=tid; r<rows; r+=nb) p+=M[r*(long)J+j]; sh[tid]=p; __syncthreads();
  for(int s=nb/2;s>0;s>>=1){ if(tid<s) sh[tid]+=sh[tid+s]; __syncthreads(); }
  if(tid==0) gb[j]=sh[0];
}

/* row-major C[M×N] = opA(A)·opB(B) + beta·C via the col-major swap (pass B then A), f64 / f32 overloads
   (same identity as the MinGRU sgemm_rm). alpha is always 1. */
static cublasStatus_t lstm_gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int M, int N, int K, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc){
  double al=1.0; return cublasDgemm(h, opB, opA, N, M, K, &al, B, ldb, A, lda, &beta, C, ldc);
}
static cublasStatus_t lstm_gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
    int M, int N, int K, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc){
  float al=1.0f; return cublasSgemm(h, opB, opA, N, M, K, &al, B, ldb, A, lda, &beta, C, ldc);
}
/* host→device upload with an f64→R cast for the f32 tier (direct memcpy when R==double). */
static void lstm_up(double* d, const double* s, size_t n){ cudaMemcpy(d,s,n*sizeof(double),cudaMemcpyHostToDevice); }
static void lstm_up(float*  d, const double* s, size_t n){ float* hbuf=(float*)malloc(n*sizeof(float));
  for(size_t i=0;i<n;i++) hbuf[i]=(float)s[i]; cudaMemcpy(d,hbuf,n*sizeof(float),cudaMemcpyHostToDevice); free(hbuf); }
/* device→host download widening R→f64 (direct memcpy when R==double). */
static void lstm_down(double* dst, const double* dsrc, size_t n){ cudaMemcpy(dst,dsrc,n*sizeof(double),cudaMemcpyDeviceToHost); }
static void lstm_down(double* dst, const float* dsrc, size_t n){ float* hbuf=(float*)malloc(n*sizeof(float));
  cudaMemcpy(hbuf,dsrc,n*sizeof(float),cudaMemcpyDeviceToHost); for(size_t i=0;i<n;i++) dst[i]=(double)hbuf[i]; free(hbuf); }

/* PERSISTENT device-buffer cache (shapes are constant across a training run ⇒ allocate once, reuse; the
   f32 tier reuses the same slots since float ≤ double bytes). Grown, never shrunk; freed at process exit. */
static void* g_lstm_buf[40]; static size_t g_lstm_sz[40];
static void* lstm_slot(int i, size_t bytes){
  if(g_lstm_sz[i]<bytes){ if(g_lstm_buf[i]) cudaFree(g_lstm_buf[i]);
    if(cudaMalloc(&g_lstm_buf[i],bytes)!=cudaSuccess){ g_lstm_buf[i]=NULL; g_lstm_sz[i]=0; return NULL; }
    g_lstm_sz[i]=bytes; }
  return g_lstm_buf[i];
}

template<typename R>
static int lstm_bptt_dev_R(const double* pp, const double* obsBd, const double* actAd,
    const double* advAd, const double* retAd, const double* oldAd, const double* termAd,
    const double* h0sd, const double* c0sd, size_t B, size_t T, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double* gOut, double* outHostOrNull){
  cublasHandle_t hbl=cu_handle();
  if(!hbl || B==0 || T==0 || H==0 || A>64) return 0;
  size_t O=A+1, H4=4*H;
  long TB=(long)T*(long)B; size_t BH=B*H, BH4=B*H4, sR=sizeof(R);
  #define LS(i,n) (R*)lstm_slot((i),(size_t)(n)*sR)
  R *dWx=LS(0,H4*D), *dWh=LS(1,H4*H), *dbih=LS(2,H4), *dWo=LS(3,O*H), *dbo=LS(4,O);
  R *dObs=LS(5,(size_t)TB*D), *dGin=LS(6,(size_t)TB*H4);
  R *dII=LS(7,(size_t)TB*H), *dFF=LS(8,(size_t)TB*H), *dGG=LS(9,(size_t)TB*H), *dOO=LS(10,(size_t)TB*H);
  R *dTC=LS(11,(size_t)TB*H), *dHP=LS(12,(size_t)TB*H), *dCP=LS(13,(size_t)TB*H), *dHT=LS(14,(size_t)TB*H);
  R *dOUT=LS(15,(size_t)TB*O), *dOutg=LS(16,(size_t)TB*O), *dHfromOut=LS(17,(size_t)TB*H), *dG=LS(18,(size_t)TB*H4);
  R *dGrec=LS(19,BH4), *dHst=LS(20,BH), *dCst=LS(21,BH), *dHnext=LS(22,BH), *dCnext=LS(23,BH), *dHprevG=LS(24,BH), *dCprevG=LS(25,BH);
  R *gWx=LS(26,H4*D), *gWh=LS(27,H4*H), *gbih=LS(28,H4), *gWo=LS(29,O*H), *gbo=LS(30,O);
  double *dAct=(double*)lstm_slot(31,(size_t)TB*8), *dAdv=(double*)lstm_slot(32,(size_t)TB*8);
  double *dRet=(double*)lstm_slot(33,(size_t)TB*8), *dOld=(double*)lstm_slot(34,(size_t)TB*8), *dTrm=(double*)lstm_slot(35,(size_t)TB*8);
  #undef LS
  if(!(dWx&&dWh&&dbih&&dWo&&dbo&&dObs&&dGin&&dII&&dFF&&dGG&&dOO&&dTC&&dHP&&dCP&&dHT&&dOUT&&dOutg&&dHfromOut&&dG
     &&dGrec&&dHst&&dCst&&dHnext&&dCnext&&dHprevG&&dCprevG&&gWx&&gWh&&gbih&&gWo&&gbo&&dAct&&dAdv&&dRet&&dOld&&dTrm))
    return 0;
  /* H2D inputs (once; the f32 tier casts on the host during the copy) */
  { const double* pWx=pp; const double* pWh=pWx+H4*D; const double* pbih=pWh+H4*H;
    const double* pWo=pbih+H4; const double* pbo=pWo+O*H;
    lstm_up(dWx,pWx,H4*D); lstm_up(dWh,pWh,H4*H); lstm_up(dbih,pbih,H4); lstm_up(dWo,pWo,O*H); lstm_up(dbo,pbo,O);
    lstm_up(dObs,obsBd,(size_t)TB*D); lstm_up(dHst,h0sd,BH); lstm_up(dCst,c0sd,BH); }
  cudaMemcpy(dAct,actAd,(size_t)TB*8,cudaMemcpyHostToDevice); cudaMemcpy(dAdv,advAd,(size_t)TB*8,cudaMemcpyHostToDevice);
  cudaMemcpy(dRet,retAd,(size_t)TB*8,cudaMemcpyHostToDevice); cudaMemcpy(dOld,oldAd,(size_t)TB*8,cudaMemcpyHostToDevice);
  cudaMemcpy(dTrm,termAd,(size_t)TB*8,cudaMemcpyHostToDevice);
  int Bk=256;
  #define GD(x) ceildiv((long)(x),Bk)
  /* ---- FORWARD ---- input projection over the WHOLE batch, then the sequential recurrent scan ---- */
  lstm_gemm(hbl,CUBLAS_OP_N,CUBLAS_OP_T,(int)TB,(int)H4,(int)D, dObs,(int)D, dWx,(int)D, (R)0, dGin,(int)H4);
  for(size_t t=0;t<T;t++){
    if(t>0) k_lstm_reset<R><<<GD(BH),Bk>>>(dHst,dCst,dTrm,(int)B,(int)H,(int)t);
    lstm_gemm(hbl,CUBLAS_OP_N,CUBLAS_OP_T,(int)B,(int)H4,(int)H, dHst,(int)H, dWh,(int)H, (R)0, dGrec,(int)H4);
    size_t sBH=(size_t)t*BH;
    k_lstm_gate_fwd<R><<<GD(BH),Bk>>>(dGin+(size_t)t*B*H4, dGrec, dbih, dHst, dCst,
      dII+sBH,dFF+sBH,dGG+sBH,dOO+sBH,dTC+sBH,dHT+sBH,dHP+sBH,dCP+sBH,(int)B,(int)H);
  }
  lstm_gemm(hbl,CUBLAS_OP_N,CUBLAS_OP_T,(int)TB,(int)O,(int)H, dHT,(int)H, dWo,(int)H, (R)0, dOUT,(int)O);
  k_lstm_add_bias<R><<<GD((long)TB*O),Bk>>>(dOUT,dbo,TB,(int)O);
  /* ---- BACKWARD ---- PPO objective + non-recurrent grads batched, the (h,c) carry sequential ---- */
  k_lstm_ppo_dout<R><<<GD(TB),Bk>>>(dOUT,dAct,dAdv,dRet,dOld,dOutg,TB,(int)A,(int)O,(R)vfCoef,(R)entCoef,(R)clipEps);
  lstm_gemm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)O,(int)H,(int)TB, dOutg,(int)O, dHT,(int)H, (R)0, gWo,(int)H);
  k_lstm_colsum<R><<<(int)O,256>>>(gbo,dOutg,TB,(int)O);
  lstm_gemm(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)TB,(int)H,(int)O, dOutg,(int)O, dWo,(int)H, (R)0, dHfromOut,(int)H);
  cudaMemset(dHnext,0,BH*sR); cudaMemset(dCnext,0,BH*sR);
  for(size_t tt=T; tt-->0; ){
    size_t t=tt, sBH=(size_t)t*BH;
    k_lstm_gate_bwd<R><<<GD(BH),Bk>>>(dII+sBH,dFF+sBH,dGG+sBH,dOO+sBH,dTC+sBH,dCP+sBH,
      dHfromOut+sBH, dHnext,dCnext, dCprevG, dG+(size_t)t*BH4, (int)B,(int)H);
    lstm_gemm(hbl,CUBLAS_OP_N,CUBLAS_OP_N,(int)B,(int)H,(int)H4, dG+(size_t)t*BH4,(int)H4, dWh,(int)H, (R)0, dHprevG,(int)H);
    k_lstm_carry<R><<<GD(BH),Bk>>>(dHnext,dCnext, dHprevG,dCprevG, dTrm,(int)B,(int)H,(int)t);
  }
  lstm_gemm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H4,(int)D,(int)TB, dG,(int)H4, dObs,(int)D, (R)0, gWx,(int)D);
  lstm_gemm(hbl,CUBLAS_OP_T,CUBLAS_OP_N,(int)H4,(int)H,(int)TB, dG,(int)H4, dHP,(int)H, (R)0, gWh,(int)H);
  k_lstm_colsum<R><<<(int)H4,256>>>(gbih,dG,TB,(int)H4);
  #undef GD
  cudaDeviceSynchronize();
  if(cudaGetLastError()!=cudaSuccess) return 0;   /* kernel/launch fault ⇒ let the caller run the CPU path */
  /* D2H the gradient in the flat [gWx|gWh|gbih|gWo|gbo] layout (widened to f64 for the f32 tier). */
  double* g=gOut;
  lstm_down(g,gWx,H4*D); g+=H4*D; lstm_down(g,gWh,H4*H); g+=H4*H; lstm_down(g,gbih,H4); g+=H4;
  lstm_down(g,gWo,O*H); g+=O*H; lstm_down(g,gbo,O);
  if(outHostOrNull) lstm_down(outHostOrNull,dOUT,(size_t)TB*O);
  return 1;
}

/* C entry called from pufferblas.c: dispatch the f64 default vs the f32 tier. */
extern "C" int cuda_lstm_ppo_grad_batch(
    const double* pp, const double* obsB, const double* actA, const double* advA,
    const double* retA, const double* oldA, const double* termA,
    const double* h0s, const double* c0s, size_t B, size_t T, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double* gOut, double* outHostOrNull, int useF32){
  if(useF32) return lstm_bptt_dev_R<float >(pp,obsB,actA,advA,retA,oldA,termA,h0s,c0s,B,T,H,D,A,vfCoef,entCoef,clipEps,gOut,outHostOrNull);
  return            lstm_bptt_dev_R<double>(pp,obsB,actA,advA,retA,oldA,termA,h0s,c0s,B,T,H,D,A,vfCoef,entCoef,clipEps,gOut,outHostOrNull);
}

