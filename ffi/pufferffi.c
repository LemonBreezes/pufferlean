/* Native FFI kernels for the puffer trainer. Each has a Lean `@[extern]` twin in
   Puffer/Float/FFI.lean, validated bit-for-bit against the verified Lean oracle
   (`dotF`) and benchmarked. Read-only on their FloatArray args (owned convention:
   we `lean_dec` them before returning). */
#include <lean/lean.h>
#include <stddef.h>
#include <string.h>

/* Slice a FloatArray: out = arr[off : off+len] as a single memcpy — the C-speed replacement for the
   interpreted `mk ((Array.range len).map (fun i => arr[off+i]!))` column extraction in the resident
   rollout glue (O(len) boxed Array walk → one contiguous copy). Owned convention (`lean_dec arr`);
   calling it several times on the SAME array is fine — Lean inserts an `inc` per extra owned use. */
LEAN_EXPORT lean_obj_res lean_ffi_slice(lean_obj_arg arrA, size_t off, size_t len){
  const double* arr = lean_float_array_cptr(arrA);
  lean_object* outO = lean_alloc_sarray(sizeof(double), len, len);
  double* out = lean_float_array_cptr(outO);
  memcpy(out, arr + off, len * sizeof(double));
  lean_dec(arrA);
  return outO;
}

/* Smoke test: unboxed double -> double. */
LEAN_EXPORT double lean_ffi_test(double x) {
    return x * 2.0 + 1.0;
}

/* Right-fold dot product, matching Lean's `dotF`:
   x0*w0 + (x1*w1 + (... + (x_{n-1}*w_{n-1} + 0))) — accumulate from the last
   element backward, so the IEEE-754 rounding order is identical to `dotF`. */
LEAN_EXPORT double lean_ffi_dot(lean_obj_arg x, lean_obj_arg w) {
    size_t nx = lean_sarray_size(x);
    size_t nw = lean_sarray_size(w);
    size_t n = nx < nw ? nx : nw;
    const double* xp = lean_float_array_cptr(x);
    const double* wp = lean_float_array_cptr(w);
    double acc = 0.0;
    for (size_t i = n; i-- > 0; ) {
        acc = xp[i] * wp[i] + acc;
    }
    lean_dec(x);
    lean_dec(w);
    return acc;
}

/* ---- MLP + PPO gradient (the training hot path) ------------------------------
   Params are one flat FloatArray, layout: W1[H*D], b1[H], W2[O*H], b2[O], O=A+1.
   Forward: z1=b1+W1·obs, h=relu(z1), out=b2+W2·h (dots right-folded to match dotF).
   Objective (identical to Puffer.RL.NNTrain.mlpGradPPO):
     lse = log Σ_{k<A} exp(out[k]);  logpA = out[a]-lse;  ratio = exp(logpA-oldLogp)
     polObj = min(adv·ratio, adv·clamp(ratio,1-ε,1+ε));  ent = lse - Σ_{k<A} pk·out[k]
     obj = polObj − vf·½·(out[A]-ret)²  + ent·ent_coef
   Backward (analytic; matches the AD tape's math): see dout below. -------------- */
#include <math.h>
#include <stdlib.h>

/* Single-transition objective PRIMAL (for finite-difference gradient checks). */
LEAN_EXPORT double lean_ffi_mlp_ppo_obj1(
    lean_obj_arg params, lean_obj_arg obsA,
    size_t H, size_t D, size_t A, size_t a,
    double adv, double ret, double oldLogp, double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1;
  const double* pp = lean_float_array_cptr(params);
  const double* W1 = pp; const double* b1 = pp + H*D;
  const double* W2 = pp + H*D + H; const double* b2 = pp + H*D + H + O*H;
  const double* obs = lean_float_array_cptr(obsA);
  double* h = (double*)malloc(sizeof(double)*H);
  double* out = (double*)malloc(sizeof(double)*O);
  for (size_t j=0;j<H;j++){ const double* w=W1+j*D; double acc=0.0; for (size_t k=D;k-->0;) acc=w[k]*obs[k]+acc; double z=b1[j]+acc; h[j]=z>0.0?z:0.0; }
  for (size_t k=0;k<O;k++){ const double* w=W2+k*H; double acc=0.0; for (size_t j=H;j-->0;) acc=w[j]*h[j]+acc; out[k]=b2[k]+acc; }
  double sumexp=0.0; for (size_t k=0;k<A;k++) sumexp+=exp(out[k]);
  double lse=log(sumexp);
  double pout=0.0; for (size_t k=0;k<A;k++){ double pk=exp(out[k]-lse); pout+=pk*out[k]; }
  double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp);
  double lo=1.0-clipEps, hi=1.0+clipEps;
  double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio);
  double surr1=adv*ratio, surr2=adv*ratioC;
  double polObj = surr1<=surr2?surr1:surr2;
  double diff=out[A]-ret; double vloss=diff*diff;
  double ent=lse-pout;
  double obj = (polObj - vfCoef*0.5*vloss) + entCoef*ent;
  free(h); free(out); lean_dec(params); lean_dec(obsA);
  return obj;
}

/* Batched gradient: sums dObj/dparams over N transitions into a FloatArray[P].
   obsB is N*D (row-major), acts/advs/rets/oldlps are length-N FloatArrays. */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_ppo_grad_batch(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg acts,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps,
    size_t N, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps) {
  size_t O = A + 1;
  size_t P = H*D + H + O*H + O;
  const double* pp = lean_float_array_cptr(params);
  const double* W1 = pp; const double* b1 = pp + H*D;
  const double* W2 = pp + H*D + H; const double* b2 = pp + H*D + H + O*H;
  const double* obsAll = lean_float_array_cptr(obsB);
  const double* actA = lean_float_array_cptr(acts);
  const double* advA = lean_float_array_cptr(advs);
  const double* retA = lean_float_array_cptr(rets);
  const double* oldA = lean_float_array_cptr(oldlps);
  lean_object* grad = lean_alloc_sarray(sizeof(double), P, P);
  double* g = lean_float_array_cptr(grad);
  for (size_t t=0;t<P;t++) g[t]=0.0;
  double* gW1=g; double* gb1=g+H*D; double* gW2=g+H*D+H; double* gb2=g+H*D+H+O*H;
  double* h=(double*)malloc(sizeof(double)*H); double* z1=(double*)malloc(sizeof(double)*H);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O);
  double* dh=(double*)malloc(sizeof(double)*H); double* pk=(double*)malloc(sizeof(double)*A);
  for (size_t i=0;i<N;i++) {
    const double* obs = obsAll + i*D;
    size_t a=(size_t)actA[i]; double adv=advA[i], ret=retA[i], oldLogp=oldA[i];
    for (size_t j=0;j<H;j++){ const double* w=W1+j*D; double acc=0.0; for (size_t k=D;k-->0;) acc=w[k]*obs[k]+acc; double z=b1[j]+acc; z1[j]=z; h[j]=z>0.0?z:0.0; }
    for (size_t k=0;k<O;k++){ const double* w=W2+k*H; double acc=0.0; for (size_t j=H;j-->0;) acc=w[j]*h[j]+acc; out[k]=b2[k]+acc; }
    double sumexp=0.0; for (size_t k=0;k<A;k++) sumexp+=exp(out[k]);
    double lse=log(sumexp);
    double pout=0.0; for (size_t k=0;k<A;k++){ pk[k]=exp(out[k]-lse); pout+=pk[k]*out[k]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp);
    double lo=1.0-clipEps, hi=1.0+clipEps;
    double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio);
    double surr1=adv*ratio, surr2=adv*ratioC;
    double dPol_dlogpA;
    if (surr1<=surr2) dPol_dlogpA=adv*ratio;
    else { double cg=(lo<ratio && ratio<hi)?1.0:0.0; dPol_dlogpA=adv*cg*ratio; }
    for (size_t k=0;k<A;k++){ double dpol=dPol_dlogpA*(((k==a)?1.0:0.0)-pk[k]); double dent=entCoef*pk[k]*(pout-out[k]); dout[k]=dpol+dent; }
    dout[A] = -vfCoef*(out[A]-ret);
    for (size_t j=0;j<H;j++) dh[j]=0.0;
    for (size_t k=0;k<O;k++){ double dk=dout[k]; double* gw=gW2+k*H; const double* w=W2+k*H; for (size_t j=0;j<H;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[k]+=dk; }
    for (size_t j=0;j<H;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*D; for (size_t d=0;d<D;d++) gw[d]+=dz*obs[d]; gb1[j]+=dz; }
  }
  free(h);free(z1);free(out);free(dout);free(dh);free(pk);
  lean_dec(params);lean_dec(obsB);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);
  return grad;
}

/* Batched gradient WITH PufferLib value-loss CLIPPING (torch_pufferl.py):
     v_clipped = V_old + clamp(V_new - V_old, -vfClip, vfClip)
     v_loss    = 0.5 * max((V_new - R)^2, (v_clipped - R)^2)
   so dObj/dV_new = -vfCoef*(V_new-R)  UNLESS the clipped branch is the max AND the clamp is
   saturated (|V_new - V_old| >= vfClip), where it is 0. Extra input `oldvals[N]` = V at
   collection (PufferLib's mb_values); `vfClip <= 0` falls back to the unclipped gradient.
   Identical to lean_ffi_mlp_ppo_grad_batch otherwise. */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_ppo_grad_batch_vclip(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg acts,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg oldvals,
    size_t N, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double vfClip) {
  size_t O = A + 1;
  size_t P = H*D + H + O*H + O;
  const double* pp = lean_float_array_cptr(params);
  const double* W1 = pp; const double* b1 = pp + H*D;
  const double* W2 = pp + H*D + H; const double* b2 = pp + H*D + H + O*H;
  const double* obsAll = lean_float_array_cptr(obsB);
  const double* actA = lean_float_array_cptr(acts);
  const double* advA = lean_float_array_cptr(advs);
  const double* retA = lean_float_array_cptr(rets);
  const double* oldA = lean_float_array_cptr(oldlps);
  const double* ovA = lean_float_array_cptr(oldvals);
  /* output = gradient[P] ++ new_logp[N] ++ new_value[N] (the last 2N let the PER loop iterate the
     ratio/value buffers from this forward, avoiding a separate Lean forward). */
  size_t OUT = P + 2*N;
  lean_object* grad = lean_alloc_sarray(sizeof(double), OUT, OUT);
  double* g = lean_float_array_cptr(grad);
  for (size_t t=0;t<OUT;t++) g[t]=0.0;
  double* gW1=g; double* gb1=g+H*D; double* gW2=g+H*D+H; double* gb2=g+H*D+H+O*H;
  double* h=(double*)malloc(sizeof(double)*H); double* z1=(double*)malloc(sizeof(double)*H);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O);
  double* dh=(double*)malloc(sizeof(double)*H); double* pk=(double*)malloc(sizeof(double)*A);
  for (size_t i=0;i<N;i++) {
    const double* obs = obsAll + i*D;
    size_t a=(size_t)actA[i]; double adv=advA[i], ret=retA[i], oldLogp=oldA[i], vold=ovA[i];
    for (size_t j=0;j<H;j++){ const double* w=W1+j*D; double acc=0.0; for (size_t k=D;k-->0;) acc=w[k]*obs[k]+acc; double z=b1[j]+acc; z1[j]=z; h[j]=z>0.0?z:0.0; }
    for (size_t k=0;k<O;k++){ const double* w=W2+k*H; double acc=0.0; for (size_t j=H;j-->0;) acc=w[j]*h[j]+acc; out[k]=b2[k]+acc; }
    double sumexp=0.0; for (size_t k=0;k<A;k++) sumexp+=exp(out[k]);
    double lse=log(sumexp);
    double pout=0.0; for (size_t k=0;k<A;k++){ pk[k]=exp(out[k]-lse); pout+=pk[k]*out[k]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp);
    double lo=1.0-clipEps, hi=1.0+clipEps;
    double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio);
    double surr1=adv*ratio, surr2=adv*ratioC;
    double dPol_dlogpA;
    if (surr1<=surr2) dPol_dlogpA=adv*ratio;
    else { double cg=(lo<ratio && ratio<hi)?1.0:0.0; dPol_dlogpA=adv*cg*ratio; }
    for (size_t k=0;k<A;k++){ double dpol=dPol_dlogpA*(((k==a)?1.0:0.0)-pk[k]); double dent=entCoef*pk[k]*(pout-out[k]); dout[k]=dpol+dent; }
    /* clipped value-loss gradient */
    double vnew=out[A];
    double dvloss_dv;
    if (vfClip > 0.0) {
      double d=vnew-vold;
      double vclip=vold + (d<-vfClip?-vfClip:(d>vfClip?vfClip:d));
      double du=(vnew-ret)*(vnew-ret), dc=(vclip-ret)*(vclip-ret);
      if (du>=dc) dvloss_dv=(vnew-ret);
      else if (d>-vfClip && d<vfClip) dvloss_dv=(vnew-ret);   /* clamp inactive => vclip=vnew */
      else dvloss_dv=0.0;                                     /* clamp saturated => no grad */
    } else dvloss_dv=(vnew-ret);
    dout[A] = -vfCoef*dvloss_dv;
    g[P+i]=logpA; g[P+N+i]=vnew;                 /* new_logp (taken action) + new_value */
    for (size_t j=0;j<H;j++) dh[j]=0.0;
    for (size_t k=0;k<O;k++){ double dk=dout[k]; double* gw=gW2+k*H; const double* w=W2+k*H; for (size_t j=0;j<H;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[k]+=dk; }
    for (size_t j=0;j<H;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*D; for (size_t d=0;d<D;d++) gw[d]+=dz*obs[d]; gb1[j]+=dz; }
  }
  free(h);free(z1);free(out);free(dout);free(dh);free(pk);
  lean_dec(params);lean_dec(obsB);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(oldvals);
  return grad;
}

/* ---- MLP forward (rollout hot path) ------------------------------------------
   out = b2 + W2·relu(b1 + W1·obs); dots right-folded to match forwardAll/dotF.
   Returns the O = numActions+1 outputs (logits then value). */
LEAN_EXPORT lean_obj_res lean_ffi_mlp_forward(
    lean_obj_arg params, lean_obj_arg obsA, size_t H, size_t D, size_t O) {
  const double* pp = lean_float_array_cptr(params);
  const double* W1=pp; const double* b1=pp+H*D; const double* W2=pp+H*D+H; const double* b2=pp+H*D+H+O*H;
  const double* obs = lean_float_array_cptr(obsA);
  double* h = (double*)malloc(sizeof(double)*H);
  lean_object* outO = lean_alloc_sarray(sizeof(double), O, O);
  double* out = lean_float_array_cptr(outO);
  for (size_t j=0;j<H;j++){ const double* w=W1+j*D; double acc=0.0; for (size_t k=D;k-->0;) acc=w[k]*obs[k]+acc; double z=b1[j]+acc; h[j]=z>0.0?z:0.0; }
  for (size_t k=0;k<O;k++){ const double* w=W2+k*H; double acc=0.0; for (size_t j=H;j-->0;) acc=w[j]*h[j]+acc; out[k]=b2[k]+acc; }
  free(h); lean_dec(params); lean_dec(obsA);
  return outO;
}

/* ---- Gaussian (continuous) head PPO gradient ---------------------------------
   dout width O = 2A+1: mean[i]=out[i], logstd_raw[i]=out[A+i] (clamped to [-5,2]),
   value=out[2A]. logp = Σ_i (-0.5 z_i^2 - logstd_i - halfLog2pi), z_i=(a_i-mean_i)*exp(-logstd_i).
   ent = Σ_i (logstd_i + halfLog2pieE). Objective as in ContVecTrain.mlpGradPPOCont;
   backward is analytic (see below). */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static const double GC_LO = -5.0, GC_HI = 2.0;

static inline void gauss_forward(const double* pp, const double* obs, size_t H, size_t D, size_t O,
                                 double* z1, double* h, double* out) {
  const double* W1=pp; const double* b1=pp+H*D; const double* W2=pp+H*D+H; const double* b2=pp+H*D+H+O*H;
  for (size_t j=0;j<H;j++){ const double* w=W1+j*D; double acc=0.0; for (size_t k=D;k-->0;) acc=w[k]*obs[k]+acc; double z=b1[j]+acc; z1[j]=z; h[j]=z>0.0?z:0.0; }
  for (size_t k=0;k<O;k++){ const double* w=W2+k*H; double acc=0.0; for (size_t j=H;j-->0;) acc=w[j]*h[j]+acc; out[k]=b2[k]+acc; }
}

LEAN_EXPORT double lean_ffi_gauss_ppo_obj1(
    lean_obj_arg params, lean_obj_arg obsA, lean_obj_arg actA,
    size_t H, size_t D, size_t A,
    double adv, double ret, double oldLogp, double vfCoef, double entCoef, double clipEps) {
  size_t O = 2*A + 1;
  const double* pp = lean_float_array_cptr(params);
  const double* obs = lean_float_array_cptr(obsA);
  const double* act = lean_float_array_cptr(actA);
  double* z1=(double*)malloc(sizeof(double)*H); double* h=(double*)malloc(sizeof(double)*H); double* out=(double*)malloc(sizeof(double)*O);
  gauss_forward(pp,obs,H,D,O,z1,h,out);
  double halfLog2pi=0.5*log(2.0*M_PI), halfLog2pieE=0.5*(1.0+log(2.0*M_PI));
  double logp=0.0, ent=0.0;
  for (size_t i=0;i<A;i++){
    double ls=out[A+i]; ls = ls<GC_LO?GC_LO:(ls>GC_HI?GC_HI:ls);
    double invStd=exp(-ls); double z=(act[i]-out[i])*invStd;
    logp += -0.5*z*z - ls - halfLog2pi;
    ent  += ls + halfLog2pieE;
  }
  double ratio=exp(logp-oldLogp);
  double lo=1.0-clipEps, hi=1.0+clipEps;
  double ratioC = ratio<lo?lo:(ratio>hi?hi:ratio);
  double surr1=adv*ratio, surr2=adv*ratioC; double polObj=surr1<=surr2?surr1:surr2;
  double diff=out[2*A]-ret; double vloss=diff*diff;
  double obj=(polObj - vfCoef*0.5*vloss) + entCoef*ent;
  free(z1);free(h);free(out); lean_dec(params);lean_dec(obsA);lean_dec(actA);
  return obj;
}

LEAN_EXPORT lean_obj_res lean_ffi_gauss_ppo_grad_batch(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg actsB,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps,
    size_t N, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps) {
  size_t O=2*A+1; size_t P=H*D+H+O*H+O;
  const double* pp=lean_float_array_cptr(params);
  const double* W2=pp+H*D+H;
  const double* obsAll=lean_float_array_cptr(obsB); const double* actAll=lean_float_array_cptr(actsB);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets); const double* oldA=lean_float_array_cptr(oldlps);
  lean_object* grad=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(grad);
  for (size_t t=0;t<P;t++) g[t]=0.0;
  double* gW1=g; double* gb1=g+H*D; double* gW2=g+H*D+H; double* gb2=g+H*D+H+O*H;
  double* z1=(double*)malloc(sizeof(double)*H); double* h=(double*)malloc(sizeof(double)*H);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O); double* dh=(double*)malloc(sizeof(double)*H);
  double halfLog2pi=0.5*log(2.0*M_PI);
  for (size_t n=0;n<N;n++){
    const double* obs=obsAll+n*D; const double* act=actAll+n*A;
    double adv=advA[n], ret=retA[n], oldLogp=oldA[n];
    gauss_forward(pp,obs,H,D,O,z1,h,out);
    double logp=0.0;
    for (size_t i=0;i<A;i++){ double ls=out[A+i]; ls=ls<GC_LO?GC_LO:(ls>GC_HI?GC_HI:ls); double invStd=exp(-ls); double z=(act[i]-out[i])*invStd; logp += -0.5*z*z - ls - halfLog2pi; }
    double ratio=exp(logp-oldLogp); double lo=1.0-clipEps, hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio, surr2=adv*ratioC;
    double dPol; if (surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for (size_t i=0;i<A;i++){
      double lsr=out[A+i]; double ls=lsr<GC_LO?GC_LO:(lsr>GC_HI?GC_HI:lsr);
      double lsGrad=(GC_LO<lsr && lsr<GC_HI)?1.0:0.0;
      double invStd=exp(-ls); double z=(act[i]-out[i])*invStd;
      dout[i] = dPol * (z*invStd);                               // d/dmean_i
      dout[A+i] = lsGrad * ( dPol*(z*z - 1.0) + entCoef );        // d/dlogstd_raw_i
    }
    dout[2*A] = -vfCoef*(out[2*A]-ret);                          // d/dvalue
    for (size_t j=0;j<H;j++) dh[j]=0.0;
    for (size_t k=0;k<O;k++){ double dk=dout[k]; double* gw=gW2+k*H; const double* w=W2+k*H; for (size_t j=0;j<H;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[k]+=dk; }
    for (size_t j=0;j<H;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*D; for (size_t d=0;d<D;d++) gw[d]+=dz*obs[d]; gb1[j]+=dz; }
  }
  free(z1);free(h);free(out);free(dout);free(dh);
  lean_dec(params);lean_dec(obsB);lean_dec(actsB);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);
  return grad;
}

/* Gaussian head gradient WITH value-loss clipping (see lean_ffi_mlp_ppo_grad_batch_vclip). */
LEAN_EXPORT lean_obj_res lean_ffi_gauss_ppo_grad_batch_vclip(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg actsB,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg oldvals,
    size_t N, size_t H, size_t D, size_t A,
    double vfCoef, double entCoef, double clipEps, double vfClip) {
  size_t O=2*A+1; size_t P=H*D+H+O*H+O;
  const double* pp=lean_float_array_cptr(params);
  const double* W2=pp+H*D+H;
  const double* obsAll=lean_float_array_cptr(obsB); const double* actAll=lean_float_array_cptr(actsB);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets); const double* oldA=lean_float_array_cptr(oldlps);
  const double* ovA=lean_float_array_cptr(oldvals);
  size_t OUT=P+2*N;   /* gradient[P] ++ new_logp[N] ++ new_value[N] for the PER ratio/value iteration */
  lean_object* grad=lean_alloc_sarray(sizeof(double),OUT,OUT); double* g=lean_float_array_cptr(grad);
  for (size_t t=0;t<OUT;t++) g[t]=0.0;
  double* gW1=g; double* gb1=g+H*D; double* gW2=g+H*D+H; double* gb2=g+H*D+H+O*H;
  double* z1=(double*)malloc(sizeof(double)*H); double* h=(double*)malloc(sizeof(double)*H);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O); double* dh=(double*)malloc(sizeof(double)*H);
  double halfLog2pi=0.5*log(2.0*M_PI);
  for (size_t n=0;n<N;n++){
    const double* obs=obsAll+n*D; const double* act=actAll+n*A;
    double adv=advA[n], ret=retA[n], oldLogp=oldA[n], vold=ovA[n];
    gauss_forward(pp,obs,H,D,O,z1,h,out);
    double logp=0.0;
    for (size_t i=0;i<A;i++){ double ls=out[A+i]; ls=ls<GC_LO?GC_LO:(ls>GC_HI?GC_HI:ls); double invStd=exp(-ls); double z=(act[i]-out[i])*invStd; logp += -0.5*z*z - ls - halfLog2pi; }
    double ratio=exp(logp-oldLogp); double lo=1.0-clipEps, hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio, surr2=adv*ratioC;
    double dPol; if (surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for (size_t i=0;i<A;i++){
      double lsr=out[A+i]; double ls=lsr<GC_LO?GC_LO:(lsr>GC_HI?GC_HI:lsr);
      double lsGrad=(GC_LO<lsr && lsr<GC_HI)?1.0:0.0;
      double invStd=exp(-ls); double z=(act[i]-out[i])*invStd;
      dout[i] = dPol * (z*invStd);
      dout[A+i] = lsGrad * ( dPol*(z*z - 1.0) + entCoef );
    }
    double vnew=out[2*A], dvloss_dv;
    if (vfClip > 0.0) { double dd=vnew-vold; double vclip=vold + (dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd)); double du=(vnew-ret)*(vnew-ret), dc=(vclip-ret)*(vclip-ret); if (du>=dc) dvloss_dv=(vnew-ret); else if (dd>-vfClip && dd<vfClip) dvloss_dv=(vnew-ret); else dvloss_dv=0.0; } else dvloss_dv=(vnew-ret);
    dout[2*A] = -vfCoef*dvloss_dv;
    g[P+n]=logp; g[P+N+n]=vnew;                 /* new_logp (Gaussian) + new_value */
    for (size_t j=0;j<H;j++) dh[j]=0.0;
    for (size_t k=0;k<O;k++){ double dk=dout[k]; double* gw=gW2+k*H; const double* w=W2+k*H; for (size_t j=0;j<H;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[k]+=dk; }
    for (size_t j=0;j<H;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*D; for (size_t d=0;d<D;d++) gw[d]+=dz*obs[d]; gb1[j]+=dz; }
  }
  free(z1);free(h);free(out);free(dout);free(dh);
  lean_dec(params);lean_dec(obsB);lean_dec(actsB);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(oldvals);
  return grad;
}

/* ---- CNN head PPO gradient (conv encoder + dense) ----------------------------
   params: convW[nF*(C*k*k)], convB[nF], W1[hidden*flatDim], b1[hidden],
           W2[O*hidden], b2[O]; O=A+1, flatDim=nF*outH*outW,
           outH=(inH-k)/s+1, outW=(inW-k)/s+1.
   Forward: feat = relu(conv(obs)) flattened, then MLP head → out (as CnnVecTrain).
   Matches Puffer.RL.NNTrain.cnnGradPPO. */
static inline size_t cnn_patch_obs(size_t idx, size_t C, size_t inH, size_t inW,
                                   size_t k, size_t s, size_t oy, size_t ox) {
  size_t c = idx/(k*k); size_t rem = idx%(k*k); size_t ky = rem/k; size_t kx = rem%k;
  return (c*inH + (oy*s+ky))*inW + (ox*s+kx);
}

/* Forward, storing conv pre-activations `pre` and features `feat` for the backward. */
static void cnn_forward(const double* pp, const double* obs,
    size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t O,
    double* pre, double* feat, double* z1, double* h, double* out) {
  size_t ck = C*k*k;
  size_t outH=(inH-k)/s+1, outW=(inW-k)/s+1, flatDim=nF*outH*outW;
  const double* convW=pp; const double* convB=pp+nF*ck;
  const double* W1=convB+nF; const double* b1=W1+hidden*flatDim;
  const double* W2=b1+hidden; const double* b2=W2+O*hidden;
  size_t fi=0;
  for (size_t f=0;f<nF;f++){ const double* wf=convW+f*ck;
    for (size_t oy=0;oy<outH;oy++) for (size_t ox=0;ox<outW;ox++){
      double acc=0.0; for (size_t idx=ck; idx-->0;) acc = wf[idx]*obs[cnn_patch_obs(idx,C,inH,inW,k,s,oy,ox)] + acc;
      double pv=convB[f]+acc; pre[fi]=pv; feat[fi]=pv>0.0?pv:0.0; fi++;
    }
  }
  for (size_t j=0;j<hidden;j++){ const double* w=W1+j*flatDim; double acc=0.0; for (size_t i=flatDim;i-->0;) acc=w[i]*feat[i]+acc; double z=b1[j]+acc; z1[j]=z; h[j]=z>0.0?z:0.0; }
  for (size_t kk=0;kk<O;kk++){ const double* w=W2+kk*hidden; double acc=0.0; for (size_t j=hidden;j-->0;) acc=w[j]*h[j]+acc; out[kk]=b2[kk]+acc; }
}

LEAN_EXPORT double lean_ffi_cnn_ppo_obj1(
    lean_obj_arg params, lean_obj_arg obsA,
    size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t A, size_t a,
    double adv, double ret, double oldLogp, double vfCoef, double entCoef, double clipEps) {
  size_t O=A+1; size_t outH=(inH-k)/s+1, outW=(inW-k)/s+1, flatDim=nF*outH*outW;
  const double* pp=lean_float_array_cptr(params); const double* obs=lean_float_array_cptr(obsA);
  double* pre=(double*)malloc(sizeof(double)*flatDim); double* feat=(double*)malloc(sizeof(double)*flatDim);
  double* z1=(double*)malloc(sizeof(double)*hidden); double* h=(double*)malloc(sizeof(double)*hidden); double* out=(double*)malloc(sizeof(double)*O);
  cnn_forward(pp,obs,C,inH,inW,nF,k,s,hidden,O,pre,feat,z1,h,out);
  double sumexp=0.0; for (size_t kk=0;kk<A;kk++) sumexp+=exp(out[kk]); double lse=log(sumexp);
  double pout=0.0; for (size_t kk=0;kk<A;kk++){ double pk=exp(out[kk]-lse); pout+=pk*out[kk]; }
  double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
  double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC; double polObj=surr1<=surr2?surr1:surr2;
  double diff=out[A]-ret; double vloss=diff*diff; double ent=lse-pout;
  double obj=(polObj-vfCoef*0.5*vloss)+entCoef*ent;
  free(pre);free(feat);free(z1);free(h);free(out); lean_dec(params);lean_dec(obsA);
  return obj;
}

LEAN_EXPORT lean_obj_res lean_ffi_cnn_ppo_grad_batch(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg acts,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps,
    size_t N, size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t A,
    double vfCoef, double entCoef, double clipEps) {
  size_t O=A+1; size_t ck=C*k*k; size_t outH=(inH-k)/s+1, outW=(inW-k)/s+1, flatDim=nF*outH*outW;
  size_t inSz=C*inH*inW;
  size_t P = nF*ck + nF + hidden*flatDim + hidden + O*hidden + O;
  const double* pp=lean_float_array_cptr(params);
  const double* W1=pp+nF*ck+nF; const double* W2=W1+hidden*flatDim+hidden;
  const double* obsAll=lean_float_array_cptr(obsB); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets); const double* oldA=lean_float_array_cptr(oldlps);
  lean_object* grad=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(grad);
  for (size_t t=0;t<P;t++) g[t]=0.0;
  double* gConvW=g; double* gConvB=g+nF*ck; double* gW1=gConvB+nF; double* gb1=gW1+hidden*flatDim; double* gW2=gb1+hidden; double* gb2=gW2+O*hidden;
  double* pre=(double*)malloc(sizeof(double)*flatDim); double* feat=(double*)malloc(sizeof(double)*flatDim);
  double* z1=(double*)malloc(sizeof(double)*hidden); double* h=(double*)malloc(sizeof(double)*hidden);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O);
  double* dh=(double*)malloc(sizeof(double)*hidden); double* dfeat=(double*)malloc(sizeof(double)*flatDim); double* pk=(double*)malloc(sizeof(double)*A);
  for (size_t n=0;n<N;n++){
    const double* obs=obsAll+n*inSz; size_t a=(size_t)actA[n]; double adv=advA[n],ret=retA[n],oldLogp=oldA[n];
    cnn_forward(pp,obs,C,inH,inW,nF,k,s,hidden,O,pre,feat,z1,h,out);
    double sumexp=0.0; for (size_t kk=0;kk<A;kk++) sumexp+=exp(out[kk]); double lse=log(sumexp);
    double pout=0.0; for (size_t kk=0;kk<A;kk++){ pk[kk]=exp(out[kk]-lse); pout+=pk[kk]*out[kk]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double dPol; if (surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for (size_t kk=0;kk<A;kk++){ double dp=dPol*(((kk==a)?1.0:0.0)-pk[kk]); double de=entCoef*pk[kk]*(pout-out[kk]); dout[kk]=dp+de; }
    dout[A]=-vfCoef*(out[A]-ret);
    for (size_t j=0;j<hidden;j++) dh[j]=0.0;
    for (size_t kk=0;kk<O;kk++){ double dk=dout[kk]; double* gw=gW2+kk*hidden; const double* w=W2+kk*hidden; for (size_t j=0;j<hidden;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[kk]+=dk; }
    for (size_t i=0;i<flatDim;i++) dfeat[i]=0.0;
    for (size_t j=0;j<hidden;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*flatDim; const double* w=W1+j*flatDim; for (size_t i=0;i<flatDim;i++){ gw[i]+=dz*feat[i]; dfeat[i]+=dz*w[i]; } gb1[j]+=dz; }
    /* conv backward */
    size_t fi=0;
    for (size_t f=0;f<nF;f++){ double* gwf=gConvW+f*ck;
      for (size_t oy=0;oy<outH;oy++) for (size_t ox=0;ox<outW;ox++){
        double dp=(pre[fi]>0.0)?dfeat[fi]:0.0;
        for (size_t idx=0;idx<ck;idx++) gwf[idx]+=dp*obs[cnn_patch_obs(idx,C,inH,inW,k,s,oy,ox)];
        gConvB[f]+=dp; fi++;
      }
    }
  }
  free(pre);free(feat);free(z1);free(h);free(out);free(dout);free(dh);free(dfeat);free(pk);
  lean_dec(params);lean_dec(obsB);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);
  return grad;
}

/* CNN head gradient WITH value-loss clipping (see lean_ffi_mlp_ppo_grad_batch_vclip). */
LEAN_EXPORT lean_obj_res lean_ffi_cnn_ppo_grad_batch_vclip(
    lean_obj_arg params, lean_obj_arg obsB, lean_obj_arg acts,
    lean_obj_arg advs, lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg oldvals,
    size_t N, size_t C, size_t inH, size_t inW, size_t nF, size_t k, size_t s, size_t hidden, size_t A,
    double vfCoef, double entCoef, double clipEps, double vfClip) {
  size_t O=A+1; size_t ck=C*k*k; size_t outH=(inH-k)/s+1, outW=(inW-k)/s+1, flatDim=nF*outH*outW;
  size_t inSz=C*inH*inW;
  size_t P = nF*ck + nF + hidden*flatDim + hidden + O*hidden + O;
  const double* pp=lean_float_array_cptr(params);
  const double* W1=pp+nF*ck+nF; const double* W2=W1+hidden*flatDim+hidden;
  const double* obsAll=lean_float_array_cptr(obsB); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets); const double* oldA=lean_float_array_cptr(oldlps);
  const double* ovA=lean_float_array_cptr(oldvals);
  size_t OUT=P+2*N;   /* gradient[P] ++ new_logp[N] ++ new_value[N] for the PER ratio/value iteration */
  lean_object* grad=lean_alloc_sarray(sizeof(double),OUT,OUT); double* g=lean_float_array_cptr(grad);
  for (size_t t=0;t<OUT;t++) g[t]=0.0;
  double* gConvW=g; double* gConvB=g+nF*ck; double* gW1=gConvB+nF; double* gb1=gW1+hidden*flatDim; double* gW2=gb1+hidden; double* gb2=gW2+O*hidden;
  double* pre=(double*)malloc(sizeof(double)*flatDim); double* feat=(double*)malloc(sizeof(double)*flatDim);
  double* z1=(double*)malloc(sizeof(double)*hidden); double* h=(double*)malloc(sizeof(double)*hidden);
  double* out=(double*)malloc(sizeof(double)*O); double* dout=(double*)malloc(sizeof(double)*O);
  double* dh=(double*)malloc(sizeof(double)*hidden); double* dfeat=(double*)malloc(sizeof(double)*flatDim); double* pk=(double*)malloc(sizeof(double)*A);
  for (size_t n=0;n<N;n++){
    const double* obs=obsAll+n*inSz; size_t a=(size_t)actA[n]; double adv=advA[n],ret=retA[n],oldLogp=oldA[n],vold=ovA[n];
    cnn_forward(pp,obs,C,inH,inW,nF,k,s,hidden,O,pre,feat,z1,h,out);
    double sumexp=0.0; for (size_t kk=0;kk<A;kk++) sumexp+=exp(out[kk]); double lse=log(sumexp);
    double pout=0.0; for (size_t kk=0;kk<A;kk++){ pk[kk]=exp(out[kk]-lse); pout+=pk[kk]*out[kk]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double dPol; if (surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for (size_t kk=0;kk<A;kk++){ double dp=dPol*(((kk==a)?1.0:0.0)-pk[kk]); double de=entCoef*pk[kk]*(pout-out[kk]); dout[kk]=dp+de; }
    double vnew=out[A], dvloss_dv;
    if (vfClip > 0.0) { double dd=vnew-vold; double vclip=vold + (dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd)); double du=(vnew-ret)*(vnew-ret), dc=(vclip-ret)*(vclip-ret); if (du>=dc) dvloss_dv=(vnew-ret); else if (dd>-vfClip && dd<vfClip) dvloss_dv=(vnew-ret); else dvloss_dv=0.0; } else dvloss_dv=(vnew-ret);
    dout[A]=-vfCoef*dvloss_dv;
    g[P+n]=logpA; g[P+N+n]=vnew;                 /* new_logp (taken action) + new_value */
    for (size_t j=0;j<hidden;j++) dh[j]=0.0;
    for (size_t kk=0;kk<O;kk++){ double dk=dout[kk]; double* gw=gW2+kk*hidden; const double* w=W2+kk*hidden; for (size_t j=0;j<hidden;j++){ gw[j]+=dk*h[j]; dh[j]+=dk*w[j]; } gb2[kk]+=dk; }
    for (size_t i=0;i<flatDim;i++) dfeat[i]=0.0;
    for (size_t j=0;j<hidden;j++){ double dz=(z1[j]>0.0)?dh[j]:0.0; double* gw=gW1+j*flatDim; const double* w=W1+j*flatDim; for (size_t i=0;i<flatDim;i++){ gw[i]+=dz*feat[i]; dfeat[i]+=dz*w[i]; } gb1[j]+=dz; }
    size_t fi=0;
    for (size_t f=0;f<nF;f++){ double* gwf=gConvW+f*ck;
      for (size_t oy=0;oy<outH;oy++) for (size_t ox=0;ox<outW;ox++){
        double dp=(pre[fi]>0.0)?dfeat[fi]:0.0;
        for (size_t idx=0;idx<ck;idx++) gwf[idx]+=dp*obs[cnn_patch_obs(idx,C,inH,inW,k,s,oy,ox)];
        gConvB[f]+=dp; fi++;
      }
    }
  }
  free(pre);free(feat);free(z1);free(h);free(out);free(dout);free(dh);free(dfeat);free(pk);
  lean_dec(params);lean_dec(obsB);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(oldvals);
  return grad;
}

/* ---- LSTM head PPO gradient with truncated BPTT -----------------------------
   params: Wx[4H*D], Wh[4H*H], bih[4H], Wo[dout*H], bo[dout] (dout=A+1); gate rows
   stacked i[0:H] f[H:2H] g[2H:3H] o[3H:4H]. One env-sequence (length T); h0,c0 the
   DETACHED BPTT-initial state; terms[t]!=0 marks a terminal at step t, so step t+1
   starts from a zeroed detached state and the gradient does NOT flow across the
   boundary (nor past t=0). Left-folded dots match RecVecTrain's `dotV`/`dotL`, so the
   forward equals the AD-tape primal. Native twin of Puffer.RL.NNTrain.recPPOGradSeq. */
static inline double lstm_sig(double x){ return 1.0/(1.0+exp(-x)); }

LEAN_EXPORT double lean_ffi_lstm_ppo_obj_seq(
    lean_obj_arg params, lean_obj_arg obsSeq, lean_obj_arg acts, lean_obj_arg advs,
    lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg terms,
    lean_obj_arg h0a, lean_obj_arg c0a,
    size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t dout=A+1, H4=4*H;
  const double* Wx=lean_float_array_cptr(params); const double* Wh=Wx+H4*D;
  const double* bih=Wh+H4*H; const double* Wo=bih+H4; const double* bo=Wo+dout*H;
  const double* obs=lean_float_array_cptr(obsSeq); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets);
  const double* oldA=lean_float_array_cptr(oldlps); const double* termA=lean_float_array_cptr(terms);
  const double* h0=lean_float_array_cptr(h0a); const double* c0=lean_float_array_cptr(c0a);
  double* h=(double*)malloc(sizeof(double)*H); double* c=(double*)malloc(sizeof(double)*H);
  double* gate=(double*)malloc(sizeof(double)*H4); double* out=(double*)malloc(sizeof(double)*dout);
  for(size_t j=0;j<H;j++){h[j]=h0[j];c[j]=c0[j];}
  double total=0.0;
  for(size_t t=0;t<T;t++){
    if(t>0 && termA[t-1]!=0.0){ for(size_t j=0;j<H;j++){h[j]=0.0;c[j]=0.0;} }
    const double* x=obs+t*D;
    for(size_t k=0;k<H4;k++){ double gx=0.0; const double* wx=Wx+k*D; for(size_t d=0;d<D;d++) gx+=wx[d]*x[d];
      double gh=0.0; const double* wh=Wh+k*H; for(size_t d=0;d<H;d++) gh+=wh[d]*h[d]; gate[k]=bih[k]+gx+gh; }
    for(size_t j=0;j<H;j++){ double ig=lstm_sig(gate[j]),fg=lstm_sig(gate[H+j]),gg=tanh(gate[2*H+j]),og=lstm_sig(gate[3*H+j]);
      double cj=fg*c[j]+ig*gg; c[j]=cj; h[j]=og*tanh(cj); }
    for(size_t m=0;m<dout;m++){ double acc=0.0; const double* w=Wo+m*H; for(size_t j=0;j<H;j++) acc+=w[j]*h[j]; out[m]=bo[m]+acc; }
    size_t a=(size_t)actA[t]; double adv=advA[t],ret=retA[t],oldLogp=oldA[t];
    double sumexp=0.0; for(size_t k=0;k<A;k++) sumexp+=exp(out[k]); double lse=log(sumexp);
    double pout=0.0; for(size_t k=0;k<A;k++){ double pk=exp(out[k]-lse); pout+=pk*out[k]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double polObj=surr1<=surr2?surr1:surr2; double diff=out[A]-ret; double vloss=diff*diff; double ent=lse-pout;
    total += (polObj - vfCoef*0.5*vloss) + entCoef*ent;
  }
  free(h);free(c);free(gate);free(out);
  lean_dec(params);lean_dec(obsSeq);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(terms);lean_dec(h0a);lean_dec(c0a);
  return total;
}

LEAN_EXPORT lean_obj_res lean_ffi_lstm_ppo_grad_seq(
    lean_obj_arg params, lean_obj_arg obsSeq, lean_obj_arg acts, lean_obj_arg advs,
    lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg terms,
    lean_obj_arg h0a, lean_obj_arg c0a,
    size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps) {
  size_t dout=A+1, H4=4*H;
  size_t P = H4*D + H4*H + H4 + dout*H + dout;
  const double* Wx=lean_float_array_cptr(params); const double* Wh=Wx+H4*D;
  const double* bih=Wh+H4*H; const double* Wo=bih+H4; const double* bo=Wo+dout*H;
  const double* obs=lean_float_array_cptr(obsSeq); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets);
  const double* oldA=lean_float_array_cptr(oldlps); const double* termA=lean_float_array_cptr(terms);
  const double* h0=lean_float_array_cptr(h0a); const double* c0=lean_float_array_cptr(c0a);
  lean_object* grad=lean_alloc_sarray(sizeof(double),P,P); double* g=lean_float_array_cptr(grad);
  for(size_t t=0;t<P;t++) g[t]=0.0;
  double* gWx=g; double* gWh=gWx+H4*D; double* gbih=gWh+H4*H; double* gWo=gbih+H4; double* gbo=gWo+dout*H;
  double* ii=(double*)malloc(sizeof(double)*T*H); double* ff=(double*)malloc(sizeof(double)*T*H);
  double* ggA=(double*)malloc(sizeof(double)*T*H); double* oo=(double*)malloc(sizeof(double)*T*H);
  double* tc=(double*)malloc(sizeof(double)*T*H); double* hp=(double*)malloc(sizeof(double)*T*H);
  double* cp=(double*)malloc(sizeof(double)*T*H); double* ht=(double*)malloc(sizeof(double)*T*H);
  double* outs=(double*)malloc(sizeof(double)*T*dout);
  double* h=(double*)malloc(sizeof(double)*H); double* c=(double*)malloc(sizeof(double)*H);
  double* gate=(double*)malloc(sizeof(double)*H4);
  for(size_t j=0;j<H;j++){h[j]=h0[j];c[j]=c0[j];}
  /* FORWARD (store activations) */
  for(size_t t=0;t<T;t++){
    if(t>0 && termA[t-1]!=0.0){ for(size_t j=0;j<H;j++){h[j]=0.0;c[j]=0.0;} }
    for(size_t j=0;j<H;j++){ hp[t*H+j]=h[j]; cp[t*H+j]=c[j]; }
    const double* x=obs+t*D;
    for(size_t k=0;k<H4;k++){ double gx=0.0; const double* wx=Wx+k*D; for(size_t d=0;d<D;d++) gx+=wx[d]*x[d];
      double gh=0.0; const double* wh=Wh+k*H; for(size_t d=0;d<H;d++) gh+=wh[d]*h[d]; gate[k]=bih[k]+gx+gh; }
    for(size_t j=0;j<H;j++){ double ig=lstm_sig(gate[j]),fg=lstm_sig(gate[H+j]),gg=tanh(gate[2*H+j]),og=lstm_sig(gate[3*H+j]);
      double cj=fg*c[j]+ig*gg; double tcj=tanh(cj);
      ii[t*H+j]=ig; ff[t*H+j]=fg; ggA[t*H+j]=gg; oo[t*H+j]=og; tc[t*H+j]=tcj;
      c[j]=cj; h[j]=og*tcj; ht[t*H+j]=h[j]; }
    for(size_t m=0;m<dout;m++){ double acc=0.0; const double* w=Wo+m*H; for(size_t j=0;j<H;j++) acc+=w[j]*h[j]; outs[t*dout+m]=bo[m]+acc; }
  }
  /* BACKWARD (reverse time) */
  double* dh_next=(double*)calloc(H,sizeof(double)); double* dc_next=(double*)calloc(H,sizeof(double));
  double* dh=(double*)malloc(sizeof(double)*H); double* dhprev=(double*)malloc(sizeof(double)*H);
  double* dcprev=(double*)malloc(sizeof(double)*H); double* dgate=(double*)malloc(sizeof(double)*H4);
  double* dout_=(double*)malloc(sizeof(double)*dout); double* pk=(double*)malloc(sizeof(double)*A);
  for(size_t tt=T; tt-->0; ){
    size_t t=tt; const double* out=outs+t*dout;
    size_t a=(size_t)actA[t]; double adv=advA[t],ret=retA[t],oldLogp=oldA[t];
    double sumexp=0.0; for(size_t k=0;k<A;k++) sumexp+=exp(out[k]); double lse=log(sumexp);
    double pout=0.0; for(size_t k=0;k<A;k++){ pk[k]=exp(out[k]-lse); pout+=pk[k]*out[k]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double dPol; if(surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for(size_t k=0;k<A;k++){ double dp=dPol*(((k==a)?1.0:0.0)-pk[k]); double de=entCoef*pk[k]*(pout-out[k]); dout_[k]=dp+de; }
    dout_[A]=-vfCoef*(out[A]-ret);
    for(size_t j=0;j<H;j++) dh[j]=dh_next[j];
    for(size_t m=0;m<dout;m++){ double dm=dout_[m]; double* gw=gWo+m*H; const double* w=Wo+m*H;
      gbo[m]+=dm; for(size_t j=0;j<H;j++){ gw[j]+=dm*ht[t*H+j]; dh[j]+=dm*w[j]; } }
    for(size_t j=0;j<H;j++){
      double tcj=tc[t*H+j], og=oo[t*H+j];
      double dcj=dc_next[j] + dh[j]*og*(1.0-tcj*tcj);
      double do_=dh[j]*tcj; double ig=ii[t*H+j], fg=ff[t*H+j], gg=ggA[t*H+j], cprev=cp[t*H+j];
      double df_=dcj*cprev, di_=dcj*gg, dg_=dcj*ig; dcprev[j]=dcj*fg;
      dgate[j]=di_*ig*(1.0-ig); dgate[H+j]=df_*fg*(1.0-fg);
      dgate[2*H+j]=dg_*(1.0-gg*gg); dgate[3*H+j]=do_*og*(1.0-og);
    }
    for(size_t d=0;d<H;d++) dhprev[d]=0.0;
    const double* x=obs+t*D; const double* hprevk=hp+t*H;
    for(size_t k=0;k<H4;k++){ double da=dgate[k]; gbih[k]+=da;
      double* gwx=gWx+k*D; for(size_t d=0;d<D;d++) gwx[d]+=da*x[d];
      double* gwh=gWh+k*H; const double* wh=Wh+k*H;
      for(size_t d=0;d<H;d++){ gwh[d]+=da*hprevk[d]; dhprev[d]+=da*wh[d]; } }
    int flow = (t>0 && termA[t-1]==0.0);
    if(flow){ for(size_t j=0;j<H;j++){ dh_next[j]=dhprev[j]; dc_next[j]=dcprev[j]; } }
    else { for(size_t j=0;j<H;j++){ dh_next[j]=0.0; dc_next[j]=0.0; } }
  }
  free(ii);free(ff);free(ggA);free(oo);free(tc);free(hp);free(cp);free(ht);free(outs);
  free(h);free(c);free(gate);free(dh_next);free(dc_next);free(dh);free(dhprev);free(dcprev);free(dgate);free(dout_);free(pk);
  lean_dec(params);lean_dec(obsSeq);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(terms);lean_dec(h0a);lean_dec(c0a);
  return grad;
}

/* LSTM BPTT gradient WITH value-loss clipping (see lean_ffi_mlp_ppo_grad_batch_vclip):
   extra oldvals[T] (V at collection) + vfClip. Otherwise identical to lean_ffi_lstm_ppo_grad_seq. */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_ppo_grad_seq_vclip(
    lean_obj_arg params, lean_obj_arg obsSeq, lean_obj_arg acts, lean_obj_arg advs,
    lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg terms,
    lean_obj_arg h0a, lean_obj_arg c0a, lean_obj_arg oldvals,
    size_t T, size_t H, size_t D, size_t A, double vfCoef, double entCoef, double clipEps, double vfClip) {
  size_t dout=A+1, H4=4*H;
  size_t P = H4*D + H4*H + H4 + dout*H + dout;
  const double* Wx=lean_float_array_cptr(params); const double* Wh=Wx+H4*D;
  const double* bih=Wh+H4*H; const double* Wo=bih+H4; const double* bo=Wo+dout*H;
  const double* obs=lean_float_array_cptr(obsSeq); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets);
  const double* oldA=lean_float_array_cptr(oldlps); const double* termA=lean_float_array_cptr(terms);
  const double* h0=lean_float_array_cptr(h0a); const double* c0=lean_float_array_cptr(c0a);
  const double* ovA=lean_float_array_cptr(oldvals);
  size_t OUT=P+2*T;   /* gradient[P] ++ new_logp[T] ++ new_value[T] for the PER ratio/value iteration */
  lean_object* grad=lean_alloc_sarray(sizeof(double),OUT,OUT); double* g=lean_float_array_cptr(grad);
  for(size_t t=0;t<OUT;t++) g[t]=0.0;
  double* gWx=g; double* gWh=gWx+H4*D; double* gbih=gWh+H4*H; double* gWo=gbih+H4; double* gbo=gWo+dout*H;
  double* ii=(double*)malloc(sizeof(double)*T*H); double* ff=(double*)malloc(sizeof(double)*T*H);
  double* ggA=(double*)malloc(sizeof(double)*T*H); double* oo=(double*)malloc(sizeof(double)*T*H);
  double* tc=(double*)malloc(sizeof(double)*T*H); double* hp=(double*)malloc(sizeof(double)*T*H);
  double* cp=(double*)malloc(sizeof(double)*T*H); double* ht=(double*)malloc(sizeof(double)*T*H);
  double* outs=(double*)malloc(sizeof(double)*T*dout);
  double* h=(double*)malloc(sizeof(double)*H); double* c=(double*)malloc(sizeof(double)*H);
  double* gate=(double*)malloc(sizeof(double)*H4);
  for(size_t j=0;j<H;j++){h[j]=h0[j];c[j]=c0[j];}
  for(size_t t=0;t<T;t++){
    if(t>0 && termA[t-1]!=0.0){ for(size_t j=0;j<H;j++){h[j]=0.0;c[j]=0.0;} }
    for(size_t j=0;j<H;j++){ hp[t*H+j]=h[j]; cp[t*H+j]=c[j]; }
    const double* x=obs+t*D;
    for(size_t k=0;k<H4;k++){ double gx=0.0; const double* wx=Wx+k*D; for(size_t d=0;d<D;d++) gx+=wx[d]*x[d];
      double gh=0.0; const double* wh=Wh+k*H; for(size_t d=0;d<H;d++) gh+=wh[d]*h[d]; gate[k]=bih[k]+gx+gh; }
    for(size_t j=0;j<H;j++){ double ig=lstm_sig(gate[j]),fg=lstm_sig(gate[H+j]),gg=tanh(gate[2*H+j]),og=lstm_sig(gate[3*H+j]);
      double cj=fg*c[j]+ig*gg; double tcj=tanh(cj);
      ii[t*H+j]=ig; ff[t*H+j]=fg; ggA[t*H+j]=gg; oo[t*H+j]=og; tc[t*H+j]=tcj;
      c[j]=cj; h[j]=og*tcj; ht[t*H+j]=h[j]; }
    for(size_t m=0;m<dout;m++){ double acc=0.0; const double* w=Wo+m*H; for(size_t j=0;j<H;j++) acc+=w[j]*h[j]; outs[t*dout+m]=bo[m]+acc; }
  }
  double* dh_next=(double*)calloc(H,sizeof(double)); double* dc_next=(double*)calloc(H,sizeof(double));
  double* dh=(double*)malloc(sizeof(double)*H); double* dhprev=(double*)malloc(sizeof(double)*H);
  double* dcprev=(double*)malloc(sizeof(double)*H); double* dgate=(double*)malloc(sizeof(double)*H4);
  double* dout_=(double*)malloc(sizeof(double)*dout); double* pk=(double*)malloc(sizeof(double)*A);
  for(size_t tt=T; tt-->0; ){
    size_t t=tt; const double* out=outs+t*dout;
    size_t a=(size_t)actA[t]; double adv=advA[t],ret=retA[t],oldLogp=oldA[t],vold=ovA[t];
    double sumexp=0.0; for(size_t k=0;k<A;k++) sumexp+=exp(out[k]); double lse=log(sumexp);
    double pout=0.0; for(size_t k=0;k<A;k++){ pk[k]=exp(out[k]-lse); pout+=pk[k]*out[k]; }
    double logpA=out[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double dPol; if(surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    for(size_t k=0;k<A;k++){ double dp=dPol*(((k==a)?1.0:0.0)-pk[k]); double de=entCoef*pk[k]*(pout-out[k]); dout_[k]=dp+de; }
    double vnew=out[A], dvloss_dv;
    if (vfClip > 0.0) { double dd=vnew-vold; double vclip=vold + (dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd)); double du=(vnew-ret)*(vnew-ret), dc=(vclip-ret)*(vclip-ret); if (du>=dc) dvloss_dv=(vnew-ret); else if (dd>-vfClip && dd<vfClip) dvloss_dv=(vnew-ret); else dvloss_dv=0.0; } else dvloss_dv=(vnew-ret);
    dout_[A]=-vfCoef*dvloss_dv;
    g[P+t]=logpA; g[P+T+t]=vnew;                 /* new_logp (taken action) + new_value at t */
    for(size_t j=0;j<H;j++) dh[j]=dh_next[j];
    for(size_t m=0;m<dout;m++){ double dm=dout_[m]; double* gw=gWo+m*H; const double* w=Wo+m*H;
      gbo[m]+=dm; for(size_t j=0;j<H;j++){ gw[j]+=dm*ht[t*H+j]; dh[j]+=dm*w[j]; } }
    for(size_t j=0;j<H;j++){
      double tcj=tc[t*H+j], og=oo[t*H+j];
      double dcj=dc_next[j] + dh[j]*og*(1.0-tcj*tcj);
      double do_=dh[j]*tcj; double ig=ii[t*H+j], fg=ff[t*H+j], gg=ggA[t*H+j], cprev=cp[t*H+j];
      double df_=dcj*cprev, di_=dcj*gg, dg_=dcj*ig; dcprev[j]=dcj*fg;
      dgate[j]=di_*ig*(1.0-ig); dgate[H+j]=df_*fg*(1.0-fg);
      dgate[2*H+j]=dg_*(1.0-gg*gg); dgate[3*H+j]=do_*og*(1.0-og);
    }
    for(size_t d=0;d<H;d++) dhprev[d]=0.0;
    const double* x=obs+t*D; const double* hprevk=hp+t*H;
    for(size_t k=0;k<H4;k++){ double da=dgate[k]; gbih[k]+=da;
      double* gwx=gWx+k*D; for(size_t d=0;d<D;d++) gwx[d]+=da*x[d];
      double* gwh=gWh+k*H; const double* wh=Wh+k*H;
      for(size_t d=0;d<H;d++){ gwh[d]+=da*hprevk[d]; dhprev[d]+=da*wh[d]; } }
    int flow = (t>0 && termA[t-1]==0.0);
    if(flow){ for(size_t j=0;j<H;j++){ dh_next[j]=dhprev[j]; dc_next[j]=dcprev[j]; } }
    else { for(size_t j=0;j<H;j++){ dh_next[j]=0.0; dc_next[j]=0.0; } }
  }
  free(ii);free(ff);free(ggA);free(oo);free(tc);free(hp);free(cp);free(ht);free(outs);
  free(h);free(c);free(gate);free(dh_next);free(dc_next);free(dh);free(dhprev);free(dcprev);free(dgate);free(dout_);free(pk);
  lean_dec(params);lean_dec(obsSeq);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(terms);lean_dec(h0a);lean_dec(c0a);lean_dec(oldvals);
  return grad;
}

/* ---- Batched LSTM forward step for the rollout (native C, one timestep, N rows) -----------------
   Replaces the per-env Lean `lstmCellF` glue (Array.range/map/push per env per timestep -- this WAS
   the trainer's bottleneck once the BPTT step moved to `lean_ffi_lstm_ppo_grad_seq`: rollout 39.6s vs
   BPTT 9.7s at a 256-env/H64/T64 config) with ONE C call over the whole N-row batch. Same left-fold
   summation order as `dotL` (dwx then dwh, `(bih[k]+dwx)+dwh`) so it is bit-exact vs `lstmCellF`, not
   tolerance-close. params flat: Wx[4H·D],Wh[4H·H],bih[4H],Wo[O·H],bo[O] (O=A+1). obs/h/c are N·D / N·H
   / N·H row-major. Returns [hN(N·H); cN(N·H); out(N·O)] -- `out` feeds straight into
   `lean_ffi_sample_actions_batch` (same O=A+1 logits-then-value format). */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_fwd_step_batch(
    lean_obj_arg pa, lean_obj_arg obsa, lean_obj_arg ha, lean_obj_arg ca,
    size_t N, size_t D, size_t H, size_t A){
  size_t O=A+1, H4=4*H;
  const double* pp=lean_float_array_cptr(pa);
  const double* Wx=pp; const double* Wh=Wx+H4*D; const double* bih=Wh+H4*H;
  const double* Wo=bih+H4; const double* bo=Wo+O*H;
  const double* obs=lean_float_array_cptr(obsa); const double* hin=lean_float_array_cptr(ha); const double* cin=lean_float_array_cptr(ca);
  size_t outSz = N*H + N*H + N*O;
  lean_object* go=lean_alloc_sarray(sizeof(double),outSz,outSz); double* out=lean_float_array_cptr(go);
  double* hN=out; double* cN=out+N*H; double* outO=out+2*N*H;
  double* gate=(double*)malloc(sizeof(double)*H4);
  for(size_t n=0;n<N;n++){
    const double* x=obs+n*D; const double* h=hin+n*H; const double* c=cin+n*H;
    for(size_t k=0;k<H4;k++){ double gx=0.0; const double* wx=Wx+k*D; for(size_t d=0;d<D;d++) gx+=wx[d]*x[d];
      double gh=0.0; const double* wh=Wh+k*H; for(size_t d=0;d<H;d++) gh+=wh[d]*h[d]; gate[k]=(bih[k]+gx)+gh; }
    double* hnrow=hN+n*H; double* cnrow=cN+n*H;
    for(size_t j=0;j<H;j++){ double ig=lstm_sig(gate[j]),fg=lstm_sig(gate[H+j]),gg=tanh(gate[2*H+j]),og=lstm_sig(gate[3*H+j]);
      double cj=fg*c[j]+ig*gg; cnrow[j]=cj; hnrow[j]=og*tanh(cj); }
    double* orow=outO+n*O;
    for(size_t m=0;m<O;m++){ double acc=0.0; const double* w=Wo+m*H; for(size_t j=0;j<H;j++) acc+=w[j]*hnrow[j]; orow[m]=bo[m]+acc; }
  }
  free(gate);
  lean_dec(pa);lean_dec(obsa);lean_dec(ha);lean_dec(ca);
  return go;
}

/* ---- MinGRU head PPO gradient with truncated BPTT ---------------------------
   PufferLib's DEFAULT net: DefaultEncoder(Linear obs->H) -> numLayers x MinGRU ->
   DefaultDecoder(Linear->A logits, Linear->1 value). Params (flattenMG layout):
     wEnc[H*obsSize], bEnc[H], layers[numLayers][3H*H], wDec[A*H], bDec[A], wVal[H], bVal[1].
   Per layer: y=W.h; z=sig(gate); g=_g(hid) (hid+0.5 if hid>=0 else sig(hid));
     o=(1-z)*prev+z*g; hg=sig(proj); hn=hg*o+(1-hg)*h; state<-o; h<-hn.
   Native twin of Puffer.RL.NNTrain.mingruGradSeq (the AD-tape oracle); value-loss
   clipping as in lean_ffi_mlp_ppo_grad_batch_vclip. One env-sequence (length T),
   zero initial state, state reset to zero after a terminal (no BPTT across). */
LEAN_EXPORT lean_obj_res lean_ffi_mingru_ppo_grad_seq(
    lean_obj_arg params, lean_obj_arg obsSeq, lean_obj_arg acts, lean_obj_arg advs,
    lean_obj_arg rets, lean_obj_arg oldlps, lean_obj_arg terms, lean_obj_arg oldvals,
    size_t T, size_t H, size_t obsSize, size_t numLayers, size_t A,
    double vfCoef, double entCoef, double clipEps, double vfClip) {
  size_t wEncSz=H*obsSize, layerSz=3*H*H;
  size_t P = wEncSz + H + numLayers*layerSz + A*H + A + H + 1;
  const double* pp=lean_float_array_cptr(params);
  const double* wEnc=pp; const double* bEnc=pp+wEncSz;
  const double* layersBase=pp+wEncSz+H;
  const double* wDec=layersBase+numLayers*layerSz; const double* bDec=wDec+A*H;
  const double* wVal=bDec+A; const double* bVal=wVal+H;
  const double* obsAll=lean_float_array_cptr(obsSeq); const double* actA=lean_float_array_cptr(acts);
  const double* advA=lean_float_array_cptr(advs); const double* retA=lean_float_array_cptr(rets);
  const double* oldA=lean_float_array_cptr(oldlps); const double* termA=lean_float_array_cptr(terms);
  const double* ovA=lean_float_array_cptr(oldvals);
  size_t OUT=P+2*T;   /* gradient[P] ++ new_logp[T] ++ new_value[T] for the PER ratio/value iteration */
  lean_object* grad=lean_alloc_sarray(sizeof(double),OUT,OUT); double* g=lean_float_array_cptr(grad);
  for(size_t i=0;i<OUT;i++) g[i]=0.0;
  double* gWEnc=g; double* gBEnc=g+wEncSz; double* gLayers=g+wEncSz+H;
  double* gWDec=gLayers+numLayers*layerSz; double* gBDec=gWDec+A*H; double* gWVal=gBDec+A; double* gBVal=gWVal+H;
  size_t TLH=T*numLayers*H;
  double* hin=(double*)malloc(sizeof(double)*TLH); double* prevS=(double*)malloc(sizeof(double)*TLH);
  double* hidA=(double*)malloc(sizeof(double)*TLH); double* gateA=(double*)malloc(sizeof(double)*TLH);
  double* projA=(double*)malloc(sizeof(double)*TLH); double* oA=(double*)malloc(sizeof(double)*TLH);
  double* hfin=(double*)malloc(sizeof(double)*T*H); double* lg=(double*)malloc(sizeof(double)*T*A);
  double* val=(double*)malloc(sizeof(double)*T);
  double* st=(double*)calloc(numLayers*H,sizeof(double));
  double* hb=(double*)malloc(sizeof(double)*H); double* yb=(double*)malloc(sizeof(double)*3*H);
  double* hnew=(double*)malloc(sizeof(double)*H);
  for(size_t t=0;t<T;t++){
    const double* obs=obsAll+t*obsSize;
    for(size_t i=0;i<H;i++){ double acc=0.0; const double* w=wEnc+i*obsSize; for(size_t k=0;k<obsSize;k++) acc+=w[k]*obs[k]; hb[i]=bEnc[i]+acc; }
    for(size_t l=0;l<numLayers;l++){
      const double* Wl=layersBase+l*layerSz; size_t base=(t*numLayers+l)*H;
      for(size_t j=0;j<H;j++){ hin[base+j]=hb[j]; prevS[base+j]=st[l*H+j]; }
      for(size_t r=0;r<3*H;r++){ double acc=0.0; const double* w=Wl+r*H; for(size_t j=0;j<H;j++) acc+=w[j]*hb[j]; yb[r]=acc; }
      for(size_t j=0;j<H;j++){
        double hidj=yb[j], gatej=yb[H+j], projj=yb[2*H+j];
        double zj=lstm_sig(gatej);
        double gj=(hidj>=0.0)?(hidj+0.5):lstm_sig(hidj);
        double prevj=st[l*H+j];
        double oj=(1.0-zj)*prevj+zj*gj;
        double hgj=lstm_sig(projj);
        hnew[j]=hgj*oj+(1.0-hgj)*hb[j];
        hidA[base+j]=hidj; gateA[base+j]=gatej; projA[base+j]=projj; oA[base+j]=oj;
        st[l*H+j]=oj;
      }
      for(size_t j=0;j<H;j++) hb[j]=hnew[j];
    }
    for(size_t j=0;j<H;j++) hfin[t*H+j]=hb[j];
    for(size_t k=0;k<A;k++){ double acc=0.0; const double* w=wDec+k*H; for(size_t j=0;j<H;j++) acc+=w[j]*hb[j]; lg[t*A+k]=bDec[k]+acc; }
    { double acc=0.0; for(size_t j=0;j<H;j++) acc+=wVal[j]*hb[j]; val[t]=bVal[0]+acc; }
    if(termA[t]!=0.0){ for(size_t x=0;x<numLayers*H;x++) st[x]=0.0; }
  }
  double* dOnext=(double*)calloc(numLayers*H,sizeof(double));
  double* dprev=(double*)malloc(sizeof(double)*numLayers*H);
  double* dhn=(double*)malloc(sizeof(double)*H); double* dhin=(double*)malloc(sizeof(double)*H);
  double* dhf=(double*)malloc(sizeof(double)*H);
  double* dyh=(double*)malloc(sizeof(double)*H); double* dyg=(double*)malloc(sizeof(double)*H); double* dyp=(double*)malloc(sizeof(double)*H);
  double* pk=(double*)malloc(sizeof(double)*A);
  for(size_t tt=T; tt-->0;){
    size_t t=tt; const double* obs=obsAll+t*obsSize; const double* logits=lg+t*A;
    double sumexp=0.0; for(size_t k=0;k<A;k++) sumexp+=exp(logits[k]); double lse=log(sumexp);
    double pout=0.0; for(size_t k=0;k<A;k++){ pk[k]=exp(logits[k]-lse); pout+=pk[k]*logits[k]; }
    size_t a=(size_t)actA[t]; double adv=advA[t],ret=retA[t],oldLogp=oldA[t],vold=ovA[t];
    double logpA=logits[a]-lse; double ratio=exp(logpA-oldLogp); double lo=1.0-clipEps,hi=1.0+clipEps;
    double ratioC=ratio<lo?lo:(ratio>hi?hi:ratio); double surr1=adv*ratio,surr2=adv*ratioC;
    double dPol; if(surr1<=surr2) dPol=adv*ratio; else { double cg=(lo<ratio&&ratio<hi)?1.0:0.0; dPol=adv*cg*ratio; }
    double vnew=val[t], dvloss_dv;
    if(vfClip>0.0){ double dd=vnew-vold; double vclip=vold+(dd<-vfClip?-vfClip:(dd>vfClip?vfClip:dd)); double du=(vnew-ret)*(vnew-ret),cc=(vclip-ret)*(vclip-ret); if(du>=cc) dvloss_dv=(vnew-ret); else if(dd>-vfClip&&dd<vfClip) dvloss_dv=(vnew-ret); else dvloss_dv=0.0;} else dvloss_dv=(vnew-ret);
    double dvalue=-vfCoef*dvloss_dv;
    g[P+t]=logpA; g[P+T+t]=vnew;                 /* new_logp (taken action) + new_value at t */
    const double* hf=hfin+t*H;
    for(size_t j=0;j<H;j++) dhf[j]=0.0;
    for(size_t k=0;k<A;k++){ double dl=dPol*(((k==a)?1.0:0.0)-pk[k])+entCoef*pk[k]*(pout-logits[k]);
      double* gw=gWDec+k*H; const double* w=wDec+k*H; for(size_t j=0;j<H;j++){ gw[j]+=dl*hf[j]; dhf[j]+=dl*w[j]; } gBDec[k]+=dl; }
    for(size_t j=0;j<H;j++){ gWVal[j]+=dvalue*hf[j]; dhf[j]+=dvalue*wVal[j]; } gBVal[0]+=dvalue;
    for(size_t j=0;j<H;j++) dhn[j]=dhf[j];
    for(size_t ll=numLayers; ll-->0;){
      size_t l=ll; size_t base=(t*numLayers+l)*H; const double* Wl=layersBase+l*layerSz; double* Wg=gLayers+l*layerSz;
      for(size_t j=0;j<H;j++) dhin[j]=0.0;
      for(size_t j=0;j<H;j++){
        double hidj=hidA[base+j], gatej=gateA[base+j], projj=projA[base+j], oj=oA[base+j], prevj=prevS[base+j], hinj=hin[base+j];
        double zj=lstm_sig(gatej), hgj=lstm_sig(projj);
        double gj=(hidj>=0.0)?(hidj+0.5):lstm_sig(hidj);
        double dhnj=dhn[j];
        double dhg=dhnj*(oj-hinj);
        double do_=dhnj*hgj + dOnext[l*H+j];
        dhin[j]+=dhnj*(1.0-hgj);
        double dz=do_*(gj-prevj);
        dprev[l*H+j]=do_*(1.0-zj);
        double dg=do_*zj;
        double dgate=dz*zj*(1.0-zj);
        double dproj=dhg*hgj*(1.0-hgj);
        double sg; if(hidj>=0.0) sg=1.0; else { double s=lstm_sig(hidj); sg=s*(1.0-s); }
        dyh[j]=dg*sg; dyg[j]=dgate; dyp[j]=dproj;
      }
      for(size_t j=0;j<H;j++){
        double dh_=dyh[j]; double* gw=Wg+j*H; const double* w=Wl+j*H; for(size_t k=0;k<H;k++){ gw[k]+=dh_*hin[base+k]; dhin[k]+=dh_*w[k]; }
        double dga=dyg[j]; double* gw2=Wg+(H+j)*H; const double* w2=Wl+(H+j)*H; for(size_t k=0;k<H;k++){ gw2[k]+=dga*hin[base+k]; dhin[k]+=dga*w2[k]; }
        double dpr=dyp[j]; double* gw3=Wg+(2*H+j)*H; const double* w3=Wl+(2*H+j)*H; for(size_t k=0;k<H;k++){ gw3[k]+=dpr*hin[base+k]; dhin[k]+=dpr*w3[k]; }
      }
      for(size_t j=0;j<H;j++) dhn[j]=dhin[j];
    }
    for(size_t i=0;i<H;i++){ gBEnc[i]+=dhn[i]; double* gw=gWEnc+i*obsSize; for(size_t k=0;k<obsSize;k++) gw[k]+=dhn[i]*obs[k]; }
    if(t>0){ double gate_=(termA[t-1]==0.0)?1.0:0.0; for(size_t x=0;x<numLayers*H;x++) dOnext[x]=dprev[x]*gate_; }
  }
  free(hin);free(prevS);free(hidA);free(gateA);free(projA);free(oA);free(hfin);free(lg);free(val);
  free(st);free(hb);free(yb);free(hnew);
  free(dOnext);free(dprev);free(dhn);free(dhin);free(dhf);free(dyh);free(dyg);free(dyp);free(pk);
  lean_dec(params);lean_dec(obsSeq);lean_dec(acts);lean_dec(advs);lean_dec(rets);lean_dec(oldlps);lean_dec(terms);lean_dec(oldvals);
  return grad;
}

/* ---- Batched categorical action sampler for the rollout -------------------------------------------
   Replaces the per-env Lean glue (softmax + sampleCat + logp + value extraction — 4 small Array allocs
   per env per timestep) with ONE C call over the whole N×O batch. `Yb` is N×O row-major (O=A+1: A policy
   logits then the value). `rng` is the starting splitmix64 STATE s; env n draws word = hash(s+(n+1)·G)
   (G = golden ratio) — exactly the per-env `rngNext` stream — so actions/logps/values are BIT-IDENTICAL
   to the Lean rollout, and the caller advances rng by N·G (O(1), since splitmix64's state is just
   s + steps·G). Returns [actions(N); logps(N); values(N)] as f64 (size 3N). Sampling matches
   softmax+sampleCat op-for-op: m=max, z=Σexp(l−m), probs[k]=exp(l−m)/z, cumulative acc, first k with
   u<acc (else A−1); logp=log(probs[a]); value=row[A]. (Behavior-preserving for deterministic-reset envs;
   for envs whose reset draws rng, the sample/reset interleaving differs but each draw is still a valid
   categorical sample — learning is unchanged.) */
static inline uint64_t sm64_hash(uint64_t s){
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z = z ^ (z >> 31);
  return z;
}
LEAN_EXPORT lean_obj_res lean_ffi_sample_actions_batch(
    lean_obj_arg Yba, size_t N, size_t A, size_t O, uint64_t rng){
  const double* Yb = lean_float_array_cptr(Yba);
  lean_object* Oo = lean_alloc_sarray(sizeof(double), 3*N, 3*N);
  double* out = lean_float_array_cptr(Oo);
  const uint64_t G = 0x9E3779B97F4A7C15ULL;
  for(size_t n=0;n<N;n++){
    const double* row = Yb + (size_t)n*O;
    double m = row[0];
    for(size_t k=1;k<A;k++) if(row[k] > m) m = row[k];
    double z = 0.0;
    for(size_t k=0;k<A;k++) z += exp(row[k] - m);
    uint64_t word = sm64_hash(rng + (uint64_t)(n+1)*G);
    double u = (double)(word >> 11) / 9007199254740992.0;      /* uniform01: top 53 bits / 2^53 */
    double acc = 0.0; size_t a = A - 1;
    for(size_t k=0;k<A;k++){ acc += exp(row[k]-m)/z; if(u < acc){ a = k; break; } }
    out[n]       = (double)a;
    out[N+n]     = log(exp(row[a]-m)/z);                        /* log(probs[a]) */
    out[2*N+n]   = row[A];                                      /* value head */
  }
  lean_dec(Yba);
  return Oo;
}

/* ---- Minibatch gather (SoA trajectory) ------------------------------------------------------------
   The SoA rollout keeps obs UNBOXED in a flat env-major column (row e·T+s), filled per timestep by
   `scatter_obs` from `xb` (the N×D batch already built for the forward). `gather_minibatch` then copies
   the shuffled minibatch rows out of the SoA columns into the contiguous `mb*` buffers the step kernels
   want — all in C, replacing the per-index Lean push loops (obs was a boxed `Array Float`). */

/* obsCol[(e·T+s)·D + j] = xb[e·D + j] for timestep s. In place: obsCol is owned and threaded linearly by
   the caller (uniquely referenced), so the buffer is mutated and the same object returned. */
LEAN_EXPORT lean_obj_res lean_ffi_scatter_obs(lean_obj_arg obsColA, lean_obj_arg xbA,
    size_t N, size_t D, size_t T, size_t s){
  double* obsCol = lean_float_array_cptr(obsColA);
  const double* xb = lean_float_array_cptr(xbA);
  for(size_t e=0;e<N;e++){
    double* dst = obsCol + (e*T+s)*D; const double* src = xb + e*D;
    for(size_t j=0;j<D;j++) dst[j] = src[j];
  }
  lean_dec(xbA);
  return obsColA;
}

/* GAE (truncated, no bootstrap) in native C — the MLP plugin trainer's buildBatchSoA, bit-identical:
   per env e (segment of T rows, row = e·T+t), backward over t from T-2 to 0:
     nnt = terms[t]?0:1;  r = clamp(rews[t],-1,1);  delta = r + gamma·vals[t+1]·nnt - vals[t];
     A = delta + gamma·lam·A·nnt;  adv[t] = A;   (adv[T-1] stays 0);  returns[t] = adv[t] + vals[t].
   O(N·T) native replaces the ~35ms/update boxed-Lean scan (the largest chunk of rollout+GAE after the
   native rollout driver). Returns [adv(N·T); returns(N·T)]. */
LEAN_EXPORT lean_obj_res lean_ffi_gae_soa(lean_obj_arg valsA, lean_obj_arg rewsA, lean_obj_arg termsA,
    size_t N, size_t T, double gamma, double lam){
  const double* vals = lean_float_array_cptr(valsA);
  const double* rews = lean_float_array_cptr(rewsA);
  const double* terms = lean_float_array_cptr(termsA);
  size_t NT = N*T;
  lean_object* outO = lean_alloc_sarray(sizeof(double), 2*NT, 2*NT);
  double* out = lean_float_array_cptr(outO);
  double* adv = out; double* ret = out + NT;
  for(size_t e=0;e<N;e++){
    size_t base = e*T;
    if(T>=1) adv[base + T-1] = 0.0;
    double lastA = 0.0;
    for(size_t i=0;i+1<T;i++){                    /* s = T-2, T-3, …, 0 */
      size_t s = T-2-i, t = base+s;
      double nnt = (terms[t]!=0.0) ? 0.0 : 1.0;
      double rr = rews[t]; double r = rr>1.0?1.0:(rr<-1.0?-1.0:rr);
      double delta = r + gamma*vals[t+1]*nnt - vals[t];
      lastA = delta + gamma*lam*lastA*nnt;
      adv[t] = lastA;
    }
  }
  for(size_t t=0;t<NT;t++) ret[t] = adv[t] + vals[t];
  lean_dec(valsA); lean_dec(rewsA); lean_dec(termsA);
  return outO;
}

/* GAE WITH a bootstrap value + advantage batch-normalization, in native C — the LSTM plugin trainer's
   `computeGAEBoot` + adv-normalize, all on TIME-MAJOR rollout columns (row = t·N+n). Per env n, backward
   over t = T-1 … 0:  nnt = term[t]>0.5?0:1;  vNext = (t+1<T)? val[t+1] : bootV[n];
   delta = rew[t] + gamma·vNext·nnt − val[t];  A = delta + gamma·lam·nnt·A;  adv = A;  ret = A + val[t].
   Then advantages are normalized over the whole N·T (mean/std, +1e-8). Identical recurrence + norm form
   to the old boxed-`Array Float` Lean loop (which was the biggest host cost after the native rollout —
   ~50ms/update of per-element boxing); returns [adv(N·T); ret(N·T)] time-major, ready for the BPTT. */
LEAN_EXPORT lean_obj_res lean_ffi_lstm_gae_boot_norm(
    lean_obj_arg valsA, lean_obj_arg rewsA, lean_obj_arg termsA, lean_obj_arg bootA,
    size_t N, size_t T, double gamma, double lam){
  const double* vals = lean_float_array_cptr(valsA);
  const double* rews = lean_float_array_cptr(rewsA);
  const double* terms = lean_float_array_cptr(termsA);
  const double* bootV = lean_float_array_cptr(bootA);
  size_t NT = N*T;
  lean_object* outO = lean_alloc_sarray(sizeof(double), 2*NT, 2*NT);
  double* out = lean_float_array_cptr(outO);
  double* adv = out; double* ret = out + NT;
  for(size_t n=0;n<N;n++){
    double lastA = 0.0;
    for(size_t i=0;i<T;i++){
      size_t t = T-1-i, idx = t*N+n;
      double nnt = (terms[idx] > 0.5) ? 0.0 : 1.0;
      double vNext = (t+1<T) ? vals[(t+1)*N+n] : bootV[n];
      double delta = rews[idx] + gamma*vNext*nnt - vals[idx];
      lastA = delta + gamma*lam*nnt*lastA;
      adv[idx] = lastA; ret[idx] = lastA + vals[idx];
    }
  }
  size_t den = NT>0?NT:1;
  double sum=0.0; for(size_t i=0;i<NT;i++) sum+=adv[i];
  double mean=sum/(double)den;
  double vsum=0.0; for(size_t i=0;i<NT;i++){ double d=adv[i]-mean; vsum+=d*d; }
  double inv=1.0/(sqrt(vsum/(double)den)+1e-8);
  for(size_t i=0;i<NT;i++) adv[i]=(adv[i]-mean)*inv;
  lean_dec(valsA); lean_dec(rewsA); lean_dec(termsA); lean_dec(bootA);
  return outO;
}

/* Flat per-epoch shuffle in C — the bit-exact twin of `epochs ×` Puffer.RL.VecTrain.shuffleIdx, replacing
   the interpreted-Lean `permFlat` loop (epochs·NT Array.push/update, the biggest host-side cost). Returns
   the f64-encoded permutation `perm[epochs·NT]`: epoch e's shuffle is perm[e·NT … (e+1)·NT). Fisher–Yates
   (backward) with splitmix64 `rngNext` (Puffer.RL.Train). The caller advances its own rng by epochs·NT·G
   (rngNext's state is s+=G/call, independent of the drawn word — so the post-shuffle rng is exactly that). */
LEAN_EXPORT lean_obj_res lean_ffi_shuffle_perm(size_t NT, size_t epochs, uint64_t rng){
  size_t total = epochs*NT;
  lean_object* Oo = lean_alloc_sarray(sizeof(double), total, total);
  double* out = lean_float_array_cptr(Oo);
  int* a = (int*)malloc(sizeof(int)*NT);
  if(!a){ for(size_t i=0;i<total;i++) out[i]=0.0; return Oo; }
  uint64_t s = rng;
  for(size_t e=0;e<epochs;e++){
    for(size_t i=0;i<NT;i++) a[i]=(int)i;
    for(size_t i=0;i<NT;i++){
      size_t j = NT-1-i;
      uint64_t sp = s + 0x9E3779B97F4A7C15ULL; s = sp;      /* rngNext: splitmix64 */
      uint64_t z = sp;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      size_t k = (size_t)(z % (uint64_t)(j + 1));
      int tmp = a[j]; a[j] = a[k]; a[k] = tmp;
    }
    double* dst = out + e*NT;
    for(size_t i=0;i<NT;i++) dst[i] = (double)a[i];
  }
  free(a);
  return Oo;
}

/* Gather rows idxs[0..Nmb) out of the SoA columns → (mbObs[Nmb·D], mbAct[Nmb], mbAdv[Nmb], mbRet[Nmb],
   mbOlp[Nmb]). idxs are f64-encoded transition indices. All inputs consumed (Lean re-incs the reused
   columns across minibatches). */
LEAN_EXPORT lean_obj_res lean_ffi_gather_minibatch(
    lean_obj_arg obsColA, lean_obj_arg actionsA, lean_obj_arg advsA, lean_obj_arg retsA,
    lean_obj_arg olpsA, lean_obj_arg idxsA, size_t Nmb, size_t D){
  const double* obsCol  = lean_float_array_cptr(obsColA);
  const double* actions = lean_float_array_cptr(actionsA);
  const double* advs    = lean_float_array_cptr(advsA);
  const double* rets    = lean_float_array_cptr(retsA);
  const double* olps    = lean_float_array_cptr(olpsA);
  const double* idxs    = lean_float_array_cptr(idxsA);
  lean_object* mbObsO = lean_alloc_sarray(sizeof(double), Nmb*D, Nmb*D);
  lean_object* mbActO = lean_alloc_sarray(sizeof(double), Nmb, Nmb);
  lean_object* mbAdvO = lean_alloc_sarray(sizeof(double), Nmb, Nmb);
  lean_object* mbRetO = lean_alloc_sarray(sizeof(double), Nmb, Nmb);
  lean_object* mbOlpO = lean_alloc_sarray(sizeof(double), Nmb, Nmb);
  double* mbObs = lean_float_array_cptr(mbObsO);
  double* mbAct = lean_float_array_cptr(mbActO);
  double* mbAdv = lean_float_array_cptr(mbAdvO);
  double* mbRet = lean_float_array_cptr(mbRetO);
  double* mbOlp = lean_float_array_cptr(mbOlpO);
  for(size_t k=0;k<Nmb;k++){
    size_t idx = (size_t)idxs[k];
    const double* orow = obsCol + idx*D; double* drow = mbObs + k*D;
    for(size_t j=0;j<D;j++) drow[j] = orow[j];
    mbAct[k] = actions[idx];
    mbAdv[k] = advs[idx];
    mbRet[k] = rets[idx];
    mbOlp[k] = olps[idx];
  }
  lean_dec(obsColA); lean_dec(actionsA); lean_dec(advsA); lean_dec(retsA); lean_dec(olpsA); lean_dec(idxsA);
  /* build (mbObs, (mbAct, (mbAdv, (mbRet, mbOlp)))) — Prod.mk is ctor 0, 2 fields */
  lean_object* t4 = lean_alloc_ctor(0, 2, 0); lean_ctor_set(t4, 0, mbRetO); lean_ctor_set(t4, 1, mbOlpO);
  lean_object* t3 = lean_alloc_ctor(0, 2, 0); lean_ctor_set(t3, 0, mbAdvO); lean_ctor_set(t3, 1, t4);
  lean_object* t2 = lean_alloc_ctor(0, 2, 0); lean_ctor_set(t2, 0, mbActO); lean_ctor_set(t2, 1, t3);
  lean_object* t1 = lean_alloc_ctor(0, 2, 0); lean_ctor_set(t1, 0, mbObsO); lean_ctor_set(t1, 1, t2);
  return t1;
}

/* Prioritized-replay weights + sampling in one C pass — replaces the two interpreted-Lean O(N·T) / O(mbSegs·N)
   host loops (prioW over ALL N segments every minibatch, then weightedSampleReplace's linear-scan search),
   which dominated the trainer at scale (~half of "other"). Bit-identical: the splitmix64 stream advances
   deterministically (state s_i = rng + i·GOLD), the float ops match the Lean order exactly, and the cumulative
   search is a lower_bound (first cum[e] ≥ u) equivalent to the Lean first-hit scan (cum is strictly increasing
   since every prob ≥ 1e-6/denom > 0). The caller advances its rng by mbSegs·GOLD afterward.
     prioW[e]   = exp(prioAlpha·log(Σ_t|adv[e·T+t]| + 1e-12))
     prioProbs  = (prioW + 1e-6)/(ΣprioW + 1e-6);  cum = prefix-sum(prioProbs);  total = Σ prioProbs
     draw i:  u = uniform01(splitmix64(rng + (i+1)·GOLD))·total;  idx = first e with cum[e] ≥ u
     mbPrio[i]  = exp(−annealBeta·log(N·prioProbs[idx] + 1e-12))
   Returns (sampledIdxF[mbSegs] (as doubles), mbPrio[mbSegs]). */
LEAN_EXPORT lean_obj_res lean_ffi_prio_sample(
    lean_obj_arg advL1A, size_t N, size_t T, size_t mbSegs,
    double prioAlpha, double annealBeta, uint64_t rng){
  (void)T;
  const double* advL1 = lean_float_array_cptr(advL1A);   /* Σ_t|adv[e,t]| per segment (computed on-device now) */
  double* prioW=(double*)malloc(N*sizeof(double)); double* prioProbs=(double*)malloc(N*sizeof(double));
  double* cum=(double*)malloc(N*sizeof(double));
  double sumW=0.0;
  for(size_t e=0;e<N;e++){ double w=exp(prioAlpha*log(advL1[e]+1e-12)); prioW[e]=w; sumW+=w; }
  double denom=sumW+1e-6, acc=0.0;
  for(size_t e=0;e<N;e++){ double p=(prioW[e]+1e-6)/denom; prioProbs[e]=p; acc+=p; cum[e]=acc; }
  double total=acc;
  lean_object* idxO=lean_alloc_sarray(sizeof(double),mbSegs,mbSegs);
  lean_object* mpO =lean_alloc_sarray(sizeof(double),mbSegs,mbSegs);
  double* idxArr=lean_float_array_cptr(idxO); double* mpArr=lean_float_array_cptr(mpO);
  const uint64_t GOLD=0x9E3779B97F4A7C15ULL;
  for(size_t i=0;i<mbSegs;i++){
    uint64_t z=rng+(uint64_t)(i+1)*GOLD;                      /* splitmix64 state for the (i+1)-th draw */
    z=(z ^ (z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z ^ (z>>27))*0x94D049BB133111EBULL;
    z=z ^ (z>>31);
    double u=(double)(z>>11)/9007199254740992.0*total;
    size_t L=0,R=N; while(L<R){ size_t mid=(L+R)>>1; if(cum[mid]>=u) R=mid; else L=mid+1; }  /* first cum≥u */
    size_t idx=(L<N)?L:(N-1);
    idxArr[i]=(double)idx;
    mpArr[i]=exp((-annealBeta)*log((double)N*prioProbs[idx]+1e-12));
  }
  free(prioW); free(prioProbs); free(cum);
  lean_dec(advL1A);
  lean_object* pr=lean_alloc_ctor(0,2,0); lean_ctor_set(pr,0,idxO); lean_ctor_set(pr,1,mpO);
  return pr;
}

/* Full MinGRU minibatch gather in one C pass — obs + the 6 scalar buffers the BPTT kernel takes, replacing
   the pobs C call plus 6 interpreted-Lean `(Array.range (T·Bmb)).map` loops (each allocated an Array.range,
   a mapped Array, and a FloatArray). For the Bmb sampled segments (segIdx[bi] = env e), timestep-major
   output (index os = t·Bmb + bi), reading the flat SoA columns at si = e·T + t:
     pobs[t·Bmb·D + bi·D + j] = obsCol[e·T·D + t·D + j]
     pact = (double)(unsigned long long)actCol[si]      (= Float.ofNat (actCol.toUInt64.toNat))
     padv = mbPrio[bi]·(advFlat[si] − advMean)/(advStd + 1e-8)
     pret = advFlat[si] + valueBuf[si];  pold = logpCol[si];  pov = valueBuf[si]
     pterm = termCol[si] > 0.5 ? 1 : 0
   Same values, same op order as the Lean loops ⇒ bit-identical. Returns (pobs,pact,padv,pret,pold,pterm,pov). */
LEAN_EXPORT lean_obj_res lean_ffi_gather_seq_minibatch(
    lean_obj_arg obsColA, lean_obj_arg actColA, lean_obj_arg logpColA, lean_obj_arg termColA,
    lean_obj_arg advFlatA, lean_obj_arg valueBufA, lean_obj_arg segIdxA, lean_obj_arg mbPrioA,
    size_t T, size_t Bmb, size_t D, size_t N){
  (void)N;
  const double* obsCol=lean_float_array_cptr(obsColA); const double* actCol=lean_float_array_cptr(actColA);
  const double* logpCol=lean_float_array_cptr(logpColA); const double* termCol=lean_float_array_cptr(termColA);
  const double* advFlat=lean_float_array_cptr(advFlatA); const double* valueBuf=lean_float_array_cptr(valueBufA);
  const double* segIdx=lean_float_array_cptr(segIdxA); const double* mbPrio=lean_float_array_cptr(mbPrioA);
  size_t nb=T*Bmb, osz=nb*D;
  /* advMean/advStd over the sampled minibatch, computed here (was two interpreted-Lean O(Bmb·T) loops).
     Same (bi outer, t inner) order and nAv=max(Bmb·T,1) as the Lean advVals build ⇒ bit-identical. */
  (void)obsCol; (void)osz; (void)D;   /* obs is now device-resident (BPTT gathers g_dMGObsTraj); no host pobs */
  double navd=(double)(nb>0?nb:1), sum=0.0;
  for(size_t bi=0; bi<Bmb; bi++){ size_t e=(size_t)segIdx[bi]; const double* ab=advFlat+e*T;
    for(size_t t=0;t<T;t++) sum += ab[t]; }
  double advMean=sum/navd, sq=0.0;
  for(size_t bi=0; bi<Bmb; bi++){ size_t e=(size_t)segIdx[bi]; const double* ab=advFlat+e*T;
    for(size_t t=0;t<T;t++){ double d=ab[t]-advMean; sq += d*d; } }
  double advStd=sqrt(sq/navd);
  /* ONE packed output [6·nb] = [act|adv|ret|old|term|ov] (was 6 lean_alloc_sarray/call — the per-call
     allocation of six fresh buffers dominated this pass; the BPTT reads the six as contiguous slices). */
  lean_object* pO=lean_alloc_sarray(sizeof(double),6*nb,6*nb); double* pk=lean_float_array_cptr(pO);
  double* pact=pk; double* padv=pk+nb; double* pret=pk+2*nb; double* pold=pk+3*nb; double* pterm=pk+4*nb; double* pov=pk+5*nb;
  double stde = advStd + 1e-8;
  for(size_t bi=0; bi<Bmb; bi++){
    size_t e=(size_t)segIdx[bi]; double mp=mbPrio[bi];
    for(size_t t=0; t<T; t++){
      size_t si=e*T+t, os=t*Bmb+bi;
      double a=advFlat[si], v=valueBuf[si];
      pact[os]=(double)(unsigned long long)actCol[si];
      padv[os]=mp*(a-advMean)/stde;
      pret[os]=a+v; pold[os]=logpCol[si]; pov[os]=v;
      pterm[os]=termCol[si]>0.5?1.0:0.0;
    }
  }
  lean_dec(obsColA);lean_dec(actColA);lean_dec(logpColA);lean_dec(termColA);
  lean_dec(advFlatA);lean_dec(valueBufA);lean_dec(segIdxA);lean_dec(mbPrioA);
  return pO;
}
