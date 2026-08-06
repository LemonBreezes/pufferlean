/* puffer_handle.h — the dlopen'd env plugin handle, shared between the runtime loader
   (ffi/puffer_loader.c) and the native rollout driver (ffi/puffercuda.cu) so the driver can
   call the env's step function directly. This header names NO specific env — it is the
   generic env-agnostic plugin ABI. */
#ifndef PUFFER_HANDLE_H
#define PUFFER_HANDLE_H
#include <stdint.h>

typedef void* (*pe_make)(int, uint64_t, const char*);
typedef void  (*pe_spec)(void*, int*, int*, int*, int*);
typedef void  (*pe_actinfo)(void*, int*, int*);
typedef int   (*pe_iscont)(void*);
typedef void  (*pe_reset)(void*, double*);
typedef void  (*pe_step)(void*, const double*, double*, double*, double*);
/* step only envs [start, start+count) — for persistent-thread env-stepping (optional; NULL on old .so). */
typedef void  (*pe_step_range)(void*, const double*, double*, double*, double*, int, int);
/* f32-obs twin of step_range: obs written as float directly (envs compute obs in float/uint8 natively, so
   this equals the f64 path + downstream (float) cast BIT-FOR-BIT, minus the double intermediate + cast
   pass on the rollout critical path). Optional; NULL on old .so. */
typedef void  (*pe_step_range_f32)(void*, const double*, float*, double*, double*, int, int);
/* reduced-precision obs transport (optional). u8: raw bytes, EXACT for byte-native envs (obs_kind()==1
   ONLY — would truncate float obs). bf16: RNE-rounded, valid for any env (tolerance-class). */
typedef void  (*pe_step_range_u8)(void*, const double*, unsigned char*, double*, double*, int, int);
typedef void  (*pe_step_range_bf16)(void*, const double*, unsigned short*, double*, double*, int, int);
typedef int   (*pe_obs_kind)(void*);
/* OPTIONAL log channel (ffi/puffer_env.h): PufferLib's per-env `Log` (episode_return/score/perf/… —
   whatever THIS env defines), aggregated over the copies and zeroed on read. NULL on a plugin that
   doesn't export it, exactly like pe_iscont. Metrics only — never touches obs/rewards/terminals. */
typedef int         (*pe_log_nfields)(void*);
typedef const char* (*pe_log_name)(void*, int);
typedef double      (*pe_log)(void*, double*, int);
typedef void  (*pe_free)(void*);

typedef struct {
  void* dl; void* env;
  pe_make make; pe_spec spec; pe_actinfo actinfo; pe_iscont iscont; pe_reset reset; pe_step step; pe_free freef;
  pe_step_range step_range;   /* NULL if the plugin predates puffer_env_step_range */
  pe_step_range_f32 step_range_f32;   /* NULL if the plugin predates puffer_env_step_range_f32 */
  pe_step_range_u8 step_range_u8; pe_step_range_bf16 step_range_bf16;   /* optional reduced-precision obs */
  int obsKind;                        /* 0=f32-native, 1=byte-native (u8 transport exact) */
  int N, obsDim, numActions, maxSteps, numAgents;   /* batch rows B = N·numAgents */
  int nHeads, headSizes[16];                          /* action-space structure */
  int isCont;                                         /* 1 = continuous (Gaussian) action space */
  pe_log_name log_name; pe_log envlog;                /* optional log channel (NULL ⇒ unsupported) */
  int logNFields;                                     /* 0 = env exports no log channel */
} Handle;

#endif
