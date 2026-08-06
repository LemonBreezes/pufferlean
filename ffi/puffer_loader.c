/* puffer_loader.c — the runtime env loader compiled into `puffer`.
 *
 * This is the ONLY env-facing code in the trainer, and it knows zero specific envs: it dlopen's
 * libenv_<name>.so by NAME at runtime and calls through the puffer_env.h ABI. So `puffer` never
 * sees an env at compile time — new envs drop into envs/<name>/ and build to their own .so.
 *
 * The env holds hidden mutable state inside the handle, so open/reset/step/close are exposed to
 * Lean as IO actions (properly sequenced, never assumed pure/CSE'd). The spec getters are pure.
 */
#include <lean/lean.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "puffer_handle.h"

/* Search PUFFER_ENV_PATH, then envs/<name>/, ./, .lake/build/lib/ for libenv_<name>.so. */
static void* try_open(const char* name) {
  char path[1024]; void* dl;
  const char* envp = getenv("PUFFER_ENV_PATH");
  if (envp) { snprintf(path,sizeof(path),"%s/libenv_%s.so",envp,name); if((dl=dlopen(path,RTLD_NOW|RTLD_LOCAL))) return dl; }
  snprintf(path,sizeof(path),"ocean/%s/libenv_%s.so",name,name); if((dl=dlopen(path,RTLD_NOW|RTLD_LOCAL))) return dl;
  snprintf(path,sizeof(path),"envs/%s/libenv_%s.so",name,name); if((dl=dlopen(path,RTLD_NOW|RTLD_LOCAL))) return dl;
  snprintf(path,sizeof(path),"./libenv_%s.so",name);            if((dl=dlopen(path,RTLD_NOW|RTLD_LOCAL))) return dl;
  snprintf(path,sizeof(path),".lake/build/lib/libenv_%s.so",name); if((dl=dlopen(path,RTLD_NOW|RTLD_LOCAL))) return dl;
  return NULL;
}

/* IO USize — 0 on failure (env not found / missing symbols). */
LEAN_EXPORT lean_obj_res lean_puffer_env_open(lean_obj_arg namea, size_t N, uint64_t seed,
                                              lean_obj_arg configa, lean_obj_arg w) {
  (void)w;
  const char* name = lean_string_cstr(namea);
  const char* config = lean_string_cstr(configa);
  size_t result = 0;
  void* dl = try_open(name);
  if (dl) {
    Handle* h = (Handle*)malloc(sizeof(Handle));
    h->dl = dl;
    h->make  = (pe_make) dlsym(dl, "puffer_env_make");
    h->spec  = (pe_spec) dlsym(dl, "puffer_env_spec");
    h->reset = (pe_reset)dlsym(dl, "puffer_env_reset");
    h->step  = (pe_step) dlsym(dl, "puffer_env_step");
    h->freef = (pe_free) dlsym(dl, "puffer_env_free");
    h->step_range = (pe_step_range) dlsym(dl, "puffer_env_step_range");   /* optional (NULL on old .so) */
    h->step_range_f32 = (pe_step_range_f32) dlsym(dl, "puffer_env_step_range_f32");   /* optional */
    h->step_range_u8 = (pe_step_range_u8) dlsym(dl, "puffer_env_step_range_u8");      /* optional */
    h->step_range_bf16 = (pe_step_range_bf16) dlsym(dl, "puffer_env_step_range_bf16");/* optional */
    h->obsKind = 0;
    if (h->make && h->spec && h->reset && h->step && h->freef) {
      h->N = (int)N;
      h->numAgents = 1;
      h->env = h->make((int)N, seed, config);
      h->spec(h->env, &h->obsDim, &h->numActions, &h->maxSteps, &h->numAgents);
      if (h->numAgents < 1) h->numAgents = 1;
      h->nHeads = 1; h->headSizes[0] = h->numActions;   /* default single discrete head */
      h->actinfo = (pe_actinfo)dlsym(dl, "puffer_env_actinfo");
      if (h->actinfo) { h->actinfo(h->env, &h->nHeads, h->headSizes); if (h->nHeads < 1) h->nHeads = 1; }
      h->isCont = 0;                                     /* default discrete */
      h->iscont = (pe_iscont)dlsym(dl, "puffer_env_iscont");
      if (h->iscont) h->isCont = h->iscont(h->env) ? 1 : 0;
      { pe_obs_kind okf = (pe_obs_kind)dlsym(dl, "puffer_env_obs_kind");
        if (okf) h->obsKind = okf(h->env); }
      /* Optional log channel — PufferLib's per-env `Log` (see puffer_env.h). All three symbols must be
         present or the env simply has no log channel, same "absent ⇒ default" rule as puffer_env_iscont. */
      h->logNFields = 0; h->envlog = NULL; h->log_name = NULL;
      { pe_log_nfields lnf = (pe_log_nfields)dlsym(dl, "puffer_env_log_nfields");
        h->log_name = (pe_log_name)dlsym(dl, "puffer_env_log_name");
        h->envlog   = (pe_log)     dlsym(dl, "puffer_env_log");
        if (lnf && h->log_name && h->envlog) { int k = lnf(h->env); h->logNFields = k > 0 ? k : 0; }
        if (h->logNFields == 0) { h->log_name = NULL; h->envlog = NULL; } }
      result = (size_t)h;
    } else { dlclose(dl); free(h); }
  }
  lean_dec(namea); lean_dec(configa);
  return lean_io_result_mk_ok(lean_box_usize(result));
}

/* Pure spec getters (fixed after open). */
LEAN_EXPORT size_t lean_puffer_env_obsdim(size_t h)     { return (size_t)((Handle*)h)->obsDim; }
LEAN_EXPORT size_t lean_puffer_env_numactions(size_t h) { return (size_t)((Handle*)h)->numActions; }
LEAN_EXPORT size_t lean_puffer_env_maxsteps(size_t h)   { return (size_t)((Handle*)h)->maxSteps; }
LEAN_EXPORT size_t lean_puffer_env_numagents(size_t h)  { return (size_t)((Handle*)h)->numAgents; }
LEAN_EXPORT size_t lean_puffer_env_nheads(size_t h)     { return (size_t)((Handle*)h)->nHeads; }
LEAN_EXPORT size_t lean_puffer_env_headsize(size_t h, size_t i) { return (size_t)((Handle*)h)->headSizes[i]; }
LEAN_EXPORT size_t lean_puffer_env_iscont(size_t h)     { return (size_t)((Handle*)h)->isCont; }
/* Number of named fields in the env's own `Log`; 0 = the plugin has no log channel. */
LEAN_EXPORT size_t lean_puffer_env_log_nfields(size_t h) { return (size_t)((Handle*)h)->logNFields; }

/* Name of log field `i` (fixed after open) — "" when out of range or unsupported. */
LEAN_EXPORT lean_obj_res lean_puffer_env_log_name(size_t hh, size_t i) {
  Handle* h = (Handle*)hh;
  const char* s = (h->log_name && (int)i < h->logNFields) ? h->log_name(h->env, (int)i) : "";
  return lean_mk_string(s ? s : "");
}

/* IO FloatArray — the env's aggregated + zeroed log, one entry per named field (paired with
   lean_puffer_env_log_name). EMPTY array when the channel is unsupported, or when no episode finished
   since the previous call (PufferLib's `n == 0` early-out — the logs are then left untouched). */
LEAN_EXPORT lean_obj_res lean_puffer_env_log(size_t hh, lean_obj_arg w) {
  (void)w;
  Handle* h = (Handle*)hh;
  int k = h->envlog ? h->logNFields : 0;
  lean_object* Oo = lean_alloc_sarray(sizeof(double), k, k > 0 ? k : 1);
  if (k > 0 && h->envlog(h->env, lean_float_array_cptr(Oo), k) == 0.0)
    lean_sarray_set_size(Oo, 0);        /* no episode completed — report "nothing to say" */
  return lean_io_result_mk_ok(Oo);
}

/* IO FloatArray — obs[B·obsDim], B = N·numAgents (each agent is a row). */
LEAN_EXPORT lean_obj_res lean_puffer_env_reset(size_t hh, lean_obj_arg w) {
  (void)w;
  Handle* h = (Handle*)hh; long B = (long)h->N * h->numAgents, n = B * h->obsDim;
  lean_object* Oo = lean_alloc_sarray(sizeof(double), n, n);
  h->reset(h->env, lean_float_array_cptr(Oo));
  return lean_io_result_mk_ok(Oo);
}

/* IO FloatArray — [obs(B·obsDim); rewards(B); terminals(B)], B = N·numAgents. `actions` is B. */
LEAN_EXPORT lean_obj_res lean_puffer_env_step(size_t hh, lean_obj_arg actsa, lean_obj_arg w) {
  (void)w;
  Handle* h = (Handle*)hh; long D = h->obsDim, B = (long)h->N * h->numAgents, total = B*D + 2*B;
  const double* acts = lean_float_array_cptr(actsa);
  lean_object* Oo = lean_alloc_sarray(sizeof(double), total, total);
  double* out = lean_float_array_cptr(Oo);
  h->step(h->env, acts, out, out + B*D, out + B*D + B);
  lean_dec(actsa);
  return lean_io_result_mk_ok(Oo);
}

/* IO Unit. */
LEAN_EXPORT lean_obj_res lean_puffer_env_close(size_t hh, lean_obj_arg w) {
  (void)w;
  Handle* h = (Handle*)hh;
  if (h) { if (h->freef && h->env) h->freef(h->env); if (h->dl) dlclose(h->dl); free(h); }
  return lean_io_result_mk_ok(lean_box(0));
}
