#include "double_pendulum.h"
#include <string.h>
#include <stdint.h>
#define ENV_T DoublePendulum
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(DoublePendulum* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->cart_mass=cfg_flt(cfg,"cart-mass",1.0f); e->gravity=cfg_flt(cfg,"gravity",9.8f);
  /* link geometry/masses were MISSING here: calloc left them 0, so max_y=0 made `height` NaN and
     `fmaxf(NaN,0)` collapsed every reward to exactly 0, while the singular physics went non-finite
     and terminated each step (c_reset zeroes terminals before the adapter reads them) — the env
     reported a permanent 0 reward / 0 terminals. Defaults are upstream config/double_pendulum.ini [env]. */
  e->link1_mass=cfg_flt(cfg,"link1-mass",0.1f); e->link2_mass=cfg_flt(cfg,"link2-mass",0.1f);
  e->link1_length=cfg_flt(cfg,"link1-length",0.5f); e->link2_length=cfg_flt(cfg,"link2-length",0.5f);
  e->force_mag=cfg_flt(cfg,"force-mag",10.0f); e->dt=cfg_flt(cfg,"dt",0.02f);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(DP_OBS_SIZE,sizeof(float)); e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float)); e->terminals=(float*)calloc(1,sizeof(float));
  init(e);
}
static int ocean_obsdim(DoublePendulum* e){ (void)e; return DP_OBS_SIZE; }
static int ocean_numactions(DoublePendulum* e){ (void)e; return 3; }
static int ocean_maxsteps(DoublePendulum* e){ (void)e; return 100000; }
static void ocean_teardown(DoublePendulum* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(x_threshold_termination) X(max_steps_termination) \
  X(hold_time) X(n)
#include "../../ffi/ocean_adapter.h"
