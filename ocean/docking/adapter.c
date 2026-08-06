#include "docking.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Docking
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Docking* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->width=cfg_int(cfg,"width",256); e->height=cfg_int(cfg,"height",192); e->max_ticks=cfg_int(cfg,"max-ticks",1024);
  e->max_speed=cfg_flt(cfg,"max-speed",6.0f); e->turn_rate=cfg_flt(cfg,"turn-rate",0.10f);
  e->accel=cfg_flt(cfg,"accel",0.55f); e->drag=cfg_flt(cfg,"drag",0.92f);
  e->dock_radius=cfg_flt(cfg,"dock-radius",18.0f); e->dock_speed_threshold=cfg_flt(cfg,"dock-speed-threshold",0.72f);
  e->dock_heading_threshold=cfg_flt(cfg,"dock-heading-threshold",0.28f);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(DOCKING_OBS_SIZE,sizeof(float)); e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float)); e->terminals=(float*)calloc(1,sizeof(float));
  c_init(e);
}
static int ocean_obsdim(Docking* e){ (void)e; return DOCKING_OBS_SIZE; }
static int ocean_numactions(Docking* e){ (void)e; return 5; }
static int ocean_maxsteps(Docking* e){ return e->max_ticks; }
static void ocean_teardown(Docking* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(success_rate) X(crash_rate) X(timeout_rate) \
  X(final_distance) X(alignment_error) X(n)
#include "../../ffi/ocean_adapter.h"
