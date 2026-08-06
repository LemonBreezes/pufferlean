#include "tetris.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Tetris
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Tetris* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->n_rows=cfg_int(cfg,"n-rows",20); e->n_cols=cfg_int(cfg,"n-cols",10);
  e->use_deck_obs=cfg_int(cfg,"use-deck-obs",1); e->n_noise_obs=cfg_int(cfg,"n-noise-obs",0);
  e->n_init_garbage=cfg_int(cfg,"n-init-garbage",4);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);
}
static int ocean_obsdim(Tetris* e){ return e->dim_obs; }   /* allocate() computes this from n_rows/n_cols/n_noise_obs (234 at defaults) */
static int ocean_numactions(Tetris* e){ (void)e; return 7; }
static int ocean_maxsteps(Tetris* e){ (void)e; return 100000; }
static void ocean_teardown(Tetris* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_length) X(episode_return) X(lines_deleted) X(avg_combo) X(atn_frac_soft_drop) \
  X(atn_frac_hard_drop) X(atn_frac_rotate) X(atn_frac_hold) X(game_level) X(ticks_per_line) X(n)
#include "../../ffi/ocean_adapter.h"
