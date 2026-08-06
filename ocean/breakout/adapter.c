/* Adapter: PufferLib Ocean `breakout` → puffer-lean plugin ABI. Runs the REAL breakout.h physics
 * (float obs 10+num_bricks, sinf/cosf paddle bounce, rand_r fire). Config defaults from breakout.ini;
 * `frameskip` overridable. */
#include "breakout.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Breakout
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Breakout* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->frameskip=cfg_int(cfg,"frameskip",4);
  e->width=576; e->height=330;
  e->initial_paddle_width=62; e->paddle_width=62; e->paddle_height=8;
  e->ball_width=32; e->ball_height=32; e->brick_width=32; e->brick_height=12;
  e->brick_rows=6; e->brick_cols=18;
  e->initial_ball_speed=256; e->max_ball_speed=448; e->paddle_speed=620;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);   /* init() + mallocs observations/actions/rewards/terminals */
}
static int ocean_obsdim(Breakout* e){ return 10 + e->num_bricks; }
static int ocean_numactions(Breakout* e){ (void)e; return 3; }
static int ocean_maxsteps(Breakout* e){ (void)e; return 100000; }
static void ocean_teardown(Breakout* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
