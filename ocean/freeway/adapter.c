/* Adapter: PufferLib Ocean freeway -> puffer-lean plugin ABI. Runs the REAL ocean/freeway/freeway.h c_reset/c_step. */
#include "freeway.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Freeway
#define OBS_T float
/* freeway is SINGLE-AGENT: binding.c my_init hardcodes num_agents=1, allocate() sizes obs/actions/rewards/
   terminals for ONE agent, and c_step only touches [0] (the human player is scripted via human_actions).
   PufferLib's num_agents=4096 is its VECTORIZATION count → maps to --num-envs here, not in-env agents.
   So NO OCEAN_NAGENTS override (defaults to 1). actions/terminals are float* -> ACT_T/TERM_T float (default). */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(ENV_T* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;                                   /* single-agent (matches binding.c my_init) */
  e->frameskip=cfg_int(cfg,"frameskip",4);
  e->width=cfg_int(cfg,"width",1216);
  e->height=cfg_int(cfg,"height",720);
  e->player_width=cfg_int(cfg,"player-width",64);
  e->player_height=cfg_int(cfg,"player-height",64);
  e->car_width=cfg_int(cfg,"car-width",64);
  e->car_height=cfg_int(cfg,"car-height",40);
  e->lane_size=cfg_int(cfg,"lane-size",64);
  e->difficulty=cfg_int(cfg,"difficulty",0);
  e->level=cfg_int(cfg,"level",-1);
  e->enable_human_player=cfg_int(cfg,"enable-human-player",0);
  e->env_randomization=cfg_int(cfg,"env-randomization",1);
  e->use_dense_rewards=cfg_int(cfg,"use-dense-rewards",1);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);   /* calls init(e): sizes the four buffers (single-agent) and sets derived state */
}
static int  ocean_obsdim(ENV_T* e){ (void)e; return 4 + NUM_LANES*MAX_ENEMIES_PER_LANE; }  /* = 34; matches allocate() */
static int  ocean_numactions(ENV_T* e){ (void)e; return 3; }
static int  ocean_maxsteps(ENV_T* e){ return (int)(GAME_LENGTH / TICK_RATE / (float)e->frameskip) + 1; }
static void ocean_teardown(ENV_T* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(up_action_frac) X(hits) X(n)
#include "../../ffi/ocean_adapter.h"
