/* Adapter: PufferLib Ocean connect4 -> puffer-lean plugin ABI. Runs the REAL ocean/connect4/connect4.h c_reset/c_step. */
#include "connect4.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Connect4
#define OBS_T float
/* actions/rewards/terminals are all float* -> ACT_T/REW_T/TERM_T default float; single-agent (num_agents=1) so OMIT OCEAN_NAGENTS. */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(ENV_T* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=cfg_int(cfg,"num-agents",1);
  e->player_pieces=(uint64_t)cfg_int(cfg,"player-pieces",0);
  e->env_pieces=(uint64_t)cfg_int(cfg,"env-pieces",0);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate_cconnect4(e);
  init(e);
}
static int  ocean_obsdim(ENV_T* e){ (void)e; return 42; }
static int  ocean_numactions(ENV_T* e){ (void)e; return 7; }
static int  ocean_maxsteps(ENV_T* e){ (void)e; return 42; }
static void ocean_teardown(ENV_T* e){ free_allocated_cconnect4(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
