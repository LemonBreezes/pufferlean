/* Adapter: PufferLib Ocean slimevolley → pufferlean plugin ABI. MULTI-DISCRETE: 3 binary heads {2,2,2}. */
#include "slimevolley.h"
#include <string.h>
#include <stdint.h>
#define ENV_T SlimeVolley
#define OBS_T float
#define OCEAN_NHEADS 3
#define OCEAN_HEADSIZES {2,2,2}
static void ocean_setup(SlimeVolley* e,uint64_t seed,int idx,const char* cfg){ (void)cfg;
  e->num_agents=1;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(12,sizeof(float));
  e->actions=(float*)calloc(3,sizeof(float));    /* num_agents × NHEADS */
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
  init(e);
}
static int  ocean_obsdim(SlimeVolley* e){ (void)e; return 12; }
static int  ocean_numactions(SlimeVolley* e){ (void)e; return 6; }   /* Σheadsizes (total logits) */
static int  ocean_maxsteps(SlimeVolley* e){ (void)e; return 100000; }
static void ocean_teardown(SlimeVolley* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
