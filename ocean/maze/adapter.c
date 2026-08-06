/* Adapter: PufferLib Ocean maze -> puffer-lean plugin ABI. Runs the REAL ocean/maze/maze.h c_reset/c_step. */
#include "maze.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Grid
#define OBS_T unsigned char
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }

/* maze levels are generated in binding.c's my_vec_init, not my_init, and c_reset dereferences
   env->levels[idx]. Build the level bank once and share it across all env copies (mirrors the
   original design where levels are shared) to avoid regenerating/OOMing per copy. */
static State* g_levels = NULL;
static int g_num_levels = 0;
static void ensure_levels(int num_maps,int map_size){
  if(g_levels) return;
  if(num_maps < 1) num_maps = 1;
  int max_size = MAX_SIZE;
  State* levels = (State*)calloc(num_maps,sizeof(State));
  unsigned int map_rng = 42;
  for(int i=0;i<num_maps;i++){
    int sz = map_size;
    if(map_size == -1) sz = 5 + (rand_r(&map_rng) % (max_size - 5));
    if(sz % 2 == 0) sz -= 1;
    State* level = &levels[i];
    level->width = sz;
    level->height = sz;
    float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
    create_maze_level(level, difficulty, i);
  }
  g_levels = levels;
  g_num_levels = num_maps;
}

static void ocean_setup(Grid* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  int num_maps = cfg_int(cfg,"num-maps",8192);
  int map_size = cfg_int(cfg,"map-size",-1);
  ensure_levels(num_maps,map_size);
  e->levels = g_levels;
  e->num_levels = g_num_levels;
  e->renderer = NULL;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(unsigned char*)calloc(WINDOW*WINDOW,sizeof(unsigned char));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(Grid* e){ (void)e; return WINDOW*WINDOW; }
static int  ocean_numactions(Grid* e){ (void)e; return 5; }
static int  ocean_maxsteps(Grid* e){ (void)e; return 2*MAX_SIZE*MAX_SIZE; }
static void ocean_teardown(Grid* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
