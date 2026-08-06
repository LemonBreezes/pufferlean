/* Adapter: PufferLib Ocean tower_climb -> pufferlean plugin ABI. Runs the REAL ocean/tower_climb/tower_climb.h c_reset/c_step. */
#include "tower_climb.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>   /* getenv for the maps.bin path override */
#define ENV_T CTowerClimb
#define OBS_T unsigned char
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
/* Upstream's my_vec_init loads the level bank ONCE from resources/tower_climb/maps.bin and shares
   the Level[]/PuzzleState[] arrays across every env copy (binding.c:21-49). We never vendored that
   file, so num_maps stayed 0 and c_reset fell into the emergency fallback (tower_climb.h:348-353):
   a single ground block with goal_location=999 — an unreachable goal, so reward_climb_row could
   never fire and training saw only the illegal-move penalty. Load + share it exactly as upstream
   does (same one-bank-for-all-copies design as ocean/maze/adapter.c). */
static Level* g_tc_levels = NULL;
static PuzzleState* g_tc_puzzles = NULL;
static int g_tc_num_maps = 0;
static void ensure_tc_levels(void){
  if(g_tc_levels) return;
  const char* env_path = getenv("PUFFER_TOWER_CLIMB_MAPS");
  const char* paths[2] = { env_path, "resources/tower_climb/maps.bin" };
  for(int i=0;i<2 && !g_tc_levels;i++){
    if(!paths[i]) continue;
    int n=0; Level* lv=load_levels_from_file(&n, paths[i]);
    if(!lv || n<=0) continue;
    PuzzleState* ps=(PuzzleState*)calloc((size_t)n,sizeof(PuzzleState));
    if(!ps){ free(lv); continue; }
    for(int k=0;k<n;k++){ init_puzzle_state(&ps[k]); levelToPuzzleState(&lv[k], &ps[k]); }
    g_tc_levels=lv; g_tc_puzzles=ps; g_tc_num_maps=n;
  }
}
static void ocean_setup(CTowerClimb* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  /* reward fields are float config; cfg_int cannot parse floats, so use config/tower_climb.ini [env] defaults */
  (void)cfg;
  e->reward_climb_row=0.205371f;
  e->reward_fall_row=0.0f;
  e->reward_illegal_move=-0.00397522f;
  e->reward_move_block=0.0f;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  ensure_tc_levels();          /* shared bank; NULL-safe — env keeps the old fallback if absent */
  e->all_levels=g_tc_levels;
  e->all_puzzles=g_tc_puzzles;
  e->num_maps=g_tc_num_maps;
  init(e);
  int t=OBS_VISION+PLAYER_OBS;
  e->observations=(unsigned char*)calloc(t,sizeof(unsigned char));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(CTowerClimb* e){ (void)e; return OBS_VISION+PLAYER_OBS; }
static int  ocean_numactions(CTowerClimb* e){ (void)e; return 6; }
static int  ocean_maxsteps(CTowerClimb* e){ (void)e; return 100000; }
static void ocean_teardown(CTowerClimb* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); c_close(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
