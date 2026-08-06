/* Adapter: PufferLib Ocean `boxoban` -> pufferlean plugin ABI. Runs the REAL ocean/boxoban/boxoban.h
 * c_reset/c_step (10x10 Sokoban, 4 uint8 obs planes AGENT/WALLS/BOXES/TARGET = 400 bytes,
 * 5 discrete actions NOOP/DOWN/UP/LEFT/RIGHT).
 *
 * Field-for-field vs upstream binding.c my_init (PufferLib/ocean/boxoban/binding.c):
 *   difficulty_id, size=10, num_agents=1, max_steps, int_r_coeff, target_loss_pen_coeff, init(env)
 * — all set below (defaults from config/boxoban.ini [env]). init() then allocates
 * intermediate_rewards and sets win=0/initialized=false; c_reset fills agent_x/agent_y/n_boxes/
 * n_targets/on_target/tick/episode_return from the puzzle bank, and client stays NULL (headless).
 * env->rng is the one field upstream leaves at calloc-0 — with rng=0 in every copy they all draw
 * the SAME puzzle sequence in lockstep; we seed it per copy like every other adapter here.
 *
 * MAP BANK (the tower_climb-class trap): c_reset does `memcpy(obs, MAP_BASE + i*PUZZLE_SIZE, ...)`
 * with `i = rand_r % PUZZLE_COUNT`. With no bank loaded that is a %0 and a memcpy from NULL, i.e. a
 * crash, not a silent misbehaviour — so the bank is mandatory. Upstream loads it from
 * `resources/boxoban/boxoban_maps_<difficulty>.bin`, generating (basic/easy) or downloading
 * (medium/hard/unfiltered, ~11MB zip from GitHub) that file on first use. We do the same, once, in
 * a process-wide static, and fall back to the always-locally-generable `basic` bank if the
 * configured difficulty cannot be prepared (e.g. no network). Paths are relative to the cwd exactly
 * as upstream's are — the CLI already requires being run from the repo root (config/<env>.ini) —
 * with an absolute repo-root path also probed, mirroring ocean/tower_climb/adapter.c. Set
 * BOXOBAN_MAP_BIN=<path> to point at a prebuilt bank instead. */
#define BOXOBAN_MAPS_IMPLEMENTATION   /* defines MAP_BASE/PUZZLE_COUNT/... + the map loader */
#include "boxoban.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#define ENV_T Boxoban
#define OBS_T unsigned char
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }

/* Load the shared puzzle bank ONCE per process (mmap'd read-only; every env copy indexes it). */
static int g_bx_tried = 0;
static void ensure_boxoban_maps(int difficulty_id){
  if(g_bx_tried) return;
  g_bx_tried = 1;
  const char* envp = getenv("BOXOBAN_MAP_BIN");
  if(envp && access(envp,F_OK)==0 && boxoban_set_map_path(envp)==0){ ensure_map_loaded(); return; }
  int order[2] = { difficulty_id, 0 };   /* configured difficulty, then `basic` (generated offline) */
  for(int t=0;t<2;t++){
    const char* name = boxoban_difficulty_name_from_id(order[t]);
    if(name == NULL) continue;
    char rel[512], out[512];
    snprintf(rel,sizeof rel,"resources/boxoban/boxoban_maps_%s.bin",name);
    const char* have = (access(rel,F_OK)==0) ? rel : NULL;
    if(have){ if(boxoban_set_map_path(have)==0){ ensure_map_loaded(); return; } continue; }
    if(boxoban_prepare_maps_for_difficulty(name,out,sizeof out)==0){ ensure_map_loaded(); return; }
    fprintf(stderr,"[boxoban] could not prepare the '%s' map bank\n",name);
  }
  fprintf(stderr,"[boxoban] FATAL: no puzzle bank available (tried difficulty %d then basic); "
                 "run from the repo root or set BOXOBAN_MAP_BIN\n",difficulty_id);
  abort();   /* c_reset would otherwise memcpy from NULL — upstream's init() aborts here too */
}

static void ocean_setup(Boxoban* e,uint64_t seed,int idx,const char* cfg){
  int difficulty = cfg_int(cfg,"difficulty",2);          /* 0 basic 1 easy 2 medium 3 hard 4 unfiltered */
  e->size=10;                                            /* the bank is 10x10; PUZZLE_OBS_BYTES==4*10*10 */
  e->num_agents=1;
  e->max_steps=cfg_int(cfg,"max-steps",150);
  e->int_r_coeff=cfg_flt(cfg,"int-r-coeff",0.25f);
  e->target_loss_pen_coeff=cfg_flt(cfg,"target-loss-pen-coeff",0.0f);
  if(e->max_steps<1) e->max_steps=1;                     /* c_reset does `rand_r % max_steps` */
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  ensure_boxoban_maps(difficulty);
  e->difficulty_id=-1;   /* bank already configured above; -1 makes init()'s own configure a no-op
                            (boxoban.h:70) instead of re-preparing it per copy */
  init(e);               /* allocates intermediate_rewards, win=0, initialized=false */
  e->difficulty_id=difficulty;                           /* restore: inert after init, but honest */
  e->observations=(unsigned char*)calloc((size_t)(4*e->size*e->size),sizeof(unsigned char));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(Boxoban* e){ return 4*e->size*e->size; }
static int  ocean_numactions(Boxoban* e){ (void)e; return 5; }
static int  ocean_maxsteps(Boxoban* e){ return e->max_steps; }
static void ocean_teardown(Boxoban* e){
  c_close(e);            /* frees intermediate_rewards (window was never opened) */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}
/* env's own PufferLib `Log` fields, in declaration order — expanded to offsetof by the shared
   adapter so the log channel reports the same statistics upstream's my_log does. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(on_targets) X(n)
#include "../../ffi/ocean_adapter.h"
