/* NOT TRAINABLE THROUGH THIS ABI TODAY — measured, not speculation.
   Upstream declares MY_ACTION_MASK 97 and hands the policy a legal-move mask; our plugin ABI has no
   mask channel, so `action_mask` stays NULL (chess.h NULL-guards it). A move needs two correct
   actions out of 97, so a random policy completes one roughly every ~250 steps and no game finishes:
   `puffer train chess --total-agents 2048 --env.max-moves 60` reports 0 eps and exactly 0.000000
   reward through update 40. The adapter itself is faithful (every field upstream's binding.c sets is
   set here); the gap is the missing mask channel plus the fact that chess obs are categorical tokens
   0..255 that upstream feeds through an embedding, not raw into an MLP/MinGRU.
   So: a flat zero here is EXPECTED, not a regression — do not chase it in env sweeps.

   WHY THERE IS NO MASK CHANNEL (decided 2026-08-04, cost/benefit): exactly TWO upstream envs declare
   MY_ACTION_MASK — chess and nethack — and nethack is unported (needs vendor/fast-nle + a custom CUDA
   encoder). Adding masks to the plugin ABI means threading a per-step mask plane through the adapter,
   the loader, the categorical sampler AND the PPO gradient in puffercuda.cu, for ONE reachable env
   that would still train poorly because its observations are categorical tokens upstream feeds through
   an embedding we also do not have. If nethack is ever ported, or an embedding encoder lands, revisit —
   the two changes together are what make chess trainable, and neither is useful alone. */
/* Adapter: PufferLib Ocean `chess` -> pufferlean plugin ABI. Runs the REAL ocean/chess/chess.h
 * c_reset/c_step (full legal-move chess: magic bitboards, castling, en passant, promotion,
 * 50-move rule, threefold repetition).
 *
 * SHAPE. uint8 obs, OBS_SIZE = 167 per agent (ego-centric board 64 + side + castle 4 + ep 9 +
 * rule50/repetition/checks/pick-phase/selected + the valid-from (16) / valid-to (32) / promo index
 * lists + pass flag), 97 discrete actions (0..63 square, 64..95 promotion cell, 96 pass), float
 * actions/rewards/terminals. A move takes TWO actions: pick a from-square, then a to-square.
 *
 * MODE / AGENT COUNT. config/chess.ini sets mode = 1 = CHESS_MODE_SELFPLAY, which is 2 agents per
 * env instance (one per colour) — both slots are learners, so the trainer's shared policy plays
 * itself. mode = 0 (CHESS_MODE_RANDOM) is 1 agent vs a built-in random opponent and is the cheap
 * sanity setting. Modes 2/3 (human) and 4 (maia, spawns an external lc0) are interactive/eval only;
 * do not train them.
 *
 * BUFFER POINTERS. Unlike every other ocean env, chess does NOT read env->observations/actions/
 * rewards/terminals — it reads per-slot pointers (obs_ptr/action_ptr/reward_ptr/terminal_ptr) that
 * PufferLib's vecenv fills in via binding.c's my_setup_perm. We allocate the four buffers in the
 * layout the shared adapter expects (agent a at stride a) and point the per-slot pointers into
 * them, which is exactly what my_setup_perm does for an identity permutation.
 *
 * ACTION MASK. Upstream declares MY_ACTION_MASK 97 and vecenv hands the policy a per-step legal
 * -action mask. Our plugin ABI has no mask channel, so env->action_mask stays NULL — chess.h
 * explicitly NULL-guards it (chess.h:1594) and simply skips mask filling. The legality information
 * is still IN the observation (O_VALID_FROM/O_VALID_TO index lists), so the policy can learn it,
 * but expect far worse sample efficiency than upstream's masked runs.
 *
 * Field-for-field vs upstream binding.c (apply_kwargs + my_init + my_vec_init): every assignment
 * there is reproduced below in the same order — see the comments. The only deliberate differences:
 *   - rng: upstream uses the env index `i`; we use the repo-wide seed+idx*golden convention.
 *   - obs_ptr[1]/action_ptr[1]/reward_ptr[1]/terminal_ptr[1] are left NULL when num_agents == 1,
 *     exactly as my_setup_perm leaves them (every use of slot 1 is under an `mode == SELFPLAY`
 *     guard).
 *   - tag / boundary_reached stay 0 (calloc): tag > 0 selects a FROZEN historical opponent bank
 *     from PufferLib's selfplay pool, which is a Python-side feature we do not have. tag = 0 is
 *     "pure selfplay", which is what upstream also uses for the non-historical fraction of envs. */
#include "chess.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#define ENV_T Chess
#define OBS_T uint8_t
#define ACT_T float
#define OCEAN_NAGENTS(e) ((e)->num_agents)

#define DEFAULT_STARTING_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }

/* Shared FEN curriculum, loaded once and shared by every copy (upstream's my_vec_init does the same
   with a file-scope SHARED_FEN_CURRICULUM). The file is NOT shipped with PufferLib, so this is
   normally a no-op and c_reset falls back to starting_fen — the same NULL-safe shape as
   ocean/tower_climb/adapter.c's map bank. */
static char** g_fens = NULL;
static int g_num_fens = 0, g_fens_tried = 0;
static char** load_fen_file(const char* path, int* num_fens_out){
  FILE* f = fopen(path,"r");
  if(f == NULL){ *num_fens_out = 0; return NULL; }
  int num_fens = 0; char line[256];
  while(fgets(line,sizeof(line),f))
    if(line[0] != '#' && line[0] != '\n' && line[0] != '\r') num_fens++;
  if(num_fens == 0){ fclose(f); *num_fens_out = 0; return NULL; }
  char** fens = (char**)malloc((size_t)num_fens*sizeof(char*));
  rewind(f);
  int idx = 0;
  while(fgets(line,sizeof(line),f) && idx < num_fens){
    if(line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    size_t len = strlen(line);
    while(len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
    fens[idx++] = strdup(line);
  }
  fclose(f);
  *num_fens_out = num_fens;
  return fens;
}
static void ensure_fens(float curric_pct){
  if(g_fens_tried) return;
  g_fens_tried = 1;
  if(curric_pct <= 0.0f) return;                       /* upstream only loads when the pct is > 0 */
  const char* paths[2] = { getenv("PUFFER_CHESS_FENS"), "resources/chess/fens.txt" };
  for(int i=0;i<2 && g_fens==NULL;i++){
    if(paths[i] == NULL) continue;
    int n = 0; char** f = load_fen_file(paths[i], &n);
    if(f && n > 0){ g_fens = f; g_num_fens = n; }
  }
}

static void ocean_setup(Chess* e,uint64_t seed,int idx,const char* cfg){
  /* ---- apply_kwargs (binding.c:66-93) ---- */
  e->max_moves                  = cfg_int(cfg,"max-moves",5000);
  e->reward_draw                = cfg_flt(cfg,"reward-draw",0.0f);
  e->reward_invalid_piece       = cfg_flt(cfg,"reward-invalid-piece",0.0f);
  e->reward_invalid_move        = cfg_flt(cfg,"reward-invalid-move",0.0f);
  e->reward_repetition          = cfg_flt(cfg,"reward-repetition",0.0f);
  e->render_fps                 = cfg_int(cfg,"render-fps",30);
  e->mode                       = cfg_int(cfg,"mode",CHESS_MODE_SELFPLAY);
  e->enable_50_move_rule        = cfg_int(cfg,"enable-50-move-rule",1);
  e->enable_threefold_repetition= cfg_int(cfg,"enable-threefold-repetition",1);
  e->random_fen                 = cfg_int(cfg,"random-fen",0);
  e->fen_curric_pct             = cfg_flt(cfg,"fen-curric-pct",0.9f);
  e->client                     = NULL;
  e->legal_dirty                = 1;
  e->human_color                = -1;
  e->log_pgn                    = 0;
  e->log_pgn_choice_made        = 1;
  e->pgn_filename[0]            = '\0';
  e->pgn_game_number            = 0;
  e->maia_pid                   = 0;
  e->maia_stdin_fd              = -1;
  e->maia_stdout_fd             = -1;
  e->maia_phase                 = 0;
  strcpy(e->starting_fen, DEFAULT_STARTING_FEN);
  strcpy(e->last_result, "Game starting...");
  /* Interactive-only modes have no policy driving them here (human input / an external lc0
     subprocess per env); clamp to the random-opponent mode so a stray --mode can't hang training. */
  if(e->mode != CHESS_MODE_RANDOM && e->mode != CHESS_MODE_SELFPLAY) e->mode = CHESS_MODE_RANDOM;
  if(e->max_moves < 1) e->max_moves = 1;

  /* ---- my_vec_init per-env block (binding.c:112-135) ---- */
  int agents_per_env = (e->mode == CHESS_MODE_SELFPLAY) ? 2 : 1;
  e->num_agents  = agents_per_env;
  e->rng         = (unsigned int)(seed + (uint64_t)idx*0x9E3779B9u);
  e->learner_color = (agents_per_env == 1) ? (idx % 2) : CHESS_WHITE;
  if(agents_per_env == 2 && (idx & 1)){          /* de-bias slot<->colour across env copies */
    e->slot_for_color[CHESS_WHITE] = 1;
    e->slot_for_color[CHESS_BLACK] = 0;
  } else {
    e->slot_for_color[CHESS_WHITE] = 0;
    e->slot_for_color[CHESS_BLACK] = 1;
  }
  ensure_fens(e->fen_curric_pct);
  e->fen_curriculum = g_fens;
  e->num_fens       = g_num_fens;
  init_bitboards();                              /* self-guarded, chess.h:727 */

  /* ---- the buffers vecenv would own, plus my_setup_perm's per-slot pointers ---- */
  int nA = agents_per_env;
  e->observations = (uint8_t*)calloc((size_t)nA*OBS_SIZE, sizeof(uint8_t));
  e->actions      = (float*)  calloc((size_t)nA, sizeof(float));
  e->rewards      = (float*)  calloc((size_t)nA, sizeof(float));
  e->terminals    = (float*)  calloc((size_t)nA, sizeof(float));
  e->action_mask  = NULL;                        /* no mask channel in this ABI; chess.h NULL-guards */
  for(int s=0;s<nA;s++){
    e->obs_ptr[s]         = e->observations + (size_t)s*OBS_SIZE;
    e->action_mask_ptr[s] = NULL;
    e->action_ptr[s]      = e->actions   + s;
    e->reward_ptr[s]      = e->rewards   + s;
    e->terminal_ptr[s]    = e->terminals + s;
  }
}
static int  ocean_obsdim(Chess* e){ (void)e; return OBS_SIZE; }
static int  ocean_numactions(Chess* e){ (void)e; return NUM_ACTIONS; }   /* 97 */
static int  ocean_maxsteps(Chess* e){ return 2*e->max_moves; }           /* 2 actions per move */
static void ocean_teardown(Chess* e){
  c_close(e);   /* render client + maia subprocess: both absent here, so this is a no-op */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
  /* The shared FEN bank is deliberately NOT freed: the CLI opens a throwaway "peek" wrap, closes
     it, then opens the real one, so a process-wide static freed on teardown would be a
     use-after-free for the second wrap (same reasoning as ocean/maze's shared level bank). */
}
/* env's own PufferLib `Log` fields, in declaration order — expanded to offsetof by the shared
   adapter so the log channel reports the same statistics upstream's my_log does. */
#define OCEAN_LOG_FIELDS(X) X(n) X(wins_as_white) X(wins_as_black) X(games_as_white) X(games_as_black) X(maia_failures)
#include "../../ffi/ocean_adapter.h"
