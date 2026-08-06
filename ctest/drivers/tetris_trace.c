// Headless trace driver for ocean/tetris. Sets up a deterministic env (rng
// seeded to 0, n_noise_obs = 0 so compute_observations never draws, and
// n_init_garbage = 0 so reset never draws garbage), calls c_reset (which
// shuffles the 7-bag deck via glibc rand_r from seed 0), then applies the action
// list. The ONLY stochasticity is the deck shuffle (Fisher-Yates over rand_r),
// which is fully reproduced in the Lean model. Garbage lines only kick in at
// tick >= GARBAGE_KICKOFF_TICK (500); every case here stays well under that, so
// no garbage rand_r is ever drawn. Stops at the first terminal (whose c_reset
// re-randomizes and leaves terminals[0] = 1); on that row the deterministic
// post-reset fields are printed alongside the terminal step's reward.
//
// Usage: tetris_trace N_ROWS N_COLS ACTION [ACTION ...]
//        actions: 0=NOOP 1=LEFT 2=RIGHT 3=ROTATE 4=SOFT_DROP 5=HARD_DROP 6=HOLD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tetris.h"

static void emit(Tetris *env, int step) {
    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.9g\t%d\n",
        step, env->tick, env->cur_tetromino, env->cur_tetromino_row,
        env->cur_tetromino_col, env->cur_tetromino_rot, env->score,
        env->lines_deleted, env->game_level, env->rewards[0],
        (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s N_ROWS N_COLS [ACTION...]\n", argv[0]); return 2; }
    Tetris env;
    memset(&env, 0, sizeof(env));
    env.n_rows = atoi(argv[1]);
    env.n_cols = atoi(argv[2]);
    env.use_deck_obs = true;
    env.n_noise_obs = 0;
    env.n_init_garbage = 0;
    allocate(&env);
    env.rng = 0;
    c_reset(&env);

    printf("step\ttick\tcur\trow\tcol\trot\tscore\tlines\tlevel\treward\tterminal\n");
    emit(&env, 0);
    for (int i = 3; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        emit(&env, i - 2);
        if (env.terminals[0]) break;
    }
    free_allocated(&env);
    return 0;
}
