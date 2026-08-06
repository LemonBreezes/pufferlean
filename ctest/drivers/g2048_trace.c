// Headless trace driver for ocean/g2048 (the "2048" tile game).
//
// g2048's RNG runs DURING the episode: every board-changing move spawns one new
// tile via place_tile_at_random_cell(get_new_tile()), both drawing from
// rand_r(&game->rng) — the POSIX reentrant PRNG seeded by the readable
// `game->rng` field. c_reset likewise seeds the starting board with RNG tiles.
// Because rand_r has a simple, portable, reproducible algorithm and a readable
// seed, the whole trace is deterministic once we fix `game->rng`.
//
// So instead of bypassing c_reset (as the pure reset-RNG envs do) we FIX THE
// SEED: env.rng = SEED before c_reset. The Lean model reproduces rand_r and
// c_reset exactly, so both sides agree on the initial board and every spawn.
// scaffolding_ratio is 0, so is_scaffolding_episode is always false (it still
// consumes exactly one rand_r for the check, which the Lean model mirrors).
//
// On a terminal step c_step calls c_reset again (re-seeding the board from the
// same advancing rng); we stop at that first terminal. The terminal row shows
// the post-reset board and score=0 — the Lean model reproduces that same reset.
//
// Columns: step, grid (16 tile-exponents, row-major, comma-joined), score,
//          reward (f32, %.9g), terminal.  A grid cell holds the tile's log2
//          exponent: 0=empty, 1=2, 2=4, ..., n=2^n.
//
// Usage: g2048_trace SEED ACTION [ACTION ...]
//        ACTION: 0=UP 1=DOWN 2=LEFT 3=RIGHT  (c_step applies action+1 internally)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "g2048.h"

static void print_row(Game* env, int step) {
    printf("%d\t", step);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d%s", env->grid[i][j], (i == SIZE - 1 && j == SIZE - 1) ? "" : ",");
        }
    }
    printf("\t%d\t%.9g\t%d\n", env->score, env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s SEED [ACTION...]\n", argv[0]); return 2; }
    unsigned int seed = (unsigned int)strtoul(argv[1], NULL, 10);

    Game env;
    memset(&env, 0, sizeof(env));
    unsigned char obs[SIZE * SIZE] = {0};
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;
    env.num_agents = 1;
    env.scaffolding_ratio = 0.0f;   // never a scaffolding episode

    init(&env);                     // lifetime_max_tile = 0, grid cleared
    env.rng = seed;                 // fix the PRNG seed (deterministic trace)
    c_reset(&env);                  // seed the starting board from `seed`

    printf("step\tgrid\tscore\treward\tterminal\n");
    print_row(&env, 0);

    for (int i = 2; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        print_row(&env, i - 1);
        if (env.terminals[0]) break;   // c_reset already re-seeded the board; stop
    }
    return 0;
}
