// Headless trace driver for ocean/drmario. Sets up the env directly
// (n_rows/n_cols/n_init_viruses from argv, rng = SEED), calls c_reset — which
// deterministically places viruses and spawns the first capsule via glibc
// rand_r(&env->rng), a readable, fully-reproducible LCG — and applies the given
// action list. Emits the deterministic observable fields each step and stops at
// the first terminal (whose c_reset would re-randomize the board).
//
// Grid checksum `gridsum` = sum over cells of grid[i]*(i+1), a position-weighted
// signature of the locked pieces + viruses (capsule-in-flight cells are NOT in
// the grid until they lock).
//
// Usage: drmario_trace ROWS COLS VIRUSES SEED [ACTION ...]
//        actions: 0=NOOP 1=LEFT 2=RIGHT 3=DOWN 4=ROT_L 5=ROT_R 6=DROP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "drmario.h"

static long grid_checksum(DrMario* env) {
    long s = 0;
    int cells = env->n_rows * env->n_cols;
    for (int i = 0; i < cells; i++) {
        s += (long)env->grid[i] * (long)(i + 1);
    }
    return s;
}

static void print_row(int step, DrMario* env) {
    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%ld\t%.9g\t%d\n",
        step, env->tick, env->tick_fall,
        env->cap_color_a, env->cap_color_b, env->cap_orient,
        env->cap_row_1, env->cap_col_1, env->cap_row_2, env->cap_col_2,
        env->viruses_remaining, env->viruses_cleared,
        env->score, grid_checksum(env),
        env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s ROWS COLS VIRUSES SEED [ACTION...]\n", argv[0]);
        return 2;
    }
    DrMario env;
    memset(&env, 0, sizeof(env));
    env.n_rows = atoi(argv[1]);
    env.n_cols = atoi(argv[2]);
    env.n_init_viruses = atoi(argv[3]);
    env.rng = (unsigned int)strtoul(argv[4], NULL, 10);

    int cells = env.n_rows * env.n_cols;
    env.dim_obs = cells * N_OBS_PLANES + N_SCALAR_OBS;
    float* obs = (float*)calloc((size_t)env.dim_obs, sizeof(float));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;

    c_init(&env);      // allocates env.grid
    c_reset(&env);     // places viruses + spawns first capsule via rand_r(SEED)

    printf("step\ttick\ttick_fall\tca\tcb\torient\tr1\tc1\tr2\tc2\tvrem\tvclr\tscore\tgridsum\treward\tterminal\n");
    print_row(0, &env);
    for (int i = 5; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        print_row(i - 4, &env);
        if (env.terminals[0]) break;
    }

    free(env.grid);
    free(obs);
    return 0;
}
