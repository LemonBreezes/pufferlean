// Headless trace driver for ocean/breakout.
//
// Determinism: c_reset is fully deterministic (paddle/ball placed by fixed
// formulas, all bricks reset, no RNG). The ONLY RNG in the env is
// `rand_r(&env->rng)` in step_frame, drawn ONCE each time a ball is fired (when
// balls_fired transitions 0->1) to pick the horizontal launch direction. It is
// glibc's reentrant LCG, reproduced exactly on the Lean side (randR) and seeded
// here from RNG_SEED. Under a fixed seed the whole trajectory is deterministic.
//
// The env config is baked to the standard breakout demo params so both sides
// agree; only the rng seed and the action list vary. rewards/terminals are
// zeroed by c_step itself, so the printed reward is the single-step value.
//
// Usage: breakout_trace RNG_SEED ACTION [ACTION ...]
//        actions: 0=NOOP 1=LEFT 2=RIGHT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "breakout.h"

static int count_bricks(Breakout* env) {
    int n = 0;
    for (int i = 0; i < env->num_bricks; i++)
        if (env->brick_states[i] == 1.0f) n++;
    return n;
}

static void print_row(Breakout* env, int step) {
    printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%d\t%d\t%d\t%.9g\t%d\n",
        step, env->paddle_x, env->ball_x, env->ball_y, env->ball_vx, env->ball_vy,
        env->ball_speed, env->paddle_width, env->score, env->num_balls,
        env->balls_fired, env->hits, count_bricks(env),
        env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s RNG_SEED ACTION [ACTION ...]\n", argv[0]);
        return 2;
    }
    Breakout env;
    memset(&env, 0, sizeof(env));
    env.frameskip           = 1;
    env.width               = 576;
    env.height              = 330;
    env.initial_paddle_width = 62;
    env.paddle_width        = 62;
    env.paddle_height       = 8;
    env.ball_width          = 32;
    env.ball_height         = 32;
    env.brick_width         = 32;
    env.brick_height        = 12;
    env.brick_rows          = 6;
    env.brick_cols          = 18;
    env.initial_ball_speed  = 256;
    env.max_ball_speed      = 448;
    env.paddle_speed        = 620;
    env.continuous          = 0;

    unsigned int seed = (unsigned int)strtoul(argv[1], NULL, 10);

    allocate(&env);      // init() + alloc obs/act/rew/term (needs config above)
    env.rng = seed;      // seed the reproducible rand_r stream (ball-fire direction)
    c_reset(&env);       // deterministic reset (does not touch env.rng)

    printf("step\tpaddle_x\tball_x\tball_y\tball_vx\tball_vy\tball_speed\tpaddle_width"
           "\tscore\tnum_balls\tballs_fired\thits\tbricks\treward\tterminal\n");
    print_row(&env, 0);
    for (int i = 2; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        print_row(&env, i - 1);
        if (env.terminals[0]) break;
    }
    free_allocated(&env);
    return 0;
}
