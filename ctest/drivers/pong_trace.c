// Headless trace driver for ocean/pong. Sets up a deterministic round state
// directly (bypassing c_reset), then applies the action list and prints the raw
// physics state each step, stopping at the first terminal (whose c_reset
// re-randomizes). All physics is float (f32); the only RNG is reset_round's
//   ball_vy = (rand_r(&rng) % 2 - 1) * ball_initial_speed_y
// which we neutralize by fixing ball_initial_speed_y = 0 (RNG multiplied out ->
// ball_vy resets to 0, deterministically). Vertical physics is still exercised
// by choosing a non-zero INITIAL ball_vy (argv[1], set directly here) and by the
// right-paddle bounce increment. Discrete controls (continuous = 0).
//
// Usage: pong_trace BALL_VY0 MAX_SCORE ACTION [ACTION ...]
//        actions: 0=STILL 1=UP(dir=+1) 2=DOWN(dir=-1)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pong.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s BALL_VY0 MAX_SCORE [ACTION...]\n", argv[0]); return 2; }
    Pong env;
    memset(&env, 0, sizeof(env));
    // fixed config (demo defaults) except ball_initial_speed_y = 0 (RNG bypass)
    env.width = 500; env.height = 640;
    env.paddle_width = 20; env.paddle_height = 70;
    env.ball_width = 32; env.ball_height = 32;
    env.paddle_speed = 8;
    env.ball_initial_speed_x = 10;
    env.ball_initial_speed_y = 0;
    env.ball_speed_y_increment = 3;
    env.ball_max_speed_y = 13;
    env.max_score = (unsigned int)atoi(argv[2]);
    env.frameskip = 1;
    env.continuous = 0;
    env.rng = 0;
    env.num_agents = 1;

    float obs[8]; memset(obs, 0, sizeof(obs));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;

    init(&env);   // min_paddle_y, max_paddle_y, paddle_dir=0, tick=0, n_bounces=0, win=0

    // deterministic round state (mirrors reset_round, but ball_vy is chosen)
    env.paddle_yl = env.height / 2 - env.paddle_height / 2;
    env.paddle_yr = env.height / 2 - env.paddle_height / 2;
    env.ball_x = env.width / 5;
    env.ball_y = env.height / 2 - env.ball_height / 2;
    env.ball_vx = env.ball_initial_speed_x;
    env.ball_vy = (float)atof(argv[1]);
    env.score_l = 0; env.score_r = 0;
    env.tick = 0; env.n_bounces = 0;

    printf("step\ttick\tpaddle_yl\tpaddle_yr\tball_x\tball_y\tball_vx\tball_vy\tscore_l\tscore_r\tn_bounces\treward\tterminal\n");
    printf("0\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%u\t%u\t%d\t%.9g\t%d\n",
        env.tick, env.paddle_yl, env.paddle_yr, env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
        env.score_l, env.score_r, env.n_bounces, env.rewards[0], (int)env.terminals[0]);

    for (int i = 3; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%u\t%u\t%d\t%.9g\t%d\n",
            i - 2, env.tick, env.paddle_yl, env.paddle_yr, env.ball_x, env.ball_y, env.ball_vx, env.ball_vy,
            env.score_l, env.score_r, env.n_bounces, env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }
    return 0;
}
