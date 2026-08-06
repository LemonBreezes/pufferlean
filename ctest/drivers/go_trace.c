// Headless trace driver for ocean/go. Sets up a small board directly (side=1 =
// agent plays black and moves first, selfplay=0, human_play=0), bypassing any
// reset RNG, and applies the given agent actions (0=pass, 1..gs*gs = point+1).
// The bot's turns consume steps too (ignoring the supplied action). The built-in
// opponent (enemy_greedy_hard) is deterministic apart from enemy_random_move's
// Fisher-Yates shuffle, which uses rand_r(&env->rng) — a readable glibc LCG
// seeded to 0 here, so the whole trace is reproducible.
//
// Emits per step: tick / reward / terminal / capture_count[0] (black) /
// capture_count[1] (white) / previous_move / legal / illegal / pass / nstones,
// stopping at the first terminal (whose end_game->c_reset wipes the board).
//
// Usage: go_trace GRID_SIZE ACTION [ACTION ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "go.h"

static int count_stones(CGo* env) {
    int n = 0;
    for (int i = 0; i < env->grid_size * env->grid_size; i++)
        if (env->board_states[i] != 0) n++;
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s GRID_SIZE [ACTION...]\n", argv[0]); return 2; }
    int grid_size = atoi(argv[1]);
    CGo env;
    memset(&env, 0, sizeof(env));
    env.grid_size = grid_size;
    env.grid_square_size = 64;
    env.width = 950; env.height = 750;
    env.board_width = 600; env.board_height = 600;
    env.komi = 7.5f;
    env.reward_move_pass = -0.518441f;
    env.reward_move_valid = 0.0f;
    env.reward_move_invalid = -0.0864746f;
    env.reward_player_capture = 0.553628f;
    env.reward_opponent_capture = -0.102283f;
    env.selfplay = 0;
    env.side = 1;
    env.human_play = 0;
    env.num_agents = 1;
    allocate(&env);
    env.rng = 0;
    c_reset(&env);

    printf("step\ttick\treward\tterminal\tcapb\tcapw\tprevmove\tlegal\tillegal\tpass\tnstones\n");
    printf("0\t%d\t%.9g\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
           (int)env.tick, env.rewards[0], (int)env.terminals[0],
           env.capture_count[0], env.capture_count[1], env.previous_move,
           env.legal_move_count, env.illegal_move_count, env.pass_move_count,
           count_stones(&env));
    for (int i = 2; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%.9g\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
               i - 1, (int)env.tick, env.rewards[0], (int)env.terminals[0],
               env.capture_count[0], env.capture_count[1], env.previous_move,
               env.legal_move_count, env.illegal_move_count, env.pass_move_count,
               count_stones(&env));
        if (env.terminals[0]) break;
    }
    free_allocated(&env);
    return 0;
}
