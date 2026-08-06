// Headless trace driver for ocean/freeway (Atari Freeway crossing game).
//
// Sets up a deterministic initial state directly (the squared/pong bypass) and
// applies an action list, printing the observable AI-player state each step,
// stopping at the first terminal (whose c_reset re-randomizes).
//
// Determinism: freeway's RNG is neutralised by config, so no seed is needed.
//   * env_randomization = 0  -> spawn_enemies adds no random lane offset (the
//     rand_r draw is discarded); enemies spawn at enemy_initial_x deterministically.
//   * levels 0-3 have SPEED_RANDOMIZATION = 0, so randomize_enemy_speed (fired
//     every 360 ticks) only advances rng and never changes an enemy's speed.
//   * enable_human_player = 0 -> the human paddle never steps / touches rewards.
// Under those conventions the whole trajectory is a deterministic function of the
// fixed initial state, matching Puffer/Env/Freeway/Model.lean.
//
// Usage: freeway_trace LEVEL DIFFICULTY FRAMESKIP ACTION [ACTION ...]
//        actions: 0=NOOP 1=UP 2=DOWN     (LEVEL in 0..3)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freeway.h"

#define N_ENEMIES (NUM_LANES * MAX_ENEMIES_PER_LANE)

static void print_row(Freeway* env, int step) {
    double sumx = 0.0;
    for (int e = 0; e < N_ENEMIES; e++) sumx += env->enemies[e].enemy_x;
    printf("%d\t%d\t%.9g\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%d\n",
        step, env->tick, env->ai_player.player_y, env->ai_player.best_lane_idx,
        env->ai_player.score, env->ai_player.ticks_stunts_left, env->ai_player.hits,
        sumx, env->enemies[0].enemy_x, env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s LEVEL DIFFICULTY FRAMESKIP ACTION...\n", argv[0]);
        return 2;
    }
    Freeway env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.frameskip = atoi(argv[3]);
    env.width = 1216;
    env.height = 720;
    env.player_width = 64;
    env.player_height = 64;
    env.car_width = 64;
    env.car_height = 40;
    env.lane_size = 64;
    env.difficulty = atoi(argv[2]);
    env.level = atoi(argv[1]);
    env.enable_human_player = 0;
    env.env_randomization = 0;
    env.use_dense_rewards = 1;

    init(&env);   // allocates enemies + human_actions, sets road/truck dims, load_level

    float obs[4 + N_ENEMIES]; memset(obs, 0, sizeof(obs));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.rng = 0;

    c_reset(&env);   // player to start line, spawn_enemies (deterministic), compute_observations

    printf("step\ttick\tplayer_y\tbest_lane\tscore\tstunts\thits\tsumx\tex0\treward\tterminal\n");
    print_row(&env, 0);
    for (int i = 4; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        env.human_actions[0] = 0;
        c_step(&env);
        print_row(&env, i - 3);
        if (env.terminals[0]) break;
    }
    c_close(&env);
    return 0;
}
