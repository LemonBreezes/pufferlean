// Headless trace driver for ocean/pacman. Builds the canonical map via init(),
// then sets up the round state DIRECTLY (bypassing reset_round's rand_range /
// rand_r), so the trajectory is fully deterministic:
//   - randomize_starting_position = false  (player never rand_r-placed)
//   - every ghost start_timeout = 0         (no rand_range draw)
//   - env->rng = 0                          (only consumed if a ghost is
//                                            frightened; reproduced by Lean randR)
// The player start (PX,PY) is an argument so distinct behaviours can be
// exercised; on any terminal, c_step internally re-runs c_reset, which restores
// the FIXED spawns (player -> map 'p' at (13,23), ghosts -> their '1234' tiles,
// direction UP, score 0, remaining_pickups 244). We stop at the first terminal.
//
// Config mirrors config/pacman.ini (randomize off):
//   min/max_start_timeout 0/49, frightened_time 35, max_mode_changes 6,
//   scatter_mode_length 70, chase_mode_length 140.
//
// Usage: pacman_trace PX PY ACTION [ACTION ...]
//        actions/dirs: 0=DOWN 1=UP 2=RIGHT 3=LEFT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pacman.h"

static void print_row(int step, PacmanEnv *env) {
    printf("%d\t%d\t%d\t%d\t%d\t%.9g\t%d\t%d",
           step, env->player_pos.x, env->player_pos.y, env->player_direction,
           env->score, env->rewards[0], (int)env->terminals[0], env->remaining_pickups);
    for (int i = 0; i < NUM_GHOSTS; i++) {
        Ghost *g = &env->ghosts[i];
        printf("\t%d\t%d\t%d", g->pos.x, g->pos.y, g->direction);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s PX PY ACTION [ACTION...]\n", argv[0]);
        return 2;
    }
    int px = atoi(argv[1]);
    int py = atoi(argv[2]);

    PacmanEnv env;
    memset(&env, 0, sizeof(env)); // rng = 0
    env.randomize_starting_position = false;
    env.min_start_timeout = 0;
    env.max_start_timeout = 49;
    env.frightened_time = 35;
    env.max_mode_changes = 6;
    env.scatter_mode_length = 70;
    env.chase_mode_length = 140;
    env.num_agents = 1;

    init(&env); // builds game_map from original_map + spawn positions

    env.observations = (float *)calloc(OBSERVATIONS_COUNT, sizeof(float));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;

    // round setup (deterministic; mirrors reset_round without the RNG)
    env.score = 0;
    env.scatter_mode = false;
    env.mode_time_left = 0;
    env.mode_changes = 0;
    env.frightened_time_left = 0;
    env.step_count = 0;
    env.remaining_pickups = NUM_DOTS + NUM_POWERUPS;
    for (int i = 0; i < NUM_DOTS + NUM_POWERUPS; i++)
        env.pickup_obs[i] = 1.0f;
    for (int i = 0; i < NUM_GHOSTS; i++) {
        Ghost *g = &env.ghosts[i];
        g->pos = g->spawn_pos;
        g->direction = UP;
        g->start_timeout = 0; // fixed (bypass rand_range)
        g->frightened = false;
        g->return_to_spawn = false;
        g->half_move = false;
    }
    env.player_pos = (Position){px, py};
    env.player_direction = RIGHT;
    update_interpolation(&env); // last positions (render only; harmless)

    printf("step\tpx\tpy\tpdir\tscore\treward\tterm\trem"
           "\tg0x\tg0y\tg0d\tg1x\tg1y\tg1d\tg2x\tg2y\tg2d\tg3x\tg3y\tg3d\n");
    print_row(0, &env);
    for (int i = 3; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        print_row(i - 2, &env);
        if (env.terminals[0])
            break;
    }
    free(env.observations);
    c_close(&env);
    return 0;
}
