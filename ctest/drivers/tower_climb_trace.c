// Headless trace driver for ocean/tower_climb (discrete 3D block-pushing puzzle).
//
// The env is PURE INTEGER logic (bitmask block grid + robot pose); there is no
// floating-point physics. The ONLY rand_r is in c_reset's map pick
// (`rand_r(&env->rng) % num_maps`) and in gen_level (init_random_level, never
// called by c_step). By leaving num_maps == 0 we force c_reset's deterministic
// fallback (a trivial default level: one ground block at index 0, spawn 0, goal
// 999) — NO rand_r ever fires. The demo's srand()/rand() live in tower_climb.c,
// which we do not compile in. So the whole trace is deterministic.
//
// We build a custom level directly (blocks / spawn / goal), bypassing the maps
// file. c_step never sets terminals[0] (it resets internally on goal/death/
// timeout>60), and with num_maps==0 every such internal reset is the same
// deterministic default level, so long (100+ step) traces stay reproducible.
//
// Usage:
//   tower_climb_trace R_CLIMB R_FALL R_ILLEGAL R_MOVEBLOCK SPAWN GOAL
//       NBLOCKS b0 b1 ... b(NBLOCKS-1)  a0 a1 ...
//   actions: 0=RIGHT 1=DOWN 2=LEFT 3=UP 4=GRAB 5=DROP  (-1=NOOP)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tower_climb.h"

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s R_CLIMB R_FALL R_ILLEGAL R_MOVEBLOCK SPAWN GOAL "
                        "NBLOCKS b0.. a0..\n", argv[0]);
        return 2;
    }
    CTowerClimb env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.num_maps   = 0;   // force c_reset's deterministic default-level fallback
    env.client     = NULL;
    env.reward_climb_row    = (float)atof(argv[1]);
    env.reward_fall_row     = (float)atof(argv[2]);
    env.reward_illegal_move = (float)atof(argv[3]);
    env.reward_move_block   = (float)atof(argv[4]);
    int spawn = atoi(argv[5]);
    int goal  = atoi(argv[6]);
    int nblocks = atoi(argv[7]);

    env.observations = (unsigned char*)calloc(OBS_VISION + PLAYER_OBS, sizeof(unsigned char));
    env.actions      = (float*)calloc(1, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));

    init(&env);   // allocs env->level (dims 10/10/100/1000) + env->state

    // Build the custom level directly (bypass maps file / rand_r).
    env.level->goal_location  = goal;
    env.level->spawn_location = spawn;
    memset(env.level->map, 0, BLOCK_BYTES);
    for (int i = 0; i < nblocks; i++) {
        int idx = atoi(argv[8 + i]);
        SET_BIT(env.level->map, idx);
    }
    memcpy(env.state->blocks, env.level->map, BLOCK_BYTES);
    env.state->robot_position    = spawn;
    env.state->robot_orientation = UP;
    env.state->robot_state       = 0;
    env.state->block_grabbed     = -1;
    env.rows_cleared = 0;
    env.goal_reached = false;
    env.buffer = (Log){0};

    int first_action = 8 + nblocks;

    // 64-bit FNV-1a over the 125-byte block bitmask, split into two 32-bit halves
    // so difftest compares them as exact (< 2^53) float64 integers.
    #define BHASH(hi, lo) do { \
        uint64_t h = 0xcbf29ce484222325ULL; \
        for (int _i = 0; _i < BLOCK_BYTES; _i++) { \
            h ^= (uint64_t)env.state->blocks[_i]; h *= 0x100000001b3ULL; } \
        (hi) = (unsigned)(h >> 32); (lo) = (unsigned)(h & 0xffffffffULL); \
    } while (0)

    unsigned bh_hi, bh_lo;
    printf("step\tpos\torient\trstate\tgrabbed\tbhash_hi\tbhash_lo\treward\tterminal\n");
    BHASH(bh_hi, bh_lo);
    printf("0\t%d\t%d\t%d\t%d\t%u\t%u\t%.9g\t%d\n",
        env.state->robot_position, env.state->robot_orientation, env.state->robot_state,
        env.state->block_grabbed, bh_hi, bh_lo, env.rewards[0], (int)env.terminals[0]);

    for (int i = first_action; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        BHASH(bh_hi, bh_lo);
        printf("%d\t%d\t%d\t%d\t%d\t%u\t%u\t%.9g\t%d\n",
            i - first_action + 1,
            env.state->robot_position, env.state->robot_orientation, env.state->robot_state,
            env.state->block_grabbed, bh_hi, bh_lo, env.rewards[0], (int)env.terminals[0]);
    }
    return 0;
}
