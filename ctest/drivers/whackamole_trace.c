// Headless trace driver for ocean/whackamole. Sets the initial mole directly
// (AT init_mole cell, bypassing c_reset's rand_r placement) and the PRNG seed,
// then applies the action list. The mid-episode mole hops and the terminal-step
// c_reset re-roll use rand_r(&env.rng), which the Lean model reproduces exactly.
//
// Emits the deterministic fields the env exposes AFTER each c_step:
//   tick, mole_r, mole_c, reward, hits, terminal.
// NOTE: on the terminating tick (tick reaches ATTEMPTS_PER_EPISODE=3) c_step
// calls c_reset, which clobbers rewards[0]=0, terminals[0]=0, tick=0 and re-rolls
// the mole; so terminal is never observed as 1 and the driver runs the full
// action list (there is no observable terminal to break on). hits is NOT reset
// by c_reset, so it accumulates across episode boundaries.
//
// Usage: whackamole_trace SEED INIT_MOLE_IDX ACTION [ACTION ...]
//        action: -1=NOOP, 0..24 = grid cell (row-major); other values pay 0.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "whackamole.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s SEED INIT_MOLE_IDX [ACTION...]\n", argv[0]); return 2; }
    unsigned int seed = (unsigned int)strtoul(argv[1], NULL, 10);
    int init_mole = atoi(argv[2]);

    Whackamole env;
    memset(&env, 0, sizeof(env));
    float* obs = (float*)calloc((size_t)TOTAL_CELLS, sizeof(float));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.num_agents = 1;
    env.rng = seed;
    env.tick = 0;
    env.hits = 0;
    env.mole_r = init_mole / GRID_SIZE;
    env.mole_c = init_mole % GRID_SIZE;
    env.client = NULL;
    if (init_mole >= 0 && init_mole < TOTAL_CELLS) obs[init_mole] = 1.0f;

    printf("step\ttick\tmole_r\tmole_c\treward\thits\tterminal\n");
    printf("0\t%d\t%d\t%d\t%.9g\t%d\t%d\n",
           env.tick, env.mole_r, env.mole_c, env.rewards[0], env.hits, (int)env.terminals[0]);
    for (int i = 3; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%d\t%d\t%.9g\t%d\t%d\n",
               i - 2, env.tick, env.mole_r, env.mole_c, env.rewards[0], env.hits, (int)env.terminals[0]);
    }
    free(obs);
    return 0;
}
