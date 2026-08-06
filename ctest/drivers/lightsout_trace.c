// Headless trace driver for ocean/lightsout. Sets up a FIXED initial grid
// directly (bypassing c_reset's rand_r() scramble), and stops at the first
// terminal (whose internal init_lightsout re-scrambles the board). Emits only
// the deterministic scalar fields: step_count, last_action, prev_action,
// reward, terminal.
//
// The board is passed as a GRID_BITS string of grid_size*grid_size chars
// ('0'/'1', row-major); lights_on is derived from it. This bypasses the RNG
// that c_reset would otherwise use to fill the grid.
//
// NOTE: lights_on is intentionally NOT printed. On a terminal step c_step calls
// init_lightsout(), which RNG-re-scrambles the grid and so clobbers lights_on
// with a non-deterministic value before we can read it. step_count/last_action/
// prev_action are reset to deterministic values (0/-1/-1) by that same call, and
// reward/terminals survive it, so those columns stay deterministic on every row.
//
// Usage: lightsout_trace GRID_SIZE MAX_STEPS GRID_BITS ACTION [ACTION ...]
//        ACTION: a cell index 0..grid_size*grid_size-1 (out-of-range => invalid)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lightsout.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s GRID_SIZE MAX_STEPS GRID_BITS [ACTION...]\n", argv[0]);
        return 2;
    }
    int grid_size = atoi(argv[1]);
    int max_steps = atoi(argv[2]);
    const char* bits = argv[3];
    int n = grid_size * grid_size;
    if ((int)strlen(bits) != n) {
        fprintf(stderr, "GRID_BITS must be %d chars, got %zu\n", n, strlen(bits));
        return 2;
    }

    LightsOut env;
    memset(&env, 0, sizeof(env));
    unsigned char* obs = (unsigned char*)calloc((size_t)n, 1);
    unsigned char* grid = (unsigned char*)calloc((size_t)n, 1);
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;

    // Static params (mirroring my_init, but grid_size is a CLI arg here).
    env.grid_size = grid_size;
    env.max_steps = max_steps;
    env.observation_size = n;
    env.num_agents = 1;
    env.client = NULL;             // headless: enables the timeout branch

    // Fixed initial state, mirroring init_lightsout's non-RNG part.
    env.grid = grid;
    env.step_count = 0;
    env.prev_action = -1;
    env.last_action = -1;
    env.episode_return = 0.0f;
    int lights_on = 0;
    for (int i = 0; i < n; i++) {
        unsigned char v = (bits[i] == '1') ? 1 : 0;
        grid[i] = v;
        lights_on += v;
    }
    env.lights_on = lights_on;

    printf("step\tstep_count\tlast_action\tprev_action\treward\tterminal\n");
    printf("0\t%d\t%d\t%d\t%.9g\t%d\n",
           env.step_count, env.last_action, env.prev_action,
           env.rewards[0], (int)env.terminals[0]);

    for (int i = 4; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%d\t%d\t%.9g\t%d\n",
               i - 3, env.step_count, env.last_action, env.prev_action,
               env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;   // internal init_lightsout re-scrambled; stop
    }

    // NB: do not call c_close (it frees env->grid); we free our own buffers.
    free(obs);
    free(grid);
    return 0;
}
