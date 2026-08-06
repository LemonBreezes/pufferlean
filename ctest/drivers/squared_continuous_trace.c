// Headless trace driver for ocean/squared_continuous. Sets up the grid directly
// (AGENT at center, TARGET at TARGET_IDX), bypassing c_reset's rand_r target
// placement, and stops at the first terminal (whose c_reset re-randomizes). Emits
// the deterministic r/c/reward/terminal fields.
//
// Continuous actions: two floats per step, (vertical, horizontal). Each is stored
// into env.actions[0]/[1] as `float` (binary32) exactly as PufferLib expects, so
// the ±0.25 threshold decision matches the Lean model's Float32-rounded compare.
//
// Usage: squared_continuous_trace SIZE TARGET_IDX VERT HORIZ [VERT HORIZ ...]
//   vertical:   > 0.25 => DOWN (r+1), < -0.25 => UP (r-1), else stationary
//   horizontal: > 0.25 => RIGHT (c+1), < -0.25 => LEFT (c-1), else stationary
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "squared_continuous.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s SIZE TARGET_IDX [VERT HORIZ...]\n", argv[0]); return 2; }
    int size = atoi(argv[1]);
    int target_idx = atoi(argv[2]);
    Squared env;
    memset(&env, 0, sizeof(env));
    unsigned char* obs = (unsigned char*)calloc((size_t)(size * size), 1);
    float act[2] = {0.0f, 0.0f};
    float rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = act; env.rewards = &rew; env.terminals = &term;
    env.size = size; env.tick = 0; env.r = size / 2; env.c = size / 2;
    obs[(size * size) / 2] = 1;   // AGENT
    obs[target_idx] = 2;          // TARGET

    printf("step\tr\tc\treward\tterminal\n");
    printf("0\t%d\t%d\t%.9g\t%d\n", env.r, env.c, env.rewards[0], (int)env.terminals[0]);
    int step = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        env.actions[0] = (float)atof(argv[i]);      // vertical
        env.actions[1] = (float)atof(argv[i + 1]);  // horizontal
        c_step(&env);
        step++;
        printf("%d\t%d\t%d\t%.9g\t%d\n", step, env.r, env.c, env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }
    free(obs);
    return 0;
}
