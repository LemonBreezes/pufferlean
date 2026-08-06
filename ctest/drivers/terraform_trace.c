// Headless trace driver for ocean/terraform (single-agent bulldozer terraforming
// over a float heightmap; continuous Euler physics with sinf/cosf).
//
// Determinism: ALL of terraform's RNG is RESET-ONLY — perlin-noise map offsets,
// the dozer's rand_r placement, and the initial `tick` (all in init()/c_reset()).
// c_step's only in-episode RNG is "teleportitis" at tick % 512 == 0. We BYPASS the
// reset RNG the `squared` way: after allocate() (which allocates the arrays), we
// overwrite the map / target / dozer pose / deltas directly from (size, mode) via
// the fixed cellH/cellT formulas, set tick = 0, reset_frequency = 0, and keep every
// trajectory < 512 steps, so teleportitis never fires and no reset ever runs mid
// trace. Hence the whole trace is a deterministic function of (size, mode, start,
// actions), matched bit-for-bit by the Lean model.
//
// terraform's c_step never sets env->terminals; the episode "terminates" when the
// top-of-c_step check `current_total_delta < 0.01f` fires (reward +1, then the RNG
// c_reset). We detect that BEFORE calling c_step and emit a synthetic terminal row
// (reward 1.0, terminal 1, pose unchanged), then stop — the `squared` "stop at the
// first terminal which re-randomizes" rule.
//
// Usage: terraform_trace SIZE MODE START_X START_Y [ACCEL STEER BUCKET]...
//   accel/steer ∈ {0..4}; bucket ∈ {0=NOOP,1=LOAD,2=UNLOAD}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "terraform.h"

// Deterministic map / target formulas (must match Puffer.Env.Terraform.cellH/cellT).
static float cellH(int size, int mode, int r, int c) {
    if (mode == 0) return (float)((r + c) % 3);
    if (mode == 1) return (r == 4 && c == 6) ? 1.0f : 0.0f;
    if (mode == 2) return (c >= size / 2) ? 12.0f : 0.0f;
    return 0.0f;
}
static float cellT(int size, int mode, int r, int c) {
    (void)size;
    if (mode == 0) return (float)((r + c) % 2);
    return 0.0f;
}

static void print_row(Terraform* env, int step, float reward, int terminal) {
    Dozer* d = &env->dozers[0];
    printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
        step, d->x, d->y, d->v, d->heading, d->load, reward,
        env->current_total_delta, env->delta_progress, terminal);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s SIZE MODE START_X START_Y [ACCEL STEER BUCKET]...\n", argv[0]);
        return 2;
    }
    int size = atoi(argv[1]);
    int mode = atoi(argv[2]);
    int startx = atoi(argv[3]);
    int starty = atoi(argv[4]);

    Terraform env;
    memset(&env, 0, sizeof(env));
    env.size = size;
    env.num_agents = 1;
    env.reset_frequency = 0;
    env.reward_scale = 0.04f;

    // allocate obs/act/rew/term (as terraform.c's allocate() does) then init().
    // init() runs the (bypassed) perlin RNG and allocates orig_map/map/target_map/
    // grid_indices/quadrants/dozers sized by size & num_agents; we overwrite the
    // contents below. (allocate/free_allocated live in terraform.c, not the header.)
    env.observations = (float*)calloc(env.num_agents * 319, sizeof(float));
    env.actions = (float*)calloc(3 * env.num_agents, sizeof(float));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (float*)calloc(env.num_agents, sizeof(float));
    init(&env);

    // Overwrite the maps with the deterministic (size, mode) formulas.
    float itd = 0.0f;
    for (int i = 0; i < size * size; i++) {
        int r = i / size, c = i % size;
        float h = cellH(size, mode, r, c);
        float t = cellT(size, mode, r, c);
        env.orig_map[i] = h;
        env.map[i] = h;
        env.target_map[i] = t;
    }
    for (int i = 0; i < size * size; i++) {
        itd += fabsf(env.map[i] - env.target_map[i]);
    }
    env.initial_total_delta = itd;
    env.current_total_delta = itd;
    env.delta_progress = 0.0f;
    env.tick = 0;
    env.rng = 12345u;                 // unused (trajectory < 512, no teleportitis/reset)

    // Set the dozer pose directly (bypass the rand_r placement).
    Dozer* d = &env.dozers[0];
    d->x = (float)startx;
    d->y = (float)starty;
    d->v = 0.0f;
    d->heading = 0.0f;
    d->load = 0.0f;
    env.rewards[0] = 0.0f;
    env.terminals[0] = 0.0f;

    printf("step\tx\ty\tv\theading\tload\treward\tctd\tprogress\tterminal\n");
    print_row(&env, 0, 0.0f, 0);

    int nacts = (argc - 5) / 3;
    for (int k = 0; k < nacts; k++) {
        // Top-of-c_step reset check (reset_frequency == 0 → delta-only).
        if (env.current_total_delta < 0.01f) {
            print_row(&env, k + 1, 1.0f, 1);
            return 0;
        }
        env.actions[0] = (float)atoi(argv[5 + 3 * k]);       // accel
        env.actions[1] = (float)atoi(argv[6 + 3 * k]);       // steer
        env.actions[2] = (float)atoi(argv[7 + 3 * k]);       // bucket
        c_step(&env);
        print_row(&env, k + 1, env.rewards[0], (int)env.terminals[0]);
    }
    free_initialized(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
