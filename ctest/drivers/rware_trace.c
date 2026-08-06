// Headless trace driver for ocean/rware (the Robotic Warehouse env), num_agents=1.
//
// Sets up a deterministic initial state directly (base map + one optional
// REQUESTED_SHELF cell, agent at a chosen pose, LCG seeded), bypassing c_reset's
// random agent placement / shelf requests, then applies the action list. The env
// has no terminal, so it runs the full list. Emits the deterministic observable
// fields plus a whole-grid checksum (so RNG-/movement-driven cell changes show).
//
// Usage: rware_trace MAP_CHOICE SEED START_LOC START_DIR START_STATE REQ_IDX ACTION [ACTION ...]
//   MAP_CHOICE: 1 tiny(11x10) 2 small(10x20) 3 medium(16x20)
//   SEED:       initial env->rng (unsigned)
//   START_LOC:  agent cell index
//   START_DIR:  0 right 1 down 2 left 3 up
//   START_STATE:0 unloaded 1 holding-requested 2 holding-empty
//   REQ_IDX:    cell forced to REQUESTED_SHELF, or -1 for none
//   actions:    0 NOOP 1 FORWARD 2 LEFT 3 RIGHT 4 TOGGLE_LOAD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rware.h"

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s MAP_CHOICE SEED START_LOC START_DIR START_STATE REQ_IDX [ACTION...]\n", argv[0]);
        return 2;
    }
    int map_choice = atoi(argv[1]);
    unsigned int seed = (unsigned int)strtoul(argv[2], NULL, 10);
    int start_loc = atoi(argv[3]);
    int start_dir = atoi(argv[4]);
    int start_state = atoi(argv[5]);
    int req_idx = atoi(argv[6]);

    CRware env;
    memset(&env, 0, sizeof(env));
    env.width = 640;
    env.height = 704;
    env.map_choice = map_choice;
    env.num_agents = 1;
    env.num_requested_shelves = 1;
    env.grid_square_size = 64;
    env.human_agent_idx = 0;
    env.reward_type = 2;

    allocate(&env);  // init() + obs/actions/rewards/terminals buffers

    // Clobber all reset randomization with a deterministic setup.
    int size = map_sizes[env.map_choice - 1];
    const int* base = maps[env.map_choice - 1];
    memcpy(env.warehouse_states, base, size * sizeof(int));
    if (req_idx >= 0) env.warehouse_states[req_idx] = REQUESTED_SHELF;
    env.agent_locations[0] = start_loc;
    env.old_agent_locations[0] = start_loc;
    env.agent_directions[0] = start_dir;
    env.agent_states[0] = start_state;
    env.scores[0] = 0.0f;
    env.rng = seed;
    reset_movement_graph(&env);

    printf("step\tloc\tdir\tstate\treward\tterminal\twsum\n");
    long wsum = 0;
    for (int i = 0; i < size; i++) wsum += (long)env.warehouse_states[i] * (i + 1);
    printf("0\t%d\t%d\t%d\t%.9g\t%d\t%ld\n",
           env.agent_locations[0], env.agent_directions[0], env.agent_states[0],
           env.rewards[0], (int)env.terminals[0], wsum);

    for (int a = 7; a < argc; a++) {
        env.actions[0] = (float)atoi(argv[a]);
        c_step(&env);
        wsum = 0;
        for (int i = 0; i < size; i++) wsum += (long)env.warehouse_states[i] * (i + 1);
        printf("%d\t%d\t%d\t%d\t%.9g\t%d\t%ld\n",
               a - 6, env.agent_locations[0], env.agent_directions[0], env.agent_states[0],
               env.rewards[0], (int)env.terminals[0], wsum);
    }
    return 0;
}
