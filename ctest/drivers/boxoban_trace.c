// Headless trace driver for ocean/boxoban. Sets up the puzzle directly from a
// MAP string (bypassing c_reset's rand_r() puzzle pick + first-episode random
// tick), then runs scripted actions and prints the deterministic observable
// fields, stopping at the first terminal.
//
// We call the env's REAL transition core `take_action` (which drives the real
// clear/move_entity/get_entity logic, the actual meat of the env) and reproduce
// only c_step's thin wrapper (tick++, reward accumulation, terminal checks)
// inline. We deliberately do NOT call c_step itself, because on a terminal
// c_step invokes c_reset -> get_random_puzzle_idx, which `rand_r % PUZZLE_COUNT`
// with an unloaded map (PUZZLE_COUNT == 0) and memcpy from a NULL MAP_BASE. That
// reset also re-randomizes to a fresh puzzle, i.e. it is non-deterministic and
// unmodellable. Stopping at the terminal keeps the trace deterministic and lets
// us print the true terminal-time state (agent/on_target/tick) before any reset.
//
// The MAP is a size*size string laid out row-major (y outer, x inner) using the
// standard Boxoban glyphs (same as parse_maps.h):
//   '#' wall   '$' box   '.' target   '*' box-on-target
//   '@' agent  '+' agent-on-target    any other char (e.g. '-' or ' ') = floor
//
// Usage: boxoban_trace SIZE MAX_STEPS INT_R_COEFF TARGET_LOSS_PEN_COEFF MAP ACTION...
//        actions: 0=NOOP 1=DOWN 2=UP 3=LEFT 4=RIGHT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BOXOBAN_MAPS_IMPLEMENTATION  // supply MAP_BASE/PUZZLE_* + map fns for the
                                     // c_reset/init defs pulled in by boxoban.h
#include "boxoban.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s SIZE MAX_STEPS INT_R_COEFF TARGET_LOSS_PEN_COEFF MAP [ACTION...]\n",
            argv[0]);
        return 2;
    }
    int size = atoi(argv[1]);
    int max_steps = atoi(argv[2]);
    float int_r_coeff = (float)atof(argv[3]);
    float target_loss_pen_coeff = (float)atof(argv[4]);
    const char* map = argv[5];
    int cells = size * size;
    if ((int)strlen(map) != cells) {
        fprintf(stderr, "MAP has %zu chars, expected %d (SIZE*SIZE)\n", strlen(map), cells);
        return 2;
    }

    Boxoban env;
    memset(&env, 0, sizeof(env));

    // Observation planes: AGENT=0, WALLS=1, BOXES=2, TARGET=3.
    unsigned char* obs = (unsigned char*)calloc((size_t)(4 * cells), 1);
    unsigned char* int_r = (unsigned char*)calloc((size_t)cells, 1);
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;
    env.intermediate_rewards = int_r;
    env.size = size;
    env.num_agents = 1;
    env.max_steps = max_steps;
    env.int_r_coeff = int_r_coeff;
    env.target_loss_pen_coeff = target_loss_pen_coeff;
    env.initialized = true;
    env.win = 0;
    env.episode_return = 0;
    env.tick = 0;

    // Decode the MAP into the four planes + meta, exactly like parse_maps.h.
    int agent_x = -1, agent_y = -1;
    int n_boxes = 0, n_targets = 0, on_target = 0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            char ch = map[y * size + x];
            int is_agent = (ch == '@' || ch == '+');
            int is_wall = (ch == '#');
            int is_box = (ch == '$' || ch == '*');
            int is_target = (ch == '.' || ch == '*' || ch == '+');
            if (is_agent) { agent_x = x; agent_y = y; }
            n_boxes += is_box;
            n_targets += is_target;
            on_target += (is_box && is_target);
            obs[AGENT * cells + y * size + x] = (unsigned char)is_agent;
            obs[WALLS * cells + y * size + x] = (unsigned char)is_wall;
            obs[BOXES * cells + y * size + x] = (unsigned char)is_box;
            obs[TARGET * cells + y * size + x] = (unsigned char)is_target;
        }
    }
    env.agent_x = agent_x;
    env.agent_y = agent_y;
    env.n_boxes = n_boxes;
    env.n_targets = n_targets;
    env.on_target = on_target;

    // c_reset seeds intermediate_rewards from the TARGET plane; mirror that.
    memcpy(env.intermediate_rewards, obs + TARGET * cells, (size_t)cells);

    printf("step\tagent_x\tagent_y\ton_target\ttick\treward\tterminal\n");
    printf("0\t%d\t%d\t%d\t%d\t%.9g\t%d\n",
           env.agent_x, env.agent_y, env.on_target, env.tick,
           env.rewards[0], (int)env.terminals[0]);

    for (int i = 6; i < argc; i++) {
        // --- inline c_step wrapper (stops before c_step's internal c_reset) ---
        env.tick += 1;
        env.terminals[0] = 0;
        env.rewards[0] = 0.0f;
        int action = atoi(argv[i]);
        float on_target_before = env.on_target;
        int int_r_new = take_action(&env, action);   // REAL env transition core
        float on_target_after = env.on_target;
        env.rewards[0] += (float)int_r_new * env.int_r_coeff;
        if (on_target_after < on_target_before) {
            env.rewards[0] -= env.target_loss_pen_coeff;
        }
        if (env.on_target == env.n_targets) {
            env.terminals[0] = 1;
            env.rewards[0] += 1.0f;
        } else if (env.tick >= env.max_steps) {
            env.terminals[0] = 1;
            env.rewards[0] -= 1.0f;
        }
        // ---------------------------------------------------------------------
        printf("%d\t%d\t%d\t%d\t%d\t%.9g\t%d\n",
               i - 5, env.agent_x, env.agent_y, env.on_target, env.tick,
               env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }

    free(obs);
    free(int_r);
    return 0;
}
