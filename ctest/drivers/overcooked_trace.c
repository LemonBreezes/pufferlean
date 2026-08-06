// Headless trace driver for ocean/overcooked. Sets up a fixed layout with the
// C-default reward config (from overcooked.c) and num_agents=2, calls the real
// init()+c_reset() (fully deterministic — the env's only rand()/srand() live in
// the interactive main() demo, NOT in c_step/c_reset), then applies a flat list
// of actions (two per step: agent0 then agent1) and prints one TSV row per step.
//
// The env never sets terminals (no termination), so we simply run all steps.
// Only per-step reward is float; every other field is an exact integer.
//
// Usage: overcooked_trace LAYOUT_ID A0 A1 A0 A1 ...
//   LAYOUT_ID: 0=cramped_room 1=asymmetric_advantages 2=forced_coordination
//              3=coordination_ring 4=counter_circuit
//   actions:   0=NOOP 1=UP 2=DOWN 3=LEFT 4=RIGHT 5=INTERACT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "overcooked.h"

static void print_row(int step, Overcooked* env) {
    Agent* a0 = &env->agents[0];
    Agent* a1 = &env->agents[1];
    CookingPot* p = &env->cooking_pots[0];
    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%d\t%d\t%d\t%d\t%d\t%d\n",
        step,
        (int)a0->x, (int)a0->y, a0->facing_direction, a0->held_item,
        a0->held_soup_onions, a0->held_soup_tomatoes, a0->held_soup_total, a0->ticks_since_reward,
        (int)a1->x, (int)a1->y, a1->facing_direction, a1->held_item,
        a1->held_soup_onions, a1->held_soup_tomatoes, a1->held_soup_total, a1->ticks_since_reward,
        env->rewards[0], env->rewards[1],
        p->cooking_state, p->cooking_progress, p->ingredient_count, p->num_onions, p->num_tomatoes,
        env->num_items);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s LAYOUT_ID [A0 A1 ...]\n", argv[0]); return 2; }
    LayoutType layout_id = (LayoutType)atoi(argv[1]);
    int num_agents = 2;
    int num_obs = 43;

    Overcooked env;
    memset(&env, 0, sizeof(env));
    env.layout_id = layout_id;
    env.num_agents = num_agents;
    env.grid_size = 100;
    env.observation_size = num_obs;
    env.rewards_config.dish_served_whole_team = 1.0f;
    env.rewards_config.dish_served_agent = 0.0f;
    env.rewards_config.pot_started = 0.15f;
    env.rewards_config.ingredient_added = 0.15f;
    env.rewards_config.ingredient_picked = 0.05f;
    env.rewards_config.plate_picked = 0.05f;
    env.rewards_config.soup_plated = 0.20f;
    env.rewards_config.wrong_dish_served = 0.0f;
    env.rewards_config.step_penalty = 0.0f;

    env.observations = (float*)calloc(num_obs * num_agents, sizeof(float));
    env.actions = (float*)calloc(num_agents, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (float*)calloc(num_agents, sizeof(float));

    init(&env);
    c_reset(&env);

    printf("step\ta0x\ta0y\ta0f\ta0h\ta0so\ta0st\ta0tt\ta0tk\ta1x\ta1y\ta1f\ta1h\ta1so\ta1st\ta1tt\ta1tk\tr0\tr1\tpst\tppr\tpic\tpon\tpto\tnit\n");
    print_row(0, &env);

    int nsteps = (argc - 2) / 2;
    for (int s = 0; s < nsteps; s++) {
        env.actions[0] = (float)atoi(argv[2 + s * 2]);
        env.actions[1] = (float)atoi(argv[2 + s * 2 + 1]);
        c_step(&env);
        print_row(s + 1, &env);
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
