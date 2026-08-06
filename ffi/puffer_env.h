/* puffer_env.h — the standard C env ABI for pufferlean plugin envs.
 *
 * Each env lives in its own folder (envs/<name>/) and compiles to a standalone shared
 * library libenv_<name>.so that exports these six symbols. `puffer` NEVER sees any env at
 * compile time — it dlopen's the .so by name at runtime (ffi/puffer_loader.c) and drives it
 * through this ABI. This mirrors PufferLib's native C (Ocean) envs: the env steps N copies in
 * a tight C loop over flat double buffers (no interpreter, no per-step allocation), while the
 * GPU runs only the policy. Each env is verified bit-exact against its Lean reference model.
 *
 * Buffers are caller-owned, row-major, contiguous:
 *   obs      : N * obs_dim   (per-env observation)
 *   actions  : N             (discrete action index, as double)
 *   rewards  : N
 *   terminals: N             (0.0 / 1.0)
 * The env auto-resets a terminated copy in place on the same step (PufferLib semantics), so
 * `step` always returns a fresh live observation for every env.
 */
#ifndef PUFFER_ENV_H
#define PUFFER_ENV_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Create a vectorized env of `num_envs` copies. `config` is a "k=v,k=v" string (env-specific
 * keys, e.g. "size=5,target=14" or "frameskip=4"); `seed` bases any per-env RNG. Returns an
 * opaque handle owned by the caller (freed via puffer_env_free). */
void* puffer_env_make(int num_envs, uint64_t seed, const char* config);

/* Report the per-agent obs size, discrete action count, episode step cap, and the number of agents PER
 * env instance (num_agents). An env with num_agents>1 emits num_agents obs/rewards/terminals per step and
 * consumes num_agents actions; the trainer flattens them so the batch is num_envs·num_agents rows. */
void  puffer_env_spec(void* env, int* obs_dim, int* num_actions, int* max_steps, int* num_agents);

/* Action-space structure: `n_heads` categorical heads with sizes head_sizes[0..n_heads-1] (caller buffer,
 * ≥16). Single discrete = 1 head; multi-discrete = k heads (PufferLib ACT_SIZES). The policy outputs
 * Σ head_sizes logits + 1 value; the trainer samples/scores each head. */
void  puffer_env_actinfo(void* env, int* n_heads, int* head_sizes);

/* Optional: return 1 if the action space is CONTINUOUS (diagonal-Gaussian, PufferLib Box / ACT_SIZES all
 * 1s), else 0. When 1, n_heads (from actinfo) is the real action dim d, the policy head is 2·d+1 (means,
 * logstds, value), and the trainer runs the Gaussian PPO path. Absent symbol ⇒ discrete (loader defaults 0). */
int   puffer_env_iscont(void* env);

/* Reset all copies; write the initial observations into obs[num_envs * obs_dim]. */
void  puffer_env_reset(void* env, double* obs);

/* Step all copies with actions[num_envs]; write next obs, rewards, terminals. Auto-resets
 * terminated copies in place. */
void  puffer_env_step(void* env, const double* actions, double* obs, double* rewards, double* terminals);

/* Release the handle. */
void  puffer_env_free(void* env);

/* ---- OPTIONAL log channel (metrics only — never touches obs/rewards/terminals/actions) --------------
 *
 * PufferLib's ocean envs each keep their OWN `Log` struct on the env (`env->log`), whose fields differ
 * per env in both NAME and ORDER (breakout: perf,score,episode_return,episode_length,n; cartpole:
 * perf,episode_length,x_threshold_termination,pole_angle_termination,max_steps_termination,n,score).
 * The env's `add_log()` accumulates into it at each EPISODE end and bumps `log.n`; PufferLib reports
 * episode statistics from it (src/vecenv.h `static_vec_aggregate_logs` / `static_vec_log`), NOT by
 * summing rewards between terminal flags. The two are DIFFERENT UNITS, and the difference is not
 * cosmetic. Measured with a random policy via `puffer env-log <env>`: fourteen of the 39 built ocean
 * envs never raise a terminal flag at all (their `c_reset` runs inside `c_step`) — blastar, convert,
 * double_pendulum, drive, enduro, hex, minimal, moba, rware, snake, target, terraform, tower_climb,
 * whackamole — so reward-summing reports a permanent 0.0 for every one of them while their logs carry
 * real episode statistics (moba `episode_return` 105.2, target 1.0, snake −0.72, …). And even where
 * terminals DO exist the episode boundary can differ: trash_pickup logs 11.06 where reward-summing
 * says 1.38, tripletriad −3.50 vs −22.5, nmmo3 −1.00 vs −3.84, go −18.46 vs −17.04. This channel
 * carries the real thing, so our numbers are comparable with upstream PufferLib's.
 *
 * The per-env field variation is handled by exporting the log as a set of NAMED scalars: the plugin
 * publishes its own field names (from its own `Log`), and the caller looks up the ones it cares about by
 * NAME. A field an env does not have is simply absent from the list. All three symbols are OPTIONAL —
 * a plugin that exports none (or not all three) has no log channel, exactly like `puffer_env_iscont`.
 *
 * Aggregation is PufferLib's, exactly: sum each field over the env copies whose `log.n > 0`, DIVIDE
 * every field by the summed `n`, then ZERO every copy's log (so each call reports the window since the
 * previous call). The field literally named "n" reports the summed episode count itself, undivided —
 * matching `static_vec_log`'s `dict_set(out, "n", n)` after `my_log`.
 *
 * Not thread-safe against stepping: call between rollouts, never concurrently with puffer_env_step*. */

/* Optional: number of named log fields (>0). Absent symbol ⇒ no log channel (loader defaults 0). */
int         puffer_env_log_nfields(void* env);

/* Optional: name of field `i` in [0, nfields) — a static NUL-terminated string owned by the plugin,
 * valid for the plugin's lifetime. Out-of-range returns "". */
const char* puffer_env_log_name(void* env, int i);

/* Optional: aggregate + ZERO. Writes min(nfields, max) doubles into out[] (out[i] pairs with
 * puffer_env_log_name(env,i)) and returns `n`, the number of episodes completed since the previous
 * call. Returns 0.0 when no episode completed — out[] is then UNTOUCHED and the logs are NOT zeroed
 * (PufferLib's `if (n == 0) return;` early-out). */
double      puffer_env_log(void* env, double* out, int max);


#ifdef __cplusplus
}
#endif
#endif
