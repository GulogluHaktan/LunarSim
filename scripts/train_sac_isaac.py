"""SAC training against `lunarsim.adapters.isaac.isaac_lander_vec_env.
IsaacLanderVecEnv` -- the real-PhysX-backed, parallel, per-episode-terrain
lander env, not the hand-integrated `AnalyticLanderEnv`
`scripts/train_sac_curriculum.py` trains against.

Parallelism: `--n-envs` really does run N vehicles simultaneously in the
ONE live PhysX scene (see `IsaacLanderVecEnv`'s module docstring -- this is
intra-process PhysX batching, not N Isaac Sim processes), each on its own
terrain tile, each independently re-generating that tile via `tile_fn`
every time IT resets (not synchronized across envs) -- the real-physics
analogue of `train_sac_curriculum.py`'s `tile_fn`-per-episode curriculum.

Run: /home/haktan/isaac-env/bin/python scripts/train_sac_isaac.py --steps-per-stage 50000
(or the equivalent `/isaac-sim/python.sh` inside the docker image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser.add_argument("--steps-per-stage", type=int, default=50_000)
parser.add_argument("--n-envs", type=int, default=16,
                     help="real parallel vehicles in the one PhysX scene -- keep this within what the "
                          "local GPU's VRAM/compute can hold; there's no subprocess-per-env cost to trade off "
                          "against here (see IsaacLanderVecEnv's module docstring), so scale it up until step "
                          "throughput stops improving, not to match core count.")
parser.add_argument("--out-dir", type=str, default="out")
parser.add_argument("--terrain-seed", type=int, default=7)
parser.add_argument("--terrain-grid-n", type=int, default=80,
                     help="heightfield grid edge length; res_m = tile_size_m / this. 80 is what every "
                          "run on disk used -- see lunarsim/rl/curriculum.py's _DEFAULT_TERRAIN_GRID_N "
                          "comment for why 160 is probably right and what it costs.")
parser.add_argument("--max-rocks-per-env", type=int, default=10,
                     help="real collidable rocks spawned per env per reset (from the terrain's own "
                          "tile.rocks field) -- capped since a full reference-density rock field can be "
                          "dozens-hundreds of prims, and this authors them fresh every episode reset.")
parser.add_argument("--only-stage", type=str, default=None,
                     help="run one stage, or a comma-separated list of stages in order")
parser.add_argument("--warm-start", type=str, default=None,
                     help="load this checkpoint .zip as the starting model instead of a fresh one")
parser.add_argument("--keep-buffer", action="store_true",
                     help="don't reset the replay buffer at stage boundaries -- see "
                          "train_sac_curriculum.py's comment on why this defaults off")
parser.add_argument("--demo-path", type=str, default=None,
                     help="seed the replay buffer with ZemZevController demo transitions from "
                          "scripts/collect_zemzev_demos.py before learn() starts on the FIRST stage only "
                          "(see _seed_replay_buffer_from_demos's docstring). NOTE: the replay buffer is "
                          "reset at every stage boundary unless --keep-buffer, so with a multi-stage run "
                          "these demos are discarded after stage 1 -- and they are orbit_descent-release "
                          "demos, which do not match ramp_20m's distribution anyway. Intended for "
                          "--only-stage orbit_descent.")
parser.add_argument("--demo-repeat", type=int, default=20,
                     help="repeat the demo file's transitions this many times when seeding, so they aren't "
                          "diluted to a negligible fraction of the buffer by --steps-per-stage online steps "
                          "(see _seed_replay_buffer_from_demos's field comment for the real run this fixes)")
parser.add_argument("--gradient-steps", type=int, default=-1,
                     help="SAC gradient steps per rollout. -1 means 'one per environment transition "
                          "collected', which is what plain single-env SAC does and what this needs. "
                          "See the REAL BUG note at the SAC(...) call site: SB3's default of 1, with "
                          "train_freq=1 and --n-envs parallel envs, silently trains at a 1/n_envs "
                          "replay ratio.")
parser.add_argument("--stage-warmup-steps", type=int, default=10_000,
                     help="uniform-random transitions to collect at the START OF EVERY STAGE before "
                          "any gradient step is taken -- see the REAL BUG note where this is applied. "
                          "~10k is about one episode per env at --n-envs 16.")
parser.add_argument("--torch-device", type=str, default="auto",
                     help="device for SAC's own MLPs (NOT the physics backend, which is PhysX's "
                          "own). 'auto' picks cuda when available. See the REAL BUG note at the "
                          "SAC(...) call site: with --gradient-steps -1 the gradient steps, not "
                          "the simulation, are the wall-clock bottleneck.")
parser.add_argument("--gamma", type=float, default=0.999,
                     help="SAC discount factor. NOT SB3's 0.99 default -- see the REAL BUG note at "
                          "the SAC(...) call site: at dt_s=0.05 a gamma of 0.99 is a 5-second "
                          "horizon against 20-60 second episodes, which makes the terminal landing "
                          "bonus invisible and stalling the optimal policy.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

# MEASURED THIS SESSION (two throughput points off the same config, one
# with --gradient-steps 1 and one with -1): at --n-envs 16 a rollout costs
# ~28.4 ms of env/PhysX time and ~4.25 ms per SAC gradient step, so with
# gradient_steps=-1 (16 steps per rollout) roughly 70% of wall-clock time
# is the MLP updates -- and `nvidia-smi` showed the GPU pinned at 0%
# utilization throughout, with the process using ~3 of 16 CPU cores. The
# simulation stays on PhysX's own backend; this moves only SAC's networks.
# Mathematically neutral (identical updates, different device), so it does
# not change what a run converges to -- only how long it takes.
if args.torch_device == "auto":
    import torch as _torch
    args.torch_device = "cuda" if _torch.cuda.is_available() else "cpu"

simulation_app = SimulationApp({"headless": args.headless})

sys.path.insert(0, args.lunarsim_root)

from stable_baselines3 import SAC  # noqa: E402

from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
# the stage table, the terrain config and the tile_fn all live in ONE place
# now -- see lunarsim/rl/curriculum.py's docstring for the four drifted
# copies that motivated it.
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.reward import RewardWeights  # noqa: E402


def _seed_replay_buffer_from_demos(model: "SAC", demo_path: str, n_envs: int, n_repeats: int = 1) -> None:
    """Push `scripts/collect_zemzev_demos.py`'s saved real-Isaac-Sim demo
    transitions into `model`'s replay buffer before any learning happens,
    so the critic sees at least some real successful (and failed) landing
    trajectories instead of relying entirely on the policy's own
    exploration to stumble into one (see collect_zemzev_demos.py's
    docstring for why this was added this session).

    Expects that script's FLAT v2 format: arrays shaped (N_transitions, ...)
    which get chunked into `n_envs`-sized `ReplayBuffer.add` calls here.
    The old v1 (T, group_size, ...) lane format is REFUSED rather than
    silently loaded -- it truncated every episode group to the shortest
    episode in it and so shipped only 4 terminal transitions in 39824 rows
    (see collect_zemzev_demos.py's docstring for the full finding). Every
    v1 file on disk has that defect; they must be re-collected.
    """
    data = np.load(demo_path, allow_pickle=False)
    version = int(data["format_version"]) if "format_version" in data else 1
    if version < 2:
        raise SystemExit(
            f"{demo_path} is a v1 demo file (lane-packed, group_size={int(data['group_size'])}). That "
            f"format truncated every episode group to its shortest episode and therefore dropped almost "
            f"all terminal transitions -- re-collect with scripts/collect_zemzev_demos.py.")

    stored_weights = str(data["reward_weights"])
    current_weights = json.dumps(asdict(RewardWeights()), sort_keys=True)
    if stored_weights != current_weights:
        raise SystemExit(
            f"{demo_path} stores rewards computed under a DIFFERENT RewardWeights than the one this run "
            f"will optimize -- seeding them would train the critic toward a reward function that no "
            f"longer exists. Re-collect the demos, or pass --demo-path '' to train without them.\n"
            f"  file:    {stored_weights}\n  current: {current_weights}")

    obs, next_obs = data["obs"], data["next_obs"]
    actions, rewards, dones, truncated = data["actions"], data["rewards"], data["dones"], data["truncated"]
    n_chunks = obs.shape[0] // n_envs  # the tail that doesn't fill a chunk is dropped
    if n_chunks == 0:
        raise SystemExit(f"{demo_path} holds only {obs.shape[0]} transitions, fewer than --n-envs={n_envs}")

    # REAL BUG FOUND (this session, a real 2M-step demo-bootstrapped training
    # run that still reached 0/16 landed_safely despite 39824 seeded demo
    # transitions): demo transitions are never evicted (buffer_size=1e6,
    # n_envs=16 means a 2M-timestep run only ever makes ~125000 add() calls,
    # far short of wrapping), but they get DILUTED -- 2489 demo add() calls
    # vs. ~125000 online ones leaves demos at under 2% of the final buffer,
    # and SAC samples minibatches uniformly at random, so that ~2% signal
    # gets swamped by the much larger volume of self-generated "hover and
    # wait" transitions as training proceeds. Repeating the demo add() calls
    # `n_repeats` times raises that floor (e.g. 20 repeats ~= 2489*20 ~=
    # 49780 demo entries against the same ~125000 online entries ~= 28% of
    # the final buffer, and a much higher fraction for all of early/mid
    # training before the online count catches up) without needing a
    # custom prioritized-replay sampler.
    #
    # NOTE for whoever reads the run that first used this: that run's own
    # demo file turned out to contain 4 terminal transitions in total (see
    # the docstring above), so "demo bootstrapping didn't help" is NOT yet
    # evidence about demo bootstrapping -- it was never actually tried with
    # completed landings in the buffer.
    n_terminal = 0
    for _ in range(max(1, n_repeats)):
        for c in range(n_chunks):
            sl = slice(c * n_envs, (c + 1) * n_envs)
            infos = [{"TimeLimit.truncated": bool(v)} for v in truncated[sl]]
            model.replay_buffer.add(obs[sl], next_obs[sl], actions[sl], rewards[sl], dones[sl], infos)
        n_terminal = int(dones[: n_chunks * n_envs].sum())
    print(f"seeded replay buffer with {n_chunks * n_envs * max(1, n_repeats)} demo transitions "
          f"({n_chunks} add() calls x {n_envs} lanes x {max(1, n_repeats)} repeats, "
          f"{n_terminal} terminal transitions per repeat) from {demo_path}")


def sanity_rollout(model, venv, stage: Stage, args, max_steps: int = 2000) -> None:
    """Deterministic 1-episode-per-env rollout on `stage`'s own release
    condition, reporting the outcome counts that actually define success.
    `left_tile` is reported separately from `truncated` on purpose: an
    episode that flew off the terrain collision mesh is not a timeout, and
    conflating the two is what made "ran the full clock" look like a
    policy choice in earlier runs when it was partly a map-size artifact.
    """
    venv.params = stage.params
    venv.tile_fn = make_tile_fn(stage, args.terrain_grid_n)
    obs = venv.reset()
    total_reward = np.zeros(args.n_envs)
    done_once = np.zeros(args.n_envs, dtype=bool)
    counts = {"landed_safely": 0, "lost_control": 0, "left_tile": 0, "other": 0}
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        venv.step_async(action)
        obs, rewards, dones, infos = venv.step_wait()
        total_reward += rewards * (~done_once)
        for i in range(args.n_envs):
            if dones[i] and not done_once[i]:
                done_once[i] = True
                info = infos[i]
                if info.get("landed_safely"):
                    counts["landed_safely"] += 1
                elif info.get("lost_control"):
                    counts["lost_control"] += 1
                elif info.get("left_tile"):
                    counts["left_tile"] += 1
                else:
                    counts["other"] += 1
                print(f"[{stage.name}] env {i} ended: landed_safely={info.get('landed_safely')}, "
                      f"lost_control={info.get('lost_control')}, left_tile={info.get('left_tile')}, "
                      f"reward={total_reward[i]:.1f}, t_s={info.get('t_s'):.1f}")
        if done_once.all():
            break
    print(f"=== SANITY {stage.name}: landed_safely={counts['landed_safely']}/{args.n_envs} "
          f"lost_control={counts['lost_control']} left_tile={counts['left_tile']} "
          f"other={counts['other']} mean_reward={total_reward.mean():.1f} ===")


def main():
    if args.only_stage is None:
        stages = list(STAGES)
    else:
        wanted = args.only_stage.split(",")
        unknown = [n for n in wanted if n not in STAGES_BY_NAME]
        if unknown:
            raise SystemExit(f"no stage(s) named {unknown!r}; choices: {list(STAGES_BY_NAME)}")
        stages = [STAGES_BY_NAME[n] for n in wanted]

    model = None
    venv = None
    for stage in stages:
        print(f"=== stage: {stage.name} (tile={stage.tile_size_m:.0f} m, alt={stage.params.spawn_altitude_m:.0f} m, "
              f"h_speed={stage.params.spawn_horizontal_speed_m_s} m/s, n_envs={args.n_envs}) ===")
        tile_fn = make_tile_fn(stage, args.terrain_grid_n)
        if venv is None:
            # the PhysX scene (N vehicles + N terrain slots) is built once,
            # here, by the first IsaacLanderVecEnv -- later stages reuse the
            # SAME instance (swap `.params`/`.tile_fn`; env spacing has to
            # fit the LARGEST stage's tile, so it's sized off STAGES, not
            # just this stage, or a later bigger-tile stage would overlap
            # neighboring env slots).
            max_tile = max(s.tile_size_m for s in STAGES)
            venv = IsaacLanderVecEnv(
                num_envs=args.n_envs, tile_fn=tile_fn, params=stage.params,
                env_spacing_m=max_tile * 1.5, seed=args.terrain_seed,
                max_rocks_per_env=args.max_rocks_per_env,
            )
        else:
            venv.params = stage.params
            venv.tile_fn = tile_fn
            venv.reset()

        first_model_creation = model is None
        if model is None:
            if args.warm_start:
                model = SAC.load(args.warm_start, env=venv, device=args.torch_device)
                # a checkpoint saved before --gradient-steps existed carries
                # SB3's default of 1; honour the CLI either way (see the
                # REAL BUG note on the fresh-model branch below).
                model.gradient_steps = args.gradient_steps
                # same for gamma: every checkpoint on disk was saved with
                # SB3's 0.99 default baked in, and loading one silently
                # restores it. See the REAL BUG note on the fresh-model
                # branch below for why that single number decided the
                # outcome of every run so far.
                model.gamma = args.gamma
                print(f"warm-started from {args.warm_start} "
                      f"(gamma={model.gamma}, gradient_steps={model.gradient_steps})")
                if not args.keep_buffer:
                    model.replay_buffer.reset()
            else:
                # REAL BUG FOUND (this session, reading SB3 2.9.0's
                # off_policy_algorithm.py against the real run config):
                # `gradient_steps` defaults to 1 and `train_freq` to 1 step.
                # With a VecEnv, ONE `train_freq=1` rollout collects
                # `num_envs` transitions (`RolloutReturn(num_collected_steps
                # * env.num_envs, ...)`, line 608) but still takes exactly
                # ONE gradient step. At --n-envs 16 that is a replay ratio
                # of 1/16, i.e. a 2M-timestep run performs only 125000
                # gradient updates -- the learning that a 125k-step
                # single-env SAC run would do, not a 2M-step one. Every
                # "it converged to a local optimum / ent_coef went flat"
                # conclusion in handover.md was drawn from runs that had
                # had 16x fewer updates than their step counts suggest.
                # gradient_steps=-1 makes SB3 use `rollout.episode_timesteps`
                # (= num_envs) instead, restoring the 1:1 ratio SAC is
                # designed around. Cost is bounded: at the ~2.9 vectorized
                # steps/s this env actually runs at, that is ~46 extra
                # MLP gradient steps/s on CPU.
                # REAL BUG FOUND (this session, scoring four strategy
                # archetypes end-to-end under the real reward -- see
                # reward.py's `reward_scale` field comment for the full
                # table): SB3's default gamma=0.99 at this env's dt_s=0.05
                # is an effective horizon of 1/(1-gamma) = 100 steps = 5.0
                # SECONDS, against episodes that run 20-60 seconds. A
                # touchdown 700 steps (35 s) after release was discounted
                # by 0.99**700 = 8.8e-4, so `landing_bonus_scale`=1800 was
                # worth 1.58 return units at the release point -- about 1%
                # of the decision. Discounted returns at the shipped
                # coefficients:
                #     gamma=0.99 : land -198.2  STALL -146.9  climb -194.1
                #     gamma=0.999: land +159.1  stall -1343.3 climb -1597.6
                # i.e. at gamma=0.99 HOVERING AT 100 m WAS THE OPTIMAL
                # POLICY, scoring better than a successful landing, with
                # climbing away tied with landing -- which is exactly the
                # behavior every run on record produced (0/16 landed_safely,
                # 11-14/16 running the full clock, telemetry showing
                # full-throttle climbs to 680-1372 m). A joint gamma/
                # coefficient sweep confirmed gamma is the binding
                # constraint, not the coefficients: at gamma=0.99 stalling
                # wins for EVERY coefficient variant tried, and at
                # gamma=0.999 landing wins for every one of them.
                # gamma=0.9995 was also tried and rejected -- it inverts
                # the required crash-worse-than-timeout ordering.
                # 0.999 = a 1000-step / 50-second horizon, matched to the
                # episode length. If `max_episode_s` or `dt_s` ever change,
                # this must move with them.
                model = SAC("MlpPolicy", venv, verbose=1, device=args.torch_device,
                             gamma=args.gamma, gradient_steps=args.gradient_steps)
        else:
            model.set_env(venv)
            if not args.keep_buffer:
                model.replay_buffer.reset()

        # REAL BUG FOUND (this session, reading SB3 2.9.0 against the real
        # multi-stage run): `learning_starts` is compared against the GLOBAL
        # `num_timesteps`, which `learn(..., reset_num_timesteps=False)`
        # deliberately carries across stages -- but the replay buffer is
        # reset at every stage boundary (just above). So from stage 2
        # onwards there is no uniform-random warmup at all AND `train()` is
        # called on the very first rollout, i.e. SAC fits its critic by
        # sampling 256-element minibatches (with replacement) out of a
        # buffer holding exactly `--n-envs` transitions, on a brand-new
        # state distribution it has never seen. Re-arming `learning_starts`
        # relative to the current step count gives every stage the same
        # warmup the first one gets.
        model.learning_starts = model.num_timesteps + args.stage_warmup_steps

        if args.demo_path and first_model_creation:
            _seed_replay_buffer_from_demos(model, args.demo_path, args.n_envs, args.demo_repeat)

        model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False)
        out_path = f"{args.out_dir}/sac_lunar_lander_isaac_{stage.name}.zip"
        model.save(out_path)
        print(f"saved: {out_path}")
        # Measure THIS stage on ITS OWN release condition before moving on.
        # Previously the only rollout in this script ran once, at the very
        # end, on the hardest stage only -- so a multi-hour curriculum run
        # produced no landed_safely signal at all until it was over, and a
        # stage that had not been learned was warm-started into the next one
        # regardless (stage advancement is a fixed step budget, it is not
        # gated on competence). This does not gate anything either; it just
        # makes the failure visible at the stage where it happens instead of
        # three hours later.
        sanity_rollout(model, venv, stage, args)

    sanity_rollout(model, venv, stages[-1], args)


if __name__ == "__main__":
    main()
    simulation_app.close()
