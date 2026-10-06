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
parser.add_argument("--tau", type=float, default=0.001,
                     help="SAC target-network Polyak rate. NOT SB3's 0.005 default -- see the REAL "
                          "BUG note at the SAC(...) call site: 0.005 gives the target net a 200-step "
                          "time constant, against the 1000-step bootstrap horizon gamma=0.999 implies. "
                          "SB3's default is tuned for gamma=0.99, where 200 > 100 and the target is "
                          "the SLOWER of the two; at our gamma that ordering inverts.")
parser.add_argument("--ent-coef", type=str, default="0.05",
                     help="SAC entropy temperature. NOT SB3's 'auto' default -- see the REAL BUG "
                          "note at the SAC(...) call site: auto-tuning ran away from 0.635 to 6.6 "
                          "over 268k steps and took critic_loss from 7e3 to 3e6 with it. 0.05 keeps "
                          "the entropy term at ~9% of the measured Q spread (~1074 units), against the "
                          "16x-of-spread the runaway reached; 0.01 was numerically safe but left the "
                          "Q term outweighing entropy ~1e4:1, i.e. sigma->0 and SAC degenerates to TD3 "
                          "without exploration noise -- the wrong side to err on for a task whose "
                          "standing failure is never sampling a landing. Accepts a float, or "
                          "'auto'/'auto_<init>' to hand control back to the tuner.")
parser.add_argument("--layer-norm", action="store_true",
                     help="weave LayerNorm into the actor and critic trunks. The "
                          "Klein et al. (2024) plasticity survey finds general "
                          "regularisation usually beats domain-specific fixes, and "
                          "Lyle et al. (2024) report normalisation also fights "
                          "overestimation. Fresh models only -- it reshapes the nets.")
parser.add_argument("--reset-every", type=int, default=0,
                     help="periodically reset part of the agent (Nikishin et al. 2022). "
                          "Pair with --checkpoint-every: the steps right after a reset "
                          "are expected to be bad, so snapshots are what you evaluate.")
parser.add_argument("--reset-alpha", type=float, default=1.0,
                     help="1.0 = full reinit of the scope, <1 pulls partway toward it "
                          "(Calibrated Partial Resets, 2026, avoids the collapse that "
                          "full reinit can cause).")
parser.add_argument("--reset-scope", type=str, default="last",
                     choices=["last", "critic", "all"])
parser.add_argument("--save-buffer", action="store_true",
                     help="save the replay buffer beside each checkpoint, and restore it "
                          "on --warm-start. Without this the buffer is EMPTY after a warm "
                          "start (SAC.load does not restore one), so the critic refits "
                          "from a narrow early window -- textbook primacy bias.")
parser.add_argument("--n-critics", type=int, default=2,
                     help="size of the SAC critic ensemble. SB3's default is 2 and it "
                          "takes the min, which already damps overestimation; more "
                          "critics damp it further (REDQ uses 5-10). NOTE: four "
                          "experiments in handover.md exonerate overestimation as the "
                          "cause of the mid-run collapse, so this is a last resort, not "
                          "the obvious next knob. Changing it reshapes the critic, so a "
                          "warm start cannot load directly -- see _expand_critic_ensemble, "
                          "which keeps the trained actor and clones the trained critics "
                          "rather than starting from scratch.")
parser.add_argument("--learning-rate", type=float, default=3e-4,
                     help="SB3's default is 3e-4, which is sized for training from "
                          "scratch. Fine-tuning an already-good policy at that rate "
                          "destroyed one: 600k steps from a checkpoint measured at "
                          "22.9%% (96 episodes) ended at 0/24 on five consecutive "
                          "snapshots, with Q inflating 151 -> 403 and critic_loss "
                          "14 -> 298. Lower it when continuing from a good policy.")
parser.add_argument("--reward-weight", type=str, action="append", default=None,
                     metavar="NAME=VALUE",
                     help="override one RewardWeights field, repeatable. Exists so a "
                          "reward A/B is a reproducible command rather than an edit to "
                          "the source: e.g. --reward-weight profile_k=35 "
                          "--reward-weight kxy_knee_m_s=1e9 reverts the vertical and "
                          "lateral fixes without touching reward.py.")
parser.add_argument("--checkpoint-every", type=int, default=0,
                     help="also save a checkpoint every N timesteps, not just at stage end. "
                          "MEASURED reason this exists: performance across runs goes up then "
                          "DOWN (42%% -> 8%% on one stage with 500k more steps), and every stage "
                          "result so far is the END of a run. If the peak is mid-run, saving only "
                          "the last step systematically keeps the over-trained tail. 0 disables.")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()


def _parse_ent_coef(raw: str):
    """SB3 takes `ent_coef` as either the string 'auto'/'auto_<init>' or a
    float. Keep both reachable from one CLI string."""
    return raw if raw.startswith("auto") else float(raw)

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

from stable_baselines3 import SAC
from stable_baselines3.common.utils import FloatSchedule  # noqa: E402

def _expand_critic_ensemble(old_model, venv, args, n_critics):
    """Warm start into a LARGER critic ensemble without losing the actor.

    Changing `n_critics` reshapes the critic, so `SAC.load` cannot restore a
    2-critic checkpoint into a 5-critic model. Training from scratch instead
    would throw away the only policy this project has that lands (22.9% on
    ramp_35m), which is not necessary: the actor is shape-compatible, and the
    extra critics can be CLONED from the trained ones rather than initialised
    randomly.

    Cloning matters. SAC takes the MIN over the ensemble, so a freshly
    initialised critic would dominate that min with untrained garbage and
    bootstrap it into the target -- the new critics have to start out agreeing
    with the trained ones and diversify through training. Each clone gets a
    small multiplicative perturbation so they are not exact duplicates (which
    would make the extra critics redundant and the min unchanged).
    """
    import torch

    model = SAC("MlpPolicy", venv, verbose=1, device=args.torch_device,
                gamma=args.gamma, gradient_steps=args.gradient_steps,
                tau=args.tau, ent_coef=_parse_ent_coef(args.ent_coef),
                learning_rate=args.learning_rate,
                policy_kwargs={"n_critics": n_critics})

    model.actor.load_state_dict(old_model.actor.state_dict())
    model.actor_target.load_state_dict(old_model.actor.state_dict())         if hasattr(model, "actor_target") else None

    old_n = len(old_model.critic.q_networks)
    gen = torch.Generator().manual_seed(args.terrain_seed)
    for attr in ("critic", "critic_target"):
        src = getattr(old_model, attr).state_dict()
        dst = {}
        for key, val in getattr(model, attr).state_dict().items():
            idx = int(key.split(".", 1)[0][2:])          # "qf3.0.weight" -> 3
            rest = key.split(".", 1)[1]
            src_val = src[f"qf{idx % old_n}.{rest}"]
            if idx < old_n:
                dst[key] = src_val.clone()
            else:
                noise = 1.0 + 0.01 * torch.randn(src_val.shape, generator=gen)
                dst[key] = src_val * noise.to(src_val.dtype)
        getattr(model, attr).load_state_dict(dst)

    if hasattr(old_model, "log_ent_coef") and old_model.log_ent_coef is not None \
            and getattr(model, "log_ent_coef", None) is not None:
        with torch.no_grad():
            model.log_ent_coef.copy_(old_model.log_ent_coef)

    model.num_timesteps = old_model.num_timesteps
    model._total_timesteps = old_model._total_timesteps
    print(f"[critic] expanded ensemble {old_n} -> {n_critics}; actor copied exactly, "
          f"extra critics cloned from the trained ones with 1% perturbation",
          flush=True)
    return model


from lunarsim.adapters.isaac.isaac_lander_vec_env import IsaacLanderVecEnv  # noqa: E402
# the stage table, the terrain config and the tile_fn all live in ONE place
# now -- see lunarsim/rl/curriculum.py's docstring for the four drifted
# copies that motivated it.
from lunarsim.rl.curriculum import STAGES, STAGES_BY_NAME, make_tile_fn  # noqa: E402
from lunarsim.rl.reward import RewardWeights  # noqa: E402

def _reward_fn_from_args(args):
    """The reward this run trains on, with any --reward-weight overrides.

    Returns None when nothing is overridden so the env keeps using its own
    default and this stays a no-op on the normal path.
    """
    if not args.reward_weight:
        return None
    from dataclasses import fields

    from lunarsim.rl.reward import make_apollo_reward_fn

    valid = {f.name: f.type for f in fields(RewardWeights)}
    overrides = {}
    for item in args.reward_weight:
        if "=" not in item:
            raise SystemExit(f"--reward-weight wants NAME=VALUE, got {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        if name not in valid:
            raise SystemExit(f"no RewardWeights field named {name!r}")
        overrides[name] = float(raw)
    print(f"[reward] overriding {overrides}", flush=True)
    return make_apollo_reward_fn(RewardWeights(**overrides))



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
    n_chunks = obs.shape[0] // n_envs
    if n_chunks == 0:
        raise SystemExit(f"{demo_path} holds only {obs.shape[0]} transitions, fewer than --n-envs={n_envs}")
    # The transitions that do not fill a whole chunk have to go somewhere.
    # Drop them off the FRONT, not the tail: a demo file is a concatenation
    # of episodes, so its last rows are the last episode's final steps --
    # i.e. a TERMINAL, the single most valuable row in the file and the
    # entire reason for seeding. The first rows are release-condition
    # states, of which there are 48 and which the online policy will
    # re-sample constantly anyway. Measured: 35521 transitions at
    # --n-envs 16 dropped exactly 1 row, and the seeding log duly reported
    # "47 terminal transitions per repeat" against 48 episodes.
    offset = obs.shape[0] - n_chunks * n_envs

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
            sl = slice(offset + c * n_envs, offset + (c + 1) * n_envs)
            infos = [{"TimeLimit.truncated": bool(v)} for v in truncated[sl]]
            model.replay_buffer.add(obs[sl], next_obs[sl], actions[sl], rewards[sl], dones[sl], infos)
        n_terminal = int(dones[offset:].sum())
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
                reward_fn=_reward_fn_from_args(args),
            )
        else:
            venv.params = stage.params
            venv.tile_fn = tile_fn
            venv.reset()

        first_model_creation = model is None
        if model is None:
            if args.warm_start:
                model = SAC.load(args.warm_start, env=venv, device=args.torch_device)
                if len(model.critic.q_networks) != args.n_critics:
                    model = _expand_critic_ensemble(model, venv, args, args.n_critics)
                # SAC.load does NOT restore a replay buffer, so without this the
                # critic starts from an empty one and refits off a narrow early
                # window every single warm start.
                _buf = Path(args.warm_start).with_suffix(".buffer.pkl")
                if args.save_buffer and _buf.is_file():
                    model.load_replay_buffer(str(_buf))
                    print(f"[buffer] restored {model.replay_buffer.size()} "
                          f"transitions from {_buf}", flush=True)
                    args.keep_buffer = True   # do not throw away what we just loaded
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
                # and the same for the entropy temperature. A checkpoint
                # saved mid-runaway carries its own `log_ent_coef` tensor,
                # so warm-starting from `out/sac_training_run_cal_v1` would
                # silently resume at alpha~6.6 no matter what --ent-coef
                # says. Only an auto-tuned checkpoint HAS that tensor, hence
                # the guard.
                # tau too: a checkpoint restores its own, and SB3's 0.005
                # default is wrong for gamma=0.999 (see the --tau help).
                model.tau = args.tau
                # and the learning rate, for the same reason -- a checkpoint
                # carries the rate it was saved with. Setting the attribute
                # alone is NOT enough: SB3's `_update_learning_rate` reads
                # `self.lr_schedule` on every train() call, so the optimizers
                # would keep the loaded rate while the attribute lied about it.
                model.learning_rate = args.learning_rate
                model.lr_schedule = FloatSchedule(args.learning_rate)
                _ec = _parse_ent_coef(args.ent_coef)
                if not isinstance(_ec, str):
                    import torch as _t
                    model.ent_coef = _ec
                    model.ent_coef_tensor = _t.tensor(float(_ec), device=model.device)
                    model.ent_coef_optimizer = None
                    model.log_ent_coef = None
                print(f"warm-started from {args.warm_start} "
                      f"(gamma={model.gamma}, gradient_steps={model.gradient_steps}, "
                      f"ent_coef={args.ent_coef})")
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
                # REAL BUG FOUND (2026-10-02, run `out/train_cal_v1.log`):
                # SB3's `ent_coef="auto"` default diverged on this task.
                # Measured over 268k steps: ent_coef 0.635 -> 6.6 and
                # critic_loss 7.15e3 -> 3.03e6, both monotonic and still
                # climbing at the end. Mechanism: alpha is tuned only
                # against `target_entropy` = -dim(A) = -4, so once a large
                # reward scale pulls the policy deterministic, alpha climbs
                # to push back -- and alpha re-enters the critic target
                # (`Q = r + gamma*(Q' - alpha*log_pi)`), which at
                # gamma=0.999 compounds over a ~1000-step horizon and
                # inflates Q, which inflates the gradient. A feedback loop.
                # reward.py's 10x scale-down addresses the cause; pinning
                # alpha removes the loop outright. 0.01 was sized against
                # the measured per-step reward on the winning controller
                # run (mean |r| = 0.0483), so the entropy term
                # `alpha*|target_entropy|` = 0.04 sits at ~0.8x the task
                # signal per step -- real exploration pressure, ~8% of the
                # discounted terminal over a typical episode, and 660x
                # below where the tuner ran to.
                # REAL BUG FOUND (2026-10-04 audit): SB3's default tau=0.005
                # gives the target network a 1/tau = 200 gradient-step time
                # constant. At gamma=0.999 the bootstrap horizon is
                # 1/(1-gamma) = 1000 steps, so the target moves 5x FASTER
                # than the horizon it exists to stabilise and bootstrap
                # error chases itself. SB3's default is calibrated for its
                # own gamma=0.99 default (horizon 100), where the target is
                # the slower of the two by 2x -- the stable ordering. This
                # is an independent divergence driver from the entropy
                # runaway and pinning alpha does not address it.
                model = SAC("MlpPolicy", venv, verbose=1, device=args.torch_device,
                             gamma=args.gamma, gradient_steps=args.gradient_steps,
                             tau=args.tau, ent_coef=_parse_ent_coef(args.ent_coef),
                             learning_rate=args.learning_rate,
                             policy_kwargs={"n_critics": args.n_critics})
                if args.layer_norm:
                    from lunarsim.rl.plasticity import add_layer_norm_to_sac
                    n_ln = add_layer_norm_to_sac(model)
                    print(f"[layernorm] rebuilt {n_ln} trunks with LayerNorm", flush=True)
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

        cbs = []
        if args.checkpoint_every > 0:
            from stable_baselines3.common.callbacks import CheckpointCallback
            cbs.append(CheckpointCallback(
                save_freq=max(1, args.checkpoint_every // max(1, args.n_envs)),
                save_path=f"{args.out_dir}/snapshots",
                name_prefix=f"{stage.name}",
                save_replay_buffer=args.save_buffer))
        if args.reset_every > 0:
            from lunarsim.rl.plasticity import PeriodicResetCallback
            cbs.append(PeriodicResetCallback(
                every=args.reset_every, alpha=args.reset_alpha,
                scope=args.reset_scope, seed=args.terrain_seed))
        cb = cbs if cbs else None
        model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False, callback=cb)
        out_path = f"{args.out_dir}/sac_lunar_lander_isaac_{stage.name}.zip"
        model.save(out_path)
        if args.save_buffer:
            model.save_replay_buffer(out_path.replace(".zip", ".buffer.pkl"))
            print(f"saved buffer: {out_path.replace('.zip', '.buffer.pkl')}")
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
