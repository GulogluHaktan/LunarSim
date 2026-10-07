"""THE curriculum stage table: one definition, imported by everything that
has to agree on what a "stage" is.

REAL DRIFT THIS FIXES: this table used to be copy-pasted into
`scripts/train_sac_isaac.py`, `scripts/diag_stage_landing_rate.py`,
`scripts/diag_policy_telemetry.py`, `scripts/collect_zemzev_demos.py` and
`scripts/test_landing_feasibility.py`, each with a comment asserting it
"mirrors scripts/train_sac_isaac.py's STAGES exactly". By this session they
did not: `test_landing_feasibility.py` -- the script whose entire job is to
certify that the stage the RL curriculum trains on is physically solvable --
was still certifying `hover_only_easy`/`hover_only`/`final_approach`, three
stages that had been DELETED from training, and had never heard of the four
`ramp_*` stages that replaced them. A feasibility reference that tests a
different scenario than the trainer is worse than none.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lunarsim.core.terrain.config import TerrainConfig
from lunarsim.core.terrain.generate import Tile, generate_tile
from lunarsim.rl.analytic_lander_env import LanderParams


@dataclass
class Stage:
    name: str
    tile_size_m: float
    params: LanderParams = field(default_factory=LanderParams)
    terrain_roughness_scale: float = 1.0


#  REMOVED EARLIER THIS SESSION, PER USER DIRECTION: the hover_only_easy ->
#  hover_only -> final_approach curriculum stages that used to precede
#  orbit_descent. RE-ADDED THIS SESSION as a DIFFERENT shape, after THREE
#  separate attempts at orbit_descent-only training all reached 0/16
#  landed_safely: (1) a 1M-step warm-started continuation of the old
#  13M-step checkpoint, (2) a 2M-step fresh-from-scratch run, (3) a
#  2M-step fresh run with SAC's replay buffer pre-seeded with real Isaac
#  Sim ZemZevController demo transitions (20x-repeated so they weren't
#  diluted to <2% of the buffer by online data -- see
#  _seed_replay_buffer_from_demos's field comment for that first-try bug).
#  All three: 11-14/16 episodes ran the FULL 60s clock without ever
#  attempting touchdown, ent_coef decayed to a low flat value, no trend
#  toward improvement. Root cause, CORRECTED this session against the real
#  constants (the earlier wording here said "this vehicle's thrust-to-weight
#  ratio is close to 1", which is NOT true and would have meant the task was
#  barely flyable at all): the DPS gives T/W = 1.84 at the full 15103 kg PDI
#  mass and 4.01 at dry mass, i.e. plenty of authority. What is close to 1
#  is the T/W at action[0] = 0 -- the exact CENTRE of the action box, and
#  the mean of an untrained squashed-Gaussian SAC policy. throttle =
#  (a+1)/2 = 0.5 maps to 4672 + 0.5*(45040-4672) = 24856 N against a PDI
#  weight of 15103*1.62 = 24467 N, i.e. T/W = 1.016, and the exact hover
#  action is a0 = -0.019. So "hover" is not merely findable under random
#  exploration, it is literally what a freshly-initialized policy outputs
#  on average, and it stays that way all episode (only ~5% of the
#  propellant burns in 60 s, so T/W at a0=0 drifts only 1.016 -> ~1.06).
#  orbit_descent's release (200m, 10-30 m/s horizontal) is far enough from
#  that attractor that the policy essentially never randomly stumbles into
#  a full successful touchdown to learn its value from. Demo-bootstrapping
#  tried to shortcut this without curriculum; it didn't work within budget
#  -- though see `_seed_replay_buffer_from_demos`, the demo FILE that run
#  used turned out to contain only 4 terminal transitions in 39824.
#
#  THIS curriculum is shaped differently from the old hover_only_easy ->
#  final_approach -> orbit_descent one (which jumped straight from
#  final_approach's 35m/0-3 m/s to orbit_descent's 200m/10-30 m/s in ONE
#  step -- a cliff, not a ramp): every stage below starts from a release
#  the PREVIOUS stage's policy should already handle reasonably (small
#  altitude/speed delta each step), so the policy is always warm-started
#  into a regime "near" what it already knows, instead of ever facing a
#  qualitatively new release condition cold. Terrain stays at FULL
#  roughness throughout (unlike the old hover_only_easy/hover_only, which
#  used roughness=0) -- the skill being ramped here is altitude/speed
#  control, not terrain handling, and the real target (orbit_descent) is
#  real terrain, so there's no reason to ease that dimension separately.
#
#  REAL BUG FOUND IN THE TILE SIZES (this session, and it was already in
#  orbit_descent's 600m tile long before the ramp stages copied its
#  method): they were sized as "fall time x top horizontal speed, +30%" --
#  but that is the distance travelled FROM the release point, which has to
#  fit inside the tile's HALF-extent, not inside its full size. Factor-of-2
#  error, compounded by using free-fall time (a CONTROLLED descent takes
#  ~2.5x longer than free fall, because braking horizontal speed is what
#  the extra time is spent on). The consequence is not a modelling nicety:
#  `sample_height_at`/`collision_mesh_height_at` CLAMP out-of-range lookups
#  to the grid edge, so past the tile boundary the env keeps reporting a
#  confident altitude while PhysX has no collider there at all (see
#  `isaac_lander_env.out_of_tile`). Measured on every real orbit_descent
#  capture on disk, 600m tile / 300m half-extent:
#    orbit_descent_margins_check_1 ended at r=650 m ("touchdown" at a
#      reported alt of -3.2 m, against nothing);
#    orbit_descent_margins_check_2 ended at r=524 m, reported alt -32.8 m;
#    orbit_descent_multiseed_33 (the PROVEN ZemZev controller) ended at
#      r=487 m, reported alt -2.4 m.
#  And measured directly, 24 ZemZev episodes per stage on AnalyticLanderEnv
#  with an unlimited time budget (the ground track does not depend on which
#  backend integrates it), max distance from the tile centre vs. the OLD
#  half-extent: ramp_20m 15 m vs 30 (fine), ramp_50m 93 vs 60, ramp_100m
#  252 vs 130, ramp_150m 469 vs 225, orbit_descent 672 vs 300 -- i.e. 7/24,
#  12/24, 15/24 and 13/24 episodes respectively flew off the terrain even
#  under the controller that is this project's proof the task is solvable.
#  Tile sizes below are now 2 x (that measured max radius) x ~1.25 margin,
#  rounded to a multiple of TERRAIN_GRID_N. This is also forced by simple
#  physics independent of any controller: killing 30 m/s with the DPS at
#  its 30 deg braking tilt gives ~1.49 m/s^2 of lateral decel
#  (45040 N / 15103 kg x sin30 deg), i.e. a 302 m minimum stopping
#  distance, so a +-300 m tile CANNOT contain orbit_descent's release
#  envelope no matter how well it is flown.
STAGES = [
    Stage(
        name="ramp_20m",
        # 60 -> 120 m (2026-10-05). MEASURED: tiles were sized for the drift
        # during a ballistic FALL, but episodes run ~4x the fall time, so a
        # vehicle that has not yet learned to brake laterally leaves the tile
        # by construction -- and `out_of_tile` truncates, so it never reaches
        # the ground and can never discover the landing bonus. The
        # sparse-reward trap was being enforced by the tile size.
        #   usable lateral room (half-width - footpad margin - spawn radius)
        #     = 30 - 4.7 - 10 = 15.3 m
        #   drift over a full 20 s episode at the 2 m/s release = 40 m
        # Measured consequence: 14/16 ramp_20m episodes ended `left_tile`.
        # At 120 m the usable room is 45.3 m, which covers it. Grid stays at
        # 80, so resolution is 1.5 m -- no PhysX cost.
        # The same mismatch exists at every other stage (ramp_50m 90.3 m
        # usable vs 240 m of episode drift, orbit_descent 755 vs 1800); they
        # are left alone for now because a competent policy brakes early and
        # the ZemZev demos never leave any tile (0/48), but a from-scratch
        # policy on those stages will hit the same wall.
        tile_size_m=120.0,
        params=LanderParams(
            spawn_altitude_m=20.0, spawn_xy_radius_m=10.0,
            # MEASURED 2026-10-05, and it was the wall seven training runs
            # hit: released at vz=0 a RANDOM policy touches the ground
            # 0/40 times -- 40/40 time out instead. The vehicle's
            # thrust-to-weight at the centre of the action box is 1.0159,
            # so "do nothing", which is exactly an untrained policy's mean
            # action, IS a hover. Nothing ever reaches the ground, so the
            # terminal reward is never sampled and cannot be learned, no
            # matter what it says.
            # Released at vz=-2.0 the same random policy touches down
            # 40/40, at a median severity of 1.67x the vz limit and a best
            # of 1.16x -- i.e. the success region is immediately adjacent
            # to random behaviour, and `crash_severity` grades every metre
            # per second it shaves off. The deleted `hover_only_easy`
            # stage, which this file's history records as "presumed
            # solved", used exactly this value; that is why it worked.
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(0.0, 2.0),
            max_episode_s=20.0,
        ),
    ),
    Stage(
        # INSERTED 2026-10-05. ramp_20m -> ramp_50m changed BOTH dimensions
        # at once -- altitude 2.5x and release lateral speed 4x -- and the
        # policy that lands 83% on ramp_20m scored 1/16 and 0/16 there, with
        # lateral speed the binding criterion both times. That is the same
        # "cliff, not a ramp" defect this file records for the curriculum
        # this ramp replaced.
        # Lateral is the hard dimension (it was also the binding criterion
        # on 10 of 12 touchdowns at ramp_20m), so ramp it alone first, at an
        # altitude the policy has already solved. Budget check: descent
        # 11.5 s + bleeding 5 m/s at the ~20 deg tilt the policy uses
        # (8.5 s), partially overlapped, needs ~14.5 s against 20.
        name="ramp_20m_fast",
        # 120 -> 260 m (2026-10-06). This stage was created with ramp_20m's
        # tile and inherited the same mismatch the moment its release speed
        # went to 5 m/s: usable lateral room 45.3 m against 100 m of drift
        # over a 20 s episode. The 42% measured on it was scored under that
        # handicap.
        # 260 -> 120 m (2026-10-06). The 260 m was sized for this stage's old
        # 5 m/s lateral release; at (1.5, 2.5) the controller's measured ground
        # track in Isaac is 23.3 m, so 260 m was ~5x oversized. That is not
        # merely wasteful: `terrain_grid_n` is global, so a 260 m tile has 3.25 m
        # cells against ramp_20m's 1.5 m, and the stage meant to introduce
        # lateral speed at ramp_20m's altitude was also silently flattening the
        # terrain relative to the stage it follows. 120 m restores the
        # one-dimension-at-a-time property and still leaves 5x the measured
        # track.
        tile_size_m=120.0,
        params=LanderParams(
            spawn_altitude_m=20.0, spawn_xy_radius_m=10.0,
            # (2.0, 5.0) -> (1.5, 2.5) (2026-10-06). This stage was not hard,
            # it was IMPOSSIBLE, and for a different reason than ramp_100m's
            # short budget: 20 m of altitude is not enough to bleed off 5 m/s
            # of lateral speed. Flown by the ZemZev controller it scored
            # 1/24, touching down at t~17 s with v_xy still at 0.87-2.50
            # against a 1.2 m/s limit. Raising the budget does NOT help --
            # 1/24 at every budget from 20 s to 34 s -- because the vehicle
            # reaches the ground before it can finish braking, not before the
            # clock runs out.
            # Measured envelope at 20 m, controller over 24 episodes:
            #   v_xy (2.0, 5.0)   1/24      v_xy (1.5, 2.5)  16/24
            #   v_xy (2.0, 3.5)   1/24      v_xy (2.0, 2.5)  11/24
            #   v_xy (2.0, 3.0)   1/24
            # and raising altitude instead: h0=28 -> 17/24, h0=30 -> 22/24,
            # which is just ramp_35m again (35 m, 2-5 m/s, 21/24). So at this
            # altitude the feasible range caps near 2.5 m/s, and the stage's
            # job -- introduce lateral speed at ramp_20m's altitude -- has to
            # fit inside it. 16/24 for the reference controller makes this a
            # real step up from ramp_20m's 24/24 while staying solvable.
            # NOTE: every rate ever measured on this stage (42%, 46%, 29%)
            # was scored on the impossible version.
            spawn_v_z_m_s=-2.0, spawn_horizontal_speed_m_s=(1.5, 2.5),
            max_episode_s=20.0,
        ),
    ),
    Stage(
        # INSERTED 2026-10-06, same reasoning that produced ramp_20m_fast:
        # ramp_20m_fast -> ramp_50m changed altitude 2.5x AND release speed
        # 1.6x at once, and ramp_50m managed only 2/16 even after its
        # reachability was fixed. This stage moves ALTITUDE alone (20 -> 35 m)
        # at ramp_20m_fast's own 2-5 m/s.
        # Every number below is derived from the three rules this file now
        # records, not picked:
        #   budget 25 s     -> needs h0/T = 1.40 m/s of average descent
        #   vz0 -2.8        =  2x that, and inside the 4.61 m/s envelope at
        #                      35 m, so the episode does not start in penalty
        #   drift 5*25=125 m -> tile >= 2*(125 + 4.7 + 15) = 289 -> 320 m
        name="ramp_35m",
        tile_size_m=320.0,
        params=LanderParams(
            spawn_altitude_m=35.0, spawn_xy_radius_m=15.0,
            spawn_v_z_m_s=-2.8, spawn_horizontal_speed_m_s=(2.0, 5.0),
            max_episode_s=25.0,
        ),
    ),
    Stage(
        name="ramp_50m",
        # 240 -> 600 m (2026-10-06). The random-policy probe on this stage
        # came back `land 0, touchdown 2, timeout 16, LEFT_TILE 22` out of 40:
        # 55% of episodes end by flying off the map before anything can be
        # learned. Same blocker ramp_20m had, same cause -- the tile is sized
        # for drift during a ballistic FALL (63 m) while the episode runs 30 s,
        # which at the 8 m/s release is 240 m of drift against 90.3 m of usable
        # lateral room (half-width minus footpad margin minus spawn radius).
        # 600 m gives 270 m of room. Grid stays 80, so cells are 7.5 m and the
        # 9.4 m footpad span spans ~1.25 of them -- leg_diff becomes nearly
        # trivial, which is acceptable only because it already passes ~100% of
        # the time on training terrain; revisit with --terrain-grid-n if that
        # stops being true.
        tile_size_m=600.0,
        params=LanderParams(
            spawn_altitude_m=50.0, spawn_xy_radius_m=25.0,
            # Bootstrap descent, sized at ~2x h0/max_episode_s (2026-10-06).
            # The first version of this TAPERED the value down as the
            # curriculum climbed, on the reasoning that later stages inherit
            # a policy that already descends. That was backwards and the
            # random-policy probe caught it: reachability needs AT LEAST
            # h0/T of average descent, and h0 grows faster than the budget
            # does, so the requirement rises stage to stage while I was
            # lowering the help.
            #   stage        h0/T needed   was given
            #   ramp_50m        1.67 m/s     -1.5   <- unreachable
            #   ramp_100m       2.50         -1.0
            #   ramp_150m       3.00         -0.5
            # Measured at ramp_50m with -1.5: a random policy gets 0/40
            # touchdowns, 40/40 timeouts, median final altitude 18.7 m --
            # the same unreachable-goal signature ramp_20m had at vz=0.
            # These values stay INSIDE the descent envelope (5.52 m/s at
            # 50 m, 7.80 at 100), so an episode does not start in penalty.
            # orbit_descent keeps 0.0: it is the real scenario, an orbital
            # release, and by then the policy has to descend on its own.
            # -3.0 -> -3.4 (2026-10-06, measured in REAL Isaac). This stage
            # had ZERO budget slack and was one step from the defect that made
            # ramp_100m and ramp_150m unsolvable. Over 48 controller episodes
            # touchdown times were 29.6 s median with a max of EXACTLY 30.0 s
            # and 9 of 48 at >= 29.8 s -- one episode landed in the final
            # control step of the clock. Anything slightly slower than the
            # reference controller (a trained policy, a rougher draw, a heavier
            # lateral correction) times out at ~1 m of clearance and is graded a
            # failure.
            #
            # The same Isaac sweep also corrected the reachability rule this
            # file states. Worst-case touchdown time is a tight multiple of the
            # free-coast time across every stage:
            #   h0/|vz0| vs t_td_max -> 1.69, 1.70, 1.82, 1.79, 1.83, 1.87
            # so the usable form is `T >= ~1.9 * h0/|vz0|`, equivalently
            # `|vz0| >= 1.9 * h0/T` -- the "~2x h0/T" heuristic used for
            # spawn_v_z_m_s was right, and the bare `>= h0/T` written elsewhere
            # in these comments is what permits an unreachable stage. By the
            # 1.9x rule this stage needed |vz0| >= 3.33; it had 3.0.
            spawn_v_z_m_s=-3.4, spawn_horizontal_speed_m_s=(2.0, 8.0),
            max_episode_s=30.0,
        ),
    ),
    Stage(
        name="ramp_100m",
        # 640 -> 800 m (2026-10-06, measured in REAL Isaac). This is the only
        # stage whose tile was never re-derived from a measured ground track.
        # The controller's max radius here is 311.2 m against a truncation
        # boundary of 315.3 m (half_extent - footpad_span/2) -- 4.1 m of margin
        # over 24 episodes. Every other stage carries 2.1-2.5x: ramp_150m is
        # 1200 m for a 561 m track, orbit_descent 1680 for 684 m. The analytic
        # figure the old comment cited (293.7 m) was both smaller than Isaac's
        # and compared against the half-extent rather than the half-extent
        # minus the footpad margin. Cost of the fix: grid stays at 80, cells go
        # 8 -> 10 m.
        tile_size_m=800.0,
        params=LanderParams(
            spawn_altitude_m=100.0, spawn_xy_radius_m=50.0,
            # Bootstrap descent, sized at ~2x h0/max_episode_s (2026-10-06).
            # The first version of this TAPERED the value down as the
            # curriculum climbed, on the reasoning that later stages inherit
            # a policy that already descends. That was backwards and the
            # random-policy probe caught it: reachability needs AT LEAST
            # h0/T of average descent, and h0 grows faster than the budget
            # does, so the requirement rises stage to stage while I was
            # lowering the help.
            #   stage        h0/T needed   was given
            #   ramp_50m        1.67 m/s     -1.5   <- unreachable
            #   ramp_100m       2.50         -1.0
            #   ramp_150m       3.00         -0.5
            # Measured at ramp_50m with -1.5: a random policy gets 0/40
            # touchdowns, 40/40 timeouts, median final altitude 18.7 m --
            # the same unreachable-goal signature ramp_20m had at vz=0.
            # These values stay INSIDE the descent envelope (5.52 m/s at
            # 50 m, 7.80 at 100), so an episode does not start in penalty.
            # orbit_descent keeps 0.0: it is the real scenario, an orbital
            # release, and by then the policy has to descend on its own.
            # 40 -> 48 s (2026-10-06). The budget, not the difficulty, was
            # what made this stage impossible. Flown by the ZemZev
            # controller -- the reference for a healthy landing -- this
            # stage scored 0/24, and every failure looked identical:
            #   ep0 TIMEOUT t=40.0 alt=4.45 vz=-0.80 vxy=0.56
            #   ep1 TIMEOUT t=40.0 alt=4.45 vz=-0.80 vxy=0.41
            #   ep2 TIMEOUT t=40.0 alt=4.47 vz=-0.80 vxy=0.02
            # The vehicle had already nulled its lateral speed and was
            # descending correctly at its 0.80 m/s terminal rate -- it
            # simply ran out of clock 4.5 m above the ground, which is
            # 5.6 s short. Measured: at T=46 the controller goes to 24/24,
            # and the extra time is spent in the terminal descent where
            # lateral speed is ~0.5 m/s, so drift does NOT grow with it
            # (293.7 -> 292.2 m against 320 m of usable room).
            # This is why the ladder was not monotonic in difficulty: the
            # controller landed 21/24 on ramp_35m and 22/24 on the FINAL
            # orbit_descent stage, but 0/24 on this one and ramp_150m.
            spawn_v_z_m_s=-4.0, spawn_horizontal_speed_m_s=(6.0, 16.0),
            max_episode_s=48.0,
        ),
    ),
    Stage(
        name="ramp_150m",
        tile_size_m=1200.0,
        params=LanderParams(
            spawn_altitude_m=150.0, spawn_xy_radius_m=65.0,
            # Bootstrap descent, sized at ~2x h0/max_episode_s (2026-10-06).
            # The first version of this TAPERED the value down as the
            # curriculum climbed, on the reasoning that later stages inherit
            # a policy that already descends. That was backwards and the
            # random-policy probe caught it: reachability needs AT LEAST
            # h0/T of average descent, and h0 grows faster than the budget
            # does, so the requirement rises stage to stage while I was
            # lowering the help.
            #   stage        h0/T needed   was given
            #   ramp_50m        1.67 m/s     -1.5   <- unreachable
            #   ramp_100m       2.50         -1.0
            #   ramp_150m       3.00         -0.5
            # Measured at ramp_50m with -1.5: a random policy gets 0/40
            # touchdowns, 40/40 timeouts, median final altitude 18.7 m --
            # the same unreachable-goal signature ramp_20m had at vz=0.
            # These values stay INSIDE the descent envelope (5.52 m/s at
            # 50 m, 7.80 at 100), so an episode does not start in penalty.
            # orbit_descent keeps 0.0: it is the real scenario, an orbital
            # release, and by then the policy has to descend on its own.
            # 50 -> 60 s (2026-10-06). Same defect as ramp_100m, same
            # signature: 0/24 with the ZemZev controller, every episode
            # timing out at alt 4.6 m with vz=-0.80 and vxy already down to
            # 0.5 m/s. 5.8 s short. At T=56 the controller goes to 24/24 and
            # drift is unchanged (529.3 -> 527.9 m against 600 m of room).
            spawn_v_z_m_s=-5.0, spawn_horizontal_speed_m_s=(10.0, 24.0),
            max_episode_s=60.0,
        ),
    ),
    # "yorunge" stage: an uncontrolled-release-scale altitude/horizontal-
    # speed regime, matching what scripts/isaaclab_static_telemetry_capture.py's
    # actual demo descents used (120-350m release, real craters/hills/rocks
    # at full roughness) -- NOT literal orbital mechanics, same scope
    # caveat as everywhere else in this codebase (see analytic_lander_env's
    # module docstring). This is the hardest, most "final descent"-like
    # stage: real obstacles (rocks, via IsaacLanderVecEnv's tile.rocks
    # spawning), a real crater/hill field, and (since this session) a tile
    # actually big enough for the ground track -- 1680 m, i.e. a +-840 m
    # half-extent against a measured 672 m worst-case ZemZev ground track
    # and a 302 m hard physical minimum stopping distance. The old 600 m
    # was derived from free-fall time (15.7 s) x 30 m/s = ~470 m "of
    # drift" compared against the FULL tile size instead of its half-
    # extent, and with free-fall rather than controlled-descent time.
    Stage(
        name="orbit_descent",
        tile_size_m=1680.0,
        params=LanderParams(
            spawn_altitude_m=200.0, spawn_xy_radius_m=80.0,
            # 0.0 -> -5.5 (MEASURED). Every ramp rung releases with a descent already
            # established -- -2.0, -2.0, -2.8, -3.4, -4.0, -5.0 -- and orbit_descent alone
            # released at 0.0, breaking the ramp it is the top of. This file already records
            # why that is fatal, for ramp_20m: "released at vz=0 a RANDOM policy touches down
            # 0/40 times -- 40/40 time out instead. The vehicle's thrust-to-weight at the
            # centre of the action box is 1.0159, so 'do nothing', which is exactly an
            # untrained policy's mean action, IS a hover. Nothing ever reaches the ground, so
            # the terminal reward is never sampled and cannot be learned, no matter what it
            # says." The fix was applied to the rungs as they were created and never to the
            # stage they lead to.
            #
            # Re-measured here on the analytic env, 20 episodes of a near-zero-mean random
            # policy per stage:
            #     ramp_20m   -2.00 m/s   touched down 20/20   median min altitude 0.0 m
            #     ramp_35m   -2.80        20/20                            0.0 m
            #     ramp_150m  -5.00        20/20                            0.0 m
            #     orbit_descent 0.00       0/20  (20/20 timed out)       200.0 m
            # On the target stage an untrained policy does not descend a single metre in 60
            # seconds, so the terminal reward is sampled zero times. The "from-scratch RL
            # scores 0%" result measured earlier in this project was therefore a statement
            # about the stage's configuration, not about RL.
            #
            # -5.5 continues the ramp (-4.0, -5.0, -5.5) and is also the more realistic end of
            # a braking burn than a hover at 200 m with 30 m/s of horizontal velocity.
            #
            # CONSEQUENCE, stated rather than buried: this changes the target task. The orbit
            # demos, the clone fitted to them and the 22.9% clone baseline were all measured at
            # vz=0 and are stale. The controller's own rate will shift too, so the comparison
            # stays fair only if BOTH are re-measured, which is the next step.
            # REVERTED to 0.0, and the measurement that forced it back is worth keeping.
            #
            # Changing it to -5.5 did fix the exploration problem -- an untrained policy went
            # from 0/20 touchdowns to 20/20 -- and broke the REFERENCE: the ZemZev controller
            # landed 0/96 in Isaac on the changed stage, every episode running the full 1201
            # steps. Swept on the analytic env, 16 episodes per value:
            #
            #     spawn vz    controller lands    random policy reaches the ground
            #       0.0           16/16                    0/16
            #      -1.0            0/16                    0/16
            #      -2.0            0/16                    0/16
            #      -3.0            0/16                    0/16
            #      -4.0            0/16                    0/16
            #      -5.5            0/16                   16/16
            #
            # There is NO value where both work. The controller is not a general guidance law
            # for this stage -- it is tuned to vz=0 tightly enough that -1.0 already fails
            # completely. So the choice is between a measurable reference and a stage
            # exploration can learn in, and keeping the reference wins: the goal is defined as
            # beating or approaching the controller, which is meaningless if the controller
            # scores 0.
            #
            # What this settles rather than leaves open: from-scratch RL is genuinely off the
            # table ON THIS STAGE, for a reason that is now measured rather than assumed -- an
            # untrained policy samples the terminal reward zero times because
            # thrust-to-weight at the action box centre is 1.0159, so its mean action is a
            # hover. That is not a statement about RL.
            #
            # It is also an argument for the two-model split: a landing specialist trains on
            # `land_handoff`, whose entry distribution includes vz in [-5, -1] where
            # exploration does reach the ground, while the approach half starts from this
            # stage's vz=0 which the controller handles. Each half gets an entry its own
            # regime supports, instead of one stage having to serve both.
            spawn_v_z_m_s=0.0, spawn_horizontal_speed_m_s=(10.0, 30.0),
            max_episode_s=60.0,
        ),
    ),
]


# Terrain feature scales, in ABSOLUTE METRES, identical at every stage.
#
# REAL BUG FOUND (this session): these used to be written as fractions of
# the tile size (`wavelength_m = size/6`, crater `d_max_m = size/10`),
# which silently made terrain DIFFICULTY a function of which curriculum
# stage you were on -- the exact opposite of this curriculum's stated
# design ("the skill being ramped here is altitude/speed control, not
# terrain handling"). A 60 m tile got 10 m-wavelength hills at the same
# 0.3 m amplitude as a 600 m tile's 100 m-wavelength hills, i.e. 10x the
# slope. Measured fraction of landing sites that satisfy
# `safe_landing_max_leg_height_diff_m = 0.16` (the footpad-span terrain
# flatness test in `landed_safely`), 3000 random sites x 6 tiles per stage,
# with the OLD size-relative scales: ramp_20m 58%, ramp_50m 88%, ramp_100m
# 99.9%, ramp_150m 100%, orbit_descent 100%. The curriculum's FIRST and
# supposedly easiest stage was the one where 42% of episodes could not be
# landed safely no matter how well they were flown. Pinning the scales to
# the values the 600 m orbit_descent tile used to produce gives: ramp_20m
# 91.5%, ramp_50m 100%, everything else 100%.
_HILL_WAVELENGTH_M = 100.0
_CRATER_D_MAX_M = 60.0

# Heightfield grid edge length. res_m = tile_size_m / TERRAIN_GRID_N, so
# this (not the tile size) is what sets the per-reset cost: the PhysX
# collision mesh `IsaacLanderVecEnv._rebuild_terrain` rebuilds on EVERY
# episode reset of EVERY env has 2*(N-1)^2 triangles.
#
# Kept at 80 -- the value every run on disk used -- so enlarging the tiles
# above changes zero about training throughput. It is NOT the value this
# scenario deserves: at 80, orbit_descent's 1680 m tile has 21 m cells, so
# the 9.4 m footpad span falls inside a single cell and
# `safe_landing_max_leg_height_diff_m` becomes inert (measured median
# footpad height diff 0.010 m against a 0.16 m limit). `--terrain-grid-n
# 160` is the natural next step and is FREE on the generation side
# (measured: generate_tile takes 80 ms at both N=80 and N=160, and only
# jumps to 315 ms at N=200) -- but it quadruples the triangle count PhysX
# has to cook per reset (12482 -> 50562), which cannot be measured without
# a GPU run. Try it with a short --steps-per-stage first and compare
# wall-clock before committing a long run to it.
_DEFAULT_TERRAIN_GRID_N = 80

# NOT part of the sequential curriculum. `STAGES` is a difficulty ramp that a single
# policy walks up; the landing specialist is one half of a two-model split and shares no
# ordering with it, so putting it in STAGES made a full-curriculum run start there. It stays
# addressable by --only-stage through STAGES_BY_NAME.
SPECIALIST_STAGES = [
    Stage(
        # LANDING SPECIALIST. Not part of the sequential curriculum's difficulty ramp --
        # it is one half of a two-model split, and it exists because of a measurement.
        #
        # The critic's action gradient, resolved by altitude on a trained orbit_descent
        # checkpoint:
        #
        #     altitude     toward the expert    |dQ/da|
        #      0-2 m            54.3%            1.297    decisive, gradient is noise
        #      2-5 m            53.8%            0.585
        #     15-40 m           63.1%            0.257
        #     80-150 m          68.6%            0.118    informative, barely pushes
        #
        # The actor is pushed hardest exactly where the critic knows least. And on
        # orbit_descent the buffer is 65% cruise-phase samples (40-210 m) against 14% in
        # the decisive 0-5 m band, so most of what the critic learns is about the regime
        # that does not decide the outcome. Gaudet/Linares/Furfaro switch regime at 15 m
        # for the same reason, and Apollo's descent guidance was phase-split outright
        # (P63 braking, P64 approach, P66 terminal descent).
        #
        # Trained alone, this stage's buffer is 100% decisive-band data -- which is the
        # condition under which this file already records from-scratch RL working: at
        # ramp_20m's release "a RANDOM policy touches down 40/40 ... the success region is
        # immediately adjacent to random behaviour".
        #
        # The ENTRY CONDITIONS ARE A DISTRIBUTION, not a point, because the specialist is
        # only useful if it accepts whatever an approach policy hands over. The ranges
        # below are deliberately wider than any single approach policy would produce, so
        # the handoff is covered rather than assumed; the envelope it actually achieves is
        # then measurable, and an approach policy can be rewarded for delivering into it.
        name="land_handoff",
        tile_size_m=120.0,
        params=LanderParams(
            spawn_altitude_m=(8.0, 25.0),
            spawn_xy_radius_m=10.0,
            spawn_v_z_m_s=(-5.0, -1.0),
            spawn_horizontal_speed_m_s=(0.0, 6.0),
            # An approach policy does not hand the vehicle over upright and still. The
            # measured controller tilts 8-13 deg routinely while braking and up to 17.6,
            # so a specialist that only ever sees near-level entries would be trained on a
            # handoff distribution that does not occur. +/-0.26 rad is 15 deg, the same
            # figure as the touchdown tilt criterion, so the entry spread reaches the edge
            # of what is still recoverable rather than stopping short of it.
            spawn_tilt_rad=(-0.26, 0.26),
            spawn_w_rad_s=(-0.15, 0.15),
            max_episode_s=20.0,
        ),
    ),
]

STAGES_BY_NAME = {s.name: s for s in list(STAGES) + SPECIALIST_STAGES}


def terrain_config(stage: Stage, seed: int, grid_n: int = _DEFAULT_TERRAIN_GRID_N) -> TerrainConfig:
    r = stage.terrain_roughness_scale
    return TerrainConfig(
        mode="fine", size_m=stage.tile_size_m, res_m=max(0.5, stage.tile_size_m / grid_n), seed=seed,
        coarse_source="procedural",
        hills={"amplitude_m": 0.3 * r, "wavelength_m": _HILL_WAVELENGTH_M, "hurst": 0.75},
        craters={"count_scale": 0.1 * r, "d_min_m": 1.0, "d_max_m": _CRATER_D_MAX_M, "b": 2.5,
                 "depth_ratio": 0.08, "age": 0.5},
        # real, PHYSICAL (collidable) rocks -- see IsaacLanderVecEnv's
        # `_rebuild_terrain` docstring on the real `tile.rocks` field this
        # spawns from (capped per-env regardless of density, for reset cost).
        rocks={"density_scale": r, "d_max_m": 1.0},
        roi={"sigma_m": stage.tile_size_m / 4.0, "centers": None},
        curvature=False,
    )


def make_tile_fn(stage: Stage, grid_n: int = _DEFAULT_TERRAIN_GRID_N):
    """A fresh-terrain-per-episode `tile_fn` for this stage."""
    def tile_fn(rng: np.random.Generator) -> Tile:
        return generate_tile(terrain_config(stage, int(rng.integers(0, 2 ** 31 - 1)), grid_n))
    return tile_fn

