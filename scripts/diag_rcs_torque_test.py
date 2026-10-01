"""Minimal diagnostic: spawn just the Apollo LM (no terrain, no cameras,
no controller) and apply a CONSTANT full RCS pitch-torque command via the
exact same PhysxForceAPI mechanism `isaaclab_policy_eval_capture.py` /
`isaaclab_controller_eval_capture.py` use, then print roll/pitch/ang_vel
every step for a few seconds. Purpose: isolate whether torque->rotation
works AT ALL in this pipeline, independent of the controller, terrain, or
observation code -- see the "frozen roll/pitch" finding from a real
controller-driven capture.

Run via: (same docker invocation pattern as the other isaaclab_* scripts,
no --enable_cameras needed)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import numpy as np
import isaaclab.sim as sim_utils
import omni.usd
from isaaclab.sensors import Imu, ImuCfg, Pva, PvaCfg
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

sys.path.insert(0, args_cli.lunarsim_root)

from lunarsim.adapters.isaac.isaac_lander_env import _quat_to_euler
from lunarsim.adapters.isaac.lander import spawn_apollo_lm
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs
from lunarsim.rl import LanderParams

p = LanderParams()
specs = ApolloLMSpecs()

sim_cfg = sim_utils.SimulationCfg(device="cpu", gravity=(0.0, 0.0, -p.gravity_m_s2))
sim = sim_utils.SimulationContext(sim_cfg)
stage = omni.usd.get_context().get_stage()

lm_asset_path = os.path.join(args_cli.lunarsim_root, "assets/models/apollo_lm/Apollo_Lunar_Module.usdz")
lm_root = spawn_apollo_lm(stage, "/World/LM", lm_asset_path, specs=specs, fuel_kg=p.initial_fuel_kg,
                           visual_only=False)
xform_api = UsdGeom.XformCommonAPI(lm_root.GetPrim())
xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 500.0))  # high in the air, no terrain, so it never lands mid-test

FORCE_TORQUE_CAL = 1.0 / 100.0
PhysxSchema.PhysxRigidBodyAPI.Apply(lm_root.GetPrim()).CreateAngularDampingAttr(p.angular_damping_per_s)

force_api = PhysxSchema.PhysxForceAPI.Apply(lm_root.GetPrim())
force_api.CreateForceEnabledAttr(True)
force_api.CreateWorldFrameEnabledAttr(False)
force_api.CreateForceAttr(Gf.Vec3f(0.0, 0.0, 0.0))  # no thrust -- isolate torque only
force_api.CreateTorqueAttr(Gf.Vec3f(0.0, 0.0, 0.0))

imu = Imu(cfg=ImuCfg(prim_path="/World/LM"))
pva = Pva(cfg=PvaCfg(prim_path="/World/LM"))

sim.reset()

max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple
print(f"max_torque={max_torque:.2f} N.m, FORCE_TORQUE_CAL={FORCE_TORQUE_CAL}, "
      f"applied_torque_x={max_torque * FORCE_TORQUE_CAL:.4f} N.m (pitch_cmd=1.0)")

print("warming up...")
for _ in range(30):
    sim.step(render=False)
    imu.update(dt=sim.get_physics_dt())
    pva.update(dt=sim.get_physics_dt())

sim_dt = sim.get_physics_dt()
print(f"sim_dt={sim_dt}")


def _run(steps, label):
    for step in range(steps):
        sim.step(render=False)
        imu.update(dt=sim_dt)
        pva.update(dt=sim_dt)
        if step % 30 == 0:
            qx, qy, qz, qw = pva.data.quat_w[0]
            qx, qy, qz, qw = (float(v) for v in (qx, qy, qz, qw))
            roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)
            ang_vel_b = imu.data.ang_vel_b[0]
            ang_vel_b = ang_vel_b.cpu().numpy() if hasattr(ang_vel_b, "cpu") else np.asarray(ang_vel_b)
            t_s = step * sim_dt
            print(f"[{label}] t={t_s:5.2f}s roll={np.degrees(roll):9.4f}deg pitch={np.degrees(pitch):9.4f}deg "
                  f"wx={ang_vel_b[0]:.6f} wy={ang_vel_b[1]:.6f} wz={ang_vel_b[2]:.6f}")


print("=== phase 1: torque via PhysxForceAPI, WorldFrameEnabled=False (original assumption) ===")
force_api.GetTorqueAttr().Set(Gf.Vec3f(max_torque * FORCE_TORQUE_CAL, 0.0, 0.0))
_run(args_cli.steps // 3, "torque_local")

print("=== phase 2: torque via PhysxForceAPI, WorldFrameEnabled=True ===")
force_api.GetTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
force_api.GetWorldFrameEnabledAttr().Set(True)
force_api.GetTorqueAttr().Set(Gf.Vec3f(max_torque * FORCE_TORQUE_CAL, 0.0, 0.0))
_run(args_cli.steps // 3, "torque_world")

print("=== phase 3: bypass torque entirely -- set angular velocity directly ===")
force_api.GetTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
rigid_body_api = UsdPhysics.RigidBodyAPI(lm_root.GetPrim())
rigid_body_api.GetAngularVelocityAttr().Set(Gf.Vec3f(30.0, 0.0, 0.0))  # deg/s, USD convention
_run(args_cli.steps // 3, "direct_angvel")

print("done")
simulation_app.close()
