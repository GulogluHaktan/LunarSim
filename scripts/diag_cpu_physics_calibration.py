"""Calibrates PhysxForceAPI force/torque scale under CPU physics (device=
"cpu") -- the GPU-mode pipeline's empirical `FORCE_TORQUE_CAL=1/100` factor
was derived under Direct GPU API semantics and doesn't apply once torque
(broken under GPU mode -- see diag_rcs_torque_test.py) forces a switch to
CPU physics. Two checks:
  A) hover: apply thrust = exact weight (mass*g), no scale factor -- vz
     should stay ~0 (net-zero vertical accel) if force needs no CPU-mode
     correction.
  B) torque: apply max RCS torque, no scale factor -- angular accel should
     match the analytic model's own expectation (max_torque/i_tilt, ~0.02
     rad/s^2 at full mass) if torque needs no CPU-mode correction either.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
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
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, moment_of_inertia
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
xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, 500.0))

mass_kg = p.dry_mass_kg + p.initial_fuel_kg
mass_api = UsdPhysics.MassAPI(lm_root.GetPrim())
mass_api.GetMassAttr().Set(mass_kg)
i_tilt, i_yaw = moment_of_inertia(mass_kg, p.body_radius_m, p.body_height_m)
mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(i_tilt, i_tilt, i_yaw))

force_api = PhysxSchema.PhysxForceAPI.Apply(lm_root.GetPrim())
force_api.CreateForceEnabledAttr(True)
force_api.CreateWorldFrameEnabledAttr(False)
force_api.CreateForceAttr(Gf.Vec3f(0.0, 0.0, 0.0))
force_api.CreateTorqueAttr(Gf.Vec3f(0.0, 0.0, 0.0))

imu = Imu(cfg=ImuCfg(prim_path="/World/LM"))
pva = Pva(cfg=PvaCfg(prim_path="/World/LM"))

sim.reset()

print(f"mass_kg={mass_kg:.1f} i_tilt={i_tilt:.1f} weight_n={mass_kg * p.gravity_m_s2:.2f}")
max_torque = p.rcs_thrust_n * p.rcs_quad_radius_m * p.rcs_jets_per_couple
print(f"max_torque={max_torque:.2f} N.m, expected max angular accel = {max_torque / i_tilt:.6f} rad/s^2")

print("warming up...")
for _ in range(30):
    sim.step(render=False)
    imu.update(dt=sim.get_physics_dt())
    pva.update(dt=sim.get_physics_dt())
sim_dt = sim.get_physics_dt()


def _sample(label):
    qx, qy, qz, qw = pva.data.quat_w[0]
    qx, qy, qz, qw = (float(v) for v in (qx, qy, qz, qw))
    roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)
    lin_vel_b = pva.data.lin_vel_b[0]
    lin_vel_b = lin_vel_b.cpu().numpy() if hasattr(lin_vel_b, "cpu") else np.asarray(lin_vel_b)
    ang_vel_b = imu.data.ang_vel_b[0]
    ang_vel_b = ang_vel_b.cpu().numpy() if hasattr(ang_vel_b, "cpu") else np.asarray(ang_vel_b)
    print(f"[{label}] roll={np.degrees(roll):9.4f}deg vz={lin_vel_b[2]:.4f} "
          f"wx={ang_vel_b[0]:.6f} wy={ang_vel_b[1]:.6f}")


print("=== A: hover force = exact weight, NO scale factor ===")
force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, mass_kg * p.gravity_m_s2))
for step in range(60):
    sim.step(render=False)
    imu.update(dt=sim_dt)
    pva.update(dt=sim_dt)
    if step % 15 == 0:
        _sample(f"hover t={step * sim_dt:.2f}s")

print("=== B: torque = max_torque (pitch axis), NO scale factor, short burst ===")
force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, mass_kg * p.gravity_m_s2))  # keep hovering, isolate torque effect
force_api.GetTorqueAttr().Set(Gf.Vec3f(max_torque, 0.0, 0.0))
for step in range(30):  # short burst -- if scale is right this should stay small; if wrong, will blow up fast
    sim.step(render=False)
    imu.update(dt=sim_dt)
    pva.update(dt=sim_dt)
    if step % 5 == 0:
        _sample(f"torque t={step * sim_dt:.2f}s")

print("done")
simulation_app.close()
