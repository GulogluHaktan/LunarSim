"""Measures the ACTUAL effective mass/inertia PhysX is using for this rigid
body (as opposed to what `UsdPhysics.MassAPI`'s override says) by applying a
known force/torque for exactly ONE physics step and reading the resulting
velocity delta directly: effective_mass = force*dt/dvz,
effective_i_tilt = torque*dt/dwx. This is the cleanest measurement (a single
step avoids damping/compounding), and pins down exactly how far off the
solver's real mass properties are from what we asked for -- see
diag_cpu_physics_calibration.py's finding that both force and torque
produced roughly unit-mass/unit-inertia-sized responses.
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

from lunarsim.adapters.isaac.lander import spawn_apollo_lm
from lunarsim.core.vehicle.apollo_lm import ApolloLMSpecs, moment_of_inertia
from lunarsim.rl import LanderParams

p = LanderParams()
specs = ApolloLMSpecs()

sim_cfg = sim_utils.SimulationCfg(device="cpu", gravity=(0.0, 0.0, 0.0))  # zero gravity -- isolate force/torque cleanly
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
for _ in range(10):
    sim.step(render=False)
    imu.update(dt=sim.get_physics_dt())
    pva.update(dt=sim.get_physics_dt())
sim_dt = sim.get_physics_dt()


def _vz():
    lin_vel_b = pva.data.lin_vel_b[0]
    return float((lin_vel_b.cpu().numpy() if hasattr(lin_vel_b, "cpu") else np.asarray(lin_vel_b))[2])


def _wx():
    ang_vel_b = imu.data.ang_vel_b[0]
    return float((ang_vel_b.cpu().numpy() if hasattr(ang_vel_b, "cpu") else np.asarray(ang_vel_b))[0])


vz0 = _vz()
TEST_FORCE_N = 1000.0
force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, TEST_FORCE_N))
sim.step(render=False)
pva.update(dt=sim_dt)
vz1 = _vz()
dvz = vz1 - vz0
effective_mass = TEST_FORCE_N * sim_dt / dvz if dvz != 0 else float("inf")
print(f"sim_dt={sim_dt}")
print(f"FORCE test: applied={TEST_FORCE_N} N, dvz={dvz:.6f} m/s over one step -> "
      f"effective_mass={effective_mass:.4f} kg (intended={mass_kg:.1f} kg, "
      f"correction_factor={effective_mass / mass_kg:.6f})")

force_api.GetForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
for _ in range(5):
    sim.step(render=False)
    imu.update(dt=sim_dt)
    pva.update(dt=sim_dt)

wx0 = _wx()
TEST_TORQUE_NM = 1000.0
force_api.GetTorqueAttr().Set(Gf.Vec3f(TEST_TORQUE_NM, 0.0, 0.0))
sim.step(render=False)
imu.update(dt=sim_dt)
wx1 = _wx()
dwx = wx1 - wx0
effective_i_tilt = TEST_TORQUE_NM * sim_dt / dwx if dwx != 0 else float("inf")
print(f"TORQUE test: applied={TEST_TORQUE_NM} N.m, dwx={dwx:.6f} rad/s over one step -> "
      f"effective_i_tilt={effective_i_tilt:.6f} kg.m^2 (intended={i_tilt:.1f} kg.m^2, "
      f"correction_factor={effective_i_tilt / i_tilt:.8f})")

print("done")
simulation_app.close()
