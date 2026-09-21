import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import numpy as np
from pxr import UsdGeom
from isaacsim.core.api import World

sys.path.insert(0, args.lunarsim_root)
from lunarsim.adapters.isaac.dust import dust_event_from_disturbance, spawn_dust_burst

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

proto = UsdGeom.Sphere.Define(stage, "/World/DustPrototypes/Speck")
proto.CreateRadiusAttr(1.0)  # PointInstancer widths attr controls actual per-particle size

height = np.zeros((201, 201))
event = dust_event_from_disturbance(origin_m=np.array([0.0, 0.0, 0.1]), intensity=0.8, seed=3)
print("n_particles:", event.n_particles, "speed_range:", event.speed_range_m_s)

instancer = spawn_dust_burst(
    stage, "/World/Dust", event, "/World/DustPrototypes/Speck",
    height, res_m=1.0, fps=30.0,
)

from pxr import Usd
n_time_samples = instancer.GetPositionsAttr().GetNumTimeSamples()
print("n_time_samples:", n_time_samples)

times = instancer.GetPositionsAttr().GetTimeSamples()
print("first/last time codes:", times[0], times[-1])

pos0 = instancer.GetPositionsAttr().Get(times[0])
pos_last = instancer.GetPositionsAttr().Get(times[-1])
pos0 = np.array(pos0)
pos_last = np.array(pos_last)

print("frame 0 mean height (should be ~ launch height, near 0.1):", pos0[:, 2].mean())
print("last frame mean height (should be ~ settled near ground, low):", pos_last[:, 2].mean())
print("max horizontal range at last frame:", np.hypot(pos_last[:,0], pos_last[:,1]).max())

scales0 = np.array(instancer.GetScalesAttr().Get(times[0]))
scales_last = np.array(instancer.GetScalesAttr().Get(times[-1]))
print("scale shrinks after settling (first vs last mean):", scales0.mean(), scales_last.mean())

ok = (
    n_time_samples > 5
    and pos_last[:, 2].mean() < pos0[:, 2].mean()
    and scales_last.mean() <= scales0.mean()
)
print("DUST TEST", "PASS" if ok else "FAIL")
simulation_app.close()
sys.exit(0 if ok else 1)
