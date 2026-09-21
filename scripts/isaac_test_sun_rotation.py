import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np
from pxr import Gf, UsdGeom
from isaacsim.core.api import World

sys.path.insert(0, args.lunarsim_root)
from lunarsim.adapters.isaac.lighting import create_sun_light
from lunarsim.core.lighting.sun import SunPosition

world = World(stage_units_in_meters=1.0)
stage = world.stage

test_cases = [
    ("zenith (elev=90)", SunPosition(elevation_deg=90.0, azimuth_deg=0.0)),
    ("north horizon (elev=0, az=0)", SunPosition(elevation_deg=0.0, azimuth_deg=0.0)),
    ("east horizon (elev=0, az=90)", SunPosition(elevation_deg=0.0, azimuth_deg=90.0)),
    ("45deg, az=45", SunPosition(elevation_deg=45.0, azimuth_deg=45.0)),
    ("low south-pole-like (elev=2, az=270)", SunPosition(elevation_deg=2.0, azimuth_deg=270.0)),
]

all_ok = True
for name, sun_pos in test_cases:
    light = create_sun_light(stage, f"/World/Sun_{name[:4].replace(' ','_')}", sun_pos, angular_diameter_deg=0.53)
    xf = UsdGeom.Xformable(light.GetPrim())
    m = xf.ComputeLocalToWorldTransform(0)
    # local -Z axis in world space = the direction the light shines
    local_minus_z = Gf.Vec3d(0, 0, -1)
    world_dir = m.TransformDir(local_minus_z)
    world_dir = np.array([world_dir[0], world_dir[1], world_dir[2]])
    world_dir /= np.linalg.norm(world_dir)

    el = np.deg2rad(sun_pos.elevation_deg)
    az = np.deg2rad(sun_pos.azimuth_deg)
    expected_dir_to_sun = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    expected_light_shine_dir = -expected_dir_to_sun  # light shines FROM sun TOWARD ground

    dot = np.dot(world_dir, expected_light_shine_dir)
    ok = dot > 0.999
    all_ok &= ok
    print(f"{name}: light shines {world_dir}, expected {expected_light_shine_dir}, dot={dot:.4f}, {'OK' if ok else 'MISMATCH'}")

print("ALL OK" if all_ok else "SOME MISMATCHES")
simulation_app.close()
sys.exit(0 if all_ok else 1)
