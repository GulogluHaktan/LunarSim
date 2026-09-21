import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--lunarsim-root", type=str, default="/workspace/LunarSim")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World

sys.path.insert(0, args.lunarsim_root)
from lunarsim.adapters.isaac.rocks import spawn_rocks
from lunarsim.core.terrain.rocks import RockField
from dataclasses import dataclass

# Build a REAL external rock asset file (as production usage expects --
# prototype_usd_paths pointing at real .usd files, not internal stage paths).
proto_path = "/tmp/rock_proto.usd"
proto_stage = Usd.Stage.CreateNew(proto_path)
sph = UsdGeom.Sphere.Define(proto_stage, "/Rock")
sph.CreateRadiusAttr(0.5)
UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
sph.GetPrim().CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
proto_stage.SetDefaultPrim(proto_stage.GetPrimAtPath("/Rock"))
proto_stage.GetRootLayer().Save()

world = World(stage_units_in_meters=1.0, physics_prim_path="/World/PhysicsScene")
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
ps = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
ps.CreateGravityMagnitudeAttr(1.62)

@dataclass
class FakeTile:
    height: object
    res_m: float
    rocks: object

height = np.zeros((21,21))
rocks = RockField(x_m=np.array([0.0]), y_m=np.array([0.0]), diameter_m=np.array([2.0]))
tile = FakeTile(height=height, res_m=1.0, rocks=rocks)

prims = spawn_rocks(stage, "/World/Rocks", tile, [proto_path], np.random.default_rng(0), add_collision=True)
rock_prim = prims[0]
print("rock prim type (resolved via external reference):", rock_prim.GetTypeName())
print("rock applied schemas:", rock_prim.GetAppliedSchemas())

xf = UsdGeom.Xformable(rock_prim)
ops = xf.GetOrderedXformOps()
ops[0].Set(Gf.Vec3d(0.0, 0.0, 1.0))  # rock center z=1.0, radius 1.0 (2.0 diameter) -> bottom 0, top 2.0

ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
ball.CreateRadiusAttr(0.2)
bxf = UsdGeom.Xformable(ball.GetPrim())
bxf.ClearXformOpOrder()
bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
UsdPhysics.MassAPI.Apply(ball.GetPrim()).CreateMassAttr(1.0)

world.reset()

for i in range(300):
    world.step(render=False)
    m2 = UsdGeom.Xformable(ball.GetPrim()).ComputeLocalToWorldTransform(0)
    p = m2.ExtractTranslation()
    if i % 30 == 0 or i == 299:
        print(f"step {i}: ball pos = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

print("expected resting z ~= 2.2 (rock top 2.0 + ball radius 0.2)")
simulation_app.close()
