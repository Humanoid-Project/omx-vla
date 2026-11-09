import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the OMX-F URDF to USD.")
parser.add_argument("--force", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

OMX_DIR = Path(__file__).resolve().parent

cfg = UrdfConverterCfg(
    asset_path=str(OMX_DIR / "description" / "urdf" / "omx_f.urdf"),
    usd_dir=str(OMX_DIR / "usd"),
    usd_file_name="omx_f.usd",
    fix_base=True,
    merge_fixed_joints=False,
    convert_mimic_joints_to_normal_joints=False,
    self_collision=False,
    collider_type="convex_decomposition",
    force_usd_conversion=args.force,
    make_instanceable=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=50.0, damping=1.0),
    ),
)
converter = UrdfConverter(cfg)

import os
import sys

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

MASSLESS_LINKS = ["world", "end_effector_link"]
HIDDEN_VISUALS = ["end_effector_link"]

stage = Usd.Stage.Open(converter.usd_path)
root = stage.GetDefaultPrim().GetPath()
for name in MASSLESS_LINKS:
    prim = stage.GetPrimAtPath(root.AppendChild(name))
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(0.001)
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-8, 1e-8, 1e-8))
    print(f"set {prim.GetPath()} mass 0.001 kg", flush=True)
physics_stage = Usd.Stage.Open(str(OMX_DIR / "usd" / "configuration" / "omx_f_physics.usd"))
moved = 0
for prim in physics_stage.Traverse():
    if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith("/colliders/"):
        holder = prim.GetParent()
        if holder.HasAPI(UsdPhysics.CollisionAPI):
            holder.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            holder.RemoveAPI(UsdPhysics.CollisionAPI)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        moved += 1
print(f"collision API on {moved} meshes", flush=True)

import numpy as np
from scipy.spatial import ConvexHull

FINGER_MESHES = {"link6": "follower_07_gripper_motorized", "link7": "follower_08_gripper_gear"}
FINGER_SLAB_EDGES_MM = [-11.0, 8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 66.0]


def surface_samples(points, counts, indices, n=6):
    tris = []
    start = 0
    for c in counts:
        face = indices[start:start + c]
        tris += [(face[0], face[k], face[k + 1]) for k in range(1, c - 1)]
        start += c
    tri = points[np.array(tris)]
    bary = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)])
    return (tri[:, None, 0] * (1 - bary.sum(1))[None, :, None] + tri[:, None, 1] * bary[None, :, 0, None] + tri[:, None, 2] * bary[None, :, 1, None]).reshape(-1, 3)


for link, mesh_name in FINGER_MESHES.items():
    src = stage.GetPrimAtPath(root.AppendPath(f"{link}/collisions/{mesh_name}/node_STL_BINARY_/mesh"))
    mesh = UsdGeom.Mesh(src)
    samples = surface_samples(
        np.array(mesh.GetPointsAttr().Get(), dtype=np.float64),
        mesh.GetFaceVertexCountsAttr().Get(),
        mesh.GetFaceVertexIndicesAttr().Get(),
    )
    local = UsdGeom.Xformable(src).GetLocalTransformation()
    target = next(
        p for p in physics_stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith("/colliders/") and mesh_name in str(p.GetPath())
    )
    for api in (PhysxSchema.PhysxConvexDecompositionCollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.CollisionAPI):
        target.RemoveAPI(api)
    holder = target.GetParent()
    for k, (lo, hi) in enumerate(zip(FINGER_SLAB_EDGES_MM[:-1], FINGER_SLAB_EDGES_MM[1:])):
        pts = samples[(samples[:, 0] >= lo) & (samples[:, 0] <= hi)]
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        remap = {v: i for i, v in enumerate(hull.vertices)}
        slab = UsdGeom.Mesh.Define(physics_stage, holder.GetPath().AppendChild(f"solid_slab_{k}"))
        slab.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        slab.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        slab.CreateFaceVertexIndicesAttr([remap[i] for s in hull.simplices for i in s])
        slab.AddTransformOp().Set(local)
        slab.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(slab.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    print(f"{link}: {len(FINGER_SLAB_EDGES_MM) - 1} solid convex slabs replace {target.GetPath()}", flush=True)
physics_stage.GetRootLayer().Save()
for name in HIDDEN_VISUALS:
    prim = stage.GetPrimAtPath(root.AppendChild(name).AppendChild("visuals"))
    UsdGeom.Imageable(prim).MakeInvisible()
    print(f"hid {prim.GetPath()}", flush=True)
stage.GetRootLayer().Save()
print(f"USD: {converter.usd_path}", flush=True)
sys.stdout.flush()
os._exit(0)
