"""Geometric clearance at the reset pose, independent of PhysX contact reporting.

Needed because the URDF `tool` link carries a <visual> but no <collision>: the Pika
gripper is invisible to physics, so it can visually intersect the stand or the table
without producing any contact force. This measures actual geometry instead.

For every arm body we transform the link's local mesh AABB by its physics world pose
and report the lowest world z, plus whether the stand's own prims even have colliders.

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/measure_clearance.py
"""

import itertools
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARM_USD = ROOT / "assets/rb3_730e_bakedrzp90/rb3_730e_bakedrzp90.usda"
STAND_USD = ROOT / "assets/dual_rb3_730e_stand_ver3/dual_rb3_730e_stand_ver3.usda"

RESET = {
    "left": [259.0, 75.6, 129.5, -55.6, -131.2, -161.7],
    "right": [-253.7, -76.9, -127.6, 65.7, 143.7, 166.9],
}
JOINTS = [
    "base_joint",
    "shoulder_joint",
    "elbow_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]
MOUNT_FRAME = {"left": "stand_left_arm_base", "right": "stand_right_arm_base"}
BODIES = ["link1", "link2", "link3", "link4", "link5", "link6", "tool"]
TABLE_TOP_Z = 0.0


def quat_to_R(q, np):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom, UsdPhysics

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    world.get_physics_context().set_gravity(0.0)
    stage = get_current_stage()

    add_reference_to_stage(usd_path=str(STAND_USD), prim_path="/World/cell/stand")
    mount_xf = {}
    for prim in stage.Traverse():
        for side, frame in MOUNT_FRAME.items():
            if prim.GetName() == frame and side not in mount_xf:
                mount_xf[side] = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
    for side, m in mount_xf.items():
        p = f"/World/cell/{side}_arm"
        UsdGeom.Xform.Define(stage, p).MakeMatrixXform().Set(m)
        add_reference_to_stage(usd_path=str(ARM_USD), prim_path=f"{p}/robot")

    n_stand_colliders = sum(
        1
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/cell/stand")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    )

    arts = {}
    for side in MOUNT_FRAME:
        a = SingleArticulation(prim_path=f"/World/cell/{side}_arm/robot", name=f"{side}_arm")
        world.scene.add(a)
        arts[side] = a
    world.reset()
    for a in arts.values():
        a.initialize()

    # Local (link-frame) geometry AABB, captured before posing -- USD bounds are static.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    local_aabb, views, has_collider = {}, {}, {}
    for side in MOUNT_FRAME:
        root = f"/World/cell/{side}_arm/robot"
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            name = prim.GetName()
            if not p.startswith(root) or name not in BODIES:
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            key = f"{side}/{name}"
            if key in views:
                continue
            world_b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            T = np.array([[m[i][j] for j in range(4)] for i in range(4)]).T
            Ti = np.linalg.inv(T)
            corners = np.array(
                [
                    Ti @ np.array([x, y, z, 1.0])
                    for x, y, z in itertools.product(
                        (world_b.GetMin()[0], world_b.GetMax()[0]),
                        (world_b.GetMin()[1], world_b.GetMax()[1]),
                        (world_b.GetMin()[2], world_b.GetMax()[2]),
                    )
                ]
            )[:, :3]
            local_aabb[key] = (corners.min(axis=0), corners.max(axis=0))
            has_collider[key] = any(
                c.HasAPI(UsdPhysics.CollisionAPI) for c in Usd.PrimRange(prim)
            )
            v = RigidPrim(prim_paths_expr=p, name=f"mv_{side}_{name}")
            v.initialize()
            views[key] = v

    for side, art in arts.items():
        names = list(art.dof_names)
        q = np.zeros(len(names), dtype=np.float32)
        for j, deg in zip(JOINTS, RESET[side]):
            q[names.index(j)] = np.deg2rad(deg)
        art.set_joint_positions(q)
        art.set_joint_velocities(np.zeros_like(q))
        art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(6):
        world.step(render=False)

    print(f"stand prims with a collider: {n_stand_colliders}")
    print(f"{'body':16s} {'min world z':>12s} {'clearance':>11s}  collider")
    worst = (1e9, None)
    for key in sorted(views):
        pos, quat = views[key].get_world_poses()
        p = np.asarray(pos)[0]
        R = quat_to_R(np.asarray(quat)[0], np)
        lo, hi = local_aabb[key]
        pts = np.array(
            [p + R @ np.array(c) for c in itertools.product(*zip(lo, hi))]
        )
        zmin = float(pts[:, 2].min())
        clear = zmin - TABLE_TOP_Z
        if clear < worst[0]:
            worst = (clear, key)
        print(
            f"{key:16s} {zmin:12.4f} {clear:11.4f}  "
            f"{'yes' if has_collider[key] else 'NO (invisible to physics)'}"
            f"{'   <-- BELOW TABLE' if clear < 0 else ''}"
        )

    print(f"\nworst clearance: {worst[0]:.4f} m at {worst[1]}  (table top z={TABLE_TOP_Z})")
    print("PASS" if worst[0] > 0 else "FAIL: geometry penetrates the table plane")
    app.close()
    return 0 if worst[0] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
