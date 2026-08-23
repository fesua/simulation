"""Check whether the arms are in collision at the InitMotion reset pose.

Reads net contact forces straight from PhysX for every arm link, so a nonzero force
means that link is actually touching something (stand, table, the other arm, itself).
Operator reported the montage pose looks like it intersects the stand/floor; this is
the measurement that settles it before step 4.

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/check_clearance.py
"""

import argparse
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
LINKS = ["link1", "link2", "link3", "link4", "link5", "link6", "tool", "tcp"]

TABLE = dict(cx=0.55, cy=0.0, hx=0.50, hy=0.55, thick=0.012)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", default=True)
    ap.add_argument("--no-table", dest="table", action="store_false")
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    # Pure-geometry check: gravity off so the pose cannot sag while we look at contacts.
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

    if args.table:
        cube = UsdGeom.Cube.Define(stage, "/World/scene/table")
        cube.CreateSizeAttr(2.0)
        UsdGeom.Xformable(cube).AddTransformOp().Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(TABLE["hx"], TABLE["hy"], TABLE["thick"]))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(TABLE["cx"], TABLE["cy"], -TABLE["thick"]))
        )
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    arts = {}
    for side in MOUNT_FRAME:
        a = SingleArticulation(prim_path=f"/World/cell/{side}_arm/robot", name=f"{side}_arm")
        world.scene.add(a)
        arts[side] = a
    world.reset()
    for a in arts.values():
        a.initialize()

    # Resolve rigid-body prim paths per arm/link, then wrap with contact tracking on.
    views = {}
    for side in MOUNT_FRAME:
        root = f"/World/cell/{side}_arm/robot"
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            name = prim.GetName()
            if not p.startswith(root) or name not in LINKS:
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            key = f"{side}/{name}"
            if key in views:
                continue
            v = RigidPrim(prim_paths_expr=p, name=f"cv_{side}_{name}", track_contact_forces=True)
            v.initialize()
            views[key] = v

    for side, art in arts.items():
        names = list(art.dof_names)
        q = np.zeros(len(names), dtype=np.float32)
        for j, deg in zip(JOINTS, RESET[side]):
            q[names.index(j)] = np.deg2rad(deg)
        art.set_joint_positions(q)
        art.set_joint_velocities(np.zeros_like(q))
        # Drive targets default to 0; without this the arm is actively driven away from
        # the reset pose and every measurement below describes a different posture.
        art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(args.steps):
        world.step(render=False)

    print(f"table collider: {'ON' if args.table else 'OFF'}   links checked: {len(views)}")
    touching = []
    for key in sorted(views):
        f = np.asarray(views[key].get_net_contact_forces())
        mag = float(np.linalg.norm(f))
        pos, _ = views[key].get_world_poses()
        z = float(np.asarray(pos)[0][2])
        flag = ""
        if mag > 1e-3:
            touching.append((key, mag, z))
            flag = "  <-- CONTACT"
        print(f"  {key:16s} |F|={mag:9.3f} N   body z={z:7.4f}{flag}")

    print()
    if touching:
        print(f"FAIL: {len(touching)} link(s) in contact at the reset pose")
        for k, m, z in touching:
            print(f"  {k}  |F|={m:.3f} N  z={z:.4f}")
    else:
        print("PASS: no arm link is in contact at the reset pose")

    app.close()
    return 1 if touching else 0


if __name__ == "__main__":
    raise SystemExit(main())
