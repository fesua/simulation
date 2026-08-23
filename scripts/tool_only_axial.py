"""Render ONLY the Pika tool, viewed exactly down its own approach axis, for each
tool-yaw candidate. Everything else (arm links, stand) is hidden, so an N-degree
rotation about the tool axis shows up as an unambiguous N-degree in-plane spin.

All candidates are instantiated in one Isaac session at the reset pose.

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/tool_only_axial.py
"""

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = [
    ("p0", ROOT / "assets/rb3_730e_toolyawp0/rb3_730e_toolyawp0.usda"),
    ("p90", ROOT / "assets/rb3_730e_toolyawp90/rb3_730e_toolyawp90.usda"),
    ("p180", ROOT / "assets/rb3_730e_toolyawp180/rb3_730e_toolyawp180.usda"),
    ("p270", ROOT / "assets/rb3_730e_toolyawp270/rb3_730e_toolyawp270.usda"),
]
JOINTS = [
    "base_joint",
    "shoulder_joint",
    "elbow_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]
RESET_LEFT = [259.0, 75.6, 129.5, -55.6, -131.2, -161.7]
W, H = 800, 800


def main() -> int:
    global CANDIDATES
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=[t for t, _ in CANDIDATES])
    a = ap.parse_args()
    # One candidate per process: stacking all four at the same world pose makes PhysX
    # push the interpenetrating arms apart, so their tool frames end up different and
    # the comparison is meaningless.
    CANDIDATES = [c for c in CANDIDATES if c[0] == a.tag]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from PIL import Image
    from pxr import UsdGeom, UsdLux, UsdPhysics

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    world.get_physics_context().set_gravity(0.0)
    stage = get_current_stage()
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1500.0)

    arts = {}
    for i, (tag, usd) in enumerate(CANDIDATES):
        root = f"/World/c{i}"
        add_reference_to_stage(usd_path=str(usd), prim_path=root)
        a = SingleArticulation(prim_path=root, name=f"arm_{tag}")
        world.scene.add(a)
        arts[tag] = (root, a)

    world.reset()
    for _, a in arts.values():
        a.initialize()

    # RigidPrim views must be created here (right after initialize); wrapping them
    # later -- after stepping and after MakeInvisible -- fails inside get_linear_velocities.
    tcp_views = {}
    for i, (tag, _) in enumerate(CANDIDATES):
        root = f"/World/c{i}"
        tcp_prim = next(
            pr
            for pr in stage.Traverse()
            if str(pr.GetPath()).startswith(root)
            and pr.GetName() == "tcp"
            and pr.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        v = RigidPrim(prim_paths_expr=str(tcp_prim.GetPath()), name=f"tcpv_{tag}")
        v.initialize()
        tcp_views[tag] = v

    for tag, (root, a) in arts.items():
        names = list(a.dof_names)
        q = np.zeros(len(names), dtype=np.float32)
        for j, deg in zip(JOINTS, RESET_LEFT):
            q[names.index(j)] = np.deg2rad(deg)
        a.set_joint_positions(q)
        a.set_joint_velocities(np.zeros_like(q))
    for _ in range(10):
        world.step(render=True)

    # Capture tool frames BEFORE hiding: MakeInvisible() invalidates the physics
    # tensor view and get_world_poses() then throws "Failed to get rigid body transforms".
    frames = {}
    for tag in tcp_views:
        pos, quat = tcp_views[tag].get_world_poses()
        frames[tag] = (np.asarray(pos)[0].copy(), np.asarray(quat)[0].copy())

    # NOTE: do NOT hide the arm links. USD visibility is inherited, and the tool lives
    # at .../link6/attachment_site/tool -- making link6 invisible hides the tool too.
    # Instead just dolly the camera in until the tool fills the frame.

    for i, (tag, _) in enumerate(CANDIDATES):
        root = f"/World/c{i}"
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if not p.startswith(root):
                continue
            name = prim.GetName()
            if name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6"):
                UsdGeom.Imageable(prim).MakeInvisible()

    for i, (tag, _) in enumerate(CANDIDATES):
        p, q = frames[tag]
        w_, x_, y_, z_ = q
        R = np.array([
            [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
            [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
            [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)],
        ])
        axis = R[:, 2]
        tgt = p - axis * 0.11
        eye = tgt + axis * 0.30
        cam = rep.create.camera(
            position=tuple(float(v) for v in eye),
            look_at=tuple(float(v) for v in tgt),
            clipping_range=(0.01, 100.0),
        )
        rp = rep.create.render_product(cam, (W, H))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        arr = np.asarray([])
        for _ in range(20):
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
            arr = np.asarray(ann.get_data())
            if arr.size:
                break
        out = ROOT / f"outputs/toolonly_{tag}.png"
        Image.fromarray(arr[..., :3].astype("uint8")).save(out)
        print(f"{tag:5s} tool axis={np.round(axis, 3)}  wrote {out.name}")
        ann.detach([rp])

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
