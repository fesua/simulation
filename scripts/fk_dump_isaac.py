"""Ladder step 2c: FK from the imported USD, evaluated at the same joint configs
as the MuJoCo reference.

Gravity is zeroed and the base is fixed, so this is pure kinematics: joint state is
written directly and the resulting link poses are read back from PhysX (authoritative,
unlike the USD prim transforms which fabric may not write back).

Reads  outputs/fk_mujoco.json  (for the joint configurations)
Writes outputs/fk_isaac.json

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/fk_dump_isaac.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
USD = ROOT / "assets/rb3_730e/rb3_730e.usda"
REF = ROOT / "outputs/fk_mujoco.json"
OUT = ROOT / "outputs/fk_isaac.json"

PRIM = "/World/robot"


def main() -> int:
    ref = json.loads(REF.read_text())
    joints = ref["joints"]
    want_frames = ref["frames"]
    configs = [r["q_deg"] for r in ref["records"]]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 240.0)
    world.get_physics_context().set_gravity(0.0)

    add_reference_to_stage(usd_path=str(USD), prim_path=PRIM)
    art = SingleArticulation(prim_path=PRIM, name="rb3")
    world.scene.add(art)
    world.reset()
    art.initialize()

    dof_names = list(art.dof_names)
    body_names = list(art._articulation_view.body_names)
    print("dof_names :", dof_names)
    print("body_names:", body_names)

    missing = [f for f in want_frames if f not in body_names]
    if missing:
        print(f"NOTE: frames absent as physics bodies (fused by PhysX): {missing}")

    dof_index = [dof_names.index(j) for j in joints]

    # Per-link poses come from RigidPrim views (the Articulation itself only
    # exposes the root pose). Link prims are nested (world/link0/link1/.../link6/
    # attachment_site/tcp) and share names with their visual-mesh children, so
    # resolve paths by looking for the applied rigid-body API rather than by name.
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    body_paths = {}
    for prim in get_current_stage().Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_paths.setdefault(prim.GetName(), str(prim.GetPath()))

    have = [f for f in want_frames if f in body_names and f in body_paths]
    unresolved = [f for f in want_frames if f in body_names and f not in body_paths]
    if unresolved:
        print(f"NOTE: no rigid-body prim resolved for: {unresolved}")
    for f in have:
        print(f"  {f:18s} -> {body_paths[f]}")

    views = {f: RigidPrim(prim_paths_expr=body_paths[f], name=f"view_{f}") for f in have}
    for v in views.values():
        v.initialize()

    records = []
    for cfg in configs:
        q = np.zeros(len(dof_names), dtype=np.float32)
        for idx, deg in zip(dof_index, cfg):
            q[idx] = np.deg2rad(deg)
        art.set_joint_positions(q)
        art.set_joint_velocities(np.zeros_like(q))
        world.step(render=False)

        frames = {}
        for name, view in views.items():
            pos, quat = view.get_world_poses()
            frames[name] = {
                "pos": [float(v) for v in np.asarray(pos)[0]],
                "quat_wxyz": [float(v) for v in np.asarray(quat)[0]],
            }
        records.append({"q_deg": cfg, "frames": frames})

    body_index = {f: i for i, f in enumerate(have)}

    OUT.write_text(
        json.dumps(
            {"source": str(USD), "frames": sorted(body_index), "records": records}, indent=1
        )
    )
    print(f"wrote {OUT}  ({len(records)} configs)")
    if "attachment_site" in body_index:
        p = records[0]["frames"]["attachment_site"]["pos"]
        print(f"zero-pose attachment_site = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
