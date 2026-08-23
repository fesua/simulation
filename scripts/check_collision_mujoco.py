"""Exact collision check at the reset pose, using MuJoCo on the dual-arm MJCF.

Why not Isaac: the URDF `tool` link has no <collision>, so PhysX cannot see the gripper
at all, and attempts to measure geometry through USD kept mixing stale/mm-scaled data
with metre-scale physics poses. MuJoCo's MJCF carries real collision hulls for the arms
AND the stand, and does exact narrow-phase collision, so contacts here are trustworthy.

Adds a ground/table plane at z=0 (the stand base plane) and reports every contact with
its penetration depth and the geom pair involved.

Run:  .venv/bin/python scripts/check_collision_mujoco.py
"""

import argparse
import pathlib
import re

import mujoco
import numpy as np

DESC = pathlib.Path.home() / "workspace/robotics_lab/rb_servo_server/descriptions"
DUAL = DESC / "mjcf/dual_rb3_730e_ver3.xml"

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
PREFIX = {"left": "dual_rb3_730e_left_", "right": "dual_rb3_730e_right_"}


def build_xml(add_plane: bool) -> str:
    xml = DUAL.read_text()
    # from_xml_string loses the file's directory, so relative asset references must be
    # absolutised (the dual model <model file=...> and the mesh search path).
    xml = xml.replace('file="rb3_730e/rb3_730e.xml"', f'file="{DESC / "mjcf/rb3_730e/rb3_730e.xml"}"')
    xml = xml.replace('meshdir="../meshes"', f'meshdir="{DESC / "meshes"}"')
    # descriptions/camera/realsense_d435f.obj is referenced but absent from the repo
    xml = re.sub(r'\s*<mesh name="realsense_d435f"[^>]*>', "", xml)
    xml = re.sub(r'\s*<geom name="[\w_]*camera[\w_]*_vis"[^>]*>', "", xml)
    xml = re.sub(r'\s*<body name="\w*camera_mount"[^>]*>\s*</body>', "", xml, flags=re.S)
    if add_plane:
        xml = xml.replace(
            "<worldbody>",
            '<worldbody>\n    <geom name="table_plane" type="plane" size="2 2 0.1" pos="0 0 0"/>',
            1,
        )
    return xml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plane", dest="plane", action="store_false", default=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_string(build_xml(args.plane), {})
    data = mujoco.MjData(model)

    for side, degs in RESET.items():
        for j, deg in zip(JOINTS, degs):
            data.joint(PREFIX[side] + j).qpos[0] = np.deg2rad(deg)
    mujoco.mj_forward(model, data)

    def gname(gid):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        return n or f"geom{gid}"

    print(f"table plane: {'ON' if args.plane else 'OFF'}   contacts: {data.ncon}")
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        depth = -float(c.dist)  # positive = penetration
        worst = max(worst, depth)
        print(f"  {gname(c.geom1):38s} <-> {gname(c.geom2):38s}  depth={depth*1000:7.2f} mm")

    # lowest point of any collision geom, as a separate sanity number
    zmin, zmin_g = 1e9, None
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        if gname(gid) == "table_plane":
            continue
        z = float(data.geom_xpos[gid][2] - model.geom_rbound[gid])
        if z < zmin:
            zmin, zmin_g = z, gname(gid)
    print(f"\nlowest collision geom (centre - bounding radius): {zmin:.4f} m at {zmin_g}")
    print(f"deepest penetration: {worst*1000:.2f} mm")
    print("PASS: reset pose is collision-free" if data.ncon == 0 else "FAIL: reset pose collides")
    return 0 if data.ncon == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
