"""Bake a rotation about z into a copy of pika_gripper.STL, and emit a URDF that
references it with an identity visual origin.

Why baking is necessary (verified 2026-08-17): with merge_fixed_joints=False the
URDF `tool` link becomes a PhysX rigid body whose pose is owned by the articulation
through `tool_joint` (rpy = 0). The visual origin rotation therefore never reaches
the rendered geometry, and a USD xform authored on the tool prim is overwritten by
physics every step. Measured: joint-6 +90 deg changes 12.5% of rendered pixels,
while tool-visual-yaw +90 deg changes 0.4% (i.e. nothing). Rotating the mesh
vertices is the only route that survives the importer.

Run:  .venv/bin/python scripts/bake_tool_mesh.py --deg 90
"""

import argparse
import math
import pathlib
import struct
import xml.etree.ElementTree as ET

import numpy as np

DESC = pathlib.Path.home() / "workspace/robotics_lab/rb_servo_server/descriptions"
SRC_STL = DESC / "meshes/robots/rb3_730e/visual/tool/pika_gripper.STL"
SRC_HULL = DESC / "meshes/robots/rb3_730e/visual/tool/pika_gripper_hull.STL"
SRC_URDF = DESC / "urdf/rb3_730e.urdf"
OUT_MESH = pathlib.Path(__file__).resolve().parent.parent / "assets/meshes"
OUT_URDF = pathlib.Path(__file__).resolve().parent.parent / "assets/urdf"


def read_binary_stl(path: pathlib.Path):
    with open(path, "rb") as f:
        header = f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50).copy()
    floats = raw[:, :48].view("<f4").reshape(n, 4, 3)
    attrs = raw[:, 48:50].copy()
    return header, floats, attrs


def write_binary_stl(path: pathlib.Path, header, floats, attrs) -> None:
    n = floats.shape[0]
    raw = np.zeros((n, 50), dtype=np.uint8)
    raw[:, :48] = floats.astype("<f4").reshape(n, 12).view(np.uint8)
    raw[:, 48:50] = attrs
    with open(path, "wb") as f:
        f.write(header[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", n))
        f.write(raw.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, required=True, help="rotation about z, degrees")
    args = ap.parse_args()

    OUT_MESH.mkdir(parents=True, exist_ok=True)
    OUT_URDF.mkdir(parents=True, exist_ok=True)

    a = math.radians(args.deg)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1.0]])

    tag = f"rz{int(round(args.deg)):+d}".replace("+", "p").replace("-", "m")

    header, floats, attrs = read_binary_stl(SRC_STL)
    # rows are [normal, v0, v1, v2]; rotating all four is correct for a pure rotation
    rotated = floats @ R.T
    mesh_out = OUT_MESH / f"pika_gripper_{tag}.STL"
    write_binary_stl(mesh_out, header, rotated, attrs)

    # The convex hull shares the visual mesh's frame, so it takes the same rotation.
    # rb3_730e.urdf gives the tool link no <collision> at all, which leaves the gripper
    # invisible to physics -- it cannot touch the table, the stand, or a bolt. Emitting
    # a collision element here closes that gap (and lets PhysX derive a sane mass,
    # silencing the negative-mass warning on the tool body).
    hdr_h, fl_h, at_h = read_binary_stl(SRC_HULL)
    hull_out = OUT_MESH / f"pika_gripper_hull_{tag}.STL"
    write_binary_stl(hull_out, hdr_h, fl_h @ R.T, at_h)

    lo = rotated[:, 1:, :].reshape(-1, 3).min(axis=0)
    hi = rotated[:, 1:, :].reshape(-1, 3).max(axis=0)
    print(f"wrote {mesh_out}  ({floats.shape[0]} tris)")
    print(f"wrote {hull_out}  ({fl_h.shape[0]} tris, collision)")
    print(f"  bbox x[{lo[0]:8.1f},{hi[0]:8.1f}] y[{lo[1]:8.1f},{hi[1]:8.1f}] (mm)")

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", str((SRC_URDF.parent / fn).resolve()))

    # rb3_730e.urdf gives `tcp` and `attachment_site` a 2.5 mm <sphere> visual as a frame
    # marker. tcp sits on the fingertip plane, so it renders as a small dot between the
    # gripper tips in every wrist-camera frame.
    # SHRINK rather than delete: these links carry no collision either, so removing the
    # visual leaves them with no geometry at all and the importer then creates no rigid
    # body -- the tcp frame disappears and anything reading its pose gets nan.
    # 1e-5 m is far below one pixel at the wrist camera's working distance.
    MARKER_R = 1e-5
    shrunk = 0
    for link in root.findall("link"):
        if link.get("name") not in ("tcp", "attachment_site"):
            continue
        for vis in link.findall("visual"):
            g = vis.find("geometry")
            sph = g.find("sphere") if g is not None else None
            if sph is not None:
                sph.set("radius", str(MARKER_R))
                shrunk += 1
    print(f"shrank {shrunk} frame-marker sphere(s) to r={MARKER_R} m (tcp / attachment_site)")

    patched = 0
    for link in root.findall("link"):
        if link.get("name") != "tool":
            continue
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            origin.set("rpy", "0.0 0.0 0.0")  # rotation now lives in the mesh
            vis.find("geometry/mesh").set("filename", str(mesh_out.resolve()))
            patched += 1
        if not link.findall("collision"):
            col = ET.SubElement(link, "collision")
            o = ET.SubElement(col, "origin")
            o.set("xyz", "0.0 0.0 0.0")
            o.set("rpy", "0.0 0.0 0.0")
            g = ET.SubElement(col, "geometry")
            m = ET.SubElement(g, "mesh")
            m.set("filename", str(hull_out.resolve()))
            m.set("scale", "0.001 0.001 0.001")
            print(f"added tool <collision> -> {hull_out.name}")
    if patched != 1:
        print(f"FAIL: patched {patched} tool visuals")
        return 1

    name = f"rb3_730e_baked{tag}"
    root.set("name", name)
    out = OUT_URDF / f"{name}.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
