"""Remove only the gripper shell that sits in front of the D405 lens.

The CAD models the camera housing, so a camera placed at the real optical centre is
enclosed by it and every wrist frame comes back black. The real camera obviously does
not see its own housing, so the visual mesh gets the occluding material carved out.

What is removed: triangles whose centroid falls inside the camera's view frustum,
starting at the lens and extending only far enough to clear the shell. The fingers
(x ~ +-50..105 mm, z 140..247.6) sit outside that cone and are untouched -- the carve is
verified by counting what falls in each region before writing.

Collision geometry is NOT carved; only the visual mesh changes.

Run:  .venv/bin/python scripts/carve_lens_view.py --deg 90
"""

import argparse
import math
import pathlib
import struct
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DESC = pathlib.Path.home() / "workspace/robotics_lab/rb_servo_server/descriptions"
OUT_MESH = ROOT / "assets/meshes"
OUT_URDF = ROOT / "assets/urdf"

# D405 optical centre in the tool frame (mm), from scripts/fit_d405.py
LENS = np.array([9.17, 46.01, 119.30])
HALF_ANGLE_DEG = 50.0   # a little wider than the 87 deg horizontal FOV
CARVE_DEPTH_MM = 30.0   # far enough to clear the shell (material reaches z~132.7)


def read_stl(path):
    with open(path, "rb") as f:
        header = f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50).copy()
    return header, raw


def tri_verts(raw):
    return raw[:, 12:48].copy().view("<f4").reshape(-1, 3, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=90.0, help="tool yaw already baked in")
    ap.add_argument("--half-angle", type=float, default=HALF_ANGLE_DEG)
    ap.add_argument("--depth", type=float, default=CARVE_DEPTH_MM)
    args = ap.parse_args()

    tag = f"rz{int(round(args.deg)):+d}".replace("+", "p").replace("-", "m")
    src = OUT_MESH / f"pika_gripper_{tag}.STL"
    if not src.exists():
        print(f"missing {src}; run scripts/bake_tool_mesh.py --deg {args.deg:g} first")
        return 1

    header, raw = read_stl(src)
    v = tri_verts(raw)
    c = v.mean(axis=1)

    d = c - LENS                      # centroid relative to the lens
    fwd = d[:, 2]                     # optical axis is +z in the tool frame
    radial = np.hypot(d[:, 0], d[:, 1])
    cone_r = np.tan(math.radians(args.half_angle)) * np.maximum(fwd, 0.0)
    inside = (fwd > 0.0) & (fwd < args.depth) & (radial <= cone_r)

    fingers = (np.abs(c[:, 0]) > 40.0) & (c[:, 2] > 135.0)
    print(f"{src.name}: {len(v)} triangles")
    print(f"  carve cone: half-angle {args.half_angle:g} deg, depth {args.depth:g} mm from the lens")
    print(f"  removed              : {inside.sum()} ({100*inside.mean():.2f}%)")
    print(f"  of which finger tris : {(inside & fingers).sum()}   <- must be 0")
    if (inside & fingers).sum():
        print("FAIL: the carve would eat finger geometry; narrow the cone or depth")
        return 1

    kept = raw[~inside]
    out = OUT_MESH / f"pika_gripper_{tag}_carved.STL"
    with open(out, "wb") as f:
        f.write(header[:80].ljust(80, b"\0"))
        f.write(struct.pack("<I", len(kept)))
        f.write(kept.tobytes())
    kv = tri_verts(kept).reshape(-1, 3)
    print(f"  wrote {out.name}  ({len(kept)} tris)")
    print(f"    bbox x[{kv[:,0].min():7.1f},{kv[:,0].max():7.1f}]"
          f" y[{kv[:,1].min():7.1f},{kv[:,1].max():7.1f}]"
          f" z[{kv[:,2].min():7.1f},{kv[:,2].max():7.1f}]")

    # URDF: carved mesh for <visual>, untouched hull for <collision>
    hull = OUT_MESH / f"pika_gripper_hull_{tag}.STL"
    tree = ET.parse(DESC / "urdf/rb3_730e.urdf")
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", str((DESC / "urdf" / fn).resolve()))
    for link in root.findall("link"):
        if link.get("name") != "tool":
            continue
        for vis in link.findall("visual"):
            vis.find("origin").set("rpy", "0.0 0.0 0.0")
            vis.find("geometry/mesh").set("filename", str(out.resolve()))
        if not link.findall("collision"):
            col = ET.SubElement(link, "collision")
            o = ET.SubElement(col, "origin")
            o.set("xyz", "0.0 0.0 0.0")
            o.set("rpy", "0.0 0.0 0.0")
            g = ET.SubElement(col, "geometry")
            m = ET.SubElement(g, "mesh")
            m.set("filename", str(hull.resolve()))
            m.set("scale", "0.001 0.001 0.001")
    name = f"rb3_730e_carved{tag}"
    root.set("name", name)
    urdf_out = OUT_URDF / f"{name}.urdf"
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)
    print(f"  wrote {urdf_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
