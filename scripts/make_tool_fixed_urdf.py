"""Write an Isaac-local copy of rb3_730e.urdf with an overridden tool visual yaw.

robotics_lab is left untouched: this writes a patched copy under simulation/assets/urdf/
with all mesh filenames rewritten to absolute paths so the copy can live anywhere.

Background (see CLAUDE.md): the tool visual origin carries rpy z=+90 deg, which commit
a0ee3f7 inherited from the older tool.STL mounting and did not re-derive when the mesh
was swapped to pika_gripper.STL. Operator reports the rendered gripper is 90 deg off.

Run:  .venv/bin/python scripts/make_tool_fixed_urdf.py --yaw-deg 0
"""

import argparse
import math
import pathlib
import xml.etree.ElementTree as ET

SRC = (
    pathlib.Path.home()
    / "workspace/robotics_lab/rb_servo_server/descriptions/urdf/rb3_730e.urdf"
)
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets/urdf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yaw-deg",
        type=float,
        required=True,
        help="net yaw (deg) for the tool link's visual origin; source URDF uses +90",
    )
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SRC)
    root = tree.getroot()
    base = SRC.parent

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", str((base / fn).resolve()))

    patched = 0
    for link in root.findall("link"):
        if link.get("name") != "tool":
            continue
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            if origin is None:
                origin = ET.SubElement(vis, "origin")
                origin.set("xyz", "0.0 0.0 0.0")
            old = origin.get("rpy", "0 0 0")
            r, p, _ = (float(v) for v in old.split())
            origin.set("rpy", f"{r} {p} {math.radians(args.yaw_deg)}")
            patched += 1
            print(f"tool visual rpy: '{old}' -> '{origin.get('rpy')}'")

    if patched != 1:
        print(f"FAIL: expected exactly one tool visual, patched {patched}")
        return 1

    tag = args.name or f"rb3_730e_toolyaw{int(args.yaw_deg):+d}".replace("+", "p").replace(
        "-", "m"
    )
    root.set("name", tag)
    out = OUT_DIR / f"{tag}.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
