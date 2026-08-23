"""Build the Isaac tool asset from the ARTICULATED Pika URDF, so the jaw can open/close.

Why this replaces the baked single-mesh asset:
  * `rb3_730e.urdf` puts the whole gripper in one rigid `pika_gripper.STL`, so the fingers
    cannot move and the policy's grip channel has nothing to drive.
  * `rb3_730e_pika_articulated.urdf` already splits it into `tool` (pika_gripper_base.STL,
    which also contains the flange adapter and the RFT64 FT body) plus `finger_left` /
    `finger_right` on PRISMATIC joints along +-X. That is exactly the open/close model
    robotics_lab's own runtime uses.
  * Those meshes are exported with the +90 deg tool rotation already baked in, so the
    rotation baking that `bake_tool_mesh.py` had to do is no longer needed.

What this script adds on top of the source URDF:
  * collision hulls for tool + both fingers (source has visuals only, so PhysX saw nothing
    and reported negative mass)
  * frame-marker spheres on `tcp` / `attachment_site` shrunk to sub-pixel -- they otherwise
    render as a dot between the gripper tips in every wrist frame. Shrink, never delete:
    these links have no other geometry, and a link with no geometry gets no rigid body,
    which makes the tcp frame vanish.

Jaw model (matches robotics_lab, see rb_servo_server/config/stack_real.yaml and
collision_monitor.cpp):
    finger_pos = (1 - grip/100) * 0.047     # grip 100 = open (mesh as authored), 0 = closed

Run:  .venv/bin/python scripts/make_articulated_urdf.py
"""

import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
DESC = pathlib.Path.home() / "workspace/robotics_lab/rb_servo_server/descriptions"
SRC = DESC / "urdf/rb3_730e_pika_articulated.urdf"
OUT_URDF = ROOT / "assets/urdf"

TOOL_DIR = DESC / "meshes/robots/rb3_730e/visual/tool"
HULLS = {
    "tool": TOOL_DIR / "pika_gripper_base_hull.STL",
    "finger_left": TOOL_DIR / "pika_finger_left_hull.STL",
    "finger_right": TOOL_DIR / "pika_finger_right_hull.STL",
}
MARKER_R = 1e-5
FINGER_TRAVEL_M = 0.047  # rb_servo_server/config/stack_real.yaml


def main() -> int:
    OUT_URDF.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(SRC)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            mesh.set("filename", str((SRC.parent / fn).resolve()))

    shrunk = added = 0
    for link in root.findall("link"):
        name = link.get("name")

        if name in ("tcp", "attachment_site"):
            for vis in link.findall("visual"):
                g = vis.find("geometry")
                sph = g.find("sphere") if g is not None else None
                if sph is not None:
                    sph.set("radius", str(MARKER_R))
                    shrunk += 1

        if name in HULLS and not link.findall("collision"):
            hull = HULLS[name]
            if not hull.exists():
                print(f"FAIL: missing hull {hull}")
                return 1
            col = ET.SubElement(link, "collision")
            o = ET.SubElement(col, "origin")
            o.set("xyz", "0.0 0.0 0.0")
            o.set("rpy", "0.0 0.0 0.0")
            g = ET.SubElement(col, "geometry")
            m = ET.SubElement(g, "mesh")
            m.set("filename", str(hull.resolve()))
            m.set("scale", "0.001 0.001 0.001")
            added += 1

    fingers = {}
    for j in root.findall("joint"):
        if j.get("name") in ("finger_left_joint", "finger_right_joint"):
            lim = j.find("limit")
            fingers[j.get("name")] = (
                j.get("type"),
                j.find("axis").get("xyz"),
                (lim.get("lower"), lim.get("upper")) if lim is not None else None,
            )

    print(f"shrank {shrunk} frame-marker sphere(s) to r={MARKER_R} m")
    print(f"added {added} collision hull(s): {', '.join(sorted(HULLS))}")
    print("finger joints:")
    for n, (t, ax, lim) in fingers.items():
        print(f"  {n:20s} {t:10s} axis={ax}  limit={lim}")
    print(f"jaw model: finger_pos = (1 - grip/100) * {FINGER_TRAVEL_M}  "
          f"[grip 100 = open, 0 = closed]")

    name = "rb3_730e_pika_articulated_sim"
    root.set("name", name)
    out = OUT_URDF / f"{name}.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
