"""Ladder step 2a: convert the canonical RB3-730E URDF to USD.

Uses the kinematics-canonical `rb3_730e.urdf` (NOT rb3_730e_pika_articulated.urdf,
which adds viewer-only prismatic finger DOFs).

merge_fixed_joints stays False so the fixed frames the rest of the rig depends on
(attachment_site, tcp, ft_sensor_*) survive into the USD and can be queried for FK.

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/import_urdf.py
"""

import argparse
import pathlib

REPO = pathlib.Path.home() / "workspace/robotics_lab/rb_servo_server/descriptions"
URDF = REPO / "urdf/rb3_730e.urdf"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=str(URDF))
    ap.add_argument("--fix-base", action="store_true", default=True)
    args = ap.parse_args()
    urdf_path = pathlib.Path(args.urdf)

    OUT_DIR.mkdir(exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    config = URDFImporterConfig(
        urdf_path=str(urdf_path),
        usd_path=str(OUT_DIR),
        merge_fixed_joints=False,
        merge_mesh=False,
        fix_base=True,
        collision_type="Convex Hull",
        joint_drive_type="acceleration",
        joint_target_type="position",
    )
    importer = URDFImporter(config)
    result = importer.import_urdf(config)
    print(f"import result: {result}")

    produced = sorted(p for p in OUT_DIR.rglob("*.usd*"))
    for p in produced:
        print(f"  {p.relative_to(OUT_DIR)}  {p.stat().st_size / 1e6:.2f} MB")

    app.close()
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
