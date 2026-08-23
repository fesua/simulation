"""Ladder step 2b: reference FK from the MJCF, for cross-checking the Isaac USD import.

MJCF (rb3_730e.xml) and URDF (rb3_730e.urdf) are separate files generated from the
same CAD. Agreement between MuJoCo-on-MJCF and Isaac-on-USD therefore validates BOTH
the MJCF<->URDF consistency and the fidelity of the URDF->USD conversion.

Frame note: the MJCF site named "tcp" sits at link6 + z 0.100, which is the URDF's
`attachment_site` -- NOT the URDF's `tcp` (which adds the Pika 0.247642 fingertip
offset on top). We compare against `attachment_site` to avoid that trap.

Emits outputs/fk_mujoco.json, which also carries the joint configurations so the
Isaac side evaluates exactly the same poses.

Run:  .venv/bin/python scripts/fk_dump_mujoco.py
"""

import json
import pathlib

import mujoco
import numpy as np

MJCF = (
    pathlib.Path.home()
    / "workspace/robotics_lab/rb_servo_server/descriptions/mjcf/rb3_730e/rb3_730e.xml"
)
OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs/fk_mujoco.json"

JOINTS = [
    "base_joint",
    "shoulder_joint",
    "elbow_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]
BODIES = ["link1", "link2", "link3", "link4", "link5", "link6"]

# InitMotion reset pose (rb_gui/rb_servo_gui/app.py:216-217), degrees.
RESET_LEFT_DEG = [259.0, 75.6, 129.5, -55.6, -131.2, -161.7]
RESET_RIGHT_DEG = [-253.7, -76.9, -127.6, 65.7, 143.7, 166.9]

# Elbow (J3) is physically limited to +/-150 deg; sampling wider would place the
# reference in a configuration the real arm cannot reach.
LIMITS_DEG = [(-180, 180), (-180, 180), (-150, 150), (-180, 180), (-180, 180), (-180, 180)]


def make_configs(n_random: int = 24, seed: int = 7) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    configs = [
        [0.0] * 6,
        RESET_LEFT_DEG,
        RESET_RIGHT_DEG,
    ]
    for _ in range(n_random):
        configs.append([float(rng.uniform(lo, hi)) for lo, hi in LIMITS_DEG])
    return configs


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)

    qadr = [model.joint(j).qposadr[0] for j in JOINTS]
    configs = make_configs()

    records = []
    for cfg in configs:
        for a, deg in zip(qadr, cfg):
            data.qpos[a] = np.deg2rad(deg)
        mujoco.mj_forward(model, data)

        frames = {}
        for b in BODIES:
            body = data.body(b)
            frames[b] = {
                "pos": [float(v) for v in body.xpos],
                "xmat": [float(v) for v in body.xmat],
            }
        # MJCF site "tcp" == URDF frame "attachment_site" (link6 + z 0.100)
        site = data.site("tcp")
        frames["attachment_site"] = {
            "pos": [float(v) for v in site.xpos],
            "xmat": [float(v) for v in site.xmat],
        }
        records.append({"q_deg": cfg, "frames": frames})

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": str(MJCF),
                "joints": JOINTS,
                "frames": BODIES + ["attachment_site"],
                "records": records,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}  ({len(records)} configs)")
    ee = records[0]["frames"]["attachment_site"]["pos"]
    print(f"zero-pose attachment_site = ({ee[0]:.6f}, {ee[1]:.6f}, {ee[2]:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
