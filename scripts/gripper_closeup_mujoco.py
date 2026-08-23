"""MuJoCo counterpart of gripper_closeup.py, using the montage's Pika build.

The montage attaches pika_gripper_base.STL + the two finger STLs at identity under
link6 + z 0.100. This renders that build from the same attachment-frame-relative
viewpoint as the Isaac version, so the two images are directly comparable.

Run:  MUJOCO_GL=egl .venv/bin/python scripts/gripper_closeup_mujoco.py
"""

import pathlib

import mujoco
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
MJCF = ROOT / "montage/rb3_730e_pika/rb3_730e_pika.xml"
OUT = ROOT / "outputs/gripper_mujoco.png"

W, H = 900, 700


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site = data.site("attachment_site")
    origin = np.array(site.xpos)
    R = np.array(site.xmat).reshape(3, 3)
    ax, ay, az = R[:, 0], R[:, 1], R[:, 2]

    target = origin + az * 0.14
    eye = target + ay * 0.62 + az * 0.05

    print(f"attachment origin = {np.round(origin, 4)}")
    print(f"  +x (finger open axis) = {np.round(ax, 3)}")
    print(f"  +y (camera side)      = {np.round(ay, 3)}")
    print(f"  +z (tool forward)     = {np.round(az, 3)}")

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = target
    d = eye - target
    cam.distance = float(np.linalg.norm(d))
    dn = d / cam.distance
    cam.azimuth = float(np.degrees(np.arctan2(dn[1], dn[0])))
    cam.elevation = float(np.degrees(np.arcsin(np.clip(dn[2], -1, 1))))

    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(OUT)
    renderer.close()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
