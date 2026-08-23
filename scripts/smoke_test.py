"""Ladder step 1 smoke test: launch Isaac Sim headless, build a trivial scene, render.

Verifies the install can (a) start the Kit app on this GPU, (b) step physics,
(c) produce RGB frames off-screen — the three capabilities every later ladder
step depends on.

Run:  .venv-isaac/bin/python scripts/smoke_test.py
"""

import argparse
import pathlib
import sys

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=60, help="physics/render steps before capture")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    # Imports below must happen AFTER SimulationApp() — Kit extensions are not
    # on the path until the app boots.
    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    from pxr import Gf, UsdLux

    world = World(stage_units_in_meters=1.0)
    GroundPlane(prim_path="/World/ground", size=10.0)

    stage = world.stage
    light = UsdLux.DistantLight.Define(stage, "/World/light")
    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(1.0)

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/cube",
            name="cube",
            position=np.array([0.0, 0.0, 1.0]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.15, 0.55, 0.35]),
        )
    )

    camera = rep.create.camera(position=(2.2, -2.2, 1.6), look_at=(0.0, 0.0, 0.25))
    render_product = rep.create.render_product(camera, (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])

    world.reset()
    for _ in range(args.frames):
        world.step(render=True)

    # The RGB annotator stays empty until the render pipeline has actually
    # produced a frame for this render product; orchestrator.step() drives that.
    arr = np.asarray([])
    for attempt in range(30):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        arr = np.asarray(rgb.get_data())
        if arr.size:
            print(f"OK  annotator populated after {attempt + 1} orchestrator step(s)")
            break

    if arr.size == 0:
        print("FAIL: renderer returned an empty frame", file=sys.stderr)
        app.close()
        return 1

    from PIL import Image

    out_path = OUT_DIR / "smoke_test.png"
    Image.fromarray(arr[..., :3].astype("uint8")).save(out_path)

    pos, _ = cube.get_world_pose()
    print(f"OK  frame={arr.shape} dtype={arr.dtype}")
    print(f"OK  cube settled at z={float(pos[2]):.4f} (dropped from 1.0)")
    print(f"OK  wrote {out_path}")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
