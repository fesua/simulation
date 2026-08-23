"""Render the Pika tool from a viewpoint fixed in the attachment_site frame.

Because the camera is defined in the tool's own frame, the resulting image is
independent of where the arm is or which world convention the scene uses -- so it
can be compared directly against the MuJoCo montage build to settle whether the
tool is mounted with the right rotation about link6 +z.

View: camera on the attachment frame's +y side looking at the tool, with world-up
replaced by the attachment frame's +z. Fingers open along attachment x.

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/gripper_closeup.py
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARM_USD = ROOT / "assets/rb3_730e/rb3_730e.usda"
OUT = ROOT / "outputs/gripper_isaac.png"

W, H = 900, 700


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Usd, UsdGeom, UsdLux

    world = World(stage_units_in_meters=1.0)
    world.get_physics_context().set_gravity(0.0)
    stage = get_current_stage()
    add_reference_to_stage(usd_path=str(ARM_USD), prim_path="/World/robot")

    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1200.0)
    key = UsdLux.DistantLight.Define(stage, "/World/key")
    key.CreateIntensityAttr(2500.0)

    world.reset()
    for _ in range(3):
        world.step(render=False)

    att = None
    for p in stage.Traverse():
        if p.GetName() == "attachment_site" and p.GetTypeName() == "Xform":
            att = p
            break
    m = UsdGeom.Xformable(att).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    T = np.array([[m[i][j] for j in range(4)] for i in range(4)]).T
    origin, ax, ay, az = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]

    # Frame on the tool's actual world bounds rather than a guessed offset.
    tool_prim = next(
        p
        for p in stage.Traverse()
        if p.GetName() == "tool" and "attachment_site" in str(p.GetPath())
    )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    rng = cache.ComputeWorldBound(tool_prim).ComputeAlignedRange()
    lo = np.array([rng.GetMin()[i] for i in range(3)])
    hi = np.array([rng.GetMax()[i] for i in range(3)])
    target = (lo + hi) / 2.0
    span = float(np.linalg.norm(hi - lo))
    eye = target + ay * (span * 1.3) + az * (span * 0.55) + ax * (span * 0.5)
    print(f"tool world bbox span = {span:.4f} m, target = {np.round(target, 4)}")
    camera = rep.create.camera(
        position=tuple(float(v) for v in eye), look_at=tuple(float(v) for v in target)
    )
    rp = rep.create.render_product(camera, (W, H))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    print(f"attachment origin = {np.round(origin, 4)}")
    print(f"  +x (finger open axis) = {np.round(ax, 3)}")
    print(f"  +y (camera side)      = {np.round(ay, 3)}")
    print(f"  +z (tool forward)     = {np.round(az, 3)}")

    arr = np.asarray([])
    for _ in range(20):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        arr = np.asarray(rgb.get_data())
        if arr.size:
            break
    if arr.size:
        from PIL import Image

        Image.fromarray(arr[..., :3].astype("uint8")).save(OUT)
        print(f"wrote {OUT}")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
