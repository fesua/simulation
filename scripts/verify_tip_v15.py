#!/usr/bin/env python3
"""Gate the v15 fingertip asset inside Isaac, before any of it is trusted for scoring.

build_tip_v15.py checks the MESHES against the design frame. This checks the ASSET: that
PhysX still parses the articulation, that the jaw still opens and closes over the same
travel, that the pinch faces land where the offline maths said they would, and that the
policy's own camera actually sees the new tip. Every one of those has failed silently on
this project at least once -- a black wrist frame scored ten episodes at 0/0 (CLAUDE.md
section 20), and a mesh frame assumed rather than checked cost three retractions
(gripper_sections.py's docstring).

Run:  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/verify_tip_v15.py
      PIKA_TIP=orig ... same script against the old tip, for the side-by-side
"""
from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIKA_TIP = os.environ.get("PIKA_TIP", "v15").lower()
ARM_USD = ROOT / ("assets/rb3_730e_pika_tip_v15/rb3_730e_pika_articulated_sim.usda"
                  if PIKA_TIP == "v15" else
                  "assets/rb3_730e_pika_articulated_sim/rb3_730e_pika_articulated_sim.usda")
OUT = ROOT / "outputs"
# Must track eval_closed_loop.py: the v15 tip is placed on the measured 98 mm full-open
# gap, so the stroke is 49 mm/side, not the old unmeasured 47.
FINGER_TRAVEL_M = float(os.environ.get(
    "FINGER_TRAVEL_M", "0.049" if PIKA_TIP == "v15" else "0.047"))
# CLAUDE.md section 14: the D405 colour optical centre in the tool frame, recovered from the
# gripper CAD (baseline 18.34 mm against the D405's nominal 18). Same numbers the eval uses.
LENS = (0.00917, 0.04601, 0.11930)


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np
    import imageio.v2 as imageio
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    print(f"== asset: {ARM_USD.relative_to(ROOT)}")
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 500.0,
                  rendering_dt=1.0 / 500.0)
    world.get_physics_context().set_gravity(0.0)
    stage = get_current_stage()
    add_reference_to_stage(usd_path=str(ARM_USD), prim_path="/World/robot")
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1500.0)

    art = SingleArticulation(prim_path="/World/robot", name="arm")
    world.reset()
    art.initialize()
    names = list(art.dof_names)
    print(f"  dofs ({len(names)}): {', '.join(names)}")
    ok = len(names) == 8 and {"finger_left_joint", "finger_right_joint"} <= set(names)
    print(f"  articulation intact: {'PASS' if ok else 'FAIL'}")
    if not ok:
        app.close()
        return 1

    # The collider prims carry purpose="guide", so a default BBoxCache ignores them. Ask for
    # guide explicitly -- otherwise this measures the VISUAL tip and would happily pass while
    # the physics jaw was somewhere else entirely.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    col = {}
    for p in stage.Traverse():
        n = p.GetName()
        if n in ("tip_v15_col", "pika_finger_left_hull", "pika_finger_right_hull"):
            side = "left" if "left" in str(p.GetPath()) else "right"
            col.setdefault(side, str(p.GetPath()))
    if len(col) != 2:
        # v15 names both sides tip_v15_col, so disambiguate by the parent link.
        col = {}
        for p in stage.Traverse():
            if p.GetName() in ("finger_left", "finger_right") and p.HasAPI(UsdPhysics.RigidBodyAPI):
                col[p.GetName().split("_")[1]] = str(p.GetPath())
    print(f"  jaw prims: {', '.join(f'{k}={v.split(chr(47))[-1]}' for k, v in sorted(col.items()))}")

    def jaw_faces():
        """Inner (facing the other jaw) X of each finger's COLLIDER, in world mm."""
        out = {}
        cache.Clear()
        for side, path in col.items():
            b = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            lo, hi = b.GetMin()[0] * 1e3, b.GetMax()[0] * 1e3
            out[side] = hi if side == "left" else lo
        return out

    n_idx = [names.index(j) for j in ("finger_left_joint", "finger_right_joint")]
    print("\n== jaw travel (world mm; the arm is at all-joints-zero, tool axis == world x)")
    print("  grip   cmd travel     left face   right face      gap")
    rows = {}
    for grip in (100.0, 50.0, 0.0):
        pos = (1.0 - grip / 100.0) * FINGER_TRAVEL_M
        q = np.zeros(len(names), dtype=np.float32)
        q[n_idx[0]], q[n_idx[1]] = pos, -pos
        art.set_joint_positions(q)
        for _ in range(3):
            world.step(render=False)
        f = jaw_faces()
        # SIGNED, deliberately. The left jaw's inner face is its max X and the right's is
        # its min X, so gap = right - left goes NEGATIVE once the two cross. Taking abs()
        # here reported a 92.9 mm stroke for a 94.0 mm travel, because the 0.54 mm of
        # overlap at the closed stop came back as +0.54.
        gap = f["right"] - f["left"]
        rows[grip] = gap
        print(f"  {grip:5.0f}   {pos * 1e3:7.2f} mm   {f['left']:9.3f}   {f['right']:9.3f}   "
              f"{gap:7.3f} mm")
    stroke = rows[100.0] - rows[0.0]
    want = 2 * FINGER_TRAVEL_M * 1e3
    print(f"  stroke {stroke:.3f} mm  vs 2 x travel {want:.1f} mm  "
          f"({'PASS' if abs(stroke - want) < 0.5 else 'FAIL'})")
    print(f"  full-open gap {rows[100.0]:.3f} mm   closed gap {rows[0.0]:+.3f} mm")
    print(f"  hardware (vernier, 2026-09-04): full-open ~98 mm, jaws close to contact "
          f"({'PASS' if abs(rows[100.0] - 98.0) < 1.0 and abs(rows[0.0]) < 0.5 else 'CHECK'})")

    # ---- does the blade actually COLLIDE? ----------------------------------------------
    # The jaw numbers above are mesh bounds: they prove placement, not contact. An SDF
    # collider that PhysX declined to build would pass every one of them and then let the
    # jaws shut straight through a bolt. So close the drive on a STATIC cylinder of known
    # diameter and read where the joints stall. The bolt is static and gravity is off, so
    # the only thing that can stop the jaws is the collider.
    tool = next(str(p.GetPath()) for p in stage.Traverse()
                if p.GetName() == "tool" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    tool_m = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(tool))
                        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T

    # The URDF import leaves the prismatic finger drives too soft to move against anything;
    # the first version of this probe reported "stalled at 98 mm" for every diameter, which
    # was not a dead collider but a dead DRIVE. Same gains eval_closed_loop.py runs.
    kps = np.full(len(names), 1.0e7, dtype=np.float32)
    kds = np.full(len(names), 1.0e5, dtype=np.float32)
    art.get_articulation_controller().set_gains(kps=kps, kds=kds)

    def probe(diam_mm=None, z_tool_mm=240.0):
        """Stall gap when the jaws close on a static cylinder of this diameter.
        diam_mm=None is the CONTROL: nothing between the jaws, so they must reach ~0."""
        path = "/World/probe"
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
        if diam_mm is not None:
            cy = UsdGeom.Cylinder.Define(stage, path)
            cy.CreateRadiusAttr(diam_mm / 2000.0)
            cy.CreateHeightAttr(0.05)
            cy.CreateAxisAttr("Y")      # across the jaw, like a bolt lying in the grasp
            m = Gf.Matrix4d(*tool_m.T.reshape(-1).tolist())
            UsdGeom.Xformable(cy).AddTransformOp().Set(
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, z_tool_mm / 1000.0)) * m)
            UsdPhysics.CollisionAPI.Apply(cy.GetPrim())
        art.set_joint_positions(np.zeros(len(names), dtype=np.float32))
        art.set_joint_velocities(np.zeros(len(names), dtype=np.float32))
        for _ in range(2):
            world.step(render=False)
        tgt = np.zeros(len(names), dtype=np.float32)
        tgt[n_idx[0]], tgt[n_idx[1]] = FINGER_TRAVEL_M, -FINGER_TRAVEL_M   # command shut
        for _ in range(1200):
            art.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))
            world.step(render=False)
        q = np.asarray(art.get_joint_positions()).reshape(-1)
        fp = float(np.mean([abs(q[i]) for i in n_idx]))
        if diam_mm is not None:
            stage.RemovePrim(path)
        return 2.0 * (FINGER_TRAVEL_M - fp) * 1e3

    print("\n== contact (static cylinder between the blades, drive commanded fully shut)")
    print("  achieved_gap = 2 x (travel - finger_pos), i.e. the FACE separation the jaw model")
    print("  reports. A dead collider returns ~0. A FLAT face stalls at the diameter. An")
    print("  arched face stalls SHORT of it, by however deep the cylinder nests per side --")
    print("  which is the retention arch doing its job, and is the whole point of the tip.")
    free = probe(None)
    print(f"  CONTROL, empty jaw                ->  closed to  {free:6.2f} mm  "
          f"({'PASS' if abs(free) < 1.0 else 'FAIL -- the drive is not moving the jaws'})")
    cont_ok = abs(free) < 1.0
    for d in (18.4, 12.0):
        g = probe(d)
        # Stalling anywhere in a plausible band proves the collider is live; the residual
        # is the measurement, not the gate.
        good = 0.5 * d <= g <= d + 1.0
        cont_ok &= good
        print(f"  cylinder {d:5.1f} mm (M12 {'head ' if d > 15 else 'shaft'})  ->  stalled at "
              f"{g:6.2f} mm   nests {(d - g) / 2:5.2f} mm/side   {'PASS' if good else 'FAIL'}")
    if not cont_ok:
        print("  FAIL: the blade collider is not stopping the jaws. If approximation=sdf is")
        print("        authored but PhysX ignored it, rebuild with TIP_COLLISION=decomp.")

    # ---- what the policy sees ----------------------------------------------------------
    cam = Camera(prim_path=f"{tool}/wrist_cam", name="d405", resolution=(640, 480))
    cam.initialize()
    cam.set_local_pose(translation=np.array(LENS),
                       orientation=np.array([0.0, 0.0, 0.0, 1.0]), camera_axes="ros")
    cam.prim.GetAttribute("focalLength").Set(11.0)
    cam.prim.GetAttribute("horizontalAperture").Set(20.955)
    cam.prim.GetAttribute("verticalAperture").Set(20.955 * 480 / 640)
    cam.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.004, 100.0))

    # Place the closeup from the TOOL's actual world pose, not from a guessed world point.
    # The first version hardcoded a spot near the origin and photographed a forearm link --
    # the arm is at all-joints-zero, so the tool sits ~1.5 m up.
    tip_w = tool_m @ np.array([0.0, 0.0, 0.2476, 1.0])      # the TCP, in world
    eye = tip_w[:3] + np.array([0.0, -0.26, 0.06])
    close = rep.create.camera(position=tuple(float(v) for v in eye),
                              look_at=tuple(float(v) for v in tip_w[:3]),
                              clipping_range=(0.005, 100.0))
    print(f"  closeup camera at {np.round(eye, 3).tolist()} looking at TCP "
          f"{np.round(tip_w[:3], 3).tolist()}")
    close_rp = rep.create.render_product(close, (900, 700))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([close_rp])

    OUT.mkdir(exist_ok=True)
    print("\n== renders")
    for grip in (100.0, 0.0):
        pos = (1.0 - grip / 100.0) * FINGER_TRAVEL_M
        q = np.zeros(len(names), dtype=np.float32)
        q[n_idx[0]], q[n_idx[1]] = pos, -pos
        art.set_joint_positions(q)
        for _ in range(3):
            world.step(render=False)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        fr = np.asarray(ann.get_data())
        if fr.size:
            p = OUT / f"tip_{PIKA_TIP}_closeup_g{int(grip):03d}.png"
            imageio.imwrite(p, fr[..., :3].astype(np.uint8))
            print(f"  {p.name}")
        w = cam.get_rgba()
        if w is not None and np.asarray(w).size:
            p = OUT / f"tip_{PIKA_TIP}_wrist_g{int(grip):03d}.png"
            imageio.imwrite(p, np.asarray(w)[..., :3].astype(np.uint8))
            print(f"  {p.name}")
        else:
            print(f"  FAIL: wrist camera returned nothing at grip {grip:.0f}")

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
