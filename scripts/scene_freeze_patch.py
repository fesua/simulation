#!/usr/bin/env python3
"""Apply the scene-freezing patch to eval_closed_loop.py.

WHY THIS EXISTS (measured, 2026-08-20). PhysX settling is not reproducible run to run. The same
seed, same layout and same 240-step settle produced different start scenes in two arms of the
same protocol: seed 101 reported a black-bolt nearest-neighbour gap of 43 mm under pi05v1 and
48 mm under siglip. Every arm has therefore been scored on a slightly different world -- exactly
the uncontrolled variable a fixed protocol exists to remove, and it matters here because grasp
success collapses over a ~20 mm band of lateral error.

WHAT IT CHANGES
  --dump-scene-states FILE   settle as usual, then write the resulting bolt poses
  --scene-states FILE        set those exact poses instead of settling

Loading restores position, orientation and zero velocity for every bolt, so all arms start from
a bit-identical world. It does NOT make a whole episode deterministic -- gripper/bolt contact
during the episode is still PhysX -- it removes the initial-condition variance only.

This is kept as a separate applier rather than edited in place because the evaluation script
must not change while a scoring queue is running: arms scored under different code are not
comparable. Run it when no queue is active, then re-baseline.
"""
from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path(__file__).with_name("eval_closed_loop.py")

# (anchor, replacement) -- each anchor must appear exactly once.
PATCHES: list[tuple[str, str]] = [
    # 1) CLI
    (
        '    ap.add_argument("--label", default=None,\n'
        '                    help="model name recorded in the summary (defaults to --tag)")\n',
        '    ap.add_argument("--label", default=None,\n'
        '                    help="model name recorded in the summary (defaults to --tag)")\n'
        '    # PhysX settling is NOT reproducible run to run (measured: same seed, same layout,\n'
        '    # 43 mm vs 48 mm nearest-neighbour gap in two arms). Freeze the settled world once and\n'
        '    # load it, so every arm is scored on a bit-identical start state.\n'
        '    ap.add_argument("--scene-states", default=None,\n'
        '                    help="JSON of settled bolt poses to load instead of settling")\n'
        '    ap.add_argument("--dump-scene-states", default=None,\n'
        '                    help="settle as usual, then write the settled poses here and exit")\n',
    ),
    # 2) reset: load frozen poses, or settle then capture
    (
        "        for _ in range(240):\n"
        "            world.step(render=False)\n",
        "        frozen = None\n"
        "        if args.scene_states:\n"
        "            frozen = _SCENE_STATES.get(str(ep_seed))\n"
        "            if frozen is None:\n"
        "                raise SystemExit(\n"
        f'                    f"--scene-states has no entry for seed {{ep_seed}}; regenerate it "\n'
        '                    "with --dump-scene-states for this layout/n-per-color")\n'
        "        if frozen is not None:\n"
        "            # Exact settled world: position + orientation + zero velocity per bolt.\n"
        "            for path, st in zip(bolt_prims, frozen[\"bolts\"]):\n"
        "                v = bolt_views[path]\n"
        "                v.set_world_poses(np.array([st[\"p\"]], dtype=float),\n"
        "                                  np.array([st[\"q\"]], dtype=float))\n"
        "                v.set_velocities(np.zeros((1, 6)))\n"
        "            world.step(render=False)\n"
        "        else:\n"
        "            for _ in range(240):\n"
        "                world.step(render=False)\n",
    ),
    # 3) after crowding is computed, optionally dump and move on
    (
        "        # Warm the renderer BEFORE the first observation.",
        "        if args.dump_scene_states:\n"
        "            P, Q = bolt_all_poses()\n"
        "            _DUMP[str(ep_seed)] = dict(\n"
        "                layout=args.layout, colors=colors,\n"
        "                bolts=[dict(p=[float(x) for x in P[i]], q=[float(x) for x in Q[i]])\n"
        "                       for i in range(len(bolt_prims))])\n"
        "            print(f\"  [freeze] seed {ep_seed}: captured {len(bolt_prims)} bolt poses\")\n"
        "            continue\n"
        "        # Warm the renderer BEFORE the first observation.",
    ),
    # 4) full-pose reader next to bolt_xyz
    (
        '    def bolt_xyz():\n        """(n_bolts, 3) world positions, in bolt_prims order."""\n',
        "    def bolt_all_poses():\n"
        '        """(pos, quat) for every bolt, in bolt_prims order -- what freezing records."""\n'
        "        P = np.stack([np.asarray(bolt_views[p].get_world_poses()[0])[0] for p in bolt_prims])\n"
        "        Q = np.stack([np.asarray(bolt_views[p].get_world_poses()[1])[0] for p in bolt_prims])\n"
        "        return P, Q\n\n"
        '    def bolt_xyz():\n        """(n_bolts, 3) world positions, in bolt_prims order."""\n',
    ),
    # 5) load the file up front; write it at the end
    (
        "    RTC_ENABLED = bool(args.rtc)",
        "    global _SCENE_STATES, _DUMP\n"
        "    _SCENE_STATES, _DUMP = {}, {}\n"
        "    if args.scene_states:\n"
        "        _SCENE_STATES = json.loads(pathlib.Path(args.scene_states).read_text())\n"
        '        print(f"[eval] frozen scenes: {len(_SCENE_STATES)} seeds from {args.scene_states}")\n'
        "    RTC_ENABLED = bool(args.rtc)",
    ),
    (
        "    summary = dict(\n        protocol=args.protocol,",
        "    if args.dump_scene_states:\n"
        "        pathlib.Path(args.dump_scene_states).write_text(json.dumps(_DUMP, indent=1))\n"
        '        print(f"[freeze] wrote {len(_DUMP)} seeds -> {args.dump_scene_states}")\n'
        "        app.close()\n"
        "        return 0\n"
        "    summary = dict(\n        protocol=args.protocol,",
    ),
    # 6) record which frozen file a score came from
    (
        "        protocol=args.protocol, label=args.label or args.tag,",
        "        protocol=args.protocol, label=args.label or args.tag,\n"
        "        scene_states=args.scene_states,",
    ),
]


def main() -> int:
    src = TARGET.read_text()
    if "--dump-scene-states" in src:
        print("already patched")
        return 0
    for i, (anchor, repl) in enumerate(PATCHES, 1):
        n = src.count(anchor)
        if n != 1:
            print(f"PATCH {i} FAILED: anchor appears {n} times (need exactly 1)")
            print(f"  anchor starts: {anchor[:80]!r}")
            return 1
        src = src.replace(anchor, repl, 1)
    TARGET.write_text(src)
    print(f"patched {TARGET} ({len(PATCHES)} hunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
