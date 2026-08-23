"""Ladder step 5: closed-loop evaluation of the served pi0.5 policy in Isaac.

Deploy contract, all verified against the running server and robotics_lab (see CLAUDE.md):

  server   pi05_pika_umi_wrist_velgrip_k1_h24_80k on :8001
  obs      observation/{left,right}_wrist_0_rgb  (full 640x480 RGB; the SERVER does
           resize_with_pad to 224x224, so do not pre-resize here)
           observation/state = 14-D velocity_grip
               [pos_vel3, rot_vel3, grip] per arm, left block then right block
           prompt = the fixed task sentence
  actions  (24, 14) per-step ee_local deltas; gripper dims 6/13 in /100 units

Two choices that are easy to get wrong and are deliberate here:

  * velproprio_source = "command". The velocity fed to the policy comes from the
    history of EMITTED absolute TCP targets, NOT from measured motion. Hold ticks
    re-append the last target (ZOH) so a stationary command decays to zero velocity.
    Feeding measured motion instead would hand the policy a feedback channel the
    demonstrations never had.
  * chunk_anchor = "command". Chunk deltas integrate onto the last COMMANDED pose,
    not the measured one.

Control is the Ruckig chunk follower (not the CM controller), with the limits from
rb_servo_server/config/stack_real.yaml `ruckig_follower`.

Run:
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$HOME/workspace/openpi/packages/openpi-client/src \\
    .venv-isaac/bin/python scripts/eval_closed_loop.py --episodes 20 --layout aligned
"""

import argparse
import json
import math
import pathlib
import time

import os
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARM_USD = ROOT / "assets/rb3_730e_pika_articulated_sim/rb3_730e_pika_articulated_sim.usda"
STAND_USD = ROOT / "assets/dual_rb3_730e_stand_ver3/dual_rb3_730e_stand_ver3.usda"

PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, "
    "then pick up the gray bolt with the left arm and put it in the left box"
)

RESET = {
    "left": [259.0, 75.6, 129.5, -55.6, -131.2, -161.7],
    "right": [-253.7, -76.9, -127.6, 65.7, 143.7, 166.9],
}
ARM_JOINTS = [
    "base_joint",
    "shoulder_joint",
    "elbow_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]
MOUNT_FRAME = {"left": "stand_left_arm_base", "right": "stand_right_arm_base"}

# ---- scene (same numbers as build_scene.py) ---------------------------------
SHAFT_R, SHAFT_L = 0.006, 0.025
HEAD_R, HEAD_L = 0.0092, 0.012
TABLE = dict(cx=0.55, cy=0.0, hx=0.50, hy=0.55, thick=0.012)
PILE_X, PILE_DY = 0.47, 0.16
BOX_X, BOX_DY = 0.72, 0.215
BOX = dict(hw=0.120, hd=0.190, wall_h=0.0525, t=0.020, floor_t=0.0065, sponge_h=0.030)
ARM_OF_COLOR = {"gray": "left", "black": "right"}
# top-view left box is green (black bolts), right is gray (gray bolts)
BOX_CY = {"gray": +BOX_DY, "black": -BOX_DY}

# ---- deploy parameters (from the operator's flow_infer command) -------------
POLICY_DT = 0.0334          # --policy-dt-sec, 30 Hz
ACTION_HORIZON = 24
CHUNK_EXECUTE_STEPS = 4     # FLOW_INFER_CHUNK_EXECUTE_STEPS
SPEED_SCALE = 1.0
# FLOW_INFER_RTC=1 in the deploy command. NOTE: with RTC on, robotics_lab sets
# _chunk_crossfade_steps = 0 (openpi_remote.py:592) -- RTC replaces the crossfade,
# it is not used alongside it. RTC pins the new chunk's first `inference_delay`
# rows to the PREVIOUS plan's unexecuted tail, which is what keeps velocity
# continuous across the 133 ms chunk boundary. Without it every boundary is a
# fresh plan with no continuity constraint -- visible as residual tremor.
RESERVE_STEPS = 4   # stack_real.yaml reserve_steps; lookahead for the central
                    # difference that gives each knot a nonzero target VELOCITY
RTC_ENABLED = False   # isolate the follower fix first (operator: solve tremor before RTC)
RTC_INFERENCE_DELAY = 4     # = execute_steps - prefetch_at (4 - 0)
FINGER_TRAVEL_M = 0.047
# "actual" = measured jaw (the deploy default), "command" = the value just sent (this rig's
# historical behaviour). See the observation builder for why the difference is not cosmetic.
GRIP_PROPRIO = os.environ.get("GRIP_PROPRIO", "command").lower()
# Steps the grip channel is shifted against its paired pose row, per arm. 0/0 is what every run
# to date used. Positive = the jaw command is taken from further ahead in the chunk, i.e. it
# leads the arm, compensating a jaw slower than the arm.
GRIP_LEAD = {"left": int(os.environ.get("GRIP_LEAD_L", "0")),
             "right": int(os.environ.get("GRIP_LEAD_R", "0"))}
# Jaw transport delay, per arm, in ms. Hardware measures 105 (left) and 209 (right); this rig
# has always had ~0 because the position drive is stiff. Without it a lead value tuned in sim
# has the WRONG SIGN for the robot: sim's jaw leads the arm, hardware's trails it.
GRIP_LAG_MS = {"left": float(os.environ.get("GRIP_LAG_L", "0")),
               "right": float(os.environ.get("GRIP_LAG_R", "0"))}
# ee_local r_align: pika_rz180 = diag(-1,-1,+1), applied to BOTH linear and angular
R_ALIGN = np.diag([-1.0, -1.0, 1.0])

# ---- Ruckig follower limits (stack_real.yaml: safety.ruckig_follower) -------
LIN_V, LIN_A, LIN_J = 0.45, 12.0, 4000.0
ANG_V, ANG_A, ANG_J = 0.90, 40.0, 8000.0

PHYSICS_DT = 1.0 / 500.0    # real servo rate
# rb3_730e.urdf gives every arm joint velocity=3.14159 rad/s. Clipping the IK step
# to that budget per substep is what stops the command outrunning the drives:
# the previous +-0.05 rad per 2 ms was 25 rad/s, ~8x the limit, so the joints
# saturated, tracking error grew, and the chunk-boundary re-anchor then jumped --
# which shows up as visible tremor (real hardware does not shake).
# Physical joint travel of the sim robot (rb3_730e.urdf). PhysX enforces these, so a
# command that runs past them cannot be reached: the joint pins at the stop while the
# IK keeps integrating, and the command-anchored chunk drifts away without bound.
# Measured before this clamp existed: left j1 spent 74% of an episode past +6.2832
# (up to 6.841) and the command ended 635 mm from the robot.
# rb_servo_server does the same thing in SafetyFilter::clampJointLimits (safety_filter.cpp:251):
#   out[i] = std::clamp(out[i], q_min_deg[i], q_max_deg[i])
JOINT_LO = np.array([-6.2832, -6.2832, -2.618, -6.2832, -6.2832, -6.2832])
JOINT_HI = -JOINT_LO
# stack_real.yaml dq_max_deg_s [170,170,170,240,240,320] -> rad/s, per joint
JOINT_VEL_LIMIT = np.deg2rad([170.0, 170.0, 170.0, 240.0, 240.0, 320.0])
DQ_MAX = JOINT_VEL_LIMIT * PHYSICS_DT
# DLS damping. 1e-4 is the historical value every published number used. Near the boxes the
# Jacobian's smallest singular value drops to ~2e-4 and the arm visibly shakes and sheds bolts;
# a heavier damping trades tracking for calm there, so it is a knob, not a new default.
IK_LAMBDA = float(os.environ.get("IK_LAMBDA", "1e-4"))
# Single-arm oracle. The oracle's job has narrowed to characterising PICK physics; bimanual
# choreography (arm-arm interference, races at the pile midline) is task realism the POLICY
# must face but pure noise for a physics instrument. ORACLE_ARM=left parks the other arm at
# all-joints-zero -- verified by offline FK to point straight up (lowest link z=0.73 m, TCP at
# z=1.71 m), fully clear of the table and the working arm -- and the active arm services BOTH
# colours, placing each bolt into the box of its own colour.
ORACLE_ARM = os.environ.get("ORACLE_ARM", "").lower() or None
ORACLE_PARKED = ({"left": "right", "right": "left"}[ORACLE_ARM] if ORACLE_ARM else None)
DIAG = os.environ.get("TREMOR_DIAG") == "1"
# EVAL_DUMP_OBS=<dir> writes the exact wrist frames handed to the policy, every 30th tick.
# The observation path is the one thing an aggregate score cannot audit -- black or stale
# frames score 0 while looking like a policy result (see CLAUDE.md section 20).
DUMP_OBS = os.environ.get("EVAL_DUMP_OBS") or ""
if DUMP_OBS:
    os.makedirs(DUMP_OBS, exist_ok=True)
# EVAL_PROBE_DUMP=<dir> writes a PERCEPTION-PROBE dataset: every PROBE_EVERY-th policy tick,
# the exact wrist frames plus ground truth no real log has -- TCP pose, the nearest bolts in the
# TOOL frame, and the policy's own chunk-endpoint aim. Purpose: decompose the closed-loop aim
# error into "the encoder cannot locate the bolt from this (drifted) viewpoint" versus "the
# features suffice but the action side loses precision" by fitting a probe from frozen encoder
# features to the tool-frame bolt position and comparing probe error with aim error per stratum.
PROBE_DUMP = os.environ.get("EVAL_PROBE_DUMP") or ""
PROBE_EVERY = int(os.environ.get("EVAL_PROBE_EVERY", "10"))
if PROBE_DUMP:
    os.makedirs(PROBE_DUMP, exist_ok=True)
# Replay drives the recorded joint-command stream verbatim, so the bolts-vs-no-bolts
# comparison differs ONLY by contact. The free-space run without this was useless:
# with no bolts the policy commanded unreachable poses (141-350 mm off), so the two
# conditions were not running the same trajectory at all.
REPLAY = os.environ.get("TREMOR_REPLAY")
_REP = np.load(REPLAY) if REPLAY else None
if _REP is not None:
    # Fail loudly at startup: a diag npz recorded before joint logging existed has no
    # *_q, and the KeyError deep in the loop left Isaac hung until the 1200 s timeout.
    _missing = [k for k in ("left_q", "left_gq", "right_q", "right_gq") if k not in _REP]
    if _missing:
        raise SystemExit(f"TREMOR_REPLAY={REPLAY} lacks {_missing}; re-record with TREMOR_DIAG=1")
IK_ITERS = 3        # DLS iterations per substep; one was not converging
SUBSTEPS = int(round(POLICY_DT / PHYSICS_DT))   # 17


def rotvec_to_mat(r):
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-6:
        # near-pi: use the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = V[:, int(np.argmax(w))]
        return axis * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2.0 * math.sin(th)))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])



# ---- analytic arm kinematics ------------------------------------------------
# Taken straight from rb3_730e.urdf. Using the URDF chain rather than PhysX
# Jacobians because SingleArticulation exposes none (only the batched Articulation
# class does), and an analytic Jacobian is both exact and far cheaper at 500 Hz.
# Each entry is (origin_xyz, origin_rpy, axis); the joint rotation applies AFTER
# the origin transform. Verified at runtime against the physics TCP pose.
URDF_CHAIN = [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.5708), None),            # link0_fixed
    ((0.0, 0.0, 0.1453), (0.0, 0.0, -1.5708), (0, 0, 1)),   # base_joint
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 1, 0)),          # shoulder_joint
    ((0.0, -0.00645, 0.286), (0.0, 0.0, 0.0), (0, 1, 0)),   # elbow_joint
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 1)),          # wrist1_joint
    ((0.0, 0.0, 0.344), (0.0, 0.0, 0.0), (0, 1, 0)),        # wrist2_joint
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 1)),          # wrist3_joint
]
TOOL_Z = 0.1 + 0.247642   # attachment_site + tcp


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_rot(axis, q):
    return rotvec_to_mat(np.array(axis, dtype=float) * q)


def fk_chain(q, T_mount):
    """Return (tcp_pos, tcp_R, joint_origins, joint_axes) in world."""
    T = np.array(T_mount, dtype=float)
    origins, axes = [], []
    qi = 0
    for xyz, rpy, ax in URDF_CHAIN:
        M = np.eye(4)
        M[:3, :3] = _rpy(*rpy)
        M[:3, 3] = xyz
        T = T @ M
        if ax is not None:
            origins.append(T[:3, 3].copy())
            axes.append((T[:3, :3] @ np.array(ax, dtype=float)).copy())
            R = np.eye(4)
            R[:3, :3] = _axis_rot(ax, q[qi])
            T = T @ R
            qi += 1
    Tt = np.eye(4)
    Tt[2, 3] = TOOL_Z
    T = T @ Tt
    return T[:3, 3].copy(), T[:3, :3].copy(), origins, axes


def jacobian(q, T_mount):
    p_e, _, origins, axes = fk_chain(q, T_mount)
    J = np.zeros((6, 6))
    for i, (o, a) in enumerate(zip(origins, axes)):
        J[:3, i] = np.cross(a, p_e - o)
        J[3:, i] = a
    return J


class RuckigArmFollower:
    """Jerk-limited 6-DOF Cartesian follower, one per arm.

    Mirrors rb_servo_server's cartesian_chunk_follower: a Ruckig p/v/a chain drives the
    Cartesian reference toward the active knot, re-targeted at each 33 ms boundary.
    Orientation is carried as a rotvec so the per-axis angular limits apply the same way
    the C++ AxisLimit does.
    """

    def __init__(self, pose0):
        from ruckig import InputParameter, OutputParameter, Ruckig

        self.otg = Ruckig(6, PHYSICS_DT)
        self.inp = InputParameter(6)
        self.out = OutputParameter(6)
        self.inp.max_velocity = [LIN_V] * 3 + [ANG_V] * 3
        self.inp.max_acceleration = [LIN_A] * 3 + [ANG_A] * 3
        self.inp.max_jerk = [LIN_J] * 3 + [ANG_J] * 3
        self.inp.current_position = list(pose0)
        self.inp.current_velocity = [0.0] * 6
        self.inp.current_acceleration = [0.0] * 6
        self.inp.target_position = list(pose0)
        self.inp.target_velocity = [0.0] * 6
        self.reference = np.array(pose0, dtype=float)

    def set_target(self, pose, vel=None):
        """Target a knot WITH a velocity.

        chunk_window.hpp keeps `reserve_R` steps of lookahead precisely so each knot can
        be given a central-difference target velocity. Leaving target_velocity at zero
        makes Ruckig decelerate to a full stop at every 33 ms knot: the arm accelerates
        and brakes 30 times a second, which is exactly the tremor the real robot does not
        have (it flows through knots at speed).
        """
        self.inp.target_position = [float(v) for v in pose]
        if vel is None or os.environ.get("TREMOR_BASELINE") == "1":
            self.inp.target_velocity = [0.0] * 6   # old behaviour: stop at every knot
        else:
            self.inp.target_velocity = [float(v) for v in vel]

    def step(self):
        from ruckig import Result

        # Ruckig raises on a strictly degenerate request: target == current to the last bit,
        # with zero velocity, gives "error in step 2 ... for t sync: 0.000000". A policy never
        # produces that (its knots always move a little), but a scripted planner that holds
        # still does, and it killed the first oracle run. Holding the reference is the correct
        # answer for a zero-length trajectory anyway.
        # 1 nm / 1 nrad is far below anything the arm can express, so treating it as "already
        # there" cannot change behaviour -- but it does cover the whole family of static-hold
        # requests a scripted planner emits, which is what crashed the first two oracle runs.
        if (np.allclose(self.inp.target_position, self.inp.current_position, atol=1e-9)
                and np.allclose(self.inp.current_velocity, 0.0, atol=1e-7)
                and np.allclose(self.inp.target_velocity, 0.0, atol=1e-7)):
            self.reference = np.array(self.inp.current_position, dtype=float)
            return True

        res = self.otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)
        self.reference = np.array(self.out.new_position, dtype=float)
        return res in (Result.Working, Result.Finished)


def bolt_poses(layout, n_per, seed):
    rng = np.random.default_rng(seed)
    out = []
    spread = float(os.environ.get("BOLT_SPREAD", "1.0"))
    if layout == "aligned":
        for color, sy in (("gray", +1.0), ("black", -1.0)):
            for _ in range(n_per):
                # BOLT_SPREAD scales the pile sigma. At 1.0 the piles match the real cell's
                # density (same-colour nearest-neighbour gap ~38 mm, well inside the 110 mm
                # jaw sweep) -- correct for POLICY scoring, but for grasp-physics calibration
                # it entangles contact parameters with neighbour interference. Sparse layouts
                # are an instrument setting only; std20/std40 stay at 1.0.
                out.append((color, PILE_X + rng.normal(0, 0.045 * spread),
                            sy * PILE_DY + rng.normal(0, 0.05 * spread), rng.uniform(0, math.pi)))
    else:
        colors = ["gray"] * n_per + ["black"] * n_per
        rng.shuffle(colors)
        for color in colors:
            out.append((color, PILE_X + rng.normal(0, 0.07),
                        rng.uniform(-0.30, 0.30), rng.uniform(0, math.pi)))
    return out


def main() -> int:
    global RTC_ENABLED, RTC_INFERENCE_DELAY, CHUNK_EXECUTE_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--layout", choices=["aligned", "random"], default="aligned")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n-per-color", type=int, default=10,
                    help="bolts per colour; the real cell has ~20 bolts piled up")
    ap.add_argument("--episode-sec", type=float, default=40.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--video", action="store_true", default=True)
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--tag", default="ep")
    # Deploy-runner timing/RTC knobs (flow_infer_sweep_run.sh equivalents). The runner
    # KICKS the next inference at chunk step `prefetch_at` and SWAPS at the execute
    # boundary, dropping the rows that elapsed since the observation (realized_delay =
    # execute - prefetch_at). Emulating that here is what makes the sim comparable to
    # the robot: a synchronous infer() at the boundary would give every executed row a
    # freshness the real robot never has.
    ap.add_argument("--rtc", action="store_true", default=False,
                    help="FLOW_INFER_RTC=1: freeze the first (execute-prefetch_at) rows to the "
                         "previous plan and inpaint the rest (server-side, zeros schedule)")
    ap.add_argument("--execute-steps", type=int, default=CHUNK_EXECUTE_STEPS)
    ap.add_argument("--prefetch-at", type=int, default=0)
    # THE fixed scoring set. Every model comparison is run under --protocol std20 so that two
    # arms can never differ by an evaluation knob; only the policy endpoint and the layout axis
    # may change. Video is ON even though it roughly doubles wall clock: recording changes the
    # number of rep.orchestrator.step() calls per tick, and an orchestrator step advances the
    # timeline (section 19's mechanism). Measured 2026-08-19 on seed 104, same code: video ON
    # reproduced the historical runs (2 placements, close |dxy| p50 10.6 mm) while video OFF did
    # not (0 placements, 260 mm) even with verified-correct wrist observations. Every validated
    # number this project has came from a video-ON run; the protocol keeps it that way.
    ap.add_argument("--protocol", choices=["std20", "std40", "free"], default="free",
                    help="std20 = fixed eval set: 20 episodes, seeds 100..119, 30 s each, "
                         "10 bolts per colour, video on. Overrides the knobs it pins.")
    ap.add_argument("--label", default=None,
                    help="model name recorded in the summary (defaults to --tag)")
    # PhysX settling is NOT reproducible run to run (measured: same seed, same layout,
    # 43 mm vs 48 mm nearest-neighbour gap in two arms). Freeze the settled world once and
    # load it, so every arm is scored on a bit-identical start state.
    ap.add_argument("--scene-states", default=None,
                    help="JSON of settled bolt poses to load instead of settling")
    ap.add_argument("--oracle", action="store_true", default=False,
                    help="replace the policy with a privileged-state scripted planner "
                         "(same scene, same controller, same scoring) to measure the CEILING")
    ap.add_argument("--dump-scene-states", default=None,
                    help="settle as usual, then write the settled poses here and exit")
    args = ap.parse_args()
    if args.protocol in ("std20", "std40"):
        n = 20 if args.protocol == "std20" else 40
        args.episodes, args.seed, args.episode_sec = n, 100, 30.0
        args.n_per_color, args.video = 10, True
        # std40 = seeds 100..139, a strict SUPERSET of std20's 100..119. A std40 run therefore
        # still contains the std20 answer, so widening the protocol does not orphan earlier
        # scores -- the first 20 seeds remain directly comparable.
        print(f"[eval] protocol={args.protocol}: {n} eps, seeds 100-{99+n}, 30 s, "
              "10 bolts/colour, video on")
    global _SCENE_STATES, _DUMP
    _SCENE_STATES, _DUMP = {}, {}
    if args.scene_states:
        _SCENE_STATES = json.loads(pathlib.Path(args.scene_states).read_text())
        print(f"[eval] frozen scenes: {len(_SCENE_STATES)} seeds from {args.scene_states}")
    RTC_ENABLED = bool(args.rtc)
    CHUNK_EXECUTE_STEPS = int(args.execute_steps)
    PREFETCH_AT = int(np.clip(args.prefetch_at, 0, CHUNK_EXECUTE_STEPS - 1))
    RTC_INFERENCE_DELAY = CHUNK_EXECUTE_STEPS - PREFETCH_AT
    print(f"[eval] rtc={RTC_ENABLED} execute={CHUNK_EXECUTE_STEPS} prefetch_at={PREFETCH_AT} "
          f"realized_delay={RTC_INFERENCE_DELAY}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import imageio.v2 as imageio
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from openpi_client import websocket_client_policy
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    outdir = ROOT / "outputs/eval"
    outdir.mkdir(parents=True, exist_ok=True)

    # rendering_dt MUST be given. It defaults to 1/60 s, so with physics_dt=1/500 every
    # render step advanced the timeline by 8.33 physics steps: recording a frame once per
    # policy tick injected a ~6.6 mm TCP jump (measured max 6.58 mm = 3.3 m/s, impossible
    # for a 0.45 m/s command) at exactly 30 Hz. Replaying the identical joint stream with
    # --no-video showed none of it (max step 1.80 mm).
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
                  rendering_dt=PHYSICS_DT)
    stage = get_current_stage()

    def mat(path, rgb, rough=0.5, metallic=0.0):
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return m

    MAT = {
        "table": mat("/World/m/table", (0.42, 0.44, 0.45), 0.42, 0.65),
        "green": mat("/World/m/green", (0.10, 0.38, 0.27), 0.45),
        "boxgray": mat("/World/m/boxgray", (0.46, 0.47, 0.49), 0.50, 0.35),
        "insert": mat("/World/m/insert", (0.26, 0.26, 0.28), 0.85),
        "bolt_gray": mat("/World/m/bgray", (0.52, 0.53, 0.56), 0.32, 1.0),
        "bolt_black": mat("/World/m/bblack", (0.07, 0.07, 0.08), 0.42, 0.9),
    }

    def sbox(path, c, h, key):
        cu = UsdGeom.Cube.Define(stage, path)
        cu.CreateSizeAttr(2.0)
        UsdGeom.Xformable(cu).AddTransformOp().Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(*h)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*c))
        )
        UsdPhysics.CollisionAPI.Apply(cu.GetPrim())
        UsdShade.MaterialBindingAPI(cu.GetPrim()).Bind(MAT[key])

    add_reference_to_stage(usd_path=str(STAND_USD), prim_path="/World/cell/stand")
    mount_xf = {}
    for prim in stage.Traverse():
        for side, frame in MOUNT_FRAME.items():
            if prim.GetName() == frame and side not in mount_xf:
                mount_xf[side] = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
    for side, m in mount_xf.items():
        p = f"/World/cell/{side}_arm"
        UsdGeom.Xform.Define(stage, p).MakeMatrixXform().Set(m)
        add_reference_to_stage(usd_path=str(ARM_USD), prim_path=f"{p}/robot")

    # FINGER EFFORT LIMIT. The real jaws close until they touch -- there is no positional stop at
    # the object, so what makes a real grasp work is that the motor STALLS on the bolt at a finite
    # force (operator, 2026-08-23). The sim drive has no such limit and keeps pushing toward the
    # commanded 0 mm gap, which squeezes the bolt out. SingleArticulation has no set_max_efforts
    # in Isaac 6.0, so set the drive attribute on the joint prims directly, at build time, before
    # PhysX parses them -- writing it later would be overwritten by the articulation.
    _maxf = os.environ.get("GRIP_MAXF")
    if _maxf:
        _n = 0
        for prim in stage.Traverse():
            if prim.GetName() in ("finger_left_joint", "finger_right_joint"):
                for tok in ("linear", "transX", "transY", "transZ"):
                    dr = UsdPhysics.DriveAPI.Get(prim, tok)
                    if dr:
                        dr.CreateMaxForceAttr().Set(float(_maxf))
                        _n += 1
                        break
                else:
                    dr = UsdPhysics.DriveAPI.Apply(prim, "linear")
                    dr.CreateMaxForceAttr().Set(float(_maxf))
                    _n += 1
        print(f"  [grip] finger drive maxForce = {_maxf} N on {_n} joint(s)")
        if _n == 0:
            raise SystemExit("ABORT: GRIP_MAXF set but no finger joints found -- the limit would "
                             "have been silently ignored and the run would look like a physics "
                             "result.")

    sbox("/World/scene/table", (TABLE["cx"], TABLE["cy"], -TABLE["thick"]),
         (TABLE["hx"], TABLE["hy"], TABLE["thick"]), "table")
    b = BOX
    for name, color, wall in (("box_gray", "gray", "boxgray"), ("box_green", "black", "green")):
        cy = BOX_CY[color]
        r = f"/World/scene/{name}"
        UsdGeom.Xform.Define(stage, r)
        sbox(f"{r}/floor", (BOX_X, cy, b["floor_t"] / 2),
             (b["hw"], b["hd"], b["floor_t"] / 2), wall)
        sbox(f"{r}/sponge", (BOX_X, cy, b["floor_t"] + b["sponge_h"] / 2),
             (b["hw"] - b["t"], b["hd"] - b["t"], b["sponge_h"] / 2), "insert")
        h = b["wall_h"]
        for tag, off, half in (("xm", (-b["hw"], 0), (b["t"], b["hd"], h)),
                               ("xp", (+b["hw"], 0), (b["t"], b["hd"], h)),
                               ("ym", (0, -b["hd"]), (b["hw"], b["t"], h)),
                               ("yp", (0, +b["hd"]), (b["hw"], b["t"], h))):
            sbox(f"{r}/w_{tag}", (BOX_X + off[0], cy + off[1], h), half, wall)

    # Optional explicit contact material for the bolts (see the binding below).
    _PHYSMAT = None
    _mu = os.environ.get("BOLT_FRICTION")
    if _mu:
        _mp = "/World/physmat_bolt"
        _m = UsdShade.Material.Define(stage, _mp)
        _api = UsdPhysics.MaterialAPI.Apply(_m.GetPrim())
        _api.CreateStaticFrictionAttr().Set(float(_mu))
        _api.CreateDynamicFrictionAttr().Set(float(_mu))
        _api.CreateRestitutionAttr().Set(float(os.environ.get("BOLT_RESTITUTION", "0.0")))
        _PHYSMAT = _m.GetPrim()
        print(f"  [phys] bolt friction {_mu}, restitution "
              f"{os.environ.get('BOLT_RESTITUTION', '0.0')}")

    n_bolts = args.n_per_color * 2
    bolt_prims = []
    for i in range(n_bolts):
        path = f"/World/scene/bolt_{i:02d}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.4, 0, 0.05 + 0.05 * i)))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.022)
        for tag, rr, hl, cx in (("shaft", SHAFT_R, SHAFT_L / 2, SHAFT_L / 2),
                                ("head", HEAD_R, HEAD_L / 2, -HEAD_L / 2)):
            cy_ = UsdGeom.Cylinder.Define(stage, f"{path}/{tag}")
            cy_.CreateRadiusAttr(rr)
            cy_.CreateHeightAttr(hl * 2)
            cy_.CreateAxisAttr("X")
            UsdGeom.Xformable(cy_).AddTranslateOp().Set(Gf.Vec3d(cx, 0, 0))
            UsdPhysics.CollisionAPI.Apply(cy_.GetPrim())
            # The bolts have never had a physics material -- friction and restitution have been
            # whatever PhysX defaults to (0.5/0.5/0). On a task that is entirely about pinching a
            # smooth cylinder, that is an unexamined parameter, so make it explicit and tunable.
            if _PHYSMAT is not None:
                UsdShade.MaterialBindingAPI.Apply(cy_.GetPrim())
                UsdShade.MaterialBindingAPI(cy_.GetPrim()).Bind(
                    UsdShade.Material(_PHYSMAT), bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                    materialPurpose="physics")
        bolt_prims.append(path)

    # FINGERTIP / TABLE FRICTION. (Placed AFTER scene build: the first version ran before
    # sbox() created the table, matched zero colliders, and the loud guard aborted the run
    # -- which is the guard doing its job; a silent version would have produced two fake
    # 'friction has no effect' results.) Operator observation from the montage: when the tips are
    # resting on the table the jaws stop closing, while on hardware they scrape along the
    # surface and still shut. The sim tips and the table have no physics material at all, so
    # they inherit PhysX's default 0.5 and the position-controlled arm presses hard enough for
    # that to lock the prismatic fingers. A bolt cannot be picked if touching down forbids
    # closing, so make the tip/table pair explicitly tunable.
    _fmu = os.environ.get("FINGER_FRICTION")
    if _fmu:
        _fm = UsdShade.Material.Define(stage, "/World/physmat_finger")
        _fa = UsdPhysics.MaterialAPI.Apply(_fm.GetPrim())
        _fa.CreateStaticFrictionAttr().Set(float(_fmu))
        _fa.CreateDynamicFrictionAttr().Set(float(_fmu))
        _fa.CreateRestitutionAttr().Set(0.0)
        _n = 0
        for prim in stage.Traverse():
            nm = prim.GetName()
            if ("finger" in nm or nm == "table") and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)
                UsdShade.MaterialBindingAPI(prim).Bind(
                    _fm, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                    materialPurpose="physics")
                _n += 1
        print(f"  [phys] finger/table friction {_fmu} on {_n} collider(s)")
        if _n == 0:
            raise SystemExit("ABORT: FINGER_FRICTION set but no finger/table colliders matched "
                             "-- the run would look like a physics result while changing nothing.")


    key = UsdLux.DistantLight.Define(stage, "/World/key")
    key.CreateIntensityAttr(600.0)
    key.CreateAngleAttr(12.0)
    UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(520.0)

    arts = {}
    for side in MOUNT_FRAME:
        a = SingleArticulation(prim_path=f"/World/cell/{side}_arm/robot", name=f"{side}_arm")
        world.scene.add(a)
        arts[side] = a
    world.reset()
    for a in arts.values():
        a.initialize()

    bolt_views = {p: RigidPrim(prim_paths_expr=p, name=f"bv{i}")
                  for i, p in enumerate(bolt_prims)}
    for v in bolt_views.values():
        v.initialize()

    # ---- T1 instrumentation: read EVERY bolt pose once per policy tick -------------------
    # The multimodality probe needs all bolt positions at 30 Hz (to see which candidate the
    # chunk endpoint is aimed at). Per-bolt reads would be 20 x 900 round trips per episode,
    # so use one batched view and remap it into bolt_prims order. Falls back to per-bolt
    # reads rather than failing: a slower eval is better than no eval.
    bolts_all, _bolt_order = None, None
    try:
        _bv = RigidPrim(prim_paths_expr="/World/scene/bolt_.*", name="bolts_all")
        _bv.initialize()
        _paths = getattr(_bv, "prim_paths", None) or getattr(_bv, "paths", None)
        _bolt_order = [bolt_prims.index(str(p)) for p in list(_paths)]
        if sorted(_bolt_order) != list(range(len(bolt_prims))):
            raise RuntimeError(f"batched view covers {len(_bolt_order)}/{len(bolt_prims)} bolts")
        bolts_all = _bv
    except Exception as exc:
        print(f"  [warn] batched bolt view unavailable ({exc}); falling back to per-bolt reads")

    def bolt_all_poses():
        """(pos, quat) for every bolt, in bolt_prims order -- what freezing records."""
        P = np.stack([np.asarray(bolt_views[p].get_world_poses()[0])[0] for p in bolt_prims])
        Q = np.stack([np.asarray(bolt_views[p].get_world_poses()[1])[0] for p in bolt_prims])
        return P, Q

    def bolt_axis_xy(i):
        """In-plane direction of bolt i's shaft, as a world-XY angle in degrees.

        The demonstrations grasp bolts across the shaft, so the jaw line (tool X -- the fingers
        are prismatic +-X) has to end up perpendicular to this. Measured 2026-08-22: EVERY
        successful grasp sits above 60 deg of separation and no close below it has ever
        succeeded, which makes this an independent necessary condition alongside lateral aim.
        Recorded at the close so it no longer has to be recovered by matching positions back to
        the frozen scene -- that matching silently drops every bolt the arms have nudged.
        """
        q = np.asarray(bolt_views[bolt_prims[i]].get_world_poses()[1])[0]
        w, x, y, z = (float(v) for v in q)
        # bolt shaft is the prim's local X (build_scene.py sets CreateAxisAttr("X"))
        ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
        return float(np.degrees(np.arctan2(ax[1], ax[0]))), float(ax[2])

    def bolt_xyz():
        """(n_bolts, 3) world positions, in bolt_prims order."""
        if bolts_all is not None:
            P = np.asarray(bolts_all.get_world_poses()[0])
            out = np.empty_like(P)
            out[_bolt_order] = P
            return out
        return np.stack([np.asarray(bolt_views[p].get_world_poses()[0])[0] for p in bolt_prims])

    tcp_views, wrist_cams = {}, {}
    LENS = (0.00917, 0.04601, 0.11930)
    for side in MOUNT_FRAME:
        root = f"/World/cell/{side}_arm/robot"
        tcp_path = next(str(p.GetPath()) for p in stage.Traverse()
                        if str(p.GetPath()).startswith(root) and p.GetName() == "tcp"
                        and p.HasAPI(UsdPhysics.RigidBodyAPI))
        v = RigidPrim(prim_paths_expr=tcp_path, name=f"tcp_{side}")
        v.initialize()
        tcp_views[side] = v
        tool_path = next(str(p.GetPath()) for p in stage.Traverse()
                         if str(p.GetPath()).startswith(root) and p.GetName() == "tool"
                         and p.HasAPI(UsdPhysics.RigidBodyAPI))
        cam = Camera(prim_path=f"{tool_path}/wrist_cam", name=f"d405_{side}",
                     resolution=(640, 480))
        cam.initialize()
        cam.set_local_pose(translation=np.array(LENS),
                           orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 180 deg roll
                           camera_axes="ros")
        cam.prim.GetAttribute("focalLength").Set(11.0)
        cam.prim.GetAttribute("horizontalAperture").Set(20.955)
        cam.prim.GetAttribute("verticalAperture").Set(20.955 * 480 / 640)
        cam.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.004, 100.0))
        wrist_cams[side] = cam

    # The overview render product is the video source only. It is 960x720 and renders on every
    # orchestrator step whether or not a frame is kept, so it is created ONLY when recording.
    ov_ann = top_ann = None
    if args.video:
        ov = rep.create.camera(position=(1.85, -1.35, 1.15), look_at=(0.55, 0.0, 0.15),
                               clipping_range=(0.01, 100.0))
        ov_rp = rep.create.render_product(ov, (960, 720))
        ov_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ov_ann.attach([ov_rp])
        if os.environ.get("EVAL_TOPVIEW"):
            # A near-top view of the bolt piles, for judging grasps by eye. Not straight down:
            # a perfectly vertical camera is degenerate here (view axis parallel to world up,
            # sec.12), so it sits back on +x and looks down at the piles -- robot at the top of
            # frame, +y to the right, matching the montage convention.
            tv = rep.create.camera(position=(1.00, 0.0, 0.95), look_at=(0.58, 0.0, 0.0),
                                   clipping_range=(0.01, 100.0))
            top_rp = rep.create.render_product(tv, (960, 720))
            top_ann = rep.AnnotatorRegistry.get_annotator("rgb")
            top_ann.attach([top_rp])

    class Oracle:
        """Scripted pick-and-place from privileged state, emitting the SAME chunk format.

        Why this exists. Every improvement arm so far sits below the control, and we have no
        upper bound: 42 placements might be 90% of what is physically achievable in 30 s or it
        might be 20%. Worse, sim converts only 11% of close commands into grasps where the
        hardware record converts ~47%, so the "precision is the bottleneck" diagnosis could be
        measuring sim contact physics rather than the model. An oracle with exact bolt poses
        separates the two: if perfect aim still rarely grasps, the rig's physics is the term to
        fix before any more training arms.

        It replaces ONLY the policy. Scene, controller (Ruckig follower + DLS IK), scoring and
        the T1 instrument are untouched, so the difference is attributable to the policy alone.

        Attitude is taken from what actually works rather than designed: successful grasps
        approach 8-9 deg off straight down with the jaw axis horizontal, and every one of the 36
        recorded grasps has the jaw within 30 deg of perpendicular to the bolt. So the template
        is tool z straight down, tool x perpendicular to the target bolt's shaft.
        """

        # heights (m) above the bolt, and the grip percentages the follower sees
        H_APPROACH = 0.080
        Z_GRASP_OFF = float(os.environ.get("ORACLE_ZOFF", "0.0026"))
        # Transit/carry height. The box walls top out at z=105 mm. The first oracle traversed
        # at TCP 89 mm (16 mm BELOW the wall) and carried at 150 mm, where a grasped bolt's
        # lowest point hangs at ~113 mm -- 8 mm of wall clearance, gone at the first shake.
        # Video review + the lifted/dropped counters (8 lifted, 4 dropped in one run) showed
        # both: wall strikes on the way to pick, and bolts clipped off on the way in.
        H_LIFT = 0.200
        TRANSIT_XY = 0.040       # descend only when this close, laterally, to the target
        # Place at the NEAR interior of the box (still inside: interior x 0.62..0.82), not the
        # centre. 60 mm less extension where the Jacobian's smallest singular value collapses
        # to 2e-4 (1/90th of its overall p10) -- the measured shake-and-drop zone.
        PLACE_X = 0.660
        # Partial opening (operator): at 50% the half-gap is 23.5 mm against an 18.4 mm
        # head -- still 2.5 mm of aim slack per side for an oracle that lands at 0.1 mm, and
        # the narrower sweep stops the descent snagging neighbouring bolts.
        OPEN = float(os.environ.get("GRIP_OPEN", "100"))
        SHUT = 0.0
        STEP_XY, STEP_Z = 0.010, 0.006          # per policy tick; well inside LIN_V*dt = 15 mm
        GAIN = 0.15                             # fraction of the remaining error per row
        # The jaws must not slam. The policy closes at ~0.8%/tick (measured over 80 transitions
        # in its own dumps) and the hardware record puts jaw lag at 105-209 ms; the first oracle
        # stepped 100 -> 0 in ONE tick, which launched properly-arrived bolts up to 237 mm. That
        # is a planner artifact, not a verdict on sim contact physics, so the rate is a knob and
        # the close phase waits for the ramp to finish.
        GRIP_RATE = float(os.environ.get("GRIP_RATE", "100"))     # percent per policy tick
        RELEASE_TICKS = 6

        @property
        def CLOSE_TICKS(self):
            return int(max(12, 100.0 / self.GRIP_RATE + 8))

        # A phase must not be able to stall forever. The first version used 4 mm XY / 10 mm Z
        # transition gates, which the offline replay clears instantly with perfect tracking but
        # the real follower never does -- so all three smoke episodes sat in "seek" and issued
        # ZERO gripper commands. Gates are now loose and every phase is force-advanced after
        # PHASE_TIMEOUT, because a ceiling measurement has to actually attempt grasps.
        PHASE_TIMEOUT = 105          # ticks (3.5 s)
        DEBUG = bool(int(os.environ.get("ORACLE_DEBUG", "0")))

        def __init__(self):
            self.phase = {"left": "seek", "right": "seek"}
            self.target = {"left": None, "right": None}
            self.timer = {"left": 0, "right": 0}
            self.age = {"left": 0, "right": 0}
            self.place_col = {"left": "gray", "right": "black"}
            self.claimed = set()
            self.clock = 0
            self.cooldown = {}          # bolt idx -> clock tick until which it is skipped

        def _go(self, side, ph, timer=0):
            if self.DEBUG:
                print(f"      [oracle] {side} {self.phase[side]} -> {ph} "
                      f"(after {self.age[side]} ticks)")
            self.phase[side], self.timer[side], self.age[side] = ph, timer, 0

        # Approach attitude, taken from what the policy's SUCCESSFUL grasps actually do rather
        # than from what looks ideal. Exactly-straight-down was tried first and is wrong twice
        # over: it is a 180 deg rotation, i.e. sitting on the rotvec representation singularity,
        # and it is at the edge of the reachable set -- the traced run converged to a 22.8 mm
        # lateral residual and then froze, because the IK was pushing into a joint limit that
        # the command clamp holds. Successful grasps sit 8-9 deg off vertical (failures 11.6 and
        # 22.5), so the oracle uses that, per arm, in the direction the measurement shows.
        TILT = {"left": np.array([0.14, 0.01, -0.99]),
                "right": np.array([0.01, -0.14, -0.99])}
        # PLACE attitude. The operator deliberately tilted RX when placing so the far reach
        # stays feasible, and the collected data carries it: measured over the control
        # policy's rollout dumps, the tool near the box (x>0.60) runs 20-26 deg off vertical
        # with tool z tipped toward +x [0.4, 0, -0.9], versus 6-9 deg over the pile. The
        # first oracle placed dead-vertical there -- exactly the ill-conditioned pose the
        # demos avoid on purpose.
        TILT_PLACE = {"left": np.array([0.42, 0.0, -0.91]),
                      "right": np.array([0.42, 0.0, -0.91])}

        @classmethod
        def _pose(cls, p_xyz, bolt_ang_deg, side="left", place=False, ref=None):
            """tool z at the measured approach tilt, tool x as perpendicular to the shaft as
            that tilt allows."""
            tilt = cls.TILT_PLACE if place else cls.TILT
            zt = tilt[side] / np.linalg.norm(tilt[side])
            b = np.array([np.cos(np.radians(bolt_ang_deg)), np.sin(np.radians(bolt_ang_deg)), 0.0])
            xt = np.cross(np.array([0.0, 0.0, 1.0]), b)
            n = np.linalg.norm(xt)
            xt = xt / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
            xt = xt - np.dot(xt, zt) * zt                  # re-orthogonalise against the tilt
            xt = xt / np.linalg.norm(xt)
            # A two-jaw gripper is 180-deg symmetric: "perpendicular to the shaft" has TWO
            # solutions a half-turn apart, and cross(up, shaft) picks between them by which
            # way the bolt HEAD points -- sending the wrist on needless half-turns between
            # picks (operator, from video). Take the solution nearer the current jaw axis;
            # the yaw move is then never more than 90 deg.
            if ref is not None and np.dot(xt, rotvec_to_mat(np.asarray(ref[3:]))[:, 0]) < 0.0:
                xt = -xt
            yt = np.cross(zt, xt)
            R = np.column_stack([xt, yt, zt])
            return np.concatenate([p_xyz, mat_to_rotvec(R)])

        def _pick(self, side, B, placed):
            """Nearest unclaimed, unplaced bolt of this arm's colour."""
            cols = (set(colors) if ORACLE_ARM == side
                    else {c for c, a_ in ARM_OF_COLOR.items() if a_ == side})
            cand = [i for i, c in enumerate(colors)
                    if c in cols and bolt_prims[i] not in placed
                    and bolt_prims[i] not in self.claimed
                    and self.cooldown.get(i, 0) <= self.clock]
            if not cand:                       # everything cooling down: take the coolest
                cand = [i for i, c in enumerate(colors)
                        if c in cols and bolt_prims[i] not in placed
                        and bolt_prims[i] not in self.claimed]
            if not cand:
                return None
            home = np.array([PILE_X, PILE_DY if side == "left" else -PILE_DY])

            def score(i):
                # Prefer ISOLATED bolts: overlapping pairs produced double-grasps and blocked
                # closes (operator, from video). Clearance to the nearest OTHER bolt counts
                # for, distance from this arm's pile counts against; clearance is capped at
                # 30 mm because beyond a jaw-width it buys nothing.
                d_others = [np.linalg.norm(B[i][:2] - B[j][:2])
                            for j in range(len(B)) if j != i]
                clear = min(min(d_others) if d_others else 1.0, 0.030)
                return 2.0 * clear - np.linalg.norm(B[i][:2] - home)

            k = max(cand, key=score)
            self.claimed.add(bolt_prims[k])
            if ORACLE_ARM != side:
                # bimanual: box = the bolt's own colour
                self.place_col[side] = colors[k]
            # single-arm: place_col stays this arm's OWN box (init value). The far box is at
            # the edge of the arm's comfortable reach and placing there contaminated the
            # measurement (operator). Cross-colour bolts then score as placed_wrong, so for
            # oracle runs the pick metric is correct+wrong combined -- read it that way.
            return k

        def plan(self, side, anchor, B, placed):
            """-> (goal pose (6,), grip pct). Called once per chunk boundary."""
            self.age[side] += CHUNK_EXECUTE_STEPS
            self.clock += CHUNK_EXECUTE_STEPS
            stuck = self.age[side] > self.PHASE_TIMEOUT
            ph = self.phase[side]
            i = self.target[side]
            if ph == "seek" and i is None:
                i = self.target[side] = self._pick(side, B, placed)
                if i is None:
                    # Nothing left to fetch: hover over this arm's pile rather than commanding
                    # the anchor itself, which is the degenerate request Ruckig rejects.
                    return self._pose(np.array([PILE_X, PILE_DY if side == "left" else -PILE_DY,
                                                self.H_LIFT]), 0.0, side), self.OPEN
            if i is not None and ph in ("seek", "descend", "close"):
                b = B[i]
                ang = bolt_axis_xy(i)[0]
                if ph == "seek" or ph == "descend":
                    if stuck:
                        # Blocked (a wall in the descent path, an unreachable pose): abandon
                        # THIS bolt for 10 s and pick another, instead of force-advancing into
                        # a pointless close and then retrying the same bolt forever -- the
                        # operator-observed loop. Deliberately no path planning: a cooldown
                        # list is all a physics instrument needs.
                        self.cooldown[i] = self.clock + 300
                        self.claimed.discard(bolt_prims[i])
                        self.target[side] = None
                        self._go(side, "seek")
                        return self._pose(np.array([b[0], b[1], self.H_LIFT]),
                                          ang, side, ref=anchor), self.OPEN
                if ph == "seek":
                    lat = float(np.linalg.norm(anchor[:2] - b[:2]))
                    # stay at transit height until overhead; descending early is what dragged
                    # the fingers through the box walls (and through the pile itself)
                    tz = b[2] + self.H_APPROACH if lat < self.TRANSIT_XY else self.H_LIFT
                    goal = self._pose(np.array([b[0], b[1], tz]), ang, side, ref=anchor)
                    if self.DEBUG and self.age[side] % 20 == 0:
                        att = np.degrees(np.linalg.norm(mat_to_rotvec(
                            rotvec_to_mat(anchor[3:]).T @ rotvec_to_mat(goal[3:]))))
                        print(f"      [oracle] {side} seek t={self.age[side]:3d} "
                              f"dxy={np.linalg.norm(anchor[:2]-b[:2])*1e3:6.1f}mm "
                              f"dz={(anchor[2]-b[2]-self.H_APPROACH)*1e3:+7.1f}mm "
                              f"datt={att:5.1f}deg")
                    if (np.linalg.norm(anchor[:2] - b[:2]) < 0.012
                            and abs(anchor[2] - (b[2] + self.H_APPROACH)) < 0.020):
                        self._go(side, "descend")
                    return goal, self.OPEN
                if ph == "descend":
                    goal = self._pose(np.array([b[0], b[1], b[2] + self.Z_GRASP_OFF]), ang, side, ref=anchor)
                    if abs(anchor[2] - (b[2] + self.Z_GRASP_OFF)) < 0.008:
                        self._go(side, "close", self.CLOSE_TICKS)
                    return goal, self.OPEN
                # close: hold still and shut, ramping the jaws rather than stepping them
                self.timer[side] -= CHUNK_EXECUTE_STEPS
                if self.timer[side] <= 0:
                    self._go(side, "lift")
                closed_for = self.CLOSE_TICKS - max(0, self.timer[side])
                g = max(self.SHUT, self.OPEN - self.GRIP_RATE * closed_for)
                return (self._pose(np.array([b[0], b[1], b[2] + self.Z_GRASP_OFF]), ang, side, ref=anchor), g)
            if ph == "lift":
                goal = anchor.copy(); goal[2] = self.H_LIFT
                if stuck or anchor[2] > self.H_LIFT - 0.020:
                    held = (i is not None
                            and np.linalg.norm(B[i] - anchor[:3]) < 0.065)
                    if held:
                        self._go(side, "carry")
                    else:
                        # Lifted nothing: an 8 s empty round-trip to the box teaches us
                        # nothing. Cool the bolt down and move on.
                        self.cooldown[i] = self.clock + 300
                        self.claimed.discard(bolt_prims[i])
                        self.target[side] = None
                        self._go(side, "seek")
                return goal, self.SHUT
            if ph == "carry":
                col = self.place_col[side]
                goal = self._pose(np.array([self.PLACE_X, BOX_CY[col], self.H_LIFT]), 0.0, side, place=True)
                if stuck or np.linalg.norm(anchor[:2] - goal[:2]) < 0.035:
                    self._go(side, "release", self.RELEASE_TICKS)
                return goal, self.SHUT
            # release
            self.timer[side] -= CHUNK_EXECUTE_STEPS
            if self.timer[side] <= 0:
                # Release the claim so a MISSED bolt is retried. Holding it would make the
                # oracle walk away from every failure, which is exactly the wrong behaviour in
                # a ceiling measurement -- the ceiling has to include retries.
                if self.target[side] is not None:
                    self.claimed.discard(bolt_prims[self.target[side]])
                self.target[side] = None
                self._go(side, "seek")
            # Hold ABOVE THE BOX while the jaws open, not at the anchor: an exact self-target
            # is the degenerate Ruckig input, and this keeps the bolt over the box as it drops.
            col_ = self.place_col[side]
            return self._pose(np.array([self.PLACE_X, BOX_CY[col_], self.H_LIFT]), 0.0, side, place=True), self.OPEN

        def chunk(self, side, anchor, B, placed):
            """Absolute goal -> H rows of the sequential body-frame deltas the loop integrates."""
            goal, grip = self.plan(side, anchor, B, placed)
            rows = np.zeros((ACTION_HORIZON, 7), dtype=float)
            cur_p, cur_R = anchor[:3].copy(), rotvec_to_mat(anchor[3:])
            R_goal = rotvec_to_mat(goal[3:])
            for k in range(ACTION_HORIZON):
                d = goal[:3] - cur_p
                lim = np.array([self.STEP_XY, self.STEP_XY, self.STEP_Z])
                # PROPORTIONAL, not saturating-then-stopping. The runner drops the rows that
                # elapsed since the observation (`drop` = 4 here), so rows 4..7 are what actually
                # executes. A step that clips straight to the goal reaches it by row 2 and leaves
                # rows 4..7 as exact zeros -- the arm then never moves again, the anchor never
                # advances, and the next chunk reproduces the same zeros. That deadlock froze
                # three oracle runs at 17-29 mm lateral error, always below the 4 x 10 mm the
                # skipped rows could have covered. A geometric approach never emits zeros.
                step = np.clip(self.GAIN * d, -lim, lim)
                nxt_p = cur_p + step
                # slew the attitude a fixed fraction per tick so it arrives with the position
                dR = cur_R.T @ R_goal
                rv = mat_to_rotvec(dR)
                nxt_R = cur_R @ rotvec_to_mat(rv * 0.25)
                # invert the loop's composition: p += R_cur @ (R_ALIGN @ dp)
                rows[k, :3] = R_ALIGN.T @ (cur_R.T @ (nxt_p - cur_p))
                rows[k, 3:6] = R_ALIGN.T @ mat_to_rotvec(cur_R.T @ nxt_R)
                # rows 4..7 are what actually executes, so the ramp has to live in the rows.
                rows[k, 6] = max(0.0, min(1.0, (grip - self.GRIP_RATE * k) / 100.0)) \
                    if grip < self.OPEN else grip / 100.0
                cur_p, cur_R = nxt_p, nxt_R
            return rows

    oracle = Oracle() if args.oracle else None
    client = None if args.oracle else \
        websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to {args.host}:{args.port}")
    if client is not None:
        print(f"server metadata: {client.get_server_metadata()}")
    else:
        print("ORACLE mode: privileged-state planner, no policy server")

    def tcp_pose(side):
        pos, quat = tcp_views[side].get_world_poses()
        p = np.asarray(pos)[0].astype(float)
        R = quat_to_mat(np.asarray(quat)[0].astype(float))
        return p, R

    def set_arm(side, q_arm, grip):
        art = arts[side]
        names = list(art.dof_names)
        q = np.array(art.get_joint_positions(), dtype=np.float32)
        for j, val in zip(ARM_JOINTS, q_arm):
            q[names.index(j)] = val
        fp = (1.0 - grip / 100.0) * FINGER_TRAVEL_M
        for jn, sgn in (("finger_left_joint", +1.0), ("finger_right_joint", -1.0)):
            if jn in names:
                q[names.index(jn)] = sgn * fp
        art.apply_action(ArticulationAction(joint_positions=q))

    T_mount = {}
    for side, m in mount_xf.items():
        T_mount[side] = np.array([[m[i][j] for j in range(4)] for i in range(4)]).T

    results = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        # ---- reset -----------------------------------------------------------
        for side, art in arts.items():
            names = list(art.dof_names)
            q = np.zeros(len(names), dtype=np.float32)
            for j, deg in zip(ARM_JOINTS, RESET[side]):
                q[names.index(j)] = np.deg2rad(deg)
            if side == ORACLE_PARKED:
                q[:] = 0.0                     # straight up, out of the working volume
            art.set_joint_positions(q)
            art.set_joint_velocities(np.zeros_like(q))
            # kp=1e7 at a 2 ms step is extremely stiff; expose it so the tremor can be
            # tested against drive stiffness instead of assumed innocent.
            _kp = float(os.environ.get("TREMOR_KP", 1.0e7))
            _kd = float(os.environ.get("TREMOR_KD", 1.0e5))
            kps, kds = np.full(len(names), _kp), np.full(len(names), _kd)
            # The FINGERS were inheriting the arm's gains: kp=1e7 on a prismatic joint is
            # 10 MN/m, and grip 0 commands 47 mm of travel -- straight through an 18.4 mm bolt
            # head. Real hardware stalls a finite-torque motor instead. Left at the inherited
            # value by default so nothing changes silently; GRIP_KP/GRIP_MAXF make it testable.
            _gkp = os.environ.get("GRIP_KP")
            _gmf = os.environ.get("GRIP_MAXF")
            fidx = [names.index(j) for j in ("finger_left_joint", "finger_right_joint")
                    if j in names]
            if _gkp and fidx:
                kps[fidx] = float(_gkp)
                kds[fidx] = float(os.environ.get("GRIP_KD", float(_gkp) * 1e-2))
            art.get_articulation_controller().set_gains(kps=kps, kds=kds)
            if _gmf and fidx:
                # Effort limits are the more physical knob (a real motor stalls), but the API
                # differs across Isaac versions. Never let it kill a run silently or otherwise;
                # GRIP_KP alone is enough to bound the force via penetration x stiffness.
                try:
                    eff = np.asarray(art.get_max_efforts(), dtype=np.float32).reshape(-1)
                    eff[fidx] = float(_gmf)
                    art.set_max_efforts(eff)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  [grip] max-effort unavailable ({exc}); using GRIP_KP only")
            art.apply_action(ArticulationAction(joint_positions=q))

        poses = bolt_poses(args.layout, args.n_per_color, ep_seed)
        colors = [c for c, *_ in poses]
        for (color, x, y, yaw), path in zip(poses, bolt_prims):
            v = bolt_views[path]
            qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
            v.set_world_poses(np.array([[x, y, SHAFT_R + 0.002]]), np.array([qz]))
            v.set_velocities(np.zeros((1, 6)))
        for i, path in enumerate(bolt_prims):
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path + "/shaft")).Bind(
                MAT["bolt_gray" if colors[i] == "gray" else "bolt_black"])
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path + "/head")).Bind(
                MAT["bolt_gray" if colors[i] == "gray" else "bolt_black"])
        frozen = None
        if args.scene_states:
            frozen = _SCENE_STATES.get(str(ep_seed))
            if frozen is None:
                raise SystemExit(
                    f"--scene-states has no entry for seed {ep_seed}; regenerate it "
                    "with --dump-scene-states for this layout/n-per-color")
        if frozen is not None:
            # Exact settled world: position + orientation + zero velocity per bolt.
            for path, st in zip(bolt_prims, frozen["bolts"]):
                v = bolt_views[path]
                v.set_world_poses(np.array([st["p"]], dtype=float),
                                  np.array([st["q"]], dtype=float))
                v.set_velocities(np.zeros((1, 6)))
            world.step(render=False)
        else:
            for _ in range(240):
                world.step(render=False)
        # T1 scene difficulty is a property of the CONDITION, so measure it on the settled
        # START scene. Measuring after the episode (as this first did) let the arms disturb the
        # pile, and the same seed reported different crowding in two runs.
        B0 = bolt_xyz()
        # A bolt that settles inside a box was never picked or placed -- it was born there.
        # Counting it inflated every arm's score (0.05/ep aligned, 0.40/ep random) and was
        # enough on its own to produce the bogus "random scores higher than aligned" reading.
        pre_in_box = {
            path for i, path in enumerate(bolt_prims)
            if any(abs(B0[i][0] - BOX_X) <= BOX["hw"] and abs(B0[i][1] - cy) <= BOX["hd"]
                   for cy in BOX_CY.values())}
        if pre_in_box:
            print(f"[ep {ep:02d}] {len(pre_in_box)} bolt(s) settled inside a box -> excluded")
        crowd = {}
        for col in set(colors):
            idx = [i for i, c in enumerate(colors) if c == col]
            if len(idx) > 1:
                P = B0[idx][:, :2]
                D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
                np.fill_diagonal(D, np.inf)
                crowd[col] = float(np.median(D.min(axis=1)))
        if args.dump_scene_states:
            P, Q = bolt_all_poses()
            _DUMP[str(ep_seed)] = dict(
                layout=args.layout, colors=colors,
                bolts=[dict(p=[float(x) for x in P[i]], q=[float(x) for x in Q[i]])
                       for i in range(len(bolt_prims))])
            print(f"  [freeze] seed {ep_seed}: captured {len(bolt_prims)} bolt poses")
            continue
        # Warm the renderer BEFORE the first observation. The per-tick orchestrator step runs
        # at the END of a tick, so without this the very first _observe() reads unfilled
        # annotators and the episode's first chunk is computed from black images. Every run
        # before 2026-08-19 had that defect (harmless-looking: one bad 0.8 s chunk at reset).
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        if ep == 0:
            for i, path in enumerate(bolt_prims):
                bp = np.asarray(bolt_views[path].get_world_poses()[0])[0]
                print(f"  bolt {i} {colors[i]:5s} -> ({bp[0]:+.3f},{bp[1]:+.3f},{bp[2]:+.3f})")

        # ---- follower + command state ---------------------------------------
        followers, cmd_hist, grip_cmd, q_cmd = {}, {}, {}, {}
        for side in MOUNT_FRAME:
            q_cmd[side] = (np.zeros(6) if side == ORACLE_PARKED
                           else np.deg2rad(np.array(RESET[side], dtype=float)))
            p_fk, R_fk, _, _ = fk_chain(q_cmd[side], T_mount[side])
            p_ph, _ = tcp_pose(side)
            d = float(np.linalg.norm(p_fk - p_ph))
            if ep == 0:
                print(f"  FK check {side:5s}: analytic vs physics TCP = {d*1000:.3f} mm")
            p, R = tcp_pose(side)
            pose6 = np.concatenate([p, mat_to_rotvec(R)])
            followers[side] = RuckigArmFollower(pose6)
            cmd_hist[side] = [pose6.copy(), pose6.copy()]
            grip_cmd[side] = 100.0

        frames = []
        n_ticks = int(args.episode_sec / POLICY_DT)
        chunk = {s: None for s in MOUNT_FRAME}
        grip_hist = {s: [] for s in MOUNT_FRAME}     # commanded grip per policy tick
        if oracle is not None:
            oracle.__init__()                 # planner state is per-episode
        chunk_idx = 0
        knots = {s_: None for s_ in MOUNT_FRAME}
        tcp_log = {s_: [] for s_ in MOUNT_FRAME}
        sub_i = -1
        dg = {s_: {k: [] for k in ("dq", "res", "smin", "grip", "refz", "cmdtcp", "q", "gq")}
              for s_ in MOUNT_FRAME}
        rtc_prev_raw = None      # reset per episode, like _rtc_prev_raw_chunk
        pending = None           # (full chunk (H,14), obs_tick) kicked at prefetch_at, swapped at boundary
        prev_grip = {s_: 100.0 for s_ in MOUNT_FRAME}
        close_events = []        # per gripper-close command: TCP-vs-nearest-bolt error, grasp outcome
        grasp_checks = []        # (event_index, bolt_path, bolt_z0, side, due_tick)
        # T1: per tick, index of the bolt nearest this arm's 24-step-ahead chunk endpoint.
        # Changes in this series ARE target switches -- the "bolt hopping" the multimodal
        # conditional would produce at a chunk boundary.
        target_track = {s_: [] for s_ in MOUNT_FRAME}
        _bxy = None
        held = {}        # bolt path -> (side, grasp tick); who is carrying what
        placed_at = {}   # bolt path -> first box entry (time, arm, transport duration)
        eject_watch = []  # [close idx, bolt idx, due tick, max displacement, pos at close]
        # PICK vs TRANSPORT. Operator observation from the oracle videos: bolts are picked and
        # then dropped on the way to the box, and the arm shakes near the box where the IK is
        # ill-conditioned. Placements therefore mix "did it grasp" with "did it survive the
        # carry", and the two rank the arms differently (up to 5 places apart on the current
        # board). Track the lift directly off bolt height so the pick stage can be read alone;
        # a bolt above LIFT_Z was carried, whatever the close-event detector thought.
        LIFT_Z = 0.060
        bolt_max_z = np.zeros(len(bolt_prims))
        smin_log = []
        held_last = {}   # bolt path -> most recent (arm, grasp tick)
        blank_obs = {s_: 0 for s_ in MOUNT_FRAME}   # empty wrist frames fed to the policy
        rtc_warned = False
        infer_ms = []

        for tick in range(n_ticks):
            # ---- observation (velproprio_source=command, fixed_step) ---------
            # velocity_grip proprio, exactly the training converter's _arm_velocity
            # (openpi_remote._proprio_state_velocity):
            #     pos_vel = R_cur^T . (p_next - p_cur)      BODY frame, previous pose
            #     rot_vel = rotvec(R_cur^T . R_next)
            # It is a PER-STEP displacement in the previous body frame -- NOT divided by
            # dt (the fixed_step window already is one ~policy_dt frame), and NOT a world
            # vector. Dividing by dt fed the policy ~30x inflated numbers. R_align maps
            # the RB TCP frame to the EE (pika tip) frame the checkpoint trained in.
            state = np.zeros(14, dtype=np.float32)
            for bi, side in enumerate(("left", "right")):
                h = cmd_hist[side]
                p_cur, p_next = h[-2][:3], h[-1][:3]
                R_cur, R_next = rotvec_to_mat(h[-2][3:]), rotvec_to_mat(h[-1][3:])
                pos_vel = R_cur.T @ (p_next - p_cur)
                rot_vel = mat_to_rotvec(R_cur.T @ R_next)
                state[bi * 7 + 0: bi * 7 + 3] = R_ALIGN @ pos_vel
                state[bi * 7 + 3: bi * 7 + 6] = R_ALIGN @ rot_vel
                # GRIPPER PROPRIO SOURCE. The deployed runner defaults to `actual` -- the
                # MEASURED jaw, which lags its command by 105-209 ms on hardware. This rig has
                # always sent the COMMAND, which has zero lag, so the grip channel the policy
                # sees here is not the channel it was trained on. That matters more than it
                # looks: the measured grip-echo pathology means the model largely predicts
                # "current grip - epsilon", so feeding it its own command closes the loop on
                # itself. Left on `command` by default so eleven arms of history stay
                # comparable; GRIP_PROPRIO=actual switches to the deploy contract.
                if GRIP_PROPRIO == "actual":
                    names = list(arts[side].dof_names)
                    q_now = np.asarray(arts[side].get_joint_positions()).reshape(-1)
                    fp = np.mean([abs(float(q_now[names.index(j)]))
                                  for j in ("finger_left_joint", "finger_right_joint")
                                  if j in names])
                    state[bi * 7 + 6] = float(np.clip(1.0 - fp / FINGER_TRAVEL_M, 0.0, 1.0))
                else:
                    state[bi * 7 + 6] = grip_cmd[side] / 100.0

            def _observe():
                imgs = {}
                for side in ("left", "right"):
                    a = wrist_cams[side].get_rgba()
                    arr = np.asarray(a) if a is not None else None
                    if arr is None or arr.size == 0:
                        # NEVER let this pass quietly: a black observation is not a policy
                        # result, it is a broken harness, and it scores 0 while looking real.
                        arr = np.zeros((480, 640, 4), dtype=np.uint8)
                        blank_obs[side] += 1
                    imgs[side] = arr[..., :3].astype(np.uint8)
                if DUMP_OBS and tick % 30 == 0:
                    for side in ("left", "right"):
                        imageio.imwrite(
                            pathlib.Path(DUMP_OBS) / f"obs_{args.tag}_{ep:02d}_{tick:04d}_{side}.png",
                            imgs[side])
                obs = {
                    "observation/left_wrist_0_rgb": imgs["left"],
                    "observation/right_wrist_0_rgb": imgs["right"],
                    "observation/state": state,
                    "prompt": PROMPT,
                }
                if RTC_ENABLED and rtc_prev_raw is not None:
                    # rtc_shift_prev_chunk: advance the cached raw chunk by the steps that
                    # will have executed when the new chunk takes over, so the freeze pins
                    # to the UNEXECUTED tail rather than replaying old actions.
                    steps = int(max(0, min(CHUNK_EXECUTE_STEPS, rtc_prev_raw.shape[0])))
                    pad = np.zeros((steps, rtc_prev_raw.shape[1]), dtype=rtc_prev_raw.dtype)
                    obs["prev_action_chunk"] = np.concatenate([rtc_prev_raw[steps:], pad], axis=0)
                    obs["inference_delay"] = int(np.clip(RTC_INFERENCE_DELAY, 0,
                                                         CHUNK_EXECUTE_STEPS))
                return obs

            def _infer(obs):
                nonlocal rtc_prev_raw, rtc_warned
                t0 = time.time()
                if oracle is not None:
                    # Same (H,14) contract the server returns: [left 7 | right 7].
                    B = bolt_xyz()
                    out = np.zeros((ACTION_HORIZON, 14), dtype=float)
                    # The parked arm must not PLAN either: its virtual planner was cycling
                    # grip ramps and polluting the close-event stream with ~30 phantom closes
                    # per run (found via the P_R5/P_RL0 side histogram). Hold it open.
                    if ORACLE_PARKED is not None:
                        pi_ = ("left", "right").index(ORACLE_PARKED)
                        out[:, pi_ * 7 + 6] = 1.0
                    for bi_o, side_o in enumerate(("left", "right")):
                        if side_o == ORACLE_PARKED:
                            continue
                        p_a, R_a, _, _ = fk_chain(q_cmd[side_o], T_mount[side_o])
                        anchor = np.concatenate([p_a, mat_to_rotvec(R_a)])
                        out[:, bi_o * 7: bi_o * 7 + 7] = oracle.chunk(
                            side_o, anchor, B, set(placed_at))
                    infer_ms.append((time.time() - t0) * 1e3)
                    return out
                res = client.infer(obs)
                infer_ms.append((time.time() - t0) * 1e3)
                if RTC_ENABLED:
                    raw = res.get("rtc_raw_actions")
                    if raw is not None:
                        rtc_prev_raw = np.asarray(raw, dtype=np.float32)
                    elif not rtc_warned:
                        print("  [rtc] server returned no 'rtc_raw_actions' -> staying vanilla")
                        rtc_warned = True
                return np.asarray(res["actions"], dtype=float)   # (H, 14)

            def _activate(act):
                for bi, side in enumerate(("left", "right")):
                    chunk[side] = act[:, bi * 7: bi * 7 + 7]

            if chunk["left"] is None:
                # first chunk of the episode: synchronous, no elapsed rows (runner does the same)
                _activate(_infer(_observe()))
                chunk_idx = 0
                if sum(blank_obs.values()):
                    raise SystemExit(
                        "ABORT: the wrist cameras returned empty frames on the FIRST observation "
                        "-- the policy would be scored on black images. The renderer is not "
                        "filling its annotators; check that rep.orchestrator.step() runs every "
                        "policy tick (it must not be gated on --video).")
            elif chunk_idx >= CHUNK_EXECUTE_STEPS:
                # boundary: swap in the prefetched plan, aligned by the ticks that elapsed
                # since its observation (flow_inference._activate_chunk source_start_index)
                if pending is None:
                    act, obs_tick = _infer(_observe()), tick   # stall fallback (never with sync infer)
                else:
                    act, obs_tick = pending
                    pending = None
                drop = int(np.clip(tick - obs_tick, 0, act.shape[0] - CHUNK_EXECUTE_STEPS))
                _activate(act[drop:])
                chunk_idx = 0
            if pending is None and chunk_idx == PREFETCH_AT:
                # kick the next inference now; it is swapped in at the next boundary
                pending = (_infer(_observe()), tick)

            # ---- integrate ee_local delta onto the COMMAND anchor -------------
            # flow_inference._target_payload_for_arm RE-ANCHORS at every chunk
            # boundary (steps_since_boundary == 0) instead of free-running the
            # accumulator. Without this the command drifts away from the robot --
            # measured here as x 0.359 -> 0.224 m over 4 s with 150-428 mm tracking
            # error. chunk_anchor="command" makes the anchor FK(q_sent).
            if chunk_idx == 0:
                for side in ("left", "right"):
                    p_a, R_a, _, _ = fk_chain(q_cmd[side], T_mount[side])
                    anchor_pose = np.concatenate([p_a, mat_to_rotvec(R_a)])
                    cmd_hist[side][-1] = anchor_pose
                    # Integrate the whole horizon into absolute knots so the follower can
                    # look ahead; knots[0] is the anchor itself.
                    ks = [anchor_pose.copy()]
                    cur_p, cur_R = anchor_pose[:3].copy(), rotvec_to_mat(anchor_pose[3:])
                    for r in chunk[side]:
                        dl = R_ALIGN @ r[:3] * SPEED_SCALE
                        da = R_ALIGN @ r[3:6] * SPEED_SCALE
                        nl, na = np.linalg.norm(dl), np.linalg.norm(da)
                        if nl > LIN_V * POLICY_DT:
                            dl = dl * (LIN_V * POLICY_DT / nl)
                        if na > ANG_V * POLICY_DT:
                            da = da * (ANG_V * POLICY_DT / na)
                        cur_p = cur_p + cur_R @ dl
                        cur_R = cur_R @ rotvec_to_mat(da)
                        ks.append(np.concatenate([cur_p, mat_to_rotvec(cur_R)]))
                    knots[side] = ks
            _bxy = bolt_xyz()                     # T1: all bolt positions, once per tick
            bolt_max_z = np.maximum(bolt_max_z, _bxy[:, 2])
            # Jacobian conditioning of the CURRENT command, once per policy tick. The place
            # boxes sit near the far edge of the reach envelope, so this is where an
            # ill-conditioned solve would shake the arm and shed the bolt.
            for _s in ("left", "right"):
                _p, _, _, _ = fk_chain(q_cmd[_s], T_mount[_s])
                smin_log.append((_s, float(np.linalg.svd(jacobian(q_cmd[_s], T_mount[_s]),
                                                         compute_uv=False)[-1]),
                                 float(np.linalg.norm(_p[:2] - np.array([BOX_X, BOX_CY[
                                     'gray' if _s == 'left' else 'black']])))))
            for side in ("left", "right"):
                ks = knots[side]
                k = min(chunk_idx + 1, len(ks) - 1)          # knot for this policy step
                new_pose = ks[k]
                # central difference over the reserve lookahead -> target velocity
                lo = max(0, k - 1)
                hi = min(len(ks) - 1, k + min(RESERVE_STEPS, 1))
                tvel = (ks[hi] - ks[lo]) / (max(1, hi - lo) * POLICY_DT)
                tvel[:3] = np.clip(tvel[:3], -LIN_V, LIN_V)
                tvel[3:] = np.clip(tvel[3:], -ANG_V, ANG_V)
                cmd_hist[side].append(new_pose.copy())
                if len(cmd_hist[side]) > 64:
                    cmd_hist[side].pop(0)
                followers[side].set_target(new_pose, tvel)
                row = chunk[side][min(chunk_idx, chunk[side].shape[0] - 1)]
                # GRIPPER LEAD. UMI records (pose_k, grip_k) as one synchronous measurement of a
                # human hand: there is no command/actual split and no latency, so grip_k means
                # "the jaw opening at the instant the tool is at pose_k". Reproducing that on a
                # robot needs the two to ARRIVE together, and they do not: the arm realises a
                # command in ~133 ms while the jaw takes 105 ms (left) and 209 ms (right). The
                # jaw command therefore has to lead the paired arm command by (L_grip - L_arm),
                # which is +2.3 steps on the right and -0.8 on the left. Issuing them on the same
                # tick -- what every run so far has done -- lands the right jaw ~76 ms late, and
                # the right arm is exactly the one with the documented 78% half-open pathology.
                # This is an EXECUTION fix, not a training one: the data contains no latency to
                # learn from.
                gi = int(np.clip(chunk_idx + GRIP_LEAD[side], 0, chunk[side].shape[0] - 1))
                grip_cmd[side] = float(np.clip(chunk[side][gi][6] * 100.0, 0.0, 100.0))
                grip_hist[side].append(grip_cmd[side])
                # ---- T1: which bolt is this arm's 24-step-ahead intent pointing at? ----
                # knots[-1] is the far end of the integrated chunk, i.e. the policy's stated
                # intent for the next 24 steps -- a much cleaner target read than the current
                # TCP, which lags it by the whole horizon.
                ep_xy = np.asarray(knots[side][-1][:3])[:2]
                target_track[side].append(
                    int(np.argmin(np.linalg.norm(_bxy[:, :2] - ep_xy, axis=1))))

                # ---- gripper-close command: where is the TCP relative to the nearest bolt?
                def _jaw_bolt(R, i):
                    b_ang, b_z = bolt_axis_xy(i)
                    jaw = R @ np.array([1.0, 0.0, 0.0])
                    d = abs(float(np.degrees(np.arctan2(jaw[1], jaw[0]))) - b_ang) % 180.0
                    return dict(bolt_axis_deg=b_ang, bolt_axis_z=b_z,
                                # folded to [0,90]: a cylinder and a two-jaw gripper are both
                                # 180-deg symmetric, so 10 deg and 170 deg are the same grasp.
                                jaw_bolt_deg=min(d, 180.0 - d))

                if prev_grip[side] >= 40.0 and grip_cmd[side] < 40.0:
                    p_t, R_t = tcp_pose(side)
                    d_all = np.linalg.norm(_bxy[:, :2] - p_t[:2], axis=1)
                    order = np.argsort(d_all)
                    bi_ = int(order[0])
                    bj_ = int(order[1]) if len(order) > 1 else bi_
                    bp, bq = _bxy[bi_], _bxy[bj_]
                    d1, d2 = float(d_all[bi_]), float(d_all[bj_])
                    # Mode-average signature: does the aim sit BETWEEN the two candidates?
                    # t_seg is the projection of the TCP onto the b1->b2 segment (0 = on the
                    # nearest bolt, 1 = on the second); perp is the offset from that line.
                    u = (bq - bp)[:2]
                    w = p_t[:2] - bp[:2]
                    gap = float(np.linalg.norm(u))
                    if gap > 1e-6:
                        t_seg = float(np.dot(w, u) / (gap * gap))
                        perp = float(abs(u[0] * w[1] - u[1] * w[0]) / gap)
                    else:
                        t_seg, perp = 0.0, 0.0
                    tgt_color = [c for c, a_ in ARM_OF_COLOR.items() if a_ == side][0]
                    same = [i for i, c in enumerate(colors) if c == tgt_color]
                    d_tc = float(np.min(d_all[same])) if same else float("nan")
                    trk = target_track[side]
                    def _switches(n):
                        seg = trk[-n:]
                        return int(sum(1 for a_, b_ in zip(seg, seg[1:]) if a_ != b_))
                    lead = 0
                    for q_ in range(len(trk) - 1, 0, -1):
                        if trk[q_ - 1] != trk[-1]:
                            break
                        lead += 1
                    ev = dict(side=side, tick=tick, t=tick * POLICY_DT,
                              tcp=[float(v) for v in p_t],
                              # Aim error has only ever been measured in XY. Roll/pitch drift
                              # would tilt the jaws off the bolt axis and is invisible there.
                              tcp_rotvec=[float(v) for v in mat_to_rotvec(R_t)],
                              **_jaw_bolt(R_t, bi_),
                              # Bolts OTHER than the target sitting inside the jaw sweep
                              # volume at the moment of closing. The jaws sweep x in +-55 mm
                              # (open half-gap 47 + finger body), are ~28 mm half-wide in y,
                              # and act near the tip plane in z. A neighbour in that box gets
                              # struck: the observed double-grasps and blocked closes.
                              # Was the DESCENT blocked? bolt_elev = target sits on other
                              # bolts instead of the table; z_cmd_gap = physics held the TCP
                              # above where the command wanted it (fingers resting on a
                              # neighbour or the pile). Either way the jaws close higher than
                              # intended, and dz alone cannot tell WHY.
                              bolt_elev_mm=float((bp[2] - 0.0091) * 1e3),
                              z_cmd_gap_mm=float((p_t[2] - fk_chain(
                                  q_cmd[side], T_mount[side])[0][2]) * 1e3),
                              neighbors_in_jaw=int(sum(
                                  1 for j_ in range(len(_bxy)) if j_ != bi_
                                  and abs((v_ := R_t.T @ (_bxy[j_] - p_t))[0]) < 0.055
                                  and abs(v_[1]) < 0.028 and abs(v_[2]) < 0.035)),
                              bolt=[float(v) for v in bp], bolt_color=colors[bi_],
                              target_color=tgt_color,
                              dx=float(p_t[0] - bp[0]), dy=float(p_t[1] - bp[1]),
                              dz=float(p_t[2] - bp[2]), dxy=d1,
                              # --- T1 multimodality fields ---
                              d2=d2, margin=d2 - d1, pair_gap=gap,
                              t_seg=t_seg, perp=perp, second_color=colors[bj_],
                              d_target_color=d_tc,
                              nearest_is_target_color=bool(colors[bi_] == tgt_color),
                              switches_1s=_switches(30), switches_2s=_switches(60),
                              commit_lead_ticks=lead,
                              grasped=None)
                    close_events.append(ev)
                    # 2.5 s, not 1.5: the event fires when grip crosses 40%, and a partial
                    # opening (GRIP_OPEN=50) crosses ~17 ticks sooner relative to the lift, so
                    # at +45 the lift had barely begun and every real grasp scored False
                    # (orcP0: grasped 0/22 while PLACING 8 -- the placements are the proof).
                    grasp_checks.append((len(close_events) - 1, bolt_prims[bi_], float(bp[2]),
                                         side, tick + 75))
                    # EJECTION probe. The finger joints run the same kp=1e7 position drive as
                    # the arm, and grip 0 commands them 47 mm inward -- straight through an
                    # 18.4 mm bolt head. A real gripper stalls its motor at finite force; this
                    # one can shoot the bolt out. Watch the target bolt's speed for the next
                    # 0.5 s: a held bolt tracks the hand (<0.3 m/s), an ejected one does not.
                    eject_watch.append([len(close_events) - 1, bi_, tick + 15,
                                        float(np.linalg.norm(_bxy[bi_] - bp)), bp.copy()])
                prev_grip[side] = grip_cmd[side]
            if PROBE_DUMP and tick % PROBE_EVERY == 0 and _bxy is not None:
                rec = dict(ep=ep, seed=ep_seed, tick=tick, t=tick * POLICY_DT)
                for side in ("left", "right"):
                    a = wrist_cams[side].get_rgba()
                    arr = np.asarray(a) if a is not None else None
                    if arr is None or arr.size == 0:
                        continue
                    fn = f"pr_{args.tag}_{ep:02d}_{tick:04d}_{side}.png"
                    imageio.imwrite(pathlib.Path(PROBE_DUMP) / fn, arr[..., :3].astype(np.uint8))
                    p_t, R_t = tcp_pose(side)
                    d_all = np.linalg.norm(_bxy[:, :2] - p_t[:2], axis=1)
                    # ALL bolts, not the nearest three: the probe that reads these has to work in
                    # clutter, and a per-patch bolt/no-bolt label needs every bolt or the
                    # unlabelled ones become false negatives.
                    order = [int(i) for i in np.argsort(d_all)]
                    rec[side] = dict(
                        img=fn,
                        tcp=[float(v) for v in p_t],
                        tcp_rotvec=[float(v) for v in mat_to_rotvec(R_t)],
                        grip=float(grip_cmd[side]),
                        endpoint=[float(v) for v in np.asarray(knots[side][-1][:3])],
                        bolts_world=[[float(v) for v in _bxy[i]] for i in order],
                        bolts_tool=[[float(v) for v in (R_t.T @ (_bxy[i] - p_t))] for i in order],
                        bolt_colors=[colors[i] for i in order],
                    )
                with open(pathlib.Path(PROBE_DUMP) / f"probe_{args.tag}.jsonl", "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

            # resolve the ejection probe: peak displacement of the bolt since the close
            still_e = []
            for w in eject_watch:
                w[3] = max(w[3], float(np.linalg.norm(_bxy[w[1]] - w[4])))
                if tick >= w[2]:
                    close_events[w[0]]["bolt_move_mm"] = w[3] * 1e3
                else:
                    still_e.append(w)
            eject_watch = still_e

            # resolve pending grasp checks: did the nearest bolt come up with the gripper?
            still = []
            for (ei, path, z0, side_, due) in grasp_checks:
                if tick >= due:
                    bz = float(np.asarray(bolt_views[path].get_world_poses()[0])[0][2])
                    ok = bool(bz - z0 > 0.03 and grip_cmd[side_] < 40.0)
                    close_events[ei]["grasped"] = ok
                    # What the jaw PHYSICALLY did 1.5 s after the command: 0 = fully shut
                    # (closed on nothing or crushed through), ~28 = stalled on a bolt head,
                    # large = never closed (jammed on the floor or a neighbour). grip_cmd
                    # cannot tell these apart; the measured finger joints can.
                    _nm = list(arts[side_].dof_names)
                    _q = np.asarray(arts[side_].get_joint_positions()).reshape(-1)
                    _fp = np.mean([abs(float(_q[_nm.index(j)]))
                                   for j in ("finger_left_joint", "finger_right_joint")
                                   if j in _nm])
                    close_events[ei]["achieved_gap_mm"] = float(
                        2.0 * (0.047 - _fp) * 1e3)
                    if ok:
                        # Remember who is carrying what, so a later box entry can be
                        # attributed to an arm and to a transport duration.
                        # Keep BOTH: a bolt that is dropped and re-grasped overwrites the
                        # latest entry, which made transport_s collapse to ~0.2 s. The first
                        # grasp measures the whole attempt, the last the actual carry.
                        held.setdefault(path, (side_, tick))
                        held_last[path] = (side_, tick)
                else:
                    still.append((ei, path, z0, side_, due))
            grasp_checks = still

            # ---- placement TIMING -------------------------------------------------
            # A count of placements says nothing about how long the robot took to get them.
            # Record the FIRST tick each bolt enters a box; that gives time-to-first-success,
            # the interval between successes (cycle time) and, via `held`, the grasp->place
            # transport time per arm. Costs nothing: _bxy is already read every tick.
            for bi_, path in enumerate(bolt_prims):
                if path in placed_at or path in pre_in_box:
                    continue
                p = _bxy[bi_]
                for col, cy in BOX_CY.items():
                    if abs(p[0] - BOX_X) <= BOX["hw"] and abs(p[1] - cy) <= BOX["hd"]:
                        side_, gt = held.get(path, (None, None))
                        side_l, gl = held_last.get(path, (None, None))
                        placed_at[path] = dict(
                            bolt=bi_, bolt_color=colors[bi_], box_color=col,
                            t=tick * POLICY_DT, tick=tick, side=(side_l or side_),
                            grasp_t=(gt * POLICY_DT if gt is not None else None),
                            attempt_s=((tick - gt) * POLICY_DT if gt is not None else None),
                            transport_s=((tick - gl) * POLICY_DT if gl is not None else None))
                        break
            chunk_idx += 1

            # ---- servo: Ruckig reference -> IK -> drives ----------------------
            for _ in range(SUBSTEPS):
                sub_i += 1
                for side in ("left", "right"):
                    if side == ORACLE_PARKED:
                        q_cmd[side] = np.zeros(6)
                        set_arm(side, q_cmd[side], 100.0)
                        continue
                    followers[side].step()
                    ref = followers[side].reference
                    qc = q_cmd[side]
                    p_fk, R_fk, _, _ = fk_chain(qc, T_mount[side])
                    err = np.concatenate([ref[:3] - p_fk,
                                          mat_to_rotvec(rotvec_to_mat(ref[3:]) @ R_fk.T)])
                    for _ in range(IK_ITERS):
                        J = jacobian(qc, T_mount[side])
                        dq = J.T @ np.linalg.solve(J @ J.T + IK_LAMBDA * np.eye(6), err)
                        qc = np.clip(qc + np.clip(dq, -DQ_MAX, DQ_MAX), JOINT_LO, JOINT_HI)
                        p_fk, R_fk, _, _ = fk_chain(qc, T_mount[side])
                        err = np.concatenate([ref[:3] - p_fk,
                                              mat_to_rotvec(rotvec_to_mat(ref[3:]) @ R_fk.T)])
                    q_cmd[side] = qc
                    if _REP is not None:
                        gi = min(sub_i, len(_REP[f"{side}_q"]) - 1)
                        q_cmd[side] = _REP[f"{side}_q"][gi].copy()
                        grip_cmd[side] = float(_REP[f"{side}_gq"][gi])
                    # The JOINTS get the delayed value; grip_cmd stays the command, which is
                    # what the close event, the proprio "command" mode and the logs mean.
                    lag_t = int(round(GRIP_LAG_MS[side] / 1000.0 / POLICY_DT))
                    h = grip_hist[side]
                    g_apply = h[max(0, len(h) - 1 - lag_t)] if h else grip_cmd[side]
                    set_arm(side, q_cmd[side], g_apply)
                    if DIAG:
                        # sigma_min exposes a near-singular wrist, where DLS damping turns
                        # a small Cartesian error into a large, sign-flipping joint step.
                        dg[side]["dq"].append(float(np.abs(dq).max()))
                        dg[side]["res"].append(float(np.linalg.norm(err[:3])))
                        dg[side]["smin"].append(
                            float(np.linalg.svd(J, compute_uv=False)[-1]))
                        dg[side]["grip"].append(grip_cmd[side])
                        dg[side]["refz"].append(float(ref[2]))
                        # The decisive split: COMMANDED tcp (analytic FK of the joint
                        # command) vs MEASURED tcp. If commanded is smooth while measured
                        # shakes, the tremor is physics/contact, not the controller.
                        dg[side]["cmdtcp"].append(
                            fk_chain(q_cmd[side], T_mount[side])[0].copy())
                        dg[side]["q"].append(q_cmd[side].copy())
                        dg[side]["gq"].append(grip_cmd[side])
                world.step(render=False)
                # Sample at the PHYSICS rate, not the policy rate. The tremor lives INSIDE
                # the 33 ms knot interval (accelerate then brake across 17 substeps);
                # sampling once per knot aliases it away completely, which is why the first
                # version of this metric was blind to the very thing it was measuring.
                for side in ("left", "right"):
                    tcp_log[side].append(tcp_pose(side)[0].copy())

            if ep == 0 and tick % 30 == 0:
                dbg = []
                for side in ("left", "right"):
                    p_ph, _ = tcp_pose(side)
                    ref = followers[side].reference[:3]
                    cmd = cmd_hist[side][-1][:3]
                    dbg.append(f"{side[0]}: cmd=({cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f}) "
                               f"track_err={np.linalg.norm(ref - p_ph)*1000:5.1f}mm "
                               f"grip={grip_cmd[side]:5.1f}")
                print(f"  t{tick:04d} " + " | ".join(dbg))

            # The WRIST cameras are replicator render products too: their annotators only fill
            # when the orchestrator steps. This step must therefore run whether or not video is
            # being recorded. Gating it on --video (as this did until 2026-08-19) left
            # Camera.get_rgba() empty, _observe() silently substituted a black image, and ten
            # std20 episodes scored 0/0 while looking like a policy result.
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
            if ov_ann is not None:
                fr = np.asarray(ov_ann.get_data())
                if not fr.size:                       # one retry, as before
                    rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
                    fr = np.asarray(ov_ann.get_data())
                if fr.size:
                    im = fr[..., :3].astype(np.uint8)
                    if top_ann is not None:
                        tf = np.asarray(top_ann.get_data())
                        if tf.size:
                            im = np.concatenate([im, tf[..., :3].astype(np.uint8)], axis=1)
                    frames.append(im)

        # ---- score ------------------------------------------------------------
        placed, misplaced = 0, 0
        for i, path in enumerate(bolt_prims):
            if path in pre_in_box:
                continue
            pos, _ = bolt_views[path].get_world_poses()
            p = np.asarray(pos)[0]
            for col, cy in BOX_CY.items():
                inside = (abs(p[0] - BOX_X) <= BOX["hw"]) and (abs(p[1] - cy) <= BOX["hd"])
                if inside:
                    if col == colors[i]:
                        placed += 1
                    else:
                        misplaced += 1
        tremor, knotband, tremor_box = {}, {}, {}
        for side in ("left", "right"):
            P = np.asarray(tcp_log[side])
            if len(P) > 64:
                d2 = np.diff(P, n=2, axis=0)
                # micrometres per physics step^2
                tremor[side] = float(np.sqrt((d2 ** 2).sum(axis=1).mean()) * 1e6)
                # The operator sees the arm shake NEAR THE BOX specifically, where the
                # Jacobian conditioning collapses (smin ~2e-4). Bucket the same jerk metric
                # by region so "the controller trembles" stops being one number that averages
                # a calm pile approach with a shaking place.
                nb = P[1:-1, 0] > 0.60
                if nb.sum() > 64:
                    tremor_box[side] = float(np.sqrt((d2[nb] ** 2).sum(axis=1).mean()) * 1e6)
                # how much of the speed signal sits at the 30 Hz knot rate and its harmonic
                spd = np.linalg.norm(np.diff(P, axis=0), axis=1) / PHYSICS_DT
                spd = spd - spd.mean()
                freq = np.fft.rfftfreq(len(spd), PHYSICS_DT)
                mag = np.abs(np.fft.rfft(spd)) ** 2
                tot = mag[freq > 0.5].sum()
                f0 = 1.0 / POLICY_DT
                band = ((np.abs(freq - f0) < 3.0) | (np.abs(freq - 2 * f0) < 3.0))
                knotband[side] = float(mag[band].sum() / tot * 100.0) if tot > 0 else 0.0
        print("          tremor RMS |d2 TCP| : "
              + "  ".join(f"{k}={v:.1f} um" for k, v in tremor.items())
              + " | knot-rate energy: "
              + "  ".join(f"{k}={v:.1f}%" for k, v in knotband.items()))
        if DIAG:
            np.savez(outdir / f"diag_{args.tag}_{ep:02d}.npz",
                     **{f"{sd}_{k}": np.asarray(v)
                        for sd, d in dg.items() for k, v in d.items()},
                     **{f"{sd}_tcp": np.asarray(tcp_log[sd]) for sd in dg})
            print(f"          diag -> diag_{args.tag}_{ep:02d}.npz")
        for (ei, path, z0, side_, due) in grasp_checks:   # unresolved at episode end
            close_events[ei]["grasped"] = False
        n_close = len(close_events)
        n_grasp = sum(1 for e in close_events if e["grasped"])
        # ---- T1: target-track compression ---------------------------------------------
        # (crowding was measured on the START scene, above)
        def _rle(tr):
            out = []
            for i, v in enumerate(tr):
                if not out or out[-1][1] != v:
                    out.append([i, int(v)])
            return out
        tswitch = {s_: max(0, len(_rle(target_track[s_])) - 1) for s_ in MOUNT_FRAME}
        rec = dict(episode=ep, seed=ep_seed, layout=args.layout, bolts=n_bolts,
                   rtc=RTC_ENABLED, execute_steps=CHUNK_EXECUTE_STEPS, prefetch_at=PREFETCH_AT,
                   tremor_um=tremor, knot_energy_pct=knotband,
                   placed_correct=placed, placed_wrong=misplaced,
                   pre_placed=len(pre_in_box),
                   lifted=int(sum(1 for i, pth in enumerate(bolt_prims)
                                  if pth not in pre_in_box and bolt_max_z[i] > LIFT_Z)),
                   dropped=int(sum(1 for i, pth in enumerate(bolt_prims)
                                   if pth not in pre_in_box and bolt_max_z[i] > LIFT_Z
                                   and pth not in placed_at)),
                   smin_p10={_s: (float(np.percentile([v for t, v, d in smin_log if t == _s], 10))
                                  if any(t == _s for t, _, _ in smin_log) else None)
                             for _s in ("left", "right")},
                   tremor_box_um=tremor_box,
                   smin_p10_near_box={_s: (float(np.percentile(
                       [v for t, v, d in smin_log if t == _s and d < 0.15], 10))
                       if any(t == _s and d < 0.15 for t, _, d in smin_log) else None)
                       for _s in ("left", "right")},
                   close_events=close_events, n_close=n_close, n_grasp=n_grasp,
                   same_color_crowding_m=crowd,
                   placements=sorted(placed_at.values(), key=lambda e: e["tick"]),
                   blank_obs=dict(blank_obs),
                   target_switches=tswitch,
                   target_track_rle={s_: _rle(target_track[s_]) for s_ in MOUNT_FRAME},
                   infer_ms_p50=float(np.median(infer_ms)) if infer_ms else None,
                   infers=len(infer_ms))
        if close_events:
            dxy = np.array([e["dxy"] for e in close_events]) * 1e3
            dz = np.array([e["dz"] for e in close_events]) * 1e3
            print(f"          close cmds={n_close} grasped={n_grasp} | TCP-vs-nearest-bolt at close: "
                  f"|dxy| p50 {np.median(dxy):.1f} mm, dz p50 {np.median(dz):+.1f} mm")
            mg = np.array([e["margin"] for e in close_events]) * 1e3
            ts = np.array([e["t_seg"] for e in close_events])
            pp = np.array([e["perp"] for e in close_events]) * 1e3
            btw = int(np.sum((ts > 0.15) & (ts < 0.85) & (pp < 40.0)))
            wrong = int(np.sum([not e["nearest_is_target_color"] for e in close_events]))
            print(f"          T1 ambiguity: margin p50 {np.median(mg):.0f} mm "
                  f"(p10 {np.percentile(mg, 10):.0f}) | between-candidates {btw}/{n_close} | "
                  f"wrong-colour-nearest {wrong}/{n_close} | switches/ep "
                  f"L{tswitch['left']} R{tswitch['right']} | crowding "
                  + " ".join(f"{k}={v*1e3:.0f}mm" for k, v in sorted(crowd.items())))
        results.append(rec)
        print(f"[ep {ep:02d}] correct={placed} wrong={misplaced} "
              f"infer_p50={rec['infer_ms_p50']:.0f}ms n_infer={len(infer_ms)}")

        if args.video and frames:
            vp = outdir / f"{args.tag}_{args.layout}_{ep:02d}.mp4"
            imageio.mimwrite(vp, frames, fps=30, quality=6)
            print(f"          video {vp.name} ({len(frames)} frames)")

    _ev = [e for r in results for e in r["close_events"]]
    def _p50(vals, scale=1.0):
        return float(np.median(vals)) * scale if len(vals) else None
    t1 = dict(
        close_events=len(_ev),
        margin_p50_mm=_p50([e["margin"] for e in _ev], 1e3),
        margin_p10_mm=(float(np.percentile([e["margin"] for e in _ev], 10)) * 1e3
                       if _ev else None),
        # aim sits between two candidates and off both: the mode-average signature
        between_candidates=sum(1 for e in _ev
                               if 0.15 < e["t_seg"] < 0.85 and e["perp"] < 0.040),
        wrong_colour_nearest=sum(1 for e in _ev if not e["nearest_is_target_color"]),
        dxy_p50_mm=_p50([e["dxy"] for e in _ev], 1e3),
        dz_p50_mm=_p50([e["dz"] for e in _ev], 1e3),
        switches_1s_p50=_p50([e["switches_1s"] for e in _ev]),
        commit_lead_ticks_p50=_p50([e["commit_lead_ticks"] for e in _ev]),
        target_switches_per_episode={
            s: _p50([r["target_switches"][s] for r in results]) for s in ("left", "right")},
    )
    if args.dump_scene_states:
        pathlib.Path(args.dump_scene_states).write_text(json.dumps(_DUMP, indent=1))
        print(f"[freeze] wrote {len(_DUMP)} seeds -> {args.dump_scene_states}")
        app.close()
        return 0
    summary = dict(
        protocol=args.protocol, label=args.label or args.tag,
        scene_states=args.scene_states,
        # Runs before 2026-08-22 counted bolts that SETTLED inside a box as placements. The
        # frozen sets make that a constant per protocol (aligned +3 correct over 40 eps, random
        # +9 correct / +5 wrong), so old summaries stay correctable -- this flag is how
        # t1_report tells which ones still need the correction applied.
        pre_placed_excluded=True,
        policy=f"{args.host}:{args.port}",
        episodes=args.episodes, layout=args.layout,
        seeds=[r["seed"] for r in results], episode_sec=args.episode_sec,
        total_correct=sum(r["placed_correct"] for r in results),
        total_wrong=sum(r["placed_wrong"] for r in results),
        episodes_with_any_correct=sum(1 for r in results if r["placed_correct"] > 0),
        rtc=RTC_ENABLED, execute_steps=CHUNK_EXECUTE_STEPS, prefetch_at=PREFETCH_AT,
        total_close=sum(r["n_close"] for r in results),
        total_grasp=sum(r["n_grasp"] for r in results),
        # If this is not 0 the run is INVALID: the policy saw black wrist frames.
        blank_obs=sum(v for r in results for v in r["blank_obs"].values()),
        t1=t1,
        records=results,
    )
    if summary["blank_obs"]:
        print(f"!! INVALID RUN: {summary['blank_obs']} blank wrist observations were fed to "
              f"the policy. Do not report these numbers.")
    (outdir / f"summary_{args.tag}_{args.layout}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
