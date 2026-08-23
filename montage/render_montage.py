"""Expected-simulation montage for the dual RB3-730E M12 bolt cell.

Builds a MuJoCo scene from robotics_lab description assets (stand + dual arm +
Pika tool) plus procedural table / green totes / M12 socket-head bolts, and
renders overview + wrist-camera views for aligned vs randomized placements.
"""
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 960

SCENE_TMPL = """
<mujoco model="bolt_cell_montage">
  <compiler angle="radian" meshdir="meshes" autolimits="true" eulerseq="xyz" />
  <option integrator="implicitfast" />
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2" />
    <quality shadowsize="4096" />
    <global offheight="{H}" offwidth="{W}" />
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.37 0.40" rgb2="0.12 0.13 0.15" width="256" height="256" />
    <material name="table_metal" rgba="0.50 0.52 0.53 1" specular="0.35" shininess="0.4" reflectance="0.02" />
    <material name="floor_mat" rgba="0.22 0.22 0.23 1" specular="0.1" shininess="0.1" />
    <material name="tote_green" rgba="0.10 0.38 0.27 1" specular="0.3" shininess="0.3" />
    <material name="box_gray" rgba="0.52 0.53 0.55 1" specular="0.2" shininess="0.2" />
    <material name="foam_gray" rgba="0.62 0.62 0.63 1" specular="0.05" shininess="0.05" />
    <material name="insert_dark" rgba="0.28 0.28 0.30 1" specular="0.05" shininess="0.05" />
    <material name="bolt_gray" rgba="0.72 0.73 0.76 1" specular="0.9" shininess="0.8" reflectance="0.25" />
    <material name="bolt_black" rgba="0.09 0.09 0.10 1" specular="0.5" shininess="0.6" reflectance="0.05" />
    <mesh name="stand_mesh" file="stands/dual_rb3_730e/dual_rb3_730e_stand_ver3.stl" scale="0.001 0.001 0.001" />
  </asset>
  <asset>
    <model name="arm" file="rb3_730e_pika/rb3_730e_pika.xml" />
  </asset>
  <worldbody>
    <light pos="0.8 -1.2 2.2" dir="-0.3 0.4 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3" castshadow="true" />
    <light pos="-0.8 -0.3 2.0" dir="0.3 -0.2 -1" diffuse="0.4 0.4 0.4" castshadow="false" />

    <geom name="floor" type="plane" size="4 4 0.1" material="floor_mat" pos="0 0 -0.75" />
    <geom name="table" type="box" size="0.75 0.55 0.012" pos="0 -0.50 -0.012" material="table_metal" />
    <geom name="table_leg1" type="box" size="0.03 0.03 0.363" pos="0.68 -0.95 -0.387" material="foam_gray" />
    <geom name="table_leg2" type="box" size="0.03 0.03 0.363" pos="-0.68 -0.95 -0.387" material="foam_gray" />
    <geom name="table_leg3" type="box" size="0.03 0.03 0.363" pos="0.68 -0.08 -0.387" material="foam_gray" />
    <geom name="table_leg4" type="box" size="0.03 0.03 0.363" pos="-0.68 -0.08 -0.387" material="foam_gray" />

    <body name="stand" pos="0 0 0" euler="0 0 0">
      <geom name="stand_vis" type="mesh" mesh="stand_mesh" pos="0 0 0.01" euler="0 0 -1.57078"
        rgba="0.45 0.45 0.48 1" contype="0" conaffinity="0" />
    </body>
    <body name="left_mount" pos="0.1601 -0.1725 0.5825" euler="0.785 2.35619 0">
      <attach model="arm" body="world" prefix="L_" />
    </body>
    <body name="right_mount" pos="-0.1601 -0.1725 0.5825" euler="0.785 -2.35619 0">
      <attach model="arm" body="world" prefix="R_" />
    </body>

    {totes}
    {bolts}

    <camera name="overview" pos="1.05 -1.55 0.95" xyaxes="0.75 0.66 0 -0.28 0.32 0.90" fovy="55" />
    <camera name="top" pos="0 -0.48 1.35" xyaxes="1 0 0 0 1 0" fovy="62" />
    <camera name="front" pos="0 -1.75 0.55" xyaxes="1 0 0 0 0.45 0.89" fovy="52" />
  </worldbody>
</mujoco>
"""


def tote(name, x, y, w=0.155, d=0.115, h=0.075, t=0.012, mat="tote_green"):
    g = []
    # 5cm-tall dark-gray insert bottom + 4 walls
    g.append(f'<geom type="box" size="{w} {d} 0.025" pos="{x} {y} 0.025" material="insert_dark" />')
    g.append(f'<geom type="box" size="{w} {t} {h}" pos="{x} {y - d} {h + t}" material="{mat}" />')
    g.append(f'<geom type="box" size="{w} {t} {h}" pos="{x} {y + d} {h + t}" material="{mat}" />')
    g.append(f'<geom type="box" size="{t} {d} {h}" pos="{x - w} {y} {h + t}" material="{mat}" />')
    g.append(f'<geom type="box" size="{t} {d} {h}" pos="{x + w} {y} {h + t}" material="{mat}" />')
    return f'<body name="{name}">' + "".join(g) + "</body>"


def bolt(i, x, y, yaw, color):
    mat = "bolt_gray" if color == "gray" else "bolt_black"
    # M12 x 25mm SHCS lying on its side: shaft r6 x l25, head r9 x h12
    return (
        f'<body name="bolt{i}" pos="{x} {y} 0.0065" euler="0 0 {yaw}">'
        f'<geom type="cylinder" size="0.006 0.0125" euler="0 1.5708 0" pos="0.0105 0 0" material="{mat}" />'
        f'<geom type="cylinder" size="0.0092 0.006" euler="0 1.5708 0" pos="-0.008 0 0.0028" material="{mat}" />'
        f"</body>"
    )


def make_bolts(mode, n_per=11, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    if mode == "aligned":
        centers = {"gray": (0.16, -0.47), "black": (-0.16, -0.47)}
        i = 0
        for color, (cx, cy) in centers.items():
            for _ in range(n_per):
                x = cx + rng.normal(0, 0.05)
                y = cy + rng.normal(0, 0.045)
                out.append(bolt(i, x, y, rng.uniform(0, np.pi), color))
                i += 1
    else:  # random: two colors intermixed over the whole strip
        colors = ["gray"] * n_per + ["black"] * n_per
        rng.shuffle(colors)
        for i, color in enumerate(colors):
            x = rng.uniform(-0.30, 0.30)
            y = -0.47 + rng.normal(0, 0.07)
            out.append(bolt(i, x, y, rng.uniform(0, np.pi), color))
    return "\n    ".join(out)


JOINTS = ["base_joint", "shoulder_joint", "elbow_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]


def ik_arm(model, data, prefix, target_pos, iters=300):
    """Damped LS IK: put tcp_tip at target with tool +z pointing world -z."""
    sid = model.site(prefix + "tcp_tip").id
    dofs = [model.joint(prefix + j).dofadr[0] for j in JOINTS]
    qadr = [model.joint(prefix + j).qposadr[0] for j in JOINTS]
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[sid]
        zax = data.site_xmat[sid].reshape(3, 3)[:, 2]
        e_pos = target_pos - pos
        e_rot = np.cross(zax, np.array([0.0, 0.0, -1.0]))
        err = np.concatenate([e_pos, 0.6 * e_rot])
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
            break
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, dofs], 0.6 * jacr[:, dofs]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        for a, d in zip(qadr, dofs):
            data.qpos[a] += 0.5 * dq[dofs.index(d)]
    return data.qpos[qadr]


# InitMotion reset pose from rb_gui/rb_servo_gui/app.py:216 (deg)
RESET_LEFT_DEG = (259.0, 75.6, 129.5, -55.6, -131.2, -161.7)
RESET_RIGHT_DEG = (-253.7, -76.9, -127.6, 65.7, 143.7, 166.9)


def render(mode):
    # place boxes past the bolt area (away from robot), side by side:
    # top-view left = green (black bolts), top-view right = gray (gray bolts)
    totes = tote("box_green", -0.17, -0.80) + tote("box_gray", 0.17, -0.80, mat="box_gray")
    xml = SCENE_TMPL.format(H=H, W=W, totes=totes, bolts=make_bolts(mode))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for prefix, degs in (("L_", RESET_LEFT_DEG), ("R_", RESET_RIGHT_DEG)):
        for jn, deg in zip(JOINTS, degs):
            data.joint(prefix + jn).qpos[0] = np.deg2rad(deg)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=H, width=W)
    shots = {}
    for cam in ("overview", "top", "front", "L_wrist", "R_wrist"):
        r.update_scene(data, camera=cam)
        shots[cam] = r.render().copy()
        Image.fromarray(shots[cam]).save(f"out_{mode}_{cam}.png")
    r.close()
    return shots


if __name__ == "__main__":
    for mode in ("aligned", "random"):
        render(mode)
    print("rendered")
