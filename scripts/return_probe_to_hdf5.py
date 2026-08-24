#!/usr/bin/env python3
"""Convert a return-episode probe dump into per-episode HDF5 + preview mp4s.

Encoding matches the training set `pika_umi_video_train_tcp_gripabs_velgrip_k1` exactly
(verified against the parquet: state[t+1] == action[t] at r=0.999):
  state[t]  = [ R_ALIGN @ (R_{t-1}^T (p_t - p_{t-1})),  R_ALIGN @ rotvec(R_{t-1}^T R_t),
                grip_t/100 ] per arm, left block then right; t=0 velocities are zero.
  action[t] = same delta taken t -> t+1, grip_{t+1}/100 (absolute).
R_ALIGN = diag(-1,-1,1) (its own inverse), the same matrix the eval loop applies when
EXECUTING policy actions -- so a policy trained on these rows drives the sim identically.
NOT wired into any training config; inspection only, per operator decision.
"""
from __future__ import annotations

import json
import pathlib
import sys

import h5py
import imageio.v2 as iio
import numpy as np

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/plaif/probe_return")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                   "/home/plaif/workspace/simulation/episodes")
R_ALIGN = np.diag([-1.0, -1.0, 1.0])


def rv2m(r):
    r = np.asarray(r, float); th = np.linalg.norm(r)
    if th < 1e-12: return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def m2rv(R):
    c = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2 * np.sin(th)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jl = sorted(SRC.glob("probe_*.jsonl"))
    assert jl, f"no dump in {SRC}"
    recs = [json.loads(l) for f in jl for l in f.read_text().splitlines()]
    eps = sorted({r["ep"] for r in recs})
    print(f"{len(recs)} ticks over {len(eps)} episodes")
    for ep in eps:
        rows = sorted((r for r in recs if r["ep"] == ep), key=lambda r: r["tick"])
        # keep only ticks where BOTH arms were sampled (they always are, but be safe)
        rows = [r for r in rows if "left" in r and "right" in r]
        T = len(rows)
        if T < 30:
            print(f"  ep {ep}: only {T} ticks, skipped"); continue
        P = {s: np.array([r[s]["tcp"] for r in rows]) for s in ("left", "right")}
        R = {s: [rv2m(r[s]["tcp_rotvec"]) for r in rows] for s in ("left", "right")}
        G = {s: np.array([r[s]["grip"] for r in rows]) / 100.0 for s in ("left", "right")}
        state = np.zeros((T - 1, 14), np.float32)
        act = np.zeros((T - 1, 14), np.float32)
        for bi, s in enumerate(("left", "right")):
            o = bi * 7
            for t in range(T - 1):
                if t > 0:
                    dp = R[s][t - 1].T @ (P[s][t] - P[s][t - 1])
                    dr = m2rv(R[s][t - 1].T @ R[s][t])
                    state[t, o:o + 3] = R_ALIGN @ dp
                    state[t, o + 3:o + 6] = R_ALIGN @ dr
                state[t, o + 6] = G[s][t]
                dp = R[s][t].T @ (P[s][t + 1] - P[s][t])
                dr = m2rv(R[s][t].T @ R[s][t + 1])
                act[t, o:o + 3] = R_ALIGN @ dp
                act[t, o + 3:o + 6] = R_ALIGN @ dr
                act[t, o + 6] = G[s][t + 1]
        imgs = {}
        for s in ("left", "right"):
            frames = [iio.imread(SRC / rows[t][s]["img"]) for t in range(T - 1)]
            imgs[s] = np.stack(frames).astype(np.uint8)[..., :3]
        f = OUT / f"return_ep{ep:04d}.hdf5"
        with h5py.File(f, "w") as h:
            h.create_dataset("observation/left_wrist_0_rgb", data=imgs["left"],
                             compression="gzip", compression_opts=2, chunks=(1, 480, 640, 3))
            h.create_dataset("observation/right_wrist_0_rgb", data=imgs["right"],
                             compression="gzip", compression_opts=2, chunks=(1, 480, 640, 3))
            h.create_dataset("observation/state", data=state)
            h.create_dataset("actions", data=act)
            h.attrs["fps"] = 30.0
            h.attrs["source"] = "isaac_oracle_return_v1"
            h.attrs["seed"] = rows[0].get("seed", -1)
            h.attrs["encoding"] = "velgrip_k1 (state[t]=delta(t-1,t)+grip_t, action[t]=delta(t,t+1)+grip_{t+1}, R_ALIGN=diag(-1,-1,1))"
        pv = OUT / f"return_ep{ep:04d}.mp4"
        with iio.get_writer(pv, fps=30, quality=5) as w:
            for t in range(0, T - 1, 1):
                w.append_data(np.concatenate([imgs["left"][t], imgs["right"][t]], axis=1))
        mag = np.linalg.norm(act[:, 0:3], axis=1) * 1000
        rot = np.degrees(np.linalg.norm(act[:, 3:6], axis=1))
        print(f"  ep {ep}: T={T-1}  pos p50/p90 {np.percentile(mag,50):.1f}/{np.percentile(mag,90):.1f}mm"
              f"  rot p90 {np.percentile(rot,90):.2f}deg  -> {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
