#!/usr/bin/env python3
"""T1 report: decompose closed-loop failure into ambiguity vs precision vs commitment.

The question this answers is the one the whole "multimodal task" framing rests on: when the
policy fails to pick up a bolt, is it because several bolts were near-tied (a genuine
multimodal conditional the head cannot express) or because the aim was simply off (precision)?

Only simulation can answer it, because only simulation knows every bolt's ground-truth pose.

The decisive plot is grasp-success-rate versus AMBIGUITY MARGIN (distance to the 2nd nearest
bolt minus distance to the nearest, at the moment the close is commanded):
  - flat curve  -> ambiguity is NOT the loss channel; fix precision/grip, not the action head
  - falls at small margin -> the tie IS the loss channel; mode commitment / head expressiveness

Supporting signals:
  between-candidates   aim sits on the segment joining the two nearest bolts, off both
                       -> the mode-AVERAGING signature (what an L2 head does to a bimodal target)
  switches_1s          target index changed in the second before the close
                       -> the mode-SWITCHING signature (chunk-boundary bolt hopping)
  wrong-colour-nearest only meaningful in --layout random; the nearest bolt is the wrong colour,
                       which the identifiability audit says is unlearnable from the current data

Usage:  t1_report.py outputs/eval/summary_<tag>_<layout>.json [more...]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

# Margin buckets in mm. The bottom bucket is the near-tie regime: two bolts within 20 mm of
# equidistant, which is inside the M12 head diameter (18.4 mm) -- a real coin flip.
BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 1e9)]


def scene_xy(rec: dict, layout: str) -> np.ndarray | None:
    """Initial bolt XY (mm) for one episode: from the record if the run stored it, else
    regenerated from (layout, seed) with the very function the scene builder used."""
    if rec.get("bolts_xy0"):
        return np.asarray(rec["bolts_xy0"], dtype=float)[:, :2] * 1e3
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bs", pathlib.Path(__file__).with_name("build_scene.py"))
        bs = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(bs)
        except SystemExit:
            pass
        n_per = rec.get("bolts", 20) // 2
        poses = bs.bolt_poses(layout, n_per, rec["seed"])
        return np.array([[x, y] for (_c, x, y, _yaw) in poses]) * 1e3
    except Exception:
        return None


def isotropic_null(docs_records, layout: str, r_obs: np.ndarray,
                   draws: int = 20000) -> tuple[float, float] | None:
    """P(theta<60) for an aim error that is ISOTROPIC around the intended bolt, in the same
    scenes and with the same error-magnitude distribution as observed.

    Why this and not "uniform theta -> 33.3%": b1 is *defined* as the bolt nearest the TCP and
    b2 as the second nearest, and that selection alone concentrates theta. Measured on this
    project's scenes the selection-only null is ~42%, not 33% -- reporting against 33% turned an
    ordinary isotropic scatter into a spurious 2.6-sigma "pull toward the second candidate".
    """
    scenes = [s for s in (scene_xy(r, layout) for r in docs_records) if s is not None]
    if not scenes or not len(r_obs):
        return None
    rng = np.random.default_rng(0)
    th = []
    per = max(1, draws // max(1, len(scenes)))
    for P in scenes:
        for _ in range(per):
            i = rng.integers(len(P))
            a = rng.uniform(0, 2 * np.pi)
            tcp = P[i] + rng.choice(r_obs) * np.array([np.cos(a), np.sin(a)])
            o = np.argsort(np.linalg.norm(P - tcp, axis=1))
            b1, b2 = P[o[0]], P[o[1]]
            u, w = b2 - b1, tcp - b1
            gap = float(np.linalg.norm(u))
            if gap < 1e-9:
                continue
            th.append(np.degrees(np.arctan2(abs(u[0] * w[1] - u[1] * w[0]) / gap,
                                            float(np.dot(w, u) / gap))))
    if not th:
        return None
    th = np.asarray(th)
    return float(np.mean(th < 60) * 100), float(np.median(th))


# Bolts that SETTLE inside a box were never picked. Runs before 2026-08-22 counted them; the
# frozen scene sets make the inflation a constant, so those runs stay comparable once it is
# subtracted rather than being thrown away. Keyed by the scene-states file the run named.
PRE_PLACED = {
    "assets/scene_states40_aligned.json": (3, 0),
    "assets/scene_states40_random.json": (9, 5),
}


def load(path: pathlib.Path) -> dict:
    d = json.loads(path.read_text())
    if "t1" not in d:
        raise SystemExit(f"{path.name}: no t1 block -- rerun with the instrumented eval script")
    if not d.get("pre_placed_excluded"):
        c, w = PRE_PLACED.get(str(d.get("scene_states") or ""), (0, 0))
        if c or w:
            d["total_correct"] -= c
            d["total_wrong"] -= w
            d["pre_placed_corrected"] = [c, w]
    return d


def report(d: dict, name: str) -> None:
    ev = [e for r in d["records"] for e in r["close_events"]]
    n_ep = d["episodes"]
    print(f"\n=== {name}  [{d.get('label')}  {d.get('policy')}  protocol={d.get('protocol')}  "
          f"layout={d['layout']}  rtc={d.get('rtc')}] ===")
    if d.get("pre_placed_corrected"):
        c, w = d["pre_placed_corrected"]
        print(f"  [corrected: -{c} correct / -{w} wrong bolts that started inside a box]")
    print(f"  placements {d['total_correct']} (wrong {d['total_wrong']}) | "
          f"episodes with >=1: {d['episodes_with_any_correct']}/{n_ep} | "
          f"closes {d['total_close']} -> grasps {d['total_grasp']} "
          f"({d['total_grasp']/max(1,d['total_close'])*100:.0f}%)")
    if not ev:
        print("  no close commands -- the policy never tried to grasp")
        return

    margin = np.array([e["margin"] for e in ev]) * 1e3
    grasped = np.array([bool(e["grasped"]) for e in ev])
    print("\n  grasp success vs AMBIGUITY MARGIN (d2nd - d1st at close):")
    print(f"    {'margin':>14}  {'n':>5}  {'grasped':>7}  rate")
    for lo, hi in BUCKETS:
        m = (margin >= lo) & (margin < hi)
        if not m.any():
            continue
        lab = f"{lo}-{hi:.0f} mm" if hi < 1e9 else f">{lo} mm"
        print(f"    {lab:>14}  {m.sum():5d}  {grasped[m].sum():7d}  "
              f"{grasped[m].mean()*100:5.1f}%")

    t_seg = np.array([e["t_seg"] for e in ev])
    perp = np.array([e["perp"] for e in ev]) * 1e3
    between = (t_seg > 0.15) & (t_seg < 0.85) & (perp < 40.0)
    sw = np.array([e["switches_1s"] for e in ev])
    lead = np.array([e["commit_lead_ticks"] for e in ev])
    dxy = np.array([e["dxy"] for e in ev]) * 1e3
    dz = np.array([e["dz"] for e in ev]) * 1e3
    wrong_col = np.array([not e["nearest_is_target_color"] for e in ev])

    # --- mode-average test, against the isotropic null ---------------------------------
    # `between_candidates` ALONE PROVES NOTHING. If the aim error were an isotropic blob around
    # the nearest bolt, a large share of events would still land "between" it and the second
    # candidate, because the error magnitude (~24 mm p50) is the same scale as the inter-bolt
    # gap (~40 mm). The test has to be DIRECTIONAL: theta = angle between the aim error
    # (tcp - b1) and the direction to the second candidate (b2 - b1). Under an isotropic error
    # theta is uniform on [0,180) deg, so P(theta < 60) = 1/3 exactly. Mode averaging pulls the
    # aim toward b2, i.e. concentrates theta near 0.
    # Caveat kept explicit: b1 is *defined* as the nearest bolt, so an error pointing at b2 can
    # relabel b1<->b2 and drop out of the sample. That truncation works AGAINST small theta,
    # which makes this test conservative -- a positive result is real, a null one is weaker.
    gap = np.array([e["pair_gap"] for e in ev])
    along = t_seg * gap * 1e3                      # mm along b1->b2
    theta = np.degrees(np.arctan2(perp, along))    # 0 = aimed straight at the 2nd candidate
    near60 = theta < 60.0
    n = len(ev)
    r_err = np.hypot([e["dx"] for e in ev], [e["dy"] for e in ev]) * 1e3
    null = isotropic_null(d["records"], d["layout"], r_err)
    print("\n  MODE-AVERAGE test (directional, vs GEOMETRY-MATCHED isotropic null):")
    print(f"    theta = angle(aim error, direction to 2nd candidate)")
    if null is None:
        print("    [!] could not build the null (no scene geometry) -- do NOT use 33.3% instead")
    else:
        p_null, med_null = null
        se = (p_null / 100 * (1 - p_null / 100) / n) ** 0.5 * 100
        diff = near60.mean() * 100 - p_null
        verdict = ("MODE PULL toward the 2nd candidate" if diff > 2 * se else
                   "consistent with isotropic aim error -- no mode averaging")
        print(f"    observed P(theta<60) = {near60.mean()*100:5.1f}%   median {np.median(theta):5.1f} deg")
        print(f"    null     P(theta<60) = {p_null:5.1f}%   median {med_null:5.1f} deg   "
              f"(same scenes, same |error| distribution)")
        print(f"    diff {diff:+5.1f} pp = {abs(diff)/se:.1f} s.e.  ->  {verdict}")
        print(f"    (the naive 'uniform theta = 33.3%' null is WRONG here: nearest/2nd-nearest "
              f"selection alone gives {p_null:.0f}%)")
    # Mode averaging should bite hardest where the two candidates are closest to tied.
    print("    by ambiguity margin:")
    for lo, hi in BUCKETS:
        m = (margin >= lo) & (margin < hi)
        if m.sum() >= 5:
            lab = f"{lo}-{hi:.0f} mm" if hi < 1e9 else f">{lo} mm"
            print(f"      {lab:>12}  n={m.sum():4d}  P(theta<60)={near60[m].mean()*100:5.1f}%  "
                  f"aim err p50 {np.median(dxy[m]):5.1f} mm  grasp {grasped[m].mean()*100:5.1f}%")

    print("\n  other signatures (share of close commands):")
    print(f"    (raw)         aim between two candidates : {between.sum():4d}/{len(ev)} "
          f"({between.mean()*100:.1f}%)  -- uncalibrated, read the directional test above")
    print(f"    mode-SWITCH   target changed within 1 s  : {(sw > 0).sum():4d}/{len(ev)} "
          f"({(sw > 0).mean()*100:.1f}%)   grasp rate there {grasped[sw > 0].mean()*100 if (sw > 0).any() else float('nan'):.1f}%")
    print(f"    PRECISION     |dxy| p50 {np.median(dxy):5.1f} mm   dz p50 {np.median(dz):+6.1f} mm "
          f"(grasped: {np.median(dxy[grasped]) if grasped.any() else float('nan'):5.1f} / "
          f"{np.median(dz[grasped]) if grasped.any() else float('nan'):+6.1f})")
    print(f"    COMMITMENT    lead ticks p50 {np.median(lead):.0f} "
          f"({np.median(lead)/30.0:.2f} s of stable intent before the close)")
    if d["layout"] == "random":
        print(f"    COLOUR        nearest bolt is wrong colour: {wrong_col.sum()}/{len(ev)} "
              f"({wrong_col.mean()*100:.1f}%)")

    # ---- throughput: a success that took 25 s is not the same result as one that took 6 s ----
    pl = [e for r in d["records"] for e in r.get("placements", [])]
    if pl:
        first = [min(e["t"] for e in r["placements"]) for r in d["records"] if r.get("placements")]
        gaps = []
        for r in d["records"]:
            ts = sorted(e["t"] for e in r.get("placements", []))
            gaps += [b - a for a, b in zip(ts, ts[1:])]
        tr = [e["transport_s"] for e in pl if e.get("transport_s") is not None]
        att = {s_: sum(1 for e in pl if e.get("side") == s_) for s_ in ("left", "right")}
        print("\n  THROUGHPUT (placement timing):")
        print(f"    placements {len(pl)} in {len(d['records'])} x {d.get('episode_sec', 30)} s"
              f"  -> {len(pl)/(len(d['records'])*float(d.get('episode_sec', 30))/60.0):.2f} /min")
        print(f"    time to FIRST placement : p50 {np.median(first):5.1f} s  "
              f"(episodes with any: {len(first)}/{len(d['records'])})")
        if gaps:
            print(f"    interval between placements (cycle): p50 {np.median(gaps):5.1f} s  n={len(gaps)}")
        if tr:
            print(f"    grasp -> place transport: p50 {np.median(tr):5.1f} s  n={len(tr)}")
        print(f"    by arm: left {att['left']}  right {att['right']}  "
              f"unattributed {len(pl) - att['left'] - att['right']}")

    crowd = [r["same_color_crowding_m"] for r in d["records"] if r.get("same_color_crowding_m")]
    if crowd:
        for col in sorted(crowd[0]):
            v = np.array([c[col] for c in crowd if col in c]) * 1e3
            print(f"  scene: same-colour nearest-neighbour gap [{col}] p50 {np.median(v):.0f} mm")
    tsw = {s: np.array([r["target_switches"][s] for r in d["records"]])
           for s in ("left", "right")}
    # Episode-level switch counts include TRANSPORT, where the chunk endpoint sits over the
    # box and "nearest bolt" is meaningless noise. The approach-phase signal is switches_1s
    # measured at the close command; this line is context, not evidence.
    print(f"  target switches per episode (incl. transport): left p50 {np.median(tsw['left']):.0f}  "
          f"right p50 {np.median(tsw['right']):.0f}")


def table(docs: list[tuple[str, dict]]) -> None:
    """One row per arm. This is the T0 authority view: rank arms in sim, compare the order
    against the real-robot labels we already hold for several of these checkpoints."""
    print(f"\n{'arm':22} {'layout':8} {'place':>5} {'ep+':>5} {'close':>6} {'grasp':>6} "
          f"{'g%':>5} {'dxy':>6} {'dz':>7} {'t1st':>6} {'cyc':>5} {'trans':>6}")
    print("-" * 104)
    for name, d in docs:
        ev = [e for r in d["records"] for e in r["close_events"]]
        t1 = d["t1"]
        nc, ng = d["total_close"], d["total_grasp"]
        f = lambda v, w=6, p=1: (f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * w)
        pl = [e for r in d["records"] for e in r.get("placements", [])]
        first = [min(e["t"] for e in r["placements"]) for r in d["records"] if r.get("placements")]
        gaps = []
        for r in d["records"]:
            ts = sorted(e["t"] for e in r.get("placements", []))
            gaps += [b - a for a, b in zip(ts, ts[1:])]
        tr = [e["transport_s"] for e in pl if e.get("transport_s") is not None]
        med = lambda v, w=6: (f"{np.median(v):{w}.1f}" if len(v) else " " * w)
        print(f"{(d.get('label') or name)[:22]:22} {d['layout']:8} "
              f"{d['total_correct']:5d} {d['episodes_with_any_correct']:3d}/{d['episodes']:<2d}"
              f"{nc:6d} {ng:6d} {ng/max(1,nc)*100:5.0f} "
              f"{f(t1['dxy_p50_mm'])} {f(t1['dz_p50_mm'],7)} "
              f"{med(first)} {med(gaps,5)} {med(tr)}")
    print("\nplace=bolts placed correctly  ep+=episodes with >=1  g%=grasps per close command")
    print("dxy/dz = p50 mm at the close command")
    print("t1st = s to first placement; cyc = s between placements; trans = s from grasp to box")

    # Grasp rate vs margin, all arms side by side -- the T1 decision plot in text form.
    print(f"\ngrasp success vs ambiguity margin\n{'arm':22}" +
          "".join(f"{(f'{lo}-{hi:.0f}' if hi < 1e9 else f'>{lo}'):>14}" for lo, hi in BUCKETS))
    print("-" * (22 + 14 * len(BUCKETS)))
    for name, d in docs:
        ev = [e for r in d["records"] for e in r["close_events"]]
        margin = np.array([e["margin"] for e in ev]) * 1e3
        grasped = np.array([bool(e["grasped"]) for e in ev])
        cells = []
        for lo, hi in BUCKETS:
            m = (margin >= lo) & (margin < hi) if len(ev) else np.zeros(0, bool)
            cells.append(f"{grasped[m].mean()*100:5.0f}% n={m.sum():<4d}" if m.any() else
                         f"{'-':>14}")
        print(f"{(d.get('label') or name)[:22]:22}" + "".join(cells))


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--table"]
    as_table = "--table" in sys.argv[1:]
    paths = [pathlib.Path(p) for p in argv]
    if not paths:
        raise SystemExit(__doc__)
    docs = [(p.name, load(p)) for p in paths]
    if as_table:
        table(docs)
    else:
        for name, d in docs:
            report(d, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
