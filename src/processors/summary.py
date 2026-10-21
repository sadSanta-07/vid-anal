import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.video_io import get_video_info, format_time
from src.utils.transcode import to_h264

OUT_DIR = Path("outputs/summaries")
OUT_W = 1280


def norm(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x / (np.percentile(x, 95) + 1e-9), 0, 1)


def build_windows(scenes, motion, tracks, fps, stride, clip, hop):
    eff_fps = fps / stride
    max_dur = scenes.duration_sec.max()
    rows = []
    for s in scenes.itertuples():
        a, b = s.start_sec, s.end_sec
        if b - a < 1.0:
            continue
        length = min(clip, b - a)
        starts = np.arange(a, b - length + 1e-6, hop)
        if len(starts) == 0:
            starts = [a]
        for t0 in starts:
            t1 = t0 + length
            m = motion[(motion.time_sec >= t0) & (motion.time_sec < t1)]
            tr = tracks[(tracks.time_sec >= t0) & (tracks.time_sec < t1)]
            rows.append(dict(scene_id=int(s.scene_id), t0=float(t0), t1=float(t1),
                             motion=float(m.object_motion.mean()) if len(m) else 0.0,
                             objects=len(tr) / max(length * eff_fps, 1),
                             scene_weight=s.duration_sec / max_dur))
    win = pd.DataFrame(rows)
    win["score"] = (0.50 * norm(win.motion) + 0.35 * norm(win.objects)
                    + 0.15 * win.scene_weight)
    return win


def select(win, target):
    chosen, total, per_scene = [], 0.0, {}
    cand = win.copy()

    def take(i, adj):
        nonlocal total, cand
        w = cand.loc[i]
        d = w.to_dict()
        d["adj_score"] = float(adj)
        chosen.append(d)
        total += w.t1 - w.t0
        sid = int(w.scene_id)
        per_scene[sid] = per_scene.get(sid, 0) + 1
        overlap = (cand.scene_id == sid) & (cand.t0 < w.t1) & (cand.t1 > w.t0)
        cand = cand[~overlap]

    # Pass 1: best window from each scene, strongest scenes first, while budget allows
    best = win.loc[win.groupby("scene_id").score.idxmax()].sort_values("score", ascending=False)
    for i, row in best.iterrows():
        if total + (row.t1 - row.t0) > target + 0.5:
            continue
        take(i, row.score)

    # Pass 2: fill the remaining budget greedily, discouraging repeats from one scene
    while total < target - 0.25 and not cand.empty:
        penalty = 0.6 ** cand.scene_id.map(per_scene).fillna(0)
        adj = cand.score * penalty
        take(adj.idxmax(), adj.max())

    return pd.DataFrame(chosen).sort_values("t0").reset_index(drop=True)

def write_summary(video_path, sel, info, out_raw):
    h = int(info.height * OUT_W / info.width)
    cap = cv2.VideoCapture(str(video_path))
    wr = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"),
                         info.fps, (OUT_W, h))
    for r in sel.itertuples():
        f0, f1 = int(round(r.t0 * info.fps)), int(round(r.t1 * info.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for f in range(f0, f1):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (OUT_W, h))
            bar = frame.copy()
            cv2.rectangle(bar, (0, h - 44), (OUT_W, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)
            cv2.putText(frame, f"Scene {r.scene_id}  |  source time {format_time(f / info.fps)}",
                        (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            wr.write(frame)
    cap.release()
    wr.release()


def plot_selection(win, sel, scenes, out_path):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    mid = (win.t0 + win.t1) / 2
    ax.plot(mid, win.score, color="#4C78A8", lw=1.6, label="Window importance score")
    for r in sel.itertuples():
        ax.axvspan(r.t0, r.t1, color="#2CA02C", alpha=0.35)
    ax.axvspan(0, 0, color="#2CA02C", alpha=0.35, label="Selected for summary")
    for s in scenes.start_sec.tolist()[1:]:
        ax.axvline(s, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_title("Summary clip selection (dashed = scene cuts)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/uploads/video.mp4")
    ap.add_argument("--scenes", default="outputs/scenes/scenes.csv")
    ap.add_argument("--motion", default="outputs/motion/motion_series.csv")
    ap.add_argument("--tracks", default="outputs/tracking/tracks.csv")
    ap.add_argument("--target", type=float, default=None, help="summary length (sec)")
    ap.add_argument("--clip", type=float, default=2.0, help="clip length (sec)")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2, help="stride used in tracking")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = get_video_info(a.video)
    scenes = pd.read_csv(a.scenes)
    motion = pd.read_csv(a.motion)
    tracks = pd.read_csv(a.tracks)
    n_scenes = len(scenes)
    target = a.target or float(np.clip(max(0.15 * info.duration_sec, 1.8 * n_scenes), 10, 60))

    win = build_windows(scenes, motion, tracks, info.fps, a.stride, a.clip, a.hop)
    sel = select(win, target)

    raw, final = OUT_DIR / "summary_raw.mp4", OUT_DIR / "summary.mp4"
    print(f"Writing summary (target {target:.0f}s, {len(sel)} clips)...")
    write_summary(a.video, sel, info, raw)
    if to_h264(raw, final):
        raw.unlink()
    else:
        raw.replace(final)
        print("  (kept mp4v encoding; may not play in browsers)")

    out = sel.copy()
    out["start"] = out.t0.apply(format_time)
    out["end"] = out.t1.apply(format_time)
    out.round(3).to_csv(OUT_DIR / "summary_clips.csv", index=False)
    plot_selection(win, sel, scenes, OUT_DIR / "summary_selection.png")

    total = float((sel.t1 - sel.t0).sum())
    print(out[["scene_id", "start", "end", "motion", "objects", "score"]]
          .round(3).to_string(index=False))
    print(f"\nSummary: {total:.1f}s from {info.duration_sec:.1f}s "
          f"({100 * total / info.duration_sec:.0f}% of the original)")
    print(f"Saved to {OUT_DIR}/")