import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.utils.video_io import get_video_info

OUT_DIR = Path("outputs/tracking")
WIDTH = 960
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
        7: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}
TRAIL_LEN = 30
MIN_DETS = 6


def color_for(gid):
    hsv = np.uint8([[[(gid * 37) % 180, 220, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def reset_trackers(model):
    pred = getattr(model, "predictor", None)
    for t in getattr(pred, "trackers", []) or []:
        t.reset()


def run_tracking(video_path, scenes_csv, model_name, stride, conf, device):
    info = get_video_info(video_path)
    starts = pd.read_csv(scenes_csv)["start_frame"].tolist()
    h = int(info.height * WIDTH / info.width)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(OUT_DIR / "tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             info.fps / stride, (WIDTH, h))
    rows, sampled = [], []
    trails = defaultdict(list)
    bg_acc, n_bg = np.zeros((h, WIDTH, 3), np.float64), 0
    offset = max_gid = 0
    prev_idx, idx, n = -1, 0, 0

    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = cv2.resize(frame, (WIDTH, h))
            bg_acc += frame
            n_bg += 1
            sampled.append(idx)

            if any(prev_idx < s <= idx for s in starts):   # new scene
                offset, trails = max_gid, defaultdict(list)
                reset_trackers(model)

            res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=conf, imgsz=640, classes=list(KEEP),
                              device=device, verbose=False)[0]
            vis = frame.copy()
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                cls = boxes.cls.int().cpu().numpy()
                cfs = boxes.conf.cpu().numpy()
                for b, tid, c, cf in zip(xyxy, ids, cls, cfs):
                    gid = int(tid) + offset
                    max_gid = max(max_gid, gid)
                    x1, y1, x2, y2 = b
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    rows.append(dict(frame=idx, time_sec=round(idx / info.fps, 2),
                                     track_id=gid, class_name=KEEP[int(c)],
                                     conf=round(float(cf), 3), x1=x1, y1=y1,
                                     x2=x2, y2=y2, cx=cx, cy=cy, foot_y=y2))
                    trails[gid].append((int(cx), int(cy)))
                    col = color_for(gid)
                    pts = np.array(trails[gid][-TRAIL_LEN:], np.int32)
                    if len(pts) > 1:
                        cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    label = f"{KEEP[int(c)]} #{gid} {cf:.2f}"
                    cv2.putText(vis, label, (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            cv2.putText(vis, f"t={idx / info.fps:6.1f}s  tracked={0 if boxes is None or boxes.id is None else len(boxes.id)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(vis)

            prev_idx = idx
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames processed ({idx / info.fps:.0f}s of {info.duration_sec:.0f}s)")
        idx += 1
    cap.release()
    writer.release()
    bg = (bg_acc / max(n_bg, 1)).astype(np.uint8)
    return pd.DataFrame(rows), bg, sampled, starts, info


def summarize(df, fps):
    g = df.groupby("track_id")
    out = []
    for tid, d in g:
        if len(d) < MIN_DETS:
            continue
        d = d.sort_values("frame")
        dist = float(np.hypot(d.cx.diff(), d.cy.diff()).sum())
        dur = float(d.time_sec.iloc[-1] - d.time_sec.iloc[0])
        out.append(dict(track_id=tid, class_name=d.class_name.mode().iloc[0],
                        first_sec=d.time_sec.iloc[0], last_sec=d.time_sec.iloc[-1],
                        duration_sec=round(dur, 2), detections=len(d),
                        distance_px=round(dist, 1),
                        avg_speed_px_s=round(dist / dur, 1) if dur > 0 else 0.0))
    return pd.DataFrame(out)


def plot_trajectories(df, summ, bg, out_path):
    canvas = (bg * 0.45).astype(np.uint8)
    keep = set(summ.track_id)
    for tid, d in df[df.track_id.isin(keep)].groupby("track_id"):
        d = d.sort_values("frame")
        pts = np.array(list(zip(d.cx, d.cy)), np.int32)
        col = color_for(int(tid))
        cv2.polylines(canvas, [pts], False, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[0]), 5, col, -1)
        cv2.putText(canvas, f"#{int(tid)}", tuple(pts[0] + [6, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def plot_heatmap(df, bg, out_path):
    h, w = bg.shape[:2]
    heat = np.zeros((h, w), np.float32)
    xs = np.clip(df.cx.astype(int), 0, w - 1)
    ys = np.clip(df.foot_y.astype(int), 0, h - 1)
    np.add.at(heat, (ys, xs), 1)
    heat = cv2.GaussianBlur(heat, (0, 0), 18)
    heat = np.clip(heat / (np.percentile(heat[heat > 0], 99) + 1e-9), 0, 1) if heat.max() > 0 else heat
    hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0.03)[..., None]
    over = np.where(mask, cv2.addWeighted(bg, 0.45, hm, 0.55, 0), bg)
    cv2.imwrite(str(out_path), over.astype(np.uint8))


def plot_counts(df, sampled, starts, fps, out_path):
    counts = (df.groupby(["frame", "class_name"]).size().unstack(fill_value=0)
              .reindex(sampled, fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    t = np.array(sampled) / fps
    for cname in counts.columns:
        ax.plot(t, counts[cname].rolling(5, min_periods=1).mean(), lw=1.8, label=cname)
    for s in starts[1:]:
        ax.axvline(s / fps, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Objects in frame")
    ax.set_title("Tracked objects over time (dashed = scene cuts)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/uploads/video.mp4")
    ap.add_argument("--scenes", default="outputs/scenes/scenes.csv")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None, help="e.g. cpu, 0")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Tracking with {a.model} (stride {a.stride})...")
    df, bg, sampled, starts, info = run_tracking(
        a.video, a.scenes, a.model, a.stride, a.conf, a.device)

    if df.empty:
        print("No tracked objects found. Try --conf 0.2")
        raise SystemExit
    df.to_csv(OUT_DIR / "tracks.csv", index=False)
    summ = summarize(df, info.fps)
    summ.to_csv(OUT_DIR / "track_summary.csv", index=False)
    plot_trajectories(df, summ, bg, OUT_DIR / "trajectories.png")
    plot_heatmap(df, bg, OUT_DIR / "path_heatmap.png")
    plot_counts(df, sampled, starts, info.fps, OUT_DIR / "object_counts.png")

    print("\nUnique tracked objects (tracks with >= %d detections):" % MIN_DETS)
    print(summ.class_name.value_counts().to_string())
    print("\nLongest tracks:")
    print(summ.sort_values("duration_sec", ascending=False).head(8).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")