import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.video_io import get_video_info

OUT_DIR = Path("outputs/motion")
WIDTH = 320


def flow_to_bgr(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def analyze(video_path, scenes_csv, step=None):
    info = get_video_info(video_path)
    step = step or max(1, round(info.fps / 10))
    cuts = pd.read_csv(scenes_csv)["start_frame"].tolist()[1:]
    h = int(info.height * WIDTH / info.width)

    cap = cv2.VideoCapture(str(video_path))
    heat = np.zeros((h, WIDTH), np.float64)
    bg_acc = np.zeros((h, WIDTH, 3), np.float64)
    n_flow, n_bg = 0, 0
    rows, prev_gray, prev_idx = [], None, -1
    peak = dict(score=-1)

    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (WIDTH, h))
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            bg_acc += small
            n_bg += 1

            crossed_cut = prev_idx >= 0 and any(prev_idx < c <= idx for c in cuts)
            if prev_gray is not None and not crossed_cut:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                dt = (idx - prev_idx) / info.fps
                cam = np.median(flow.reshape(-1, 2), axis=0)
                resid = flow - cam
                mag = np.linalg.norm(resid, axis=2) / dt     # px/sec
                cam_mag = float(np.linalg.norm(cam) / dt)

                heat += mag
                n_flow += 1
                score = float(mag.mean())
                rows.append(dict(frame=idx, time_sec=round(idx / info.fps, 2),
                                 object_motion=score, camera_motion=cam_mag,
                                 moving_pct=float((mag > 15).mean() * 100)))
                if score > peak["score"]:
                    peak = dict(score=score, frame=small.copy(),
                                flow=flow - cam, t=idx / info.fps)
                if n_flow % 200 == 0:
                    print(f"  processed {n_flow} flow fields ({idx / info.fps:.0f}s)")
            prev_gray, prev_idx = gray, idx
        idx += 1
    cap.release()
    return rows, heat / max(n_flow, 1), (bg_acc / max(n_bg, 1)).astype(np.uint8), peak, cuts, info


def save_heatmap(heat, bg, out_path):
    hm = cv2.GaussianBlur(heat, (0, 0), 3)
    hm = np.clip(hm / (np.percentile(hm, 99) + 1e-9), 0, 1)
    hm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(bg, 0.5, hm, 0.5, 0)
    over = cv2.resize(over, (960, int(960 * over.shape[0] / over.shape[1])))
    cv2.imwrite(str(out_path), over)


def save_timeline(df, cuts, fps, out_path):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    t = df["time_sec"]
    ax.plot(t, df["camera_motion"].rolling(5, min_periods=1).mean(),
            color="#999999", lw=1.2, label="Camera motion")
    ax.plot(t, df["object_motion"].rolling(5, min_periods=1).mean(),
            color="#D62728", lw=1.8, label="Object motion")
    for c in cuts:
        ax.axvline(c / fps, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion (px/s, 320px-wide)")
    ax.set_title("Motion intensity over time (dashed = scene cuts)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_peak(peak, out_path):
    if peak["score"] < 0:
        return
    side = np.hstack([peak["frame"], flow_to_bgr(peak["flow"])])
    cv2.putText(side, f"Peak object motion  t={peak['t']:.1f}s", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    side = cv2.resize(side, (side.shape[1] * 2, side.shape[0] * 2))
    cv2.imwrite(str(out_path), side)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/uploads/video.mp4")
    ap.add_argument("--scenes", default="outputs/scenes/scenes.csv")
    ap.add_argument("--step", type=int, default=None, help="analyze every Nth frame")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing optical flow...")
    rows, heat, bg, peak, cuts, info = analyze(args.video, args.scenes, args.step)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "motion_series.csv", index=False)
    save_heatmap(heat, bg, OUT_DIR / "motion_heatmap.png")
    save_timeline(df, cuts, info.fps, OUT_DIR / "motion_timeline.png")
    save_peak(peak, OUT_DIR / "peak_flow.png")

    print(f"\n{len(df)} flow samples")
    print(f"Mean object motion : {df.object_motion.mean():.1f} px/s")
    print(f"Peak object motion : {df.object_motion.max():.1f} px/s at "
          f"{df.loc[df.object_motion.idxmax(), 'time_sec']}s")
    print(f"Mean camera motion : {df.camera_motion.mean():.1f} px/s")
    print(f"Saved to {OUT_DIR}/")