import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scenedetect import detect, ContentDetector, AdaptiveDetector

from src.utils.video_io import get_video_info, format_time

OUT_DIR = Path("outputs/scenes")


def detect_scenes(video_path, method="content", threshold=None, min_scene_len_sec=1.0):
    info = get_video_info(video_path)
    min_len = max(1, int(min_scene_len_sec * info.fps))

    if method == "adaptive":
        detector = AdaptiveDetector(
            adaptive_threshold=threshold or 3.0, min_scene_len=min_len
        )
    else:
        detector = ContentDetector(
            threshold=threshold or 27.0, min_scene_len=min_len
        )

    scene_list = detect(str(video_path), detector)

    rows = []
    if not scene_list:  # no cuts found: whole video is one scene
        rows.append(dict(scene_id=1, start_frame=0, end_frame=info.frame_count,
                         start_sec=0.0, end_sec=info.duration_sec))
    else:
        for i, (start, end) in enumerate(scene_list, 1):
            rows.append(dict(scene_id=i,
                             start_frame=start.frame_num,
                             end_frame=end.frame_num,
                             start_sec=round(start.seconds, 2),
                             end_sec=round(end.seconds, 2)))

    df = pd.DataFrame(rows)
    df["duration_sec"] = (df["end_sec"] - df["start_sec"]).round(2)
    df["start_time"] = df["start_sec"].apply(format_time)
    df["end_time"] = df["end_sec"].apply(format_time)
    return df, info


def plot_timeline(df, total_sec, out_path):
    fig, ax = plt.subplots(figsize=(12, 2.4))
    colors = ["#4C78A8", "#F58518"]
    for i, r in df.iterrows():
        ax.barh(0, r.duration_sec, left=r.start_sec, height=0.6,
                color=colors[i % 2], edgecolor="white")
        if r.duration_sec > total_sec * 0.015:
            ax.text(r.start_sec + r.duration_sec / 2, 0, str(int(r.scene_id)),
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, total_sec)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Scene timeline: {len(df)} scenes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/uploads/video.mp4")
    ap.add_argument("--method", choices=["content", "adaptive"], default="content")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_len", type=float, default=1.0, help="min scene length (sec)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, info = detect_scenes(args.video, args.method, args.threshold, args.min_len)
    df.to_csv(OUT_DIR / "scenes.csv", index=False)
    plot_timeline(df, info.duration_sec, OUT_DIR / "scene_timeline.png")

    print(df[["scene_id", "start_time", "end_time", "duration_sec"]].to_string(index=False))
    print(f"\n{len(df)} scenes found. Saved to {OUT_DIR}/")