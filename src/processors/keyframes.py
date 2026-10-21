import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.video_io import get_video_info, format_time

OUT_DIR = Path("outputs/keyframes")


def extract_features(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, int(h * 320 / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                        [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    return sharp, hist.astype(np.float32)


def choose_sample_indices(scenes, fps, step_sec, margin_sec):
    step = max(1, int(step_sec * fps))
    margin = int(margin_sec * fps)
    wanted = {}
    for r in scenes.itertuples():
        length = r.end_frame - r.start_frame
        m = margin if length > 4 * margin else 0
        for idx in range(r.start_frame + m, r.end_frame - m, step):
            wanted[idx] = r.scene_id
        if not any(v == r.scene_id for v in wanted.values()):
            wanted[(r.start_frame + r.end_frame) // 2] = r.scene_id
    return wanted


def scan_video(video_path, wanted):
    cap = cv2.VideoCapture(str(video_path))
    records, idx, last = [], 0, max(wanted)
    while idx <= last:
        if not cap.grab():
            break
        if idx in wanted:
            ok, frame = cap.retrieve()
            if ok:
                sharp, hist = extract_features(frame)
                records.append(dict(frame=idx, scene_id=wanted[idx],
                                    sharp=sharp, hist=hist))
        idx += 1
    cap.release()
    return records


def pick_keyframes(recs, duration_sec):
    k = 1 if duration_sec < 5 else 2 if duration_sec < 12 else 3
    k = min(k, len(recs))
    H = np.array([r["hist"] for r in recs], dtype=np.float32)
    sharp = np.array([r["sharp"] for r in recs])
    sharp_n = sharp / (sharp.max() + 1e-9)

    if k == 1:
        labels = np.zeros(len(recs), dtype=int)
        centers = H.mean(axis=0, keepdims=True)
    else:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        _, lab, centers = cv2.kmeans(H, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        labels = lab.ravel()

    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(H[members] - centers[c], axis=1)
        d_n = d / (d.max() + 1e-9)
        score = 0.6 * sharp_n[members] + 0.4 * (1 - d_n)
        best = members[int(np.argmax(score))]
        chosen.append((recs[best], float(score.max())))
    chosen.sort(key=lambda x: x[0]["frame"])
    return chosen


def contact_sheet(items, out_path, cols=4, thumb_w=320):
    thumb_h = thumb_w * 9 // 16
    cell_h = thumb_h + 28
    rows = (len(items) + cols - 1) // cols
    sheet = np.zeros((rows * cell_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, (img, caption) in enumerate(items):
        r, c = divmod(i, cols)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        y, x = r * cell_h, c * thumb_w
        sheet[y:y + thumb_h, x:x + thumb_w] = thumb
        cv2.putText(sheet, caption, (x + 6, y + thumb_h + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), sheet)


def run(video_path, scenes_csv, step_sec=0.5, margin_sec=0.3):
    info = get_video_info(video_path)
    scenes = pd.read_csv(scenes_csv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUT_DIR.glob("*.jpg"):
        f.unlink()

    wanted = choose_sample_indices(scenes, info.fps, step_sec, margin_sec)
    print(f"Scoring {len(wanted)} sampled frames across {len(scenes)} scenes...")
    records = scan_video(video_path, wanted)

    cap = cv2.VideoCapture(str(video_path))
    rows, sheet_items = [], []
    for r in scenes.itertuples():
        recs = [x for x in records if x["scene_id"] == r.scene_id]
        if not recs:
            continue
        for n, (rec, score) in enumerate(pick_keyframes(recs, r.duration_sec), 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            t = rec["frame"] / info.fps
            name = f"scene{r.scene_id:02d}_kf{n}_t{t:07.2f}s.jpg"
            cv2.imwrite(str(OUT_DIR / name), frame)
            rows.append(dict(scene_id=r.scene_id, keyframe=n, frame=rec["frame"],
                             time_sec=round(t, 2), time=format_time(t),
                             sharpness=round(rec["sharp"], 1), score=round(score, 3),
                             file=name))
            sheet_items.append((frame, f"S{r.scene_id} #{n}  {format_time(t)}"))
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "keyframes.csv", index=False)
    contact_sheet(sheet_items, OUT_DIR / "contact_sheet.jpg")
    print(df[["scene_id", "keyframe", "time", "sharpness", "score"]].to_string(index=False))
    print(f"\n{len(df)} keyframes saved to {OUT_DIR}/ (see contact_sheet.jpg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/uploads/video.mp4")
    ap.add_argument("--scenes", default="outputs/scenes/scenes.csv")
    ap.add_argument("--step", type=float, default=0.5, help="sample interval (sec)")
    args = ap.parse_args()
    run(args.video, args.scenes, args.step)