from pathlib import Path

import numpy as np
import pandas as pd

TRK = Path("outputs/tracking")
MIN_DETS = 6


def merge_tracks(df, cuts_sec, max_gap=1.5, base_dist=100, speed_dist=100):
    info = {}
    for tid, d in df.groupby("track_id"):
        if len(d) < MIN_DETS:
            continue
        d = d.sort_values("time_sec")
        info[tid] = dict(cls=d.class_name.mode().iloc[0],
                         t0=d.time_sec.iloc[0], t1=d.time_sec.iloc[-1],
                         p0=d[["cx", "cy"]].iloc[:3].mean().to_numpy(),
                         p1=d[["cx", "cy"]].iloc[-3:].mean().to_numpy())
    parent = {t: t for t in info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for a, A in info.items():
        for b, B in info.items():
            if a == b or A["cls"] != B["cls"]:
                continue
            gap = B["t0"] - A["t1"]
            if not (0 < gap <= max_gap):
                continue
            if any(A["t1"] < c <= B["t0"] for c in cuts_sec):
                continue
            dist = float(np.linalg.norm(A["p1"] - B["p0"]))
            if dist <= base_dist + speed_dist * gap:
                pairs.append((dist, a, b))

    used_out, used_in = set(), set()
    for dist, a, b in sorted(pairs):
        if a in used_out or b in used_in:
            continue
        used_out.add(a)
        used_in.add(b)
        parent[find(b)] = find(a)
    return {t: find(t) for t in info}


if __name__ == "__main__":
    df = pd.read_csv(TRK / "tracks.csv")
    scenes = pd.read_csv("outputs/scenes/scenes.csv")
    cuts = scenes.start_sec.tolist()[1:]

    mapping = merge_tracks(df, cuts)
    df["merged_id"] = df.track_id.map(mapping)
    df = df.dropna(subset=["merged_id"]).astype({"merged_id": int})
    df.to_csv(TRK / "tracks_merged.csv", index=False)

    rows = []
    for mid, d in df.groupby("merged_id"):
        d = d.sort_values("time_sec")
        dist = float(np.hypot(d.cx.diff(), d.cy.diff()).sum())
        dur = float(d.time_sec.iloc[-1] - d.time_sec.iloc[0])
        rows.append(dict(object_id=mid, class_name=d.class_name.mode().iloc[0],
                         first_sec=d.time_sec.iloc[0], last_sec=d.time_sec.iloc[-1],
                         duration_sec=round(dur, 2),
                         fragments=d.track_id.nunique(), detections=len(d),
                         distance_px=round(dist, 1),
                         avg_speed_px_s=round(dist / dur, 1) if dur > 0 else 0.0))
    uniq = pd.DataFrame(rows)
    uniq.to_csv(TRK / "unique_objects.csv", index=False)

    raw = pd.read_csv(TRK / "track_summary.csv")
    print(f"Track IDs (>= {MIN_DETS} detections): {len(raw)}  ->  "
          f"unique objects after merging: {len(uniq)}\n")
    print(pd.DataFrame({"before": raw.class_name.value_counts(),
                        "after": uniq.class_name.value_counts()}).fillna(0).astype(int).to_string())
    print("\nMost-merged objects:")
    print(uniq.sort_values("fragments", ascending=False).head(5).to_string(index=False))