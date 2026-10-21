import subprocess
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.video_io import get_video_info, format_time
from src.utils.transcode import to_h264

ROOT = Path(__file__).parent
UPLOADS = ROOT / "data" / "uploads"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

st.set_page_config(page_title="Intelligent Video Analysis", layout="wide")


# ---------- helpers ----------
@st.cache_data
def _read_csv(path, mtime):
    return pd.read_csv(path)


def load_csv(path):
    path = Path(path)
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def show_image(path, caption=None):
    path = Path(path)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Not generated yet: {path.name}")


def h264_copy(src):
    src = Path(src)
    dst = src.with_name(src.stem + "_h264.mp4")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        with st.spinner("Encoding video for browser playback..."):
            if not to_h264(src, dst):
                return src
    return dst


def show_video(path):
    path = Path(path)
    if path.exists():
        st.video(str(h264_copy(path)))
    else:
        st.info(f"Not generated yet: {path.name}")


@st.cache_data
def read_frame(video, t_sec, mtime):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (960, int(h * 960 / w)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_pipeline(video, p):
    v = str(video)
    steps = [
        ("Scene boundaries", ["src.processors.scene_detection", "--video", v,
                              "--threshold", str(p["thr"]), "--min_len", str(p["min_len"])]),
        ("Keyframes", ["src.processors.keyframes", "--video", v, "--step", str(p["kf_step"])]),
        ("Motion map", ["src.processors.motion", "--video", v]),
        ("Object tracking", ["src.tracking.tracker", "--video", v, "--model", p["model"],
                             "--stride", str(p["stride"]), "--conf", str(p["conf"])]),
        ("Merge fragmented tracks", ["src.tracking.cleanup"]),
        ("Summary video", ["src.processors.summary", "--video", v, "--stride", str(p["stride"])]
         + (["--target", str(p["target"])] if p["target"] else [])),
    ]
    with st.status("Running pipeline...", expanded=True) as status:
        for label, args in steps:
            st.write(f"Running: {label}...")
            t0 = time.time()
            proc = subprocess.run([sys.executable, "-m", *args], cwd=ROOT,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                status.update(label=f"Failed at: {label}", state="error")
                st.code((proc.stdout + proc.stderr)[-3000:])
                return False
            st.write(f"Done: {label} ({time.time() - t0:.0f}s)")
        status.update(label="Pipeline complete", state="complete")
    return True


# ---------- sidebar ----------
st.sidebar.title("Video Analysis")
UPLOADS.mkdir(parents=True, exist_ok=True)

up = st.sidebar.file_uploader("Upload a video", type=[e[1:] for e in VIDEO_EXT])
if up is not None:
    (UPLOADS / up.name).write_bytes(up.getbuffer())
    st.sidebar.success(f"Saved {up.name}")

files = sorted(f.name for f in UPLOADS.iterdir() if f.suffix.lower() in VIDEO_EXT)
if not files:
    st.title("Intelligent Video Analysis")
    st.warning("Put a video in data/uploads/ or upload one from the sidebar.")
    st.stop()

result_dirs = sorted(d.name for d in ROOT.glob("outputs*") if d.is_dir()) or ["outputs"]
OUT = ROOT / st.sidebar.selectbox("Results folder", result_dirs)
src_file = OUT / "source.txt"
src_name = src_file.read_text().strip() if src_file.exists() else None
default = files.index(src_name) if src_name in files else (
    files.index("video.mp4") if "video.mp4" in files else 0)
video = UPLOADS / st.sidebar.selectbox("Video", files, index=default, key=f"video_{OUT.name}")

with st.sidebar.expander("Pipeline settings"):
    p = dict(
        thr=st.slider("Scene cut threshold", 10.0, 50.0, 27.0, 1.0),
        min_len=st.slider("Min scene length (s)", 0.5, 5.0, 1.0, 0.5),
        kf_step=st.slider("Keyframe sample step (s)", 0.25, 2.0, 0.5, 0.25),
        model=st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=1),
        stride=st.slider("Tracking frame stride", 1, 6, 2),
        conf=st.slider("Detection confidence", 0.1, 0.8, 0.3, 0.05),
    )
    auto = st.checkbox("Auto summary length", value=True)
    p["target"] = None if auto else st.slider("Summary length (s)", 5, 120, 20)

if st.sidebar.button("Run full pipeline", type="primary", width="stretch"):
    if run_pipeline(video, p):
        (ROOT / "outputs" / "source.txt").write_text(video.name)
        st.rerun()
if OUT.name != "outputs":
    st.sidebar.caption("Note: a pipeline run writes to the 'outputs' folder. Select it to see fresh results.")

# ---------- main ----------
info = get_video_info(video)
st.title("Intelligent Video Analysis")
st.caption(f"{video.name}  |  {info.width}x{info.height}  |  {info.fps:.0f} FPS  |  "
           f"{format_time(info.duration_sec)}  |  {info.codec}")

scenes = load_csv(OUT / "scenes" / "scenes.csv")
kfs = load_csv(OUT / "keyframes" / "keyframes.csv")
motion = load_csv(OUT / "motion" / "motion_series.csv")
tracks = load_csv(OUT / "tracking" / "tracks.csv")
tsum = load_csv(OUT / "tracking" / "track_summary.csv")
uniq = load_csv(OUT / "tracking" / "unique_objects.csv")
clips = load_csv(OUT / "summaries" / "summary_clips.csv")

tabs = st.tabs(["Overview", "Scenes", "Keyframes", "Motion", "Tracking", "Summary"])

with tabs[0]:
    c = st.columns(5)
    c[0].metric("Scenes", len(scenes) if scenes is not None else "-")
    c[1].metric("Keyframes", len(kfs) if kfs is not None else "-")
    c[2].metric("Peak motion (px/s)", f"{motion.object_motion.max():.0f}" if motion is not None else "-")
    if uniq is not None:
        c[3].metric("Unique objects", len(uniq), f"{len(tsum)} raw track IDs", delta_color="off")
    else:
        c[3].metric("Tracked objects", len(tsum) if tsum is not None else "-")
    if clips is not None:
        L = float((clips.t1 - clips.t0).sum())
        c[4].metric("Summary", f"{L:.1f}s", f"{100 * L / info.duration_sec:.0f}% of video")
    else:
        c[4].metric("Summary", "-")
    st.subheader("Original video")
    st.video(str(video))
    if (uniq if uniq is not None else tsum) is not None:
        st.subheader("Objects by class")
        st.bar_chart((uniq if uniq is not None else tsum).class_name.value_counts())

with tabs[1]:
    show_image(OUT / "scenes" / "scene_timeline.png")
    if scenes is not None:
        st.dataframe(scenes[["scene_id", "start_time", "end_time", "duration_sec"]],
                     hide_index=True)
        st.subheader("Frame explorer")
        t = st.slider("Time (s)", 0.0, float(info.duration_sec) - 0.1, 0.0, 0.1)
        sid = scenes[(scenes.start_sec <= t) & (scenes.end_sec > t)].scene_id
        frame = read_frame(str(video), t, video.stat().st_mtime)
        if frame is not None:
            st.image(frame, caption=f"t = {format_time(t)}   |   scene "
                     f"{int(sid.iloc[0]) if len(sid) else '-'}", width="stretch")

with tabs[2]:
    show_image(OUT / "keyframes" / "contact_sheet.jpg", "All keyframes")
    if kfs is not None:
        st.dataframe(kfs[["scene_id", "keyframe", "time", "sharpness", "score"]], hide_index=True)

with tabs[3]:
    if motion is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=motion.time_sec, y=motion.object_motion,
                                 name="Object motion", line=dict(color="#D62728")))
        fig.add_trace(go.Scatter(x=motion.time_sec, y=motion.camera_motion,
                                 name="Camera motion", line=dict(color="#999999")))
        if scenes is not None:
            for s in scenes.start_sec.tolist()[1:]:
                fig.add_vline(x=s, line_dash="dash", line_width=1, opacity=0.4)
        fig.update_layout(title="Motion intensity over time", xaxis_title="Time (s)",
                          yaxis_title="px/s", height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        show_image(OUT / "motion" / "motion_heatmap.png", "Object-motion heatmap")
    with b:
        show_image(OUT / "motion" / "peak_flow.png", "Peak motion: frame and optical flow")

with tabs[4]:
    show_video(OUT / "tracking" / "tracked.mp4")
    a, b = st.columns(2)
    with a:
        show_image(OUT / "tracking" / "trajectories.png", "Trajectories")
    with b:
        show_image(OUT / "tracking" / "path_heatmap.png", "Where objects spent time")
    if tracks is not None:
        cnt = tracks.groupby(["time_sec", "class_name"]).size().reset_index(name="count")
        st.plotly_chart(px.line(cnt, x="time_sec", y="count", color="class_name",
                                title="Tracked objects over time", height=360),
                        width="stretch")
    if uniq is not None:
        st.dataframe(uniq.sort_values("duration_sec", ascending=False), hide_index=True)
        st.caption("Objects after merging fragmented tracks (same class, close in time and space, no scene cut between). "
                   "A heuristic, not true re-identification.")
    elif tsum is not None:
        st.dataframe(tsum.sort_values("duration_sec", ascending=False), hide_index=True)
        st.caption("Counts are track IDs, not distinct individuals: occlusions can create new IDs.")

with tabs[5]:
    show_video(OUT / "summaries" / "summary.mp4")
    show_image(OUT / "summaries" / "summary_selection.png")
    if clips is not None:
        st.dataframe(clips[["scene_id", "start", "end", "motion", "objects", "score"]],
                     hide_index=True)
