
import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Intelligent Video Analysis",
    page_icon="🎬",
    layout="wide",
)

st.title("Intelligent Video Analysis")
st.caption(
    "Video processing, scene detection, motion analysis, "
    "object tracking, and summarization."
)

st.subheader("Upload your video")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is not None:
    video_bytes = uploaded_video.getvalue()

    st.video(video_bytes)

    video_array = np.frombuffer(video_bytes, dtype=np.uint8)
    frame = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

    if frame is not None:
        height, width = frame.shape[:2]

        col1, col2 = st.columns(2)
        col1.metric("Frame width", f"{width} px")
        col2.metric("Frame height", f"{height} px")

        st.success("Video uploaded and decoded successfully.")
    else:
        st.error("Could not decode this video.")
else:
    st.info("Upload a video to begin your analysis.")
