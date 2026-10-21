from dataclasses import dataclass
from pathlib import Path
import sys

import cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str


def format_time(seconds: float) -> str:
    """Seconds -> HH:MM:SS.s"""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def get_video_info(path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    info = VideoInfo(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        frame_count=frame_count,
        duration_sec=frame_count / fps if fps else 0.0,
        codec=codec,
    )
    cap.release()
    return info


def save_first_frame(path, out_path) -> bool:
    """Sanity check that frames actually decode."""
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if ok:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame)
    return ok


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/video.mp4"
    info = get_video_info(video)
    print("-" * 40)
    print(f"File       : {info.path}")
    print(f"Resolution : {info.width} x {info.height}")
    print(f"FPS        : {info.fps:.2f}")
    print(f"Frames     : {info.frame_count}")
    print(f"Duration   : {format_time(info.duration_sec)}")
    print(f"Codec      : {info.codec}")
    print("-" * 40)
    ok = save_first_frame(video, "outputs/first_frame.jpg")
    print("First frame decoded and saved to outputs/first_frame.jpg" if ok
          else "WARNING: could not decode first frame")