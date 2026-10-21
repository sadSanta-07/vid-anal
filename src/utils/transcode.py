import subprocess


def to_h264(src, dst) -> bool:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"  (imageio-ffmpeg unavailable: {e})")
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-an",
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-preset", "veryfast", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-500:])
        return False
    return True