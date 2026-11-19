import io
import re
import tempfile
import zipfile
import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st

st.set_page_config(page_title="Moraxtv AI Clipper", page_icon="🎮", layout="centered")

st.title("🎮 Moraxtv AI Clipper")
st.caption("Mobile-friendly gaming highlight finder • v1.0")

st.info(
    "Upload a gameplay recording. This first version finds possible highlights "
    "from loud audio and visual activity. It does not yet recognize specific "
    "events such as kills or headshots."
)

with st.sidebar:
    st.header("Clip settings")
    clip_len = st.slider("Clip length (seconds)", 15, 60, 30, 5)
    number_clips = st.slider("Number of clips", 1, 5, 3)
    separation = st.slider("Minimum gap between clips (seconds)", 0, 120, 20, 5)

uploaded = st.file_uploader("Upload gameplay video", type=["mp4", "mov", "mkv", "avi", "webm"])

def ffmpeg(args):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    return subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error"] + args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )

def duration_seconds(path):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [exe, "-i", str(path), "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError("Could not read duration. Try an MP4 video.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)

def normalize(scores):
    if not scores.size:
        return scores
    low, high = np.percentile(scores, 10), np.percentile(scores, 95)
    return np.clip((scores - low) / max(float(high - low), 1.0), 0, 1)

def audio_activity(path, duration):
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        ffmpeg(["-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "wav", str(wav_path)])
        with wave.open(str(wav_path), "rb") as wf:
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
            rate = wf.getframerate()
    values = []
    for sec in range(int(np.ceil(duration))):
        chunk = samples[int(sec * rate):int((sec + 1) * rate)]
        values.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
    return normalize(np.asarray(values, dtype=np.float32))

def visual_activity(path, duration):
    cap = cv2.VideoCapture(str(path))
    total = int(np.ceil(duration))
    values = np.zeros(total, dtype=np.float32)
    if not cap.isOpened():
        return values
    previous = None
    for sec in range(total):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            values[sec] = float(np.mean(cv2.absdiff(gray, previous)))
        previous = gray
    cap.release()
    return normalize(values)

def select_windows(audio, visual, duration, length, count, gap):
    n = min(len(audio), len(visual), int(np.ceil(duration)))
    if n <= 0:
        return [(0.0, min(duration, length))]
    length = min(length, n)
    combined = 0.65 * audio[:n] + 0.35 * visual[:n]
    combined = np.convolve(combined, np.ones(5, dtype=np.float32) / 5, mode="same")
    candidates = []
    for start in range(max(1, n - length + 1)):
        candidates.append((float(np.mean(combined[start:start + length])), start))
    candidates.sort(reverse=True)
    chosen = []
    for score, start in candidates:
        end = min(start + length, duration)
        if all(end + gap <= old_start or start >= old_end + gap for old_start, old_end in chosen):
            chosen.append((float(start), float(end)))
            if len(chosen) == count:
                break
    return sorted(chosen or [(0.0, min(duration, length))])

def render_clip(source, destination, start, end):
    ffmpeg([
        "-y", "-ss", f"{start:.2f}", "-i", str(source), "-t", f"{end-start:.2f}",
        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "25", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(destination)
    ])

if uploaded:
    st.video(uploaded)
    if st.button("✨ Find highlights", type="primary", use_container_width=True):
        # Streamlit Cloud has limited memory and processing time. Keep uploads modest.
        if uploaded.size > 180 * 1024 * 1024:
            st.error("This free-hosted demo is limited to 180 MB per upload. Try a shorter or lower-resolution MP4.")
            st.stop()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            source = folder / ("gameplay" + Path(uploaded.name).suffix.lower())
            source.write_bytes(uploaded.getvalue())
            try:
                with st.spinner("Checking video..."):
                    duration = duration_seconds(source)
                if duration < 5:
                    st.error("Please upload a video at least 5 seconds long.")
                    st.stop()
                with st.spinner("Analyzing sound and visual activity. Please keep this page open..."):
                    audio = audio_activity(source, duration)
                    visual = visual_activity(source, duration)
                    windows = select_windows(audio, visual, duration, clip_len, number_clips, separation)
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, (start, end) in enumerate(windows, 1):
                        output = folder / f"Moraxtv_Highlight_{i:02d}.mp4"
                        with st.spinner(f"Rendering clip {i}/{len(windows)}..."):
                            render_clip(source, output, start, end)
                        zf.write(output, output.name)
                        st.write(f"**Clip {i}** · {int(start//60):02d}:{int(start%60):02d} – {int(end//60):02d}:{int(end%60):02d}")
                        st.video(output.read_bytes())
                st.download_button(
                    "⬇️ Download all clips (ZIP)", archive.getvalue(),
                    file_name="Moraxtv_Clips.zip", mime="application/zip",
                    use_container_width=True
                )
            except subprocess.CalledProcessError:
                st.error("Video processing failed. Try a standard MP4 file.")
            except Exception as exc:
                st.error(f"Processing error: {exc}")

st.divider()
st.caption("Moraxtv AI Clipper • Free prototype. Videos are processed by the hosting server when deployed online.")
st.caption("Please review all generated clips before publishing. The detector estimates activity; it does not understand game events.")
