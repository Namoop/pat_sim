"""Passive sim-time MP4 recording for the visualizer."""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

RECORDINGS_DIR = Path("recordings")
FINAL_SLOT_KEY = -1
FIRST_CAPTURE_SLOT = 1
_TIME_EPS = 1e-9
RECORD_PLAYBACK_SLOWDOWN = 4.0  # MP4 plays this many times slower than capture fps


_UNSAFE_STEM_RE = re.compile(r'[<>:"/\\|?*\x00]')


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def require_ffmpeg() -> str:
    """Return the ffmpeg executable path (bundled via imageio-ffmpeg)."""
    try:
        return _ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "Recording requires imageio-ffmpeg. "
            'Install visualization extras: pip install -e ".[viz]"'
        ) from exc


def slot_index(t: float, start_t: float, fps: float) -> int:
    return int(np.floor((t - start_t) * fps + _TIME_EPS))


def slot_time(k: int, start_t: float, fps: float) -> float:
    return start_t + k / fps


def sanitize_recording_stem(name: str) -> str:
    stem = _UNSAFE_STEM_RE.sub("_", name.strip())
    if not stem:
        raise ValueError("scenario name is empty after sanitization")
    return stem


def resolve_recording_path(scenario_name: str) -> Path:
    stem = sanitize_recording_stem(scenario_name)
    base = RECORDINGS_DIR / f"{stem}.mp4"
    if not base.exists():
        return base
    n = 1
    while (RECORDINGS_DIR / f"{stem}_{n}.mp4").exists():
        n += 1
    return RECORDINGS_DIR / f"{stem}_{n}.mp4"


def _end_aligned(start_t: float, end_t: float, fps: float) -> bool:
    steps = (end_t - start_t) * fps
    return abs(steps - round(steps)) < 1e-6


def slots_to_capture(
    prev_t: float,
    new_t: float,
    *,
    start_t: float,
    end_t: float,
    fps: float,
    captured: set[int],
    first_capture_slot: int = FIRST_CAPTURE_SLOT,
) -> list[int]:
    """Return slot keys to capture on a forward t advance (keep-first)."""
    if new_t < prev_t - _TIME_EPS:
        return []

    due: list[int] = []
    k = first_capture_slot
    while True:
        t_k = slot_time(k, start_t, fps)
        if t_k > end_t + _TIME_EPS:
            break
        if k in captured:
            k += 1
            continue
        if prev_t < t_k <= new_t + _TIME_EPS:
            due.append(k)
        k += 1

    if (
        FINAL_SLOT_KEY not in captured
        and new_t >= end_t - _TIME_EPS
        and not _end_aligned(start_t, end_t, fps)
    ):
        due.append(FINAL_SLOT_KEY)

    return due


def pixmap_to_rgb_array(pixmap: Any) -> np.ndarray:
    from PyQt6.QtGui import QImage

    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    rgba = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return np.ascontiguousarray(rgba[..., :3])


def ensure_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Return H×W×3 uint8 RGB, dropping alpha if present."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected H×W×3 or H×W×4 uint8 array, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def crop_frame_to_even(rgb: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Crop to even width/height required by libx264 yuv420p."""
    height, width = rgb.shape[:2]
    even_w = width - (width % 2)
    even_h = height - (height % 2)
    if even_w < 2 or even_h < 2:
        raise ValueError(
            f"recording frame too small for H.264: {width}x{height}"
        )
    if even_w != width or even_h != height:
        rgb = np.ascontiguousarray(rgb[:even_h, :even_w])
    return even_w, even_h, rgb


def _ffmpeg_report(
    process: subprocess.Popen[bytes],
    *,
    prefix: str,
    stderr_body: bytes = b"",
) -> RuntimeError:
    msg = stderr_body.decode(errors="replace").strip()
    if not msg and process.stderr is not None:
        try:
            msg = process.stderr.read().decode(errors="replace").strip()
        except Exception:
            msg = ""
    detail = msg or f"exit code {process.poll()}"
    return RuntimeError(f"{prefix} {detail}")


class AsyncMp4Recorder:
    """Producer-consumer MP4 writer: frames queued from UI, FFmpeg encodes via stdin."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if fps <= 0:
            raise ValueError("record fps must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("recording dimensions must be positive")
        require_ffmpeg()
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._closed = False
        self._frame_count = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._writer_worker,
            name="mp4-recorder",
            daemon=False,
        )
        self._thread.start()

    @property
    def has_frames(self) -> bool:
        return self._frame_count > 0

    @property
    def error(self) -> BaseException | None:
        return self._error

    def add_frame(self, frame_array: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("AsyncMp4Recorder is closed")
        expected = (self._height, self._width, 3)
        if frame_array.dtype != np.uint8 or frame_array.shape != expected:
            raise ValueError(
                f"expected uint8 array with shape {expected}, got {frame_array.dtype} {frame_array.shape}"
            )
        if not frame_array.flags.c_contiguous:
            frame_array = np.ascontiguousarray(frame_array)
        self._queue.put(frame_array.copy())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _writer_worker(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = require_ffmpeg()
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(self._path),
        ]
        process: subprocess.Popen[bytes] | None = None
        stderr_thread: threading.Thread | None = None
        stderr_body = b""
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None

            def _drain_stderr() -> None:
                nonlocal stderr_body
                assert process is not None and process.stderr is not None
                stderr_body = process.stderr.read()

            stderr_thread = threading.Thread(
                target=_drain_stderr,
                name="ffmpeg-stderr",
                daemon=True,
            )
            stderr_thread.start()

            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError as exc:
                    process.stdin.close()
                    process.wait()
                    if stderr_thread is not None:
                        stderr_thread.join(timeout=5.0)
                    raise _ffmpeg_report(
                        process,
                        prefix=f"ffmpeg pipe broken while writing {self._path}:",
                        stderr_body=stderr_body,
                    ) from exc
                self._frame_count += 1
            process.stdin.close()
            rc = process.wait()
            if stderr_thread is not None:
                stderr_thread.join(timeout=5.0)
            if rc != 0:
                raise _ffmpeg_report(
                    process,
                    prefix=f"ffmpeg failed while writing {self._path}:",
                    stderr_body=stderr_body,
                )
        except BaseException as exc:
            self._error = exc
            if process is not None and process.poll() is None:
                process.kill()
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break


class RecordingSampler:
    """Passive sim-time frame sampler with keep-first slot capture."""

    def __init__(
        self,
        *,
        fps: float,
        start_t: float,
        end_t: float,
        scenario_name: str,
    ) -> None:
        if fps <= 0:
            raise ValueError("record fps must be positive")
        require_ffmpeg()
        self.fps = fps
        self._encode_fps = fps / RECORD_PLAYBACK_SLOWDOWN
        self.start_t = start_t
        self.end_t = end_t
        self.scenario_name = scenario_name
        self._captured_slots: set[int] = set()
        self._encoder: AsyncMp4Recorder | None = None
        self._finalized = False
        self._output_path: Path | None = None

    @property
    def has_frames(self) -> bool:
        return self._encoder is not None and self._encoder.has_frames

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _submit_frame(self, frame: np.ndarray) -> None:
        width, height, rgb = crop_frame_to_even(ensure_rgb_uint8(frame))
        if self._encoder is None:
            path = resolve_recording_path(self.scenario_name)
            self._output_path = path
            self._encoder = AsyncMp4Recorder(
                path,
                width=width,
                height=height,
                fps=self._encode_fps,
            )
        self._encoder.add_frame(rgb)

    def _timeline_complete(self) -> bool:
        if not self._captured_slots:
            return False
        if _end_aligned(self.start_t, self.end_t, self.fps):
            last_k = slot_index(self.end_t, self.start_t, self.fps)
            if last_k < FIRST_CAPTURE_SLOT:
                return False
            return last_k in self._captured_slots
        return FINAL_SLOT_KEY in self._captured_slots

    def on_t_advanced(
        self,
        prev_t: float,
        new_t: float,
        *,
        grab_fn: Callable[[], np.ndarray],
    ) -> bool:
        """Capture newly crossed slots after the view has been rendered."""
        if self._finalized:
            return False

        due = slots_to_capture(
            prev_t,
            new_t,
            start_t=self.start_t,
            end_t=self.end_t,
            fps=self.fps,
            captured=set(self._captured_slots),
        )
        for key in due:
            if key not in self._captured_slots:
                self._submit_frame(grab_fn())
                self._captured_slots.add(key)

        return self._timeline_complete()

    def begin_finalize(self) -> None:
        """Stop capture and signal the encoder to shut down (non-blocking)."""
        if self._finalized:
            return
        self._finalized = True
        if self._encoder is not None:
            self._encoder.close()

    def complete_finalize(self) -> Path | None:
        """Wait for FFmpeg to finish; return the output path."""
        if self._encoder is None:
            return None
        self._encoder.join()
        if self._encoder.error is not None:
            raise self._encoder.error
        if not self._encoder.has_frames:
            return None
        return self._output_path

    def finalize(self) -> Path | None:
        """Flush queued frames to MP4. Returns path or None if empty."""
        self.begin_finalize()
        return self.complete_finalize()
