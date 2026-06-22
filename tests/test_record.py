"""Recording sampler and MP4 path tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualize.record import (
    FINAL_SLOT_KEY,
    RECORD_PLAYBACK_SLOWDOWN,
    AsyncMp4Recorder,
    RecordingSampler,
    crop_frame_to_even,
    pil_to_rgb_array,
    require_ffmpeg,
    resolve_recording_path,
    sanitize_recording_stem,
    slot_index,
    slot_time,
    slots_to_capture,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_slot_index_and_time():
    assert slot_time(0, 0.0, 10.0) == 0.0
    assert slot_time(3, 0.0, 10.0) == pytest.approx(0.3)
    assert slot_index(0.25, 0.0, 10.0) == 2


def test_slots_to_capture_forward_crossing():
    captured: set[int] = set()
    due = slots_to_capture(0.0, 0.25, start_t=0.0, end_t=1.0, fps=10.0, captured=captured)
    assert due == [1, 2]
    assert 0 not in due

    captured.update(due)
    due = slots_to_capture(0.25, 0.55, start_t=0.0, end_t=1.0, fps=10.0, captured=captured)
    assert due == [3, 4, 5]


def test_slots_to_capture_skips_slot_zero_at_start():
    captured: set[int] = set()
    due = slots_to_capture(0.0, 0.0, start_t=0.0, end_t=1.0, fps=10.0, captured=captured)
    assert due == []


def test_slots_to_capture_backward_scrub_ignored():
    captured = {1, 2, 3}
    due = slots_to_capture(0.5, 0.2, start_t=0.0, end_t=1.0, fps=10.0, captured=captured)
    assert due == []


def test_slots_to_capture_unaligned_end():
    captured: set[int] = set()
    due = slots_to_capture(0.9, 1.05, start_t=0.0, end_t=1.05, fps=10.0, captured=captured)
    assert FINAL_SLOT_KEY in due


def test_require_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        require_ffmpeg()


def test_pil_to_rgb_array():
    img = Image.new("RGB", (20, 10), color=(255, 0, 0))
    frame = pil_to_rgb_array(img)
    assert frame.shape == (10, 20, 3)
    assert frame.dtype == np.uint8


def test_crop_frame_to_even():
    odd = np.zeros((11, 15, 3), dtype=np.uint8)
    width, height, cropped = crop_frame_to_even(odd)
    assert (width, height) == (14, 10)
    assert cropped.shape == (10, 14, 3)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_async_mp4_recorder_writes_file(tmp_path):
    path = tmp_path / "stream.mp4"
    recorder = AsyncMp4Recorder(path, width=8, height=8, fps=10.0)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for color in colors:
        arr = np.full((8, 8, 3), color, dtype=np.uint8)
        recorder.add_frame(arr)
    recorder.close()
    recorder.join()
    assert recorder.error is None
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_async_mp4_recorder_odd_dimensions(tmp_path):
    """Odd grab sizes are cropped to even before encoding (libx264 yuv420p)."""
    path = tmp_path / "odd.mp4"
    recorder = AsyncMp4Recorder(path, width=14, height=10, fps=10.0)
    arr = np.zeros((10, 14, 3), dtype=np.uint8)
    recorder.add_frame(arr)
    recorder.close()
    recorder.join()
    assert recorder.error is None
    assert path.stat().st_size > 0


def test_recording_sampler_playback_slowdown():
    sampler = RecordingSampler(
        fps=20.0,
        start_t=0.0,
        end_t=1.0,
        scenario_name="slow",
    )
    assert sampler._encode_fps == pytest.approx(20.0 / RECORD_PLAYBACK_SLOWDOWN)


def test_recording_sampler_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        RecordingSampler(fps=10.0, start_t=0.0, end_t=0.2, scenario_name="test")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_recording_sampler_keep_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    grabs: list[int] = []

    def grab():
        grabs.append(len(grabs))
        return Image.new("RGB", (4, 4), color=(grabs[-1], 0, 0))

    sampler = RecordingSampler(fps=10.0, start_t=0.0, end_t=0.2, scenario_name="test")
    sampler.on_t_advanced(-1.0, 0.0, grab_fn=grab)
    assert len(sampler._captured_slots) == 0
    assert sampler.has_frames is False
    sampler.on_t_advanced(0.15, 0.05, grab_fn=grab)
    assert len(sampler._captured_slots) == 0
    assert sampler.has_frames is False


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_recording_sampler_crops_odd_grab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def grab():
        return Image.new("RGB", (15, 11), color=(128, 64, 32))

    sampler = RecordingSampler(fps=10.0, start_t=0.0, end_t=0.1, scenario_name="odd")
    sampler.on_t_advanced(0.0, 0.1, grab_fn=grab)
    path = sampler.finalize()
    assert path is not None
    assert path.stat().st_size > 0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_recording_sampler_completes_at_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames: list[Image.Image] = []

    def grab():
        img = Image.new("RGB", (2, 2), color=(0, 0, 0))
        frames.append(img)
        return img

    sampler = RecordingSampler(fps=10.0, start_t=0.0, end_t=0.2, scenario_name="complete")
    assert sampler.on_t_advanced(-1.0, 0.0, grab_fn=grab) is False
    assert sampler.on_t_advanced(0.0, 0.2, grab_fn=grab) is True
    path = sampler.finalize()
    assert path is not None
    assert path.suffix == ".mp4"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_resolve_recording_path_collision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = tmp_path / "recordings"
    rec.mkdir()
    (rec / "demo.mp4").write_bytes(b"x")
    (rec / "demo_1.mp4").write_bytes(b"x")

    assert resolve_recording_path("demo") == Path("recordings/demo_2.mp4")
    assert resolve_recording_path("new") == Path("recordings/new.mp4")


def test_sanitize_recording_stem_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        sanitize_recording_stem("   ")
