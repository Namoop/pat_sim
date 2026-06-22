"""Recording sampler and MP4 path tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualize.record import (
    FINAL_SLOT_KEY,
    RECORD_REFERENCE_PLAYBACK_SLOWDOWN,
    RECORD_REFERENCE_T_STEP,
    AsyncMp4Recorder,
    RecordingSampler,
    crop_frame_to_even,
    ensure_rgb_uint8,
    recording_capture_fps,
    recording_encode_fps,
    recording_playback_slowdown,
    require_ffmpeg,
    resolve_recording_path,
    sanitize_recording_stem,
    slot_index,
    slot_time,
    slots_to_capture,
)


def _has_ffmpeg() -> bool:
    try:
        from visualize.record import require_ffmpeg

        require_ffmpeg()
        return True
    except RuntimeError:
        return False


HAS_FFMPEG = _has_ffmpeg()


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
    def _missing() -> str:
        raise ImportError("imageio_ffmpeg not installed")

    monkeypatch.setattr("visualize.record._ffmpeg_exe", _missing)
    with pytest.raises(RuntimeError, match="Recording requires imageio-ffmpeg"):
        require_ffmpeg()


def test_ensure_rgb_uint8_drops_alpha():
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgb = ensure_rgb_uint8(rgba)
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == np.uint8


def test_crop_frame_to_even():
    odd = np.zeros((11, 15, 3), dtype=np.uint8)
    width, height, cropped = crop_frame_to_even(odd)
    assert (width, height) == (14, 10)
    assert cropped.shape == (10, 14, 3)


@pytest.mark.skipif(not HAS_FFMPEG, reason="imageio-ffmpeg not available")
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


@pytest.mark.skipif(not HAS_FFMPEG, reason="imageio-ffmpeg not available")
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


def test_recording_capture_fps_default_stride():
    assert recording_capture_fps(2, 0.01) == pytest.approx(50.0)


def test_recording_playback_slowdown_scales_with_t_step():
    assert recording_playback_slowdown(RECORD_REFERENCE_T_STEP) == pytest.approx(
        RECORD_REFERENCE_PLAYBACK_SLOWDOWN
    )
    assert recording_playback_slowdown(0.0001) == pytest.approx(400.0)


def test_recording_encode_fps_matches_reference_slowdown():
    stride = 2.0
    t_step = 0.01
    capture = recording_capture_fps(stride, t_step)
    assert recording_encode_fps(stride, t_step) == pytest.approx(
        capture / RECORD_REFERENCE_PLAYBACK_SLOWDOWN
    )


def test_recording_sampler_encode_fps():
    sampler = RecordingSampler(
        stride=2,
        t_step=0.01,
        start_t=0.0,
        end_t=1.0,
        scenario_name="slow",
    )
    assert sampler.fps == pytest.approx(50.0)
    assert sampler._encode_fps == pytest.approx(12.5)


def test_recording_sampler_requires_ffmpeg(monkeypatch):
    def _missing() -> str:
        raise ImportError("imageio_ffmpeg not installed")

    monkeypatch.setattr("visualize.record._ffmpeg_exe", _missing)
    with pytest.raises(RuntimeError, match="Recording requires imageio-ffmpeg"):
        RecordingSampler(
            stride=2, t_step=0.01, start_t=0.0, end_t=0.2, scenario_name="test"
        )


@pytest.mark.skipif(not HAS_FFMPEG, reason="imageio-ffmpeg not available")
def test_recording_sampler_keep_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    grabs: list[int] = []

    def grab():
        grabs.append(len(grabs))
        return np.full((4, 4, 3), (grabs[-1], 0, 0), dtype=np.uint8)

    sampler = RecordingSampler(
        stride=2, t_step=0.01, start_t=0.0, end_t=0.2, scenario_name="test"
    )
    sampler.on_t_advanced(-1.0, 0.0, grab_fn=grab)
    assert len(sampler._captured_slots) == 0
    assert sampler.has_frames is False
    sampler.on_t_advanced(0.15, 0.05, grab_fn=grab)
    assert len(sampler._captured_slots) == 0
    assert sampler.has_frames is False


@pytest.mark.skipif(not HAS_FFMPEG, reason="imageio-ffmpeg not available")
def test_recording_sampler_crops_odd_grab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def grab():
        return np.full((11, 15, 3), (128, 64, 32), dtype=np.uint8)

    sampler = RecordingSampler(
        stride=2, t_step=0.01, start_t=0.0, end_t=0.1, scenario_name="odd"
    )
    sampler.on_t_advanced(0.0, 0.1, grab_fn=grab)
    path = sampler.finalize()
    assert path is not None
    assert path.stat().st_size > 0


@pytest.mark.skipif(not HAS_FFMPEG, reason="imageio-ffmpeg not available")
def test_recording_sampler_completes_at_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def grab():
        return np.zeros((2, 2, 3), dtype=np.uint8)

    sampler = RecordingSampler(
        stride=2, t_step=0.01, start_t=0.0, end_t=0.2, scenario_name="complete"
    )
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
