import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.video.clipper import (
    FFMPEG_TIMEOUT_SECONDS,
    FFmpegClipper,
    check_ffmpeg_available,
    concat_entry,
    ffmpeg_executable,
)

FFMPEG_AVAILABLE = check_ffmpeg_available()
FFPROBE_AVAILABLE = shutil.which("ffprobe") is not None


def test_ffmpeg_probe_returns_a_boolean():
    available = check_ffmpeg_available()
    assert isinstance(available, bool)
    if available:
        assert ffmpeg_executable()


def test_concat_entry_escapes_apostrophes_ffmpeg_style():
    # FFmpeg syntax: close the quote, backslash-escaped quote, reopen ('\''),
    # not the SQL-style doubled apostrophe ('').
    directive = concat_entry(Path("/tmp/it's a test.mp4"))
    assert directive == "file '/tmp/it'\\''s a test.mp4'\n"


def test_run_raises_runtime_error_on_nonzero_exit(tmp_path: Path):
    clipper = FFmpegClipper(executable="/bin/sh")
    with pytest.raises(RuntimeError, match="boom"):
        clipper._run(["/bin/sh", "-c", "echo boom >&2; exit 3"])
    assert FFMPEG_TIMEOUT_SECONDS == 120


def test_run_decodes_bytes_stderr_on_timeout(monkeypatch: pytest.MonkeyPatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 0, stderr=b"late boom \xff bytes")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clipper = FFmpegClipper(executable="ffmpeg")
    with pytest.raises(RuntimeError, match="late boom"):
        clipper._run(["ffmpeg", "-version"])


def _encode_tiny_clip(ffmpeg: str, destination: Path, frequency: int, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=128x96:rate=10",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(
    not (FFMPEG_AVAILABLE and FFPROBE_AVAILABLE),
    reason="ffmpeg/ffprobe not available on PATH",
)
def test_merge_clips_with_apostrophes_and_spaces_in_paths(tmp_path: Path):
    ffmpeg = ffmpeg_executable()
    assert ffmpeg
    clips_dir = tmp_path / "it's clip dir"
    clip_a = clips_dir / "take one's.mp4"
    clip_b = clips_dir / "take two's.mp4"
    _encode_tiny_clip(ffmpeg, clip_a, frequency=440, duration=0.4)
    _encode_tiny_clip(ffmpeg, clip_b, frequency=880, duration=0.6)

    clipper = FFmpegClipper()
    merged = clipper.merge_clips(
        [clip_a, clip_b],
        tmp_path / "it's output dir" / "high light's.mp4",
    )
    assert merged is not None and merged.exists()

    ffprobe = shutil.which("ffprobe")
    assert ffprobe
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(merged),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)
    assert {"video", "audio"} <= {stream["codec_type"] for stream in info["streams"]}
    assert 0.8 <= float(info["format"]["duration"]) <= 1.5
