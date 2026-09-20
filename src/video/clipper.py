"""FFmpeg-backed event clipping and highlight concatenation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from src.events.types import RawEvent

FFMPEG_TIMEOUT_SECONDS = 120


def concat_entry(path: Path) -> str:
    """Format one ``file`` directive for the FFmpeg concat demuxer."""
    # FFmpeg (not SQL) quoting: close the single-quoted string, emit a
    # backslash-escaped quote, then reopen. Apostrophe -> '\''.
    escaped = path.as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def ffmpeg_executable() -> str | None:
    return shutil.which("ffmpeg")


def check_ffmpeg_available() -> bool:
    return ffmpeg_executable() is not None


class FFmpegClipper:
    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or ffmpeg_executable()

    def _require_executable(self) -> str:
        if not self.executable:
            raise RuntimeError(
                "FFmpeg is not available on PATH. Install it with `brew install ffmpeg` "
                "or see README.md."
            )
        return self.executable

    def _run(self, command: list[str]) -> None:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            stderr = error.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr = (stderr or "").strip()
            raise RuntimeError(
                f"FFmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s: {' '.join(command)}"
                f"{': ' + stderr if stderr else ''}"
            ) from error
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"FFmpeg exited with code {result.returncode}: {' '.join(command)}"
                f"{': ' + stderr if stderr else ''}"
            )

    def clip_event(
        self,
        input_path: str | Path,
        event: RawEvent,
        output_path: str | Path,
        pre_buffer: float = 2.0,
        post_buffer: float = 2.0,
    ) -> Path:
        executable = self._require_executable()
        source = Path(input_path).resolve()
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        start = max(0.0, event.start_time - max(0.0, pre_buffer))
        end = max(start + 0.1, event.end_time + max(0.0, post_buffer))
        duration = end - start
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(target),
        ]
        self._run(command)
        return target

    def merge_clips(self, clip_paths: Iterable[str | Path], output_path: str | Path) -> Path | None:
        executable = self._require_executable()
        paths = [Path(path).resolve() for path in clip_paths if Path(path).exists()]
        if not paths:
            return None
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="solo-director-concat-") as temp_dir:
            concat_file = Path(temp_dir) / "concat.txt"
            concat_file.write_text(
                "".join(concat_entry(path) for path in paths),
                encoding="utf-8",
            )
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(target),
            ]
            self._run(command)
        return target
