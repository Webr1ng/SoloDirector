"""OpenCV video reader with deterministic timestamp-aware sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(slots=True)
class VideoFrame:
    frame_idx: int
    timestamp: float
    frame: np.ndarray
    fps: float


class VideoReader:
    """Read BGR frames while preserving source frame indexes and timestamps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Input video does not exist: {self.path}")
        self._capture: cv2.VideoCapture | None = None

    def info(self) -> VideoInfo:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(self.path, width, height, fps, frame_count, duration)

    def iter_frames(
        self,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield frames at or below ``target_fps`` without losing source indexes."""

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open video: {self.path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        effective_target = float(target_fps or source_fps)
        effective_target = min(effective_target, source_fps)
        interval = 1.0 / max(effective_target, 0.001)
        next_sample_time = 0.0
        emitted = 0
        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_idx / source_fps
                should_emit = target_fps is None or source_fps <= target_fps
                if not should_emit:
                    should_emit = timestamp + (1.0 / source_fps) * 0.5 >= next_sample_time
                if should_emit:
                    yield VideoFrame(frame_idx, timestamp, frame, source_fps)
                    emitted += 1
                    next_sample_time += interval
                    if max_frames is not None and emitted >= max_frames:
                        break
                frame_idx += 1
        finally:
            capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_: object) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
