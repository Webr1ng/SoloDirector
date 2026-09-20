"""Streaming jump/motion-peak detector based on normalized vertical motion."""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any

from src.pose.detector import FramePose, PersonPose

from .types import RawEvent
from .utils import clamp01, select_subject, visible

LEFT_HIP, RIGHT_HIP = 11, 12


class JumpDetector:
    """Detect a fast rise followed by a descent of the body center.

    Only hips drive the vertical signal: mixing ankles and hips makes the
    measurement jump when keypoint visibility flips. Frames without both hips
    consistently visible are ignored, a missing person resets the active jump
    state and history, and large timestamp gaps break the trajectory instead of
    producing a false motion peak.
    """

    event_type = "jump"

    def __init__(self, config: dict[str, Any]) -> None:
        event_config = config.get("events", {}).get("jump", {})
        self.lookback_seconds = float(event_config.get("lookback_seconds", 0.8))
        self.minimum_rise = float(event_config.get("minimum_rise", 0.08))
        self.minimum_descent = float(event_config.get("minimum_descent", 0.05))
        self.max_event_seconds = float(event_config.get("max_event_seconds", 2.5))
        self.cooldown_seconds = float(event_config.get("cooldown_seconds", 0.8))
        self.min_person_confidence = float(event_config.get("min_person_confidence", 0.35))
        self.keypoint_confidence = float(event_config.get("keypoint_confidence", 0.25))
        self.max_frame_gap_seconds = float(event_config.get("max_frame_gap_seconds", 0.5))
        self.history: deque[tuple[float, float, float]] = deque(maxlen=256)
        self._start: float | None = None
        self._apex_time: float | None = None
        self._apex_y: float | None = None
        self._baseline_y: float | None = None
        self._last_event_time: float | None = None
        self._last_timestamp: float | None = None

    @staticmethod
    def _hip_position(person: PersonPose, hip_confidence: float) -> float | None:
        hips_visible = visible(person, LEFT_HIP, hip_confidence) and visible(
            person, RIGHT_HIP, hip_confidence
        )
        if not hips_visible:
            return None
        left_y = float(person.keypoints[LEFT_HIP][1])
        right_y = float(person.keypoints[RIGHT_HIP][1])
        return (left_y + right_y) / 2.0

    def _reset_active(self) -> None:
        self._start = None
        self._apex_time = None
        self._apex_y = None
        self._baseline_y = None

    def _reset_all(self) -> None:
        self._reset_active()
        self.history.clear()
        self._last_timestamp = None

    def update(self, frame_pose: FramePose) -> list[RawEvent]:
        person = select_subject(frame_pose, self.min_person_confidence)
        if person is None:
            self._reset_all()
            return []
        timestamp = frame_pose.timestamp
        if (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp > self.max_frame_gap_seconds
        ):
            # Tracking was interrupted; the previous trajectory cannot be continued.
            self._reset_all()
        self._last_timestamp = timestamp
        position = self._hip_position(person, self.keypoint_confidence)
        if position is None:
            # Hip visibility dropped out: the vertical signal is interrupted, so the
            # stale trajectory and baseline must not be mixed with future frames.
            self._reset_all()
            return []
        self.history.append((timestamp, position, person.confidence))
        while self.history and timestamp - self.history[0][0] > max(
            self.lookback_seconds * 2.5, 2.5
        ):
            self.history.popleft()

        if (
            self._last_event_time is not None
            and timestamp - self._last_event_time < self.cooldown_seconds
        ):
            return []

        if self._start is None:
            baseline_values = [
                value
                for time_value, value, _ in self.history
                if timestamp - time_value >= self.lookback_seconds * 0.45
            ]
            if len(baseline_values) < 2:
                return []
            baseline = median(baseline_values)
            rise = baseline - position
            if rise >= self.minimum_rise:
                self._start = timestamp
                self._baseline_y = baseline
                self._apex_y = position
                self._apex_time = timestamp
            return []

        if self._apex_y is None or position < self._apex_y:
            self._apex_y = position
            self._apex_time = timestamp
            return []

        descent = position - self._apex_y
        if descent >= self.minimum_descent:
            baseline = self._baseline_y if self._baseline_y is not None else position
            rise = max(0.0, baseline - self._apex_y)
            confidence = clamp01(0.45 + rise / max(self.minimum_rise * 3.0, 0.001))
            event = RawEvent(
                event_type=self.event_type,
                start_time=self._start,
                peak_time=self._apex_time or self._start,
                end_time=timestamp,
                confidence=confidence,
                features={
                    "vertical_amplitude": rise,
                    "descent_amplitude": descent,
                    "person_confidence": person.confidence,
                    "motion_peak": True,
                },
            )
            self._last_event_time = timestamp
            self._reset_active()
            return [event]

        if timestamp - self._start > self.max_event_seconds:
            self._reset_active()
        return []

    def flush(self, end_time: float | None = None) -> list[RawEvent]:
        del end_time
        self._reset_active()
        return []
