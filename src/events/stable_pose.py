"""Stable-pose detector for hands-free solo shooting."""

from __future__ import annotations

from typing import Any

from src.pose.detector import FramePose, PersonPose

from .types import RawEvent
from .utils import mean_motion, select_subject


class StablePoseDetector:
    """Emit one event per continuous stable segment held for at least 3 seconds.

    The segment stays open while the subject remains within ``movement_threshold`` of
    the pose captured at the segment start (the anchor). Comparing against the anchor
    instead of the previous frame prevents slow accumulated drift from posing as one
    long stable hold. The segment is closed — and the event emitted with the whole
    retained span — by real motion, disappearance, or flush.
    """

    event_type = "stable_pose"

    def __init__(self, config: dict[str, Any]) -> None:
        event_config = config.get("events", {}).get("stable_pose", {})
        self.duration_seconds = float(event_config.get("duration_seconds", 3.0))
        self.movement_threshold = float(event_config.get("movement_threshold", 0.035))
        self.min_completeness = float(event_config.get("min_completeness", 0.65))
        self.min_confidence = float(event_config.get("min_person_confidence", 0.35))
        self._start: float | None = None
        self._last_stable_time: float | None = None
        self._peak_time: float | None = None
        self._anchor: PersonPose | None = None
        self._movements: list[float] = []
        self._completeness: list[float] = []
        self._confidences: list[float] = []

    def update(self, frame_pose: FramePose) -> list[RawEvent]:
        person = select_subject(frame_pose, self.min_confidence, self.min_completeness)
        movement = mean_motion(self._anchor, person)
        is_stable = person is not None and (movement is None or movement <= self.movement_threshold)
        emitted: list[RawEvent] = []
        if is_stable and person is not None:
            if self._start is None:
                self._start = frame_pose.timestamp
                self._peak_time = frame_pose.timestamp
                self._anchor = person
            self._last_stable_time = frame_pose.timestamp
            if movement is not None:
                self._movements.append(movement)
            self._completeness.append(person.completeness())
            self._confidences.append(person.confidence)
            # The segment intentionally stays open past duration_seconds so the event
            # covers the entire hold; emission waits for motion, disappearance, or flush.
        else:
            closed = self._close()
            if closed is not None:
                emitted.append(closed)
        return emitted

    def _close(self) -> RawEvent | None:
        if self._start is None or self._last_stable_time is None:
            self._reset()
            return None
        end_time = self._last_stable_time
        duration = end_time - self._start
        event = None
        if duration >= self.duration_seconds:
            avg_conf = sum(self._confidences) / max(len(self._confidences), 1)
            avg_completeness = sum(self._completeness) / max(len(self._completeness), 1)
            max_movement = max(self._movements, default=0.0)
            event = RawEvent(
                event_type=self.event_type,
                start_time=self._start,
                peak_time=self._peak_time or self._start,
                end_time=end_time,
                confidence=max(0.0, min(1.0, avg_conf)),
                features={
                    "duration_seconds": duration,
                    "movement_amplitude": max_movement,
                    "person_completeness": avg_completeness,
                    "person_confidence": avg_conf,
                },
            )
        self._reset()
        return event

    def flush(self, end_time: float | None = None) -> list[RawEvent]:
        del end_time  # The last observed stable timestamp is the accurate segment end.
        event = self._close()
        return [event] if event is not None else []

    def _reset(self) -> None:
        self._start = None
        self._last_stable_time = None
        self._peak_time = None
        self._anchor = None
        self._movements.clear()
        self._completeness.clear()
        self._confidences.clear()
