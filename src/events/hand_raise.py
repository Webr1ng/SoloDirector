"""Rule-based hand-raise detector with a clear extension point for waving."""

from __future__ import annotations

from typing import Any

from src.pose.detector import FramePose, PersonPose

from .types import RawEvent
from .utils import clamp01, select_subject, visible

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10


class HandRaiseDetector:
    """Detect a wrist held above its same-side shoulder for consecutive frames."""

    event_type = "hand_raise"

    def __init__(self, config: dict[str, Any]) -> None:
        event_config = config.get("events", {}).get("hand_raise", {})
        self.minimum_duration = float(event_config.get("minimum_duration_seconds", 0.35))
        self.shoulder_margin = float(event_config.get("shoulder_margin", 0.04))
        self.keypoint_confidence = float(event_config.get("keypoint_confidence", 0.25))
        self.min_person_confidence = float(event_config.get("min_person_confidence", 0.35))
        self._start: float | None = None
        self._last_time: float | None = None
        self._hands: set[str] = set()
        self._raise_amounts: list[float] = []
        self._confidences: list[float] = []

    def _raised_hands(self, person: PersonPose) -> tuple[set[str], float]:
        raised: set[str] = set()
        amounts: list[float] = []
        pairs = {
            "left": (LEFT_SHOULDER, LEFT_WRIST),
            "right": (RIGHT_SHOULDER, RIGHT_WRIST),
        }
        for name, (shoulder_index, wrist_index) in pairs.items():
            if not (
                visible(person, shoulder_index, self.keypoint_confidence)
                and visible(person, wrist_index, self.keypoint_confidence)
            ):
                continue
            shoulder_y = float(person.keypoints[shoulder_index][1])
            wrist_y = float(person.keypoints[wrist_index][1])
            amount = shoulder_y - wrist_y
            if amount >= self.shoulder_margin:
                raised.add(name)
                amounts.append(amount)
        return raised, max(amounts, default=0.0)

    def update(self, frame_pose: FramePose) -> list[RawEvent]:
        person = select_subject(frame_pose, self.min_person_confidence)
        hands, amount = self._raised_hands(person) if person is not None else (set(), 0.0)
        emitted: list[RawEvent] = []
        if hands:
            if self._start is None:
                self._start = frame_pose.timestamp
            self._last_time = frame_pose.timestamp
            self._hands.update(hands)
            self._raise_amounts.append(amount)
            self._confidences.append(person.confidence if person else 0.0)
        else:
            event = self._close()
            if event is not None:
                emitted.append(event)
        return emitted

    def _close(self) -> RawEvent | None:
        if self._start is None or self._last_time is None:
            self._reset()
            return None
        end_time = self._last_time
        duration = end_time - self._start
        event = None
        if duration >= self.minimum_duration:
            avg_confidence = sum(self._confidences) / max(len(self._confidences), 1)
            max_raise = max(self._raise_amounts, default=0.0)
            event = RawEvent(
                event_type=self.event_type,
                start_time=self._start,
                peak_time=self._start + duration / 2.0,
                end_time=end_time,
                confidence=clamp01(avg_confidence * (0.75 + min(max_raise / 0.25, 1.0) * 0.25)),
                features={
                    "hands": sorted(self._hands),
                    "raise_amplitude": max_raise,
                    "person_confidence": avg_confidence,
                    "duration_seconds": duration,
                    "wave_extension_ready": True,
                },
            )
        self._reset()
        return event

    def flush(self, end_time: float | None = None) -> list[RawEvent]:
        del end_time
        event = self._close()
        return [event] if event is not None else []

    def _reset(self) -> None:
        self._start = None
        self._last_time = None
        self._hands.clear()
        self._raise_amounts.clear()
        self._confidences.clear()
