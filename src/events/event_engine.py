"""Unified streaming event engine and event merging."""

from __future__ import annotations

from typing import Any, Iterable

from src.pose.detector import FramePose

from .hand_raise import HandRaiseDetector
from .jump import JumpDetector
from .stable_pose import StablePoseDetector
from .types import RawEvent


def merge_raw_events(events: Iterable[RawEvent], merge_gap_seconds: float = 0.6) -> list[RawEvent]:
    """Merge adjacent segments of the same type while preserving explainable features."""

    # Keep independent type streams while merging. This matters when, for example, a
    # hand_raise candidate is emitted between two fragments of the same jump.
    by_type: dict[str, list[RawEvent]] = {}
    for incoming in sorted(events, key=lambda item: (item.start_time, item.end_time)):
        stream = by_type.setdefault(incoming.event_type, [])
        if not stream or incoming.start_time > stream[-1].end_time + merge_gap_seconds:
            stream.append(incoming)
            continue
        current = stream[-1]
        current.end_time = max(current.end_time, incoming.end_time)
        if incoming.confidence >= current.confidence:
            current.peak_time = incoming.peak_time
        current.confidence = max(current.confidence, incoming.confidence)
        for key, value in incoming.features.items():
            if key not in current.features:
                current.features[key] = value
            elif isinstance(value, (int, float)) and isinstance(
                current.features[key], (int, float)
            ):
                current.features[key] = max(float(current.features[key]), float(value))
    return sorted(
        (event for stream in by_type.values() for event in stream), key=lambda item: item.start_time
    )


class EventEngine:
    """Run stable-pose, hand-raise, and jump rules over the sampled pose stream."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.detectors = [
            StablePoseDetector(config),
            HandRaiseDetector(config),
            JumpDetector(config),
        ]
        self._events: list[RawEvent] = []
        self._last_timestamp = 0.0

    def update(self, frame_pose: FramePose) -> list[RawEvent]:
        self._last_timestamp = frame_pose.timestamp
        emitted: list[RawEvent] = []
        for detector in self.detectors:
            emitted.extend(detector.update(frame_pose))
        self._events.extend(emitted)
        return emitted

    def finalize(self) -> list[RawEvent]:
        for detector in self.detectors:
            self._events.extend(detector.flush(self._last_timestamp))
        gap = float(self.config.get("events", {}).get("merge_gap_seconds", 0.6))
        self._events = merge_raw_events(self._events, gap)
        return list(self._events)

    def process(self, frames: Iterable[FramePose]) -> list[RawEvent]:
        for frame in frames:
            self.update(frame)
        return self.finalize()

    def reset(self) -> None:
        self._events.clear()
        self._last_timestamp = 0.0
        self.__init__(self.config)
