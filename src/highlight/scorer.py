"""Explainable highlight scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2

from src.analysis_types import FrameObservation
from src.events.types import RawEvent


@dataclass(slots=True)
class HighlightScore:
    score: float
    reason: str
    features: dict[str, float]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _event_observations(
    event: RawEvent, observations: Iterable[FrameObservation]
) -> list[FrameObservation]:
    return [
        observation
        for observation in observations
        if event.start_time - 0.001 <= observation.timestamp <= event.end_time + 0.001
    ]


def _composition_score(observation: FrameObservation | None) -> float:
    if observation is None or observation.pose.dominant_person is None:
        return 0.0
    person = observation.pose.dominant_person
    x1, y1, x2, y2 = person.bbox
    center_x, center_y = person.center
    center_score = 1.0 - min(1.0, abs(center_x - 0.5) * 1.8 + abs(center_y - 0.5) * 0.8)
    edge_margin = min(x1, y1, 1.0 - x2, 1.0 - y2)
    framing_score = _clamp(edge_margin / 0.08) if edge_margin >= 0 else 0.0
    area_score = _clamp((person.area - 0.03) / 0.35)
    return _clamp(center_score * 0.45 + framing_score * 0.25 + area_score * 0.30)


def _sharpness_score(observation: FrameObservation | None, reference: float) -> float:
    if observation is None or observation.frame is None:
        return 0.0
    gray = cv2.cvtColor(observation.frame, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return _clamp(variance / max(reference, 1.0))


def score_event(
    event: RawEvent,
    observations: Iterable[FrameObservation],
    config: dict[str, Any],
) -> HighlightScore:
    """Return a 0-100 score with feature values that explain the result."""

    event_frames = _event_observations(event, observations)
    peak = (
        min(event_frames, key=lambda item: abs(item.timestamp - event.peak_time))
        if event_frames
        else None
    )
    completeness = max(
        (
            observation.pose.dominant_person.completeness()
            for observation in event_frames
            if observation.pose.dominant_person
        ),
        default=float(event.features.get("person_completeness", 0.0)),
    )
    motion_value = max(
        float(event.features.get(key, 0.0))
        for key in (
            "movement_amplitude",
            "raise_amplitude",
            "vertical_amplitude",
            "descent_amplitude",
        )
    )
    # Motion values are normalized image coordinates; 0.25 is already a visibly strong action.
    motion_amplitude = _clamp(motion_value / 0.25)
    confidence = _clamp(event.confidence)
    composition = _composition_score(peak)
    clarity_reference = float(config.get("highlight", {}).get("clarity_reference", 500.0))
    clarity = _sharpness_score(peak, clarity_reference)
    weights = config.get("highlight", {}).get("weights", {})
    weighted = (
        confidence * float(weights.get("confidence", 0.30))
        + motion_amplitude * float(weights.get("motion_amplitude", 0.20))
        + _clamp(completeness) * float(weights.get("completeness", 0.20))
        + composition * float(weights.get("composition", 0.15))
        + clarity * float(weights.get("clarity", 0.15))
    )
    score = round(_clamp(weighted) * 100.0, 2)
    features = {
        "confidence": round(confidence, 4),
        "motion_amplitude": round(motion_amplitude, 4),
        "person_completeness": round(_clamp(completeness), 4),
        "composition": round(composition, 4),
        "clarity": round(clarity, 4),
    }
    reason = (
        f"动作置信度 {confidence:.2f}；人物完整度 {completeness:.2f}；"
        f"动作幅度 {motion_amplitude:.2f}；构图 {composition:.2f}；清晰度 {clarity:.2f}"
    )
    return HighlightScore(score=score, reason=reason, features=features)
