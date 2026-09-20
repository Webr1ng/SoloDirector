"""Best-frame selection within an event window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.analysis_types import FrameObservation
from src.events.types import RawEvent


@dataclass(slots=True)
class BestFrameSelection:
    observation: FrameObservation | None
    score: float
    features: dict[str, float]


def _sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(
    event: RawEvent,
    observations: list[FrameObservation],
    config: dict,
) -> BestFrameSelection:
    """Pick the clearest, best-framed frame near the event peak."""

    buffer_seconds = float(config.get("video", {}).get("post_buffer_seconds", 2.0))
    start = max(0.0, event.start_time - min(buffer_seconds, 1.0))
    end = event.end_time + min(buffer_seconds, 1.0)
    candidates = [item for item in observations if start <= item.timestamp <= end]
    if not candidates:
        return BestFrameSelection(None, 0.0, {})

    reference = float(config.get("highlight", {}).get("clarity_reference", 500.0))
    scored: list[tuple[float, FrameObservation, dict[str, float]]] = []
    for observation in candidates:
        person = observation.pose.dominant_person
        if person is None:
            continue
        clarity = max(0.0, min(1.0, _sharpness(observation.frame) / max(reference, 1.0)))
        completeness = person.completeness()
        center_x, center_y = person.center
        composition = max(
            0.0, 1.0 - min(1.0, abs(center_x - 0.5) * 1.7 + abs(center_y - 0.5) * 0.6)
        )
        peak_proximity = max(
            0.0, 1.0 - min(1.0, abs(observation.timestamp - event.peak_time) / 1.5)
        )
        score = clarity * 0.40 + completeness * 0.25 + composition * 0.20 + peak_proximity * 0.15
        scored.append(
            (
                score,
                observation,
                {
                    "clarity": clarity,
                    "person_completeness": completeness,
                    "composition": composition,
                    "peak_proximity": peak_proximity,
                },
            )
        )
    if not scored:
        return BestFrameSelection(None, 0.0, {})
    score, observation, features = max(scored, key=lambda item: item[0])
    return BestFrameSelection(observation, round(score, 4), features)


def save_best_frame(selection: BestFrameSelection, output_path: str | Path) -> str | None:
    if selection.observation is None:
        return None
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), selection.observation.frame):
        raise RuntimeError(f"OpenCV could not write best frame: {target}")
    return str(target)
