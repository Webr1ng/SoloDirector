"""Small helpers shared by rule-based temporal detectors."""

from __future__ import annotations

import numpy as np

from src.pose.detector import FramePose, PersonPose


def select_subject(
    frame_pose: FramePose,
    min_confidence: float = 0.0,
    min_completeness: float = 0.0,
) -> PersonPose | None:
    eligible = [
        person
        for person in frame_pose.persons
        if person.confidence >= min_confidence and person.completeness() >= min_completeness
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item.confidence * (0.5 + item.completeness() / 2))


def visible(person: PersonPose, index: int, threshold: float = 0.25) -> bool:
    if index >= len(person.keypoints) or index >= len(person.keypoint_confidence):
        return False
    return float(person.keypoint_confidence[index]) >= threshold


def mean_motion(previous: PersonPose | None, current: PersonPose | None) -> float | None:
    if previous is None or current is None:
        return None
    count = min(len(previous.keypoints), len(current.keypoints))
    if count == 0:
        return None
    prev = np.asarray(previous.keypoints[:count], dtype=float)
    curr = np.asarray(current.keypoints[:count], dtype=float)
    valid = np.ones(count, dtype=bool)
    if len(previous.keypoint_confidence) >= count:
        valid &= np.asarray(previous.keypoint_confidence[:count]) > 0.1
    if len(current.keypoint_confidence) >= count:
        valid &= np.asarray(current.keypoint_confidence[:count]) > 0.1
    if not np.any(valid):
        return None
    return float(np.mean(np.linalg.norm(curr[valid] - prev[valid], axis=1)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
