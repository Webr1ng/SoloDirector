"""Ultralytics YOLO Pose adapter.

This module deliberately exposes a small, stable internal representation.  Ultralytics
objects never leave this boundary, which lets event rules and the API stay independent
of the exact YOLO model version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import resolve_path

try:  # Import errors are reported when a detector is instantiated, not on --help.
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised only before environment setup
    YOLO = None  # type: ignore[assignment,misc]


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(slots=True)
class PersonPose:
    """One person in normalized image coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray
    keypoint_confidence: np.ndarray = field(default_factory=lambda: np.ones(17, dtype=float))
    class_id: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def completeness(self, threshold: float = 0.25) -> float:
        """Fraction of the 17 COCO keypoints visible above a confidence threshold."""

        if self.keypoint_confidence.size == 0:
            return 0.0
        return float(np.mean(self.keypoint_confidence >= threshold))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(float(v), 6) for v in self.bbox],
            "confidence": round(float(self.confidence), 6),
            "keypoints": np.asarray(self.keypoints, dtype=float).round(6).tolist(),
            "keypoint_confidence": np.asarray(self.keypoint_confidence, dtype=float)
            .round(6)
            .tolist(),
            "completeness": round(self.completeness(), 6),
        }


@dataclass(slots=True)
class FramePose:
    frame_idx: int
    timestamp: float
    persons: list[PersonPose]
    image_width: int
    image_height: int

    @property
    def dominant_person(self) -> PersonPose | None:
        if not self.persons:
            return None
        return max(
            self.persons, key=lambda person: person.confidence * (0.5 + person.completeness() / 2)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(float(self.timestamp), 6),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "persons": [person.to_dict() for person in self.persons],
        }


def select_device(requested: str = "auto") -> str:
    """Choose CUDA, Apple MPS, or CPU without hard-coding a platform."""

    if requested and requested.lower() != "auto":
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PoseDetector:
    """Thin YOLO Pose wrapper that returns :class:`FramePose` records."""

    def __init__(self, config: dict[str, Any], model: Any | None = None) -> None:
        if model is None and YOLO is None:
            raise RuntimeError(
                "Ultralytics is not installed. Run `python -m pip install -r requirements.txt`."
            )
        model_config = config.get("model", {})
        self.config = config
        self.person_confidence = float(model_config.get("person_confidence", 0.35))
        self.keypoint_confidence = float(model_config.get("keypoint_confidence", 0.25))
        self.max_persons = int(model_config.get("max_persons", 8))
        self.device = select_device(str(model_config.get("device", "auto")))
        self.model_path = resolve_path(config, model_config.get("path", "models/yolo11n-pose.pt"))

        if model is not None:
            self.model = model
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        download_if_missing = bool(model_config.get("download_if_missing", True))
        if not self.model_path.exists() and not download_if_missing:
            raise FileNotFoundError(
                f"Pose checkpoint not found at {self.model_path}; "
                "set model.download_if_missing=true "
                "or place an official checkpoint there."
            )
        # Ultralytics downloads official weights when passed a known checkpoint name.
        # No checkpoint is committed to this repository.
        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,), dtype=float)
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def infer(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> FramePose:
        """Run pose inference for one BGR OpenCV frame."""

        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.person_confidence,
            device=self.device,
            verbose=False,
            classes=[0],
        )
        if not results:
            return FramePose(frame_idx, timestamp, [], width, height)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            return FramePose(frame_idx, timestamp, [], width, height)

        xyxy = self._to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
        box_confidence = self._to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        class_ids = self._to_numpy(getattr(boxes, "cls", None)).reshape(-1)
        normalized = self._to_numpy(getattr(keypoints, "xyn", None))
        if normalized.size == 0:
            raw_points = self._to_numpy(getattr(keypoints, "xy", None))
            if raw_points.size:
                normalized = raw_points.reshape(raw_points.shape[0], -1, 2).copy()
                normalized[..., 0] /= max(width, 1)
                normalized[..., 1] /= max(height, 1)
        normalized = normalized.reshape((-1, 17, 2)) if normalized.size else np.empty((0, 17, 2))
        point_confidence = self._to_numpy(getattr(keypoints, "conf", None))
        if point_confidence.size:
            point_confidence = point_confidence.reshape((-1, 17))

        persons: list[PersonPose] = []
        for index, box in enumerate(xyxy[: self.max_persons]):
            x1, y1, x2, y2 = [float(value) for value in box]
            bbox = (
                max(0.0, min(1.0, x1 / max(width, 1))),
                max(0.0, min(1.0, y1 / max(height, 1))),
                max(0.0, min(1.0, x2 / max(width, 1))),
                max(0.0, min(1.0, y2 / max(height, 1))),
            )
            points = normalized[index] if index < len(normalized) else np.zeros((17, 2))
            point_scores = (
                point_confidence[index]
                if index < len(point_confidence)
                else np.ones(17, dtype=float)
            )
            confidence = float(box_confidence[index]) if index < len(box_confidence) else 0.0
            class_id = int(class_ids[index]) if index < len(class_ids) else 0
            persons.append(
                PersonPose(
                    bbox=bbox,
                    confidence=confidence,
                    keypoints=np.asarray(points, dtype=float),
                    keypoint_confidence=np.asarray(point_scores, dtype=float),
                    class_id=class_id,
                )
            )
        return FramePose(frame_idx, timestamp, persons, width, height)
