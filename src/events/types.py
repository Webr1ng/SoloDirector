"""Internal event candidates emitted by temporal detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RawEvent:
    event_type: str
    start_time: float
    peak_time: float
    end_time: float
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)
