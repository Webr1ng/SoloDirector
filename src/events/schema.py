"""Stable, frontend-facing Pydantic event schema.

The frontend must depend on this model rather than the internal YOLO output.
Validation targets Pydantic v2: event types are restricted to the known
``EventType`` members, non-finite floats (NaN/Infinity) are rejected, and the
event timeline must satisfy ``start_time <= peak_time <= end_time``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

_allow_inf_nan = False  # shared by every float field below


class EventType(str, Enum):
    STABLE_POSE = "stable_pose"
    HAND_RAISE = "hand_raise"
    JUMP = "jump"
    MOTION_PEAK = "motion_peak"


class EventRecord(BaseModel):
    """The only event shape exposed to the web client."""

    event_id: int = Field(ge=1)
    event_type: EventType
    start_time: float = Field(ge=0, allow_inf_nan=_allow_inf_nan)
    peak_time: float = Field(ge=0, allow_inf_nan=_allow_inf_nan)
    end_time: float = Field(ge=0, allow_inf_nan=_allow_inf_nan)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=_allow_inf_nan)
    highlight_score: float = Field(
        default=0.0, ge=0, le=100, allow_inf_nan=_allow_inf_nan
    )
    reason: str = ""
    best_frame: str | None = None
    clip: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="after")
    def _validate_timeline_order(self) -> "EventRecord":
        """Reject timelines where peak falls outside [start, end]."""

        if self.peak_time < self.start_time:
            raise ValueError("peak_time must be >= start_time")
        if self.end_time < self.peak_time:
            raise ValueError("end_time must be >= peak_time")
        return self

    def as_dict(self) -> dict[str, Any]:
        """Serialize with enum members already stored as their raw values."""

        return self.model_dump(mode="json")
