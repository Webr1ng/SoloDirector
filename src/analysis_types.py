"""Internal types shared by the streaming pipeline and highlight modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameObservation:
    """One sampled video frame and the pose result produced for it."""

    frame_idx: int
    timestamp: float
    frame: np.ndarray
    pose: Any
