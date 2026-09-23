from __future__ import annotations

import cv2
import numpy as np

from .config import IdentityConfig
from .types import BBox


class Embedder:
    def __init__(self, cfg: IdentityConfig):
        self.cfg = cfg

    def crop(self, frame: np.ndarray, bbox: BBox, *, wrap: bool = False) -> np.ndarray:
        height, width = frame.shape[:2]
        y1, y2 = max(0, int(bbox.y1)), min(height, int(np.ceil(bbox.y2)))
        if y2 <= y1 or bbox.w <= 0:
            return frame[:0, :0]
        if wrap:
            x1 = int(bbox.x1) % width
            length = min(width, int(np.ceil(bbox.w)))
            xs = np.arange(x1, x1 + length) % width
            return frame[y1:y2][:, xs]
        x1, x2 = max(0, int(bbox.x1)), min(width, int(np.ceil(bbox.x2)))
        return frame[y1:y2, x1:x2]

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                            [0, 180, 0, 256, 0, 256]).flatten()
        norm = np.linalg.norm(hist)
        return (hist / norm if norm > 0 else hist).astype(np.float32)
