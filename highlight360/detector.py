from __future__ import annotations

import numpy as np

from .config import DetectConfig
from .types import BBox


class Detector:
    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        try:
            from ultralytics import YOLO
            self._model = YOLO(cfg.model)
        except Exception as exc:
            raise RuntimeError("无法加载YOLO人物检测器；请检查ultralytics及权重文件，不能以空检测代替成功") from exc

    def detect(self, image: np.ndarray) -> list[BBox]:
        result = self._model.predict(
            image, conf=self.cfg.conf, classes=list(self.cfg.classes),
            device=self.cfg.device, verbose=False,
        )[0]
        return [BBox(*map(float, b.xyxy[0].tolist()), score=float(b.conf[0]), cls="person")
                for b in result.boxes]
