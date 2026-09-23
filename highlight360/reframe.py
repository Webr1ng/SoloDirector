"""按候选/高光中保存的固定 ViewSpec 渲染，不根据第一条轨迹重新取景。"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .projection import perspective_from_equirect
from .types import ViewSpec


def render_view(frame: np.ndarray, view: ViewSpec, size: int) -> np.ndarray:
    """全景透视或平面ROI等比补边，输出方形BGR，人物不拉伸。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv-python")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size 必须是正整数")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0:
        raise ValueError("frame 必须是非空三通道 BGR 图像")
    if view.projection == "equirectangular":
        return perspective_from_equirect(frame, view.yaw_deg, view.pitch_deg,
                                          view.fov_deg, size)
    if view.projection != "flat":
        raise ValueError("projection 必须是 flat 或 equirectangular")
    if len(view.flat_box) != 4 or not all(map(math.isfinite, view.flat_box)):
        raise ValueError("flat_box 必须是有限的归一化 xyxy")
    x1, y1, x2, y2 = view.flat_box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("flat_box 必须在 [0,1] 内且有正面积")
    height, width = frame.shape[:2]
    left, top = math.floor(x1 * width), math.floor(y1 * height)
    right, bottom = math.ceil(x2 * width), math.ceil(y2 * height)
    crop = frame[top:bottom, left:right]
    scale = min(size / crop.shape[1], size / crop.shape[0])
    out_width = min(size, max(1, round(crop.shape[1] * scale)))
    out_height = min(size, max(1, round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (out_width, out_height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.zeros((size, size, 3), dtype=frame.dtype)
    x, y = (size - out_width) // 2, (size - out_height) // 2
    result[y:y + out_height, x:x + out_width] = resized
    return result
