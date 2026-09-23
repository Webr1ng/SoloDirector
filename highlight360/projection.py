"""等距柱状全景与方形透视视窗；yaw 向右为正，pitch 向上为正。

全景 bbox 使用连续的 x 区间，允许 x2 > width 表示跨接缝。
像素坐标为图像边界坐标，投影、检测框回映射和取景共用同一旋转。
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .types import BBox


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    return ry @ rx


def shortest_covering_arc(intervals: list[tuple[float, float]],
                          period: float = 360.0) -> tuple[float, float]:
    """覆盖所有连续圆弧的最短圆弧；返回 0 <= start < period 的展开区间。

    输入 end >= start；点可以表示成 (x, x)。使用区间而非仅端点，
    避免把宽框的内部错误当作可删除的最大空隙。
    """
    if not intervals or not math.isfinite(period) or period <= 0:
        raise ValueError("圆弧需要非空区间及正周期")
    pieces = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("圆弧坐标必须有限且 end >= start")
        span = end - start
        if span >= period:
            return 0.0, period
        start %= period
        end = start + span
        if end <= period:
            pieces.append((start, end))
        else:
            pieces.extend(((start, period), (0.0, end - period)))
    merged: list[list[float]] = []
    for start, end in sorted(pieces):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [(merged[(i + 1) % len(merged)][0]
             + (period if i == len(merged) - 1 else 0) - end, i)
            for i, (_, end) in enumerate(merged)]
    gap, i = max(gaps)
    start = merged[(i + 1) % len(merged)][0] % period
    return start, start + period - gap


def _minimum_sinusoid(a: float, b: float, low: float, high: float) -> float:
    """min(a*sin(t) + b*cos(t))，包含区间内部的极值。"""
    result = min(a * math.sin(low) + b * math.cos(low),
                 a * math.sin(high) + b * math.cos(high))
    minimum = math.atan2(a, b) + math.pi
    minimum += math.ceil((low - minimum) / math.tau) * math.tau
    if minimum <= high + 1e-12:
        result = min(result, -math.hypot(a, b))
    return result


def spherical_bounds_margin(bounds: tuple[float, float, float, float],
                            yaw_deg: float, pitch_deg: float,
                            fov_deg: float) -> float:
    """球面矩形到四个透视裁剪面的最小余量；>=0 表示整框在视窗内。

    bounds=(yaw_min, pitch_min, yaw_max, pitch_max)，角度可展开跨缝，
    pitch 在 [-90,90]。解析检查边界和内部极值，不仅检查四角；
    方形 gnomonic 的横、纵 FOV 相同，也会排除相机背面的点。
    """
    lon1, pitch1, lon2, pitch2 = map(math.radians, bounds)
    lat1, lat2 = -pitch2, -pitch1  # 相机/图像的 y 轴向下
    tangent = math.tan(math.radians(fov_deg) / 2.0)
    normals = np.array([[1, 0, tangent], [-1, 0, tangent],
                        [0, 1, tangent], [0, -1, tangent]], dtype=float)
    normals = normals @ _rotation_matrix(yaw_deg, pitch_deg).T
    margins = []
    for nx, ny, nz in normals:
        # cos(latitude) >= 0，因此可依次对 longitude、latitude 求最小值。
        horizontal = _minimum_sinusoid(float(nx), float(nz), lon1, lon2)
        margins.append(_minimum_sinusoid(float(ny), horizontal, lat1, lat2))
    return min(margins) / math.hypot(1.0, tangent)


def perspective_from_equirect(equi: np.ndarray, yaw_deg: float, pitch_deg: float,
                              fov_deg: float, out_size: int) -> np.ndarray:
    """从等距柱状图生成方形透视视窗，正 pitch 朝图像上方。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv-python")
    if equi.ndim < 2 or min(equi.shape[:2]) == 0:
        raise ValueError("全景帧不能为空")
    if out_size <= 0 or not all(map(math.isfinite, (yaw_deg, pitch_deg, fov_deg))):
        raise ValueError("视窗尺寸必须为正，角度必须有限")
    if not 0 < fov_deg < 180 or not -90 <= pitch_deg <= 90:
        raise ValueError("透视 FOV 必须在 (0,180)，pitch 必须在 [-90,90]")
    h, w = equi.shape[:2]
    f = 0.5 * out_size / math.tan(math.radians(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(out_size), np.arange(out_size))
    x = j + 0.5 - out_size / 2.0
    y = i + 0.5 - out_size / 2.0
    vec = np.stack([x, y, np.full_like(x, f)], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    vec = vec @ _rotation_matrix(yaw_deg, pitch_deg).T
    lon = np.arctan2(vec[..., 0], vec[..., 2])
    lat = np.arcsin(np.clip(vec[..., 1], -1, 1))
    # 横向 wrap，但南北极不能互相 wrap；像素中心相对边界坐标偏移 0.5。
    map_x = (((lon / math.tau + 0.5) * w - 0.5) % w + 1).astype(np.float32)
    map_y = np.clip((lat / math.pi + 0.5) * h - 0.5, 0, h - 1).astype(np.float32)
    padded = np.concatenate((equi[:, -1:], equi, equi[:, :1]), axis=1)
    return cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _x_segments(box: BBox, width: float) -> list[tuple[float, float]]:
    if box.w >= width:
        return [(0.0, width)]
    start = box.x1 % width
    end = start + box.w
    if end <= width:
        return [(start, end)]
    return [(start, width), (0.0, end - width)]


def deduplicate_boxes(boxes: list[BBox], width: int, wrap: bool,
                      iou_threshold: float = 0.5) -> list[BBox]:
    """按检测置信度执行同类别 NMS；wrap=True 使用圆柱面区间交并比。

    保留原 BBox 对象和坐标，不把跨缝框扩大为几乎整幅全景的框。
    """
    if width <= 0 or not 0 <= iou_threshold <= 1:
        raise ValueError("width 必须为正，IoU 阈值必须在 [0,1]")
    kept: list[BBox] = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        if not all(math.isfinite(v) for v in (box.x1, box.y1, box.x2, box.y2, box.score)):
            raise ValueError("检测框坐标与置信度必须有限")
        if box.w <= 0 or box.h <= 0:
            raise ValueError("检测框必须有正面积；跨缝请使用 x2 > width")
        duplicate = False
        for other in kept:
            if box.cls != other.cls:
                continue
            if wrap:
                iw = sum(max(0.0, min(b, d) - max(a, c))
                         for a, b in _x_segments(box, width)
                         for c, d in _x_segments(other, width))
                ih = max(0.0, min(box.y2, other.y2) - max(box.y1, other.y1))
                intersection = iw * ih
                union = min(box.w, width) * box.h + min(other.w, width) * other.h - intersection
                iou = intersection / union if union > 0 else 0.0
            else:
                iou = box.iou(other)
            if iou > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


class PanoProjector:
    """水平 num_views 个视窗加南北极视窗，并负责坐标回映射。"""

    def __init__(self, num_views: int, fov_deg: float, view_size: int):
        if num_views <= 0 or view_size <= 0 or not 0 < fov_deg < 180:
            raise ValueError("视窗数量、尺寸必须为正，FOV 必须在 (0,180)")
        self.num_views = num_views
        self.fov_deg = fov_deg
        self.view_size = view_size
        self.yaws = [i * 360.0 / num_views for i in range(num_views)]

    def split(self, equi: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
        """返回 [(yaw, pitch, 视窗图), ...]，正 pitch 朝北极。"""
        angles = [(yaw, 0.0) for yaw in self.yaws] + [(0.0, 90.0), (0.0, -90.0)]
        return [(yaw, pitch, perspective_from_equirect(
            equi, yaw, pitch, self.fov_deg, self.view_size)) for yaw, pitch in angles]

    def view_bbox_to_equirect(self, bbox: BBox, yaw_deg: float,
                              equi_w: int, equi_h: int, pitch_deg: float = 0.0) -> BBox:
        """完整采样四边并纳入纬度极值；含极点的框覆盖全经度。

        每条边按连续经度展开，避免把大于半圆的覆盖误作接缝外的空隙。
        除框内极点外，球面纬度的极值必在边缘，不可仅映射角点。
        """
        if equi_w <= 0 or equi_h <= 0 or bbox.w <= 0 or bbox.h <= 0:
            raise ValueError("图像和检测框必须有正尺寸")
        if (not all(map(math.isfinite, (yaw_deg, pitch_deg, bbox.x1, bbox.y1, bbox.x2, bbox.y2)))
                or not -90 <= pitch_deg <= 90):
            raise ValueError("检测框和角度必须有限，pitch 必须在 [-90,90]")
        center = self.view_size / 2.0
        f = center / math.tan(math.radians(self.fov_deg) / 2.0)
        rotation = _rotation_matrix(yaw_deg, pitch_deg)
        poles = []
        for world_y, equi_y in ((-1.0, 0.0), (1.0, float(equi_h))):
            camera = np.array([0.0, world_y, 0.0]) @ rotation
            if camera[2] > 1e-12:
                px, py = center + f * camera[:2] / camera[2]
                if (bbox.x1 - 1e-9 <= px <= bbox.x2 + 1e-9
                        and bbox.y1 - 1e-9 <= py <= bbox.y2 + 1e-9):
                    poles.append(equi_y)
        corners = np.array([(bbox.x1, bbox.y1), (bbox.x2, bbox.y1),
                            (bbox.x2, bbox.y2), (bbox.x1, bbox.y2)], dtype=float)
        rays = np.column_stack((corners - center, np.full(4, f))) @ rotation.T
        arcs, latitudes = [], list(poles)
        for index in range(4):
            a = rays[index]
            d = rays[(index + 1) % 4] - a
            samples = np.linspace(0, 1, max(3, math.ceil(float(np.linalg.norm(d)) / 4) + 1))
            # d/dt ((a_y+t*d_y)/|a+t*d|) 的分子为常数 + 一次项。
            aa, ad, dd = float(a @ a), float(a @ d), float(d @ d)
            constant = d[1] * aa - a[1] * ad
            slope = d[1] * ad - a[1] * dd
            if slope != 0:
                extremum = -constant / slope
                if 0 < extremum < 1:
                    samples = np.sort(np.append(samples, extremum))
            vec = a + samples[:, None] * d
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            ey = (np.arcsin(np.clip(vec[:, 1], -1, 1)) / math.pi + 0.5) * equi_h
            latitudes.extend((float(ey.min()), float(ey.max())))
            if not poles:
                lon = np.unwrap(np.arctan2(vec[:, 0], vec[:, 2]))
                ex = (lon / math.tau + 0.5) * equi_w
                arcs.append((float(ex.min()), float(ex.max())))
        x1, x2 = (0.0, float(equi_w)) if poles else shortest_covering_arc(arcs, equi_w)
        return BBox(x1, min(latitudes), x2, max(latitudes), score=bbox.score, cls=bbox.cls)

    @staticmethod
    def equirect_bbox_to_yaw_pitch(bbox: BBox, equi_w: int, equi_h: int) -> tuple[float, float]:
        """全景框中心 -> (yaw, pitch)，跨缝中心取 mod，正 pitch 朝上。"""
        if equi_w <= 0 or equi_h <= 0:
            raise ValueError("全景尺寸必须为正")
        yaw = ((bbox.cx % equi_w) / equi_w - 0.5) * 360.0
        pitch = (0.5 - bbox.cy / equi_h) * 180.0
        return yaw, pitch
