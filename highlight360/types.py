"""贯穿全流程的数据结构。所有模块只依赖这里的类型，方便各自替换实现。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """像素坐标下的检测框 (等距柱状全景图坐标系)。"""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0
    cls: str = "person"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


@dataclass
class TrackPoint:
    frame_idx: int
    bbox: BBox
    time_sec: float = 0.0


@dataclass
class Track:
    """一段连续可见的轨迹，不等同于人物身份。"""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # 该轨迹的聚合身份特征
    person_id: Optional[int] = None  # 聚类后归属的人物 id


@dataclass
class Person:
    """身份聚类后的一个人物, 由多条 Track 合并而来。"""

    person_id: int
    track_ids: list[int] = field(default_factory=list)
    name: str = ""  # 用户最后手动改名; 默认 "人物A" 之类

    def display_name(self) -> str:
        return self.name or f"人物{chr(ord('A') + self.person_id)}"


@dataclass
class ViewSpec:
    projection: str
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    fov_deg: float = 90.0
    flat_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass
class CandidateRegion:
    candidate_id: int
    start_sec: float
    end_sec: float
    view: ViewSpec
    track_ids: list[int] = field(default_factory=list)
    clip_path: str = ""
    rendered_duration_sec: float = 0.0


@dataclass
class ModelHighlight:
    start_sec: float
    end_sec: float
    best_sec: float
    reason: str
    title: str


@dataclass
class HighlightEvent:
    event_id: int
    candidate_id: int
    start_sec: float
    end_sec: float
    best_sec: float
    view: ViewSpec
    reason: str
    caption: str
    involved_track_ids: list[int] = field(default_factory=list)
    person_ids: list[int] = field(default_factory=list)
    clip_path: str = ""
    photo_path: str = ""
    clip_start_sec: float = 0.0
    clip_end_sec: float = 0.0
    candidate_start_sec: float = 0.0
    candidate_end_sec: float = 0.0
    jury_report_path: str = ""
    consensus_support: float = 0.0
    source_candidate_ids: list[int] = field(default_factory=list)
