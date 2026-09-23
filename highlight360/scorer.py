"""按时空邻接生成固定取景的候选视频区域，不计算精彩分数。"""
from __future__ import annotations

import math

from .config import RegionConfig
from .projection import (PanoProjector, shortest_covering_arc,
                         spherical_bounds_margin)
from .types import BBox, CandidateRegion, Track, TrackPoint, ViewSpec


_MIN_DURATION = 2.0
_EPS = 1e-9


def _angular_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    yaw1, pitch1 = map(math.radians, a)
    yaw2, pitch2 = map(math.radians, b)
    cosine = (math.sin(pitch1) * math.sin(pitch2)
              + math.cos(pitch1) * math.cos(pitch2) * math.cos(yaw1 - yaw2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _groups(points: dict[int, list[TrackPoint]], width: int, height: int,
            projection: str, cfg: RegionConfig) -> list[list[int]]:
    """只在同一采样帧建立邻接关系，不把先后路过同位置的人误当互动。"""
    parents = {tid: tid for tid in points}

    def root(tid: int) -> int:
        while parents[tid] != tid:
            parents[tid] = parents[parents[tid]]
            tid = parents[tid]
        return tid

    frames: dict[int, list[tuple[int, BBox]]] = {}
    for tid, samples in points.items():
        for sample in samples:
            frames.setdefault(sample.frame_idx, []).append((tid, sample.bbox))
    for items in frames.values():
        for i, (tid, a) in enumerate(items):
            for other, b in items[i + 1:]:
                if root(tid) == root(other):
                    continue
                if projection == "equirectangular":
                    near = _angular_distance(
                        PanoProjector.equirect_bbox_to_yaw_pitch(a, width, height),
                        PanoProjector.equirect_bbox_to_yaw_pitch(b, width, height)
                    ) <= cfg.gap_deg + _EPS
                else:
                    # 归一化二维框间距：同 x、但位于上下不同场景的框不会误并。
                    dx = max(a.x1 - b.x2, b.x1 - a.x2, 0.0) / width
                    dy = max(a.y1 - b.y2, b.y1 - a.y2, 0.0) / height
                    near = math.hypot(dx, dy) <= cfg.flat_gap + _EPS
                if near:
                    r1, r2 = root(tid), root(other)
                    parents[max(r1, r2)] = min(r1, r2)
    components: dict[int, list[int]] = {}
    for tid in sorted(points):
        components.setdefault(root(tid), []).append(tid)
    return list(components.values())


def _pano_bounds(boxes: list[BBox], width: int, height: int,
                 context: float) -> tuple[float, float, float, float]:
    left, right = shortest_covering_arc([(b.x1, b.x2) for b in boxes], width)
    lon1 = left / width * 360.0 - 180.0 - context
    lon2 = right / width * 360.0 - 180.0 + context
    if lon2 - lon1 > 360.0:
        center = (lon1 + lon2) / 2.0
        lon1, lon2 = center - 180.0, center + 180.0
    bottom = max(-90.0, (0.5 - max(b.y2 for b in boxes) / height) * 180.0 - context)
    top = min(90.0, (0.5 - min(b.y1 for b in boxes) / height) * 180.0 + context)
    return lon1, bottom, lon2, top


def _panorama_view(boxes: list[BBox], width: int, height: int,
                   cfg: RegionConfig) -> ViewSpec | None:
    """用联合球面矩形保留人物间背景，并检查完整区域是否落在透视裁剪面内。"""
    bounds = _pano_bounds(boxes, width, height, cfg.context_deg)
    yaw = ((bounds[0] + bounds[2]) / 2.0 + 180.0) % 360.0 - 180.0
    pitch = (bounds[1] + bounds[3]) / 2.0

    def margin(p: float, fov: float = cfg.max_fov_deg) -> float:
        return spherical_bounds_margin(bounds, yaw, p, fov)

    if margin(pitch) < -_EPS:
        # 固定 yaw 下只搜索 pitch。先保留全范围最优采样，再细化其邻域。
        guesses = [pitch, bounds[1], bounds[3], *range(-90, 91, 5)]
        pitch = max(guesses, key=margin)
        low, high = max(-90.0, pitch - 5.0), min(90.0, pitch + 5.0)
        for _ in range(36):
            p1, p2 = (2 * low + high) / 3.0, (low + 2 * high) / 3.0
            if margin(p1) < margin(p2):
                low = p1
            else:
                high = p2
        refined = (low + high) / 2.0
        pitch = max((pitch, refined), key=margin)
        if margin(pitch) < -_EPS:
            return None
    low, high = cfg.min_fov_deg, cfg.max_fov_deg
    if margin(pitch, low) >= 0:
        high = low
    else:
        for _ in range(30):
            middle = (low + high) / 2.0
            if margin(pitch, middle) >= 0:
                high = middle
            else:
                low = middle
        # 不用四舍五入缩小边界，微小余量也不突破配置上限。
        high = min(cfg.max_fov_deg, high + 1e-5)
    return ViewSpec(projection="equirectangular", yaw_deg=yaw,
                    pitch_deg=pitch, fov_deg=high)


def _flat_view(boxes: list[BBox], width: int, height: int,
               cfg: RegionConfig) -> ViewSpec:
    # 普通图像没有物理 yaw。边距使用二维邻近阈值的一半（归一化图像单位），
    # 不把 context_deg 或像素位置伪装成真实相机角度。
    margin = cfg.flat_gap / 2.0
    box = (max(0.0, min(b.x1 for b in boxes) / width - margin),
           max(0.0, min(b.y1 for b in boxes) / height - margin),
           min(1.0, max(b.x2 for b in boxes) / width + margin),
           min(1.0, max(b.y2 for b in boxes) / height + margin))
    return ViewSpec(projection="flat", flat_box=box)


def _windows(duration: float, window: float, stride: float):
    start = 0.0
    while start < duration:
        end = min(duration, start + window)
        # 不生成不足 2 秒的尾片；必要时向前扩展而不是丢掉最后事件。
        start = min(start, end - _MIN_DURATION)
        yield start, end
        # 第一个到达视频末尾的窗口后就停止，不再产生完全被包含的尾窗。
        if end >= duration:
            break
        start += stride


def _validate(tracks: list[Track], duration: float, width: int, height: int,
              projection: str, cfg: RegionConfig) -> None:
    if projection not in ("flat", "equirectangular"):
        raise ValueError("projection 必须是 flat 或 equirectangular")
    if width <= 0 or height <= 0 or not math.isfinite(duration) or duration < 0:
        raise ValueError("分析图尺寸必须为正，视频时长必须有限且非负")
    values = (cfg.window_sec, cfg.stride_sec, cfg.gap_deg, cfg.flat_gap,
              cfg.context_deg, cfg.min_fov_deg, cfg.max_fov_deg)
    if not all(map(math.isfinite, values)):
        raise ValueError("区域配置必须为有限数值")
    if cfg.window_sec < _MIN_DURATION or not 0 < cfg.stride_sec <= cfg.window_sec:
        raise ValueError("window_sec 必须 >= 2，stride_sec 必须在 (0,window_sec]，以免漏片")
    if not 0 <= cfg.gap_deg <= 180 or cfg.flat_gap < 0 or cfg.context_deg < 0:
        raise ValueError("邻近阈值/上下文必须非负，gap_deg 不得超过 180")
    if not 0 < cfg.min_fov_deg <= cfg.max_fov_deg <= 125:
        raise ValueError("需要 0 < min_fov_deg <= max_fov_deg <= 125")
    ids = set()
    has_points = False
    for track in tracks:
        if track.track_id in ids:
            raise ValueError("track_id 必须唯一")
        ids.add(track.track_id)
        for point in track.points:
            has_points = True
            if not math.isfinite(point.time_sec) or not 0 <= point.time_sec <= duration + _EPS:
                raise ValueError(f"轨迹 {track.track_id} 的 time_sec 必须是视频内的秒数")
            box = point.bbox
            if not all(map(math.isfinite, (box.x1, box.y1, box.x2, box.y2))):
                raise ValueError("bbox 坐标必须有限")
            if box.w <= 0 or box.h <= 0:
                raise ValueError("bbox 必须有正面积；全景跨缝框请使用 x2 > width")
            if box.y2 <= 0 or box.y1 >= height:
                raise ValueError("bbox 必须与分析图有交集")
            if projection == "flat" and (box.x2 <= 0 or box.x1 >= width):
                raise ValueError("平面 bbox 必须与分析图有交集")
    if has_points and duration < _MIN_DURATION:
        raise ValueError("视频短于 2 秒，无法生成最短 2 秒的候选")


def build_candidates(tracks: list[Track], duration_sec: float, width: int,
                     height: int, projection: str,
                     cfg: RegionConfig) -> list[CandidateRegion]:
    """过宽区域分组或细分时间；最短2秒仍无法完整容纳时明确报错。"""
    _validate(tracks, duration_sec, width, height, projection, cfg)
    if not any(t.points for t in tracks):
        return []
    source = {t.track_id: t.points for t in tracks if t.points}
    candidates: list[CandidateRegion] = []

    def select(points: dict[int, list[TrackPoint]], start: float,
               end: float) -> dict[int, list[TrackPoint]]:
        result = {}
        for tid, samples in points.items():
            selected = [p for p in samples if start - _EPS <= p.time_sec <= end + _EPS]
            if selected:
                result[tid] = selected
        return result

    def emit(ids: list[int], start: float, end: float, view: ViewSpec) -> None:
        candidates.append(CandidateRegion(candidate_id=0, start_sec=start,
                                          end_sec=end, view=view, track_ids=sorted(ids)))

    def process(points: dict[int, list[TrackPoint]], start: float, end: float) -> None:
        for ids in _groups(points, width, height, projection, cfg):
            cache: dict[tuple[int, ...], ViewSpec | None] = {}

            def fit(members: list[int]) -> ViewSpec | None:
                key = tuple(sorted(members))
                if key not in cache:
                    boxes = [p.bbox for tid in key for p in points[tid]]
                    cache[key] = (_flat_view(boxes, width, height, cfg)
                                  if projection == "flat" else
                                  _panorama_view(boxes, width, height, cfg))
                return cache[key]

            view = fit(ids)
            if view is not None:
                emit(ids, start, end, view)
                continue
            too_wide = [tid for tid in ids if fit([tid]) is None]
            if too_wide:
                length = end - start
                if length <= _MIN_DURATION + _EPS:
                    print(f"[scorer] 轨迹 {too_wide} 在 [{start:.1f},{end:.1f}]秒内移动范围超过"
                          f"{cfg.max_fov_deg:g}°FOV，跳过该轨迹", flush=True)
                    continue
                if length >= 2 * _MIN_DURATION:
                    middle = (start + end) / 2.0
                    intervals = [(start, middle), (middle, end)]
                else:
                    intervals = [(start, start + _MIN_DURATION),
                                 (end - _MIN_DURATION, end)]
                subset = {tid: points[tid] for tid in ids}
                for sub_start, sub_end in intervals:
                    process(select(subset, sub_start, sub_end), sub_start, sub_end)
                continue

            # 每条轨迹都能单独完整容纳。以各条为种子扩张，保留不同的最大子组，
            # 例如 A-B-C 过宽时优先保留 A-B 和 B-C，而不是把互动链硬切断。
            representatives = {}
            for tid in ids:
                bounds = _pano_bounds([p.bbox for p in points[tid]], width, height, 0.0)
                representatives[tid] = ((bounds[0] + bounds[2]) / 2.0,
                                        (bounds[1] + bounds[3]) / 2.0)
            covers: list[list[int]] = []
            for seed in ids:
                members = [seed]
                neighbors = sorted((tid for tid in ids if tid != seed), key=lambda tid: (
                    _angular_distance(representatives[seed], representatives[tid]), tid))
                for neighbor in neighbors:
                    if fit(members + [neighbor]) is not None:
                        members.append(neighbor)
                chosen = set(members)
                if any(chosen <= set(cover) for cover in covers):
                    continue
                covers = [cover for cover in covers if not set(cover) < chosen]
                covers.append(sorted(members))
            for members in covers:
                view = fit(members)
                assert view is not None  # 已逐个验证，不以 max_fov 截断计算结果。
                emit(members, start, end, view)

    for start, end in _windows(duration_sec, cfg.window_sec, cfg.stride_sec):
        process(select(source, start, end), start, end)
    # 时间细分可能从重叠滑窗生成同一个候选；只去掉完全相同的窗口/轨迹集合。
    unique = {}
    for candidate in candidates:
        key = (round(candidate.start_sec, 9), round(candidate.end_sec, 9),
               tuple(candidate.track_ids))
        unique.setdefault(key, candidate)
    result = [unique[key] for key in sorted(unique)]
    for index, candidate in enumerate(result):
        candidate.candidate_id = index
    return result
