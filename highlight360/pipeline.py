from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .clusterer import IdentityClusterer
from .config import Config
from .detector import Detector
from .embedder import Embedder
from .export import export_live_photos, render_candidates, write_timeline
from .experts import review_requirements
from .io_video import VideoReader
from .projection import PanoProjector, deduplicate_boxes
from .scorer import build_candidates
from .tracker import IoUTracker
from .types import BBox, HighlightEvent, ViewSpec
from .panel import ExpertPanel


def _views_close(a: ViewSpec, b: ViewSpec) -> bool:
    """只合并几乎相同的构图；共享旁观者不代表同一方向的事件。"""
    if a.projection != b.projection:
        return False
    if a.projection == "flat":
        return BBox(*a.flat_box).iou(BBox(*b.flat_box)) >= 0.9
    if a.projection != "equirectangular":
        return False
    if not all(map(math.isfinite, (a.yaw_deg, a.pitch_deg, a.fov_deg,
                                   b.yaw_deg, b.pitch_deg, b.fov_deg))):
        return False
    yaw_gap = abs((a.yaw_deg - b.yaw_deg + 180) % 360 - 180)
    if (yaw_gap > 5 or abs(a.pitch_deg - b.pitch_deg) > 5
            or abs(a.fov_deg - b.fov_deg) > 5):
        return False
    # 同时检查球面中心夹角；极区还保留 yaw 门槛以防旋转构图被误并。
    p1, p2 = math.radians(a.pitch_deg), math.radians(b.pitch_deg)
    cosine = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(math.radians(yaw_gap))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) <= 5 + 1e-9


def _event_priority(event: HighlightEvent) -> tuple[float, float, int, float, int]:
    return (event.consensus_support, event.end_sec - event.start_sec,
            len(event.involved_track_ids), event.view.fov_deg, -event.candidate_id)


def deduplicate_events(events: list[HighlightEvent]) -> list[HighlightEvent]:
    kept = []
    for event in sorted(events, key=lambda e: (e.start_sec, e.candidate_id)):
        duplicate_index = None
        for index, previous in enumerate(kept):
            if not _views_close(event.view, previous.view):
                continue
            if not set(event.involved_track_ids).intersection(previous.involved_track_ids):
                continue
            intersection = min(event.end_sec, previous.end_sec) - max(event.start_sec, previous.start_sec)
            shorter = min(event.end_sec - event.start_sec, previous.end_sec - previous.start_sec)
            if intersection / shorter >= 0.7 and abs(event.best_sec - previous.best_sec) <= 0.75:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(event)
        elif _event_priority(event) > _event_priority(kept[duplicate_index]):
            kept[duplicate_index] = event
    kept.sort(key=lambda event: (event.start_sec, event.candidate_id))
    for index, event in enumerate(kept):
        event.event_id = index
    return kept


def _storyboard_view(group: list[HighlightEvent]) -> ViewSpec | None:
    """同一时刻只取一个镜头；相近全景方向扩大构图以覆盖各候选中心。"""
    views = [event.view for event in group]
    if len(views) == 1:
        return views[0]
    if all(view.projection == "flat" for view in views):
        boxes = [view.flat_box for view in views]
        if any(len(box) != 4 or not all(map(math.isfinite, box)) for box in boxes):
            return None
        return ViewSpec("flat", flat_box=(min(box[0] for box in boxes),
                                          min(box[1] for box in boxes),
                                          max(box[2] for box in boxes),
                                          max(box[3] for box in boxes)))
    if not all(view.projection == "equirectangular" and all(map(
            math.isfinite, (view.yaw_deg, view.pitch_deg, view.fov_deg))) for view in views):
        return None
    angles = sorted(view.yaw_deg % 360 for view in views)
    gaps = [right - left for left, right in zip(angles, angles[1:])]
    gaps.append(angles[0] + 360 - angles[-1])
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    arc = 360 - gaps[gap_index]
    pitches = [view.pitch_deg for view in views]
    if arc > 80 or max(pitches) - min(pitches) > 30:
        return None  # 相反方向无法用一个方形镜头完整呈现，保留最有共识的一段。
    center = (angles[(gap_index + 1) % len(angles)] + arc / 2 + 180) % 360 - 180
    fov = min(125.0, max(max(view.fov_deg for view in views), arc + 60,
                         max(pitches) - min(pitches) + 60))
    return ViewSpec("equirectangular", yaw_deg=center,
                    pitch_deg=(min(pitches) + max(pitches)) / 2, fov_deg=fov)


def assemble_storyboard(events: list[HighlightEvent]) -> list[HighlightEvent]:
    """同一源时间只播一次；重合动作合并时间轴并从全景中选一个覆盖视角。"""
    ordered = sorted(events, key=lambda event: (event.start_sec, event.end_sec, event.candidate_id))
    if any(not all(map(math.isfinite, (event.start_sec, event.end_sec, event.best_sec)))
           or event.end_sec <= event.start_sec for event in ordered):
        raise ValueError("高光时间轴包含无效区间")
    groups: list[list[HighlightEvent]] = []
    for event in ordered:
        if groups and event.start_sec < max(item.end_sec for item in groups[-1]) - 1e-7:
            groups[-1].append(event)
        else:
            groups.append([event])
    storyboard = []
    for group in groups:
        best = max(group, key=_event_priority)
        view = _storyboard_view(group)
        if view is None:
            merged = replace(best)
            sources = [best]
        else:
            merged = replace(best, start_sec=min(item.start_sec for item in group),
                             end_sec=max(item.end_sec for item in group), view=view,
                             candidate_start_sec=min(item.candidate_start_sec for item in group),
                             candidate_end_sec=max(item.candidate_end_sec for item in group),
                             involved_track_ids=sorted({track for item in group
                                                        for track in item.involved_track_ids}),
                             person_ids=sorted({person for item in group for person in item.person_ids}))
            sources = group
        merged.event_id = len(storyboard)
        merged.source_candidate_ids = sorted({item.candidate_id for item in sources})
        storyboard.append(merged)
    return storyboard


def _validate_output_directory(root: Path, input_path: str) -> None:
    """允许 GUI 将本次输入源片与结果放在同一新目录；拒绝覆盖其他内容。"""
    root = root.resolve()
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError("输出路径不是目录；请用 --out 指定新的输出目录")

    entries = list(root.iterdir())
    if not entries:
        return

    source = Path(input_path).resolve()
    if (len(entries) == 1 and source.is_file() and source.parent == root
            and entries[0] == source and entries[0].is_file() and not entries[0].is_symlink()):
        return

    raise ValueError("输出目录必须为空或仅包含当前输入视频，请用 --out 指定新的目录；不会覆盖已有结果")


def run(cfg: Config, progress: Callable[[dict], None] | None = None) -> dict:
    def emit(stage, done, total, unit, **fields):
        if progress is not None:
            progress({"stage": stage, "done": done, "total": total, "unit": unit, **fields})

    root = Path(cfg.export.out_dir).resolve()
    _validate_output_directory(root, cfg.input_path)
    with VideoReader(cfg.input_path, cfg.input.projection) as reader:
        meta = reader.meta
        if meta.duration_sec < 2:
            raise ValueError("输入视频不足2秒，无法提交视频模型分析")
        judge = None
        if not cfg.prepare_only:
            judge = ExpertPanel(cfg.panel)
            judge.validate_setup()
        detector = Detector(cfg.detect)
        projector = PanoProjector(cfg.projection.num_views, cfg.projection.view_fov_deg,
                                  cfg.projection.view_size) if meta.is_equirectangular else None
        tracker = None
        embedder = Embedder(cfg.identity) if cfg.identity.enabled else None
        sample_counts: dict[int, int] = {}
        analysis_width = analysis_height = 0
        detect_total = math.ceil(meta.duration_sec * cfg.detect.detect_fps)
        detected = 0
        emit("detect", detected, detect_total, "frames")
        for frame_idx, (time_sec, frame) in enumerate(reader.iter_frames(
            sample_fps=cfg.detect.detect_fps, max_width=cfg.input.analysis_width,
        )):
            analysis_height, analysis_width = frame.shape[:2]
            if tracker is None:
                tracker = IoUTracker(cfg.track, analysis_width if meta.is_equirectangular else None)
            detections: list[BBox] = []
            if projector is not None:
                for yaw, pitch, view in projector.split(frame):
                    detections.extend(projector.view_bbox_to_equirect(
                        box, yaw, analysis_width, analysis_height, pitch_deg=pitch)
                        for box in detector.detect(view))
            else:
                detections = detector.detect(frame)
            detections = deduplicate_boxes(detections, analysis_width, meta.is_equirectangular)
            current = tracker.update(detections, frame_idx, time_sec)
            if embedder is not None:
                for track in current:
                    count = sample_counts.get(track.track_id, 0)
                    if count < cfg.identity.samples_per_track:
                        vector = embedder.embed_crop(embedder.crop(frame, track.points[-1].bbox,
                                                                   wrap=meta.is_equirectangular))
                        track.embedding = vector if count == 0 else track.embedding + vector
                        sample_counts[track.track_id] = count + 1
            detected += 1
            emit("detect", detected, detect_total, "frames")
        emit("detect", detected, detect_total, "frames")
        if tracker is None:
            raise RuntimeError("视频未解码出任何可用帧")
        tracks = tracker.finalize()
        print(f"检测完成：有效轨迹 {len(tracks)}，投影 {cfg.input.projection}", flush=True)
        candidates = build_candidates(tracks, meta.duration_sec, analysis_width, analysis_height,
                                      cfg.input.projection, cfg.region)
        emit("regions", len(candidates), len(candidates), "candidates")
        if len(candidates) > cfg.panel.max_candidates:
            raise ValueError(f"候选共{len(candidates)}段，超过预算{cfg.panel.max_candidates}；"
                             "请缩短输入或显式提高 --max-candidates，不会静默丢弃候选")
        if judge is not None:
            judge.check_budget(len(candidates))
        people = []
        if cfg.identity.enabled and tracks:
            for track in tracks:
                norm = np.linalg.norm(track.embedding)
                if norm > 0:
                    track.embedding = track.embedding / norm
            people = IdentityClusterer(cfg.identity).cluster(tracks)
        track_people = {t.track_id: t.person_id for t in tracks}
        root.mkdir(parents=True, exist_ok=True)
        panel_size = len(judge.experts) if judge is not None else 0
        print(f"候选分区 {len(candidates)} 段；"
              f"{'仅本地预览' if cfg.prepare_only else f'每段由{panel_size}位视觉评委并行评审'}",
              flush=True)
        if judge is not None:
            (root / "jury").mkdir()
        events = []
        candidate_done = calls_done = failed_done = degraded_candidates = 0
        calls_total = len(candidates) * panel_size if judge is not None else 0
        with tempfile.TemporaryDirectory(prefix="h360-regions-", dir=str(root)) as temporary:
            candidate_dir = root / "candidates" if cfg.prepare_only else Path(temporary)
            candidate_dir.mkdir(exist_ok=True)
            for batch in render_candidates(reader, candidates, cfg.region, candidate_dir, progress):
                try:
                    for candidate in batch:
                        clip = candidate_dir / f"candidate{candidate.candidate_id:04d}.mp4"
                        duration = candidate.end_sec - candidate.start_sec
                        if cfg.prepare_only:
                            candidate.clip_path = clip.relative_to(root).as_posix()
                        else:
                            local_done = local_failed = 0
                            local_total = panel_size

                            def judging(update):
                                nonlocal local_done, local_failed, local_total, calls_total
                                local_done = int(update["done"])
                                local_failed = int(update.get("failed", 0))
                                calls_total += int(update["total"]) - local_total
                                local_total = int(update["total"])
                                public_fields = ("id", "model", "status", "attempt",
                                                 "max_attempts", "error_code")
                                experts = [{key: row[key] for key in public_fields if key in row}
                                           for row in update.get("experts", [])
                                           if isinstance(row, dict)]
                                detail = {"detail": update["detail"]} if "detail" in update else {}
                                emit("judging", calls_done + local_done, calls_total, "calls",
                                     experts=experts, failed=failed_done + local_failed,
                                     candidate_done=candidate_done + int(local_done == local_total),
                                     candidate_index=candidate_done + 1, candidate_total=len(candidates),
                                     calls_done=calls_done + local_done,
                                     calls_total=calls_total, **detail)

                            report_path = root / "jury" / f"candidate{candidate.candidate_id:04d}.json"
                            judge.last_report = {}
                            selections = None
                            try:
                                selections = judge.analyze(str(clip), duration, progress=judging,
                                                           remaining_candidates=len(candidates) - candidate_done - 1)
                            finally:
                                report = {**judge.last_report, "candidate_id": candidate.candidate_id,
                                          "source_start_sec": candidate.start_sec,
                                          "source_end_sec": candidate.end_sec,
                                          "rendered_duration_sec": candidate.rendered_duration_sec,
                                          "timestamps": "relative to candidate; add source_start_sec for source time"}
                                if selections is None:
                                    report["status"] = "incomplete"
                                reviews = report.get("reviews", [])
                                report["degraded"] = bool(report.get("degraded") or len(reviews) < panel_size
                                                          or any(r.get("status") != "ok" for r in reviews))
                                degraded_candidates += int(report["degraded"])
                                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                                  allow_nan=False), encoding="utf-8")
                            calls_done += local_done
                            failed_done += local_failed
                            consensus = judge.last_report.get("consensus", [])
                            for selection_index, selection in enumerate(selections):
                                people_ids = sorted({track_people[tid] for tid in candidate.track_ids
                                                     if track_people.get(tid) is not None})
                                agreement = (consensus[selection_index]
                                             if selection_index < len(consensus) else {})
                                events.append(HighlightEvent(
                                    event_id=len(events), candidate_id=candidate.candidate_id,
                                    start_sec=candidate.start_sec + selection.start_sec,
                                    end_sec=candidate.start_sec + selection.end_sec,
                                    best_sec=candidate.start_sec + selection.best_sec,
                                    view=candidate.view, reason=selection.reason, caption=selection.title,
                                    involved_track_ids=candidate.track_ids.copy(), person_ids=people_ids,
                                    candidate_start_sec=candidate.start_sec, candidate_end_sec=candidate.end_sec,
                                    jury_report_path=report_path.relative_to(root).as_posix(),
                                    consensus_support=agreement.get("conservative_min_support", 0.0),
                                ))
                        candidate_done += 1
                        print(f"已处理候选 {candidate_done}/{len(candidates)}", flush=True)
                finally:
                    if not cfg.prepare_only:
                        for candidate in batch:
                            (candidate_dir / f"candidate{candidate.candidate_id:04d}.mp4").unlink(missing_ok=True)
        if not candidates or cfg.prepare_only:
            emit("judging", 0, 0, "calls", experts=[], failed=0, candidate_done=candidate_done,
                 candidate_total=len(candidates), calls_done=0, calls_total=0)
        events = assemble_storyboard(deduplicate_events(events))
        export_live_photos(cfg.export, reader, events, progress=progress)
        source = {**asdict(meta), "path": str(Path(cfg.input_path).resolve()),
                  "analysis_width": analysis_width, "analysis_height": analysis_height,
                  "projection_declared_by_user": True}
        timeline = root / ("candidates.json" if cfg.prepare_only else "全场高光时间轴.json")
        required_review_count, required_family_count = review_requirements(judge.experts) if judge else (0, 0)
        panel_summary = None if judge is None else {
            "experts": judge.experts, "weight_policy": judge.weight_report,
            "api_calls": judge.calls_made, "max_api_calls": cfg.panel.max_api_calls,
            "consensus_threshold": cfg.panel.consensus_threshold,
            "required_review_count": required_review_count,
            "configured_review_count": len(judge.experts),
            "required_family_count": required_family_count,
            "token_limit": "provider_default", "degraded": bool(degraded_candidates),
            "degraded_candidate_count": degraded_candidates,
        }
        write_timeline(timeline, source, candidates, events, people,
                       status="prepared" if cfg.prepare_only else "analyzed",
                       panel=panel_summary, model_fps=cfg.panel.sample_fps,
                       audio=meta.has_audio)
        return {"candidates": len(candidates), "events": len(events),
                "people": len(people), "timeline": str(timeline)}
