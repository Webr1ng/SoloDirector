from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .audio import RATE as _AUDIO_RATE
from .config import ExportConfig, RegionConfig
from .io_video import VideoReader, VideoWriter, mux_audio, read_audio_pcm
from .reframe import render_view
from .types import CandidateRegion, HighlightEvent, Person


def _candidate_frame_count(candidate: CandidateRegion, fps: int) -> int:
    duration = candidate.end_sec - candidate.start_sec
    if duration < 2 - 1e-9:
        raise ValueError("分区片段不足2秒，无法作为视频模型输入")
    return math.ceil((duration - 1e-9) * fps)


def _render_batch(reader: VideoReader, candidates: list[CandidateRegion],
                  cfg: RegionConfig, paths: list[Path], frame_done: Callable | None = None) -> None:
    first = candidates[0]
    count = _candidate_frame_count(first, cfg.fps)
    with ExitStack() as stack:
        writers = [stack.enter_context(VideoWriter(str(path), cfg.fps, (cfg.size, cfg.size)))
                   for path in paths]
        for _, frame in reader.iter_frames(start_sec=first.start_sec, end_sec=first.end_sec,
                                           sample_fps=cfg.fps):
            for candidate, writer in zip(candidates, writers):
                writer.write(render_view(frame, candidate.view, cfg.size))
            if frame_done is not None:
                frame_done()
        for writer in writers:
            if writer.frame_count != count:
                raise RuntimeError(f"分区视频解码帧数异常：期望{count}，实际{writer.frame_count}")
    for candidate, path in zip(candidates, paths):
        candidate.rendered_duration_sec = count / cfg.fps
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("分区视频超过模型本地上传100MB限制，请缩短分析窗口")


def render_candidates(reader: VideoReader, candidates: list[CandidateRegion], cfg: RegionConfig,
                      directory: Path, progress: Callable[[dict], None] | None = None
                      ) -> Iterator[list[CandidateRegion]]:
    """同窗口流式共用源帧，每批至多八路编码；批次产出后才处理下一批。"""
    groups: dict[tuple[float, float], list[CandidateRegion]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.start_sec, candidate.end_sec), []).append(candidate)
    batches = [group[offset:offset + 8] for group in groups.values()
               for offset in range(0, len(group), 8)]
    total = sum(_candidate_frame_count(batch[0], cfg.fps) for batch in batches)
    done = 0

    def emit():
        if progress is not None:
            progress({"stage": "render", "done": done, "total": total, "unit": "frames"})

    def frame_done():
        nonlocal done
        done += 1
        emit()

    emit()
    for batch in batches:
        paths = [Path(directory) / f"candidate{candidate.candidate_id:04d}.mp4" for candidate in batch]
        _render_batch(reader, batch, cfg, paths, frame_done)
        yield batch


def render_candidate(reader: VideoReader, candidate: CandidateRegion,
                     cfg: RegionConfig, path: Path,
                     progress: Callable[[dict], None] | None = None) -> float:
    """单候选包装器：写入调用者指定的 path，合同与批量渲染一致。"""
    count = _candidate_frame_count(candidate, cfg.fps)
    done = 0

    def emit():
        if progress is not None:
            progress({"stage": "render", "done": done, "total": count, "unit": "frames"})

    def frame_done():
        nonlocal done
        done += 1
        emit()

    emit()
    _render_batch(reader, [candidate], cfg, [path], frame_done)
    return candidate.rendered_duration_sec


def clip_range(event: HighlightEvent, min_sec: float, source_duration: float) -> tuple[float, float]:
    """保留评委选出的完整高光区间；短于 min_sec 时围绕最佳时刻补足上下文。"""
    values = (event.start_sec, event.end_sec, event.best_sec, min_sec, source_duration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("高光剪辑边界和时长必须是有限数值")
    if min_sec <= 0 or source_duration < 2 - 1e-7:
        raise ValueError("最小时长必须为正，原视频至少需要2秒")

    start = max(0.0, min(source_duration, event.start_sec))
    end = max(0.0, min(source_duration, event.end_sec))
    if end <= start:
        raise ValueError("大模型选出的高光区间不在原视频有效范围内")

    duration = min(min_sec, source_duration)
    if end - start < duration - 1e-7:
        center = max(start, min(end, event.best_sec))
        # 可移动的补足窗口必须同时包住模型选中的完整动作区间。
        earliest = max(0.0, end - duration)
        latest = min(start, source_duration - duration)
        start = max(earliest, min(center - duration / 2, latest))
        end = start + duration
    return start, end


def storyboard_clip_ranges(events: list[HighlightEvent], min_sec: float,
                           source_duration: float) -> list[tuple[HighlightEvent, float, float]]:
    """仅允许明确标记的不同方向镜头回放源时间；组与组之间只裁补足的上下文。"""
    grouped: dict[tuple[str, int], list[list]] = {}
    for index, event in enumerate(events):
        key = (("replay", event.replay_group_id) if event.replay_group_id >= 0
               else ("event", index))
        grouped.setdefault(key, []).append([event, *clip_range(event, min_sec, source_duration)])
    groups = sorted(grouped.values(), key=lambda plans: (
        min(plan[0].start_sec for plan in plans), min(plan[0].event_id for plan in plans)))
    for plans in groups:
        plans.sort(key=lambda plan: plan[0].event_id)
    for index in range(1, len(groups)):
        previous, current = groups[index - 1], groups[index]
        previous_selected_end = max(plan[0].end_sec for plan in previous)
        current_selected_start = min(plan[0].start_sec for plan in current)
        if previous_selected_end > current_selected_start + 1e-7:
            raise ValueError("高光事件仍有重叠，请先合并同一时间段的分镜")
        previous_clip_end = max(plan[2] for plan in previous)
        current_clip_start = min(plan[1] for plan in current)
        if previous_clip_end <= current_clip_start + 1e-9:
            continue
        lower = max(previous_selected_end, current_clip_start)
        upper = min(current_selected_start, previous_clip_end)
        boundary = max(lower, min((previous_selected_end + current_selected_start) / 2, upper))
        for plan in previous:
            plan[2] = min(plan[2], boundary)
        for plan in current:
            plan[1] = max(plan[1], boundary)
    return [(event, start, end) for plans in groups for event, start, end in plans]


def export_live_photos(cfg: ExportConfig, reader: VideoReader,
                       events: list[HighlightEvent], progress: Callable[[dict], None] | None = None) -> None:
    plans = []
    if cfg.export_video:
        for event, start, end in storyboard_clip_ranges(
                events, cfg.live_photo_sec, reader.meta.duration_sec):
            # 最后一张画面也须覆盖模型选中的动作终点；浮点误差不能多算一帧。
            count = int(math.ceil((end - start) * cfg.fps - 1e-7))
            if count <= 0:
                raise ValueError("高光分镜短于一帧，无法导出视频")
            plans.append((event, start, end, count))
    # total 只统计各事件短片帧数；合集(reel)与事件帧同源同帧写出，不重复计数，
    # 否则 done<=total 的进度合同会被破坏。
    total = sum(count for _, _, _, count in plans)
    done = 0

    def emit():
        if progress is not None:
            progress({"stage": "export", "done": done, "total": total, "unit": "frames"})

    emit()
    root = Path(cfg.out_dir)
    reel_path = None
    reel_pcm: list = []
    with ExitStack() as stack:
        reel = None
        for event, start, end, count in plans:
            path = root / f"event{event.event_id}_live.mp4"
            if cfg.make_reel and reel is None:
                reel_path = root / "highlight_reel.mp4"
                reel = stack.enter_context(VideoWriter(str(reel_path),
                                                       cfg.fps, (cfg.size, cfg.size)))
            with VideoWriter(str(path), cfg.fps, (cfg.size, cfg.size)) as writer:
                for _, frame in reader.iter_frames(start_sec=start, end_sec=end, sample_fps=cfg.fps):
                    # 短片与照片始终从原片全分辨率裁切，不复用低清候选片段。
                    view = render_view(frame, event.view, cfg.size)
                    writer.write(view)
                    if reel is not None:
                        reel.write(view)
                    done += 1
                    emit()
                if writer.frame_count != count:
                    raise RuntimeError(f"事件{event.event_id}导出帧数不完整")
            pcm = read_audio_pcm(reader.path, start, end) if reader.meta.has_audio else None
            if pcm is not None:
                temporary = path.with_name(path.name + ".muxing.mp4")
                try:
                    mux_audio(str(path), pcm, _AUDIO_RATE, str(temporary))
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                reel_pcm.append(pcm)
            still = render_view(reader.read_at(event.best_sec), event.view, cfg.size)
            ok, encoded = cv2.imencode(".jpg", still, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"事件{event.event_id}照片编码失败")
            photo_path = root / f"event{event.event_id}_still.jpg"
            encoded.tofile(str(photo_path))
            event.clip_path, event.photo_path = path.name, photo_path.name
            event.clip_start_sec, event.clip_end_sec = start, end
    if reel_path is not None and reel_pcm:
        temporary = reel_path.with_name(reel_path.name + ".muxing.mp4")
        try:
            mux_audio(str(reel_path), np.concatenate(reel_pcm), _AUDIO_RATE, str(temporary))
            temporary.replace(reel_path)
        finally:
            temporary.unlink(missing_ok=True)


def write_timeline(path: Path, source: dict, candidates: list[CandidateRegion],
                   events: list[HighlightEvent], people: list[Person],
                   *, status: str, panel: dict | None, model_fps: float,
                   audio: bool = False) -> None:
    data = {
        "schema_version": 3,
        "source": source,
        "analysis": {
            "status": status,
            "method": "parallel_visual_panel" if panel else "regions_only",
            "panel": panel,
            "visual_input": "timestamped_image_sequence" if panel else None,
            "model_sample_fps": model_fps if panel else None,
            "timestamps": "seconds from first displayed source video frame; end-exclusive",
            "audio_analyzed": False,
            "manual_highlight_weights": False,
        },
        "summary": {"candidate_count": len(candidates), "event_count": len(events)},
        "candidates": [asdict(c) for c in candidates],
        "events": [asdict(e) for e in events],
        "people": [dict(person_id=p.person_id, name=p.display_name(), track_ids=p.track_ids,
                        event_ids=[e.event_id for e in events if p.person_id in e.person_ids]) for p in people],
        "media": {"codec": "h264", "pixel_format": "yuv420p", "audio": audio,
                  "color_space": "bt709", "color_range": "limited", "crf": 16, "preset": "veryfast",
                  "native_live_photo": False,
                  "note": "独立MP4与JPG，不是原生Live Photo；评委输入为无声帧序列；"
                          + ("导出含对应时间段音轨" if audio else "全链路无音轨")},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
