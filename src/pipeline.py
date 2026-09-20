"""End-to-end offline analysis pipeline.

The pipeline intentionally keeps the stages explicit:

    VideoReader -> PoseDetector -> EventEngine -> HighlightScore -> BestFrame -> FFmpeg

Only the final ``EventRecord`` objects are serialized for the frontend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis_types import FrameObservation
from src.config import ensure_runtime_directories, load_config, output_paths
from src.events.event_engine import EventEngine
from src.events.schema import EventRecord
from src.highlight.best_frame import save_best_frame, select_best_frame
from src.highlight.scorer import score_event
from src.pose.detector import PoseDetector
from src.video.clipper import FFmpegClipper
from src.video.reader import VideoReader


@dataclass(slots=True)
class AnalysisRunResult:
    input_path: Path
    events: list[EventRecord]
    events_json: Path
    highlight_video: Path | None
    video_info: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "events_json": str(self.events_json),
            "highlight_video": str(self.highlight_video) if self.highlight_video else None,
            "video_info": self.video_info,
            "events": [event.as_dict() for event in self.events],
        }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_events_json(
    output_path: Path,
    events: list[EventRecord],
    input_path: Path,
    video_info: dict[str, Any],
    root: Path,
) -> None:
    payload = {
        "schema_version": "1.0",
        "project": "SoloDirector",
        "source": _relative_or_absolute(input_path, root),
        "video": video_info,
        "events": [event.as_dict() for event in events],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_video(
    input_path: str | Path,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> AnalysisRunResult:
    """Analyze one local video and write events, best frames, and clips."""

    active_config = config or load_config(config_path)
    ensure_runtime_directories(active_config)
    root = Path(str(active_config["project"]["root"])).resolve()
    source = Path(input_path).expanduser()
    if not source.is_absolute():
        source = (root / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source}")

    reader = VideoReader(source)
    video_info_obj = reader.info()
    video_info = {
        "width": video_info_obj.width,
        "height": video_info_obj.height,
        "fps": video_info_obj.fps,
        "frame_count": video_info_obj.frame_count,
        "duration_seconds": video_info_obj.duration_seconds,
    }
    detector = PoseDetector(active_config)
    engine = EventEngine(active_config)
    observations: list[FrameObservation] = []
    video_config = active_config.get("video", {})
    target_fps = float(video_config.get("sample_fps", 8.0))
    max_frames_value = video_config.get("max_frames")
    max_frames = int(max_frames_value) if max_frames_value is not None else None

    for video_frame in reader.iter_frames(target_fps=target_fps, max_frames=max_frames):
        frame_pose = detector.infer(video_frame.frame, video_frame.frame_idx, video_frame.timestamp)
        observations.append(
            FrameObservation(
                frame_idx=video_frame.frame_idx,
                timestamp=video_frame.timestamp,
                frame=video_frame.frame,
                pose=frame_pose,
            )
        )
        engine.update(frame_pose)

    raw_events = engine.finalize()
    paths = output_paths(active_config)
    output_events: list[EventRecord] = []
    clipper = FFmpegClipper()
    clip_paths: list[Path] = []
    pre_buffer = float(video_config.get("pre_buffer_seconds", 2.0))
    post_buffer = float(video_config.get("post_buffer_seconds", 2.0))

    for event_id, raw_event in enumerate(raw_events, start=1):
        highlight = score_event(raw_event, observations, active_config)
        best_selection = select_best_frame(raw_event, observations, active_config)
        best_frame_path = paths["frames_dir"] / f"event_{event_id:04d}_{raw_event.event_type}.jpg"
        saved_frame = save_best_frame(best_selection, best_frame_path)
        saved_frame_ref = _relative_or_absolute(Path(saved_frame), root) if saved_frame else None

        clip_path: Path | None = None
        clip_error: str | None = None
        target_clip = paths["clips_dir"] / f"event_{event_id:04d}_{raw_event.event_type}.mp4"
        try:
            clip_path = clipper.clip_event(
                source,
                raw_event,
                target_clip,
                pre_buffer=pre_buffer,
                post_buffer=post_buffer,
            )
            clip_paths.append(clip_path)
        except (RuntimeError, OSError, ValueError) as error:
            # Missing FFmpeg should not discard the useful JSON and best frame.
            clip_error = str(error)
        except Exception as error:  # subprocess failures are still recorded per event
            clip_error = f"FFmpeg clip failed: {error}"

        features = dict(raw_event.features)
        features["score_features"] = highlight.features
        features["best_frame_score"] = best_selection.score
        if best_selection.features:
            features["best_frame_features"] = best_selection.features
        if clip_error:
            features["clip_error"] = clip_error
        output_events.append(
            EventRecord(
                event_id=event_id,
                event_type=raw_event.event_type,
                start_time=round(raw_event.start_time, 4),
                peak_time=round(raw_event.peak_time, 4),
                end_time=round(raw_event.end_time, 4),
                confidence=round(max(0.0, min(1.0, raw_event.confidence)), 4),
                highlight_score=highlight.score,
                reason=highlight.reason,
                best_frame=saved_frame_ref,
                clip=_relative_or_absolute(clip_path, root) if clip_path else None,
                features=features,
            )
        )

    highlight_video: Path | None = None
    if clip_paths:
        target_highlight = paths["highlights_dir"] / "highlights.mp4"
        try:
            highlight_video = clipper.merge_clips(clip_paths, target_highlight)
        except (RuntimeError, OSError, ValueError):
            highlight_video = None

    events_json = paths["events_dir"] / "events.json"
    _write_events_json(events_json, output_events, source, video_info, root)
    return AnalysisRunResult(source, output_events, events_json, highlight_video, video_info)
