"""Deterministic synthetic tests for the stable-pose and jump temporal detectors."""

from __future__ import annotations

import numpy as np
import pytest

from src.events.jump import JumpDetector
from src.events.stable_pose import StablePoseDetector
from src.pose.detector import FramePose, PersonPose

CONF = 0.9
SAMPLE_FPS = 8.0  # Matches config.yaml video.sample_fps.


def make_person(
    hip_y: float = 0.5,
    offset: float = 0.0,
    confidence: float = CONF,
    visible_hips: bool = True,
    visible_ankles: bool = True,
) -> PersonPose:
    """Build a full-body person; ``offset`` moves every keypoint horizontally."""

    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[:, 0] += offset
    keypoints[11:13, 1] = hip_y  # hips
    keypoints[13:15, 1] = hip_y + 0.25  # knees
    keypoints[15:17, 1] = hip_y + 0.5  # ankles
    keypoint_confidence = np.full(17, CONF, dtype=float)
    if not visible_hips:
        keypoint_confidence[11:13] = 0.0
    if not visible_ankles:
        keypoint_confidence[15:17] = 0.0
    return PersonPose(
        bbox=(0.4 + offset, 0.1, 0.6 + offset, hip_y + 0.55),
        confidence=confidence,
        keypoints=keypoints,
        keypoint_confidence=keypoint_confidence,
    )


def make_frame(idx: int, timestamp: float, *persons: PersonPose) -> FramePose:
    return FramePose(
        frame_idx=idx, timestamp=timestamp, persons=list(persons), image_width=1, image_height=1
    )


def run_frames(detector, frames):
    emitted = []
    for frame in frames:
        emitted.extend(detector.update(frame))
    return emitted


def stable_stream(start: float, seconds: float, fps: float = SAMPLE_FPS, **person_kwargs):
    frames = []
    count = int(seconds * fps)
    for i in range(count + 1):
        frames.append(make_frame(i, start + i / fps, make_person(**person_kwargs)))
    return frames


def cfg(**overrides) -> dict:
    return {"events": {"stable_pose": overrides, "jump": overrides}}


# --------------------------------------------------------------------------
# StablePoseDetector
# --------------------------------------------------------------------------


def test_stable_long_hold_emits_full_segment_on_flush():
    detector = StablePoseDetector(cfg())
    frames = stable_stream(10.0, 6.0)
    emitted = run_frames(detector, frames)
    assert emitted == []  # No per-frame or 3-second tick emission; segment stays open.
    emitted = detector.flush()
    assert len(emitted) == 1
    event = emitted[0]
    assert event.event_type == "stable_pose"
    assert event.start_time == pytest.approx(10.0)
    assert event.end_time == pytest.approx(10.0 + 6.0, abs=1 / SAMPLE_FPS)
    assert event.features["duration_seconds"] == pytest.approx(6.0, abs=1 / SAMPLE_FPS)
    assert event.features["movement_amplitude"] <= 0.035


def test_stable_segment_closes_on_motion_and_retains_full_tail():
    detector = StablePoseDetector(cfg())
    emitted = run_frames(detector, stable_stream(0.0, 4.0))
    assert emitted == []
    # A real movement (far beyond the movement threshold) ends the segment.
    emitted = run_frames(detector, [make_frame(99, 4.125, make_person(offset=0.5))])
    assert len(emitted) == 1
    event = emitted[0]
    assert event.start_time == pytest.approx(0.0)
    assert event.end_time == pytest.approx(4.0, abs=1 / SAMPLE_FPS)  # Last stable frame kept.
    assert event.features["duration_seconds"] == pytest.approx(4.0, abs=1 / SAMPLE_FPS)
    # After motion, the next hold starts fresh instead of extending the old segment.
    emitted = run_frames(detector, stable_stream(4.25, 4.0, offset=0.5))
    assert emitted == []
    emitted = detector.flush()
    assert len(emitted) == 1
    assert emitted[0].start_time == pytest.approx(4.25)


def test_stable_accumulated_drift_is_rejected_against_anchor():
    detector = StablePoseDetector(cfg())
    # Each frame drifts a little toward +x; every per-frame step stays under the
    # threshold, but the accumulated displacement from the anchor grows past it.
    step = 0.01
    frames = []
    for i in range(90):  # 90 frames ≈ 11.25 s at 8 fps: far beyond duration_seconds.
        offset = step * i
        frames.append(make_frame(i, i / SAMPLE_FPS, make_person(offset=offset)))
    emitted = run_frames(detector, frames)
    assert emitted == []
    assert detector.flush() == []  # Drifted poses must never become a stable event.
    # Sanity: total displacement (0.89) is far above the threshold, final step is not.
    assert step * 89 > 0.035 > step


def test_stable_disappearance_closes_segment_and_resets():
    detector = StablePoseDetector(cfg())
    emitted = run_frames(detector, stable_stream(0.0, 4.0))
    assert emitted == []
    # Person vanishes for one frame.
    emitted = run_frames(detector, [make_frame(33, 4.125)])
    assert len(emitted) == 1
    assert emitted[0].end_time == pytest.approx(4.0, abs=1 / SAMPLE_FPS)
    # A fresh hold shorter than duration_seconds must not be emitted.
    emitted = run_frames(detector, stable_stream(5.0, 1.0))
    assert emitted == []
    assert detector.flush() == []


def test_stable_short_hold_not_emitted():
    detector = StablePoseDetector(cfg())
    emitted = run_frames(detector, stable_stream(0.0, 2.0))
    assert emitted == []
    assert detector.flush() == []


# --------------------------------------------------------------------------
# JumpDetector
# --------------------------------------------------------------------------


def jump_stream(start: float = 0.0, fps: float = SAMPLE_FPS):
    """Stand still, rise, then come back down — one synthetic jump."""

    frames = []
    hip_ys = [0.5] * 4 + [0.44, 0.38, 0.33, 0.33, 0.40, 0.48, 0.5, 0.5, 0.5]
    for i, hip_y in enumerate(hip_ys):
        frames.append(make_frame(i, start + i / fps, make_person(hip_y=hip_y)))
    return frames


def test_jump_true_jump_emits_single_event():
    detector = JumpDetector(cfg())
    frames = jump_stream()
    # Two identical jumps separated by standing still; only windows may differ.
    emitted = run_frames(detector, frames)
    assert len(emitted) == 1
    event = emitted[0]
    assert event.event_type == "jump"
    assert event.features["vertical_amplitude"] >= 0.08
    assert event.features["descent_amplitude"] >= 0.05
    assert event.peak_time > event.start_time
    assert event.end_time > event.peak_time


def test_jump_missing_person_resets_state():
    detector = JumpDetector(cfg())
    frames = jump_stream()
    # Insert frames without a person in the middle of the descent.
    with_gap = frames[:6] + [make_frame(100, frames[6].timestamp + 0.125)] + frames[6:]
    emitted = run_frames(detector, with_gap)
    assert emitted == []
    assert detector.flush() == []
    # After the reset the detector can still detect a normal jump.
    fresh = JumpDetector(cfg())
    assert len(run_frames(fresh, jump_stream(start=20.0))) == 1


def test_jump_large_timestamp_gap_rejected():
    detector = JumpDetector(cfg())
    frames = jump_stream()
    # Stretch the gap between rise start and apex far beyond max_frame_gap_seconds.
    for frame in frames:
        if frame.timestamp > 0.5:
            frame.timestamp += 5.0
    emitted = run_frames(detector, frames)
    assert emitted == []
    assert detector.flush() == []


def test_jump_changing_ankle_visibility_does_not_distort_signal():
    detector = JumpDetector(cfg())
    frames = []
    hip_ys = [0.5] * 4 + [0.38, 0.33, 0.40, 0.5, 0.5, 0.5]
    for i, hip_y in enumerate(hip_ys):
        # Ankles flicker visible/invisible every frame; hips stay visible.
        person = make_person(hip_y=hip_y, visible_ankles=bool(i % 2 == 0))
        frames.append(make_frame(i, i / SAMPLE_FPS, person))
    emitted = run_frames(detector, frames)
    assert len(emitted) == 1  # Hips-only signal is unaffected by the ankle flicker.
    event = emitted[0]
    assert event.features["vertical_amplitude"] == pytest.approx(0.17, abs=0.01)


def test_jump_hips_hidden_frames_ignored():
    detector = JumpDetector(cfg())
    frames = []
    hip_ys = [0.5, 0.5, 0.38, 0.33, 0.33, 0.40, 0.5, 0.5, 0.5]
    for i, hip_y in enumerate(hip_ys):
        person = make_person(hip_y=hip_y)
        if i in (2, 3, 4):  # Hips invisible exactly during the rise/apex.
            person.keypoint_confidence[11:13] = 0.0
        frames.append(make_frame(i, i / SAMPLE_FPS, person))
    emitted = run_frames(detector, frames)
    assert emitted == []
    assert detector.flush() == []

