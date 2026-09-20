import numpy as np

from src.analysis_types import FrameObservation
from src.events.types import RawEvent
from src.highlight.scorer import score_event
from src.pose.detector import FramePose, PersonPose


def _observation(timestamp: float) -> FrameObservation:
    keypoints = np.array([[0.5, 0.5]] * 17, dtype=float)
    confidences = np.ones(17, dtype=float)
    person = PersonPose((0.25, 0.1, 0.75, 0.95), 0.9, keypoints, confidences)
    pose = FramePose(0, timestamp, [person], 320, 240)
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    frame[80:160, 120:200] = 255
    return FrameObservation(0, timestamp, frame, pose)


def test_highlight_score_is_explainable_and_bounded():
    config = {
        "highlight": {
            "clarity_reference": 500,
            "weights": {
                "confidence": 0.3,
                "motion_amplitude": 0.2,
                "completeness": 0.2,
                "composition": 0.15,
                "clarity": 0.15,
            },
        }
    }
    event = RawEvent("jump", 0.0, 0.5, 1.0, 0.9, {"vertical_amplitude": 0.2})
    result = score_event(event, [_observation(0.5)], config)
    assert 0 <= result.score <= 100
    assert "人物完整度" in result.reason
    assert result.features["person_completeness"] == 1.0
