import pytest
from pydantic import ValidationError

from src.events.schema import EventRecord, EventType


def test_event_schema_serializes_frontend_contract():
    event = EventRecord(
        event_id=1,
        event_type=EventType.HAND_RAISE,
        start_time=1.0,
        peak_time=1.4,
        end_time=2.0,
        confidence=0.9,
        highlight_score=88.5,
        reason="visible hand raise",
        best_frame="outputs/frames/event_0001.jpg",
        clip="outputs/clips/event_0001.mp4",
    )
    payload = event.as_dict()
    assert payload["event_type"] == "hand_raise"
    assert payload["highlight_score"] == 88.5
    assert {"event_id", "start_time", "peak_time", "end_time", "confidence", "reason"}.issubset(
        payload
    )


def test_event_type_is_strict_enum():
    """Only known EventType members are accepted, raw string or enum."""

    for member in EventType:
        event = _minimal_event(event_type=member.value)
        assert event.event_type == member.value

    with pytest.raises(ValidationError):
        _minimal_event(event_type="dancing")


def test_nan_and_infinity_rejected():
    """Non-finite floats must fail validation on every float field."""

    non_finite = [float("nan"), float("inf"), float("-inf")]
    fields = ("start_time", "peak_time", "end_time", "confidence", "highlight_score")

    for value in non_finite:
        for field in fields:
            kwargs = {field: value}
            with pytest.raises(ValidationError):
                _minimal_event(**kwargs)


def test_timeline_order_enforced():
    """peak_time must lie within [start_time, end_time]."""

    _minimal_event(start_time=1.0, peak_time=1.4, end_time=2.0)  # valid, no raise
    _minimal_event(start_time=1.0, peak_time=1.0, end_time=1.0)  # degenerate, valid

    with pytest.raises(ValidationError, match="peak_time"):
        _minimal_event(start_time=2.0, peak_time=1.0, end_time=3.0)

    with pytest.raises(ValidationError, match="peak_time"):
        _minimal_event(start_time=1.0, peak_time=3.0, end_time=2.0)


def _minimal_event(**overrides) -> EventRecord:
    """Build a schema-valid baseline event with selective overrides."""

    data = {
        "event_id": 1,
        "event_type": EventType.MOTION_PEAK.value,
        "start_time": 1.0,
        "peak_time": 1.4,
        "end_time": 2.0,
        "confidence": 0.9,
    }
    data.update(overrides)
    return EventRecord(**data)
