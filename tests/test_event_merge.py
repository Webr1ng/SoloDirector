from src.events.event_engine import merge_raw_events
from src.events.types import RawEvent


def test_adjacent_same_type_events_merge():
    events = merge_raw_events(
        [
            RawEvent("jump", 1.0, 1.2, 1.5, 0.7, {"vertical_amplitude": 0.1}),
            RawEvent("jump", 1.7, 1.9, 2.2, 0.9, {"vertical_amplitude": 0.2}),
            RawEvent("hand_raise", 1.6, 1.8, 2.0, 0.8),
        ],
        merge_gap_seconds=0.3,
    )
    assert len(events) == 2
    jump = next(event for event in events if event.event_type == "jump")
    assert jump.start_time == 1.0
    assert jump.end_time == 2.2
    assert jump.confidence == 0.9
    assert jump.features["vertical_amplitude"] == 0.2
