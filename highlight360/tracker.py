from __future__ import annotations

from .config import TrackConfig
from .types import BBox, Track, TrackPoint


class _LiveTrack:
    def __init__(self, track_id: int, bbox: BBox, frame_idx: int, time_sec: float):
        self.track = Track(track_id=track_id, points=[TrackPoint(frame_idx, bbox, time_sec)])
        self.last_bbox = bbox
        self.missing = 0


class IoUTracker:
    def __init__(self, cfg: TrackConfig, wrap_width: int | None = None):
        self.cfg = cfg
        self.wrap_width = wrap_width
        self._next_id = 0
        self._live: list[_LiveTrack] = []
        self._finished: list[Track] = []

    def _iou(self, a: BBox, b: BBox) -> float:
        shifts = (0, -self.wrap_width, self.wrap_width) if self.wrap_width else (0,)
        return max(a.iou(BBox(b.x1 + dx, b.y1, b.x2 + dx, b.y2)) for dx in shifts)

    def update(self, dets: list[BBox], frame_idx: int, time_sec: float) -> list[Track]:
        pairs = [(self._iou(lt.last_bbox, d), li, di)
                 for li, lt in enumerate(self._live) for di, d in enumerate(dets)]
        matched_live, matched_det = set(), set()
        current = []
        for iou, li, di in sorted(pairs, reverse=True):
            if iou < self.cfg.iou_threshold or li in matched_live or di in matched_det:
                continue
            lt = self._live[li]
            lt.track.points.append(TrackPoint(frame_idx, dets[di], time_sec))
            lt.last_bbox, lt.missing = dets[di], 0
            matched_live.add(li)
            matched_det.add(di)
            current.append(lt.track)
        still_live = []
        for li, lt in enumerate(self._live):
            if li not in matched_live:
                lt.missing += 1
            if lt.missing > self.cfg.max_age:
                self._finalize(lt.track)
            else:
                still_live.append(lt)
        self._live = still_live
        for di, d in enumerate(dets):
            if di not in matched_det:
                lt = _LiveTrack(self._next_id, d, frame_idx, time_sec)
                self._next_id += 1
                self._live.append(lt)
                current.append(lt.track)
        return current

    def _finalize(self, track: Track) -> None:
        if len(track.points) >= self.cfg.min_track_len:
            self._finished.append(track)

    def finalize(self) -> list[Track]:
        for lt in self._live:
            self._finalize(lt.track)
        self._live = []
        return self._finished
