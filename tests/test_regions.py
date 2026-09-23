"""只使用合成轨迹/图像，无 YOLO、模型权重或云 API。"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from highlight360.config import RegionConfig, TrackConfig
from highlight360.projection import (PanoProjector, deduplicate_boxes,
                                     shortest_covering_arc, spherical_bounds_margin)
from highlight360.reframe import render_view
from highlight360.scorer import build_candidates
from highlight360.tracker import IoUTracker
from highlight360.types import BBox, Track, TrackPoint, ViewSpec


WIDTH, HEIGHT = 1024, 512


def pano_box(yaw: float, pitch: float = 0.0, angular_width: float = 8.0,
             angular_height: float = 20.0) -> BBox:
    x = ((yaw + 180.0) % 360.0) / 360.0 * WIDTH
    y = (0.5 - pitch / 180.0) * HEIGHT
    dx, dy = angular_width / 360.0 * WIDTH / 2, angular_height / 180.0 * HEIGHT / 2
    return BBox(x - dx, y - dy, x + dx, y + dy, score=0.0)


def track(tid: int, times: list[float], boxes: list[BBox]) -> Track:
    return Track(tid, [TrackPoint(frame_idx=round(t * 100), bbox=b, time_sec=t)
                       for t, b in zip(times, boxes)])


def stationary(tid: int, yaw: float, times=(0.0, 4.0, 7.99), **kwargs) -> Track:
    return track(tid, list(times), [pano_box(yaw, **kwargs) for _ in times])


def camera_samples(box: BBox, view: ViewSpec, nx=71, ny=51) -> np.ndarray:
    """独立于生产旋转函数的全框密集采样，包含四角、边缘和内部点。"""
    x, y = np.meshgrid(np.linspace(box.x1, box.x2, nx),
                       np.linspace(box.y1, box.y2, ny))
    longitude = (x.ravel() / WIDTH - 0.5) * math.tau
    latitude = (0.5 - y.ravel() / HEIGHT) * math.pi
    world = np.column_stack((np.sin(longitude) * np.cos(latitude),
                             -np.sin(latitude), np.cos(longitude) * np.cos(latitude)))
    yaw, pitch = math.radians(view.yaw_deg), math.radians(view.pitch_deg)
    right = [math.cos(yaw), 0, -math.sin(yaw)]
    down = [math.sin(yaw) * math.sin(pitch), math.cos(pitch), math.cos(yaw) * math.sin(pitch)]
    forward = [math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch)]
    return world @ np.array([right, down, forward]).T


class ProjectionTests(unittest.TestCase):
    def test_shortest_arc_across_180(self):
        start, end = shortest_covering_arc([(175, 185), (-178, -170)])
        self.assertAlmostEqual(start, 175)
        self.assertAlmostEqual(end, 190)
        self.assertAlmostEqual((start + end) / 2 % 360, 182.5)

    def test_covering_arc_preserves_wide_interval_interior(self):
        self.assertEqual(shortest_covering_arc([(10, 350)]), (10, 350))
        self.assertEqual(shortest_covering_arc([(0, 360)]), (0, 360))
        self.assertEqual(shortest_covering_arc([(990, 1040), (0, 30)], 1024), (990, 1054))

    def test_wrap_nms_modulo_and_vertical_overlap(self):
        a = BBox(990, 100, 1040, 200, 0.9)
        same = BBox(-34, 100, 16, 200, 0.8)
        shifted = BBox(992, 100, 1042, 200, 0.7)
        below = BBox(990, 210, 1040, 310, 0.6)
        different_class = BBox(990, 100, 1040, 200, 0.5, cls="dog")
        kept = deduplicate_boxes([same, below, a, shifted, different_class], WIDTH, True)
        self.assertEqual(kept, [a, below, different_class])
        self.assertEqual(deduplicate_boxes([a, same], WIDTH, False), [a, same])

    def test_nms_full_width_and_shifted_copies(self):
        a = BBox(0, 0, WIDTH, 100, 0.9)
        b = BBox(-50, 0, WIDTH - 50, 100, 0.8)
        self.assertEqual(deduplicate_boxes([b, a], WIDTH, True), [a])
        self.assertEqual(deduplicate_boxes([], WIDTH, True), [])

    def test_mapping_wrap_and_center_modulo(self):
        projector = PanoProjector(4, 100, 640)
        a = projector.view_bbox_to_equirect(BBox(280, 180, 360, 460, 0.9), 179, WIDTH, HEIGHT)
        b = projector.view_bbox_to_equirect(BBox(280, 180, 360, 460, 0.8), -179, WIDTH, HEIGHT)
        self.assertLess(a.w, WIDTH / 8)
        self.assertGreater(a.x2, WIDTH)
        self.assertEqual(deduplicate_boxes([a, b], WIDTH, True), [a])
        yaw, pitch = projector.equirect_bbox_to_yaw_pitch(BBox(1000, 100, 1080, 180), WIDTH, HEIGHT)
        self.assertAlmostEqual(yaw, (16 / WIDTH - 0.5) * 360)
        self.assertGreater(pitch, 0)

    def test_mapping_samples_horizontal_edge_extrema(self):
        projector = PanoProjector(4, 100, 640)
        mapped = projector.view_bbox_to_equirect(BBox(10, 0, 610, 300), 0, WIDTH, HEIGHT)
        # 上边缘的光轴交点纬度为 50°；仅映射对角点会错失这一最高点。
        self.assertAlmostEqual(mapped.y1, (0.5 - 50 / 180) * HEIGHT, places=7)
        self.assertGreater(mapped.h, 100)

    def test_split_includes_high_latitude_targets_in_both_polar_views(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:24, :, 0] = 240
        frame[-24:, :, 2] = 240
        projector = PanoProjector(4, 100, 65)
        views = projector.split(frame)
        self.assertEqual([(yaw, pitch) for yaw, pitch, _ in views],
                         [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)])
        self.assertTrue(all(not image.any() for _, _, image in views[:4]))
        for yaw, pitch, image in views[4:]:
            channel = 0 if pitch > 0 else 2
            yy, xx = np.where(image[:, :, channel] > 100)
            self.assertGreater(len(xx), 0)
            box = BBox(float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1))
            mapped = projector.view_bbox_to_equirect(box, yaw, WIDTH, HEIGHT, pitch_deg=pitch)
            self.assertEqual((mapped.x1, mapped.x2), (0, WIDTH))
            if pitch > 0:
                self.assertEqual(mapped.y1, 0)
                self.assertLess(mapped.y2, HEIGHT / 4)
            else:
                self.assertEqual(mapped.y2, HEIGHT)
                self.assertGreater(mapped.y1, HEIGHT * 3 / 4)
            # 从极区检测到单次轨迹再到候选，不能因为寿命短或极点框丢失目标。
            tracker = IoUTracker(TrackConfig(), wrap_width=WIDTH)
            tracker.update([mapped], 0, 0)
            candidates = build_candidates(tracker.finalize(), 2, WIDTH, HEIGHT,
                                          "equirectangular", RegionConfig())
            self.assertEqual(len(candidates), 1)
            self.assertGreater(candidates[0].view.pitch_deg * pitch, 0)

    def test_tilted_mapping_covers_edges_and_interior_including_off_axis_poles(self):
        projector = PanoProjector(4, 100, 640)
        for pitch, box in ((35, BBox(31, 19, 603, 287)),
                           (65, BBox(281, 160, 367, 240)),
                           (-65, BBox(281, 400, 367, 480)),
                           (90, BBox(350, 11, 611, 627)),
                           (90, BBox(280, 250, 380, 390)),
                           (-90, BBox(280, 250, 380, 390))):
            with self.subTest(pitch=pitch, box=box):
                yaw = 179
                mapped = projector.view_bbox_to_equirect(box, yaw, WIDTH, HEIGHT, pitch_deg=pitch)
                # 独立生成密集框内光线，验证所有边缘/内部点的经纬度都被覆盖。
                xs, ys = np.meshgrid(np.linspace(box.x1, box.x2, 151),
                                     np.linspace(box.y1, box.y2, 151))
                f = 320 / math.tan(math.radians(50))
                camera = np.column_stack((xs.ravel() - 320, ys.ravel() - 320,
                                          np.full(xs.size, f)))
                y, p = math.radians(yaw), math.radians(pitch)
                axes = np.array([[math.cos(y), 0, -math.sin(y)],
                                 [math.sin(y) * math.sin(p), math.cos(p), math.cos(y) * math.sin(p)],
                                 [math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)]])
                world = camera @ axes
                world /= np.linalg.norm(world, axis=1, keepdims=True)
                ex = (np.arctan2(world[:, 0], world[:, 2]) / math.tau + 0.5) * WIDTH
                ey = (np.arcsin(np.clip(world[:, 1], -1, 1)) / math.pi + 0.5) * HEIGHT
                offset = (ex - mapped.x1) % WIDTH
                offset[np.isclose(offset, WIDTH, rtol=0, atol=1e-7)] = 0
                self.assertLessEqual(float(offset.max()), mapped.w + 1e-7)
                self.assertGreaterEqual(float(ey.min()), mapped.y1 - 1e-7)
                self.assertLessEqual(float(ey.max()), mapped.y2 + 1e-7)
                if abs(pitch) >= 65 and box.x1 < 320 < box.x2:
                    self.assertEqual((mapped.x1, mapped.x2), (0, WIDTH))
                    self.assertEqual(mapped.y1 if pitch > 0 else mapped.y2, 0 if pitch > 0 else HEIGHT)

    def test_positive_pitch_mapping_keeps_seam_deduplication(self):
        projector = PanoProjector(4, 100, 640)
        box = BBox(280, 220, 360, 400, 0.9)
        a = projector.view_bbox_to_equirect(box, 179, WIDTH, HEIGHT, pitch_deg=40)
        b = projector.view_bbox_to_equirect(replace(box, score=0.8), -179, WIDTH, HEIGHT, pitch_deg=40)
        self.assertGreater(a.x2, WIDTH)
        self.assertLess(a.w, WIDTH / 4)
        self.assertLess(a.y2, HEIGHT / 2)
        self.assertEqual(deduplicate_boxes([b, a], WIDTH, True), [a])

    def test_pitch_sign_matches_rendered_center(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:HEIGHT // 2, :, 0] = 240
        frame[HEIGHT // 2:, :, 2] = 240
        up = render_view(frame, ViewSpec("equirectangular", pitch_deg=35, fov_deg=70), 65)
        down = render_view(frame, ViewSpec("equirectangular", pitch_deg=-35, fov_deg=70), 65)
        self.assertEqual(tuple(up[32, 32]), (240, 0, 0))
        self.assertEqual(tuple(down[32, 32]), (0, 0, 240))

    def test_panorama_sampling_does_not_wrap_north_to_south(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:3, :, 1] = 200
        frame[-3:, :, 2] = 250
        north = render_view(frame, ViewSpec("equirectangular", pitch_deg=90), 65)
        self.assertEqual(tuple(north[32, 32]), (0, 200, 0))

    def test_exact_margin_checks_edge_interior_not_just_corners(self):
        # 朝北极的视窗：最大离轴角在低纬水平边内部，也纳入解析检查。
        bounds = (-100.0, 20.0, 100.0, 70.0)
        view = ViewSpec("equirectangular", yaw_deg=0, pitch_deg=65, fov_deg=110)
        box = BBox((bounds[0] + 180) / 360 * WIDTH,
                   (0.5 - bounds[3] / 180) * HEIGHT,
                   (bounds[2] + 180) / 360 * WIDTH,
                   (0.5 - bounds[1] / 180) * HEIGHT)
        points = camera_samples(box, view, nx=301, ny=201)
        tangent = math.tan(math.radians(view.fov_deg / 2))
        sampled = np.min(tangent * points[:, 2] - np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1])))
        sampled /= math.hypot(1, tangent)
        exact = spherical_bounds_margin(bounds, view.yaw_deg, view.pitch_deg, view.fov_deg)
        self.assertLessEqual(exact, sampled + 1e-10)
        self.assertAlmostEqual(exact, sampled, places=4)


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RegionConfig()

    def build(self, tracks, duration=8, projection="equirectangular", cfg=None):
        return build_candidates(tracks, duration, WIDTH, HEIGHT, projection, cfg or self.cfg)

    def assert_box_contained(self, box, view):
        points = camera_samples(box, view)
        self.assertGreater(float(points[:, 2].min()), 0)
        tangent = math.tan(math.radians(view.fov_deg / 2))
        self.assertLessEqual(float(np.max(np.abs(points[:, :2]) / points[:, 2, None])), tangent + 1e-7)

    def assert_all_tracks_covered(self, tracks, candidates):
        for item in tracks:
            for sample in item.points:
                matching = [c for c in candidates if item.track_id in c.track_ids
                            and c.start_sec - 1e-9 <= sample.time_sec <= c.end_sec + 1e-9]
                self.assertTrue(matching, (item.track_id, sample.time_sec))
                for candidate in matching:
                    self.assert_box_contained(sample.bbox, candidate.view)

    def test_no_weights_quality_or_motion_filter_for_solo(self):
        item = stationary(1, 20)
        candidates = self.build([item])
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.track_ids, [1])
        self.assertEqual((candidate.start_sec, candidate.end_sec), (0, 8))
        self.assertEqual(candidate.clip_path, "")
        self.assertEqual(candidate.rendered_duration_sec, 0)
        self.assert_all_tracks_covered([item], candidates)

    def test_single_observation_is_not_discarded(self):
        candidates = self.build([stationary(1, 0, times=(7.99,))])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].end_sec, 8)

    def test_default_tracker_to_candidates_preserves_single_detection(self):
        self.assertEqual(TrackConfig().min_track_len, 1)
        for expire in (False, True):
            with self.subTest(expire=expire):
                tracker = IoUTracker(TrackConfig(max_age=0), wrap_width=WIDTH)
                detected = tracker.update([pano_box(25)], 0, 1.0)[0]
                if expire:
                    tracker.update([], 1, 1.2)
                tracks = tracker.finalize()
                self.assertEqual(len(tracks), 1)
                self.assertEqual(len(tracks[0].points), 1)
                candidates = self.build(tracks)
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0].track_ids, [detected.track_id])
                self.assert_all_tracks_covered(tracks, candidates)

    def test_tracker_minimum_length_remains_configurable(self):
        tracker = IoUTracker(TrackConfig(min_track_len=2))
        tracker.update([pano_box(0)], 0, 0)
        self.assertEqual(tracker.finalize(), [])

    def test_two_simultaneous_directions_are_separate(self):
        items = [stationary(1, -90), stationary(2, 90)]
        candidates = self.build(items)
        self.assertEqual([c.track_ids for c in candidates], [[1], [2]])
        self.assert_all_tracks_covered(items, candidates)

    def test_seam_neighbors_share_stable_view(self):
        items = [stationary(1, 177), stationary(2, -177)]
        candidates = self.build(items)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].track_ids, [1, 2])
        self.assertGreater(abs(candidates[0].view.yaw_deg), 175)
        self.assertLessEqual(candidates[0].view.fov_deg, 125)
        self.assert_all_tracks_covered(items, candidates)

    def test_only_contemporaneous_people_form_edges(self):
        items = [stationary(1, 0, times=(0, 1)), stationary(2, 2, times=(6, 7))]
        self.assertEqual([c.track_ids for c in self.build(items)], [[1], [2]])

    def test_connected_groups_use_all_times_not_only_midpoint(self):
        items = [stationary(1, -25, times=(0,)),
                 stationary(2, 0, times=(0, 7)),
                 stationary(3, 25, times=(7,))]
        candidates = self.build(items)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].track_ids, [1, 2, 3])

    def test_flat_groups_use_both_axes_and_union_all_boxes(self):
        a = track(1, [0, 7], [BBox(100, 30, 150, 90), BBox(50, 20, 160, 100)])
        b = track(2, [0], [BBox(200, 40, 250, 95)])
        c = track(3, [0], [BBox(100, 420, 150, 480)])
        candidates = self.build([a, b, c], projection="flat")
        self.assertEqual([item.track_ids for item in candidates], [[1, 2], [3]])
        box = candidates[0].view.flat_box
        self.assertEqual(box[0], 0)
        self.assertEqual(box[1], 0)
        self.assertGreaterEqual(box[2], 250 / WIDTH)
        self.assertGreaterEqual(box[3], 100 / HEIGHT)
        for candidate in candidates:
            self.assertEqual(candidate.view.yaw_deg, 0)
            self.assertEqual(candidate.view.pitch_deg, 0)
            self.assertTrue(all(0 <= value <= 1 for value in candidate.view.flat_box))

    def test_flat_far_diagonal_boxes_do_not_merge(self):
        a = track(1, [0], [BBox(20, 20, 60, 60)])
        b = track(2, [0], [BBox(200, 120, 240, 160)])
        candidates = self.build([a, b], projection="flat", cfg=replace(self.cfg, flat_gap=0.16))
        self.assertEqual(len(candidates), 2)

    def test_time_sec_not_frame_idx_is_used(self):
        item = Track(7, [TrackPoint(99999, pano_box(0), 9.9)])
        candidates = self.build([item], duration=10)
        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0].start_sec, candidates[0].end_sec), (4, 10))

    def test_clipped_tail_no_fully_repeated_end_windows(self):
        for duration, expected in [(8, [(0, 8)]), (10, [(0, 8), (4, 10)]),
                                   (12, [(0, 8), (4, 12)]),
                                   (13, [(0, 8), (4, 12), (8, 13)])]:
            with self.subTest(duration=duration):
                times = list(range(int(duration))) + [duration]
                item = stationary(1, 0, times=times)
                candidates = self.build([item], duration=duration)
                self.assertEqual([(c.start_sec, c.end_sec) for c in candidates], expected)
                self.assert_all_tracks_covered([item], candidates)

    def test_tail_under_two_seconds_is_extended_backward(self):
        cfg = replace(self.cfg, window_sec=2, stride_sec=2)
        item = stationary(1, 0, times=(0, 2, 4, 4.1))
        candidates = self.build([item], duration=4.1, cfg=cfg)
        self.assertAlmostEqual(candidates[-1].start_sec, 2.1)
        self.assertAlmostEqual(candidates[-1].end_sec, 4.1)
        self.assertTrue(all(c.end_sec - c.start_sec >= 2 - 1e-9 for c in candidates))
        self.assert_all_tracks_covered([item], candidates)

    def test_full_trajectory_bounds_not_just_middle_box(self):
        item = track(1, [0, 4, 7.99], [pano_box(-32, angular_width=12),
                                     pano_box(0), pano_box(32, angular_width=16)])
        candidates = self.build([item])
        self.assertEqual(len(candidates), 1)
        self.assertGreater(candidates[0].view.fov_deg, 95)
        self.assert_all_tracks_covered([item], candidates)

    def test_geometry_includes_vertical_corners_and_context(self):
        # 横宽 80°、纵高 70°，透视角落需要 >80°，不能只用水平角宽。
        box = pano_box(20, angular_width=80, angular_height=70)
        item = track(1, [0, 7], [box, box])
        cfg = replace(self.cfg, context_deg=0)
        candidate = self.build([item], cfg=cfg)[0]
        self.assertGreater(candidate.view.fov_deg, 80)
        self.assert_box_contained(box, candidate.view)
        with_context = self.build([item], cfg=replace(cfg, context_deg=5))[0]
        self.assertGreater(with_context.view.fov_deg, candidate.view.fov_deg)
        self.assert_box_contained(pano_box(20, angular_width=90, angular_height=80), with_context.view)

    def test_off_equator_and_near_pole_geometry(self):
        for yaw, pitch, aw, ah in [(179, 45, 75, 45), (-50, -48, 90, 30), (10, 75, 210, 20)]:
            with self.subTest(pitch=pitch):
                item = track(1, [0, 7], [pano_box(yaw, pitch, aw, ah)] * 2)
                candidates = self.build([item], cfg=replace(self.cfg, context_deg=3))
                self.assertEqual(len(candidates), 1)
                self.assert_all_tracks_covered([item], candidates)

    def test_wide_group_splits_with_overlapping_interaction_context(self):
        items = [stationary(i, yaw) for i, yaw in enumerate((-60, -30, 0, 30, 60), 1)]
        candidates = self.build(items)
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(c.view.fov_deg <= 125 for c in candidates))
        self.assertTrue(any(set(a.track_ids) & set(b.track_ids)
                            for i, a in enumerate(candidates) for b in candidates[i + 1:]))
        self.assertEqual(set().union(*(set(c.track_ids) for c in candidates)), {1, 2, 3, 4, 5})
        self.assert_all_tracks_covered(items, candidates)

    def test_moving_single_track_splits_time_without_losing_end(self):
        times = [0, 2, 4, 6, 8]
        item = track(1, times, [pano_box(yaw) for yaw in (-80, -40, 0, 40, 80)])
        candidates = self.build([item])
        self.assertEqual([(c.start_sec, c.end_sec) for c in candidates], [(0, 4), (4, 8)])
        self.assert_all_tracks_covered([item], candidates)

    def test_three_second_movement_uses_overlapping_two_second_windows(self):
        item = track(1, [0, 1.5, 3], [pano_box(yaw) for yaw in (-65, 0, 65)])
        candidates = self.build([item], duration=3)
        self.assertEqual([(c.start_sec, c.end_sec) for c in candidates], [(0, 2), (1, 3)])
        self.assert_all_tracks_covered([item], candidates)

    def test_impossible_two_second_fixed_view_skips_not_clips(self):
        item = track(1, [0, 2], [pano_box(-80), pano_box(80)])
        candidates = self.build([item], duration=2)
        self.assertEqual(len(candidates), 0)

    def test_oversize_single_box_skips_not_silently_capped(self):
        item = stationary(1, 0, times=(0, 1), angular_width=160)
        candidates = self.build([item], duration=2)
        self.assertEqual(len(candidates), 0)

    def test_invalid_configuration_and_too_short_video(self):
        item = stationary(1, 0, times=(0,))
        with self.assertRaisesRegex(ValueError, "短于 2"):
            self.build([item], duration=1)
        with self.assertRaises(ValueError):
            self.build([item], cfg=replace(self.cfg, stride_sec=9))
        with self.assertRaises(ValueError):
            self.build([item], cfg=replace(self.cfg, max_fov_deg=140))
        self.assertEqual(self.build([], duration=0), [])

    def test_candidates_deterministic_with_unique_ids(self):
        items = [stationary(2, 179), stationary(1, -179), stationary(3, 0)]
        result = self.build(items)
        self.assertEqual(result, self.build(list(reversed(items))))
        self.assertEqual([c.candidate_id for c in result], list(range(len(result))))


class RenderTests(unittest.TestCase):
    def test_wide_roi_letterbox_preserves_square_marker(self):
        frame = np.full((100, 300, 3), 40, dtype=np.uint8)
        frame[25:75, 125:175] = (0, 240, 0)
        view = ViewSpec("flat", flat_box=(1 / 6, 0, 5 / 6, 1))
        rendered = render_view(frame, view, 200)
        self.assertEqual(rendered.shape, (200, 200, 3))
        self.assertTrue(np.all(rendered[:50] == 0))
        self.assertTrue(np.all(rendered[150:] == 0))
        yy, xx = np.where(rendered[:, :, 1] == 240)
        self.assertEqual(xx.max() - xx.min(), yy.max() - yy.min())
        self.assertEqual(tuple(rendered[100, 100]), (0, 240, 0))

    def test_tall_roi_letterbox_preserves_aspect(self):
        frame = np.full((300, 100, 3), (10, 20, 30), dtype=np.uint8)
        rendered = render_view(frame, ViewSpec("flat"), 120)
        self.assertTrue(np.all(rendered[:, :40] == 0))
        self.assertTrue(np.all(rendered[:, 80:] == 0))
        self.assertEqual(tuple(rendered[60, 60]), (10, 20, 30))

    def test_roi_outward_rounding_handles_tiny_valid_box(self):
        frame = np.full((10, 10, 3), (20, 40, 80), dtype=np.uint8)
        rendered = render_view(frame, ViewSpec("flat", flat_box=(0.501, 0.501, 0.502, 0.502)), 16)
        self.assertTrue(np.all(rendered == (20, 40, 80)))

    def test_bad_roi_and_projection_raise(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        for view in [ViewSpec("fisheye"), ViewSpec("flat", flat_box=(0.5, 0, 0.1, 1)),
                     ViewSpec("flat", flat_box=(-0.1, 0, 1, 1))]:
            with self.subTest(view=view), self.assertRaises(ValueError):
                render_view(frame, view, 16)


if __name__ == "__main__":
    unittest.main()
