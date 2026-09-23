import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from highlight360.config import Config, ExportConfig, PanelConfig, RegionConfig, TrackConfig
from highlight360.export import clip_range, export_live_photos, render_candidate, render_candidates
from highlight360.io_video import VideoReader, VideoWriter
from highlight360.panel import _frames
from highlight360.panel_protocol import parse_visual_review
from highlight360.pipeline import deduplicate_events, run
from highlight360.tracker import IoUTracker
from highlight360.types import BBox, CandidateRegion, HighlightEvent, ModelHighlight, ViewSpec


class FakeDetector:
    def __init__(self, cfg):
        pass

    def detect(self, image):
        return [BBox(20, 40, 50, 140, 0.9), BBox(250, 40, 280, 140, 0.9)]


class FakeJudge:
    calls = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_report = {}
        self.experts = [{"id": f"expert{i}", "model": f"model{i}", "family": f"family{i % 2}",
                         "role": "visual"} for i in range(5)]
        self.weight_report = {}
        self.calls_made = 0

    def validate_setup(self):
        pass

    def check_budget(self, count):
        if count * 5 > self.cfg.max_api_calls:
            raise ValueError("超过API调用预算")

    def complete_reviews(self, progress=None, failed=0):
        self.last_report = {"status": "incomplete", "reviews": []}
        experts = [{"id": e["id"], "model": e["model"], "status": "running"} for e in self.experts]
        errors = 0

        def emit(done):
            if progress is not None:
                progress({"stage": "judging", "done": done, "total": 5,
                          "experts": [dict(e) for e in experts], "failed": errors})

        emit(0)
        for done, index in enumerate((2, 0, 4, 1, 3), 1):
            status = "error" if done <= failed else "ok"
            errors += int(status == "error")
            experts[index]["status"] = status
            self.calls_made += 1
            self.last_report["reviews"].append({**self.experts[index], "status": status})
            emit(done)
        self.last_report["status"] = "complete" if failed <= 1 else "incomplete"
        self.last_report["degraded"] = failed > 0

    def analyze(self, path, duration_sec, progress=None, *, remaining_candidates=0):
        with VideoReader(path, "flat") as video:
            assert video.meta.codec == "h264"
            assert 2 <= duration_sec <= video.meta.duration_sec + 0.01
        self.calls.append((path, duration_sec))
        self.complete_reviews(progress)
        return [ModelHighlight(1, 2, 1.5, "人物展示动作", "互动时刻")]


def event(**kwargs):
    data = dict(event_id=0, candidate_id=0, start_sec=1.0, end_sec=2.0, best_sec=1.5,
                view=ViewSpec("flat"), reason="可见互动", caption="互动", involved_track_ids=[0],
                candidate_start_sec=0, candidate_end_sec=4)
    return HighlightEvent(**(data | kwargs))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Windows 临时目录可能是 8.3 短名；resolve 后才能与 reader/writer 内部路径比较。
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "普通视频.mp4"
        with VideoWriter(str(self.video), 10, (320, 180)) as writer:
            for _ in range(40):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                frame[40:140, 20:50] = (0, 0, 220)
                frame[40:140, 250:280] = (220, 0, 0)
                writer.write(frame)
        self.cfg = Config(input_path=str(self.video))
        self.cfg.input.projection = "flat"
        self.cfg.export.out_dir = str(self.root / "results")
        self.cfg.region.size = 128
        self.cfg.export.size = 128
        self.cfg.export.fps = 10
        FakeJudge.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_only_never_constructs_cloud_client(self):
        self.cfg.prepare_only = True
        updates = []
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel") as cloud:
            result = run(self.cfg, progress=updates.append)
        render = [u for u in updates if u["stage"] == "render"]
        self.assertEqual((render[-1]["done"], render[-1]["total"]), (40, 40))
        self.assertTrue(all(u["total"] == 0 for u in updates if u["stage"] == "judging"))
        cloud.assert_not_called()
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertEqual(data["analysis"]["status"], "prepared")
        self.assertEqual(result["events"], 0)
        self.assertEqual(result["candidates"], 2)
        for candidate in data["candidates"]:
            self.assertTrue((Path(self.cfg.export.out_dir) / candidate["clip_path"]).is_file())
        self.assertFalse(data["analysis"]["manual_highlight_weights"])

    def test_polar_detection_pipeline_preserves_view_pitch(self):
        frame = np.zeros((128, 256, 3), dtype=np.uint8)
        frame[:5, :, 1] = 240
        self.video.unlink()
        with VideoWriter(str(self.video), 10, (256, 128)) as writer:
            for _ in range(20):
                writer.write(frame)
        self.cfg.input.projection = "equirectangular"
        self.cfg.projection.view_size = 65
        self.cfg.prepare_only = True

        class PolarDetector:
            def __init__(self, cfg):
                pass

            def detect(self, image):
                yy, xx = np.where(image[:, :, 1] > 100)
                if not len(xx):
                    return []
                return [BBox(float(xx.min()), float(yy.min()),
                             float(xx.max() + 1), float(yy.max() + 1), 0.9)]

        with patch("highlight360.pipeline.Detector", PolarDetector), \
                patch("highlight360.pipeline.ExpertPanel") as cloud:
            result = run(self.cfg)
        cloud.assert_not_called()
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertEqual(result["candidates"], 1)
        self.assertGreater(data["candidates"][0]["view"]["pitch_deg"], 60)

    def test_model_pipeline_outputs_cropped_video_and_reason(self):
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge):
            result = run(self.cfg)
        self.assertEqual(result["events"], 2)
        self.assertEqual(len(FakeJudge.calls), 2)
        self.assertTrue(all(not Path(path).exists() for path, _ in FakeJudge.calls))
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        for item in data["events"]:
            self.assertEqual(item["reason"], "人物展示动作")
            self.assertGreaterEqual(item["start_sec"], 0)
            self.assertLessEqual(item["end_sec"], 4)
            with VideoReader(str(Path(self.cfg.export.out_dir) / item["clip_path"]), "flat") as video:
                self.assertEqual(video.meta.width, 128)
                self.assertAlmostEqual(video.meta.duration_sec, 3.0, places=2)
        self.assertFalse(data["media"]["native_live_photo"])
        self.assertFalse(data["media"]["audio"])
        self.assertNotIn("score", data["events"][0])

    def test_candidate_relative_times_map_to_source(self):
        candidate = CandidateRegion(0, 1.0, 3.0, ViewSpec("flat"), [0])
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch("highlight360.pipeline.build_candidates", return_value=[candidate]):
            result = run(self.cfg)
        item = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))["events"][0]
        self.assertEqual((item["start_sec"], item["end_sec"], item["best_sec"]), (2.0, 3.0, 2.5))
        self.assertEqual((item["clip_start_sec"], item["clip_end_sec"]), (1.0, 4.0))

    def test_fractional_clip_start_does_not_encode_extra_frame(self):
        candidate = CandidateRegion(0, 0.2000000000000003, 3.200000000000001, ViewSpec("flat"), [0])
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch("highlight360.pipeline.build_candidates", return_value=[candidate]):
            result = run(self.cfg)
        self.assertEqual(result["events"], 1)

    def test_fractional_windows_keep_full_model_coverage_without_crossing_source_end(self):
        self.video.unlink()
        with VideoWriter(str(self.video), 10, (320, 180)) as writer:
            for _ in range(41):
                writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
        self.cfg.region.window_sec = self.cfg.region.stride_sec = 2.05
        self.cfg.region.fps = 10
        self.cfg.export.export_video = False
        source_samples, observations = [], []
        original_iter = VideoReader.iter_frames

        def traced_frames(reader, *args, **kwargs):
            for timestamp, frame in original_iter(reader, *args, **kwargs):
                if Path(reader.path) == self.video and kwargs.get("sample_fps") == 10:
                    source_samples.append((kwargs["start_sec"], kwargs["end_sec"], timestamp))
                yield timestamp, frame

        def analyze(judge, path, duration_sec, progress=None, *, remaining_candidates=0):
            judge.complete_reviews(progress)
            with VideoReader(path, "flat") as clip:
                self.assertAlmostEqual(clip.meta.duration_sec, 2.1)
                self.assertEqual(clip.meta.frame_count, 21)
            self.assertAlmostEqual(duration_sec, 2.05)
            _, audit = _frames(path, duration_sec,
                               PanelConfig(sample_fps=10, frame_limit=64, frame_width=64))
            observations.append(audit)
            proposal = {"start_sec": 2.0, "end_sec": duration_sec, "best_sec": 2.025,
                        "reason": "尾段可见动作", "title": "尾段"}
            facts = [{"start_sec": 2.0, "end_sec": duration_sec, "description": "尾段有人抬手"}]
            review = parse_visual_review(json.dumps({"facts": facts, "highlights": [proposal]}), duration_sec)
            with self.assertRaises(ValueError):
                parse_visual_review(json.dumps({"facts": facts, "highlights": [proposal | {"end_sec": 2.1}]}), duration_sec)
            return [ModelHighlight(**review["highlights"][0])]

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch.object(FakeJudge, "analyze", analyze), \
                patch.object(VideoReader, "iter_frames", traced_frames):
            result = run(self.cfg)
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertEqual(len(data["candidates"]), 4)
        for tid in (0, 1):
            windows = [c for c in data["candidates"] if tid in c["track_ids"]]
            self.assertEqual([(c["start_sec"], c["end_sec"]) for c in windows],
                             [(0.0, 2.05), (2.05, 4.1)])
        self.assertTrue(all(abs(c["rendered_duration_sec"] - 2.1) < 1e-9 for c in data["candidates"]))
        self.assertEqual(len(observations), 4)
        self.assertTrue(all(a["frame_timestamps"][-1] == 2.0 for a in observations))
        self.assertTrue(all(t < a["duration_sec"] for a in observations for t in a["frame_timestamps"]))
        self.assertEqual(len(source_samples), 42)
        self.assertTrue(all(start <= time < end <= 4.1 for start, end, time in source_samples))
        self.assertAlmostEqual(max(time for _, _, time in source_samples), 4.05)
        self.assertEqual(max(e["end_sec"] for e in data["events"]), 4.1)
        for item in data["events"]:
            report = json.loads((Path(self.cfg.export.out_dir) / item["jury_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["rendered_duration_sec"], 2.1)
            self.assertLessEqual(item["end_sec"], report["source_end_sec"])

    def test_render_rejects_short_effective_duration_even_if_encoding_rounds_to_two(self):
        candidate = CandidateRegion(0, 0, 1.95, ViewSpec("flat"), [0])
        with VideoReader(str(self.video), "flat") as reader, \
                self.assertRaisesRegex(ValueError, "不足2秒"):
            render_candidate(reader, candidate, self.cfg.region, self.root / "short.mp4")
        self.assertEqual(candidate.end_sec, 1.95)
        self.assertFalse((self.root / "short.mp4").exists())

    def test_api_failure_does_not_write_fake_success(self):
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch.object(FakeJudge, "analyze", side_effect=RuntimeError("API failure")):
            with self.assertRaisesRegex(RuntimeError, "API failure"):
                run(self.cfg)
        self.assertFalse((Path(self.cfg.export.out_dir) / "全场高光时间轴.json").exists())
        self.assertFalse(list(Path(self.cfg.export.out_dir).glob("h360-regions-*")))

    def test_budget_is_checked_before_any_upload(self):
        self.cfg.panel.max_candidates = 1
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge):
            with self.assertRaisesRegex(ValueError, "超过预算"):
                run(self.cfg)
        self.assertFalse(FakeJudge.calls)

    def test_existing_output_is_not_overwritten(self):
        output = Path(self.cfg.export.out_dir)
        output.mkdir()
        sentinel = output / "user-data.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            run(self.cfg)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_output_directory_may_contain_only_its_input_video(self):
        output = self.root / "recording-results"
        output.mkdir()
        source = output / "source.mp4"
        self.video.replace(source)
        original = source.read_bytes()
        self.cfg.input_path = str(source)
        self.cfg.export.out_dir = str(output)
        self.cfg.prepare_only = True

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel") as cloud:
            result = run(self.cfg)

        cloud.assert_not_called()
        self.assertTrue(source.is_file())
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue(Path(result["timeline"]).is_file())
        self.assertEqual(Path(result["timeline"]).parent, output)

    def test_output_directory_with_input_and_other_files_is_still_rejected(self):
        output = self.root / "recording-results"
        output.mkdir()
        source = output / "source.mp4"
        self.video.replace(source)
        sentinel = output / "user-data.txt"
        sentinel.write_text("keep")
        self.cfg.input_path = str(source)
        self.cfg.export.out_dir = str(output)

        with self.assertRaisesRegex(ValueError, "不会覆盖"):
            run(self.cfg)

        self.assertTrue(source.is_file())
        self.assertEqual(sentinel.read_text(), "keep")

    def test_no_video_still_analyzes_candidate_clips(self):
        self.cfg.export.export_video = False
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge):
            result = run(self.cfg)
        self.assertEqual(result["events"], 2)
        self.assertEqual(len(FakeJudge.calls), 2)
        self.assertFalse(list(Path(self.cfg.export.out_dir).glob("*.mp4")))

    def test_progress_counts_real_work_and_reuses_source_for_reel(self):
        updates, source_reads, written = [], [], []
        original_iter, original_write = VideoReader.iter_frames, VideoWriter.write

        def traced_frames(reader, *args, **kwargs):
            if Path(reader.path) == self.video:
                source_reads.append(kwargs)
            yield from original_iter(reader, *args, **kwargs)

        def traced_write(writer, frame):
            original_write(writer, frame)
            # 进度只统计事件短片帧；合集(reel)与事件帧同源写出，不计入 done/total。
            if (Path(writer.path).parent == Path(self.cfg.export.out_dir)
                    and Path(writer.path).name != "highlight_reel.mp4"):
                written.append(writer.path)

        def progress(update):
            updates.append(update)
            if update["stage"] == "export":
                self.assertEqual(update["done"], len(written))

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch.object(VideoReader, "iter_frames", traced_frames), \
                patch.object(VideoWriter, "write", traced_write):
            result = run(self.cfg, progress=progress)
        stages = list(dict.fromkeys(u["stage"] for u in updates))
        self.assertEqual(stages, ["detect", "regions", "render", "judging", "export"])
        for stage, total in (("detect", 20), ("render", 40), ("judging", 10), ("export", 60)):
            rows = [u for u in updates if u["stage"] == stage]
            self.assertEqual(rows[0]["done"], 0)
            self.assertEqual(rows[-1]["done"], total)
            self.assertTrue(all(u["total"] == total for u in rows))
            self.assertEqual([u["done"] for u in rows], sorted(u["done"] for u in rows))
        judging = [u for u in updates if u["stage"] == "judging"]
        self.assertEqual([u["done"] for u in judging], list(range(6)) + list(range(5, 11)))
        self.assertTrue(all(len(u["experts"]) == 5 and u["unit"] == "calls" for u in judging))
        self.assertEqual(judging[-1]["candidate_done"], 2)
        self.assertEqual(judging[-1]["calls_done"], judging[-1]["calls_total"])
        self.assertTrue(all(e["status"] == "running" for e in judging[0]["experts"]))
        self.assertEqual(judging[1]["experts"][2]["status"], "ok")
        self.assertEqual(len(source_reads), 4)
        self.assertEqual(sum(r.get("end_sec") == 4 and r.get("start_sec") == 0 for r in source_reads), 1)
        self.assertTrue(all("max_width" not in r for r in source_reads[1:]))
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertEqual(data["analysis"]["method"], "parallel_visual_panel")
        panel = data["analysis"]["panel"]
        self.assertEqual(panel["configured_review_count"], 5)
        self.assertEqual(panel["required_review_count"], 4)
        self.assertEqual(panel["token_limit"], "provider_default")
        self.assertEqual(panel["api_calls"], 10)
        self.assertFalse(panel["degraded"])
        self.assertNotIn("requires_all_judges", panel)

    def test_real_panel_retry_progress_counts_calls_not_reviewers(self):
        from highlight360.experts import EXPERTS
        counts, updates = {}, []
        self.cfg.panel.allow_cloud_upload = True
        self.cfg.panel.max_api_calls = 12
        self.cfg.panel.frame_limit = 4

        class Client:
            def __init__(self, *args):
                pass

            def complete(self, model, content):
                counts[model] = counts.get(model, 0) + 1
                bad = model in {EXPERTS[1]["model"], EXPERTS[4]["model"]} and counts[model] == 1
                text = "invalid" if bad else json.dumps({
                    "facts": [{"start_sec": 0, "end_sec": 4, "description": "人物做手势"}],
                    "highlights": [{"start_sec": 1, "end_sec": 2, "best_sec": 1.5,
                                    "reason": "可见表达", "title": "手势"}]})
                return {"text": text, "usage": {"total_tokens": 10}, "latency_sec": 0.1,
                        "response_model": model, "request_id": "offline"}

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.panel.ChatClient", Client), \
                patch("highlight360.panel.resolve_api_environment", return_value=("https://example.invalid/v1", "fake")), \
                patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")):
            result = run(self.cfg, progress=updates.append)
        self.assertEqual(result["events"], 2)
        judging = [u for u in updates if u["stage"] == "judging"]
        self.assertEqual([u["done"] for u in judging], sorted(u["done"] for u in judging))
        self.assertTrue(all(u["done"] <= u["total"] for u in judging))
        self.assertTrue(any(u["done"] == 5 and u["candidate_done"] == 0 for u in judging))
        self.assertEqual((judging[-1]["done"], judging[-1]["total"], judging[-1]["candidate_done"]), (12, 12, 2))
        timeline = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertEqual(timeline["analysis"]["panel"]["api_calls"], 12)
        report = json.loads((Path(self.cfg.export.out_dir) / "jury/candidate0000.json").read_text(encoding="utf-8"))
        self.assertEqual(report["retry_count"], 2)
        self.assertEqual(sum(len(r["attempts"]) for r in report["reviews"]), 7)

    def test_zero_candidates_reports_zero_totals_without_requests(self):
        updates = []
        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch("highlight360.pipeline.build_candidates", return_value=[]):
            result = run(self.cfg, progress=updates.append)
        self.assertEqual(result["candidates"], 0)
        self.assertFalse(FakeJudge.calls)
        for stage in ("regions", "render", "judging", "export"):
            rows = [u for u in updates if u["stage"] == stage]
            self.assertTrue(rows)
            self.assertTrue(all(u["done"] == u["total"] == 0 for u in rows))
        self.assertNotIn("server_complete", [u["stage"] for u in updates])

    def test_model_failure_audits_completed_requests_and_cleans_batch(self):
        updates = []

        def fail(judge, path, duration_sec, progress=None, *, remaining_candidates=0):
            judge.complete_reviews(progress, failed=2)
            raise RuntimeError("not enough reviewers")

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch.object(FakeJudge, "analyze", fail), self.assertRaisesRegex(RuntimeError, "reviewers"):
            run(self.cfg, progress=updates.append)
        root = Path(self.cfg.export.out_dir)
        report = json.loads((root / "jury/candidate0000.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(report["degraded"])
        self.assertEqual(sum(r["status"] == "error" for r in report["reviews"]), 2)
        self.assertEqual((updates[-1]["done"], updates[-1]["total"], updates[-1]["failed"]), (5, 10, 2))
        self.assertFalse(list(root.glob("h360-regions-*")))
        self.assertFalse(list(root.glob("*.mp4")))
        self.assertFalse((root / "全场高光时间轴.json").exists())

    def test_degraded_success_is_audited_and_progress_is_public_only(self):
        updates = []

        def analyze(judge, path, duration_sec, progress=None, *, remaining_candidates=0):
            def noisy_progress(update):
                for expert in update["experts"]:
                    expert["private"] = "private-value"
                progress({**update, "api_key": "private-value", "text": "long-model-response"})
            judge.complete_reviews(noisy_progress, failed=1)
            return []

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch.object(FakeJudge, "analyze", analyze):
            result = run(self.cfg, progress=updates.append)
        data = json.loads(Path(result["timeline"]).read_text(encoding="utf-8"))
        self.assertTrue(data["analysis"]["panel"]["degraded"])
        self.assertEqual(data["analysis"]["panel"]["degraded_candidate_count"], 2)
        self.assertNotIn("private-value", json.dumps(updates))
        self.assertNotIn("long-model-response", json.dumps(updates))
        judging = [u for u in updates if u["stage"] == "judging"]
        self.assertEqual(judging[-1]["failed"], 2)
        for path in Path(self.cfg.export.out_dir).glob("jury/*.json"):
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["degraded"])

    def test_batch_render_limits_open_writers_without_dropping_candidates(self):
        candidates = [CandidateRegion(i * 2, 0, 4, ViewSpec("flat"), [i]) for i in range(9)]
        self.cfg.region.fps = 2
        active, peaks, reads, updates = set(), [], [], []
        original_iter = VideoReader.iter_frames

        class CountedWriter(VideoWriter):
            def __init__(writer, *args):
                super().__init__(*args)
                active.add(writer.path)
                peaks.append(len(active))

            def close(writer):
                try:
                    super().close()
                finally:
                    active.discard(writer.path)

        def traced_frames(reader, **kwargs):
            reads.append(kwargs)
            yield from original_iter(reader, **kwargs)

        with VideoReader(str(self.video), "flat") as reader, \
                patch("highlight360.export.VideoWriter", CountedWriter), \
                patch.object(VideoReader, "iter_frames", traced_frames):
            batches = render_candidates(reader, candidates, self.cfg.region, self.root, updates.append)
            self.assertEqual(len(next(batches)), 8)
            self.assertEqual(len(reads), 1)
            self.assertFalse(active)
            self.assertEqual(len(next(batches)), 1)
            self.assertEqual(list(batches), [])
        self.assertEqual(max(peaks), 8)
        self.assertEqual(len(reads), 2)
        self.assertEqual((updates[-1]["done"], updates[-1]["total"]), (16, 16))
        for candidate in candidates:
            self.assertTrue((self.root / f"candidate{candidate.candidate_id:04d}.mp4").is_file())
            self.assertEqual(candidate.rendered_duration_sec, 4)

    def test_previous_group_is_cleaned_before_next_render(self):
        candidates = [CandidateRegion(i, start, start + 2, ViewSpec("flat"), [i])
                      for i, start in enumerate((0, 0, 2))]
        self.cfg.export.export_video = False
        original_iter = VideoReader.iter_frames

        def traced_frames(reader, **kwargs):
            if Path(reader.path) == self.video and kwargs.get("start_sec") == 2:
                self.assertFalse(list(Path(self.cfg.export.out_dir).glob("h360-regions-*/candidate000[01].mp4")))
            yield from original_iter(reader, **kwargs)

        with patch("highlight360.pipeline.Detector", FakeDetector), \
                patch("highlight360.pipeline.ExpertPanel", FakeJudge), \
                patch("highlight360.pipeline.build_candidates", return_value=candidates), \
                patch.object(VideoReader, "iter_frames", traced_frames):
            run(self.cfg)
        self.assertEqual(len(FakeJudge.calls), 3)

    def test_export_without_reel_counts_writes_and_uses_jpeg_quality_95(self):
        import cv2
        self.cfg.export.out_dir = str(self.root)
        self.cfg.export.make_reel = False
        updates = []
        with VideoReader(str(self.video), "flat") as reader, \
                patch("highlight360.export.cv2.imencode", wraps=cv2.imencode) as encode, \
                patch.object(reader, "iter_frames", wraps=reader.iter_frames) as frames, \
                patch.object(reader, "read_at", wraps=reader.read_at) as still:
            export_live_photos(self.cfg.export, reader, [event()], updates.append)
        self.assertEqual([u["done"] for u in updates], list(range(31)))
        self.assertTrue(all(u["total"] == 30 for u in updates))
        self.assertEqual(frames.call_count, 1)
        self.assertNotIn("max_width", frames.call_args.kwargs)
        self.assertNotIn("max_width", still.call_args.kwargs)
        self.assertEqual(encode.call_args.args[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertEqual(encode.call_args.args[1].shape, (128, 128, 3))
        self.assertFalse((self.root / "highlight_reel.mp4").exists())

    def test_single_candidate_wrapper_writes_fractional_tail(self):
        candidate = CandidateRegion(12, .25, 2.3, ViewSpec("flat"), [0])
        path = self.root / "custom-name.mp4"
        with VideoReader(str(self.video), "flat") as reader:
            duration = render_candidate(reader, candidate, self.cfg.region, path)
        self.assertAlmostEqual(duration, 2.1)
        self.assertEqual(candidate.end_sec, 2.3)
        with VideoReader(str(path), "flat") as reader:
            self.assertEqual(reader.meta.frame_count, 21)

    def test_cli_requires_projection_and_cloud_consent(self):
        from main import build_config
        with self.assertRaises(SystemExit):
            build_config(["--input", "video.mp4", "--prepare-only"])
        with self.assertRaises(SystemExit):
            build_config(["--input", "video.mp4", "--projection", "flat"])
        cfg = build_config(["--input", "video.mp4", "--projection", "flat", "--prepare-only"])
        self.assertTrue(cfg.prepare_only)
        selected = build_config(["--input", "video.mp4", "--projection", "flat", "--prepare-only",
                                "--judges", "v1", "v4"])
        self.assertEqual(selected.panel.selected_expert_ids, ("v1", "v4"))
        with self.assertRaises(SystemExit):
            build_config(["--input", "video.mp4", "--projection", "flat", "--prepare-only",
                          "--judges", "v1"])
        with self.assertRaises(SystemExit):
            build_config(["--input", "video.mp4", "--projection", "flat", "--prepare-only",
                          "--judges", "v1", "v1"])
        with self.assertRaises(SystemExit):
            build_config(["--input", "video.mp4", "--projection", "flat", "--prepare-only",
                          "--clip-sec", "nan"])


class TimelineTests(unittest.TestCase):
    def test_quality_defaults_without_large_encoding_fixture(self):
        self.assertEqual(RegionConfig().size, 960)
        self.assertEqual(PanelConfig().frame_width, 960)
        self.assertEqual(ExportConfig().size, 1440)

    def test_long_model_selected_interval_is_not_shortened(self):
        selected = event(start_sec=0.4, end_sec=3.8, best_sec=1.1)
        self.assertEqual(clip_range(selected, 3, 4), (0.4, 3.8))

    def test_short_model_selected_interval_is_padded_without_losing_action(self):
        selected = event(start_sec=0.2, end_sec=0.9, best_sec=0.7)
        start, end = clip_range(selected, 3, 4)
        self.assertEqual((start, end), (0, 3))
        self.assertLessEqual(start, selected.start_sec)
        self.assertGreaterEqual(end, selected.end_sec)

    def test_padding_respects_both_source_edges_and_short_source(self):
        near_end = event(start_sec=3.5, end_sec=3.8, best_sec=3.7)
        self.assertEqual(clip_range(near_end, 3, 4), (1, 4))
        short_source = event(start_sec=1.0, end_sec=1.4, best_sec=1.2)
        self.assertEqual(clip_range(short_source, 3, 2.5), (0, 2.5))

    def test_invalid_model_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "有效范围"):
            clip_range(event(start_sec=3, end_sec=2, best_sec=2.5), 3, 4)

    def test_different_regions_keep_simultaneous_events(self):
        events = [event(), event(event_id=1, candidate_id=1, involved_track_ids=[1])]
        self.assertEqual(len(deduplicate_events(events)), 2)

    def test_overlapping_window_duplicate_removed(self):
        events = [event(), event(candidate_id=1, start_sec=1.1, end_sec=2.1, best_sec=1.6)]
        self.assertEqual(len(deduplicate_events(events)), 1)

    def test_shared_bystander_does_not_merge_different_compositions(self):
        views = [
            (ViewSpec("equirectangular", yaw_deg=-45), ViewSpec("equirectangular", yaw_deg=45)),
            (ViewSpec("flat", flat_box=(0, 0, 0.6, 1)), ViewSpec("flat", flat_box=(0.4, 0, 1, 1))),
        ]
        for left, right in views:
            with self.subTest(projection=left.projection):
                events = [event(view=left, involved_track_ids=[0, 1]),
                          event(candidate_id=1, view=right, involved_track_ids=[1, 2])]
                self.assertEqual(len(deduplicate_events(events)), 2)

    def test_near_seam_same_composition_still_deduplicates(self):
        events = [event(view=ViewSpec("equirectangular", yaw_deg=179)),
                  event(candidate_id=1, view=ViewSpec("equirectangular", yaw_deg=-179),
                        start_sec=1.1, end_sec=2.1, best_sec=1.6)]
        self.assertEqual(len(deduplicate_events(events)), 1)

    def test_deduplication_requires_close_fov_pitch_and_spherical_center(self):
        original = ViewSpec("equirectangular")
        for other in (ViewSpec("equirectangular", fov_deg=110),
                      ViewSpec("equirectangular", pitch_deg=15),
                      ViewSpec("equirectangular", yaw_deg=4, pitch_deg=4),
                      ViewSpec("flat")):
            with self.subTest(other=other):
                self.assertEqual(len(deduplicate_events([event(view=original),
                                 event(candidate_id=1, view=other)])), 2)
        # 极点中心相同，yaw 相差很大仍会旋转构图，不当成同一事件。
        self.assertEqual(len(deduplicate_events([
            event(view=ViewSpec("equirectangular", pitch_deg=90)),
            event(candidate_id=1, view=ViewSpec("equirectangular", yaw_deg=90, pitch_deg=90)),
        ])), 2)

    def test_flat_deduplication_requires_ninety_percent_roi_overlap(self):
        for right, count in ((0.95, 1), (0.85, 2)):
            with self.subTest(right=right):
                self.assertEqual(len(deduplicate_events([
                    event(), event(candidate_id=1, view=ViewSpec("flat", flat_box=(0, 0, right, 1))),
                ])), count)

    def test_tracker_preserves_seam_and_timestamps(self):
        tracker = IoUTracker(TrackConfig(min_track_len=2), wrap_width=100)
        a = tracker.update([BBox(90, 10, 110, 50)], 0, 0.0)[0]
        b = tracker.update([BBox(0, 10, 20, 50)], 1, 0.2)[0]
        self.assertEqual(a.track_id, b.track_id)
        tracks = tracker.finalize()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].points[-1].time_sec, 0.2)


if __name__ == "__main__":
    unittest.main()
