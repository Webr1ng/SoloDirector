"""GUI 离线冒烟测试：offscreen 平台 + 假帧源/假流水线，覆盖进度事件、完成与取消路径。"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets

import gui


def tiny_frames(count=8):
    for _ in range(count):
        yield np.zeros((1440, 2880, 3), np.uint8)


def slow_frames(count=60, delay=0.05):
    for _ in range(count):
        time.sleep(delay)
        yield np.zeros((1440, 2880, 3), np.uint8)


def fake_pipeline_run(cfg, progress=None):
    """模拟本地流水线：发几个阶段的进度事件，写出时间轴与评审记录。"""
    out = pathlib.Path(cfg.export.out_dir)
    (out / "jury").mkdir(parents=True, exist_ok=True)

    def emit(stage, done, total, **fields):
        if progress is not None:
            progress({"stage": stage, "done": done, "total": total, "unit": "calls", **fields})

    emit("detect", 3, 3)
    emit("judging", 5, 5, experts=[{"id": "v1", "model": "qwen3.5-omni-plus", "status": "ok"}],
         candidate_done=1, candidate_total=1)
    (out / "jury" / "candidate0000.json").write_text(json.dumps(
        {"reviews": [{"expert_id": "v1", "status": "ok"}]}), encoding="utf-8")
    (out / "全场高光时间轴.json").write_text(json.dumps(
        {"summary": {"event_count": 1}}), encoding="utf-8")
    emit("export", 2, 2)
    return {"candidates": 1, "events": 1, "people": 0, "timeline": str(out / "全场高光时间轴.json")}


def Path_tmp():
    return pathlib.Path(tempfile.mkdtemp(prefix="h360-gui-test-"))


class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _run_worker(self, cancel_after=None, frames=tiny_frames, run=fake_pipeline_run,
                    duration=1, allow_cloud=True, selected_expert_ids=None):
        destination = Path_tmp()
        events, done, errors = [], [], []
        worker = gui.CaptureWorker(duration, allow_cloud, destination=str(destination),
                                   frame_factory=frames, pipeline_run=run,
                                   selected_expert_ids=selected_expert_ids)
        worker.progress.connect(lambda e: events.append(e), QtCore.Qt.DirectConnection)
        worker.completed.connect(done.append, QtCore.Qt.DirectConnection)
        worker.failed.connect(errors.append, QtCore.Qt.DirectConnection)
        worker.start()
        if cancel_after is not None:
            time.sleep(cancel_after)
            worker.cancel()
        worker.wait(30000)
        return worker, events, done, errors, destination

    def test_local_run_writes_video_analyzes_and_completes(self):
        worker, events, done, errors, destination = self._run_worker()
        self.assertEqual(errors, [])
        self.assertEqual(len(done), 1)
        run_dir = pathlib.Path(done[0])
        self.assertEqual(run_dir.parent, destination)
        self.assertTrue((run_dir / "source.mp4").is_file())
        self.assertTrue((run_dir / "全场高光时间轴.json").is_file())
        stages = {e.get("stage") for e in events}
        self.assertIn("recording", stages)
        self.assertIn("write", stages)
        self.assertIn("detect", stages)
        self.assertIn("judging", stages)
        self.assertIn("export", stages)
        for e in events:
            self.assertLessEqual(e.get("done", 0), e.get("total", 0))
        recording = [e for e in events if e.get("stage") == "recording"]
        self.assertTrue(recording[-1].get("finished"))

    def test_pipeline_config_is_local_equirectangular(self):
        seen = {}

        def spy_run(cfg, progress=None):
            seen.update(input_path=cfg.input_path, out_dir=cfg.export.out_dir,
                        projection=cfg.input.projection, prepare_only=cfg.prepare_only,
                        allow_cloud=cfg.panel.allow_cloud_upload,
                        selected_expert_ids=cfg.panel.selected_expert_ids,
                        source_exists_during_analysis=pathlib.Path(cfg.input_path).is_file())
            return fake_pipeline_run(cfg, progress)

        worker, events, done, errors, destination = self._run_worker(
            run=spy_run, allow_cloud=False, selected_expert_ids=("v1", "v4"))
        self.assertEqual(errors, [])
        self.assertEqual(seen["projection"], "equirectangular")
        self.assertTrue(seen["prepare_only"])
        self.assertFalse(seen["allow_cloud"])
        self.assertTrue(seen["input_path"].endswith(".mp4"))
        self.assertEqual(pathlib.Path(seen["out_dir"]).parent, destination)
        self.assertEqual(pathlib.Path(seen["input_path"]),
                         pathlib.Path(seen["out_dir"]) / "source.mp4")
        self.assertTrue(seen["source_exists_during_analysis"])
        self.assertEqual(seen["selected_expert_ids"], ("v1", "v4"))
        self.assertTrue(pathlib.Path(seen["input_path"]).is_file())

    def test_early_stop_progress_reports_saved_frames_as_complete(self):
        window = gui.MainWindow()
        try:
            window.on_progress({"stage": "recording", "done": 79, "total": 150,
                                "unit": "frames", "finished": True})
            self.assertEqual(window.record_progress.value(), 1000)
            self.assertIn("已正常停拍，保存 79 帧", window.record_progress.format())
            window.on_progress({"stage": "write", "done": 79, "total": 79,
                                "unit": "frames", "finished": True})
            self.assertEqual(window.write_progress.value(), 1000)
            self.assertIn("完成，保存 79 帧", window.write_progress.format())
        finally:
            window.close()

    def test_cancel_during_capture_keeps_partial_and_skips_analysis(self):
        called = []

        def spy_run(cfg, progress=None):
            called.append(True)
            return fake_pipeline_run(cfg, progress)

        worker, events, done, errors, destination = self._run_worker(
            cancel_after=0.4, frames=slow_frames, run=spy_run, duration=5)
        self.assertEqual(done, [])
        self.assertEqual(called, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("取消", errors[0])
        partials = list(destination.glob("cancelled-*.mp4"))
        self.assertEqual(len(partials), 1)
        self.assertGreater(partials[0].stat().st_size, 0)

    def test_cancel_during_analysis_stops_pipeline(self):
        started = threading.Event()

        def slow_run(cfg, progress=None):
            started.set()
            for index in range(1000):
                if progress is not None:
                    progress({"stage": "detect", "done": index + 1, "total": 1000, "unit": "frames"})
                time.sleep(0.005)
            return fake_pipeline_run(cfg, progress)

        destination = Path_tmp()
        errors, done = [], []
        worker = gui.CaptureWorker(1, True, destination=str(destination),
                                   frame_factory=tiny_frames, pipeline_run=slow_run)
        worker.completed.connect(done.append, QtCore.Qt.DirectConnection)
        worker.failed.connect(errors.append, QtCore.Qt.DirectConnection)
        worker.start()
        self.assertTrue(started.wait(10))
        worker.cancel()
        worker.wait(30000)
        self.assertEqual(done, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("已取消", errors[0])
        run_dirs = [p for p in destination.iterdir() if p.is_dir()]
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "source.mp4").is_file())  # 原片保留

    def test_pipeline_failure_is_reported_and_source_kept(self):
        def broken_run(cfg, progress=None):
            raise RuntimeError("有效评委不足4/5，预算内有限补试后仍失败")

        worker, events, done, errors, destination = self._run_worker(run=broken_run)
        self.assertEqual(done, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("有效评委不足4/5", errors[0])
        run_dirs = [p for p in destination.iterdir() if p.is_dir()]
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "source.mp4").is_file())

    def test_audio_note_reports_silent_when_no_recorder(self):
        notes = []
        worker = gui.CaptureWorker(1, False, destination=str(Path_tmp()),
                                   frame_factory=tiny_frames, pipeline_run=fake_pipeline_run)
        worker.audio_note.connect(notes.append, QtCore.Qt.DirectConnection)
        worker.start()
        worker.wait(30000)
        self.assertEqual(notes, ["无声视频（未检测到X5麦克风）"])

    def test_main_window_has_cancel_button(self):
        window = gui.MainWindow()
        self.assertTrue(hasattr(window, "cancel_btn"))
        self.assertFalse(window.cancel_btn.isEnabled())
        self.assertTrue(window.cloud_check.isChecked())
        self.assertEqual(window.selected_expert_ids(), ("v1", "v2", "v3", "v4", "v5"))
        self.assertEqual(window.judge_selection_summary.text(), "已选 5/5 位 · 至少 4 位有效")
        window.close()

    def test_judge_selection_controls_selected_reviewers_and_requires_two(self):
        from unittest.mock import patch
        window = gui.MainWindow()
        try:
            window.judge_checks[1].setChecked(False)
            window.judge_checks[2].setChecked(False)
            self.assertEqual(window.selected_expert_ids(), ("v1", "v4", "v5"))
            self.assertEqual(window.judge_selection_summary.text(),
                             "已选 3/5 位 · 至少 3 位有效")
            for check in window.judge_checks[2:]:
                check.setChecked(False)
            with patch.object(gui.QtWidgets.QMessageBox, "warning") as warning:
                window.start_capture()
            warning.assert_called_once()
            self.assertIsNone(window.worker)
        finally:
            window.close()

    def test_retry_progress_is_visible_in_native_window(self):
        from PyQt5.QtTest import QTest
        window = gui.MainWindow()
        try:
            window.show()
            window.on_progress({"stage": "judging", "done": 5, "total": 12,
                                "candidate_done": 0, "candidate_total": 2,
                                "candidate_index": 1,
                                "detail": "补试评委 v2、v5（第2/2次）",
                                "experts": [{"id": "v2", "status": "running", "attempt": 2,
                                             "max_attempts": 2}]})
            QTest.qWait(20)
            self.assertIn("补试评委", window.status_label.text())
            self.assertIn("候选 0/2", window.progress.format())
            self.assertEqual(window.judges.item(1, 2).text(), "1/2 · 2/2次")
            window.on_progress({"stage": "judging", "done": 12, "total": 12,
                                "candidate_done": 1, "candidate_index": 2, "candidate_total": 2,
                                "detail": "补试完成", "experts": [{"id": "v2", "status": "ok"}]})
            self.assertEqual(window.progress.value(), 1000)
            self.assertEqual(window.judges.item(1, 2).text(), "2/2 · 完成")
            self.assertFalse(window.grab().isNull())
        finally:
            window.close()

    def test_start_capture_passes_selected_judges_and_clears_preview_until_camera_is_ready(self):
        from unittest.mock import patch
        from PyQt5 import QtGui
        class Signal:
            def connect(self, _callback):
                pass

        class WorkerStub:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                self.frame_ready = Signal()
                self.status = Signal()
                self.audio_note = Signal()
                self.progress = Signal()
                self.completed = Signal()
                self.failed = Signal()
                self.finished = Signal()
                self.started = False

            def start(self):
                self.started = True

            def isRunning(self):
                return False

        window = gui.MainWindow()
        try:
            window.cloud_check.setChecked(False)
            window.last_image = QtGui.QImage(20, 20, QtGui.QImage.Format_RGB888)
            with patch.object(gui, "CaptureWorker", WorkerStub):
                window.start_btn.click()
            self.assertEqual(window.worker.kwargs["selected_expert_ids"], ("v1", "v2", "v3", "v4", "v5"))
            self.assertTrue(window.worker.started)
            self.assertIsNone(window.last_image)
            self.assertIn("正在准备相机", window.preview.text())
            self.assertFalse(window.start_btn.isEnabled())
            window.on_worker_finished()
            self.assertTrue(window.start_btn.isEnabled())
        finally:
            window.close()

    def test_clear_results_button_removes_local_output_only(self):
        import shutil
        from unittest.mock import patch
        root = Path_tmp()
        (root / "run-1").mkdir(parents=True)
        (root / "run-1" / "event0_live.mp4").write_bytes(b"x")
        (root / "loose.mp4").write_bytes(b"y")
        window = gui.MainWindow()
        try:
            window.output_dir = root
            window.result_dir = str(root / "run-1")
            window.results_btn.setEnabled(True)
            with patch.object(gui.QtWidgets.QMessageBox, "question",
                              return_value=gui.QtWidgets.QMessageBox.Yes):
                window.clear_results()
            self.assertEqual(sorted(p.name for p in root.iterdir()), [])
            self.assertIsNone(window.result_dir)
            self.assertFalse(window.results_btn.isEnabled())
            with patch.object(gui.QtWidgets.QMessageBox, "question",
                              return_value=gui.QtWidgets.QMessageBox.No):
                (root / "keep").mkdir()
                window.clear_results()
            self.assertTrue((root / "keep").is_dir())
            shutil.rmtree(root, ignore_errors=True)
        finally:
            window.close()

    def test_stale_judge_rows_close_after_judging_and_follow_jury_report(self):
        window = gui.MainWindow()
        try:
            window.on_progress({"stage": "judging", "done": 4, "total": 5, "unit": "calls",
                                "candidate_index": 1, "candidate_total": 5,
                                "experts": [{"id": "v1", "model": "qwen3.5-omni-plus", "status": "ok"},
                                            {"id": "v2", "model": "qwen3.8-omni-flash", "status": "running"},
                                            {"id": "v4", "model": "kimi-k2.6", "status": "error"}]})
            self.assertEqual([window.judges.item(r, 2).text() for r in (0, 1, 3)],
                             ["1/5 · 完成", "1/5 · 1/2次", "1/5 · 弃权"])
            # 导出阶段不可能还有评委在跑：漏掉的终态快照不能把行冻在"分析中"。
            window.on_progress({"stage": "export", "done": 194, "total": 270})
            self.assertEqual([window.judges.item(r, 2).text() for r in (0, 1, 3)],
                             ["1/5 · 完成", "已结束", "1/5 · 弃权"])
            root = Path_tmp()
            (root / "jury").mkdir()
            (root / "jury" / "candidate0004.json").write_text(json.dumps(
                {"reviews": [{"expert_id": "v2", "status": "ok"},
                             {"expert_id": "v4", "status": "error"}]}), encoding="utf-8")
            window.on_completed(str(root))
            self.assertEqual([window.judges.item(r, 2).text() for r in (0, 1, 3)],
                             ["0/1完成·未处理1", "1/1完成", "0/1完成·弃权1"])
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
