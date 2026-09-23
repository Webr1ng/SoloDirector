"""Offline five-judge panel tests; only generated temporary fixtures and fake clients."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock, get_ident
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import cv2
import numpy as np

from highlight360.chat_api import APIRequestError
from highlight360.config import PanelConfig
from highlight360.experts import EXPERTS, panel_weights, review_requirements, select_experts
from highlight360.io_video import VideoWriter
from highlight360.panel import ExpertPanel, PanelIncompleteError


class FakeClient:
    calls = []
    init_args = []
    failures = set()
    invalid = set()
    empty_models = set()
    gate = None
    wait_for_fast = None
    lock = Lock()
    active = peak = 0
    before_call = None

    def __init__(self, *args, **kwargs):
        type(self).init_args.append((args, kwargs))

    def complete(self, model, content):
        with self.lock:
            self.calls.append((model, content))
            call_number = len(self.calls)
            type(self).active += 1
            type(self).peak = max(self.peak, self.active)
        try:
            if self.before_call is not None:
                type(self).before_call(model)
            if self.gate is not None and (self.gate.parties == 5 or call_number <= 4):
                self.gate.wait()
            if self.wait_for_fast is not None and model != EXPERTS[-1]["model"]:
                if not self.wait_for_fast.wait(5):
                    raise RuntimeError("Coordinator did not report the fast judge first")
            if model in self.failures:
                raise APIRequestError("ProductNotActivated", 400)
            highlight = dict(start_sec=0.5, end_sec=1.5, best_sec=1,
                             reason="人物展示物品并得到可见回应", title="展示与回应")
            review = {"highlights": [] if model in self.empty_models else [highlight],
                      "facts": [{"start_sec": 0, "end_sec": 2, "description": "人物向另一人展示物品。"}]}
            text = "private-invalid-json" if model in self.invalid else json.dumps(review, ensure_ascii=False)
            return {"text": text, "usage": {"total_tokens": 100}, "latency_sec": 0.01,
                    "request_id": "test-request", "response_model": model}
        finally:
            with self.lock:
                type(self).active -= 1


class ProbeClient:
    """check_panel --call-api 的离线替身：只接受合成多帧请求，绝不联网。"""

    calls = []
    init_args = []
    failures = set()
    invalid = set()
    gate = None
    lock = Lock()
    active = peak = 0

    def __init__(self, *args, **kwargs):
        type(self).init_args.append((args, kwargs))

    def complete(self, model, content):
        with self.lock:
            type(self).calls.append((model, content))
            type(self).active += 1
            type(self).peak = max(self.peak, self.active)
        try:
            if self.gate is not None:
                self.gate.wait()  # 只有五个请求真的并发时才能全部通过。
            if model in self.failures:
                raise APIRequestError("ProductNotActivated", 400)
            if model in self.invalid:
                return {"text": "private-invalid-review", "usage": {"total_tokens": 10},
                        "latency_sec": 0.25, "request_id": "probe-request",
                        "response_model": model}
            review = {"facts": [{"start_sec": 0.0, "end_sec": 2.0,
                                 "description": "白色背景上先后出现红色与蓝色色块。"}],
                      "highlights": []}
            return {"text": json.dumps(review, ensure_ascii=False),
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240},
                    "latency_sec": 0.5, "request_id": "probe-request", "response_model": model}
        finally:
            with self.lock:
                type(self).active -= 1


class PanelTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls, FakeClient.failures, FakeClient.invalid = [], set(), set()
        FakeClient.init_args, FakeClient.empty_models = [], set()
        FakeClient.gate = FakeClient.wait_for_fast = FakeClient.before_call = None
        FakeClient.active = FakeClient.peak = 0
        # Replace, rather than copy, environment to avoid reading real credentials.
        patchers = (
            patch.object(os, "environ", {}),
            patch("highlight360.panel.resolve_api_environment", return_value=("https://example.invalid/v1", "fake-key")),
            patch("highlight360.panel.ChatClient", FakeClient),
            patch("requests.sessions.Session.request", side_effect=AssertionError("Offline tests cannot use network")),
        )
        for patcher in patchers:
            result = patcher.start()
            self.addCleanup(patcher.stop)
            if patcher is patchers[1]:
                self.resolve = result
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "fixture.mp4"
        with VideoWriter(str(self.video), 10, (128, 96)) as writer:
            for _ in range(20):
                writer.write(np.zeros((96, 128, 3), np.uint8))
        self.cfg = PanelConfig(allow_cloud_upload=True, frame_limit=4, frame_width=64)

    def test_prompt_names_solo_actions_as_highlight_evidence(self):
        from highlight360.panel import _VISUAL_PROMPT
        for token in ("起立", "鼓掌", "朝镜头", "只有一人", "不同方向"):
            self.assertIn(token, _VISUAL_PROMPT)

    def test_selected_roster_scales_calls_weights_and_quorum(self):
        self.cfg.selected_expert_ids = ("v1", "v2", "v5")
        panel = ExpertPanel(self.cfg)
        progress = []
        self.assertEqual([expert["id"] for expert in panel.experts], ["v1", "v2", "v5"])
        self.assertEqual((panel.required_ok_reviews, panel.required_families), (3, 1))
        self.assertAlmostEqual(sum(panel.weights.values()), 1)
        self.assertEqual(len(panel.analyze(str(self.video), 2, progress.append)), 1)
        self.assertEqual(len(FakeClient.calls), 3)
        self.assertEqual({model for model, _ in FakeClient.calls},
                         {expert["model"] for expert in panel.experts})
        self.assertEqual(panel.last_report["consensus_policy"]["configured_reviewers"], 3)
        self.assertEqual(panel.last_report["consensus_policy"]["required_ok_reviews"], 3)
        self.assertEqual(panel.last_report["consensus_policy"]["required_families"], 1)
        for update in progress:
            self.assertEqual([entry["id"] for entry in update["experts"]], ["v1", "v2", "v5"])
            self.assertTrue(all(entry["max_attempts"] == 2 for entry in update["experts"]))
        self.assertTrue(all(entry["status"] == "ok"
                            for entry in progress[-1]["experts"]))

    def test_selected_roster_rejects_invalid_size_duplicate_or_unknown_ids(self):
        for ids in (("v1",), ("v1", "v1"), ("v1", "unknown")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_experts(ids)

    def test_five_independent_jpeg_reviews_and_complete_audit(self):
        panel = ExpertPanel(self.cfg)
        with patch("highlight360.panel.cv2.imencode", wraps=cv2.imencode) as encode:
            events = panel.analyze(str(self.video), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(panel.calls_made, 5)
        report = panel.last_report
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["experts"]), 5)
        self.assertEqual(len(report["reviews"]), 5)
        self.assertEqual([r["expert_id"] for r in report["reviews"]], [e["id"] for e in EXPERTS])
        self.assertEqual((report["success_count"], report["failed_count"]), (5, 0))
        self.assertEqual(report["eligible_weights"], panel.weights)
        self.assertEqual((report["eligible_weight"], report["total_weight"]), (1, 1))
        self.assertEqual(report["eligible_families"], ["kimi", "qwen"])
        self.assertEqual((report["cloud_calls_before"], report["cloud_calls_after"]), (0, 5))
        self.assertEqual(report["required_ok_reviews"], 4)
        policy = report["consensus_policy"]
        self.assertEqual(policy["configured_reviewers"], 5)
        self.assertEqual(policy["required_ok_reviews"], 4)
        self.assertEqual(policy["required_families"], 2)
        self.assertEqual(policy["minimum_supporters"], 2)
        self.assertEqual(policy["minimum_supporting_families"], 2)
        self.assertEqual(policy["denominator"], "full_panel_weight")
        self.assertEqual(policy["failure_vote"], "abstain")
        self.assertTrue(policy["failed_reviewers_abstain"])
        self.assertEqual(policy["token_limit"], "provider_default")
        weights_report = report["weights"]
        self.assertEqual(weights_report["configured_reviewers"], 5)
        self.assertEqual(weights_report["required_ok_reviews"], 4)
        self.assertTrue(weights_report["failed_reviewers_abstain"])
        self.assertEqual(weights_report["token_limit"], "provider_default")
        self.assertEqual(weights_report["family_weights"], {"qwen": 0.8, "kimi": 0.2})
        for review in report["reviews"]:
            self.assertEqual(review["status"], "ok")
            self.assertEqual(review["usage"], {"total_tokens": 100})
            self.assertEqual(review["latency_sec"], 0.01)
            self.assertEqual(review["request_id"], "test-request")
            self.assertNotIn("error", review)
        self.assertEqual(report["consensus"][0]["ok_count"], 5)
        self.assertEqual(report["consensus"][0]["families"], ["kimi", "qwen"])
        self.assertEqual(report["highlights"], [
            {"start_sec": 0.5, "end_sec": 1.5, "best_sec": 1,
             "reason": "人物展示物品并得到可见回应", "title": "展示与回应"}])
        self.assertEqual((events[0].start_sec, events[0].end_sec, events[0].best_sec), (0.5, 1.5, 1))
        audit = report["input"]
        self.assertEqual(audit["frame_timestamps"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(audit["video_sha256"], hashlib.sha256(self.video.read_bytes()).hexdigest())
        self.assertEqual((audit["duration_sec"], audit["encoded_duration_sec"], audit["encoding_fps"]), (2, 2, 10))
        self.assertEqual(audit["jpeg_quality"], 90)
        self.assertEqual(audit["frame_width_limit"], 64)
        self.assertFalse(audit["audio"])
        self.assertEqual(encode.call_count, 4)
        for call in encode.call_args_list:
            self.assertEqual(call.args[0], ".jpg")
            self.assertEqual(call.args[2], [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertEqual({model for model, _ in FakeClient.calls}, {e["model"] for e in EXPERTS})
        for _, content in FakeClient.calls:
            self.assertIsInstance(content, list)
            images = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(images), 4)
            for image in images:
                uri = image["image_url"]["url"]
                self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
                raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
                self.assertTrue(raw.startswith(b"\xff\xd8"))
                self.assertEqual(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape[1], 64)
            text = " ".join(c["text"] for c in content if c["type"] == "text")
            self.assertNotIn("文字评委", text)
            self.assertNotIn("人物展示物品并得到可见回应", text)
        serialized = json.dumps(report, allow_nan=False)
        for forbidden in ("data:image", "fake-key", "text_evidence", "requires_all_judges",
                          "max_tokens", "max_completion_tokens"):
            self.assertNotIn(forbidden, serialized)

    def test_client_is_built_with_timeout_only_and_requests_carry_no_token_limits(self):
        panel = ExpertPanel(self.cfg)
        panel.analyze(str(self.video), 2)
        self.assertEqual(FakeClient.init_args,
                         [(("https://example.invalid/v1", "fake-key", 120), {})])
        self.assertEqual(len(FakeClient.calls), 5)
        for _, content in FakeClient.calls:
            self.assertTrue(all(set(part) <= {"type", "text", "image_url"} for part in content))
            self.assertNotIn("max_tokens", json.dumps(content))
            self.assertNotIn("max_completion_tokens", json.dumps(content))
            self.assertNotIn("thinking", json.dumps(content))
        self.cfg.timeout_sec = 45
        ExpertPanel(self.cfg).analyze(str(self.video), 2)
        self.assertEqual(FakeClient.init_args[-1], (("https://example.invalid/v1", "fake-key", 45), {}))

    def test_five_real_concurrent_calls_using_barrier(self):
        self.assertEqual(self.cfg.workers, 5)
        FakeClient.gate = Barrier(5, timeout=5)
        panel = ExpertPanel(self.cfg)
        self.assertEqual(len(panel.analyze(str(self.video), 2)), 1)
        self.assertEqual(FakeClient.peak, 5)
        self.assertEqual(panel.last_report["success_count"], 5)
        self.assertEqual(len(FakeClient.calls), 5)

    def test_worker_limit_is_respected_and_never_exceeds_five(self):
        for workers, expected in ((2, 2), (10, 5)):
            with self.subTest(workers=workers):
                FakeClient.calls, FakeClient.active, FakeClient.peak = [], 0, 0
                FakeClient.gate = Barrier(expected, timeout=5)
                self.cfg.workers = workers
                panel = ExpertPanel(self.cfg)
                panel.analyze(str(self.video), 2)
                self.assertEqual(FakeClient.peak, expected)
                self.assertEqual(panel.last_report["success_count"], 5)

    def test_as_completed_fast_judge_progress_on_coordinator_thread(self):
        FakeClient.gate = Barrier(5, timeout=5)
        FakeClient.wait_for_fast = Event()
        self.addCleanup(FakeClient.wait_for_fast.set)
        panel = ExpertPanel(self.cfg)
        events, thread_ids, running_ids = [], [], set()
        coordinator = get_ident()

        def before_call(model):
            expert_id = next(e["id"] for e in EXPERTS if e["model"] == model)
            if expert_id not in running_ids:
                raise AssertionError("Request submitted before its running event")

        FakeClient.before_call = before_call

        def progress(event):
            thread_ids.append(get_ident())
            events.append(event)  # No copying: snapshots must remain independent.
            running_ids.update(e["id"] for e in event["experts"] if e["status"] == "running")
            if event["done"] == 1:
                self.assertEqual(event["experts"][-1]["status"], "ok")
                self.assertTrue(all(e["status"] == "running" for e in event["experts"][:-1]))
                FakeClient.wait_for_fast.set()

        panel.analyze(str(self.video), 2, progress)
        self.assertEqual(thread_ids, [coordinator] * 10)
        self.assertEqual([event["done"] for event in events], [0] * 5 + [1, 2, 3, 4, 5])
        for index, event in enumerate(events[:5]):
            self.assertEqual([e["status"] for e in event["experts"]],
                             ["running"] * (index + 1) + ["pending"] * (4 - index))
        for event in events:
            self.assertEqual(set(event), {"stage", "done", "total", "failed", "experts",
                                          "calls_done", "calls_total"})
            self.assertEqual((event["stage"], event["total"], event["failed"]), ("judging", 5, 0))
            self.assertEqual((event["calls_total"], event["calls_done"]), (5, event["done"]))
            self.assertEqual([e["id"] for e in event["experts"]], [e["id"] for e in EXPERTS])
            self.assertEqual([e["model"] for e in event["experts"]], [e["model"] for e in EXPERTS])
            self.assertTrue(all(set(e) == {"id", "model", "status", "attempt",
                                            "max_attempts", "error_code"}
                                for e in event["experts"]))
            self.assertTrue(all(e["max_attempts"] == 2 for e in event["experts"]))
            self.assertTrue(all(e["attempt"] in (0, 1) for e in event["experts"]))
        self.assertTrue(all(e["status"] == "ok" for e in events[-1]["experts"]))
        # 事件是独立快照：消费者改动事件不会污染面板状态，也不会污染其它事件。
        self.assertEqual(len({id(event["experts"]) for event in events}), len(events))
        events[-1]["experts"][0]["status"] = "mutated"
        events[-1]["done"] = 99
        self.assertEqual([r["status"] for r in panel.last_report["reviews"]], ["ok"] * 5)
        self.assertEqual((panel.last_report["success_count"], panel.last_report["failed_count"]), (5, 0))

    def test_four_ok_degraded_without_renormalizing_and_progress_counts_failure(self):
        FakeClient.failures = {EXPERTS[-1]["model"]}
        FakeClient.empty_models = {EXPERTS[1]["model"], EXPERTS[2]["model"]}
        panel, progress = ExpertPanel(self.cfg), []
        # 等权下 v1+v4=0.4 不足 0.5；若除以有效权重 0.8 就会假通过。
        self.assertEqual(panel.analyze(str(self.video), 2, progress.append), [])
        report = panel.last_report
        self.assertEqual(report["status"], "degraded")
        self.assertEqual((report["success_count"], report["failed_count"]), (4, 1))
        self.assertEqual(report["eligible_weights"], {k: v for k, v in panel.weights.items() if k != "v5"})
        self.assertEqual((report["eligible_weight"], report["total_weight"], report["abstaining_weight"]), (0.8, 1, 0.2))
        self.assertEqual(report["abstaining_ids"], ["v5"])
        self.assertEqual(len(report["reviews"]), 5)
        self.assertEqual((progress[-1]["done"], progress[-1]["failed"]), (5, 1))
        for event in progress:
            self.assertEqual(event["failed"], sum(e["status"] == "error" for e in event["experts"]))
            self.assertEqual(event["done"], sum(e["status"] in ("ok", "error") for e in event["experts"]))
            self.assertEqual((event["calls_done"], event["calls_total"]), (event["done"], 5))
        FakeClient.empty_models = set()
        self.assertEqual(len(panel.analyze(str(self.video), 2)), 1)
        self.assertEqual(panel.last_report["consensus"][0]["ok_count"], 4)
        self.assertEqual(panel.last_report["consensus"][0]["abstaining_ids"], ["v5"])
        segment = panel.last_report["consensus"][0]["segments"][0]
        self.assertEqual(segment["support_weight"], 0.8)
        self.assertEqual(segment["abstaining_ids"], ["v5"])
        self.assertEqual(len(FakeClient.calls), 10)  # Exactly five per candidate; no retries.

    def test_three_ok_incomplete_retains_full_audit_and_no_false_success(self):
        FakeClient.failures = {EXPERTS[1]["model"], EXPERTS[2]["model"]}
        panel, progress = ExpertPanel(self.cfg), []
        with self.assertRaises(PanelIncompleteError):
            panel.analyze(str(self.video), 2, progress.append)
        report = panel.last_report
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual((report["success_count"], report["failed_count"]), (3, 2))
        self.assertEqual(report["eligible_families"], ["kimi", "qwen"])
        self.assertEqual(set(report["eligible_weights"]), {"v1", "v4", "v5"})
        self.assertEqual(report["total_weight"], 1)
        self.assertEqual(len(report["reviews"]), 5)
        self.assertEqual(report["highlights"], [])
        self.assertEqual(report["consensus"], [])
        self.assertIn("video_sha256", report["input"])
        self.assertEqual((progress[-1]["done"], progress[-1]["failed"]), (5, 2))
        self.assertEqual(panel.calls_made, 5)

    def test_missing_family_degrades_but_still_consents_with_four_ok(self):
        FakeClient.failures = {e["model"] for e in EXPERTS if e["family"] == "kimi"}
        panel = ExpertPanel(self.cfg)
        self.assertEqual(len(panel.analyze(str(self.video), 2)), 1)
        self.assertEqual(panel.last_report["eligible_families"], ["qwen"])
        self.assertEqual(panel.last_report["status"], "degraded")
        self.assertEqual(panel.last_report["consensus"][0]["segments"][0]["family_count"], 1)

    def test_invalid_visual_review_is_failed_and_keeps_usage_not_raw_text(self):
        FakeClient.invalid = {EXPERTS[0]["model"]}
        panel = ExpertPanel(self.cfg)
        panel.analyze(str(self.video), 2)
        self.assertEqual(panel.last_report["status"], "degraded")
        failed = panel.last_report["reviews"][0]
        self.assertEqual(failed["error"]["code"], "invalid_review")
        self.assertEqual(failed["usage"], {"total_tokens": 100})
        self.assertNotIn("private-invalid-json", json.dumps(panel.last_report))

    def test_truncated_reviews_keep_only_safe_consumption_metadata(self):
        metadata = {"usage": {"prompt_tokens": 50, "completion_tokens": 40000, "total_tokens": 40050},
                    "latency_sec": 1.25, "request_id": "safe-truncated-request", "response_model": "test-model"}
        error = APIRequestError("truncated", 200, metadata=metadata | {
            "text": "private-response-marker", "raw": "private-response-marker"})
        panel = ExpertPanel(self.cfg)
        with patch.object(FakeClient, "complete", side_effect=error), self.assertRaises(PanelIncompleteError):
            panel.analyze(str(self.video), 2)
        self.assertEqual(panel.last_report["status"], "incomplete")
        self.assertEqual((panel.last_report["success_count"], panel.last_report["failed_count"]), (0, 5))
        for review in panel.last_report["reviews"]:
            self.assertEqual(review["error"], {"code": "truncated", "http_status": 200})
            for key, value in metadata.items():
                self.assertEqual(review[key], value)
            self.assertNotIn("review", review)
        self.assertNotIn("private-response-marker", json.dumps(panel.last_report))

    def test_unexpected_and_permanent_errors_never_leak_or_retry(self):
        for error in (APIRequestError("Unauthorized", 401), ValueError("fake-key private-prompt"),
                      RuntimeError("fake-key private-prompt")):
            with self.subTest(error_type=type(error)):
                panel = ExpertPanel(self.cfg)
                with patch.object(FakeClient, "complete", side_effect=error) as complete, \
                        self.assertRaises(PanelIncompleteError):
                    panel.analyze(str(self.video), 2)
                self.assertEqual(complete.call_count, 5)
                self.assertTrue(all("usage" not in r for r in panel.last_report["reviews"]))
                for private in ("fake-key", "private-prompt"):
                    self.assertNotIn(private, json.dumps(panel.last_report))

    def test_two_invalid_reviews_recover_without_recalling_successes(self):
        from collections import Counter
        original = FakeClient.complete
        counts, lock = Counter(), Lock()
        broken = {EXPERTS[1]["model"], EXPERTS[4]["model"]}

        def complete(client, model, content):
            response = original(client, model, content)
            with lock:
                counts[model] += 1
                first = counts[model] == 1
            if model in broken and first:
                response["text"] = '{"facts":["private-invalid-marker"],"highlights":[]}'
            return response

        events, threads = [], []
        coordinator = get_ident()

        def progress(update):
            threads.append(get_ident())
            events.append(update)

        panel = ExpertPanel(self.cfg)
        with patch.object(FakeClient, "complete", complete):
            self.assertEqual(len(panel.analyze(str(self.video), 2, progress)), 1)
        self.assertEqual(threads, [coordinator] * len(events))
        self.assertEqual(counts, {e["model"]: 2 if e["model"] in broken else 1 for e in EXPERTS})
        report = panel.last_report
        self.assertEqual((panel.calls_made, report["retry_count"], report["success_count"]), (7, 2, 5))
        self.assertEqual((events[-1]["done"], events[-1]["total"]), (7, 7))
        self.assertEqual([e["done"] for e in events], sorted(e["done"] for e in events))
        self.assertTrue(all(0 <= e["done"] <= e["total"] for e in events))
        self.assertTrue(any(e["done"] == 5 and e["total"] == 7 for e in events))
        for review in report["reviews"]:
            expected = ["error", "ok"] if review["model"] in broken else ["ok"]
            self.assertEqual([a["status"] for a in review["attempts"]], expected)
            if len(expected) == 2:
                self.assertEqual(review["attempts"][0]["error"],
                                 {"code": "invalid_review", "validation": "facts",
                                  "detail": "fields"})
                self.assertEqual(sum(a["usage"]["total_tokens"] for a in review["attempts"]), 200)
            self.assertNotIn("error", review)
        retry_requests = [content for _, content in FakeClient.calls if "重新独立观察" in content[-1].get("text", "")]
        self.assertEqual(len(retry_requests), 2)
        first_images = [p for p in FakeClient.calls[0][1] if p["type"] == "image_url"]
        for content in retry_requests:
            self.assertEqual([p for p in content if p["type"] == "image_url"], first_images)
            self.assertIn("每条必须恰好包含start_sec、end_sec、description", content[-1]["text"])
            self.assertNotIn("private-invalid-marker", json.dumps(content))
        self.assertNotIn("private-invalid-marker", json.dumps(report))

    def test_permanent_invalid_reviews_stop_after_one_retry(self):
        FakeClient.invalid = {EXPERTS[1]["model"], EXPERTS[4]["model"]}
        panel = ExpertPanel(self.cfg)
        with self.assertRaises(PanelIncompleteError):
            panel.analyze(str(self.video), 2)
        self.assertEqual(panel.calls_made, 7)
        self.assertEqual(panel.last_report["success_count"], 3)
        self.assertEqual(panel.last_report["highlights"], [])
        self.assertEqual(panel.last_report["total_weight"], 1)
        for review in panel.last_report["reviews"]:
            if review["status"] == "error":
                self.assertEqual(len(review["attempts"]), 2)

    def test_retry_budget_reserves_all_later_candidates_first_calls(self):
        self.cfg.max_api_calls = 11
        original = FakeClient.complete
        counts = {}

        def complete(client, model, content):
            response = original(client, model, content)
            counts[model] = counts.get(model, 0) + 1
            if model in {EXPERTS[1]["model"], EXPERTS[4]["model"]} and counts[model] == 1:
                response["text"] = "invalid"
            return response

        panel = ExpertPanel(self.cfg)
        with patch.object(FakeClient, "complete", complete):
            panel.analyze(str(self.video), 2, remaining_candidates=1)
            self.assertEqual(panel.last_report["status"], "degraded")
            self.assertEqual(panel.calls_made, 6)
            self.assertEqual(panel.last_report["retry_skipped_ids"], ["v5"])
            panel.analyze(str(self.video), 2)
        self.assertEqual(panel.calls_made, 11)
        self.assertEqual(panel.last_report["status"], "complete")
        self.assertEqual(panel.last_report["cloud_calls_before"], 6)

    def test_budget_exhaustion_does_not_overspend_or_force_consensus(self):
        self.cfg.max_api_calls = 5
        FakeClient.invalid = {EXPERTS[1]["model"], EXPERTS[4]["model"]}
        panel = ExpertPanel(self.cfg)
        with self.assertRaises(PanelIncompleteError):
            panel.analyze(str(self.video), 2)
        self.assertEqual(panel.calls_made, 5)
        self.assertEqual(panel.last_report["retry_skipped_ids"], ["v2", "v5"])
        self.assertEqual(panel.last_report["retry_count"], 0)

    def test_transient_errors_retry_once_but_timeouts_and_auth_do_not(self):
        for error, expected in ((APIRequestError("connection_error"), 10),
                                (APIRequestError("timeout"), 5),
                                (APIRequestError("HTTP", 408), 5),
                                (APIRequestError("HTTP", 504), 5),
                                (APIRequestError("RateLimit", 429), 10),
                                (APIRequestError("HTTP", 503), 10),
                                (APIRequestError("Forbidden", 403), 5),
                                (APIRequestError("insufficient_quota", 429), 5)):
            panel = ExpertPanel(self.cfg)
            with self.subTest(code=error.code), patch("highlight360.panel.time.sleep") as wait, \
                    patch.object(FakeClient, "complete", side_effect=error) as call, \
                    self.assertRaises(PanelIncompleteError):
                panel.analyze(str(self.video), 2)
            self.assertEqual(call.call_count, expected)
            self.assertEqual(panel.calls_made, expected)
            self.assertEqual(wait.call_count, int(expected == 10))

    def test_bad_remaining_candidate_count_never_calls_api(self):
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ExpertPanel(self.cfg).analyze(str(self.video), 2, remaining_candidates=count)
        self.assertEqual(FakeClient.calls, [])

    def test_all_negative_is_complete_not_failure(self):
        FakeClient.empty_models = {e["model"] for e in EXPERTS}
        panel = ExpertPanel(self.cfg)
        self.assertEqual(panel.analyze(str(self.video), 2), [])
        self.assertEqual(panel.last_report["status"], "complete")
        self.assertEqual(panel.calls_made, 5)
        self.assertEqual(panel.last_report["retry_count"], 0)

    def test_five_call_budget_preflight_and_global_count_across_candidates(self):
        self.cfg.max_api_calls = 10
        panel = ExpertPanel(self.cfg)
        panel.check_budget(2)
        with self.assertRaises(ValueError):
            panel.check_budget(3)
        for count in (-1, True, 1.5, "2"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                panel.check_budget(count)
        for before in (0, 5):
            panel.analyze(str(self.video), 2)
            self.assertEqual(panel.last_report["cloud_calls_before"], before)
            self.assertEqual(panel.last_report["cloud_calls_after"], before + 5)
        with patch("highlight360.panel._frames") as frames, self.assertRaises(ValueError):
            panel.analyze(str(self.video), 2)
        frames.assert_not_called()
        self.assertEqual(len(FakeClient.calls), 10)
        self.assertEqual(panel.calls_made, 10)
        self.cfg.max_api_calls = 5
        ExpertPanel(self.cfg).validate_setup()  # One candidate is a valid budget.

    def test_no_consent_never_reads_credentials_frames_or_calls_api(self):
        self.cfg.allow_cloud_upload = False
        panel = ExpertPanel(self.cfg)
        with patch("highlight360.panel._frames") as frames, self.assertRaises(ValueError):
            panel.analyze(str(self.video), 2)
        self.resolve.assert_not_called()
        frames.assert_not_called()
        self.assertEqual(FakeClient.calls, [])
        self.assertEqual(panel.calls_made, 0)
        self.assertEqual(panel.last_report["cloud_calls_after"], 0)

    def test_duration_mismatch_never_calls_api(self):
        with self.assertRaises(ValueError):
            ExpertPanel(self.cfg).analyze(str(self.video), 3)
        self.assertEqual(FakeClient.calls, [])

    def test_encoded_tail_tolerance_and_effective_duration_sampling(self):
        self.video = self.root / "padded.mp4"
        with VideoWriter(str(self.video), 10, (128, 96)) as writer:
            for _ in range(21):
                writer.write(np.zeros((96, 128, 3), np.uint8))
        self.cfg.sample_fps, self.cfg.frame_limit = 10, 64
        for duration in (2.01, 2.05):
            with self.subTest(duration=duration):
                panel = ExpertPanel(self.cfg)
                panel.analyze(str(self.video), duration)
                audit = panel.last_report["input"]
                self.assertEqual(audit["duration_sec"], duration)
                self.assertAlmostEqual(audit["encoded_duration_sec"], 2.1)
                self.assertAlmostEqual(audit["encoding_fps"], 10)
                self.assertEqual(audit["frame_timestamps"][-1], 2.0)
                self.assertTrue(all(t < duration for t in audit["frame_timestamps"]))
                self.assertIn(f"duration={duration}秒", FakeClient.calls[0][1][0]["text"])
                FakeClient.calls = []

    def test_padding_over_one_frame_or_short_encoded_file_is_rejected(self):
        with self.assertRaises(ValueError):
            ExpertPanel(self.cfg).analyze(str(self.video), 2.01)
        self.video = self.root / "too-long.mp4"
        with VideoWriter(str(self.video), 30, (128, 96)) as writer:
            for _ in range(62):
                writer.write(np.zeros((96, 128, 3), np.uint8))
        with self.assertRaises(ValueError):
            ExpertPanel(self.cfg).analyze(str(self.video), 2.02)
        self.assertEqual(FakeClient.calls, [])

    def test_sampling_downscales_with_frame_budget(self):
        self.cfg.frame_limit = 2
        panel = ExpertPanel(self.cfg)
        panel.analyze(str(self.video), 2)
        self.assertEqual(panel.last_report["input"]["frame_timestamps"], [0.0, 1.0])

    def test_invalid_config_and_duration_rejected_offline(self):
        for field, value in (("workers", 0), ("workers", True), ("workers", 11),
                             ("timeout_sec", 0), ("timeout_sec", 601), ("timeout_sec", 1.5),
                             ("frame_limit", 1), ("frame_limit", 1000),
                             ("frame_width", 63), ("frame_width", 1281),
                             ("max_api_calls", 4), ("sample_fps", float("nan")),
                             ("sample_fps", True), ("sample_fps", "2"), ("sample_fps", 0.05),
                             ("sample_fps", 10.5),
                             ("consensus_threshold", 0.2), ("consensus_threshold", True),
                             ("consensus_threshold", 1.01), ("consensus_threshold", float("nan")),
                             ("consensus_threshold", float("inf"))):
            with self.subTest(field=field, value=value):
                cfg = PanelConfig(allow_cloud_upload=True)
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    ExpertPanel(cfg).validate_setup()
        self.resolve.assert_not_called()
        for duration in (True, "2", None, float("nan"), float("inf"), 10 ** 400, 1, 31):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                ExpertPanel(self.cfg).analyze(str(self.video), duration)
        self.assertEqual(FakeClient.calls, [])

    def test_config_boundaries_are_accepted(self):
        for field, value in (("workers", 1), ("workers", 10), ("timeout_sec", 1),
                             ("timeout_sec", 600), ("frame_limit", 2), ("frame_limit", 64),
                             ("frame_width", 64), ("frame_width", 1280), ("max_api_calls", 5),
                             ("sample_fps", 0.1), ("sample_fps", 10),
                             ("consensus_threshold", 0.5), ("consensus_threshold", 1)):
            with self.subTest(field=field, value=value):
                cfg = PanelConfig(allow_cloud_upload=True)
                setattr(cfg, field, value)
                panel = ExpertPanel(cfg)
                panel.validate_setup()
                self.assertIsInstance(panel.client, FakeClient)

    def test_threshold_one_is_valid_and_passes_no_segment(self):
        self.cfg.consensus_threshold = 1
        panel = ExpertPanel(self.cfg)
        self.assertEqual(panel.analyze(str(self.video), 2), [])
        self.assertEqual(panel.last_report["status"], "complete")
        self.assertEqual(panel.last_report["consensus_threshold"], 1)
        self.assertEqual(panel.last_report["consensus"], [])

    def test_progress_must_be_callable(self):
        for progress in (5, "progress", []):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                ExpertPanel(self.cfg).analyze(str(self.video), 2, progress)
        self.assertEqual(FakeClient.calls, [])

    def test_incomplete_error_has_a_single_source(self):
        from highlight360 import panel_protocol
        self.assertIs(PanelIncompleteError, panel_protocol.PanelIncompleteError)
        self.assertTrue(issubclass(PanelIncompleteError, RuntimeError))
        self.assertFalse(issubclass(PanelIncompleteError, ValueError))


class WeightAndCheckTests(unittest.TestCase):
    def setUp(self):
        ProbeClient.calls, ProbeClient.init_args = [], []
        ProbeClient.failures, ProbeClient.invalid = set(), set()
        ProbeClient.gate, ProbeClient.active, ProbeClient.peak = None, 0, 0
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_five_visual_reviewers_get_equal_neutral_weights(self):
        self.assertEqual(len(EXPERTS), 5)
        self.assertEqual([expert["id"] for expert in EXPERTS], ["v1", "v2", "v3", "v4", "v5"])
        self.assertTrue(all(expert["role"] == "visual" for expert in EXPERTS))
        self.assertEqual(len({expert["model"] for expert in EXPERTS}), 5)
        self.assertEqual({expert["family"] for expert in EXPERTS}, {"qwen", "kimi"})
        weights, report = panel_weights()
        for expert in EXPERTS:
            self.assertAlmostEqual(weights[expert["id"]], 0.2)
        self.assertAlmostEqual(report["family_weights"]["qwen"], 0.8)
        self.assertAlmostEqual(report["family_weights"]["kimi"], 0.2)
        self.assertEqual(set(weights), {expert["id"] for expert in EXPERTS})
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertEqual(report["weights"], weights)
        self.assertEqual(report["policy"], "equal_weight_neutral")
        self.assertFalse(report["calibrated"])
        self.assertIsNone(report["calibration"])
        self.assertEqual(report["configured_reviewers"], 5)
        self.assertEqual(report["required_ok_reviews"], 4)
        self.assertTrue(report["failed_reviewers_abstain"])
        self.assertEqual(report["token_limit"], "provider_default")
        self.assertFalse(report["public_evidence"]["weight_eligible"])
        self.assertTrue(report["public_evidence"]["references"])
        self.assertNotIn("十位", json.dumps(report, ensure_ascii=False))

    def test_selected_roster_weights_and_eighty_percent_quorum_scale(self):
        for count, required in ((2, 2), (3, 3), (4, 4), (5, 4)):
            roster = select_experts(tuple(expert["id"] for expert in EXPERTS[:count]))
            self.assertEqual(review_requirements(roster)[0], required)
            weights, report = panel_weights(experts=roster)
            self.assertEqual(set(weights), {expert["id"] for expert in roster})
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertEqual(report["configured_reviewers"], count)
            self.assertEqual(report["required_ok_reviews"], required)

    def test_calibration_rescales_globally_and_keeps_sum_one(self):
        scores = [1.0, 0.5, 0.5, 0.8, 0.2]
        reliability = {expert["model"]: scores[index] for index, expert in enumerate(EXPERTS)}
        data = {"metric": "human_highlight_agreement", "dataset_id": "party-highlights-20",
                "sample_count": 20, "evaluated_at": "2026-09-22", "reliability": reliability}
        path = self.root / "reliability.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        weights, report = panel_weights(str(path))
        total = sum(reliability.values())
        for expert in EXPERTS:
            self.assertAlmostEqual(weights[expert["id"]], reliability[expert["model"]] / total)
        self.assertAlmostEqual(report["family_weights"]["qwen"],
                               sum(reliability[e["model"]] for e in EXPERTS if e["family"] == "qwen") / total)
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertEqual(report["policy"], "equal_weight_task_calibration")
        self.assertTrue(report["calibrated"])
        self.assertEqual(report["calibration"], data)
        self.assertEqual(report["configured_reviewers"], 5)
        self.assertEqual(report["required_ok_reviews"], 4)
        self.assertTrue(report["failed_reviewers_abstain"])
        self.assertEqual(report["token_limit"], "provider_default")
        last = EXPERTS[-1]["model"]
        for change in ({"metric": "generic_video_qa"}, {"sample_count": 19},
                       {"sample_count": 20.0}, {"dataset_id": ""}, {"evaluated_at": None},
                       {"reliability": dict(reliability, **{last: 0})},
                       {"reliability": dict(reliability, **{last: 1.5})},
                       {"reliability": dict(reliability, **{last: "0.5"})},
                       {"reliability": dict(reliability, **{last: None})},
                       {"reliability": {key: 1.0 for key in list(reliability)[:-1]}},
                       {"reliability": dict(reliability, **{"unknown-model": 1.0})},
                       {"extra": 1}):
            with self.subTest(change=list(change)):
                path.write_text(json.dumps({**data, **change}, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    panel_weights(str(path))
        with self.assertRaises((ValueError, OSError)):
            panel_weights(str(self.root / "missing.json"))
        # 中性权重不读取任何文件。
        with patch.object(Path, "read_text", side_effect=AssertionError("Do not read files")):
            self.assertEqual(panel_weights()[0], panel_weights(None)[0])

    def test_check_panel_default_is_offline_and_reports_provider_default_token_limit(self):
        import check_panel
        output = io.StringIO()
        report_path = self.root / "panel_check.json"
        with patch.object(os, "environ", {}), \
                patch.object(check_panel, "resolve_api_environment",
                             side_effect=AssertionError("No keys")), \
                patch.object(check_panel, "ChatClient", ProbeClient), \
                patch("requests.sessions.Session.request",
                      side_effect=AssertionError("No network")), redirect_stdout(output):
            self.assertEqual(check_panel.main(["--out", str(report_path)]), 0)
        self.assertEqual(ProbeClient.calls, [])
        self.assertFalse(report_path.exists())
        text = output.getvalue()
        for expected in ("全景拾光", "五位视觉评委", "默认5并发", "token_limit=provider_default",
                         "max_tokens", "max_completion_tokens", "未调用API",
                         "required_ok_reviews=4", "failed_reviewers_abstain=True",
                         "family=qwen", "family=kimi", "vote_weight="):
            self.assertIn(expected, text)
        for expert in EXPERTS:
            self.assertEqual(text.count(expert["model"]), 1)
        self.assertEqual(text.count("vote_weight="), 5)
        self.assertNotIn("十", text)
        self.assertFalse(hasattr(check_panel, "VideoWriter"))
        self.assertFalse(hasattr(check_panel, "ExpertPanel"))

    def test_check_panel_call_api_probes_five_models_once_with_two_synthetic_jpegs(self):
        import check_panel
        ProbeClient.gate = Barrier(5, timeout=10)
        output, report_path = io.StringIO(), self.root / "panel_check.json"
        with patch.object(os, "environ", {}), \
                patch.object(check_panel, "resolve_api_environment",
                             return_value=("https://example.invalid/v1", "fake-key")), \
                patch.object(check_panel, "ChatClient", ProbeClient), \
                patch("requests.sessions.Session.request",
                      side_effect=AssertionError("No network")), redirect_stdout(output):
            self.assertEqual(check_panel.main(["--call-api", "--out", str(report_path),
                                               "--timeout", "30"]), 0)
        self.assertEqual(ProbeClient.init_args, [(("https://example.invalid/v1", "fake-key", 30), {})])
        self.assertEqual(ProbeClient.peak, 5)  # 五位评委真的并发，而不是串行。
        self.assertEqual(len(ProbeClient.calls), 5)  # 每位评委一次，无重试。
        self.assertEqual({model for model, _ in ProbeClient.calls}, {e["model"] for e in EXPERTS})
        for _, content in ProbeClient.calls:
            images = [part for part in content if part["type"] == "image_url"]
            self.assertEqual(len(images), 2)
            for image in images:
                uri = image["image_url"]["url"]
                self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
                raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
                self.assertTrue(raw.startswith(b"\xff\xd8"))
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
            text = " ".join(part["text"] for part in content if part["type"] == "text")
            self.assertIn(f"duration={check_panel.DURATION_SEC}秒", text)
            self.assertNotIn("max_tokens", json.dumps(content))
        printed = output.getvalue()
        for expert in EXPERTS:
            self.assertIn(f"{expert['model']:24}  status=ok", printed)
            self.assertIn("latency_sec=0.5", printed)
            self.assertIn("usage={'prompt_tokens': 1200, 'completion_tokens': 40", printed)
        self.assertIn("实际API调用计数：5", printed)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual((payload["api_calls"], payload["ok_count"], payload["retries"]), (5, 5, 0))
        self.assertEqual(payload["ok_families"], ["kimi", "qwen"])
        self.assertEqual(payload["required_ok_reviews"], 4)
        self.assertEqual(payload["required_families"], 2)
        self.assertEqual(payload["token_limit"], "provider_default")
        self.assertEqual(payload["frame_count"], 2)
        self.assertEqual(payload["weights"]["token_limit"], "provider_default")
        self.assertEqual([row["id"] for row in payload["results"]], [e["id"] for e in EXPERTS])
        for row in payload["results"]:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["usage"]["total_tokens"], 1240)
            self.assertEqual(row["latency_sec"], 0.5)
            self.assertEqual(row["facts"], 1)
            self.assertEqual(row["highlights"], 0)
            self.assertNotIn("review", row)
            self.assertNotIn("text", row)
        serialized = json.dumps(payload, ensure_ascii=False)
        for forbidden in ("data:image", "fake-key", "白色背景", "max_tokens"):
            self.assertNotIn(forbidden, serialized)

    def test_check_panel_call_api_reports_failures_without_retrying(self):
        import check_panel
        ProbeClient.failures = {EXPERTS[0]["model"], EXPERTS[1]["model"]}
        ProbeClient.invalid = {EXPERTS[2]["model"]}
        output, errors = io.StringIO(), io.StringIO()
        report_path = self.root / "panel_check.json"
        with patch.object(os, "environ", {}), \
                patch.object(check_panel, "resolve_api_environment",
                             return_value=("https://example.invalid/v1", "fake-key")), \
                patch.object(check_panel, "ChatClient", ProbeClient), \
                patch("requests.sessions.Session.request",
                      side_effect=AssertionError("No network")), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(check_panel.main(["--call-api", "--out", str(report_path)]), 1)
        self.assertEqual(len(ProbeClient.calls), 5)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual((payload["api_calls"], payload["ok_count"]), (5, 2))
        self.assertEqual(payload["ok_families"], ["kimi", "qwen"])
        by_id = {row["id"]: row for row in payload["results"]}
        self.assertEqual(by_id["v1"]["error"], {"code": "ProductNotActivated", "http_status": 400})
        self.assertEqual(by_id["v3"]["error"], {"code": "invalid_review"})
        self.assertEqual(by_id["v3"]["usage"], {"total_tokens": 10})
        self.assertEqual(by_id["v4"]["status"], "ok")
        self.assertNotIn("error", by_id["v4"])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("private-invalid-review", serialized)
        self.assertNotIn("fake-key", serialized)
        printed = output.getvalue()
        self.assertIn("status=error", printed)
        self.assertIn("status=ok", printed)
        self.assertIn("接口联调未达标", errors.getvalue())

    def test_check_panel_refuses_existing_report_and_bad_timeout(self):
        import check_panel
        existing = self.root / "exists.json"
        existing.write_text("{}", encoding="utf-8")
        for argv in (["--call-api", "--out", str(existing)],
                     ["--call-api", "--out", str(self.root / "missing-dir" / "x.json")],
                     ["--call-api", "--out", str(self.root / "ok.json"), "--timeout", "0"],
                     ["--call-api", "--out", str(self.root / "ok.json"), "--timeout", "601"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as caught, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                check_panel.main(argv)
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(ProbeClient.calls, [])
        self.assertFalse((self.root / "ok.json").exists())

    def test_check_panel_without_credentials_fails_closed_and_still_writes_report(self):
        import check_panel
        output, report_path = io.StringIO(), self.root / "panel_check.json"
        with patch.object(os, "environ", {}), \
                patch.object(check_panel, "resolve_api_environment",
                             side_effect=ValueError("OPENAI_BASE_URL or DASHSCOPE_HTTP_BASE_URL is required.")), \
                patch.object(check_panel, "ChatClient", ProbeClient), \
                patch("requests.sessions.Session.request",
                      side_effect=AssertionError("No network")), \
                redirect_stdout(output), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(check_panel.main(["--call-api", "--out", str(report_path)]), 1)
        self.assertEqual(ProbeClient.calls, [])
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual((payload["api_calls"], payload["ok_count"], payload["results"]), (0, 0, []))
        self.assertIn("评委联调失败", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
