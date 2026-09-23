"""全景拾光五视觉评委协议离线测试；不读取凭据、不联网。"""
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import patch

from highlight360 import panel_protocol
from highlight360.experts import EXPERTS, REQUIRED_FAMILIES, REQUIRED_OK_REVIEWS, panel_weights
from highlight360.panel_protocol import (PanelIncompleteError, aggregate_reviews,
                                         parse_visual_review)
from highlight360.types import ModelHighlight


def highlight(**changes):
    return {"start_sec": 1, "end_sec": 4, "best_sec": 2.5,
            "title": "展示与回应", "reason": "一人展示作品，另一人以可见的手势回应。", **changes}


def fact(**changes):
    return {"start_sec": 0, "end_sec": 10, "description": "一人展示作品，另一人抬手。", **changes}


def encode(data):
    return json.dumps(data, ensure_ascii=False)


def visual(*items, facts=None):
    return {"facts": [fact()] if facts is None else facts, "highlights": list(items)}


def panel():
    experts = [dict(e) for e in EXPERTS]
    reviews = [{"expert_id": e["id"], "model": e["model"], "family": e["family"],
                "role": e["role"], "status": "ok", "review": visual(facts=[])} for e in experts]
    return experts, reviews, panel_weights()[0]


def nominate(reviews, ids, *items):
    for row in reviews:
        if row["expert_id"] in ids:
            row["review"] = visual(*copy.deepcopy(items))


class ParseReviewTests(unittest.TestCase):
    def test_raw_json_types_and_empty_reviews_are_preserved(self):
        for data in (visual(highlight()), visual(facts=[]), visual()):
            self.assertEqual(parse_visual_review(encode(data), 10), data)
        self.assertIs(type(parse_visual_review(encode(visual(highlight())), 10)
                           ["highlights"][0]["start_sec"]), int)
        with self.assertRaises(ValueError):
            parse_visual_review(encode(visual(highlight(), facts=[])), 10)

    def test_fact_count_and_description_limits(self):
        data = visual(*[highlight()] * 5, facts=[fact(description="字" * 300)] * 6)
        self.assertEqual(parse_visual_review(encode(data), 10), data)
        invalid = [visual(facts=[fact()] * 7), visual(*[highlight()] * 6)]
        invalid.extend(visual(facts=[fact(description=value)])
                       for value in ("", " \n\t", "字" * 301, None, True, 1, [], {}))
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_visual_review(encode(data), 10)

    def test_fact_and_top_level_schema_are_strict(self):
        invalid = [None, [], True, 1, "text", {}, {"facts": []}, {"highlights": []},
                   {"facts": [], "highlights": [], "model": "private-marker"},
                   {"facts": [], "highlights": None}, {"facts": [], "highlights": {}}]
        invalid.extend({"facts": value, "highlights": []} for value in (None, {}, "facts", True, 1))
        entries = [None, [], "fact", True, {}, fact(weight=0.5)]
        for field in fact():
            item = fact()
            del item[field]
            entries.append(item)
        invalid.extend(visual(facts=[item]) for item in entries)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_visual_review(encode(data), 10)

    def test_times_reject_bool_nonfinite_wrong_types_and_out_of_range(self):
        values = [True, False, None, "1", [], {}, float("nan"), float("inf"),
                  -float("inf"), 10 ** 400]
        for field in ("start_sec", "end_sec"):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    parse_visual_review(encode(visual(facts=[fact(**{field: value})])), 10)
            for token in ("NaN", "Infinity", "-Infinity", "1e999"):
                text = encode(visual(facts=[fact(**{field: "TOKEN"})])).replace('"TOKEN"', token)
                with self.subTest(token=token), self.assertRaises(ValueError):
                    parse_visual_review(text, 10)
        for changes in ({"start_sec": -1e-12}, {"start_sec": 10}, {"end_sec": 0},
                        {"end_sec": math.nextafter(10.0, math.inf)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_visual_review(encode(visual(facts=[fact(**changes)])), 10)
        for best in (0, math.nextafter(10.0, 0.0)):
            data = visual(highlight(start_sec=0, end_sec=10, best_sec=best))
            self.assertEqual(parse_visual_review(encode(data), 10), data)

    def test_highlight_schema_remains_strict(self):
        entries = [None, {}, [], highlight(extra="private-marker"), highlight(best_sec=4),
                   highlight(best_sec=0.999999999), highlight(start_sec=-1e-12),
                   highlight(end_sec=math.nextafter(10.0, math.inf))]
        for field in highlight():
            item = highlight()
            del item[field]
            entries.append(item)
        for field in ("start_sec", "end_sec", "best_sec"):
            entries.extend(highlight(**{field: value}) for value in (
                True, False, None, "1", float("nan"), float("inf"), 10 ** 400))
        for field, limit in (("title", 80), ("reason", 500)):
            entries.extend(highlight(**{field: value}) for value in (
                "", " \n", True, None, [], 1, "字" * (limit + 1)))
        for item in entries:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_visual_review(encode(visual(highlight(), item)), 10)

    def test_bad_json_duplicates_and_fences_are_not_repaired(self):
        texts = ["", " ", "not JSON", "```json\n{}\n```", "{'highlights': []}",
                 '{"facts":[],"highlights":[],}', '{}{}', '{} trailing',
                 '[' * 1100 + '0' + ']' * 1100,
                 '{"facts":[],"facts":[],"highlights":[]}',
                 '{"facts":[],"highlights":[],"highlights":[]}',
                 '{"facts":[{"start_sec":0,"end_sec":1,"description":"a",'
                 '"description":"b"}],"highlights":[]}',
                 encode(visual(highlight())).replace('"best_sec": 2.5', '"best_sec":2,"best_sec":2.5')]
        for text in texts:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                parse_visual_review(text, 10)

    def test_response_size_input_types_and_duration(self):
        data = visual(facts=[])
        base = encode(data)
        text = base + " " * (32768 - len(base))
        self.assertEqual(parse_visual_review(text, 10), data)
        for value in (text + " ", None, b"{}", True, 1, {}):
            with self.subTest(kind=type(value)), self.assertRaises(ValueError):
                parse_visual_review(value, 10)
        for duration in (0, -1, True, False, None, "10", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                parse_visual_review(base, duration)

    def test_validation_categories_are_safe_and_do_not_relax_the_contract(self):
        cases = [("invalid private", "json"), (encode({"facts": []}), "top_level"),
                 (encode(visual(facts=["private"])), "facts"),
                 (encode(visual(highlight(best_sec=4))), "highlights"),
                 (encode(visual(highlight(), facts=[])), "missing_facts")]
        for text, category in cases:
            with self.subTest(category=category), self.assertRaises(panel_protocol.ReviewValidationError) as caught:
                parse_visual_review(text, 10)
            self.assertEqual(caught.exception.code, category)
            self.assertNotIn("private", str(caught.exception))

    def test_fact_failure_detail_is_safe_and_specific(self):
        cases = [([fact()] * 7, "shape"),
                 ([fact(extra="private-marker")], "fields"),
                 ([fact(end_sec=11)], "time"),
                 ([fact(description="private-marker" * 100)], "description")]
        for facts, detail in cases:
            with self.subTest(detail=detail), self.assertRaises(panel_protocol.ReviewValidationError) as caught:
                parse_visual_review(encode(visual(facts=facts)), 10)
            self.assertEqual((caught.exception.code, caught.exception.detail), ("facts", detail))
            self.assertNotIn("private-marker", str(caught.exception))

    def test_errors_do_not_echo_untrusted_text(self):
        with self.assertRaises(ValueError) as caught:
            parse_visual_review('private-marker: {invalid-json}', 10)
        self.assertNotIn("private-marker", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.experts, self.reviews, self.weights = panel()
        patcher = patch("requests.sessions.Session.request", side_effect=AssertionError("API forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def aggregate(self, **changes):
        return aggregate_reviews(**({"reviews": self.reviews, "experts": self.experts,
                                     "weights": self.weights, "duration": 10} | changes))

    def vote(self, ids, *items):
        nominate(self.reviews, ids, *items)

    def fail(self, ids, status="error"):
        for row in self.reviews:
            if row["expert_id"] in ids:
                row["status"] = status
                row.pop("review", None)

    def test_text_stage_helpers_are_gone(self):
        for name in ("build_text_evidence", "parse_text_review"):
            self.assertFalse(hasattr(panel_protocol, name))
        self.assertTrue(issubclass(PanelIncompleteError, RuntimeError))
        self.assertFalse(issubclass(PanelIncompleteError, ValueError))
        self.assertEqual((REQUIRED_OK_REVIEWS, REQUIRED_FAMILIES), (4, 2))

    def test_five_visual_members_complete_audit_and_order_independence(self):
        self.assertEqual(len(self.experts), 5)
        self.assertTrue(all(e["role"] == "visual" for e in self.experts))
        self.vote(["v1", "v2", "v4"], highlight())
        original = copy.deepcopy((self.experts, self.reviews, self.weights))
        events, audit = self.aggregate()
        self.assertEqual(events, [ModelHighlight(**highlight())])
        detail = audit[0]
        self.assertAlmostEqual(detail["conservative_min_support"], 0.6)
        self.assertEqual(detail["supporting_ids"], ["v1", "v2", "v4"])
        self.assertEqual(detail["opposing_ids"], ["v3", "v5"])
        self.assertEqual(detail["abstaining_ids"], [])
        self.assertEqual(detail["supporting_families"], ["kimi", "qwen"])
        self.assertEqual(detail["ok_count"], 5)
        self.assertEqual(detail["families"], ["kimi", "qwen"])
        self.assertEqual(detail["text_source_id"], "v1")
        segment = detail["segments"][0]
        self.assertEqual(segment["supporter_count"], 3)
        self.assertEqual(segment["family_count"], 2)
        self.assertEqual(segment["supporting_families"], ["kimi", "qwen"])
        self.assertEqual(segment["abstaining_ids"], [])
        self.assertEqual(segment["eligible_weight"], 1)
        self.assertEqual(segment["total_weight"], 1)
        self.assertEqual(set(detail), {"start_sec", "end_sec", "conservative_min_support",
                                       "supporting_ids", "opposing_ids", "abstaining_ids",
                                       "supporting_families", "ok_count", "families",
                                       "text_source_id", "segments"})
        self.assertEqual((self.experts, self.reviews, self.weights), original)
        self.assertEqual(self.aggregate(reviews=list(reversed(self.reviews)),
                                        experts=list(reversed(self.experts))), (events, audit))

    def test_four_ok_keep_original_denominator_and_failed_votes_abstain(self):
        self.fail(["v5"])
        self.vote(["v1", "v4"], highlight())
        # 等权下 0.4 不足 0.5；若除以有效权重 0.8 就会假通过；分母必须保持完整配置权重。
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v2"], highlight())
        events, audit = self.aggregate()
        self.assertEqual(len(events), 1)
        self.assertEqual(audit[0]["ok_count"], 4)
        self.assertEqual(audit[0]["families"], ["kimi", "qwen"])
        self.assertEqual(audit[0]["abstaining_ids"], ["v5"])
        segment = audit[0]["segments"][0]
        self.assertAlmostEqual(segment["support_weight"], 0.6)
        self.assertEqual(segment["eligible_weight"], 0.8)
        self.assertEqual(segment["total_weight"], 1)
        self.assertEqual(segment["abstaining_ids"], ["v5"])
        self.assertEqual(segment["opposing_ids"], ["v3"])

    def test_not_called_is_also_an_abstention(self):
        self.fail(["v5"], "not_called")
        self.vote(["v1", "v2", "v4"], highlight())
        events, audit = self.aggregate()
        self.assertEqual(len(events), 1)
        self.assertEqual(audit[0]["ok_count"], 4)
        self.assertEqual(audit[0]["abstaining_ids"], ["v5"])
        self.assertEqual(audit[0]["segments"][0]["abstaining_ids"], ["v5"])
        rows = copy.deepcopy(self.reviews)
        rows[-1]["status"] = "not_called"
        rows[-1]["review"] = visual(highlight())
        with self.assertRaises(ValueError):
            self.aggregate(reviews=rows)  # 弃权记录不得夹带评审结果

    def test_three_successes_are_incomplete_even_with_majority_and_both_families(self):
        self.vote(["v1", "v4", "v5"], highlight())
        self.fail(["v2", "v3"])
        with self.assertRaises(PanelIncompleteError) as caught:
            self.aggregate()
        self.assertNotIn("private", str(caught.exception))
        self.assertIn("3/5", str(caught.exception))

    def test_two_ok_or_one_ok_are_incomplete(self):
        for ok in (["v1", "v4"], ["v1"]):
            with self.subTest(ok=ok):
                self.experts, self.reviews, self.weights = panel()
                self.vote(ok, highlight())
                self.fail([row["expert_id"] for row in self.reviews if row["expert_id"] not in ok])
                with self.assertRaises(PanelIncompleteError):
                    self.aggregate()

    def test_zero_ok_is_incomplete_not_empty_consensus(self):
        self.fail([row["expert_id"] for row in self.reviews])
        with self.assertRaises(PanelIncompleteError):
            self.aggregate()

    def test_four_successes_single_family_pass_when_missing_family_failed(self):
        for index, expert in enumerate(self.experts):
            family = "a" if index < 4 else "b"
            expert["family"] = self.reviews[index]["family"] = family
        self.vote(self.weights, highlight())
        self.fail(["v5"])
        # 唯一 b 家族评委失败时，家族交叉要求自适应为 1，允许 a 家族内部共识。
        events, audit = self.aggregate()
        self.assertEqual(len(events), 1)
        self.assertEqual(audit[0]["families"], ["a"])
        self.assertEqual(audit[0]["segments"][0]["family_count"], 1)

    def test_missing_unknown_duplicate_and_unfinished_records_fail(self):
        for count in (0, 1, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.aggregate(reviews=self.reviews[:count])
        for status in ("timeout", "failed", "pending", "running", None, True):
            rows = copy.deepcopy(self.reviews)
            rows[-1]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.aggregate(reviews=rows)
        for expert_id in ("unknown", "v1", None):
            rows = copy.deepcopy(self.reviews)
            rows[-1]["expert_id"] = expert_id
            with self.subTest(expert_id=expert_id), self.assertRaises(ValueError):
                self.aggregate(reviews=rows)
        with self.assertRaises(ValueError):
            self.aggregate(reviews=self.reviews + [self.reviews[0]])

    def test_roster_and_identity_validation_includes_failed_records(self):
        for experts in (None, {}, self.experts[:-1], self.experts + [self.experts[0]]):
            with self.subTest(experts=experts), self.assertRaises(ValueError):
                self.aggregate(experts=experts)
        for field, value in (("model", ""), ("family", None), ("id", True),
                             ("id", "v2"), ("role", "other"), ("role", "text")):
            experts = copy.deepcopy(self.experts)
            experts[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.aggregate(experts=experts)
        for status in ("ok", "error", "not_called"):
            for field in ("model", "family", "role"):
                rows = copy.deepcopy(self.reviews)
                rows[0].update({field: "spoofed", "status": status})
                rows[0].pop("review", None)
                with self.subTest(field=field, status=status), self.assertRaises(ValueError):
                    self.aggregate(reviews=rows)

    def test_invalid_successful_review_fails_even_if_not_supporting(self):
        self.vote(["v1", "v2", "v4"], highlight())
        invalid = [None, {"highlights": []}, visual(highlight(), facts=[]),
                   visual(highlight(end_sec=11)), visual(facts=[fact(description="")]),
                   {"facts": (), "highlights": []}, {"facts": [], "highlights": ()},
                   {"facts": [], "highlights": [], "extra": "private-marker"},
                   visual(highlight(best_sec=True)), visual(*[highlight()] * 6)]
        for bad in invalid:
            rows = copy.deepcopy(self.reviews)
            rows[4]["review"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.aggregate(reviews=rows)

    def test_weights_and_numeric_arguments_are_strict(self):
        invalid = [None, {}, {key: value for key, value in self.weights.items() if key != "v5"},
                   dict(self.weights, unknown=0.1)]
        for value in (0, -0.1, True, None, "0.1", float("nan"), float("inf"), 10 ** 400, 0.3):
            invalid.append(dict(self.weights, v1=value))
        for weights in invalid:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.aggregate(weights=weights)
        for duration in (0, -1, True, None, "10", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.aggregate(duration=duration)
        for threshold in (0, math.nextafter(0.5, 0), 1.01, True, None, "0.5", float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.aggregate(threshold=threshold)

    def test_strict_majority_and_two_supporting_families(self):
        for ids in (["v1", "v2", "v3"], ["v4", "v5"]):
            self.experts, self.reviews, self.weights = panel()
            self.vote(ids, highlight())
            self.assertEqual(self.aggregate(), ([], []))  # Exactly 0.5 is not enough.
        self.vote(["v1"], highlight())
        self.assertEqual(len(self.aggregate()[0]), 1)
        support = math.fsum(self.weights[key] for key in ("v1", "v4", "v5"))
        self.assertEqual(self.aggregate(threshold=support), ([], []))
        self.assertEqual(self.aggregate(threshold=1), ([], []))
        # Even artificial heavy weights cannot bypass the two-visual/two-family gate.
        self.experts, self.reviews, _ = panel()
        self.weights = {"v1": 0.6, "v2": 0.1, "v3": 0.1, "v4": 0.1, "v5": 0.1}
        self.vote(["v1"], highlight())
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v2"], highlight())
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v4"], highlight())
        self.assertEqual(len(self.aggregate()[0]), 1)

    def test_overlaps_count_once_and_each_half_open_segment_needs_consensus(self):
        self.vote(["v1"], *[highlight(start_sec=i / 10, end_sec=6, best_sec=2) for i in range(5)])
        self.vote(["v4"], highlight(start_sec=0, end_sec=6))
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v2"], highlight(start_sec=0, end_sec=6))
        events, audit = self.aggregate()
        self.assertEqual([(e.start_sec, e.end_sec) for e in events], [(0, 6)])
        for segment in audit[0]["segments"]:
            self.assertEqual(segment["supporting_ids"].count("v1"), 1)
            self.assertAlmostEqual(segment["support_weight"], 0.6)
        self.experts, self.reviews, self.weights = panel()
        self.vote(["v1", "v2", "v3"], highlight(start_sec=0, end_sec=5))
        self.vote(["v4", "v5"], highlight(start_sec=5, end_sec=10, best_sec=5))
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v4"], highlight(start_sec=0, end_sec=10, best_sec=5))
        events, _ = self.aggregate()
        self.assertEqual([(e.start_sec, e.end_sec) for e in events], [(0, 5)])
        self.assertLess(events[0].best_sec, 5)

    def test_adjacent_segments_merge_without_losing_vote_changes(self):
        self.vote(["v1", "v4"], highlight(start_sec=0, end_sec=10))
        self.vote(["v2"], highlight(start_sec=0, end_sec=5))
        self.vote(["v5"], highlight(start_sec=5, end_sec=10, best_sec=7))
        events, audit = self.aggregate()
        self.assertEqual([(e.start_sec, e.end_sec) for e in events], [(0, 10)])
        detail = audit[0]
        self.assertAlmostEqual(detail["conservative_min_support"], 0.6)
        self.assertEqual([(s["start_sec"], s["end_sec"]) for s in detail["segments"]], [(0, 5), (5, 10)])
        self.assertIn("v2", detail["supporting_ids"])
        self.assertIn("v2", detail["opposing_ids"])
        for segment in detail["segments"]:
            self.assertEqual(set(segment["supporting_ids"]) | set(segment["opposing_ids"]), set(self.weights))

    def test_separated_highlights_and_tiny_rejected_gaps_do_not_merge(self):
        for start in (8, math.nextafter(5.0, math.inf)):
            items = [highlight(start_sec=0, end_sec=5, best_sec=0),
                     highlight(start_sec=start, end_sec=10, best_sec=9)]
            self.vote(self.weights, *items)
            events, audit = self.aggregate()
            self.assertEqual(events, [ModelHighlight(**item) for item in items])
            self.assertEqual(len(audit), 2)

    def test_empty_is_valid_and_best_is_weighted_median_with_authentic_text(self):
        self.assertEqual(self.aggregate(), ([], []))
        self.vote(["v1", "v2"], highlight(best_sec=1.5))
        self.vote(["v4"], highlight(best_sec=3.5, title="原始标题", reason="原始理由"))
        custom = {"v1": 0.2, "v2": 0.2, "v3": 0.2, "v4": 0.3, "v5": 0.1}
        event = self.aggregate(weights=custom)[0][0]
        self.assertEqual(event.best_sec, 1.5)
        self.assertEqual((event.title, event.reason), ("原始标题", "原始理由"))
        self.assertEqual(self.aggregate(weights=custom)[1][0]["text_source_id"], "v4")

    def test_only_one_best_per_expert_and_only_in_event_candidates(self):
        self.vote(["v4"], *[highlight(start_sec=0, end_sec=10, best_sec=1)] * 5)
        self.vote(["v1", "v2"], highlight(start_sec=0, end_sec=10, best_sec=9))
        self.assertEqual(self.aggregate()[0][0].best_sec, 9)
        self.experts, self.reviews, self.weights = panel()
        self.vote(["v1", "v4"], highlight(start_sec=0, end_sec=10, best_sec=8),
                  highlight(start_sec=2, end_sec=4, best_sec=2.25))
        self.vote(["v2"], highlight(start_sec=2, end_sec=4, best_sec=3))
        event = self.aggregate()[0][0]
        self.assertEqual((event.start_sec, event.end_sec, event.best_sec), (2, 4, 2.25))

    def test_midpoint_fallback_excludes_event_end_and_handles_rounding(self):
        for start, end, best in ((2, 4, 3),
                                 (math.nextafter(1.0, math.inf), math.nextafter(math.nextafter(1.0, math.inf), math.inf),
                                  math.nextafter(1.0, math.inf))):
            self.vote(["v1", "v2"], highlight(start_sec=0, end_sec=end, best_sec=0))
            self.vote(["v4"], highlight(start_sec=start, end_sec=8, best_sec=end))
            event = self.aggregate()[0][0]
            self.assertEqual((event.start_sec, event.end_sec, event.best_sec), (start, end, best))
        self.vote(self.weights, highlight(start_sec=1.6e308, end_sec=1.7e308, best_sec=1.65e308))
        self.assertTrue(math.isfinite(self.aggregate(duration=1.7e308)[0][0].best_sec))

    def test_audit_does_not_forward_arbitrary_metadata(self):
        self.vote(self.weights, highlight())
        for expert in self.experts:
            expert["credential"] = "synthetic-secret-marker"
        for row in self.reviews:
            row.update({"credential": "synthetic-secret-marker", "usage": object()})
        self.assertNotIn("synthetic-secret-marker", encode(self.aggregate()[1]))


if __name__ == "__main__":
    unittest.main()
