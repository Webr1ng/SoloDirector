import json
import unittest

from highlight360 import vlm
from highlight360.types import ModelHighlight


def highlight(**changes):
    item = {"start_sec": 1.0, "end_sec": 4.0, "best_sec": 2.5,
            "reason": "人物展示作品后，大家以可见的手势回应。", "title": "作品展示"}
    item.update(changes)
    return item


def encode(*items):
    return json.dumps({"highlights": list(items)}, ensure_ascii=False)


class ParseHighlightsTests(unittest.TestCase):
    def test_empty_is_valid(self):
        self.assertEqual(vlm.parse_highlights(' {"highlights": []}\n', 8), [])

    def test_maps_every_field_without_time_offset(self):
        item = highlight()
        result = vlm.parse_highlights(encode(item), 8)
        self.assertEqual(result, [ModelHighlight(**item)])
        self.assertIsInstance(result[0].start_sec, float)

    def test_multiple_highlights_preserve_order(self):
        items = [highlight(start_sec=i, end_sec=i + 1, best_sec=i + 0.5) for i in range(5)]
        self.assertEqual(vlm.parse_highlights(encode(*items), 8), [ModelHighlight(**i) for i in items])

    def test_rejects_more_than_five(self):
        with self.assertRaises(ValueError):
            vlm.parse_highlights(encode(*[highlight()] * 6), 8)

    def test_accepts_exact_time_boundaries(self):
        for best in (0, 7.999):
            result = vlm.parse_highlights(encode(highlight(start_sec=0, end_sec=8, best_sec=best)), 8)
            self.assertEqual(result[0].best_sec, best)
        with self.assertRaises(ValueError):
            vlm.parse_highlights(encode(highlight(start_sec=0, end_sec=8, best_sec=8)), 8)

    def test_rejects_bad_json_without_repair(self):
        for text in ("", " ", "not JSON", '```json\n{"highlights":[]}\n```',
                     '{"highlights":[],}', '{"highlights":[]} trailing',
                     "{'highlights': []}", '{"highlights":[]}{"highlights":[]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                vlm.parse_highlights(text, 8)

    def test_rejects_non_text_input_and_oversized_response(self):
        for text in (None, b'{"highlights":[]}', {}, 1, True, " " * (vlm.MAX_RESPONSE_CHARS + 1)):
            with self.subTest(kind=type(text)), self.assertRaises(ValueError):
                vlm.parse_highlights(text, 8)

    def test_rejects_top_level_schema_errors(self):
        for data in ([], None, True, 1, "text", {}, {"highlights": None},
                     {"highlights": {}}, {"highlights": "[]"}, {"highlights": [], "reason": "extra"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                vlm.parse_highlights(json.dumps(data), 8)

    def test_rejects_missing_extra_and_non_object_entries(self):
        invalid = [None, [], "text", True, 1, {}]
        for key in highlight():
            item = highlight()
            del item[key]
            invalid.append(item)
        invalid.extend([highlight(score=0.5), highlight(path="elsewhere.mp4"),
                        highlight(clip_path="../../output.mp4")])
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                vlm.parse_highlights(encode(item), 8)

    def test_rejects_duplicate_json_keys(self):
        for text in ('{"highlights":[],"highlights":[]}',
                     '{"highlights":[{"start_sec":1,"start_sec":2,"end_sec":4,'
                     '"best_sec":3,"reason":"理由","title":"标题"}]}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                vlm.parse_highlights(text, 8)

    def test_rejects_nan_infinity_overflow_and_wrong_numeric_types(self):
        invalid = [float("nan"), float("inf"), -float("inf"), True, False,
                   None, "2", [], {}, 10 ** 400]
        for key in ("start_sec", "end_sec", "best_sec"):
            for value in invalid:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    vlm.parse_highlights(encode(highlight(**{key: value})), 8)
            for token in ("NaN", "Infinity", "-Infinity", "1e999"):
                text = encode(highlight(**{key: "TOKEN"})).replace('"TOKEN"', token)
                with self.assertRaises(ValueError):
                    vlm.parse_highlights(text, 8)

    def test_rejects_invalid_times_including_tiny_overrun(self):
        changes = [{"start_sec": -0.000001}, {"start_sec": 4}, {"start_sec": 5},
                   {"end_sec": 0}, {"end_sec": 8.000001}, {"end_sec": 8.05},
                   {"best_sec": 0.999999}, {"best_sec": 4.000001}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                vlm.parse_highlights(encode(highlight(**change)), 8)

    def test_rejects_invalid_description_types_and_lengths(self):
        for key, limit in (("reason", vlm.MAX_REASON_CHARS), ("title", vlm.MAX_TITLE_CHARS)):
            for value in (None, True, 5, [], {}, "", " \n\t", "字" * (limit + 1)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    vlm.parse_highlights(encode(highlight(**{key: value})), 8)
            item = highlight(**{key: "字" * limit})
            self.assertEqual(vlm.parse_highlights(encode(item), 8), [ModelHighlight(**item)])

    def test_invalid_later_item_does_not_return_partial_success(self):
        with self.assertRaises(ValueError):
            vlm.parse_highlights(encode(highlight(), highlight(end_sec=9)), 8)

    def test_validates_duration_even_for_empty_output(self):
        for duration in (0, -1, True, "8", None, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                vlm.parse_highlights('{"highlights":[]}', duration)


if __name__ == "__main__":
    unittest.main()
