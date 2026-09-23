"""模型高光响应的严格数据校验，不执行网络请求。"""
from __future__ import annotations

import json
import math

from .types import ModelHighlight


MAX_HIGHLIGHTS = 5
MAX_REASON_CHARS = 500
MAX_TITLE_CHARS = 80
MAX_RESPONSE_CHARS = 32_768
_HIGHLIGHT_FIELDS = {"start_sec", "end_sec", "best_sec", "reason", "title"}


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} 必须为有限数字，不能为布尔值或字符串。")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} 必须为有限数字。") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须为有限数字。")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("JSON 不允许 NaN 或 Infinity。")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 不允许重复字段。")
        result[key] = value
    return result


def parse_highlights(text: str, duration_sec: float) -> list[ModelHighlight]:
    duration = _finite_number(duration_sec, "duration_sec")
    if duration <= 0:
        raise ValueError("duration_sec 必须大于 0。")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型输出必须为非空 JSON 文本。")
    if len(text) > MAX_RESPONSE_CHARS:
        raise ValueError("模型输出文本过长。")
    try:
        data = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise ValueError("模型输出不是合法的严格 JSON（禁止非有限数或重复字段）。") from None
    if not isinstance(data, dict) or set(data) != {"highlights"}:
        raise ValueError("JSON 顶层必须为仅含 highlights 字段的对象。")
    items = data["highlights"]
    if not isinstance(items, list) or len(items) > MAX_HIGHLIGHTS:
        raise ValueError(f"highlights 必须为最多 {MAX_HIGHLIGHTS} 条的数组。")
    highlights = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != _HIGHLIGHT_FIELDS:
            raise ValueError(f"highlights[{index}] 字段缺失或含额外字段。")
        start = _finite_number(item["start_sec"], "start_sec")
        end = _finite_number(item["end_sec"], "end_sec")
        best = _finite_number(item["best_sec"], "best_sec")
        if not (0 <= start < end <= duration and start <= best < end):
            raise ValueError(f"highlights[{index}] 时间无效或越界（不允许越界误差）。")
        for name, limit in (("reason", MAX_REASON_CHARS), ("title", MAX_TITLE_CHARS)):
            value = item[name]
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"highlights[{index}].{name} 必须为非空字符串且不超过 {limit} 字符。")
        highlights.append(ModelHighlight(start_sec=start, end_sec=end, best_sec=best,
                                         reason=item["reason"], title=item["title"]))
    return highlights
