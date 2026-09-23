"""全景拾光五视觉评委的离线校验与区间共识；失败弃权，不缩小分母。"""
from __future__ import annotations

import json
import math

from .experts import REQUIRED_FAMILIES, review_requirements
from .types import ModelHighlight
from .vlm import MAX_RESPONSE_CHARS, parse_highlights


MAX_FACTS = 6
MAX_DESCRIPTION_CHARS = 300
_FACT_FIELDS = {"start_sec", "end_sec", "description"}
# 弃权状态：请求失败(error)或因预算/未提交而根本没调用(not_called)。
ABSTAIN_STATUS = ("error", "not_called")


class PanelIncompleteError(RuntimeError):
    """成功评委或家族覆盖不足；完整审计已保留，绝不缩小分母冒充共识。"""


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} 必须为有限数字，不能为布尔值。")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} 必须为有限数字。") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须为有限数字。")
    return result


def _duration(value: object) -> float:
    result = _number(value, "duration")
    if result <= 0:
        raise ValueError("duration 必须大于 0。")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("JSON 不允许非有限数字。")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 不允许重复字段。")
        result[key] = value
    return result


def _load_json(text: str) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_RESPONSE_CHARS:
        raise ValueError("评审必须为非空 JSON 文本且不超过 32768 字符。")
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        # 不回显不可信响应或异常原文。
        raise ValueError("评审不是合法的严格 JSON（禁止重复字段及非有限数字）。") from None


def _json(data: object) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("评审必须可序列化为严格 JSON。") from None


def _facts(items: object, duration: float) -> None:
    if not isinstance(items, list) or len(items) > MAX_FACTS:
        raise ValueError("facts 必须为最多 6 条的数组。")
    for item in items:
        if not isinstance(item, dict) or set(item) != _FACT_FIELDS:
            raise ValueError("fact 必须仅含 start_sec、end_sec、description。")
        start = _number(item["start_sec"], "start_sec")
        end = _number(item["end_sec"], "end_sec")
        if not 0 <= start < end <= duration:
            raise ValueError("fact 时间无效或越界。")
        description = item["description"]
        if (not isinstance(description, str) or not description.strip()
                or len(description) > MAX_DESCRIPTION_CHARS):
            raise ValueError("description 必须为非空字符串且不超过 300 字符。")


class ReviewValidationError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__("评委结果不符合协议：" + code)


def parse_visual_review(text: str, duration: float) -> dict:
    """严格检查观察和高光，返回原始 JSON 对象；不修复围栏、字段或时间。"""
    duration = _duration(duration)
    try:
        data = _load_json(text)
    except ValueError:
        raise ReviewValidationError("json") from None
    if not isinstance(data, dict) or set(data) != {"facts", "highlights"}:
        raise ReviewValidationError("top_level")
    try:
        _facts(data["facts"], duration)
    except ValueError:
        raise ReviewValidationError("facts") from None
    try:
        highlights = parse_highlights(_json({"highlights": data["highlights"]}), duration)
    except ValueError:
        raise ReviewValidationError("highlights") from None
    if highlights and not data["facts"]:
        raise ReviewValidationError("missing_facts")
    return data


def _panel(reviews: list[dict], experts: list[dict], weights: dict[str, float],
           duration: float) -> tuple[dict[str, dict], dict[str, float], dict[str, list[ModelHighlight]],
                                     list[str]]:
    if not isinstance(experts, list) or not 2 <= len(experts) <= 5:
        raise ValueError("必须指定2~5位视觉评委。")
    roster = {}
    for expert in experts:
        if not isinstance(expert, dict):
            raise ValueError("expert 必须为对象。")
        for field in ("id", "model", "family", "role"):
            value = expert.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("expert 的 id/model/family/role 必须为非空字符串。")
        if expert["role"] != "visual" or expert["id"] in roster:
            raise ValueError("评委 id 重复或不是视觉评委。")
        roster[expert["id"]] = expert
    families = {expert["family"] for expert in roster.values()}
    if not 1 <= len(families) <= REQUIRED_FAMILIES:
        raise ValueError("所选评委模型家族数量无效。")
    required_ok_reviews, _ = review_requirements(experts)
    roster = dict(sorted(roster.items()))
    if not isinstance(weights, dict) or set(weights) != set(roster):
        raise ValueError("weights 必须恰好覆盖所选评委 id。")
    checked_weights = {key: _number(weights[key], "weight") for key in roster}
    if any(not 0 < value <= 1 for value in checked_weights.values()):
        raise ValueError("所有权重必须为正有限数且总和为 1。")
    if not math.isclose(math.fsum(checked_weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("权重总和必须为 1；不自动归一化。")
    if not isinstance(reviews, list) or len(reviews) != len(roster):
        raise ValueError("每位所选评委必须各有一条成功或失败记录；不能省略失败评委。")
    parsed, seen = {}, set()
    for item in reviews:
        if not isinstance(item, dict):
            raise ValueError("review 记录必须为对象。")
        expert_id = item.get("expert_id")
        if not isinstance(expert_id, str) or expert_id not in roster or expert_id in seen:
            raise ValueError("评审包含未知或重复评委。")
        seen.add(expert_id)
        expert = roster[expert_id]
        if any(item.get(key) != expert[key] for key in ("model", "family", "role")):
            raise ValueError("评审身份元数据与指定评委不一致。")
        status = item.get("status")
        if status in ABSTAIN_STATUS:
            # 失败或未调用都是弃权：保留完整权重分母，不视作反对票。
            if "review" in item:
                raise ValueError("弃权评委不能携带评审结果。")
            continue
        if status != "ok":
            raise ValueError("评审状态必须为 ok、error 或 not_called。")
        data = item.get("review")
        if (not isinstance(data, dict) or not isinstance(data.get("highlights"), list)
                or any(not isinstance(entry, dict) for entry in data["highlights"])):
            raise ValueError("评审必须包含合法 highlights 数组。")
        # 检查 Python 容器类型，避免 JSON 序列化修复非法 tuple。
        _facts(data.get("facts"), duration)
        parse_visual_review(_json(data), duration)
        parsed[expert_id] = parse_highlights(_json({"highlights": data["highlights"]}), duration)
    ok_families = sorted({roster[key]["family"] for key in parsed})
    if len(parsed) < required_ok_reviews:
        raise PanelIncompleteError(
            f"成功评委{len(parsed)}/{len(roster)}（至少需要{required_ok_reviews}位）；"
            "失败评委弃权，不缩小分母冒充共识。")
    return roster, checked_weights, parsed, ok_families


def _midpoint(start: float, end: float) -> float:
    middle = start + (end - start) / 2
    # 相邻浮点数之间可能没有可表示的中点；起点仍属于此半开基本段。
    return middle if middle < end else start


def _event(segments: list[dict], roster: dict[str, dict], weights: dict[str, float],
           reviews: dict[str, list[ModelHighlight]], ok_count: int,
           ok_families: list[str]) -> tuple[ModelHighlight, dict]:
    start, end = segments[0]["start_sec"], segments[-1]["end_sec"]
    supporting = sorted({key for segment in segments for key in segment["supporting_ids"]})
    opposing = sorted({key for segment in segments for key in segment["opposing_ids"]})
    abstaining = sorted({key for segment in segments for key in segment["abstaining_ids"]})
    relevant = {
        key: [item for item in reviews[key] if item.start_sec < end and start < item.end_sec]
        for key in supporting
    }
    candidates = []
    for key in supporting:
        # 每位评委最多一个 best；按其原始提名顺序选首个落在事件内的候选。
        for item in relevant[key]:
            if start <= item.best_sec < end:
                candidates.append((item.best_sec, key, weights[key]))
                break
    best = _midpoint(start, end)
    if candidates:
        candidates.sort()
        half = math.fsum(candidate[2] for candidate in candidates) / 2
        running = []
        for candidate_best, _, weight in candidates:
            running.append(weight)
            if math.fsum(running) >= half:
                best = candidate_best
                break
    # 文案取权重最高的支持者原文，不改写、不拼接、不代填。
    source = min(supporting, key=lambda key: (-weights[key], key))
    text = relevant[source][0]
    event = ModelHighlight(start_sec=start, end_sec=end, best_sec=best,
                           reason=text.reason, title=text.title)
    audit = {
        "start_sec": start, "end_sec": end,
        "conservative_min_support": min(segment["support_weight"] for segment in segments),
        "supporting_ids": supporting, "opposing_ids": opposing, "abstaining_ids": abstaining,
        "supporting_families": sorted({roster[key]["family"] for key in supporting}),
        "ok_count": ok_count, "families": list(ok_families),
        "text_source_id": source, "segments": segments,
    }
    return event, audit


def aggregate_reviews(reviews: list[dict], experts: list[dict], weights: dict[str, float],
                      duration: float, threshold: float = 0.5) -> tuple[list[ModelHighlight], list[dict]]:
    """所选评委各一条记录；成功人数按面板大小动态设置，至少2位且不低于约80%。

    分母始终是完整配置权重之和：失败/未调用评委弃权，不重新归一化，也不当作反对票。
    每个半开基本段必须严格满足 support > threshold（threshold 不低于0.5），
    且支持者不少于2位、支持家族不少于2个；相邻通过段合并为一个事件。
    best_sec 取支持者候选 best 的加权中位数（每位支持者最多一个落在事件内的 best），
    title/reason 取权重最高支持者的原文。
    不满足成功人数或家族覆盖时抛出 PanelIncompleteError（本模块为唯一定义处）。
    事件级 supporting_ids/opposing_ids 是子段并集，精确投票及弃权见 segments。
    """
    duration = _duration(duration)
    threshold = _number(threshold, "threshold")
    if not 0.5 <= threshold <= 1:
        raise ValueError("threshold 必须在 [0.5, 1] 内。")
    roster, weights, parsed, ok_families = _panel(reviews, experts, weights, duration)
    total_weight = math.fsum(weights.values())
    eligible_weight = math.fsum(weights[key] for key in parsed)
    abstaining = [key for key in roster if key not in parsed]
    boundaries = {0.0, duration}
    for items in parsed.values():
        for item in items:
            boundaries.update((item.start_sec, item.end_sec))
    boundaries = sorted(boundaries)
    groups = []
    current = []
    for start, end in zip(boundaries, boundaries[1:]):
        middle = _midpoint(start, end)
        supporting = [key for key in roster if key in parsed and any(
            item.start_sec <= middle < item.end_sec for item in parsed[key]
        )]
        # 分母固定为完整配置权重之和，弃权评委的权重不被重新分配。
        support = math.fsum(weights[key] for key in supporting) / total_weight
        supporting_families = sorted({roster[key]["family"] for key in supporting})
        # 家族交叉要求按"在场家族"自适应：某家族全员失败时允许单家族共识。
        required_families = min(REQUIRED_FAMILIES, len(ok_families))
        if (support > threshold and len(supporting) >= 2
                and len(supporting_families) >= required_families):
            current.append({
                "start_sec": start, "end_sec": end, "support_weight": support,
                "total_weight": total_weight, "eligible_weight": eligible_weight,
                "supporting_ids": supporting,
                "opposing_ids": [key for key in roster if key in parsed and key not in supporting],
                "abstaining_ids": list(abstaining),
                "supporter_count": len(supporting), "family_count": len(supporting_families),
                "supporting_families": supporting_families,
            })
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    events, audit = [], []
    for segments in groups:
        event, detail = _event(segments, roster, weights, parsed, len(parsed), ok_families)
        events.append(event)
        audit.append(detail)
    return events, audit
