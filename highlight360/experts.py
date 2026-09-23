from __future__ import annotations

import json
import math
from pathlib import Path


EXPERTS = (
    {"id": "v1", "model": "qwen3.5-omni-plus", "family": "qwen", "role": "visual"},
    {"id": "v2", "model": "qwen3.8-omni-flash", "family": "qwen", "role": "visual"},
    {"id": "v3", "model": "qwen3-vl-plus", "family": "qwen", "role": "visual"},
    {"id": "v4", "model": "kimi-k2.6", "family": "kimi", "role": "visual"},
    {"id": "v5", "model": "qwen3.8-flash", "family": "qwen", "role": "visual"},
)

REQUIRED_OK_REVIEWS = 4
REQUIRED_FAMILIES = 2


def select_experts(expert_ids=None) -> list[dict]:
    """Resolve a user-selected subset while keeping the fixed roster order."""
    if expert_ids is None:
        selected = {expert["id"] for expert in EXPERTS}
    else:
        if not isinstance(expert_ids, (list, tuple)) or any(
                not isinstance(expert_id, str) for expert_id in expert_ids):
            raise ValueError("selected_expert_ids 必须是评委 id 列表")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("selected_expert_ids 不能重复")
        known = {expert["id"] for expert in EXPERTS}
        selected = set(expert_ids)
        if not selected or selected - known:
            raise ValueError("至少选择一位且只能选择列表中的评委")
    experts = [dict(expert) for expert in EXPERTS if expert["id"] in selected]
    if not 2 <= len(experts) <= len(EXPERTS):
        raise ValueError("至少选择2位评委，才能形成高光共识")
    return experts


def review_requirements(experts: list[dict]) -> tuple[int, int]:
    """Require at least two successes and preserve an 80% reviewer quorum."""
    if not 2 <= len(experts) <= len(EXPERTS):
        raise ValueError("评委数量必须为2~5位")
    family_count = len({expert["family"] for expert in experts})
    return max(2, math.ceil(len(experts) * 0.8)), min(REQUIRED_FAMILIES, family_count)

PUBLIC_EVIDENCE = {
    "reviewed_at": "2026-09-22",
    "weight_eligible": False,
    "reason": "公开榜单的任务、版本和输入条件不一致，不能换算为聚会高光准确率；权重只用厂商均衡中性先验。",
    "references": [
        {"url": "https://video-mme.github.io/home_page.html", "scope": "video_qa",
         "limitation": "通用视频问答，不等于高光偏好"},
        {"url": "https://arxiv.org/html/2602.00288v2", "scope": "temporal_video_understanding",
         "limitation": "时序理解基准，不等于高光偏好"},
        {"url": "https://artificialanalysis.ai/evaluations/mmmu-pro", "scope": "static_image_reasoning",
         "limitation": "静态图像推理，不等于视频高光"},
    ],
}


def panel_weights(reliability_file: str | None = None,
                  experts: list[dict] | None = None) -> tuple[dict[str, float], dict]:
    roster = [dict(expert) for expert in (EXPERTS if experts is None else experts)]
    reliability = {expert["model"]: 1.0 for expert in EXPERTS}
    calibration = None
    if reliability_file is not None:
        data = json.loads(Path(reliability_file).read_text(encoding="utf-8"))
        fields = {"metric", "dataset_id", "sample_count", "evaluated_at", "reliability"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("可靠性文件必须包含metric/dataset_id/sample_count/evaluated_at/reliability")
        if data["metric"] != "human_highlight_agreement":
            raise ValueError("只接受同一批人工标注高光的校准数据")
        if type(data["sample_count"]) is not int or data["sample_count"] < 20:
            raise ValueError("校准样本至少20条")
        for field in ("dataset_id", "evaluated_at"):
            if not isinstance(data[field], str) or not data[field].strip() or len(data[field]) > 200:
                raise ValueError("校准数据必须说明数据集和评测日期")
        scores = data["reliability"]
        if not isinstance(scores, dict) or set(scores) != set(reliability):
            raise ValueError("校准数据必须完整覆盖当前五位评委模型")
        for score in scores.values():
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 < score <= 1:
                raise ValueError("可靠性必须为(0,1]的有限数值")
        reliability = scores
        calibration = data
    if (not roster or len({expert["id"] for expert in roster}) != len(roster)
            or any(expert["model"] not in reliability for expert in roster)):
        raise ValueError("评委名单无效")
    families = sorted({expert["family"] for expert in roster})
    weights = {}
    total = sum(reliability[expert["model"]] for expert in roster)
    for expert in roster:
        weights[expert["id"]] = reliability[expert["model"]] / total
    return weights, {
        "policy": "equal_weight_neutral" if calibration is None else "equal_weight_task_calibration",
        "calibrated": calibration is not None,
        "calibration": calibration,
        "public_evidence": PUBLIC_EVIDENCE,
        "weights": weights,
        "family_weights": {family: sum(weights[e["id"]] for e in roster if e["family"] == family)
                           for family in families},
        "configured_reviewers": len(roster),
        "required_ok_reviews": review_requirements(roster)[0],
        "required_families": review_requirements(roster)[1],
        "failed_reviewers_abstain": True,
        "token_limit": "provider_default",
        "warning": "权重是中性先验，不是高光准确率；失败评委弃权，不重新归一化分母。",
    }
