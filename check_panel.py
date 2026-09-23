"""全景拾光五评委接口检查；默认离线，显式 --call-api 才发起最多五次合成多帧请求。

检查只上传本地合成的两张无人物色块 JPEG（base64），不读取、不上传任何真实视频，
每位评委各一次请求、5并发、不重试；报告只写状态/延迟/用量/错误码，不写响应原文。
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from highlight360.chat_api import APIRequestError, ChatClient, resolve_api_environment
from highlight360.experts import EXPERTS, REQUIRED_FAMILIES, REQUIRED_OK_REVIEWS, panel_weights
from highlight360.panel import _VISUAL_PROMPT
from highlight360.panel_protocol import parse_visual_review


FRAME_COUNT = 2
FRAME_WIDTH, FRAME_HEIGHT = 128, 96
DURATION_SEC = 2.0
_SAFE_FIELDS = ("usage", "latency_sec", "response_model", "request_id")


def synthetic_content() -> list[dict]:
    """本地合成两张色块 JPEG，结构与真实评审请求一致（时间戳文本 + data URI 图像）。"""
    content = [{"type": "text", "text": _VISUAL_PROMPT + f"\nduration={DURATION_SEC}秒。"}]
    for index in range(FRAME_COUNT):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)
        color = (0, 0, 255) if index == 0 else (255, 0, 0)
        left = 16 + index * 40
        cv2.rectangle(frame, (left, 24), (left + 32, 64), color, -1)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("合成JPEG编码失败")
        timestamp = index * DURATION_SEC / FRAME_COUNT
        content.extend([
            {"type": "text", "text": f"观察帧时间={timestamp:.6f}秒"},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}},
        ])
    return content


def probe(client: ChatClient, expert: dict, content: list[dict]) -> dict:
    """单个评委一次合成请求；失败只记录公开安全的错误码与用量。"""
    result = {"id": expert["id"], "model": expert["model"], "family": expert["family"],
              "status": "error"}
    try:
        response = client.complete(expert["model"], content)
        result.update({key: response[key] for key in _SAFE_FIELDS if key in response})
        review = parse_visual_review(response["text"], DURATION_SEC)
        result["status"] = "ok"
        result["facts"] = len(review["facts"])
        result["highlights"] = len(review["highlights"])
    except APIRequestError as exc:
        result.update(exc.metadata)
        result["error"] = {"code": exc.code, "http_status": exc.status_code}
    except ValueError:
        # 解析或传输异常可能包含私密输入，绝不回显原文。
        result["error"] = {"code": "invalid_review"}
    except Exception:
        result["error"] = {"code": "probe_failed"}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-api", action="store_true",
                        help="授权发送本地合成色块图（无真实视频、无人物），五位评委各一次")
    parser.add_argument("--out", default="panel_check.json", help="接口测试报告路径，必须不存在")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    weights, report = panel_weights()
    print("全景拾光 · 五位视觉评委 · 默认5并发")
    print(f"token_limit={report['token_limit']}（客户端不发送 max_tokens/max_completion_tokens，"
          f"只保留网络读取超时{args.timeout}秒）")
    for expert in EXPERTS:
        print(f"{expert['id']:>2}  {expert['role']:6}  {expert['model']:24}  "
              f"family={expert['family']:5}  vote_weight={weights[expert['id']]:.6f}")
    print("权重策略：", report["policy"],
          f"；等权中性先验（家族权重按成员数：qwen={report['family_weights']['qwen']:.2f}、"
          f"kimi={report['family_weights']['kimi']:.2f}），configured_reviewers={report['configured_reviewers']}"
          f"，required_ok_reviews={report['required_ok_reviews']}"
          f"，failed_reviewers_abstain={report['failed_reviewers_abstain']}；不是高光准确率")
    if not args.call_api:
        print("仅离线显示名单与权重，未读取凭据、未调用API；使用 --call-api 可做合成素材联调。")
        return 0
    if type(args.timeout) is not int or not 1 <= args.timeout <= 600:
        parser.error("--timeout 必须在1~600之间")
    output = Path(args.out).resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("报告文件必须不存在，且父目录必须已存在")
    results, status = [], 0
    try:
        base, key = resolve_api_environment()
        client = ChatClient(base, key, args.timeout)
        content = synthetic_content()
        with ThreadPoolExecutor(max_workers=len(EXPERTS)) as pool:
            futures = {pool.submit(probe, client, expert, content): expert for expert in EXPERTS}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                usage = result.get("usage", {})
                print(f"{result['id']:>2}  {result['model']:24}  status={result['status']:5}  "
                      f"latency_sec={result.get('latency_sec', '')}  usage={usage or {}}  "
                      f"error={result.get('error', '')}")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"评委联调失败：{exc}", file=sys.stderr)
        status = 1
    results.sort(key=lambda row: row["id"])
    ok = [row for row in results if row["status"] == "ok"]
    families = {row["family"] for row in ok}
    payload = {"fixture": "synthetic_colored_shapes_no_people_no_video",
               "frame_count": FRAME_COUNT, "duration_sec": DURATION_SEC,
               "is_highlight_accuracy_test": False, "retries": 0,
               "api_calls": len(results), "ok_count": len(ok),
               "ok_families": sorted(families),
               "required_ok_reviews": REQUIRED_OK_REVIEWS, "required_families": REQUIRED_FAMILIES,
               "token_limit": report["token_limit"], "weights": report, "results": results}
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    if status == 0 and (len(ok) < REQUIRED_OK_REVIEWS or len(families) < REQUIRED_FAMILIES):
        print(f"接口联调未达标：{len(ok)}/5成功、家族{sorted(families)}", file=sys.stderr)
        status = 1
    print(f"报告：{output}；实际API调用计数：{len(results)}（每位评委一次，无重试）")
    return status


if __name__ == "__main__":
    sys.exit(main())
