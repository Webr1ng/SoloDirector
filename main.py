"""离线整片分析入口；本地摄像头采集写片用 capture.py，图形界面用 gui.py。"""
from __future__ import annotations

import argparse
import sys

from highlight360.config import Config
from highlight360.experts import EXPERTS
from highlight360.pipeline import run


def build_config(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(
        description="360°人物分区视频分析：先裁出完整场景，再由视频模型判断高光",
        epilog=".insv/.lrv 不能直接分析；请先用 Insta360 Studio 导出已拼接的单目全景、SDR MP4。",
    )
    ap.add_argument("--input", required=True, help="本地 MP4/MOV/MKV/M4V/AVI 视频路径")
    ap.add_argument("--projection", required=True, choices=("equirectangular", "flat"),
                    help="已拼接单目2:1全景选 equirectangular；普通视频选 flat，不按宽高比猜测")
    ap.add_argument("--out", default="output",
                    help="新的或空的输出目录（也可仅含当前输入视频），不会覆盖已有结果")
    ap.add_argument("--prepare-only", action="store_true", help="只导出分区候选视频与清单，不调用云端")
    ap.add_argument("--allow-cloud-upload", action="store_true",
                    help="允许将分区视频发送到阿里云模型，涉及隐私和API费用")
    ap.add_argument("--model-fps", type=float, default=2.0, help="所选视觉评委共用的时间戳抽帧率，0.1~10")
    ap.add_argument("--max-api-calls", type=int, default=100,
                    help="含补试的整次调用预算；可恢复失败每位最多补试一次，优先预留后续候选首轮费用")
    ap.add_argument("--judge-workers", type=int, default=5, help="同阶段评委最大并发数，1~10")
    ap.add_argument("--judges", nargs="+", choices=[expert["id"] for expert in EXPERTS],
                    help="参与评审的评委ID，默认全部；至少选择2位，例如 --judges v1 v4")
    ap.add_argument("--judge-timeout", type=int, default=120, help="单次评委请求读取超时秒数")
    ap.add_argument("--reliability-file", help="可选的同数据集人工高光校准JSON；未提供时使用厂商均衡中性权重")
    ap.add_argument("--max-candidates", type=int, default=100, help="候选数量预算；超出会报错，不静默漏掉")
    ap.add_argument("--window-sec", type=float, default=8.0, help="每段分析上下文长度，2~30秒")
    ap.add_argument("--stride-sec", type=float, default=4.0, help="分析步长，须不大于窗口长度")
    ap.add_argument("--detect-fps", type=float, default=5.0, help="人物检测采样频率，0.5~30")
    ap.add_argument("--views", type=int, default=4, help="全景水平检测视窗数4~12，另加上下极区两个视窗")
    ap.add_argument("--region-gap", type=float, default=35.0, help="人群方位邻近间隔，5~90度；不是精彩权重")
    ap.add_argument("--device", default=None, help="YOLO设备，例如 cpu 或 0；默认由Ultralytics选择")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO权重路径；默认权重首次加载可能联网下载")
    ap.add_argument("--clip-sec", type=float, default=3.0,
                    help="模型选中区间的最短成片时长；更长的完整动作区间会全部保留，2~4秒")
    ap.add_argument("--no-video", action="store_true", help="不导出最终短片和照片；模型分析仍需临时视频")
    ap.add_argument("--group-by-person", action="store_true", help="附加颜色特征人物分组，不是人脸识别")
    ap.add_argument("--n-people", type=int, default=0, help="人物分组数，0=自动")
    args = ap.parse_args(argv)
    checks = [
        (0.1 <= args.model_fps <= 10, "--model-fps 必须在0.1~10之间"),
        (args.max_candidates > 0, "--max-candidates 必须为正整数"),
        (10 <= args.max_api_calls <= 10000, "--max-api-calls 必须在10~10000之间"),
        (1 <= args.judge_workers <= 10, "--judge-workers 必须在1~10之间"),
        (args.judges is None or 2 <= len(args.judges) <= len(EXPERTS),
         "--judges 至少选择2位，最多选择5位"),
        (args.judges is None or len(set(args.judges)) == len(args.judges),
         "--judges 不能重复选择评委"),
        (1 <= args.judge_timeout <= 600, "--judge-timeout 必须在1~600之间"),
        (2 <= args.window_sec <= 30, "--window-sec 必须在2~30之间"),
        (0 < args.stride_sec <= args.window_sec, "--stride-sec 必须大于0且不超过窗口长度"),
        (0.5 <= args.detect_fps <= 30, "--detect-fps 必须在0.5~30之间"),
        (4 <= args.views <= 12, "--views 必须在4~12之间"),
        (5 <= args.region_gap <= 90, "--region-gap 必须在5~90之间"),
        (2 <= args.clip_sec <= 4, "--clip-sec 必须在2~4之间"),
        (args.n_people >= 0, "--n-people 不能为负数"),
        (args.prepare_only or args.allow_cloud_upload,
         "先用 --prepare-only 检查分区；调用模型须显式指定 --allow-cloud-upload"),
    ]
    for ok, message in checks:
        if not ok:
            ap.error(message)
    cfg = Config(input_path=args.input, prepare_only=args.prepare_only)
    cfg.input.projection = args.projection
    cfg.export.out_dir = args.out
    cfg.export.live_photo_sec = args.clip_sec
    cfg.export.export_video = not args.no_video
    cfg.detect.detect_fps = args.detect_fps
    cfg.detect.model = args.weights
    cfg.detect.device = args.device
    cfg.projection.num_views = args.views
    cfg.region.window_sec = args.window_sec
    cfg.region.stride_sec = args.stride_sec
    cfg.region.gap_deg = args.region_gap
    cfg.panel.sample_fps = args.model_fps
    cfg.panel.max_candidates = args.max_candidates
    cfg.panel.max_api_calls = args.max_api_calls
    cfg.panel.workers = args.judge_workers
    cfg.panel.selected_expert_ids = None if args.judges is None else tuple(args.judges)
    cfg.panel.timeout_sec = args.judge_timeout
    cfg.panel.reliability_file = args.reliability_file
    cfg.panel.allow_cloud_upload = args.allow_cloud_upload
    cfg.identity.enabled = args.group_by_person
    cfg.identity.n_people = args.n_people
    return cfg


if __name__ == "__main__":
    try:
        result = run(build_config())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        sys.exit(1)
    print(f"完成：候选 {result['candidates']}，高光 {result['events']}；清单 {result['timeline']}")
