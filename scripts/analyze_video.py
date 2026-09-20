#!/usr/bin/env python3
"""Run SoloDirector's offline video analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import analyze_video  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze an Insta360 video with YOLO Pose and generate event highlights."
    )
    parser.add_argument("--input", required=True, help="Input MP4/MOV video path")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "config.yaml"),
        help="YAML config path (default: config/config.yaml)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = analyze_video(args.input, config_path=args.config)
    except Exception as error:
        print(f"SoloDirector analysis failed: {error}", file=sys.stderr)
        return 1
    print(f"Input: {result.input_path}")
    print(f"Events: {len(result.events)}")
    print(f"Event JSON: {result.events_json}")
    print(f"Highlight video: {result.highlight_video or 'not generated'}")
    for event in result.events:
        print(
            f"  #{event.event_id} {event.event_type} "
            f"{event.start_time:.2f}s-{event.end_time:.2f}s score={event.highlight_score:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
