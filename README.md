# SoloDirector / 影石智拍导演

SoloDirector 是本次 AI+影像产品开发赛道的首版比赛工程：它把 Insta360 相机素材接入
人体姿态理解，自动识别稳定姿态、举手、跳跃/明显动作，计算可解释的高光分数，输出最佳帧
与事件片段，并通过 FastAPI 给前端提供统一事件 JSON。

项目当前优先保证一条可以现场演示和离线复盘的链路：

```text
Insta360 Camera SDK / 本地视频
        |
        v
Camera Bridge (C++ CLI/API seam) --> data/input/*.mp4
        |
        v
OpenCV VideoReader (frame_idx, fps, timestamp, sampling)
        |
        v
Ultralytics YOLO Pose (CUDA / Apple MPS / CPU auto selection)
        |
        v
StablePose + HandRaise + Jump temporal rules
        |
        v
EventEngine --> EventRecord JSON (frontend contract)
        |
        +--> Explainable Highlight Scorer
        +--> Best Frame (Laplacian clarity + pose/composition)
        +--> FFmpeg event clips + merged highlights.mp4
        |
        v
FastAPI health / video selection / upload / synchronous analysis / job result
```

## 工程结构

```text
config/config.yaml                 统一阈值、模型、输出和 API 配置
insta360/camera_bridge/            C++17 Mock + 官方 SDK 接入占位层
src/video/                         OpenCV reader、FFmpeg clipper
src/pose/                          YOLO Pose adapter 和统一人体数据结构
src/events/                        stable_pose / hand_raise / jump / merge / schema
src/highlight/                     explainable scoring 和 best-frame 选择
src/api/                           FastAPI server
models/                            本地模型权重（Git 忽略）
data/input/                        输入视频（Git 忽略）
outputs/{events,frames,clips,...}  分析产物（Git 忽略）
tests/                             基础单元测试
scripts/                           CLI 分析和 API 启动脚本
```

## 环境安装

项目优先支持 Python 3.10/3.11。当前开发机可用的是 Python 3.12，因此使用 Python 3.12
创建虚拟环境也可以；不要把 Python 3.9 作为本项目运行时。

```bash
# 在仓库根目录执行
python --version                 # 当前机器应为 3.12.x
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`torch` 没有被固定为 CUDA 专用轮子。安装时会使用当前平台可用的 PyTorch 构建；运行时
由 `config/config.yaml` 的 `model.device: auto` 自动选择 CUDA、Apple MPS 或 CPU。

FFmpeg 和 CMake 是系统工具，不由 Python 依赖管理：

```bash
brew install ffmpeg cmake
ffmpeg -version
cmake --version
```

## 离线分析

把一个 MP4/MOV 放入 `data/input/`，或直接传绝对路径：

```bash
python scripts/analyze_video.py --help
python scripts/analyze_video.py --input data/input/demo.mp4
```

首次运行时，Ultralytics 会按照配置下载轻量 `models/yolo11n-pose.pt`（若比赛提供了指定
权重，也可以把路径改到 `config/config.yaml`）。当前默认采样 8 FPS，重点是让比赛演示在
普通笔记本上有可控延迟；调试时可以把 `video.sample_fps` 调到 4，追求精度时再调高。

产物包括：

```text
outputs/events/events.json
outputs/frames/event_0001_<type>.jpg
outputs/clips/event_0001_<type>.mp4
outputs/highlights/highlights.mp4
```

事件 JSON 是前端唯一依赖的格式，示例：

```json
{
  "schema_version": "1.0",
  "source": "data/input/demo.mp4",
  "events": [
    {
      "event_id": 1,
      "event_type": "hand_raise",
      "start_time": 12.4,
      "peak_time": 12.8,
      "end_time": 13.2,
      "confidence": 0.91,
      "highlight_score": 86.7,
      "reason": "动作置信度 0.91；人物完整度 0.94；动作幅度 0.72；构图 0.84；清晰度 0.68",
      "best_frame": "outputs/frames/event_0001_hand_raise.jpg",
      "clip": "outputs/clips/event_0001_hand_raise.mp4",
      "features": {}
    }
  ]
}
```

## FastAPI

启动服务：

```bash
python scripts/run_api.py
```

默认地址是 `http://127.0.0.1:8000`，可用接口：

```text
GET  /health
GET  /api/v1/videos
POST /api/v1/analyze          {"input_path":"data/input/demo.mp4"}
POST /api/v1/analyze/upload   multipart file=<video>
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/result
```

首版为了方便比赛联调是同步执行的，但路由通过 `AnalysisService` 调用，后续可将执行体
替换为任务队列而不改变前端事件 schema。

## Insta360 SDK 接入状态

`insta360/camera_bridge/` 是正式保留的 C++ 接入模块，不会用普通视频导入冒充影石 SDK。
它包含：

- `CameraBackend` 抽象接口：`connect / status / start_record / stop_record / list_files / download / latest`
- `MockCameraBackend`：无硬件时的可编译、可运行 smoke backend
- `Insta360CameraBackend`：不伪造官方函数名的占位 adapter
- CMake 构建文件和稳定 CLI 命令契约

先验证 Mock：

```bash
cmake -S . -B insta360/camera_bridge/build
cmake --build insta360/camera_bridge/build
find insta360/camera_bridge/build -type f -name solo_director_camera_bridge -print
```

拿到比赛提供的官方 Desktop Camera SDK 头文件、库文件和授权说明后，只需在
`src/insta360_camera_backend.cpp` 里完成真机调用，并把 SDK 路径作为本地 CMake 参数传入。
闭源二进制默认被 `.gitignore` 忽略，许可不明确时不提交。真实相机能力的验收顺序建议是：

```text
connect -> status -> start_record -> stop_record -> list_files -> latest -> download
```

## 测试与检查

```bash
python -m pytest
python -m compileall src scripts tests
python scripts/analyze_video.py --help
git status --short
```

测试覆盖配置加载、事件 schema、事件合并、高光评分和 FFmpeg 探测。没有输入视频时，
`--help` 仍然可用；真正分析需要视频、YOLO 权重和可用的 Python 依赖。

## 当前明确不做的 P1

首版暂不加入大规模训练、ReID/复杂多目标跟踪、Qwen 多模态、音频分析或复杂剪辑模板。
先让基础事件在比赛视频上稳定触发，再根据现场设备和时间把 SDK 真机控制接到这条离线链路
的输入端。
