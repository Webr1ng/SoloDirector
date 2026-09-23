# SoloDirector｜全景拾光

> 影石 2026 BoldMaker 智能影像挑战赛「AI+影像产品开发赛道」作品

<p align="center">
  <img src="assets/boldmaker-poster.png" alt="SoloDirector 团队影石 2026 BoldMaker 智能影像挑战赛海报" width="720">
</p>

SoloDirector 是面向 Insta360 X5 和本地视频的 AI 高光剪辑工具。它在本机完成视频采集、人物检测、场景分区和视频导出；可选的视觉评委会分析候选片段，判断精彩动作从开始到结束的完整区间，再据此生成高光短片和合集。

## 功能

- 从 X5 USB 摄像头实时预览并录制全景视频，支持正常提前停拍。
- 对已拼接的 360° 全景或普通视频进行人物检测、跟踪和场景分区。
- 在 GUI 中勾选 v1–v5 评委；默认全选。界面显示每位评委当前处理的候选、尝试次数和结果。
- 根据评委共识保留完整动作区间。区间短于设定的最短时长时才补足上下文，避免精彩动作被固定时长裁断。
- 在本地导出评审报告、事件短片、照片及高光合集。

人物检测、分区和视频剪辑在本机运行。启用云端评审后，候选视频帧会发送给配置的模型服务并产生相应费用；不启用时可使用本地候选准备流程。

## 环境安装

需要 Python 3.10 或更新版本、FFmpeg，以及可用的摄像头或本地视频文件。项目已在 macOS 和 Python 3.12 环境中验证。

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

开发和测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

macOS 可用 Homebrew 安装 FFmpeg：

```bash
brew install ffmpeg
ffmpeg -version
```

首次运行时，Ultralytics 可能会下载默认人物检测权重 `yolov8n.pt`。权重保存在本地，不应提交到 Git。

## 启动图形界面

连接 X5 并切换到 USB 摄像头模式后，在仓库根目录运行：

```bash
.venv/bin/python gui.py
```

GUI 默认启用云端评审并默认选中五位评委。可以取消不参与的评委，但至少选择两位才能提交云端评审。有效评审门槛按“至少 80% 且至少两位”计算：选 2、3、4、5 位时，分别需要 2、3、4、4 位有效评审。超时评委直接弃权；其他可恢复错误最多补试一次，因此每位评委最多请求两次。

界面会显示当前候选进度和每位评委的尝试状态。处理完成后，各评委行会显示其成功、弃权或未处理的候选数量。GUI 会为每次拍摄创建独立结果目录，保存在仓库下的 `output/` 目录。

## 配置模型服务

复制安全模板并填写所用服务商的凭据：

```bash
cp .env.example .env
chmod 600 .env
```

在 `.env` 中填写一组 OpenAI 兼容配置或 DashScope 配置：

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=

# 或使用 DashScope：
DASHSCOPE_API_KEY=
DASHSCOPE_HTTP_BASE_URL=
```

`.env` 已加入 Git 忽略规则。不要将密钥写入源代码、README、命令行历史或提交记录。启用云端评审前，请先检查所选服务商、请求地址、评委数量及预期费用。

评委及其模型：

| 评委 | 模型 |
| --- | --- |
| v1 | `qwen3.5-omni-plus` |
| v2 | `qwen3.8-omni-flash` |
| v3 | `qwen3-vl-plus` |
| v4 | `kimi-k2.6` |
| v5 | `qwen3.8-flash` |

## 命令行处理

先只在本机准备候选片段，不调用云端模型：

```bash
.venv/bin/python main.py \
  --input /path/to/video.mp4 \
  --projection equirectangular \
  --prepare-only \
  --out output/prepare
```

普通平面视频将投影参数改为 `flat`。确认候选片段后，可显式启用云端评审并指定评委；省略 `--judges` 时默认使用全部五位：

```bash
.venv/bin/python main.py \
  --input /path/to/video.mp4 \
  --projection equirectangular \
  --allow-cloud-upload \
  --judges v1 v4 \
  --out output/reviewed
```

命令行完整选项可通过 `.venv/bin/python main.py --help` 查看。`--projection` 必须明确指定；尚未拼接的 `.insv` / `.lrv` 素材请先用 Insta360 Studio 导出为拼接视频。

## 输出内容

GUI 每次处理会创建独立结果目录。命令行请用 `--out` 指定新的或空的结果目录；程序不会覆盖已有结果。完整云端处理通常会生成：

- `全场高光时间轴.json`：源视频、候选、评委配置和共识结果。
- `jury/candidate*.json`：按候选保存的评委回复状态、尝试记录和审计信息。
- `event*_live.mp4` 与对应照片：各条高光短片及关键帧。
- `highlight_reel.mp4`：按时间顺序拼接的高光合集。

本地准备模式会输出 `candidates.json` 和候选片段，不会创建云端评审结果。

## 测试

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

GUI 测试在无屏幕环境中使用 Qt 的 offscreen 平台；其余单元测试使用合成视频和离线假模型，不会调用外部 API。
