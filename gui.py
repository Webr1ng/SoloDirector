"""全景拾光：本地拍摄、本地写片，可选视觉评委并行分析；不连接任何服务器。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

# venv 的 python 是指向 conda 基础解释器的符号链接，Qt 按可执行文件路径找到的
# 是 /opt/anaconda3/bin/qt.conf，插件搜索路径被误导到 conda 目录；
# 显式把 PyQt5 自带插件目录加进搜索路径，真实窗口和 offscreen 测试都依赖它。
_QT_PLUGINS = Path(QtCore.__file__).resolve().parent / "Qt5" / "plugins"
if _QT_PLUGINS.is_dir():
    QtCore.QCoreApplication.addLibraryPath(str(_QT_PLUGINS))

from capture import camera_frames, find_camera_index, record
from highlight360.audio import RATE, AudioRecorder, extract_pcm, find_microphone
from highlight360.config import Config
from highlight360.experts import EXPERTS, review_requirements, select_experts


PROJECT_ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FPS = 2880, 1440, 15


class _Cancelled(Exception):
    """用户取消：从进度回调或帧源内部中断本地流程，已生成文件保留。"""


def _default_pipeline_run():
    from highlight360.pipeline import run
    return run


class CaptureWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status = QtCore.pyqtSignal(str)
    audio_note = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(dict)
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, duration: int, allow_cloud: bool, camera_index: int = 1,
                 destination: str | None = None, *, selected_expert_ids=None,
                 frame_factory=None, pipeline_run=None):
        super().__init__()
        self.duration, self.allow_cloud = duration, allow_cloud
        self.camera_index = camera_index
        self.selected_expert_ids = (None if selected_expert_ids is None
                                    else tuple(selected_expert_ids))
        self.destination = Path(destination) if destination else PROJECT_ROOT / "output"
        self.stop_flag = threading.Event()
        self.cancel_flag = threading.Event()
        self.frame_factory = frame_factory
        self.pipeline_run = pipeline_run
        self._last_frame = 0.0
        self._timeline: list = []
        self._recorder = None

    def stop(self):
        self.stop_flag.set()

    def cancel(self):
        self.cancel_flag.set()
        self.stop_flag.set()
        if self._recorder is not None:
            self._recorder.stop()

    def _emit(self, event):
        if self.cancel_flag.is_set():
            raise _Cancelled("任务已取消：本地分析不再继续，已生成文件保留不删除")
        self.progress.emit(dict(event))

    def _preview_frames(self):
        source = (self.frame_factory() if self.frame_factory else
                  camera_frames(self.camera_index, WIDTH, HEIGHT, FPS, self.duration,
                                self.stop_flag, timeline=self._timeline,
                                on_ready=lambda: self.status.emit("相机画面已稳定，开始正式拍摄")))
        try:
            for frame in source:
                if self.cancel_flag.is_set():
                    raise _Cancelled("已取消拍摄：不提交分析，已采集片段保留")
                now = time.monotonic()
                if now - self._last_frame >= 0.1:
                    rgb = cv2.cvtColor(cv2.resize(frame, (1440, 720), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    image = QtGui.QImage(rgb.data, 1440, 720, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
                    self.frame_ready.emit(image)
                    self._last_frame = now
                yield frame
            if self.cancel_flag.is_set():
                raise _Cancelled("已取消拍摄：不提交分析，已采集片段保留")
        finally:
            source.close()

    def _unique_run_dir(self) -> Path:
        stamp = time.strftime("run-%Y%m%d-%H%M%S")
        candidate = self.destination / stamp
        suffix = 1
        while candidate.exists():
            candidate = self.destination / f"{stamp}-{suffix}"
            suffix += 1
        return candidate

    def _record(self, temp: Path) -> tuple[Path, int]:
        """采集并写本地 mp4；有录音器时按采集时间轴混音。返回 (成片路径, 帧数)。"""
        total = max(1, round(self.duration * FPS))
        silent = temp / "source.mp4"
        self._recording_file = (silent.with_name("source.nosound.mp4")
                                if self._recorder is not None else silent)
        written = {"count": 0}

        def progress(update: dict):
            if update.get("stage") == "recording":
                written["count"] = int(update.get("done", 0))
                self._recording_frames = written["count"]
            self._emit(update)
            if update.get("stage") == "recording":
                self._emit({"stage": "write", "done": written["count"],
                            "total": int(update.get("total", total)), "unit": "frames"})

        source = Path(record(str(silent), self._preview_frames(), FPS, (WIDTH, HEIGHT),
                             total, progress, self._recorder, self._timeline))
        self._recording_file = source
        if written["count"]:
            self._emit({"stage": "write", "done": written["count"],
                        "total": written["count"], "unit": "frames", "finished": True})
        return source, written["count"]

    def _build_config(self, source: Path, run_dir: Path) -> Config:
        cfg = Config(input_path=str(source), prepare_only=not self.allow_cloud)
        cfg.input.projection = "equirectangular"
        cfg.export.out_dir = str(run_dir)
        cfg.panel.allow_cloud_upload = self.allow_cloud
        cfg.panel.selected_expert_ids = self.selected_expert_ids
        cfg.panel.max_api_calls = 200
        cfg.panel.workers = 5
        cfg.panel.timeout_sec = 120
        return cfg

    def run(self):
        temp = None
        run_dir = None
        source = None
        self._recording_file = None
        self._recording_frames = 0
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
            self.status.emit("正在准备相机；画面稳定后开始计时")
            if self.frame_factory is None:
                device = find_microphone()
                if device is not None:
                    try:
                        self._recorder = AudioRecorder(device)
                        self._recorder.start()
                        self.status.emit("检测到X5 USB麦克风，本次录制带声")
                    except Exception:
                        self._recorder = None
                if self._recorder is None:
                    self.status.emit("未检测到X5麦克风，本次录制为无声")
            self.audio_note.emit("有声视频（X5 USB麦克风）" if self._recorder
                                 else "无声视频（未检测到X5麦克风）")
            temp = Path(tempfile.mkdtemp(prefix="h360-gui-"))
            source, written = self._record(temp)
            if written == 0:
                raise RuntimeError("未采集到任何有效帧，不生成视频也不提交分析")
            self.status.emit("拍摄结束，视频已写入本地，开始本地分析")
            run_dir = self._unique_run_dir()
            run_dir.mkdir(parents=True)
            final_source = run_dir / "source.mp4"
            shutil.move(str(source), str(final_source))
            source = final_source
            self._recording_file = source
            cfg = self._build_config(source, run_dir)
            pipeline_run = self.pipeline_run or _default_pipeline_run()
            pipeline_run(cfg, progress=self._emit)
            source = None
            self.completed.emit(str(run_dir))
        except Exception as exc:
            partial = source if source is not None else self._recording_file
            if self._recording_frames == 0 and source is None:
                partial = None  # 一帧未写成的容器无效，不保留
            if partial is not None and Path(partial).is_file():
                keep = None
                try:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    if run_dir is not None:
                        keep = run_dir / "source.mp4"
                    elif isinstance(exc, _Cancelled):
                        keep = self.destination / f"cancelled-{stamp}.mp4"
                    else:
                        keep = self.destination / f"partial-{stamp}.mp4"
                    if keep is not None and not keep.exists():
                        shutil.move(str(partial), str(keep))
                except OSError:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if self._recorder is not None:
                self._recorder.stop()
            if temp is not None:
                shutil.rmtree(temp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 纯展示层：样式表与自定义绘制。不承载任何业务逻辑；业务槽只读写标准控件 API。
# ---------------------------------------------------------------------------

_QSS = """
QWidget#mainWindow {
    background: #1a1d23;
    color: #e6e9ef;
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    font-size: 13px;
}
QLabel { background: transparent; color: #e6e9ef; }
QLabel#brandTitle { font-size: 20px; font-weight: 700; color: #f2f5f9; }
QLabel#brandSubtitle { font-size: 12px; color: #6b7383; }
QLabel#envBadge {
    font-size: 11px; color: #34c98e;
    background: rgba(52, 201, 142, 0.10);
    border: 1px solid rgba(52, 201, 142, 0.35);
    border-radius: 11px; padding: 3px 10px;
}
QFrame#card { background: #22262e; border: 1px solid #30363f; border-radius: 12px; }
QLabel#cardTitle { font-size: 12px; font-weight: 600; color: #9aa3b2; letter-spacing: 2px; }
QLabel#controlLabel { color: #9aa3b2; font-size: 12px; }
QLabel#preview {
    background: #10131a; border: 1px solid #30363f; border-radius: 12px;
    color: #5b6474; font-size: 13px;
}
QLabel#noteLabel { color: #6b7383; font-size: 11px; }
QLabel#statusLabel { color: #9aa3b2; font-size: 12px; }
QLabel#stageChip { font-size: 11px; border-radius: 11px; }
QLabel#stageChip[state="pending"] { color: #6b7383; background: #161a21; border: 1px solid #2c313c; }
QLabel#stageChip[state="active"] { color: #6ea8ff; background: rgba(61, 123, 253, 0.14); border: 1px solid rgba(61, 123, 253, 0.55); }
QLabel#stageChip[state="done"] { color: #34c98e; background: rgba(52, 201, 142, 0.12); border: 1px solid rgba(52, 201, 142, 0.45); }
QLabel#stageChip[state="failed"] { color: #ef5566; background: rgba(239, 85, 102, 0.12); border: 1px solid rgba(239, 85, 102, 0.45); }
QLabel#stageArrow { color: #3a4150; font-size: 11px; }

QPushButton {
    min-height: 32px; padding: 0 16px; border-radius: 8px;
    font-size: 13px; border: 1px solid transparent;
}
QPushButton#startButton { background: #3d7bfd; color: #ffffff; font-weight: 600; }
QPushButton#startButton:hover { background: #2f6ae0; }
QPushButton#startButton:pressed { background: #2a5cc7; }
QPushButton#startButton:disabled { background: #2c313c; color: #5b6474; }
QPushButton#stopButton {
    background: rgba(224, 161, 62, 0.10);
    border: 1px solid rgba(224, 161, 62, 0.45); color: #e0a13e;
}
QPushButton#stopButton:hover { background: rgba(224, 161, 62, 0.18); }
QPushButton#stopButton:disabled { background: transparent; border-color: #2c313c; color: #5b6474; }
QPushButton#cancelButton {
    background: rgba(239, 85, 102, 0.08);
    border: 1px solid rgba(239, 85, 102, 0.40); color: #ef5566;
}
QPushButton#cancelButton:hover { background: rgba(239, 85, 102, 0.16); }
QPushButton#cancelButton:disabled { background: transparent; border-color: #2c313c; color: #5b6474; }
QPushButton#resultsButton { background: #2c313c; border: 1px solid #3a4150; color: #dfe4ee; }
QPushButton#resultsButton:hover { background: #343a47; }
QPushButton#resultsButton:disabled { background: #242832; border-color: #2c313c; color: #5b6474; }
QPushButton#clearButton { background: transparent; border: 1px solid #4a2c31; color: #c2575f; }
QPushButton#clearButton:hover { background: rgba(239, 85, 102, 0.10); }
QPushButton#clearButton:disabled { border-color: #2c313c; color: #5b6474; }

QSpinBox {
    background: #10131a; border: 1px solid #30363f; border-radius: 6px;
    padding: 3px 8px; color: #e6e9ef; min-height: 24px;
}
QSpinBox:focus { border-color: #3d7bfd; }
QSpinBox:disabled { color: #5b6474; background: #161a21; }
QSpinBox::up-button, QSpinBox::down-button { width: 18px; background: transparent; border: none; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #2c313c; }

QCheckBox { color: #dfe4ee; font-size: 13px; spacing: 6px; }
QCheckBox:disabled { color: #5b6474; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid #3a4150; background: #10131a;
}
QCheckBox::indicator:hover { border-color: #3d7bfd; }
QCheckBox::indicator:checked { background: #3d7bfd; border-color: #3d7bfd; }

QProgressBar {
    background: #10131a; border: 1px solid #30363f; border-radius: 8px;
    min-height: 18px; max-height: 18px;
    text-align: center; color: #c9d2e0; font-size: 11px;
}
QProgressBar::chunk { border-radius: 7px; background: #3d7bfd; }
QProgressBar#recordProgress::chunk { background: #34c98e; }
QProgressBar#writeProgress::chunk { background: #2fbfce; }
QProgressBar#processingProgress::chunk { background: #3d7bfd; }

QTableWidget#judgesTable { background: transparent; border: none; gridline-color: transparent; }
QTableWidget#judgesTable::item { border: none; }
QLabel#tableHead { color: #6b7383; font-size: 11px; padding-bottom: 2px; }

QPlainTextEdit#logView {
    background: #10131a; border: 1px solid #30363f; border-radius: 8px;
    color: #8f97a6; font-family: "Menlo", "SF Mono", "Consolas", monospace;
    font-size: 11px; padding: 6px 8px;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #3a4150; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #4a5160; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #2c313c; color: #dfe4ee; border: 1px solid #3a4150; padding: 4px 8px; }
"""


_STAGE_ORDER = ("detect", "regions", "judging", "export")
_STAGE_LABELS = {"detect": "检测", "regions": "分区", "judging": "评审", "export": "导出"}
# on_progress 写入进度条的阶段标签 → 阶段键；render 归入分区卡片展示。
_BAR_STAGE = {"人物检测": "detect", "场景分区": "regions", "区域画面生成": "regions",
              "评委并行评审": "judging", "导出高光": "export"}


class _StageProgressBar(QtWidgets.QProgressBar):
    """纯展示扩展：从自身 setFormat 文本解析阶段标签并广播，业务槽代码保持不变。"""

    stage_changed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stage = "idle"

    def setFormat(self, text):
        super().setFormat(text)
        text = str(text)
        head, sep, _ = text.partition("：")
        if sep and head in _BAR_STAGE:
            stage = _BAR_STAGE[head]
        elif "失败" in text:
            stage = "failed"
        elif "完成" in text:
            stage = "done"
        else:
            stage = "idle"
        if stage != self._stage:
            self._stage = stage
            self.stage_changed.emit(stage)


class _StageChips(QtWidgets.QWidget):
    """检测→分区→评审→导出 阶段徽章：当前蓝色、已过绿色、失败红色、未到灰色。纯展示。"""

    def __init__(self):
        super().__init__()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._chips = {}
        for index, key in enumerate(_STAGE_ORDER):
            if index:
                arrow = QtWidgets.QLabel("→")
                arrow.setObjectName("stageArrow")
                row.addWidget(arrow)
            chip = QtWidgets.QLabel(_STAGE_LABELS[key])
            chip.setObjectName("stageChip")
            chip.setAlignment(QtCore.Qt.AlignCenter)
            chip.setFixedSize(64, 22)
            chip.setProperty("state", "pending")
            row.addWidget(chip)
            self._chips[key] = chip
        row.addStretch(1)

    def set_active(self, stage: str):
        if stage == "idle":
            self._apply({key: "pending" for key in _STAGE_ORDER})
        elif stage == "done":
            self._apply({key: "done" for key in _STAGE_ORDER})
        elif stage == "failed":
            self._apply({key: ("failed" if chip.property("state") == "active"
                               else chip.property("state"))
                         for key, chip in self._chips.items()})
        elif stage in _STAGE_ORDER:
            current = _STAGE_ORDER.index(stage)
            self._apply({key: ("done" if i < current else "active" if i == current else "pending")
                         for i, key in enumerate(_STAGE_ORDER)})

    def _apply(self, states):
        for key, state in states.items():
            chip = self._chips[key]
            if chip.property("state") != state:
                chip.setProperty("state", state)
                chip.style().unpolish(chip)
                chip.style().polish(chip)


_JUDGE_BADGE = {
    "等待": ((138, 147, 163, 36), "#8a93a3"),
    "分析中": ((61, 123, 253, 46), "#6ea8ff"),
    "完成": ((52, 201, 142, 40), "#34c98e"),
    "失败/弃权": ((239, 85, 102, 40), "#ef5566"),
    "未启用": ((138, 147, 163, 30), "#9aa3b2"),
    "已结束": ((138, 147, 163, 30), "#9aa3b2"),
    "已中止": ((239, 85, 102, 30), "#c98790"),
}


class _JudgeDelegate(QtWidgets.QStyledItemDelegate):
    """评委表纯绘制：第1列等宽字体 id+模型名，第2列状态色徽章。"""

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = option.rect.adjusted(8, 0, -8, 0)
        text = index.data() or ""
        if index.column() == 0:
            super().paint(painter, option, index)
        elif index.column() == 2:
            if "弃权" in text or "失败" in text:
                bg, fg = _JUDGE_BADGE["失败/弃权"]
            elif "完成" in text:
                bg, fg = _JUDGE_BADGE["完成"]
            elif text == "未启用":
                bg, fg = _JUDGE_BADGE["未启用"]
            elif text == "等待":
                bg, fg = _JUDGE_BADGE["等待"]
            elif text in ("已结束", "已中止"):
                bg, fg = _JUDGE_BADGE[text]
            else:
                bg, fg = _JUDGE_BADGE["分析中"]
            font = QtGui.QFont(option.font)
            font.setPointSizeF(10.5)
            painter.setFont(font)
            width = painter.fontMetrics().horizontalAdvance(text) + 20
            badge = QtCore.QRect(rect.left(), rect.center().y() - 10, width, 20)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(*bg))
            painter.drawRoundedRect(badge, 10, 10)
            painter.setPen(QtGui.QColor(fg))
            painter.drawText(badge, QtCore.Qt.AlignCenter, text)
        else:
            ident, sep, model = text.partition(" · ")
            mono = QtGui.QFont("Menlo")
            mono.setStyleHint(QtGui.QFont.TypeWriter)
            mono.setPointSizeF(10.5)
            if sep:
                id_font = QtGui.QFont(mono)
                id_font.setBold(True)
                painter.setFont(id_font)
                painter.setPen(QtGui.QColor("#6b7383"))
                painter.drawText(rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, ident)
                offset = painter.fontMetrics().horizontalAdvance(ident + " · ")
                painter.setFont(mono)
                painter.setPen(QtGui.QColor("#dfe4ee"))
                painter.drawText(rect.adjusted(offset, 0, 0, 0),
                                 QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, model)
            else:
                painter.setFont(mono)
                painter.setPen(QtGui.QColor("#dfe4ee"))
                painter.drawText(rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)
        painter.restore()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全景拾光 · 拍摄控制台")
        self.resize(1180, 820)
        self.worker = None
        self.result_dir = None
        self.output_dir = PROJECT_ROOT / "output"
        self.last_image = None
        self.preview_refreshes = 0
        self.pending_close = False
        self._failed = False
        self._init_ui()

    def _note(self, audio_text: str) -> str:
        return (f"原始采集 2880×1440｜{audio_text}｜客户端不设 Token 上限，"
                f"服务商仍有限制｜进度按实际任务计算")

    def _init_ui(self):
        self.setObjectName("mainWindow")
        self.setStyleSheet(_QSS)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)
        brand = QtWidgets.QVBoxLayout()
        brand.setSpacing(2)
        title = QtWidgets.QLabel("全景拾光")
        title.setObjectName("brandTitle")
        brand.addWidget(title)
        subtitle = QtWidgets.QLabel("X5 全景拍摄 · 本地写片 · 可选评委并行评审")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(subtitle)
        header.addLayout(brand)
        header.addStretch(1)
        badge = QtWidgets.QLabel("本地运行 · 无需服务器")
        badge.setObjectName("envBadge")
        header.addWidget(badge, 0, QtCore.Qt.AlignVCenter)
        root.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(8)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)
        self.preview = QtWidgets.QLabel("连接 X5 并选择 USB 摄像头模式后，点击开始拍摄")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(640, 320)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        left.addWidget(self.preview, 1)
        self.preview_status = QtWidgets.QLabel("实时预览 · 等待画面")
        self.preview_status.setObjectName("noteLabel")
        left.addWidget(self.preview_status)
        self.note_label = QtWidgets.QLabel(self._note("声音待拍摄开始确认"))
        self.note_label.setObjectName("noteLabel")
        self.note_label.setWordWrap(True)
        left.addWidget(self.note_label)
        body.addLayout(left, 1)

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(8)

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("card")
        controls = QtWidgets.QVBoxLayout(controls_card)
        controls.setContentsMargins(16, 12, 16, 16)
        controls.setSpacing(8)
        controls_title = QtWidgets.QLabel("拍摄控制")
        controls_title.setObjectName("cardTitle")
        controls.addWidget(controls_title)
        params = QtWidgets.QHBoxLayout()
        params.setSpacing(8)
        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(2, 3600)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" 秒")
        self.camera_spin = QtWidgets.QSpinBox()
        self.camera_spin.setRange(0, 20)
        camera_index = find_camera_index()
        self.camera_spin.setValue(camera_index if camera_index is not None else 1)
        device_label = QtWidgets.QLabel("设备编号")
        device_label.setObjectName("controlLabel")
        params.addWidget(device_label)
        params.addWidget(self.camera_spin)
        duration_label = QtWidgets.QLabel("最长拍摄")
        duration_label.setObjectName("controlLabel")
        params.addWidget(duration_label)
        params.addWidget(self.duration_spin, 1)
        controls.addLayout(params)
        self.cloud_check = QtWidgets.QCheckBox("所选评委并行分析（云端计费）")
        self.cloud_check.setChecked(True)
        controls.addWidget(self.cloud_check)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = QtWidgets.QPushButton("开始拍摄")
        self.stop_btn = QtWidgets.QPushButton("停止拍摄")
        self.cancel_btn = QtWidgets.QPushButton("取消任务")
        self.start_btn.setObjectName("startButton")
        self.stop_btn.setObjectName("stopButton")
        self.cancel_btn.setObjectName("cancelButton")
        self.stop_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("立即停止采集与本地分析；已生成文件保留，不删除")
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        self.cancel_btn.clicked.connect(self.cancel_task)
        buttons.addWidget(self.start_btn, 1)
        buttons.addWidget(self.stop_btn, 1)
        buttons.addWidget(self.cancel_btn, 1)
        controls.addLayout(buttons)
        side.addWidget(controls_card)

        judges_card = QtWidgets.QFrame()
        judges_card.setObjectName("card")
        judges_box = QtWidgets.QVBoxLayout(judges_card)
        judges_box.setContentsMargins(16, 12, 16, 12)
        judges_box.setSpacing(4)
        judges_title = QtWidgets.QLabel("评委选择 · 每人进度")
        judges_title.setObjectName("cardTitle")
        judges_box.addWidget(judges_title)
        table_head = QtWidgets.QHBoxLayout()
        table_head.setContentsMargins(8, 0, 0, 0)
        table_head.setSpacing(0)
        head_use = QtWidgets.QLabel("选用")
        head_use.setObjectName("tableHead")
        head_use.setFixedWidth(42)
        table_head.addWidget(head_use)
        head_model = QtWidgets.QLabel("评委 · 模型")
        head_model.setObjectName("tableHead")
        table_head.addWidget(head_model, 1)
        head_state = QtWidgets.QLabel("候选 · 次数")
        head_state.setObjectName("tableHead")
        head_state.setFixedWidth(152)
        table_head.addWidget(head_state)
        judges_box.addLayout(table_head)
        self.judges = QtWidgets.QTableWidget(len(EXPERTS), 3)
        self.judges.setObjectName("judgesTable")
        self.judges.horizontalHeader().setVisible(False)
        self.judges.verticalHeader().setVisible(False)
        self.judges.verticalHeader().setDefaultSectionSize(30)
        self.judges.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.judges.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.judges.setFocusPolicy(QtCore.Qt.NoFocus)
        self.judges.setShowGrid(False)
        self.judges.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.judges.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.judges.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self.judges.setColumnWidth(0, 42)
        self.judges.setColumnWidth(2, 152)
        self.judges.setFixedHeight(30 * len(EXPERTS) + 4)
        self.judges.setItemDelegate(_JudgeDelegate(self.judges))
        self.judge_checks = []
        for row, expert in enumerate(EXPERTS):
            self.judges.setItem(row, 0, QtWidgets.QTableWidgetItem())
            check = QtWidgets.QCheckBox()
            check.setChecked(True)
            check.setObjectName(f"judgeCheck_{expert['id']}")
            check.setFocusPolicy(QtCore.Qt.NoFocus)
            check.setToolTip(f"启用评委 {expert['id']}")
            self.judges.setCellWidget(row, 0, check)
            self.judge_checks.append(check)
            model_item = QtWidgets.QTableWidgetItem(f"{expert['id']} · {expert['model']}")
            model_item.setToolTip(f"{expert['id']} · {expert['model']}")
            self.judges.setItem(row, 1, model_item)
            self.judges.setItem(row, 2, QtWidgets.QTableWidgetItem("等待"))
            check.toggled.connect(self._update_judge_selection_summary)
        judges_box.addWidget(self.judges)
        self.judge_selection_summary = QtWidgets.QLabel()
        self.judge_selection_summary.setObjectName("noteLabel")
        judges_box.addWidget(self.judge_selection_summary)
        self._update_judge_selection_summary()
        self.judges.setToolTip("评委请求一次性返回；进度显示当前候选和尝试次数")
        side.addWidget(judges_card)
        side.addStretch(1)
        side_w = QtWidgets.QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(384)
        body.addWidget(side_w)
        root.addLayout(body, 1)

        progress_card = QtWidgets.QFrame()
        progress_card.setObjectName("card")
        progress_box = QtWidgets.QVBoxLayout(progress_card)
        progress_box.setContentsMargins(16, 12, 16, 16)
        progress_box.setSpacing(8)
        progress_title = QtWidgets.QLabel("任务进度")
        progress_title.setObjectName("cardTitle")
        progress_box.addWidget(progress_title)
        self.record_progress = self._bar(progress_box, "拍摄", "recordProgress")
        self.write_progress = self._bar(progress_box, "写入本地视频", "writeProgress")
        self._stage_chips = _StageChips()
        progress_box.addWidget(self._stage_chips)
        self.progress = self._bar(progress_box, "本地分析", "processingProgress", stage_bar=True)
        self.progress.stage_changed.connect(self._stage_chips.set_active)
        root.addWidget(progress_card)

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)
        initial_status = ("就绪：采集与分析全部在本机完成，无需服务器"
                          if camera_index is not None else
                          "未检测到 X5；请检查 USB 连接和 Webcam 模式，连接后重新打开窗口")
        self.status_label = QtWidgets.QLabel(initial_status)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.results_btn = QtWidgets.QPushButton("打开结果目录")
        self.results_btn.setObjectName("resultsButton")
        self.results_btn.setEnabled(False)
        self.results_btn.clicked.connect(self.open_results)
        footer.addWidget(self.results_btn)
        self.clear_btn = QtWidgets.QPushButton("清除结果")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.setToolTip("删除本地output下的全部分析结果与源片副本，不可恢复")
        self.clear_btn.clicked.connect(self.clear_results)
        footer.addWidget(self.clear_btn)
        root.addLayout(footer)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMinimumHeight(56)
        self.log.setMaximumHeight(88)
        root.addWidget(self.log)

    @staticmethod
    def _bar(layout, label, object_name="", stage_bar=False):
        bar = _StageProgressBar() if stage_bar else QtWidgets.QProgressBar()
        if object_name:
            bar.setObjectName(object_name)
        bar.setRange(0, 1000)
        bar.setValue(0)
        bar.setFormat(label + "：等待")
        layout.addWidget(bar)
        return bar

    def start_capture(self):
        if self.worker and self.worker.isRunning():
            return
        selected_expert_ids = self.selected_expert_ids()
        if self.cloud_check.isChecked() and len(selected_expert_ids) < 2:
            QtWidgets.QMessageBox.warning(self, "评委数量不足", "至少选择2位评委，才能形成高光共识。")
            self.on_status("至少选择2位评委，才能形成高光共识")
            return
        for widget in (self.start_btn, self.duration_spin, self.camera_spin, self.cloud_check):
            widget.setEnabled(False)
        for check in self.judge_checks:
            check.setEnabled(False)
        for row, expert in enumerate(EXPERTS):
            self.judges.item(row, 2).setText(
                "等待" if expert["id"] in selected_expert_ids else "未启用")
        self.stop_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.results_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)  # P1：运行中禁用清除，避免删除正在写入的run目录导致源片丢失
        self.note_label.setText(self._note("声音待拍摄开始确认"))
        self.last_image = None
        self.preview_refreshes = 0
        self.preview_status.setText("实时预览 · 等待画面")
        self.preview.clear()
        self.preview.setText("正在准备相机，画面稳定后开始录制…")
        self._failed = False
        for bar in (self.record_progress, self.write_progress, self.progress):
            bar.setValue(0)
            bar.setFormat("等待真实进度")
        self.worker = CaptureWorker(self.duration_spin.value(), self.cloud_check.isChecked(),
                                    self.camera_spin.value(), selected_expert_ids=selected_expert_ids)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.audio_note.connect(lambda text: self.note_label.setText(self._note(text)))
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def stop_capture(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.on_status("正在正常停拍，随后在本机完成分析；不会重复提交")

    def cancel_task(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
            self.on_status("取消任务：停止采集与本地分析；已生成文件保留不删除")

    def on_frame(self, image):
        self.last_image = image
        self.preview_refreshes += 1
        self.preview_status.setText(
            f"实时预览 · 已刷新 {self.preview_refreshes} 次 · 最近 {time.strftime('%H:%M:%S')}")
        self._draw_preview()

    def _draw_preview(self):
        if self.last_image is not None:
            pixmap = QtGui.QPixmap.fromImage(self.last_image)
            self.preview.setPixmap(pixmap.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw_preview()

    def on_status(self, message):
        self.status_label.setText(message)
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def on_progress(self, event):
        stage = event.get("stage", "")
        done, total = event.get("done", 0), event.get("total", 0)
        if type(done) not in (int, float) or type(total) not in (int, float):
            return
        if not math.isfinite(done) or not math.isfinite(total) or not 0 <= done <= total:
            return
        labels = {"recording": "拍摄", "write": "写入本地视频", "detect": "人物检测",
                  "regions": "场景分区", "render": "区域画面生成", "judging": "评委并行评审",
                  "export": "导出高光"}
        bar = (self.record_progress if stage == "recording"
               else self.write_progress if stage == "write" else self.progress)
        ratio = done / total if total else 0
        bar.setValue(1000 if stage in ("recording", "write") and event.get("finished")
                     else round(ratio * 1000))
        counts = f"{done:g}/{total:g}"
        text = f"{labels.get(stage, stage)}：{counts}（{ratio:.0%}）" if total else f"{labels.get(stage, stage)}：无待处理任务"
        if event.get("finished") and stage == "recording":
            text = (f"拍摄：已正常停拍，保存 {done:g} 帧（最长 {total:g} 帧）"
                    if done < total else f"拍摄：完成，保存 {done:g} 帧")
        elif event.get("finished") and stage == "write":
            text = f"写入本地视频：完成，保存 {done:g} 帧"
        if stage == "judging":
            text += f" · 候选 {event.get('candidate_done', 0)}/{event.get('candidate_total', '?')}"
            if event.get("detail"):
                text += " · " + event["detail"]
        bar.setFormat(text)
        if stage == "recording" and event.get("finished"):
            self.stop_btn.setEnabled(False)
        if stage not in ("recording", "write"):
            self.status_label.setText(text)
        states = {"pending": "等待", "running": "分析中", "ok": "完成", "error": "失败/弃权"}
        candidate_index = event.get("candidate_index", event.get("candidate_done", 0) + 1)
        candidate_total = event.get("candidate_total", "?")
        for expert in event.get("experts", []):
            for row, item in enumerate(EXPERTS):
                if expert.get("id") == item["id"]:
                    status = expert.get("status")
                    attempt = expert.get("attempt", 1)
                    max_attempts = expert.get("max_attempts", 2)
                    if status == "running":
                        label = f"{candidate_index}/{candidate_total} · {attempt}/{max_attempts}次"
                    elif status == "ok":
                        label = f"{candidate_index}/{candidate_total} · 完成"
                    elif status == "error":
                        result = ("超时弃权" if expert.get("error_code") in {
                            "timeout", "http_408", "http_504"}
                                  else "弃权")
                        label = f"{candidate_index}/{candidate_total} · {result}"
                    else:
                        label = states.get(status, "未知")
                    self.judges.item(row, 2).setText(label)
        if stage == "export":
            # 评审阶段已结束；进度是节流快照，可能漏掉最后一帧全终态事件，
            # 仍显示"分析中"的行不可能真的在跑；完成后按评审记录校正。
            for row in range(self.judges.rowCount()):
                status = self.judges.item(row, 2).text()
                if status == "分析中" or ("次" in status and "弃权" not in status):
                    self.judges.item(row, 2).setText("已结束")
                elif status == "等待":
                    self.judges.item(row, 2).setText("未启用")

    def on_completed(self, directory):
        self.result_dir = directory
        self.progress.setValue(1000)
        self.progress.setFormat("本地分析完成，结果已保存 · 100%")
        self.results_btn.setEnabled(True)
        message = f"结果已保存：{directory}"
        timeline = Path(directory) / "全场高光时间轴.json"
        if timeline.is_file():
            try:
                data = json.loads(timeline.read_text(encoding="utf-8"))
                message += f"；共识高光 {data['summary']['event_count']} 个"
            except (ValueError, KeyError):
                pass
        else:
            candidates = Path(directory) / "candidates.json"
            if candidates.is_file():
                try:
                    data = json.loads(candidates.read_text(encoding="utf-8"))
                    message += f"；本地候选 {len(data.get('candidates', []))} 段（未调用云端）"
                except (ValueError, KeyError):
                    pass
        jury = sorted((Path(directory) / "jury").glob("candidate*.json"))
        reviews_by_id = {expert["id"]: [] for expert in EXPERTS}
        for report_path in jury:
            try:
                reviews = json.loads(report_path.read_text(encoding="utf-8")).get("reviews", [])
            except (ValueError, OSError):
                reviews = []
            for review in reviews:
                if review.get("expert_id") in reviews_by_id:
                    reviews_by_id[review["expert_id"]].append(review)
        candidate_count = len(jury)
        for row, expert in enumerate(EXPERTS):
            if not self.judge_checks[row].isChecked() or not self.cloud_check.isChecked():
                label = "未启用"
            elif candidate_count == 0:
                label = "未开始"
            else:
                rows = reviews_by_id[expert["id"]]
                successes = sum(item.get("status") == "ok" for item in rows)
                failures = len(rows) - successes
                missing = max(0, candidate_count - len(rows))
                label = f"{successes}/{candidate_count}完成"
                if failures:
                    label += f"·弃权{failures}"
                if missing:
                    label += f"·未处理{missing}"
            self.judges.item(row, 2).setText(label)
        self.on_status(message)

    def on_error(self, message):
        self._failed = True
        self.stop_btn.setEnabled(False)
        for row in range(self.judges.rowCount()):
            status = self.judges.item(row, 2).text()
            if status in ("分析中", "等待"):
                self.judges.item(row, 2).setText(
                    "已中止" if self.judge_checks[row].isChecked() and self.cloud_check.isChecked()
                    else "未启用")
        self.on_status("处理未完成：" + message)
        self.progress.setFormat("处理失败，保留最后实测进度；请查看日志")

    def on_worker_finished(self):
        for widget in (self.start_btn, self.duration_spin, self.camera_spin, self.cloud_check):
            widget.setEnabled(True)
        for check in self.judge_checks:
            check.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.clear_btn.setEnabled(True)  # 与 start_capture 的禁用配对；成功/失败/取消都会走到这里
        if self.pending_close:
            self.close()

    def open_results(self):
        if self.result_dir:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.result_dir))

    def selected_expert_ids(self) -> tuple[str, ...]:
        return tuple(expert["id"] for expert, check in zip(EXPERTS, self.judge_checks)
                     if check.isChecked())

    def _update_judge_selection_summary(self):
        selected = self.selected_expert_ids()
        count = len(selected)
        if count < 2:
            text = f"已选 {count}/5 位 · 至少选择2位以形成共识"
        else:
            required, _ = review_requirements(select_experts(selected))
            text = f"已选 {count}/5 位 · 至少 {required} 位有效"
        self.judge_selection_summary.setText(text)

    def clear_results(self):
        answer = QtWidgets.QMessageBox.question(
            self, "清除结果",
            f"删除 {self.output_dir} 下全部本地结果与源片副本？\n删除后不可恢复。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        root = Path(self.output_dir)
        if not root.is_dir():
            self.on_status("没有本地结果可清除")
            return
        removed = 0
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        self.result_dir = None
        self.results_btn.setEnabled(False)
        self.on_status(f"已清除本地结果 {removed} 项：{root}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.pending_close = True
            self.worker.cancel()
            self.on_status("正在安全关闭，请稍候；已生成文件不会删除")
            event.ignore()
        else:
            event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("全景拾光")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
