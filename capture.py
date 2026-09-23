"""电脑端本地采集：摄像头（macOS AVFoundation / Windows DirectShow）或文件模拟，直接写本地 mp4。

不再连接任何服务器；离线整片分析用 main.py，图形界面用 gui.py。
摄像头模式检测到 X5 USB 麦克风时同步录音，并按视频时间轴混音进成片；文件模拟一律无声。
"""
from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path
import queue
import re
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

from highlight360.audio import RATE, AudioRecorder, extract_pcm, find_microphone


def _is_glitch_black(frame) -> bool:
    """X5 Webcam流会间歇性吐出纯黑故障帧；真实暗场仍有噪点和结构，不会均值方差同零。"""
    sample = frame[::8, ::8]
    return float(sample.mean()) < 6 and float(sample.std()) < 6


_CAMERA_STABLE_SEC = 1.0
_CAMERA_START_TIMEOUT_SEC = 12.0
_CAMERA_QUEUE_FRAMES = 8


def find_camera_index(hint: str = "Insta360 X5") -> int | None:
    """查找 macOS AVFoundation 视频设备编号；编号会随 USB 插拔变化。"""
    if sys.platform != "darwin":
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    video_devices = result.stderr.split("AVFoundation audio devices:", 1)[0]
    for line in video_devices.splitlines():
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match and hint.casefold() in match.group(2).casefold():
            return int(match.group(1))
    return None


def _camera_backend():
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    raise RuntimeError("摄像头采集仅支持 macOS(AVFoundation) 或 Windows(DirectShow)；"
                       "其他平台请先用 --file 验证流程")


def _read_exact(stream, size: int, timeout_sec: float = 5.0) -> bytes:
    """从管道读满一帧；相机停流时及时报错，避免预览永久停在旧帧。"""
    chunks = bytearray()
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        fd = None
    deadline = time.monotonic() + timeout_sec
    while len(chunks) < size:
        if fd is None:
            chunk = stream.read(size - len(chunks))
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise RuntimeError("X5画面超过5秒没有送出完整新帧；请检查USB连接和相机Webcam模式")
            chunk = os.read(fd, min(size - len(chunks), 1024 * 1024))
        if not chunk:
            return b""
        chunks.extend(chunk)
    return bytes(chunks)


def _open_camera_reader(index: int, width: int, height: int):
    """返回 (read, close)。macOS 经 FFmpeg 读 AVFoundation，避开 OpenCV 设备索引漂移。"""
    if sys.platform == "darwin":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("macOS摄像头采集需要已安装并可在PATH找到FFmpeg")
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-f", "avfoundation", "-framerate", "30", "-video_size",
                   f"{width}x{height}", "-i", f"{index}:", "-an", "-pix_fmt",
                   "bgr24", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       bufsize=0)
        except OSError as exc:
            raise RuntimeError(f"无法启动FFmpeg摄像头采集：{exc}") from exc
        if process.stdout is None:
            process.kill()
            raise RuntimeError("FFmpeg没有提供原始视频输出管道")
        frame_bytes = width * height * 3

        def read():
            raw = _read_exact(process.stdout, frame_bytes)
            if not raw:
                code = process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"FFmpeg摄像头采集失败（退出码{code}，设备索引{index}，"
                                       f"请求尺寸{width}x{height}）")
                return False, None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            return True, frame

        def close():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            process.stdout.close()

        return read, close

    backend = _camera_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("不能打开指定摄像头，请确认设备处于Webcam模式并检查设备索引")
    for key, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                       (cv2.CAP_PROP_FRAME_WIDTH, width),
                       (cv2.CAP_PROP_FRAME_HEIGHT, height), (cv2.CAP_PROP_FPS, 30)):
        if not math.isclose(cap.get(key), value, rel_tol=0, abs_tol=0.001):
            cap.set(key, value)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap.read, cap.release


def _read_camera(index, width, height, fps, seconds, stopped, timeline, on_ready):
    if stopped():
        return
    read, close = _open_camera_reader(index, width, height)
    try:
        origin = previous = usable_at = open_at = None
        warmup_at = stable_at = last_received = None
        index_out = 0
        while not stopped():
            ok, current = read()
            now = time.monotonic()
            if stopped():
                return
            if not ok:
                raise RuntimeError("摄像头断流，不把断流视为正常停拍")
            if current.shape[:2] != (height, width):
                raise RuntimeError(f"摄像头实际输出{current.shape[1]}x{current.shape[0]}，"
                                   f"不是请求的{width}x{height}；不能自动猜测为完整全景")
            dark = _is_glitch_black(current)
            if origin is None:
                if warmup_at is None:
                    warmup_at = now
                if now - warmup_at > _CAMERA_START_TIMEOUT_SEC:
                    raise RuntimeError("相机启动后画面持续不稳定，未开始录制；请检查USB连接")
                gap = last_received is not None and now - last_received > max(0.2, 2 / fps)
                if dark or gap:
                    stable_at = None
                elif stable_at is None:
                    stable_at = now
                last_received = now
                if stable_at is None or now - stable_at < _CAMERA_STABLE_SEC:
                    continue
                origin = usable_at = open_at = now
                previous = current
                if timeline is not None:
                    timeline.append((now, now))
                if on_ready is not None:
                    on_ready()
                continue
            if now - usable_at > 5:
                raise RuntimeError("摄像头停顿超过5秒，停止采集以免伪造连续画面")
            if dark:
                continue
            if now - usable_at > 0.5:
                if timeline is not None:
                    timeline[-1] = (open_at, usable_at)
                    timeline.append((now, now))
                origin += now - usable_at
                open_at = now
            usable_at = now
            if timeline is not None:
                timeline[-1] = (open_at, now)
            received_at = now - origin
            while index_out / fps < received_at and not stopped():
                if index_out / fps >= seconds:
                    return
                index_out += 1
                yield previous
            previous = current
    finally:
        close()


def camera_frames(index: int, width: int, height: int, fps: int, seconds: float,
                  stop: threading.Event, timeline: list | None = None, on_ready=None):
    frames = queue.Queue(maxsize=_CAMERA_QUEUE_FRAMES)
    closing, finished = threading.Event(), threading.Event()
    errors = []

    def produce():
        source = _read_camera(index, width, height, fps, seconds,
                              lambda: closing.is_set() or stop.is_set(), timeline, on_ready)
        try:
            for frame in source:
                try:
                    frames.put_nowait(frame)
                except queue.Full:
                    raise RuntimeError("视频编码跟不上采集，帧缓冲已满；停止而不是丢帧或伪造连续画面") from None
        except Exception as exc:
            errors.append(exc)
        finally:
            source.close()
            finished.set()

    thread = threading.Thread(target=produce, name="h360-camera", daemon=True)
    thread.start()
    try:
        while True:
            if errors:
                raise errors[0]
            try:
                frame = frames.get(timeout=0.1)
            except queue.Empty:
                if finished.is_set():
                    if errors:
                        raise errors[0]
                    return
                continue
            yield frame
    finally:
        closing.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("相机驱动仍未释放，请暂勿重新开始拍摄")


def file_frames(path: str, projection: str, fps: int, seconds: float,
                stop: threading.Event, realtime: bool):
    from highlight360.io_video import VideoReader

    with VideoReader(path, projection) as reader:
        origin = time.monotonic()
        for timestamp, frame in reader.iter_frames(end_sec=min(seconds, reader.meta.duration_sec), sample_fps=fps):
            if realtime and stop.wait(max(0.0, origin + timestamp - time.monotonic())):
                break
            if stop.is_set():
                break
            yield frame


def _stop_on_enter(stop: threading.Event) -> None:
    try:
        input("录制中按回车正常结束；Ctrl+C视为中断，保留已写入的片段。\n")
    except EOFError:
        return
    stop.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera-index", type=int, help="Webcam设备索引（macOS走AVFoundation）")
    source.add_argument("--file", help="用已有视频模拟采集，不访问摄像头；输出无声")
    parser.add_argument("--projection", required=True, choices=("flat", "equirectangular"),
                        help="已拼接单目2:1全景选 equirectangular；普通视频选 flat，不按宽高比猜测")
    parser.add_argument("--out", required=True, help="输出 mp4 路径；必须不存在，绝不覆盖")
    parser.add_argument("--width", type=int, default=2880)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--fps", type=int, default=15, help="保存帧率；摄像头协商30fps后按接收时间采样")
    parser.add_argument("--duration", type=float, help="自动停拍秒数；未指定则回车结束，最多录制1小时")
    parser.add_argument("--fast-file", action="store_true", help="文件模拟不按真实时间等待，摄像头不适用")
    parser.add_argument("--no-audio", action="store_true", help="摄像头模式不录音，直接输出无声视频")
    return parser


def make_progress_printer():
    """CLI进度：按阶段打印一行；同一阶段内按百分比节流，避免逐帧刷屏。"""
    last = None

    def print_progress(update: dict) -> None:
        nonlocal last
        stage = str(update.get("stage", ""))
        done = int(update.get("done", 0))
        total = int(update.get("total", 0))
        unit = str(update.get("unit", ""))
        key = (stage, total, done if total <= 0 else done * 100 // total)
        if key == last:
            return
        last = key
        suffix = "（完成）" if total > 0 and done >= total else ""
        print(f"进度 {stage}：{done}/{total} {unit}{suffix}", flush=True)

    return print_progress


class _FFmpegWriter:
    """有界线程的实时 H.264 编码；隔离相机进程和 PyAV 动态库。"""

    def __init__(self, path, fps, size):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("macOS视频编码需要FFmpeg")
        self.errors = tempfile.TemporaryFile()
        self.closed = False
        try:
            self.process = subprocess.Popen(
                [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
                 f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "pipe:0",
                 "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                 "-threads", "2", "-pix_fmt", "yuv420p", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors)
        except BaseException:
            self.errors.close()
            raise

    def isOpened(self):
        return self.process.poll() is None

    def write(self, frame):
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
        except BrokenPipeError as exc:
            raise RuntimeError("FFmpeg视频编码进程提前退出") from exc

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired as exc:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("FFmpeg编码结束超时") from exc
            if code:
                self.errors.seek(0)
                detail = self.errors.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg编码失败：{detail.strip()}")
        finally:
            self.errors.close()


def record(out: str, frames, fps: int, size: tuple[int, int], total_frames: int,
           progress, recorder: AudioRecorder | None, timeline: list) -> str:
    """把帧序列写成 mp4；有录音器时按时间轴提取PCM并混音。返回最终文件路径。"""
    target = Path(out)
    silent = target.with_name(target.stem + ".nosound.mp4") if recorder is not None else target
    written = 0
    try:
        if target.exists() or silent.exists():
            raise FileExistsError(f"输出已存在，不能覆盖：{target}")
        writer = (_FFmpegWriter(silent, fps, size) if sys.platform == "darwin" else
                  cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"avc1"), fps, size))
        try:
            if not writer.isOpened():
                raise RuntimeError("macOS原生H264编码器启动失败，不能继续录制")
            for frame in frames:
                writer.write(frame)
                written += 1
                progress({"stage": "recording", "done": written, "total": total_frames,
                          "unit": "frames"})
            progress({"stage": "recording", "done": written, "total": total_frames,
                      "unit": "frames", "finished": True})
        finally:
            writer.release()
    finally:
        try:
            if hasattr(frames, "close"):
                frames.close()
        finally:
            if recorder is not None:
                recorder.stop()
    if recorder is not None:
        if written == 0:
            silent.unlink(missing_ok=True)
            raise RuntimeError("未写入任何帧，不生成带声视频")
        pcm = extract_pcm(recorder.snapshot(), timeline, 0.0, written / fps)
        if _mux_audio(str(silent), pcm, RATE, str(target)) and target.exists():
            silent.unlink()
            return str(target)
        return str(silent)
    return str(target)


def _mux_audio(video_path: str, pcm: np.ndarray, rate: int, out_path: str) -> bool:
    """子进程混音，避免PyAV与OpenCV的macOS AVFoundation动态库冲突。"""
    raw = Path(video_path).with_suffix(".pcm.raw")
    try:
        raw.write_bytes(np.ascontiguousarray(pcm, np.int16).tobytes())
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, numpy as np; from highlight360.io_video import mux_audio; "
             "mux_audio(sys.argv[1], np.fromfile(sys.argv[2], dtype=np.int16), "
             "int(sys.argv[3]), sys.argv[4])",
             str(video_path), str(raw), str(rate), str(out_path)],
            cwd=Path(__file__).resolve().parent,
            text=True, capture_output=True, check=True,
        )
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        print(f"警告：音频混流失败，保留无声视频。{message}", file=sys.stderr, flush=True)
        return False
    finally:
        raw.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    duration = args.duration if args.duration is not None else 3600.0
    checks = [
        (math.isfinite(duration) and 2 <= duration <= 3600, "录制时长须为2~3600秒"),
        (1 <= args.fps <= 60, "fps须在1~60之间"),
        (args.camera_index is None or args.camera_index >= 0, "摄像头索引不能为负数"),
    ]
    for valid, message in checks:
        if not valid:
            parser.error(message)
    if args.camera_index is not None and not sys.stdin.isatty() and args.duration is None:
        parser.error("无交互终端的摄像头采集必须指定 --duration，不能用stdin EOF作为正常停拍")
    stop = threading.Event()
    width, height = args.width, args.height
    recorder, timeline = None, []
    if args.file:
        from highlight360.io_video import VideoReader
        with VideoReader(args.file, args.projection) as reader:
            width, height = reader.meta.width, reader.meta.height
            total_frames = max(1, round(min(duration, reader.meta.duration_sec) * args.fps))
        frames = file_frames(args.file, args.projection, args.fps, duration, stop, not args.fast_file)
    else:
        if width < 2 or height < 2 or width % 2 or height % 2 or width * height > 40_000_000:
            parser.error("尺寸须为偶数正整数，总像素不超过4000万")
        if args.projection == "equirectangular" and width != height * 2:
            parser.error("已拼接单目全景必须为2:1")
        total_frames = max(1, round(duration * args.fps))
        if not args.no_audio:
            device = find_microphone()
            if device is not None:
                recorder = AudioRecorder(device)
                recorder.start()
                print("检测到X5 USB麦克风，本次录制带声", flush=True)
            else:
                print("未检测到X5麦克风，本次录制为无声", flush=True)
        frames = camera_frames(args.camera_index, width, height, args.fps, duration, stop,
                               timeline=timeline)
    if args.camera_index is not None and sys.stdin.isatty() and args.duration is None:
        threading.Thread(target=_stop_on_enter, args=(stop,), daemon=True).start()
    try:
        path = record(args.out, frames, args.fps, (width, height), total_frames,
                      make_progress_printer(), recorder, timeline)
    except Exception:
        if recorder is not None:
            recorder.stop()
        raise
    print(f"完成：{path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已中断；已写入的片段保留在输出文件中。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"采集失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
