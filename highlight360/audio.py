"""X5 USB麦克风录音与按视频时间轴切片；无设备或未装依赖时整条链路走无声路径。

视频时间轴会扣除摄像头断流间隔（见 capture.camera_frames 的 timeline），
音频按同一组区间映射回墙钟取样本，因此成片音画对齐而不是按墙钟硬贴。
"""
from __future__ import annotations

import math
import threading
import time
import wave

import numpy as np

try:
    import sounddevice
except ImportError:  # 客户端环境可能没装音频依赖；此时一律无声会话。
    sounddevice = None

RATE = 48000
DEVICE_HINT = "Insta360"
CHUNK_TOLERANCE_SEC = 0.06


def find_microphone(hint: str = DEVICE_HINT):
    """返回匹配hint的录音设备索引；没有录音设备或依赖缺失时返回None。"""
    if sounddevice is None:
        return None
    try:
        devices = sounddevice.query_devices()
    except Exception:
        return None
    for index, device in enumerate(devices):
        name = str(device.get("name", "")).lower()
        if device.get("max_input_channels", 0) > 0 and hint.lower() in name:
            return index
    return None


class AudioRecorder:
    """回调线程追加 (monotonic起始秒, int16单声道样本)；停止后按块切片。"""

    def __init__(self, device: int, rate: int = RATE):
        if sounddevice is None:
            raise RuntimeError("未安装 sounddevice，不能录音")
        self.device, self.rate = int(device), int(rate)
        self._blocks: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self._origin = None
        self._samples_received = 0
        self._error = None
        self._stream = sounddevice.InputStream(
            samplerate=self.rate, channels=1, dtype="int16",
            blocksize=self.rate // 100, device=self.device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        now = time.monotonic()
        samples = np.frombuffer(indata, dtype=np.int16).reshape(frames, -1)[:, 0].copy()
        with self._lock:
            if status.input_overflow:
                self._error = "麦克风采集溢出，音频样本已丢失；请重新录制"
            if self._error:
                return
            if self._origin is None:
                self._origin = now + time_info.inputBufferAdcTime - time_info.currentTime
            # MME会批量交付回调；后续块只按样本计时，不能把调度间隙变成静音。
            stamp = self._origin + self._samples_received / self.rate
            self._blocks.append((stamp, samples))
            self._samples_received += len(samples)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
            return list(self._blocks)


def extract_pcm(blocks: list[tuple[float, np.ndarray]], intervals: list[tuple[float, float]],
                start: float, end: float, rate: int = RATE) -> np.ndarray:
    """取视频时间轴 [start,end) 对应的单声道int16样本，长度精确为 round((end-start)*rate)。

    intervals 为 [(墙钟起, 墙钟止), ...]，第 i 段对应时间轴基线为前面各段时长之和；
    录音块之间的缺口补静音，保证时间轴长度不被丢块缩短。
    """
    if end < start or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("音频切片窗口必须有限且 end >= start")
    target = int(round((end - start) * rate))
    out = np.zeros(target, np.int16)
    base = 0.0
    for wall_start, wall_end in intervals:
        length = wall_end - wall_start
        if length <= 0:
            continue
        begin, finish = max(start, base), min(end, base + length)
        wall_begin = wall_start + begin - base
        base += length
        if begin >= finish:
            continue
        dst_start = round((begin - start) * rate)
        dst_end = min(target, round((finish - start) * rate))
        cursor = dst_start
        for stamp, samples in blocks:
            position = dst_start + round((stamp - wall_begin) * rate)
            left = max(cursor, position)
            right = min(dst_end, position + len(samples))
            if left < right:
                out[left:right] = samples[left - position:right - position]
                cursor = right
    return out


def write_wav(path, pcm: np.ndarray, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.ascontiguousarray(pcm, np.int16).tobytes())


def parse_wav_pcm(path, rate: int = RATE) -> np.ndarray:
    """读取PCM16单声道WAV；格式不符直接报错，不做静默转换。"""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != rate:
            raise ValueError(f"WAV必须是{rate}Hz单声道PCM16")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).copy()


def audio_part_seconds_ok(pcm: np.ndarray, seconds: float, rate: int = RATE,
                          tolerance: float = CHUNK_TOLERANCE_SEC) -> bool:
    return abs(len(pcm) / rate - seconds) <= tolerance
