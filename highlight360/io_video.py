"""本地SDR视频的时间戳读取与无声H.264输出，不负责全景拼接。"""
from __future__ import annotations

import math
import re
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator


_SUPPORTED = {".mp4", ".mov", ".mkv", ".m4v", ".avi"}


def _dependencies():
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("视频 I/O 需要本地安装 PyAV（av）和 numpy，请检查运行环境。") from exc
    return av, np


def _local_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("视频路径必须是非空的本地文件路径。")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", path) or path.startswith(("//", "\\\\")):
        raise ValueError("只接受本地视频文件，不接受 URL 或网络共享路径。")
    return Path(path).expanduser().resolve()


def _seconds(value: float, name: str) -> Fraction:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return Fraction(str(number))


def _frame_time(frame: Any) -> Fraction:
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        raise ValueError("视频帧缺少有效展示时间戳 PTS，无法保证原速，请重新导出视频。")
    return frame.pts * frame.time_base


def _metadata_check(metadata: dict) -> None:
    """只检查实际可见的标签，不声称能识别画面中的双鱼眼/立体内容。"""
    items = list(metadata.items())
    for value in metadata.values():
        # 可见 XMP 中的 GSpherical 元素/属性；不解析外部 XML 实体。
        items.extend(re.findall(
            r"\b(stereomode|projectiontype|stitched|rotation)\b\s*(?:=\s*[\"']|>\s*)([^<\"']+)",
            str(value).lower(),
        ))
    for key, value in items:
        key = str(key).lower()
        value = str(value).lower().strip()
        text = f"{key}={value}"
        compact = re.sub(r"[\s_\-]", "", text)
        if "rotate" in key or "rotation" in key:
            try:
                rotated = not math.isclose(float(value) % 360, 0.0, abs_tol=1e-6)
            except ValueError:
                rotated = True
            if rotated:
                raise ValueError("视频含旋转元数据，请先物理旋转画面并清除旋转标签后重新导出。")
        if any(word in compact for word in ("fisheye", "vr180", "halfequirectangular")):
            raise ValueError("检测到双鱼眼或 VR180 元数据；请先导出已拼接的单目 360° 等距柱状视频。")
        if "stereo" in key and value not in {"mono", "monoscopic", "2d", "0"}:
            raise ValueError("检测到立体视频元数据；仅支持普通平面或已拼接单目全景。")
        if re.search(r"stereomode[^a-z0-9]+(left[-_ ]?right|top[-_ ]?bottom|side[-_ ]?by[-_ ]?side)", text):
            raise ValueError("检测到立体全景元数据；请先导出单目视频。")
        if "stitched" in key and value in {"0", "false", "no"}:
            raise ValueError("视频元数据表明尚未拼接，请先完成单目全景拼接。")
        if "projection" in key and value not in {"equirectangular", "rectangular", "flat", "0"}:
            raise ValueError("检测到不支持的投影元数据；请导出普通平面或完整单目等距柱状视频。")
        if any(word in compact for word in ("smpte2084", "aribstdb67", "dolbyvision")) or (
            any(word in key for word in ("transfer", "color_trc", "hdr")) and value in {"pq", "hlg", "16", "18"}
        ):
            raise ValueError("不支持 HDR（PQ/HLG/Dolby Vision），请先色调映射并导出 SDR Rec709 视频。")


def _validate_frame(frame: Any, stream: Any, size: tuple[int, int]) -> None:
    if frame.is_corrupt:
        raise ValueError("视频解码得到损坏帧，请检查原文件或重新导出。")
    _frame_time(frame)
    if (frame.width, frame.height) != size:
        raise ValueError("视频中途改变分辨率，不支持此类视频，请按固定分辨率重新导出。")
    if stream.codec_context.color_trc in (16, 18):
        raise ValueError("不支持 HDR（PQ/HLG），请先色调映射并导出 SDR Rec709 视频。")
    if getattr(frame, "rotation", 0) % 360:
        raise ValueError("视频含旋转元数据，请先物理旋转画面并清除旋转标签后重新导出。")
    for side in frame.side_data:
        kind = side.type.name
        data = bytes(side)
        if kind == "STEREO3D" and (len(data) < 4 or struct.unpack_from("=i", data)[0] != 0):
            raise ValueError("检测到立体视频元数据；仅支持单目视频。")
        if kind == "SPHERICAL":
            # AVSphericalMapping: projection, yaw/pitch/roll, 四个裁切边界。
            if len(data) < 32 or struct.unpack_from("=i", data)[0] != 0 or any(struct.unpack_from("=4I", data, 16)):
                raise ValueError("检测到非完整等距柱状投影（可能为 VR180/裁切全景），请先重新导出单目 360° 视频。")
        if kind == "DISPLAYMATRIX":
            identity = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)
            if len(data) != 36 or struct.unpack("=9i", data) != identity:
                raise ValueError("视频含旋转/翻转显示矩阵，请先物理旋转画面并清除显示矩阵后重新导出。")
        if kind in {"MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL", "DYNAMIC_HDR_PLUS", "DOVI_RPU_BUFFER", "DOVI_METADATA", "DYNAMIC_HDR_VIVID"}:
            raise ValueError("检测到 HDR 元数据，请先色调映射并导出 SDR Rec709 视频。")


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    duration_sec: float
    projection: str
    codec: str
    pixel_format: str
    container: str
    has_audio: bool
    frame_count: int = 0  # 容器报告值；VFR/部分容器可为 0，不用于时间换算。
    start_time_sec: float = 0.0  # 首个可展示帧的原始 PTS 秒值。

    @property
    def is_equirectangular(self) -> bool:
        return self.projection == "equirectangular"


class VideoReader:
    def __init__(self, path: str, projection: str):
        if projection not in {"flat", "equirectangular"}:
            raise ValueError("projection 必须显式指定为 flat 或 equirectangular。")
        file = _local_path(path)
        if file.suffix.lower() in {".insv", ".insp", ".lrv"}:
            raise ValueError("不支持 .insv/.insp/.lrv 原始或代理文件，请先用 Insta360 软件导出已拼接单目 MP4。")
        if file.suffix.lower() not in _SUPPORTED:
            raise ValueError("不支持此视频后缀；仅接受 mp4、mov、mkv、m4v、avi，本模块不保证仅凭扩展名即可解码。")
        if not file.is_file():
            raise FileNotFoundError(f"本地视频文件不存在或不是文件：{file}")
        if file.stat().st_size == 0:
            raise ValueError(f"视频文件为空：{file}")
        self.path = str(file)
        self._av, _ = _dependencies()
        self._closed = False
        self._active_containers = set()
        self.meta = self._probe(projection)

    def _open(self):
        return self._av.open(self.path, mode="r", options={"err_detect": "explode"})

    @contextmanager
    def _decoding_container(self):
        self._check_open()
        container = self._open()
        self._active_containers.add(container)
        try:
            yield container
        finally:
            self._active_containers.discard(container)
            container.close()

    def _probe(self, projection: str) -> VideoMeta:
        try:
            with self._open() as container:
                videos = container.streams.video
                if len(videos) != 1:
                    raise ValueError(f"需要且仅支持一条视频轨道，当前为 {len(videos)} 条；请先导出单视频轨道文件。")
                stream = videos[0]
                self._stream_index = stream.index
                ctx = stream.codec_context
                width, height = ctx.width, ctx.height
                if width < 2 or height < 2:
                    raise ValueError("视频分辨率无效，宽高必须至少为 2 像素。")
                if projection == "equirectangular" and width != 2 * height:
                    raise ValueError("equirectangular 必须严格为 2:1，且由调用者确认是已拼接单目 360° 全景；不能据画面自动识别。")
                if stream.time_base is None or stream.time_base <= 0:
                    raise ValueError("视频缺少有效时间基准，无法按时间戳读取。")
                rate = stream.average_rate or stream.guessed_rate or stream.base_rate
                if rate is None or not math.isfinite(float(rate)) or rate <= 0:
                    raise ValueError("视频帧率无效，请按有效帧率重新导出。")
                _metadata_check(container.metadata)
                _metadata_check(stream.metadata)
                frames = iter(container.decode(stream))
                first = next(frames, None)
                if first is None:
                    raise ValueError("视频没有任何可解码的视频帧，可能为空或已损坏。")
                _validate_frame(first, stream, (width, height))
                self._origin = _frame_time(first)
                pixel_format = first.format.name
                stream_start = stream.start_time * stream.time_base if stream.start_time is not None else self._origin
                self._duration = (stream_start + stream.duration * stream.time_base - self._origin) if stream.duration is not None else None
                if stream.start_time is None and self._origin != 0:
                    # MKV 的 DURATION 可能是绝对结束时间，不能当作首帧起算时长。
                    self._duration = None
                if self._duration is None:
                    # 轨道时长缺失或含糊时只流式扫描时间戳，不写整段中间副本。
                    last_time = self._origin
                    last_duration = first.duration * first.time_base if first.duration > 0 else None
                    last_step = None
                    del first
                    for frame in frames:
                        _validate_frame(frame, stream, (width, height))
                        stamp = _frame_time(frame)
                        if stamp <= last_time:
                            raise ValueError("视频展示时间戳不递增，无法保证原速，请重新导出。")
                        last_step = stamp - last_time
                        last_time = stamp
                        last_duration = frame.duration * frame.time_base if frame.duration > 0 else None
                    self._duration = last_time - self._origin + (last_duration or last_step or 1 / rate)
                if self._duration <= 0 or not math.isfinite(float(self._duration)):
                    raise ValueError("视频有效时长必须大于零，请检查文件或重新导出。")
                return VideoMeta(
                    width=width, height=height, fps=float(rate), duration_sec=float(self._duration),
                    projection=projection, codec=ctx.name, pixel_format=pixel_format,
                    container=container.format.name, has_audio=bool(container.streams.audio),
                    frame_count=stream.frames or 0, start_time_sec=float(self._origin),
                )
        except self._av.FFmpegError as exc:
            raise RuntimeError(f"无法解码视频（后缀不保证编码受支持，文件也可能损坏）：{self.path}；{exc}") from exc

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("VideoReader 已关闭，不能继续读取。")

    def _output_size(self, max_width: int | None) -> tuple[int, int]:
        width, height = self.meta.width, self.meta.height
        if max_width is not None:
            if isinstance(max_width, bool) or not isinstance(max_width, Integral) or max_width < 2:
                raise ValueError("max_width 必须是至少为 2 的整数。")
            limit = min(width, int(max_width))
        else:
            limit = width
        out_width = limit // 2 * 2
        out_height = (height * out_width // width) // 2 * 2
        if out_height < 2:
            raise ValueError("max_width 太小，无法在不放大的前提下得到有效偶数宽高。")
        return out_width, out_height

    def _decoded(self, start: Fraction, seek: bool) -> Iterator[tuple[Fraction, Any]]:
        """每次迭代独占一个 demuxer/decoder；不共享 seek 状态。"""
        try:
            with self._decoding_container() as container:
                stream = container.streams[self._stream_index]
                if seek:
                    absolute = self._origin + start
                    container.seek(math.floor(absolute / stream.time_base), stream=stream, backward=True, any_frame=False)
                last_time = None
                last_duration = None
                for frame in container.decode(stream):
                    self._check_open()
                    _validate_frame(frame, stream, (self.meta.width, self.meta.height))
                    stamp = _frame_time(frame) - self._origin
                    if last_time is None and seek and stamp > start:
                        # 某些容器索引只能落在目标之后，重新打开后顺序解码，不用 fps 索引猜测。
                        del frame
                        container.close()
                        yield from self._decoded(start, seek=False)
                        return
                    if last_time is not None and stamp <= last_time:
                        raise ValueError("视频展示时间戳不递增，无法保证原速，请重新导出。")
                    last_time = stamp
                    last_duration = frame.duration * frame.time_base if frame.duration > 0 else None
                    yield stamp, frame
                if last_time is None:
                    raise RuntimeError("视频没有可解码帧或 seek 后无法解码，请检查文件是否损坏。")
                if last_duration is not None and last_time + last_duration + 2 * stream.time_base < self._duration:
                    raise RuntimeError("视频提前结束：解码时长短于声明时长，文件可能被截断或损坏。")
        except self._av.FFmpegError as exc:
            raise RuntimeError(f"视频时间戳定位/解码失败：{self.path}；{exc}") from exc

    def _samples(self, start: Fraction, end: Fraction, step: Fraction, size: tuple[int, int], seek: bool):
        decoded = self._decoded(start, seek)
        try:
            before = next(decoded)
            after = next(decoded, None)
            index = 0
            while True:
                self._check_open()
                target = start + index * step
                if target >= end or abs(float(target - end)) < 1e-9:
                    break
                while after is not None and after[0] <= target:
                    before = after
                    after = next(decoded, None)
                if before[0] > target:
                    raise RuntimeError("目标时间之前没有可展示的视频帧，请检查时间戳。")
                # 目标时间处展示的是最近一张 PTS <= target 的帧，而非未来帧。
                # VFR 的长帧按其展示区间保持，不能用平均 fps 当作帧索引。
                yield float(target), before[1].to_ndarray(width=size[0], height=size[1], format="bgr24")
                index += 1
        except self._av.FFmpegError as exc:
            raise RuntimeError(f"视频帧转换失败：{exc}") from exc
        finally:
            decoded.close()

    def iter_frames(self, start_sec: float = 0.0, end_sec: float | None = None,
                    sample_fps: float = 5.0, max_width: int | None = None) -> Iterator[tuple[float, Any]]:
        """按原片PTS采样 [start,end)，返回首展示帧起算的时间，VFR长帧允许重复。"""
        self._check_open()
        start = _seconds(start_sec, "start_sec")
        end = self._duration if end_sec is None else _seconds(end_sec, "end_sec")
        rate = _seconds(sample_fps, "sample_fps")
        if start < 0 or start > self._duration:
            raise ValueError("start_sec 必须在视频有效时长范围内。")
        if end < start:
            raise ValueError("end_sec 不能小于 start_sec。")
        if rate <= 0:
            raise ValueError("sample_fps 必须大于零。")
        size = self._output_size(max_width)
        end = min(end, self._duration)
        if start < end:
            yield from self._samples(start, end, 1 / rate, size, seek=start > 0)

    def read_at(self, time_sec: float, max_width: int | None = None):
        """按时间戳 seek 到前一关键帧再解码，读取目标时间正在展示的 BGR 帧。"""
        self._check_open()
        target = _seconds(time_sec, "time_sec")
        if target < 0 or target >= self._duration:
            raise ValueError("time_sec 必须满足 0 <= time_sec < 视频有效时长。")
        samples = self._samples(target, self._duration, self._duration, self._output_size(max_width), seek=True)
        try:
            return next(samples)[1]
        finally:
            samples.close()

    def close(self) -> None:
        self._closed = True
        # 即使调用者保留尚未耗尽的生成器，退出 reader 上下文也释放文件句柄。
        for container in tuple(self._active_containers):
            container.close()
        self._active_containers.clear()

    def __enter__(self) -> "VideoReader":
        self._check_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_audio_pcm(path: str, start: float, end: float, rate: int = 48000):
    """解码源文件音频 [start,end) 并重采样为单声道PCM16；无音轨返回None。"""
    av, np = _dependencies()
    if end < start or not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("音频读取窗口必须有限且 end >= start")
    target = int(round((end - start) * rate))
    pcm = np.zeros(target, np.int16)
    with av.open(str(_local_path(path)), mode="r") as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        video = container.streams.video[0] if container.streams.video else None
        origin = video.start_time * video.time_base if video is not None and video.start_time is not None else Fraction(0)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        cursor = 0

        def take(frame):
            nonlocal cursor
            for out in resampler.resample(frame):
                if out.pts is None or out.time_base is None:
                    raise ValueError("音频缺少有效PTS，无法保证音画同步")
                position = round((out.pts * out.time_base - origin - Fraction(str(start))) * rate)
                samples = out.to_ndarray().reshape(-1)
                left, right = max(0, cursor, position), min(target, position + len(samples))
                if left < right:
                    pcm[left:right] = samples[left - position:right - position]
                    cursor = right

        for frame in container.decode(stream):
            take(frame)
        take(None)
    return pcm


def mono_audio_frame(pcm_block, rate: int, pts: int):
    """显式单声道PCM帧；from_ndarray会按默认双声道推断形状，不能用于mono。"""
    av, np = _dependencies()
    frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm_block))
    frame.sample_rate = rate
    frame.pts = pts
    frame.time_base = Fraction(1, rate)
    np.frombuffer(frame.planes[0], dtype=np.int16)[:] = pcm_block
    return frame


def mux_audio(video_path: str, pcm, rate: int, out_path: str) -> None:
    """视频包原样复制，PCM编码为AAC后按PTS交织写入新MP4。"""
    av, np = _dependencies()
    source = _local_path(video_path)
    target = _local_path(out_path)
    if not pcm.dtype == np.int16 or pcm.ndim != 1:
        raise ValueError("音频必须是一维int16样本")
    with av.open(str(source), mode="r") as incoming, av.open(
            str(target), "w", format="mp4", options={"movflags": "+faststart"}) as output:
        vsrc = incoming.streams.video[0]
        if len(incoming.streams.video) != 1:
            raise ValueError("待混音视频必须只有一条视频轨道")
        vdst = output.add_stream_from_template(vsrc)
        adst = output.add_stream("aac", rate=rate)
        adst.codec_context.layout = "mono"
        adst.codec_context.bit_rate = 64000
        output.start_encoding()
        encoded = []
        for start in range(0, len(pcm), 1024):
            block = pcm[start:start + 1024]
            frame = mono_audio_frame(block, rate, start)
            for packet in adst.encode(frame):
                encoded.append((packet.pts * packet.time_base, packet))
        for packet in adst.encode(None):
            encoded.append((packet.pts * packet.time_base, packet))
        encoded.sort(key=lambda pair: pair[0])
        pending = iter(encoded)
        held = next(pending, None)
        for packet in incoming.demux(vsrc):
            if not packet.size:
                continue
            stamp = packet.pts * packet.time_base if packet.pts is not None else Fraction(0)
            while held is not None and held[0] <= stamp:
                held[1].stream = adst
                output.mux(held[1])
                held = next(pending, None)
            packet.stream = vdst
            output.mux(packet)
        while held is not None:
            held[1].stream = adst
            output.mux(held[1])
            held = next(pending, None)


class VideoWriter:
    """流式写无声 libx264/yuv420p/faststart MP4；不是原生 Live Photo。"""
    def __init__(self, path: str, fps: int, size: tuple[int, int]):
        file = _local_path(path)
        if file.suffix.lower() != ".mp4":
            raise ValueError("VideoWriter 仅输出 .mp4 格式的无声 H.264 视频。")
        if not file.parent.is_dir():
            raise FileNotFoundError(f"输出视频的父目录不存在：{file.parent}")
        if file.exists():
            raise FileExistsError(f"输出视频已存在，不会覆盖：{file}")
        if isinstance(fps, bool) or not isinstance(fps, Integral) or fps <= 0:
            raise ValueError("输出 fps 必须是正整数。")
        if not isinstance(size, tuple) or len(size) != 2 or any(
            isinstance(x, bool) or not isinstance(x, Integral) or x < 2 or x % 2 for x in size
        ):
            raise ValueError("输出 size 必须是 (宽, 高)，且宽高为至少 2 的偶数整数。")
        self.path = str(file)
        self.fps = int(fps)
        self.size = tuple(map(int, size))
        self._av, self._np = _dependencies()
        self._container = None
        self._closed = False
        self._frame_count = 0
        try:
            self._container = self._av.open(self.path, mode="w", format="mp4", options={"movflags": "+faststart"})
            self._stream = self._container.add_stream("libx264", rate=self.fps)
            self._stream.width, self._stream.height = self.size
            self._stream.pix_fmt = "yuv420p"
            self._stream.time_base = Fraction(1, self.fps)
            self._stream.options = {"crf": "16", "preset": "veryfast"}
            ctx = self._stream.codec_context
            ctx.color_primaries = ctx.color_trc = ctx.colorspace = 1  # SDR BT.709
            ctx.color_range = 1  # limited / MPEG
            ctx.open()  # 在写入前明确发现 libx264 缺失等编码器错误。
        except (self._av.FFmpegError, ValueError) as exc:
            if self._container is not None:
                self._container.close()
            raise RuntimeError(f"无法创建 libx264 MP4 编码器，请检查本地 PyAV 编码支持：{exc}") from exc

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write(self, bgr) -> None:
        if self._closed:
            raise RuntimeError("VideoWriter 已关闭，不能写入。")
        if not isinstance(bgr, self._np.ndarray) or bgr.dtype != self._np.uint8 or bgr.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"写入帧必须是 uint8 BGR 数组，尺寸为 {self.size[0]}x{self.size[1]}，三通道。")
        try:
            frame = self._av.VideoFrame.from_ndarray(self._np.ascontiguousarray(bgr), format="bgr24")
            frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG")
            frame.pts = self._frame_count
            frame.time_base = Fraction(1, self.fps)
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self._frame_count += 1
        except self._av.FFmpegError as exc:
            raise RuntimeError(f"H.264 视频编码/写入失败：{exc}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._frame_count:
                    for packet in self._stream.encode(None):
                        self._container.mux(packet)
            finally:
                self._container.close()
        except self._av.FFmpegError as exc:
            raise RuntimeError(f"H.264 编码器刷新或 MP4 收尾失败：{exc}") from exc
        if not self._frame_count:
            raise ValueError("未写入任何视频帧，不能生成有效的 MP4 视频。")

    def __enter__(self) -> "VideoWriter":
        if self._closed:
            raise RuntimeError("VideoWriter 已关闭。")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
