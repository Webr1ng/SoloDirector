"""离线 unittest；所有视频 fixture 均在临时目录内用 PyAV 生成。"""
from __future__ import annotations

import dataclasses
import struct
import tempfile
import unittest
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from highlight360.io_video import VideoReader, VideoWriter, _metadata_check, _validate_frame


def make_video(path, size=(64, 48), pts=None, fps=10, metadata=None, hdr=None, tracks=1):
    pts = list(range(0, 1000, 100)) if pts is None else pts
    values = [24 + 20 * i for i in range(len(pts))]
    with av.open(str(path), "w") as container:
        if metadata:
            container.metadata.update(metadata)
        streams = []
        for _ in range(tracks):
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = size
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.codec_context.max_b_frames = 0
            stream.codec_context.gop_size = 2
            stream.options = {"crf": "0", "preset": "ultrafast"}
            if hdr:
                stream.codec_context.color_trc = hdr
                stream.codec_context.color_primaries = 9
                stream.codec_context.colorspace = 9
            streams.append(stream)
        for index, stamp in enumerate(pts):
            for stream in streams:
                frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), values[index], np.uint8), format="bgr24")
                frame.pts = stamp
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
        for stream in streams:
            for packet in stream.encode(None):
                container.mux(packet)
    return values


class VideoIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="视频IO测试_")
        cls.root = Path(cls.temp.name)
        cls.flat = cls.root / "普通视频.mp4"
        cls.equi = cls.root / "已拼接全景.mp4"
        make_video(cls.flat)
        make_video(cls.equi, size=(64, 32))
        cls.vfr = cls.root / "变帧率.mp4"
        cls.shifted = cls.root / "非零起始时间.mp4"
        cls.mkv = cls.root / "非零起始时间.mkv"
        cls.points = [0, 100, 400, 450, 900]
        make_video(cls.vfr, pts=cls.points)
        make_video(cls.shifted, pts=[2000 + p for p in cls.points])
        make_video(cls.mkv, pts=[2000 + p for p in cls.points])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_shade(self, frame, index):
        self.assertLess(abs(float(frame.mean()) - (24 + 20 * index)), 4)

    def test_explicit_projection_and_dataclass_contract(self):
        with self.assertRaises(TypeError):
            VideoReader(str(self.flat))
        for projection in ("auto", "", "stereo"):
            with self.subTest(projection=projection), self.assertRaisesRegex(ValueError, "显式"):
                VideoReader(str(self.flat), projection)
        with VideoReader(str(self.equi), "equirectangular") as reader:
            self.assertTrue(dataclasses.is_dataclass(reader.meta))
            names = {field.name for field in dataclasses.fields(reader.meta)}
            self.assertTrue({"width", "height", "fps", "duration_sec", "projection", "codec", "pixel_format", "container", "has_audio"} <= names)
            self.assertTrue(reader.meta.is_equirectangular)
            self.assertEqual((reader.meta.width, reader.meta.height), (64, 32))
            self.assertEqual(reader.meta.codec, "h264")
        # 普通 2:1 内容仍可以显式当作 flat；不做图像内容推断。
        with VideoReader(str(self.equi), "flat") as reader:
            self.assertFalse(reader.meta.is_equirectangular)
        with self.assertRaisesRegex(ValueError, "严格为 2:1"):
            VideoReader(str(self.flat), "equirectangular")

    def test_local_suffix_and_existence_validation(self):
        for suffix in (".insv", ".INSP", ".lrv"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "Insta360"):
                VideoReader(str(self.root / ("原始" + suffix)), "flat")
        with self.assertRaisesRegex(ValueError, "后缀"):
            VideoReader(str(self.root / "test.webm"), "flat")
        with self.assertRaises(FileNotFoundError):
            VideoReader(str(self.root / "missing.mp4"), "flat")
        for path in ("https://example.com/a.mp4", "file:///tmp/a.mp4", "//server/a.mp4", "\\\\server\\a.mp4"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "本地"):
                VideoReader(path, "flat")

    def test_empty_and_corrupt_files_are_errors(self):
        empty = self.root / "空文件.mp4"
        empty.write_bytes(b"")
        broken = self.root / "损坏文件.mp4"
        broken.write_bytes(b"this is not an MP4 video")
        with self.assertRaisesRegex(ValueError, "为空"):
            VideoReader(str(empty), "flat")
        with self.assertRaisesRegex(RuntimeError, "解码"):
            VideoReader(str(broken), "flat")

    def test_no_audio_and_chinese_path(self):
        with VideoReader(str(self.flat), "flat") as reader:
            self.assertFalse(reader.meta.has_audio)
            self.assertAlmostEqual(reader.meta.fps, 10)
            self.assertAlmostEqual(reader.meta.duration_sec, 1)
            self.assertEqual(reader.read_at(0).shape, (48, 64, 3))

    def test_sampling_is_end_exclusive_and_absolute(self):
        with VideoReader(str(self.flat), "flat") as reader:
            samples = list(reader.iter_frames())
            self.assertEqual([t for t, _ in samples], [0, .2, .4, .6, .8])
            for index, (_, frame) in enumerate(samples):
                self.assert_shade(frame, index * 2)
            first = list(reader.iter_frames(0, .4, 5))
            second = list(reader.iter_frames(.4, 1, 5))
            self.assertEqual([t for t, _ in first + second], [0, .2, .4, .6, .8])
            self.assertEqual([t for t, _ in reader.iter_frames(.15, .65, 4)], [.15, .4])
            self.assertEqual([t for t, _ in reader.iter_frames(.8, 10, 5)], [.8])
            self.assertEqual(list(reader.iter_frames(1, 1)), [])

    def test_vfr_uses_real_pts_not_average_fps(self):
        with av.open(str(self.vfr)) as container:
            raw = [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]
        np.testing.assert_allclose(raw, [p / 1000 for p in self.points])
        with VideoReader(str(self.vfr), "flat") as reader:
            self.assertAlmostEqual(reader.meta.duration_sec, 1)
            samples = list(reader.iter_frames(sample_fps=10))
            self.assertEqual([t for t, _ in samples], [i / 10 for i in range(10)])
            for (_, frame), expected in zip(samples, [0, 1, 1, 1, 2, 3, 3, 3, 3, 4]):
                self.assert_shade(frame, expected)
            self.assert_shade(reader.read_at(.399), 1)
            self.assert_shade(reader.read_at(.4), 2)
            self.assert_shade(reader.read_at(.449), 2)
            self.assert_shade(reader.read_at(.45), 3)
            self.assert_shade(reader.read_at(.999), 4)

    def test_nonzero_origin_mp4_and_mkv(self):
        for path in (self.shifted, self.mkv):
            with self.subTest(path=path):
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    if path.suffix == ".mp4":
                        self.assertGreater(stream.start_time, 0)
                    # MKV 可以完全没有 stream.start_time；参考点仍是首帧 PTS。
                    raw = [float(f.pts * f.time_base) for f in container.decode(stream)]
                    np.testing.assert_allclose(raw, [2 + p / 1000 for p in self.points])
                with VideoReader(str(path), "flat") as reader:
                    self.assertAlmostEqual(reader.meta.start_time_sec, 2)
                    self.assertAlmostEqual(reader.meta.duration_sec, 1)
                    samples = list(reader.iter_frames(.2, .8, sample_fps=5))
                    self.assertEqual([t for t, _ in samples], [.2, .4, .6])
                    for (_, frame), expected in zip(samples, [1, 2, 3]):
                        self.assert_shade(frame, expected)
                    self.assert_shade(reader.read_at(0), 0)
                    self.assert_shade(reader.read_at(.8), 3)
                    self.assert_shade(reader.read_at(.95), 4)

    def test_read_at_does_not_disturb_iteration(self):
        with VideoReader(str(self.shifted), "flat") as reader:
            first = reader.iter_frames(sample_fps=5)
            second = reader.iter_frames(.4, .8, sample_fps=5)
            try:
                self.assert_shade(next(first)[1], 0)
                self.assert_shade(reader.read_at(.95), 4)
                self.assert_shade(next(second)[1], 2)
                self.assert_shade(next(first)[1], 1)
                self.assertEqual([t for t, _ in first], [.4, .6, .8])
                self.assertEqual([t for t, _ in second], [.6])
                self.assertEqual(len(list(reader.iter_frames())), 5)
            finally:
                first.close()
                second.close()

    def test_read_at_invokes_timestamp_seek(self):
        with VideoReader(str(self.shifted), "flat") as reader:
            opened = []
            original = reader._open

            class TrackedContainer:
                def __init__(self):
                    self.container = original()
                    self.streams = self.container.streams
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    self.container.close()
                def close(self):
                    self.container.close()
                def seek(self, offset, **kwargs):
                    # time_base 必须在调用时刻捕获：容器关闭后 PyAV 的
                    # stream.time_base 可能返回 None，事后再读是不稳定的。
                    opened.append((offset, kwargs, kwargs["stream"].time_base))
                    return self.container.seek(offset, **kwargs)
                def decode(self, stream):
                    return self.container.decode(stream)

            with patch.object(reader, "_open", side_effect=TrackedContainer):
                self.assert_shade(reader.read_at(.45), 3)
            self.assertEqual(len(opened), 1)
            offset, kwargs, time_base = opened[0]
            self.assertEqual(offset, int(Fraction(245, 100) / time_base))
            self.assertTrue(kwargs["backward"])
            self.assertFalse(kwargs["any_frame"])

    def test_resize_even_dimensions_without_upscaling(self):
        with VideoReader(str(self.flat), "flat") as reader:
            self.assertEqual(reader.read_at(0, max_width=31).shape, (22, 30, 3))
            self.assertEqual(reader.read_at(0, max_width=128).shape, (48, 64, 3))
            for _, frame in reader.iter_frames(max_width=33):
                self.assertEqual(frame.shape, (24, 32, 3))
                self.assertEqual(frame.dtype, np.uint8)
            for width in (0, 1, -3, 10.5, True):
                with self.subTest(width=width), self.assertRaises(ValueError):
                    reader.read_at(0, max_width=width)

    def test_invalid_times_and_closed_reader(self):
        reader = VideoReader(str(self.flat), "flat")
        for kwargs in ({"start_sec": -1}, {"start_sec": 2}, {"end_sec": -1}, {"sample_fps": 0}, {"sample_fps": float("nan")}, {"end_sec": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                list(reader.iter_frames(**kwargs))
        for time in (-.01, 1, float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                reader.read_at(time)
        reader.close()
        reader.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            reader.read_at(0)

    def test_close_releases_suspended_iterator_and_reader_creates_no_files(self):
        files_before = set(self.root.iterdir())
        with VideoReader(str(self.flat), "flat") as reader:
            samples = reader.iter_frames(sample_fps=100)
            next(samples)
            self.assertEqual(len(reader._active_containers), 1)
        self.assertFalse(reader._active_containers)
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            next(samples)
        samples.close()
        self.assertEqual(set(self.root.iterdir()), files_before)

    def test_invalid_resolution_rate_duration_and_missing_pts(self):
        # 损坏头信息通常被 FFmpeg 自行修复；直接模拟这些不可用的探测结果。
        for problem in ("resolution", "rate", "duration", "pts", "time_base"):
            with self.subTest(problem=problem):
                ctx = SimpleNamespace(width=0 if problem == "resolution" else 64, height=48, color_trc=1, name="h264")
                stream = SimpleNamespace(
                    index=0, codec_context=ctx, time_base=Fraction(0 if problem == "time_base" else 1, 10),
                    average_rate=0 if problem == "rate" else 10, guessed_rate=None, base_rate=None,
                    start_time=0, duration=0 if problem == "duration" else 10, metadata={}, frames=1,
                )
                frame = SimpleNamespace(
                    is_corrupt=False, pts=None if problem == "pts" else 0, time_base=Fraction(1, 10),
                    width=64, height=48, duration=1, rotation=0, side_data=[], format=SimpleNamespace(name="yuv420p"),
                )
                container = SimpleNamespace(
                    streams=SimpleNamespace(video=[stream], audio=[]), metadata={},
                    decode=lambda _: iter([frame]), format=SimpleNamespace(name="mp4"),
                )
                reader = VideoReader.__new__(VideoReader)
                reader._av = av
                reader._open = lambda: nullcontext(container)
                with self.assertRaises(ValueError):
                    reader._probe("flat")

    def test_hdr_rejected_from_actual_bitstream(self):
        for transfer in (16, 18):
            path = self.root / f"HDR_{transfer}.mp4"
            make_video(path, hdr=transfer)
            with self.subTest(transfer=transfer), self.assertRaisesRegex(ValueError, "SDR Rec709"):
                VideoReader(str(path), "flat")

    def test_multiple_and_missing_video_tracks_rejected(self):
        multiple = self.root / "双视频轨道.mp4"
        make_video(multiple, tracks=2)
        with self.assertRaisesRegex(ValueError, "2 条"):
            VideoReader(str(multiple), "flat")
        audio = self.root / "仅音频.mp4"
        with av.open(str(audio), "w", format="mp4") as container:
            stream = container.add_stream("aac", rate=48000)
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
            frame.sample_rate = 48000
            frame.pts = 0
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        with self.assertRaisesRegex(ValueError, "0 条"):
            VideoReader(str(audio), "flat")

    def test_discoverable_projection_and_rotation_metadata_rejected(self):
        for metadata in ({"stereo_mode": "left_right"}, {"projection": "VR180"}, {"projection": "dual_fisheye"}, {"rotate": "90"}, {"stitched": "false"}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                _metadata_check(metadata)
        _metadata_check({"stereo_mode": "mono", "rotate": "0", "projection": "equirectangular"})
        for xml in (
            '<GSpherical:StereoMode>top-bottom</GSpherical:StereoMode>',
            '<rdf:Description GSpherical:Stitched="false"/>',
            '<GSpherical:ProjectionType>cubemap</GSpherical:ProjectionType>',
        ):
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                _metadata_check({"xmp": xml})
        _metadata_check({"xmp": '<GSpherical:StereoMode>mono</GSpherical:StereoMode>'})
        # Matroska 标签由实际容器保存并由 Reader 检查。
        stereo = self.root / "立体标签.mkv"
        make_video(stereo, size=(64, 32), metadata={"stereo_mode": "left_right"})
        with self.assertRaisesRegex(ValueError, "立体"):
            VideoReader(str(stereo), "equirectangular")
        rotated = self.root / "旋转标签.mp4"
        make_video(rotated)
        data = bytearray(rotated.read_bytes())
        tkhd = data.index(b"tkhd")
        self.assertEqual(data[tkhd + 4], 0)  # version 0 的 tkhd matrix 位于 type+44。
        struct.pack_into(">9i", data, tkhd + 44, 0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824)
        rotated.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "旋转"):
            VideoReader(str(rotated), "flat")

    def test_decode_failure_never_silently_returns_empty(self):
        with VideoReader(str(self.flat), "flat") as reader:
            with patch.object(reader, "_open", side_effect=av.error.InvalidDataError(1, "broken packet")):
                with self.assertRaisesRegex(RuntimeError, "解码失败"):
                    list(reader.iter_frames())
        # 即使解码器只标记 corrupt 而没有抛出 FFmpegError，也必须拒绝。
        with self.assertRaisesRegex(ValueError, "损坏帧"):
            _validate_frame(SimpleNamespace(is_corrupt=True), None, (64, 48))

    def test_writer_h264_decode_faststart_and_frame_count(self):
        output = self.root / "模型输入_无声.mp4"
        with VideoWriter(str(output), fps=10, size=(64, 48)) as writer:
            self.assertEqual(writer.frame_count, 0)
            for index in range(7):
                writer.write(np.full((48, 64, 3), 24 + index * 20, dtype=np.uint8))
            self.assertEqual(writer.frame_count, 7)
        writer.close()
        with av.open(str(output)) as container:
            self.assertEqual(container.format.name, "mov,mp4,m4a,3gp,3g2,mj2")
            self.assertEqual(len(container.streams.video), 1)
            self.assertFalse(container.streams.audio)
            stream = container.streams.video[0]
            self.assertEqual(stream.codec_context.name, "h264")
            self.assertEqual(stream.codec_context.pix_fmt, "yuv420p")
            self.assertEqual(stream.codec_context.color_trc, 1)
            self.assertEqual(stream.codec_context.color_primaries, 1)
            self.assertEqual(stream.codec_context.colorspace, 1)
            self.assertEqual(stream.codec_context.color_range, 1)
            self.assertEqual(stream.average_rate, 10)
            frames = list(container.decode(stream))
            self.assertEqual(len(frames), 7)  # close 必须刷新 B 帧延迟。
            np.testing.assert_allclose([f.time for f in frames], [i / 10 for i in range(7)])
        atoms = output.read_bytes()
        self.assertLess(atoms.index(b"moov"), atoms.index(b"mdat"))
        with VideoReader(str(output), "flat") as reader:
            self.assertAlmostEqual(reader.meta.duration_sec, .7)
            self.assertEqual(len(list(reader.iter_frames(sample_fps=10))), 7)
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            writer.write(np.zeros((48, 64, 3), np.uint8))

    def test_writer_refuses_existing_output_without_changing_bytes(self):
        output = self.root / "保留已有输出.mp4"
        output.write_bytes(b"existing-user-video")
        with self.assertRaisesRegex(FileExistsError, "不会覆盖"):
            VideoWriter(str(output), fps=10, size=(64, 48))
        self.assertEqual(output.read_bytes(), b"existing-user-video")

    def test_writer_uses_crf_16_and_keeps_fast_preset(self):
        output = self.root / "编码质量.mp4"
        with VideoWriter(str(output), fps=10, size=(64, 48)) as writer:
            writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
        bitstream = output.read_bytes()
        self.assertIn(b"crf=16.0", bitstream)
        self.assertIn(b"subme=2", bitstream)

    def test_writer_validation_and_empty_output(self):
        output = str(self.root / "参数校验.mp4")
        for fps, size in ((0, (64, 48)), (2.5, (64, 48)), (10, (63, 48)), (10, (64, 0))):
            with self.subTest(fps=fps, size=size), self.assertRaises(ValueError):
                VideoWriter(output, fps=fps, size=size)
        with VideoWriter(output, fps=10, size=(64, 48)) as writer:
            for frame in (np.zeros((46, 64, 3), np.uint8), np.zeros((48, 64, 3), np.float32), np.zeros((48, 64), np.uint8)):
                with self.assertRaisesRegex(ValueError, "BGR"):
                    writer.write(frame)
            writer.write(np.zeros((48, 64, 3), np.uint8))
        empty = VideoWriter(str(self.root / "无帧输出.mp4"), fps=10, size=(64, 48))
        with self.assertRaisesRegex(ValueError, "未写入"):
            empty.close()
        empty.close()


if __name__ == "__main__":
    unittest.main()
