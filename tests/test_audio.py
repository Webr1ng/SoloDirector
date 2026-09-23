"""音频时间轴切片、WAV边车合同与导出混音的离线测试；不访问麦克风。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import wave

import numpy as np

from highlight360.audio import (RATE, audio_part_seconds_ok, extract_pcm,
                                parse_wav_pcm, write_wav)


def block(start: float, value: int, count: int, rate: int = 100):
    return (start, np.full(count, value, np.int16))


class ExtractPcmTests(unittest.TestCase):
    def test_cancel_and_worker_cleanup_stop_audio_stream_only_once(self):
        from unittest.mock import patch
        from highlight360.audio import AudioRecorder
        with patch("highlight360.audio.sounddevice") as device:
            recorder = AudioRecorder(0)
            recorder.start()
            recorder.stop()
            recorder.stop()
            device.InputStream.return_value.stop.assert_called_once()
            device.InputStream.return_value.close.assert_called_once()

    def test_batched_callbacks_use_sample_clock_not_arrival_times(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from highlight360.audio import AudioRecorder
        original = ((np.arange(RATE * 2) % 10000) + 1).astype(np.int16)
        stamps = [100 + (i // 3) * 0.03 + (i % 3) * 0.0001 for i in range(200)]
        with patch("highlight360.audio.sounddevice"), \
                patch("highlight360.audio.time.monotonic", side_effect=stamps):
            recorder = AudioRecorder(0)
            for i in range(200):
                info = SimpleNamespace(inputBufferAdcTime=10 + (i % 3) * .01, currentTime=10.02)
                recorder._callback(original[i * 480:(i + 1) * 480, None], 480, info,
                                   SimpleNamespace(input_overflow=False))
            blocks = recorder.snapshot()
            recorder.stop()
        origin = blocks[0][0]
        self.assertAlmostEqual(origin, 99.98)
        self.assertTrue(np.allclose(np.diff([t for t, _ in blocks]), .01))
        actual = extract_pcm(blocks, [(origin, origin + 2)], 0, 2)
        np.testing.assert_array_equal(actual, original)
        self.assertEqual(int(np.count_nonzero(actual == 0)), 0)

    def test_actual_microphone_overflow_is_reported_not_hidden_by_silence(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from highlight360.audio import AudioRecorder
        with patch("highlight360.audio.sounddevice"):
            recorder = AudioRecorder(0)
            recorder._callback(np.ones((480, 1), np.int16), 480,
                               SimpleNamespace(inputBufferAdcTime=0, currentTime=0),
                               SimpleNamespace(input_overflow=True))
            with self.assertRaisesRegex(RuntimeError, "采集溢出"):
                recorder.snapshot()
            recorder.stop()

    def test_missing_interval_tail_does_not_pull_next_interval_audio_forward(self):
        blocks = [block(0, 1, 80), block(2, 3, 100)]
        actual = extract_pcm(blocks, [(0, 1), (2, 3)], 0, 2, rate=100)
        np.testing.assert_array_equal(actual, np.r_[np.ones(80), np.zeros(20), np.full(100, 3)])

    def test_overlap_is_placed_at_timestamp_without_duplicating_samples(self):
        blocks = [block(0, 1, 100), (.5, np.arange(200, 300, dtype=np.int16))]
        actual = extract_pcm(blocks, [(0, 1.5)], 0, 1.5, rate=100)
        np.testing.assert_array_equal(actual, np.r_[np.ones(100), np.arange(250, 300)])

    def test_single_interval_slices_by_wall_time(self):
        blocks = [block(0.0, 1, 100), block(1.0, 2, 100)]
        out = extract_pcm(blocks, [(0.0, 2.0)], 0.5, 1.5, rate=100)
        self.assertEqual(len(out), 100)
        self.assertTrue((out[:50] == 1).all())
        self.assertTrue((out[50:] == 2).all())

    def test_stall_intervals_are_skipped_and_shortfall_is_silence(self):
        # 墙钟1~2秒是断流间隔：时间轴1秒对应墙钟2~3秒，超出部分补静音。
        blocks = [block(0.0, 1, 100), block(1.0, 9, 100), block(2.0, 3, 100)]
        out = extract_pcm(blocks, [(0.0, 1.0), (2.0, 3.0)], 0.5, 2.5, rate=100)
        self.assertEqual(len(out), 200)
        self.assertTrue((out[:50] == 1).all())
        self.assertTrue((out[50:150] == 3).all())
        self.assertTrue((out[150:] == 0).all())

    def test_missing_recording_blocks_become_silence_not_shorter_timeline(self):
        blocks = [block(0.0, 1, 50), block(0.7, 1, 30)]
        out = extract_pcm(blocks, [(0.0, 1.0)], 0.0, 1.0, rate=100)
        self.assertEqual(len(out), 100)
        self.assertTrue((out[:50] == 1).all())
        self.assertTrue((out[50:70] == 0).all())
        self.assertTrue((out[70:] == 1).all())

    def test_window_validation(self):
        for start, end in ((1.0, 0.5), (float("nan"), 1.0), (0.0, float("inf"))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                extract_pcm([], [(0.0, 1.0)], start, end, rate=100)


class WavSidecarTests(unittest.TestCase):
    def test_roundtrip_keeps_samples(self):
        pcm = (np.sin(np.linspace(0, 90, 4800)) * 6000).astype(np.int16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "part.wav"
            write_wav(path, pcm, RATE)
            self.assertTrue(np.array_equal(parse_wav_pcm(path, RATE), pcm))
            self.assertTrue(audio_part_seconds_ok(pcm, 0.1))
            self.assertFalse(audio_part_seconds_ok(pcm, 0.2))

    def test_foreign_wav_layouts_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stereo.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(RATE)
                handle.writeframes(b"\x00\x00" * 9600)
            with self.assertRaises(ValueError):
                parse_wav_pcm(path, RATE)


class MuxRoundtripTests(unittest.TestCase):
    def test_audio_pts_gaps_and_video_origin_are_preserved(self):
        import av
        from fractions import Fraction
        from highlight360.io_video import mono_audio_frame, read_audio_pcm
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timed.mkv"
            with av.open(str(path), "w") as c:
                video = c.add_stream("libx264", rate=10)
                video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
                audio = c.add_stream("pcm_s16le", rate=RATE)
                audio.codec_context.layout = "mono"
                for index in range(30):
                    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="bgr24")
                    frame.pts, frame.time_base = index + 20, Fraction(1, 10)
                    for packet in video.encode(frame):
                        c.mux(packet)
                for packet in video.encode(None):
                    c.mux(packet)
                for start, value in ((2.5, 1000), (3.5, 2000)):
                    frame = mono_audio_frame(np.full(RATE // 2, value, np.int16), RATE, round(start * RATE))
                    for packet in audio.encode(frame):
                        c.mux(packet)
                for packet in audio.encode(None):
                    c.mux(packet)
            pcm = read_audio_pcm(str(path), 0, 3)
            expected = np.zeros(RATE * 3, np.int16)
            expected[RATE // 2:RATE] = 1000
            expected[RATE * 3 // 2:RATE * 2] = 2000
            np.testing.assert_array_equal(pcm, expected)

    def test_mux_audio_then_read_window_matches(self):
        from highlight360.io_video import VideoReader, VideoWriter, mux_audio, read_audio_pcm
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip = root / "clip.mp4"
            with VideoWriter(str(clip), 10, (64, 48)) as writer:
                for index in range(20):
                    writer.write(np.full((48, 64, 3), index * 3, np.uint8))
            self.assertIsNone(read_audio_pcm(str(clip), 0, 1))
            rate = 8000
            seconds = np.arange(rate * 2) / rate
            pcm = (np.sin(2 * np.pi * 440 * seconds) * 9000).astype(np.int16)
            out = root / "out.mp4"
            mux_audio(str(clip), pcm, rate, str(out))
            with VideoReader(str(out), "flat") as reader:
                self.assertTrue(reader.meta.has_audio)
            back = read_audio_pcm(str(out), 0.25, 1.25, rate)
            self.assertEqual(len(back), rate)
            reference = pcm[int(0.25 * rate):int(1.25 * rate)]
            self.assertGreater(np.corrcoef(back.astype(float), reference.astype(float))[0, 1], .97)


if __name__ == "__main__":
    unittest.main()
