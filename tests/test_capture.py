import contextlib
import io
import os
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

import capture


class CameraCaptureTests(unittest.TestCase):
    def setUp(self):
        warmup = patch.object(capture, "_CAMERA_STABLE_SEC", 0)
        warmup.start()
        self.addCleanup(warmup.stop)

    def test_unsupported_platform_does_not_probe_device(self):
        with patch.object(capture.sys, "platform", "linux"), patch.object(capture.cv2, "VideoCapture") as open_camera:
            with self.assertRaisesRegex(RuntimeError, "AVFoundation"):
                list(capture.camera_frames(1, 4, 2, 2, 2, threading.Event()))
            open_camera.assert_not_called()

    def test_macos_camera_uses_ffmpeg_avfoundation(self):
        process = Mock()
        process.stdout = io.BytesIO()
        process.poll.return_value = None
        with patch.object(capture.sys, "platform", "darwin"), \
                patch.object(capture.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                patch.object(capture.subprocess, "Popen", return_value=process) as popen:
            _, close = capture._open_camera_reader(1, 2880, 1440)
            close()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-f") + 1], "avfoundation")
        self.assertEqual(command[command.index("-video_size") + 1], "2880x1440")
        self.assertEqual(command[command.index("-i") + 1], "1:")
        self.assertEqual(command[command.index("-fps_mode") + 1], "passthrough")
        process.terminate.assert_called_once()
        process.stdout.close()

    def test_ffmpeg_reader_yields_complete_bgr_frames_and_releases(self):
        expected = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        process = Mock()
        process.stdout = io.BytesIO(expected.tobytes())
        process.poll.return_value = None
        with patch.object(capture.sys, "platform", "darwin"), \
                patch.object(capture.shutil, "which", return_value="ffmpeg"), \
                patch.object(capture.subprocess, "Popen", return_value=process):
            read, close = capture._open_camera_reader(1, 4, 2)
            ok, frame = read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(frame, expected)
            self.assertEqual(read(), (False, None))
            close()
        process.terminate.assert_called_once()

    def test_ffmpeg_pipe_stall_reports_error(self):
        reader_fd, writer_fd = os.pipe()
        try:
            with os.fdopen(reader_fd, "rb", buffering=0) as stream:
                with self.assertRaisesRegex(RuntimeError, "没有送出完整新帧"):
                    capture._read_exact(stream, 1, timeout_sec=0.1)
        finally:
            os.close(writer_fd)

    def test_camera_resolution_is_not_silently_guessed(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.return_value = (True, np.zeros((4, 4, 3), np.uint8))
        with patch.object(capture.sys, "platform", "win32"), patch.object(capture.cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(RuntimeError, "不是请求"):
                list(capture.camera_frames(1, 4, 2, 2, 2, threading.Event()))
        cap.release.assert_called_once()

    def test_camera_failure_does_not_become_normal_eof(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.side_effect = [(True, np.zeros((2, 4, 3), np.uint8)), (False, None)]
        with patch.object(capture.sys, "platform", "win32"), patch.object(capture.cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(RuntimeError, "断流"):
                list(capture.camera_frames(1, 4, 2, 2, 2, threading.Event()))
        cap.release.assert_called_once()

    def test_cfr_sampling_uses_receipt_times_instead_of_accelerating(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.side_effect = [(True, np.full((2, 4, 3), i + 20, np.uint8)) for i in range(6)]
        timeline = []
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap), \
                patch.object(capture.time, "monotonic",
                             side_effect=[100, 100.2, 100.7, 101.2, 101.7, 102.2, 102.7]):
            frames = list(capture.camera_frames(1, 4, 2, 2, 2, threading.Event(), timeline))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 21, 22, 23])
        self.assertEqual(timeline, [(100, 102.2)])
        cap.release.assert_called_once()

    def test_pure_black_glitch_frames_never_reach_the_timeline(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        stop = threading.Event()
        black = np.zeros((2, 4, 3), np.uint8)
        reads = [(True, np.full((2, 4, 3), 20, np.uint8)), (True, black),
                 (True, np.full((2, 4, 3), 21, np.uint8)),
                 (True, np.full((2, 4, 3), 22, np.uint8))]

        def read():
            if not reads:
                stop.set()
                return True, black  # 结束帧必须是会被丢弃的黑帧，不能混进时间轴
            return reads.pop(0)

        cap.read.side_effect = read
        timeline = []
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap), \
                patch.object(capture.time, "monotonic",
                             side_effect=[100, 100.05, 100.1, 100.6, 100.7, 100.8]):
            frames = list(capture.camera_frames(1, 4, 2, 2, 10, stop, timeline))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 21])
        self.assertEqual(timeline, [(100, 100.6)])

    def test_stream_stall_shifts_the_timeline_instead_of_freezing_old_frames(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        stop = threading.Event()
        black = np.zeros((2, 4, 3), np.uint8)
        reads = [(True, np.full((2, 4, 3), value, np.uint8)) for value in (20, 21, 22, 23, 24)]

        def read():
            if not reads:
                stop.set()
                return True, black
            return reads.pop(0)

        cap.read.side_effect = read
        # 1.5秒断流：旧实现会补出3帧重复旧画面；新实现时间轴后移，只按真实画面出帧。
        timeline = []
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap), \
                patch.object(capture.time, "monotonic",
                             side_effect=[100, 100.1, 101.6, 101.7, 102.2, 102.3, 102.4]):
            frames = list(capture.camera_frames(1, 4, 2, 2, 10, stop, timeline))
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [20, 23])
        self.assertEqual(timeline, [(100, 100.1), (101.6, 102.2)])

    def test_black_only_stream_is_a_stall_not_silent_success(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.return_value = (True, np.zeros((2, 4, 3), np.uint8))
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap), \
                patch.object(capture, "_CAMERA_START_TIMEOUT_SEC", 5), \
                patch.object(capture.time, "monotonic", side_effect=[100, 101, 102, 103, 104, 105, 106]):
            with self.assertRaisesRegex(RuntimeError, "未开始录制"):
                list(capture.camera_frames(1, 4, 2, 2, 10, threading.Event()))

    def test_startup_waits_for_continuous_frames_before_recording(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        stamps = [0.0, 0.05, 1.05] + [1.1 + i / 10 for i in range(16)]
        values = [20, 21, 0] + list(range(30, 46))
        cap.read.side_effect = [(True, np.full((2, 4, 3), v, np.uint8)) for v in values]
        ready, timeline = [], []
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture, "_CAMERA_STABLE_SEC", 1), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap), \
                patch.object(capture.time, "monotonic", side_effect=stamps):
            frames = list(capture._read_camera(1, 4, 2, 10, 0.2, lambda: False, timeline,
                                              lambda: ready.append(timeline[0][0])))
        self.assertEqual(len(ready), 1)
        self.assertGreaterEqual(ready[0], 2.1)
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(int(f[0, 0, 0]) >= 40 for f in frames))
        cap.release.assert_called_once()

    def test_matching_camera_properties_are_not_renegotiated(self):
        cap = Mock()
        cap.isOpened.return_value = True
        properties = {capture.cv2.CAP_PROP_FOURCC: capture.cv2.VideoWriter_fourcc(*"MJPG"),
                      capture.cv2.CAP_PROP_FRAME_WIDTH: 4, capture.cv2.CAP_PROP_FRAME_HEIGHT: 2,
                      capture.cv2.CAP_PROP_FPS: 30.00003}
        cap.get.side_effect = properties.get
        stop = threading.Event()

        def read():
            stop.set()
            return True, np.full((2, 4, 3), 20, np.uint8)

        cap.read.side_effect = read
        with patch.object(capture.sys, "platform", "win32"), \
                patch.object(capture.cv2, "VideoCapture", return_value=cap):
            self.assertEqual(list(capture._read_camera(1, 4, 2, 2, 2, stop.is_set, [], None)), [])
        cap.set.assert_called_once_with(capture.cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.release.assert_called_once()

    def test_camera_thread_reads_ahead_while_encoder_is_busy(self):
        produced, closed = threading.Event(), threading.Event()

        def source(*args):
            try:
                yield np.full((2, 4, 3), 20, np.uint8)
                yield np.full((2, 4, 3), 21, np.uint8)
                produced.set()
            finally:
                closed.set()

        with patch.object(capture, "_read_camera", source):
            stream = capture.camera_frames(1, 4, 2, 2, 2, threading.Event())
            self.assertEqual(int(next(stream)[0, 0, 0]), 20)
            self.assertTrue(produced.wait(2))
            self.assertEqual([int(f[0, 0, 0]) for f in stream], [21])
        self.assertTrue(closed.is_set())

    def test_camera_overflow_fails_instead_of_silently_dropping_frames(self):
        release, closed = threading.Event(), threading.Event()

        def source(*args):
            try:
                yield np.full((2, 4, 3), 20, np.uint8)
                release.wait(2)
                for _ in range(capture._CAMERA_QUEUE_FRAMES + 1):
                    yield np.full((2, 4, 3), 21, np.uint8)
            finally:
                closed.set()

        with patch.object(capture, "_read_camera", source):
            stream = capture.camera_frames(1, 4, 2, 2, 2, threading.Event())
            try:
                next(stream)
                release.set()
                self.assertTrue(closed.wait(2))
                with self.assertRaisesRegex(RuntimeError, "帧缓冲已满"):
                    next(stream)
            finally:
                release.set()
                stream.close()

    def test_cancel_before_camera_start_does_not_open_device(self):
        stop = threading.Event()
        stop.set()
        with patch.object(capture.sys, "platform", "win32"), patch.object(capture.cv2, "VideoCapture") as open_camera:
            self.assertEqual(list(capture.camera_frames(1, 4, 2, 2, 2, stop)), [])
        open_camera.assert_not_called()

    def test_x5_index_uses_video_list_after_usb_reordering(self):
        listing = ("AVFoundation video devices:\n[0] FaceTime高清相机\n"
                   "[2] Insta360 X5\nAVFoundation audio devices:\n[1] Insta360 X5\n")
        with patch.object(capture.sys, "platform", "darwin"), \
                patch.object(capture.shutil, "which", return_value="ffmpeg"), \
                patch.object(capture.subprocess, "run", return_value=Mock(stderr=listing)):
            self.assertEqual(capture.find_camera_index(), 2)

    def test_console_eof_is_not_explicit_stop(self):
        stop = threading.Event()
        with patch("builtins.input", side_effect=EOFError):
            capture._stop_on_enter(stop)
        self.assertFalse(stop.is_set())
        with patch("builtins.input", return_value=""):
            capture._stop_on_enter(stop)
        self.assertTrue(stop.is_set())

    def test_capture_requires_one_source_projection_and_out(self):
        parser = capture.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in ([], ["--camera-index", "1"],
                         ["--camera-index", "1", "--file", "a.mp4",
                          "--projection", "flat", "--out", "b.mp4"],
                         ["--camera-index", "1", "--projection", "flat"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    parser.parse_args(argv)
        args = parser.parse_args(["--file", "a.mp4", "--projection", "equirectangular",
                                  "--out", "b.mp4"])
        self.assertEqual(args.file, "a.mp4")
        self.assertEqual(args.projection, "equirectangular")
        self.assertEqual(args.out, "b.mp4")
        self.assertIsNone(args.camera_index)


if __name__ == "__main__":
    unittest.main()
