from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np

from highlight360.io_video import VideoReader


class VideoFormatTests(unittest.TestCase):
    def test_hevc_mp4_and_mov_decode_with_real_timestamps(self):
        if "libx265" not in av.codecs_available:
            self.skipTest("PyAV build has no libx265 fixture encoder")
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".mp4", ".mov"):
                with self.subTest(suffix=suffix):
                    path = Path(tmp) / ("已拼接SDR全景" + suffix)
                    with av.open(str(path), "w") as container:
                        stream = container.add_stream("libx265", rate=8)
                        stream.width, stream.height = 128, 64
                        stream.pix_fmt = "yuv420p"
                        stream.options = {"x265-params": "log-level=error:pools=1:frame-threads=1"}
                        stream.codec_context.color_trc = 1
                        stream.time_base = Fraction(1, 8)
                        for i in range(16):
                            image = np.full((64, 128, 3), i * 10, dtype=np.uint8)
                            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                            frame.pts = i
                            frame.time_base = Fraction(1, 8)
                            for packet in stream.encode(frame):
                                container.mux(packet)
                        for packet in stream.encode(None):
                            container.mux(packet)
                    with VideoReader(str(path), "equirectangular") as reader:
                        self.assertEqual(reader.meta.codec, "hevc")
                        self.assertAlmostEqual(reader.meta.duration_sec, 2.0, places=2)
                        frames = list(reader.iter_frames(sample_fps=4))
                        self.assertEqual(len(frames), 8)
                        self.assertEqual(frames[0][0], 0)
                        self.assertEqual(frames[-1][0], 1.75)


if __name__ == "__main__":
    unittest.main()
