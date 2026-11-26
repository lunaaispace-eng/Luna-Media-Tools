"""The encoder arguments: quality must govern H.265 and AV1 on NVENC; SDR is BT.709.

hevc_nvenc and av1_nvenc without -maxrate stop at a level-derived ceiling (measured 2026-11-26 at 3840x2160:
HEVC cq 10 and cq 20 both ~24 Mbit/s; AV1 cq 4 and cq 10 both ~130 Mbit/s). h264_nvenc is not capped.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
video_nodes = sys.modules[PACK.__name__ + ".video_nodes"]
args = video_nodes._encoder_arguments


class NvencArgumentTests(unittest.TestCase):
    def test_h264_unchanged(self):
        self.assertEqual(args("H.264", "h264_nvenc", 8, 14, 18),
                         ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "14", "-pix_fmt", "yuv420p"])
        self.assertEqual(args("H.264", "h264_nvenc", 10, 14, 18),
                         ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "14", "-pix_fmt", "yuv420p10le"])

    def test_hevc_ceiling_lifted(self):
        self.assertEqual(args("H.265 (HEVC)", "hevc_nvenc", 10, 10, 18),
                         ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "10",
                          "-tier", "high", "-maxrate", "400M", "-bufsize", "800M", "-pix_fmt", "yuv420p10le"])

    def test_av1_ceiling_lifted(self):
        self.assertEqual(args("AV1", "av1_nvenc", 10, 20, 30),
                         ["-c:v", "av1_nvenc", "-preset", "p5", "-cq", "20",
                          "-maxrate", "400M", "-bufsize", "800M", "-pix_fmt", "yuv420p10le"])

    def test_cpu_encoders_unchanged(self):
        self.assertEqual(args("H.265 (HEVC)", "libx265", 10, 10, 18),
                         ["-c:v", "libx265", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p10le"])


class SdrColorTests(unittest.TestCase):
    """SDR is converted with the BT.709 matrix and tagged BT.709 (2026-11-26). Untagged, FFmpeg used BT.601 and
    DaVinci Resolve read the file as Rec.709, shifted warm."""

    def test_sdr_8bit(self):
        self.assertEqual(video_nodes._color_arguments("SDR", 8), [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                   "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"])

    def test_sdr_10bit(self):
        self.assertIn("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le,",
                      video_nodes._color_arguments("SDR", 10)[1])

    def test_hdr_unchanged(self):
        self.assertEqual(video_nodes._color_arguments("HDR PQ (BT.2020)", 10), [
            "-vf", "scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le,"
                   "setparams=range=tv:colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084",
            "-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-color_range", "tv"])


if __name__ == "__main__":
    unittest.main()
