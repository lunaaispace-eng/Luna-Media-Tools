"""media_io fixes from the 2026-10-16 review: the online frame picker and the
annotation tag. (The Asset Loader / H3 parts of that file stay with the H3 pack.)

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from media_io import FramePicker, pick_frames, strip_annotation  # noqa: E402


def times(fps, seconds):
    return [i / fps for i in range(int(round(fps * seconds)))]


class OnlinePickerTests(unittest.TestCase):
    """The decoder keeps only the frames it will use; it must pick exactly what
    pick_frames picks, as each frame arrives."""

    def run_picker(self, src, out_fps, start, count, nth=1):
        p = FramePicker(out_fps, start, count, nth)
        kept = []
        for i, t in enumerate(src):
            kept += p.feed(t, i)
            if p.done:
                break
        kept += p.finish()
        return kept

    def check(self, src, out_fps, start, count, nth=1):
        self.assertEqual(self.run_picker(src, out_fps, start, count, nth),
                         pick_frames(src, out_fps, start, count)[::nth])

    def test_same_rate(self):
        self.check(times(24, 2), 24, 0.0, 24)

    def test_30_to_24(self):
        self.check(times(30, 3), 24, 0.4, 39)

    def test_12_to_24_holds(self):
        self.check(times(12, 2), 24, 0.0, 30)

    def test_runs_out(self):
        self.check(times(24, 1), 24, 0.5, 50)

    def test_every_nth(self):
        self.check(times(30, 3), 24, 0.0, 39, nth=2)

    def test_tail_ends_on_last_frame(self):
        self.check(times(30, 12.4), 24, 12.4 - 1 / 30 - 21 / 24, 22)


class AnnotationTests(unittest.TestCase):
    def test_only_a_trailing_folder_tag_is_removed(self):
        self.assertEqual(strip_annotation("song.mp3 [input]"), ("song.mp3", "input"))
        self.assertEqual(strip_annotation("clip [v2].mp4"), ("clip [v2].mp4", None))
        self.assertEqual(strip_annotation("a/b.wav"), ("a/b.wav", None))


if __name__ == "__main__":
    unittest.main()
