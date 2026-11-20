"""media_io.py is identical in Luna-Media-Tools and Luna-MinimaxH3 (Luna-MinimaxH3 `tests/test_media_io.py` is the
other side of the check), with no node name of its own in a message.

The check reads the sibling pack from env LUNA_PACKS_DIR (default D:\\Claude) and skips when it is not there.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
"""

import filecmp
import os
import sys
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
import luna_units  # noqa: E402
import media_io  # noqa: E402

PACKS_DIR = os.environ.get("LUNA_PACKS_DIR", r"D:\Claude")
OTHER = "Luna-MinimaxH3"


class CopyTests(unittest.TestCase):
    def test_copy_is_identical_in_luna_minimaxh3(self):
        other = os.path.join(PACKS_DIR, OTHER, "media_io.py")
        if not os.path.isfile(other):
            self.skipTest(f"{OTHER} is not beside this pack")
        self.assertTrue(filecmp.cmp(os.path.join(PACK, "media_io.py"), other, shallow=False), f"{OTHER}/media_io.py differs")

    def test_labels_come_from_who(self):
        with open(os.path.join(PACK, "media_io.py"), encoding="utf-8") as fh:
            body = fh.read().split('"""', 2)[2]
        self.assertNotIn("[Luna-", body)
        self.assertEqual(media_io.WHO, "media_io")

    def test_on_the_shared_frame_rule(self):
        self.assertEqual(media_io.TRAINED_MAX_FRAMES, luna_units.H3_TRAINED_MAX)
        self.assertEqual((media_io.FRAME_STEP, media_io.FRAME_PLUS), luna_units.GRIDS["h3"])
        for n in range(1, 400):
            self.assertEqual(media_io.grid_frames(n, "h3"), luna_units.frames_up(n, "h3"))
            self.assertEqual(media_io.snap_frames_down(n), luna_units.frames_down(n, "h3"))


if __name__ == "__main__":
    unittest.main()
