"""The node's visible run report counts grid losses before every-Nth sampling."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_registration import load_pack


class GridReportTests(unittest.TestCase):
    def test_grid_and_free_reports_for_a_real_clip(self):
        import importlib
        loader = importlib.import_module(load_pack().__name__ + ".load_video")
        with tempfile.TemporaryDirectory() as temp:
            clip = str(Path(temp) / "thirty.mkv")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:r=24",
                            "-frames:v", "30", "-c:v", "ffv1", clip], check=True)
            node = loader.LunaLoadVideo()
            args = dict(source="path / URL", path=clip, every_nth=2, audio="drop")
            for grid, count, text in [("H3 (24 fps, 17k+5)",11,"dropped 8 frames"), ("free",15,None)]:
                with self.subTest(grid=grid):
                    result = node.load(**args, frame_grid=grid)
                    self.assertEqual(result["result"][2], [count])
                    message = result["ui"]["luna_load_video"][0]
                    if text:
                        self.assertIn(text, message)
                    else:
                        self.assertNotIn("dropped", message)
            actual = loader.probe_media(clip)
            with patch.object(loader, "probe_media", return_value={**actual, "duration": 3.0}):
                result = node.load(**args, frame_grid="H3 (24 fps, 17k+5)")
                self.assertIn("dropped 8 frames", result["ui"]["luna_load_video"][0])
                self.assertIn("ended early", result["ui"]["luna_load_video"][0])


if __name__ == "__main__":
    unittest.main()
