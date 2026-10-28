"""Luna-Degrid's maths core is pure torch: a clean image passes through, a gridded one is cleaned.

    E:/ComfyUI-Easy-Install/python_embeded/python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degrid_core import degrid  # noqa: E402


class DegridCoreTests(unittest.TestCase):
    def test_flat_image_is_left_alone(self):
        img = torch.full((1, 64, 64, 3), 0.5)
        out, vis, stats = degrid(img, mode="auto", limit=0.02, grid_gain=10.0, grid_view="full frame",
                                 skip_when_clean=True)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(torch.allclose(out, img))
        self.assertEqual(len(stats), 1)

    def test_checker_grid_is_reduced(self):
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        grid = (((yy + xx) % 2).float() * 2 - 1) * 0.01
        img = (0.5 + grid).view(1, 64, 64, 1).repeat(1, 1, 1, 3)
        out, _, _ = degrid(img, mode="auto", limit=0.02, grid_gain=10.0, grid_view="full frame",
                           skip_when_clean=True)
        self.assertLess(float((out - 0.5).abs().mean()), float((img - 0.5).abs().mean()))


if __name__ == "__main__":
    unittest.main()
