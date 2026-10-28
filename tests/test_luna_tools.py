"""The four ported nodes: Luna-Watermark, Luna-StatusSwitch, Luna-RTXUpscaler, Luna-RTXFrameGen.

Pure logic, no GPU and no NVIDIA RTX VFX needed. Needs ComfyUI on sys.path (folder_paths, comfy.utils);
test_registration.load_pack() puts it there.
"""

import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import PACK, load_pack  # noqa: E402

load_pack()
import Luna_Media_Tools_under_test.luna_tools.luna_watermark as wm  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_batch_output as bo  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_logging as lg  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_node_status_switch as sw  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_upscaler as up  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_frame_gen as fg  # noqa: E402


class WatermarkMathTests(unittest.TestCase):
    def test_position_corners(self):
        self.assertEqual(wm._position_xy("top-left", 100, 80, 10, 10, 5, 7), (5, 7))
        self.assertEqual(wm._position_xy("top-right", 100, 80, 10, 10, 5, 7), (85, 7))
        self.assertEqual(wm._position_xy("bottom-left", 100, 80, 10, 10, 5, 7), (5, 63))
        self.assertEqual(wm._position_xy("bottom-right", 100, 80, 10, 10, 5, 7), (85, 63))
        self.assertEqual(wm._position_xy("center", 100, 80, 10, 10, 5, 7), (45, 35))

    def test_corner_sequence_starts_where_asked_and_is_stable(self):
        seq = wm._corner_sequence("top-left", 5, 123)
        self.assertEqual(len(seq), 6)
        self.assertEqual(seq[0], "top-left")
        self.assertEqual(seq, wm._corner_sequence("top-left", 5, 123))
        for a, b in zip(seq, seq[1:]):
            self.assertNotEqual(a, b)

    def test_rotation_zero_is_identity_and_90_swaps_size(self):
        x = torch.rand(1, 4, 6, 10)
        self.assertIs(wm._rotate_bicubic_expand(x, 0), x)
        y = wm._rotate_bicubic_expand(x, 90)
        self.assertEqual(tuple(y.shape[-2:]), (10, 6))

    def test_premultiplied_rgba_is_clamped(self):
        x = torch.tensor([[[[2.0]], [[0.9]], [[-1.0]], [[0.5]]]])
        out = wm._sanitize_premultiplied_rgba(x)
        self.assertLessEqual(float(out[0, 0, 0, 0]), 0.5)      # rgb never above alpha
        self.assertEqual(float(out[0, 2, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 3, 0, 0]), 0.5)

    def test_as_bool(self):
        self.assertTrue(wm._as_bool("yes"))
        self.assertFalse(wm._as_bool("off"))
        self.assertTrue(wm._as_bool(True))


class WatermarkRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mark = os.path.join(self.tmp.name, "mark.png")
        Image.new("RGBA", (16, 16), (255, 255, 255, 255)).save(self.mark)
        self._orig = wm.folder_paths.get_annotated_filepath
        wm.folder_paths.get_annotated_filepath = lambda name: self.mark

    def tearDown(self):
        wm.folder_paths.get_annotated_filepath = self._orig
        self.tmp.cleanup()

    def run_node(self, images, **kw):
        args = dict(watermark_path="mark.png", position="top-left", scale=0.25, resampling="nearest",
                    transparency=1.0, rotation=0, padding_x=0, padding_y=0, optical_padding=False,
                    optical_strength=0.4)
        args.update(kw)
        buf = io.StringIO()
        with redirect_stdout(buf):
            (out,) = wm.LunaWatermark().apply_watermark(images, **args)
        return out, buf.getvalue()

    def test_white_mark_lands_in_the_corner(self):
        images = torch.zeros(2, 32, 32, 3)
        out, log = self.run_node(images)
        self.assertEqual(tuple(out.shape), (2, 32, 32, 3))
        self.assertTrue(torch.all(out[:, 0:8, 0:8, :] == 1.0))
        self.assertTrue(torch.all(out[:, 8:, :, :] == 0.0))
        self.assertTrue(torch.all(out[:, :, 8:, :] == 0.0))
        self.assertIn("[Luna-Watermark]", log)

    def test_input_batch_is_not_modified(self):
        images = torch.zeros(1, 32, 32, 3)
        self.run_node(images)
        self.assertEqual(float(images.sum()), 0.0)

    def test_half_transparency_blends(self):
        out, _ = self.run_node(torch.zeros(1, 32, 32, 3), transparency=0.5)
        self.assertAlmostEqual(float(out[0, 2, 2, 0]), 0.5, places=3)

    def test_bottom_right_with_padding(self):
        out, _ = self.run_node(torch.zeros(1, 32, 32, 3), position="bottom-right", padding_x=4, padding_y=4)
        self.assertTrue(torch.all(out[:, 20:28, 20:28, :] == 1.0))
        self.assertEqual(float(out[0, 31, 31, 0]), 0.0)

    def test_fade_starts_invisible(self):
        out, _ = self.run_node(torch.zeros(10, 32, 32, 3), fade=True, fade_margin=0.2)
        self.assertEqual(float(out[0, 2, 2, 0]), 0.0)       # first frame: fade multiplier 0
        self.assertEqual(float(out[5, 2, 2, 0]), 0.0)       # middle frames: fade multiplier 0 (as upstream)

    def test_alpha_batch_channel_is_dropped(self):
        out, _ = self.run_node(torch.zeros(1, 32, 32, 4))
        self.assertEqual(out.shape[-1], 3)

    def test_node_shape(self):
        cls = wm.LunaWatermark
        self.assertEqual(cls.CATEGORY, "Luna-Media-Tools")
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))
        spec = cls.INPUT_TYPES()["required"]
        for name, (kind, opts) in [(k, v if len(v) > 1 else (v[0], {})) for k, v in spec.items()]:
            self.assertNotIn("description", opts, name)
        self.assertIn("tooltip", spec["scale"][1])
        self.assertEqual(wm.NODE_CLASS_MAPPINGS, {"Luna-Watermark": wm.LunaWatermark})


class BatchOutputTests(unittest.TestCase):
    def test_small_output_lives_in_ram(self):
        with tempfile.TemporaryDirectory() as d:
            out, path = bo.allocate_cpu_output((2, 4, 4, 3), torch.float16, d)
        self.assertEqual(tuple(out.shape), (2, 4, 4, 3))
        self.assertEqual(out.dtype, torch.float16)
        self.assertIsNone(path)

    def test_forced_mmap_uses_a_luna_file_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            out, path = bo.allocate_cpu_output((2, 4, 4, 3), torch.float32, d, force_mmap=True)
            self.assertTrue(os.path.basename(path).startswith("luna_output_"))
            self.assertTrue(os.path.exists(path))
            out[:] = 1.0
            self.assertEqual(float(out.sum()), 2 * 4 * 4 * 3)
            del out
            import gc
            gc.collect()
            self.assertFalse(os.path.exists(path))

    def test_too_big_output_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                bo.allocate_cpu_output((1, 100000, 100000, 3), torch.float32, d, has_free_disk_space=lambda *a: True)

    def test_tensor_nbytes(self):
        self.assertEqual(bo.tensor_nbytes((2, 3), torch.float32), 24)


class LoggingTests(unittest.TestCase):
    def test_prefix_is_the_node_name(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            lg.log_luna("Luna-Watermark", "hello")
        self.assertIn("[Luna-Watermark]", buf.getvalue())
        self.assertIn("hello", buf.getvalue())


class StatusSwitchTests(unittest.TestCase):
    def test_passes_enabled_through(self):
        node = sw.LunaStatusSwitch()
        self.assertEqual(node.execute(True, "true → active", "mute"), (True,))
        self.assertEqual(node.execute(0, "false → active", "bypass", unique_id="5", target_01=object()), (False,))

    def test_shape(self):
        cls = sw.LunaStatusSwitch
        req = cls.INPUT_TYPES()["required"]
        self.assertEqual(list(req), ["enabled", "trigger_on", "action"])
        self.assertEqual(req["action"][0], ["mute", "bypass"])
        self.assertEqual(cls.RETURN_TYPES, ("BOOLEAN",))
        self.assertEqual(cls.RETURN_NAMES, ("enabled_out",))
        self.assertEqual(cls.CATEGORY, "Luna-Media-Tools")
        self.assertTrue(cls.VALIDATE_INPUTS(target_01="anything"))

    def test_any_type_matches_everything(self):
        self.assertTrue(sw.ANY == "IMAGE")
        self.assertFalse(sw.ANY != "LATENT")

    def test_ids(self):
        self.assertEqual(sw.NODE_CLASS_MAPPINGS, {"Luna-StatusSwitch": sw.LunaStatusSwitch})


# the upstream pack names, assembled so that this file does not itself spell them out
FOREIGN = r"da" r"siwa|\bde" r"no(?![a-z])|install_rtx_vfx|\.bat"


class RTXNodeShapeTests(unittest.TestCase):
    """Both RTX nodes: ids, category, tooltips, help text, lazy SDK. Behaviour tests are in test_rtx_*.py."""

    def test_node_shape(self):
        for cls, node_id in ((up.LunaRTXUpscaler, "Luna-RTXUpscaler"), (fg.LunaRTXFrameGen, "Luna-RTXFrameGen")):
            self.assertEqual(cls.CATEGORY, "Luna-Media-Tools")
            self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))
            self.assertTrue(cls.OUTPUT_TOOLTIPS)
            self.assertIn(node_id, cls.DESCRIPTION)
            self.assertGreater(len(cls.DESCRIPTION), 200)
            schema = cls.INPUT_TYPES()
            for section in schema.values():
                for name, spec in section.items():
                    self.assertNotIn("description", spec[1], f"{node_id}.{name}")
                    self.assertTrue(len(spec[1].get("tooltip", "")) >= 20, f"{node_id}.{name}")

    def test_ids(self):
        self.assertEqual(up.NODE_CLASS_MAPPINGS, {"Luna-RTXUpscaler": up.LunaRTXUpscaler})
        self.assertEqual(fg.NODE_CLASS_MAPPINGS, {"Luna-RTXFrameGen": fg.LunaRTXFrameGen})

    def test_upscaler_keeps_the_three_stages_and_memory_controls(self):
        schema = up.LunaRTXUpscaler.INPUT_TYPES()
        for name in ("denoise", "denoise_quality", "deblur", "deblur_quality", "upscale", "upscale_quality",
                     "resize_type", "scale", "megapixels", "width", "height", "divisible_by", "ratio_preset",
                     "resize_method", "device_id"):
            self.assertIn(name, schema["required"])
        for name in ("empty_cache", "use_mmap", "auto_unload_models", "chunking", "lossless_fp16", "chunk_frames",
                     "denoise_strength", "deblur_strength", "upscale_strength", "hdr_enabled", "image_encoding"):
            self.assertIn(name, schema["optional"])
        self.assertEqual(schema["required"]["upscale"][0][:3], ["Off", "VSR", "High Bitrate"])

    def test_all_stages_off_is_a_passthrough(self):
        images = torch.rand(2, 8, 8, 3)
        (out,) = up.LunaRTXUpscaler().execute(
            images, False, "Ultra", False, "Ultra", "Off", "Ultra", "Scale", 2.0, 2.0, 1920, 1080, "8",
            "16:9", "Center Crop (Fill)", 0, lossless_fp16=False)
        self.assertTrue(torch.equal(out, images))

    def test_missing_nvvfx_error_describes_this_pack(self):
        saved = sys.modules.get("nvvfx")
        sys.modules["nvvfx"] = None          # makes "import nvvfx" raise ImportError
        try:
            with self.assertRaises(RuntimeError) as ctx:
                up._import_vfx()
            with self.assertRaises(RuntimeError) as ctx2:
                fg._luna_fg_sdk()
        finally:
            if saved is None:
                sys.modules.pop("nvvfx", None)
            else:
                sys.modules["nvvfx"] = saved
        for msg in (str(ctx.exception), str(ctx2.exception)):
            self.assertIn("Luna-Media-Tools", msg)
            self.assertIn("pip install nvidia-vfx", msg)
            self.assertIsNone(re.search(FOREIGN, msg, re.I), msg)

    def test_importing_the_nodes_does_not_import_nvvfx(self):
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_registration as t; "
                "t.load_pack(); print('nvvfx' in sys.modules)") % (str(Path(__file__).parent), t_comfy())
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "False", out.stderr[-500:])


def t_comfy():
    from test_registration import COMFY
    return COMFY


class NamingTests(unittest.TestCase):
    """The packs a file was ported from are named only in its licence/credit header."""

    PATTERN = re.compile(r"dasiwa|\bdeno(?![a-z])", re.I)

    def test_no_foreign_names_outside_the_credit_header(self):
        files = list((Path(PACK) / "luna_tools").glob("*.py")) + list((Path(PACK) / "web").glob("luna_*.js"))
        files += list((Path(PACK) / "tests").glob("test_rtx_*.py")) + [Path(PACK) / "tests" / "test_rtx_ui.mjs"]
        files += [Path(PACK) / "web" / "luna_pack_theme.js"]
        bad = []
        for f in files:
            if f.name.startswith("luna_help") or f.name in ("luna_theme.mjs", "luna_collapse.mjs"):
                continue                 # shared kit files, not edited here
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                if self.PATTERN.search(line) and i >= 12:
                    bad.append(f"{f.name}:{i + 1}: {line.strip()[:80]}")
        self.assertEqual(bad, [])

    def test_ported_files_keep_their_credit_header(self):
        for name in ("luna_watermark.py", "luna_node_status_switch.py", "luna_batch_output.py", "luna_logging.py"):
            head = (Path(PACK) / "luna_tools" / name).read_text(encoding="utf-8")[:600]
            self.assertIn("darksidewalker/ComfyUI-DaSiWa-Nodes", head, name)
            self.assertIn("ab364c5", head, name)
            self.assertIn("GPL-3.0", head, name)
        for name in ("luna_rtx_upscaler.py", "luna_rtx_frame_gen.py", "luna_rtx_ui.js"):
            folder = "web" if name.endswith(".js") else "luna_tools"
            head = (Path(PACK) / folder / name).read_text(encoding="utf-8")[:600]
            self.assertIn("darksidewalker/ComfyUI-DaSiWa-Nodes", head, name)
            self.assertIn("ab364c5", head, name)
            self.assertIn("GPL-3.0", head, name)


if __name__ == "__main__":
    unittest.main()
