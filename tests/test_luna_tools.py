"""The four ported nodes: Luna-Watermark, Luna-StatusSwitch, Luna-RTXUpscale, Luna-RTXVideoFinisher.

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
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_runtime as rt  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_upscale as up  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_video_finisher as fin  # noqa: E402
import Luna_Media_Tools_under_test.luna_tools.luna_resolution_common as rc  # noqa: E402


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


class RTXSizeTests(unittest.TestCase):
    def test_quality_names(self):
        self.assertEqual(up._quality_attr("VSR Ultra"), "ULTRA")
        self.assertEqual(up._quality_attr("High Bitrate Low"), "HIGHBITRATE_LOW")
        self.assertEqual(up._quality_attr("Denoise High"), "DENOISE_HIGH")
        self.assertEqual(up._quality_attr("Deblur Medium"), "DEBLUR_MEDIUM")
        for mode in up.QUALITY_LEVELS:
            self.assertTrue(up._quality_attr(mode))

    def test_same_size_modes(self):
        self.assertTrue(up._same_size_only("Denoise Low"))
        self.assertTrue(up._same_size_only("Deblur Ultra"))
        self.assertFalse(up._same_size_only("VSR High"))

    def test_target_size_scale(self):
        t = lambda **k: up._target_size(**{**dict(source_width=640, source_height=360, mode="VSR Medium",
                                                   resize_type="Scale", scale=2.0, megapixels=2.0, width=1920,
                                                   height=1080, divisible_by=1, ratio_preset="16:9"), **k})
        self.assertEqual(t(), (1280, 720))
        self.assertEqual(t(scale=1.5, divisible_by=8), (960, 544))
        self.assertEqual(t(resize_type="Manual", width=1000, height=500), (1000, 500))
        self.assertEqual(t(resize_type="Same Size"), (640, 360))
        self.assertEqual(t(mode="Denoise High"), (640, 360))       # same-size modes ignore the resize type
        w, h = t(resize_type="Keep Ratio", megapixels=2.0)
        self.assertAlmostEqual(w / h, 640 / 360, places=2)
        self.assertAlmostEqual(w * h / 1e6, 2.0, delta=0.05)
        w, h = t(resize_type="Preset Ratio", ratio_preset="1:1", megapixels=1.0, divisible_by=16)
        self.assertEqual(w, h)
        self.assertEqual(w % 16, 0)

    def test_safe_divisible_by(self):
        self.assertEqual(up._safe_divisible_by("16"), 16)
        self.assertEqual(up._safe_divisible_by(7), 1)
        self.assertEqual(up._safe_divisible_by("x"), 1)

    def test_output_width_aligned_to_8(self):
        self.assertEqual(up._vfx_output_width(1280), 1280)
        self.assertEqual(up._vfx_output_width(1281), 1288)

    def test_fit_frame_crop_and_pad(self):
        frame = torch.zeros(3, 90, 160)
        crop = up._fit_frame_to_target_aspect(frame, 100, 100, "Center Crop (Fill)")
        self.assertEqual(tuple(crop.shape), (3, 90, 90))
        pad = up._fit_frame_to_target_aspect(frame, 100, 100, "Fit (Letterbox/Pillarbox)")
        self.assertEqual(tuple(pad.shape), (3, 160, 160))
        same = up._fit_frame_to_target_aspect(frame, 320, 180, "Center Crop (Fill)")
        self.assertEqual(tuple(same.shape), (3, 90, 160))

    def test_validate_inputs(self):
        v = up.LunaRTXUpscale.VALIDATE_INPUTS
        self.assertIs(v(mode="VSR Low", resize_type="Scale", divisible_by="8", ratio_preset="16:9",
                        resize_method="Center Crop (Fill)"), True)
        self.assertIsNot(v(mode="nonsense"), True)
        self.assertIsNot(v(mode="VSR Low", resize_type="Preset Ratio", ratio_preset="7:3"), True)
        self.assertIs(v(mode="Denoise Low", resize_type="whatever"), True)

    def test_resolution_helpers(self):
        self.assertEqual(rc.simplify_ratio(1920, 1080), "16:9")
        self.assertEqual(rc.round_up(1000, 64), 1024)
        w, h = rc.compute_aligned_ratio_dims("16:9", 2.0, 16)
        self.assertEqual((w % 16, h % 16), (0, 0))

    def test_node_shape(self):
        for cls, node_id in ((up.LunaRTXUpscale, "Luna-RTXUpscale"), (fin.LunaRTXVideoFinisher, "Luna-RTXVideoFinisher")):
            self.assertEqual(cls.CATEGORY, "Luna-Media-Tools")
            self.assertEqual(cls.RETURN_TYPES, ("IMAGE",))
            self.assertTrue(cls.OUTPUT_TOOLTIPS)
            self.assertIn(node_id, cls.DESCRIPTION)
            for name, spec in cls.INPUT_TYPES()["required"].items():
                if name != "images":
                    self.assertIn("tooltip", spec[1], f"{node_id}.{name}")


class FinisherTests(unittest.TestCase):
    def test_both_passes_off_is_a_true_passthrough(self):
        images = torch.rand(2, 8, 8, 3)
        (out,) = fin.LunaRTXVideoFinisher().apply_finisher(
            images, "Off", "Ultra", "Off", "High", "Keep Ratio", 2.0, 4.0, 3840, 2160, "1", "16:9",
            "Center Crop (Fill)")
        self.assertIs(out, images)

    def test_validate_inputs(self):
        v = fin.LunaRTXVideoFinisher.VALIDATE_INPUTS
        self.assertIs(v(first_pass="Deblur", first_quality="Ultra", upscale_pass="High Bitrate",
                        upscale_quality="High", resize_type="Keep Ratio", divisible_by="1", ratio_preset="16:9",
                        resize_method="Center Crop (Fill)"), True)
        self.assertIsNot(v(first_pass="Sharpen"), True)
        self.assertIsNot(v(first_pass="Off", upscale_pass="VSR", upscale_quality="Max"), True)
        self.assertIs(v(first_pass="Off", upscale_pass="Off"), True)

    def test_choices(self):
        req = fin.LunaRTXVideoFinisher.INPUT_TYPES()["required"]
        self.assertEqual(req["first_pass"][0], ["Off", "Denoise", "Deblur"])
        self.assertEqual(req["upscale_pass"][0], ["Off", "VSR", "High Bitrate"])
        self.assertEqual(req["first_pass"][1]["default"], "Deblur")
        self.assertEqual(req["upscale_pass"][1]["default"], "High Bitrate")

    def test_off_pass_creates_no_effect(self):
        with fin._maybe_vfx_effect(object(), False, None, 0, 10, 10) as effect:
            self.assertIsNone(effect)


class RTXRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(rt.RUNTIME_ENV_VAR, None)

    def tearDown(self):
        os.environ.pop(rt.RUNTIME_ENV_VAR, None)
        if self._env is not None:
            os.environ[rt.RUNTIME_ENV_VAR] = self._env

    def test_marker_lives_in_this_pack_with_the_luna_name(self):
        marker = rt.runtime_marker_path()
        self.assertEqual(marker.name, "Luna_RTX_VFX_runtime_path.txt")
        self.assertEqual(marker.parent, Path(PACK) / "luna_tools" / "tools")

    def test_no_marker_means_no_override(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(rt.read_rtx_vfx_runtime_path(Path(d)))

    def test_marker_with_matching_python_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Path(d) / rt.expected_python_runtime_segment() / "r1"
            (runtime / "nvvfx").mkdir(parents=True)
            (Path(d) / "tools").mkdir()
            (Path(d) / "tools" / rt.RUNTIME_MARKER_NAME).write_text(f'"{runtime}"\n', encoding="utf-8")
            self.assertEqual(rt.read_rtx_vfx_runtime_path(Path(d)), runtime)

    def test_marker_for_another_python_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Path(d) / "py27" / "r1"
            (runtime / "nvvfx").mkdir(parents=True)
            (Path(d) / "tools").mkdir()
            (Path(d) / "tools" / rt.RUNTIME_MARKER_NAME).write_text(str(runtime), encoding="utf-8")
            self.assertIsNone(rt.read_rtx_vfx_runtime_path(Path(d)))

    def test_environment_variable_names_the_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Path(d) / rt.expected_python_runtime_segment() / "r2"
            (runtime / "nvvfx").mkdir(parents=True)
            os.environ[rt.RUNTIME_ENV_VAR] = str(runtime)
            self.assertEqual(rt.read_rtx_vfx_runtime_path(Path(d)), runtime)

    def test_finding_the_installed_runtime_does_not_import_it(self):
        sys.modules.pop("nvvfx", None)
        rt.find_installed_nvvfx()
        rt.installed_nvvfx_version()
        self.assertNotIn("nvvfx", sys.modules)

    def test_prefer_runtime_path_moves_it_to_the_front(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = Path(d) / rt.expected_python_runtime_segment() / "r3"
            (runtime / "nvvfx").mkdir(parents=True)
            os.environ[rt.RUNTIME_ENV_VAR] = str(runtime)
            saved = list(sys.path)
            try:
                self.assertEqual(rt.prefer_rtx_vfx_runtime_path(Path(d)), runtime)
                self.assertEqual(sys.path[0], str(runtime))
            finally:
                sys.path[:] = saved

    def test_missing_nvvfx_error_describes_this_pack(self):
        saved = sys.modules.get("nvvfx")
        sys.modules["nvvfx"] = None          # makes "from nvvfx import ..." raise ImportError
        try:
            with self.assertRaises(RuntimeError) as ctx:
                up._import_vfx()
        finally:
            if saved is None:
                sys.modules.pop("nvvfx", None)
            else:
                sys.modules["nvvfx"] = saved
        msg = str(ctx.exception)
        self.assertIn("Luna-Media-Tools", msg)
        self.assertIn("pip install", msg)
        self.assertIn("nvidia-vfx", msg)
        self.assertIsNone(re.search(r"deno(?!ise)", msg, re.I), msg)
        self.assertIsNone(re.search(r"install_rtx_vfx|\.bat", msg), msg)


class NamingTests(unittest.TestCase):
    """DaSiWa and Deno appear only in the licence/credit header of a ported file."""

    PATTERN = re.compile(r"dasiwa|\bdeno(?![a-z])", re.I)

    def test_no_foreign_names_outside_the_credit_header(self):
        files = list((Path(PACK) / "luna_tools").glob("*.py")) + list((Path(PACK) / "web").glob("luna_*.js"))
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
        for name in ("luna_rtx_upscale.py", "luna_rtx_video_finisher.py", "luna_rtx_runtime.py", "luna_resolution_common.py"):
            head = (Path(PACK) / "luna_tools" / name).read_text(encoding="utf-8")[:600]
            self.assertIn("Deno2026/comfyui-deno-custom-nodes", head, name)
            self.assertIn("0cd75f4", head, name)
            self.assertIn("GPL-3.0", head, name)


if __name__ == "__main__":
    unittest.main()
