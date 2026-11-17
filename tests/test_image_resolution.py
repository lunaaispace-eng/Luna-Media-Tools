"""Luna-ImageResolution: presets, portrait, size modes, snap, model rules, the wired image, latent shapes, info."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
ir = sys.modules[PACK.__name__ + ".image_resolution"]
Node = ir.LunaImageResolution


def run(**kw):
    args = dict(category="Aspect ratio", preset="1:1", portrait=False, size_mode="native", megapixels=1.0,
                longest_side=1024, snap="16", batch=1, image=None)
    args.update(kw)
    return Node().run(**args)["result"]


def test_outputs():
    assert Node.RETURN_NAMES == ("width", "height", "latent", "info")
    assert len(Node.OUTPUT_TOOLTIPS) == 4
    assert Node.CATEGORY == "Luna-Media-Tools"


def test_ratio_at_one_mp():
    w, h, latent, info = run(preset="16:9")
    assert (w, h) == (1360, 768)
    assert w % 16 == 0 and h % 16 == 0
    assert tuple(latent["samples"].shape) == (1, 4, 96, 170)
    assert latent["downscale_ratio_spacial"] == 8
    assert info.splitlines()[0].startswith("1360×768")


def test_portrait_turns_a_size_but_not_a_fixed_one():
    assert run(category="Display", preset="1080p", portrait=True, snap="8")[:2] == (1080, 1920)
    assert run(category="Social media", preset="X header", portrait=True, snap="8")[:2] == (1504, 496)


def test_native_preset_snaps_and_says_so():
    w, h, _, info = run(category="Display", preset="1080p", snap="64")
    assert (w, h) == (1920, 1088)
    assert "snapped from 1920x1080" in info


def test_model_raises_the_snap_and_uses_its_latent():
    w, h, latent, info = run(category="Qwen-Image 2.1", preset="16:9 (2752x1536)", snap="8")
    assert (w, h) == (2752, 1536)
    assert tuple(latent["samples"].shape) == (1, 64, 96, 172)
    assert latent["downscale_ratio_spacial"] == 16
    assert "snap raised to 32" in info


def test_h3_image_latent_is_3d_24ch():
    w, h, latent, _ = run(category="MiniMax H3 Image", preset="1:1 (4096x4096)", batch=2)
    assert (w, h) == (4096, 4096)
    assert tuple(latent["samples"].shape) == (2, 24, 1, 256, 256)


def test_krea_latent_is_5d_wan():
    w, h, latent, _ = run(category="Krea 2", preset="1:1")
    assert (w, h) == (1024, 1024)
    assert tuple(latent["samples"].shape) == (1, 16, 1, 128, 128)


def test_megapixels_and_longest_side():
    assert run(preset="1:1", size_mode="megapixels", megapixels=4.0)[:2] == (2048, 2048)
    assert run(preset="3:2", size_mode="longest_side", longest_side=1536)[:2] == (1536, 1024)


def test_out_of_range_warning():
    info = run(category="SDXL", preset="1:1", size_mode="megapixels", megapixels=4.0)[3]
    assert "outside SDXL" in info


def test_unknown_preset_falls_back_with_a_warning():
    w, h, _, info = run(category="Fibo 2.0", preset="nope")
    assert (w, h) == (1024, 1024) and "not a Fibo 2.0 preset" in info


def test_image_shape_wins():
    img = torch.zeros(1, 1350, 1080, 3)
    w, h, _, info = run(image=img, size_mode="megapixels", megapixels=1.0)
    assert h > w and abs(w / h - 0.8) < 0.02
    w, h, _, info = run(category="Fibo 2.0", preset="1:1 (1024x1024)", image=torch.zeros(1, 1080, 1920, 3))
    assert (w, h) == (1424, 800) and "from image 1920x1080" in info
    w, h, _, _ = run(category="Fibo 2.0", preset="1:1 (1024x1024)", image=torch.zeros(1, 1920, 1080, 3))
    assert (w, h) == (800, 1424)


def test_custom_file(tmp_path):
    p = tmp_path / "resolutions.json"
    p.write_text(json.dumps([{"label": "Poster", "width": 2400, "height": 3000}, {"label": "Wide", "value": "3000x1000"},
                             {"label": "bad"}]), encoding="utf-8")
    custom = ir.read_custom(str(p))
    assert [c[0] for c in custom] == ["Poster", "Wide"]
    r = ir.compute("Custom", "Poster (2400x3000)", True, snap=8, custom=custom)
    assert (r["width"], r["height"]) == (2400, 3000)   # custom sizes keep their orientation


def test_classifier_marks_inexact_ratios():
    assert ir.classify(1920, 1080) == ("16:9", True)
    assert ir.classify(1000, 700)[1] is False


def test_every_category_has_a_working_default():
    for c in ir.CATEGORIES:
        r = ir.compute(c, "", custom=[])
        assert r["width"] > 0 and r["height"] > 0, c
        assert r["width"] % r["snap"] == 0 and r["height"] % r["snap"] == 0, c


def test_readout_line_one_splits_for_the_frontend():
    """web/luna_image_resolution.js splits line 1 on runs of 2+ spaces into size, ratio, MP."""
    import re
    for c in ir.CATEGORIES:
        parts = re.split(r"\s{2,}", ir.compute(c, "", custom=[])["lines"][0])
        assert len(parts) == 3 and parts[2].endswith("MP"), (c, parts)


def test_routes_are_registered():
    import server
    paths = {getattr(r, "path", None) for r in server.PromptServer.instance.routes}
    assert {"/luna_media/resolution", "/luna_media/resolution/presets"} <= paths


def test_wired_native_preset_is_snapped():
    r = ir.compute("Social media", "", size_mode="native", snap=16, image_size=(1080, 1350), custom=[])
    assert r["width"] % 16 == 0 and r["height"] % 16 == 0


def test_wired_edit_sizes_match_the_core_encoders():
    """Mirrors of TextEncodeQwenImage21 (resolution 1024), TextEncodeFibo2 trained_size and MiniMaxH3ImageConditioning."""
    import importlib
    import math as m
    comfy = os.environ.get("COMFYUI_DIR", r"E:\ComfyUI-Easy-Install\ComfyUI")
    if comfy not in sys.path:
        sys.path.insert(0, comfy)
    fibo = importlib.import_module("comfy_extras.nodes_fibo2")
    h3 = importlib.import_module("comfy_extras.nodes_minimax_h3")
    for iw, ih in [(1600, 1000), (1080, 1350), (4000, 3000), (640, 480), (3000, 1000)]:
        q = ir.compute("Qwen-Image 2.1", "", image_size=(iw, ih))
        ratio = iw / ih
        assert (q["width"], q["height"]) == (max(32, round(m.sqrt(1024 * 1024 * ratio) / 32) * 32),
                                             max(32, round(m.sqrt(1024 * 1024 / ratio) / 32) * 32))
        f = ir.compute("Fibo 2.0", "", image_size=(iw, ih))
        assert (f["width"], f["height"]) == tuple(fibo.trained_size(iw / ih))
        hh = ir.compute("MiniMax H3 Image", "", image_size=(iw, ih))
        assert (hh["width"], hh["height"]) == tuple(h3._size_short_edge(iw, ih, h3.IMAGE_REF_SHORT_EDGE,
                                                                        h3.IMAGE_REF_VAE_MAX_SHORT_EDGE))
