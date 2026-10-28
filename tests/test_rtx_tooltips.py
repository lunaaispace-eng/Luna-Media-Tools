# Derived from tests/test_rtx_tooltips.py in ComfyUI-DaSiWa-Nodes (https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes),
# commit ab364c5, GPL-3.0 (see THIRD_PARTY_NOTICES.md). Changes: the nodes under test are Luna-RTXUpscaler /
# Luna-RTXFrameGen (luna_tools/luna_rtx_upscaler.py, luna_rtx_frame_gen.py), loaded through the pack package.
"""Native hover coverage and frozen pre-tooltip workflow ABI (no GPU required)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registration import load_pack  # noqa: E402

load_pack()
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_frame_gen as fg
import Luna_Media_Tools_under_test.luna_tools.luna_rtx_upscaler as rtx

# Types/choices, input ordering, defaults, limits and capability flags from the
# pre-tooltip schema. Documentation fields are intentionally excluded.
EXPECTED = {('upscaler', False): [('required',
                        [('images', ('IMAGE', {})),
                         ('denoise', ('BOOLEAN', {'default': False})),
                         ('denoise_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                         ('deblur', ('BOOLEAN', {'default': False})),
                         ('deblur_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                         ('upscale', (['Off', 'VSR', 'High Bitrate'], {'default': 'VSR'})),
                         ('upscale_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                         ('resize_type',
                          (['Keep Ratio', 'Manual', 'Preset Ratio', 'Scale', 'Same Size'],
                           {'default': 'Scale'})),
                         ('scale', ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.05})),
                         ('megapixels', ('FLOAT', {'default': 2.0, 'min': 0.01, 'max': 64.0, 'step': 0.01})),
                         ('width', ('INT', {'default': 1920, 'min': 64, 'max': 8192, 'step': 8})),
                         ('height', ('INT', {'default': 1080, 'min': 64, 'max': 8192, 'step': 8})),
                         ('divisible_by', (['8', '16', '32', '64', '128'], {'default': '8'})),
                         ('ratio_preset', (['1:1', '4:3', '3:2', '16:9', '21:9'], {'default': '16:9'})),
                         ('resize_method',
                          (['Center Crop (Fill)', 'Letterbox (Fit)'], {'default': 'Center Crop (Fill)'})),
                         ('device_id', ('INT', {'default': 0, 'min': 0, 'max': 8, 'step': 1}))]),
                       ('optional',
                        [('empty_cache', ('BOOLEAN', {'default': False})),
                         ('use_mmap', ('BOOLEAN', {'default': False})),
                         ('auto_unload_models', ('BOOLEAN', {'default': True})),
                         ('chunking', ('BOOLEAN', {'default': True})),
                         ('lossless_fp16', ('BOOLEAN', {'default': True})),
                         ('chunk_frames', ('INT', {'default': 16, 'min': 1, 'max': 1024, 'step': 1})),
                         ('image_encoding',
                          (['8-bit RGB'], {'default': '8-bit RGB', 'luna_unavailable': True})),
                         ('denoise_strength',
                          ('FLOAT',
                           {'default': 1.0,
                            'min': 0.0,
                            'max': 1.0,
                            'step': 0.01,
                            'luna_unavailable': True})),
                         ('deblur_strength',
                          ('FLOAT',
                           {'default': 1.0,
                            'min': 0.0,
                            'max': 1.0,
                            'step': 0.01,
                            'luna_unavailable': True})),
                         ('upscale_strength',
                          ('FLOAT',
                           {'default': 1.0,
                            'min': 0.0,
                            'max': 1.0,
                            'step': 0.01,
                            'luna_unavailable': True})),
                         ('preserve_alpha', ('BOOLEAN', {'default': False})),
                         ('channel_integrity_check', ('BOOLEAN', {'default': False})),
                         ('hdr_enabled', ('BOOLEAN', {'default': False, 'luna_unavailable': True})),
                         ('hdr_contrast',
                          ('INT',
                           {'default': 100, 'min': 0, 'max': 200, 'step': 1, 'luna_unavailable': True})),
                         ('hdr_saturation',
                          ('INT',
                           {'default': 100, 'min': 0, 'max': 200, 'step': 1, 'luna_unavailable': True})),
                         ('hdr_middle_gray',
                          ('INT',
                           {'default': 50, 'min': 10, 'max': 100, 'step': 1, 'luna_unavailable': True})),
                         ('hdr_luminance',
                          ('INT',
                           {'default': 650, 'min': 400, 'max': 2000, 'step': 1, 'luna_unavailable': True})),
                         ('hdr_debanding', ('BOOLEAN', {'default': True, 'luna_unavailable': True}))])],
 ('upscaler', True): [('required',
                       [('images', ('IMAGE', {})),
                        ('denoise', ('BOOLEAN', {'default': False})),
                        ('denoise_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                        ('deblur', ('BOOLEAN', {'default': False})),
                        ('deblur_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                        ('upscale',
                         (['Off', 'VSR', 'High Bitrate', 'Streaming Medium', 'Streaming Ultra'],
                          {'default': 'VSR'})),
                        ('upscale_quality', (['Low', 'Medium', 'High', 'Ultra'], {'default': 'Ultra'})),
                        ('resize_type',
                         (['Keep Ratio', 'Manual', 'Preset Ratio', 'Scale', 'Same Size'],
                          {'default': 'Scale'})),
                        ('scale', ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': 0.05})),
                        ('megapixels', ('FLOAT', {'default': 2.0, 'min': 0.01, 'max': 64.0, 'step': 0.01})),
                        ('width', ('INT', {'default': 1920, 'min': 64, 'max': 8192, 'step': 8})),
                        ('height', ('INT', {'default': 1080, 'min': 64, 'max': 8192, 'step': 8})),
                        ('divisible_by', (['8', '16', '32', '64', '128'], {'default': '8'})),
                        ('ratio_preset', (['1:1', '4:3', '3:2', '16:9', '21:9'], {'default': '16:9'})),
                        ('resize_method',
                         (['Center Crop (Fill)', 'Letterbox (Fit)'], {'default': 'Center Crop (Fill)'})),
                        ('device_id', ('INT', {'default': 0, 'min': 0, 'max': 8, 'step': 1}))]),
                      ('optional',
                       [('empty_cache', ('BOOLEAN', {'default': False})),
                        ('use_mmap', ('BOOLEAN', {'default': False})),
                        ('auto_unload_models', ('BOOLEAN', {'default': True})),
                        ('chunking', ('BOOLEAN', {'default': True})),
                        ('lossless_fp16', ('BOOLEAN', {'default': True})),
                        ('chunk_frames', ('INT', {'default': 16, 'min': 1, 'max': 1024, 'step': 1})),
                        ('image_encoding',
                         (['8-bit RGB', '10-bit RGB (RGB10A2)'],
                          {'default': '8-bit RGB', 'luna_unavailable': False})),
                        ('denoise_strength',
                         ('FLOAT',
                          {'default': 1.0,
                           'min': 0.0,
                           'max': 1.0,
                           'step': 0.01,
                           'luna_unavailable': False})),
                        ('deblur_strength',
                         ('FLOAT',
                          {'default': 1.0,
                           'min': 0.0,
                           'max': 1.0,
                           'step': 0.01,
                           'luna_unavailable': False})),
                        ('upscale_strength',
                         ('FLOAT',
                          {'default': 1.0,
                           'min': 0.0,
                           'max': 1.0,
                           'step': 0.01,
                           'luna_unavailable': False})),
                        ('preserve_alpha', ('BOOLEAN', {'default': False})),
                        ('channel_integrity_check', ('BOOLEAN', {'default': False})),
                        ('hdr_enabled', ('BOOLEAN', {'default': False, 'luna_unavailable': False})),
                        ('hdr_contrast',
                         ('INT',
                          {'default': 100, 'min': 0, 'max': 200, 'step': 1, 'luna_unavailable': False})),
                        ('hdr_saturation',
                         ('INT',
                          {'default': 100, 'min': 0, 'max': 200, 'step': 1, 'luna_unavailable': False})),
                        ('hdr_middle_gray',
                         ('INT',
                          {'default': 50, 'min': 10, 'max': 100, 'step': 1, 'luna_unavailable': False})),
                        ('hdr_luminance',
                         ('INT',
                          {'default': 650, 'min': 400, 'max': 2000, 'step': 1, 'luna_unavailable': False})),
                        ('hdr_debanding', ('BOOLEAN', {'default': True, 'luna_unavailable': False}))])],
 ('framegen', False): [('required',
                        [('images', ('IMAGE', {})),
                         ('generation_type', (['frame rate multiplier', 'specific timestep'], {})),
                         ('multiplier', ('INT', {'default': 2, 'min': 2, 'max': 16})),
                         ('timestep', ('FLOAT', {'default': 0.5, 'min': 0.01, 'max': 0.99, 'step': 0.01})),
                         ('mode', (['LOW', 'MEDIUM', 'HIGH'], {'default': 'HIGH'})),
                         ('automatic_shot_change_detection', ('BOOLEAN', {'default': True})),
                         ('shot_change', ('BOOLEAN', {'default': False})),
                         ('device_id', ('INT', {'default': 0, 'min': 0, 'max': 8}))]),
                       ('optional',
                        [('image_encoding', (['8-bit RGB'], {'default': '8-bit RGB'})),
                         ('auto_unload_models', ('BOOLEAN', {'default': True})),
                         ('use_mmap', ('BOOLEAN', {'default': False}))])],
 ('framegen', True): [('required',
                       [('images', ('IMAGE', {})),
                        ('generation_type', (['frame rate multiplier', 'specific timestep'], {})),
                        ('multiplier', ('INT', {'default': 2, 'min': 2, 'max': 16})),
                        ('timestep', ('FLOAT', {'default': 0.5, 'min': 0.01, 'max': 0.99, 'step': 0.01})),
                        ('mode', (['LOW', 'MEDIUM', 'HIGH'], {'default': 'HIGH'})),
                        ('automatic_shot_change_detection', ('BOOLEAN', {'default': True})),
                        ('shot_change', ('BOOLEAN', {'default': False})),
                        ('device_id', ('INT', {'default': 0, 'min': 0, 'max': 8}))]),
                      ('optional',
                       [('image_encoding', (['8-bit RGB', '10-bit RGB (RGB10A2)'], {'default': '8-bit RGB'})),
                        ('auto_unload_models', ('BOOLEAN', {'default': True})),
                        ('use_mmap', ('BOOLEAN', {'default': False}))])]}

@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("name", ["upscaler", "framegen"])
def test_tooltips_cover_every_input_and_preserve_abi(monkeypatch, modern, name):
    monkeypatch.setattr(rtx, "_luna_rtx_capabilities", lambda: {
        "encoding": modern, "strength": modern, "hdr": modern,
        "streaming": ["Streaming Medium", "Streaming Ultra"] if modern else [],
    })
    monkeypatch.setattr(fg, "_luna_fg_sdk", lambda: (None, None, None, modern))
    cls = rtx.LunaRTXUpscaler if name == "upscaler" else fg.LunaRTXFrameGen
    schema = cls.INPUT_TYPES()
    actual = []
    tips = []
    for section, inputs in schema.items():
        rows = []
        for input_name, (kind, options) in inputs.items():
            help_text = options.get("tooltip", "")
            assert isinstance(help_text, str) and len(help_text.strip()) >= 20, input_name
            tips.append(help_text)
            if isinstance(kind, list):
                for choice in kind:
                    assert str(choice) in help_text, (input_name, choice)
            rows.append((input_name, (kind, {k: v for k, v in options.items()
                                           if k not in ("tooltip", "description")})))
        actual.append((section, rows))
    assert len(tips) == len(set(tips)), "Each control needs distinct meaningful help"
    assert actual == EXPECTED[name, modern]
    assert cls.RETURN_TYPES == ("IMAGE",)
    assert cls.RETURN_NAMES == ("images",)
    assert cls.FUNCTION == ("execute" if name == "upscaler" else "luna_rtx_generate_frames")
    module = rtx if name == "upscaler" else fg
    assert module.NODE_CLASS_MAPPINGS["Luna-RTXUpscaler" if name == "upscaler" else "Luna-RTXFrameGen"] is cls
