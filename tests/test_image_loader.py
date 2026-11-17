"""Luna-ImageLoader: active count, roles, own sizes (no bars), outputs and the refs payload."""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
il = sys.modules[PACK.__name__ + ".image_loader"]
Node = il.LunaImageLoader


def _pic(folder, name, w, h, mode="RGB", color=(200, 100, 50)):
    path = os.path.join(str(folder), name)
    Image.new(mode, (w, h), color if mode == "RGB" else (*color, 0)).save(path)
    return name


@pytest.fixture(autouse=True)
def _input_dir(tmp_path, monkeypatch):
    # ComfyUI refuses absolute paths: pictures are file names in the input folder, as on the node.
    monkeypatch.setattr(il.folder_paths, "get_input_directory", lambda *a, **k: str(tmp_path))


def _roles(*roles):
    return json.dumps({"pictures": [{"role": r} for r in roles]})


def test_outputs_order_and_names():
    assert Node.RETURN_NAMES[:4] == ("width", "height", "asset_brief", "refs")
    assert Node.RETURN_NAMES[4:] == tuple(f"image_{i}" for i in range(1, 10))
    assert Node.RETURN_TYPES[3] == "LUNA_H3_REFS"
    assert len(Node.OUTPUT_TOOLTIPS) == len(Node.RETURN_NAMES)
    assert Node.CATEGORY == "Luna-Media-Tools"


def test_each_picture_keeps_its_own_size(tmp_path):
    a = _pic(tmp_path, "a.png", 1920, 1080)
    b = _pic(tmp_path, "b.png", 832, 1216)
    out = Node().load(f"{a}\n{b}", 2, "", _roles("edit", "ref"))
    w, h, brief, refs = out[:4]
    assert (w, h) == (1920, 1080)
    assert tuple(out[4].shape) == (1, 1080, 1920, 3)
    assert tuple(out[5].shape) == (1, 1216, 832, 3)
    assert out[6] is None
    assert brief == "Image 1 (edit base, 1920x1080) · Image 2 (reference, 832x1216)"
    assert refs["roles"] == ["edit base", "reference"]
    assert [tuple(t.shape) for t in refs["images"]] == [(1, 1080, 1920, 3), (1, 1216, 832, 3)]
    assert refs["items"]["images"] == ["edit base, 1920x1080", "reference, 832x1216"]
    assert refs["videos"] == [] and refs["audios"] == [] and refs["brief"] == brief


def test_width_follows_the_edit_base(tmp_path):
    a = _pic(tmp_path, "a.png", 640, 480)
    b = _pic(tmp_path, "b.png", 512, 768)
    w, h = Node().load(f"{a}\n{b}", 2, "", _roles("style", "edit"))[:2]
    assert (w, h) == (512, 768)


def test_cards_above_the_active_count_are_not_sent(tmp_path):
    paths = "\n".join(_pic(tmp_path, f"{i}.png", 64 + i * 8, 64) for i in range(3))
    out = Node().load(paths, 2, "", "")
    assert out[4] is not None and out[5] is not None and out[6] is None
    assert len(out[3]["images"]) == 2
    assert "Image 3" not in out[2]


def test_zero_active_is_empty_not_an_error(tmp_path):
    a = _pic(tmp_path, "a.png", 64, 64)
    out = Node().load(a, 0, "", "")
    assert out[:3] == (0, 0, "No pictures.")
    assert out[3]["images"] == [] and out[3]["roles"] == []
    assert all(x is None for x in out[4:])


def test_second_edit_base_becomes_a_reference(tmp_path):
    a = _pic(tmp_path, "a.png", 64, 64)
    b = _pic(tmp_path, "b.png", 64, 64)
    out = Node().load(f"{a}\n{b}", 2, "", _roles("edit", "edit"))
    assert out[3]["roles"] == ["edit base", "reference"]
    assert "Image 2 is also marked edit base" in out[2]


def test_unknown_role_is_reference():
    roles, warnings = il.parse_roles(_roles("hero", "composition"), 3)
    assert roles == ["ref", "composition", "ref"] and warnings == []


def test_resize_state_and_brief_note(tmp_path):
    a = _pic(tmp_path, "a.png", 2000, 1000)
    state = json.dumps({"items": [{"mode": "longest_side", "longest_side": 1000, "snap": 16}]})
    out = Node().load(a, 1, state, "")
    assert tuple(out[4].shape) == (1, 496, 992, 3)
    assert out[:2] == (992, 496)
    assert "[resized from 2000x1000]" in out[2]


def test_crop_to_ratio_never_pads():
    img = Image.new("RGB", (1600, 900))
    got = il.apply_state(img, dict(il._DEFAULT_ITEM, ratio="1:1"))
    assert got.size == (900, 900)


def test_snap_goes_down_and_never_stretches():
    assert il.snap_size(1023.99999999, 777, 32) == (1024, 768)
    img = Image.new("RGB", (1000, 700))
    got = il.apply_state(img, dict(il._DEFAULT_ITEM, snap=64))
    assert got.size == (960, 640)


def test_transparent_picture_is_filled_and_noted(tmp_path):
    a = _pic(tmp_path, "a.png", 32, 32, mode="RGBA")
    out = Node().load(a, 1, "", "")
    assert "transparent areas filled with #000000" in out[2]
    assert float(out[4].max()) == 0.0


def test_too_many_lines_fail_validation():
    assert Node.VALIDATE_INPUTS("\n".join(f"{i}.png" for i in range(10)), 9) is not True


def test_missing_file_names_the_node(tmp_path):
    try:
        Node().load("nope.png", 1, "", "")
    except RuntimeError as e:
        assert "[Luna-ImageLoader]" in str(e)
    else:
        raise AssertionError("no error")


def test_changed_picture_changes_the_hash(tmp_path):
    a = _pic(tmp_path, "a.png", 16, 16)
    first = Node.IS_CHANGED(a, 1)
    Image.fromarray(np.full((16, 16, 3), 7, np.uint8)).save(os.path.join(str(tmp_path), a))
    assert Node.IS_CHANGED(a, 1) != first


def test_snap_wider_than_the_picture_does_not_upscale():
    got = il.apply_state(Image.new("RGB", (32, 24)), dict(il._DEFAULT_ITEM, snap=64))
    assert got.size == (32, 24)
    up = il.apply_state(Image.new("RGB", (32, 24)), dict(il._DEFAULT_ITEM, snap=64, allow_upscale=True))
    assert up.size == (64, 64)


CASES = [
    ((1921, 1081), {"mode": "scale_factor", "scale_factor": 0.5}),
    ((2000, 1000), {"mode": "longest_side", "longest_side": 1000, "snap": 16}),
    ((32, 24), {"snap": 64}),
    ((1600, 900), {"ratio": "1:1"}),
    ((1600, 900), {"ratio": "1:1", "ratio_action": "pad"}),
    ((1601, 899), {"ratio": "16:9", "ratio_action": "pad"}),
    ((1920, 1080), {"mode": "max_mp", "max_mp": 0}),
    ((3000, 2001), {"mode": "max_mp", "max_mp": 1.0, "snap": 32}),
    ((1000, 700), {"snap": 64}),
    ((833, 1217), {"ratio": "2:3", "mode": "longest_side", "longest_side": 768, "snap": 8}),
]


def test_js_size_prediction_matches_python():
    """The panel's predictSize (web/luna_cards.mjs) gives the size apply_state delivers."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node not on PATH")
    kit = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "luna_cards.mjs")
    script = ("const k = await import(process.argv[1]); const cases = JSON.parse(process.argv[2]);"
              "console.log(JSON.stringify(cases.map(([s, st]) => k.predictSize(s[0], s[1], {...k.DEFAULT_ITEM, ...st}))));")
    url = "file:///" + kit.replace("\\", "/")
    out = subprocess.run(["node", "--input-type=module", "-e", script, url, json.dumps(CASES)],
                         capture_output=True, text=True, check=True).stdout
    js = [tuple(x) for x in json.loads(out)]
    py = [il.apply_state(Image.new("RGB", size), dict(il._DEFAULT_ITEM, **st)).size for size, st in CASES]
    assert js == py
