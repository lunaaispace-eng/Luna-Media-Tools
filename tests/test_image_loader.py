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


def _roles(*roles, target=None):
    state = {"pictures": [{"role": r} for r in roles]}
    if target is not None:
        state["settings"] = {"target": target}
    return json.dumps(state)


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
    # No ⚙ Target (no settings in media_state) = General: card order, as before, plus the names line.
    assert brief == ("Image 1 (edit base, 1920x1080) · Image 2 (reference, 832x1216) · "
                     "@ names: @image1 = Image 1; @image2 = Image 2.")
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
    assert "Card 2 (@image2) is also marked edit base" in out[2]


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


# --- ⚙ Target and the @ names (luna_naming) ------------------------------------------------------

naming = sys.modules[PACK.__name__ + ".luna_naming"]
A_ROLES = ("ref", "style", "edit")
A_SIZES = ((64, 48), (80, 60), (96, 72))   # card 1, 2, 3: told apart by their shapes


def _three(folder):
    return "\n".join(_pic(folder, f"c{i + 1}.png", w, h) for i, (w, h) in enumerate(A_SIZES))


def _shape(card):
    w, h = A_SIZES[card - 1]
    return (1, h, w, 3)


def test_a_minimax_image_sends_the_edit_base_first(tmp_path):
    out = Node().load(_three(tmp_path), 3, "", _roles(*A_ROLES, target="minimax_image"))
    w, h, brief, refs = out[:4]
    assert [tuple(t.shape) for t in out[4:7]] == [_shape(3), _shape(1), _shape(2)]
    assert out[7] is None
    assert [tuple(t.shape) for t in refs["images"]] == [_shape(3), _shape(1), _shape(2)]
    assert refs["naming"]["map"] == {"@image1": ("image", 2), "@image2": ("image", 3), "@image3": ("image", 1)}
    assert refs["naming"]["row"] == "minimax" and refs["naming"]["order"] == [2, 0, 1]
    assert refs["roles"] == ["edit base", "reference", "style"]
    assert refs["items"]["images"] == ["edit base, 96x72", "reference, 64x48", "style, 80x60"]
    assert (w, h) == (96, 72) and (refs["width"], refs["height"]) == (96, 72)
    assert brief == ("Image 1 (edit base, 96x72, card 3) · Image 2 (reference, 64x48, card 1) · "
                     "Image 3 (style, 80x60, card 2) · "
                     "@ names: @image1 = <Picture 2>; @image2 = <Picture 3>; @image3 = <Picture 1>.")
    assert refs["brief"] == brief


def test_f2_the_writers_read_the_map_back(tmp_path):
    brief = Node().load(_three(tmp_path), 3, "", _roles(*A_ROLES, target="minimax_image"))[2]
    read = naming.read_names_line(brief)
    assert read["map"] == {"@image1": ("image", 2), "@image2": ("image", 3), "@image3": ("image", 1)}
    assert read["row"] == "minimax"


def test_b_qwen_same_order_its_own_tags(tmp_path):
    out = Node().load(_three(tmp_path), 3, "", _roles(*A_ROLES, target="qwen"))
    brief, refs = out[2], out[3]
    assert [tuple(t.shape) for t in out[4:7]] == [_shape(3), _shape(1), _shape(2)]
    assert refs["roles"] == ["edit base", "reference", "style"]
    assert refs["naming"]["row"] == "qwen"
    assert refs["naming"]["map"] == {"@image1": ("image", 2), "@image2": ("image", 3), "@image3": ("image", 1)}
    assert "@image3 = <image1>" in brief
    assert brief.endswith("@ names: @image1 = <image2>; @image2 = <image3>; @image3 = <image1>.")


@pytest.mark.parametrize("media", [_roles(*A_ROLES, target="general"), _roles(*A_ROLES),
                                   _roles(*A_ROLES, target="bogus")])
def test_c_general_and_old_saved_nodes_keep_card_order(tmp_path, media):
    out = Node().load(_three(tmp_path), 3, "", media)
    w, h, brief, refs = out[:4]
    assert [tuple(t.shape) for t in out[4:7]] == [_shape(1), _shape(2), _shape(3)]
    assert [tuple(t.shape) for t in refs["images"]] == [_shape(1), _shape(2), _shape(3)]
    assert refs["roles"] == ["reference", "style", "edit base"]
    assert (w, h) == (96, 72)   # still the edit base's size
    before = "Image 1 (reference, 64x48) · Image 2 (style, 80x60) · Image 3 (edit base, 96x72)"
    assert brief == before + " · @ names: @image1 = Image 1; @image2 = Image 2; @image3 = Image 3."
    assert refs["naming"]["row"] == "cloud" and refs["naming"]["order"] == [0, 1, 2]


def test_d_no_edit_base_keeps_card_order(tmp_path):
    out = Node().load(_three(tmp_path), 3, "", _roles("ref", "style", "composition", target="minimax_image"))
    brief, refs = out[2], out[3]
    assert [tuple(t.shape) for t in out[4:7]] == [_shape(1), _shape(2), _shape(3)]
    assert refs["naming"]["map"] == {"@image1": ("image", 1), "@image2": ("image", 2), "@image3": ("image", 3)}
    assert "card " not in brief
    assert out[:2] == (64, 48)   # no edit base: the first picture sent


def test_e_zero_active_has_a_naming_of_zero_cards(tmp_path):
    a = _pic(tmp_path, "a.png", 64, 64)
    out = Node().load(a, 0, "", _roles("edit", target="qwen"))
    assert out[:3] == (0, 0, "No pictures.")
    assert all(x is None for x in out[4:])
    refs = out[3]
    assert refs["naming"]["map"] == {} and refs["naming"]["counts"]["image"] == 0
    assert refs["naming"]["row"] == "qwen"
    assert naming.from_refs(refs) is refs["naming"]
    assert set(refs) == set(il.empty_refs("x")), "no key added or lost beside naming"


# The Luna packs' bundle (Luna-MinimaxH3 SPEC.md § 2), the Asset Loader's keys, plus naming and the keyframe list
# (Luna-Media-Tools SPEC.md § 1, 2026-11-20).
REFS_KEYS = {"images", "roles", "videos", "video_audios", "audios", "song", "vocals", "keyframes", "brief", "items",
             "width", "height", "naming"}


def test_refs_keys_only_gain_naming(tmp_path):
    refs = Node().load(_three(tmp_path), 2, "", _roles(*A_ROLES, target="minimax_image"))[3]
    assert set(refs) == REFS_KEYS
    assert refs["vocals"] is None and refs["keyframes"] == []
    # The edit base (card 3) is not active: card order, and its @ name is not in the map, so it warns.
    assert refs["naming"]["map"] == {"@image1": ("image", 1), "@image2": ("image", 2)}
    assert naming.replace("@image3 in the style of @image2", refs["naming"])[1]


def test_empty_refs_have_the_bundle_keys(tmp_path):
    for refs in (il.empty_refs("x"), il.empty_refs("x", naming.make("qwen")),
                 Node().load(_pic(tmp_path, "a.png", 8, 8), 0, "", "")[3]):
        assert set(refs) == REFS_KEYS
        assert refs["vocals"] is None and refs["keyframes"] == []
    # A fresh list each call: one run's keyframes never leak into another's.
    assert il.empty_refs("x")["keyframes"] is not il.empty_refs("x")["keyframes"]


def test_parse_target():
    assert [il.parse_target(_roles(target=t)) for t in il.TARGETS] == list(il.TARGETS)
    assert il.parse_target("") == "general" and il.parse_target("{bad") == "general"
    assert il.parse_target(json.dumps({"settings": {"target": "Qwen"}})) == "general"
    assert il.parse_target(json.dumps({"settings": "qwen"})) == "general"
    assert {naming.row_for(t) for t in il.TARGETS} == {"minimax", "qwen", "cloud"}


def test_inputs_keep_their_order():
    types = Node.INPUT_TYPES()
    assert list(types["required"]) == ["image_paths", "image_count"]
    assert list(types["optional"]) == ["asset_state", "media_state"]


def _js_card_names(cases):
    """cardNames (web/luna_image_loader.js) for each (target, roles, active), through the node's real file."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node not on PATH")
    web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "luna_image_loader.js")
    script = (
        "import { registerHooks } from 'node:module'; import fs from 'node:fs'; import { fileURLToPath } from 'node:url';"
        "registerHooks({"
        " resolve(s, c, n) { return s.endsWith('/scripts/app.js') ? { url: 'data:text/javascript,export const app = "
        "{ registerExtension() {} };', shortCircuit: true } : n(s, c); },"
        " load(u, c, n) { return u.endsWith('/web/luna_image_loader.js') ? { format: 'module', source: "
        "fs.readFileSync(fileURLToPath(u), 'utf8'), shortCircuit: true } : n(u, c); } });"
        "const L = await import(process.argv[1]); const cases = JSON.parse(fs.readFileSync(0, 'utf8'));"
        "console.log(JSON.stringify(cases.map(([t, r, a]) => { const x = L.cardNames(t, r, a);"
        " return [x.order, x.cards.map((c) => c.tag)]; })));"
    )
    url = "file:///" + web.replace("\\", "/")
    # The cases go on stdin: as an argument they pass the Windows command-line limit.
    out = subprocess.run(["node", "--no-warnings", "--input-type=module", "-e", script, url], input=json.dumps(cases),
                         capture_output=True, text=True, check=True).stdout
    return [tuple(x) for x in json.loads(out)]


def test_js_card_rule_matches_python():
    """The cards' @ name tags (cardNames) are the ones load() sends: every role list up to 3 cards, every active count,
    every Target and an unknown one."""
    import itertools
    role_lists = [list(r) for n in range(4) for r in itertools.product(il.ROLES, repeat=n)]
    cases = [(t, roles, a) for t in (*il.TARGETS, "bogus") for roles in role_lists for a in range(len(roles) + 1)]
    py = []
    for t, roles, a in cases:
        sent, _ = il.parse_roles(_roles(*roles), a)
        row = naming.row_for(il.parse_target(_roles(target=t)))
        made = naming.make(row, sent)
        tags = [naming.tag(row, "image", made["map"][f"@image{i + 1}"][1]) if i < a else None
                for i in range(len(roles))]
        py.append((made["order"], tags))
    assert _js_card_names(cases) == [tuple(x) for x in py]
