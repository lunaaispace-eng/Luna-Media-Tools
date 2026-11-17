"""Luna-ImageLoader — the pictures for image work: edits, style transfer, text-to-image from references.

- The picture part of Luna-H3-AssetLoader (Luna-MinimaxH3 `asset_loader.py`, `asset_images.py`), without sound, clips,
  timeline or song, and with nothing MiniMax-specific. Same widget names and state JSON, so the card UI is the same.
- Up to 9 picture cards in order; card N is Image N everywhere downstream (the user prompt says @image1 … @image9).
- `image_count` is the number of ACTIVE pictures: cards 1…N are loaded and sent; a card above N stays in the node, off.
- Each card has a role, stored in `media_state` `pictures[i].role`: edit (edit base, at most one), ref (reference, default),
  style, composition.
- Each card has its own resize state in `asset_state` (the Asset Loader's schema). Unset, a picture goes out as loaded.
- No padding and no bars on any output: there is no letterboxed batch. Each picture keeps its own size.
- Fixed outputs first, `image_1…image_9` last: the frontend removes trailing outputs, and ComfyUI maps outputs by index.

Per-image resize state follows ComfyUI-Pixaroma's LoadImageMini state (https://github.com/pixaroma/ComfyUI-Pixaroma, MIT,
Copyright (c) 2026 pixaroma): key names and defaults. The code is ported from Luna-MinimaxH3 `asset_images.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import node_helpers

MAX_IMAGES = 9

# Stored role -> the words every consumer reads (asset_brief, refs["roles"], the writers' "Image N (role, WxH)").
ROLES = {"edit": "edit base", "ref": "reference", "style": "style", "composition": "composition"}

_DEFAULT_ITEM = {
    "mode": "off",           # off | max_mp | longest_side | scale_factor
    "max_mp": 2.0,
    "longest_side": 1024,
    "scale_factor": 1.0,
    "ratio": "",             # "" keeps the source aspect; else "16:9", "2:3", ...
    "ratio_action": "crop",  # crop | pad
    "crop_anchor": "center",
    "pad_color": "#000000",
    "snap": 0,               # 0 = off; 8, 16, 32, 64
    "allow_upscale": False,
}

# Pillow modes that carry transparency, and the 16/32-bit greyscale ones.
_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
_WIDE_GREY = ("I;16", "I;16B", "I;16L", "I;16N", "I")
# Float slack for "w * scale" landing a hair under a grid line (1023.9999 -> 1024).
_EPS = 1e-6


# --- picture handling (ported from Luna-MinimaxH3 asset_images.py) -------------------------------

def to_rgb(img: Image.Image, background: str = "#000000") -> tuple[Image.Image, str | None]:
    """Any Pillow picture -> 8-bit RGB, plus a note when transparent areas were filled with `background`."""
    note = None
    if img.mode in _WIDE_GREY:
        arr = np.asarray(img).astype(np.float64)
        top = 65535.0 if img.mode != "I" or arr.max(initial=0) > 255 else 255.0
        img = Image.fromarray(np.clip(arr / top * 255.0 + 0.5, 0, 255).astype(np.uint8))
    elif img.mode == "F":
        arr = np.asarray(img, dtype=np.float64)
        img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
    has_alpha = img.mode in _ALPHA_MODES or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            canvas = Image.new("RGBA", rgba.size, background or "#000000")
            canvas.alpha_composite(rgba)
            note = f"transparent areas filled with {background or '#000000'}"
            return canvas.convert("RGB"), note
        img = rgba
    return img.convert("RGB"), note


def _grid_floor(v: float, step: int) -> int:
    return max(step, int(math.floor(v / step + _EPS)) * step)


def snap_size(w: float, h: float, step: int) -> tuple[int, int]:
    """Each edge goes DOWN to the grid (snapping never upscales); the difference is cropped, never stretched."""
    if step and step > 1:
        return _grid_floor(w, step), _grid_floor(h, step)
    return max(1, round(w)), max(1, round(h))


def _anchored(span: int, keep: int, anchor: str, lo: str, hi: str) -> int:
    if lo in anchor:
        return 0
    if hi in anchor:
        return span - keep
    return (span - keep) // 2


def apply_state(img: Image.Image, st: dict) -> Image.Image:
    """Aspect first, then size, then snap: cropping after scaling would undo the size asked for."""
    if st.get("mode") == "off" and not st.get("ratio") and not st.get("snap"):
        return img
    w, h = img.size
    anchor = str(st.get("crop_anchor") or "center")

    ratio = str(st.get("ratio") or "").strip()
    if ratio and ":" in ratio:
        try:
            rw, rh = (float(x) for x in ratio.split(":", 1))
            target = rw / rh
        except Exception:
            target = None
        if target and target > 0 and abs(w / h - target) > 0.001:
            if st.get("ratio_action") == "pad":
                nw, nh = (w, round(w / target)) if w / h > target else (round(h * target), h)
                canvas = Image.new("RGB", (max(nw, w), max(nh, h)), st.get("pad_color") or "#000000")
                canvas.paste(img, ((max(nw, w) - w) // 2, (max(nh, h) - h) // 2))
                img = canvas
            else:
                nw, nh = (round(h * target), h) if w / h > target else (w, round(w / target))
                left = _anchored(w, nw, anchor, "left", "right")
                top = _anchored(h, nh, anchor, "top", "bottom")
                img = img.crop((left, top, left + nw, top + nh))
            w, h = img.size

    mode = str(st.get("mode") or "off")
    scale = 1.0
    if mode == "max_mp":
        budget = float(st.get("max_mp", 2.0)) * 1_000_000
        if budget > 0:
            scale = math.sqrt(budget / (w * h))
    elif mode == "longest_side":
        scale = float(st.get("longest_side", 1024)) / max(w, h)
    elif mode == "scale_factor":
        scale = float(st.get("scale_factor", 1.0))
    if not st.get("allow_upscale", False):
        scale = min(1.0, scale)

    step = int(st.get("snap") or 0)
    # A grid wider than an edge would force that edge UP to one grid step; with upscale off, the picture is not snapped.
    if step > 1 and not st.get("allow_upscale", False) and min(w * scale, h * scale) + _EPS < step:
        step = 0
    tw, th = snap_size(w * scale, h * scale, step)
    if (tw, th) == (w, h):
        return img
    cover = max(tw / w, th / h)
    cw, ch = max(tw, math.ceil(w * cover - _EPS)), max(th, math.ceil(h * cover - _EPS))
    if (cw, ch) != (w, h):
        img = img.resize((cw, ch), Image.LANCZOS)
    if (cw, ch) != (tw, th):
        left = _anchored(cw, tw, anchor, "left", "right")
        top = _anchored(ch, th, anchor, "top", "bottom")
        img = img.crop((left, top, left + tw, top + th))
    return img


# --- state ---------------------------------------------------------------------------------------

def split_paths(raw: str) -> list[str]:
    """One file name per line, as on the Asset Loader."""
    out: list[str] = []
    for line in (raw or "").replace("\r", "\n").split("\n"):
        line = line.strip().strip('"')
        if line:
            out.append(line)
    return out


def parse_state(raw: str, count: int) -> list[dict]:
    """asset_state JSON -> one settled resize state per picture. {"all": {...}} for every picture, {"items": [...]} per
    picture (items win). Malformed JSON falls back to defaults: a bad state string must not cost a run."""
    base = dict(_DEFAULT_ITEM)
    items: list[dict] = []
    try:
        parsed = json.loads(raw) if raw and raw.strip() else {}
        if isinstance(parsed, dict):
            shared = parsed.get("all")
            if isinstance(shared, dict):
                base.update({k: v for k, v in shared.items() if k in _DEFAULT_ITEM})
            listed = parsed.get("items")
            if isinstance(listed, list):
                items = [x if isinstance(x, dict) else {} for x in listed]
    except Exception:
        pass
    out = []
    for i in range(count):
        st = dict(base)
        if i < len(items):
            st.update({k: v for k, v in items[i].items() if k in _DEFAULT_ITEM})
        out.append(st)
    return out


def parse_roles(raw: str, count: int) -> tuple[list[str], list[str]]:
    """media_state JSON -> (one stored role per picture, warnings). Missing or unknown = ref. Only the first card marked
    edit stays the edit base; a later one is sent as a reference and a warning says so."""
    try:
        data = json.loads(raw) if raw and raw.strip() else {}
    except (TypeError, ValueError):
        data = {}
    listed = data.get("pictures") if isinstance(data, dict) else None
    listed = listed if isinstance(listed, list) else []
    roles, warnings, edit_seen = [], [], False
    for i in range(count):
        p = listed[i] if i < len(listed) and isinstance(listed[i], dict) else {}
        role = p.get("role") if p.get("role") in ROLES else "ref"
        if role == "edit":
            if edit_seen:
                warnings.append(f"Image {i + 1} is also marked edit base; only one picture can be, so it is sent as a reference.")
                role = "ref"
            edit_seen = True
        roles.append(role)
    return roles, warnings


def describe(n: int, role: str, size: tuple[int, int], original: tuple[int, int], note: str | None) -> str:
    """'Image 2 (reference, 832x1216 [resized from 1664x2432])' — the same words on asset_brief and in refs."""
    w, h = size
    text = f"{ROLES[role]}, {w}x{h}"
    if size != original:
        text += f" [resized from {original[0]}x{original[1]}]"
    if note:
        text += f" [{note}]"
    return f"Image {n} ({text})"


def _load(path: str, st: dict) -> tuple[torch.Tensor, tuple[int, int], str | None]:
    """One picture -> ([1, H, W, 3] float 0-1, original (w, h), note or None)."""
    resolved = folder_paths.get_annotated_filepath(path)
    if not resolved or not os.path.isfile(resolved):
        raise RuntimeError(f"[Luna-ImageLoader] Image not found: {path}. Upload it on the node, or put it in ComfyUI's "
                           f"input folder and use the file name, one per line.")
    img = node_helpers.pillow(Image.open, resolved)
    img, note = to_rgb(ImageOps.exif_transpose(img), st.get("pad_color") or "#000000")
    original = img.size
    img = apply_state(img, st)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...], original, note


def empty_refs(brief: str) -> dict:
    """A well-formed LUNA_H3_REFS with nothing in it (the Asset Loader's keys, plus `roles`)."""
    return {"images": [], "videos": [], "video_audios": [], "audios": [], "width": 0, "height": 0, "brief": brief,
            "song": None, "roles": [],
            "items": {"images": [], "videos": [], "video_audios": [], "audios": [], "song": "", "extra": []}}


class LunaImageLoader:
    DESCRIPTION = (
        "Load up to 9 pictures for image work: an edit, style transfer, or a new image from reference pictures. Card N is "
        "Image N everywhere downstream; in the prompt write @image1, @image2 … Each card has a role (Edit base, Reference, "
        "Style, Composition) and its own size settings; unset, a picture goes out as loaded. The − N + stepper sets how "
        "many pictures are ACTIVE: cards above N stay in the node, off, and are not sent. No output is padded: every "
        "picture keeps its own shape."
    )
    CATEGORY = "Luna-Media-Tools"
    FUNCTION = "load"

    RETURN_TYPES = ("INT", "INT", "STRING", "LUNA_H3_REFS") + ("IMAGE",) * MAX_IMAGES
    RETURN_NAMES = ("width", "height", "asset_brief", "refs") + tuple(f"image_{i + 1}" for i in range(MAX_IMAGES))
    OUTPUT_TOOLTIPS = (
        "The edit base's width after its resize, else Image 1's; 0 with no active picture. For Luna-ImageResolution's "
        "image input or Luna-LLMPrompt / Luna-LLMPromptAPI width.",
        "The edit base's height after its resize, else Image 1's; 0 with no active picture. See width.",
        "One line naming each active picture: 'Image 1 (edit base, 1920x1080) · Image 2 (reference, 832x1216)'. For "
        "Luna-LLMPrompt / Luna-LLMPromptAPI asset_brief, so the writer knows which picture is which.",
        "Every active picture at its own size, with its number and role, on one wire. For Luna-LLMPrompt / "
        "Luna-LLMPromptAPI refs, or Luna-H3-RefsOut.",
    ) + tuple(
        f"Image {i + 1} at its own size (after its card's resize), for an encoder's image input "
        f"(MiniMaxH3ImageConditioning, TextEncodeQwenImage21 image{i + 1}). Empty when fewer than {i + 1} pictures are active."
        for i in range(MAX_IMAGES)
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One image file name per line, from ComfyUI's input folder. The cards write it. Line 1 is "
                               "Image 1, line 2 is Image 2, and so on.",
                }),
                "image_count": ("INT", {
                    "default": 1, "min": 0, "max": MAX_IMAGES,
                    "tooltip": "How many pictures are ACTIVE, set with the − N + stepper. Cards 1…N are sent and show "
                               "image_1…image_N; a card above N stays loaded, greyed and off: not in refs, asset_brief or "
                               "any output. A socket you have wired is never removed.",
                }),
            },
            "optional": {
                "asset_state": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Per-picture resize, as JSON, written by the card panel (the Asset Loader's schema). "
                               "{\"all\": {...}} for every picture; {\"items\": [{...}, ...]} per picture, items win. "
                               "Keys: mode (off|max_mp|longest_side|scale_factor), max_mp, longest_side, scale_factor, "
                               "ratio (\"16:9\"), ratio_action (crop|pad), crop_anchor, pad_color, snap, allow_upscale.",
                }),
                "media_state": ("STRING", {
                    "default": "",
                    "tooltip": "The cards' roles, as JSON, written by the card UI: {\"pictures\": [{\"role\": \"edit\"}, "
                               "{\"role\": \"ref\"}]}. Roles: edit (edit base, at most one), ref, style, composition.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image_paths, image_count, asset_state="", media_state="", **kwargs):
        # The file contents, not just the names: a changed picture re-uploaded under the same name must re-run.
        m = hashlib.sha256()
        m.update(f"{image_count}|{asset_state}|{media_state}".encode("utf-8"))
        for p in split_paths(image_paths)[:max(0, int(image_count))]:
            resolved = folder_paths.get_annotated_filepath(p)
            m.update(p.encode("utf-8"))
            if resolved and os.path.isfile(resolved):
                with open(resolved, "rb") as fh:
                    m.update(fh.read())
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image_paths, image_count, **kwargs):
        paths = split_paths(image_paths)
        if len(paths) > MAX_IMAGES:
            return f"Luna-ImageLoader takes at most {MAX_IMAGES} pictures, but {len(paths)} lines are listed."
        return True

    def load(self, image_paths, image_count, asset_state="", media_state="", **kwargs):
        paths = split_paths(image_paths)[:max(0, min(MAX_IMAGES, int(image_count)))]
        if not paths:
            print("[Luna-ImageLoader] No active picture: empty outputs.")
            return (0, 0, "No pictures.", empty_refs("No pictures.")) + (None,) * MAX_IMAGES

        states = parse_state(asset_state, len(paths))
        roles, warnings = parse_roles(media_state, len(paths))
        loaded = [_load(p, st) for p, st in zip(paths, states)]
        frames = [f for f, _, _ in loaded]
        sizes = [(int(f.shape[2]), int(f.shape[1])) for f in frames]  # (w, h) as delivered
        lines = [describe(i + 1, roles[i], sizes[i], loaded[i][1], loaded[i][2]) for i in range(len(paths))]

        main = roles.index("edit") if "edit" in roles else 0
        brief = " · ".join(lines)
        if warnings:
            brief += " " + " ".join(warnings)
        refs = empty_refs(brief)
        refs.update({"images": frames, "width": sizes[main][0], "height": sizes[main][1],
                     "roles": [ROLES[r] for r in roles]})
        # Each picture's description without "Image N", as the Asset Loader keeps them: Luna-H3-RefsOut renumbers.
        refs["items"]["images"] = [line[line.index("(") + 1:-1] for line in lines]
        print(f"[Luna-ImageLoader] {brief}")
        outs = tuple(frames) + (None,) * (MAX_IMAGES - len(frames))
        return (sizes[main][0], sizes[main][1], brief, refs) + outs


NODE_CLASS_MAPPINGS = {"Luna-ImageLoader": LunaImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-ImageLoader": "Luna-ImageLoader"}
