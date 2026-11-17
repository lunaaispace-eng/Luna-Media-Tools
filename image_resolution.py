"""Luna-ImageResolution — the size of the generated image, and an empty latent in the target model's own shape.

- The image counterpart of Luna-H3-Canvas (Luna-MinimaxH3 `h3_canvas.py`), without its video settings.
- Face: `category`, `preset`, `portrait`, and a live readout. Gear panel: `size_mode`, `megapixels`, `longest_side`, `snap`,
  `batch`; custom sizes come from one JSON file in the ComfyUI user folder, `user/luna/resolutions.json`, shared by every node.
- Wired `image`: the frame follows that picture's shape (as on Luna-H3-Canvas): native mode takes the listed size whose ratio
  is nearest, turned to the picture's orientation; megapixels and longest side use its exact aspect.
- Model sizes, grids, ranges and latent shapes: Luna-Research
  `packs-and-nodes\\2026-11-16-image-model-resolutions\\RESOLUTIONS.md` (DOCUMENTED there per model; the rule set is PROPOSED).
- The latent is built in the chosen model's own shape (channels, downscale, dimensions): a plain 4-channel latent is adapted
  only by core samplers and comes out twice as large on /16 models through several third-party samplers (MEASURED there).
- `compute()` is the one source of the maths; the frontend asks it through the `/luna_media/resolution` route.
"""

from __future__ import annotations

import json
import math
import os

import torch

MP = 1024 * 1024    # 1 MP = 1024 x 1024, as core ResolutionSelector and MOSS-Image count it
SNAPS = (8, 16, 32, 64)
SIZE_MODES = ("native", "megapixels", "longest_side")

# --- presets -------------------------------------------------------------------------------------
# (label, width, height[, fixed]) landscape or square; `fixed` = one orientation only (portrait does not turn it).

RATIOS = ["1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9", "32:9"]

SOCIAL = [
    ("Instagram post 1:1", 1080, 1080), ("Instagram portrait 4:5", 1080, 1350, True),
    ("Instagram / TikTok story 9:16", 1080, 1920, True), ("Instagram landscape 1.91:1", 1080, 566),
    ("Facebook post", 1200, 630), ("Facebook cover", 851, 315, True),
    ("X post 16:9", 1600, 900), ("X header", 1500, 500, True),
    ("YouTube thumbnail", 1280, 720), ("YouTube banner", 2560, 1440, True),
    ("Pinterest pin 2:3", 1000, 1500, True), ("LinkedIn post", 1200, 627), ("LinkedIn banner", 1584, 396, True),
]
PRINT = [  # 300 dpi
    ("A3 300 dpi", 4961, 3508), ("A4 300 dpi", 3508, 2480), ("A5 300 dpi", 2480, 1748),
    ("Letter 300 dpi", 3300, 2550), ("4x6 in 300 dpi", 1800, 1200), ("5x7 in 300 dpi", 2100, 1500),
    ("8x10 in 300 dpi", 3000, 2400),
]
CINEMA = [
    ("DCI 2K flat 1.85:1", 1998, 1080), ("DCI 2K scope 2.39:1", 2048, 858), ("DCI 2K full", 2048, 1080),
    ("DCI 4K flat 1.85:1", 3996, 2160), ("DCI 4K scope 2.39:1", 4096, 1716), ("Academy 1.37:1", 1480, 1080),
    ("Super 35 1.33:1", 1440, 1080),
]
DISPLAY = [
    ("720p", 1280, 720), ("1080p", 1920, 1080), ("1440p", 2560, 1440), ("4K UHD", 3840, 2160),
    ("5K", 5120, 2880), ("Ultrawide 1440", 3440, 1440), ("Ultrawide 1080", 2560, 1080), ("WUXGA 16:10", 1920, 1200),
]


def _unnamed(pairs):
    # Lists without names: the preset shows only its size.
    return [("", w, h) for w, h in pairs]


# One entry per image model. snap = the model's pixel grid; mp = (low, high) binary MP outside which a warning shows;
# latent = (channels, downscale, 3-D); native = the maker's list (empty = sizes from megapixels); default_mp for the
# megapixels mode and for a model without a list.
MODELS = {
    "MiniMax H3 Image": {
        "snap": 32, "mp": (3.8, 19.1), "latent": (24, 16, True), "default_mp": 16.0,
        "native": [("1:1", 4096, 4096), ("5:4", 4576, 3648), ("4:3", 4736, 3552), ("3:2", 5024, 3360),
                   ("16:9", 5472, 3072), ("21:9", 6272, 2688)],
        "note": "maker table about 16 MP (5:4 from the same 4096² area rule); core node tuned for 2048² area",
    },
    "Qwen-Image 2.1": {
        "snap": 32, "mp": None, "latent": (64, 16, False), "default_mp": 4.0,
        "native": [("1:1", 2048, 2048), ("4:3", 2400, 1792), ("3:2", 2528, 1696), ("16:9", 2752, 1536)],
        "note": "maker table; sizes for other ratios not published",
    },
    "MOSS-Image": {
        "snap": 16, "mp": (0.25, 4.0), "latent": (128, 16, False), "default_mp": 1.0,
        "native": [(f"{r} {mp} MP", w, h) for mp, rows in (
            ("1", [("1:1", 1024, 1024), ("4:3", 1184, 880), ("3:2", 1248, 832), ("16:9", 1360, 768), ("21:9", 1568, 672)]),
            ("1.5", [("1:1", 1248, 1248), ("4:3", 1456, 1088), ("3:2", 1536, 1024), ("16:9", 1680, 944), ("21:9", 1920, 816)]),
            ("2", [("1:1", 1456, 1456), ("4:3", 1680, 1248), ("3:2", 1776, 1184), ("16:9", 1936, 1088), ("21:9", 2208, 944)]),
        ) for r, w, h in rows],
        "note": "maker fit_size at 1, 1.5 and 2 MP (recommended)",
    },
    "Fibo 2.0": {
        "snap": 16, "mp": (0.9, 1.1), "latent": (128, 16, False), "default_mp": 1.0,
        "native": [("1:1", 1024, 1024), ("5:4", 1120, 896), ("4:3", 1152, 864), ("3:2", 1248, 832),
                   ("16:9", 1424, 800), ("2:1", 1440, 720), ("21:9", 1568, 672)],
        "note": "trained at about 1 MP (core trained_size)",
    },
    "Krea 2": {
        "snap": 16, "mp": (0.95, 4.0), "latent": (16, 8, True), "default_mp": 1.0,
        "native": [], "note": "no maker ratio list; Turbo 1k-2k, Raw up to 1k",
    },
    "Z-Image Turbo": {
        "snap": 16, "mp": (0.25, 4.0), "latent": (16, 8, False), "default_mp": 1.0,
        "native": [(f"{r} base {b}", w, h) for b, rows in (
            (1024, [("1:1", 1024, 1024), ("9:7", 1152, 896), ("4:3", 1152, 864), ("3:2", 1248, 832), ("16:9", 1280, 720), ("21:9", 1344, 576)]),
            (1280, [("1:1", 1280, 1280), ("9:7", 1440, 1120), ("4:3", 1472, 1104), ("3:2", 1536, 1024), ("16:9", 1536, 864), ("21:9", 1680, 720)]),
            (1536, [("1:1", 1536, 1536), ("9:7", 1728, 1344), ("4:3", 1728, 1296), ("3:2", 1872, 1248), ("16:9", 2048, 1152), ("21:9", 2016, 864)]),
        ) for r, w, h in rows],
        "note": "maker Space list",
    },
    "Flux.1": {
        "snap": 16, "mp": (0.1, 2.0), "latent": (16, 8, False), "default_mp": 1.0,
        "native": _unnamed([(1024, 1024), (1104, 944), (1184, 880), (1248, 832), (1328, 800), (1392, 752),
                               (1456, 720), (1504, 688), (1568, 672)]),
        "note": "Kontext preferred list (core); 0.1-2 MP",
    },
    "Flux.2": {
        "snap": 16, "mp": (0.004, 2.0), "latent": (128, 16, False), "default_mp": 1.0,
        "native": [], "note": "any ratio, multiples of 16; at most 4 MP, 2 MP or less recommended",
    },
    "SDXL": {
        "snap": 64, "mp": (0.9, 1.1), "latent": (4, 8, False), "default_mp": 1.0,
        "native": _unnamed([(1024, 1024), (1088, 960), (1152, 896), (1216, 832), (1280, 768), (1344, 768),
                               (1408, 704), (1472, 704), (1536, 640), (1600, 640), (1664, 576), (1728, 576)]),
        "note": "training buckets, about 1 MP on /64",
    },
}

GENERAL = {"Aspect ratio": [], "Social media": SOCIAL, "Print": PRINT, "Cinema": CINEMA, "Display": DISPLAY}
CATEGORIES = list(GENERAL) + list(MODELS) + ["Custom"]
CUSTOM_FILE = ("luna", "resolutions.json")


# --- custom sizes ----------------------------------------------------------------------------------

def custom_path() -> str:
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.getcwd(), "user")
    return os.path.join(base, *CUSTOM_FILE)


def read_custom(path: str | None = None) -> list[tuple]:
    """The shared file: a list of {"label", "width", "height"} or {"label", "value": "WxH"}. Missing or bad = empty."""
    try:
        with open(path or custom_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        w, h = item.get("width"), item.get("height")
        if (not w or not h) and isinstance(item.get("value"), str) and "x" in item["value"].lower():
            try:
                w, h = (int(x) for x in item["value"].lower().split("x", 1))
            except ValueError:
                continue
        try:
            w, h = int(w), int(h)
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            out.append((str(item.get("label") or f"{w}x{h}"), w, h, True))
    return out


def presets(category: str, custom: list | None = None) -> list[tuple]:
    """(label, w, h, fixed) for a category, as the preset list shows them."""
    if category == "Aspect ratio":
        return [(r, 0, 0, False) for r in RATIOS]
    if category in GENERAL:
        return [(p[0], p[1], p[2], len(p) > 3 and p[3]) for p in GENERAL[category]]
    if category in MODELS:
        native = MODELS[category]["native"]
        if native:
            return [(f"{label} ({w}x{h})" if label else f"{w}x{h}", w, h, False) for label, w, h in native]
        return [(r, 0, 0, False) for r in RATIOS]
    if category == "Custom":
        return [(f"{label} ({w}x{h})", w, h, True) for label, w, h, _ in (read_custom() if custom is None else custom)]
    return []


def all_preset_labels() -> list[str]:
    seen, out = set(), []
    for c in CATEGORIES:
        for p in presets(c):
            if p[0] not in seen:
                seen.add(p[0])
                out.append(p[0])
    return out


# --- maths ------------------------------------------------------------------------------------------

_COMMON = [(1, 1), (5, 4), (4, 3), (3, 2), (16, 10), (16, 9), (2, 1), (21, 9), (32, 9), (4, 5), (3, 4), (2, 3),
           (10, 16), (9, 16), (1, 2), (9, 21)]


def classify(w: int, h: int) -> tuple[str, bool]:
    """The nearest common ratio and whether it is exact within 2% (Resolution Master's classifier rule)."""
    r = w / h
    a, b = min(_COMMON, key=lambda c: abs(c[0] / c[1] - r))
    return f"{a}:{b}", abs(a / b - r) / r <= 0.02


def _ratio_value(text: str) -> float:
    a, b = text.split(" ")[0].split(":")
    return float(a) / float(b)


def _snap(v: float, step: int) -> int:
    return max(step, int(round(v / step)) * step)


def _by_area(ratio: float, mp: float, step: int) -> tuple[int, int]:
    area = mp * MP
    return _snap(math.sqrt(area * ratio), step), _snap(math.sqrt(area / ratio), step)


# Edit sizing of the core encoders, mirrored (ComfyUI comfy_extras). A wired picture on these models in native mode gets the
# output size the encoder computes for image_1 at its defaults, so the latent and width/height match the reference.

def _trained_size(aspect_ratio: float) -> tuple[int, int]:
    # Fibo 2.0, nodes_fibo2.trained_size: of the multiples of 16 within 10% of 1 MP, the least cropping, then closest to 1 MP.
    area = 1024 * 1024
    sizes = [(w, h) for w in range(64, 2 * area // 64, 16)
             for h in range(max(64, 16 * math.ceil(0.9 * area / w / 16)), 16 * math.floor(1.1 * area / w / 16) + 1, 16)]
    return min(sizes, key=lambda s: (round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9),
                                     abs(s[0] * s[1] - area), -s[0]))


def _qwen21_edit(w: int, h: int) -> tuple[int, int, str]:
    # TextEncodeQwenImage21 at resolution 1024 (its default): about 1024² at the picture's ratio, sides on /32.
    ratio = w / h
    return (max(32, round(math.sqrt(1024 * 1024 * ratio) / 32) * 32),
            max(32, round(math.sqrt(1024 * 1024 / ratio) / 32) * 32), "as TextEncodeQwenImage21 at resolution 1024")


def _fibo2_edit(w: int, h: int) -> tuple[int, int, str]:
    tw, th = _trained_size(w / h)
    return tw, th, "as TextEncodeFibo2 (trained size)"


def _h3_image_edit(w: int, h: int) -> tuple[int, int, str]:
    # MiniMaxH3ImageConditioning with width/height 0: image_1's VAE size, short edge held in 1024-2048, sides on /32.
    scale = min(max(min(w, h), 1024), 2048) / min(w, h)
    return (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32),
            "as MiniMaxH3ImageConditioning (image_1's reference size)")


EDIT_SIZE = {"Qwen-Image 2.1": _qwen21_edit, "Fibo 2.0": _fibo2_edit, "MiniMax H3 Image": _h3_image_edit}


def compute(category: str, preset: str, portrait: bool = False, size_mode: str = "native", megapixels: float = 1.0,
            longest_side: int = 1024, snap: int = 16, image_size: tuple[int, int] | None = None,
            custom: list | None = None) -> dict:
    """Everything the node outputs except the latent: width, height, the three readout lines and the latent spec."""
    warnings: list[str] = []
    model = MODELS.get(category)
    step = max(int(snap) if int(snap) in SNAPS else 16, model["snap"] if model else 8)
    if model and int(snap) < model["snap"]:
        warnings.append(f"snap raised to {model['snap']}, {category}'s pixel grid")
    plist = presets(category, custom)
    pick = next((p for p in plist if p[0] == preset), None)
    if pick is None and plist:
        pick = plist[0]
        if preset:
            warnings.append(f"'{preset}' is not a {category} preset; {pick[0]} used")
    if pick is None:
        pick = ("1:1", 0, 0, False)
        if category == "Custom":
            warnings.append(f"no custom sizes in {custom_path()}")
    label, pw, ph, fixed = pick
    mp_default = model["default_mp"] if model else 1.0

    if image_size and image_size[0] > 0 and image_size[1] > 0:
        iw, ih = image_size
        ratio = iw / ih
        source = f"from image {iw}x{ih}"
        if size_mode == "native" and category in EDIT_SIZE:
            # An edit model: the size its core encoder gives image_1, or the edit shifts against its reference.
            w, h, how = EDIT_SIZE[category](iw, ih)
            source += f" → {how}"
        elif size_mode == "native":
            if plist and pw and ph:
                # The listed size whose ratio is nearest to the picture's, turned to its orientation, then snapped.
                want = max(ratio, 1 / ratio)
                label, pw, ph, fixed = min((p for p in plist if p[1]), key=lambda p: abs(max(p[1], p[2]) / min(p[1], p[2]) - want))
                w, h = (max(pw, ph), min(pw, ph)) if ratio >= 1 else (min(pw, ph), max(pw, ph))
                w, h = _snap(w, step), _snap(h, step)
                source += f" → {category} {label}"
            else:
                w, h = _snap(iw, step), _snap(ih, step)
        elif size_mode == "megapixels":
            w, h = _by_area(ratio, megapixels, step)
        else:
            s = longest_side / max(iw, ih)
            w, h = _snap(iw * s, step), _snap(ih * s, step)
    else:
        if pw and ph:
            ratio = pw / ph
        else:
            ratio = _ratio_value(label)
        turn = portrait and not fixed and ratio != 1
        if turn:
            ratio = 1 / ratio
        if size_mode == "native" and pw and ph:
            w, h = (ph, pw) if turn else (pw, ph)
            w, h = _snap(w, step), _snap(h, step)
            source = f"{category} {label}" + (" (portrait)" if turn else "")
        elif size_mode == "longest_side":
            long_edge = int(longest_side)
            w, h = (long_edge, long_edge / ratio) if ratio >= 1 else (long_edge * ratio, long_edge)
            w, h = _snap(w, step), _snap(h, step)
            source = f"{category} {label}, longest side {long_edge}" + (" (portrait)" if turn else "")
        else:
            mp = megapixels if size_mode == "megapixels" else mp_default
            w, h = _by_area(ratio, mp, step)
            source = f"{category} {label} at {mp:g} MP" + (" (portrait)" if turn else "")
        if pw and ph and size_mode == "native" and (w, h) != ((ph, pw) if turn else (pw, ph)):
            warnings.append(f"snapped from {pw}x{ph} to /{step}")

    mp_out = w * h / MP
    if model and model["mp"] and not (model["mp"][0] <= mp_out <= model["mp"][1]):
        warnings.append(f"{mp_out:.2f} MP is outside {category}'s {model['mp'][0]:g}-{model['mp'][1]:g} MP")
    name, exact = classify(w, h)
    line1 = f"{w}×{h}  {name if exact else '≈ ' + name}  {mp_out:.2f} MP"
    line2 = source + (f" · {model['note']}" if model and size_mode == "native" else "")
    latent = model["latent"] if model else (4, 8, False)
    return {"width": int(w), "height": int(h), "lines": [line1, line2] + (["; ".join(warnings)] if warnings else []),
            "warnings": warnings, "latent": latent, "snap": step}


def empty_latent(width: int, height: int, batch: int, spec: tuple[int, int, bool]) -> dict:
    """A zero latent in the model's own shape. The downscale key stays, so core samplers see a matching ratio."""
    ch, ds, three_d = spec
    t = torch.zeros([max(1, int(batch)), ch, height // ds, width // ds])
    if three_d:
        t = t.unsqueeze(2)
    return {"samples": t, "downscale_ratio_spacial": ds}


# --- node -------------------------------------------------------------------------------------------

class LunaImageResolution:
    DESCRIPTION = (
        "The size of the image to generate, from a category (aspect ratio, social media, print, cinema, display, one per "
        "image model, or your own sizes) and a preset; `portrait` turns it on its side. The ⚙ panel sets how the size is "
        "found: the preset's own size, a megapixel target, or the longest side, and the pixel grid it snaps to. A model "
        "category uses that model's native sizes and grid and makes the latent in its own shape. Wire a picture into "
        "`image` (for example the edit base) and the frame follows its shape. Custom sizes come from "
        "ComfyUI/user/luna/resolutions.json: [{\"label\": \"Poster\", \"width\": 2400, \"height\": 3000}]."
    )
    CATEGORY = "Luna-Media-Tools"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT", "LATENT", "STRING")
    RETURN_NAMES = ("width", "height", "latent", "info")
    OUTPUT_TOOLTIPS = (
        "The width to generate at, on the grid. For Luna-LLMPrompt / Luna-LLMPromptAPI width, and an encoder's width "
        "(MiniMaxH3ImageConditioning, TextEncodeQwenImage21).",
        "The height to generate at, on the grid. See width.",
        "An empty latent at width x height and batch, in the chosen model's own shape (4 channels /8 for the general "
        "categories). For a sampler's latent_image.",
        "The readout as text: size, ratio, MP, where it came from, and any warning. For Luna-SaveImage or a note.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (CATEGORIES, {"default": "Aspect ratio",
                                          "tooltip": "Which list the preset comes from. A model entry uses that model's "
                                                     "native sizes, pixel grid and latent shape."}),
                "preset": (all_preset_labels(), {"default": "1:1",
                                                 "tooltip": "The size or ratio, landscape. The list follows the category."}),
                "portrait": ("BOOLEAN", {"default": False,
                                         "tooltip": "Turn the size on its side. Sizes with one orientation (stories, "
                                                    "banners, custom) stay as listed."}),
                "size_mode": (list(SIZE_MODES), {"default": "native",
                                                 "tooltip": "native: the preset's own size (a ratio uses the model's or 1 "
                                                            "MP). megapixels: same ratio at `megapixels`. longest_side: "
                                                            "same ratio, longest side = `longest_side`."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 64.0, "step": 0.05,
                                         "tooltip": "Target area for size_mode megapixels; 1 MP = 1024 x 1024."}),
                "longest_side": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8,
                                         "tooltip": "Longest side for size_mode longest_side."}),
                "snap": ([str(s) for s in SNAPS], {"default": "16",
                                                   "tooltip": "Both sides go to the nearest multiple of this. A model "
                                                              "category raises it to the model's own grid."}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 64,
                                  "tooltip": "How many latents in the batch."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "A picture whose shape the frame follows, for example the edit base from "
                                               "Luna-ImageLoader image_1. In native mode on MiniMax H3 Image, Qwen-Image "
                                               "2.1 or Fibo 2.0 the size is the one that model's core encoder gives "
                                               "image_1 at its defaults (TextEncodeQwenImage21 resolution 1024), so the "
                                               "edit does not shift."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, preset, **kwargs):
        # Listed here so ComfyUI skips its own combo check: custom sizes come from a file that can change after load.
        return True

    def run(self, category, preset, portrait, size_mode, megapixels, longest_side, snap, batch, image=None):
        size = (int(image.shape[2]), int(image.shape[1])) if image is not None else None
        r = compute(category, preset, bool(portrait), size_mode, float(megapixels), int(longest_side), int(snap), size)
        info = "\n".join(r["lines"])
        print(f"[Luna-ImageResolution] {info}")
        latent = empty_latent(r["width"], r["height"], batch, r["latent"])
        # The picture's size goes to the frontend too, so the readout can follow later settings without another run.
        ui = {"luna_resolution": [info], "luna_image_size": [list(size)] if size else []}
        return {"ui": ui, "result": (r["width"], r["height"], latent, info)}


def _register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        return

    @routes.get("/luna_media/resolution/presets")
    async def _presets(request):
        custom = read_custom()
        return web.json_response({"categories": {c: [p[0] for p in presets(c, custom)] for c in CATEGORIES},
                                  "custom_file": custom_path()})

    @routes.post("/luna_media/resolution")
    async def _compute(request):
        q = await request.json()
        try:
            size = q.get("image_size")
            r = compute(str(q.get("category", "Aspect ratio")), str(q.get("preset", "")), bool(q.get("portrait")),
                        str(q.get("size_mode", "native")), float(q.get("megapixels", 1.0)),
                        int(q.get("longest_side", 1024)), int(q.get("snap", 16)),
                        tuple(size) if isinstance(size, list) and len(size) == 2 else None)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({k: r[k] for k in ("width", "height", "lines", "warnings", "snap")})


_register_routes()

NODE_CLASS_MAPPINGS = {"Luna-ImageResolution": LunaImageResolution}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-ImageResolution": "Luna-ImageResolution"}
