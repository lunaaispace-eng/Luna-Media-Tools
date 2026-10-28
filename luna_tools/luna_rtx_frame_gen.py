# Derived from ComfyUI-DaSiWa-Nodes (https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes),
# commit ab364c5, GPL-3.0 (see THIRD_PARTY_NOTICES.md). Source file: nodes/nodes_rtx_frame_generation.py
# Changes: renamed to Luna-RTXFrameGen (category Luna-Media-Tools); helper imports point at luna_batch_output / luna_logging;
# DESCRIPTION rewritten; input 'description' keys became 'tooltip'; log prefix and messages use the node's own name;
# install help describes this pack; registered as a V1 node. Behaviour, inputs and memory handling as upstream.
"""Pair-based NVIDIA RTX interpolation; no SDK/GPU work at module import."""
import contextlib
import inspect
import math
import os

import torch

from .luna_rtx_upscaler import (
    INSTALL_HELP as _INSTALL_HELP,
    _allocate_output_tensor as _luna_fg_allocate_output,
    _close_vfx_effect as _luna_fg_close_effect,
    _luna_rtx_pack_rgb10a2 as _luna_fg_pack,
    _luna_rtx_unpack_rgb10a2 as _luna_fg_unpack,
    _model_management as _luna_fg_model_management,
)
from .luna_logging import log_luna as _luna_fg_log

NODE_NAME = "Luna-RTXFrameGen"


def _luna_fg_sdk():
    """Lazy symbol/signature capability check, without effect construction."""
    try:
        import nvvfx
    except ImportError as exc:
        raise RuntimeError(f"{NODE_NAME} needs nvidia-vfx with VideoFrameGeneration support. " + _INSTALL_HELP) from exc
    effect = getattr(nvvfx, "VideoFrameGeneration", None)
    if effect is None or not callable(getattr(effect, "run_at_timestep", None)):
        raise RuntimeError("The installed nvidia-vfx lacks VideoFrameGeneration. " + _INSTALL_HELP)
    mode = getattr(effect, "Mode", None)
    encoding = getattr(nvvfx, "ImageEncoding", None)
    if mode is None or encoding is None:
        raise RuntimeError(f"{NODE_NAME}: the SDK mode/image encoding API is unavailable.")
    packed = ("image_encoding" in inspect.signature(effect).parameters
              and hasattr(encoding, "RGB10A2"))
    return effect, mode, encoding, packed


def _luna_fg_interrupt():
    if _luna_fg_model_management is not None:
        _luna_fg_model_management.throw_exception_if_processing_interrupted()


@contextlib.contextmanager
def _luna_fg_stream(device):
    """Use blocking default-stream SDK calls, handing off the caller explicitly.

    No VSR single-input runner is used. Pair interpolation has its own stream
    validation. Device synchronization conservatively covers native SDK work.
    """
    torch.cuda.synchronize(device)
    with torch.cuda.stream(torch.cuda.default_stream(device)):
        try:
            yield
        finally:
            torch.cuda.synchronize(device)


def _luna_fg_prepare(image, device, packed):
    frame = image[..., :3].permute(2, 0, 1).contiguous().to(device=device, dtype=torch.float32)
    return _luna_fg_pack(frame) if packed else frame


def _luna_fg_copy_pair_output(result, row, device, packed):
    """Copy borrowed SDK output before another run or close can reuse it."""
    borrowed = torch.from_dlpack(result.image)
    rgb = _luna_fg_unpack(borrowed) if packed else borrowed
    row.copy_(rgb.permute(1, 2, 0).clamp(0, 1))
    torch.cuda.synchronize(device)
    if row.device.type == "cuda" and row.device != device:
        torch.cuda.synchronize(row.device)


class LunaRTXFrameGen:
    DESCRIPTION = (
        "Luna-RTXFrameGen: adds in-between frames to an image batch with NVIDIA RTX frame generation "
        "(interpolation between each pair of neighbouring frames). RGB only; alpha is discarded.\n"
        "frame rate multiplier: inserts M-1 frames per pair, so N frames become (N-1)*M+1; every source frame and both "
        "endpoints are kept. Encode at source FPS * M to keep the speed. specific timestep: inserts one frame per pair "
        "at the given position, 2*N-1 frames; only 0.5 is evenly spaced.\n"
        "mode picks the SDK quality tier (LOW, MEDIUM, HIGH); shot-change detection avoids blending across cuts. The whole "
        "output is one IMAGE batch kept in VRAM, then RAM, then (only with use_mmap) a temp file. A single frame passes "
        "through untouched. Needs an NVIDIA RTX GPU and the nvidia-vfx package in ComfyUI's Python.\n"
        "No FPS, timestamps or audio are passed through."
    )
    OUTPUT_TOOLTIPS = ("The IMAGE batch with the generated frames inserted; set the encoder FPS to source FPS * multiplier.",)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "luna_rtx_generate_frames"
    CATEGORY = "Luna-Media-Tools"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            packed = _luna_fg_sdk()[3]
        except Exception:
            packed = False
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Ordered IMAGE batch: still images or decoded video frames, not a VIDEO object. RGB only; alpha discarded. No FPS, timestamps or audio are passed through."}),
            "generation_type": (["frame rate multiplier", "specific timestep"], {"tooltip": "frame rate multiplier: evenly insert M-1 frames per pair, output (N-1)*M+1; encode at source FPS*M. specific timestep: insert one frame per pair, output 2*N-1; only 0.5 is uniformly spaced (encode at source FPS*2). Both preserve endpoints; container duration can differ by a final frame interval."}),
            "multiplier": ("INT", {"default": 2, "min": 2, "max": 16, "tooltip": "Active only for frame rate multiplier: M subdivisions per source interval, M-1 generated frames per pair. Output (N-1)*M+1; set encoder FPS to source FPS*M. Ignored for specific timestep."}),
            "timestep": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01, "tooltip": "Active only for specific timestep: fractional position between previous (0) and next (1) source frame. 0.5 is the midpoint; other values produce unequal temporal spacing that a constant-FPS IMAGE encoder cannot reproduce exactly. Ignored for multiplier mode."}),
            "mode": (["LOW", "MEDIUM", "HIGH"], {"default": "HIGH", "tooltip": "SDK interpolation quality preset: LOW favors speed, MEDIUM is the middle tier, HIGH favors quality. These are frame-generation modes, not VSR levels; cost and visual quality depend on SDK, GPU and content, with no fixed speed or quality guarantee."}),
            "automatic_shot_change_detection": ("BOOLEAN", {"default": True, "tooltip": "On: ask the SDK to detect cuts automatically between source frames. Off: disable automatic detection; the separate shot_change flag still applies. Cut handling is SDK-defined, not a per-frame cut list."}),
            "shot_change": ("BOOLEAN", {"default": False, "tooltip": "On: explicitly mark every submitted source pair as a cut (useful for a single pair). Off: do not force cuts; automatic detection can still apply. This is not a list of cut indices."}),
            "device_id": ("INT", {"default": 0, "min": 0, "max": 8, "tooltip": "CUDA device index for NVIDIA interpolation (0 is the first GPU). Must identify an available supported NVIDIA device; does not choose encoder or audio device."}),
        }, "optional": {
            "image_encoding": (["8-bit RGB", "10-bit RGB (RGB10A2)"] if packed else ["8-bit RGB"],
                               {"default": "8-bit RGB", "tooltip": "8-bit RGB: default SDK encoding, retains source floating output dtype. 10-bit RGB (RGB10A2): packed 10-bit SDR SDK processing, float32 output; not HDR or smaller IMAGE storage. Original frames are copied directly."}),
            "auto_unload_models": ("BOOLEAN", {"default": True, "tooltip": "On: reclaim targeted GPU memory and, under RAM output pressure, native execution caches and model pins; re-check before allocation. Off: no reclamation. Never resets the running executor or globally offloads GPU models into RAM. Native cache eviction requires ComfyUI RAM-pressure cache support; full batches still need storage."}),
            "use_mmap": ("BOOLEAN", {"default": False, "tooltip": "Off: never allocate disk output. On: allow a temporary mmap as the last output tier if budgeted VRAM/RAM cannot fit the complete IMAGE batch. Requires disk space for the full output; not streaming input/output."}),
        }}

    def luna_rtx_generate_frames(self, images, generation_type="frame rate multiplier", multiplier=2,
                                  timestep=0.5, mode="HIGH", automatic_shot_change_detection=True,
                                  shot_change=False, device_id=0, image_encoding="8-bit RGB",
                                  auto_unload_models=True, use_mmap=False):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError(f"{NODE_NAME} requires a BHWC RGB/RGBA IMAGE tensor.")
        if not images.is_floating_point() or any(dim < 1 for dim in images.shape):
            raise ValueError(f"{NODE_NAME} requires a nonempty floating-point IMAGE batch.")
        if generation_type not in ("frame rate multiplier", "specific timestep"):
            raise ValueError(f"{NODE_NAME}: unsupported generation_type.")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or not 2 <= multiplier <= 16:
            raise ValueError("multiplier must be an integer from 2 to 16.")
        if not math.isfinite(timestep) or not 0.01 <= timestep <= 0.99:
            raise ValueError("timestep must be finite and between 0.01 and 0.99.")
        if mode not in ("LOW", "MEDIUM", "HIGH"):
            raise ValueError(f"{NODE_NAME}: unsupported mode.")
        if image_encoding not in ("8-bit RGB", "10-bit RGB (RGB10A2)"):
            raise ValueError(f"{NODE_NAME}: unsupported image_encoding.")
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a nonnegative CUDA index.")
        packed = image_encoding != "8-bit RGB"
        # Single-frame passthrough needs neither SDK nor GPU. RGB-only policy.
        if images.shape[0] == 1:
            return (images[..., :3].float() if packed else images[..., :3],)
        effect_class, modes, encodings, supports_packed = _luna_fg_sdk()
        if packed and not supports_packed:
            raise RuntimeError(f"The installed SDK does not support RGB10A2 in {NODE_NAME}.")
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise RuntimeError(f"{NODE_NAME} requires an available NVIDIA CUDA device.")
        device = torch.device(f"cuda:{device_id}")
        count, height, width, _ = images.shape
        step = multiplier if generation_type == "frame rate multiplier" else 2
        shape = ((count - 1) * step + 1, height, width, 3)
        if generation_type == "specific timestep" and timestep != 0.5:
            _luna_fg_log(NODE_NAME, "Warning: specific timestep produces nonuniform temporal spacing; IMAGE has no timestamps.")
        # Two staged input frames, conversions and SDK output, plus openly
        # heuristic opaque FG workspace headroom. This is not an OOM guarantee.
        reserve = 1024 ** 3 + height * width * (96 if packed else 72)
        out = previous = current = row = result = effect = None
        mmap_path = None
        try:
            out, mmap_path = _luna_fg_allocate_output(shape, torch.float32 if packed else images.dtype,
                images.device, allow_mmap=use_mmap, auto_unload_models=auto_unload_models,
                working_reserve_bytes=reserve, compute_device=device)
            with torch.cuda.device(device), torch.inference_mode(), _luna_fg_stream(device):
                # Inputs/output may be on a different CUDA device from compute.
                if images.device.type == "cuda" and images.device != device:
                    torch.cuda.synchronize(images.device)
                effect = effect_class(mode=getattr(modes, mode),
                    automatic_shot_change_detection_enabled=automatic_shot_change_detection,
                    device=device_id, **({"image_encoding": encodings.RGB10A2 if packed else encodings.RGB8}
                        if "image_encoding" in inspect.signature(effect_class).parameters else {}))
                try:
                    effect.input_width, effect.input_height = width, height
                    effect.frame_multiplier = step
                    effect.load()
                    out[0].copy_(images[0, ..., :3])
                    previous = _luna_fg_prepare(images[0], device, packed)
                    output_index = 1
                    for pair in range(1, count):
                        _luna_fg_interrupt()
                        current = _luna_fg_prepare(images[pair], device, packed)
                        torch.cuda.synchronize(device)
                        for frame_index in range(1, step):
                            _luna_fg_interrupt()
                            if generation_type == "frame rate multiplier":
                                result = effect.run(previous, current, frame_index=frame_index,
                                                    shot_change=shot_change, non_blocking=False, stream_ptr=0)
                            else:
                                result = effect.run_at_timestep(previous, current, timestep,
                                                    shot_change=shot_change, non_blocking=False, stream_ptr=0)
                            row = out[output_index]
                            _luna_fg_copy_pair_output(result, row, device, packed)
                            row = result = None
                            output_index += 1
                        out[output_index].copy_(images[pair, ..., :3])
                        output_index += 1
                        previous, current = current, None
                    if out.device.type == "cuda" and out.device != device:
                        torch.cuda.synchronize(out.device)
                finally:
                    torch.cuda.synchronize(device)
                    if effect is not None:
                        _luna_fg_close_effect(effect)
                        effect = None
            return (out,)
        except BaseException:
            # Tracebacks must not retain the full failed mmap batch or pair views.
            out = previous = current = row = result = None
            if mmap_path:
                try:
                    os.unlink(mmap_path)
                except FileNotFoundError:
                    pass
            raise


NODE_CLASS_MAPPINGS = {"Luna-RTXFrameGen": LunaRTXFrameGen}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-RTXFrameGen": "Luna-RTXFrameGen"}
