# Derived from the Deno RTX VFX nodes, comfyui-deno-custom-nodes
# (https://github.com/Deno2026/comfyui-deno-custom-nodes), commit 0cd75f4, GPL-3.0
# (see THIRD_PARTY_NOTICES.md). Source file: deno_rtx_vfx_video_finisher.py
# Changes: renamed to Luna-RTXVideoFinisher (category Luna-Media-Tools); imports point at the luna_*
# modules; DESCRIPTION rewritten for this pack; input tooltips and OUTPUT_TOOLTIPS added;
# registered as a V1 node. The processing code is as upstream.
"""Luna-RTXVideoFinisher.

Runs an optional same-size NVIDIA RTX pass (Denoise / Deblur) and an optional
upscale pass (VSR / High Bitrate) frame-by-frame inside a single node so the
full intermediate Denoise/Deblur batch is never materialized. In Low RAM mode
the final tensor is preallocated on CPU and written per-frame instead of
collecting a Python list and calling torch.stack(), which removes the final
memory peak that hurts low-spec video / LTX workflows.

This node intentionally reuses the runtime/import/error helpers from
luna_rtx_upscale so Luna-RTXUpscale stays untouched and backward compatible.
"""

import contextlib

import torch

from .luna_resolution_common import COMMON_RATIOS, RESIZE_METHODS, validate_combo_choice, validate_active_ratio_preset
from .luna_rtx_upscale import (
    RTX_VFX_DEFAULT_DIVISIBLE_BY,
    RTX_VFX_DIVISIBLE_BY_VALUES,
    RTX_VFX_INSTALL_STEPS,
    _create_vfx_effect,
    _fit_frame_to_target_aspect,
    _import_vfx,
    _quality_attr,
    _run_vfx_frame,
    _safe_cuda_device_index,
    _safe_divisible_by,
    _target_size,
    _vfx_output_width,
)

FIRST_PASS_CHOICES = ["Off", "Denoise", "Deblur"]
UPSCALE_PASS_CHOICES = ["Off", "VSR", "High Bitrate"]
QUALITY_CHOICES = ["Low", "Medium", "High", "Ultra"]
FINISHER_RESIZE_TYPES = ["Keep Ratio", "Manual", "Preset Ratio", "Scale", "Same Size"]


@contextlib.contextmanager
def _maybe_vfx_effect(VideoSuperRes, enabled, mode, device_index, out_width, out_height):
    """Create an RTX VFX effect only when the pass is enabled.

    Yields None when the pass is "Off" so the caller can keep one flat loop.
    """
    if not enabled:
        yield None
        return
    quality = getattr(VideoSuperRes.QualityLevel, _quality_attr(mode))
    with _create_vfx_effect(VideoSuperRes, quality, device_index, mode) as effect:
        effect.output_width = _vfx_output_width(out_width)
        effect.output_height = int(out_height)
        effect.load()
        yield effect


class LunaRTXVideoFinisher:
    DESCRIPTION = (
        "Luna-RTXVideoFinisher: cleans up and upscales video frames with NVIDIA RTX VFX in two passes, "
        "frame by frame, inside one node.\n\n"
        "Pass 1 (first_pass) is an optional same-size Denoise or Deblur. Pass 2 (upscale_pass) is an optional "
        "VSR or High Bitrate upscale to the size set by resize_type. Setting both passes to Off returns the "
        "frames unchanged. The finished batch is written frame by frame into one preallocated tensor, so "
        "the full in-between batch is never held in memory.\n\n"
        "Needs an NVIDIA RTX GPU and the NVIDIA RTX VFX package (`nvvfx`, from nvidia-vfx 0.2.0.0 or newer). "
        "The pack loads without it; the node reports what is missing when it runs.\n\n"
        + RTX_VFX_INSTALL_STEPS
    )
    OUTPUT_TOOLTIPS = ("The finished frame batch.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input video frame batch to finish with one or two RTX VFX passes."}),
                "first_pass": (FIRST_PASS_CHOICES, {"default": "Deblur", "tooltip": "Optional same-size first pass, Denoise or Deblur. Off skips it."}),
                "first_quality": (QUALITY_CHOICES, {"default": "Ultra", "tooltip": "Quality/strength of the first pass."}),
                "upscale_pass": (UPSCALE_PASS_CHOICES, {"default": "High Bitrate", "tooltip": "Optional second pass: VSR or High Bitrate upscale. Off skips it."}),
                "upscale_quality": (QUALITY_CHOICES, {"default": "High", "tooltip": "Quality of the upscale pass."}),
                "resize_type": (FINISHER_RESIZE_TYPES, {"default": "Keep Ratio", "tooltip": "How the final output size is chosen."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Scale multiplier used by Scale resizing."}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.01, "max": 64.0, "step": 0.01, "tooltip": "Target total output size in megapixels for Keep Ratio and Preset Ratio resizing."}),
                "width": ("INT", {"default": 3840, "min": 64, "max": 8192, "step": 8, "tooltip": "Manual output width in pixels."}),
                "height": ("INT", {"default": 2160, "min": 64, "max": 8192, "step": 8, "tooltip": "Manual output height in pixels."}),
                "divisible_by": (RTX_VFX_DIVISIBLE_BY_VALUES, {"default": RTX_VFX_DEFAULT_DIVISIBLE_BY, "tooltip": "Round the output size to a multiple the VFX runtime handles well."}),
                "ratio_preset": (COMMON_RATIOS, {"default": "16:9", "tooltip": "Target aspect ratio used in Preset Ratio resizing."}),
                "resize_method": (RESIZE_METHODS, {"default": "Center Crop (Fill)", "tooltip": "Crop-fill or fit with black bars, when the output ratio differs from the input."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_finisher"
    CATEGORY = "Luna-Media-Tools"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        first_pass=None,
        first_quality=None,
        upscale_pass=None,
        upscale_quality=None,
        resize_type=None,
        divisible_by=None,
        ratio_preset=None,
        resize_method=None,
    ):
        first_pass_result = validate_combo_choice("first_pass", first_pass, FIRST_PASS_CHOICES)
        if first_pass_result is not True:
            return first_pass_result
        if str(first_pass or "").strip() in {"Denoise", "Deblur"}:
            first_quality_result = validate_combo_choice("first_quality", first_quality, QUALITY_CHOICES)
            if first_quality_result is not True:
                return first_quality_result

        upscale_pass_result = validate_combo_choice("upscale_pass", upscale_pass, UPSCALE_PASS_CHOICES)
        if upscale_pass_result is not True:
            return upscale_pass_result
        if str(upscale_pass or "").strip() not in {"VSR", "High Bitrate"}:
            return True
        for result in (
            validate_combo_choice("upscale_quality", upscale_quality, QUALITY_CHOICES),
            validate_combo_choice("resize_type", resize_type, FINISHER_RESIZE_TYPES),
            validate_combo_choice("divisible_by", divisible_by, RTX_VFX_DIVISIBLE_BY_VALUES),
            validate_combo_choice("resize_method", resize_method, RESIZE_METHODS),
        ):
            if result is not True:
                return result
        return validate_active_ratio_preset(resize_type, ratio_preset)

    def apply_finisher(
        self,
        images,
        first_pass: str,
        first_quality: str,
        upscale_pass: str,
        upscale_quality: str,
        resize_type: str,
        scale: float,
        megapixels: float,
        width: int,
        height: int,
        divisible_by: int,
        ratio_preset: str,
        resize_method: str,
    ):
        # A fully disabled finisher is a true pass-through. Return before any
        # CUDA / RTX VFX probing so CPU-only workflows can leave the node in
        # place without changing tensor identity, dtype, device, or channels.
        if first_pass == "Off" and upscale_pass == "Off":
            return (images,)

        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA RTX VFX requires CUDA. This ComfyUI Python does not currently see CUDA.")

        if images.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape [batch, height, width, channels], got {tuple(images.shape)}"
            )

        batch, source_height, source_width, channels = images.shape
        if channels < 3:
            raise ValueError("NVIDIA RTX VFX requires RGB images with 3 channels.")

        first_enabled = first_pass in {"Denoise", "Deblur"}
        upscale_enabled = upscale_pass in {"VSR", "High Bitrate"}
        first_mode = f"{first_pass} {first_quality}" if first_enabled else None
        upscale_mode = f"{upscale_pass} {upscale_quality}" if upscale_enabled else None

        alignment = _safe_divisible_by(divisible_by)
        if upscale_enabled:
            target_width, target_height = _target_size(
                source_width=int(source_width),
                source_height=int(source_height),
                mode=upscale_mode,
                resize_type=resize_type,
                scale=scale,
                megapixels=megapixels,
                width=width,
                height=height,
                divisible_by=alignment,
                ratio_preset=ratio_preset,
            )
        else:
            target_width, target_height = int(source_width), int(source_height)

        VideoSuperRes = _import_vfx()
        device_index = _safe_cuda_device_index(0)
        cuda_device = torch.device(f"cuda:{device_index}")

        # Full-quality float32 output. RAM is handled at the workflow level
        # (VHS Meta Batch chunks the whole graph); this node stays lossless.
        # Preallocate + write per frame (no Python list / torch.stack peak).
        out_device = images.device
        out_dtype = images.dtype
        out = torch.empty(
            (int(batch), int(target_height), int(target_width), 3),
            device=out_device,
            dtype=out_dtype,
        )

        with torch.inference_mode():
            with _maybe_vfx_effect(
                VideoSuperRes, first_enabled, first_mode, device_index,
                int(source_width), int(source_height),
            ) as first_effect:
                with _maybe_vfx_effect(
                    VideoSuperRes, upscale_enabled, upscale_mode, device_index,
                    int(target_width), int(target_height),
                ) as second_effect:
                    for index in range(int(batch)):
                        frame = (
                            images[index, :, :, :3]
                            .to(device=cuda_device, dtype=torch.float32)
                            .permute(2, 0, 1)
                            .contiguous()
                        )

                        if first_effect is not None:
                            frame = _run_vfx_frame(
                                first_effect, frame, int(source_width), int(source_height), same_size=True,
                            )

                        if second_effect is not None:
                            frame = _fit_frame_to_target_aspect(
                                frame, int(target_width), int(target_height), resize_method
                            )
                            frame = _run_vfx_frame(
                                second_effect, frame, int(target_width), int(target_height),
                            )

                        enhanced = frame.permute(1, 2, 0).contiguous()
                        # clamp in the source (float32) precision first, then a
                        # single cast for storage. Clamping after a CPU float16
                        # cast is the slow path (CPUs have no native fp16 math),
                        # so this keeps the Low-RAM (fp16) saving without the
                        # extra per-frame speed penalty.
                        out[index].copy_(
                            enhanced.clamp(0.0, 1.0).to(device=out_device, dtype=out_dtype)
                        )

                        del frame, enhanced

        return (out,)


NODE_CLASS_MAPPINGS = {"Luna-RTXVideoFinisher": LunaRTXVideoFinisher}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-RTXVideoFinisher": "Luna-RTXVideoFinisher"}
