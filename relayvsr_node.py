"""Luna-RelayVSR — RelayVSR video upscale straight to a video file.

ComfyUI's native RelayVSR graph (RelayVSR Conditioning -> KSampler at 1 step, cfg 1 -> VSR Decode -> Save Video)
holds the whole upscaled video in RAM as float32 before it is saved: 15 s of 4K is about 36-45 GB. This node runs
the same computation and keeps only one keyframe window of 4K frames at a time:

1. The keyframe pass: every 15th frame through the RelayVSR model, exactly as RelayVSR Conditioning plus a
   KSampler (1 step, cfg 1, denoise 1, sampler and scheduler on the node: euler / simple by default) do it.
2. The decode: the Flash decoder already rebuilds the frames one window (keyframe to keyframe) at a time and starts
   each window fresh from its two keyframe latents. Here each window is decoded on its own through the VAE's normal
   decode, and its frames go to FFmpeg at once. The pixels are the ones VSR Decode would make.
3. The file: Luna-SaveVideo's encoder (`video_nodes.encode_video_file`), with its codecs, audio and metadata.
"""

import os

import torch

import comfy.model_management
import comfy.utils

from .video_nodes import (
    _AUDIO_BITRATE_OPTIONS,
    _AUDIO_CODEC_OPTIONS,
    _CODEC_OPTIONS,
    _MIME_TYPES,
    _iter_frame_byte_chunks,
    _selected_bit_depth,
    encode_video_file,
)

NODE_ID = "Luna-RelayVSR"
_CONTAINERS = ["Auto", "WebM", "MKV", "MP4"]


def _sampler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _scheduler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


def _log(message):
    print(f"[{NODE_ID}] {message}")


def _check_inputs(images, model, vae):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3 or images.shape[0] < 1:
        raise ValueError("images must be an IMAGE batch [frames, height, width, RGB] with at least one frame.")
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not hasattr(diffusion_model, "positive_conditioning") or not hasattr(diffusion_model, "latent_stride"):
        raise ValueError("model is not a RelayVSR model. Load relayvsr_x2 or relayvsr_x4 with Load Diffusion Model.")
    decoder = getattr(vae, "first_stage_model", None)
    if not hasattr(decoder, "prepare_condition") or not hasattr(decoder, "upscale_factor"):
        raise ValueError("vae is not a RelayVSR Flash decoder. Load relayvsr_flash_vae_x2 or _x4 with Load VAE.")
    return diffusion_model, decoder


def relay_windows(frame_count):
    """Keyframe indices and the (first, last) frame span of each decode window, as the Flash decoder walks them."""
    from comfy.ldm.wan.model_vsr import relay_keyframes

    keyframes = relay_keyframes(frame_count)
    windows = list(zip(keyframes, keyframes[1:])) or [(keyframes[0], keyframes[0])]
    return keyframes, windows


def sample_keyframes(model, images, seed, sampler_name="euler", scheduler="simple"):
    """The latent RelayVSR Conditioning + KSampler (1 step, cfg 1, denoise 1, the given sampler and scheduler) make."""
    import nodes as comfy_nodes
    from comfy.ldm.wan.model_vsr import relay_latent_size

    diffusion_model = model.model.diffusion_model
    keyframes, _ = relay_windows(images.shape[0])
    positive = [[diffusion_model.positive_conditioning.unsqueeze(0).clone(), {"lq_images": images[keyframes]}]]
    latent_size = relay_latent_size(images.shape[1], images.shape[2], diffusion_model.latent_stride)
    latent = torch.zeros(
        [1, model.model.latent_format.latent_channels, len(keyframes), *latent_size],
        device=comfy.model_management.intermediate_device(),
    )
    return comfy_nodes.common_ksampler(
        model, seed, 1, 1.0, sampler_name, scheduler, positive, positive, {"samples": latent}, denoise=1.0,
    )[0]["samples"]


def decode_windows(vae, samples, images, detail):
    """Yield the upscaled frames one window at a time, in order, each frame once.

    A window is decoded with only its two keyframe latents and its own low-quality frames, which is what the
    Flash decoder's loop does inside one call: the attention history restarts at every window. The keyframe two
    windows share is output by the earlier one and dropped from the later one, as in that loop.
    """
    keyframes, windows = relay_windows(images.shape[0])
    for index, (first, last) in enumerate(windows):
        comfy.model_management.throw_exception_if_processing_interrupted()
        span = samples[:, :, index:index + 2] if len(keyframes) > 1 else samples[:, :, :1]
        pixels = vae.decode(span, vae_options={"lq_images": images[first:last + 1], "detail": detail})
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
        yield pixels if index == 0 else pixels[1:]


class LunaRelayVSR:
    DESCRIPTION = (
        "RelayVSR video upscale in one node, written straight to a video file. Same models and same result as the "
        "native RelayVSR Conditioning -> KSampler -> VSR Decode graph, but the upscaled frames go to the file one "
        "15-frame window at a time instead of being held in RAM: a 15 s clip to 4K needs about the input video's "
        "memory, not 40+ GB. The x2 model and x2 decoder take 1080p to 4K; x4 takes 540p to 4K."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The low-quality video at its own size, from Luna-LoadVideo.video_frames "
                                                "or any IMAGE batch. Do not resize it first: the model sets the factor."}),
                "model": ("MODEL", {"tooltip": "The RelayVSR model from Load Diffusion Model (UNETLoader), "
                                               "relayvsr_x2 or relayvsr_x4."}),
                "vae": ("VAE", {"tooltip": "The RelayVSR Flash decoder from Load VAE, the same factor as the model "
                                           "(relayvsr_flash_vae_x2 with relayvsr_x2)."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01,
                    "tooltip": "Playback speed of the file. Wire Luna-LoadVideo.fps here so it matches the source."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                    "tooltip": "Noise seed of the one sampling step, as the KSampler's seed in the native graph."}),
                "detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How much of the detail the decoder adds on top of a plain bicubic upscale is kept. "
                               "1 = full model output, 0 = bicubic only. Same as VSR Decode's detail."}),
                "codec": (_CODEC_OPTIONS, {"default": "Auto",
                    "tooltip": "As in Luna-SaveVideo. Auto tries AV1, then VP9, then H.264 and keeps the first this "
                               "machine encodes. H.265 is smaller at 4K; H.264 plays everywhere."}),
                "container": (_CONTAINERS, {"default": "Auto",
                    "tooltip": "The file type. Auto picks one that fits the codec. MP4 is the safest to share."}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto",
                    "tooltip": "Colour precision. Auto: 8-bit with codec Auto; with a chosen codec it is judged from "
                               "the source frames. 10-bit needs AV1, H.265 or VP9."}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51,
                    "tooltip": "Lower = better picture, bigger file. 18-23 looks clean; 20 is the default."}),
                "filename_prefix": ("STRING", {"default": "relayvsr_%date:hhmmss%",
                    "tooltip": "Filename, numbered: relayvsr_120455_00001.mp4. Date tokens as in Luna-SaveVideo."}),
                "output_path": ("STRING", {"default": "", "placeholder": "empty = ComfyUI output folder",
                    "tooltip": "Empty = ComfyUI output folder. Relative = a subfolder of it. Absolute = exactly there. "
                               "The node only previews files inside the output folder."}),
                "save_output": ("BOOLEAN", {"default": True,
                    "tooltip": "Off = write to ComfyUI's temp folder, deleted on the next restart."}),
                "save_metadata": ("BOOLEAN", {"default": True,
                    "tooltip": "Embed the workflow in the file, so it can be dragged back into ComfyUI."}),
                "crop_to_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "With audio wired: on = the file ends when the audio ends."}),
                "audio_codec": (_AUDIO_CODEC_OPTIONS, {"default": "Auto",
                    "tooltip": "Auto uses Opus for WebM and AAC for MKV / MP4."}),
                "audio_bitrate": (_AUDIO_BITRATE_OPTIONS, {"default": "192k",
                    "tooltip": "Audio quality. 192k is clean for music and speech."}),
                # Appended last, so saved workflows keep their widget positions.
                "sampler_name": (_sampler_names(), {"default": "euler",
                    "tooltip": "Sampler of the one keyframe step, as the KSampler's sampler_name in the native graph "
                               "(euler, or er_sde as in Peter's relayvsr_x2_video KJ)."}),
                "scheduler": (_scheduler_names(), {"default": "simple",
                    "tooltip": "Scheduler of the one keyframe step, as the KSampler's scheduler in the native graph."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The source sound, from Luna-LoadVideo.audio. Copied into the file."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "VIDEO")
    RETURN_NAMES = ("filename", "video")
    OUTPUT_TOOLTIPS = (
        "Full path of the saved video file.",
        "The saved file as a native ComfyUI VIDEO, sound included, for any VIDEO input.",
    )
    FUNCTION = "upscale"
    OUTPUT_NODE = True
    CATEGORY = "Luna-Media-Tools"

    def upscale(
        self, images, model, vae, frame_rate, seed, detail, codec, container, bit_depth, quality, filename_prefix,
        output_path, save_output, save_metadata, crop_to_audio, audio_codec, audio_bitrate, sampler_name="euler",
        scheduler="simple", audio=None, prompt=None, extra_pnginfo=None,
    ):
        _, decoder = _check_inputs(images, model, vae)
        frame_count = images.shape[0]
        factor = decoder.upscale_factor
        height, width = images.shape[1] * factor, images.shape[2] * factor
        keyframes, windows = relay_windows(frame_count)
        _log(f"{frame_count} frames {images.shape[2]}x{images.shape[1]} -> {width}x{height} (x{factor}), "
             f"{len(keyframes)} keyframes, {len(windows)} windows.")

        samples = sample_keyframes(model, images, seed, sampler_name, scheduler)

        selected_bit_depth = _selected_bit_depth(codec, bit_depth, images)
        progress = comfy.utils.ProgressBar(frame_count)
        passes = []

        def frame_chunks():
            passes.append(1)
            if len(passes) > 1:
                _log("That encoder failed; decoding again from the start for the next one.")
            done = 0
            for pixels in decode_windows(vae, samples, images, detail):
                yield from _iter_frame_byte_chunks(pixels, selected_bit_depth, False)
                done += pixels.shape[0]
                progress.update_absolute(done, frame_count)
                del pixels

        saved = encode_video_file(
            frame_chunks, width, height, frame_count, frame_rate, codec, container, selected_bit_depth, quality,
            save_metadata, filename_prefix, output_path, save_output, audio, crop_to_audio, audio_codec,
            audio_bitrate, prompt, extra_pnginfo,
        )
        video_path = saved["path"]
        ui = {}
        if saved["subfolder"] is not None:
            # The same player data Luna-SaveVideo gives (web/save_video_simple_preview.js plays both).
            video = {"filename": os.path.basename(video_path), "subfolder": saved["subfolder"], "type": saved["type"],
                     "format": _MIME_TYPES[saved["container"]], "codec": saved["codec"], "bit_depth": selected_bit_depth,
                     "container": saved["container"], "width": width, "height": height}
            ui = {"images": [video], "gifs": [{**video, "fps": frame_rate}]}
        from comfy_api.latest import InputImpl

        return {"ui": ui, "result": (video_path, InputImpl.VideoFromFile(video_path))}


NODE_CLASS_MAPPINGS = {NODE_ID: LunaRelayVSR}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: NODE_ID}
