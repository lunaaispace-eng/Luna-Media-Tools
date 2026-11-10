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

With `windows_ahead` above 0, a second thread decodes the next windows on the GPU while FFmpeg takes the current
one, instead of the two waiting for each other. The frames and their order are the same.
"""

import os
import queue
import threading
import time

import torch

import comfy.model_management
import comfy.utils

from .video_nodes import (
    _AUDIO_BITRATE_OPTIONS,
    _AUDIO_CODEC_OPTIONS,
    _CODEC_OPTIONS,
    _MIME_TYPES,
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


def window_bytes_on_device(pixels, bit_depth):
    """[frames, H, W, 3] pixels -> FFmpeg's raw rgb24 / rgb48le bytes as a CPU uint8 tensor.

    The same arithmetic as Luna-SaveVideo's `_frame_bytes`, done where the pixels are (the GPU), so only the 8-bit
    (or 16-bit) result crosses to the CPU instead of float32. Works in place on `pixels`.
    """
    frames = pixels[..., :3].float().clamp_(0, 1)
    if bit_depth == 10:
        values = frames.mul_(1023).round_().to(torch.int32).mul_(64)
        # little-endian uint16, as numpy writes it on this machine
        packed = torch.stack(((values & 255).to(torch.uint8), (values >> 8).to(torch.uint8)), dim=-1)
    else:
        packed = frames.mul_(255).round_().to(torch.uint8)
    return packed.contiguous().cpu()


def decode_windows_to_bytes(vae, samples, images, detail, bit_depth):
    """Yield (frame count, CPU uint8 bytes tensor) per window, in order, each frame once.

    The Flash decoder runs exactly as VSR Decode runs it (same latent, frames, dtype, detail), but writes into a
    buffer on the GPU instead of ComfyUI's CPU float32 output; the 8/10-bit conversion happens there too.
    """
    decoder = vae.first_stage_model
    keyframes, windows = relay_windows(images.shape[0])
    out_dtype = vae.vae_output_dtype()
    out_size = torch.empty((), dtype=out_dtype).element_size()
    for index, (first, last) in enumerate(windows):
        comfy.model_management.throw_exception_if_processing_interrupted()
        span = samples[:, :, index:index + 2] if len(keyframes) > 1 else samples[:, :, :1]
        lq_images = images[first:last + 1]
        shape = decoder.decode_output_shape(span.shape, lq_images)
        # the decoder's own need, plus the output buffer and its conversion living on the GPU
        vae.prepare_decode(span.shape, vae.memory_used_decode(span.shape, vae.vae_dtype) + 2 * torch.Size(shape).numel() * out_size)
        with comfy.model_management.cuda_device_context(vae.device):
            out = torch.empty(shape, device=vae.device, dtype=out_dtype)
            decoder.decode(span.to(device=vae.device, dtype=vae.vae_dtype), lq_images=lq_images, output_buffer=out,
                           detail=detail)
            pixels = out.movedim(1, -1).reshape(-1, *out.shape[-2:], out.shape[1])
            if index:
                pixels = pixels[1:]
            count = pixels.shape[0]
            packed = window_bytes_on_device(pixels, bit_depth)
            del out, pixels
        yield count, packed


def byte_chunks(packed, max_chunk_bytes=64 * 1024 * 1024):
    """A window's bytes tensor as FFmpeg pipe chunks, without copying."""
    frame_bytes = packed[0].numel() if packed.shape[0] else 1
    step = max(1, max_chunk_bytes // frame_bytes)
    array = packed.numpy()
    for start in range(0, packed.shape[0], step):
        yield memoryview(array[start:start + step]).cast("B")


_DONE = object()


class _Failed:
    def __init__(self, error):
        self.error = error


def run_ahead(produce, depth):
    """Yield what `produce()` yields, made in a thread that keeps up to `depth` items ready.

    The thread decodes while the caller writes. A failure in the thread (an interrupt too) is raised in the caller.
    Closing this generator stops the thread and waits for it, so two decodes never run at once.
    """
    items = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def worker():
        produced = produce()
        try:
            # Inference mode is per thread; ComfyUI sets it only on the thread that runs the node.
            with torch.inference_mode():
                for item in produced:
                    if not put(item):
                        return
            put(_DONE)
        except BaseException as error:
            put(_Failed(error))
        finally:
            produced.close()

    thread = threading.Thread(target=worker, name="luna-relayvsr-decode", daemon=True)
    thread.start()
    try:
        while True:
            item = items.get()
            if item is _DONE:
                return
            if isinstance(item, _Failed):
                raise item.error
            yield item
    finally:
        stop.set()
        thread.join()


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
                "windows_ahead": ("INT", {"default": 2, "min": 0, "max": 8,
                    "tooltip": "How many 15-frame windows the GPU may decode ahead while FFmpeg writes. 0 = the GPU "
                               "and FFmpeg take turns (slower). Each window held costs about 0.4 GB of RAM at 4K "
                               "8-bit (0.8 GB at 10-bit). Same frames at any value."}),
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
        scheduler="simple", windows_ahead=2, audio=None, prompt=None, extra_pnginfo=None,
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
        open_pipelines = []
        windows_ahead = max(0, int(windows_ahead))

        def timed_windows(timing):
            # One window at a time: decoded and converted to bytes on the GPU, then only the bytes come to the CPU.
            decoded = decode_windows_to_bytes(vae, samples, images, detail, selected_bit_depth)
            try:
                while True:
                    started = time.perf_counter()
                    item = next(decoded, None)
                    timing["decode"] += time.perf_counter() - started
                    if item is None:
                        return
                    yield item
            finally:
                decoded.close()

        def frame_chunks():
            passes.append(1)
            if len(passes) > 1:
                _log("That encoder failed; decoding again from the start for the next one.")
            for pipeline in open_pipelines:
                pipeline.close()
            open_pipelines.clear()
            timing = {"decode": 0.0, "wait": 0.0, "write": 0.0}
            started = time.perf_counter()
            done = 0
            if windows_ahead:
                pipeline = run_ahead(lambda: timed_windows(timing), windows_ahead)
            else:
                pipeline = timed_windows(timing)
            open_pipelines.append(pipeline)
            while True:
                waited = time.perf_counter()
                item = next(pipeline, None)
                timing["wait"] += time.perf_counter() - waited
                if item is None:
                    break
                count, packed = item
                writing = time.perf_counter()
                yield from byte_chunks(packed)
                timing["write"] += time.perf_counter() - writing
                del packed, item
                done += count
                progress.update_absolute(done, frame_count)
            total = time.perf_counter() - started
            mode = f"{windows_ahead} ahead" if windows_ahead else "GPU and FFmpeg in turn"
            _log(f"Decode + write {total:.1f} s for {len(windows)} windows ({mode}): GPU decode {timing['decode']:.1f} s, "
                 f"FFmpeg write {timing['write']:.1f} s, waited for the GPU {timing['wait']:.1f} s.")

        try:
            saved = encode_video_file(
                frame_chunks, width, height, frame_count, frame_rate, codec, container, selected_bit_depth, quality,
                save_metadata, filename_prefix, output_path, save_output, audio, crop_to_audio, audio_codec,
                audio_bitrate, prompt, extra_pnginfo,
            )
        finally:
            for pipeline in open_pipelines:
                pipeline.close()
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
