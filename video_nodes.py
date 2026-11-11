"""Luna-SaveVideo — an IMAGE batch to a video file, with a prompt sidecar.

Derived from `nodes_enhanced_video_combine.py` in ComfyUI-DaSiWa-Nodes
(https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes), Apache License 2.0 (the version this was taken from;
the upstream repo is GPL-3.0 since its 08-04 licence change).
This pack is GPL-3.0; Apache-2.0 code may be combined into a GPL-3.0 work, and this notice stays.

Changes from the original, as Apache-2.0 section 4(b) requires them to be stated:
  - renamed the node (now Luna-SaveVideo) and moved it to the Luna-Media-Tools category
  - added an optional `prompt_text` input and a `save_prompt` toggle, writing the
    final prompt to a .txt beside the video, matching how Luna-SaveImage does it
    for stills
  - dropped the dependency on the DaSiWa logging helper
  - appended a native VIDEO output wrapping the saved video, including audio
  - added an optional `output_path` (empty / relative / absolute, as in Luna-SaveImage)
  - rewrote every input's help as a `tooltip` (the original's `description` keys
    are not shown by ComfyUI) and added OUTPUT_TOOLTIPS
  - moved the encoding into `encode_video_file`, which takes the frames as a stream
    of raw bytes and is shared with Luna-RelayVSR
  - added a `meta_batch` input (Luna-MetaBatch): one file written across queued runs,
    the encoder picked by a one-frame probe, the sound added when the file closes
  - added an optional `latent` input, saved beside the video as a safetensors .latent
"""

import asyncio
import datetime
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time

import torch
from PIL import Image

import folder_paths
try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # imported outside ComfyUI (tests)
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message


def log_luna(scope, message):
    print(f"[Luna-SaveVideo] {message}")


_CODEC_OPTIONS = ["Auto", "AV1", "VP9", "H.265 (HEVC)", "H.264"]
_AUTO_CODEC_CANDIDATES = ("AV1", "H.265 (HEVC)", "VP9", "H.264")
_CONTAINER_OPTIONS = ["Auto", "WebM", "MKV", "MP4", "Animated WebP", "Animated AVIF"]
_ENCODER_NAMES = {
    "H.264": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "libx264"),
    "H.265 (HEVC)": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_vaapi", "libx265"),
    "AV1": ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1"),
    "VP9": ("vp9_qsv", "vp9_vaapi", "libvpx-vp9"),
}
_CONTAINER_EXTENSIONS = {"WebM": ".webm", "MKV": ".mkv", "MP4": ".mp4"}
_ANIMATED_IMAGE_SETTINGS = {
    "Animated WebP": (".webp", "libwebp_anim", "image/webp"),
    "Animated AVIF": (".avif", "libaom-av1", "image/avif"),
}
_ANIMATED_AVIF_ENCODERS = ("av1_nvenc", "av1_qsv", "av1_amf", "av1_vaapi", "libsvtav1", "libaom-av1")
_AUDIO_CODEC_OPTIONS = ["Auto", "AAC", "Opus", "MP3"]
_AUDIO_ENCODERS = {"AAC": "aac", "Opus": "libopus", "MP3": "libmp3lame"}
_AUDIO_BITRATE_OPTIONS = ["64k", "96k", "128k", "160k", "192k", "256k", "320k"]
_MAX_RAW_FRAME_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_FFMPEG_STDERR_BYTES = 4 * 1024 * 1024


def _log(message):
    log_luna("Enhanced Video Combine", message)


def find_ffmpeg():
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _preview_source_path(filename, subfolder, output_type):
    """Resolve a ComfyUI asset without allowing traversal outside its asset root."""
    if not filename or filename != os.path.basename(filename) or ".." in subfolder:
        return None
    output_dir = folder_paths.get_directory_by_type(output_type)
    if not output_dir:
        return None
    root = os.path.abspath(output_dir)
    candidate = os.path.abspath(os.path.join(root, subfolder, filename))
    if os.path.commonpath((candidate, root)) != root or not os.path.isfile(candidate):
        return None
    return candidate


if PromptServer is not None:
    @PromptServer.instance.routes.get("/lunasave/video-preview")
    async def enhanced_video_preview(request):
        """Transcode unsupported outputs to fragmented H.264 MP4 without a sidecar file."""
        source = _preview_source_path(
            request.rel_url.query.get("filename", ""),
            request.rel_url.query.get("subfolder", ""),
            request.rel_url.query.get("type", "output"),
        )
        ffmpeg = find_ffmpeg()
        if source is None:
            return web.Response(status=404)
        if ffmpeg is None:
            return web.Response(status=503, text="FFmpeg is unavailable")
        process = await asyncio.create_subprocess_exec(
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", source,
            "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", "pipe:1", stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        response = web.StreamResponse(headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})
        await response.prepare(request)
        try:
            while chunk := await process.stdout.read(256 * 1024):
                await response.write(chunk)
        except (ConnectionResetError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
        finally:
            await process.wait()
        return response


def detect_bit_depth(images):
    if not torch.is_floating_point(images):
        return 10 if images.element_size() >= 2 else 8

    # Sample first, convert after: converting the whole batch to float32 first cost 108 GB for 1076 frames of
    # 4K fp16 (2026-11-09). A contiguous batch gives the same strided sample as before, as a view.
    values = images.detach()
    if values.numel() == 0:
        return 8
    if not values.is_contiguous():
        picks = torch.linspace(0, values.shape[0] - 1, min(values.shape[0], 8)).round().long().unique()
        values = values[picks.to(values.device)]
    values = values.reshape(-1)
    if values.numel() > 250_000:
        values = values[:: max(1, values.numel() // 250_000)]

    values = values.to(device="cpu", dtype=torch.float32).clamp(0, 1)
    error_8 = torch.mean(torch.abs(values * 255 - torch.round(values * 255)))
    error_10 = torch.mean(torch.abs(values * 1023 - torch.round(values * 1023)))
    return 10 if error_10 < error_8 * 0.8 else 8


def _container_candidates(codec, container):
    if container != "Auto":
        return (container,)
    if codec in {"AV1", "VP9"}:
        return ("WebM", "MKV", "MP4")
    return ("MP4", "MKV")


def _auto_container_candidates(codec, container):
    if container != "Auto":
        return _container_candidates(codec, container)
    return {"AV1": ("WebM",), "VP9": ("WebM",), "H.264": ("MP4",)}[codec]


def _codec_candidates(codec):
    # H.265 is intentionally explicit-only: unlike AV1/VP9/H.264 it is not a
    # broadly browser-compatible WebM/MP4 fallback.
    return ("AV1", "VP9", "H.264") if codec == "Auto" else (codec,)


# HDR save (Peter, 2026-11-11: "do it"). The frames arrive already HDR-coded, e.g. from RTX Video HDR ("HDR" = BT.2020
# with HLG, "HDR PQ" = BT.2020 with PQ). The file gets 10-bit, the BT.2020 conversion matrix and the HDR tags, so an HDR
# screen shows it as HDR. SDR writes exactly what it wrote before (no tags).
COLOR_OPTIONS = ["SDR", "HDR HLG (BT.2020)", "HDR PQ (BT.2020)"]
_HDR_TRANSFER = {"HDR HLG (BT.2020)": "arib-std-b67", "HDR PQ (BT.2020)": "smpte2084"}
COLOR_TOOLTIP = (
    "SDR: a normal video, as before. HDR HLG / HDR PQ: for frames that are already HDR, e.g. from RTX Video HDR "
    "(its output_colorspace HDR = HLG, HDR PQ = PQ; pick the same here). The file is written 10-bit H.265 (AV1 in WebM) "
    "with the BT.2020 HDR tags, so an HDR screen shows it as HDR. Codec and bit_depth are set for you.")


def is_hdr(color):
    return color in _HDR_TRANSFER


def _color_arguments(color):
    """FFmpeg output options for `color`: the RGB to YUV matrix and the tags. Empty for SDR."""
    transfer = _HDR_TRANSFER.get(color)
    if transfer is None:
        return []
    # The frames carry the tags (setparams) as well as the output options: FFmpeg takes the frames' own colour
    # properties over -color_* when the filter chain sets any (MEASURED 2026-11-11: matrix only, primaries/trc lost).
    return ["-vf", "scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le,"
                   f"setparams=range=tv:colorspace=bt2020nc:color_primaries=bt2020:color_trc={transfer}",
            "-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", transfer, "-color_range", "tv"]


def _hdr_codec(codec, container):
    """HDR needs 10-bit H.265 or AV1: AV1 stays, a WebM container takes AV1, anything else becomes H.265."""
    if codec == "AV1" or container == "WebM":
        return "AV1"
    return "H.265 (HEVC)"


def _selected_bit_depth(codec, bit_depth, images, color="SDR"):
    if is_hdr(color):
        return 10
    requested_bit_depth = {"8-bit": 8, "10-bit": 10}.get(bit_depth)
    if requested_bit_depth is not None:
        return requested_bit_depth
    # Auto uses AV1 first and must therefore avoid 10-bit browser decoder gaps.
    return 8 if codec == "Auto" else detect_bit_depth(images)


def _animated_image_settings(container):
    settings = _ANIMATED_IMAGE_SETTINGS.get(container)
    return settings[:2] if settings else None


def _animated_image_encoder_candidates(container):
    if container == "Animated AVIF":
        return _ANIMATED_AVIF_ENCODERS
    settings = _animated_image_settings(container)
    return (settings[1],) if settings else ()


def _available_encoders(ffmpeg):
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return set()
    return {line.split()[1] for line in result.stdout.splitlines() if line.startswith(" V")}


def _encoder_arguments(codec, encoder, bit_depth, cq, crf):
    pixel_format = "yuv420p10le" if bit_depth == 10 else "yuv420p"
    if encoder.endswith("_nvenc"):
        return ["-c:v", encoder, "-preset", "p5", "-cq", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_qsv"):
        return ["-c:v", encoder, "-global_quality", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_amf"):
        return ["-c:v", encoder, "-quality", "quality", "-qp_i", str(cq), "-qp_p", str(cq), "-pix_fmt", pixel_format]
    if encoder.endswith("_vaapi"):
        return ["-c:v", encoder, "-qp", str(cq), "-pix_fmt", pixel_format]
    if codec in {"H.264", "H.265 (HEVC)"}:
        return ["-c:v", encoder, "-crf", str(crf), "-preset", "medium", "-pix_fmt", pixel_format]
    if encoder == "libsvtav1":
        return ["-c:v", encoder, "-crf", str(crf), "-preset", "6", "-pix_fmt", pixel_format]
    if codec == "AV1":
        return ["-c:v", encoder, "-crf", str(crf), "-b:v", "0", "-cpu-used", "6", "-pix_fmt", pixel_format]
    return ["-c:v", encoder, "-crf", str(crf), "-b:v", "0", "-deadline", "good", "-pix_fmt", pixel_format]


def _frame_bytes(images, bit_depth):
    frames = images[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1)
    if bit_depth == 10:
        return torch.round(frames * 1023).to(torch.int32).mul_(64).to(torch.uint16).numpy().tobytes()
    return torch.round(frames * 255).to(torch.uint8).numpy().tobytes()


def _frames_per_chunk(images, bit_depth, max_chunk_bytes):
    bytes_per_frame = images.shape[1] * images.shape[2] * 3 * (2 if bit_depth == 10 else 1)
    return max(1, min(32, max_chunk_bytes // bytes_per_frame))


def _iter_frame_byte_chunks(images, bit_depth, pingpong, max_chunk_bytes=_MAX_RAW_FRAME_CHUNK_BYTES):
    frames_per_chunk = _frames_per_chunk(images, bit_depth, max_chunk_bytes)
    for start in range(0, len(images), frames_per_chunk):
        yield _frame_bytes(images[start:start + frames_per_chunk], bit_depth)
    if pingpong:
        for stop in range(len(images) - 1, 1, -frames_per_chunk):
            start = max(1, stop - frames_per_chunk)
            yield _frame_bytes(images[start:stop].flip(0), bit_depth)


def _encoded_frame_count(images, pingpong):
    return len(images) + (len(images) - 2 if pingpong and len(images) >= 3 else 0)


def _write_prompt_sidecar(video_path, prompt_text):
    """Write the final prompt to a .txt with the video's base name.

    Derived from the video path rather than counted separately, so the pair can
    never drift apart the way two independent filename counters would.
    """
    text = (prompt_text or "").strip()
    if not text:
        return None
    sidecar_path = os.path.splitext(video_path)[0] + ".txt"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return sidecar_path


def _save_frame_exports(images, output_path, save_first_frame, save_last_frame, pingpong=False):
    """Write selected source frames beside the encoded video without browser downloads."""
    if not (save_first_frame or save_last_frame):
        return []

    frame_stem = os.path.splitext(output_path)[0]
    exports = []
    last_frame = images[1] if pingpong and len(images) >= 3 else images[-1]
    for suffix, frame, enabled in (
        ("first", images[0], save_first_frame),
        ("last", last_frame, save_last_frame),
    ):
        if not enabled:
            continue
        path = f"{frame_stem}-{suffix}-frame.png"
        pixels = torch.round(frame[..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0, 1) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels, mode="RGB").save(path, "PNG")
        exports.append(path)
    return exports


def _pingpong_frames(images, pingpong):
    if not pingpong or len(images) < 3:
        return images
    return torch.cat((images, images[1:-1].flip(0)), dim=0)


def _format_filename_prefix(filename_prefix):
    """Expand ComfyUI-style date placeholders before requesting an output path."""
    now = datetime.datetime.now()

    def replace_date(match):
        fmt = match.group(1) or "yyyyMMdd_HHmmss"
        fmt = fmt.replace("yyyy", "%Y").replace("yy", "%y")
        fmt = fmt.replace("MM", "%m").replace("dd", "%d").replace("DD", "%d")
        fmt = fmt.replace("HH", "%H").replace("hh", "%H")
        fmt = fmt.replace("mm", "%M").replace("ss", "%S")
        return now.strftime(fmt)

    return re.sub(r"%date(?::([^%]+))?%", replace_date, filename_prefix)


def _subfolder_under(folder, root):
    """`folder` relative to `root` with forward slashes, or None when outside it."""
    folder, root = os.path.abspath(folder), os.path.abspath(root)
    try:
        if os.path.normcase(os.path.commonpath([folder, root])) != os.path.normcase(root):
            return None
    except ValueError:  # different drives on Windows
        return None
    relative = os.path.relpath(folder, root)
    return "" if relative == "." else relative.replace("\\", "/")


def _output_filename(filename, counter, extension):
    return f"{filename}_{counter:05}.{extension.lstrip('.')}"


def _metadata_file(prompt, extra_pnginfo):
    if prompt is None and not extra_pnginfo:
        return None

    metadata = {}
    if prompt is not None:
        metadata["prompt"] = prompt
    if extra_pnginfo:
        metadata.update(extra_pnginfo)

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ffmeta", delete=False, encoding="utf-8")
    try:
        handle.write(";FFMETADATA1\n")
        for key, value in metadata.items():
            escaped = json.dumps(value, separators=(",", ":"))
            escaped = escaped.replace("\\", "\\\\").replace(";", "\\;")
            escaped = escaped.replace("#", "\\#").replace("=", "\\=").replace("\n", "\\\n")
            handle.write(f"{key}={escaped}\n")
    finally:
        handle.close()
    return handle.name


def _audio_file(audio):
    if audio is None:
        return None, None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio must be a ComfyUI AUDIO value containing waveform and sample_rate.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("audio waveform must be a torch.Tensor.")
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError("audio waveform must have shape [batch, channels, samples].")

    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0 or waveform.shape[-1] == 0:
        raise ValueError("audio must have a positive sample rate and at least one sample.")
    channels = waveform.shape[1]
    interleaved = waveform.transpose(1, 2).reshape(-1, channels).clamp_(-1, 1).numpy()
    handle = tempfile.NamedTemporaryFile(suffix=".f32le", delete=False)
    try:
        handle.write(interleaved.tobytes())
    finally:
        handle.close()
    return (handle.name, sample_rate, channels), len(interleaved) / sample_rate


def _audio_encoder(audio_codec, container):
    if isinstance(audio_codec, bool):
        audio_codec = "Auto"
    if audio_codec == "Auto":
        return "libopus" if container == "WebM" else "aac"
    return _AUDIO_ENCODERS[audio_codec]


def _audio_encoder_candidates(audio_codec, container):
    requested = _audio_encoder(audio_codec, container)
    fallback = {
        "WebM": ("libopus",),
        "MKV": ("aac", "libopus", "libmp3lame", "pcm_s16le"),
        "MP4": ("aac", "libmp3lame"),
    }[container]
    return tuple(dict.fromkeys((requested, *fallback)))


def _run_ffmpeg(command, frame_chunks, progress_callback=None):
    """Run FFmpeg and translate its throttled progress stream into frame progress."""
    if isinstance(frame_chunks, bytes):
        if progress_callback is None:
            return subprocess.run(command, input=frame_chunks, capture_output=True, timeout=3600)
        frame_chunks = lambda: (frame_chunks,)
    if progress_callback is None:
        progress_callback = lambda _encoded_seconds: None

    process = subprocess.Popen(
        [*command[:-1], "-progress", "pipe:2", "-nostats", command[-1]],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr_tail = bytearray()
    last_report = 0.0

    def read_stderr():
        nonlocal last_report
        for raw_line in iter(process.stderr.readline, b""):
            stderr_tail.extend(raw_line)
            if len(stderr_tail) > _MAX_FFMPEG_STDERR_BYTES:
                del stderr_tail[:-_MAX_FFMPEG_STDERR_BYTES]
            try:
                key, value = raw_line.decode(errors="replace").strip().split("=", 1)
                if key == "out_time_us":
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        progress_callback(int(value) / 1_000_000)
                        last_report = now
            except ValueError:
                pass

    stderr_thread = threading.Thread(target=read_stderr, name="dasiwa-ffmpeg-progress", daemon=True)
    stderr_thread.start()
    stdin = process.stdin
    if stdin is None:
        process.kill()
        process.wait()
        stderr_thread.join()
        raise RuntimeError("Could not open FFmpeg stdin.")
    try:
        for chunk in frame_chunks():
            stdin.write(chunk)
        stdin.close()
        returncode = process.wait(timeout=3600)
    except BrokenPipeError:
        try:
            stdin.close()
        except BrokenPipeError:
            pass
        returncode = process.wait(timeout=3600)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        stderr_thread.join()
    return subprocess.CompletedProcess(command, returncode, stderr=bytes(stderr_tail))


def _encode_with_available_encoder(
    ffmpeg, codec, bit_depth, width, height, frame_rate, frame_chunks, output_path,
    container, cq, crf, metadata_path, audio_path=None, audio_duration=None, crop_to_audio=False,
    audio_codec="Auto", audio_bitrate="192k", progress_callback=None, encoder_names=None, color="SDR",
):
    available = _available_encoders(ffmpeg)
    attempts = []
    candidates = _ENCODER_NAMES[codec]
    if encoder_names:
        # A meta-batch stream is pinned to the encoder its probe found; a fallback would restart the stream midway.
        candidates = tuple(name for name in encoder_names if name in candidates)
    for encoder in candidates:
        if encoder not in available:
            continue
        audio_encoders = _audio_encoder_candidates(audio_codec, container) if audio_path else (None,)
        for selected_audio_encoder in audio_encoders:
            command = [ffmpeg, "-y", "-v", "error"]
            if metadata_path:
                command.extend(["-f", "ffmetadata", "-i", metadata_path])
            command.extend([
                "-f", "rawvideo", "-pix_fmt", "rgb48le" if bit_depth == 10 else "rgb24",
                "-s", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "-",
            ])
            video_input_index = 1 if metadata_path else 0
            if audio_path:
                command.extend(["-f", "f32le", "-ar", str(audio_path[1]), "-ac", str(audio_path[2]), "-i", audio_path[0]])
            if metadata_path:
                command.extend(["-map", f"{video_input_index}:v:0", "-map_metadata", "0"])
            elif audio_path:
                command.extend(["-map", f"{video_input_index}:v:0"])
            if audio_path:
                command.extend(["-map", f"{video_input_index + 1}:a:0", "-c:a", selected_audio_encoder, "-b:a", audio_bitrate])
            command.extend(_encoder_arguments(codec, encoder, bit_depth, cq, crf))
            command.extend(_color_arguments(color))
            if crop_to_audio and audio_duration is not None:
                command.extend(["-t", f"{audio_duration:.9f}"])
            if container == "MP4":
                command.extend(["-movflags", "+use_metadata_tags"])
            result = _run_ffmpeg(command + [output_path], frame_chunks, progress_callback)
            if result.returncode == 0:
                if selected_audio_encoder and selected_audio_encoder != _audio_encoder(audio_codec, container):
                    _log(f"Audio fallback: {selected_audio_encoder}.")
                audio_details = f"; audio={selected_audio_encoder}/{audio_bitrate}" if selected_audio_encoder else ""
                _log(f"Encoded {codec}/{container} via {encoder}{audio_details} -> {os.path.basename(output_path)}.")
                return encoder
            error_lines = result.stderr.decode(errors="replace").splitlines()
            error = error_lines[0][:180] if error_lines else "unknown FFmpeg error"
            attempts.append(f"{encoder}/{selected_audio_encoder or 'no-audio'}: {error}")
    raise RuntimeError("No usable encoder was found. " + " | ".join(attempts))


def _encode_animated_image(ffmpeg, container, bit_depth, width, height, frame_rate, frame_chunks, output_path, quality, progress_callback=None):
    available = _available_encoders(ffmpeg)
    attempts = []
    for encoder in _animated_image_encoder_candidates(container):
        if encoder not in available:
            continue
        command = [
            ffmpeg, "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb48le" if bit_depth == 10 else "rgb24",
            "-s", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "-",
        ]
        if container == "Animated WebP":
            command.extend(["-c:v", encoder, "-loop", "0", "-q:v", str(quality)])
        else:
            command.extend(_encoder_arguments("AV1", encoder, bit_depth, quality, quality))
            command.extend(["-still-picture", "0", "-f", "avif"])
        result = _run_ffmpeg(command + [output_path], frame_chunks, progress_callback)
        if result.returncode == 0:
            _log(f"Encoded {container} via {encoder} -> {os.path.basename(output_path)}.")
            return encoder
        error_lines = result.stderr.decode(errors="replace").splitlines()
        error = error_lines[0][:180] if error_lines else "unknown FFmpeg error"
        attempts.append(f"{encoder}: {error}")
    if not attempts:
        raise RuntimeError(f"FFmpeg does not provide a usable encoder for {container}.")
    raise RuntimeError(f"{container} encode failed. " + " | ".join(attempts))


class SaveVideoSimple:
    DESCRIPTION = (
        "Combines an IMAGE batch into a high-quality video with automatic NVENC/GPU "
        "selection, optional ping-pong playback, ComfyUI workflow metadata, and MP4 fallback."
    )

    @classmethod
    def INPUT_TYPES(cls):
        # `tooltip`, not `description`: ComfyUI shows `tooltip` on hover and the ⓘ panel
        # lists it; a `description` key on an input is silently ignored.
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames to encode, in order. One IMAGE batch = one video."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.01,
                    "tooltip": "Playback speed in frames per second. Match the model that made the frames "
                               "(MiniMax H3 and Wan: 24; LTX: 24 or 25), or the video plays too fast or too slow."}),
                "codec": (_CODEC_OPTIONS, {"default": "Auto",
                    "tooltip": "Auto tries AV1, then VP9, then H.264, and keeps the first one this machine can "
                               "encode (NVENC on an NVIDIA card). H.264 plays everywhere; pick it for sharing. "
                               "H.265 is smaller but some browsers will not preview it."}),
                "container": (_CONTAINER_OPTIONS, {"default": "Auto",
                    "tooltip": "The file type. Auto picks one that fits the codec (WebM, MKV or MP4). MP4 is the "
                               "safest to share. Animated WebP / AVIF make an image animation instead: codec is "
                               "ignored, there is no audio and no VIDEO output."}),
                "bit_depth": (["Auto", "8-bit", "10-bit"], {"default": "Auto",
                    "tooltip": "Colour precision. Auto: 8-bit when codec is Auto; with a chosen codec, 10-bit only if "
                               "the frames carry 10-bit detail. 10-bit reduces banding "
                               "in skies and gradients but needs AV1, H.265 or VP9 and a player that supports it."}),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51,
                    "tooltip": "Lower = better picture, bigger file. 0 is near-lossless and huge; 18-23 looks "
                               "clean; above 30 shows blocks. 20 is the recommended default."}),
                "log_level": (["Standard", "Verbose"], {"default": "Standard",
                    "tooltip": "Kept so old workflows still load. Has no effect: the console log is always short."}),
                "pingpong": ("BOOLEAN", {"default": False,
                    "tooltip": "Play forward, then backward, so the clip loops without a jump. Almost doubles the "
                               "length. Audio is not reversed."}),
                "save_metadata": ("BOOLEAN", {"default": True,
                    "tooltip": "Embed the workflow and prompt in the video file, so a workflow-aware loader can "
                               "restore the graph from it. Off = a clean file with no workflow inside."}),
                "filename_prefix": ("STRING", {"default": "video_%date:hhmmss%",
                    "tooltip": "Filename. Files are numbered: video_120455_00001.mp4. Date tokens: %date%, "
                               "%date:hhmmss%, %date:yyyy-MM-dd%. Put folders in output_path (a folder here "
                               "still works, but only inside the output folder)."}),
                # Right under filename_prefix. Moving a widget shifts positional
                # widgets_values; js/save_video_simple_preview.js maps old saves by name.
                "output_path": ("STRING", {"default": "",
                    "placeholder": "empty = ComfyUI output folder",
                    "tooltip": "Where to save. Empty = ComfyUI output folder. A relative name = a subfolder of "
                               "it (renders/h3). An absolute path = save exactly there (D:/Videos/h3). Date "
                               "tokens work: renders/%date:yyyy-MM-dd%. The in-node player only plays files "
                               "inside the output folder."}),
                "save_output": ("BOOLEAN", {"default": True,
                    "tooltip": "On = keep the video in the output folder (or output_path). Off = write it to "
                               "ComfyUI's temp folder for preview only; it is deleted on the next restart."}),
                "pass_frames": ("BOOLEAN", {"default": False,
                    "tooltip": "On = the frames output carries the encoded frames (ping-pong applied). Off = it is "
                               "empty, which saves memory when nothing is wired to it."}),
                "crop_to_audio": ("BOOLEAN", {"default": False,
                    "tooltip": "With audio connected. On = the file ends when the audio ends (extra frames are cut). "
                               "Off = the file runs to whichever is longer, the frames or the audio."}),
                "audio_codec": (_AUDIO_CODEC_OPTIONS, {"default": "Auto",
                    "tooltip": "Auto uses Opus for WebM and AAC for MKV / MP4. Leave on Auto unless a player "
                               "rejects the file."}),
                "audio_bitrate": (_AUDIO_BITRATE_OPTIONS, {"default": "192k",
                    "tooltip": "Audio quality. 192k is clean for music and speech; 320k for critical music; "
                               "128k for small files."}),
                "save_first_frame": ("BOOLEAN", {"default": False,
                    "tooltip": "Also save the first frame as a PNG beside the video. Useful as a thumbnail or as "
                               "the start image of the next clip."}),
                "save_last_frame": ("BOOLEAN", {"default": False,
                    "tooltip": "Also save the last frame as a PNG beside the video. Feed it to the next clip's "
                               "start image to continue a shot."}),
                "save_prompt": ("BOOLEAN", {"default": True,
                    "tooltip": "Write the prompt to a .txt beside the video, same name. Needs prompt_text "
                               "connected; with nothing wired, no .txt is written."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional sound to put into the video. Not used for Animated "
                                               "WebP / AVIF, which cannot hold audio."}),
                "prompt_text": ("STRING", {"forceInput": True,
                    "tooltip": "The final prompt, wired from whatever produced it (e.g. the LLM node). Written "
                               "word for word to the .txt when save_prompt is on."}),
                "latent": ("LATENT", {"tooltip": "Optional: the sampler's latent of these frames (Luna-H3-Sampler.latent). "
                    "Wired = saved beside the video as <name>.latent (with meta_batch: <name>_clip_NNNN.latent per run), "
                    "the whole latent, to decode or upscale it later without sampling again. About 60 MB per 8 s H3 "
                    "clip at 2 MP."}),
                "meta_batch": ("VHS_BatchManager", {"tooltip": "Luna-MetaBatch.meta_batch: write ONE file across "
                    "all the runs it queues (each Luna-H3-Chain clip). Every run adds its frames; the run after the "
                    "last clip closes the file, adds the sound wired at that run (Luna-H3-ChainJoin gives the whole "
                    "track then) and gives filename and video. Not with pingpong or Animated WebP / AVIF."}),
                # Last, so saved workflows keep their widget positions.
                "color": (COLOR_OPTIONS, {"default": "SDR", "tooltip": COLOR_TOOLTIP}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "VIDEO")
    RETURN_NAMES = ("frames", "filename", "video")
    OUTPUT_TOOLTIPS = (
        "The frames as encoded (ping-pong applied). Empty unless pass_frames is on.",
        "Full path of the saved video file.",
        "The saved file as a native ComfyUI VIDEO, sound included. Empty for Animated WebP / AVIF.",
    )
    FUNCTION = "combine"
    OUTPUT_NODE = True
    CATEGORY = "Luna-Media-Tools"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always encode again when the output node is queued with unchanged frames."""
        return float("nan")

    def validate_inputs(self, *args, **kwargs):
        return True

    def combine(
        self, images, frame_rate, codec, container, bit_depth, quality, pingpong,
        save_metadata, filename_prefix, save_output, pass_frames, crop_to_audio=False, audio_codec="Auto",
        audio_bitrate="192k", log_level="Standard", save_first_frame=False, save_last_frame=False, save_prompt=True,
        audio=None, prompt_text="", output_path="", prompt=None, extra_pnginfo=None, meta_batch=None, unique_id=None,
        latent=None, color="SDR",
    ):
        if images.ndim != 4 or images.shape[-1] < 3:
            raise ValueError("images must be an IMAGE batch shaped [frames, height, width, channels] with RGB channels.")
        if meta_batch is not None:
            return self._combine_batch(
                images, meta_batch, unique_id, prompt, extra_pnginfo, frame_rate, codec, container, bit_depth, quality,
                pingpong, save_metadata, filename_prefix, output_path, save_output, pass_frames, crop_to_audio,
                audio_codec, audio_bitrate, save_first_frame, save_last_frame, save_prompt, audio, prompt_text,
                latent, color,
            )

        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(_encoded_frame_count(images, pingpong))
        except ImportError:
            progress_bar = None

        def report_encode_progress(encoded_seconds):
            if progress_bar is not None:
                progress_bar.update_absolute(min(_encoded_frame_count(images, pingpong), max(0, int(encoded_seconds * frame_rate))))

        if progress_bar is not None:
            progress_bar.update_absolute(0)
        selected_bit_depth = _selected_bit_depth(codec, bit_depth, images, color)
        height, width = images.shape[1:3]
        frame_chunks = lambda: _iter_frame_byte_chunks(images, selected_bit_depth, pingpong)
        saved = encode_video_file(
            frame_chunks, width, height, _encoded_frame_count(images, pingpong), frame_rate, codec, container,
            selected_bit_depth, quality, save_metadata, filename_prefix, output_path, save_output, audio,
            crop_to_audio, audio_codec, audio_bitrate, prompt, extra_pnginfo, report_encode_progress, color=color,
        )
        output_frames = _pingpong_frames(images, pingpong) if pass_frames else images[:0]
        if progress_bar is not None:
            progress_bar.update_absolute(_encoded_frame_count(images, pingpong))
        if latent is not None:
            _save_latent(latent, os.path.splitext(saved["path"])[0] + ".latent")
        return _saved_result(
            saved, selected_bit_depth, width, height, frame_rate, images, pingpong, save_first_frame, save_last_frame,
            save_prompt, prompt_text, output_frames,
        )

    def _combine_batch(
        self, images, meta_batch, unique_id, prompt, extra_pnginfo, frame_rate, codec, container, bit_depth, quality,
        pingpong, save_metadata, filename_prefix, output_path, save_output, pass_frames, crop_to_audio, audio_codec,
        audio_bitrate, save_first_frame, save_last_frame, save_prompt, audio, prompt_text, latent=None, color="SDR",
    ):
        """One run of a Luna-MetaBatch chain: this run's frames go into the file the first run opened.

        The first run picks the encoder with a one-frame test and starts the encoder thread; every run feeds it; the
        run after which the batch source has closed its inputs ends the file, adds the sound and returns the outputs.
        """
        from .meta_batch import requeue_workflow

        _check_manager(prompt, unique_id)
        if _animated_image_settings(container):
            raise ValueError(f"[Luna-SaveVideo] {container} cannot be written across batches; pick a video container.")
        if pingpong:
            raise ValueError("[Luna-SaveVideo] pingpong cannot run across batches; turn it off with meta_batch.")
        key = str(unique_id)
        height, width = images.shape[1:3]
        entry = meta_batch.outputs.get(key)
        if entry is None:
            depth = _selected_bit_depth(codec, bit_depth, images, color)
            probe = _probe_encoder(images[:1], depth, frame_rate, codec, container, quality, color)
            total = meta_batch.total_frames
            stream = _MetaBatchStream(dict(
                width=width, height=height, frame_count=int(total) if total != float("inf") else 0,
                frame_rate=frame_rate, codec=probe["codec"], container=probe["container"], selected_bit_depth=depth,
                quality=quality, save_metadata=save_metadata, filename_prefix=filename_prefix, output_path=output_path,
                save_output=save_output, prompt=prompt, extra_pnginfo=extra_pnginfo, encoder_names=(probe["encoder"],),
                color=color,
            ))
            handle = _stream_handle(stream)
            next(handle)
            state = {"stream": stream, "depth": depth, "size": (height, width), "frames": 0, "runs": 0,
                     "first": images[:1].clone() if save_first_frame else None}
            meta_batch.outputs[key] = (state, handle)
        else:
            state, handle = entry
            if (height, width) != state["size"]:
                raise ValueError(f"[Luna-SaveVideo] this batch is {width}x{height}, the file is "
                                 f"{state['size'][1]}x{state['size'][0]}; every batch must have the same size.")
        for chunk in _iter_frame_byte_chunks(images, state["depth"], False):
            state["stream"].put(chunk)
        state["frames"] += images.shape[0]
        if latent is not None:
            stem = os.path.splitext(state["stream"].wait_for_path())[0]
            _save_latent(latent, f"{stem}_clip_{state['runs']:04d}.latent")
        state["runs"] += 1
        output_frames = images if pass_frames else images[:0]
        requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
        if not meta_batch.has_closed_inputs:
            _log(f"Batch written: {state['frames']} frames so far.")
            return {"ui": {"unfinished_batch": [True]},
                    "result": (output_frames, ExecutionBlocker(None), ExecutionBlocker(None))}

        meta_batch.outputs.pop(key, None)
        saved = state["stream"].finish()
        handle.close()
        if len(meta_batch.outputs) == 0:
            meta_batch.reset()
        if audio is not None:
            _mux_audio(saved["path"], audio, saved["container"], audio_codec, audio_bitrate, crop_to_audio)
        ends = torch.cat([state["first"] if state["first"] is not None else images[:1], images[-1:]])
        _log(f"Meta-batch file closed: {state['frames']} frames.")
        return _saved_result(
            saved, state["depth"], width, height, frame_rate, ends, False, save_first_frame, save_last_frame,
            save_prompt, prompt_text, output_frames,
        )


def _saved_result(
    saved, selected_bit_depth, width, height, frame_rate, frame_images, pingpong, save_first_frame, save_last_frame,
    save_prompt, prompt_text, output_frames,
):
    """The node's ui and outputs for a finished file. `frame_images` holds at least the first and last frame."""
    output_path, subfolder, output_type = saved["path"], saved["subfolder"], saved["type"]
    selected_codec, selected_container = saved["codec"], saved["container"]
    frame_exports = _save_frame_exports(frame_images, output_path, save_first_frame, save_last_frame, pingpong)
    prompt_sidecar = _write_prompt_sidecar(output_path, prompt_text) if save_prompt else None
    animated_settings = _animated_image_settings(selected_container)
    output_mime_type = _MIME_TYPES[selected_container]
    assets = [{"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": output_mime_type, "width": width, "height": height, "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container}]
    assets.extend(
        {"filename": os.path.basename(path), "subfolder": subfolder, "type": output_type, "format": "image/png", "width": width, "height": height}
        for path in frame_exports
    )
    if subfolder is None:
        # Saved outside the output folder: nothing the player may fetch.
        assets = []
        _log("Saved outside the ComfyUI output folder - no in-node preview.")
    ui = {"images": assets}
    if not animated_settings and subfolder is not None:
        ui["gifs"] = [{"filename": os.path.basename(output_path), "subfolder": subfolder, "type": output_type, "format": output_mime_type, "codec": selected_codec, "bit_depth": selected_bit_depth, "container": selected_container, "width": width, "height": height, "fps": frame_rate}]
    for path in frame_exports:
        _log(f"Frame export: {path}.")
    if prompt_sidecar:
        _log(f"Prompt sidecar: {prompt_sidecar}.")
    elif save_prompt:
        _log("save_prompt is on but prompt_text is empty - no sidecar written.")
    # Wrap the actual encoded file: includes audio, crop and ping-pong.
    # Animated image formats are not native VIDEO containers.
    saved_video = None
    if not animated_settings:
        from comfy_api.latest import InputImpl
        saved_video = InputImpl.VideoFromFile(output_path)
    return {"ui": ui, "result": (output_frames, output_path, saved_video)}


def _latent_tensors(latent):
    """A LATENT's tensors by name: an H3 audio+video latent as video / audio (as Luna-H3-ChainRemember writes them)."""
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        if len(parts) == 2 and parts[0].ndim == 5:
            names = ("video", "audio")
        else:
            names = tuple(f"part_{index}" for index in range(len(parts)))
    else:
        parts, names = (samples,), ("samples",)
    return {name: part.detach().cpu().contiguous() for name, part in zip(names, parts)}


def _save_latent(latent, path):
    import safetensors.torch

    safetensors.torch.save_file(_latent_tensors(latent), path,
                                metadata={"format": "luna-latent-1", "saved_by": "Luna-SaveVideo"})
    _log(f"Latent saved: {path}.")
    return path


def _check_manager(prompt, unique_id):
    """Only Luna-MetaBatch counts Luna-SaveVideo when it decides to queue the next batch."""
    from .meta_batch import MANAGER_IDS

    link = ((prompt or {}).get(str(unique_id)) or {}).get("inputs", {}).get("meta_batch")
    if isinstance(link, list) and (prompt.get(str(link[0])) or {}).get("class_type") not in MANAGER_IDS:
        raise ValueError("[Luna-SaveVideo] meta_batch must come from Luna-MetaBatch. VHS's Meta Batch Manager does not "
                         "count Luna-SaveVideo, so the next batch would never be queued.")


def _probe_encoder(frame, depth, frame_rate, codec, container, quality, color="SDR"):
    """Pick codec, container and encoder once, by encoding one frame to the temp folder, so the real file never has to
    restart halfway through a chain."""
    saved = encode_video_file(
        lambda: _iter_frame_byte_chunks(frame, depth, False), frame.shape[2], frame.shape[1], 1, frame_rate, codec,
        container, depth, quality, False, "luna_metabatch_probe", "", False, color=color,
    )
    try:
        os.remove(saved["path"])
    except OSError:
        pass
    return saved


_STREAM_END = object()


class _MetaBatchStream:
    """One file written across the runs of a Luna-MetaBatch chain: an encoder thread fed through a small queue."""

    def __init__(self, encode_arguments):
        self.queue = queue.Queue(maxsize=4)
        self.result = None
        self.error = None
        self.path = None
        self.path_ready = threading.Event()
        encode_arguments = dict(encode_arguments, on_output_path=self._set_path)
        self.thread = threading.Thread(target=self._run, args=(encode_arguments,), name="luna-metabatch-encode",
                                       daemon=True)
        self.thread.start()

    def _chunks(self):
        while True:
            item = self.queue.get()
            if item is _STREAM_END:
                return
            yield item

    def _run(self, encode_arguments):
        try:
            self.result = encode_video_file(lambda: self._chunks(), **encode_arguments)
        except BaseException as error:  # handed to the run that waits on it
            self.error = error

    def _set_path(self, path):
        self.path = path
        self.path_ready.set()

    def wait_for_path(self):
        """The file's path, known once the encoder has started on it."""
        while not self.path_ready.wait(timeout=0.5):
            if self.error is not None or not self.thread.is_alive():
                raise RuntimeError(f"[Luna-SaveVideo] the encoder stopped: {self.error}")
            _raise_if_interrupted()
        return self.path

    def put(self, chunk):
        while True:
            if self.error is not None or not self.thread.is_alive():
                raise RuntimeError(f"[Luna-SaveVideo] the encoder stopped: {self.error}")
            try:
                self.queue.put(chunk, timeout=0.5)
                return
            except queue.Full:
                _raise_if_interrupted()

    def finish(self):
        while self.thread.is_alive():
            try:
                self.queue.put(_STREAM_END, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"[Luna-SaveVideo] the encoder failed: {self.error}") from self.error
        return self.result


def _stream_handle(stream):
    """What Luna-MetaBatch keeps for this output. Its reset (a stopped or restarted chain) closes the file."""
    yield
    try:
        saved = stream.finish()
        _log(f"Meta-batch stopped early; the file keeps the frames written so far: {saved['path']}.")
    except Exception as error:
        _log(f"Meta-batch stopped early and the file could not be closed: {error}")


def _raise_if_interrupted():
    try:
        import comfy.model_management
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def _mux_audio(video_path, audio, container, audio_codec, audio_bitrate, crop_to_audio):
    """Add the sound to a finished file without touching the picture (the last run brings the whole track)."""
    ffmpeg = find_ffmpeg()
    audio_path, audio_duration = _audio_file(audio)
    stem, extension = os.path.splitext(video_path)
    muxed = f"{stem}.mux{extension}"
    attempts = []
    try:
        for encoder in _audio_encoder_candidates(audio_codec, container):
            command = [
                ffmpeg, "-y", "-v", "error", "-i", video_path,
                "-f", "f32le", "-ar", str(audio_path[1]), "-ac", str(audio_path[2]), "-i", audio_path[0],
                "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
                "-c:a", encoder, "-b:a", audio_bitrate,
            ]
            if crop_to_audio and audio_duration is not None:
                command.extend(["-t", f"{audio_duration:.9f}"])
            if container == "MP4":
                command.extend(["-movflags", "+use_metadata_tags"])
            result = subprocess.run(command + [muxed], capture_output=True, timeout=3600)
            if result.returncode == 0:
                os.replace(muxed, video_path)
                _log(f"Sound added: {encoder}/{audio_bitrate}.")
                return encoder
            lines = result.stderr.decode(errors="replace").splitlines()
            attempts.append(f"{encoder}: {lines[0][:180] if lines else 'unknown FFmpeg error'}")
    finally:
        os.unlink(audio_path[0])
        if os.path.exists(muxed):
            os.unlink(muxed)
    raise RuntimeError("[Luna-SaveVideo] the sound could not be added: " + " | ".join(attempts))


_MIME_TYPES = {
    "WebM": "video/webm", "MKV": "video/x-matroska", "MP4": "video/mp4",
    **{name: settings[2] for name, settings in _ANIMATED_IMAGE_SETTINGS.items()},
}


def encode_video_file(
    frame_chunks, width, height, frame_count, frame_rate, codec, container, selected_bit_depth, quality,
    save_metadata, filename_prefix, output_path, save_output, audio=None, crop_to_audio=False,
    audio_codec="Auto", audio_bitrate="192k", prompt=None, extra_pnginfo=None, report_encode_progress=None,
    encoder_names=None, on_output_path=None, color="SDR",
):
    """Encode raw frames to a file: output folder, codec and container fallback, audio, metadata.

    `frame_chunks` is a callable returning an iterator of raw rgb24 (8-bit) or rgb48le (10-bit) bytes,
    so the frames can be produced while FFmpeg encodes them and never sit in memory all at once.
    A failed encoder attempt calls it again. Shared by Luna-SaveVideo and Luna-RelayVSR.
    `color` HDR (COLOR_OPTIONS) needs 10-bit frames and writes H.265 / AV1 with the HDR tags, never the H.264 fallback.
    """
    hdr = is_hdr(color)
    if hdr:
        if selected_bit_depth != 10:
            raise ValueError(f"[Luna-SaveVideo] {color} needs 10-bit frames; got {selected_bit_depth}-bit.")
        if _animated_image_settings(container):
            raise ValueError(f"[Luna-SaveVideo] {container} cannot hold HDR; pick a video container or SDR.")
        wanted = _hdr_codec(codec, container)
        if wanted != codec:
            _log(f"{color}: codec {codec} -> {wanted} (HDR needs 10-bit H.265 or AV1).")
        codec = wanted
    output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    output_type = "output" if save_output else "temp"
    filename_prefix = _format_filename_prefix(filename_prefix)
    # output_path works as in Luna-SaveImage: empty = output (or temp) folder,
    # relative = subfolder of it, absolute = exactly there. The prefix may
    # still hold folders, as before, but only below the target directory.
    save_dir = _format_filename_prefix(output_path or "").strip()
    target_dir = output_dir
    if save_dir:
        target_dir = save_dir if os.path.isabs(save_dir) else os.path.join(output_dir, save_dir)
    os.makedirs(target_dir, exist_ok=True)
    output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, target_dir, width, height)
    # The player can only fetch files inside the output/temp folder.
    subfolder = _subfolder_under(output_folder, output_dir)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("No FFmpeg executable was found. Install FFmpeg or imageio-ffmpeg for the mandatory H.264/MP4 fallback.")

    metadata_path = _metadata_file(prompt, extra_pnginfo) if save_metadata else None
    audio_path, audio_duration = _audio_file(audio)
    attempts = []
    _log(
        f"Encode {frame_count}f {width}x{height}@{frame_rate:g}fps {selected_bit_depth}-bit; "
        f"codec={codec}, container={container}, audio={'yes' if audio_path else 'no'}"
        + (f", {color}." if hdr else ".")
    )
    try:
        animated_settings = _animated_image_settings(container)
        if animated_settings:
            if audio_path:
                _log(f"{container} does not support audio; connected audio is omitted.")
            output_path = os.path.join(output_folder, _output_filename(filename, counter, animated_settings[0]))
            encoder = _encode_animated_image(
                ffmpeg, container, selected_bit_depth, width, height, frame_rate, frame_chunks, output_path, quality,
                report_encode_progress,
            )
            selected_container = container
            selected_codec = container
        else:
            for selected_codec in _codec_candidates(codec):
                container_candidates = _auto_container_candidates(selected_codec, container) if codec == "Auto" else _container_candidates(selected_codec, container)
                for selected_container in container_candidates:
                    if codec == "Auto":
                        _log(f"Auto test: {selected_codec}/{selected_container}.")
                    output_path = os.path.join(
                        output_folder,
                        _output_filename(filename, counter, _CONTAINER_EXTENSIONS[selected_container]),
                    )
                    if on_output_path is not None:
                        on_output_path(output_path)
                    try:
                        encoder = _encode_with_available_encoder(
                            ffmpeg, selected_codec, selected_bit_depth, width, height, frame_rate, frame_chunks,
                            output_path, selected_container, quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                            audio_codec, audio_bitrate, report_encode_progress, encoder_names, color,
                        )
                        break
                    except RuntimeError as error:
                        attempts.append(f"{selected_codec}/{selected_container}: {error}")
                        if codec == "Auto":
                            _log(f"Auto miss: {selected_codec}/{selected_container}; trying next.")
                else:
                    continue
                break
            else:
                if hdr:  # H.264 cannot carry this HDR file; say why instead of writing an SDR-looking one
                    raise RuntimeError(f"{color}: no {codec} encoder worked. " + " | ".join(attempts))
                fallback_path = os.path.join(output_folder, _output_filename(filename, counter, ".mp4"))
                if on_output_path is not None:
                    on_output_path(fallback_path)
                encoder = _encode_with_available_encoder(
                    ffmpeg, "H.264", selected_bit_depth, width, height, frame_rate, frame_chunks,
                    fallback_path, "MP4", quality, quality, metadata_path, audio_path, audio_duration, crop_to_audio,
                    audio_codec, audio_bitrate, report_encode_progress, encoder_names,
                )
                output_path = fallback_path
                selected_container = "MP4"
                selected_codec = "H.264"
    finally:
        if metadata_path:
            os.unlink(metadata_path)
        if audio_path:
            os.unlink(audio_path[0])

    _log(f"Output: {output_path} ({selected_codec}, {encoder}, {selected_bit_depth}-bit{', ' + color if hdr else ''}).")
    return {"path": output_path, "subfolder": subfolder, "type": output_type,
            "codec": selected_codec, "container": selected_container, "encoder": encoder}


NODE_CLASS_MAPPINGS = {"Luna-SaveVideo": SaveVideoSimple}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-SaveVideo": "Luna-SaveVideo"}
