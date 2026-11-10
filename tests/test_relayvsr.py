"""Luna-RelayVSR and the encoder it shares with Luna-SaveVideo.

The per-window decode must give exactly the frames ComfyUI's native single VSR Decode gives, and the node must
write every frame to the file at the model's size. Runs on the CPU with a tiny RelayVSR Flash decoder of random
weights: no model files, no ComfyUI server.
"""

import os
import sys
import tempfile
import threading
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
relay = sys.modules[PACK.__name__ + ".relayvsr_node"]
video_nodes = sys.modules[PACK.__name__ + ".video_nodes"]

from comfy.ldm.wan.model_vsr import RelayFlash, relay_keyframes, relay_latent_size  # noqa: E402


def tiny_decoder(factor=2):
    torch.manual_seed(0)
    decoder = RelayFlash(factor, latent_channels=48, dim=128, depth=2, num_heads=2, kv_groups=1, hidden_dim=128,
                         head_dim=64)
    for parameter in decoder.parameters():
        parameter.data = torch.randn_like(parameter) * 0.05
    return decoder.eval()


class FakeVAE:
    """What comfy.sd.VAE.decode does around the Flash decoder, without the memory management."""

    def __init__(self, decoder):
        self.first_stage_model = decoder
        self.calls = 0

    def decode(self, samples, vae_options={}):
        self.calls += 1
        with torch.no_grad():
            return self.first_stage_model.decode(samples, **vae_options).movedim(1, -1)


class FakeModel:
    def __init__(self, latent_stride):
        diffusion_model = type("DM", (), {})()
        diffusion_model.positive_conditioning = torch.zeros(4, 8)
        diffusion_model.latent_stride = latent_stride
        self.model = type("M", (), {})()
        self.model.diffusion_model = diffusion_model


def clip(frames, height=32, width=48):
    torch.manual_seed(frames)
    return torch.rand(frames, height, width, 3)


def latent_for(images, decoder):
    keyframes = relay_keyframes(images.shape[0])
    height, width = relay_latent_size(images.shape[1], images.shape[2], decoder.latent_stride)
    torch.manual_seed(1)
    return torch.randn(1, 48, len(keyframes), height, width)


class WindowTests(unittest.TestCase):
    def test_windows_give_every_frame_once_in_order(self):
        for frames in range(1, 200):
            _, windows = relay.relay_windows(frames)
            out = []
            for index, (first, last) in enumerate(windows):
                span = list(range(first, last + 1))
                out += span if index == 0 else span[1:]
            self.assertEqual(out, list(range(frames)), frames)

    def test_window_decode_equals_native_decode(self):
        decoder = tiny_decoder()
        for frames in (1, 2, 15, 16, 17, 31, 33, 46):
            images = clip(frames)
            samples = latent_for(images, decoder)
            vae = FakeVAE(decoder)
            native = vae.decode(samples, {"lq_images": images, "detail": 1.0})
            native = native.reshape(-1, *native.shape[-3:])
            windowed = torch.cat(list(relay.decode_windows(vae, samples, images, 1.0)))
            self.assertEqual(windowed.shape, native.shape, frames)
            self.assertTrue(torch.equal(windowed, native), f"{frames} frames differ")

    def test_detail_is_passed_through(self):
        decoder = tiny_decoder()
        images = clip(17)
        samples = latent_for(images, decoder)
        vae = FakeVAE(decoder)
        native = vae.decode(samples, {"lq_images": images, "detail": 0.3})
        windowed = torch.cat(list(relay.decode_windows(vae, samples, images, 0.3)))
        self.assertTrue(torch.equal(windowed, native.reshape(-1, *native.shape[-3:])))


class NodeTests(unittest.TestCase):
    def setUp(self):
        import folder_paths

        self.folder_paths = folder_paths
        self.old_output = folder_paths.get_output_directory()
        self.tmp = tempfile.TemporaryDirectory()
        folder_paths.set_output_directory(self.tmp.name)
        self.old_sample = relay.sample_keyframes

    def tearDown(self):
        relay.sample_keyframes = self.old_sample
        self.folder_paths.set_output_directory(self.old_output)
        self.tmp.cleanup()

    def frames_in(self, path):
        from comfy_api.latest import InputImpl

        return InputImpl.VideoFromFile(path).get_components().images

    def test_node_writes_every_frame_at_the_model_size(self):
        decoder = tiny_decoder()
        images = clip(33)
        relay.sample_keyframes = lambda model, imgs, seed, sampler, scheduler: latent_for(imgs, decoder)
        vae = FakeVAE(decoder)
        result = relay.LunaRelayVSR().upscale(
            images, FakeModel(decoder.latent_stride), vae, 24.0, 0, 1.0, "H.264", "MP4", "8-bit", 18,
            "relay_test", "", True, False, False, "Auto", "192k",
        )
        path, video = result["result"]
        self.assertTrue(os.path.isfile(path))
        # an encoder FFmpeg lists but cannot open costs one decoded window before the next one is tried
        self.assertGreaterEqual(vae.calls, len(relay.relay_windows(33)[1]))
        written = self.frames_in(path)
        self.assertEqual(tuple(written.shape), (33, 64, 96, 3))
        self.assertEqual(result["ui"]["gifs"][0]["filename"], os.path.basename(path))
        self.assertEqual((result["ui"]["gifs"][0]["width"], result["ui"]["gifs"][0]["fps"]), (96, 24.0))
        del video, written

    def test_sampler_and_scheduler_reach_the_keyframe_pass(self):
        decoder = tiny_decoder()
        seen = []
        relay.sample_keyframes = lambda model, imgs, seed, sampler, scheduler: seen.append((sampler, scheduler)) or latent_for(imgs, decoder)
        relay.LunaRelayVSR().upscale(
            clip(16), FakeModel(decoder.latent_stride), FakeVAE(decoder), 24.0, 0, 1.0, "H.264", "MP4", "8-bit", 18,
            "relay_sampler", "", True, False, False, "Auto", "192k", "er_sde", "simple",
        )
        self.assertEqual(seen, [("er_sde", "simple")])

    def test_new_widgets_come_last(self):
        names = list(relay.LunaRelayVSR.INPUT_TYPES()["required"])
        self.assertEqual(names[-3:], ["sampler_name", "scheduler", "windows_ahead"])
        self.assertEqual(relay.LunaRelayVSR.INPUT_TYPES()["required"]["windows_ahead"][1]["default"], 2)
        self.assertIn("er_sde", relay.LunaRelayVSR.INPUT_TYPES()["required"]["sampler_name"][0])

    def test_wrong_model_is_refused(self):
        decoder = tiny_decoder()
        with self.assertRaises(ValueError):
            relay.LunaRelayVSR().upscale(
                clip(3), object(), FakeVAE(decoder), 24.0, 0, 1.0, "H.264", "MP4", "8-bit", 18, "x", "", True,
                False, False, "Auto", "192k",
            )

    def test_save_video_still_encodes_after_the_encoder_moved(self):
        images = clip(9, 64, 64)
        result = video_nodes.SaveVideoSimple().combine(
            images, 24.0, "H.264", "MP4", "8-bit", 18, False, False, "save_test", True, False,
        )
        frames, path, video = result["result"]
        self.assertEqual(frames.shape[0], 0)
        self.assertEqual(tuple(self.frames_in(path).shape), (9, 64, 64, 3))
        self.assertTrue(result["ui"]["gifs"])
        del video


class WindowsAheadTests(unittest.TestCase):
    """windows_ahead overlaps the GPU decode with the FFmpeg write; the bytes FFmpeg receives must not change."""

    def setUp(self):
        self.old_sample = relay.sample_keyframes
        self.old_encode = relay.encode_video_file

    def tearDown(self):
        relay.sample_keyframes = self.old_sample
        relay.encode_video_file = self.old_encode

    def fed_bytes(self, windows_ahead, frames=46, retry_after=None, bit_depth="8-bit"):
        """Every byte the node hands the encoder. retry_after = chunks read before a failed encoder gives up."""
        decoder = tiny_decoder()
        relay.sample_keyframes = lambda model, imgs, seed, sampler, scheduler: latent_for(imgs, decoder)
        fed = []

        def fake_encode(frame_chunks, *args, **kwargs):
            if retry_after is not None:
                attempt = frame_chunks()
                for _ in range(retry_after):
                    next(attempt)
                attempt.close()
            fed.append(b"".join(frame_chunks()))
            return {"path": os.path.join(tempfile.gettempdir(), "relay_fake.mp4"), "subfolder": None}

        relay.encode_video_file = fake_encode
        relay.LunaRelayVSR().upscale(
            clip(frames), FakeModel(decoder.latent_stride), FakeVAE(decoder), 24.0, 0, 1.0, "H.264", "MP4",
            bit_depth, 18, "x", "", True, False, False, "Auto", "192k", "euler", "simple", windows_ahead,
        )
        return fed[0]

    def test_same_bytes_at_any_depth(self):
        serial = self.fed_bytes(0)
        self.assertEqual(len(serial), 46 * 64 * 96 * 3)
        for depth in (1, 2, 8):
            self.assertEqual(self.fed_bytes(depth), serial, depth)
        self.assertEqual(self.fed_bytes(2, bit_depth="10-bit"), self.fed_bytes(0, bit_depth="10-bit"))

    def test_failed_encoder_restarts_from_the_first_frame(self):
        self.assertEqual(self.fed_bytes(2, retry_after=1), self.fed_bytes(0))
        self.assertFalse([t for t in threading.enumerate() if t.name == "luna-relayvsr-decode"])

    def test_error_in_the_decode_thread_reaches_the_node(self):
        def produce():
            yield 1
            raise RuntimeError("decode failed")

        got = []
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            for item in relay.run_ahead(produce, 2):
                got.append(item)
        self.assertEqual(got, [1])

    def test_closing_stops_the_decode_thread(self):
        made = []

        def produce():
            for index in range(100):
                made.append(index)
                yield index

        pipeline = relay.run_ahead(produce, 2)
        self.assertEqual(next(pipeline), 0)
        pipeline.close()
        self.assertFalse([t for t in threading.enumerate() if t.name == "luna-relayvsr-decode"])
        self.assertLessEqual(len(made), 5)

    def test_decode_thread_runs_in_inference_mode(self):
        def produce():
            yield torch.is_inference_mode_enabled()

        self.assertEqual(list(relay.run_ahead(produce, 1)), [True])


class BitDepthTests(unittest.TestCase):
    """detect_bit_depth samples before it converts, so a long 4K batch costs no copy of itself."""

    def old_detect(self, images):
        values = images.detach().to(device="cpu", dtype=torch.float32).flatten()
        if values.numel() > 250_000:
            values = values[:: max(1, values.numel() // 250_000)]
        values = values.clamp(0, 1)
        error_8 = torch.mean(torch.abs(values * 255 - torch.round(values * 255)))
        error_10 = torch.mean(torch.abs(values * 1023 - torch.round(values * 1023)))
        return 10 if error_10 < error_8 * 0.8 else 8

    def test_same_answer_as_before(self):
        torch.manual_seed(3)
        eight = torch.round(torch.rand(40, 64, 96, 3) * 255) / 255
        ten = torch.round(torch.rand(40, 64, 96, 3) * 1023) / 1023
        for images in (eight, ten, eight.half(), ten.to(torch.bfloat16), torch.rand(3, 8, 8, 3)):
            self.assertEqual(video_nodes.detect_bit_depth(images), self.old_detect(images))

    def test_huge_fp16_batch_is_not_copied(self):
        # expand() gives a 4K fp16 batch of 1076 frames (about 54 GB) that takes no memory; converting it whole would.
        frame = (torch.round(torch.rand(1, 2176, 3840, 3) * 1023) / 1023).half()
        images = frame.expand(1076, -1, -1, -1)
        self.assertEqual(video_nodes.detect_bit_depth(images), 10)
        self.assertEqual(video_nodes.detect_bit_depth(images[:2].contiguous()), 10)

    def test_empty_and_integer_batches(self):
        self.assertEqual(video_nodes.detect_bit_depth(torch.zeros(0, 4, 4, 3)), 8)
        self.assertEqual(video_nodes.detect_bit_depth(torch.zeros(2, 4, 4, 3, dtype=torch.uint8)), 8)
        self.assertEqual(video_nodes.detect_bit_depth(torch.zeros(2, 4, 4, 3, dtype=torch.int16)), 10)


if __name__ == "__main__":
    unittest.main()
