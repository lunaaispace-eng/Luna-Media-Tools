"""Luna-RelayVSR and the encoder it shares with Luna-SaveVideo.

The per-window decode must give exactly the frames ComfyUI's native single VSR Decode gives, and the node must
write every frame to the file at the model's size. Runs on the CPU with a tiny RelayVSR Flash decoder of random
weights: no model files, no ComfyUI server.
"""

import os
import sys
import tempfile
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
        relay.sample_keyframes = lambda model, imgs, seed: latent_for(imgs, decoder)
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
        self.assertEqual(result["ui"]["animated"], (True,))
        del video, written

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


if __name__ == "__main__":
    unittest.main()
