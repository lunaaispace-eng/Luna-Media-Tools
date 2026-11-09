"""Luna-SaveVideo with Luna-MetaBatch: one file across three queued runs, the sound added when it closes, the next
run queued after every run but the last, and a stopped chain still leaves a playable file.
Real FFmpeg (libx264 on 64x64 frames), no ComfyUI server: the queue is a stand-in with ComfyUI's item shape.
"""

import os
import sys
import tempfile
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
meta = sys.modules[PACK.__name__ + ".meta_batch"]
video_nodes = sys.modules[PACK.__name__ + ".video_nodes"]


class FakeQueue:
    def __init__(self, item):
        self.currently_running = {0: item}
        self.put_items = []

    def put(self, item):
        self.put_items.append(item)


def graph(requeue, manager_class="Luna-MetaBatch"):
    return {
        "1": {"class_type": manager_class, "inputs": {"frames_per_batch": 10, "requeue": requeue}},
        "10": {"class_type": "Luna-SaveVideo", "inputs": {"images": ["5", 0], "meta_batch": ["1", 0]}},
    }


def frames(start, count=10):
    torch.manual_seed(start)
    return torch.rand(count, 64, 64, 3)


def sound(seconds, rate=48000):
    return {"waveform": torch.rand(1, 2, int(seconds * rate)) * 0.2 - 0.1, "sample_rate": rate}


class SaveVideoMetaBatchTests(unittest.TestCase):
    def setUp(self):
        import folder_paths
        import server

        self.server = server
        self.instance_before = getattr(server.PromptServer, "instance", None)
        self.folder_paths = folder_paths
        self.output_before = folder_paths.get_output_directory()
        self.tmp = tempfile.TemporaryDirectory()
        folder_paths.set_output_directory(self.tmp.name)
        meta._guard = [None, 0, 0, {}]
        self.node = video_nodes.SaveVideoSimple()
        self.manager = meta.LunaMetaBatch()

    def tearDown(self):
        self.manager.reset()
        self.server.PromptServer.instance = self.instance_before
        self.folder_paths.set_output_directory(self.output_before)
        self.tmp.cleanup()

    def run_batch(self, requeue, images, audio=None, last=False, run_number=None, **settings):
        prompt = graph(requeue)
        queue = FakeQueue((run_number if run_number is not None else 100 + requeue, "pid", prompt, {}, ["10"], {}))
        self.server.PromptServer.instance = types.SimpleNamespace(prompt_queue=queue, number=1, routes=None)
        self.manager.update_batch(10, prompt=prompt, unique_id="1")
        self.manager.has_closed_inputs = last
        arguments = dict(codec="H.264", container="MP4", bit_depth="8-bit", quality=18, pingpong=False,
                         save_metadata=False, filename_prefix="chain", save_output=True, pass_frames=False)
        arguments.update(settings)
        result = self.node.combine(images, 24.0, audio=audio, prompt=prompt, meta_batch=self.manager, unique_id="10",
                                   **arguments)
        return result, queue

    def components(self, path):
        from comfy_api.latest import InputImpl

        return InputImpl.VideoFromFile(path).get_components()

    def test_three_runs_make_one_file_with_sound(self):
        blocked = []
        for requeue in (0, 1):
            result, queue = self.run_batch(requeue, frames(requeue))
            self.assertEqual(result["ui"], {"unfinished_batch": [True]})
            blocked.append(type(result["result"][1]).__name__)
            self.assertEqual(len(queue.put_items), 1, requeue)
            self.assertEqual(queue.put_items[0][2]["1"]["inputs"]["requeue"], requeue + 1)
        self.assertEqual(blocked, ["ExecutionBlocker", "ExecutionBlocker"])
        result, queue = self.run_batch(2, frames(2), audio=sound(30 / 24), last=True, save_last_frame=True)
        self.assertEqual(queue.put_items, [])
        _, path, video = result["result"]
        self.assertTrue(path.endswith(".mp4"), path)
        parts = self.components(path)
        self.assertEqual(parts.images.shape[0], 30)
        self.assertIsNotNone(parts.audio)
        self.assertAlmostEqual(parts.audio["waveform"].shape[-1] / parts.audio["sample_rate"], 30 / 24, delta=0.1)
        self.assertEqual(self.manager.outputs, {})
        self.assertTrue(os.path.isfile(os.path.splitext(path)[0] + "-last-frame.png"))
        self.assertEqual([name for name in os.listdir(self.tmp.name) if ".mux" in name], [])
        del video, parts

    def test_frames_keep_their_order_across_runs(self):
        flat = [torch.full((10, 64, 64, 3), value) for value in (0.1, 0.5, 0.9)]
        for requeue, images in enumerate(flat):
            result, _ = self.run_batch(requeue, images, last=requeue == 2, quality=0)
        written = self.components(result["result"][1]).images
        for index, value in enumerate((0.1, 0.5, 0.9)):
            self.assertAlmostEqual(written[index * 10:(index + 1) * 10].mean().item(), value, delta=0.03)

    def test_stopped_chain_still_closes_the_file(self):
        self.run_batch(0, frames(0))
        self.run_batch(1, frames(1))
        self.manager.reset()   # what a fresh queue does after a stop
        files = [name for name in os.listdir(self.tmp.name) if name.endswith(".mp4")]
        self.assertEqual(len(files), 1)
        self.assertEqual(self.components(os.path.join(self.tmp.name, files[0])).images.shape[0], 20)

    def test_vhs_manager_is_refused(self):
        prompt = graph(0, manager_class="VHS_BatchManager")
        with self.assertRaises(ValueError):
            self.node.combine(frames(0), 24.0, "H.264", "MP4", "8-bit", 18, False, False, "x", True, False,
                              prompt=prompt, meta_batch=self.manager, unique_id="10")

    def test_pingpong_and_animated_images_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_batch(0, frames(0), pingpong=True)
        with self.assertRaises(ValueError):
            self.run_batch(0, frames(0), container="Animated WebP")

    def h3_latent(self, seed):
        from comfy.nested_tensor import NestedTensor

        torch.manual_seed(seed)
        return {"samples": NestedTensor([torch.randn(1, 48, 3, 4, 4), torch.randn(1, 64, 20)])}

    def load(self, path):
        import safetensors.torch

        return safetensors.torch.load_file(path)

    def test_each_run_saves_its_latent(self):
        latents = [self.h3_latent(seed) for seed in range(3)]
        for requeue in range(3):
            result, _ = self.run_batch(requeue, frames(requeue), last=requeue == 2, latent=latents[requeue])
        stem = os.path.splitext(result["result"][1])[0]
        for run, latent in enumerate(latents):
            saved = self.load(f"{stem}_clip_{run:04d}.latent")
            self.assertEqual(sorted(saved), ["audio", "video"])
            self.assertTrue(torch.equal(saved["video"], latent["samples"].tensors[0]))
            self.assertTrue(torch.equal(saved["audio"], latent["samples"].tensors[1]))

    def test_plain_save_writes_one_latent_beside_the_video(self):
        latent = {"samples": torch.randn(1, 16, 8, 8)}
        result = self.node.combine(frames(0), 24.0, "H.264", "MP4", "8-bit", 18, False, False, "plain", True, False,
                                   latent=latent)
        saved = self.load(os.path.splitext(result["result"][1])[0] + ".latent")
        self.assertTrue(torch.equal(saved["samples"], latent["samples"]))

    def test_no_latent_wired_no_file(self):
        result = self.node.combine(frames(0), 24.0, "H.264", "MP4", "8-bit", 18, False, False, "plain", True, False)
        self.assertFalse(os.path.exists(os.path.splitext(result["result"][1])[0] + ".latent"))

    def test_without_meta_batch_nothing_changes(self):
        result = self.node.combine(frames(0), 24.0, "H.264", "MP4", "8-bit", 18, False, False, "plain", True, False)
        self.assertEqual(self.components(result["result"][1]).images.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
