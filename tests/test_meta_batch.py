"""Luna-MetaBatch: the manager resets on a fresh queue, keeps its state across re-queues, and the graph is queued
again only after every Luna-SaveVideo on it has asked, with the manager's requeue count one higher.
No ComfyUI server: the queue is a stand-in with the same item shape.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_registration import load_pack  # noqa: E402

PACK = load_pack()
meta = sys.modules[PACK.__name__ + ".meta_batch"]


class FakeQueue:
    def __init__(self, item):
        self.currently_running = {0: item}
        self.put_items = []

    def put(self, item):
        self.put_items.append(item)


def graph(outputs=1, requeue=0):
    prompt = {
        "1": {"class_type": "Luna-MetaBatch", "inputs": {"frames_per_batch": 22, "requeue": requeue}},
        "2": {"class_type": "Luna-H3-Chain", "inputs": {"meta_batch": ["1", 0], "clips": 3}},
    }
    for index in range(outputs):
        prompt[str(10 + index)] = {"class_type": "Luna-SaveVideo", "inputs": {"images": ["5", 0], "meta_batch": ["1", 0]}}
    return prompt


class MetaBatchTests(unittest.TestCase):
    def setUp(self):
        self.server = sys.modules.get("server")
        self.instance_before = getattr(self.server.PromptServer, "instance", None)
        meta._guard = [None, 0, 0, {}]

    def tearDown(self):
        self.server.PromptServer.instance = self.instance_before

    def install(self, prompt, run_number=7):
        queue = FakeQueue((run_number, "pid", prompt, {"client_id": "c"}, ["10"], {"key": "secret"}))
        self.server.PromptServer.instance = types.SimpleNamespace(prompt_queue=queue, number=5, routes=None)
        return queue

    def test_fresh_queue_resets_and_requeue_keeps_state(self):
        node = meta.LunaMetaBatch()
        (manager,) = node.update_batch(22, prompt=graph(), unique_id="1")
        self.assertIs(manager, node)
        manager.outputs["10"] = (1, None)
        manager.total_frames = 66
        node.update_batch(22, prompt=graph(requeue=1), unique_id="1")
        self.assertIn("10", manager.outputs)
        node.update_batch(22, prompt=graph(requeue=0), unique_id="1")
        self.assertEqual(manager.outputs, {})
        self.assertEqual(manager.frames_per_batch, 22)

    def test_reset_closes_an_open_output(self):
        closed = []

        def writer():
            while True:
                frame = yield
                if frame is None:
                    closed.append(True)
                    return

        open_writer = writer()
        open_writer.send(None)
        node = meta.LunaMetaBatch(22)
        node.outputs["10"] = (1, open_writer)
        node.reset()
        self.assertEqual(closed, [True])
        self.assertEqual(node.outputs, {})

    def test_requeue_waits_for_every_managed_output(self):
        queue = self.install(graph(outputs=2))
        meta.requeue_workflow(("1", True))
        self.assertEqual(queue.put_items, [])
        meta.requeue_workflow(("1", True))
        self.assertEqual(len(queue.put_items), 1)
        number, prompt_id, prompt, extra, outputs, sensitive = queue.put_items[0]
        self.assertEqual(number, -5)
        self.assertEqual(prompt["1"]["inputs"]["requeue"], 1)
        self.assertEqual(prompt["2"]["inputs"], graph()["2"]["inputs"])
        self.assertEqual((extra, outputs, sensitive), ({"client_id": "c"}, ["10"], {"key": "secret"}))
        self.assertNotEqual(prompt_id, "pid")

    def test_running_prompt_is_not_changed(self):
        running = graph()
        self.install(running)
        meta.requeue_workflow(("1", True))
        self.assertEqual(running["1"]["inputs"]["requeue"], 0)

    def test_no_requeue_after_the_last_batch(self):
        queue = self.install(graph())
        meta.requeue_workflow(("1", False))
        self.assertEqual(queue.put_items, [])


if __name__ == "__main__":
    unittest.main()
