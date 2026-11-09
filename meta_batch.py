"""Luna-MetaBatch — run the graph once per batch, so one file can be written across several queued runs.

Port of the Meta Batch Manager and its re-queue from ComfyUI-VideoHelperSuite
(https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), GPL-3.0, commit 4d907be:
`videohelpersuite/nodes.py` class BatchManager and `videohelpersuite/utils.py` requeue_workflow /
requeue_workflow_unchecked.

Changes from the original:
  - renamed the node (now Luna-MetaBatch) and moved it to the Luna-Media-Tools category; the socket type stays
    `VHS_BatchManager`, so Luna-H3-Chain and saved workflows connect to either manager unchanged
  - the re-queue counts Luna-SaveVideo nodes wired to a Luna-MetaBatch (VHS counts VHS Video Combine on its own manager)
  - VHS's frontend `count` widget, which made the manager run again on every user queue, is replaced by IS_CHANGED
  - the queue is read from ComfyUI's server when the re-queue runs, not when the module is imported

How a meta-batch run goes: the manager runs first in every run. On a fresh queue (`requeue` 0) it resets. A batch source
(Luna-H3-Chain) reads `requeue` as its batch number and sets `has_closed_inputs` on the last one. The output node
(Luna-SaveVideo) keeps its open file in `outputs[unique_id]` and calls `requeue_workflow`, which queues the same graph
again with `requeue` + 1 on the manager, until the inputs are closed.
"""

NODE_ID = "Luna-MetaBatch"
SOCKET_TYPE = "VHS_BatchManager"
MANAGER_IDS = (NODE_ID,)
MANAGED_OUTPUT_IDS = ("Luna-SaveVideo",)
BIGMAX = 2 ** 53 - 1


class LunaMetaBatch:
    """The manager. ComfyUI keeps one instance per node across runs, and that instance is what flows on the socket."""

    DESCRIPTION = (
        "Runs the graph again and again, once per batch, so one video can be written across several queued runs. "
        "Wire meta_batch into Luna-H3-Chain (it counts the clips and says when the last one is done) and into "
        "Luna-SaveVideo (it keeps one file open until then). Port of VHS's Meta Batch Manager; same socket type."
    )

    def __init__(self, frames_per_batch=-1):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
        self.total_frames = float("inf")

    def reset(self):
        """Close every input and output still open (a stopped or failed run) and start clean."""
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)

    def has_open_inputs(self):
        return len(self.inputs) > 0

    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_per_batch": ("INT", {"default": 16, "min": 1, "max": BIGMAX, "step": 1,
                    "tooltip": "Frames in each run. For Luna-H3-Chain: every clip's length (17k + 5 frames)."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (SOCKET_TYPE,)
    RETURN_NAMES = ("meta_batch",)
    OUTPUT_TOOLTIPS = ("The manager. Connect to Luna-H3-Chain.meta_batch and Luna-SaveVideo.meta_batch.",)
    FUNCTION = "update_batch"
    CATEGORY = "Luna-Media-Tools"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A new queue from the user must reach update_batch with requeue 0 and reset the manager.
        return float("nan")

    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
        requeue = 0
        if unique_id is not None and prompt is not None:
            requeue = int(prompt[str(unique_id)]["inputs"].get("requeue", 0))
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        else:
            batches = (self.total_frames + self.frames_per_batch - 1) // frames_per_batch
            print(f"[{NODE_ID}] batch {requeue + 1} of {batches}")
        return (self,)


def _prompt_queue():
    import server

    return server.PromptServer.instance


def _running(queue):
    value = next(iter(queue.currently_running.values()))
    # ComfyUI's queue items are (number, prompt_id, prompt, extra_data, outputs_to_execute[, sensitive]).
    return value


def requeue_workflow_unchecked():
    """Queue the running graph again, with every Luna-MetaBatch's requeue count one higher."""
    instance = _prompt_queue()
    value = _running(instance.prompt_queue)
    prompt, extra_data, outputs_to_execute = value[2], value[3], value[4]
    sensitive = value[5] if len(value) > 5 else {}
    prompt = prompt.copy()
    for uid in prompt:
        if prompt[uid]["class_type"] in MANAGER_IDS:
            node = dict(prompt[uid])
            node["inputs"] = dict(node["inputs"])
            node["inputs"]["requeue"] = node["inputs"].get("requeue", 0) + 1
            prompt[uid] = node
    # Negative numbers go to the front of the queue, ahead of anything the user queued meanwhile.
    number = -instance.number
    instance.number += 1
    import uuid

    instance.prompt_queue.put((number, str(uuid.uuid4()), prompt, extra_data, outputs_to_execute, sensitive))


_guard = [None, 0, 0, {}]


def requeue_workflow(requeue_required=(-1, True)):
    """Called once per run by every managed output. The graph is queued again when the last of them has called and
    at least one still needs another batch."""
    global _guard
    queue = _prompt_queue().prompt_queue
    if len(queue.currently_running) != 1:
        raise RuntimeError(f"[{NODE_ID}] needs exactly one running prompt to queue the next batch.")
    value = _running(queue)
    run_number, prompt = value[0], value[2]
    if _guard[0] != run_number:
        managed = 0
        for manager_uid in prompt:
            if prompt[manager_uid]["class_type"] not in MANAGER_IDS:
                continue
            for output_uid in prompt:
                if prompt[output_uid]["class_type"] in MANAGED_OUTPUT_IDS:
                    managed += sum(1 for value_in in prompt[output_uid]["inputs"].values()
                                   if value_in == [manager_uid, 0])
        _guard = [run_number, 0, managed, {}]
    _guard[1] += 1
    _guard[3][requeue_required[0]] = requeue_required[1]
    if _guard[1] == _guard[2] and max(_guard[3].values()):
        requeue_workflow_unchecked()


NODE_CLASS_MAPPINGS = {NODE_ID: LunaMetaBatch}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: NODE_ID}
