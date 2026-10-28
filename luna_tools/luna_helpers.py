"""Small helpers the watermark node needs (the originals lived in ComfyUI-DaSiWa-Nodes'
helper_batch_output.py and helper_logging.py, which were never copied into LunaTools).
Written fresh here, same call shape."""

import os
import tempfile

import numpy as np
import torch

# An output bigger than this is backed by a file in ComfyUI's temp folder instead of RAM.
DISK_BACKED_ABOVE_BYTES = 2 * 1024 ** 3

_NP_DTYPES = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
              torch.uint8: np.uint8}


def allocate_cpu_output(shape, dtype, temp_dir=None):
    """An empty CPU tensor of `shape`. Returns (tensor, mmap_path); mmap_path is None
    when the tensor lives in RAM, else the path of the backing file (ComfyUI's temp
    folder is cleaned on restart)."""
    shape = tuple(int(x) for x in shape)
    nbytes = int(np.prod(shape, dtype=np.int64)) * torch.empty(0, dtype=dtype).element_size()
    np_dtype = _NP_DTYPES.get(dtype)
    if nbytes <= DISK_BACKED_ABOVE_BYTES or np_dtype is None:
        return torch.empty(shape, dtype=dtype, device="cpu"), None
    folder = temp_dir or tempfile.gettempdir()
    os.makedirs(folder, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="luna_wm_", suffix=".bin", dir=folder)
    os.close(fd)
    arr = np.memmap(path, dtype=np_dtype, mode="w+", shape=shape)
    return torch.from_numpy(arr), path


def log_luna(scope, message):
    print(f"[Luna-Media-Tools/{scope}] {message}")
