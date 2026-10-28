"""Luna-StatusSwitch, Luna-Watermark, Luna-RTXUpscaler, Luna-RTXFrameGen.

The RTX nodes need NVIDIA RTX VFX (`nvvfx`) only when they run (nvvfx is imported inside
the functions that use it), so this package loads without it."""

from .luna_node_status_switch import (NODE_CLASS_MAPPINGS as _SWITCH_C,
                                      NODE_DISPLAY_NAME_MAPPINGS as _SWITCH_D)
from .luna_watermark import NODE_CLASS_MAPPINGS as _WM_C, NODE_DISPLAY_NAME_MAPPINGS as _WM_D
from .luna_rtx_upscaler import NODE_CLASS_MAPPINGS as _UP_C, NODE_DISPLAY_NAME_MAPPINGS as _UP_D
from .luna_rtx_frame_gen import NODE_CLASS_MAPPINGS as _FG_C, NODE_DISPLAY_NAME_MAPPINGS as _FG_D

NODE_CLASS_MAPPINGS = {**_SWITCH_C, **_WM_C, **_UP_C, **_FG_C}
NODE_DISPLAY_NAME_MAPPINGS = {**_SWITCH_D, **_WM_D, **_UP_D, **_FG_D}
