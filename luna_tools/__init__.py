"""Luna-StatusSwitch, Luna-Watermark, Luna-RTXUpscale, Luna-RTXVideoFinisher.

The RTX nodes need NVIDIA RTX VFX (`nvvfx`) only when they RUN: it is imported inside
_import_vfx() in luna_rtx_upscale.py, so this package loads without it."""

from .luna_node_status_switch import (NODE_CLASS_MAPPINGS as _SWITCH_C,
                                      NODE_DISPLAY_NAME_MAPPINGS as _SWITCH_D)
from .luna_watermark import NODE_CLASS_MAPPINGS as _WM_C, NODE_DISPLAY_NAME_MAPPINGS as _WM_D
from .luna_rtx_upscale import NODE_CLASS_MAPPINGS as _UP_C, NODE_DISPLAY_NAME_MAPPINGS as _UP_D
from .luna_rtx_video_finisher import (NODE_CLASS_MAPPINGS as _FIN_C,
                                      NODE_DISPLAY_NAME_MAPPINGS as _FIN_D)

NODE_CLASS_MAPPINGS = {**_SWITCH_C, **_WM_C, **_UP_C, **_FIN_C}
NODE_DISPLAY_NAME_MAPPINGS = {**_SWITCH_D, **_WM_D, **_UP_D, **_FIN_D}
