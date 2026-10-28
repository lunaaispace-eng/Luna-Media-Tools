"""Luna-Media-Tools — general image and video tools for ComfyUI.

Ten nodes, node id = display name, category "Luna-Media-Tools":
  Luna-SaveImage, Luna-SaveVideo, Luna-LoadVideo, Luna-ImagePrecision, Luna-Degrid,
  Luna-Ideogram, Luna-Watermark, Luna-StatusSwitch, Luna-RTXUpscale, Luna-RTXVideoFinisher.

All V1 mappings (Luna-Ideogram is a V3 class registered through the V1 mapping, as the
standalone pack did). The frontend lives in one folder, web/.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .video_nodes import (
    NODE_CLASS_MAPPINGS as _VIDEO_C,
    NODE_DISPLAY_NAME_MAPPINGS as _VIDEO_D,
)
from .load_video import (
    NODE_CLASS_MAPPINGS as _LOAD_C,
    NODE_DISPLAY_NAME_MAPPINGS as _LOAD_D,
)
from .image_precision import (
    NODE_CLASS_MAPPINGS as _PRECISION_C,
    NODE_DISPLAY_NAME_MAPPINGS as _PRECISION_D,
)
from .degrid_nodes import (
    NODE_CLASS_MAPPINGS as _DEGRID_C,
    NODE_DISPLAY_NAME_MAPPINGS as _DEGRID_D,
)
from .ideogram4_builder import Ideogram4PromptBuilderLuna
from .luna_tools import (
    NODE_CLASS_MAPPINGS as _TOOLS_C,
    NODE_DISPLAY_NAME_MAPPINGS as _TOOLS_D,
)

for _c, _d in (
    (_VIDEO_C, _VIDEO_D),
    (_LOAD_C, _LOAD_D),
    (_PRECISION_C, _PRECISION_D),
    (_DEGRID_C, _DEGRID_D),
    (_TOOLS_C, _TOOLS_D),
):
    NODE_CLASS_MAPPINGS.update(_c)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d)

NODE_CLASS_MAPPINGS["Luna-Ideogram"] = Ideogram4PromptBuilderLuna
NODE_DISPLAY_NAME_MAPPINGS["Luna-Ideogram"] = "Luna-Ideogram"

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
