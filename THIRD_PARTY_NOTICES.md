# Third-party notices

Luna-Media-Tools is licensed GPL-3.0 (see [LICENSE](LICENSE)). It contains code derived
from the projects below. Each header in the source files names what was changed.

| Project | Licence | What is derived from it | Where |
|---|---|---|---|
| [ComfyUI-DaSiWa-Nodes](https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes), commit `ab364c5` (2026-10-26) | GPL-3.0 (its `LICENSE`; the repo moved from Apache-2.0 to GPL-3.0 on 08-04) | Luna-Watermark (`nodes/nodes_watermark.py`), its helpers `nodes/helper_batch_output.py` and `nodes/helper_logging.py`, Luna-StatusSwitch (`nodes/nodes_status_switch.py`, `js/node_status_switch.js`), Luna-RTXUpscaler (`nodes/nodes_rtx_upscaler_refiner.py`), Luna-RTXFrameGen (`nodes/nodes_rtx_frame_generation.py`), the RTX widget logic (`js/dasiwa_ui.js`, RTX part only) and the RTX tests (`tests/test_rtx_regular.py`, `test_rtx_frame_generation.py`, `test_rtx_ram_reclamation.py`, `test_rtx_tooltips.py`, `test_rtx_ui.mjs`) | `luna_tools/luna_watermark.py`, `luna_tools/luna_batch_output.py`, `luna_tools/luna_logging.py`, `luna_tools/luna_node_status_switch.py`, `web/luna_node_status_switch.js`, `luna_tools/luna_rtx_upscaler.py`, `luna_tools/luna_rtx_frame_gen.py`, `web/luna_rtx_ui.js`, `tests/test_rtx_upscaler.py`, `tests/test_rtx_frame_gen.py`, `tests/test_rtx_ram_reclamation.py`, `tests/test_rtx_tooltips.py`, `tests/test_rtx_ui.mjs` |
| ComfyUI-DaSiWa-Nodes, an earlier Apache-2.0 version | Apache-2.0 | Luna-SaveVideo (from `nodes_enhanced_video_combine.py`) and its preview script | `video_nodes.py`, `web/save_video_simple_preview.js` |
| Load Video Pixaroma, in [ComfyUI-Pixaroma](https://github.com/pixaroma/ComfyUI-Pixaroma) (c) 2026 pixaroma | MIT | Luna-LoadVideo: the job, the seven outputs and their order, the player idea. Decoding and the rest are this pack's own | `load_video.py`, `web/luna_load_video.js` |
| Ideogram 4 Prompt Builder in KJNodes, by Kijai | GPL-3.0 | Luna-Ideogram is a fork of it | `ideogram4_builder.py`, `web/ideogram4_prompt_builder.js`, `web/utility.js`, `fonts/FreeMono.ttf` |

The RTX nodes drive NVIDIA RTX VFX (`nvvfx`), which is NVIDIA's software under its own
terms and is not part of this pack.

## Licence of the shared frontend kit

`web/luna_theme.mjs`, `web/luna_help.mjs` and `web/luna_collapse.mjs` (and `web/luna_pack_theme.js`,
which uses them) are Apache-2.0, Copyright 2026 Luna AI Space. They stay Apache-2.0 inside this
GPL-3.0 pack; Apache-2.0 code may be combined into a GPL-3.0 work. Copy them into another pack under
Apache-2.0.

## What each licence asks

- Apache-2.0 (the earlier DaSiWa version Luna-SaveVideo was taken from): keep the notice and state changes.
  The change list of Luna-SaveVideo is in the header of `video_nodes.py`.
- MIT (Pixaroma): keep the copyright notice and permission notice. The changes are listed in the
  header of `load_video.py`.
- GPL-3.0 (KJNodes, DaSiWa): the combined work is GPL-3.0 and its source stays available.
