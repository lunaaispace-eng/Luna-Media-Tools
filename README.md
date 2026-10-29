# Luna-Media-Tools

General image and video tools for ComfyUI, ten nodes in one pack: save an image, save a video on the
GPU, load a video with its sound, cast a batch to fp16, remove the VAE pixel grid, draw an Ideogram 4
prompt on a canvas, overlay a watermark, switch other nodes on and off, upscale with NVIDIA RTX VFX, and add
in-between frames with RTX frame generation.

Every node's id is its display name, and every node sits in the category `Luna-Media-Tools`.
Each node has an (i) help button on its title bar, and the big ones a chevron that folds the settings away.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/lunaaispace-eng/Luna-Media-Tools
```

Dependencies are in `requirements.txt` (Pillow, numpy, psutil; everything else comes with ComfyUI). The two RTX
nodes also need NVIDIA RTX VFX (`nvvfx`, the `nvidia-vfx` package, 0.2.0.0 or newer) when they run; the pack
loads without it. Install it into the Python that runs ComfyUI, then restart ComfyUI:

```
python_embeded\python.exe -m pip install nvidia-vfx
```

(`python -m pip install nvidia-vfx` for a venv install). They need an NVIDIA RTX GPU and a current driver. If
`nvvfx` is missing, the node's error message gives the same steps.

## Node tree

- **Luna-SaveImage**: Saves images, with optional workflow metadata.
- **Luna-SaveVideo**: Encodes an image batch to video on the GPU (NVENC).
- **Luna-LoadVideo**: Loads a video as frames with its sound; shows the cost first.
- **Luna-ImagePrecision**: Casts an image batch to fp16 or fp32 to save RAM.
- **Luna-Degrid**: Removes the 2px VAE pixel grid; wire after VAE Decode.
- **Luna-Ideogram**: Draw regions on a canvas; builds the Ideogram 4 JSON prompt.
- **Luna-Watermark**: Overlays a watermark on images or video frames.
- **Luna-StatusSwitch**: Mutes or bypasses connected nodes from one on/off switch.
- **Luna-RTXUpscaler**: RTX denoise, deblur and upscale, each with its own strength.
- **Luna-RTXFrameGen**: Adds in-between frames with NVIDIA RTX frame generation.

## Replaces

Luna-Media-Tools replaces ComfyUI-SaveSimple (general nodes), ComfyUI-Ideogram4-Luna and ComfyUI-LunaTools,
and carries a copy of the VAE de-grid from ComfyUI-DeGrid (that standalone pack stays available). Node ids changed
with the rename, so workflows saved with the old nodes need those nodes swapped for the new ones. Remove the old
folders from `custom_nodes` when you install this pack (except ComfyUI-DeGrid, whose node id differs).

## Tests

Run from the `tests` folder with pytest:

```
python -m pytest . --rootdir=.
```

(`python -m unittest discover -s tests` runs a part of them.) The registration test needs a ComfyUI folder on the
path; set `COMFYUI_DIR` if it is not `E:\ComfyUI-Easy-Install\ComfyUI`.

## Licence

GPL-3.0, see [LICENSE](LICENSE). Derived code and its licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The three `luna_*.mjs` frontend kit files are Apache-2.0.
