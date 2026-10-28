"""The pack registers exactly its 10 nodes, node id = display name, one category,
and loads without NVIDIA RTX VFX (nvvfx) installed.

    E:/ComfyUI-Easy-Install/python_embeded/python.exe -m unittest discover -s tests -v
Needs ComfyUI on sys.path (folder_paths, comfy_api, server); the check inserts it.
"""

import importlib.util
import os
import sys
import tempfile
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_DIR", r"E:\ComfyUI-Easy-Install\ComfyUI")

IDS = ["Luna-SaveImage", "Luna-SaveVideo", "Luna-LoadVideo", "Luna-ImagePrecision", "Luna-Degrid",
       "Luna-Ideogram", "Luna-Watermark", "Luna-StatusSwitch", "Luna-RTXUpscale", "Luna-RTXVideoFinisher"]


def load_pack():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    from aiohttp import web
    import server
    if not hasattr(server.PromptServer, "instance"):   # no running ComfyUI in a test
        server.PromptServer.instance = type("PS", (), {"routes": web.RouteTableDef(),
                                                       "send_sync": lambda *a, **k: None})()
    spec = importlib.util.spec_from_file_location("Luna_Media_Tools_under_test",
                                                  os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("nvvfx", None)
        cls.pack = load_pack()

    def test_ten_ids(self):
        self.assertEqual(sorted(self.pack.NODE_CLASS_MAPPINGS), sorted(IDS))

    def test_id_is_display_name(self):
        self.assertEqual(dict(self.pack.NODE_DISPLAY_NAME_MAPPINGS), {i: i for i in IDS})

    def test_one_category(self):
        for node_id, cls in self.pack.NODE_CLASS_MAPPINGS.items():
            self.assertEqual(getattr(cls, "CATEGORY", None), "Luna-Media-Tools", node_id)

    def test_rtx_runtime_is_lazy(self):
        self.assertNotIn("nvvfx", sys.modules)

    def test_web_folder_exists(self):
        self.assertTrue(os.path.isdir(os.path.join(PACK, self.pack.WEB_DIRECTORY)))

    def test_every_js_node_name_is_a_registered_id(self):
        import re
        web = os.path.join(PACK, "web")
        wanted = {"save_video_simple_preview.js": "Luna-SaveVideo", "luna_load_video.js": "Luna-LoadVideo",
                  "luna_node_status_switch.js": "Luna-StatusSwitch", "ideogram4_prompt_builder.js": "Luna-Ideogram"}
        for name, node_id in wanted.items():
            with open(os.path.join(web, name), encoding="utf8") as f:
                self.assertIn(f'"{node_id}"', f.read(), name)
        with open(os.path.join(web, "luna_pack_theme.js"), encoding="utf8") as f:
            listed = set(re.findall(r'"(Luna-[A-Za-z]+)"', "\n".join(
                l for l in f.read().splitlines() if not l.lstrip().startswith("//"))))
        self.assertTrue(listed <= set(IDS), listed - set(IDS))


class HelperTests(unittest.TestCase):
    def test_allocate_cpu_output_small_is_ram(self):
        import torch
        load_pack()
        from Luna_Media_Tools_under_test.luna_tools.luna_helpers import allocate_cpu_output
        out, path = allocate_cpu_output((2, 4, 4, 3), torch.float16, tempfile.gettempdir())
        self.assertEqual(tuple(out.shape), (2, 4, 4, 3))
        self.assertEqual(out.dtype, torch.float16)
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
