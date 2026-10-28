"""The pack registers exactly its 10 nodes, node id = display name, one category,
and loads without NVIDIA RTX VFX (nvvfx) installed.

    E:/ComfyUI-Easy-Install/python_embeded/python.exe -m unittest discover -s tests -v
Needs ComfyUI on sys.path (folder_paths, comfy_api, server); the check inserts it.
"""

import importlib.util
import os
import sys
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_DIR", r"E:\ComfyUI-Easy-Install\ComfyUI")

IDS = ["Luna-SaveImage", "Luna-SaveVideo", "Luna-LoadVideo", "Luna-ImagePrecision", "Luna-Degrid",
       "Luna-Ideogram", "Luna-Watermark", "Luna-StatusSwitch", "Luna-RTXUpscaler", "Luna-RTXFrameGen"]


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
                  "luna_node_status_switch.js": "Luna-StatusSwitch", "ideogram4_prompt_builder.js": "Luna-Ideogram",
                  "luna_rtx_ui.js": "Luna-RTXUpscaler"}
        for name, node_id in wanted.items():
            with open(os.path.join(web, name), encoding="utf8") as f:
                self.assertIn(f'"{node_id}"', f.read(), name)
        with open(os.path.join(web, "luna_pack_theme.js"), encoding="utf8") as f:
            listed = set(re.findall(r'"(Luna-[A-Za-z]+)"', "\n".join(
                l for l in f.read().splitlines() if not l.lstrip().startswith("//"))))
        self.assertTrue(listed <= set(IDS), listed - set(IDS))

    def test_every_node_has_help_and_collapse(self):
        import re
        with open(os.path.join(PACK, "web", "luna_pack_theme.js"), encoding="utf8") as f:
            code = "\n".join(l for l in f.read().splitlines() if not l.lstrip().startswith("//"))
        help_part = code[code.index("registerLunaHelp(app"):code.index("registerLunaCollapse(app")]
        collapse_part = code[code.index("registerLunaCollapse(app"):]
        self.assertEqual(set(re.findall(r'"(Luna-[A-Za-z]+)"', help_part)), set(IDS))
        for node_id in ("Luna-Ideogram", "Luna-Watermark", "Luna-StatusSwitch", "Luna-RTXUpscaler",
                        "Luna-RTXFrameGen"):
            self.assertIn(f'"{node_id}"', collapse_part)

    def test_every_node_has_a_description(self):
        for node_id, cls in self.pack.NODE_CLASS_MAPPINGS.items():
            desc = getattr(cls, "DESCRIPTION", None)
            if desc is None and hasattr(cls, "GET_SCHEMA"):
                desc = cls.GET_SCHEMA().description
            self.assertTrue(desc and len(str(desc).strip()) > 20, node_id)


if __name__ == "__main__":
    unittest.main()
