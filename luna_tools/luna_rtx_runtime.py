# Derived from the Deno RTX VFX nodes, comfyui-deno-custom-nodes
# (https://github.com/Deno2026/comfyui-deno-custom-nodes), commit 0cd75f4, GPL-3.0
# (see THIRD_PARTY_NOTICES.md). Source file: deno_rtx_vfx_runtime.py
# Changes: marker file renamed to Luna_RTX_VFX_runtime_path.txt (looked for in this pack's
# luna_tools/tools folder); the LUNA_RTX_VFX_RUNTIME environment variable can name the runtime
# folder instead; added find_installed_nvvfx() / installed_nvvfx_version(), which look at the
# nvvfx that is already installed in this Python WITHOUT importing it. Nothing here imports
# nvvfx, so the pack loads without it.
import os
import sys
from pathlib import Path
from typing import Optional


RUNTIME_MARKER_NAME = "Luna_RTX_VFX_runtime_path.txt"
RUNTIME_ENV_VAR = "LUNA_RTX_VFX_RUNTIME"


def expected_python_runtime_segment() -> str:
    return f"py{sys.version_info[0]}{sys.version_info[1]}"


def runtime_path_matches_current_python(runtime_path: Path) -> bool:
    expected_segment = expected_python_runtime_segment()
    return any(part.lower() == expected_segment for part in runtime_path.parts)


def runtime_marker_path(package_dir: Optional[Path] = None) -> Path:
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    return root / "tools" / RUNTIME_MARKER_NAME


def find_installed_nvvfx() -> Optional[Path]:
    """The folder of the nvvfx package that this Python already has, found without importing it
    (importing nvvfx loads native DLLs, which must wait until a node runs)."""
    if "nvvfx" in sys.modules:
        return current_nvvfx_package_path()
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvvfx")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0])
    return Path(spec.origin).parent if spec.origin else None


def installed_nvvfx_version() -> Optional[str]:
    """Version of the nvidia-vfx package installed in this Python, or None."""
    try:
        import importlib.metadata as metadata

        return metadata.version("nvidia-vfx")
    except Exception:
        return None


def read_rtx_vfx_runtime_path(package_dir: Optional[Path] = None) -> Optional[Path]:
    """The prepared runtime folder (the one that contains `nvvfx`), if one was set up.
    Set by the LUNA_RTX_VFX_RUNTIME environment variable, else by the marker file. When neither
    is set the nvvfx already installed in this Python is used."""
    raw_path = os.environ.get(RUNTIME_ENV_VAR, "").strip().strip('"')
    if not raw_path:
        marker = runtime_marker_path(package_dir)
        try:
            raw_path = marker.read_text(encoding="utf-8").strip().strip('"')
        except OSError:
            return None

    if not raw_path:
        return None

    runtime_path = Path(os.path.expandvars(raw_path))
    if not (runtime_path / "nvvfx").is_dir():
        return None
    if not runtime_path_matches_current_python(runtime_path):
        return None

    return runtime_path


def _norm_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def is_path_relative_to(child: Path, parent: Path) -> bool:
    return _is_relative_to(child, parent)


def current_nvvfx_package_path() -> Optional[Path]:
    module = sys.modules.get("nvvfx")
    if module is None:
        return None

    module_paths = getattr(module, "__path__", None)
    if module_paths:
        for module_path in module_paths:
            return Path(module_path)

    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).parent

    return None


def loaded_nvvfx_module_paths() -> dict[str, str]:
    module_paths = {}
    for module_name in ("nvvfx", "nvvfx._ext", "nvvfx._lib_loader"):
        module = sys.modules.get(module_name)
        if module is None:
            continue

        module_file = getattr(module, "__file__", None)
        module_path = getattr(module, "__path__", None)
        if module_file:
            module_paths[module_name] = str(module_file)
        elif module_path:
            module_paths[module_name] = ";".join(str(path) for path in module_path)
        else:
            module_paths[module_name] = "loaded"

    return module_paths


def loaded_process_module_paths() -> list[str]:
    if sys.platform != "win32":
        return []

    try:
        import ctypes
        from ctypes import wintypes

        hmodule = getattr(wintypes, "HMODULE", wintypes.HANDLE)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EnumProcessModulesEx.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(hmodule),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
        ]
        psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, hmodule, wintypes.LPWSTR, wintypes.DWORD]
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD
        handle = kernel32.GetCurrentProcess()

        list_modules_all = 0x03
        needed = wintypes.DWORD()
        modules = (hmodule * 2048)()
        if not psapi.EnumProcessModulesEx(
            handle,
            modules,
            ctypes.sizeof(modules),
            ctypes.byref(needed),
            list_modules_all,
        ):
            return []

        count = min(int(needed.value / ctypes.sizeof(hmodule)), len(modules))
        paths = []
        for module in modules[:count]:
            buffer = ctypes.create_unicode_buffer(32768)
            length = psapi.GetModuleFileNameExW(handle, module, buffer, len(buffer))
            if length:
                paths.append(buffer.value)
        return paths
    except Exception:
        return []


def loaded_broadcast_vfx_module_paths() -> list[str]:
    broadcast_paths = []
    for path in loaded_process_module_paths():
        lowered = _norm_path(Path(path))
        if "\\programdata\\nvidia\\ngx\\models\\nvbcast\\" in lowered or "\\nvbcast\\versions\\" in lowered:
            broadcast_paths.append(path)
    return broadcast_paths


def _move_to_front(runtime_path: Path) -> None:
    runtime_path_text = str(runtime_path)
    runtime_key = _norm_path(runtime_path)
    sys.path[:] = [entry for entry in sys.path if _norm_path(Path(entry or os.curdir)) != runtime_key]
    sys.path.insert(0, runtime_path_text)


def prefer_rtx_vfx_runtime_path(
    package_dir: Optional[Path] = None,
) -> Optional[Path]:
    runtime_path = read_rtx_vfx_runtime_path(package_dir)
    if runtime_path is None:
        return None

    _move_to_front(runtime_path)
    return runtime_path
