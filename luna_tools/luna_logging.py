# Derived from ComfyUI-DaSiWa-Nodes (https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes),
# commit ab364c5, GPL-3.0 (see THIRD_PARTY_NOTICES.md). Source file: nodes/helper_logging.py
# Changes: log_dasiwa -> log_luna; the prefix is the node's own name ("[Luna-Watermark]");
# the upstream startup banner is not carried over.
import sys

# Windows consoles default to cp1252. UTF-8 with replacement keeps printing non-ASCII
# safe on any console code page.
for _stream in (sys.__stdout__, sys.__stderr__):
    if _stream is not None:
        _stream.reconfigure(encoding="utf-8", errors="replace")

_DARK_YELLOW = "\033[38;5;136m"
_RESET = "\033[0m"


def log_luna(node_name: str, message: str) -> None:
    """Print `message` prefixed with the node's own name, e.g. log_luna("Luna-Watermark", "...")."""
    print(f"{_DARK_YELLOW}[{node_name}]{_RESET} {message}")
