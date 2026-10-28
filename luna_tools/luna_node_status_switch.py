# Derived from ComfyUI-DaSiWa-Nodes (https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes),
# commit ab364c5, GPL-3.0 (see THIRD_PARTY_NOTICES.md). Source file: nodes/nodes_status_switch.py
# Changes: renamed to Luna-StatusSwitch (category Luna-Media-Tools); DESCRIPTION rewritten; input
# 'description' keys became 'tooltip'; OUTPUT_TOOLTIPS added; registered as a V1 node.
"""
Luna-StatusSwitch

Mutes or bypasses target nodes based on a boolean toggle.

Targets are identified by wiring any output from the node you want
to control into one of this node's target inputs.  Inputs appear
dynamically: only one empty slot is shown at a time.  When you
connect it, the next slot appears (up to 99).

The switch exposes a single BOOLEAN output (`enabled_out`) carrying
its effective `enabled` value so a chain of switches can be driven
from a single upstream toggle.  Each switch in a chain still applies
its own `trigger_on` and `action` independently.
"""


class AnyType(str):
    """Matches any ComfyUI type so target inputs accept anything."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return hash("*")


ANY = AnyType("*")


class LunaStatusSwitch:
    DESCRIPTION = (
        "Luna-StatusSwitch: one on/off switch that turns other nodes on or off.\n\n"
        "Wire any output of a node you want to control into a target socket (a new empty socket appears each "
        "time you connect one). While `enabled` matches `trigger_on` the targets run normally. Otherwise they are "
        "muted (they do not run, and their outputs are missing) or bypassed (they are skipped and their input "
        "passes straight through), as chosen by `action`. The change is applied when you queue the prompt.\n\n"
        "The enabled_out output carries the switch state, so one toggle can drive a chain of switches; each "
        "switch applies its own `trigger_on` and `action`."
    )
    OUTPUT_TOOLTIPS = ("The switch state, to drive another Luna-StatusSwitch.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Master toggle to activate or deactivate the switch."}),
                "trigger_on": (
                    ["true \u2192 active", "false \u2192 active"],
                    {"default": "true \u2192 active", "tooltip": "Define if the switch is active when the boolean is True or False."},
                ),
                "action": (["mute", "bypass"], {"default": "bypass", "tooltip": "Choose whether to Mute (stop execution) or Bypass (pass through) targets."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("enabled_out",)
    FUNCTION = "execute"
    CATEGORY = "Luna-Media-Tools"
    OUTPUT_NODE = False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Legacy ComfyUI validation hook. Target sockets are frontend-only
        # control links, so this node accepts whatever reaches validation.
        return True

    def validate_inputs(self, *args, **kwargs):
        # New ComfyUI validation hook. Kept as a catch-all for compatibility
        # with internal-node validation and future ComfyUI core updates.
        return True

    def execute(self, enabled, trigger_on, action, unique_id=None, **kwargs):
        # Pass the effective enabled value through to any chained switch.
        # Downstream switches apply their own trigger_on / action logic.
        return (bool(enabled),)


NODE_CLASS_MAPPINGS = {"Luna-StatusSwitch": LunaStatusSwitch}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-StatusSwitch": "Luna-StatusSwitch"}
