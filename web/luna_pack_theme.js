// Applies the Luna house theme to this pack's nodes.
//
// Roles come from the section tints in luna_theme.mjs: a loader is an input, the
// save nodes are outputs. One line per node — copy this file into another pack
// and change the map.

import { app } from "../../scripts/app.js";
import { registerLunaTheme } from "./luna_theme.mjs";
import { registerLunaHelp } from "./luna_help.mjs";
import { registerLunaCollapse } from "./luna_collapse.mjs";

// Title-bar tinting is off for now — Peti prefers ComfyUI's default red header,
// with the widget's own amber/green palette inside. The role map stays here so
// turning it back on is uncommenting one call; the colours are also editable in
// Settings > Luna > Node colours.
//
// registerLunaTheme(app, {
//     "Luna-LoadVideo": "input",
//     "Luna-SaveImage": "output",
//     "Luna-SaveVideo": "output",
//     "Luna-Ideogram": "processing",
//     "Luna-Watermark": "processing",
//     "Luna-StatusSwitch": "processing",
//     "Luna-RTXUpscaler": "processing",
//     "Luna-RTXFrameGen": "processing",
// }, "LunaMediaTools.Theme");

// The ⓘ on the title bar. Content is each node's Python DESCRIPTION plus its
// input tooltips, so there is no separate help file to keep up to date.
registerLunaHelp(app, [
    "Luna-LoadVideo", "Luna-SaveImage", "Luna-SaveVideo", "Luna-ImagePrecision", "Luna-Degrid",
    "Luna-Ideogram", "Luna-Watermark", "Luna-StatusSwitch", "Luna-RTXUpscaler", "Luna-RTXFrameGen",
], "LunaMediaTools.Help");

// The chevron beside it folds the settings away, leaving sockets and whatever the
// node draws for itself — Luna-SaveImage keeps its preview, Luna-SaveVideo keeps the
// player, the frame checkboxes, Autoplay and Download. Luna-Ideogram keeps its canvas editor,
// Luna-RTXUpscaler and Luna-RTXFrameGen have only ordinary widgets, so folding leaves just the sockets.
registerLunaCollapse(app, [
    "Luna-SaveImage", "Luna-SaveVideo", "Luna-LoadVideo",
    "Luna-Ideogram", "Luna-Watermark", "Luna-StatusSwitch", "Luna-RTXUpscaler", "Luna-RTXFrameGen",
], "LunaMediaTools.Collapse");
