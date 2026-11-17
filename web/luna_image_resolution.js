// Luna-ImageResolution — compact face with a live readout, as on Luna-H3-Canvas.
//
// Face: category, preset, and the readout (line 1 the size large, ratio and MP beside it; line 2 what produced it, with a
// portrait chip and ⚙; line 3 in red, only for warnings). The ⚙ panel (luna_cards.mjs openSettings) holds size_mode,
// megapixels, longest_side, snap and batch, hidden from the face, and shows the custom-size file.
//
// Python is the one source of the maths: the readout asks image_resolution.compute() through POST /luna_media/resolution,
// and the preset lists come from GET /luna_media/resolution/presets. Nothing here may throw into the canvas draw.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LUNA } from "./luna_theme.mjs";
import { injectCardsCSS, openSettings, widget } from "./luna_cards.mjs";

const NODE = "Luna-ImageResolution";
const PANEL_WIDGETS = ["portrait", "size_mode", "megapixels", "longest_side", "snap", "batch"];
const SNAPS = ["8", "16", "32", "64"];

let _presets = null;            // { categories: {name: [labels]}, custom_file }
let _presetsLoading = null;     // the one request in flight, shared by every caller
function loadPresets(force = false) {
    if (_presets && !force) return Promise.resolve(_presets);
    if (_presetsLoading) return _presetsLoading;
    _presetsLoading = api.fetchApi("/luna_media/resolution/presets")
        .then(async (r) => { if (r.ok) _presets = await r.json(); return _presets; })
        .catch(() => _presets)   // the readout says the server is unreachable
        .finally(() => { _presetsLoading = null; });
    return _presetsLoading;
}

// Wired image: its size is known from the last run (ui.luna_image_size); until then the readout says so.
const imageKnown = (node) => !imageWired(node) || !!node._lunaImageSize;

const val = (node, name) => widget(node, name)?.value;
const imageWired = (node) => node.inputs?.some((i) => i.name === "image" && i.link != null);

function guard(label, fn, fallback) {
    try { return fn(); } catch (e) {
        if (!guard.seen.has(label)) { guard.seen.add(label); console.error(`[${NODE}] ${label} failed.`, e); }
        return fallback;
    }
}
guard.seen = new Set();

// The preset list follows the category; a value that left the list moves to the first entry.
async function syncPresets(node) {
    const p = await loadPresets();
    const w = widget(node, "preset");
    const labels = p?.categories?.[val(node, "category")];
    if (!w || !labels) return;
    if (w.options) w.options.values = labels.length ? labels : [""];
    if (!labels.includes(w.value)) w.value = labels[0] ?? "";
}

function hidePanelWidgets(node) {
    for (const name of PANEL_WIDGETS) {
        const w = widget(node, name);
        if (w) w.hidden = true;
    }
}

// Ask Python for the result, at most one request in flight; a change during it sends one more after.
function refresh(node) {
    node._lunaRes = node._lunaRes || { lines: ["…"], warnings: [] };
    if (node._lunaBusy) { node._lunaAgain = true; return; }
    node._lunaBusy = true;
    const body = {
        category: val(node, "category"), preset: val(node, "preset"), portrait: !!val(node, "portrait"),
        size_mode: val(node, "size_mode"), megapixels: Number(val(node, "megapixels")),
        longest_side: Number(val(node, "longest_side")), snap: Number(val(node, "snap")),
        image_size: imageWired(node) ? (node._lunaImageSize || null) : null,
    };
    api.fetchApi("/luna_media/resolution", { method: "POST", body: JSON.stringify(body) })
        .then((r) => r.json())
        .then((res) => { node._lunaRes = res.error ? { lines: ["—", "", res.error], warnings: [res.error] } : res; })
        .catch(() => { node._lunaRes = { lines: ["—", "", "server not reachable"], warnings: ["server not reachable"] }; })
        .finally(() => {
            node._lunaBusy = false;
            // The warning line adds a row: grow or shrink the node with it.
            guard("resize", () => {
                const h = node.computeSize()[1];
                if (Math.abs(h - node.size[1]) > 1) node.setSize([node.size[0], h]);
            }, undefined);
            node.setDirtyCanvas(true, true);
            if (node._lunaAgain) { node._lunaAgain = false; refresh(node); }
        });
}

const changed = (node) => guard("sync", () => { syncPresets(node).then(() => refresh(node)); }, undefined);

function setWidget(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = value;
    w.callback?.(value);
    changed(node);
}

function openPanel(node) {
    injectCardsCSS();
    loadPresets(true).then(() => {
        openSettings(app, node, `${NODE} settings`, (body, chips, head) => {
            head("▦", "Size");
            chips("Size from", [["native", "Native"], ["megapixels", "Megapixels"], ["longest_side", "Longest side"]],
                  val(node, "size_mode"), (v) => setWidget(node, "size_mode", v),
                  "Native: the preset's own size (a ratio uses the model's or 1 MP).");
            const num = (label, name, step, note) => {
                const r = document.createElement("div");
                r.className = "rw";
                r.innerHTML = `<span class="l"></span><span class="ch"><input class="nb"></span>`;
                r.querySelector(".l").textContent = label;
                const nb = r.querySelector(".nb");
                nb.value = String(val(node, name));
                nb.step = step;
                nb.onchange = () => { const v = Number(nb.value); if (Number.isFinite(v) && v > 0) setWidget(node, name, v); else nb.value = String(val(node, name)); };
                nb.addEventListener("keydown", (e) => { if (e.key === "Enter") nb.blur(); });
                body.appendChild(r);
                if (note) { const n = document.createElement("div"); n.className = "nt"; n.textContent = note; body.appendChild(n); }
            };
            const mode = val(node, "size_mode");
            if (mode === "megapixels") num("Megapixels", "megapixels", "0.05", "1 MP = 1024 × 1024.");
            if (mode === "longest_side") num("Longest side", "longest_side", "8");
            chips("Snap", SNAPS.map((s) => [s, s]), String(val(node, "snap")), (v) => setWidget(node, "snap", v),
                  "Both sides go to the nearest multiple. A model raises it to its own grid.");
            head("▣", "Latent");
            num("Batch", "batch", "1");
            head("✎", "Custom sizes");
            const p = document.createElement("div");
            p.className = "nt path";
            p.style.marginLeft = "0";
            p.textContent = _presets?.custom_file || "user/luna/resolutions.json";
            body.appendChild(p);
            const n = document.createElement("div");
            n.className = "nt";
            n.style.marginLeft = "0";
            n.textContent = "A list of {\"label\", \"width\", \"height\"}; read each time this panel opens. Category Custom.";
            body.appendChild(n);
        }, () => {
            for (const [name, v] of [["size_mode", "native"], ["megapixels", 1.0], ["longest_side", 1024], ["snap", "16"], ["batch", 1]])
                setWidget(node, name, v);
        });
        changed(node);
    });
}

/* ------------------------------------------------------------------ readout */

const FONT = "'Segoe UI', system-ui, sans-serif";

function fit(ctx, text, max) {
    if (ctx.measureText(text).width <= max) return text;
    let t = text;
    while (t.length > 1 && ctx.measureText(t + "…").width > max) t = t.slice(0, -1);
    return t + "…";
}

function readoutHeight(node) {
    const r = node._lunaRes;
    return r?.warnings?.length ? 82 : 64;
}

function drawReadout(ctx, node, y) {
    const r = node._lunaRes || { lines: ["…"], warnings: [] };
    const x = 14, pad = 10;
    const w = Math.max(160, (Number(node.size?.[0]) || 260) - x * 2);
    const h = readoutHeight(node) - 8;
    ctx.save();
    try {
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, 6); else ctx.rect(x, y, w, h);
        ctx.fillStyle = LUNA.panel; ctx.fill();
        ctx.strokeStyle = LUNA.border; ctx.lineWidth = 1; ctx.stroke();

        const [size, ratio, mp] = (r.lines?.[0] || "").split(/\s{2,}/);
        ctx.textBaseline = "alphabetic";
        ctx.textAlign = "left";
        ctx.fillStyle = LUNA.accent;
        ctx.font = `600 16px ${FONT}`;
        ctx.fillText(imageKnown(node) ? (size || "…") : "follows image", x + pad, y + 23);
        ctx.textAlign = "right";
        ctx.fillStyle = LUNA.muted;
        ctx.font = `11px ${FONT}`;
        if (imageKnown(node)) ctx.fillText(`${ratio || ""}   ${mp || ""}`, x + w - pad, y + 23);

        // Line 2: the source, then the portrait chip and the gear at the right.
        const gear = [x + w - pad - 18, y + 30, 18, 18];
        const chipText = val(node, "portrait") ? "▯ portrait" : "▭ landscape";
        ctx.font = `10.5px ${FONT}`;
        const cw = ctx.measureText(chipText).width + 12;
        const chip = [gear[0] - 6 - cw, y + 30, cw, 18];
        node._lunaHits = { gear, chip };
        const box = (rc, on) => {
            ctx.beginPath();
            if (typeof ctx.roundRect === "function") ctx.roundRect(rc[0], rc[1], rc[2], rc[3], 4); else ctx.rect(...rc);
            ctx.fillStyle = on ? LUNA.accent : LUNA.bg; ctx.fill();
            ctx.strokeStyle = on ? LUNA.accent : LUNA.border; ctx.stroke();
        };
        box(chip, !!val(node, "portrait"));
        ctx.textAlign = "center";
        ctx.fillStyle = val(node, "portrait") ? LUNA.bg : LUNA.text;
        ctx.fillText(chipText, chip[0] + chip[2] / 2, chip[1] + 13);
        box(gear, false);
        ctx.fillStyle = LUNA.accent;
        ctx.font = `12px ${FONT}`;
        ctx.fillText("⚙", gear[0] + 9, gear[1] + 13.5);

        ctx.textAlign = "left";
        ctx.fillStyle = LUNA.text;
        ctx.font = `11px ${FONT}`;
        const src = imageKnown(node) ? (r.lines?.[1] || "") : "the image size is known after a run";
        ctx.fillText(fit(ctx, src, chip[0] - (x + pad) - 6), x + pad, y + 43);

        if (r.warnings?.length) {
            ctx.fillStyle = LUNA.danger;
            ctx.fillText(fit(ctx, r.warnings.join("; "), w - pad * 2), x + pad, y + 62);
        }
    } finally {
        ctx.restore();
    }
}

const inRect = (p, r) => r && p[0] >= r[0] && p[0] <= r[0] + r[2] && p[1] >= r[1] && p[1] <= r[1] + r[3];

app.registerExtension({
    name: "LunaMediaTools.ImageResolution",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            hidePanelWidgets(this);
            this.addCustomWidget({
                type: "luna_resolution_readout",
                name: "readout",
                serialize: false,
                value: null,
                options: { serialize: false },
                computeSize() { return [0, guard("size", () => readoutHeight(node), 64)]; },
                draw(ctx, _n, _w, y) {
                    guard("draw", () => { node._lunaReadoutY = y; drawReadout(ctx, node, y); }, undefined);
                },
                mouse(event, pos) {
                    if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
                    const hits = node._lunaHits || {};
                    if (inRect(pos, hits.gear)) { openPanel(node); return true; }
                    if (inRect(pos, hits.chip)) { setWidget(node, "portrait", !val(node, "portrait")); return true; }
                    return false;
                },
            });
            for (const w of this.widgets || []) {
                if (w.type === "luna_resolution_readout") continue;
                const prev = w.callback;
                w.callback = function (...args) {
                    const out = prev?.apply(this, args);
                    changed(node);
                    return out;
                };
            }
            guard("init", () => { this.setSize(this.computeSize()); changed(this); }, undefined);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            hidePanelWidgets(this);
            changed(this);
            return r;
        };

        // A run with an image wired: keep that picture's size, so the readout follows later settings with it.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const size = message?.luna_image_size?.[0];
            if (Array.isArray(size) && size.length === 2 && imageWired(this)) {
                this._lunaImageSize = [Number(size[0]), Number(size[1])];
                changed(this);
            }
            return r;
        };

        // Unwiring or rewiring `image` forgets the size: a new picture is only known after its run.
        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, link, ioSlot) {
            const r = onConn?.apply(this, arguments);
            if (ioSlot?.name === "image") this._lunaImageSize = null;
            setTimeout(() => changed(this), 0);
            return r;
        };
    },
});
