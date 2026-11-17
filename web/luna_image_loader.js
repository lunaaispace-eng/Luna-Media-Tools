// Luna-ImageLoader — the pictures for image work, on the Asset Loader's card UI (luna_cards.mjs).
//
// Three differences from Luna-H3-AssetLoader, nothing else:
//   1. the role row is Edit base · Reference · Style · Composition (stored in media_state pictures[i].role:
//      edit, ref, style, composition); at most one edit base;
//   2. the cards are "Image N";
//   3. the − N + stepper sets how many pictures are ACTIVE (0-9): cards 1…N are sent and show image_1…image_N,
//      a card above N stays in the node, greyed and off.
//
// ComfyUI has no dynamic outputs: Python declares all nine image_N and this removes the trailing ones. The four
// fixed outputs (width, height, asset_brief, refs) come first, so their indices never move.

import { app } from "../../scripts/app.js";
import { keepNodeWidth } from "./luna_widget_width.mjs";
import {
    COUNT_WIDGET, MEDIA_WIDGET, PATHS_WIDGET, PREVIEW_H, PREVIEW_SIZES, STATE_WIDGET,
    buildPictureSection, ctlHit, drawCtl, gridHeight, injectCardsCSS, insertItems, isInternalWrite,
    openSettings, readPaths, uploadFiles, widget, writePaths,
} from "./luna_cards.mjs";

const NODE = "Luna-ImageLoader";
const MAX = 9;
const MIN = 0;
const FIXED = 4;                                  // width, height, asset_brief, refs
const ROLES = [["edit", "Edit base"], ["ref", "Reference"], ["style", "Style"], ["composition", "Composition"]];
const BADGE = { edit: "edit", style: "style", composition: "layout", ref: "" };

/* ------------------------------------------------------------------ media_state: roles and node settings */

function readMedia(node) {
    try {
        const m = JSON.parse(String(widget(node, MEDIA_WIDGET)?.value || "") || "{}");
        return {
            pictures: Array.isArray(m.pictures) ? m.pictures.map((p) => p || {}) : [],
            settings: m.settings && typeof m.settings === "object" ? m.settings : {},
        };
    } catch { return { pictures: [], settings: {} }; }
}

function writeMedia(node, m) {
    const w = widget(node, MEDIA_WIDGET);
    if (!w) return;
    const out = {};
    if (m.pictures?.some((p) => p?.role && p.role !== "ref")) out.pictures = m.pictures;
    if (m.settings && Object.keys(m.settings).length) out.settings = m.settings;
    w.value = Object.keys(out).length ? JSON.stringify(out) : "";
    w.callback?.(w.value);
    node.graph?.setDirtyCanvas(true, true);
}

const roleOf = (node, i) => {
    const r = readMedia(node).pictures[i]?.role;
    return ROLES.some(([v]) => v === r) ? r : "ref";
};

function setRole(node, i, role) {
    const m = readMedia(node);
    while (m.pictures.length <= i) m.pictures.push({});
    // One edit base: choosing it here turns any other edit base into a reference.
    if (role === "edit") m.pictures.forEach((p, k) => { if (k !== i && p?.role === "edit") p.role = "ref"; });
    m.pictures[i] = { ...m.pictures[i], role };
    writeMedia(node, m);
}

const previewH = (node) => PREVIEW_SIZES[readMedia(node).settings?.node?.preview] || PREVIEW_H;

/* ------------------------------------------------------------------ active count and outputs */

const countOf = (node) => {
    const v = Number(widget(node, COUNT_WIDGET)?.value);
    return Math.max(MIN, Math.min(MAX, Number.isFinite(v) ? v : 1));
};

// Returns { target, shown, blocked }; blocked = the 1-based image_N that stopped a shrink because it is wired.
function applyCount(node) {
    if (!node.outputs || !widget(node, COUNT_WIDGET)) return null;
    const target = countOf(node);
    let current = node.outputs.length - FIXED;
    let blocked = 0;
    while (current > target) {
        const idx = FIXED + current - 1;
        if (node.outputs[idx]?.links?.length) { blocked = current; break; }
        node.removeOutput(idx);
        current--;
    }
    while (current < target) {
        node.addOutput(`image_${current + 1}`, "IMAGE");
        current++;
    }
    node.setDirtyCanvas(true, true);
    return { target, shown: current, blocked };
}

function setCount(node, value) {
    const cw = widget(node, COUNT_WIDGET);
    if (!cw) return null;
    const next = Math.max(MIN, Math.min(MAX, Number.isFinite(Number(value)) ? Number(value) : 1));
    if (Number(cw.value) !== next) {
        cw.value = next;
        cw.callback?.(next);
    }
    const res = applyCount(node);
    if (res?.blocked) node._lunaNotice?.(`image_${res.blocked} is wired — outputs stop there.`);
    node._lunaRender?.();
    return res;
}

function promptCount(node, cur, e) {
    const apply = (v) => { const n = parseInt(v, 10); if (Number.isFinite(n)) setCount(node, n); };
    const canvas = app.canvas || node.graph?.list_of_graphcanvas?.[0];
    if (canvas?.prompt) canvas.prompt("Active pictures", cur, apply, e);
    else apply(window.prompt(`Active pictures (${MIN}-${MAX})`, String(cur)));
}

/* ------------------------------------------------------------------ the node body */

function buildUI(node) {
    injectCardsCSS();
    const root = document.createElement("div");
    root.className = "luna-al";
    const bar = document.createElement("div");
    bar.className = "luna-al-bar";

    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.multiple = true;
    picker.style.display = "none";

    const section = buildPictureSection(node, {
        max: MAX,
        roles: ROLES,
        roleOf: (i) => roleOf(node, i),
        setRole: (i, v) => setRole(node, i, v),
        roleBadge: (i) => BADGE[roleOf(node, i)] || "",
        label: (i) => `Image ${i + 1}`,
        active: (i) => i < countOf(node),
        previewH: () => previewH(node),
        // A picture's role travels with it; so does the active count when an active card goes.
        onRemove: (i) => {
            const m = readMedia(node);
            if (m.pictures.length > i) { m.pictures.splice(i, 1); writeMedia(node, m); }
            if (i < countOf(node)) setCount(node, countOf(node) - 1);
        },
        onMove: (from, to) => {
            const m = readMedia(node);
            if (!m.pictures.length) return;
            while (m.pictures.length <= Math.max(from, to)) m.pictures.push({});
            const [p] = m.pictures.splice(from, 1);
            m.pictures.splice(to, 0, p);
            writeMedia(node, m);
        },
        onChange: () => render(),
    });
    section.add.onclick = () => picker.click();
    root.append(bar, section.el, picker);

    const mkBtn = (label, cls, fn) => {
        const b = document.createElement("div");
        b.className = `luna-al-btn${cls ? " " + cls : ""}`;
        b.textContent = label;
        b.onclick = fn;
        bar.appendChild(b);
        return b;
    };
    mkBtn("Upload", "", () => picker.click());
    mkBtn("Clear", "danger", () => {
        generation++;
        writePaths(node, []);
        node._lunaPathCount = 0;
        const sw = widget(node, STATE_WIDGET);
        if (sw) { sw.value = ""; sw.callback?.(""); }
        const m = readMedia(node);
        writeMedia(node, { pictures: [], settings: m.settings });
        setCount(node, 0);
        render();
    });

    // The raw widgets are hidden once the cards own them; Fields brings them back.
    let fieldsShown = false;
    const setFieldsVisible = (show) => {
        for (const name of [PATHS_WIDGET, COUNT_WIDGET, STATE_WIDGET, MEDIA_WIDGET]) {
            const w = widget(node, name);
            if (!w) continue;
            w.hidden = !show;
            if (show) {
                w.type = w._lunaType ?? w.type;
                delete w.computeSize;
            } else {
                w._lunaType = w._lunaType ?? w.type;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
            if (w.element) {
                w.element.hidden = !show;
                w.element.style.display = show ? "" : "none";
            }
        }
        node.setDirtyCanvas(true, true);
    };
    const gearBtn = mkBtn("⚙", "", () => openSettings(app, node, `${NODE} settings`, (body, chips, head) => {
        head("▦", "Node");
        chips("Preview height", [["S", "S"], ["M", "M"], ["L", "L"]], readMedia(node).settings?.node?.preview || "M", (v) => {
            const m = readMedia(node);
            m.settings = { ...m.settings, node: { ...(m.settings.node || {}), preview: v } };
            writeMedia(node, m);
            render();
        });
    }, () => { const m = readMedia(node); m.settings = {}; writeMedia(node, m); render(); }));
    gearBtn.title = "Settings: preview height";
    const fieldsBtn = mkBtn("Fields", "", () => {
        fieldsShown = !fieldsShown;
        setFieldsVisible(fieldsShown);
        fieldsBtn.textContent = fieldsShown ? "Hide fields" : "Fields";
    });
    fieldsBtn.title = "Show the raw image_paths / image_count / asset_state / media_state widgets";
    requestAnimationFrame(() => setFieldsVisible(false));

    const note = document.createElement("div");
    note.className = "luna-al-note";
    bar.appendChild(note);
    let noteTimer = 0;
    node._lunaNotice = (msg) => {
        note.textContent = msg || "";
        clearTimeout(noteTimer);
        if (msg) noteTimer = setTimeout(() => { note.textContent = ""; }, 3000);
    };
    const counter = document.createElement("div");
    counter.className = "luna-al-count";
    bar.appendChild(counter);

    // Clear bumps this, so an upload still on its way when the cards were cleared is dropped, not restored.
    let generation = 0;
    picker.onchange = async () => {
        const files = Array.from(picker.files || []).filter((f) => (f.type || "").startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(f.name || ""));
        picker.value = "";
        if (!files.length) return;
        const gen = generation;
        const room = MAX - readPaths(node).length;
        const names = await uploadFiles(files.slice(0, Math.max(0, room)));
        if (files.length > room) node._lunaNotice(`Room for ${MAX} pictures; ${files.length - Math.max(0, room)} not added.`);
        if (!names.length || gen !== generation) return;
        // Read the cards again after the wait: they may have changed meanwhile.
        const now = readPaths(node);
        const add = names.slice(0, Math.max(0, MAX - now.length));
        if (!add.length) return;
        // New pictures go in right after the active ones and are active; parked cards stay parked, below them.
        const at = Math.min(countOf(node), now.length);
        insertItems(node, at, add.length);
        const m = readMedia(node);
        if (m.pictures.length > at) { m.pictures.splice(at, 0, ...add.map(() => ({}))); writeMedia(node, m); }
        writePaths(node, [...now.slice(0, at), ...add, ...now.slice(at)]);
        node._lunaPathCount = readPaths(node).length;
        setCount(node, Math.min(MAX, at + add.length));
        section.select(at);
        render();
    };

    function render() {
        const n = readPaths(node).length, a = Math.min(countOf(node), n);
        node._lunaPathCount = n;   // the Fields hook compares a typed list with this
        counter.textContent = n ? `${a} of ${n} active` : "";
        section.render();
        node.setDirtyCanvas(true, true);
    }

    return { root, render, panelExtraRows: section.panelExtraRows, hasCards: () => readPaths(node).length > 0 };
}

app.registerExtension({
    name: "LunaMediaTools.ImageLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const ui = buildUI(this);
            // Heights from the Asset Loader's calibration (frontend 1.55.17): root padding + toolbar + one gap, the
            // section heading, the grid, the preview and the panel (heading, role, Size, Ratio, Snap, buttons).
            const ROOT = 12 + 27 + 6, SEC = 7 + 21 + 6, PANEL = 125, MINH = 150;
            const cardsHeight = () => {
                const n = readPaths(this).length;
                const editor = n ? 6 + previewH(this) + 6 + PANEL + ui.panelExtraRows() * 26 : 0;
                return Math.max(MINH, ROOT + SEC + gridHeight(this, n) + editor + 16);
            };
            const w = this.addDOMWidget("luna_image_cards", "div", ui.root, {
                serialize: false, hideOnZoom: false, getHeight: cardsHeight,
            });
            w.computeSize = (width) => [width, cardsHeight()];
            keepNodeWidth(w);
            this._lunaRender = () => { ui.render(); this.setDirtyCanvas(true, true); };

            const cw = widget(this, COUNT_WIDGET);
            if (cw) {
                const orig = cw.callback;
                const self = this;
                cw.callback = function () {
                    const out = orig ? orig.apply(this, arguments) : undefined;
                    const res = applyCount(self);
                    if (res?.blocked) self._lunaNotice?.(`image_${res.blocked} is wired — outputs stop there.`);
                    return out;
                };
            }
            // Typing into image_paths (Fields) rebuilds the cards. Lines added are active (the count grows by them);
            // lines removed lower it to at most the list's length. Parked cards stay parked.
            const pw = widget(this, PATHS_WIDGET);
            if (pw) {
                const origPaths = pw.callback;
                const self = this;
                pw.callback = function () {
                    const out = origPaths ? origPaths.apply(this, arguments) : undefined;
                    if (!isInternalWrite()) {
                        const n = readPaths(self).length, before = self._lunaPathCount ?? n;
                        const cur = countOf(self);
                        setCount(self, Math.min(n, n > before ? cur + (n - before) : cur));
                        self._lunaPathCount = n;
                        self._lunaRender?.();
                    }
                    return out;
                };
            }
            if (this.size[1] < 320) this.size[1] = 320;
            requestAnimationFrame(() => { applyCount(this); ui.render(); });
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => { applyCount(this); this._lunaRender?.(); });
            return r;
        };

        const onDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDraw?.apply(this, arguments);
            if (!this.flags?.collapsed && ctx) drawCtl(this, ctx, { label: "Active", min: MIN, max: MAX });
            return r;
        };

        // Returning true consumes the click, so LiteGraph does not drag the node from under a button press.
        const onDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos) {
            if (!this.flags?.collapsed) {
                const part = ctlHit(this, pos);
                if (part) {
                    const cur = countOf(this);
                    if (part === "minus") setCount(this, cur - 1);
                    else if (part === "plus") setCount(this, cur + 1);
                    else promptCount(this, cur, e);
                    return true;
                }
            }
            return onDown?.apply(this, arguments);
        };
        const onDbl = nodeType.prototype.onDblClick;
        nodeType.prototype.onDblClick = function (e, pos) {
            if (!this.flags?.collapsed && ctlHit(this, pos)) return true;
            return onDbl?.apply(this, arguments);
        };
        const onMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, pos) {
            const part = this.flags?.collapsed ? "" : ctlHit(this, pos);
            if (part !== (this._lunaCtlHover || "")) {
                this._lunaCtlHover = part;
                this.setDirtyCanvas(true, false);
            }
            return onMove?.apply(this, arguments);
        };
    },
});
