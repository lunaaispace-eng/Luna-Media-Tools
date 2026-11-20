// Luna-ImageLoader — the pictures for image work, on the Asset Loader's card UI (luna_cards.mjs).
//
// Four differences from Luna-H3-AssetLoader, nothing else:
//   1. the role row is Edit base · Reference · Style · Composition (stored in media_state pictures[i].role:
//      edit, ref, style, composition); at most one edit base;
//   2. card N is @imageN; each active card shows the tag its ⚙ Target model gets for it ("@image3 · <Picture 1>");
//   3. ⚙ Target (media_state settings.target: minimax_image | qwen | general, missing = general) picks the shared
//      naming row of luna_naming.py: MiniMax Image and Qwen send the edit base first, General card order
//      (cardNames below; image_loader.py sends in the same order, and tests/test_image_loader.py compares the two);
//   4. the − N + stepper sets how many pictures are ACTIVE (0-9): cards 1…N are sent as image_1…image_N,
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

/* ------------------------------------------------------------------ naming: the ⚙ Target and the @ names */

// The ⚙ Target -> its row of luna_naming.py ROWS (pictures only): tag pattern, first number, edit base first.
export const TARGETS = [["minimax_image", "MiniMax Image"], ["qwen", "Qwen"], ["general", "General"]];
const ROWS = {
    minimax_image: { image: "<Picture {n}>", first: 1, editFirst: true },
    qwen: { image: "<image{n}>", first: 1, editFirst: true },
    general: { image: "Image {n}", first: 1, editFirst: false },
};

// Own keys only: a saved "constructor" or "__proto__" is an unknown Target too, so it falls back to General as parse_target does.
export const targetName = (t) => (typeof t === "string" && Object.prototype.hasOwnProperty.call(ROWS, t) ? t : "general");

// The order-and-tag rule the cards show, the same as image_loader.py + luna_naming.make:
//   target: settings.target; roles: one stored role per card; active: how many cards are sent (cards 0…active-1).
// Only the first card marked edit is the edit base (a later one is sent as a reference, as parse_roles does).
// Returns { target, order: [card index, in sent order], cards: [{ name, place, tag, label }] } with place 1-based
// (null for a card that is not active) and label "@image3 · <Picture 1>" (or "@image4 · off").
export function cardNames(target, roles, active) {
    target = targetName(target);
    const row = ROWS[target];
    const n = Math.max(0, Math.min(roles.length, Number.isFinite(Number(active)) ? Number(active) : 0));
    let edit = -1;
    for (let i = 0; i < n; i++) if (roles[i] === "edit") { edit = i; break; }
    const cardOrder = Array.from({ length: n }, (_, i) => i);
    const order = edit >= 0 && row.editFirst ? [edit, ...cardOrder.filter((i) => i !== edit)] : cardOrder;
    const place = new Map(order.map((c, p) => [c, p + 1]));
    const cards = roles.map((_, i) => {
        const name = `@image${i + 1}`;
        const p = place.get(i) ?? null;
        const tag = p === null ? null : row.image.replace("{n}", String(p - 1 + row.first));
        return { name, place: p, tag, label: `${name} · ${tag ?? "off"}` };
    });
    return { target, order, cards };
}

/* ------------------------------------------------------------------ media_state: roles and node settings */

export function readMedia(node) {
    try {
        const m = JSON.parse(String(widget(node, MEDIA_WIDGET)?.value || "") || "{}");
        return {
            pictures: Array.isArray(m.pictures) ? m.pictures.map((p) => p || {}) : [],
            settings: m.settings && typeof m.settings === "object" ? m.settings : {},
        };
    } catch { return { pictures: [], settings: {} }; }
}

export function writeMedia(node, m) {
    const w = widget(node, MEDIA_WIDGET);
    if (!w) return;
    const out = {};
    if (m.pictures?.some((p) => p?.role && p.role !== "ref")) out.pictures = m.pictures;
    if (m.settings && Object.keys(m.settings).length) out.settings = m.settings;
    w.value = Object.keys(out).length ? JSON.stringify(out) : "";
    w.callback?.(w.value);
    node.graph?.setDirtyCanvas(true, true);
}

export const roleOf = (node, i) => {
    const r = readMedia(node).pictures[i]?.role;
    return ROLES.some(([v]) => v === r) ? r : "ref";
};

export function setRole(node, i, role) {
    const m = readMedia(node);
    while (m.pictures.length <= i) m.pictures.push({});
    // One edit base: choosing it here turns any other edit base into a reference.
    if (role === "edit") m.pictures.forEach((p, k) => { if (k !== i && p?.role === "edit") p.role = "ref"; });
    m.pictures[i] = { ...m.pictures[i], role };
    writeMedia(node, m);
}

export const targetOf = (node) => targetName(readMedia(node).settings?.target);

export function setTarget(node, target) {
    const m = readMedia(node);
    m.settings = { ...m.settings, target: targetName(target) };
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

    // The @ names and tags of every card, from the node's current Target, roles and active count.
    const names = () => {
        const n = readPaths(node).length;
        return cardNames(targetOf(node), Array.from({ length: n }, (_, i) => roleOf(node, i)), countOf(node)).cards;
    };
    const named = (i) => names()[i] || { name: `@image${i + 1}`, tag: null, label: `@image${i + 1}` };
    const section = buildPictureSection(node, {
        max: MAX,
        roles: ROLES,
        roleOf: (i) => roleOf(node, i),
        setRole: (i, v) => setRole(node, i, v),
        // The tag the Target model gets for this card, then its role: "<Picture 1> · edit".
        roleBadge: (i) => [named(i).tag, BADGE[roleOf(node, i)]].filter(Boolean).join(" · "),
        pic: (i) => named(i).name,
        label: (i) => named(i).label,
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
        head("@", "Naming");
        chips("Target", TARGETS, targetOf(node), (v) => { setTarget(node, v); render(); },
            "The model the pictures go to. MiniMax Image (<Picture n>) and Qwen (<imageN>) edit image_1: the edit base " +
            "is sent first. General (Image N) sends card order. The cards show each @ name's tag.");
        head("▦", "Node");
        chips("Preview height", [["S", "S"], ["M", "M"], ["L", "L"]], readMedia(node).settings?.node?.preview || "M", (v) => {
            const m = readMedia(node);
            m.settings = { ...m.settings, node: { ...(m.settings.node || {}), preview: v } };
            writeMedia(node, m);
            render();
        });
    }, () => { const m = readMedia(node); m.settings = {}; writeMedia(node, m); render(); }));
    gearBtn.title = "Settings: Target (the model the pictures go to: order and @ name tags), preview height";
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
