// Luna picture cards — the picture part of Luna-H3-AssetLoader's UI as one kit file.
//
// Source: Luna-MinimaxH3 web/luna_asset_loader.js (CSS, readPaths / writePaths, the per-picture resize state, predictSize,
// describeAR, uploadFiles, the count stepper, the card row, the preview and the resize panel). Same class names, same
// look, same state JSON, so the two nodes are sisters. Hand-copied between packs like luna_theme / luna_help /
// luna_collapse; Luna-H3-AssetLoader still carries its own copy of this code.
//
// The per-picture resize state's key names and defaults follow ComfyUI-Pixaroma (MIT, Copyright (c) 2026 pixaroma),
// nodes/_resize_helpers.py.
//
// What a node passes in (buildPictureCards): its roles, its card label, which cards are active, and its node-wide
// settings. Everything else is shared.

import { LUNA } from "./luna_theme.mjs";

export const PATHS_WIDGET = "image_paths";
export const COUNT_WIDGET = "image_count";
export const STATE_WIDGET = "asset_state";
export const MEDIA_WIDGET = "media_state";
export const PREVIEW_H = 170;
export const PREVIEW_SIZES = { S: 120, M: PREVIEW_H, L: 260 };

/* ------------------------------------------------------------------ styles */

// NOTE: no backticks anywhere in this block; it is a template literal.
const CSS = `
.luna-al { --luna-accent:${LUNA.accent}; --luna-accent-soft:${LUNA.accentSoft}; --luna-panel:${LUNA.panel};
           --luna-border:${LUNA.border}; --luna-text:${LUNA.text}; --luna-muted:${LUNA.muted};
           --luna-input:${LUNA.input}; --luna-danger:${LUNA.danger}; --luna-bg:${LUNA.bg};
           --luna-radius:${LUNA.radius}px;
           display:flex; flex-direction:column; gap:6px; padding:6px; box-sizing:border-box;
           font-family:"Segoe UI", system-ui, sans-serif; font-size:11px; color:var(--luna-text); }
.luna-al-bar { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.luna-al-btn { flex:0 0 auto; padding:5px 10px; border-radius:6px; cursor:pointer;
               background:var(--luna-panel); border:1px solid var(--luna-border);
               color:var(--luna-text); transition:background .1s, border-color .1s, color .1s; }
.luna-al-btn:hover { background:var(--luna-accent); border-color:var(--luna-accent); color:var(--luna-bg); }
.luna-al-btn.danger:hover { background:var(--luna-danger); border-color:var(--luna-danger); color:#fff; }
.luna-al-count { margin-left:auto; color:var(--luna-muted); }
.luna-al-note { color:var(--luna-accent); }
.luna-al-cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(112px, 1fr));
                 gap:8px; overflow-y:auto; flex:1 1 auto; align-content:start; padding:2px; }
.luna-al-card { position:relative; border-radius:8px; overflow:hidden; cursor:grab;
                background:var(--luna-panel); border:1px solid var(--luna-border); transition:border-color .1s; }
.luna-al-card:hover { border-color:var(--luna-accent); }
.luna-al-card.drag { opacity:.35; }
.luna-al-card.over { border-color:var(--luna-accent); box-shadow:0 0 0 1px var(--luna-accent) inset; }
.luna-al-card.off { filter:grayscale(1); opacity:.45; }
.luna-al-thumb { display:block; width:100%; aspect-ratio:1/1; object-fit:contain;
                 background:var(--luna-bg); }
.luna-al-pic { position:absolute; top:4px; left:4px; padding:1px 6px; border-radius:5px;
               background:var(--luna-accent); color:var(--luna-bg); font-weight:700; font-size:10px; }
.luna-al-role { position:absolute; top:22px; left:4px; padding:0 5px; border-radius:5px; font-size:9.5px;
                background:rgba(20,21,26,.85); color:var(--luna-text); }
.luna-al-x { position:absolute; top:4px; right:4px; width:18px; height:18px; line-height:17px;
             text-align:center; border-radius:5px; cursor:pointer; font-size:11px;
             background:rgba(20,21,26,.85); color:var(--luna-muted); opacity:0; transition:opacity .1s; }
.luna-al-card:hover .luna-al-x { opacity:1; }
.luna-al-x:hover { background:var(--luna-danger); color:#fff; }
.luna-al-cap { position:absolute; left:0; right:0; bottom:0; padding:3px 5px; font-size:10px;
               background:linear-gradient(to top, rgba(12,12,15,.95), rgba(12,12,15,0)); }
.luna-al-dim { color:var(--luna-text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.luna-al-name { opacity:.45; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.luna-al-preview { position:relative; border:1px solid var(--luna-border); border-radius:8px;
                   background:var(--luna-bg); overflow:hidden; flex:0 0 auto; }
.luna-al-preview img { display:block; width:100%; height:100%; object-fit:contain; }
.luna-al-pvlbl { position:absolute; left:6px; bottom:5px; padding:1px 6px; border-radius:5px;
                 background:rgba(20,21,26,.82); font-size:10px; color:var(--luna-text); }
.luna-al-empty { opacity:.55; padding:10px 4px; }
.luna-al-warn { color:var(--luna-accent); padding:2px 4px; }
.luna-al-card.sel { border-color:var(--luna-accent); box-shadow:0 0 0 1px var(--luna-accent) inset; }
.luna-al-panel { border-top:1px solid var(--luna-border); padding-top:6px; display:flex;
                 flex-direction:column; gap:5px; }
.luna-al-row { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
.luna-al-lbl { color:var(--luna-muted); min-width:44px; }
.luna-al-seg { flex:1 1 auto; display:flex; gap:3px; }
.luna-al-opt { flex:1 1 auto; text-align:center; padding:3px 5px; border-radius:5px; cursor:pointer;
               background:var(--luna-panel); border:1px solid var(--luna-border); color:var(--luna-text); font-size:10px;
               white-space:nowrap; }
.luna-al-opt:hover { border-color:var(--luna-accent); }
.luna-al-opt.on { background:var(--luna-accent); border-color:var(--luna-accent); color:var(--luna-bg); }
.luna-al-num { width:64px; padding:3px 5px; border-radius:5px; background:var(--luna-bg);
               border:1px solid var(--luna-border); color:var(--luna-text); font-size:10px; }
.luna-al-io { opacity:.75; font-size:10px; }
.luna-al-io b { color:var(--luna-accent); font-weight:600; }
.luna-al-sec { display:flex; flex-direction:column; gap:6px; border-top:1px solid var(--luna-border); padding-top:6px; }
.luna-al-sech { display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:600; }
.luna-al-sec.k-p .luna-al-sech .tt { color:var(--luna-accent); }
.luna-al-sech .n { color:var(--luna-muted); font-weight:400; }
.luna-al-sech .sp { flex:1 1 auto; }
.luna-al-sech .add { padding:2px 9px; font-weight:400; }
.luna-al-sec .luna-al-empty { padding:4px 2px; font-size:10.5px; }
.luna-al-sp { position:fixed; z-index:1300; width:330px; transform-origin:0 0; background:#1c1e25;
              border:1px solid #e0a45866; border-radius:8px; box-shadow:0 10px 30px #000a;
              font:12px "Segoe UI", system-ui, sans-serif; color:#e7e5df; }
.luna-al-sp .hd { display:flex; align-items:center; gap:8px; padding:9px 12px; border-bottom:1px solid #2e313c; }
.luna-al-sp .hd .t { flex:1; font-weight:600; font-size:13px; }
.luna-al-sp .x, .luna-al-sp .rs, .luna-al-sp .dn { cursor:pointer; color:#8b8e99; }
.luna-al-sp .x:hover, .luna-al-sp .rs:hover, .luna-al-sp .dn:hover { color:#e0a458; }
.luna-al-sp .bd { padding:2px 12px 10px; }
.luna-al-sp .sh { color:#e0a458; font-weight:600; font-size:12px; margin:10px 0 5px; }
.luna-al-sp .rw { display:grid; grid-template-columns:110px 1fr; align-items:center; gap:8px; margin:5px 0; }
.luna-al-sp .ch { display:flex; gap:4px; flex-wrap:wrap; align-items:center; color:#8b8e99; }
.luna-al-sp .ch span { padding:2px 8px; border:1px solid #2e313c; border-radius:4px; color:#e7e5df; cursor:pointer; }
.luna-al-sp .ch span:hover { border-color:#8b8e99; }
.luna-al-sp .ch span.on { border-color:#e0a458; color:#e0a458; background:#e0a45814; }
.luna-al-sp .nb { width:64px; background:#14151a; border:1px solid #2e313c; border-radius:4px; color:#e7e5df; text-align:center; padding:2px 0; }
.luna-al-sp .path { font:10.5px ui-monospace, Consolas, monospace; color:#8b8e99; word-break:break-all; }
.luna-al-sp .nt { font-size:11px; color:#8b8e99; margin:0 0 4px 118px; line-height:1.4; }
.luna-al-sp .ft { border-top:1px solid #2e313c; padding:7px 12px; display:flex; justify-content:space-between; }
`;

export function injectCardsCSS() {
    // Its own id: Luna-H3-AssetLoader injects a different sheet as "luna-al-css", and one id would let the first node
    // loaded suppress the other's styles. The rules both define are the same, so both sheets can sit side by side.
    if (document.getElementById("luna-cards-css")) return;
    const s = document.createElement("style");
    s.id = "luna-cards-css";
    s.textContent = CSS;
    document.head.appendChild(s);
}

/* ------------------------------------------------------------------- utils */

export const widget = (node, name) => node.widgets?.find((w) => w.name === name);

export function readPaths(node) {
    return String(widget(node, PATHS_WIDGET)?.value ?? "")
        .replace(/\r/g, "\n").split("\n")
        .map((s) => s.trim().replace(/^"|"$/g, ""))
        .filter(Boolean);
}

// Set while writePaths fires the widget callback, so a hand-edit hook on image_paths can tell typing from the cards.
let internalWrite = false;
export const isInternalWrite = () => internalWrite;

export function writePaths(node, list) {
    const w = widget(node, PATHS_WIDGET);
    if (!w) return;
    w.value = list.join("\n");
    internalWrite = true;
    try { w.callback?.(w.value); } finally { internalWrite = false; }
    node.graph?.setDirtyCanvas(true, true);
}

// One cache-busting token per file per page load; uploads never overwrite, so a stable URL cannot show a stale picture.
const _viewTok = new Map();
export const viewURL = (name) => {
    if (!_viewTok.has(name)) _viewTok.set(name, Math.random().toString(36).slice(2));
    return `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=&rand=${_viewTok.get(name)}`;
};

export async function uploadFiles(files) {
    const saved = [];
    for (const file of files) {
        const form = new FormData();
        form.append("image", file);
        form.append("type", "input");
        form.append("overwrite", "false");
        const resp = await fetch("/upload/image", { method: "POST", body: form });
        if (!resp.ok) {
            alert(`Upload failed for ${file.name}: ${resp.status} ${resp.statusText}`);
            continue;
        }
        const data = await resp.json();
        saved.push(data.subfolder ? `${data.subfolder}/${data.name}` : data.name);
    }
    return saved;
}

/* ------------------------------------------------------- per-picture resize state */

export const DEFAULT_ITEM = {
    mode: "off", max_mp: 2.0, longest_side: 1024, scale_factor: 1.0,
    ratio: "", ratio_action: "crop", crop_anchor: "center", pad_color: "#000000",
    snap: 0, allow_upscale: false,
};

export function readState(node) {
    try {
        const p = JSON.parse(String(widget(node, STATE_WIDGET)?.value || "") || "{}");
        return { all: p.all || null, items: Array.isArray(p.items) ? p.items : [] };
    } catch { return { all: null, items: [] }; }
}

export function itemAt(node, i) {
    const st = readState(node);
    return { ...DEFAULT_ITEM, ...(st.all || {}), ...(st.items[i] || {}) };
}

function writeState(node, st) {
    const w = widget(node, STATE_WIDGET);
    if (!w) return;
    const out = {};
    if (st.all) out.all = st.all;
    if (st.items.length) out.items = st.items;
    w.value = Object.keys(out).length ? JSON.stringify(out) : "";
    w.callback?.(w.value);
}

export function writeItem(node, i, patch, toAll = false) {
    const st = readState(node);
    if (toAll) {
        st.all = { ...(st.all || {}), ...patch };
        st.items = [];           // "apply to all" clears per-card overrides
    } else {
        while (st.items.length <= i) st.items.push({});
        st.items[i] = { ...st.items[i], ...patch };
    }
    writeState(node, st);
}

// A card's resize state travels with its picture when cards are removed or reordered.
export function moveItem(node, from, to) {
    const st = readState(node);
    if (!st.items.length) return;
    while (st.items.length <= Math.max(from, to)) st.items.push({});
    const [it] = st.items.splice(from, 1);
    if (to >= 0) st.items.splice(to, 0, it);
    while (st.items.length && !Object.keys(st.items[st.items.length - 1] || {}).length) st.items.pop();
    writeState(node, st);
}

// Python's round(): halves go to the even neighbour (960.5 -> 960), unlike Math.round.
const roundEven = (v) => {
    const f = Math.floor(v), d = v - f;
    if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
    return Math.round(v);
};

// n new cards inserted at `at`: the per-card states from `at` on move down with their pictures.
export function insertItems(node, at, n) {
    const st = readState(node);
    if (!st.items.length || at >= st.items.length || n <= 0) return;
    st.items.splice(at, 0, ...Array.from({ length: n }, () => ({})));
    writeState(node, st);
}

// What the Python side produces (image_loader.apply_state), so the panel shows INPUT -> OUTPUT before a run.
export function predictSize(w, h, st) {
    if (!w || !h) return [0, 0];
    let W = w, H = h;
    if (st.ratio && st.ratio.includes(":")) {
        const [rw, rh] = st.ratio.split(":").map(Number);
        const t = rw / rh;
        if (t > 0 && Math.abs(W / H - t) > 0.001) {
            if (st.ratio_action === "pad") {
                if (W / H > t) H = Math.max(H, roundEven(W / t)); else W = Math.max(W, roundEven(H * t));
            } else {
                if (W / H > t) W = roundEven(H * t); else H = roundEven(W / t);
            }
        }
    }
    let s = 1;
    if (st.mode === "max_mp") { const b = Number(st.max_mp) * 1e6; if (b > 0) s = Math.sqrt(b / (W * H)); }
    else if (st.mode === "longest_side") s = Number(st.longest_side) / Math.max(W, H);
    else if (st.mode === "scale_factor") s = Number(st.scale_factor);
    if (!st.allow_upscale) s = Math.min(1, s);
    let snap = Number(st.snap) || 0;
    // As in Python: with upscale off, a grid wider than an edge is not applied.
    if (snap > 1 && !st.allow_upscale && Math.min(W * s, H * s) + 1e-6 < snap) snap = 0;
    const fix = (v) => (snap > 1 ? Math.max(snap, Math.floor(v / snap + 1e-6) * snap) : Math.max(1, roundEven(v)));
    return [fix(W * s), fix(H * s)];
}

const COMMON_AR = [[1,1],[2,3],[3,2],[3,4],[4,3],[4,5],[5,4],[9,16],[16,9],[1,2],[2,1],[21,9]];

// Short form for the ~120 px card, long form for the tooltip.
export function describeAR(w, h) {
    if (!w || !h) return { short: "", long: "" };
    const r = w / h;
    let best = COMMON_AR[0], bd = Infinity;
    for (const [a, b] of COMMON_AR) {
        const d = Math.abs(a / b - r);
        if (d < bd) { bd = d; best = [a, b]; }
    }
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = gcd(w, h) || 1;
    const ew = w / g, eh = h / g;
    const nearest = `${best[0]}:${best[1]}`;
    if (ew === best[0] && eh === best[1]) return { short: nearest, long: nearest };
    if (Math.max(ew, eh) <= 32)
        return { short: `${ew}:${eh}`, long: `exactly ${ew}:${eh}, nearest ${nearest}` };
    return { short: `≈${nearest}`, long: `${r.toFixed(2)}:1, nearest ${nearest}` };
}

/* ---------------------------------------------------------------- settings panel */

// The ⚙ panel: fixed on document.body, follows the node, closes on ⚙, ✕, Done or Esc.
let _sp = null;
export function closeSettings() {
    if (!_sp) return;
    const p = _sp;
    _sp = null;
    if (p.el.contains(document.activeElement)) document.activeElement.blur();
    p.stop();
    p.el.remove();
}

// draw(body, chips, head) fills the body; reset() puts the defaults back.
export function openSettings(app, node, title, draw, reset) {
    if (_sp && _sp.node === node) { closeSettings(); return; }
    closeSettings();
    const el = document.createElement("div");
    el.className = "luna-al-sp";
    el.innerHTML = `<div class="hd"><span>⚙</span><span class="t"></span><span class="x" title="Close">✕</span></div>
        <div class="bd"></div><div class="ft"><span class="rs" title="Back to the defaults">reset</span><span class="dn">Done</span></div>`;
    el.querySelector(".t").textContent = title;
    const body = el.querySelector(".bd");
    const chips = (label, opts, cur, f, note) => {
        const r = document.createElement("div");
        r.className = "rw";
        r.innerHTML = `<span class="l"></span><span class="ch"></span>`;
        r.querySelector(".l").textContent = label;
        for (const [v, text] of opts) {
            const c = document.createElement("span");
            c.textContent = text;
            if (String(v) === String(cur)) c.className = "on";
            c.onclick = () => { f(v); redraw(); };
            r.querySelector(".ch").appendChild(c);
        }
        body.appendChild(r);
        if (note) { const n = document.createElement("div"); n.className = "nt"; n.textContent = note; body.appendChild(n); }
    };
    const head = (icon, text) => { const h = document.createElement("div"); h.className = "sh"; h.textContent = `${icon}  ${text}`; body.appendChild(h); };
    const redraw = () => { body.replaceChildren(); draw(body, chips, head); };
    redraw();
    el.querySelector(".x").onclick = closeSettings;
    el.querySelector(".dn").onclick = closeSettings;
    el.querySelector(".rs").onclick = () => { reset(); redraw(); };
    el.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closeSettings(); } else e.stopPropagation(); });
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
    el.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
    document.body.appendChild(el);

    let raf = 0, alive = true;
    const follow = () => {
        if (!alive) return;
        if (!node.graph || app.canvas?.graph !== node.graph) { closeSettings(); return; }
        const ds = app.canvas.ds, r = app.canvas.canvas.getBoundingClientRect();
        const th = window.LiteGraph?.NODE_TITLE_HEIGHT || 30;
        el.style.left = `${r.left + (node.pos[0] + node.size[0] + 10 + ds.offset[0]) * ds.scale}px`;
        el.style.top = `${r.top + (node.pos[1] - th + ds.offset[1]) * ds.scale}px`;
        el.style.transform = `scale(${ds.scale})`;
        raf = requestAnimationFrame(follow);
    };
    follow();
    const onEsc = (e) => { if (e.key === "Escape" && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName || "")) closeSettings(); };
    document.addEventListener("keydown", onEsc, true);
    _sp = { el, node, stop() { alive = false; cancelAnimationFrame(raf); document.removeEventListener("keydown", onEsc, true); } };
}

/* ------------------------------------------------------- the count stepper */

// Drawn in the node BODY, top-left, beside the fixed output rows: the one spot that does not slide down as output
// sockets are added, and not on the title bar, where a double-click opens the rename editor.
const CTL = { left: 12, top: 8, boxH: 28, stepW: 30, numW: 40, gap: 5, labelGap: 10, minWidthForLabel: 300 };

export function countCtlRects(node, ctx, label) {
    let x = CTL.left;
    let lbl = null;
    if (label && node.size[0] >= CTL.minWidthForLabel) {
        const w = ctx ? ctx.measureText(label).width : label.length * 6.2;
        lbl = [x, 0, w, CTL.boxH];
        x += w + CTL.labelGap;
    }
    const sockets = (node.inputs || []).filter((i) => !i.widget).length;
    const y = CTL.top + (sockets ? sockets * (window.LiteGraph?.NODE_SLOT_HEIGHT || 20) + 4 : 0);
    const r = (rx, rw) => [rx, y, rw, CTL.boxH];
    if (lbl) lbl[1] = y;
    return {
        label: lbl,
        minus: r(x, CTL.stepW),
        num:   r(x + CTL.stepW + CTL.gap, CTL.numW),
        plus:  r(x + CTL.stepW + CTL.gap + CTL.numW + CTL.gap, CTL.stepW),
    };
}

const inRect = (pos, r) => pos[0] >= r[0] && pos[0] <= r[0] + r[2] && pos[1] >= r[1] && pos[1] <= r[1] + r[3];

export function ctlHit(node, pos) {
    const rects = node._lunaCtlRects || countCtlRects(node, null, "");
    if (inRect(pos, rects.minus)) return "minus";
    if (inRect(pos, rects.num)) return "num";
    if (inRect(pos, rects.plus)) return "plus";
    return "";
}

export function drawCtl(node, ctx, { label, min, max }) {
    const FONT = "'Segoe UI', system-ui, sans-serif";
    ctx.save();
    ctx.font = `12px ${FONT}`;
    const rects = countCtlRects(node, ctx, label);
    node._lunaCtlRects = rects;
    const raw = Number(widget(node, COUNT_WIDGET)?.value);
    const value = Math.max(min, Math.min(max, Number.isFinite(raw) ? raw : min));
    const hot = node._lunaCtlHover || "";
    if (rects.label) {
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = LUNA.muted;
        ctx.fillText(label, rects.label[0], rects.label[1] + rects.label[3] / 2 + 0.5);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const box = (r, glyph, key) => {
        const live = hot === key;
        const dead = (key === "minus" && value <= min) || (key === "plus" && value >= max);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(r[0], r[1], r[2], r[3], 5);
        else ctx.rect(r[0], r[1], r[2], r[3]);
        ctx.fillStyle = live && !dead ? LUNA.accent : LUNA.panel;
        ctx.fill();
        ctx.strokeStyle = dead ? LUNA.border : live ? LUNA.accent : LUNA.accentSoft;
        ctx.lineWidth = live ? 1.6 : 1.2;
        ctx.stroke();
        ctx.fillStyle = dead ? LUNA.muted : live ? LUNA.bg : (key === "num" ? LUNA.text : LUNA.accent);
        ctx.font = `bold ${key === "num" ? 14 : 17}px ${FONT}`;
        ctx.fillText(glyph, r[0] + r[2] / 2, r[1] + r[3] / 2 + (key === "num" ? 0.5 : 0));
    };
    box(rects.minus, "−", "minus");
    box(rects.num, String(value), "num");
    box(rects.plus, "+", "plus");
    ctx.restore();
}

/* ------------------------------------------------------------------- the picture section */

// The Pictures section: heading with "+ add" and the count, the card row (drag to reorder, ✕ to remove, click to
// select), the preview of the selected card, and its panel (role row, Size, Ratio, Snap, Apply to all, Reset).
//
// opts:
//   roles: [[value, text], ...]             the role row; roleOf(i) / setRole(i, value) read and write it
//   roleBadge(i): text or ""                a short role tag on the card ("" = none)
//   label(i): "Image 1"                     the card's name everywhere on the node
//   pic(i): text                            the number chip on the card (default i + 1)
//   active(i): bool                         false greys the card out (it is not sent)
//   previewH(): px                          the preview height
//   onRemove(i), onMove(from, to)           move the node's own per-card data (roles) with the picture
//   onChange()                              after any change, so the node re-renders and re-sizes
//   accept: "image/*"
export function buildPictureSection(node, opts) {
    const sec = document.createElement("div");
    sec.className = "luna-al-sec k-p";
    const h = document.createElement("div");
    h.className = "luna-al-sech";
    h.innerHTML = `<span class="tt">Pictures</span><span class="n"></span><span class="sp"></span>`;
    const add = document.createElement("div");
    add.className = "luna-al-btn add";
    add.textContent = "+ add";
    add.title = "Add pictures";
    h.appendChild(add);
    const countEl = h.querySelector(".n");

    const cards = document.createElement("div");
    cards.className = "luna-al-cards";
    const preview = document.createElement("div");
    preview.className = "luna-al-preview";
    const previewImg = document.createElement("img");
    const previewLbl = document.createElement("div");
    previewLbl.className = "luna-al-pvlbl";
    preview.append(previewImg, previewLbl);
    const panel = document.createElement("div");
    panel.className = "luna-al-panel";
    sec.append(h, cards, preview, panel);

    let selected = 0;
    let dragFrom = -1;
    const natural = new Map();        // filename -> [w, h], filled as thumbs decode
    const max = opts.max || 9;

    function render() {
        const paths = readPaths(node);
        cards.replaceChildren();
        countEl.textContent = `${paths.length} / ${max}`;
        if (!paths.length) {
            const e = document.createElement("div");
            e.className = "luna-al-empty";
            e.textContent = "No pictures. + add, or type file names in image_paths (Fields).";
            cards.appendChild(e);
        }
        if (paths.length > max) {
            const warn = document.createElement("div");
            warn.className = "luna-al-warn";
            warn.textContent = `At most ${max} pictures.`;
            cards.appendChild(warn);
        }
        paths.forEach((name, i) => {
            const card = document.createElement("div");
            card.className = "luna-al-card" + (i === selected ? " sel" : "") + (opts.active(i) ? "" : " off");
            card.draggable = true;
            card.onclick = () => { selected = i; render(); };
            const img = document.createElement("img");
            img.className = "luna-al-thumb";
            img.src = viewURL(name);
            const pic = document.createElement("div");
            pic.className = "luna-al-pic";
            pic.textContent = opts.pic ? opts.pic(i) : i + 1;
            pic.title = opts.label(i);
            const badge = opts.roleBadge ? opts.roleBadge(i) : "";
            const cap = document.createElement("div");
            cap.className = "luna-al-cap";
            const dim = document.createElement("div");
            dim.className = "luna-al-dim";
            dim.textContent = "…";
            const nm = document.createElement("div");
            nm.className = "luna-al-name";
            nm.textContent = name;
            nm.title = name;
            cap.append(dim, nm);
            img.onload = () => {
                const w = img.naturalWidth, hh = img.naturalHeight;
                const ar = describeAR(w, hh);
                dim.textContent = `${w}×${hh} ${ar.short}`;
                card.title = `${opts.label(i)}  —  ${w} × ${hh} (${ar.long})${opts.active(i) ? "" : "  · off"}\n${name}`;
                const had = natural.has(name);
                natural.set(name, [w, hh]);
                if (i === selected && !had) { renderPreview(); renderPanel(); }
            };
            img.onerror = () => { dim.textContent = "missing from input folder"; };
            const x = document.createElement("div");
            x.className = "luna-al-x";
            x.textContent = "✕";
            x.title = "Remove";
            x.onclick = (ev) => {
                ev.stopPropagation();
                const next = readPaths(node);
                next.splice(i, 1);
                moveItem(node, i, -1);
                opts.onRemove?.(i);
                writePaths(node, next);
                if (selected >= next.length) selected = Math.max(0, next.length - 1);
                opts.onChange?.();
            };
            card.addEventListener("dragstart", () => { dragFrom = i; card.classList.add("drag"); });
            card.addEventListener("dragend", () => { dragFrom = -1; card.classList.remove("drag"); render(); });
            card.addEventListener("dragover", (ev) => { ev.preventDefault(); card.classList.add("over"); });
            card.addEventListener("dragleave", () => card.classList.remove("over"));
            card.addEventListener("drop", (ev) => {
                ev.preventDefault();
                card.classList.remove("over");
                if (dragFrom < 0 || dragFrom === i) return;
                const next = readPaths(node);
                const [moved] = next.splice(dragFrom, 1);
                next.splice(i, 0, moved);
                moveItem(node, dragFrom, i);
                opts.onMove?.(dragFrom, i);
                writePaths(node, next);   // reorder IS the renumbering
                selected = i;
                opts.onChange?.();
            });
            card.append(img, pic);
            if (badge) {
                const b = document.createElement("div");
                b.className = "luna-al-role";
                b.textContent = badge;
                card.appendChild(b);
            }
            card.append(x, cap);
            cards.appendChild(card);
        });
        renderPreview();
        renderPanel();
    }

    function renderPreview() {
        const paths = readPaths(node);
        if (!paths.length) {
            preview.style.display = "none";
            previewImg.removeAttribute("src");
            previewImg.dataset.name = "";
            previewLbl.textContent = "";
            selected = 0;
            return;
        }
        if (selected >= paths.length) selected = 0;
        preview.style.display = "";
        preview.style.height = opts.previewH() + "px";
        const name = paths[selected];
        if (previewImg.dataset.name !== name) {
            previewImg.dataset.name = name;
            previewImg.src = viewURL(name);
        }
        const nat = natural.get(name);
        const out = nat ? predictSize(nat[0], nat[1], itemAt(node, selected)) : null;
        previewLbl.textContent = nat
            ? `${opts.label(selected)}  ${nat[0]}×${nat[1]} → ${out[0]}×${out[1]}${opts.active(selected) ? "" : "  · off"}`
            : opts.label(selected);
    }

    function renderPanel() {
        panel.replaceChildren();
        const paths = readPaths(node);
        panel.style.display = paths.length ? "" : "none";
        if (!paths.length) return;
        if (selected >= paths.length) selected = 0;
        const st = itemAt(node, selected);
        const nat = natural.get(paths[selected]);

        const row = (label) => {
            const r = document.createElement("div");
            r.className = "luna-al-row";
            if (label !== null) {
                const l = document.createElement("div");
                l.className = "luna-al-lbl";
                l.textContent = label;
                r.appendChild(l);
            }
            panel.appendChild(r);
            return r;
        };
        const seg = (parent, list, current, onPick) => {
            const s = document.createElement("div");
            s.className = "luna-al-seg";
            for (const [val, text] of list) {
                const o = document.createElement("div");
                o.className = "luna-al-opt" + (val === current ? " on" : "");
                o.textContent = text;
                o.onclick = () => onPick(val);
                s.appendChild(o);
            }
            parent.appendChild(s);
            return s;
        };
        const set = (patch) => { writeItem(node, selected, patch); opts.onChange?.(); };
        const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

        const head = row(null);
        const who = document.createElement("div");
        who.className = "luna-al-io";
        const out = nat ? predictSize(nat[0], nat[1], st) : null;
        who.innerHTML = nat
            ? `<b>${esc(opts.label(selected))}</b> &nbsp; loaded ${nat[0]}×${nat[1]} → output <b>${out[0]}×${out[1]}</b>`
            : `<b>${esc(opts.label(selected))}</b>`;
        head.appendChild(who);

        const rr = row("Role");
        seg(rr, opts.roles, opts.roleOf(selected), (v) => { opts.setRole(selected, v); opts.onChange?.(); });

        const r1 = row("Size");
        seg(r1, [["off", "Off"], ["max_mp", "Max MP"], ["longest_side", "Longest"],
                 ["scale_factor", "Scale ×"]], st.mode, (v) => set({ mode: v }));
        if (st.mode !== "off") {
            const r2 = row("");
            const inp = document.createElement("input");
            inp.className = "luna-al-num";
            inp.type = "number";
            inp.step = st.mode === "max_mp" ? "0.1" : (st.mode === "scale_factor" ? "0.05" : "32");
            inp.value = st.mode === "max_mp" ? st.max_mp
                      : st.mode === "longest_side" ? st.longest_side : st.scale_factor;
            inp.onchange = () => {
                const k = st.mode === "max_mp" ? "max_mp" : st.mode === "longest_side" ? "longest_side" : "scale_factor";
                set({ [k]: Number(inp.value) });
            };
            inp.addEventListener("keydown", (e) => e.stopPropagation());
            r2.appendChild(inp);
            const up = document.createElement("div");
            up.className = "luna-al-opt" + (st.allow_upscale ? " on" : "");
            up.textContent = "Upscale";
            up.title = "Off: a picture is never made bigger than it was loaded";
            up.onclick = () => set({ allow_upscale: !st.allow_upscale });
            r2.appendChild(up);
        }
        const r3 = row("Ratio");
        seg(r3, [["", "Source"], ["1:1", "1:1"], ["2:3", "2:3"], ["3:2", "3:2"],
                 ["9:16", "9:16"], ["16:9", "16:9"]], st.ratio, (v) => set({ ratio: v }));
        if (st.ratio) {
            const r4 = row("");
            seg(r4, [["crop", "Crop to fill"], ["pad", "Pad"]], st.ratio_action, (v) => set({ ratio_action: v }));
        }
        const r5 = row("Snap");
        seg(r5, [[0, "Off"], [8, "8"], [16, "16"], [32, "32"], [64, "64"]], Number(st.snap) || 0, (v) => set({ snap: v }));

        const r6 = row(null);
        const all = document.createElement("div");
        all.className = "luna-al-btn";
        all.textContent = "Apply to all";
        all.title = "Use these settings for every picture and drop per-picture overrides";
        all.onclick = () => {
            const { mode, max_mp, longest_side, scale_factor, ratio, ratio_action, crop_anchor, pad_color, snap, allow_upscale } = st;
            writeItem(node, 0, { mode, max_mp, longest_side, scale_factor, ratio, ratio_action, crop_anchor, pad_color,
                                 snap, allow_upscale }, true);
            opts.onChange?.();
        };
        const reset = document.createElement("div");
        reset.className = "luna-al-btn danger";
        reset.textContent = "Reset";
        reset.onclick = () => { const w = widget(node, STATE_WIDGET); if (w) { w.value = ""; w.callback?.(""); } opts.onChange?.(); };
        r6.append(all, reset);
    }

    return {
        el: sec, add, render,
        selected: () => selected,
        select: (i) => { selected = i; },
        // Panel rows that come and go, for the node's height arithmetic.
        panelExtraRows: () => {
            const st = itemAt(node, selected);
            return (st.mode !== "off" ? 1 : 0) + (st.ratio ? 1 : 0);
        },
    };
}

// Card grid height for n cards at the node's width (arithmetic on node.size, never measured from the layout).
export function gridHeight(node, n) {
    const GRID_SIDE = 35, CARD_MIN = 112, CARD_GAP = 8, EMPTY = 26;
    if (!n) return EMPTY;
    const gw = Math.max(CARD_MIN, node.size[0] - GRID_SIDE);
    const cols = Math.max(1, Math.floor((gw + CARD_GAP) / (CARD_MIN + CARD_GAP)));
    const card = (gw - (cols - 1) * CARD_GAP) / cols + 2;
    const rows = Math.ceil(n / cols);
    return rows * card + (rows - 1) * CARD_GAP + 4;
}
