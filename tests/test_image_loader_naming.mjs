// Luna-ImageLoader card names: the order-and-tag rule the cards show (cardNames) and the ⚙ Target's round trip through
// the node's own media_state helpers.   Run from the pack root: node --test tests\test_image_loader_naming.mjs
//
// web/luna_image_loader.js imports ComfyUI's ../../scripts/app.js; a resolve hook gives it a stub app, so the test runs the
// node's real file. The Python side (image_loader.py) is compared with this rule in tests/test_image_loader.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

registerHooks({
    resolve(spec, ctx, next) {
        if (spec.endsWith("/scripts/app.js")) {
            return { url: "data:text/javascript,export const app = { registerExtension() {} };", shortCircuit: true };
        }
        return next(spec, ctx);
    },
    load(url, ctx, next) {
        if (url.startsWith("file:") && url.endsWith("/web/luna_image_loader.js")) {
            return { format: "module", source: fs.readFileSync(fileURLToPath(url), "utf8"), shortCircuit: true };
        }
        return next(url, ctx);
    },
});
const L = await import("../web/luna_image_loader.js");

const tags = (r) => r.cards.map((c) => c.tag);
const labels = (r) => r.cards.map((c) => c.label);
const A = ["ref", "style", "edit"];

test("A: MiniMax Image sends the edit base first", () => {
    const r = L.cardNames("minimax_image", A, 3);
    assert.deepEqual(r.order, [2, 0, 1]);
    assert.deepEqual(tags(r), ["<Picture 2>", "<Picture 3>", "<Picture 1>"]);
    assert.deepEqual(labels(r), ["@image1 · <Picture 2>", "@image2 · <Picture 3>", "@image3 · <Picture 1>"]);
});

test("B: Qwen, the same order, its own tags", () => {
    const r = L.cardNames("qwen", A, 3);
    assert.deepEqual(r.order, [2, 0, 1]);
    assert.deepEqual(labels(r), ["@image1 · <image2>", "@image2 · <image3>", "@image3 · <image1>"]);
});

test("C: General keeps card order; a missing or unknown target is General", () => {
    for (const t of ["general", undefined, "", "sdxl", "constructor", "toString", "__proto__", "hasOwnProperty", 3, {}]) {
        const r = L.cardNames(t, A, 3);
        assert.equal(r.target, "general");
        assert.deepEqual(r.order, [0, 1, 2]);
        assert.deepEqual(tags(r), ["Image 1", "Image 2", "Image 3"]);
    }
});

test("D: no edit base, MiniMax Image keeps card order", () => {
    const r = L.cardNames("minimax_image", ["ref", "style", "composition"], 3);
    assert.deepEqual(r.order, [0, 1, 2]);
    assert.deepEqual(tags(r), ["<Picture 1>", "<Picture 2>", "<Picture 3>"]);
});

// A node with the media_state widget, as ComfyUI gives it.
const mkNode = (value = "") => ({ widgets: [{ name: "media_state", value }] });
// The rule over the node's own state, as the cards call it.
const onNode = (node, n) => L.cardNames(L.targetOf(node), Array.from({ length: n }, (_, i) => L.roleOf(node, i)), n);

test("a role change and a Target change give the new names", () => {
    const node = mkNode();
    A.forEach((r, i) => L.setRole(node, i, r));
    L.setTarget(node, "qwen");
    assert.deepEqual(labels(onNode(node, 3)), ["@image1 · <image2>", "@image2 · <image3>", "@image3 · <image1>"]);
    // Card 1 set to edit base: setRole turns card 3 into a reference.
    L.setRole(node, 0, "edit");
    assert.deepEqual([0, 1, 2].map((i) => L.roleOf(node, i)), ["edit", "style", "ref"]);
    const q = onNode(node, 3);
    assert.deepEqual(q.order, [0, 1, 2]);
    assert.deepEqual(labels(q), ["@image1 · <image1>", "@image2 · <image2>", "@image3 · <image3>"]);
    // Back to A's roles, then Target qwen -> general: card order, Image N.
    L.setRole(node, 2, "edit");
    assert.deepEqual(onNode(node, 3).order, [2, 0, 1]);
    L.setTarget(node, "general");
    const g = onNode(node, 3);
    assert.deepEqual(g.order, [0, 1, 2]);
    assert.deepEqual(labels(g), ["@image1 · Image 1", "@image2 · Image 2", "@image3 · Image 3"]);
});

test("cards above the active count are off; a second edit base is a reference", () => {
    const r = L.cardNames("minimax_image", ["ref", "ref", "edit", "ref"], 2);
    assert.deepEqual(r.order, [0, 1]);
    assert.deepEqual(labels(r), ["@image1 · <Picture 1>", "@image2 · <Picture 2>", "@image3 · off", "@image4 · off"]);
    const twice = L.cardNames("minimax_image", ["ref", "edit", "edit"], 3);
    assert.deepEqual(twice.order, [1, 0, 2]);
    assert.deepEqual(L.cardNames("qwen", [], 0), { target: "qwen", order: [], cards: [] });
});

test("F: settings.target survives the node's write -> read helpers", () => {
    const node = mkNode();
    assert.equal(L.targetOf(node), "general");                       // no settings: General
    L.setTarget(node, "qwen");
    assert.deepEqual(JSON.parse(node.widgets[0].value), { settings: { target: "qwen" } });
    assert.equal(L.targetOf(node), "qwen");
    // Roles and other settings stay beside it.
    const m = L.readMedia(node);
    m.pictures = [{ role: "ref" }, { role: "style" }, { role: "edit" }];
    m.settings.node = { preview: "L" };
    L.writeMedia(node, m);
    const back = L.readMedia(node);
    assert.equal(back.settings.target, "qwen");
    assert.equal(back.settings.node.preview, "L");
    assert.deepEqual(back.pictures.map((p) => p.role), ["ref", "style", "edit"]);
    L.setTarget(node, "nonsense");
    assert.equal(L.targetOf(node), "general");
    // A saved node from before the Target: roles only, no settings.
    assert.equal(L.targetOf(mkNode(JSON.stringify({ pictures: [{ role: "edit" }] }))), "general");
    assert.equal(L.targetOf(mkNode("{not json")), "general");
});
