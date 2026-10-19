// The site picker on the request form: a plan-view map of an example site, drawn on a 2D canvas,
// where a visitor selects parcels or draws an area. Ported from the "Fleet Home" design artifact
// like `scene.js` (whose header says what changed); it needs no WebGL, so it loads with the page.

import WORLD from "./world.json";

// Site picker for the request form (plan-view map of an example site)
function startPicker(Wd, { features = {} } = {}) {
  const $ = (id) => document.getElementById(id);
  const cv = $("pk-map"),
    ctx = cv.getContext("2d"),
    wrap = cv.parentElement;
  const L = Wd.LAND;
  const PARCELS = [
    {
      id: "ridge",
      name: "North Ridge",
      addr: "1 Ridgeline Rd",
      ring: [
        [-800, -500],
        [-120, -500],
        [-160, -80],
        [-800, -40],
      ],
    },
    {
      id: "bench",
      name: "East Bench",
      addr: "40 Upper Bench Rd",
      ring: [
        [-120, -500],
        [800, -500],
        [800, -60],
        [260, -70],
        [-160, -80],
      ],
    },
    {
      id: "meadow",
      name: "West Meadow",
      addr: "7 Meadow Ln",
      ring: [
        [-800, -40],
        [-160, -80],
        [-320, 120],
        [-320, 300],
        [-800, 320],
      ],
    },
    {
      id: "home",
      name: "Home Ranch",
      addr: "100 Ranch Rd",
      ring: [
        [-160, -80],
        [260, -70],
        [220, 110],
        [120, 300],
        [-320, 300],
        [-320, 120],
      ],
    },
    {
      id: "creek",
      name: "Creek Flats",
      addr: "22 Creekside Dr",
      ring: [
        [260, -70],
        [800, -60],
        [800, 500],
        [400, 500],
        [120, 300],
        [220, 110],
      ],
    },
    {
      id: "south",
      name: "South Lots",
      addr: "9 Quarry Rd",
      ring: [
        [-800, 320],
        [-320, 300],
        [120, 300],
        [400, 500],
        [-800, 500],
      ],
    },
  ];
  const M2_AC = 4046.856;
  const area = (r) => {
    let a = 0;
    for (let i = 0; i < r.length; i++) {
      const p = r[i],
        q = r[(i + 1) % r.length];
      a += p[0] * q[1] - q[0] * p[1];
    }
    return Math.abs(a) / 2;
  };
  const inside = ([x, y], r) => {
    let c = false;
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i],
        [xj, yj] = r[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  const centroid = (r) =>
    r.reduce((a, p) => [a[0] + p[0] / r.length, a[1] + p[1] / r.length], [0, 0]);
  const fmt = (m2) => {
    const ac = m2 / M2_AC;
    return ac >= 100 ? ac.toFixed(0) : ac.toFixed(1);
  };

  // state
  let mode = "select",
    sel = new Set(["home"]),
    shapes = [],
    draft = [],
    hover = null,
    example = true,
    nShape = 0;
  let W = 1,
    H = 1,
    dpr = 1;
  const view = { cx: 0, cy: 0, s: 1 }; // px per metre
  let anim = null;
  const fit = (r, pad = 0.12) => {
    const xs = r.map((p) => p[0]),
      ys = r.map((p) => p[1]);
    const x0 = Math.min(...xs),
      x1 = Math.max(...xs),
      y0 = Math.min(...ys),
      y1 = Math.max(...ys);
    return {
      cx: (x0 + x1) / 2,
      cy: (y0 + y1) / 2,
      s: Math.min(W / ((x1 - x0) * (1 + 2 * pad)), H / ((y1 - y0) * (1 + 2 * pad))),
    };
  };
  const landRing = [
    [L.x, L.y],
    [L.x + L.w, L.y],
    [L.x + L.w, L.y + L.d],
    [L.x, L.y + L.d],
  ];
  const toS = ([x, y]) => [(x - view.cx) * view.s + W / 2, (y - view.cy) * view.s + H / 2];
  const toW = (sx, sy) => [(sx - W / 2) / view.s + view.cx, (sy - H / 2) / view.s + view.cy];
  const clampView = () => {
    const sMin = fit(landRing, 0.04).s * 0.8;
    view.s = Math.max(sMin, Math.min(12, view.s));
    view.cx = Math.max(L.x, Math.min(L.x + L.w, view.cx));
    view.cy = Math.max(L.y, Math.min(L.y + L.d, view.cy));
  };
  function flyTo(t) {
    const a = { ...view },
      t0 = performance.now(),
      D = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 520;
    anim = (now) => {
      const u = Math.min(1, (now - t0) / D),
        e = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
      view.cx = a.cx + (t.cx - a.cx) * e;
      view.cy = a.cy + (t.cy - a.cy) * e;
      view.s = Math.exp(Math.log(a.s) + (Math.log(t.s) - Math.log(a.s)) * e);
      draw();
      if (u < 1) requestAnimationFrame(anim);
      else anim = null;
    };
    requestAnimationFrame(anim);
  }

  // colours from the page tokens (read on every draw so the theme toggle just works)
  const tok = () => {
    const cs = getComputedStyle(document.documentElement),
      g = (n) => cs.getPropertyValue(n).trim();
    return {
      bg: g("--stage"),
      tile: g("--tile"),
      ink: g("--ink"),
      text: g("--text"),
      muted: g("--muted"),
      line: g("--line"),
      line2: g("--line2"),
      acc: g("--accent"),
    };
  };

  function path(r) {
    ctx.beginPath();
    r.forEach((p, i) => {
      const [x, y] = toS(p);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.closePath();
  }
  function draw() {
    const c = tok();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, W, H);
    // the land: fill, contour terraces, creek, road, trees
    path(landRing);
    ctx.fillStyle = c.tile;
    ctx.fill();
    ctx.strokeStyle = c.line2;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.strokeStyle = c.line2;
    ctx.lineWidth = 1;
    for (const h of Wd.HILLS)
      for (const l of h.levels) {
        path(l.ring);
        ctx.stroke();
      }
    ctx.save();
    path([...Wd.bankL, ...[...Wd.bankR].reverse()]);
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = c.acc;
    ctx.fill();
    ctx.globalAlpha = 0.6;
    ctx.strokeStyle = c.acc;
    ctx.stroke();
    ctx.restore();
    path(Wd.road);
    ctx.fillStyle = c.line;
    ctx.fill();
    ctx.strokeStyle = c.line2;
    ctx.stroke();
    ctx.strokeStyle = c.muted;
    ctx.lineWidth = 0.8;
    const tr = Math.max(1.2, 3.2 * view.s);
    for (const t of Wd.TREES) {
      const [x, y] = toS([t.x, t.y]);
      if (x < -10 || y < -10 || x > W + 10 || y > H + 10) continue;
      ctx.beginPath();
      ctx.arc(x, y, tr * (t.s || 1), 0, 6.283);
      ctx.stroke();
    }
    // assets on the site
    ctx.strokeStyle = c.text;
    ctx.fillStyle = c.line;
    ctx.lineWidth = 1;
    for (const r of features.rects || []) {
      path(r);
      ctx.fill();
      ctx.stroke();
    }
    for (const pl of features.lines || []) {
      ctx.save();
      if (pl.dash) ctx.setLineDash(pl.dash);
      ctx.strokeStyle = pl.tone === "acc" ? c.acc : c.text;
      ctx.lineWidth = pl.w || 1;
      ctx.beginPath();
      pl.pts.forEach((p, i) => {
        const [x, y] = toS(p);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }
    for (const p of features.dots || []) {
      const [x, y] = toS(p);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 6.283);
      ctx.fillStyle = c.text;
      ctx.fill();
    }
    // parcels
    ctx.font = '500 11px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const p of PARCELS) {
      const on = sel.has(p.id),
        hv = hover === p.id && mode === "select";
      path(p.ring);
      if (on || hv) {
        ctx.save();
        ctx.globalAlpha = on ? 0.2 : 0.08;
        ctx.fillStyle = c.acc;
        ctx.fill();
        ctx.restore();
      }
      ctx.save();
      if (!on) ctx.setLineDash([5, 4]);
      ctx.strokeStyle = on ? c.acc : c.muted;
      ctx.lineWidth = on ? 2 : 1;
      ctx.stroke();
      ctx.restore();
      const [x, y] = toS(centroid(p.ring));
      ctx.fillStyle = on ? c.acc : c.muted;
      ctx.fillText(p.name.toUpperCase(), x, y - 7);
      ctx.fillStyle = on ? c.ink : c.muted;
      ctx.fillText(fmt(area(p.ring)) + " ac", x, y + 8);
    }
    // drawn areas
    for (const sh of shapes) {
      path(sh.ring);
      ctx.save();
      ctx.globalAlpha = 0.2;
      ctx.fillStyle = c.acc;
      ctx.fill();
      ctx.restore();
      ctx.strokeStyle = c.acc;
      ctx.lineWidth = 2;
      ctx.stroke();
      const [x, y] = toS(centroid(sh.ring));
      ctx.fillStyle = c.ink;
      ctx.fillText(sh.name.toUpperCase() + " · " + fmt(area(sh.ring)) + " ac", x, y);
    }
    if (draft.length) {
      ctx.strokeStyle = c.acc;
      ctx.lineWidth = 2;
      ctx.beginPath();
      draft.forEach((p, i) => {
        const [x, y] = toS(p);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      if (hover && hover.pt) {
        const [x, y] = toS(hover.pt);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      draft.forEach((p, i) => {
        const [x, y] = toS(p);
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 6 : 4, 0, 6.283);
        ctx.fillStyle = i === 0 ? c.acc : c.tile;
        ctx.fill();
        ctx.strokeStyle = c.acc;
        ctx.stroke();
      });
      if (draft.length > 2) {
        const [x, y] = toS(centroid(draft));
        ctx.fillStyle = c.ink;
        ctx.fillText(fmt(area(draft)) + " ac", x, y);
      }
    }
    // scale bar (metres and feet) and north
    const nice = [10, 20, 50, 100, 200, 500, 1000],
      m = nice.find((v) => v * view.s > 70) || 1000,
      px = m * view.s;
    ctx.textAlign = "left";
    ctx.fillStyle = c.text;
    ctx.strokeStyle = c.text;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(14, H - 16);
    ctx.lineTo(14 + px, H - 16);
    ctx.moveTo(14, H - 21);
    ctx.lineTo(14, H - 11);
    ctx.moveTo(14 + px, H - 21);
    ctx.lineTo(14 + px, H - 11);
    ctx.stroke();
    ctx.fillText(`${m} m · ${Math.round(m * 3.281)} ft`, 14, H - 30);
    ctx.textAlign = "center";
    ctx.fillText("N", W - 22, 22);
    ctx.beginPath();
    ctx.moveTo(W - 22, 32);
    ctx.lineTo(W - 22, 50);
    ctx.moveTo(W - 27, 37);
    ctx.lineTo(W - 22, 31);
    ctx.lineTo(W - 17, 37);
    ctx.stroke();
  }

  // ── list, total, acres field ──
  const list = $("pk-list"),
    total = $("pk-total"),
    acres = $("f-acres"),
    chip = $("pk-ex"),
    hint = $("pk-hint");
  function update() {
    const items = [
      ...PARCELS.filter((p) => sel.has(p.id)).map((p) => ({
        k: "p:" + p.id,
        name: p.name,
        sub: p.addr,
        a: area(p.ring),
      })),
      ...shapes.map((s) => ({ k: "s:" + s.id, name: s.name, sub: "Drawn area", a: area(s.ring) })),
    ];
    list.replaceChildren(
      ...items.map((it) => {
        const li = document.createElement("li");
        const t = document.createElement("span");
        t.innerHTML = "<b></b><em></em>";
        t.querySelector("b").textContent = it.name;
        t.querySelector("em").textContent = it.sub;
        const a = document.createElement("span");
        a.className = "ac";
        a.textContent = fmt(it.a) + " ac";
        const x = document.createElement("button");
        x.type = "button";
        x.className = "rm";
        x.setAttribute("aria-label", "Remove " + it.name);
        x.textContent = "×";
        x.addEventListener("click", () => {
          touch();
          const [kind, id] = it.k.split(":");
          if (kind === "p") sel.delete(id);
          else shapes = shapes.filter((s) => String(s.id) !== id);
          update();
          draw();
        });
        li.append(t, a, x);
        return li;
      }),
    );
    // overlapping parcels and drawn areas count once: rasterize the union when there is more than one item
    const rings = [
      ...PARCELS.filter((p) => sel.has(p.id)).map((p) => p.ring),
      ...shapes.map((s) => s.ring),
    ];
    let sum = items.reduce((s, it) => s + it.a, 0);
    if (rings.length > 1) {
      const pip = ([x, y], r) => {
        let c = false;
        for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
          const [xi, yi] = r[i],
            [xj, yj] = r[j];
          if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
        }
        return c;
      };
      const all = rings.flat(),
        x0 = Math.min(...all.map((p) => p[0])),
        x1 = Math.max(...all.map((p) => p[0])),
        y0 = Math.min(...all.map((p) => p[1])),
        y1 = Math.max(...all.map((p) => p[1]));
      const c = Math.max(0.5, Math.sqrt(((x1 - x0) * (y1 - y0)) / 90000));
      let n = 0;
      for (let y = y0 + c / 2; y < y1; y += c)
        for (let x = x0 + c / 2; x < x1; x += c) if (rings.some((r) => pip([x, y], r))) n++;
      const u = n * c * c;
      if (u < sum * 0.995) sum = u;
    }
    total.textContent = items.length
      ? `${fmt(sum)} acres · ${(sum / 1e4).toFixed(1)} ha`
      : "Nothing selected";
    acres.value = items.length ? fmt(sum) : "";
    $("pk-empty").hidden = items.length > 0;
    chip.hidden = !example;
  }
  const touch = () => {
    if (example) {
      example = false;
      chip.hidden = true;
    }
  };

  // ── modes ──
  const modeBtns = [...document.querySelectorAll("[data-pk-mode]")];
  const finishBtn = $("pk-finish"),
    undoBtn = $("pk-undo");
  function setMode(m) {
    mode = m;
    draft = [];
    hover = null;
    modeBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.pkMode === m)));
    wrap.classList.toggle("drawing", m === "draw");
    hint.textContent =
      m === "draw"
        ? "Click the map to place corners. Click the first corner, or Finish, to close the shape."
        : "Click parcels to add or remove them. Drag to move, scroll or pinch to zoom.";
    syncDraft();
    draw();
  }
  function syncDraft() {
    finishBtn.disabled = draft.length < 3;
    undoBtn.disabled = draft.length === 0;
  }
  function finish() {
    if (draft.length < 3) return;
    nShape++;
    shapes.push({ id: nShape, name: "Area " + nShape, ring: draft });
    draft = [];
    touch();
    update();
    setMode("select");
  }
  modeBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.pkMode)));
  finishBtn.addEventListener("click", finish);
  undoBtn.addEventListener("click", () => {
    draft.pop();
    syncDraft();
    draw();
  });
  $("pk-clear").addEventListener("click", () => {
    touch();
    sel.clear();
    shapes = [];
    draft = [];
    syncDraft();
    update();
    draw();
  });
  $("pk-in").addEventListener("click", () => zoomAt(W / 2, H / 2, 1.6));
  $("pk-out").addEventListener("click", () => zoomAt(W / 2, H / 2, 1 / 1.6));
  $("pk-fit").addEventListener("click", () => flyTo(fit(landRing, 0.04)));
  addEventListener("keydown", (e) => {
    if (
      mode !== "draw" ||
      (!wrap.contains(document.activeElement) && document.activeElement !== document.body)
    )
      return;
    if (e.key === "Escape") setMode("select");
    if (e.key === "Enter") finish();
  });

  // ── pointer: click to select / place, drag to pan, wheel and pinch to zoom ──
  function zoomAt(sx, sy, k) {
    const [wx, wy] = toW(sx, sy);
    view.s *= k;
    clampView();
    view.cx = wx - (sx - W / 2) / view.s;
    view.cy = wy - (sy - H / 2) / view.s;
    clampView();
    draw();
  }
  const pts = new Map();
  let drag = null,
    pinch = null;
  const local = (e) => {
    const r = cv.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  cv.addEventListener("pointerdown", (e) => {
    try {
      cv.setPointerCapture(e.pointerId);
    } catch {
      // a pointer that is already gone: nothing to capture
    }
    pts.set(e.pointerId, local(e));
    if (pts.size === 1) drag = { p: local(e), moved: false };
    else if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]) };
      drag = null;
    }
  });
  cv.addEventListener("pointermove", (e) => {
    const p = local(e);
    if (pts.has(e.pointerId)) pts.set(e.pointerId, p);
    if (pinch && pts.size === 2) {
      const [a, b] = [...pts.values()],
        d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / pinch.d);
      pinch.d = d;
      return;
    }
    if (drag) {
      const dx = p[0] - drag.p[0],
        dy = p[1] - drag.p[1];
      if (drag.moved || Math.hypot(dx, dy) > 5) {
        drag.moved = true;
        view.cx -= dx / view.s;
        view.cy -= dy / view.s;
        clampView();
        drag.p = p;
        draw();
        return;
      }
    }
    const w = toW(...p);
    if (mode === "select") {
      const h = PARCELS.find((q) => inside(w, q.ring));
      const id = h ? h.id : null;
      if (id !== hover) {
        hover = id;
        draw();
      }
      cv.style.cursor = id ? "pointer" : "grab";
    } else {
      hover = { pt: w };
      cv.style.cursor = "crosshair";
      if (draft.length) draw();
    }
  });
  const up = (e) => {
    const was = drag;
    pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    if (!was || was.moved || pts.size) {
      drag = null;
      return;
    }
    drag = null;
    const w = toW(...local(e));
    if (mode === "select") {
      const h = PARCELS.find((q) => inside(w, q.ring));
      if (!h) return;
      touch();
      if (sel.has(h.id)) sel.delete(h.id);
      else sel.add(h.id);
      update();
      draw();
    } else {
      if (draft.length > 2) {
        const [fx, fy] = toS(draft[0]),
          [sx, sy] = local(e);
        if (Math.hypot(fx - sx, fy - sy) < 12) {
          finish();
          return;
        }
      }
      if (!inside(w, landRing)) return;
      draft.push(w);
      syncDraft();
      draw();
    }
  };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", (e) => {
    pts.delete(e.pointerId);
    drag = null;
    pinch = null;
  });
  cv.addEventListener("dblclick", (e) => {
    if (mode === "draw") {
      e.preventDefault();
      draft.pop();
      finish();
    }
  });
  cv.addEventListener("pointerleave", () => {
    if (mode === "select" && hover) {
      hover = null;
      draw();
    }
  });
  cv.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const [sx, sy] = local(e);
      zoomAt(sx, sy, Math.exp(-e.deltaY * 0.0015));
    },
    { passive: false },
  );

  // ── demo address search ──
  const q = $("pk-q"),
    res = $("pk-res");
  function showResults() {
    const s = q.value.trim().toLowerCase();
    const hits = PARCELS.filter((p) => !s || (p.name + " " + p.addr).toLowerCase().includes(s));
    const shown = hits.length ? hits : PARCELS;
    res.replaceChildren(
      ...(hits.length
        ? []
        : [
            Object.assign(document.createElement("li"), {
              className: "note",
              textContent: "No match in the demo. Sample parcels:",
            }),
          ]),
      ...shown.map((p) => {
        const li = document.createElement("li");
        const b = document.createElement("button");
        b.type = "button";
        b.innerHTML = "<b></b><em></em>";
        b.querySelector("b").textContent = p.addr;
        b.querySelector("em").textContent = `${p.name} · ${fmt(area(p.ring))} ac`;
        b.addEventListener("click", () => {
          touch();
          sel = new Set([p.id]);
          q.value = p.addr;
          res.hidden = true;
          setMode("select");
          update();
          flyTo(fit(p.ring));
        });
        li.append(b);
        return li;
      }),
    );
    res.hidden = false;
  }
  q.addEventListener("focus", showResults);
  q.addEventListener("input", showResults);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Escape") res.hidden = true;
    if (e.key === "Enter") {
      e.preventDefault();
      res.querySelector("button")?.click();
    }
  });
  document.addEventListener("pointerdown", (e) => {
    if (!res.contains(e.target) && e.target !== q) res.hidden = true;
  });

  // ── size, theme ──
  function resize() {
    const r = cv.getBoundingClientRect();
    const first = W === 1;
    W = Math.max(1, r.width);
    H = Math.max(1, r.height);
    dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    if (first) Object.assign(view, fit(landRing, 0.04));
    clampView();
    draw();
  }
  new ResizeObserver(resize).observe(cv);
  new MutationObserver(() => draw()).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => draw());
  document.fonts?.ready.then(() => draw());
  setMode("select");
  update();
  resize();
  return { summary: () => total.textContent, count: () => sel.size + shapes.length };
}

const FEATURES = {
  rects: [
    [
      [34, 245.5],
      [47, 245.5],
      [47, 254.5],
      [34, 254.5],
    ],
    [
      [-22, 158.2],
      [11.4, 158.2],
      [11.4, 160.2],
      [-22, 160.2],
    ],
    [
      [-22, 165.4],
      [11.4, 165.4],
      [11.4, 167.4],
      [-22, 167.4],
    ],
    [
      [-22, 172.6],
      [11.4, 172.6],
      [11.4, 174.6],
      [-22, 174.6],
    ],
    [
      [-22, 179.8],
      [11.4, 179.8],
      [11.4, 181.8],
      [-22, 181.8],
    ],
    [
      [-74.2, 240.3],
      [-67, 240.3],
      [-67, 246.5],
      [-74.2, 246.5],
    ],
    [
      [-330, 137],
      [-130, 137],
      [-130, 225.2],
      [-330, 225.2],
    ],
    [
      [-330, 137],
      [-130, 137],
      [-130, 221.2],
      [-330, 221.2],
    ],
    [
      [-330, 137],
      [-130, 137],
      [-130, 217.2],
      [-330, 217.2],
    ],
  ],
  lines: [
    {
      pts: [
        [-65, 191.5],
        [315, 191.5],
      ],
      dash: [6, 4],
      tone: "acc",
      w: 1.5,
    },
    {
      pts: [
        [-339.4, -118.8],
        [-297, -76.4],
        [-254.5, -33.9],
        [-212.1, 8.5],
        [-169.7, 50.9],
        [-127.3, 93.3],
        [-84.8, 135.8],
        [-42.4, 178.2],
        [0, 220.6],
        [42.5, 263.1],
        [84.9, 305.5],
        [127.3, 347.9],
        [169.7, 390.3],
        [212.2, 432.8],
      ],
    },
    {
      pts: [
        [-202, 230],
        [-128.6, 230],
      ],
    },
    {
      pts: [
        [-202, 232.4],
        [-128.6, 232.4],
      ],
    },
    {
      pts: [
        [-202, 234.8],
        [-128.6, 234.8],
      ],
    },
    {
      pts: [
        [-202, 237.2],
        [-128.6, 237.2],
      ],
    },
    {
      pts: [
        [-202, 239.6],
        [-128.6, 239.6],
      ],
    },
    {
      pts: [
        [-202, 242],
        [-128.6, 242],
      ],
    },
    {
      pts: [
        [-255, 152.6],
        [-135, 152.6],
      ],
      dash: [2, 3],
    },
  ],
  dots: [
    [-339.4, -118.8],
    [-297, -76.4],
    [-254.5, -33.9],
    [-212.1, 8.5],
    [-169.7, 50.9],
    [-127.3, 93.3],
    [-84.8, 135.8],
    [-42.4, 178.2],
    [0, 220.6],
    [42.5, 263.1],
    [84.9, 305.5],
    [127.3, 347.9],
    [169.7, 390.3],
    [212.2, 432.8],
  ],
};

/** The picker over the bundled world data, with the demo site's features. */
export const startSitePicker = () => startPicker(WORLD, { features: FEATURES });
