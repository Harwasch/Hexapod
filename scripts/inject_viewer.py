#!/opt/hw-py/bin/python
"""Inject an orbit/zoom 3-D viewer of the concept skeletons into the review
page (docs/review/artifact.html), at the top of the Vision tab.

review-artifact builds the page from the repo but has no notion of a 3-D
model (see the friction log).  This reads build/models/manifest.json from
scripts/export_models.py, embeds the STL data as base64, and adds a viewer
that loads three.js from cdnjs — the one script host the Artifact sandbox
allows besides jsdelivr.

    /opt/hw-py/bin/python scripts/inject_viewer.py
"""
import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "review", "artifact.html")
MANIFEST = os.path.join(ROOT, "build", "models", "manifest.json")
THREE = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"

# Colours follow the 2-D figures in 01-sizing.md: femur red, tibia blue, coxa dark.
GROUPS = [
    ("body", "Body", "#9aa0a6"),
    ("actuators", "Actuators", "#0f9b8e"),
    ("coxa", "Coxa", "#3a3a3a"),
    ("femur", "Femur", "#c0392b"),
    ("tibia", "Tibia + foot", "#2980b9"),
    ("figure", "6 ft figure", "#c9c4b8"),
]

with open(MANIFEST) as f:
    manifest = json.load(f)

models = {}
for name, m in manifest.items():
    models[name] = {"title": m["title"], "ground_z": m["ground_z"], "stl": {}}
    for g, rel in m["files"].items():
        with open(os.path.join(ROOT, rel), "rb") as f:
            models[name]["stl"][g] = base64.b64encode(f.read()).decode("ascii")

concept_buttons = "".join(
    f'<button type="button" class="v3d-seg" data-concept="{n}" aria-pressed="{"true" if i == 0 else "false"}">{m["title"]}</button>'
    for i, (n, m) in enumerate(models.items()))
legend = "".join(f'<span class="v3d-chip"><i style="background:{c}"></i>{label}</span>' for g, label, c in GROUPS)

BLOCK = f"""
<!-- v3d:begin -->
<style>
.v3d{{margin:0 0 28px;border:1px solid var(--rule);border-radius:6px;background:var(--panel);overflow:hidden}}
.v3d-bar{{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--rule-soft);font-size:13px}}
.v3d-group{{display:flex;gap:4px;align-items:center}}
.v3d-label{{font-family:var(--cond);text-transform:uppercase;letter-spacing:.06em;font-size:11px;color:var(--muted);margin-right:4px}}
.v3d-seg,.v3d-view{{appearance:none;font:inherit;font-size:13px;color:var(--ink-2);background:var(--sunk);border:1px solid var(--rule-soft);border-radius:4px;padding:4px 10px;cursor:pointer}}
.v3d-seg[aria-pressed="true"],.v3d-view[aria-pressed="true"]{{color:var(--ink);border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}}
.v3d-seg:focus-visible,.v3d-view:focus-visible,.v3d input:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.v3d-check{{display:inline-flex;gap:6px;align-items:center;color:var(--ink-2);cursor:pointer}}
.v3d-stage{{position:relative;background:var(--sunk)}}
.v3d canvas{{display:block;width:100%;height:auto;touch-action:none;cursor:grab}}
.v3d canvas:active{{cursor:grabbing}}
.v3d-foot{{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between;padding:8px 12px;border-top:1px solid var(--rule-soft);font-size:12px;color:var(--muted)}}
.v3d-legend{{display:flex;flex-wrap:wrap;gap:6px 14px}}
.v3d-chip{{display:inline-flex;gap:6px;align-items:center;color:var(--ink-2)}}
.v3d-chip i{{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid rgba(0,0,0,.15)}}
.v3d-err{{padding:14px;color:var(--stop);font-size:13px}}
</style>
<div class="v3d" id="v3d">
  <div class="v3d-bar">
    <div class="v3d-group"><span class="v3d-label">Concept</span>{concept_buttons}</div>
    <div class="v3d-group"><span class="v3d-label">View</span>
      <button type="button" class="v3d-view" data-view="iso" aria-pressed="true">Iso</button>
      <button type="button" class="v3d-view" data-view="front" aria-pressed="false">Front</button>
      <button type="button" class="v3d-view" data-view="side" aria-pressed="false">Side</button>
      <button type="button" class="v3d-view" data-view="top" aria-pressed="false">Top</button>
    </div>
    <label class="v3d-check"><input type="checkbox" id="v3d-figure" checked> 6 ft figure</label>
    <label class="v3d-check"><input type="checkbox" id="v3d-grid" checked> 0.5 m grid</label>
  </div>
  <div class="v3d-stage"><canvas id="v3d-canvas" width="1200" height="720" aria-label="Interactive 3-D view of the hexapod skeleton concept"></canvas></div>
  <div class="v3d-foot"><div class="v3d-legend">{legend}</div><div>Drag to orbit · scroll or pinch to zoom · shift-drag or right-drag to pan · real geometry from <code>concepts/</code>, units mm</div></div>
</div>
<script src="{THREE}"></script>
<script>
(function(){{
  var MODELS = {json.dumps(models)};
  var GROUPS = {json.dumps([[g, c] for g, _, c in GROUPS])};
  var root = document.getElementById('v3d'), canvas = document.getElementById('v3d-canvas');
  if (!window.THREE) {{ root.querySelector('.v3d-stage').innerHTML = '<div class="v3d-err">three.js did not load, so the 3-D view is unavailable. The rendered images below are the same geometry.</div>'; return; }}

  function b64ToBuf(b64) {{ var bin = atob(b64), n = bin.length, u = new Uint8Array(n); for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i); return u.buffer; }}
  function parseSTL(buf) {{
    var dv = new DataView(buf), n = dv.getUint32(80, true), pos = new Float32Array(n * 9), o = 84;
    for (var i = 0; i < n; i++) {{ o += 12; for (var k = 0; k < 9; k++) {{ pos[i * 9 + k] = dv.getFloat32(o, true); o += 4; }} o += 2; }}
    var g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.computeVertexNormals(); return g;
  }}
  function cssVar(name, fallback) {{ var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v || fallback; }}

  var renderer = new THREE.WebGLRenderer({{canvas: canvas, antialias: true}});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(35, 1200 / 720, 10, 40000);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 0.9));
  var sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(1500, -2500, 3000); scene.add(sun);
  var fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-2000, 1500, 800); scene.add(fill);

  var built = {{}}, current = null, grid = null, meshes = {{}};
  function buildConcept(name) {{
    if (built[name]) return built[name];
    var m = MODELS[name], group = new THREE.Group(), parts = {{}};
    GROUPS.forEach(function (gc) {{
      var g = gc[0]; if (!m.stl[g]) return;
      var mat = new THREE.MeshStandardMaterial({{color: new THREE.Color(gc[1]), roughness: 0.6, metalness: g === 'actuators' ? 0.3 : 0.05, flatShading: true}});
      var mesh = new THREE.Mesh(parseSTL(b64ToBuf(m.stl[g])), mat); mesh.name = g; group.add(mesh); parts[g] = mesh;
    }});
    var gh = new THREE.GridHelper(4000, 8, 0x888888, 0x888888); gh.rotation.x = Math.PI / 2; gh.position.z = m.ground_z;
    gh.material.transparent = true; gh.material.opacity = 0.35; group.add(gh); parts.grid = gh;
    built[name] = {{group: group, parts: parts}}; return built[name];
  }}

  // Orbit state: spherical about a target, z up.
  var target = new THREE.Vector3(0, 0, 0), radius = 3200, az = 0.6, el = 0.45;
  var VIEWS = {{iso: [0.6, 0.42], front: [0, 0.08], side: [Math.PI / 2, 0.08], top: [0, Math.PI / 2 - 0.001]}};
  function applyCamera() {{
    var x = target.x + radius * Math.cos(el) * Math.cos(az), y = target.y + radius * Math.cos(el) * Math.sin(az), z = target.z + radius * Math.sin(el);
    camera.position.set(x, y, z); camera.lookAt(target); render();
  }}
  function fitTo(name) {{
    var b = built[name], box = new THREE.Box3();
    ['body', 'actuators', 'coxa', 'femur', 'tibia'].forEach(function (g) {{ if (b.parts[g]) box.expandByObject(b.parts[g]); }});
    box.getCenter(target); target.z = (box.min.z + box.max.z) / 2;
    radius = box.getSize(new THREE.Vector3()).length() * 1.15 + 600;
  }}
  function applyTheme() {{
    renderer.setClearColor(new THREE.Color(cssVar('--sunk', '#f4f3f0')));
    var gridC = new THREE.Color(cssVar('--rule', '#c3c2b7'));
    Object.keys(built).forEach(function (n) {{ built[n].parts.grid.material.color = gridC; }});
  }}
  function render() {{ renderer.render(scene, camera); }}
  function showConcept(name) {{
    if (current) scene.remove(built[current].group);
    var b = buildConcept(name); scene.add(b.group); current = name;
    b.parts.figure.visible = document.getElementById('v3d-figure').checked;
    b.parts.grid.visible = document.getElementById('v3d-grid').checked;
    applyTheme(); fitTo(name); applyCamera();
    root.querySelectorAll('.v3d-seg').forEach(function (bt) {{ bt.setAttribute('aria-pressed', bt.dataset.concept === name ? 'true' : 'false'); }});
  }}
  function setView(v) {{
    az = VIEWS[v][0]; el = VIEWS[v][1]; fitTo(current); applyCamera();
    root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.setAttribute('aria-pressed', bt.dataset.view === v ? 'true' : 'false'); }});
  }}
  function clearViewButtons() {{ root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.setAttribute('aria-pressed', 'false'); }}); }}

  // Pointer interaction
  var drag = null, pinch = null;
  canvas.addEventListener('contextmenu', function (e) {{ e.preventDefault(); }});
  canvas.addEventListener('pointerdown', function (e) {{ canvas.setPointerCapture(e.pointerId); drag = {{x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, id: e.pointerId}}; }});
  canvas.addEventListener('pointerup', function () {{ drag = null; pinch = null; }});
  canvas.addEventListener('pointercancel', function () {{ drag = null; pinch = null; }});
  canvas.addEventListener('pointermove', function (e) {{
    if (!drag || e.pointerId !== drag.id) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    var w = canvas.clientWidth || 1200;
    if (drag.pan) {{
      var right = new THREE.Vector3(), up = new THREE.Vector3(); camera.matrix.extractBasis(right, up, new THREE.Vector3());
      var k = radius * 0.0018 * (1200 / w) * 0.6; target.addScaledVector(right, -dx * k); target.addScaledVector(up, dy * k);
    }} else {{ az -= dx * 0.008 * (1200 / w) * 0.75; el = Math.max(-1.5, Math.min(1.5, el + dy * 0.008 * (1200 / w) * 0.75)); }}
    clearViewButtons(); applyCamera();
  }});
  canvas.addEventListener('wheel', function (e) {{ e.preventDefault(); radius = Math.max(300, Math.min(30000, radius * Math.exp(e.deltaY * 0.0012))); applyCamera(); }}, {{passive: false}});
  canvas.addEventListener('touchstart', function (e) {{ if (e.touches.length === 2) {{ pinch = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); drag = null; }} }}, {{passive: true}});
  canvas.addEventListener('touchmove', function (e) {{ if (e.touches.length === 2 && pinch) {{ var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); radius = Math.max(300, Math.min(30000, radius * pinch / d)); pinch = d; applyCamera(); e.preventDefault(); }} }}, {{passive: false}});

  root.querySelectorAll('.v3d-seg').forEach(function (bt) {{ bt.addEventListener('click', function () {{ showConcept(bt.dataset.concept); }}); }});
  root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.addEventListener('click', function () {{ setView(bt.dataset.view); }}); }});
  document.getElementById('v3d-figure').addEventListener('change', function (e) {{ built[current].parts.figure.visible = e.target.checked; render(); }});
  document.getElementById('v3d-grid').addEventListener('change', function (e) {{ built[current].parts.grid.visible = e.target.checked; render(); }});

  function resize() {{
    var w = canvas.clientWidth || 1200, h = Math.round(w * 0.6); renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); render();
  }}
  window.addEventListener('resize', resize);
  var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  if (mq && mq.addEventListener) mq.addEventListener('change', function () {{ applyTheme(); render(); }});
  new MutationObserver(function () {{ applyTheme(); render(); }}).observe(document.documentElement, {{attributes: true, attributeFilter: ['data-theme']}});
  // Tabs hide the panel; a hidden canvas has zero size, so re-fit when it becomes visible.
  new MutationObserver(function () {{ if (canvas.clientWidth) resize(); }}).observe(document.getElementById('p-vision'), {{attributes: true, attributeFilter: ['hidden', 'style', 'class']}});

  try {{ showConcept(Object.keys(MODELS)[0]); resize(); }}
  catch (err) {{ root.querySelector('.v3d-stage').innerHTML = '<div class="v3d-err">The 3-D view could not start (' + err.message + '). The rendered images below are the same geometry.</div>'; }}
}})();
</script>
<!-- v3d:end -->
"""

with open(PAGE) as f:
    page = f.read()
page = re.sub(r"\n<!-- v3d:begin -->.*?<!-- v3d:end -->\n", "\n", page, flags=re.S)   # idempotent
m = re.search(r'<section class="panel" role="tabpanel" id="p-vision"[^>]*>', page)
if not m:
    raise SystemExit("no vision panel in the review page")
# insert at the top of the vision panel's content: before the prose (the
# vision document), after the phase head and the questions box
at = page.find('<div class="prose">', m.end())
if at < 0:
    at = page.find("<figure", m.end())
if at < 0:
    raise SystemExit("no prose or figure in the vision panel")
page = page[:at] + BLOCK.lstrip("\n") + page[at:]
with open(PAGE, "w") as f:
    f.write(page)
print(f"injected 3-D viewer into {os.path.relpath(PAGE, ROOT)} ({os.path.getsize(PAGE)//1024} kB)")
