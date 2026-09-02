#!/opt/hw-py/bin/python
"""Inject an orbit/zoom 3-D viewer of the actuator CAD into the review page's
Actuator tab, the way scripts/inject_viewer.py does for the skeleton in the
Vision tab.  Units come from build/cad/<tag>/{group,group-cut}.stl written by
cad/actuator/actuator.py; the viewer offers the whole unit or a quarter cut,
per colour group, for each unit tag listed in UNITS.

    /opt/hw-py/bin/python scripts/inject_actuator_viewer.py
"""
import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "review", "artifact.html")
THREE = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
UNITS = [("femur", "Femur / knee unit — two stators (canonical)"), ("femur-1s", "Single-stator variant (configuration B)")]
GROUPS = [("housing", "Housing: base, clamp rings, cover", "#9aa5ad"), ("rotor", "Rotor cup and shaft", "#d98c3a"),
          ("magnets", "Halbach magnets", "#c0392b"), ("stator", "Stator boards", "#0f9b8e"),
          ("reducer", "Cycloid: discs, pins, output flange", "#3a3a3a"), ("bearings", "Bearings", "#e0e0e0")]

models = {}
for tag, title in UNITS:
    d = os.path.join(ROOT, "build", "cad", tag)
    if not os.path.isdir(d):
        continue
    models[tag] = {"title": title, "stl": {}, "cut": {}}
    for g, _, _ in GROUPS:
        for key, suffix in (("stl", ""), ("cut", "-cut")):
            p = os.path.join(d, f"{g}{suffix}.stl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    models[tag][key][g] = base64.b64encode(f.read()).decode("ascii")
if not models:
    raise SystemExit("no actuator STL groups under build/cad; run cad/actuator/actuator.py first")

unit_buttons = "".join(
    f'<button type="button" class="v3d-seg" data-unit="{t}" aria-pressed="{"true" if i == 0 else "false"}">{m["title"]}</button>'
    for i, (t, m) in enumerate(models.items()))
legend = "".join(f'<span class="v3d-chip"><i style="background:{c}"></i>{label}</span>' for g, label, c in GROUPS)
group_checks = "".join(f'<label class="v3d-check"><input type="checkbox" class="v3da-group" data-group="{g}" checked> {label.split(":")[0]}</label>' for g, label, c in GROUPS)

BLOCK = f"""
<!-- v3da:begin -->
<div class="v3d" id="v3da">
  <div class="v3d-bar">
    <div class="v3d-group"><span class="v3d-label">Unit</span>{unit_buttons}</div>
    <div class="v3d-group"><span class="v3d-label">View</span>
      <button type="button" class="v3d-view" data-view="iso" aria-pressed="true">Iso</button>
      <button type="button" class="v3d-view" data-view="below" aria-pressed="false">Mounting face</button>
      <button type="button" class="v3d-view" data-view="side" aria-pressed="false">Side</button>
      <button type="button" class="v3d-view" data-view="top" aria-pressed="false">Top</button>
    </div>
    <label class="v3d-check"><input type="checkbox" id="v3da-cut" checked> Quarter cut</label>
    <div class="v3d-group">{group_checks}</div>
  </div>
  <div class="v3d-stage"><canvas id="v3da-canvas" width="1200" height="720" aria-label="Interactive 3-D view of the actuator CAD"></canvas></div>
  <div class="v3d-foot"><div class="v3d-legend">{legend}</div><div>Drag to orbit · scroll or pinch to zoom · shift-drag or right-drag to pan · the build123d model from <code>cad/actuator/actuator.py</code>, units mm, mounting face at z = 0</div></div>
</div>
<script src="{THREE}"></script>
<script>
(function(){{
  var MODELS = {json.dumps(models)};
  var GROUPS = {json.dumps([[g, c] for g, _, c in GROUPS])};
  var root = document.getElementById('v3da'), canvas = document.getElementById('v3da-canvas');
  if (!window.THREE) {{ root.querySelector('.v3d-stage').innerHTML = '<div class="v3d-err">three.js did not load, so the 3-D view is unavailable. The section and cutaway images below are the same geometry.</div>'; return; }}
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
  var camera = new THREE.PerspectiveCamera(30, 1200 / 720, 1, 5000);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 0.9));
  var sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(150, -250, 300); scene.add(sun);
  var fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-200, 150, 80); scene.add(fill);
  var built = {{}}, current = null;
  function buildUnit(tag) {{
    if (built[tag]) return built[tag];
    var m = MODELS[tag], group = new THREE.Group(), parts = {{whole: {{}}, cut: {{}}}};
    GROUPS.forEach(function (gc) {{
      var g = gc[0];
      [['whole', m.stl], ['cut', m.cut]].forEach(function (kv) {{
        if (!kv[1][g]) return;
        var mat = new THREE.MeshStandardMaterial({{color: new THREE.Color(gc[1]), roughness: 0.55, metalness: g === 'bearings' ? 0.5 : 0.15, flatShading: true, side: THREE.DoubleSide}});
        var mesh = new THREE.Mesh(parseSTL(b64ToBuf(kv[1][g])), mat); mesh.name = g; group.add(mesh); parts[kv[0]][g] = mesh;
      }});
    }});
    built[tag] = {{group: group, parts: parts}}; return built[tag];
  }}
  var target = new THREE.Vector3(0, 0, 30), radius = 500, az = -0.8, el = 0.5;
  var VIEWS = {{iso: [-0.8, 0.5], below: [-0.8, -0.6], side: [0, 0.02], top: [0, Math.PI / 2 - 0.001]}};
  function applyCamera() {{
    var x = target.x + radius * Math.cos(el) * Math.cos(az), y = target.y + radius * Math.cos(el) * Math.sin(az), z = target.z + radius * Math.sin(el);
    camera.position.set(x, y, z); camera.lookAt(target); render();
  }}
  function fitTo(tag) {{
    var b = built[tag], box = new THREE.Box3(); box.expandByObject(b.group);
    box.getCenter(target); radius = box.getSize(new THREE.Vector3()).length() * 1.6;
  }}
  function applyVisibility() {{
    if (!current) return;
    var cut = document.getElementById('v3da-cut').checked, b = built[current];
    var on = {{}}; root.querySelectorAll('.v3da-group').forEach(function (c) {{ on[c.dataset.group] = c.checked; }});
    Object.keys(b.parts.whole).forEach(function (g) {{ b.parts.whole[g].visible = !cut && on[g]; }});
    Object.keys(b.parts.cut).forEach(function (g) {{ b.parts.cut[g].visible = cut && on[g]; }});
    render();
  }}
  function applyTheme() {{ renderer.setClearColor(new THREE.Color(cssVar('--sunk', '#f4f3f0'))); }}
  function render() {{ renderer.render(scene, camera); }}
  function showUnit(tag) {{
    if (current) scene.remove(built[current].group);
    var b = buildUnit(tag); scene.add(b.group); current = tag;
    applyTheme(); fitTo(tag); applyVisibility(); applyCamera();
    root.querySelectorAll('.v3d-seg').forEach(function (bt) {{ bt.setAttribute('aria-pressed', bt.dataset.unit === tag ? 'true' : 'false'); }});
  }}
  function setView(v) {{
    az = VIEWS[v][0]; el = VIEWS[v][1]; fitTo(current); applyCamera();
    root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.setAttribute('aria-pressed', bt.dataset.view === v ? 'true' : 'false'); }});
  }}
  function clearViewButtons() {{ root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.setAttribute('aria-pressed', 'false'); }}); }}
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
  canvas.addEventListener('wheel', function (e) {{ e.preventDefault(); radius = Math.max(60, Math.min(3000, radius * Math.exp(e.deltaY * 0.0012))); applyCamera(); }}, {{passive: false}});
  canvas.addEventListener('touchstart', function (e) {{ if (e.touches.length === 2) {{ pinch = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); drag = null; }} }}, {{passive: true}});
  canvas.addEventListener('touchmove', function (e) {{ if (e.touches.length === 2 && pinch) {{ var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); radius = Math.max(60, Math.min(3000, radius * pinch / d)); pinch = d; applyCamera(); e.preventDefault(); }} }}, {{passive: false}});
  root.querySelectorAll('.v3d-seg').forEach(function (bt) {{ bt.addEventListener('click', function () {{ showUnit(bt.dataset.unit); }}); }});
  root.querySelectorAll('.v3d-view').forEach(function (bt) {{ bt.addEventListener('click', function () {{ setView(bt.dataset.view); }}); }});
  document.getElementById('v3da-cut').addEventListener('change', applyVisibility);
  root.querySelectorAll('.v3da-group').forEach(function (c) {{ c.addEventListener('change', applyVisibility); }});
  function resize() {{ var w = canvas.clientWidth || 1200, h = Math.round(w * 0.6); renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); render(); }}
  window.addEventListener('resize', resize);
  var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  if (mq && mq.addEventListener) mq.addEventListener('change', function () {{ applyTheme(); render(); }});
  new MutationObserver(function () {{ applyTheme(); render(); }}).observe(document.documentElement, {{attributes: true, attributeFilter: ['data-theme']}});
  new MutationObserver(function () {{ if (canvas.clientWidth) resize(); }}).observe(document.getElementById('p-actuator'), {{attributes: true, attributeFilter: ['hidden', 'style', 'class']}});
  try {{ showUnit(Object.keys(MODELS)[0]); resize(); }}
  catch (err) {{ root.querySelector('.v3d-stage').innerHTML = '<div class="v3d-err">The 3-D view could not start (' + err.message + '). The section and cutaway images below are the same geometry.</div>'; }}
}})();
</script>
<!-- v3da:end -->
"""

with open(PAGE) as f:
    page = f.read()
page = re.sub(r"\n<!-- v3da:begin -->.*?<!-- v3da:end -->\n", "\n", page, flags=re.S)   # idempotent
m = re.search(r'<section class="panel" role="tabpanel" id="p-actuator"[^>]*>', page)
if not m:
    raise SystemExit("no actuator panel in the review page")
at = page.find('<div class="prose">', m.end())
if at < 0:
    at = page.find("<figure", m.end())
if at < 0:
    raise SystemExit("no prose or figure in the actuator panel")
page = page[:at] + BLOCK.lstrip("\n") + page[at:]
with open(PAGE, "w") as f:
    f.write(page)
print(f"injected actuator 3-D viewer into {os.path.relpath(PAGE, ROOT)} ({os.path.getsize(PAGE)//1024} kB)")
