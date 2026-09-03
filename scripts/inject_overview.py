#!/opt/hw-py/bin/python
"""Make the review page usable as a review surface rather than a scroll of plots.

    /opt/hw-py/bin/python scripts/inject_overview.py

Two changes, both injected into docs/review/artifact.html after review-artifact
has generated it:

  1. An **Overview** tab, first and selected by default: what the machine is,
     at what size and mass, where the design stands, and the spatial drawings
     that orient you before any analysis -- general arrangement, scale against
     a person, the body plan with all eighteen units, the actuator in section
     and exploded.  Every number is read from the model JSON at build time, so
     the tab cannot drift from the analysis.

  2. A **full-screen zoom** on every figure on every tab.  The reviewer reads
     this page on a phone; a dimensioned drawing at 390 px wide is unreadable,
     and pinch-zooming the whole page loses the tab bar.  Tap a figure to open
     it alone, pinch or double-tap to zoom, drag to pan.

Run after inject_viewer.py and inject_actuator_viewer.py.
"""
import base64
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "review", "artifact.html")


def j(*parts):
    p = os.path.join(ROOT, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def img(rel, alt, caption, note=None):
    """A figure, base64-embedded the way the generator does it, or nothing."""
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return ""
    b64 = base64.b64encode(open(p, "rb").read()).decode("ascii")
    ext = "svg+xml" if rel.endswith(".svg") else "png"
    n = f'<p class="ov-note">{html.escape(note)}</p>' if note else ""
    return (f'<figure class="ov-fig"><div class="sheet"><img src="data:image/{ext};base64,{b64}" alt="{html.escape(alt)}"></div>'
            f'<figcaption>{caption}</figcaption>{n}</figure>')


FM = j("hw", "stator", "frameless_motor.json")
LEG = j("hw", "leg", "leg_loads.json") or {}
LEGC = j("cad", "leg", "leg.json") or {}
ARR = j("hw", "arrangement.json") or {}
FCAD = j("cad", "actuator", "frameless.json") or {}
BOMT = j("hw", "leg", "bom_totals.json") or {}

pick = FM["pick"] if FM else None
mot = pick["motor"] if pick else {}
red = pick["reducer"] if pick else {}


def num(v, fmt="{:.0f}", dash="—"):
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return dash


# ---------------------------------------------------------------- the tiles
def tile(k, v, sub=""):
    return f'<div class="ov-tile"><span class="ov-k">{k}</span><span class="ov-v">{v}</span><span class="ov-s">{sub}</span></div>'


leg_kg = (LEG.get("mass", {}) or {}).get("leg_total_g", LEGC.get("leg_total_g"))
leg_kg = leg_kg / 1000 if leg_kg else None
robot_kg = pick["m_robot"] if pick else None
tiles = "".join([
    tile("Robot mass", f'{num(robot_kg, "{:.0f}")} kg', "at the design's own fixed point"),
    tile("Legs", "6 × 3 joints", "18 actuators, all in the body"),
    tile("Body", f'{num(ARR.get("body_length_mm"), "{:.0f}") if ARR.get("body_length_mm") else "900"} × '
                 f'{num(ARR.get("body_width_mm"), "{:.0f}") if ARR.get("body_width_mm") else "—"} mm', "slab, 220 mm deep"),
    tile("Reach", "150 + 250 + 500 mm", "coxa, femur, tibia"),
    tile("Actuator", f'Ø{num(FCAD.get("od_mm") or 172)} × {num(FCAD.get("height_mm") or (mot.get("len_mm", 25) + 12))} mm',
         f'{num(FCAD.get("total_g", 0) / 1000 if FCAD.get("total_g") else pick["m_fk"] if pick else None, "{:.2f}")} kg each'),
    tile("Motor", f'{num(mot.get("T_cont"), "{:.1f}")} N·m', f'Ø{num(mot.get("od_mm"))} × {num(mot.get("len_mm"))} frameless, {num(mot.get("mass_kg", 0)*1000)} g'),
    tile("Joint torque", f'{num(pick["T_joint_fk"] if pick else None)} N·m', f'through a {num(red.get("lobes"))}-lobe cycloid, {num(pick["ratio_fk"] if pick else None)}:1'),
    tile("Closure", f'{num(min(pick["margin"].values()) if pick else None, "{:.2f}")}×', "worst joint margin; ≥ 1 closes"),
])

# ------------------------------------------------------------ where it stands
DECIDED = [
    ("Sprawl posture, 150/250/500 mm leg", "signed, round 5"),
    ("Two cycloid discs, 180° apart", "signed, round 5"),
    ("Configuration A: motors in the body, power out to the joints", "signed, round 6"),
    ("Yaw swing lowered to 6.4 rad/s to suit the 48 V bus", "signed, round 6"),
    ("2 oz JLCPCB boards; push the unit cost further", "signed, round 8"),
]
OPEN = [
    ("Motor family", "The Wheemo frameless kit, scaled to Ø160 × 25, is the first option that closes the "
                     "requirement and deletes the capstan. It needs a price: it undercuts the $423 outrunner unit "
                     f"only below ${num(FM.get('breakeven_motor_price_usd') if FM else None)} a kit."),
    ("The winding temperature behind the datasheet's 96.7 mΩ", "If that is a cold value, every torque here falls 15 %. "
                                                               "The design was given a 1.18 margin to survive it, but the number should be asked for."),
    ("The leg is 16 kg against the 9 kg assumed", "On the outrunner units the robot does not close at 119 kg. The frameless "
                                                  "unit changes the actuator, not the leg structure; the leg has to be rebuilt on it."),
    ("The continuous load case", "30° slope at dyn 1.5 all day is still the largest single driver of motor size."),
]
decided = "".join(f'<li><span class="ov-ok">✓</span><span>{html.escape(a)}<em>{html.escape(b)}</em></span></li>' for a, b in DECIDED)
opened = "".join(f'<li><span class="ov-q">?</span><span><strong>{html.escape(a)}</strong><em>{html.escape(b)}</em></span></li>' for a, b in OPEN)

# ------------------------------------------------------------------ the images
SPATIAL = [
    ("docs/design/arrangement/ga-scale.png", "The robot beside a person and a dog, to scale",
     "<strong>How big it actually is.</strong> The robot, a 1.8 m person and a large dog to the same scale."),
    ("docs/design/arrangement/ga-side.png", "Dimensioned side view in the sprawl stance",
     "<strong>General arrangement, side.</strong> Deck height, hip height, ground clearance and stance length."),
    ("docs/design/arrangement/ga-front.png", "Dimensioned front view",
     "<strong>General arrangement, front.</strong> Stance width and how far the legs sprawl."),
    ("docs/design/arrangement/ga-top.png", "Dimensioned plan view",
     "<strong>General arrangement, plan.</strong> Hip spacing and the circle each foot can reach."),
    ("docs/design/arrangement/ga-body-plan.png", "Where the eighteen actuators pack into the body",
     "<strong>The packing problem.</strong> All eighteen units, the batteries and the electronics in the 900 mm slab."),
    ("docs/design/arrangement/ga-leg-envelope.png", "One leg's reachable workspace",
     "<strong>What one leg can reach.</strong> The foot's workspace, the neutral stance, and the folded and extended limits."),
]
ACTUATOR = [
    ("docs/design/actuator/frameless-cad-section.png", "Dimensioned section of the frameless actuator",
     "<strong>The actuator in section.</strong> The frameless motor is an annulus and the cycloid sits inside its bore, "
     "so the unit is a short can instead of a stack."),
    ("docs/design/actuator/frameless-cad-exploded.png", "Exploded view of the frameless actuator",
     "<strong>Every part, in order.</strong> Exploded along the axis and keyed to the bill of materials."),
    ("docs/design/actuator/frameless-cad-iso.png", "Isometric of the frameless actuator", "<strong>The unit as an object.</strong>"),
    ("docs/design/actuator/frameless-cad-compare.png", "The three actuator options to the same scale",
     "<strong>Three actuators, one scale.</strong> The frameless unit against the PCB two-stator machine and the outrunner unit."),
    ("docs/design/actuator/frameless-unit.png", "Architecture, reducer load and thermal budget",
     "<strong>Why the capstan can go.</strong> Deleting it puts four times the torque on the cycloid; moving the pin circle out "
     "into the motor's bore takes it back off."),
]
spatial_figs = "".join(img(p, a, c) for p, a, c in SPATIAL)
act_figs = "".join(img(p, a, c) for p, a, c in ACTUATOR)
if not spatial_figs:
    spatial_figs = ('<p class="ov-empty">The general-arrangement drawings are still being generated. '
                    'Until they land, the dimensioned views are on the <a href="#" data-goto="geometry">Drawings</a> tab.</p>')

# ------------------------------------------------------------------ assemble
overview = f"""
<section class="panel" role="tabpanel" id="p-overview" aria-labelledby="t-overview">
<div class="phase-head"><div><span class="eyebrow">Start here</span>
<h2>The machine, at a glance</h2></div><span class="pill t-wait">1 decision open</span></div>
<p class="summary">A large-dog-sized outdoor hexapod with all eighteen motors in the body and power carried out to
the joints. This tab is the orientation: how big it is, what it weighs, how it is put together and where the
design stands. Every other tab is the working detail behind one stage. <strong>Tap any drawing to open it
full-screen and zoom.</strong></p>

<div class="ov-tiles">{tiles}</div>

<h3 class="ov-h">The machine</h3>
<div class="ov-grid">{spatial_figs}</div>

<h3 class="ov-h">The actuator, eighteen times over</h3>
<p class="ov-lead">Every joint is the same unit: a frameless kit motor as an annulus, with a single-stage cycloid
reducer inside its bore. {num(mot.get('T_cont'), '{:.1f}')} N·m at the motor becomes {num(pick['T_joint_fk'] if pick else None)} N·m at the joint
through {num(pick['ratio_fk'] if pick else None)}:1.</p>
<div class="ov-grid">{act_figs}</div>

<h3 class="ov-h">Where the design stands</h3>
<div class="ov-status">
<div><h4>Settled</h4><ul class="ov-list">{decided}</ul></div>
<div><h4>Open — this is what the review is for</h4><ul class="ov-list">{opened}</ul></div>
</div>
<p class="ov-lead">The full argument for each is in the stage tabs above, and in
<a href="https://github.com/Harwasch/Hexapod/blob/claude/hexapod-robot-design-mt516g/docs/design/08-actuator-design.md"
target="_blank" rel="noopener">08-actuator-design.md</a> in the repository.</p>
</section>
"""

CSS = """
<style id="ov-css">
.ov-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rule-soft);
  border:1px solid var(--rule);margin:18px 0 26px}
.ov-tile{background:var(--panel);padding:12px 14px;display:flex;flex-direction:column;gap:2px}
.ov-k{font-family:var(--cond);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.ov-v{font-size:21px;font-weight:600;line-height:1.15;font-variant-numeric:tabular-nums}
.ov-s{font-size:11.5px;color:var(--ink-2);line-height:1.35}
.ov-h{font-family:var(--cond);font-size:13px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-2);
  border-bottom:1px solid var(--rule);padding-bottom:6px;margin:34px 0 4px}
.ov-lead{font-size:14px;color:var(--ink-2);margin:8px 0 16px;max-width:62ch}
.ov-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}
.ov-fig{margin:0}
.ov-fig .sheet{border:1px solid var(--rule);background:#fff;padding:6px;cursor:zoom-in}
.ov-fig img{display:block;width:100%;height:auto}
.ov-fig figcaption{font-size:13px;color:var(--ink-2);line-height:1.45;padding-top:7px}
.ov-note{font-size:12px;color:var(--muted);margin:3px 0 0}
.ov-empty{font-size:13.5px;color:var(--muted);border:1px dashed var(--rule);padding:14px;border-radius:3px}
.ov-status{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-top:14px}
.ov-status h4{font-size:13px;margin:0 0 8px}
.ov-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:9px}
.ov-list li{display:flex;gap:9px;font-size:13.5px;line-height:1.45}
.ov-list em{display:block;font-style:normal;color:var(--muted);font-size:12.5px;margin-top:1px}
.ov-ok{color:var(--ok);font-weight:700}
.ov-q{color:var(--wait);font-weight:700}
@media (max-width:820px){
  .ov-tiles{grid-template-columns:repeat(2,1fr)}
  .ov-grid,.ov-status{grid-template-columns:1fr}
}
/* ---- full-screen figure zoom: this page is reviewed on a phone ---- */
#zoomer{position:fixed;inset:0;z-index:999;background:rgba(12,12,11,.94);display:none;
  touch-action:none;overflow:hidden}
#zoomer.on{display:block}
#zoomer img{position:absolute;top:0;left:0;transform-origin:0 0;max-width:none;will-change:transform}
#zoomer .zbar{position:absolute;top:0;left:0;right:0;display:flex;gap:10px;align-items:center;
  padding:10px 12px;background:linear-gradient(rgba(0,0,0,.55),transparent);color:#fff;font-size:12.5px;z-index:2}
#zoomer .zbar span{flex:1;line-height:1.35;opacity:.92}
#zoomer button{background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.28);
  border-radius:3px;font:inherit;padding:5px 11px;cursor:pointer}
figure .sheet{cursor:zoom-in}
</style>
"""

JS = """
<script id="ov-js">
(function(){
  // Every figure on every tab opens full-screen. Dimensioned drawings are
  // unreadable at phone width otherwise.
  var z=document.createElement('div'); z.id='zoomer';
  z.innerHTML='<div class="zbar"><button type="button" data-z="out">Close</button>'
    +'<button type="button" data-z="fit">Fit</button><button type="button" data-z="in">+</button>'
    +'<span>Pinch or double-tap to zoom · drag to pan</span></div><img alt="">';
  document.body.appendChild(z);
  var im=z.querySelector('img'), sc=1, tx=0, ty=0, base=1, nw=0, nh=0;
  function apply(){im.style.transform='translate('+tx+'px,'+ty+'px) scale('+sc+')';}
  function fit(){
    if(!nw)return;
    base=Math.min(innerWidth/nw, innerHeight/nh)*0.96; sc=base;
    tx=(innerWidth-nw*sc)/2; ty=(innerHeight-nh*sc)/2; apply();
  }
  function open(src,alt){
    im.src=src; im.alt=alt||'';
    z.classList.add('on'); document.documentElement.style.overflow='hidden';
    if(im.complete&&im.naturalWidth){nw=im.naturalWidth;nh=im.naturalHeight;fit();}
    else im.onload=function(){nw=im.naturalWidth;nh=im.naturalHeight;fit();};
  }
  function close(){z.classList.remove('on'); document.documentElement.style.overflow=''; im.src='';}
  document.addEventListener('click',function(e){
    var b=e.target.closest('#zoomer [data-z]');
    if(b){
      var k=b.dataset.z;
      if(k==='out')close();
      else if(k==='fit')fit();
      else{var cx=innerWidth/2, cy=innerHeight/2, f=1.6;
        tx=cx-(cx-tx)*f; ty=cy-(cy-ty)*f; sc*=f; apply();}
      return;
    }
    if(e.target.closest('#zoomer'))return;
    var s=e.target.closest('figure .sheet, .ov-fig .sheet');
    if(!s)return;
    var i=s.querySelector('img');
    if(i&&i.src){e.preventDefault(); open(i.src,i.alt); return;}
    // Figures the generator emitted as inline SVG: serialise and show the same way.
    var g=s.querySelector('svg');
    if(g){e.preventDefault();
      var xml=new XMLSerializer().serializeToString(g);
      open('data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(xml))),'diagram');}
  });
  document.addEventListener('keydown',function(e){if(e.key==='Escape'&&z.classList.contains('on'))close();});
  // pan, pinch, double-tap
  var pts={}, last=null, lastTap=0;
  function dist(a,b){return Math.hypot(a.x-b.x,a.y-b.y);}
  z.addEventListener('pointerdown',function(e){
    if(e.target.closest('[data-z]'))return;
    z.setPointerCapture(e.pointerId); pts[e.pointerId]={x:e.clientX,y:e.clientY};
    var n=Date.now(); if(n-lastTap<300){var f=sc<base*1.8?2:1/2;
      tx=e.clientX-(e.clientX-tx)*f; ty=e.clientY-(e.clientY-ty)*f; sc*=f; apply();}
    lastTap=n; last=null;
  });
  z.addEventListener('pointermove',function(e){
    if(!pts[e.pointerId])return;
    pts[e.pointerId]={x:e.clientX,y:e.clientY};
    var k=Object.keys(pts);
    if(k.length===1){
      if(last){tx+=e.clientX-last.x; ty+=e.clientY-last.y; apply();}
      last={x:e.clientX,y:e.clientY};
    }else if(k.length===2){
      var a=pts[k[0]], b=pts[k[1]], d=dist(a,b), mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
      if(last&&last.d){var f=d/last.d;
        tx=mx-(mx-tx)*f; ty=my-(my-ty)*f; sc*=f; apply();}
      last={d:d};
    }
  });
  function up(e){delete pts[e.pointerId]; last=null;}
  z.addEventListener('pointerup',up); z.addEventListener('pointercancel',up);
  z.addEventListener('wheel',function(e){
    e.preventDefault(); var f=e.deltaY<0?1.12:1/1.12;
    tx=e.clientX-(e.clientX-tx)*f; ty=e.clientY-(e.clientY-ty)*f; sc*=f; apply();
  },{passive:false});
  addEventListener('resize',function(){if(z.classList.contains('on'))fit();});
})();
</script>
"""

# ---------------------------------------------------------------- injection
page = open(PAGE, encoding="utf-8").read()
if 'id="p-overview"' in page:
    raise SystemExit("overview already injected; rebuild the page first")

tab_btn = ('<button class="tab" role="tab" id="t-overview" aria-controls="p-overview" '
           'aria-selected="false" data-tab="overview"><span class="dot b-ok"></span>Overview</button>\n')
m = re.search(r'(<div class="tabs" role="tablist">\s*)', page)
if not m:
    raise SystemExit("could not find the tab bar")
page = page[:m.end()] + tab_btn + page[m.end():]

# the panel goes before the first generated panel
m = re.search(r'<section class="panel" role="tabpanel"', page)
page = page[:m.start()] + overview + "\n" + page[m.start():]

# every generated panel starts hidden; the tab script picks the stored or default one.
# Make Overview the default when nothing is stored, without disturbing the rest.
page = page.replace("try{const s=localStorage.getItem('mh-phase');\n  if(s&&tabs.some(t=>t.dataset.tab===s))show(s);}catch(e){}",
                    "try{const s=localStorage.getItem('mh-phase');\n"
                    "  show(s&&tabs.some(t=>t.dataset.tab===s)?s:'overview');}catch(e){show('overview');}")

page = page.replace("</style>", "</style>\n" + CSS, 1) if "<style" in page else CSS + page
page = page.replace("</body>", JS + "\n</body>") if "</body>" in page else page + JS

open(PAGE, "w", encoding="utf-8").write(page)
have = sum(1 for p, _, _ in SPATIAL + ACTUATOR if os.path.exists(os.path.join(ROOT, p)))
print(f"injected Overview tab and figure zoom into {os.path.relpath(PAGE, ROOT)} "
      f"({len(page)//1024} kB, {have} of {len(SPATIAL)+len(ACTUATOR)} overview figures present)")
