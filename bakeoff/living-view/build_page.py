"""The living-view comparison page, from the bake-off's artifacts.

    build_page.py OUT RUN [RUN ...]

Each RUN is an unpacked `living-view` artifact of `.github/workflows/living-view.yml`
(`summary.json`, `starts/`, `clips/<arm>/<start>.mp4`, `upscaled/<upscaler>/<arm>/<start>.mp4`);
a later run's results replace an earlier one's. For every start and arm it writes, under
`OUT/clips/`, the model's pixels at the render's size (A, bicubic), the upscaler's (A-up), and
our render moved by the model's plant motion (B: edge-aware, and the bilinear variant), with
`tools/captures/living_view.py`; then `OUT/index.html`, which refers to them by relative path.
The page follows the Artifact page contract (no document skeleton: the publisher adds it).
"""

from __future__ import annotations

import html
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "captures"))

import living_view as lv

ARM_ORDER = ["ltx", "causal", "flf", "wan"]
ARM_TITLES = {
    "ltx": "LTX-2.5 + Cinemagraph LoRA",
    "causal": "Causal Forcing",
    "causal~ctx3": "Causal Forcing, three-frame still context",
    "flf": "Wan 2.1 first-last-frame",
    "wan": "Wan 2.2 TI2V-5B (control)",
}
SCENE_TITLES = {"tree": "Minnetonka tree", "camp": "Camp shrubs"}
#: As in infra/modal/living_bakeoff.py ARMS, also for rows whose arm did not run.
LICENCES = {
    "ltx": "LTX-2.x Community License (commercial use free under $10M revenue)",
    "causal": "Apache-2.0",
    "flf": "Apache-2.0",
    "wan": "Apache-2.0",
}
#: The upscalers of A-up: the file suffix of each one's clips under OUT/clips, and its name.
UPSCALER_SUFFIX = {"flashvsr": "u", "seedvr2": "s"}
UPSCALER_TITLES = {"flashvsr": "FlashVSR v1.1", "seedvr2": "SeedVR2-3B"}
CAP_DOLLARS = 10.0
PAGE_CRF = 26
LEDGER = HERE / "cost-ledger.md"


def load_runs(runs: list[Path]) -> dict:
    """The runs merged: starts, results per arm, upscaled clips, costs, and where each clip is."""
    merged: dict = {"starts": {}, "results": {}, "upscaled": {}, "costs": [], "files": {}}
    for run in runs:
        summary: dict = {}
        for name in ("summary.json", "upscale-summary.json"):
            if (run / name).exists():
                part = json.loads((run / name).read_text())
                summary["costs"] = summary.get("costs", []) + part.get("costs", [])
                summary |= {k: v for k, v in part.items() if k != "costs"}
        merged["costs"] += [dict(row, run=run.name) for row in summary.get("costs", [])]
        for scene in summary.get("starts", []):
            for view in scene.get("views", []):
                merged["starts"][view["start"]] = view | {"dir": str(run / "starts")}
        for arm, result in summary.get("results", {}).items():
            if "clips" in result:
                merged["results"][arm] = result
                for name in result["clips"]:
                    merged["files"][(arm, name)] = run / "clips" / arm / f"{name}.mp4"
            elif arm not in merged["results"]:
                merged["results"][arm] = result
        for upscaler, result in summary.get("upscaled", {}).items():
            merged["upscaled"].setdefault(upscaler, {"clips": {}})
            if "clips" in result:
                merged["upscaled"][upscaler] |= {k: v for k, v in result.items() if k != "clips"}
                for key, clip in result["clips"].items():
                    merged["upscaled"][upscaler]["clips"][key] = clip | {
                        "path": str(run / "upscaled" / upscaler / f"{key}.mp4")
                    }
    return merged


def run_dollars(costs: list[dict], call: str) -> float:
    rows = [r for r in costs if r["call"] == call]
    return rows[-1]["dollars"] if rows else 0.0


def ledger(path: Path) -> tuple[dict[str, float], float | None]:
    """What the cost ledger booked: dollars per call, summed over the rows whose "what" opens
    with it ("arm wan: ...", "upscale flashvsr: ..."; a failed attempt's row does not), and the
    last running total. The ledger, not the runs' summaries, is the record: it corrects wall
    times that over-counted."""
    booked: dict[str, float] = {}
    total = None
    if not path.exists():
        return booked, total
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not cells[0][:1].isdigit():
            continue
        try:
            dollars, total = float(cells[5]), float(cells[6])
        except ValueError:
            continue
        call = re.match(r"(arm [\w~]+|upscale \w+):", cells[2])
        if call:
            booked[call.group(1)] = booked.get(call.group(1), 0.0) + dollars
    return booked, total


def jpeg(png: Path, out: Path, quality: int = 90) -> None:
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.open(png).convert("RGB").save(out, quality=quality, optimize=True)


def reencode(src: Path, dst: Path, crf: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]  # fmt: skip
    subprocess.run(command, check=True)


def process_all(merged: dict, out: Path, crf: int) -> dict:
    """Every start x arm through `living_view.process`, the upscaled clips re-encoded; the
    numbers by (start, arm)."""
    numbers: dict = {}
    clips = out / "clips"
    for (arm, name), clip in sorted(merged["files"].items()):
        start = name.split("~")[0]
        row = arm if "~" not in name else f"{arm}~{name.split('~')[1]}"
        view = merged["starts"].get(start)
        if view is None:
            continue
        base = Path(view["dir"])
        key = f"{start}-{row.replace('~', '-')}"
        share = base / f"{start}-share.png"
        result = lv.process(
            base / f"{start}.png",
            base / f"{start}-mask.png",
            clip,
            clips,
            key,
            share_path=share if share.exists() else None,
            crf=crf,
        )
        for upscaler, data in merged["upscaled"].items():
            up = data["clips"].get(f"{arm}/{name}")
            if up and Path(up["path"]).exists():
                suffix = UPSCALER_SUFFIX[upscaler]
                reencode(Path(up["path"]), clips / f"{key}-{suffix}.mp4", crf)
                frames, _fps = lv.read_clip(Path(up["path"]))
                render = lv.read_png(base / f"{start}.png")
                soft = lv.read_mask(base / f"{start}-mask.png")
                result[f"{upscaler}PsnrOutsideDb"] = round(lv.psnr_outside(frames, render, soft), 2)
        numbers[(start, row)] = result
        print(json.dumps(result), flush=True)
    return numbers


# --- the page --------------------------------------------------------------------------------

STYLE = """
/* Layout: one column of starts; each start is its still, then a row per arm with the model's
   pixels, the upscaled pixels and the motion-only clip side by side (stacked on a phone). */
:root {
  --bg: #f3f5f2; --surface: #fbfcfa; --ink: #1d2420; --muted: #5d6a62; --rule: #d5ddd6;
  --accent: #2f6a47; --accent-soft: #e2eee5; --mask: #b8337a; --warn: #9a5b00;
  --display: "Schibsted Grotesk", "Helvetica Neue", Arial, sans-serif;
  --body: "Public Sans", "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121714; --surface: #18201b; --ink: #e3ebe5; --muted: #9aa99f; --rule: #2c3830;
    --accent: #8cc9a0; --accent-soft: #1f3327; --mask: #f07bb8; --warn: #e3a64a;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121714; --surface: #18201b; --ink: #e3ebe5; --muted: #9aa99f; --rule: #2c3830;
  --accent: #8cc9a0; --accent-soft: #1f3327; --mask: #f07bb8; --warn: #e3a64a;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body);
  padding-inline: 16px; padding-block: 24px 48px; }
main { max-width: 1480px; margin: 0 auto; display: grid; gap: 40px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(1.7rem, 3.4vw, 2.5rem); font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: clamp(1.25rem, 2.2vw, 1.6rem); font-weight: 650; }
h3 { font-size: 1rem; font-weight: 650; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 600 0.72rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--muted); }
header { display: grid; gap: 14px; }
.legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.legend div { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 12px 14px; min-width: 0; }
.legend b { font-family: var(--mono); color: var(--accent); margin-right: 6px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.controls button { font: 600 0.85rem var(--body); color: var(--ink); background: var(--surface);
  border: 1px solid var(--rule); border-radius: 999px; padding: 6px 14px; cursor: pointer; }
.controls button[aria-pressed="true"] { background: var(--accent); color: var(--bg);
  border-color: var(--accent); }
.controls button:focus-visible, video:focus-visible { outline: 2px solid var(--accent);
  outline-offset: 2px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px;
  background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule);
  vertical-align: top; }
th { font: 600 0.72rem var(--mono); letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted); white-space: nowrap; }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
.start { display: grid; gap: 16px; padding-top: 8px; border-top: 2px solid var(--ink); }
.start-head { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline;
  justify-content: space-between; }
.still { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 16px;
  align-items: start; }
.still figure { margin: 0; position: relative; }
.still img { display: block; width: 100%; height: auto; border-radius: 4px; }
.still img.overlay { position: absolute; inset: 0; opacity: 0; transition: opacity 0.2s; }
.show-mask .still img.overlay { opacity: 1; }
.still .notes { display: grid; gap: 8px; font-size: 0.88rem; color: var(--muted); min-width: 0; }
.arm { display: grid; grid-template-columns: minmax(170px, 1fr) repeat(3, minmax(0, 2fr));
  gap: 10px; align-items: start; padding-block: 10px; border-top: 1px solid var(--rule); }
.arm .label { display: grid; gap: 6px; font-size: 0.82rem; min-width: 0; }
.arm .label dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 0; }
.arm .label dt { color: var(--muted); }
.arm .label dd { margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.cell { display: grid; gap: 4px; min-width: 0; }
.cell .tag { font: 600 0.72rem var(--mono); letter-spacing: 0.06em; color: var(--muted); }
.cell video { width: 100%; height: auto; aspect-ratio: 1280 / 704; background: var(--rule);
  border-radius: 3px; display: block; max-width: 100%; }
.cell .missing { aspect-ratio: 1280 / 704; display: grid; place-items: center; text-align: center;
  padding: 12px; border: 1px dashed var(--rule); border-radius: 3px; color: var(--muted);
  font-size: 0.82rem; }
.chip { display: inline-block; font: 600 0.7rem var(--mono); padding: 2px 8px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); }
.chip.warn { background: transparent; border: 1px solid var(--warn); color: var(--warn); }
.zoom .cell video { transform: scale(2); transform-origin: var(--zx, 50%) var(--zy, 30%); }
.cell .frame { overflow: hidden; border-radius: 3px; }
footer { font-size: 0.85rem; color: var(--muted); display: grid; gap: 8px; }
@media (max-width: 860px) {
  .arm { grid-template-columns: minmax(0, 1fr); }
  .still { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) { .still img.overlay { transition: none; } }
"""

SCRIPT = """
(() => {
  const root = document.documentElement;
  const videos = [...document.querySelectorAll('video[data-src]')];
  const load = (v) => { if (!v.src) { v.src = v.dataset.src; } };
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) { load(v); v.play().catch(() => {}); } else { v.pause(); }
    }
  }, { rootMargin: '200px 0px' });
  videos.forEach((v) => seen.observe(v));
  const toggle = (id, apply) => {
    const b = document.getElementById(id);
    if (!b) { return; }
    b.addEventListener('click', () => {
      const on = b.getAttribute('aria-pressed') !== 'true';
      b.setAttribute('aria-pressed', String(on));
      apply(on);
    });
  };
  toggle('b-bilinear', (on) => {
    document.querySelectorAll('video[data-guided]').forEach((v) => {
      const next = on ? v.dataset.bilinear : v.dataset.guided;
      v.dataset.src = next;
      if (v.src) { v.src = next; v.play().catch(() => {}); }
    });
    document.querySelectorAll('.b-name').forEach((s) => {
      s.textContent = on ? 'B · motion only, bilinear flow' : 'B · motion only, edge-aware flow';
    });
  });
  toggle('u-seedvr', (on) => {
    document.querySelectorAll('video[data-seedvr2]').forEach((v) => {
      const next = on ? v.dataset.seedvr2 : (v.dataset.flashvsr || v.dataset.seedvr2);
      v.dataset.src = next;
      if (v.src) { v.src = next; v.play().catch(() => {}); }
      const tag = v.closest('.cell').querySelector('.u-name');
      if (tag) {
        tag.textContent = on || !v.dataset.flashvsr
          ? 'A↑ · upscaled by SeedVR2' : 'A↑ · upscaled by FlashVSR';
      }
    });
  });
  toggle('show-mask', (on) => root.classList.toggle('show-mask', on));
  toggle('zoom', (on) => root.classList.toggle('zoom', on));
  toggle('pause', (on) => videos.forEach((v) => (on ? v.pause() : v.src && v.play().catch(() => {}))));
  document.querySelectorAll('.arm').forEach((row) => {
    row.addEventListener('dblclick', () => {
      row.querySelectorAll('video').forEach((v) => { v.currentTime = 0; v.play().catch(() => {}); });
    });
  });
})();
"""


def fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "–"
    if value == float("inf"):
        return "∞"
    return f"{value:.{digits}f}{unit}"


def build_html(merged: dict, numbers: dict, out: Path) -> str:
    esc = html.escape
    results = merged["results"]
    rows = [a for a in ARM_ORDER if a in results]
    extra = sorted({row for (_s, row) in numbers if "~" in row})
    upscale = merged["upscaled"].get("flashvsr", {})
    second = merged["upscaled"].get("seedvr2", {})
    booked, total = ledger(LEDGER)

    def cost(call: str) -> float:
        return booked[call] if call in booked else run_dollars(merged["costs"], call)

    up_cost = cost("upscale flashvsr")
    up_count = max(1, len(upscale.get("clips", {})))
    sv_cost = cost("upscale seedvr2")
    sv_count = max(1, len(second.get("clips", {})))
    spent = total if total is not None else sum(r["dollars"] for r in merged["costs"])

    def arm_cost(arm: str) -> float:
        return cost(f"arm {arm.split('~')[0]}")

    # The summary table: one line per arm.
    lines = []
    for arm in rows + extra:
        base = arm.split("~")[0]
        result = results.get(base, {})
        variant = arm.partition("~")[2]
        clips = {k: v for k, v in result.get("clips", {}).items() if k.partition("~")[2] == variant}
        first = [c["firstMotionSeconds"] for c in clips.values()]
        gen = [c["seconds"] for c in clips.values()]
        n = max(1, len(result.get("clips", {})))
        ups = [
            c["seconds"] for k, c in upscale.get("clips", {}).items() if k.startswith(f"{base}/")
        ]
        svs = [c["seconds"] for k, c in second.get("clips", {}).items() if k.startswith(f"{base}/")]
        status = (
            '<span class="chip">ran</span>'
            if clips
            else f'<span class="chip warn">did not run</span> {esc(str(result.get("error", ""))[:300])}'
        )
        lines.append(
            "<tr>"
            f"<td><b>{esc(ARM_TITLES.get(arm, arm))}</b><br>{status}</td>"
            f"<td>{esc(result.get('model', ''))}<br>"
            f"<span class='eyebrow'>{esc(result.get('licence', LICENCES.get(base, '')))}</span></td>"
            f"<td class='num'>{fmt(statistics.median(first) if first else None, ' s')}</td>"
            f"<td class='num'>{fmt(statistics.median(gen) if gen else None, ' s')}</td>"
            f"<td class='num'>{fmt(result.get('loadSeconds'), ' s', 0)}</td>"
            f"<td class='num'>${arm_cost(arm):.2f} · ${arm_cost(arm) / n:.2f}/clip</td>"
            f"<td class='num'>{fmt(statistics.median(ups) if ups else None, ' s')}</td>"
            + (
                f"<td class='num'>{fmt(statistics.median(svs) if svs else None, ' s')}</td>"
                if second
                else ""
            )
            + "</tr>"
        )
    table = (
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>model · licence</th>"
        "<th>first motion</th><th>generation</th><th>cold load</th><th>GPU $</th>"
        "<th>FlashVSR (A↑)</th>"
        + ("<th>SeedVR2 (A↑)</th>" if second else "")
        + "</tr></thead><tbody>"
        + "".join(lines)
        + "</tbody></table></div>"
    )

    sections = []
    for start in sorted(merged["starts"], key=lambda s: (s.split("-")[0] != "tree", s)):
        view = merged["starts"][start]
        scene, index = start.split("-")
        jpeg(Path(view["dir"]) / f"{start}.png", out / "stills" / f"{start}.jpg")
        jpeg(Path(view["dir"]) / f"{start}-overlay.png", out / "stills" / f"{start}-mask.jpg", 82)
        eye = ", ".join(f"{v:.1f}" for v in view["eye"])
        notes = (
            f"<p>Camera at ({eye}) m in the scan's frame, {view['candidate']}; plants fill "
            f"{100 * view.get('maskShare', view.get('plantFraction', 0)):.0f} % of the frame "
            f"(median plant depth {view.get('medianPlantDepth', 0):.1f} m).</p>"
            "<p>Rendered from the published splat with gsplat at 1280 × 704; the pink tint "
            "(Show plant mask) is the label render of the plant gaussians, feathered.</p>"
        )
        arms_html = []
        for arm in rows + [e for e in extra if True]:
            key = (start, arm)
            result = results.get(arm.split("~")[0], {})
            clip_name = start if "~" not in arm else f"{start}~{arm.split('~')[1]}"
            clip = result.get("clips", {}).get(clip_name)
            stem = f"clips/{start}-{arm.replace('~', '-')}"
            if key not in numbers or clip is None:
                why = esc(str(result.get("error", "no clip"))[:200])
                cells = "".join(
                    f"<div class='cell'><span class='tag'>{t}</span><div class='missing'>{why}</div></div>"
                    for t in ("A", "A↑", "B")
                )
                arms_html.append(
                    f"<div class='arm'><div class='label'><h3>{esc(ARM_TITLES.get(arm, arm))}</h3></div>{cells}</div>"
                )
                continue
            m = numbers[key]
            up = upscale.get("clips", {}).get(f"{arm.split('~')[0]}/{clip_name}")
            sv = second.get("clips", {}).get(f"{arm.split('~')[0]}/{clip_name}")
            label = (
                f"<div class='label'><h3>{esc(ARM_TITLES.get(arm, arm))}</h3><dl>"
                f"<dt>first motion</dt><dd>{fmt(clip['firstMotionSeconds'], ' s')}</dd>"
                f"<dt>generation</dt><dd>{fmt(clip['seconds'], ' s')} · {clip['frames']} f @ {clip['fps']:g} fps</dd>"
                f"<dt>model size</dt><dd>{clip['size'][0]} × {clip['size'][1]}</dd>"
                f"<dt>$ (arm/clips)</dt><dd>${arm_cost(arm) / max(1, len(result.get('clips', {}))):.2f}</dd>"
                f"<dt>licence</dt><dd>{esc(LICENCES.get(arm.split('~')[0], '').split(' (')[0])}</dd>"
                f"<dt>plant motion</dt><dd>{fmt(m['insideMeanPx'], '', 2)} mean · {fmt(m['insideP95Px'], '', 2)} p95 px</dd>"
                + (
                    f"<dt>elsewhere</dt><dd>{fmt(m['outsideMeanPx'], '', 2)} mean · {fmt(m['outsideP95Px'], '', 2)} p95 px</dd>"
                    f"<dt>camera creep</dt><dd>{fmt(m['cameraCreepPx'], ' px', 2)}</dd>"
                    if m["backgroundPx"]
                    else "<dt>elsewhere</dt><dd>– (only sky: nothing textured to measure)</dd>"
                    "<dt>camera creep</dt><dd>– (not removable: no background)</dd>"
                )
                + f"<dt>PSNR not-plant</dt><dd>A {fmt(m.get('psnrOutsideDb'), ' dB')}"
                + (f" · FlashVSR {fmt(m.get('flashvsrPsnrOutsideDb'), ' dB')}" if up else "")
                + (f" · SeedVR2 {fmt(m.get('seedvr2PsnrOutsideDb'), ' dB')}" if sv else "")
                + "</dd>"
                f"<dt>sky halo (B)</dt><dd>{fmt(m['haloPx']['guided'], '', 2)} edge-aware · {fmt(m['haloPx']['bilinear'], '', 2)} bilinear px</dd>"
                + (
                    f"<dt>FlashVSR</dt><dd>{fmt(up['seconds'], ' s')} · {fmt(up['framesPerSecond'], ' fps')} · ×{up['scale']} · ${up_cost / up_count:.3f}</dd>"
                    if up
                    else ""
                )
                + (
                    f"<dt>SeedVR2</dt><dd>{fmt(sv['seconds'], ' s')} · {fmt(sv['framesPerSecond'], ' fps')} · ${sv_cost / sv_count:.3f}</dd>"
                    if sv
                    else ""
                )
                + "</dl></div>"
            )

            def cell(tag: str, src: str, name: str = "", where: str = f"{start} {arm}") -> str:
                return (
                    f"<div class='cell'><span class='tag'>{name or tag}</span><div class='frame'>"
                    f"<video muted loop playsinline preload='none' data-src='{src}'"
                    f" aria-label='{esc(where)} {esc(tag)}'></video></div></div>"
                )

            has_u = (out / f"{stem}-u.mp4").exists()
            has_s = (out / f"{stem}-s.mp4").exists()
            first_up = f"{stem}-u.mp4" if has_u else f"{stem}-s.mp4"
            up_cell = (
                "<div class='cell'><span class='tag u-name'>"
                + ("A↑ · upscaled by FlashVSR" if has_u else "A↑ · upscaled by SeedVR2")
                + "</span><div class='frame'>"
                f"<video muted loop playsinline preload='none' data-src='{first_up}'"
                + (f" data-flashvsr='{stem}-u.mp4'" if has_u else "")
                + (f" data-seedvr2='{stem}-s.mp4'" if has_s else "")
                + f" aria-label='{esc(start)} {esc(arm)} A↑'></video></div></div>"
                if has_u or has_s
                else "<div class='cell'><span class='tag'>A↑</span><div class='missing'>not upscaled</div></div>"
            )
            b_cell = (
                "<div class='cell'><span class='tag b-name'>B · motion only, edge-aware flow</span>"
                f"<div class='frame'><video muted loop playsinline preload='none' data-src='{stem}-g.mp4'"
                f" data-guided='{stem}-g.mp4' data-bilinear='{stem}-b.mp4'"
                f" aria-label='{esc(start)} {esc(arm)} B'></video></div></div>"
            )
            arms_html.append(
                "<div class='arm'>"
                + label
                + cell("A", f"{stem}-a.mp4", name="A · model pixels")
                + up_cell
                + b_cell
                + "</div>"
            )
        sections.append(
            f"<section class='start' id='{esc(start)}'><div class='start-head'>"
            f"<h2>{esc(SCENE_TITLES.get(scene, scene))} · view {esc(index)}</h2>"
            f"<span class='eyebrow'>start {esc(start)}</span></div>"
            f"<div class='still'><figure><img src='stills/{esc(start)}.jpg' alt='Our render, {esc(start)}' width='1280' height='704'>"
            f"<img class='overlay' src='stills/{esc(start)}-mask.jpg' alt='' width='1280' height='704'></figure>"
            f"<div class='notes'><span class='eyebrow'>the still: our render</span>{notes}</div></div>"
            + "".join(arms_html)
            + "</section>"
        )

    head = f"""<title>Living View Bake-off</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Public+Sans:wght@400;600&family=Schibsted+Grotesk:wght@600;700&display=swap">
<style>{STYLE}</style>
"""
    body = f"""<main>
<header>
  <span class="eyebrow">Hexapod digital twin · Living Survey</span>
  <h1>Wind on a still scan: which video model, pixels or motion?</h1>
  <p>When the camera stops, a video model adds slight wind to the plants, anchored on the exact
  render of our splat, in short segments that always restart from that render. Each start below
  is that render; each row is one model given it with the same prompt (a gentle breeze, leaves
  and thin branches sway slightly, a locked-off tripod camera, nothing else changes), one seed.
  Clips loop; double-click a row to restart it in sync.</p>
  <div class="legend">
    <div><b>A</b>The model's own frames, resized to our 1280 × 704 render with plain bicubic.
    Everything you see is the model's drawing, including what it redrew around the plants.</div>
    <div><b>A↑</b>The same frames upscaled to 1280 × 704 by a video super-resolution model
    (FlashVSR v1.1{"; SeedVR2-3B with the toggle" if second else ""}): sharper pixels, still the
    model's drawing.</div>
    <div><b>B</b>Motion only: our own render, warped by the plant motion measured in the model's
    clip (OpenCV DIS optical flow, camera creep removed, outside the plants zeroed, eased to rest
    at the loop). Every pixel is the measured scan; only the movement is generated.</div>
  </div>
  <div class="controls" role="group" aria-label="Display">
    <button id="b-bilinear" type="button" aria-pressed="false">B with bilinear flow</button>
    {'<button id="u-seedvr" type="button" aria-pressed="false">A↑ by SeedVR2-3B</button>' if second else ""}
    <button id="show-mask" type="button" aria-pressed="false">Show plant mask</button>
    <button id="zoom" type="button" aria-pressed="false">Zoom 2×</button>
    <button id="pause" type="button" aria-pressed="false">Pause all</button>
  </div>
  {table}
  <p class="eyebrow">GPU spend for the whole bake-off ${spent:.2f} of the ${CAP_DOLLARS:.0f} cap ·
  first motion: time from the request to the first new frame on a warm GPU (whole clip for the
  non-streaming models) · elsewhere: motion of the textured pixels that are not plant
  (ground, buildings; not flat sky), before the camera is removed ·
  sky halo: B's motion on the ring of sky round the plants.</p>
</header>
{"".join(sections)}
<footer>
  <p>Made by infra/modal/living_bakeoff.py (the starts, the arms and the upscaler on Modal) and
  bakeoff/living-view/build_page.py with tools/captures/living_view.py (A, B, the numbers),
  branch bakeoff-living-view. Costs per call are in bakeoff/living-view/cost-ledger.md.</p>
</footer>
</main>
<script>{SCRIPT}</script>
"""
    return head + body


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    runs = [Path(a) for a in argv[1:]]
    out.mkdir(parents=True, exist_ok=True)
    merged = load_runs(runs)
    numbers = process_all(merged, out, PAGE_CRF)
    (out / "numbers.json").write_text(
        json.dumps({f"{s}|{a}": v for (s, a), v in numbers.items()}, indent=1)
    )
    (out / "index.html").write_text(build_html(merged, numbers, out), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"page: {out / 'index.html'}  total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
