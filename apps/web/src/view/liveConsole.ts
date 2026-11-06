/** DOM and plots for training; the renderer and snapshot lifecycle stay in live.ts. */
import type { LiveState } from "@twin/contracts";
import {
  blockCountText,
  effectiveCap,
  lossLines,
  snapshotLag,
  scrubPosition,
  seedsLabel,
  finalReadouts,
  chartDomain,
  chartPoints,
  ceilingText,
  compact,
  cost,
  count,
  eventsDisplay,
  isNewAttempt,
  lastStep,
  money,
  nearestIndex,
  phaseOf,
  pillText,
  qualityReadouts,
  snapshotAt,
  stopRule,
  timeText,
  type ChartId,
} from "./liveTrain";

const el = (id: string): HTMLElement => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing live console element: ${id}`);
  return node;
};
const text = (id: string, value: string): void => {
  el(id).textContent = value;
};
interface Chart {
  id: ChartId;
  title: string;
  color: string;
  format: (v: number) => string;
  optional?: boolean;
  mini?: boolean;
}
const CHARTS: Chart[] = [
  { id: "gs", title: "Gaussians", color: "#7fd8c0", format: compact },
  { id: "loss", title: "Training loss", color: "#d8c39a", format: (v) => v.toFixed(3) },
  { id: "psnr", title: "Held-out PSNR", color: "#b9a3f0", format: (v) => `${v.toFixed(2)} dB` },
  { id: "lr", title: "Position LR", color: "#7fd8c0", format: (v) => v.toExponential(1) },
  { id: "its", title: "Speed it/s", color: "#d8c39a", format: (v) => v.toFixed(1) },
  {
    id: "far",
    title: "Far-field",
    color: "#7fd8c0",
    format: (v) => `${v.toFixed(1)}%`,
    optional: true,
  },
  {
    id: "elongation",
    title: "Elongated",
    color: "#b9a3f0",
    format: (v) => `${v.toFixed(1)}%`,
    optional: true,
  },
  { id: "mem", title: "GPU memory", color: "#e06a5a", format: (v) => `${v.toFixed(1)} GB` },
];

const TRENDS: Chart[] = [
  { id: "refine", title: "Densify", color: "#7fd8c0", format: compact, mini: true },
  { id: "gain", title: "Gain", color: "#d8c39a", format: (v) => `${v.toFixed(3)} dB`, mini: true },
  { id: "ceiling", title: "Ceiling", color: "#b9a3f0", format: compact, mini: true },
];
const chartMarkup = (c: Chart): string =>
  `<div class="lc-panel lc-chart${c.mini ? " lc-mini" : ""}"><div class="lc-chart-head"><h2>${c.title}</h2><span id="lc-now-${c.id}"></span></div><div id="lc-legend-${c.id}" class="lc-legend"></div><svg data-training-chart="${c.id}" id="lc-chart-${c.id}" viewBox="0 0 300 150" preserveAspectRatio="none" role="img" aria-label="${c.title} over steps"></svg></div>`;

export function liveConsole(actions: {
  scrub: (step: number | null) => void;
  color: (distance: boolean) => void;
  view: (top: boolean) => void;
  layers: (mode: "splat" | "seeds" | "both", cameras: boolean) => void;
}): {
  update: (state: LiveState) => boolean;
  snapshot: (step: number) => void;
  stop: () => void;
} {
  const root = el("live-console"),
    section = el("live");
  const abort = new AbortController(),
    options = { signal: abort.signal };
  let state: LiveState | null = null,
    active = false,
    scrub: number | null = null,
    hover: number | null = null;
  let shown = -1;
  let layerMode: "splat" | "seeds" | "both" = "splat",
    camerasVisible = true;
  root.hidden = true;
  el("lc-kpis").innerHTML = ["Step", "Gaussians", "Held-out PSNR", "Phase", "Time", "GPU cost"]
    .map(
      (label, i) =>
        `<div class="lc-kpi"><span class="lc-label">${label}</span><span id="lc-kpi-${String(i)}" class="lc-value">–</span></div>`,
    )
    .join("");
  el("lc-charts").innerHTML = CHARTS.filter((c) => !c.optional)
    .map(chartMarkup)
    .join("");
  el("lc-stop-trends").innerHTML = TRENDS.map(chartMarkup).join("");
  const timeline = el("lc-timeline") as HTMLCanvasElement;
  const maxStep = (): number =>
    Math.max(
      1,
      state?.train?.plan?.stepsMax ??
        Math.max(lastStep(state?.train), state?.stage?.progress?.total ?? 1),
    );
  const x = (step: number): number => 36 + Math.max(0, Math.min(1, step / maxStep())) * 258;
  function drawCharts(): void {
    const train = state?.train,
      plan = train?.plan;
    for (const chart of [...CHARTS, ...TRENDS]) {
      if (chart.mini && !plan?.rule) continue;
      const points = chartPoints(train, chart.id);
      let svg = document.getElementById(`lc-chart-${chart.id}`);
      if (chart.optional && !points.length) {
        svg?.parentElement?.remove();
        continue;
      }
      if (!svg) {
        el("lc-charts").insertAdjacentHTML("beforeend", chartMarkup(chart));
        svg = el(`lc-chart-${chart.id}`);
      }
      const lines =
        chart.id === "loss" ? lossLines(train) : [{ name: chart.id, color: chart.color, points }];
      const reference =
        chart.id === "gs"
          ? effectiveCap(plan)
          : chart.id === "mem"
            ? plan?.cardGb
            : chart.id === "refine"
              ? plan?.refineStop
              : chart.id === "gain"
                ? plan?.rule?.minGainDb
                : chart.id === "ceiling"
                  ? plan?.stepsMax
                  : null;
      const logarithmic = chart.id === "lr";
      const [lo, hi] = chartDomain(
        lines.flatMap((line) => line.points),
        logarithmic,
        reference,
        chart.id === "gs" || chart.id === "mem" || chart.mini === true,
      );
      const y = (v: number): number =>
        8 + (1 - ((logarithmic ? Math.log10(v) : v) - lo) / (hi - lo)) * 124;
      let markup = "";
      if (plan?.refineStop != null)
        markup += `<rect x="${String(x(plan.refineStop))}" y="8" width="${String(294 - x(plan.refineStop))}" height="124" fill="#f0b254" opacity=".06"/><line x1="${String(x(plan.refineStop))}" x2="${String(x(plan.refineStop))}" y1="8" y2="132" stroke="#f0b254" opacity=".4" stroke-dasharray="2 3"/>`;
      for (const fraction of [0, 0.5, 1]) {
        const value = lo + (hi - lo) * fraction,
          yy = 132 - fraction * 124;
        const label = logarithmic
          ? `1e${value.toFixed(1)}`
          : chart.format(value).replace(/ (dB|GB)$/, "");
        markup += `<line x1="36" x2="294" y1="${String(yy)}" y2="${String(yy)}" stroke="currentColor" opacity=".08"/><text x="32" y="${String(yy + 3)}" text-anchor="end">${label}</text>`;
        markup += `<text x="${String(x(maxStep() * fraction))}" y="146" text-anchor="${fraction === 0 ? "start" : fraction === 1 ? "end" : "middle"}">${compact(maxStep() * fraction)}</text>`;
      }
      for (const line of lines) {
        if (!line.points.length) continue;
        const path = line.points
          .map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(2)} ${y(p.value).toFixed(2)}`)
          .join("");
        markup += `<path data-series="${line.name}" d="${path}" fill="none" stroke="${line.color}" stroke-width="1.6"/>`;
        for (const p of chart.id === "psnr" || chart.id === "gain"
          ? line.points
          : line.points.slice(-1))
          markup += `<circle cx="${String(x(p.step))}" cy="${String(y(p.value))}" r="2" fill="${line.color}"/>`;
      }
      if (chart.id === "loss")
        el("lc-legend-loss").innerHTML = lines
          .map((line) => `<span style="color:${line.color}">${line.name}</span>`)
          .join("");
      // References go above the series, with an ink underlay so equality cannot hide the dashes.
      if (reference != null) {
        const label =
          chart.id === "gs"
            ? "cap"
            : chart.id === "refine"
              ? "refine_stop"
              : chart.id === "gain"
                ? `${String(reference)} dB stop`
                : chart.id === "ceiling"
                  ? "ceiling"
                  : chart.format(reference);
        markup += `<g data-reference="${chart.id === "gs" ? "cap" : chart.id}"><line x1="36" x2="294" y1="${String(y(reference))}" y2="${String(y(reference))}" stroke="#12170f" stroke-width="3"/><line x1="36" x2="294" y1="${String(y(reference))}" y2="${String(y(reference))}" stroke="#f2f6ef" stroke-dasharray="4 3"/><text class="lc-reference-label" x="292" y="${String(Math.max(16, Math.min(128, y(reference) - 4)))}" text-anchor="end">${label}</text></g>`;
      }
      const mark = hover ?? scrub;
      if (mark != null)
        markup += `<line class="lc-crosshair" x1="${String(x(mark))}" x2="${String(x(mark))}" y1="8" y2="132" stroke="#f2f6ef" opacity=".6"/>`;
      const stop = train?.ended ?? train?.stop;
      if (stop)
        markup += `<line data-stop="${String(stop.step)}" x1="${String(x(stop.step))}" x2="${String(x(stop.step))}" y1="8" y2="132" stroke="#f0b254"/>`;
      svg.innerHTML = markup;
      svg.dataset.crosshair = mark == null ? "" : String(Math.round(mark));
      const point =
        points[
          nearestIndex(
            points.map((p) => p.step),
            mark ?? lastStep(train),
          )
        ];
      if (chart.mini) {
        const step = lastStep(train),
          gain = train?.evals?.findLast((e) => e.step <= step)?.gain;
        const ok =
          chart.id === "gain"
            ? gain != null &&
              reference != null &&
              gain < reference &&
              stopRule(train).checks[1]?.ok === true
            : reference != null && step >= reference;
        const value =
          chart.id === "gain"
            ? gain == null
              ? "–"
              : chart.format(gain)
            : `${compact(step)} / ${compact(reference)}`;
        text(`lc-now-${chart.id}`, `${ok ? "✓ " : train?.ended ? "" : "… "}${value}`);
        el(`lc-now-${chart.id}`).dataset.ok = String(ok);
      } else text(`lc-now-${chart.id}`, point ? chart.format(point.value) : "–");
    }
  }
  function drawTimeline(): void {
    if (!active) return;
    const rect = timeline.getBoundingClientRect(),
      dpr = Math.min(2, window.devicePixelRatio || 1);
    timeline.width = Math.max(1, Math.round(rect.width * dpr));
    timeline.height = 54 * dpr;
    const ctx = timeline.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (s: number): number => 6 + Math.min(1, s / maxStep()) * (rect.width - 12);
    const plan = state?.train?.plan,
      start = plan?.refineStart ?? 0,
      end = plan?.refineStop ?? maxStep();
    ctx.font = "10px 'Azeret Mono', monospace";
    for (const band of [
      { a: 0, b: start, color: "#677365", label: "warm-up" },
      { a: start, b: end, color: "#7fd8c0", label: "densify" },
      { a: end, b: maxStep(), color: "#f0b254", label: "settle" },
    ]) {
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = band.color;
      ctx.fillRect(X(band.a), 20, Math.max(0, X(band.b) - X(band.a) - 1), 12);
      ctx.globalAlpha = 1;
      if (X(band.b) - X(band.a) > 64) ctx.fillText(band.label, X(band.a) + 3, 13);
    }
    ctx.fillStyle = "#b9a3f0";
    const latest = lastStep(state?.train);
    for (const e of state?.train?.evals ?? [])
      if (!state?.train?.ended || e.step <= latest) ctx.fillRect(X(e.step), 35, 1.5, 6);
    ctx.fillStyle = "#f2f6ef";
    for (const s of state?.train?.snapshots ?? []) {
      if (state?.train?.ended && s.step > latest) continue;
      ctx.beginPath();
      ctx.arc(X(s.step), 26, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#f2f6ef";
    ctx.fillRect(X(scrub ?? lastStep(state?.train)) - 1, 15, 2, 24);
    const stop = state?.train?.ended ?? state?.train?.stop;
    if (stop) {
      ctx.fillStyle = "#f0b254";
      ctx.fillRect(X(stop.step) - 1, 16, 2, 22);
    }
    timeline.setAttribute("aria-valuemax", String(lastStep(state?.train)));
    timeline.setAttribute("aria-valuenow", String(Math.round(scrub ?? lastStep(state?.train))));
    text("lc-follow", scrub == null ? "live" : `step ${count(scrub)}`);
  }
  function select(step: number, release = false): void {
    const latest = lastStep(state?.train);
    scrub = scrubPosition(
      step,
      latest,
      maxStep(),
      timeline.getBoundingClientRect().width,
      release,
      state?.train?.ended != null,
    );
    actions.scrub(scrub);
    drawTimeline();
    drawCharts();
  }
  const pointStep = (event: PointerEvent): number => {
    const rect = timeline.getBoundingClientRect();
    return Math.round(
      Math.max(
        0,
        Math.min(
          lastStep(state?.train),
          ((event.clientX - rect.left - 6) / (rect.width - 12)) * maxStep(),
        ),
      ),
    );
  };
  timeline.addEventListener(
    "pointerdown",
    (event) => {
      timeline.setPointerCapture(event.pointerId);
      select(pointStep(event));
    },
    options,
  );
  timeline.addEventListener(
    "pointermove",
    (event) => {
      if (timeline.hasPointerCapture(event.pointerId)) select(pointStep(event));
    },
    options,
  );
  timeline.addEventListener(
    "pointerup",
    (event) => {
      select(pointStep(event), true);
      if (timeline.hasPointerCapture(event.pointerId))
        timeline.releasePointerCapture(event.pointerId);
    },
    options,
  );
  timeline.addEventListener(
    "keydown",
    (event) => {
      const steps = [
        ...new Set([
          0,
          ...(state?.train?.snapshots ?? []).map((s) => s.step),
          lastStep(state?.train),
        ]),
      ]
        .filter((s) => s <= lastStep(state?.train))
        .sort((a, b) => a - b);
      const at = scrub ?? lastStep(state?.train);
      const next =
        event.key === "ArrowLeft"
          ? (steps.filter((s) => s < at).at(-1) ?? 0)
          : event.key === "ArrowRight"
            ? (steps.find((s) => s > at) ?? lastStep(state?.train))
            : event.key === "Home"
              ? 0
              : event.key === "End"
                ? lastStep(state?.train)
                : null;
      if (next != null) {
        event.preventDefault();
        select(next);
      }
    },
    options,
  );
  root.addEventListener(
    "pointermove",
    (event) => {
      const svg = (event.target as Element).closest("svg[data-training-chart]");
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      hover = Math.max(
        0,
        Math.min(
          state?.train?.ended ? lastStep(state.train) : maxStep(),
          ((((event.clientX - rect.left) / rect.width) * 300 - 36) / 258) * maxStep(),
        ),
      );
      drawCharts();
    },
    options,
  );
  root.addEventListener(
    "pointerleave",
    () => {
      hover = null;
      drawCharts();
    },
    options,
  );
  for (const group of ["color", "view"] as const) {
    const buttons = root.querySelectorAll<HTMLButtonElement>(`[data-live-${group}]`);
    buttons.forEach((button) =>
      button.addEventListener(
        "click",
        () => {
          buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
          if (group === "color") {
            const distance = button.dataset.liveColor === "distance";
            el("lc-distance").hidden = !distance;
            actions.color(distance);
          } else actions.view(button.dataset.liveView === "top");
        },
        options,
      ),
    );
  }
  function snapshotLabel(): void {
    text("lc-snapshot", shown < 0 ? "splat" : `splat · step ${count(shown)}`);
    const lag = snapshotLag(shown, lastStep(state?.train));
    text("lc-lag", lag ?? "");
    el("lc-lag").hidden = lag == null;
  }
  root.querySelectorAll<HTMLButtonElement>("[data-live-layer]").forEach((button) =>
    button.addEventListener(
      "click",
      () => {
        const mode = button.dataset.liveLayer;
        if (mode !== "splat" && mode !== "seeds" && mode !== "both") return;
        layerMode = mode;
        root
          .querySelectorAll("[data-live-layer]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
        el("lc-seeds-count").hidden = mode === "splat" || !state?.train?.seeds;
        actions.layers(mode, camerasVisible);
      },
      options,
    ),
  );
  el("lc-cameras").addEventListener(
    "click",
    () => {
      camerasVisible = !camerasVisible;
      el("lc-cameras").setAttribute("aria-pressed", String(camerasVisible));
      actions.layers(layerMode, camerasVisible);
    },
    options,
  );
  function quality(): void {
    const snapshot = snapshotAt(state?.train?.snapshots ?? [], shown);
    const labels = qualityReadouts(snapshot, state?.train?.quality);
    text("lc-quality", labels.join(" · "));
    el("lc-quality").hidden = !labels.length;
  }
  const observer = new ResizeObserver(drawTimeline);
  observer.observe(timeline);
  return {
    update(next): boolean {
      const retry = isNewAttempt(state?.train, next.train);
      state = next;
      active ||= next.stage?.stageId === "train" || next.train != null;
      if (!active) return false;
      root.hidden = false;
      section.classList.add("training-console");
      if (retry) {
        layerMode = "splat";
        root
          .querySelectorAll<HTMLElement>("[data-live-layer]")
          .forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.liveLayer === "splat")));
        actions.layers(layerMode, camerasVisible);
        scrub = null;
        shown = -1;
        actions.scrub(null);
      }
      if (next.train?.ended) {
        scrub = Math.min(scrub ?? lastStep(next.train), lastStep(next.train));
        if (hover != null) hover = Math.min(hover, lastStep(next.train));
        actions.scrub(scrub);
      }
      text("lc-name", next.captureName);
      text("lc-pill", pillText(next));
      el("lc-pill").dataset.done = String(
        next.train?.ended != null ||
          pillText(next).startsWith("Finished") ||
          pillText(next) === "Stopped",
      );
      text("lc-error", next.status === "error" ? (next.error ?? "") : "");
      el("lc-error").hidden = next.status !== "error" || !next.error;
      text("lc-scene-title", next.train?.ended ? "Splat" : "Live splat");
      const link = el("lc-final") as HTMLAnchorElement;
      link.hidden = !next.siteId;
      if (next.siteId) link.href = `#${next.siteId}`;
      const train = next.train,
        series = train?.series,
        plan = train?.plan;
      const step = lastStep(train),
        index = series?.step?.findLastIndex((s) => s <= step) ?? -1,
        t = index < 0 ? null : series?.t?.[index],
        gaussians = index < 0 ? null : series?.gs?.[index],
        psnr = train?.evals?.findLast((e) => e.step <= step)?.psnr;
      const values = [
        count(train?.ended || train?.run ? step : step || next.stage?.progress?.done),
        compact(gaussians),
        psnr == null ? "–" : `${psnr.toFixed(2)} dB`,
        phaseOf(step, plan, gaussians, train?.ended),
        timeText(t),
        money(cost(t, plan?.usdPerHour)),
      ];
      values.forEach((value, i) => text(`lc-kpi-${String(i)}`, value));
      const rule = stopRule(train);
      el("lc-stop").hidden = !plan?.rule;
      text("lc-rule", rule.line);
      const entries = [
        ["Splat budget", compact(plan?.cap)],
        ["Step ceiling", `${count(plan?.stepsMax)} · ${ceilingText(plan)}`],
        ["Densify until", count(plan?.refineStop)],
        ["Chunking", blockCountText(plan?.blocks)],
      ];
      const dense = train?.quality?.denseInitPoints ?? null,
        added = train?.quality?.denseInitAddedPoints ?? null;
      if (plan?.sparse != null || dense != null || added != null)
        entries.push([
          "Init",
          [
            plan?.sparse != null ? `${compact(plan.sparse)} SfM` : null,
            dense != null ? `${compact(dense)} dense` : null,
            added != null ? `+${compact(added)}` : null,
          ]
            .filter(Boolean)
            .join(" · "),
        ]);
      el("lc-plan").replaceChildren(
        ...entries.flatMap(([label, value]) => {
          const dt = document.createElement("dt"),
            dd = document.createElement("dd");
          dt.textContent = label ?? "";
          dd.textContent = value ?? "";
          if (label === "Chunking")
            dd.insertAdjacentHTML(
              "afterbegin",
              '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M1 1h18v18H1zM7 1v18M13 1v18M1 7h18M1 13h18" fill="none" stroke="#7fd8c0"/></svg>',
            );
          return [dt, dd];
        }),
      );
      el("lc-events").replaceChildren(
        ...eventsDisplay(train).map((event) => {
          const row = document.createElement("tr");
          for (const value of [event.step, event.kind, event.text]) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.append(cell);
          }
          return row;
        }),
      );
      el("lc-layer-controls").hidden = !train?.seeds;
      text("lc-seeds-count", seedsLabel(train?.seeds));
      el("lc-seeds-count").hidden = !train?.seeds || layerMode === "splat";
      const final = finalReadouts(train);
      el("lc-final-row").hidden = !final.length;
      text("lc-final-scores", final.join(" · "));
      snapshotLabel();
      const distance = root.querySelector<HTMLButtonElement>('[data-live-color="distance"]');
      if (!distance) return true;
      distance.disabled = !plan?.metresPerUnit || !next.cameras?.final || !next.cameras.cameraCount;
      if (distance.disabled && distance.getAttribute("aria-pressed") === "true")
        root.querySelector<HTMLButtonElement>('[data-live-color="photo"]')?.click();
      drawCharts();
      drawTimeline();
      quality();
      return true;
    },
    snapshot(step): void {
      shown = step;
      snapshotLabel();
      quality();
    },
    stop(): void {
      abort.abort();
      observer.disconnect();
      section.classList.remove("training-console");
      root.hidden = true;
    },
  };
}
