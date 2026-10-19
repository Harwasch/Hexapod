import type {
  components,
  LandEvidence,
  LandFinding,
  LandMapGeometry,
  ResearchEvent,
  ResearchRun,
} from "@twin/contracts";

import type { LandPoint } from "@/state/land";

import { anchorOf } from "../geometryAnchor";

export type EventLocation = components["schemas"]["EventLocation"];

/** One line in the agent's progress: what it is doing, in plain words, and where. */
export interface AgentStep {
  key: string;
  text: string;
  /** The source it is reading, as people name it ("USGS elevation"). */
  source?: string;
  /** How that source answered ("No records here"). */
  note?: string;
  state: "done" | "active" | "finding";
  location: EventLocation | null;
  findingId?: string;
  artifactId?: string;
}

/** Sources by the names people use for them. Unknown ids fall back to the id itself. */
const SOURCE_NAMES: Record<string, string> = {
  "usgs-elevation": "USGS elevation",
  "nasa-power": "NASA climate data",
  "gbif-occurrences": "GBIF species records",
  "usda-soils": "USDA soil survey",
  "fema-flood-zones": "FEMA flood maps",
  "epa-ecoregions": "EPA ecoregions",
  "usda-ecological-sites": "USDA ecological sites",
  "col-taxonomy": "Catalogue of Life",
  "commons-place-images": "Wikimedia Commons photos",
  "usgs-historical-maps": "USGS historical maps",
  "mapped-assets": "OpenStreetMap features",
  "land-documents": "Your records",
  "cop-dem-glo-30": "Copernicus elevation",
  "esa-worldcover-2021": "ESA land cover",
  "sentinel-2-ndvi": "Sentinel-2 imagery",
  "nasa-power-hourly": "NASA hourly weather",
  "public-web-search": "Public web search",
  "saved-archive-image": "Saved archive image",
  "saved-solar-assessment": "Saved solar estimate",
  "field-survey": "Your field survey",
  "private-inventory": "Your assets",
};

export function sourceName(provider: string): string {
  return SOURCE_NAMES[provider] ?? provider;
}

const STATUS_NOTES: Record<string, string> = {
  available: "Found data",
  empty: "Nothing recorded here",
  uncovered: "Doesn't cover this land",
  unavailable: "Couldn't be reached",
};

const OUTPUT_NAMES: Record<string, string> = {
  map: "a map",
  chart: "a chart",
  table: "a table",
  raster: "a map layer",
  gallery: "a photo set",
  solar: "a solar estimate",
  document: "a write-up",
  timeline: "a timeline",
  calculation: "a calculation",
};

/** Tools whose following `looking`/`source` events belong to the step that chose them. */
const LOOKING_ACTIONS = new Set([
  "retrieve_source",
  "search_mapped_assets",
  "match_taxon",
  "search_public_sources",
  "inspect_archive_image",
  "analyze_raster",
  "analyze_solar",
]);

const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

interface Draft extends AgentStep {
  open: boolean;
  transient?: boolean;
  action?: string;
  provider?: string;
}

/**
 * The agent's events as a short list of plain steps. Bookkeeping events (empty progress,
 * tool results, the run starting) never become lines of their own: a tool's lookup and its
 * answer join the step that chose it, and "thinking" shows only while it is the latest thing.
 */
export function stepsFromEvents(events: ResearchEvent[], running: boolean): AgentStep[] {
  const steps: Draft[] = [];
  const last = () => steps.at(-1);
  const dropTransient = () => {
    while (last()?.transient) steps.pop();
  };
  const openStep = () => {
    const step = last();
    return step?.open && !step.transient ? step : null;
  };
  const closeAll = () => steps.forEach((step) => (step.open = false));
  const push = (step: Omit<Draft, "key" | "state"> & { state?: AgentStep["state"] }, key: number) =>
    steps.push({ state: "done", ...step, key: `${key}` });

  for (const event of events) {
    const payload = event.payload;
    const location = event.location ?? null;
    switch (event.kind) {
      case "queued":
        dropTransient();
        push({ text: "Waiting to start", open: true, transient: true, location }, event.sequence);
        break;
      case "started":
        dropTransient();
        push(
          {
            text: payload.resumed ? "Picking up where it left off" : "Starting",
            open: true,
            transient: true,
            location: null,
          },
          event.sequence,
        );
        break;
      case "model":
        dropTransient();
        closeAll();
        push(
          { text: "Thinking about the next step", open: true, transient: true, location: null },
          event.sequence,
        );
        break;
      case "progress": {
        const message = text(payload.message);
        if (!message) break; // A saved checkpoint, not a step.
        dropTransient();
        closeAll();
        push(
          { text: message, open: true, location, action: text(payload.action) || undefined },
          event.sequence,
        );
        break;
      }
      case "looking": {
        dropTransient();
        const provider = text(payload.provider);
        const step = openStep();
        if (step?.action && LOOKING_ACTIONS.has(step.action) && !step.provider) {
          step.provider = provider;
          step.source = sourceName(provider);
          step.location = location ?? step.location;
          break;
        }
        closeAll();
        push(
          {
            text: text(payload.message) || `Checking ${sourceName(provider)}`,
            source: sourceName(provider),
            provider,
            open: true,
            location,
          },
          event.sequence,
        );
        break;
      }
      case "source": {
        dropTransient();
        const provider = text(payload.provider);
        const note = STATUS_NOTES[text(payload.status)];
        const step = [...steps].reverse().find((item) => item.open && item.provider === provider);
        if (step) {
          step.note = note;
          step.location = step.location ?? location;
          if (!step.action) step.open = false;
          break;
        }
        const open = openStep();
        if (open?.action && LOOKING_ACTIONS.has(open.action) && !open.provider) {
          open.provider = provider;
          open.source = sourceName(provider);
          open.note = note;
          open.location = open.location ?? location;
          break;
        }
        push(
          {
            text: `Checked ${sourceName(provider)}`,
            source: sourceName(provider),
            provider,
            note,
            open: false,
            location,
          },
          event.sequence,
        );
        break;
      }
      case "search": {
        dropTransient();
        const query = text(payload.query);
        const step = openStep();
        if (step?.action === "search_public_sources") {
          step.source = sourceName("public-web-search");
          step.location = step.location ?? location;
          break;
        }
        closeAll();
        push(
          {
            text: query ? `Searching public sources for “${query}”` : "Searching public sources",
            source: sourceName("public-web-search"),
            open: true,
            location,
          },
          event.sequence,
        );
        break;
      }
      case "analysis":
      case "image": {
        dropTransient();
        const step = openStep();
        if (step?.action && LOOKING_ACTIONS.has(step.action)) {
          step.location = location ?? step.location;
          break;
        }
        closeAll();
        push(
          {
            text:
              text(payload.message) ||
              (event.kind === "image" ? "Looking at an archive image" : "Running an analysis"),
            open: true,
            location,
          },
          event.sequence,
        );
        break;
      }
      case "tool-result": {
        const step = openStep();
        if (step) step.open = false;
        break;
      }
      case "finding": {
        dropTransient();
        const step = openStep();
        // The agent said what it found as it chose to publish it; keep its words.
        if (step?.action === "publish_finding" && !step.findingId) {
          step.state = "finding";
          step.findingId = text(payload.id) || undefined;
          step.location = location ?? step.location;
          step.open = false;
          break;
        }
        push(
          {
            text: `Found: ${text(payload.title) || "a finding"}`,
            state: "finding",
            open: false,
            location,
            findingId: text(payload.id) || undefined,
          },
          event.sequence,
        );
        break;
      }
      case "artifact": {
        dropTransient();
        const kind = OUTPUT_NAMES[text(payload.kind)] ?? "a result";
        const title = text(payload.title);
        const step = openStep();
        if (step?.action === "publish_artifact" && !step.artifactId) {
          step.artifactId = text(payload.id) || undefined;
          step.location = location ?? step.location;
          step.open = false;
          break;
        }
        push(
          {
            text: title ? `Made ${kind}: ${title}` : `Made ${kind}`,
            open: false,
            location,
            artifactId: text(payload.id) || undefined,
          },
          event.sequence,
        );
        break;
      }
      case "succeeded":
      case "partial":
      case "failed":
      case "cancelled":
        dropTransient();
        closeAll();
        break;
      default: {
        const message = text(payload.message);
        if (!message) break;
        dropTransient();
        closeAll();
        push({ text: message, open: true, location }, event.sequence);
      }
    }
  }
  if (!running) {
    dropTransient();
    closeAll();
  }
  return steps.map((step, index) => ({
    key: step.key,
    text: step.text,
    source: step.source,
    note: step.note,
    location: step.location,
    findingId: step.findingId,
    artifactId: step.artifactId,
    state:
      step.state === "finding"
        ? "finding"
        : running && index === steps.length - 1
          ? "active"
          : "done",
  }));
}

/** How far through its budget a run is: the larger of steps taken and time spent. */
export function runProgress(
  run: ResearchRun | null | undefined,
  events: ResearchEvent[],
  now = Date.now(),
): number {
  if (!run) return 0;
  if (run.status !== "queued" && run.status !== "running") return 1;
  const steps = events.filter((event) => event.kind === "model").length;
  const stepShare = steps / Math.max(1, run.budget.maxSteps ?? 16);
  const started = run.startedAt ? Date.parse(run.startedAt) : NaN;
  const timeShare = Number.isFinite(started)
    ? (now - started) / 1000 / Math.max(1, run.budget.maxSeconds ?? 180)
    : 0;
  return Math.min(0.95, Math.max(0.04, stepShare, timeShare));
}

function mediaGeometry(evidence: LandEvidence | undefined): LandMapGeometry | null {
  return evidence?.media?.location ?? null;
}

/** The point a leader line ends at for a location: the point itself or the box's middle. */
export function locationAnchor(
  location: EventLocation | null | undefined,
  evidence: LandEvidence[] = [],
): LandPoint | null {
  if (!location) return null;
  if (location.point) return [location.point[0], location.point[1]];
  if (location.bbox) {
    const [west, south, east, north] = location.bbox;
    return [(west + east) / 2, (south + north) / 2];
  }
  const id = location.ref?.evidenceId;
  const geometry = id ? mediaGeometry(evidence.find((item) => item.id === id)) : null;
  return geometry ? anchorOf(geometry) : null;
}

/** A circle as a closed ring, for a search radius around a point. */
function circle([lon, lat]: LandPoint, radiusM: number, sides = 48): number[][] {
  const dLat = radiusM / 111_320;
  const dLon = radiusM / (111_320 * Math.max(0.01, Math.cos((lat * Math.PI) / 180)));
  const ring: number[][] = [];
  for (let i = 0; i <= sides; i++) {
    const angle = (i / sides) * Math.PI * 2;
    ring.push([lon + dLon * Math.cos(angle), lat + dLat * Math.sin(angle)]);
  }
  return ring;
}

/** The area a location covers, to outline on the map: a box, a search circle, or nothing. */
export function locationArea(
  location: EventLocation | null | undefined,
  evidence: LandEvidence[] = [],
): LandMapGeometry | null {
  if (!location) return null;
  if (location.bbox) {
    const [west, south, east, north] = location.bbox;
    if (west === east || south === north) return null;
    return {
      type: "Polygon",
      coordinates: [
        [
          [west, south],
          [east, south],
          [east, north],
          [west, north],
          [west, south],
        ],
      ],
    };
  }
  if (location.point && location.radiusM)
    return { type: "Polygon", coordinates: [circle(location.point, location.radiusM)] };
  const id = location.ref?.evidenceId;
  const geometry = id ? mediaGeometry(evidence.find((item) => item.id === id)) : null;
  return geometry && geometry.type !== "Point" ? geometry : null;
}

/** The shape to fly to and outline for a location, when it has one. */
export function locationGeometry(
  location: EventLocation | null | undefined,
  evidence: LandEvidence[] = [],
): LandMapGeometry | null {
  const area = locationArea(location, evidence);
  if (area) return area;
  const anchor = locationAnchor(location, evidence);
  return anchor ? { type: "Point", coordinates: anchor } : null;
}

/** Where each finding of a run came from, by finding id, from its `finding` events. */
export function findingLocations(events: ResearchEvent[]): Map<string, EventLocation> {
  const places = new Map<string, EventLocation>();
  for (const event of events) {
    const id = text(event.payload.id);
    if (event.kind === "finding" && id && event.location) places.set(id, event.location);
  }
  return places;
}

/** A finding's place: its own boundary when the agent drew one, else where it came from. */
export function findingPlace(
  finding: LandFinding,
  places: Map<string, EventLocation>,
  evidence: LandEvidence[] = [],
): { geometry: LandMapGeometry; anchor: LandPoint } | null {
  if (finding.boundary)
    return { geometry: finding.boundary as LandMapGeometry, anchor: anchorOf(finding.boundary) };
  const location = places.get(finding.id);
  const geometry = locationGeometry(location, evidence);
  const anchor = locationAnchor(location, evidence);
  return geometry && anchor ? { geometry, anchor } : null;
}

const UUID = /([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi;

/** The agent's answer without its citation tail, plus the evidence it cited. */
export function splitAnswer(content: string): { text: string; evidenceIds: string[] } {
  const [body = "", tail = ""] = content.split(/\n\nEvidence: /);
  const cited = [...tail.matchAll(UUID), ...body.matchAll(UUID)].map((match) => match[1] ?? "");
  return { text: body.trim(), evidenceIds: [...new Set(cited.filter(Boolean))] };
}

export type CitationPart = { kind: "text"; text: string } | { kind: "source"; evidenceId: string };

/** Text with evidence ids in it, as alternating plain parts and cited ids. */
export function citationParts(content: string): CitationPart[] {
  return content
    .split(UUID)
    .map((part, index): CitationPart =>
      index % 2 === 1 ? { kind: "source", evidenceId: part } : { kind: "text", text: part },
    )
    .filter((part) => part.kind === "source" || part.text.length > 0);
}
