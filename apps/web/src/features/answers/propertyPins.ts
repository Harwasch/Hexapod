import { env } from "@/app/env";
import { fetchApi } from "@/api/coldStart";

export interface AnswerPin {
  id: string;
  siteId: string;
  scanId: string;
  label: string;
  observedAt: string;
  uncertaintyM: number | null;
  geo: { lat: number; lon: number; alt: number };
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Unlocated observations stay in memory but cannot be drawn as precise map pins. */
export function parseAnswerPin(value: unknown, siteId: string): AnswerPin | null {
  const row = record(value),
    entity = record(row?.entity),
    anchor = record(entity?.anchor);
  const geo = record(anchor?.geo);
  if (
    row?.siteId !== siteId ||
    typeof row.id !== "string" ||
    typeof row.scanId !== "string" ||
    typeof row.observedAt !== "string" ||
    !entity ||
    typeof entity.label !== "string" ||
    !geo
  )
    return null;
  const { lat, lon, alt } = geo;
  if (
    typeof lat !== "number" ||
    !Number.isFinite(lat) ||
    Math.abs(lat) > 90 ||
    typeof lon !== "number" ||
    !Number.isFinite(lon) ||
    Math.abs(lon) > 180 ||
    typeof alt !== "number" ||
    !Number.isFinite(alt)
  )
    return null;
  const uncertainty = row.uncertaintyM;
  return {
    id: row.id,
    siteId,
    scanId: row.scanId,
    label: entity.label,
    observedAt: row.observedAt,
    uncertaintyM:
      typeof uncertainty === "number" && Number.isFinite(uncertainty) && uncertainty >= 0
        ? uncertainty
        : null,
    geo: { lat, lon, alt },
  };
}

export async function loadAnswerPins(siteId: string, signal: AbortSignal): Promise<AnswerPin[]> {
  const pins = new Map<string, AnswerPin>();
  let offset = 0;
  for (let page = 0; page < 200; page++) {
    const url = `${env.apiBaseUrl}/api/v1/properties/${encodeURIComponent(siteId)}/entities?limit=500&offset=${offset}`;
    const response = await fetchApi(url, { signal });
    if (!response.ok) throw new Error(`Property memory: HTTP ${response.status}`);
    const body = record(await response.json());
    if (!body || !Array.isArray(body.items)) throw new Error("Invalid property memory page");
    for (const item of body.items) {
      const pin = parseAnswerPin(item, siteId);
      if (pin) pins.set(pin.id, pin);
    }
    if (body.nextOffset === null) return [...pins.values()];
    if (
      typeof body.nextOffset !== "number" ||
      !Number.isSafeInteger(body.nextOffset) ||
      body.nextOffset <= offset
    )
      throw new Error("Invalid property memory cursor");
    offset = body.nextOffset;
  }
  throw new Error("Property memory page limit reached");
}
