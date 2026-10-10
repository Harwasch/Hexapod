import type { components } from "@twin/contracts";
export type ViewCapture = components["schemas"]["LandViewCreate"];
export const viewDraftKey = (scope: string, landId: string) =>
  `living-world-land-draft:${encodeURIComponent(scope)}:view:${landId}`;
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("The captured view has invalid fields.");
  return value as Record<string, unknown>;
}
function identifier(value: unknown) {
  if (typeof value !== "string" || !UUID.test(value))
    throw new Error("The captured view has an invalid record reference.");
}
function number(value: unknown, min: number, max: number, integer = false) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max ||
    (integer && !Number.isInteger(value))
  )
    throw new Error("The captured view has an invalid camera or layer value.");
}
export function serializeViewCapture(landId: string, payload: ViewCapture) {
  return JSON.stringify({ version: 1, landId, payload });
}
export function parseViewCapture(text: string, landId: string): ViewCapture {
  if (text.length > 64_000) throw new Error("This captured view exceeds the recovery limit.");
  const snapshot = object(JSON.parse(text));
  if (snapshot.version !== 1 || snapshot.landId !== landId)
    throw new Error("This capture belongs to other land or uses an unsupported format.");
  const payload = object(snapshot.payload),
    state = object(payload.state),
    camera = object(state.camera);
  identifier(payload.requestKey);
  if (typeof payload.name !== "string" || !payload.name.trim() || payload.name.length > 160)
    throw new Error("The captured view needs a valid name.");
  number(state.boundaryRevision, 1, Number.MAX_SAFE_INTEGER, true);
  for (const [key, min, max] of [
    ["longitude", -180, 180],
    ["latitude", -90, 90],
    ["height", -12000, 100_000_000],
    ["heading", -360, 360],
    ["pitch", -90, 90],
    ["roll", -360, 360],
  ] as const)
    number(camera[key], min, max);
  if (
    state.section !== undefined &&
    !["discover", "records", "inventory", "ecology", "scenarios", "actions"].includes(
      typeof state.section === "string" ? state.section : "",
    )
  )
    throw new Error("The captured workspace tab is unsupported.");
  for (const key of ["investigationId", "inventoryId", "surveyId", "solarId"])
    if (state[key] != null) identifier(state[key]);
  if (state.inventoryVisible !== undefined && typeof state.inventoryVisible !== "boolean")
    throw new Error("The captured inventory setting is invalid.");
  const artifacts: unknown = state.artifactIds ?? [],
    rasters: unknown = state.rasters ?? [];
  if (
    !Array.isArray(artifacts) ||
    artifacts.length > 20 ||
    !Array.isArray(rasters) ||
    rasters.length > 2
  )
    throw new Error("The captured view has too many layers.");
  artifacts.forEach(identifier);
  const ids = new Set<unknown>();
  for (const item of rasters) {
    const raster = object(item);
    identifier(raster.id);
    if (ids.has(raster.id)) throw new Error("The captured view repeats a raster layer.");
    ids.add(raster.id);
    if (
      raster.kind !== undefined &&
      !["raster", "archive-alignment"].includes(typeof raster.kind === "string" ? raster.kind : "")
    )
      throw new Error("The captured imagery type is unsupported.");
    number(raster.band ?? 1, 1, 16, true);
    number(raster.opacity ?? 0.8, 0, 1);
  }
  return payload as ViewCapture;
}
export function downloadViewCapture(text: string, landId: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `land-${landId}-view-capture.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
