export function formatDate(
  iso: string | null | undefined,
  style: "short" | "long" = "short",
): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(
    undefined,
    style === "long"
      ? { year: "numeric", month: "long", day: "numeric" }
      : { year: "numeric", month: "short", day: "numeric" },
  );
}

export function formatYear(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : String(date.getUTCFullYear());
}

export function representationLabel(value: string): string {
  switch (value) {
    case "gaussian-splat":
      return "Splat";
    case "mesh":
      return "Mesh";
    case "point-cloud":
      return "Points";
    case "terrain":
      return "Terrain";
    case "imagery":
      return "Imagery";
    default:
      return value;
  }
}

export function categoryLabel(value: string): string {
  return value
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function sourceLabel(source: { type: string } & Record<string, unknown>): string {
  switch (source.type) {
    case "cesium-ion-terrain":
    case "cesium-ion-imagery":
    case "cesium-ion-3d-tiles":
      return `Cesium ion asset ${String(source.assetId)}`;
    case "cesium-ion":
      return `Cesium ion asset ${String(source.assetId)}`;
    case "google-photorealistic":
      return "Google Photorealistic 3D Tiles via Cesium ion";
    case "3d-tiles-url":
      return `3D Tiles · ${hostOf(String(source.url))}`;
    case "xyz":
      return `XYZ tiles · ${hostOf(String(source.urlTemplate))}`;
    case "mvt":
      return `Vector tiles · ${hostOf(String(source.urlTemplate))}`;
    case "wmts":
      return `WMTS · ${hostOf(String(source.url))}`;
    case "wms":
      return `WMS · ${hostOf(String(source.url))}`;
    case "arcgis-mapserver":
      return `ArcGIS MapServer · ${hostOf(String(source.url))}`;
    case "geojson":
      return `GeoJSON · ${hostOf(String(source.url))}`;
    case "czml":
      return `CZML · ${hostOf(String(source.url))}`;
    case "stac":
      return `STAC ${typeof source.kind === "string" ? source.kind : "item"} · ${hostOf(String(source.url))}`;
    default:
      return source.type;
  }
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];

/** Bytes in the units a person reads them in; 1024-based, three significant figures. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return "—";
  if (bytes < 1000) return `${Math.max(0, Math.round(bytes))} B`;
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < BYTE_UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const rounded = value >= 100 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded} ${BYTE_UNITS[unit] ?? "B"}`;
}

/** A short elapsed time: "4s", "2m 10s", "1h 04m". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}
