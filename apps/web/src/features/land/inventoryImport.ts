import type { components, LandMapGeometry } from "@twin/contracts";
import { closeInventoryShape, openInventoryShape } from "./inventoryGeometry";
export type ImportCategory = components["schemas"]["LandFeatureCreate"]["category"];
export const inventoryCategories: ImportCategory[] = [
  "building",
  "power",
  "water",
  "transport",
  "equipment",
  "vegetation",
  "other",
];
export interface ImportMapping {
  name: string;
  id: string;
  category: string;
  longitude: string;
  latitude: string;
}
export const defaultImportMapping: ImportMapping = {
  name: "name",
  id: "id",
  category: "category",
  longitude: "longitude",
  latitude: "latitude",
};
export interface ImportRow {
  rowId: string;
  name: string;
  recordId: string;
  generatedId: boolean;
  category: ImportCategory;
  geometry: LandMapGeometry | null;
  attributes: Record<string, string | number | boolean | null>;
  warnings: string[];
  error: string | null;
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Expected a GeoJSON object.");
  return value as Record<string, unknown>;
}
function checkCrs(value: Record<string, unknown>) {
  if (!value.crs) return;
  const crs = object(value.crs),
    props = object(crs.properties);
  if (
    crs.type !== "name" ||
    !["EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84", "urn:ogc:def:crs:EPSG::4326"].includes(
      String(props.name),
    )
  )
    throw new Error(
      "Reproject this file to WGS 84 longitude/latitude (EPSG:4326) before importing.",
    );
}
function geometry(value: unknown): LandMapGeometry {
  const g = object(value);
  checkCrs(g);
  const depth =
    g.type === "Point"
      ? 0
      : g.type === "LineString"
        ? 1
        : g.type === "Polygon"
          ? 2
          : g.type === "MultiPolygon"
            ? 3
            : -1;
  if (depth < 0) throw new Error(`Unsupported geometry: ${String(g.type)}.`);
  let count = 0;
  const visit = (v: unknown, level: number): void => {
    if (!Array.isArray(v) || !v.length) throw new Error("Geometry coordinates are missing.");
    if (level) {
      v.forEach((p) => visit(p, level - 1));
      return;
    }
    if (
      v.length < 2 ||
      v.length > 3 ||
      !v.every((n: unknown) => typeof n === "number" && Number.isFinite(n)) ||
      Math.abs(Number(v[0])) > 180 ||
      Math.abs(Number(v[1])) > 90
    )
      throw new Error("Coordinates must be finite WGS 84 longitude/latitude.");
    if (++count > 20_000) throw new Error("A feature can have at most 20,000 vertices.");
  };
  visit(g.coordinates, depth);
  const typed = g as unknown as LandMapGeometry;
  if (typed.type === "Polygon" || typed.type === "MultiPolygon") {
    const parts = typed.type === "Polygon" ? [typed.coordinates] : typed.coordinates;
    if (
      parts.some((part) =>
        part.some(
          (ring) =>
            ring.length < 4 ||
            ring[0]?.[0] !== ring.at(-1)?.[0] ||
            ring[0]?.[1] !== ring.at(-1)?.[1],
        ),
      )
    )
      throw new Error("Every polygon ring must be closed and contain at least four positions.");
  }
  return closeInventoryShape(openInventoryShape(typed));
}
/** RFC 4180-style quoted cells, including escaped quotes and embedded newlines. */
export function csvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [],
    cell = "",
    quoted = false,
    closed = false;
  for (let i = 0; i < text.length; i++) {
    const char = text.charAt(i);
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          quoted = false;
          closed = true;
        }
      } else cell += char;
    } else if (char === '"' && !cell && !closed) quoted = true;
    else if (char === "," || char === "\r" || char === "\n") {
      row.push(cell);
      cell = "";
      closed = false;
      if (char !== ",") {
        if (row.some((c) => c.trim())) rows.push(row);
        row = [];
        if (char === "\r" && text[i + 1] === "\n") i++;
      }
      if (rows.length > 201 || row.length > 200)
        throw new Error("Import at most 200 records and 200 columns.");
    } else {
      if (closed || char === '"') throw new Error("Malformed quoted CSV cell.");
      cell += char;
    }
  }
  if (quoted) throw new Error("CSV has an unclosed quoted cell.");
  row.push(cell);
  if (row.length > 200) throw new Error("Import at most 200 columns.");
  if (row.some((c) => c.trim())) rows.push(row);
  if (rows.length > 201) throw new Error("Import at most 200 records.");
  return rows;
}
export function parseInventoryImport(
  text: string,
  format: "geojson" | "csv",
  fields: ImportMapping,
): ImportRow[] {
  if (new TextEncoder().encode(text).length > 5 * 1024 * 1024)
    throw new Error("Choose a file no larger than 5 MiB.");
  let features: unknown[];
  if (format === "csv") {
    const records = csvRows(text.replace(/^\uFEFF/, "")),
      header = records.shift()?.map((s) => s.trim()) ?? [];
    if (!header.length || new Set(header).size !== header.length || header.some((h) => !h))
      throw new Error("CSV needs unique, nonempty column headers.");
    if (!header.includes(fields.longitude) || !header.includes(fields.latitude))
      throw new Error("Choose the CSV longitude and latitude columns.");
    features = records.map((row) => {
      const props = Object.fromEntries(header.map((key, i) => [key, row[i] ?? ""]));
      const lon = props[fields.longitude]?.trim(),
        lat = props[fields.latitude]?.trim();
      return {
        type: "Feature",
        properties: props,
        geometry: {
          type: "Point",
          coordinates: [lon ? Number(lon) : NaN, lat ? Number(lat) : NaN],
        },
        importError:
          row.length !== header.length ? "CSV row does not match its column headers." : null,
      };
    });
  } else {
    const input = object(JSON.parse(text.replace(/^\uFEFF/, "")) as unknown);
    checkCrs(input);
    features =
      input.type === "FeatureCollection" && Array.isArray(input.features)
        ? input.features
        : [input];
  }
  if (!features.length || features.length > 200)
    throw new Error("Choose a file containing 1–200 features.");
  const rows: ImportRow[] = [];
  features.forEach((input, index) => {
    const base: ImportRow = {
      rowId: String(index + 1),
      name: `Imported feature ${index + 1}`,
      recordId: String(index + 1),
      generatedId: true,
      category: "other",
      geometry: null,
      attributes: {},
      warnings: [],
      error: null,
    };
    try {
      const feature = object(input);
      checkCrs(feature);
      if (typeof feature.importError === "string") throw new Error(feature.importError);
      const properties = feature.properties == null ? {} : object(feature.properties);
      const name = properties[fields.name];
      if (typeof name === "string" || typeof name === "number")
        base.name = String(name).slice(0, 200);
      const id = properties[fields.id] ?? feature.id;
      if (typeof id === "string" || typeof id === "number") {
        base.recordId = String(id);
        base.generatedId = false;
      }
      if (!base.recordId.trim() || base.recordId.length > 180)
        throw new Error("Record IDs must contain 1–180 characters.");
      const categoryValue = properties[fields.category];
      const category = typeof categoryValue === "string" ? categoryValue.toLowerCase() : "other";
      if (inventoryCategories.includes(category as ImportCategory))
        base.category = category as ImportCategory;
      else base.warnings.push(`Category “${category.slice(0, 80)}” mapped to other.`);
      for (const [key, value] of Object.entries(properties)) {
        if (
          Object.keys(base.attributes).length >= 100 ||
          key.length > 100 ||
          (typeof value === "string" && value.length > 2000) ||
          !(
            value === null ||
            typeof value === "string" ||
            typeof value === "boolean" ||
            (typeof value === "number" && Number.isFinite(value))
          )
        ) {
          if (
            !base.warnings.includes(
              "Some properties exceed limits or contain nested values and are not imported.",
            )
          )
            base.warnings.push(
              "Some properties exceed limits or contain nested values and are not imported.",
            );
        } else base.attributes[key] = value;
      }
      const raw = object(feature.type === "Feature" ? feature.geometry : feature);
      checkCrs(raw);
      if (
        (raw.type === "MultiPoint" || raw.type === "MultiLineString") &&
        Array.isArray(raw.coordinates)
      ) {
        if (!raw.coordinates.length) throw new Error("Geometry has no parts.");
        raw.coordinates.forEach((part: unknown, i: number) => {
          if (rows.length >= 200) throw new Error("Expanded multipart features exceed 200 assets.");
          const row = {
            ...base,
            rowId: `${base.rowId}:${i + 1}`,
            recordId: `${base.recordId}:part:${i + 1}`,
            name: `${base.name.slice(0, 175)} · part ${i + 1}`,
            warnings: [...base.warnings, "Multipart point/line expanded into individual assets."],
          };
          try {
            row.geometry = geometry({
              type: raw.type === "MultiPoint" ? "Point" : "LineString",
              coordinates: part,
            });
          } catch (error) {
            row.error = error instanceof Error ? error.message : "Invalid geometry.";
          }
          rows.push(row);
        });
      } else {
        base.geometry = geometry(raw);
        rows.push(base);
      }
    } catch (error) {
      base.error = error instanceof Error ? error.message : "Invalid feature.";
      rows.push(base);
    }
  });
  if (rows.length > 200)
    throw new Error(
      "Expanded multipart features exceed 200 assets. Split this file into smaller batches.",
    );
  return rows;
}
