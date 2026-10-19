import type { components, LandArea, LandEvidence, LandMapGeometry } from "@twin/contracts";

/**
 * Plain words for the Records lens, computed from the data alone: where the outline came from,
 * what the documents are, and the old maps and photos found near the land.
 */

export type LandDocument = components["schemas"]["LandDocumentRead"];
export type DocumentKind = NonNullable<LandDocument["kind"]>;
export type DocumentLink = components["schemas"]["DocumentLinkRead"];
export type ArchiveMedia = components["schemas"]["ArchiveMedia"];
type Source = LandArea["source"];
type Polygon = components["schemas"]["Polygon"];

/** The investigation every archive run gets; `useLandAnalysis` finds it by this title. */
export const ARCHIVE_TITLE = "Photographs and historical maps";
export const ARCHIVE_QUESTION =
  "Discover openly licensed photographs and historical map sheets relevant to this land.";

export const DOCUMENT_KINDS: { value: DocumentKind; label: string }[] = [
  { value: "deed", label: "Deed" },
  { value: "easement", label: "Easement" },
  { value: "survey", label: "Survey" },
  { value: "mineral", label: "Mineral rights" },
  { value: "water", label: "Water rights" },
  { value: "historical", label: "Historical record" },
  { value: "report", label: "Report" },
  { value: "other", label: "Other" },
];

export function kindLabel(kind: LandDocument["kind"]): string {
  return DOCUMENT_KINDS.find((item) => item.value === kind)?.label ?? "Other";
}

export const STATUS_LABEL: Record<LandDocument["status"], string> = {
  "awaiting-upload": "Upload not finished",
  ready: "Text ready",
  "needs-ocr": "Scanned, needs reading",
  unreadable: "Text could not be read",
};

export const RELATIONS: { value: DocumentLink["relation"]; label: string }[] = [
  { value: "related", label: "is related to" },
  { value: "amends", label: "amends" },
  { value: "supersedes", label: "replaces" },
  { value: "conflicts-with", label: "conflicts with" },
  { value: "mentions", label: "mentions" },
  { value: "parcel-lineage", label: "is in the parcel history of" },
];

export function relationLabel(relation: DocumentLink["relation"]): string {
  return RELATIONS.find((item) => item.value === relation)?.label ?? relation;
}

function plural(count: number, one: string, many = `${one}s`): string {
  return `${count.toLocaleString()} ${count === 1 ? one : many}`;
}

/** The parcel record numbers behind the outline: its own, or the parcels it was combined from. */
export function parcelIds(source: Source): string[] {
  const ids = [
    source.method === "parcel" ? source.recordId : null,
    ...(source.references ?? [])
      .filter((reference) => reference.method === "parcel")
      .map((reference) => reference.recordId),
  ];
  return [...new Set(ids.filter((id): id is string => Boolean(id?.trim())))];
}

const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Whether a document lists one of the land's parcel numbers ("019-210-034" = "019210034"). */
export function namesParcel(document: LandDocument, ids: string[]): boolean {
  if (!ids.length) return false;
  const wanted = new Set(ids.map(normalize).filter(Boolean));
  return (document.parcelReferences ?? []).some((reference) => wanted.has(normalize(reference)));
}

/** Where the outline came from, as the start of the headline. */
export function boundaryPhrase(source: Source): string {
  const ids = parcelIds(source);
  switch (source.method) {
    case "parcel":
      if (source.recordId) return `From a county parcel record (parcel ${source.recordId}).`;
      return ids.length > 1
        ? `Combined from ${ids.length} county parcel records.`
        : "From county parcel records.";
    case "drawn":
      return source.label === "Adjusted on the map"
        ? "Outlined and adjusted by you on the map."
        : "Drawn by you on the map.";
    case "corridor":
      return "A corridor you drew along a line.";
    case "imported":
      return source.label ? `Imported from ${source.label}.` : "Imported from a file.";
    case "mapped-feature":
      return /openstreetmap/i.test(source.label)
        ? "Traced from OpenStreetMap."
        : "Traced from a mapped feature.";
    case "imagery":
      return "Traced from imagery.";
    default:
      return "Outlined by you.";
  }
}

/** "9 old maps and photos", "4 old maps", "6 old photos". */
export function archivePhrase(maps: number, photos: number): string {
  if (maps && photos) return plural(maps + photos, "old map and photo", "old maps and photos");
  if (maps) return plural(maps, "old map");
  if (photos) return plural(photos, "old photo");
  return "";
}

export function recordsHeadline(
  source: Source,
  documents: number | null,
  maps: number,
  photos: number,
): string {
  const parts = [
    documents ? plural(documents, "document") : "",
    archivePhrase(maps, photos),
  ].filter(Boolean);
  const counts = parts.length ? ` ${parts.join(" and ")}.` : "";
  return `${boundaryPhrase(source)}${counts}`;
}

/** The first plausible year in free catalog text ("circa 1890–1895" → 1890). */
export function yearOf(text: string | null | undefined): number | null {
  const match = /\b(1[5-9]\d\d|20\d\d)\b/.exec(text ?? "");
  if (!match) return null;
  const year = Number(match[1]);
  return year <= new Date().getFullYear() + 1 ? year : null;
}

/** A catalog title without file prefixes, extensions, scales or trailing years. */
export function shortTitle(title: string): string {
  const cleaned = title
    .replace(/^file:/i, "")
    .replace(/\.(jpe?g|png|tiff?|webp|gif)$/i, "")
    .replace(/_/g, " ")
    .replace(/^USGS\s+[\d:,.-]+\s*-?\s*scale\s+quadrangle\s+for\s+/i, "")
    .replace(/\s*\(?1:\s?[\d,]+\)?\s*$/i, "")
    .replace(/,?\s+(1[5-9]\d\d|20\d\d)$/, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || title;
}

export function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1).trimEnd()}…` : text;
}

/** One old photo or map sheet found near the land. */
export interface ArchiveItem {
  id: string;
  evidence: LandEvidence;
  media: ArchiveMedia;
  kind: ArchiveMedia["kind"];
  year: number | null;
  /** A short title for pins and lists. */
  title: string;
  geometry: LandMapGeometry | null;
  /** A photo whose catalog point is inside the boundary; a map sheet always covers the land. */
  onLand: boolean;
}

export function toArchiveItem(evidence: LandEvidence): ArchiveItem | null {
  const media = evidence.media;
  if (!media) return null;
  return {
    id: evidence.id,
    evidence,
    media,
    kind: media.kind,
    year: yearOf(media.sourceDate),
    title: shortTitle(media.title),
    geometry: media.location ?? null,
    onLand: media.kind === "historical-map" || evidence.spatialRelevance === "within",
  };
}

const byYear = (a: ArchiveItem, b: ArchiveItem) =>
  (a.year ?? 9999) - (b.year ?? 9999) || a.title.localeCompare(b.title);

/** Map sheets oldest first, then photos: those on the land first, each group oldest first. */
export function archiveItems(evidence: LandEvidence[] | null | undefined): ArchiveItem[] {
  const items = (evidence ?? [])
    .map(toArchiveItem)
    .filter((item): item is ArchiveItem => item !== null);
  const seen = new Set<string>();
  const unique = items.filter((item) => !seen.has(item.id) && seen.add(item.id));
  const maps = unique.filter((item) => item.kind === "historical-map").sort(byYear);
  const photos = unique
    .filter((item) => item.kind === "photograph")
    .sort((a, b) => Number(b.onLand) - Number(a.onLand) || byYear(a, b));
  return [...maps, ...photos];
}

/** Map sheets that share a catalog footprint: editions of one quadrangle. */
export interface SheetGroup {
  key: string;
  footprint: Polygon;
  items: ArchiveItem[];
}

function bboxKey(geometry: LandMapGeometry): string | null {
  if (geometry.type !== "Polygon") return null;
  const ring = geometry.coordinates[0] ?? [];
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const [lon = 0, lat = 0] of ring) {
    west = Math.min(west, lon);
    east = Math.max(east, lon);
    south = Math.min(south, lat);
    north = Math.max(north, lat);
  }
  return [west, south, east, north].map((value) => value.toFixed(4)).join(",");
}

export function sheetGroups(items: ArchiveItem[]): SheetGroup[] {
  const groups = new Map<string, SheetGroup>();
  for (const item of items) {
    if (item.kind !== "historical-map" || item.geometry?.type !== "Polygon") continue;
    const key = bboxKey(item.geometry);
    if (!key) continue;
    const group = groups.get(key);
    if (group) group.items.push(item);
    else groups.set(key, { key, footprint: item.geometry, items: [item] });
  }
  return [...groups.values()];
}

/** "1894", "1894–1956", or "Undated". */
export function yearRange(items: ArchiveItem[]): string {
  const years = items.map((item) => item.year).filter((year): year is number => year !== null);
  if (!years.length) return "Undated";
  const first = Math.min(...years);
  const last = Math.max(...years);
  return first === last ? String(first) : `${first}–${last}`;
}

/** The newest document by its own date, else when it was recorded, else when it was added. */
export function latestDocument(documents: LandDocument[]): LandDocument | null {
  const when = (document: LandDocument) =>
    document.documentDate ?? document.recordedDate ?? document.createdAt;
  return documents.reduce<LandDocument | null>(
    (latest, document) => (!latest || when(document) > when(latest) ? document : latest),
    null,
  );
}

/** "1998" from the document's date, else its recorded date. */
export function documentYear(document: LandDocument): number | null {
  return yearOf(document.documentDate ?? document.recordedDate ?? null);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Saves a blob through a temporary link. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The alignment draft kept in this browser while matching points (`…:alignment:<id>:<sha>`). */
export function alignmentDraftKey(scope: string, evidenceId: string, sha256: string): string {
  return `living-world-land-draft:${encodeURIComponent(scope)}:alignment:${evidenceId}:${sha256}`;
}

const PROVIDERS: Record<string, string> = {
  "commons-place-images": "Wikimedia Commons",
  "usgs-historical-maps": "USGS historical maps",
  "usgs-elevation": "USGS elevation",
  "nasa-power": "NASA POWER climate",
  "gbif-occurrences": "GBIF species records",
  "usda-soils": "USDA soil survey",
  "fema-flood-zones": "FEMA flood maps",
  "epa-ecoregions": "EPA ecoregions",
  "usda-ecological-sites": "USDA ecological sites",
  "col-taxonomy": "Catalogue of Life",
};

/** A source's provider in words ("usda-soils" → "USDA soil survey"). */
export function providerLabel(provider: string): string {
  return PROVIDERS[provider] ?? provider.replace(/-/g, " ");
}
