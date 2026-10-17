import { useQuery } from "@tanstack/react-query";
import { useCallback, useMemo } from "react";

import type {
  DatedImageryCatalog,
  DatedImageryProviderStatus,
  DatedImagerySource,
  StatedAccuracy,
} from "@twin/contracts";
import { boundsOf } from "@twin/geo";

import { api, unwrap } from "@/api/client";
import { env } from "@/app/env";
import type { LanternSource, LanternSourceKind } from "@/cesium/lanternTypes";
import { useLand } from "@/state/land";
import { useLantern, type LanternBbox, type LanternKind } from "@/state/lantern";

/** Said of every source, every time: nothing here was measured against the Google mesh. */
export const NOT_CHECKED = "not checked against today's 3D world";
/** The catalog route takes areas this size at most (degrees a side). */
const MAX_SIDE_DEG = 0.25;
/**
 * A saved land's bounds reach the public catalog route only this coarsely (degrees), so that
 * route never receives the land's own extent. It does not hide where the operator looks: the
 * lens's tiles (USGS sheets from this API's public tile route down to level 17, about 150 m;
 * NAIP and Sentinel-2 from Planetary Computer) are asked for the ground under it, as any
 * map's tiles are.
 */
const PRIVATE_ROUNDING_DEG = 0.05;
const ROUNDING_DEG = 0.01;
const SIX_HOURS_MS = 6 * 3600 * 1000;
/** A list some source could not answer is asked again this soon (the server keeps it 30 s). */
const INCOMPLETE_STALE_MS = 30 * 1000;
const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const KIND_OF: Record<LanternSourceKind, LanternKind> = {
  "aerial-photo": "aerial",
  satellite: "satellite",
  "historical-map": "maps",
  "aligned-map": "maps",
};

/** An area rounded outward to `step` degrees, and kept within the route's size. */
export function roundBbox(bbox: LanternBbox, step = ROUNDING_DEG): LanternBbox {
  const down = (value: number) => Math.floor(Number((value / step).toFixed(6))) * step;
  const up = (value: number) => Math.ceil(Number((value / step).toFixed(6))) * step;
  let [west, south, east, north] = [down(bbox[0]), down(bbox[1]), up(bbox[2]), up(bbox[3])];
  if (east - west > MAX_SIDE_DEG) {
    const centre = (bbox[0] + bbox[2]) / 2;
    west = down(centre - MAX_SIDE_DEG / 2 + step);
    east = west + MAX_SIDE_DEG - step;
  }
  if (north - south > MAX_SIDE_DEG) {
    const centre = (bbox[1] + bbox[3]) / 2;
    south = down(centre - MAX_SIDE_DEG / 2 + step);
    north = south + MAX_SIDE_DEG - step;
  }
  const fix = (value: number) => Number(value.toFixed(6));
  return [fix(west), fix(south), fix(east), fix(north)];
}

/**
 * A calendar day ("2016-06-15") as the catalog writes its own labels ("15 June 2016"), so the
 * rim and the strip never show one date two ways; no time zone moves it.
 */
export function formatDay(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  const month = match ? MONTHS[Number(match[2]) - 1] : undefined;
  if (!match || !month) return iso;
  return `${Number(match[3])} ${month} ${match[1]}`;
}

const metres = (value: number) =>
  value >= 10 ? Math.round(value).toLocaleString() : value.toFixed(1).replace(/\.0$/, "");

/**
 * How far off a source may be, in a few words, and always that nobody checked it against the
 * 3D world: a measured fit; the record's own figure; the standard it was made to; or, where
 * nothing is stated, the source's own words for that (never a figure borrowed from elsewhere).
 * A map moved from an older datum adds how well that move is stated to be.
 */
export function accuracyLabel(
  accuracy: StatedAccuracy,
  shift?: DatedImagerySource["datumShift"],
): string {
  const value = accuracy.valueM;
  let stated: string;
  if (accuracy.kind === "measured-fit" && value != null) stated = `fit RMS ${metres(value)} m`;
  else if (accuracy.kind === "stated-by-source" && value != null)
    stated = `stated ±${metres(value)} m by its record`;
  else if (accuracy.kind === "standard" && value != null)
    stated = `NMAS ±${metres(value)} m (90%) if compliant`;
  else if (accuracy.basis.length <= 90) stated = accuracy.basis.replace(/\.$/, "");
  else stated = "accuracy not stated";
  if (shift)
    stated += shift.accuracyM
      ? ` · datum shift ±${metres(shift.accuracyM)} m`
      : " · datum shift accuracy not stated";
  return `${stated} · ${NOT_CHECKED}`;
}

/** The API's source as the lantern draws it: API-relative tiles made absolute. */
export function toLanternSource(source: DatedImagerySource): LanternSource {
  const template = source.tiles.urlTemplate.startsWith("/")
    ? `${env.apiBaseUrl}${source.tiles.urlTemplate}`
    : source.tiles.urlTemplate;
  const parts = source.parts.map((part) => ({
    footprint: part.footprint,
    dateLabel: part.observedOn ? formatDay(part.observedOn) : source.dateLabel,
    accuracyLabel: accuracyLabel(part.accuracy, source.datumShift),
    recordUrl: part.recordUrl,
  }));
  const [west = 0, south = 0, east = 0, north = 0] = source.bounds;
  return {
    id: source.id,
    kind: source.kind,
    title: source.title,
    dateLabel: source.dateLabel,
    dateMeaning: source.dateMeaning,
    sortDate: source.sortDate,
    resolutionM: source.resolutionM ?? null,
    mapScale: source.mapScale ?? null,
    license: source.license,
    licenseUrl: source.licenseUrl ?? null,
    attribution: source.attribution,
    accuracyLabel:
      parts[0]?.accuracyLabel ??
      accuracyLabel({ kind: "not-stated", basis: "accuracy not stated" }, source.datumShift),
    bounds: [west, south, east, north],
    parts,
    tiles: {
      scheme: source.tiles.scheme,
      urlTemplate: template,
      maximumLevel: source.tiles.maximumLevel,
    },
    warnings: source.warnings ?? [],
  };
}

/** Which kind chip a source answers to. */
export function kindOf(source: Pick<LanternSource, "kind">): LanternKind {
  return KIND_OF[source.kind];
}

/** Whether a catalog lacks a source that may answer if asked again. */
export function incomplete(catalog: Pick<DatedImageryCatalog, "providers"> | undefined): boolean {
  return Boolean(catalog?.providers.some((provider) => provider.status === "unavailable"));
}

/** Where the list is for, in the words the strip uses ("Move over … to look back"). */
export type LanternAreaName =
  { kind: "chosen" } | { kind: "land"; name: string } | { kind: "view" };

export interface DatedSources {
  /** The area listed, rounded; null until something says where. */
  bbox: LanternBbox | null;
  /** What that area is: the place a descent chose, the selected land, or the view's centre. */
  area: LanternAreaName | null;
  /** What the kind chips let through, newest first: the rail, and what the lens steps through. */
  sources: LanternSource[];
  /** Every source found, whatever the chips say. */
  all: LanternSource[];
  providers: DatedImageryProviderStatus[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  /** Some source could not be reached: the list may be missing what it would have listed. */
  incomplete: boolean;
  /** Asking again now. */
  refreshing: boolean;
  /** Asks the catalog again. */
  retry: () => void;
  /** Asks again only if the list is stale (30 s for an incomplete one, else 6 h). */
  refreshIfStale: () => void;
}

/**
 * The dated imagery for where the lantern looks: the area a descent chose, else the selected
 * land (rounded outward to 0.05°, so the public catalog route never gets its extent; the lens's
 * tiles still name the ground viewed, see PRIVATE_ROUNDING_DEG), else around the view's centre
 * when the lantern opened. (A land selected after a descent takes over from the
 * descent's area: LanternBridge clears it.) Asked for only once the lantern has been opened, or
 * a descent has landed; cached for 6 h, as the server does, or 30 s when a source could not be
 * reached.
 */
export function useDatedSources(): DatedSources {
  const requested = useLantern((s) => s.requested);
  const override = useLantern((s) => s.bboxOverride);
  const auto = useLantern((s) => s.autoBbox);
  const kinds = useLantern((s) => s.kinds);
  const land = useLand((s) => s.active);
  const bbox = useMemo<LanternBbox | null>(() => {
    if (override) return roundBbox(override);
    if (land) {
      const b = boundsOf(land.boundary);
      return roundBbox([b.west, b.south, b.east, b.north], PRIVATE_ROUNDING_DEG);
    }
    return auto ? roundBbox(auto) : null;
  }, [override, land, auto]);
  const landName = land?.name ?? null;
  const area = useMemo<LanternAreaName | null>(() => {
    if (override) return { kind: "chosen" };
    if (landName !== null) return { kind: "land", name: landName };
    return auto ? { kind: "view" } : null;
  }, [override, landName, auto]);
  const query = useQuery({
    queryKey: ["dated-imagery", ...(bbox ?? [])],
    queryFn: ({ signal }) => {
      const [west = 0, south = 0, east = 0, north = 0] = bbox ?? [];
      return unwrap(
        api.GET("/api/v1/imagery/dated", {
          params: { query: { west, south, east, north } },
          signal,
        }),
      );
    },
    enabled: requested && bbox !== null,
    staleTime: (q) => (incomplete(q.state.data) ? INCOMPLETE_STALE_MS : SIX_HOURS_MS),
    gcTime: SIX_HOURS_MS,
    retry: 1,
  });
  const all = useMemo(() => (query.data?.sources ?? []).map(toLanternSource), [query.data]);
  const sources = useMemo(() => all.filter((source) => kinds[kindOf(source)]), [all, kinds]);
  const { refetch, isStale } = query;
  const retry = useCallback(() => void refetch(), [refetch]);
  const refreshIfStale = useCallback(() => {
    if (isStale && query.data) void refetch();
  }, [isStale, query.data, refetch]);
  return {
    bbox,
    area,
    sources,
    all,
    providers: query.data?.providers ?? [],
    status:
      !requested || !bbox ? "idle" : query.isError ? "error" : query.data ? "ready" : "loading",
    error: query.error ? String(query.error.message || query.error) : null,
    incomplete: incomplete(query.data),
    refreshing: query.isFetching && query.data !== undefined,
    retry,
    refreshIfStale,
  };
}
