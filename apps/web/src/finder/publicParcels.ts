/**
 * The finder's parcel lookup: `POST /api/v1/public/parcels/near`, the public, rate-limited
 * route (apps/api/app/api/v1/public.py). Its own client, without the app's middleware: that
 * attaches the workspace write token to every POST, and this page must never send one.
 */
import createClient from "openapi-fetch";

import type { CandidateResult, Footprint, LandCandidate, paths } from "@twin/contracts";
import { footprintAreaM2 } from "@twin/geo";

import { env } from "@/app/env";

import type { FoundLand } from "./handoff";

const client = createClient<paths>({ baseUrl: env.apiBaseUrl });

export async function findPublicParcels(
  point: [number, number],
  signal: AbortSignal,
): Promise<CandidateResult> {
  const { data, response } = await client.POST("/api/v1/public/parcels/near", {
    body: { point: { type: "Point", coordinates: point } },
    signal,
  });
  if (response.status === 429)
    throw new Error("That was a lot of lookups at once. Wait a few seconds, then tap again.");
  if (!data) throw new Error(`Parcel records could not be looked up (HTTP ${response.status}).`);
  return data;
}

/** The chosen parcels as one boundary: a polygon, or every polygon as one MultiPolygon. */
export function parcelsBoundary(chosen: LandCandidate[]): Footprint {
  const polygons = chosen.flatMap((candidate) =>
    candidate.geometry.type === "Polygon"
      ? [candidate.geometry.coordinates]
      : candidate.geometry.type === "MultiPolygon"
        ? candidate.geometry.coordinates
        : [],
  );
  const [only] = polygons;
  if (polygons.length === 1 && only) return { type: "Polygon", coordinates: only };
  return { type: "MultiPolygon", coordinates: polygons };
}

/** What the finder hands back for parcels chosen from county records. */
export function foundParcels(chosen: LandCandidate[]): FoundLand {
  const boundary = parcelsBoundary(chosen);
  const attributions = [
    ...new Set(chosen.map((candidate) => candidate.source.attribution).filter(Boolean)),
  ] as string[];
  return {
    method: "parcels",
    boundary,
    parcelIds: chosen.map((candidate) => candidate.id),
    label: chosen.length === 1 ? (chosen[0]?.label ?? "1 parcel") : `${chosen.length} parcels`,
    areaM2: footprintAreaM2(boundary),
    attributions,
  };
}
