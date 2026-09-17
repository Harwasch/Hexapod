/**
 * "Click the ground you mean": turns one clicked point into an area. Map data first (the
 * smallest OpenStreetMap area that contains the point), the model's eyes second (it outlines
 * the feature in a picture of the view, when the API has a key), and a rough square around
 * the point last, labelled as the guess it is. Every result is editable on the map.
 */

import type { Footprint, GroundOutline } from "@twin/contracts";
import { footprintAreaM2, toAcres, type LonLat } from "@twin/geo";

import { rectangleFootprint, zoneFromFootprint } from "./areas";
import { OSM_ATTRIBUTION, fetchOsmContaining } from "./osm";
import type { Zone } from "./types";

export type GroundSource = "osm" | "vision" | "rough";

export interface ResolvedGround {
  zone: Zone;
  source: GroundSource;
  /** What the agent says about it, e.g. "a 42-acre field from OpenStreetMap". */
  note: string;
}

/** A picture of the view and the tools to map it back onto the ground. */
export interface ViewSnapshot {
  image: string;
  width: number;
  height: number;
}

export interface GroundResolver {
  /** The click in the view, normalized 0..1 (x right, y down), when it came from the map. */
  point: { x: number; y: number } | null;
  snapshot: () => Promise<ViewSnapshot | null>;
  /** Screen pixel → ground, or null when the pixel is sky. */
  unproject: (x: number, y: number) => LonLat | null;
  outline: (body: {
    image: string;
    width: number;
    height: number;
    point: { x: number; y: number };
    hint: string;
  }) => Promise<GroundOutline>;
  /** True when the API has a model to look with. */
  vision: boolean;
}

/** Half-size of the rough square when nothing better is known: a 14-acre working area. */
export const ROUGH_HALF_M = 120;
const MIN_VISION_ACRES = 0.2;

function acresOf(footprint: Footprint): number {
  return toAcres(footprintAreaM2(footprint));
}

/** The outline's image points on the ground: a polygon when at least three land. */
export function projectOutline(
  outline: GroundOutline,
  snapshot: ViewSnapshot,
  unproject: GroundResolver["unproject"],
): Footprint | null {
  const ring: [number, number][] = [];
  for (const p of outline.points) {
    const ground = unproject(p.x * snapshot.width, p.y * snapshot.height);
    if (ground) ring.push([ground.longitude, ground.latitude]);
  }
  const first = ring[0];
  if (ring.length < 3 || !first) return null;
  ring.push([first[0], first[1]]);
  const footprint: Footprint = { type: "Polygon", coordinates: [ring] };
  return acresOf(footprint) >= MIN_VISION_ACRES ? footprint : null;
}

export async function resolveGround(
  point: LonLat,
  id: string,
  hint: string,
  resolver: GroundResolver,
): Promise<ResolvedGround> {
  // 1. Map data: the mapped area under the click.
  try {
    const mapped = await fetchOsmContaining(point);
    if (mapped) {
      const zone: Zone = {
        ...zoneFromFootprint(id, mapped.name, mapped.footprint),
        attribution: OSM_ATTRIBUTION,
        note: OSM_ATTRIBUTION,
      };
      return {
        zone,
        source: "osm",
        note: `${mapped.name}, ${Math.round(mapped.acres).toLocaleString()} ac, outline from OpenStreetMap.`,
      };
    }
  } catch {
    // OpenStreetMap unreachable: fall through to the next way of looking.
  }
  // 2. The model looks at the view.
  if (resolver.vision && resolver.point) {
    try {
      const snapshot = await resolver.snapshot();
      if (snapshot) {
        const outline = await resolver.outline({
          image: snapshot.image,
          width: snapshot.width,
          height: snapshot.height,
          point: resolver.point,
          hint,
        });
        const footprint = projectOutline(outline, snapshot, resolver.unproject);
        if (footprint) {
          const name = outline.label.charAt(0).toUpperCase() + outline.label.slice(1);
          const zone: Zone = {
            ...zoneFromFootprint(id, name, footprint),
            note: `Outlined from imagery by the agent (confidence ${Math.round(outline.confidence * 100)}%). ${outline.note}`.trim(),
            shaped: true,
          };
          return {
            zone,
            source: "vision",
            note: `${name}, about ${Math.round(acresOf(footprint)).toLocaleString()} ac, traced from the imagery${outline.note ? `: ${outline.note}` : "."} Drag the corners to fit.`,
          };
        }
      }
    } catch {
      // The model could not outline it: the rough square below still works.
    }
  }
  // 3. A rough square, honestly labelled.
  const footprint = rectangleFootprint(point, ROUGH_HALF_M, ROUGH_HALF_M);
  const zone: Zone = {
    ...zoneFromFootprint(id, "Rough area", footprint),
    note: "A rough square around the click; drag the corners to fit.",
    shaped: true,
  };
  return {
    zone,
    source: "rough",
    note: `No mapped feature there, so I took a rough ${Math.round(acresOf(footprint))}-acre square around your click. Drag the corners to fit.`,
  };
}
