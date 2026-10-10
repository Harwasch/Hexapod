import type { Footprint } from "@twin/contracts";
import { polygonsOf } from "@twin/geo";
import type { LandPoint } from "@/state/land";

export interface BoundaryVertex {
  polygon: number;
  ring: number;
  vertex: number;
}
/** Nearest point on the GeoJSON boundary; the renderer applies a screen-pixel tolerance. */
export function nearestBoundaryPoint(
  boundary: Footprint,
  point: LandPoint,
  excluded: BoundaryVertex | null = null,
): LandPoint | null {
  let best: LandPoint | null = null,
    distance = Infinity;
  const scale = Math.max(1e-6, Math.cos((point[1] * Math.PI) / 180));
  const consider = (x: number, y: number) => {
    const squared = ((x - point[0]) * scale) ** 2 + (y - point[1]) ** 2;
    if (squared < distance) {
      distance = squared;
      best = [x, y];
    }
  };
  polygonsOf(boundary).forEach((polygon, pi) =>
    polygon.forEach((ring, ri) => {
      const skip = excluded?.polygon === pi && excluded.ring === ri ? excluded.vertex : -1;
      for (let i = 0; i < ring.length - 1; i++) {
        const a = ring[i],
          b = ring[i + 1];
        if (!a || !b) continue;
        const ax = a[0],
          ay = a[1],
          bx = b[0],
          by = b[1];
        if (ax === undefined || ay === undefined || bx === undefined || by === undefined) continue;
        const next = (i + 1) % (ring.length - 1);
        if (i !== skip) consider(ax, ay);
        if (i === skip || next === skip) continue;
        const dx = (bx - ax) * scale,
          dy = by - ay;
        const length = dx * dx + dy * dy;
        if (!length) continue;
        const t = Math.max(
          0,
          Math.min(1, ((point[0] - ax) * scale * dx + (point[1] - ay) * dy) / length),
        );
        consider(ax + t * (bx - ax), ay + t * dy);
      }
    }),
  );
  return best;
}
