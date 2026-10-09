import RBush, { type BBox } from "rbush";
import type { LandMapGeometry } from "@twin/contracts";
import type { LandContextLayer } from "@/state/landContext";
import type { LandPoint } from "@/state/land";

interface Edge extends BBox {
  a: LandPoint;
  b: LandPoint;
  label: string;
}
export interface MappedSnap {
  point: LandPoint;
  label: string;
  distance: number;
}
export class MappedSnapIndex {
  get empty(): boolean {
    return this.layers.size === 0;
  }
  private layers = new Map<string, { features: LandContextLayer["features"]; tree: RBush<Edge> }>();

  sync(layers: Record<string, LandContextLayer>): boolean {
    const eligible = Object.values(layers).filter(
      (layer) =>
        Boolean(layer.researchArtifactId) ||
        layer.id === "inventory" ||
        layer.id === "drawing-guides",
    );
    const ids = new Set(eligible.map((layer) => layer.id));
    let changed = false;
    for (const id of this.layers.keys())
      if (!ids.has(id)) {
        this.layers.delete(id);
        changed = true;
      }
    for (const layer of eligible) {
      if (this.layers.get(layer.id)?.features === layer.features) continue;
      const edges: Edge[] = [];
      const add = (first: number[], second: number[], label: string) => {
        const [ax, ay] = first,
          [bx, by] = second;
        if (
          ax === undefined ||
          ay === undefined ||
          bx === undefined ||
          by === undefined ||
          ![ax, ay, bx, by].every(Number.isFinite) ||
          Math.abs(ax - bx) > 180
        )
          return;
        edges.push({
          a: [ax, ay],
          b: [bx, by],
          label,
          minX: Math.min(ax, bx),
          minY: Math.min(ay, by),
          maxX: Math.max(ax, bx),
          maxY: Math.max(ay, by),
        });
      };
      const line = (points: number[][], label: string) => {
        for (let index = 1; index < points.length; index++) {
          const a = points[index - 1],
            b = points[index];
          if (a && b) add(a, b, label);
        }
      };
      const geometry = (value: LandMapGeometry, label: string) => {
        if (value.type === "Point") add(value.coordinates, value.coordinates, label);
        else if (value.type === "LineString") line(value.coordinates, label);
        else if (value.type === "Polygon") value.coordinates.forEach((ring) => line(ring, label));
        else value.coordinates.forEach((polygon) => polygon.forEach((ring) => line(ring, label)));
      };
      layer.features.forEach((feature) =>
        geometry(feature.geometry, `${layer.title}: ${feature.label}`),
      );
      this.layers.set(layer.id, { features: layer.features, tree: new RBush<Edge>().load(edges) });
      changed = true;
    }
    return changed;
  }

  nearest(
    point: LandPoint,
    bounds: BBox,
    distance: (target: LandPoint) => number,
    tolerance = 12,
  ): MappedSnap | null {
    let best: MappedSnap | null = null;
    const scale = Math.max(1e-6, Math.cos((point[1] * Math.PI) / 180));
    for (const { tree } of this.layers.values())
      for (const edge of tree.search(bounds)) {
        const [ax, ay] = edge.a,
          [bx, by] = edge.b;
        const dx = (bx - ax) * scale,
          dy = by - ay,
          length = dx * dx + dy * dy;
        const t = length
          ? Math.max(0, Math.min(1, ((point[0] - ax) * scale * dx + (point[1] - ay) * dy) / length))
          : 0;
        const target: LandPoint = [ax + t * (bx - ax), ay + t * dy];
        const pixels = distance(target);
        if (Number.isFinite(pixels) && pixels <= tolerance && (!best || pixels < best.distance))
          best = { point: target, label: edge.label, distance: pixels };
      }
    return best;
  }
}
