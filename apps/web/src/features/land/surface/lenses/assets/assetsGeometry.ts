import type { LandMapGeometry } from "@twin/contracts";

export type Vertex = [number, number];
/** Open rings while editing; the closing coordinate is derived on submission. */
export interface InventoryShape {
  type: LandMapGeometry["type"];
  parts: Vertex[][][];
}
export function openInventoryShape(geometry: LandMapGeometry): InventoryShape {
  const point = (p: number[]): Vertex => [p[0] ?? NaN, p[1] ?? NaN];
  if (geometry.type === "Point")
    return { type: geometry.type, parts: [[[point(geometry.coordinates)]]] };
  if (geometry.type === "LineString")
    return { type: geometry.type, parts: [[geometry.coordinates.map(point)]] };
  const parts = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  return {
    type: geometry.type,
    parts: parts.map((part) => part.map((ring) => ring.slice(0, -1).map(point))),
  };
}
export function vertexCount(shape: InventoryShape): number {
  return shape.parts.reduce(
    (total, part) => total + part.reduce((n, ring) => n + ring.length, 0),
    0,
  );
}
export function validVertex(p: Vertex): boolean {
  return (
    Number.isFinite(p[0]) && Number.isFinite(p[1]) && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90
  );
}
export function closeInventoryShape(shape: InventoryShape): LandMapGeometry {
  if (!shape.parts.length || shape.parts.some((part) => !part.length))
    throw new Error("Each area needs an outer ring.");
  const rings = shape.parts.flat();
  if (rings.some((ring) => ring.some((p) => !validVertex(p))))
    throw new Error(
      "Enter a longitude from −180 to 180 and latitude from −90 to 90 for every vertex.",
    );
  const polygon = shape.type === "Polygon" || shape.type === "MultiPolygon";
  if (vertexCount(shape) + (polygon ? rings.length : 0) > 20_000)
    throw new Error("Use at most 20,000 vertices, including ring closures.");
  const line = shape.parts[0]?.[0] ?? [];
  if (shape.type === "Point") {
    if (line.length !== 1 || !line[0]) throw new Error("A point needs one location.");
    return { type: "Point", coordinates: line[0] };
  }
  if (shape.type === "LineString") {
    if (new Set(line.map((p) => p.join(","))).size < 2)
      throw new Error("A line needs at least two distinct vertices.");
    return { type: "LineString", coordinates: line };
  }
  if (rings.some((ring) => new Set(ring.map((p) => p.join(","))).size < 3))
    throw new Error("Every ring needs at least three distinct vertices.");
  const parts = shape.parts.map((part) => part.map((ring) => [...ring, ...ring.slice(0, 1)]));
  return shape.type === "Polygon"
    ? { type: "Polygon", coordinates: parts[0] ?? [] }
    : { type: "MultiPolygon", coordinates: parts };
}
export function keepGeometryHistory(values: InventoryShape[]): InventoryShape[] {
  let vertices = 0;
  return values
    .slice(-50)
    .reverse()
    .filter((shape) => (vertices += vertexCount(shape)) <= 100_000)
    .reverse();
}
