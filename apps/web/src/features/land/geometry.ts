import type { Footprint } from "@twin/contracts";

/** Accept one boundary or a collection of polygon features without dropping holes. */
export function importBoundary(text: string): Footprint {
  const input: unknown = JSON.parse(text);
  const polygons: number[][][][] = [];
  let vertices = 0;
  const visit = (value: unknown): void => {
    if (!value || typeof value !== "object" || !("type" in value))
      throw new Error("Choose a GeoJSON polygon, feature, or feature collection.");
    if (value.type === "Feature" && "geometry" in value) return visit(value.geometry);
    if (
      value.type === "FeatureCollection" &&
      "features" in value &&
      Array.isArray(value.features)
    ) {
      value.features.forEach(visit);
      return;
    }
    if (!("coordinates" in value) || !Array.isArray(value.coordinates))
      throw new Error("The boundary has no coordinates.");
    const shapes: unknown[] =
      value.type === "Polygon"
        ? [value.coordinates]
        : value.type === "MultiPolygon"
          ? value.coordinates
          : [];
    if (!shapes.length) throw new Error("The file must contain polygons only.");
    for (const polygon of shapes) {
      if (!Array.isArray(polygon) || !polygon.length)
        throw new Error("A polygon needs an outer ring.");
      const rings: number[][][] = [];
      for (const ring of polygon) {
        if (!Array.isArray(ring) || ring.length < 4)
          throw new Error("Each ring needs at least four positions, including its closing point.");
        const positions: number[][] = [];
        for (const point of ring) {
          if (
            !Array.isArray(point) ||
            point.length < 2 ||
            point.length > 3 ||
            !point.every((v: unknown) => typeof v === "number" && Number.isFinite(v))
          )
            throw new Error("Invalid GeoJSON position.");
          const [lon, lat] = point as number[];
          if (lon === undefined || lat === undefined || Math.abs(lon) > 180 || Math.abs(lat) > 90)
            throw new Error("Coordinates must be longitude and latitude in WGS 84.");
          positions.push([lon, lat]);
          if (++vertices > 20_000) throw new Error("Import at most 20,000 boundary vertices.");
        }
        const first = positions[0],
          last = positions.at(-1);
        if (!first || !last || first[0] !== last[0] || first[1] !== last[1])
          throw new Error("Every boundary ring must be closed.");
        rings.push(positions);
      }
      polygons.push(rings);
    }
  };
  visit(input);
  if (!polygons.length) throw new Error("The file contains no land areas.");
  return { type: "MultiPolygon", coordinates: polygons };
}
