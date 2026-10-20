/**
 * The change map on the ground (ADR 0018 § Web): each class's areas between two dates --
 * raised, lowered, not comparable -- as `GroundPrimitive`s, one per class and confidence
 * (lib/changeDrape.ts `changeBuckets`), classified onto terrain and 3D tiles alike.
 *
 * Ground geometry, not splats: CesiumJS draws it whichever renderer draws the scan, so it is
 * offered under every one. Never picked (`allowPicking: false`), so a tap goes through to what
 * is underneath. Confident areas are filled in their class's colour; low-confidence ones (a
 * date's confidence does not vouch for them) are hatched in the same colour, in screen space,
 * so the hatching reads the same on a one-cell area and a whole yard.
 */
import {
  ClassificationType,
  Color,
  GeometryInstance,
  GroundPrimitive,
  Material,
  MaterialAppearance,
  PolygonGeometry,
  PolygonHierarchy,
  type Scene,
  Cartesian3,
} from "cesium";
import type { Position } from "geojson";

import type { ChangeMap } from "@twin/contracts";
import { flattenRing } from "@twin/geo";

import { CHANGE_COLORS, changeBuckets, type ChangeClass } from "@/lib/changeDrape";

/** The property each drape primitive carries, saying what it draws (tests, e2e). */
export const CHANGE_DRAPE_TAG = "changeDrape";

export interface ChangeDrapeTag {
  cls: ChangeClass;
  lowConfidence: boolean;
  polygons: number;
  hatched: boolean;
}

/** Fill opacity: the scan stays readable through the drape. Not comparable is quieter. */
const ALPHA: Record<ChangeClass, number> = { raised: 0.6, lowered: 0.6, insufficient: 0.4 };

/**
 * Diagonal stripes in screen space, `spacing` pixels apart: the colour at full opacity on the
 * stripes, at a quarter between them.
 */
const HATCH_SOURCE = `czm_material czm_getMaterial(czm_materialInput materialInput) {
  czm_material material = czm_getDefaultMaterial(materialInput);
  float stripe = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / spacing));
  material.diffuse = color.rgb;
  material.alpha = color.a * mix(0.25, 1.0, stripe);
  return material;
}`;

function materialFor(cls: ChangeClass, hatched: boolean): Material {
  const color = Color.fromCssColorString(CHANGE_COLORS[cls]).withAlpha(ALPHA[cls]);
  if (!hatched) return Material.fromType("Color", { color });
  return new Material({
    fabric: { type: "ChangeHatch", uniforms: { color, spacing: 10 }, source: HATCH_SOURCE },
    translucent: true,
  });
}

/** A polygon's rings (outer first) as a hierarchy, or null for one too short to draw. */
function hierarchyOf([outer, ...holes]: Position[][]): PolygonHierarchy | null {
  if (!outer || outer.length < 4) return null;
  return new PolygonHierarchy(
    Cartesian3.fromDegreesArray(flattenRing(outer)),
    holes
      .filter((hole) => hole.length >= 4)
      .map((hole) => new PolygonHierarchy(Cartesian3.fromDegreesArray(flattenRing(hole)))),
  );
}

export class ChangeDrapeLayer {
  private primitives: { primitive: GroundPrimitive; tag: ChangeDrapeTag }[] = [];
  private destroyed = false;

  constructor(private readonly scene: Scene) {}

  /** Draws `map`'s areas, replacing whatever was drawn before. */
  show(map: Pick<ChangeMap, "areas">): void {
    this.clear();
    if (this.destroyed) return;
    for (const bucket of changeBuckets(map)) {
      const hierarchies = bucket.polygons
        .map(hierarchyOf)
        .filter((h): h is PolygonHierarchy => h !== null);
      if (!hierarchies.length) continue;
      const hatched = bucket.lowConfidence;
      const primitive = new GroundPrimitive({
        geometryInstances: hierarchies.map(
          (polygonHierarchy) =>
            new GeometryInstance({
              geometry: new PolygonGeometry({
                polygonHierarchy,
                vertexFormat: MaterialAppearance.MaterialSupport.TEXTURED.vertexFormat,
              }),
            }),
        ),
        appearance: new MaterialAppearance({
          material: materialFor(bucket.cls, hatched),
          flat: true,
          translucent: true,
        }),
        classificationType: ClassificationType.BOTH,
        allowPicking: false,
        asynchronous: true,
      });
      const tag: ChangeDrapeTag = {
        cls: bucket.cls,
        lowConfidence: bucket.lowConfidence,
        polygons: hierarchies.length,
        hatched,
      };
      Object.assign(primitive, { [CHANGE_DRAPE_TAG]: tag });
      this.scene.groundPrimitives.add(primitive);
      this.primitives.push({ primitive, tag });
    }
    this.scene.requestRender();
  }

  /** What is drawn, one entry per primitive, in drawing order. */
  drawn(): ChangeDrapeTag[] {
    return this.primitives.map(({ tag }) => tag);
  }

  /** Takes everything drawn off the ground. */
  clear(): void {
    if (!this.primitives.length) return;
    // `remove` destroys each primitive (the collection's `destroyPrimitives` default).
    for (const { primitive } of this.primitives) this.scene.groundPrimitives.remove(primitive);
    this.primitives = [];
    this.scene.requestRender();
  }

  destroy(): void {
    this.clear();
    this.destroyed = true;
  }
}
