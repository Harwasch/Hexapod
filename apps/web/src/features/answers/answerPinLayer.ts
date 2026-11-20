import { Cartesian3, Color, type Entity, type EntityCollection } from "cesium";
import type { AnswerPin } from "./propertyPins";

/** Own only this layer's entities: cleanup must not remove scans or another feature's pins. */
export function addAnswerPins(entities: EntityCollection, pins: readonly AnswerPin[]): () => void {
  const added: Entity[] = [];
  entities.suspendEvents();
  try {
    for (const pin of pins) {
      added.push(
        entities.add({
          id: `property-answer:${pin.siteId}:${pin.id}`,
          name: pin.label,
          position: Cartesian3.fromDegrees(pin.geo.lon, pin.geo.lat, pin.geo.alt),
          point: { pixelSize: 12, color: Color.ORANGE, outlineColor: Color.BLACK, outlineWidth: 2 },
          properties: {
            propertyAnswerId: pin.id,
            propertySiteId: pin.siteId,
            captureId: pin.scanId,
          },
        }),
      );
    }
  } finally {
    entities.resumeEvents();
  }
  return () => {
    for (const entity of added) entities.remove(entity);
  };
}
