import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Entity, ScreenSpaceEventHandler, ScreenSpaceEventType, type Cartesian2 } from "cesium";
import { useScene } from "@/cesium/SceneContext";
import { repaintUntilDrawn } from "@/cesium/repaintUntilDrawn";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";
import { useAsCaptured } from "@/state/asCaptured";
import { addAnswerPins } from "./answerPinLayer";
import { loadAnswerPins, type AnswerPin } from "./propertyPins";

/** Mounted only by the Cesium globe. No tool-rail button or gallery dependency. */
export function AnswerPinBridge() {
  const host = useScene();
  const siteId = useSites((s) => s.activeSiteId);
  const hidden = useAsCaptured((s) => s.on);
  const [selected, setSelected] = useState<AnswerPin | null>(null);
  const query = useQuery({
    queryKey: ["property-answer-pins", siteId],
    queryFn: ({ signal }) => (siteId ? loadAnswerPins(siteId, signal) : Promise.resolve([])),
    enabled: Boolean(siteId) && !hidden,
    staleTime: 30_000,
    refetchInterval: 60_000,
    retry: false,
  });
  useEffect(() => {
    if (!host || host.isDestroyed || hidden || !query.data) return;
    const remove = addAnswerPins(host.viewer.entities, query.data);
    const stopRepaint = repaintUntilDrawn(host);
    const handler = new ScreenSpaceEventHandler(host.scene.canvas);
    handler.setInputAction((event: { position: Cartesian2 }) => {
      const ui = useUi.getState();
      if (ui.measureMode || ui.exploreMode) return;
      const picked: unknown = host.scene.pick(event.position);
      const entity = picked && typeof picked === "object" && "id" in picked ? picked.id : null;
      const pin =
        entity instanceof Entity
          ? query.data.find((p) => entity.id === `property-answer:${p.siteId}:${p.id}`)
          : undefined;
      setSelected(pin ?? null);
    }, ScreenSpaceEventType.LEFT_CLICK);
    return () => {
      handler.destroy();
      stopRepaint();
      if (!host.isDestroyed) {
        remove();
        host.scene.requestRender();
      }
    };
  }, [host, hidden, query.data]);
  const pin =
    !hidden && selected?.siteId === siteId ? query.data?.find((p) => p.id === selected.id) : null;
  if (!pin) return null;
  return (
    <aside
      aria-label="Answer"
      style={{
        position: "absolute",
        bottom: 72,
        right: 20,
        zIndex: 20,
        padding: 16,
        background: "#182126",
        borderRadius: 12,
        color: "white",
      }}
    >
      <strong>{pin.label}</strong>
      <p>
        <time dateTime={pin.observedAt}>{new Date(pin.observedAt).toLocaleDateString()}</time>
      </p>
      {pin.uncertaintyM !== null && <p>±{pin.uncertaintyM.toFixed(1)} m</p>}
      <a href={`/view.html#${encodeURIComponent(pin.siteId)}`}>Scan</a>{" "}
      <button type="button" aria-label="Close answer" onClick={() => setSelected(null)}>
        ×
      </button>
    </aside>
  );
}
