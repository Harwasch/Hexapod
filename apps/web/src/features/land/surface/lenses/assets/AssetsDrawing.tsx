import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Check, Undo2 } from "lucide-react";

import type { LandMapGeometry } from "@twin/contracts";
import { footprintAreaM2, formatLength } from "@twin/geo";
import { GlassButton } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { tapOrClick } from "@/lib/pointer";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useSettings } from "@/state/settings";

import { areaLabel } from "../../landFlow";
import { useLensLayer } from "../../useLensMap";
import type { Vertex } from "./assetsGeometry";
import { lineLengthM, plural, shapeOf, type ShapeKind } from "./assetsModel";

const MAX_POINTS = 2000;
const STYLE = { color: "#f1cb7a", fill: 0.2, width: 3 };

/**
 * Drawing an asset on the map: each tap or click on the ground adds a point (one for a point
 * asset), the shape so far is drawn on the map, and Undo, Done and Cancel sit with the count and
 * the live length or area. Escape pauses; "Keep drawing" picks up again.
 */
export function AssetsDrawing({
  owner,
  kind,
  what,
  enabled,
  busy,
  error,
  onDone,
  onCancel,
}: {
  /** The point-picker owner, so another tool's pick is never taken over. */
  owner: string;
  kind: ShapeKind;
  /** What is being drawn, lower case ("the well"), for the instructions. */
  what: string;
  /** False while the land panel is closed: picking stops. */
  enabled: boolean;
  busy: boolean;
  error: string | null;
  onDone: (geometry: LandMapGeometry) => void;
  onCancel: () => void;
}) {
  const scene = useScene();
  const units = useSettings((s) => s.units);
  const [points, setPoints] = useState<Vertex[]>([]);
  const picking = useLandContext((s) => s.pointPicker === owner);
  const pointsRef = useRef<Vertex[]>([]);
  const ticket = useRef(0);
  const done = useRef(onDone);
  useEffect(() => {
    done.current = onDone;
  });

  const release = () => {
    ticket.current++;
    if (useLandContext.getState().pointPicker === owner) {
      scene?.areas.cancelPick();
      useLandContext.getState().setPointPicker(null);
    }
  };

  const pick = async () => {
    if (!scene || !enabled) return;
    const current = ++ticket.current;
    for (;;) {
      useLandContext.getState().setPointPicker(owner);
      const point = await scene.areas.pickGround();
      if (current !== ticket.current || useLandContext.getState().pointPicker !== owner) return;
      if (!point) {
        useLandContext.getState().setPointPicker(null);
        return;
      }
      const value: Vertex = [point.longitude, point.latitude];
      const next = kind === "Point" ? [value] : [...pointsRef.current, value];
      pointsRef.current = next;
      setPoints(next);
      if (kind === "Point" || next.length >= MAX_POINTS) {
        useLandContext.getState().setPointPicker(null);
        if (kind === "Point") done.current({ type: "Point", coordinates: value });
        return;
      }
    }
  };

  // Start picking straight away (before paint, so the card opens already listening); leaving
  // the lens or closing the panel ends the pick.
  useLayoutEffect(() => {
    if (!enabled) return;
    void pick();
    return release;
    // `pick` and `release` read refs only; restarting on their identity would re-pick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, owner, scene]);

  const layer = useMemo<LandContextLayer | null>(() => {
    if (!points.length) return null;
    const ring = kind === "Polygon" && points.length >= 3 ? [...points, points[0] ?? []] : points;
    return {
      id: "assets-drawing",
      title: "Asset being drawn",
      style: STYLE,
      features: [
        ...(ring.length >= 2
          ? [
              {
                id: "path",
                label: "Shape so far",
                geometry: { type: "LineString" as const, coordinates: ring },
              },
            ]
          : []),
        ...points.map((coordinates, index) => ({
          id: `v${index}`,
          label: `Point ${index + 1}`,
          geometry: { type: "Point" as const, coordinates },
        })),
      ],
    };
  }, [points, kind]);
  useLensLayer(layer);

  const shape = shapeOf(kind, points);
  const measure =
    kind === "LineString" && points.length >= 2
      ? formatLength(lineLengthM(points), units)
      : kind === "Polygon" && shape?.type === "Polygon"
        ? areaLabel(footprintAreaM2(shape))
        : null;
  const status =
    kind === "Point"
      ? points.length && busy
        ? "Checking the spot…"
        : `${tapOrClick()} the map where ${what} is.`
      : !points.length
        ? kind === "LineString"
          ? `${tapOrClick()} along ${what} on the map, then Done.`
          : `${tapOrClick()} the corners of ${what} on the map, then Done.`
        : [
            plural(points.length, kind === "LineString" ? "point" : "corner"),
            measure,
            !picking && "paused",
          ]
            .filter(Boolean)
            .join(" · ");

  const undo = () => {
    const next = pointsRef.current.slice(0, -1);
    pointsRef.current = next;
    setPoints(next);
  };

  return (
    <div className="assets-draw" data-testid="assets-drawing">
      <p className="assets-draw__status" aria-live="polite">
        {status}
      </p>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      <div className="land-row">
        {kind !== "Point" && (
          <GlassButton
            size="sm"
            disabled={!points.length || busy}
            leadingIcon={<Undo2 size={15} aria-hidden="true" />}
            onClick={undo}
          >
            Undo
          </GlassButton>
        )}
        {!picking && !busy && (
          <GlassButton size="sm" disabled={!scene || !enabled} onClick={() => void pick()}>
            {kind === "Point" && points.length ? "Pick again" : "Keep drawing"}
          </GlassButton>
        )}
        {kind !== "Point" && (
          <GlassButton
            size="sm"
            variant="primary"
            disabled={!shape || busy}
            loading={busy}
            leadingIcon={<Check size={15} aria-hidden="true" />}
            onClick={() => {
              if (!shape) return;
              release();
              onDone(shape);
            }}
          >
            Done
          </GlassButton>
        )}
        <GlassButton
          size="sm"
          variant="ghost"
          onClick={() => {
            release();
            onCancel();
          }}
        >
          Cancel
        </GlassButton>
      </div>
    </div>
  );
}
