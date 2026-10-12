import { useEffect, useMemo, useRef } from "react";

import type { Footprint } from "@twin/contracts";

import { useScene } from "@/cesium/SceneContext";
import type { LandContextFeature, LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";

import { anchorOf } from "../../geometryAnchor";
import { useFlyTo } from "../../mapHooks";
import { useLensLayer, useLensPins } from "../../useLensMap";
import { percent, plotTopSpecies, type EcologyContext } from "./ecologyData";
import type { SurveyDraft, SurveyRead } from "./useEcologySurveys";

/** The lens's greens: every saved plot quietly, the selected survey bright, a draft in progress. */
export const ECOLOGY_COLORS = {
  surveys: "#93d37f",
  selected: "#d7f7bd",
  draft: "#b5e88f",
  sketch: "#ffe08a",
} as const;

const MAX_PINS = 12;

/**
 * Plots are ground-clamped shapes that Cesium builds over several frames. The scene renders
 * only on request, so after the layers change keep asking for frames until every shape is
 * built (at most a few seconds); otherwise a plot appears one change late.
 */
function useRepaintUntilDrawn(layers: (LandContextLayer | null)[]) {
  const scene = useScene();
  const [others, selected, draft, sketch] = layers;
  useEffect(() => {
    if (!scene) return;
    let frame = 0;
    let count = 0;
    const pump = () => {
      try {
        if (scene.isDestroyed) return;
        scene.scene.requestRender();
        count += 1;
        if (count < 300 && (count < 10 || !scene.viewer.dataSourceDisplay.ready))
          frame = requestAnimationFrame(pump);
      } catch {
        /* A scene without a viewer (a test double) has nothing to repaint. */
      }
    };
    frame = requestAnimationFrame(pump);
    return () => cancelAnimationFrame(frame);
  }, [scene, others, selected, draft, sketch]);
}

/**
 * A short key for a shape. The map bridge redraws a layer by removing and re-adding entities
 * in one batch, and Cesium keeps drawing the old shape when an id comes back; ids that change
 * with the geometry make every new shape a new entity.
 */
function shapeKey(geometry: { coordinates: unknown }): string {
  const text = JSON.stringify(geometry.coordinates);
  let hash = 5381;
  for (let i = 0; i < text.length; i++) hash = ((hash << 5) + hash + text.charCodeAt(i)) | 0;
  return (hash >>> 0).toString(36);
}

function plotFeatures(prefix: string, plots: { label: string; boundary: Footprint }[]) {
  return plots.map((plot, index): LandContextFeature => ({
    id: `${prefix}${index}-${shapeKey(plot.boundary)}`,
    label: plot.label,
    geometry: plot.boundary,
  }));
}

/**
 * What the Ecology lens draws: saved survey plots (the selected survey highlighted), the plots
 * and corners of a survey being recorded, and pins for plots, surveys and the one point the
 * soil-linked plant communities were sampled at. Species records are never placed on the map;
 * their positions are withheld to protect sensitive species.
 */
export function useEcologyMap({
  context,
  surveys,
  selected,
  draft,
  corners,
  onSelectSurvey,
}: {
  context: EcologyContext | null;
  surveys: SurveyRead[];
  selected: SurveyRead | null;
  draft: SurveyDraft | null;
  corners: number[][];
  onSelectSurvey: (id: string) => void;
}) {
  const fly = useFlyTo();
  const flyRef = useRef(fly);
  const selectRef = useRef(onSelectSurvey);
  useEffect(() => {
    flyRef.current = fly;
    selectRef.current = onSelectSurvey;
  });

  const recording = Boolean(draft);
  const others = useMemo(
    () => (recording ? [] : surveys.filter((survey) => survey.id !== selected?.id)),
    [recording, surveys, selected?.id],
  );

  const othersLayer = useMemo<LandContextLayer | null>(
    () =>
      others.length
        ? {
            id: "ecology-surveys",
            title: "Field survey plots",
            features: others.flatMap((survey) =>
              plotFeatures(`${survey.id}:`, survey.plots).map((feature) => ({
                ...feature,
                label: `${survey.name} · ${feature.label}`,
              })),
            ),
            style: { color: ECOLOGY_COLORS.surveys, fill: 0.12, width: 2 },
          }
        : null,
    [others],
  );
  const selectedLayer = useMemo<LandContextLayer | null>(
    () =>
      selected && !recording
        ? {
            id: "ecology-survey",
            title: selected.name,
            features: plotFeatures(`${selected.id}:`, selected.plots),
            style: { color: ECOLOGY_COLORS.selected, fill: 0.34, width: 3 },
          }
        : null,
    [selected, recording],
  );
  const draftLayer = useMemo<LandContextLayer | null>(
    () =>
      draft?.plots.length
        ? {
            id: "ecology-draft",
            title: "Survey plots being recorded",
            features: plotFeatures("plot-", draft.plots),
            style: { color: ECOLOGY_COLORS.draft, fill: 0.24, width: 2.5 },
          }
        : null,
    [draft?.plots],
  );
  const sketchLayer = useMemo<LandContextLayer | null>(() => {
    if (!draft || !corners.length) return null;
    const first = corners[0] ?? [0, 0];
    const outline: LandContextFeature[] =
      corners.length >= 3
        ? [
            {
              id: `outline-${shapeKey({ coordinates: corners })}`,
              label: "Plot being drawn",
              geometry: { type: "Polygon", coordinates: [[...corners, first]] },
            },
          ]
        : corners.length === 2
          ? [
              {
                id: `line-${shapeKey({ coordinates: corners })}`,
                label: "Plot being drawn",
                geometry: { type: "LineString", coordinates: corners },
              },
            ]
          : [];
    return {
      id: "ecology-sketch",
      title: "Plot corners",
      features: [
        ...outline,
        ...corners.map((corner, index) => ({
          id: `corner-${index}-${shapeKey({ coordinates: corner })}`,
          label: `Corner ${index + 1}`,
          geometry: { type: "Point" as const, coordinates: corner },
        })),
      ],
      style: { color: ECOLOGY_COLORS.sketch, fill: 0.16, width: 2 },
    };
  }, [draft, corners]);

  useLensLayer(othersLayer);
  useLensLayer(selectedLayer);
  useLensLayer(draftLayer);
  useLensLayer(sketchLayer);
  useRepaintUntilDrawn([othersLayer, selectedLayer, draftLayer, sketchLayer]);

  const samplePoint = context?.sites?.samplePoint ?? null;
  const siteName = context?.sites?.names[0] ?? null;
  const pins = useMemo<MapPin[] | null>(() => {
    const out: MapPin[] = [];
    if (draft) {
      draft.plots.forEach((plot, index) => {
        const top = plotTopSpecies(plot, draft.method);
        out.push({
          id: `ecology-draft-${index}`,
          position: anchorOf(plot.boundary),
          label: top ? `${plot.label} · ${top.taxon} ${percent(top.cover)}` : plot.label,
        });
      });
      return out.slice(0, MAX_PINS);
    }
    if (samplePoint)
      out.push({
        id: "ecology-sample",
        position: samplePoint,
        label: siteName ? `Soil sample · ${siteName}` : "Soil sample point",
        tone: "neutral",
      });
    if (selected)
      selected.plots.forEach((plot, index) => {
        const top = plotTopSpecies(plot, selected.method);
        out.push({
          id: `ecology-plot-${index}`,
          position: anchorOf(plot.boundary),
          label: top ? `${plot.label} · ${top.taxon} ${percent(top.cover)}` : plot.label,
          onSelect: () => flyRef.current(plot.boundary),
        });
      });
    for (const survey of others) {
      const first = survey.plots[0];
      if (!first || out.length >= MAX_PINS) break;
      out.push({
        id: `ecology-survey-${survey.id}`,
        position: anchorOf(first.boundary),
        label: survey.name,
        tone: "neutral",
        onSelect: () => selectRef.current(survey.id),
      });
    }
    return out.slice(0, MAX_PINS);
  }, [draft, samplePoint, siteName, selected, others]);
  useLensPins("ecology", pins);
}
