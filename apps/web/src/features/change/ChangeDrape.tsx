import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { ChangeMap } from "@twin/contracts";
import { GlassButton, GlassPanel, GlassSwitch } from "@twin/ui";

import { ApiError, api, unwrap } from "@/api/client";
import { useSite } from "@/api/queries";
import { ChangeDrapeLayer } from "@/cesium/changeDrape";
import { useScene } from "@/cesium/SceneContext";
import {
  calibrationNote,
  CHANGE_COLORS,
  changeError,
  changeLabel,
  changeOffer,
  legendRows,
  NEEDS_TOKEN,
  detectableNote,
  measuredLimit,
  SMALL_OBJECTS,
  vegetationNote,
  verdictNote,
  VEGETATION_TITLE,
  type ChangeLayer,
  type ChangeLayerName,
} from "@/lib/changeDrape";
import { formatEpochDate, siteEpochs } from "@/lib/epochs";
import { useChangeDrape, type ChangeDrapeRequest } from "@/state/changeDrape";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useUi } from "@/state/ui";

const cells = (n: number) => `${n.toLocaleString("en-US")} ${n === 1 ? "cell" : "cells"}`;

/** One layer's key: each class's cells, its low-confidence ones (hatched), unchanged counted. */
function LayerRows({ layer, testId }: { layer: ChangeLayer; testId: string }) {
  const colors = CHANGE_COLORS[layer.layer];
  return (
    <ul className="change-drape__rows">
      {legendRows(layer).map((row) => (
        <li key={row.key} data-testid={`${testId}-${row.key}`}>
          <span
            className="change-drape__swatch"
            style={row.key === "unchanged" ? undefined : { background: colors[row.key] }}
            aria-hidden="true"
          />
          <span className="change-drape__name">{row.label}</span>
          <span>
            {cells(row.cells)}
            {!row.drawn && " · counted, not drawn"}
          </span>
          {row.lowConfidence > 0 && (
            <span className="change-drape__low">
              <span
                className="change-drape__swatch change-drape__swatch--hatched"
                style={row.key === "unchanged" ? undefined : { color: colors[row.key] }}
                aria-hidden="true"
              />
              {row.lowConfidence.toLocaleString("en-US")} low confidence
              {row.drawn ? ", hatched" : ""}
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

/** What a layer could see between these dates, what it cannot see yet, and its measured best. */
function LayerLimits({ layer, testId }: { layer: ChangeLayer; testId: string }) {
  return (
    <ul className="change-drape__notes" data-testid={testId}>
      <li>{detectableNote(layer)}</li>
      <li>{SMALL_OBJECTS}</li>
      <li>{measuredLimit(layer.layer)}</li>
    </ul>
  );
}

/** Where a layer was outlined a level up (its map too large to outline), saying so. */
function outlinedNote(map: ChangeMap, layer: ChangeLayer) {
  return layer.areaLevel > map.level ? (
    <p className="change-drape__note">
      Outlined at {layer.areaCellM} m cells; counted at {map.cellM} m.
    </p>
  ) : null;
}

/** One layer of the map on the ground and over the scan, while `on` and the map is in. */
function useLayerDrawing(name: ChangeLayerName, layer: ChangeLayer | undefined, on: boolean) {
  const scene = useScene();
  const drawing = useRef<ChangeDrapeLayer | null>(null);
  useEffect(() => {
    if (!scene || !on) return;
    const made = new ChangeDrapeLayer(scene.scene, { layer: name });
    drawing.current = made;
    return () => {
      made.destroy();
      drawing.current = null;
    };
  }, [scene, name, on]);
  useEffect(() => {
    if (layer) drawing.current?.show(layer);
    else drawing.current?.clear();
  }, [layer, scene, on]);
}

/** Ends the drape: nothing asked for, nothing refused. */
function endChangeDrape(): void {
  useChangeDrape.setState({ request: null, error: null });
}

/**
 * A short fingerprint of the write token for the query key, so a token entered or changed asks
 * again (even while a request without it is in flight) without the token itself in the key.
 */
function tokenKey(token: string): string {
  let hash = 5381;
  for (let i = 0; i < token.length; i++) hash = ((hash * 33) ^ token.charCodeAt(i)) >>> 0;
  return token ? hash.toString(36) : "";
}

function useChangeMap(request: ChangeDrapeRequest, token: string, enabled: boolean) {
  return useQuery({
    queryKey: ["change-map", request.siteId, request.from, request.to, tokenKey(token)],
    queryFn: () =>
      unwrap<ChangeMap>(
        api.GET("/api/v1/sites/{site_id}/change", {
          params: {
            path: { site_id: request.siteId },
            query: { from: request.from, to: request.to },
          },
        }),
      ),
    enabled,
    retry: false,
    // Derived per request from stored maps that do not change under a revision: kept a while.
    staleTime: 5 * 60_000,
  });
}

function refusal(error: unknown): string {
  return error instanceof ApiError
    ? changeError(error.status, error.problem?.code)
    : changeError(0);
}

/**
 * The change map between two dates of the active site on the ground (cesium/changeDrape.ts),
 * and its legend. The ground layer, the primary product, is drawn and keyed; vegetation (the
 * top surface, at its own higher threshold, never ground change) is a layer apart, behind its
 * own switch, keyed in its own section and never counted in the ground's. Each key says what
 * each colour is and how many cells, which are low confidence (hatched), that it is derived
 * from height surfaces and not observed, the least change that layer can show between those
 * dates, and what is uncalibrated; the legend says when either date is not survey-grade.
 * Asked for from Compare › Dates (`useChangeDrape`); mounted by the shell (AppShell) while asked.
 *
 * It ends with its site (another becomes the active one) or from its own Hide. A date that
 * no longer lists a surface, or a refusal (the write token, a 409), is said here and in
 * Compare, with nothing drawn: never a blank drape. Ground geometry: drawn under every splat renderer, beside the date swipe and a
 * per-cell view rule alike, which draw splats and leave the ground to it.
 */
export function ChangeDrape() {
  const request = useChangeDrape((s) => s.request);
  return request ? <Drape request={request} /> : null;
}

function Drape({ request }: { request: ChangeDrapeRequest }) {
  const activeSiteId = useSites((s) => s.activeSiteId);
  const site = useSite(request.siteId).data;
  const token = useSettings((s) => s.writeToken.trim());
  const epochs = siteEpochs(site);
  const offer = site ? changeOffer(epochs, request.from, request.to) : null;
  const query = useChangeMap(request, token, offer?.offered === true);
  const map = offer?.offered ? query.data : undefined;
  // Vegetation is apart from the ground and off until asked for, each time the map is drawn.
  const [vegetation, setVegetation] = useState(false);
  const error = offer && !offer.offered ? offer.why : query.isError ? refusal(query.error) : null;

  // Its site's: another site taking the globe ends it.
  useEffect(() => {
    if (activeSiteId !== request.siteId) endChangeDrape();
  }, [activeSiteId, request.siteId]);

  // The ground, the primary product, always; vegetation only while its switch is on.
  useLayerDrawing("ground", map?.ground, true);
  useLayerDrawing("vegetation", map?.vegetation, vegetation);

  // Compare says why too, from the same words.
  useEffect(() => {
    useChangeDrape.setState({ error });
  }, [error]);

  const needsToken = error === NEEDS_TOKEN;

  const byId = (id: string) => epochs.find((e) => e.id === id);
  const [from, to] = [byId(request.from), byId(request.to)];
  const verdicts =
    map && from && to
      ? verdictNote(
          map.uncertainty,
          from,
          to,
          map.relativeFit,
          map.relativeFit ? byId(map.relativeFit.referenceEpochId) : undefined,
        )
      : null;
  const dates =
    from && to ? `${formatEpochDate(from.observedAt)} → ${formatEpochDate(to.observedAt)}` : "";

  return (
    <GlassPanel
      strong
      padding="sm"
      className="change-drape"
      role="region"
      aria-label="Change map"
      data-testid="change-drape"
    >
      <div className="change-drape__head">
        <p className="glass-eyebrow">Change · {dates}</p>
        <GlassButton
          size="sm"
          variant="ghost"
          iconOnly
          aria-label="Hide change map"
          onClick={endChangeDrape}
        >
          <X size={14} aria-hidden="true" />
        </GlassButton>
      </div>
      {error ? (
        <div className="change-drape__state" role="alert" data-testid="change-drape-error">
          <span>{error}</span>
          {needsToken && (
            <GlassButton
              size="sm"
              variant="ghost"
              onClick={() => useUi.getState().setSettingsOpen(true)}
            >
              Open Settings
            </GlassButton>
          )}
        </div>
      ) : !map ? (
        <p className="change-drape__state">Loading the change map…</p>
      ) : (
        <>
          <p className="glass-eyebrow">Ground</p>
          <p className="change-drape__label" data-testid="change-drape-label">
            {changeLabel(map.ground)}
          </p>
          <LayerRows layer={map.ground} testId="change-row" />
          <LayerLimits layer={map.ground} testId="change-detectable" />
          {verdicts && (
            <p className="change-drape__note" data-testid="change-drape-placement">
              {verdicts}
            </p>
          )}
          <p className="change-drape__note">{calibrationNote(map, from && to ? [from, to] : [])}</p>
          {outlinedNote(map, map.ground)}
          <div className="setting change-drape__layer">
            <span className="setting__label" id="change-vegetation-switch-label">
              {VEGETATION_TITLE}
            </span>
            <GlassSwitch
              checked={vegetation}
              onCheckedChange={setVegetation}
              aria-labelledby="change-vegetation-switch-label"
            />
          </div>
          {vegetation && (
            <section
              className="change-drape__vegetation"
              aria-label="Vegetation change"
              data-testid="change-vegetation"
            >
              <p className="change-drape__label" data-testid="change-vegetation-label">
                {changeLabel(map.vegetation)}
              </p>
              <LayerRows layer={map.vegetation} testId="change-vegetation-row" />
              <LayerLimits layer={map.vegetation} testId="change-vegetation-detectable" />
              <p className="change-drape__note">{vegetationNote(map, map.vegetation)}</p>
              {outlinedNote(map, map.vegetation)}
            </section>
          )}
          <p className="change-drape__note">
            Appearance-only change: not computed ({map.appearance.reason}).
          </p>
        </>
      )}
    </GlassPanel>
  );
}
