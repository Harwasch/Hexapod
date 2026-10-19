import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Minus, Plus } from "lucide-react";

import type { components, LandEvidence } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, ApiError, unwrap } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import { tapOrClick, touchFirst } from "@/lib/pointer";
import { useLand } from "@/state/land";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useUi } from "@/state/ui";

import { invalidateLandDerived } from "../../../landDerived";
import { lensMeta, useLens } from "../../lenses";
import { useLensLayer } from "../../useLensMap";
import {
  MAX_POINTS,
  persistAlignmentDraft,
  restoreAlignmentDraft,
  type AlignmentDraft,
} from "./alignmentDraft";
import { alignmentDraftKey, shortTitle, yearOf } from "./recordsModel";
import { useRecords } from "./recordsStore";
import { AlignmentFit, RecordsSavedAlignments } from "./RecordsSavedAlignments";
import { RecordsThenNow } from "./RecordsThenNow";
import { useArchiveImage } from "./useArchiveImage";

type Fit = components["schemas"]["ImageRegistrationResult"];
type Request = components["schemas"]["ImageRegistrationCreate"];
type Snapshot = components["schemas"]["ArchiveImageRead"];
interface Pixel {
  x: string;
  y: string;
}

const MIN_POINTS = 4;
const ZOOMS = [1, 1.5, 2, 3, 4];

/**
 * Lines a historical map up with the land: the person matches spots on the old map to the
 * same spots on today's map, checks how well they agree, and saves an overlay. Unfinished
 * matches are kept in this browser per map image (`…:alignment:<evidenceId>:<sha256>`).
 */
export function RecordsAlignment({ evidence }: { evidence: LandEvidence }) {
  const scope = useLandScope();
  const image = useArchiveImage(evidence.id, Boolean(evidence.media));
  const media = evidence.media;
  if (!media) return <p className="land-muted">This source has no image to line up.</p>;
  if (image.statusError)
    return (
      <p className="land-alert" role="alert">
        The saved map could not be loaded.{" "}
        <button type="button" className="land-link" onClick={image.retry}>
          Try again
        </button>
      </p>
    );
  if (!image.metadata)
    return image.statusPending ? (
      <p className="land-muted" role="status">
        Loading the saved map…
      </p>
    ) : (
      <div className="land-lens-empty">
        <p className="land-lede">Save a copy of this map to the workspace to line it up.</p>
        {image.canEdit ? (
          <div className="land-row">
            <GlassButton
              variant="primary"
              size="sm"
              loading={image.busy}
              onClick={() => void image.capture()}
            >
              Save a copy
            </GlassButton>
          </div>
        ) : (
          <p className="land-muted">An editor can save a copy.</p>
        )}
        {image.error && (
          <p className="land-alert" role="alert">
            {image.error}
          </p>
        )}
      </div>
    );
  if (!image.url)
    return (
      <p className="land-muted" role="status">
        {image.error ?? "Loading the saved map…"}
      </p>
    );
  return (
    <AlignmentEditor
      key={`${scope}:${evidence.id}:${image.metadata.sha256}`}
      id={evidence.id}
      scope={scope}
      image={image.metadata}
      url={image.url}
      title={media.title}
      year={yearOf(media.sourceDate)}
      attribution={evidence.attribution}
    />
  );
}

function AlignmentEditor({
  id,
  scope,
  image,
  url,
  title,
  year,
  attribution,
}: {
  id: string;
  scope: string;
  image: Snapshot;
  url: string;
  title: string;
  year: number | null;
  attribution: string;
}) {
  const scene = useScene();
  const ready = useLandAccessReady();
  const canEdit = useLandCanEdit();
  const cache = useQueryClient();
  const key = alignmentDraftKey(scope, id, image.sha256);
  const label = `${year ? `${year} · ` : ""}${shortTitle(title)}`;
  const [draft, setDraft] = useState(() => restoreAlignmentDraft(key, title));
  const [pixel, setPixel] = useState<Pixel>({ x: "", y: "" });
  const [location, setLocation] = useState({ longitude: "", latitude: "" });
  const [editing, setEditing] = useState<string | null>(null);
  const [result, setResult] = useState<Fit | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState(false);
  const [saved, setSaved] = useState(false);
  const [zoom, setZoom] = useState(1);
  const lens = useLens((s) => s.lens);
  const panel = useUi((s) => s.activePanel);
  const picker = useLandContext((s) => s.pointPicker);
  const visible = lens === "records" && panel === "land" && ready;
  const picking = picker === id;
  const mounted = useRef(true);
  const work = useRef<AbortController | null>(null);
  const pickTicket = useRef(0);
  const pickButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      work.current?.abort();
    };
  }, []);
  // A pick belongs to this lens on the open Land panel; leaving either cancels it.
  useEffect(() => {
    if (!visible) return;
    return () => {
      pickTicket.current += 1;
      if (useLandContext.getState().pointPicker === id) {
        useLandContext.getState().setPointPicker(null);
        scene?.areas.cancelPick();
      }
    };
  }, [visible, id, scene]);
  const controls = useMemo<LandContextLayer | null>(
    () =>
      visible && draft.points.length
        ? {
            id: "records-alignment",
            title: "Alignment matches",
            style: { color: lensMeta("records").color },
            features: draft.points.map((point) => ({
              id: point.label,
              label: point.label,
              geometry: { type: "Point", coordinates: [point.longitude, point.latitude] },
            })),
          }
        : null,
    [visible, draft.points],
  );
  useLensLayer(controls);

  const update = (change: Partial<AlignmentDraft>) => {
    work.current?.abort();
    setBusy(false);
    setResult(null);
    setSaved(false);
    setError(null);
    const next = { ...draft, ...change, requestKey: crypto.randomUUID() };
    setDraft(next);
    setStorageError(!persistAlignmentDraft(key, next, title));
  };
  const pixelValid = (at: Pixel) =>
    at.x.trim() !== "" &&
    at.y.trim() !== "" &&
    Number.isFinite(+at.x) &&
    Number.isFinite(+at.y) &&
    +at.x >= 0 &&
    +at.x <= image.width &&
    +at.y >= 0 &&
    +at.y <= image.height;
  const validPixel = pixelValid(pixel);
  const addPoint = (longitude: number, latitude: number, at: Pixel = pixel) => {
    if (
      !pixelValid(at) ||
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      Math.abs(longitude) > 180 ||
      Math.abs(latitude) > 90
    )
      return;
    const retained = draft.points.filter((point) => point.label !== editing);
    if (retained.length >= MAX_POINTS) {
      setError(`Use at most ${MAX_POINTS} matches.`);
      return;
    }
    if (
      retained.some(
        (p) =>
          (p.imageX === +at.x && p.imageY === +at.y) ||
          (p.longitude === longitude && p.latitude === latitude),
      )
    ) {
      setError("Each match needs its own spot on the old map and on the land.");
      return;
    }
    let n = 1;
    while (retained.some((p) => p.label === `Point ${n}`)) n++;
    const point = {
      label: editing ?? `Point ${n}`,
      imageX: +at.x,
      imageY: +at.y,
      longitude,
      latitude,
    };
    update({
      points: editing
        ? draft.points.map((p) => (p.label === editing ? point : p))
        : [...draft.points, point],
    });
    setPixel({ x: "", y: "" });
    setLocation({ longitude: "", latitude: "" });
    setEditing(null);
  };
  const pick = async (at: Pixel) => {
    if (!scene || !visible || !pixelValid(at)) return;
    const ticket = ++pickTicket.current;
    useUi.getState().setMeasureMode(null);
    useUi.getState().setExploreMode(false);
    // AreaEditor cancels any earlier one-shot pick before taking ownership.
    const promise = scene.areas.pickGround();
    useLandContext.getState().setPointPicker(id);
    try {
      const point = await promise;
      if (!mounted.current || ticket !== pickTicket.current) return;
      if (point) addPoint(point.longitude, point.latitude, at);
    } finally {
      if (useLandContext.getState().pointPicker === id && ticket === pickTicket.current) {
        useLandContext.getState().setPointPicker(null);
        pickButton.current?.focus();
      }
    }
  };
  const execute = async (save: boolean) => {
    const controller = new AbortController();
    work.current?.abort();
    work.current = controller;
    setBusy(true);
    setError(null);
    setSaved(false);
    const body: Request = { ...draft, imageSha256: image.sha256 };
    try {
      if (save) {
        const response = await unwrap(
          api.POST("/api/v1/research/evidence/{evidence_id}/image/registrations", {
            params: { path: { evidence_id: id } },
            body,
            signal: controller.signal,
          }),
        );
        if (controller.signal.aborted) return;
        setResult(response.result);
        setSaved(true);
        useRecords.getState().showOverlay({
          id: response.id,
          label,
          bounds: response.result.bounds,
          attribution,
        });
        void cache.invalidateQueries({ queryKey: ["land-image-alignments", scope, id] });
        // An aligned map is dated evidence with a footprint: the fog over the land is asked again.
        void invalidateLandDerived(cache, useLand.getState().active?.id ?? null);
      } else {
        const response = await unwrap(
          api.POST("/api/v1/research/evidence/{evidence_id}/image/registrations/preview", {
            params: { path: { evidence_id: id } },
            body,
            signal: controller.signal,
          }),
        );
        if (!controller.signal.aborted) setResult(response);
      }
    } catch (cause) {
      if (!controller.signal.aborted)
        setError(
          cause instanceof ApiError && cause.fieldErrors.length
            ? cause.fieldErrors.join(". ")
            : describeError(cause),
        );
    } finally {
      if (mounted.current && !controller.signal.aborted) setBusy(false);
    }
  };
  const valid = draft.points.length >= MIN_POINTS && draft.name.trim().length > 0 && !editing;
  // A finger has no Escape key: on a touch screen the way out is the Cancel map pick button.
  const status = picking
    ? touchFirst()
      ? "Tap the matching place on the globe, or press Cancel map pick."
      : "Now click the same spot on the map. Press Escape to cancel."
    : validPixel
      ? editing
        ? `Editing ${editing}. Pick its spot on the map, or type it below.`
        : "Spot chosen. Pick the same spot on the map."
      : draft.points.length < MIN_POINTS
        ? `${draft.points.length} of ${MIN_POINTS} matches. ${tapOrClick()} a spot you can find on today's map: a road crossing, a bend in a creek, a corner.`
        : `${draft.points.length} matches. Check the fit, or add more spread across the map.`;
  const zoomIndex = ZOOMS.indexOf(zoom);

  return (
    <section className="records-align" aria-label="Line up this map">
      <p className="records-align__status" aria-live="polite">
        {status}
      </p>
      <div className="records-align__zoom">
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Zoom out of the old map"
          disabled={zoomIndex <= 0}
          onClick={() => setZoom(ZOOMS[Math.max(0, zoomIndex - 1)] ?? 1)}
        >
          <Minus size={14} />
        </GlassButton>
        <span className="land-muted">{Math.round(zoom * 100)}%</span>
        <GlassButton
          iconOnly
          size="sm"
          variant="ghost"
          aria-label="Zoom into the old map"
          disabled={zoomIndex >= ZOOMS.length - 1}
          onClick={() => setZoom(ZOOMS[Math.min(ZOOMS.length - 1, zoomIndex + 1)] ?? 1)}
        >
          <Plus size={14} />
        </GlassButton>
      </div>
      <div className="records-align__viewport">
        <button
          type="button"
          className="records-align__image"
          style={{ width: `${zoom * 100}%` }}
          aria-label="Choose a spot on the old map"
          disabled={picking}
          onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const keyboard = event.detail === 0 || !box.width || !box.height;
            const next = keyboard
              ? { x: String(image.width / 2), y: String(image.height / 2) }
              : {
                  x: (((event.clientX - box.left) / box.width) * image.width).toFixed(2),
                  y: (((event.clientY - box.top) / box.height) * image.height).toFixed(2),
                };
            setPixel(next);
            // A pointer click goes straight on to the map; the keyboard path uses the button.
            if (!keyboard) void pick(next);
          }}
        >
          <img src={url} alt={title} draggable={false} />
          {draft.points.map((p) => (
            <span
              key={p.label}
              className="records-align__marker"
              style={{
                left: `${(p.imageX / image.width) * 100}%`,
                top: `${(p.imageY / image.height) * 100}%`,
              }}
              aria-hidden="true"
            >
              {p.label.replace("Point ", "")}
            </span>
          ))}
          {validPixel && (
            <span
              className="records-align__marker is-pending"
              style={{
                left: `${(+pixel.x / image.width) * 100}%`,
                top: `${(+pixel.y / image.height) * 100}%`,
              }}
              aria-hidden="true"
            >
              +
            </span>
          )}
        </button>
      </div>
      <div className="land-row">
        <GlassButton
          ref={pickButton}
          size="sm"
          disabled={!validPixel || picking || !scene || !visible}
          onClick={() => void pick(pixel)}
        >
          Pick matching map point
        </GlassButton>
        {picking && (
          <GlassButton size="sm" variant="ghost" onClick={() => scene?.areas.cancelPick()}>
            Cancel map pick
          </GlassButton>
        )}
        {editing && (
          <GlassButton
            size="sm"
            variant="ghost"
            disabled={picking}
            onClick={() => {
              setEditing(null);
              setPixel({ x: "", y: "" });
              setLocation({ longitude: "", latitude: "" });
            }}
          >
            Cancel point edit
          </GlassButton>
        )}
      </div>
      <details className="records-more">
        <summary>Type positions instead</summary>
        <fieldset className="records-grid" disabled={picking}>
          <legend className="land-visually-hidden">
            {editing ? `Edit ${editing}` : `New match (${draft.points.length} of ${MAX_POINTS})`}
          </legend>
          <label className="land-field">
            Image X
            <input
              className="glass-input"
              type="number"
              min={0}
              max={image.width}
              step="any"
              value={pixel.x}
              onChange={(e) => setPixel({ ...pixel, x: e.target.value })}
            />
          </label>
          <label className="land-field">
            Image Y
            <input
              className="glass-input"
              type="number"
              min={0}
              max={image.height}
              step="any"
              value={pixel.y}
              onChange={(e) => setPixel({ ...pixel, y: e.target.value })}
            />
          </label>
          <label className="land-field">
            Longitude
            <input
              className="glass-input"
              type="number"
              min={-180}
              max={180}
              step="any"
              value={location.longitude}
              onChange={(e) => setLocation({ ...location, longitude: e.target.value })}
            />
          </label>
          <label className="land-field">
            Latitude
            <input
              className="glass-input"
              type="number"
              min={-90}
              max={90}
              step="any"
              value={location.latitude}
              onChange={(e) => setLocation({ ...location, latitude: e.target.value })}
            />
          </label>
        </fieldset>
        <p className="land-muted">
          Pixels count from the top left of the saved {image.width} × {image.height} image.
          Coordinates are WGS84 degrees.
        </p>
        <div className="land-row">
          <GlassButton
            size="sm"
            disabled={
              !validPixel ||
              picking ||
              !location.longitude.trim() ||
              !location.latitude.trim() ||
              !Number.isFinite(+location.longitude) ||
              !Number.isFinite(+location.latitude) ||
              Math.abs(+location.longitude) > 180 ||
              Math.abs(+location.latitude) > 90
            }
            onClick={() => addPoint(+location.longitude, +location.latitude)}
          >
            {editing ? "Update point match" : "Add point match"}
          </GlassButton>
        </div>
      </details>
      {draft.points.length > 0 && (
        <ol className="records-matches" aria-label="Point matches">
          {draft.points.map((point) => (
            <li key={point.label}>
              <span className="records-matches__n" aria-hidden="true">
                {point.label.replace("Point ", "")}
              </span>
              <span className="records-matches__where">
                {point.latitude.toFixed(5)}°, {point.longitude.toFixed(5)}°
              </span>
              <button
                type="button"
                className="land-link"
                disabled={picking}
                onClick={() => {
                  setEditing(point.label);
                  setPixel({ x: String(point.imageX), y: String(point.imageY) });
                  setLocation({
                    longitude: String(point.longitude),
                    latitude: String(point.latitude),
                  });
                }}
              >
                <span className="land-visually-hidden">Edit {point.label}</span>
                <span aria-hidden="true">Edit</span>
              </button>
              <button
                type="button"
                className="land-link"
                disabled={picking}
                onClick={() => {
                  update({ points: draft.points.filter((p) => p.label !== point.label) });
                  if (editing === point.label) setEditing(null);
                }}
              >
                <span className="land-visually-hidden">Remove {point.label}</span>
                <span aria-hidden="true">Remove</span>
              </button>
            </li>
          ))}
        </ol>
      )}
      <details className="records-more">
        <summary>Name and notes</summary>
        <label className="land-field">
          Alignment name
          <input
            className="glass-input"
            disabled={picking}
            value={draft.name}
            maxLength={200}
            onChange={(e) => update({ name: e.target.value })}
          />
        </label>
        <label className="land-field">
          Alignment notes
          <textarea
            className="glass-input"
            disabled={picking}
            value={draft.notes}
            maxLength={2000}
            placeholder="Which places you matched, what looked uncertain or stretched"
            onChange={(e) => update({ notes: e.target.value })}
          />
        </label>
      </details>
      {storageError && (
        <p className="land-note">
          This browser could not keep your unfinished matches. Save before you leave.
        </p>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          disabled={!valid || busy || picking || !ready}
          onClick={() => void execute(false)}
        >
          {busy && !result ? "Checking…" : "Check alignment"}
        </GlassButton>
        <GlassButton
          size="sm"
          variant="primary"
          disabled={!valid || !result || busy || picking || !ready || !canEdit || saved}
          onClick={() => void execute(true)}
        >
          Save and show on the land
        </GlassButton>
      </div>
      {!canEdit && <p className="land-muted">An editor can save this alignment.</p>}
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      {saved && (
        <p className="land-muted" role="status">
          Saved. The old map is on the land; earlier alignments are kept.
        </p>
      )}
      {result && <AlignmentFit result={result} />}
      <RecordsThenNow />
      <RecordsSavedAlignments
        evidenceId={id}
        label={label}
        attribution={attribution}
        refiningDisabled={picking}
        onRefine={(row) => {
          update({
            name: row.request.name,
            notes: row.request.notes ?? "",
            points: row.request.points,
          });
          setEditing(null);
          setPixel({ x: "", y: "" });
        }}
      />
    </section>
  );
}
