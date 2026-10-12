import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, Plus } from "lucide-react";

import type { LandArea, LandMapGeometry } from "@twin/contracts";
import { GlassButton } from "@twin/ui";

import { api, ApiError, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandAccessReady, useLandCanEdit, useLandScope } from "@/state/landIdentity";
import { useLandContext, type LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";
import { useUi } from "@/state/ui";

import { anchorOf } from "../geometryAnchor";
import { LensCard, LensDetails, LensEmpty } from "../LensCard";
import { useLens } from "../lenses";
import { useFlyTo } from "../mapHooks";
import { Fact } from "../mapLinks";
import { useLensLayer, useLensPins } from "../useLensMap";
import {
  ASSET_PAGE,
  CONDITION_LIMIT,
  useAssetList,
  useAssetRecord,
  useLatestInspections,
} from "./assets/assetsApi";
import { AssetsChooser } from "./assets/AssetsChooser";
import { AssetsDetail } from "./assets/AssetsDetail";
import { downloadAssetDraft } from "./assets/assetsDraft";
import { AssetsDrawing } from "./assets/AssetsDrawing";
import { AssetsEditor, type Placement } from "./assets/AssetsEditor";
import { AssetsFinder } from "./assets/AssetsFinder";
import { AssetsImport } from "./assets/AssetsImport";
import { AssetsList } from "./assets/AssetsList";
import {
  assetsHeadline,
  CATEGORIES,
  CONDITION_ORDER,
  CONDITIONS,
  nextName,
  shortLabel,
  summarizeAssets,
  whenLabel,
  type Asset,
  type AssetPreset,
  type AssetSummary,
  type ShapeKind,
} from "./assets/assetsModel";
import { useAssetDraft, type FeatureDraft } from "./assets/useAssetDraft";
import "./assets/assets.css";

/** The asset layer keeps its pre-lens id: map clicks and edge snapping look for it. */
const INVENTORY = "inventory";
const LAYER_STYLE = { color: CATEGORIES.other.color, fill: 0.22, width: 2 };
const DRAFT_STYLE = { color: "#f1cb7a", fill: 0.25, width: 3 };
const MAX_PINS = 12;

type ColorBy = "kind" | "condition";

interface Drawing {
  purpose: "new" | "redraw";
  preset: AssetPreset;
}

function kindOf(geometry: LandMapGeometry): ShapeKind {
  return geometry.type === "MultiPolygon" ? "Polygon" : geometry.type;
}

/** "the well", or the plain shape for something unnamed. */
function drawWhat(drawing: Drawing): string {
  if (drawing.purpose === "redraw") return "the asset";
  const { id, label } = drawing.preset;
  if (id === "point") return "it";
  if (id === "line") return "the line";
  if (id === "area") return "the area";
  return `the ${label.toLowerCase()}`;
}

/**
 * Like `useLensLayer`, but updates the layer in place: a changed selection then only recolours
 * the picked asset instead of rebuilding every entity. Removed when the lens closes.
 */
function useInventoryLayer(layer: LandContextLayer | null) {
  useEffect(() => {
    if (layer) useLandContext.getState().setLayer(layer);
    else useLandContext.getState().removeLayer(INVENTORY);
  }, [layer]);
  useEffect(() => () => useLandContext.getState().removeLayer(INVENTORY), []);
}

function Kinds({ summary }: { summary: AssetSummary }) {
  const top = summary.categories.slice(0, 3);
  const rest = summary.categories.length - top.length;
  return (
    <span className="assets-kinds-fact">
      {top.map(({ category, count }) => (
        <span key={category}>
          <i style={{ background: CATEGORIES[category].color }} aria-hidden="true" />
          {CATEGORIES[category].label} {count}
        </span>
      ))}
      {rest > 0 && <span className="land-muted">{rest} more</span>}
    </span>
  );
}

/** Names the first two assets needing attention: "Well 2 (poor), East fence (critical) and 3 more". */
function names(items: AssetSummary["attention"]): string {
  const shown = items
    .slice(0, 2)
    .map(
      ({ asset, inspection }) =>
        `${asset.name} (${CONDITIONS[inspection.condition].label.toLowerCase()})`,
    )
    .join(", ");
  return items.length > 2 ? `${shown} and ${items.length - 2} more` : shown;
}

export function AssetsLens({ land }: { land: LandArea }) {
  const scope = useLandScope();
  return <Assets key={`${scope}:${land.id}`} land={land} scope={scope} />;
}

/**
 * The things on the land, on the map: each asset drawn in its kind's colour (or its condition),
 * pins on the selected one and on any that need attention, and a card that says how many there
 * are and which need a look. Adding one is choosing what it is and drawing it on the map.
 */
function Assets({ land, scope }: { land: LandArea; scope: string }) {
  const canEdit = useLandCanEdit();
  const ready = useLandAccessReady();
  const panelOpen = useUi((s) => s.activePanel === "land");
  const selectedId = useLandContext((s) => s.selectedInventoryId);
  const visible = useLandContext((s) => s.inventoryVisible);
  const fly = useFlyTo();
  const [offset, setOffset] = useState(0);
  const list = useAssetList(land, offset);
  const assets = useMemo(() => list.data ?? [], [list.data]);
  const active = useMemo(() => assets.filter((asset) => asset.status !== "retired"), [assets]);
  const latest = useLatestInspections(land, active);
  const paged = assets.length === ASSET_PAGE || offset > 0;
  const summary = useMemo(
    () => summarizeAssets(active, latest.byId, paged),
    [active, latest.byId, paged],
  );
  const record = useAssetRecord(land, selectedId);
  const selected =
    (record.data?.id === selectedId ? record.data : null) ??
    assets.find((asset) => asset.id === selectedId) ??
    null;
  const [colorBy, setColorBy] = useState<ColorBy>("kind");
  const [choosing, setChoosing] = useState(false);
  const [drawing, setDrawing] = useState<Drawing | null>(null);
  const [drawBusy, setDrawBusy] = useState(false);
  const [drawError, setDrawError] = useState<string | null>(null);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const state = useAssetDraft(land, scope, (draft) => fly(draft.geometry));
  const draft = state.draft;
  const select = (id: string | null) => useLandContext.getState().selectInventory(id);

  // Leaving for another lens lets go of the selection, so picking the same asset again from
  // elsewhere (a cited record) still opens this lens.
  useEffect(
    () => () =>
      queueMicrotask(() => {
        if (useLens.getState().lens !== "assets") useLandContext.getState().selectInventory(null);
      }),
    [],
  );

  // ---- On the map ----------------------------------------------------------
  const conditions = colorBy === "condition" ? latest.byId : null;
  const colorOf = useCallback(
    (asset: Asset) =>
      conditions
        ? CONDITIONS[conditions[asset.id]?.condition ?? "none"].color
        : CATEGORIES[asset.category].color,
    [conditions],
  );
  const features = useMemo(
    () =>
      active.map((asset) => ({
        id: asset.id,
        label: asset.name,
        geometry: asset.geometry,
        color: colorOf(asset),
      })),
    [active, colorOf],
  );
  const layer = useMemo<LandContextLayer | null>(
    () =>
      visible && features.length
        ? {
            id: INVENTORY,
            title: "Assets",
            features,
            selectedIds: selectedId ? [selectedId] : [],
            style: LAYER_STYLE,
          }
        : null,
    [visible, features, selectedId],
  );
  useInventoryLayer(layer);

  const draftGeometry = drawing ? null : state.geometryEditing ? null : draft?.geometry;
  const draftLabel = state.editing?.name ?? "New asset";
  const draftLayer = useMemo<LandContextLayer | null>(
    () =>
      draftGeometry
        ? {
            id: "assets-draft",
            title: "Asset being added",
            style: DRAFT_STYLE,
            features: [{ id: "draft", label: draftLabel, geometry: draftGeometry }],
          }
        : null,
    [draftGeometry, draftLabel],
  );
  useLensLayer(draftLayer);

  const pins = useMemo<MapPin[] | null>(() => {
    if (drawing || draft) return null;
    const out: MapPin[] = [];
    if (selected)
      out.push({
        id: `asset-${selected.id}`,
        position: anchorOf(selected.geometry),
        label: shortLabel(selected.name),
      });
    for (const { asset, inspection } of summary.attention) {
      if (out.length >= MAX_PINS) break;
      if (asset.id === selected?.id) continue;
      out.push({
        id: `asset-${asset.id}`,
        position: anchorOf(asset.geometry),
        label: `${shortLabel(asset.name, 16)} · ${CONDITIONS[inspection.condition].label.toLowerCase()}`,
        tone: "warn",
        onSelect: () => useLandContext.getState().selectInventory(asset.id),
      });
    }
    return out;
  }, [drawing, draft, selected, summary.attention]);
  useLensPins("assets", pins);

  // ---- Adding and changing -------------------------------------------------
  const startNew = (next: FeatureDraft, where: Placement | null) => {
    setChoosing(false);
    setPlacement(where);
    state.startNew(next);
  };

  const finishDrawing = async (geometry: LandMapGeometry) => {
    if (!drawing) return;
    const current = drawing;
    setDrawBusy(true);
    setDrawError(null);
    try {
      const preview = await unwrap(
        api.POST("/api/v1/land/{land_id}/features/geometry/preview", {
          params: { path: { land_id: land.id } },
          body: { geometry },
        }),
      );
      if (current.purpose === "new")
        startNew(
          {
            name: nextName(
              current.preset.name,
              assets.map((asset) => asset.name),
            ),
            category: current.preset.category,
            status: "candidate",
            geometry: preview.geometry,
            source: { method: "drawn", meaning: "physical-feature", label: "Drawn on the map" },
          },
          preview,
        );
      else {
        state.update({ geometry: preview.geometry });
        setPlacement(preview);
      }
      setDrawing(null);
    } catch (cause) {
      setDrawError(
        cause instanceof ApiError && cause.fieldErrors.length
          ? cause.fieldErrors.join(". ")
          : describeError(cause),
      );
    } finally {
      setDrawBusy(false);
    }
  };

  const notices = (
    <>
      {state.recovery && canEdit && !draft && (
        <div className="land-note assets-recovery">
          <p>An unsaved asset is waiting in this browser.</p>
          <div className="land-row">
            <GlassButton size="sm" disabled={state.busy} onClick={() => void state.recover()}>
              Recover asset draft
            </GlassButton>
            <GlassButton
              size="sm"
              variant="ghost"
              onClick={() => state.recovery && downloadAssetDraft(state.recovery)}
            >
              Download
            </GlassButton>
            <GlassButton size="sm" variant="ghost" onClick={state.clearRecovery}>
              Discard
            </GlassButton>
          </div>
        </div>
      )}
      {!draft && state.error && (
        <p className="land-alert" role="alert">
          {state.error}
        </p>
      )}
      {!draft && state.notice && (
        <p className="land-muted" role="status">
          {state.notice}
        </p>
      )}
    </>
  );

  // ---- The card ------------------------------------------------------------
  if (drawing)
    return (
      <LensCard
        lens="assets"
        headline={
          drawing.purpose === "redraw"
            ? "Redraw the shape"
            : drawing.preset.kind === "Point"
              ? `Where is ${drawWhat(drawing)}?`
              : `Draw ${drawWhat(drawing)}`
        }
      >
        <AssetsDrawing
          owner={`assets-draw:${land.id}`}
          kind={drawing.preset.kind}
          what={drawWhat(drawing)}
          enabled={panelOpen && canEdit && ready}
          busy={drawBusy}
          error={drawError}
          onDone={(geometry) => void finishDrawing(geometry)}
          onCancel={() => {
            setDrawing(null);
            setDrawError(null);
          }}
        />
      </LensCard>
    );

  if (draft && canEdit)
    return (
      <LensCard lens="assets" headline={state.editing ? `Edit ${state.editing.name}` : "New asset"}>
        <AssetsEditor
          land={land}
          state={state}
          placement={placement}
          active={panelOpen}
          onPlacement={setPlacement}
          onRedraw={() => {
            setDrawError(null);
            setDrawing({
              purpose: "redraw",
              preset: {
                id: "redraw",
                label: "shape",
                name: "",
                category: draft.category,
                kind: kindOf(draft.geometry),
              },
            });
          }}
        />
      </LensCard>
    );

  const back = (
    <GlassButton
      size="sm"
      variant="ghost"
      leadingIcon={<ArrowLeft size={14} aria-hidden="true" />}
      onClick={() => select(null)}
    >
      All assets
    </GlassButton>
  );

  if (selectedId)
    return (
      <LensCard lens="assets" headline={selected?.name} action={back}>
        {notices}
        {selected ? (
          <AssetsDetail
            key={selected.id}
            land={land}
            asset={selected}
            canEdit={canEdit && ready}
            onEdit={() => {
              setPlacement({
                intersectsLand: selected.intersectsLand,
                distanceM: selected.distanceM,
              });
              state.startEdit(selected);
            }}
          />
        ) : record.isError ? (
          <p className="land-alert" role="alert">
            This asset could not be loaded.{" "}
            <button type="button" className="land-link" onClick={() => void record.refetch()}>
              Try again
            </button>
          </p>
        ) : (
          <p className="land-muted">Loading the asset…</p>
        )}
      </LensCard>
    );

  const colorToggle = active.length > 0 && (
    <div className="assets-colorby" role="group" aria-label="Colour the map by">
      {(["kind", "condition"] as const).map((value) => (
        <button
          key={value}
          type="button"
          aria-pressed={colorBy === value}
          onClick={() => setColorBy(value)}
        >
          {value === "kind" ? "Kind" : "Condition"}
        </button>
      ))}
    </div>
  );

  const addButton = canEdit && ready && (
    <GlassButton
      variant="primary"
      disabled={Boolean(state.recovery)}
      leadingIcon={<Plus size={16} aria-hidden="true" />}
      onClick={() => setChoosing(true)}
    >
      Add an asset
    </GlassButton>
  );

  const tools = (
    <>
      <AssetsImport land={land} active={panelOpen} />
      {canEdit && ready && !state.recovery && (
        <details className="land-details">
          <summary>Find mapped features nearby</summary>
          <div className="land-details__body">
            <AssetsFinder
              land={land}
              onDraft={(next) => {
                startNew(next, null);
                fly(next.geometry);
              }}
            />
          </div>
        </details>
      )}
      <LensDetails summary="Map and sources">
        <label className="assets-check">
          <input
            type="checkbox"
            checked={visible}
            onChange={(event) =>
              useLandContext.getState().setInventoryVisible(event.target.checked)
            }
          />
          Show assets on the map
        </label>
        {paged && (
          <div className="land-row">
            <span className="land-muted">
              Showing assets {(offset + 1).toLocaleString()}–
              {(offset + assets.length).toLocaleString()}, newest changes first.
            </span>
            <GlassButton
              size="sm"
              variant="ghost"
              disabled={offset === 0}
              onClick={() => {
                select(null);
                setOffset(Math.max(0, offset - ASSET_PAGE));
              }}
            >
              Previous
            </GlassButton>
            <GlassButton
              size="sm"
              variant="ghost"
              disabled={assets.length < ASSET_PAGE}
              onClick={() => {
                select(null);
                setOffset(offset + ASSET_PAGE);
              }}
            >
              Next
            </GlassButton>
          </div>
        )}
        {latest.limited && (
          <p className="land-muted">
            Condition is checked for the {CONDITION_LIMIT} most recently changed assets.
          </p>
        )}
        {latest.failed && (
          <p className="land-muted">
            Some inspections could not be read, so conditions may be missing.
          </p>
        )}
        <p className="land-muted">
          Assets are records you keep: drawn, imported or found in open map data. Their condition
          comes from the newest inspection. Confirming what something is does not establish who owns
          it.
        </p>
      </LensDetails>
    </>
  );

  if (!ready || list.isPending)
    return (
      <LensCard lens="assets">
        <p className="land-muted" aria-live="polite">
          Loading assets…
        </p>
      </LensCard>
    );

  if (list.isError)
    return (
      <LensCard lens="assets" headline="Assets could not be loaded.">
        <p className="land-alert" role="alert">
          {describeError(list.error)}{" "}
          <button type="button" className="land-link" onClick={() => void list.refetch()}>
            Try again
          </button>
        </p>
      </LensCard>
    );

  if (!active.length)
    return (
      <LensCard lens="assets" headline="No assets mapped yet.">
        {notices}
        {choosing ? (
          <AssetsChooser
            onCancel={() => setChoosing(false)}
            onChoose={(preset) => {
              setChoosing(false);
              setDrawError(null);
              setDrawing({ purpose: "new", preset });
            }}
          />
        ) : (
          <LensEmpty action={addButton}>
            Map fences, gates, wells and buildings. They show here and on the map, with their
            condition.
          </LensEmpty>
        )}
        {assets.length > 0 && (
          <AssetsList
            assets={assets}
            latest={latest.byId}
            colorOf={colorOf}
            selectedId={selectedId}
            onSelect={(asset) => select(asset.id)}
          />
        )}
        {tools}
      </LensCard>
    );

  return (
    <LensCard lens="assets" headline={assetsHeadline(summary)} action={colorToggle}>
      {notices}
      <ul className="land-facts">
        <Fact label="What's here" value={<Kinds summary={summary} />} />
        {summary.attention[0] && (
          <Fact
            label="Needs attention"
            tone="warn"
            value={names(summary.attention)}
            target={{
              key: `asset:${summary.attention[0].asset.id}`,
              geometry: summary.attention[0].asset.geometry,
            }}
          />
        )}
        {summary.last && (
          <Fact
            label="Last inspection"
            value={`${summary.last.asset.name}, ${CONDITIONS[
              summary.last.inspection.condition
            ].label.toLowerCase()}, ${whenLabel(summary.last.inspection.observedAt)}`}
            target={{
              key: `asset:${summary.last.asset.id}`,
              geometry: summary.last.asset.geometry,
            }}
          />
        )}
      </ul>
      {colorBy === "condition" && (
        <p className="assets-legend" aria-label="Condition colours">
          {[...CONDITION_ORDER, "none" as const].map((value) => (
            <span key={value}>
              <i style={{ background: CONDITIONS[value].color }} aria-hidden="true" />
              {CONDITIONS[value].label}
            </span>
          ))}
        </p>
      )}
      {choosing ? (
        <AssetsChooser
          onCancel={() => setChoosing(false)}
          onChoose={(preset) => {
            setChoosing(false);
            setDrawError(null);
            setDrawing({ purpose: "new", preset });
          }}
        />
      ) : (
        <>
          {addButton && <div className="land-row">{addButton}</div>}
          <AssetsList
            assets={assets}
            latest={latest.byId}
            colorOf={colorOf}
            selectedId={selectedId}
            onSelect={(asset) => select(asset.id)}
          />
        </>
      )}
      {tools}
    </LensCard>
  );
}
