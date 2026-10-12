import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus } from "lucide-react";

import type { LandArea, LandCandidate } from "@twin/contracts";
import { boundsOf } from "@twin/geo";
import { GlassButton, GlassSelect } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import type { LandContextLayer } from "@/state/landContext";
import { useSelection } from "@/state/selection";

import { MapLink } from "../../mapLinks";
import { useLensLayer } from "../../useLensMap";
import { draftOfCandidate } from "./assetsModel";
import type { FeatureDraft } from "./useAssetDraft";

type Kind = "line" | "building" | "point";

/**
 * Assets that open map data already knows about (buildings, lines, poles) within 2 km of the
 * land or of the place inspected with "What's here", outlined on the map; one tap starts an
 * unconfirmed asset from a record. The inspected place itself can be recorded too.
 */
export function AssetsFinder({
  land,
  onDraft,
}: {
  land: LandArea;
  onDraft: (draft: FeatureDraft) => void;
}) {
  const id = useId();
  const inspected = useSelection((s) => s.selection);
  const [kind, setKind] = useState<Kind>("building");
  const [nearInspected, setNearInspected] = useState(false);
  const [found, setFound] = useState<LandCandidate[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ticket = useRef(0);
  useEffect(
    () => () => {
      ticket.current++;
    },
    [],
  );

  const layer = useMemo<LandContextLayer | null>(
    () =>
      found.length
        ? {
            id: "assets-found",
            title: "Mapped features nearby",
            style: { color: "#d8dde3", fill: 0.12, width: 2 },
            features: found.map((candidate) => ({
              id: candidate.id,
              label: candidate.label,
              geometry: candidate.geometry,
            })),
          }
        : null,
    [found],
  );
  useLensLayer(layer);

  const search = async () => {
    const current = ++ticket.current;
    setBusy(true);
    setError(null);
    const b = boundsOf(land.boundary);
    const point =
      nearInspected && inspected
        ? [inspected.longitude, inspected.latitude]
        : [(b.west + b.east) / 2, (b.south + b.north) / 2];
    try {
      const response = await unwrap(
        api.POST("/api/v1/land/selection/candidates", {
          body: { kind, point: { type: "Point", coordinates: point }, radiusM: 2000 },
        }),
      );
      if (ticket.current !== current) return;
      setFound(response.candidates);
      setMessage(
        `${response.message}${response.truncated ? " There may be more; inspect a closer point to narrow it." : ""}`,
      );
    } catch (cause) {
      if (ticket.current === current) setError(describeError(cause));
    } finally {
      if (ticket.current === current) setBusy(false);
    }
  };

  return (
    <div className="assets-finder">
      <label className="land-field" htmlFor={`${id}-kind`}>
        Mapped feature type
        <GlassSelect
          id={`${id}-kind`}
          value={kind}
          onChange={(event) => setKind(event.target.value as Kind)}
        >
          <option value="building">Buildings</option>
          <option value="line">Lines, roads and waterways</option>
          <option value="point">Poles, towers and equipment</option>
        </GlassSelect>
      </label>
      <label className="assets-check">
        <input
          type="checkbox"
          checked={nearInspected}
          disabled={!inspected}
          onChange={(event) => setNearInspected(event.target.checked)}
        />
        Search near the inspected map location
      </label>
      <div className="land-row">
        <GlassButton size="sm" loading={busy} disabled={busy} onClick={() => void search()}>
          Find mapped features within 2 km
        </GlassButton>
        <GlassButton
          size="sm"
          variant="ghost"
          disabled={!inspected}
          onClick={() => {
            if (!inspected) return;
            onDraft({
              name: inspected.title,
              category: "other",
              status: "candidate",
              geometry: { type: "Point", coordinates: [inspected.longitude, inspected.latitude] },
              source: {
                method: "drawn",
                meaning: "physical-feature",
                label: "Inspected map location",
              },
            });
          }}
        >
          Record inspected point
        </GlassButton>
      </div>
      <p className="land-muted">
        Searches around the middle of your land, or near a place you inspect on the map.
      </p>
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="land-muted" role="status">
          {message}
        </p>
      )}
      {found.length > 0 && (
        <ul className="assets-list">
          {found.map((candidate) => (
            <MapLink
              key={candidate.id}
              as="li"
              fly={false}
              className="assets-row"
              target={{ key: `found:${candidate.id}`, geometry: candidate.geometry }}
            >
              <span className="assets-row__text">
                <span className="assets-row__name">{candidate.label}</span>
                <span className="assets-row__sub">
                  {candidate.source.label} · {Math.round(candidate.distanceM)} m away
                </span>
              </span>
              <GlassButton
                size="sm"
                aria-label={`Add ${candidate.label}`}
                leadingIcon={<Plus size={14} aria-hidden="true" />}
                onClick={() => onDraft(draftOfCandidate(candidate))}
              >
                Add
              </GlassButton>
            </MapLink>
          ))}
        </ul>
      )}
    </div>
  );
}
