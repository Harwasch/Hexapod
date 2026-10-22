import { useMemo } from "react";

import type { LandEvidence, LandFinding, LandMapGeometry } from "@twin/contracts";

import type { LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";

import { useLensLayer, useLensPins } from "../useLensMap";
import { findingPlace, locationArea, type EventLocation } from "./agentEvents";
import { useAgentUi } from "./agentStore";

/** The agent's colour on the map (`--agent`). */
export const AGENT_COLOR = "#9db8f0";
const MAX_PINS = 12;

function shortLabel(title: string): string {
  return title.length > 28 ? `${title.slice(0, 27).trimEnd()}…` : title;
}

/**
 * What the agent found, on the map: a pin per place (findings at the same spot share one),
 * and the outline of any finding that has a boundary of its own. Pinned findings come first;
 * dismissed ones are left off.
 */
export function useAgentFindingsOnMap(
  findings: LandFinding[],
  places: Map<string, EventLocation>,
  evidence: LandEvidence[],
): void {
  const pins = useMemo<MapPin[]>(() => {
    const groups = new Map<string, { anchor: [number, number]; findings: LandFinding[] }>();
    const ordered = findings
      .filter((finding) => finding.disposition !== "dismissed")
      .sort((a, b) => Number(b.disposition === "pinned") - Number(a.disposition === "pinned"));
    for (const finding of ordered) {
      const place = findingPlace(finding, places, evidence);
      if (!place) continue;
      const key = place.anchor.map((value) => value.toFixed(6)).join(",");
      const group = groups.get(key) ?? { anchor: place.anchor, findings: [] };
      group.findings.push(finding);
      groups.set(key, group);
    }
    return [...groups.values()]
      .slice(0, MAX_PINS)
      .flatMap(({ anchor, findings: [first, ...rest] }) =>
        first
          ? [
              {
                id: `agent-${first.id}`,
                position: anchor,
                label: rest.length
                  ? `${shortLabel(first.title)} +${rest.length}`
                  : shortLabel(first.title),
                tone: "agent" as const,
                onSelect: () => useAgentUi.getState().setHighlight(first.id),
              },
            ]
          : [],
      );
  }, [findings, places, evidence]);
  useLensPins("agent", pins.length ? pins : null);

  const outlines = useMemo<LandContextLayer | null>(() => {
    const features = findings
      .filter((finding) => finding.boundary && finding.disposition !== "dismissed")
      .map((finding) => ({
        id: finding.id,
        label: finding.title,
        geometry: finding.boundary as LandMapGeometry,
      }));
    return features.length
      ? {
          id: "agent-findings",
          title: "What the agent found",
          agent: true,
          features,
          style: { color: AGENT_COLOR, fill: 0.14, width: 2 },
        }
      : null;
  }, [findings]);
  useLensLayer(outlines);
}

/** The area the agent is looking at right now (a query box, a search circle), outlined. */
export function useAgentLookingOnMap(
  location: EventLocation | null,
  evidence: LandEvidence[],
): void {
  const area = locationArea(location, evidence);
  const signature = area ? JSON.stringify(area.coordinates) : null;
  const layer = useMemo<LandContextLayer | null>(
    () =>
      area
        ? {
            id: "agent-looking",
            title: "Where the agent is looking",
            agent: true,
            features: [{ id: "looking", label: "Where the agent is looking", geometry: area }],
            style: { color: AGENT_COLOR, fill: 0.06, width: 1.5 },
          }
        : null,
    // The area is rebuilt from the location each render; its coordinates are its identity.
    [signature],
  );
  useLensLayer(layer);
}
