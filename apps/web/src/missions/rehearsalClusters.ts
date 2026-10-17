/**
 * Raised hands gathered for the map. On steep ground a rehearsal can raise a hundred hands or
 * more (one per lane and rule), and a hundred markers say less than a few that each count
 * theirs. Hands of one kind join a cluster when they stand within `radiusM` of each other or
 * are on neighbouring lanes of one zone (lane numbers one apart: the next pass, or the next
 * stretch of a pass a hole split); joining is transitive, so a band of steep lanes is one
 * cluster. When that still leaves more than `maxClusters`, the radius doubles until it does
 * not. Every hand is in exactly one cluster: clustering is for the map only, and nothing is
 * dropped from a count.
 *
 * Pure: no Cesium. `cesium/RehearsalManager.ts` draws a cluster as one marker with its count,
 * at its first hand (the first a ghost reaches), and a click on it opens that hand.
 */

import { DEG_TO_RAD, EARTH_RADIUS_M } from "@twin/geo";

import type { HandKind, RaisedHand } from "./rehearsal";

/** Hands this close (metres) join one cluster. */
export const CLUSTER_RADIUS_M = 80;
/** At most this many markers stand on the map at once. */
export const MAX_CLUSTERS = 15;

export interface HandCluster {
  /** The kind every hand in it shares. */
  kind: HandKind;
  /** Its hands in the order they go up. */
  hands: RaisedHand[];
  /** The first hand: where the marker stands, and what a click on it opens. */
  first: RaisedHand;
}

/** How each kind of hand is named where hands are grouped: the card's list and the markers. */
export const HAND_KIND_TITLE: Record<HandKind, string> = {
  "cross-slope": "Too steep across",
  grade: "Too steep up or downhill",
  "creek-buffer": "Inside a creek buffer",
};

/** The order kinds are listed in. */
export const HAND_KINDS: readonly HandKind[] = ["cross-slope", "grade", "creek-buffer"];

function distanceM(a: RaisedHand, b: RaisedHand): number {
  const kx = Math.cos(((a.at.latitude + b.at.latitude) / 2) * DEG_TO_RAD) * DEG_TO_RAD;
  const dx = (b.at.longitude - a.at.longitude) * kx * EARTH_RADIUS_M;
  const dy = (b.at.latitude - a.at.latitude) * DEG_TO_RAD * EARTH_RADIUS_M;
  return Math.hypot(dx, dy);
}

function neighbours(a: RaisedHand, b: RaisedHand): boolean {
  return a.zoneId === b.zoneId && Math.abs(a.number - b.number) <= 1;
}

/** The hands of one kind in groups: within `radiusM` or on neighbouring lanes, transitively. */
function groupsOf(hands: readonly RaisedHand[], radiusM: number): RaisedHand[][] {
  const parent = hands.map((_, i) => i);
  const root = (i: number): number => {
    let r = i;
    while (parent[r] !== r) r = parent[r] ?? r;
    // Path compression keeps later look-ups short.
    let j = i;
    while (parent[j] !== r) {
      const next = parent[j] ?? r;
      parent[j] = r;
      j = next;
    }
    return r;
  };
  for (let i = 0; i < hands.length; i++)
    for (let j = i + 1; j < hands.length; j++) {
      const a = hands[i];
      const b = hands[j];
      if (!a || !b || root(i) === root(j)) continue;
      if (neighbours(a, b) || distanceM(a, b) <= radiusM) parent[root(j)] = root(i);
    }
  const groups = new Map<number, RaisedHand[]>();
  hands.forEach((hand, i) => {
    const r = root(i);
    groups.set(r, [...(groups.get(r) ?? []), hand]);
  });
  return [...groups.values()];
}

/**
 * The hands as clusters, in the order their first hands go up. Deterministic for the same
 * hands. The kinds never mix: a cluster's count is lanes over one rule.
 */
export function clusterHands(
  hands: readonly RaisedHand[],
  options: { radiusM?: number; maxClusters?: number } = {},
): HandCluster[] {
  const maxClusters = Math.max(HAND_KINDS.length, options.maxClusters ?? MAX_CLUSTERS);
  const byKind = HAND_KINDS.map((kind) => hands.filter((h) => h.kind === kind)).filter(
    (list) => list.length > 0,
  );
  let radiusM = options.radiusM ?? CLUSTER_RADIUS_M;
  let groups: RaisedHand[][] = [];
  // 40 doublings reach past the Earth's size: by then each kind is one cluster.
  for (let step = 0; step < 40; step++) {
    groups = byKind.flatMap((list) => groupsOf(list, radiusM));
    if (groups.length <= maxClusters) break;
    radiusM *= 2;
  }
  const order = (a: RaisedHand, b: RaisedHand) => a.t - b.t || a.id.localeCompare(b.id);
  return groups
    .map((group) => {
      const sorted = [...group].sort(order);
      const first = sorted[0];
      return first ? { kind: first.kind, hands: sorted, first } : null;
    })
    .filter((cluster): cluster is HandCluster => cluster !== null)
    .sort((a, b) => order(a.first, b.first));
}

/**
 * A cluster's label on the map once `raised` of its hands are up: the hand's own words for
 * one, the rule and the count for more ("12 lanes too steep across · limit 25 % (assumed)").
 */
export function clusterLabel(cluster: HandCluster, raised: number): string {
  if (raised <= 1) return cluster.first.label;
  const limits = [...new Set(cluster.hands.map((h) => h.limit))];
  const limit = limits.length === 1 ? (limits[0] ?? 0) : null;
  const rule =
    cluster.kind === "creek-buffer"
      ? `inside a creek buffer${limit !== null ? ` · ${limit} m (assumed)` : ""}`
      : `${HAND_KIND_TITLE[cluster.kind].toLowerCase()}${limit !== null ? ` · limit ${Math.round(limit)} % (assumed)` : ""}`;
  return `${raised} lanes ${rule}`;
}
