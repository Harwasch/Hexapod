import { DEG_TO_RAD, EARTH_RADIUS_M, type LonLat } from "@twin/geo";
import { describe, expect, it } from "vitest";

import type { HandKind, RaisedHand } from "@/missions/rehearsal";
import {
  clusterHands,
  clusterLabel,
  HAND_KIND_TITLE,
  MAX_CLUSTERS,
} from "@/missions/rehearsalClusters";

const origin = { longitude: -120.04, latitude: 34.698 };
const kx = Math.cos(origin.latitude * DEG_TO_RAD) * DEG_TO_RAD * EARTH_RADIUS_M;
const ky = DEG_TO_RAD * EARTH_RADIUS_M;
const at = (x: number, y: number): LonLat => ({
  longitude: origin.longitude + x / kx,
  latitude: origin.latitude + y / ky,
});

function hand(
  number: number,
  kind: HandKind,
  x: number,
  y: number,
  extra: Partial<RaisedHand> = {},
): RaisedHand {
  return {
    id: `Z-14:${number - 1}:${kind}`,
    kind,
    zoneId: "Z-14",
    index: number - 1,
    number,
    machineId: "TR-04",
    zoneLabel: "Z-14",
    at: at(x, y),
    t: number * 60,
    value: kind === "creek-buffer" ? 4 : 31,
    limit: kind === "creek-buffer" ? 15 : 25,
    label: `lane ${number} ${kind}`,
    rule: "",
    source: "",
    ...extra,
  };
}

describe("clusterHands", () => {
  it("joins hands of one kind that stand close, and never mixes kinds", () => {
    const clusters = clusterHands([
      hand(1, "cross-slope", 0, 0),
      hand(5, "cross-slope", 50, 30),
      hand(9, "grade", 20, 0),
      hand(20, "cross-slope", 900, 0),
    ]);
    expect(clusters.map((c) => [c.kind, c.hands.map((h) => h.number)])).toEqual([
      ["cross-slope", [1, 5]],
      ["grade", [9]],
      ["cross-slope", [20]],
    ]);
    // A cluster stands at, and opens, the first of its hands to go up.
    expect(clusters[0]?.first.number).toBe(1);
  });

  it("joins neighbouring lanes of a zone however far apart their worst points are", () => {
    const clusters = clusterHands([
      hand(3, "cross-slope", 0, -600),
      hand(4, "cross-slope", 40, 600),
      hand(5, "cross-slope", 80, -600),
      // Two lanes on: a separate band.
      hand(7, "cross-slope", 160, 600),
      // The same lane numbers in another zone are not neighbours.
      hand(4, "cross-slope", 3000, 0, { zoneId: "Z-08", zoneLabel: "Z-08", id: "Z-08:3" }),
    ]);
    expect(clusters.map((c) => c.hands.map((h) => `${h.zoneId}/${h.number}`))).toEqual([
      ["Z-14/3", "Z-14/4", "Z-14/5"],
      ["Z-08/4"],
      ["Z-14/7"],
    ]);
  });

  it("keeps the map to a handful of markers, and every hand in exactly one", () => {
    // 150 hands spread over a 3 km ranch, every third lane, a kilometre between some.
    const hands: RaisedHand[] = [];
    for (let i = 0; i < 150; i++) {
      const kind: HandKind = i % 3 === 0 ? "cross-slope" : i % 3 === 1 ? "grade" : "creek-buffer";
      hands.push(hand(i * 3 + 1, kind, (i % 10) * 300, Math.floor(i / 10) * 200));
    }
    const clusters = clusterHands(hands);
    expect(clusters.length).toBeLessThanOrEqual(MAX_CLUSTERS);
    expect(clusters.length).toBeGreaterThanOrEqual(3);
    const ids = clusters.flatMap((c) => c.hands.map((h) => h.id));
    expect(ids).toHaveLength(150);
    expect(new Set(ids).size).toBe(150);
    for (const cluster of clusters)
      expect(cluster.hands.every((h) => h.kind === cluster.kind)).toBe(true);
    // The same hands, the same clusters.
    expect(clusterHands([...hands].reverse())).toEqual(clusters);
  });

  it("names a cluster by its rule and count, and a single hand by its own label", () => {
    const [steep] = clusterHands([hand(1, "cross-slope", 0, 0), hand(2, "cross-slope", 10, 0)]);
    if (!steep) throw new Error("expected a cluster");
    expect(clusterLabel(steep, 1)).toBe("lane 1 cross-slope");
    expect(clusterLabel(steep, 2)).toBe("2 lanes too steep across · limit 25 % (assumed)");
    const [creek] = clusterHands([hand(1, "creek-buffer", 0, 0), hand(2, "creek-buffer", 5, 0)]);
    if (!creek) throw new Error("expected a cluster");
    expect(clusterLabel(creek, 2)).toBe("2 lanes inside a creek buffer · 15 m (assumed)");
    expect(HAND_KIND_TITLE.grade).toBe("Too steep up or downhill");
  });
});
