import { useMemo } from "react";

import type { Footprint } from "@twin/contracts";
import { boundsOf } from "@twin/geo";

import type { LandPoint } from "@/state/land";
import type { LandContextLayer } from "@/state/landContext";
import type { MapPin } from "@/state/landMap";

import { lensMeta } from "../../lenses";
import { useLensLayer, useLensPins } from "../../useLensMap";
import { sheetGroups, truncate, yearRange, type ArchiveItem } from "./recordsModel";
import { useRecords } from "./recordsStore";

const MAX_PINS = 12;
const MAX_SHEET_PINS = 4;
const MAX_PHOTO_PINS = 8;
const SELECTED = "#f1cb7a";

const openItem = (id: string) => useRecords.getState().open({ kind: "archive", id });

/**
 * Old maps and photos on the map: each map sheet's catalog outline with a year pin, each
 * photo as a pin at its catalog point. The open one turns gold. Everything comes off the map
 * when the lens closes.
 */
export function useRecordsMap(
  boundary: Footprint,
  items: ArchiveItem[],
  selectedId: string | null,
): Map<string, LandPoint> {
  const groups = useMemo(() => sheetGroups(items), [items]);
  const color = lensMeta("records").color;

  const sheets = useMemo<LandContextLayer | null>(
    () =>
      groups.length
        ? {
            id: "records-sheets",
            title: "Old map sheets",
            legend: "Catalog extent of each historical map sheet",
            style: { color, fill: 0.03, width: 1.5 },
            features: groups.map((group) => ({
              id: group.key,
              label: `${yearRange(group.items)} · ${group.items[0]?.title ?? "Map sheet"}`,
              geometry: group.footprint,
            })),
          }
        : null,
    [groups, color],
  );
  useLensLayer(sheets);

  const selectedGroup = useMemo(
    () => groups.find((group) => group.items.some((item) => item.id === selectedId)) ?? null,
    [groups, selectedId],
  );
  const selectedLayer = useMemo<LandContextLayer | null>(
    () =>
      selectedGroup
        ? {
            id: "records-selected",
            title: "Open map sheet",
            style: { color: SELECTED, fill: 0.06, width: 3 },
            features: [
              { id: selectedGroup.key, label: "Open map sheet", geometry: selectedGroup.footprint },
            ],
          }
        : null,
    [selectedGroup],
  );
  useLensLayer(selectedLayer);

  const land = useMemo(() => boundsOf(boundary), [boundary]);
  // Sheets usually cover far more than the land; each sheet's pin (and the end of any line to
  // it) sits along the land's north edge, inside the sheet, where it shows at the land's zoom.
  const anchors = useMemo(() => {
    const result = new Map<string, LandPoint>();
    groups.forEach((group, index) => {
      const sheet = boundsOf(group.footprint);
      const spread = land.west + ((land.east - land.west) * (index + 1)) / (groups.length + 1);
      const point: LandPoint = [
        Math.min(sheet.east, Math.max(sheet.west, spread)),
        Math.min(sheet.north, Math.max(sheet.south, land.north)),
      ];
      result.set(group.key, point);
      for (const item of group.items) result.set(item.id, point);
    });
    for (const item of items)
      if (item.kind === "photograph" && item.geometry?.type === "Point") {
        const [lon = 0, lat = 0] = item.geometry.coordinates;
        result.set(item.id, [lon, lat]);
      }
    return result;
  }, [groups, items, land]);
  const pins = useMemo<MapPin[]>(() => {
    const sheetPins = groups.slice(0, MAX_SHEET_PINS).map((group): MapPin => {
      const first = group.items[0];
      const count = group.items.length > 1 ? ` (${group.items.length})` : "";
      return {
        id: `records-sheet-${group.key}`,
        position: anchors.get(group.key) ?? [land.west, land.north],
        label: `${yearRange(group.items)} · ${truncate(first?.title ?? "Map", 18)}${count}`,
        tone: group.items.some((item) => item.id === selectedId) ? "warn" : "accent",
        onSelect: () => first && openItem(first.id),
      };
    });
    const located = items.filter(
      (item) => item.kind === "photograph" && item.geometry?.type === "Point",
    );
    const shownPhotos = located.slice(0, Math.min(MAX_PHOTO_PINS, MAX_PINS - sheetPins.length));
    // The open photo always has its pin, in place of the last one if it was not shown.
    const open = located.find((item) => item.id === selectedId);
    if (open && shownPhotos.length && !shownPhotos.includes(open))
      shownPhotos.splice(shownPhotos.length - 1, 1, open);
    const photoPins = shownPhotos.map((item): MapPin => ({
      id: `records-photo-${item.id}`,
      position: anchors.get(item.id) ?? [0, 0],
      label: `${item.year ?? "Undated"} · ${truncate(item.title, 18)}`,
      tone: item.id === selectedId ? "warn" : "neutral",
      onSelect: () => openItem(item.id),
    }));
    return [...sheetPins, ...photoPins];
  }, [groups, items, land, anchors, selectedId]);
  useLensPins("records", pins);
  return anchors;
}
