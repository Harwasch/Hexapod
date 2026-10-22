/**
 * Two dates compared side by side under PlayCanvas (SiteManager `setCompareDates`; ADR 0017),
 * loaded on demand with the first comparison (through compositorSession.ts, the rules' module,
 * whose loader SiteManager already has).
 *
 * Under CesiumJS the comparison is the tilesets' own `splitDirection`, and this session hands
 * nothing anyone reads. Under PlayCanvas the right-hand date is the scan PlayCanvas draws
 * (SiteManager `scanTarget`), and this hands that renderer the left-hand date -- its tileset's
 * URL, its root's extras (its inferred layers and split objects), where it is in the scan's
 * frame -- and the divider (`scene.splitPosition`), checked every frame and handed on when it
 * moves (scanView/scanCompare.ts). It lends the renderer the globe's streaming classes
 * (scanView/ruleKit.ts) as a rule's session does, since the left date is streamed as a rule's
 * dates are (scanView/ruleDates.ts).
 *
 * One session per comparison: it lets go, and takes its view back, once the site's comparison
 * is another pair or none, at the next frame or at once by its disposer (SiteManager calls it
 * however the comparison ends).
 */

import type { CompareDates } from "@/lib/epochs";
import type { Epoch, Representation, SiteAsset } from "@twin/contracts";
import type { Cesium3DTileset, Matrix4, Scene } from "cesium";

import { epochOptions } from "@/lib/epochs";

import { compareRoomKey, useSplatRoom } from "@/state/splatRoom";

import { absoluteUrl, placedIn, rootGaussians } from "./datePlacement";
import type { RuleKit } from "./scanView/ruleKit";
import { publishScanCompare, scanCompareOf, type ScanCompareView } from "./scanView/scanCompare";
import { lendRuleKit } from "./scanView/scanRule";

/** The loaded site, as far as the comparison reads it (SiteManager's own record of it). */
export interface CompareSite {
  compare: CompareDates | null;
  representation: Representation;
  epochs: readonly Epoch[];
  site: { id?: string; assets: readonly SiteAsset[] };
  handles: ReadonlyMap<string, { tileset: Cesium3DTileset | null }>;
}

/** The comparisons a session runs for (SiteManager makes a new pair object for each). */
const running = new WeakSet<object>();

/**
 * Hands the renderer drawing `entry`'s right-hand date its comparison, until it ends: the next
 * frame after the pair changes or ends, or at once by the disposer returned (SiteManager calls
 * it as the comparison ends, so a site torn down with no frame after leaves nothing published).
 */
export function startCompare(
  scene: Pick<Scene, "preUpdate" | "splitPosition">,
  entry: CompareSite,
  kit: RuleKit,
  end: () => void = () => undefined,
): () => void {
  const pair = entry.compare;
  if (!pair || running.has(pair)) return () => undefined;
  running.add(pair);
  lendRuleKit(kit);
  let published: { assetId: string; view: ScanCompareView; key: string } | null = null;
  const unpublish = (): void => {
    // Only what this session put there: a newer pair's session may have the asset now.
    const mine = published;
    published = null;
    if (mine === null) return;
    if (scanCompareOf(mine.assetId) === mine.view) publishScanCompare(mine.assetId, null);
  };
  const stop = (): void => {
    off();
    running.delete(pair);
    unpublish();
  };
  const sync = (): void => {
    if (entry.compare !== pair) {
      stop();
      return;
    }
    const offered = epochOptions(entry.epochs, entry.site.assets, entry.representation);
    const left = offered.find((o) => o.epoch.id === pair.left)?.asset;
    const right = offered.find((o) => o.epoch.id === pair.right)?.asset;
    const leftSet = left && entry.handles.get(left.id)?.tileset;
    const rightSet = right && entry.handles.get(right.id)?.tileset;
    const url = leftSet ? absoluteUrl(leftSet) : undefined;
    if (
      !left ||
      !right ||
      !leftSet ||
      !rightSet ||
      leftSet.isDestroyed() ||
      rightSet.isDestroyed() ||
      url === undefined ||
      entry.representation !== "gaussian-splat"
    ) {
      unpublish();
      return;
    }
    const { placement, root } = placedIn(rightSet, leftSet);
    const extras = (leftSet.root as { extras?: unknown } | undefined)?.extras;
    const split = scene.splitPosition;
    const key = JSON.stringify([right.id, left.id, url, placement, root, split]);
    if (published?.key === key) return;
    if (published && published.assetId !== right.id) unpublish();
    const view: ScanCompareView = {
      // It owns no cell of a plane: the comparison splits the budget evenly (no plane).
      left: {
        url,
        assetId: left.id,
        extras,
        placement,
        root,
        owns: [],
        ...(rootGaussians(leftSet) === undefined ? {} : { rootGaussians: rootGaussians(leftSet) }),
      },
      ...(rootGaussians(rightSet) === undefined ? {} : { scanRoot: rootGaussians(rightSet) }),
      // The two roots do not fit the device's budget (the renderer's own check): the comparison
      // ends, and the control holds this pair off, saying so.
      refuse: (dates) => {
        useSplatRoom.setState({
          compare: { key: compareRoomKey(entry.site.id ?? "", pair.left, pair.right), dates },
        });
        end();
      },
      split,
      toWorld: () =>
        rightSet.isDestroyed()
          ? undefined
          : (rightSet.root as { computedTransform?: Matrix4 } | undefined)?.computedTransform,
    };
    published = { assetId: right.id, view, key };
    publishScanCompare(right.id, view);
  };
  const off = scene.preUpdate.addEventListener(sync);
  sync();
  return stop;
}
