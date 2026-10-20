/**
 * Which of a site's models is shown, now that a site can hold several dated captures (epochs,
 * decision D3; docs/DATA_MODEL.md "Epochs on the globe"), and how an epoch is named on screen.
 *
 * One rule, used everywhere a model is picked: the globe (`SiteManager.pickAsset`), the React
 * controls that describe the shown model (the Splat / Mesh switcher, the load pill, the
 * Inspector, the capture-date control) and view.html. Pure, and it imports types only: view.html
 * must never load CesiumJS (scripts/check-bundle.mjs).
 *
 * The epochs come with the site's own record (`SiteRead.epochs`). A record without them -- an
 * older API, the offline catalog, a test's partial site -- reads as a site with none.
 */

import type { Epoch, EpochPlatform, Representation, Site, SiteAsset } from "@twin/contracts";

/** What the scene was asked to show: an epoch (null or absent: the latest), or an asset by id
 *  (the load pill's Retry). */
export interface ShownSelection {
  epochId?: string | null;
  assetId?: string | null;
}

/** An epoch offered for a representation, with the asset that shows it. */
export interface EpochOption {
  epoch: Epoch;
  asset: SiteAsset;
}

/** A site's epochs, or none for a record that does not carry them. */
export function siteEpochs(site: Pick<Site, "epochs"> | null | undefined): readonly Epoch[] {
  const listed: unknown = (site as { epochs?: unknown } | null | undefined)?.epochs;
  return Array.isArray(listed) ? (listed as Epoch[]) : [];
}

function time(iso: string | null | undefined): number {
  const at = iso ? Date.parse(iso) : Number.NaN;
  return Number.isFinite(at) ? at : Number.NEGATIVE_INFINITY;
}

function compareIds(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Whether `a` was created after `b`: `createdAt`, then id, as the API orders revisions. */
function newer(a: SiteAsset, b: SiteAsset): boolean {
  const at = time(a.createdAt);
  const bt = time(b.createdAt);
  return at > bt || (at === bt && compareIds(a.id, b.id) > 0);
}

/** Oldest first: when it was observed, then when it was recorded, then id. */
function byObserved(a: Epoch, b: Epoch): number {
  return (
    time(a.observedAt) - time(b.observedAt) ||
    time(a.createdAt) - time(b.createdAt) ||
    compareIds(a.id, b.id)
  );
}

/** The order epochs joined their site: ordinal, then when recorded, then id. */
function byOrdinal(a: Epoch, b: Epoch): number {
  return a.ordinal - b.ordinal || time(a.createdAt) - time(b.createdAt) || compareIds(a.id, b.id);
}

/** Whether an asset is drawn as itself: an observed capture, not a superseded revision. */
function isLive(asset: SiteAsset): boolean {
  return (asset.validTo ?? null) === null && asset.sourceKind !== "inferred";
}

/**
 * The epoch's current asset of `representation`, or null when it has none.
 *
 * The server's rule first (`EpochRead.currentAssetId`, apps/api services/epochs.py
 * `current_asset_of`), when that asset is of this representation. Otherwise the same rule
 * recomputed here, for the representation asked: the epoch's newest observed asset with no
 * `validTo`, by `createdAt`, then id.
 */
export function currentAsset(
  epoch: Epoch,
  assets: readonly SiteAsset[],
  representation: Representation,
): SiteAsset | null {
  const named = epoch.currentAssetId
    ? assets.find((asset) => asset.id === epoch.currentAssetId)
    : undefined;
  if (named?.representation === representation && isLive(named)) return named;
  let best: SiteAsset | null = null;
  for (const asset of assets) {
    if (asset.epochId !== epoch.id || asset.representation !== representation) continue;
    if (isLive(asset) && (!best || newer(asset, best))) best = asset;
  }
  return best;
}

/** The published epochs with a current asset of `representation`, oldest first. */
export function epochOptions(
  epochs: readonly Epoch[],
  assets: readonly SiteAsset[],
  representation: Representation,
): EpochOption[] {
  const options: EpochOption[] = [];
  for (const epoch of [...epochs].sort(byObserved)) {
    if (epoch.status !== "published") continue;
    const asset = currentAsset(epoch, assets, representation);
    if (asset) options.push({ epoch, asset });
  }
  return options;
}

/** The option shown for a choice: the chosen epoch when it is offered, else the latest. */
export function shownOption(
  options: readonly EpochOption[],
  epochId: string | null | undefined,
): EpochOption | null {
  const chosen = epochId ? options.find((option) => option.epoch.id === epochId) : undefined;
  return chosen ?? options.at(-1) ?? null;
}

/**
 * The epoch every other epoch of the site is drawn with (SiteManager `followAnchor`): the
 * offered epoch with the lowest ordinal. Under the API's rules ordinal 1 is the capture that
 * made the site's frame (new-site, or an explicit join to a frameless site), so the frame's
 * height error is measured on the capture it belongs to, and publishing a new epoch never
 * moves the ones already there. Null when nothing of `representation` is offered.
 */
export function frameAnchor(
  epochs: readonly Epoch[],
  assets: readonly SiteAsset[],
  representation: Representation,
): EpochOption | null {
  let anchor: EpochOption | null = null;
  for (const option of epochOptions(epochs, assets, representation)) {
    if (!anchor || byOrdinal(option.epoch, anchor.epoch) < 0) anchor = option;
  }
  return anchor;
}

/** Whether an epoch is the one that made its site's frame: the lowest ordinal of them all. */
export function definesFrame(epoch: Epoch, epochs: readonly Epoch[]): boolean {
  return epochs.every((other) => other.ordinal >= epoch.ordinal);
}

/**
 * THE pick rule: which asset of a site is shown for `representation`.
 *
 * 1. Candidates: the site's assets of the representation. An epoch's asset stays only while it
 *    is the current asset of a published epoch; when the record lists no epochs at all, while it
 *    has no `validTo` and is observed. Assets of no epoch (seeded, hand-made, legacy) always stay.
 * 2. A pinned asset among them wins (the load pill's Retry asks for one by id).
 * 3. A site with epochs on offer shows the chosen one, else the latest.
 * 4. Epoch assets but no epoch list: the newest-created of them, so the site is never empty.
 * 5. Anything else, the rule from before epochs: the default-visible asset, else the newest
 *    observed, else the first.
 */
export function pickShownAsset(
  assets: readonly SiteAsset[],
  epochs: readonly Epoch[],
  representation: Representation,
  selection: ShownSelection = {},
): SiteAsset | null {
  const byId = new Map(epochs.map((epoch) => [epoch.id, epoch]));
  const candidates = assets.filter((asset) => {
    if (asset.representation !== representation) return false;
    if (!asset.epochId) return true;
    if (epochs.length === 0) return isLive(asset);
    const epoch = byId.get(asset.epochId);
    return (
      epoch?.status === "published" && currentAsset(epoch, assets, representation)?.id === asset.id
    );
  });
  if (selection.assetId) {
    const pinned = candidates.find((asset) => asset.id === selection.assetId);
    if (pinned) return pinned;
  }
  const shown = shownOption(epochOptions(epochs, assets, representation), selection.epochId);
  if (shown) return shown.asset;
  const ofEpochs = candidates.filter((asset) => asset.epochId);
  if (ofEpochs.length > 0)
    return ofEpochs.reduce((latest, asset) => (newer(asset, latest) ? asset : latest));
  return (
    candidates.find((asset) => asset.defaultVisible) ??
    [...candidates].sort((a, b) => (b.observedAt ?? "").localeCompare(a.observedAt ?? ""))[0] ??
    null
  );
}

/** The epoch an asset belongs to, from the site's list, or null. */
export function epochOf(
  asset: SiteAsset | null | undefined,
  epochs: readonly Epoch[],
): Epoch | null {
  return asset?.epochId ? (epochs.find((epoch) => epoch.id === asset.epochId) ?? null) : null;
}

// ── Labels ──────────────────────────────────────────────────────────────────────────────

const PLATFORM_LABEL: Record<EpochPlatform, string> = {
  "rtk-drone": "RTK drone",
  drone: "Drone",
  "phone-video": "Phone video",
  "phone-photo": "Phone photos",
  unknown: "Unknown platform",
};

/** What captured an epoch, in words; a platform this build does not know is shown as written. */
export function platformLabel(platform: string | null | undefined): string {
  if (!platform) return PLATFORM_LABEL.unknown;
  return (PLATFORM_LABEL as Record<string, string | undefined>)[platform] ?? platform;
}

/** A capture date, in UTC so that every viewer, anywhere, names the same day. */
export function formatEpochDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** A capture's date and time of day, in UTC, said to be UTC. */
export function formatEpochDateTime(iso: string): string {
  const at = new Date(iso).toLocaleString(undefined, {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${at} UTC`;
}

/** Whether the epoch's date is only when it was uploaded: the capture time was not recorded. */
export function observedAtFromUpload(epoch: Epoch): boolean {
  return epoch.observedAtSource === "created";
}

export interface AlignmentLabel {
  /** A few words, under the date in the capture-date control. */
  short: string;
  /** The sentence the Inspector and the control's accessible name give. */
  long: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** `, ±X m`, X to two significant figures, or nothing when X is not a number. */
function plusMinus(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "";
  return `, ±${String(Number(value.toPrecision(2)))} m`;
}

const NOT_YET: AlignmentLabel = {
  short: "not aligned",
  long: "Not aligned yet: placed by its own GPS",
};

/**
 * How an epoch is placed against the rest of its site, in words.
 *
 * Read from the alignment record M3 writes (decision D4; `epochs.alignment`), of which the web
 * reads three keys only: `verdict` (`survey` | `coarse` | `unaligned`), `uncertaintyM`, and a
 * ready-made `label`, which is preferred when there is one. A coarse epoch is "aligned
 * visually", never "surveyed", and only an `rtk-drone` epoch is ever survey tier. The epoch
 * that made the site's frame (`anchor`, when it has the
 * lowest ordinal of all `epochs`) defines it rather than being aligned to it. An epoch with no
 * record, or a verdict this build does not know, is not aligned yet, and says so.
 */
export function alignmentLabel(
  epoch: Epoch,
  anchor: Epoch | null,
  epochs: readonly Epoch[] = [],
): AlignmentLabel {
  if (anchor?.id === epoch.id && definesFrame(epoch, epochs))
    return { short: "site frame", long: "Defines the site frame" };
  const alignment: unknown = epoch.alignment;
  if (!isRecord(alignment)) return NOT_YET;
  const label =
    typeof alignment.label === "string" && alignment.label.trim() ? alignment.label.trim() : null;
  const x = plusMinus(alignment.uncertaintyM);
  // Only the survey tier's own platform is ever called survey-grade (D4): anything else whose
  // record says `survey` is shown at the coarse tier, and a ready-made label claiming more is
  // not repeated.
  if (alignment.verdict === "survey" && epoch.platform !== "rtk-drone")
    return {
      short: "aligned visually",
      long: `${platformLabel(epoch.platform)}, aligned visually${x}`,
    };
  switch (alignment.verdict) {
    case "survey": {
      const to = anchor ? formatEpochDate(anchor.observedAt) : "the site frame";
      return { short: "survey tier", long: label ?? `Aligned to ${to} (survey tier${x})` };
    }
    case "coarse":
      return {
        short: "aligned visually",
        long: label ?? `${platformLabel(epoch.platform)}, aligned visually${x}`,
      };
    case "unaligned":
      return { short: "not aligned", long: label ?? "Not aligned: placed by its own GPS" };
    default:
      return NOT_YET;
  }
}
