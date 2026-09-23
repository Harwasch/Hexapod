/**
 * Which captures have a motion rig, and where it sits.
 *
 * **Still an explicit claim, no longer an explicit table.** Until A9 this file held
 * `LIVING_RIGS`, a hand-written slug → rig-path map compiled into the bundle, and its
 * comment defended two properties worth keeping:
 *
 *  1. **No probe.** Asking every splat site for a `source/rig.json` it almost certainly
 *     does not have would mean a 404 per site per load.
 *  2. **A rig is a claim, not an accident.** "This site can move" must be a decision
 *     someone made and wrote down, not the side effect of a file happening to exist.
 *
 * Both survive. The claim moved to where a capture is described — `renderConfig.rigUrl`
 * on the asset, written by whoever registered it (`app/schemas/asset.py`) — so it is
 * still written down, still read rather than discovered, and still costs no request for
 * a site that has no rig. What it stopped costing is a frontend rebuild per capture,
 * which is the thing a table in a bundle cannot avoid and the thing this sprint exists
 * to remove.
 *
 * The path is relative to the tileset URL, so the rig travels with the tiles whether
 * they are served from the development static mount or from a bucket. Nothing here
 * reaches the network; resolution is pure string work and unit-tested.
 */

/**
 * The rig URL for a loaded splat asset, or null when that capture has no rig.
 *
 * Only Gaussian splats can deform: the deformer rewrites a splat attribute texture, and a
 * mesh or a point cloud has nothing of the kind. An ion-hosted asset has no URL to resolve
 * against and is therefore never living either — `sourceUrl` is null for those.
 *
 * @param rigPath the catalog's claim, relative to `sourceUrl`, or null for no rig
 */
export function rigUrlFor(
  rigPath: string | null,
  representation: string,
  sourceUrl: string | null,
): string | null {
  if (representation !== "gaussian-splat") return null;
  if (rigPath === null || rigPath === "" || sourceUrl === null) return null;
  try {
    const base =
      typeof window === "undefined" ? new URL(sourceUrl) : new URL(sourceUrl, window.location.href);
    return new URL(rigPath, base).toString();
  } catch {
    return null;
  }
}
