/**
 * Which captures have a motion rig, and where it sits.
 *
 * An explicit table rather than a probe. Asking every splat site for a `source/rig.json` it
 * almost certainly does not have would mean a 404 per site per load, and — worse — it would make
 * "this site can move" a property of a missing file rather than a decision someone made. A rig
 * is a claim about a capture; claims get written down.
 *
 * The rig lives beside the tileset in the capture folder the API serves statically
 * (`data/tiles/<slug>/`), so the path is relative to the tileset URL. Nothing here reaches the
 * network; resolution is pure string work and unit-tested.
 */

/** Rig path per capture slug, relative to that capture's tileset URL. */
export const LIVING_RIGS: Readonly<Record<string, string>> = {
  // Emitted by tools/captures/synthetic_tree.py beside the tiles it converts.
  "synthetic-tree": "../source/rig.json",
  "synthetic-tree-large": "../source/rig.json",
};

/**
 * The rig URL for a loaded splat asset, or null when that capture has no rig.
 *
 * Only Gaussian splats can deform: the deformer rewrites a splat attribute texture, and a mesh
 * or a point cloud has nothing of the kind. An ion-hosted asset has no URL to resolve against
 * and is therefore never living either — `sourceUrl` is null for those.
 */
export function rigUrlFor(
  slug: string,
  representation: string,
  sourceUrl: string | null,
): string | null {
  if (representation !== "gaussian-splat") return null;
  const relative = LIVING_RIGS[slug];
  if (relative === undefined || sourceUrl === null) return null;
  try {
    const base =
      typeof window === "undefined" ? new URL(sourceUrl) : new URL(sourceUrl, window.location.href);
    return new URL(relative, base).toString();
  } catch {
    return null;
  }
}
