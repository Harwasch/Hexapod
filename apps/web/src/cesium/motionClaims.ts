/**
 * Which instances a driver has taken, so another leaves them alone: a scene object moved by
 * telemetry (`telemetry.ts`) must not also be swayed by the wind (`skinWind.ts`) -- both
 * would write the same skin's handles, and a driven body is not a plant.
 *
 * Claims are per scan (asset id) and per owner; an instance is claimed while any owner holds
 * it. The telemetry driver claims each bound instance with its ancestors and descendants, so
 * a skin that covers any of its splats is left to it.
 */

const CLAIMS = new Map<string, Map<object, ReadonlySet<number>>>();

/** Claims `ids` of `assetId` for `owner` (replacing its earlier claim); returns the release. */
export function claimInstances(
  assetId: string,
  owner: object,
  ids: ReadonlySet<number>,
): () => void {
  let owners = CLAIMS.get(assetId);
  if (!owners) {
    owners = new Map();
    CLAIMS.set(assetId, owners);
  }
  owners.set(owner, ids);
  return () => {
    const now = CLAIMS.get(assetId);
    if (now?.get(owner) !== ids) return;
    now.delete(owner);
    if (now.size === 0) CLAIMS.delete(assetId);
  };
}

/** Whether a driver holds instance `instanceId` of `assetId`. */
export function isClaimed(assetId: string, instanceId: number): boolean {
  const owners = CLAIMS.get(assetId);
  if (!owners) return false;
  for (const ids of owners.values()) if (ids.has(instanceId)) return true;
  return false;
}

/** The claim test for one scan, as a driver takes it. */
export function claimsOf(assetId: string): (instanceId: number) => boolean {
  return (instanceId) => isClaimed(assetId, instanceId);
}
