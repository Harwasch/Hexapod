import { ARTICULATION_FILE, besideTileset } from "./weatherWindPlan";

/** Per rig address: whether it is published. Only real answers are kept, for the page. */
const rigProbes = new Map<string, Promise<boolean>>();

/** The baked rig's address beside a tileset ("" when it cannot be read). */
export function rigUrlOf(tileset: string): string {
  return besideTileset(tileset, ARTICULATION_FILE);
}

/**
 * Asks whether `articulation.json` is published beside the tiles. A host that answers every path
 * with its HTML page (a single-page fallback) is not a rig, so the type must be JSON. A 4xx is
 * an answer and is kept; a 5xx, a network failure, a timeout or `signal` aborting is not, and the
 * next ask tries again. Keyed by address, so a scan republished elsewhere is asked afresh.
 */
export function probeRig(tileset: string, signal?: AbortSignal): Promise<boolean> {
  const url = rigUrlOf(tileset);
  if (!url) return Promise.resolve(false);
  let probe = rigProbes.get(url);
  if (!probe) {
    const timeout = AbortSignal.timeout(8000);
    const forget = () => {
      rigProbes.delete(url);
      return false;
    };
    probe = fetch(url, {
      method: "HEAD",
      signal: signal ? AbortSignal.any([timeout, signal]) : timeout,
    })
      .then((r) =>
        r.status >= 500 ? forget() : r.ok && (r.headers.get("content-type") ?? "").includes("json"),
      )
      .catch(forget);
    rigProbes.set(url, probe);
  }
  return probe;
}

/** A rig that answered the probe but would not load: treated as absent from now on. */
export function rigFailed(rigUrl: string): void {
  rigProbes.set(rigUrl, Promise.resolve(false));
}

/** Test hook: forget earlier probes. */
export function forgetRigProbes(): void {
  rigProbes.clear();
}
