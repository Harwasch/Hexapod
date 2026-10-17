/**
 * Map tiles that wait out a busy server, for the Time Lantern's dated imagery.
 *
 * The USGS map tiler has a few sheet readers and answers 503 with Retry-After when every one is
 * reading and its queue is full (apps/api/app/services/usgs_topo_tiles.py); a public tiler may
 * answer 429. Cesium would mark such a tile failed for as long as the layer lives, and the lens
 * would show "no record" where there is one. Here a busy tile is asked again after the time the
 * server gave (a little more, so a sweep's tiles do not all come back at once), a few times,
 * while Cesium shows the coarser tile it has. Cesium's own error retry asks again at once, which
 * a busy server only refuses again.
 */
import {
  NeverTileDiscardPolicy,
  Request,
  RequestErrorEvent,
  RequestType,
  UrlTemplateImageryProvider,
  type ImageryTypes,
} from "cesium";

/** How long to wait when a busy answer does not say (s): the USGS tiler's own Retry-After. */
const DEFAULT_WAIT_S = 2;
/** A Retry-After is honoured up to this (s): past it the tile is left to Cesium's coarser one. */
const MAX_WAIT_S = 30;
/** Times a busy tile is asked again before it counts as failed. */
const ATTEMPTS = 3;
/**
 * While the browser's requests to that server are all in flight (Cesium's per-server limit), a
 * tile due again looks for a free one this often (ms), for at most this long: that is the
 * browser's queue, not the server's answer, so it is not one of the attempts.
 */
const THROTTLED_MS = 250;
const MAX_THROTTLED_MS = 20_000;

/** How long a failed tile request asks to wait (s), or null when it was not "busy". */
export function busyWait(error: unknown): number | null {
  if (!(error instanceof RequestErrorEvent)) return null;
  if (error.statusCode !== 503 && error.statusCode !== 429) return null;
  const headers = (error.responseHeaders ?? {}) as Record<string, unknown>;
  const raw = Object.entries(headers).find(([name]) => name.toLowerCase() === "retry-after")?.[1];
  const seconds = Number(raw);
  if (!Number.isFinite(seconds) || seconds < 0) return DEFAULT_WAIT_S;
  return Math.min(Math.max(seconds, 1), MAX_WAIT_S);
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export class BusyAwareImageryProvider extends UrlTemplateImageryProvider {
  constructor(options: UrlTemplateImageryProvider.ConstructorOptions) {
    // A discard policy makes Cesium fetch tiles as blobs, whose errors carry the HTTP status
    // and headers (an <img> error says neither); this one discards nothing.
    super({ ...options, tileDiscardPolicy: new NeverTileDiscardPolicy() });
  }

  override requestImage(
    x: number,
    y: number,
    level: number,
    request?: Request,
  ): Promise<ImageryTypes> | undefined {
    return super
      .requestImage(x, y, level, request)
      ?.catch((error: unknown) => this.again(x, y, level, error));
  }

  private async again(x: number, y: number, level: number, error: unknown): Promise<ImageryTypes> {
    let last = error;
    for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
      const wait = busyWait(last);
      if (wait === null) throw last;
      await sleep(wait * 1000 * (1 + Math.random() * 0.5));
      let next = this.ask(x, y, level);
      for (let queued = 0; !next && queued < MAX_THROTTLED_MS; queued += THROTTLED_MS) {
        await sleep(THROTTLED_MS);
        next = this.ask(x, y, level);
      }
      if (!next) throw last;
      try {
        return await next;
      } catch (failed) {
        last = failed;
      }
    }
    throw last;
  }

  /** The tile asked again, as Cesium asks for imagery; undefined while that server is full. */
  private ask(x: number, y: number, level: number): Promise<ImageryTypes> | undefined {
    return super.requestImage(
      x,
      y,
      level,
      new Request({ throttle: false, throttleByServer: true, type: RequestType.IMAGERY }),
    );
  }
}
