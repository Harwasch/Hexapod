/**
 * Telemetry sources: what hands the driver readings (`cesium/telemetry.ts`). One interface for
 * every kind, so the synthetic yard loop the e2e drives goes through the same path a robot's
 * stream does:
 *
 * - **synthetic**: a deterministic path sampled on the driver's clock (`@twin/world`
 *   `syntheticReading`), delivered with a set delay, jitter and dropouts; `pump(now)` releases
 *   what has arrived by `now`. The same clock gives the same readings.
 * - **sse** / **websocket**: JSON messages from a server (`parsePoseMessages`), each stamped on
 *   arrival with the driver's clock.
 */

import {
  poseFromScan,
  syntheticReading,
  type PoseFrame,
  type ScanFrame,
  type SyntheticTelemetry,
} from "@twin/world";

import {
  parsePoseMessages,
  type StreamSample,
  type SyntheticSourceConfig,
  type TelemetrySourceConfig,
} from "./telemetry";

/** Called once per reading, with the local time it arrived (ms, the driver's clock). */
export type SampleListener = (sample: StreamSample, arrivalMs: number) => void;

export interface TelemetrySource {
  readonly id: string;
  readonly kind: TelemetrySourceConfig["kind"];
  subscribe(listener: SampleListener): () => void;
  /** Called every frame with the driver's clock: a polled source releases what has arrived. */
  pump?(nowMs: number): void;
  /** Forgets what it sent: the next pump starts again from the clock (a replay). */
  reset?(): void;
  close(): void;
}

/** What a source may need from where it runs. */
export interface SourceContext {
  /** The driver's clock, ms. */
  readonly clock: () => number;
  /** Where the scan sits (a synthetic source emitting in `ecef` or `geodetic` needs it). */
  scan: ScanFrame | undefined;
}

/** How far back a synthetic source's first pump (or a jump in the clock) reaches, ms. */
const SYNTHETIC_CATCH_UP_MS = 10_000;

class Listeners {
  readonly #set = new Set<SampleListener>();

  add(listener: SampleListener): () => void {
    this.#set.add(listener);
    return () => this.#set.delete(listener);
  }

  emit(sample: StreamSample, arrival: number): void {
    for (const listener of [...this.#set]) listener(sample, arrival);
  }

  clear(): void {
    this.#set.clear();
  }
}

/** A deterministic path on the driver's clock. */
export class SyntheticSource implements TelemetrySource {
  readonly kind = "synthetic";
  readonly id: string;
  readonly config: SyntheticSourceConfig;
  readonly #listeners = new Listeners();
  readonly #scan: ScanFrame | undefined;
  /** The next reading index to release, once started. */
  #next: number | undefined;
  #lastNow = -Infinity;
  #muted = false;

  constructor(id: string, config: SyntheticSourceConfig, scan: ScanFrame | undefined) {
    this.id = id;
    this.config = config;
    this.#scan = scan;
  }

  /** Silences the source (a dropout made on demand), or lets it speak again. */
  set muted(value: boolean) {
    this.#muted = value;
  }

  get muted(): boolean {
    return this.#muted;
  }

  subscribe(listener: SampleListener): () => void {
    return this.#listeners.add(listener);
  }

  pump(nowMs: number): void {
    const c = this.config;
    const period = 1000 / c.rateHz;
    const start = c.startMs ?? 0;
    if (nowMs < this.#lastNow) this.#next = undefined;
    this.#lastNow = nowMs;
    // Readings exist from the path's start; a first pump (or a jump) reaches back a little.
    const earliest = Math.max(0, Math.ceil((nowMs - SYNTHETIC_CATCH_UP_MS - start) / period));
    let k = Math.max(this.#next ?? earliest, earliest);
    const telemetry: SyntheticTelemetry = {
      path: c.path,
      rateHz: c.rateHz,
      delayMs: c.delayMs,
      jitterMs: c.jitterMs,
      dropouts: c.dropouts,
      startMs: start,
    };
    for (;;) {
      const t = start + k * period;
      if (t > nowMs) break;
      const reading = syntheticReading(telemetry, k);
      if (reading !== null) {
        if (reading.arrival > nowMs) break;
        const pose = poseFromScan(reading.pose, c.frame, this.#scan);
        if (pose && !this.#muted) {
          this.#listeners.emit({ t: reading.t, frame: c.frame, ...pose }, reading.arrival);
        }
      }
      k += 1;
    }
    this.#next = k;
  }

  reset(): void {
    this.#next = undefined;
    this.#lastNow = -Infinity;
  }

  close(): void {
    this.#listeners.clear();
  }
}

/** The part of `EventSource` and `WebSocket` a stream source uses. */
interface MessageChannel {
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: ((event: unknown) => void) | null;
  close(): void;
}

export type ChannelFactory = (kind: "sse" | "websocket", url: string) => MessageChannel;

const browserChannel: ChannelFactory = (kind, url) =>
  (kind === "sse" ? new EventSource(url) : new WebSocket(url)) as unknown as MessageChannel;

/** Readings pushed by a server, JSON a message. */
export class StreamSource implements TelemetrySource {
  readonly kind: "sse" | "websocket";
  readonly id: string;
  readonly #listeners = new Listeners();
  readonly #channel: MessageChannel;
  #errors = 0;

  constructor(
    id: string,
    kind: "sse" | "websocket",
    url: string,
    frame: PoseFrame,
    clock: () => number,
    channel: ChannelFactory = browserChannel,
  ) {
    this.id = id;
    this.kind = kind;
    this.#channel = channel(kind, url);
    this.#channel.onmessage = (event) => {
      const arrival = clock();
      let raw: unknown;
      try {
        raw = typeof event.data === "string" ? JSON.parse(event.data) : event.data;
      } catch {
        return;
      }
      for (const sample of parsePoseMessages(raw, frame)) this.#listeners.emit(sample, arrival);
    };
    this.#channel.onerror = () => {
      this.#errors += 1;
    };
  }

  /** Transport errors seen (an EventSource reconnects by itself). */
  get errors(): number {
    return this.#errors;
  }

  subscribe(listener: SampleListener): () => void {
    return this.#listeners.add(listener);
  }

  close(): void {
    this.#listeners.clear();
    this.#channel.onmessage = null;
    this.#channel.onerror = null;
    this.#channel.close();
  }
}

/** A source for `config`. */
export function createTelemetrySource(
  id: string,
  config: TelemetrySourceConfig,
  context: SourceContext,
  channel?: ChannelFactory,
): TelemetrySource {
  if (config.kind === "synthetic") return new SyntheticSource(id, config, context.scan);
  return new StreamSource(id, config.kind, config.url, config.frame, context.clock, channel);
}
