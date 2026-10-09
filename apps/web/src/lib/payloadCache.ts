/**
 * Scan payloads kept in memory for a while once fetched -- a skin (`skin.json` and its
 * `skin.bin`, 75 MB for the Minnetonka tree's limbs), an objects variant's `instances.json`, a
 * fill's `supersedes.json` -- so that switching between methods and back (Compare methods:
 * Today, Limbs, Today, Limbs) is instant after the first time, instead of a download and a
 * parse every time. The browser's own cache cannot be relied on for this: a file that large is
 * past what a disk cache keeps in one entry, and the bucket's own URL says nothing about how
 * long a copy may be kept.
 *
 * Keyed by URL, least recently used first out, within a size cap by bytes (`payloadCacheBytes`:
 * a few hundred MB on a desktop, less on a phone). What is kept is the parsed document, shared
 * by whoever loads the same URL: the documents are read, never written, after they are parsed.
 * A load in flight is shared too; a load that fails is not kept, so the next asks again.
 */

import { isHandheld } from "./detail";

/** What a desktop keeps (bytes). */
export const DESKTOP_PAYLOAD_BYTES = 384 * 1024 * 1024;
/** What a phone or tablet keeps (bytes). */
export const HANDHELD_PAYLOAD_BYTES = 128 * 1024 * 1024;

/** The cap for this device. */
export function payloadCacheBytes(): number {
  return isHandheld() ? HANDHELD_PAYLOAD_BYTES : DESKTOP_PAYLOAD_BYTES;
}

interface Entry {
  value: Promise<unknown>;
  /** Its size once loaded; 0 while it loads. */
  bytes: number;
}

export class PayloadCache {
  private readonly entries = new Map<string, Entry>();
  private total = 0;
  /** Loads that went to the network (or whatever `load` does), and those served from here. */
  misses = 0;
  hits = 0;

  constructor(private readonly maxBytes: () => number = payloadCacheBytes) {}

  /** Bytes held now. */
  get bytes(): number {
    return this.total;
  }

  /** Keys held now, least recently used first (tests, diagnostics). */
  get keys(): string[] {
    return [...this.entries.keys()];
  }

  /**
   * The payload at `key`: kept, in flight, or loaded now by `load`, which resolves with the
   * value and its size in bytes. A value larger than the whole cap is handed back, not kept.
   */
  get<T>(key: string, load: () => Promise<{ value: T; bytes: number }>): Promise<T> {
    const kept = this.entries.get(key);
    if (kept) {
      this.hits += 1;
      // Most recently used: to the back of the line.
      this.entries.delete(key);
      this.entries.set(key, kept);
      return kept.value as Promise<T>;
    }
    this.misses += 1;
    const entry: Entry = { value: Promise.resolve(), bytes: 0 };
    const value = load().then(
      (loaded) => {
        if (this.entries.get(key) === entry) {
          entry.bytes = Math.max(0, loaded.bytes);
          this.total += entry.bytes;
          this.trim(key);
        }
        return loaded.value;
      },
      (error: unknown) => {
        if (this.entries.get(key) === entry) this.entries.delete(key);
        throw error;
      },
    );
    entry.value = value;
    this.entries.set(key, entry);
    return value;
  }

  /** Forgets everything. */
  clear(): void {
    this.entries.clear();
    this.total = 0;
  }

  /**
   * Lets the oldest go until the rest fit. `newest`, just loaded, is not kept at all when it
   * alone is larger than the cap (and nothing is let go for it).
   */
  private trim(newest: string): void {
    const cap = this.maxBytes();
    const added = this.entries.get(newest);
    if (added && added.bytes > cap) {
      this.entries.delete(newest);
      this.total -= added.bytes;
      return;
    }
    for (const [key, entry] of this.entries) {
      if (this.total <= cap) return;
      // Still loading: its size is not known yet, and dropping it would only load it twice.
      if (entry.bytes === 0 || key === newest) continue;
      this.entries.delete(key);
      this.total -= entry.bytes;
    }
  }
}

/** The one cache every scan payload loader shares. */
export const scanPayloads = new PayloadCache();

/** A JSON document's size in memory, roughly: its text, twice (UTF-16 strings and objects). */
export function jsonBytes(text: string): number {
  return text.length * 2;
}
