/**
 * Which tileset each splat of an incremental splat primitive comes from: its own (source 0,
 * the host) or one of the tilesets it draws as companions (`companions.ts`) -- another date of
 * the site, an inferred layer. The per-cell view rules (`splatCompositor.ts`) draw a splat only
 * where its source owns the cell it sits in, so every slot must know its source.
 *
 * One texture of our own, never the engine's: one byte a slot (R8, `SOURCE_TEXTURE_WIDTH`
 * slots a row), the source's index in the list it was given (`setSources`), `NO_SOURCE` for a
 * slot no listed tileset holds. It is rewritten from the patch's `primitive._tileSlots`
 * whenever that map is replaced (each published batch): a tile whose range is unchanged costs
 * nothing, a tile that left has its range cleared, one that arrived or moved is written, and
 * only the rows touched are uploaded -- the ids-texture pattern of `splatInstances.ts`. A reset
 * of the slots (a model-matrix change, or the texture growing: the patch starts every slot
 * over) reads as every tile having moved, so it needs no case of its own.
 *
 * Not slot-range uniforms (`splatSlots.ts`): those hold eight ranges, and a site's dates hold
 * hundreds of tiles.
 *
 * Why R8 read back as `round(r * 255)` rather than R8UI: this CesiumJS's `Texture` has no
 * internal format for `RED_INTEGER` with unsigned bytes (`PixelFormat.toInternalFormat`), and a
 * normalised byte is exact for 0..255.
 */

import type { SplatPrimitive } from "./splatInternals";

/** Slots a row of the texture holds. */
export const SOURCE_TEXTURE_WIDTH = 4096;
/** A slot no listed tileset holds: never drawn under a rule. */
export const NO_SOURCE = 255;
/** Sources a primitive can tell apart: the host and fifteen companions. */
export const MAX_SOURCES = 16;

/** A texture we own. Structural, so a test can supply a fake. */
export interface SourceTexture {
  copyFrom(options: {
    source: { width: number; height: number; arrayBufferView: ArrayBufferView };
    xOffset?: number;
    yOffset?: number;
  }): void;
  isDestroyed(): boolean;
  destroy(): void;
}

/** How the texture is made. The real one wraps CesiumJS (`splatCompositor.ts`). */
export interface SourceGpu {
  /** An `R8` texture, read normalised, sampled nearest. */
  createR8(context: unknown, width: number, height: number, data: Uint8Array): SourceTexture;
}

interface Placed {
  start: number;
  count: number;
  source: number;
}

/** GLSL: the source of slot `splatIndex`, `NO_SOURCE` past what the texture covers. */
export function sourceGlsl(fn: string, texture: string, covered: string): string {
  const shift = String(Math.log2(SOURCE_TEXTURE_WIDTH));
  return `
int ${fn}(uint splatIndex) {
    if (float(splatIndex) >= ${covered}) {
        return ${String(NO_SOURCE)};
    }
    int i = int(splatIndex);
    return int(texelFetch(${texture}, ivec2(i & ${String(SOURCE_TEXTURE_WIDTH - 1)}, i >> ${shift}), 0).r * 255.0 + 0.5);
}
`;
}

/** The per-slot source texture of one primitive. */
export class SplatSources {
  readonly #gpu: SourceGpu;
  /** Source index by tileset. */
  #index = new Map<unknown, number>();
  #bytes = new Uint8Array(SOURCE_TEXTURE_WIDTH).fill(NO_SOURCE);
  #placed = new Map<object, Placed>();
  /** The `_tileSlots` map last read, and whether the list changed since. */
  #slotsOf: unknown;
  #stale = true;
  #covered = 0;
  #dirty: [number, number] | undefined;
  #texture: SourceTexture | undefined;
  #rows = 0;
  /** Bumped whenever a byte changes. */
  version = 0;

  constructor(gpu: SourceGpu) {
    this.#gpu = gpu;
  }

  /**
   * The tilesets slots are told apart by: index 0 the host's own, then its companions, at most
   * `MAX_SOURCES` (the rest are `NO_SOURCE`). Every slot is written again on the next `sync`
   * when the list changed. Returns whether it did.
   */
  setSources(tilesets: readonly unknown[]): boolean {
    const next = new Map<unknown, number>();
    tilesets.slice(0, MAX_SOURCES).forEach((t, i) => {
      if (!next.has(t)) next.set(t, i);
    });
    const same =
      next.size === this.#index.size && [...next].every(([t, i]) => this.#index.get(t) === i);
    if (same) return false;
    this.#index = next;
    this.#stale = true;
    return true;
  }

  /** The source of `tileset`, or `NO_SOURCE`. */
  sourceOf(tileset: unknown): number {
    return this.#index.get(tileset) ?? NO_SOURCE;
  }

  /** The source slot `slot` has now (the CPU side of the texture). */
  sourceAt(slot: number): number {
    return slot < this.#covered ? (this.#bytes[slot] ?? NO_SOURCE) : NO_SOURCE;
  }

  /** Slots the texture covers: past them the shader reads `NO_SOURCE`. */
  get covered(): number {
    return this.#covered;
  }

  /**
   * Brings the bytes up to `primitive`'s slots (`_tileSlots`). Cheap when nothing changed:
   * the map is replaced, never mutated, so its identity says. Returns whether a byte changed.
   */
  sync(primitive: Pick<SplatPrimitive, "_tileSlots"> | undefined): boolean {
    const slots = primitive?._tileSlots;
    if (slots === this.#slotsOf && !this.#stale) return false;
    this.#slotsOf = slots;
    const before = this.version;
    if (this.#stale) {
      this.#stale = false;
      this.#fill(0, this.#covered, NO_SOURCE);
      this.#placed.clear();
    }
    const next = new Map<object, Placed>();
    let covered = 0;
    for (const [tile, slot] of slots ?? []) {
      next.set(tile, { start: slot.start, count: slot.count, source: this.sourceOf(tile.tileset) });
      covered = Math.max(covered, slot.start + slot.count);
    }
    const same = (a: Placed | undefined, b: Placed | undefined) =>
      a !== undefined && b?.start === a.start && b.count === a.count && b.source === a.source;
    // Ranges that left or moved are cleared before any is written: a slot freed by one tile
    // may be taken by another in the same batch.
    for (const [tile, was] of this.#placed)
      if (!same(was, next.get(tile))) this.#fill(was.start, was.start + was.count, NO_SOURCE);
    this.#reserve(covered);
    for (const [tile, now] of next)
      if (!same(now, this.#placed.get(tile)))
        this.#fill(now.start, now.start + now.count, now.source);
    this.#placed = next;
    if (covered !== this.#covered) {
      this.#covered = covered;
      this.version += 1;
    }
    return this.version !== before;
  }

  /** The texture, created or brought up to date for `context`; undefined before a context. */
  texture(context: unknown): SourceTexture | undefined {
    if (context === undefined) return this.#texture;
    const rows = this.#bytes.length / SOURCE_TEXTURE_WIDTH;
    if (this.#texture === undefined || this.#texture.isDestroyed() || this.#rows !== rows) {
      this.#texture?.destroy();
      this.#texture = this.#gpu.createR8(context, SOURCE_TEXTURE_WIDTH, rows, this.#bytes);
      this.#rows = rows;
      this.#dirty = undefined;
    } else if (this.#dirty) {
      const [first, last] = this.#dirty;
      this.#texture.copyFrom({
        source: {
          width: SOURCE_TEXTURE_WIDTH,
          height: last - first + 1,
          arrayBufferView: this.#bytes.subarray(
            first * SOURCE_TEXTURE_WIDTH,
            (last + 1) * SOURCE_TEXTURE_WIDTH,
          ),
        },
        xOffset: 0,
        yOffset: first,
      });
      this.#dirty = undefined;
    }
    return this.#texture;
  }

  /** Rows waiting to be uploaded, `[first, last]`, for tests. */
  get dirtyRows(): readonly [number, number] | undefined {
    return this.#dirty;
  }

  destroy(): void {
    if (this.#texture && !this.#texture.isDestroyed()) this.#texture.destroy();
    this.#texture = undefined;
  }

  #fill(start: number, end: number, source: number): void {
    if (end <= start) return;
    this.#reserve(end);
    this.#bytes.fill(source, start, end);
    this.version += 1;
    const first = Math.floor(start / SOURCE_TEXTURE_WIDTH);
    const last = Math.floor((end - 1) / SOURCE_TEXTURE_WIDTH);
    const dirty = this.#dirty;
    this.#dirty = dirty ? [Math.min(dirty[0], first), Math.max(dirty[1], last)] : [first, last];
  }

  /** Grows the bytes (by half again, whole rows) to hold `slots`. */
  #reserve(slots: number): void {
    if (slots <= this.#bytes.length) return;
    const rows = Math.ceil(Math.max(slots, this.#bytes.length * 1.5) / SOURCE_TEXTURE_WIDTH);
    const grown = new Uint8Array(rows * SOURCE_TEXTURE_WIDTH).fill(NO_SOURCE);
    grown.set(this.#bytes);
    this.#bytes = grown;
  }
}
