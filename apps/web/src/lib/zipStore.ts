/**
 * A store-only ZIP writer (PKZIP method 0, no compression): enough for a KMZ, whose few small
 * XML files gain little from deflate. No dependency: fflate is only a transitive package here
 * and pnpm does not let the app import it. Entries keep their order; a `null` body is a
 * directory. Names are UTF-8 (general purpose flag bit 11).
 */

export interface ZipEntry {
  name: string;
  /** The file's bytes; null for a directory (its name should end in "/"). */
  data: Uint8Array | null;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** CRC-32 (IEEE 802.3), as zip and zlib compute it. */
export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) crc = (CRC_TABLE[(crc ^ byte) & 0xff] ?? 0) ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** MS-DOS time and date fields, local time, 2-second resolution, from 1980. */
function dosDateTime(date: Date): { time: number; day: number } {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2),
    day: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

const LOCAL_HEADER = 0x04034b50;
const CENTRAL_HEADER = 0x02014b50;
const END_OF_CENTRAL = 0x06054b50;
const UTF8_NAMES = 0x0800;
const VERSION = 20;
const MAX_ENTRIES = 0xffff;
const MAX_BYTES = 0xffffffff;

export function zipStore(entries: readonly ZipEntry[], date = new Date()): Uint8Array {
  if (entries.length > MAX_ENTRIES)
    throw new Error("A store-only zip holds at most 65,535 entries.");
  const encoder = new TextEncoder();
  const { time, day } = dosDateTime(date);
  const prepared = entries.map((entry) => {
    const name = encoder.encode(entry.name);
    const data = entry.data ?? new Uint8Array(0);
    return { name, data, directory: entry.data === null, crc: crc32(data) };
  });
  const localSize = prepared.reduce((sum, e) => sum + 30 + e.name.length + e.data.length, 0);
  const centralSize = prepared.reduce((sum, e) => sum + 46 + e.name.length, 0);
  const total = localSize + centralSize + 22;
  if (total > MAX_BYTES) throw new Error("A store-only zip holds at most 4 GiB.");
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  let at = 0;
  const offsets: number[] = [];
  for (const e of prepared) {
    offsets.push(at);
    view.setUint32(at, LOCAL_HEADER, true);
    view.setUint16(at + 4, VERSION, true);
    view.setUint16(at + 6, UTF8_NAMES, true);
    view.setUint16(at + 8, 0, true); // method 0: stored
    view.setUint16(at + 10, time, true);
    view.setUint16(at + 12, day, true);
    view.setUint32(at + 14, e.crc, true);
    view.setUint32(at + 18, e.data.length, true);
    view.setUint32(at + 22, e.data.length, true);
    view.setUint16(at + 26, e.name.length, true);
    view.setUint16(at + 28, 0, true);
    out.set(e.name, at + 30);
    out.set(e.data, at + 30 + e.name.length);
    at += 30 + e.name.length + e.data.length;
  }
  const centralStart = at;
  prepared.forEach((e, index) => {
    view.setUint32(at, CENTRAL_HEADER, true);
    view.setUint16(at + 4, VERSION, true); // made by: MS-DOS, 2.0
    view.setUint16(at + 6, VERSION, true);
    view.setUint16(at + 8, UTF8_NAMES, true);
    view.setUint16(at + 10, 0, true);
    view.setUint16(at + 12, time, true);
    view.setUint16(at + 14, day, true);
    view.setUint32(at + 16, e.crc, true);
    view.setUint32(at + 20, e.data.length, true);
    view.setUint32(at + 24, e.data.length, true);
    view.setUint16(at + 28, e.name.length, true);
    view.setUint16(at + 30, 0, true); // extra
    view.setUint16(at + 32, 0, true); // comment
    view.setUint16(at + 34, 0, true); // disk
    view.setUint16(at + 36, 0, true); // internal attributes
    view.setUint32(at + 38, e.directory ? 0x10 : 0, true); // external: MS-DOS directory bit
    view.setUint32(at + 42, offsets[index] ?? 0, true);
    out.set(e.name, at + 46);
    at += 46 + e.name.length;
  });
  view.setUint32(at, END_OF_CENTRAL, true);
  view.setUint16(at + 4, 0, true);
  view.setUint16(at + 6, 0, true);
  view.setUint16(at + 8, prepared.length, true);
  view.setUint16(at + 10, prepared.length, true);
  view.setUint32(at + 12, at - centralStart, true);
  view.setUint32(at + 16, centralStart, true);
  view.setUint16(at + 20, 0, true);
  return out;
}
