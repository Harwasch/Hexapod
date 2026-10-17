import { crc32 as nodeCrc32 } from "node:zlib";

import { describe, expect, it } from "vitest";

import { crc32, zipStore } from "@/lib/zipStore";

/** Reads a store-only zip back with nothing but its own headers. */
function readZip(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const end = bytes.length - 22;
  expect(view.getUint32(end, true)).toBe(0x06054b50);
  const count = view.getUint16(end + 10, true);
  const centralSize = view.getUint32(end + 12, true);
  let at = view.getUint32(end + 16, true);
  expect(at + centralSize).toBe(end);
  const decoder = new TextDecoder();
  const entries = [];
  for (let i = 0; i < count; i++) {
    expect(view.getUint32(at, true)).toBe(0x02014b50);
    const crc = view.getUint32(at + 16, true);
    const size = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const external = view.getUint32(at + 38, true);
    const offset = view.getUint32(at + 42, true);
    const name = decoder.decode(bytes.subarray(at + 46, at + 46 + nameLength));
    expect(view.getUint32(offset, true)).toBe(0x04034b50);
    expect(view.getUint16(offset + 8, true)).toBe(0); // stored
    expect(view.getUint16(offset + 6, true) & 0x0800).toBe(0x0800); // UTF-8 names
    const localName = view.getUint16(offset + 26, true);
    const start = offset + 30 + localName;
    entries.push({ name, crc, external, data: bytes.slice(start, start + size) });
    at += 46 + nameLength;
  }
  return { count, entries };
}

describe("zipStore", () => {
  it("writes the KMZ layout: a directory and two stored files with true CRCs", () => {
    const encoder = new TextEncoder();
    const template = encoder.encode("<kml>template · é</kml>");
    const waylines = encoder.encode("<kml>waylines</kml>".repeat(50));
    const zip = zipStore(
      [
        { name: "wpmz/", data: null },
        { name: "wpmz/template.kml", data: template },
        { name: "wpmz/waylines.wpml", data: waylines },
      ],
      new Date(2026, 9, 16, 12, 30, 10),
    );
    const { count, entries } = readZip(zip);
    expect(count).toBe(3);
    expect(entries.map((e) => e.name)).toEqual([
      "wpmz/",
      "wpmz/template.kml",
      "wpmz/waylines.wpml",
    ]);
    expect(entries[0]?.external).toBe(0x10);
    expect(Array.from(entries[1]?.data ?? [])).toEqual(Array.from(template));
    expect(Array.from(entries[2]?.data ?? [])).toEqual(Array.from(waylines));
    expect(entries[1]?.crc).toBe(nodeCrc32(template));
    expect(entries[2]?.crc).toBe(nodeCrc32(waylines));
  });

  it("computes CRC-32 as zlib does", () => {
    const sample = new TextEncoder().encode("The quick brown fox jumps over the lazy dog");
    expect(crc32(sample)).toBe(0x414fa339);
    expect(crc32(new Uint8Array(0))).toBe(0);
  });

  it("refuses more entries than a zip without zip64 can list", () => {
    const many = Array.from({ length: 65_536 }, (_, i) => ({ name: `${i}`, data: null }));
    expect(() => zipStore(many)).toThrow(/65,535/);
  });
});
