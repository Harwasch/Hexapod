import { describe, expect, it } from "vitest";
import { createReconstructionBundle, manifestFor } from "./bundle";

function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(new Error("Could not read test blob"));
    reader.readAsArrayBuffer(blob);
  });
}

describe("portable reconstruction source export", () => {
  it("creates a standard tar with recoverable manifest and exact media bytes", async () => {
    const media = new Uint8Array([0, 255, 13, 10, 42]);
    const blob = createReconstructionBundle(
      [
        {
          id: "private-id",
          name: "/Users/alice/private.png",
          blob: new Blob([media], { type: "image/png" }),
          timestampMs: 450,
        },
      ],
      "matrix-game-3",
    );
    const bytes = await bytesOf(blob);
    const decoder = new TextDecoder();
    const names: string[] = [];
    const contents: Uint8Array[] = [];
    for (let offset = 0; bytes[offset];) {
      const header = bytes.slice(offset, offset + 512);
      names.push(decoder.decode(header.slice(0, 100)).replace(/\0.*$/, ""));
      const size = parseInt(decoder.decode(header.slice(124, 136)), 8);
      const checksum = parseInt(decoder.decode(header.slice(148, 154)), 8);
      header.fill(32, 148, 156);
      expect(header.reduce((sum, value) => sum + value, 0)).toBe(checksum);
      contents.push(bytes.slice(offset + 512, offset + 512 + size));
      offset += 512 + Math.ceil(size / 512) * 512;
    }
    expect(names).toEqual(["manifest.json", "sources/00001.png"]);
    expect(Array.from(contents[1] ?? [])).toEqual(Array.from(media));
    const manifest = decoder.decode(contents[0]);
    expect(manifest).toContain('"timestampMs": 450');
    expect(manifest).toContain('"geometry": "unverified"');
    expect(manifest).not.toContain("alice");
    expect(manifest).not.toContain("private-id");
    expect(bytes.length % 512).toBe(0);
    expect(bytes.slice(-1024).every((value) => value === 0)).toBe(true);
  });

  it("never fabricates camera poses, and preserves native poses when present", () => {
    const source = {
      id: "frame",
      name: "frame",
      blob: new Blob(["image"], { type: "image/jpeg" }),
    };
    expect(manifestFor([source]).frames[0]).not.toHaveProperty("cameraPose");
    expect(manifestFor([{ ...source, cameraPose: [1, 0, 0] }]).frames[0]?.cameraPose).toEqual([
      1, 0, 0,
    ]);
    expect(() => createReconstructionBundle([])).toThrow("Select a replay");
  });
});
