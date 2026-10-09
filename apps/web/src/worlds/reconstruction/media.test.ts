import { Blob as NodeBlob } from "node:buffer";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { boundedArtifactBlob, validatePlyHeader, validateSpz } from "./media";

afterEach(() => vi.unstubAllGlobals());

describe("untrusted geometry limits", () => {
  it("stops a download that exceeds its limit without a Content-Length", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(3));
        controller.enqueue(new Uint8Array(3));
      },
      cancel() {
        cancelled = true;
      },
    });
    await expect(boundedArtifactBlob(new Response(body), 4)).rejects.toThrow("too large");
    expect(cancelled).toBe(true);
  });

  it("keeps exact bytes for a bounded download", async () => {
    const blob = await boundedArtifactBlob(new Response(new Uint8Array([1, 2, 3])), 4);
    expect(blob.size).toBe(3);
  });

  it("refuses unbounded PLY allocations and list properties before parsing", () => {
    const encode = (value: string) => new TextEncoder().encode(value).buffer;
    expect(() =>
      validatePlyHeader(encode("ply\nformat ascii 1.0\nelement vertex 999999999\nend_header\n")),
    ).toThrow("2 million");
    expect(() =>
      validatePlyHeader(
        encode("ply\nformat ascii 1.0\nelement vertex 1\nproperty list uint float x\nend_header\n"),
      ),
    ).toThrow("list properties");
    expect(
      validatePlyHeader(
        encode("ply\nformat ascii 1.0\nelement vertex 1\nproperty float f_dc_0\nend_header\n"),
      ),
    ).toEqual({ gaussian: true });
  });

  it("rejects a compressed SPZ whose header declares excessive points", async () => {
    vi.stubGlobal("Blob", NodeBlob);
    const header = new Uint8Array(16);
    const view = new DataView(header.buffer);
    view.setUint32(0, 0x5053474e, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, 2_000_001, true);
    const compressed = new Uint8Array(gzipSync(header)).buffer;
    await expect(validateSpz(compressed)).rejects.toThrow("2 million");
  });
});
