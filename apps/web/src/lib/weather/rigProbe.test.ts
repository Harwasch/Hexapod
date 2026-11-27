import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { forgetRigProbes, probeRig, rigFailed } from "./rigProbe";

const TILESET = "https://tiles.example/grove/tileset.json?sig=1";
const RIG = "https://tiles.example/grove/articulation.json?sig=1";
const head = vi.fn<typeof fetch>();
const json = () =>
  new Response(null, { status: 200, headers: { "content-type": "application/json" } });

beforeEach(() => {
  forgetRigProbes();
  head.mockReset();
  vi.stubGlobal("fetch", head);
});
afterEach(() => vi.unstubAllGlobals());

describe("baked rig probe", () => {
  it("asks beside the tiles, signed query kept, and keeps a real answer", async () => {
    head.mockResolvedValue(new Response(null, { status: 404 }));
    expect(await probeRig(TILESET)).toBe(false);
    expect(await probeRig(TILESET)).toBe(false);
    expect(head).toHaveBeenCalledTimes(1);
    expect(head.mock.calls[0]?.[0]).toBe(RIG);
  });

  it("asks again after a server error or a cancelled request", async () => {
    head.mockResolvedValueOnce(new Response(null, { status: 503 }));
    expect(await probeRig(TILESET)).toBe(false);
    head.mockRejectedValueOnce(new DOMException("aborted", "AbortError"));
    expect(await probeRig(TILESET)).toBe(false);
    head.mockResolvedValueOnce(json());
    expect(await probeRig(TILESET)).toBe(true);
    expect(head).toHaveBeenCalledTimes(3);
  });

  it("a rig that would not load counts as absent", async () => {
    head.mockResolvedValue(json());
    rigFailed(RIG);
    expect(await probeRig(TILESET)).toBe(false);
    expect(head).not.toHaveBeenCalled();
  });
});
