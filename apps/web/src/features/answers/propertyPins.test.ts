import { describe, expect, it, vi } from "vitest";
import { Cartesian3, EntityCollection, JulianDate } from "cesium";
import { addAnswerPins } from "./answerPinLayer";
import { loadAnswerPins, parseAnswerPin } from "./propertyPins";
import { fetchApi } from "@/api/coldStart";
vi.mock("@/api/coldStart", () => ({ fetchApi: vi.fn() }));
const row = {
  id: "e",
  siteId: "s",
  scanId: "c",
  observedAt: "2026-01-01T00:00:00Z",
  uncertaintyM: 2,
  entity: { label: "Oak", anchor: { geo: { lat: 45, lon: -122, alt: 100 } } },
};

describe("property answer pins", () => {
  it("refuses unlocated, invalid and wrong-property observations", () => {
    expect(parseAnswerPin(row, "other")).toBeNull();
    expect(parseAnswerPin({ ...row, entity: { label: "Oak", anchor: {} } }, "s")).toBeNull();
    for (const geo of [
      { lat: 91, lon: 0, alt: 0 },
      { lat: 0, lon: NaN, alt: 0 },
    ])
      expect(parseAnswerPin({ ...row, entity: { label: "Oak", anchor: { geo } } }, "s")).toBeNull();
    expect(parseAnswerPin(row, "s")?.geo.alt).toBe(100);
  });
  it("reads pages with cancellation and refuses a looping cursor", async () => {
    const signal = new AbortController().signal;
    vi.mocked(fetchApi)
      .mockResolvedValueOnce(new Response(JSON.stringify({ items: [row], nextOffset: 1 })))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ items: [{ ...row, id: "e2" }], nextOffset: null })),
      );
    expect(await loadAnswerPins("s", signal)).toHaveLength(2);
    expect(fetchApi).toHaveBeenLastCalledWith(expect.stringContaining("offset=1"), { signal });
    vi.mocked(fetchApi).mockResolvedValueOnce(
      new Response(JSON.stringify({ items: [], nextOffset: 0 })),
    );
    await expect(loadAnswerPins("s", signal)).rejects.toThrow("cursor");
    vi.mocked(fetchApi).mockResolvedValueOnce(new Response("", { status: 503 }));
    await expect(loadAnswerPins("s", signal)).rejects.toThrow("503");
  });
  it("places WGS84 pins and removes only its own entities", () => {
    const entities = new EntityCollection();
    entities.add({ id: "scan" });
    const pin = parseAnswerPin(row, "s")!;
    const cleanup = addAnswerPins(entities, [pin]);
    const rendered = entities.getById("property-answer:s:e")!;
    expect(rendered.position?.getValue(JulianDate.now())).toEqual(
      Cartesian3.fromDegrees(-122, 45, 100),
    );
    expect(rendered.properties?.getValue(JulianDate.now())).toMatchObject({ captureId: "c" });
    cleanup();
    cleanup();
    expect(entities.values.map((e) => e.id)).toEqual(["scan"]);
  });
});
