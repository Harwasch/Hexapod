import { describe, expect, it } from "vitest";

import {
  parseFootprintText,
  validateAssetId,
  validateDate,
  validateHttpUrl,
  validateName,
} from "@/lib/validation";

describe("Add Data validation", () => {
  it("validates ion asset ids", () => {
    expect(validateAssetId("4547222")).toEqual({ ok: true, value: 4547222 });
    expect(validateAssetId("12abc").ok).toBe(false);
    expect(validateAssetId("0").ok).toBe(false);
  });

  it("validates URLs, schemes and placeholders", () => {
    expect(validateHttpUrl("https://example.com/tileset.json", { endsWith: [".json"] }).ok).toBe(
      true,
    );
    expect(validateHttpUrl("ftp://example.com/a").ok).toBe(false);
    expect(validateHttpUrl("javascript:alert(1)").ok).toBe(false);
    expect(validateHttpUrl("https://user:pw@example.com/a").ok).toBe(false);
    expect(
      validateHttpUrl("https://example.com/{z}/{x}.png", { placeholders: ["{z}", "{x}", "{y}"] }),
    ).toMatchObject({ ok: false, error: "The template must contain {y}." });
    expect(validateHttpUrl("https://example.com/a.geojson", { endsWith: [".json"] }).ok).toBe(
      false,
    );
  });

  it("validates names, dates and footprints", () => {
    expect(validateName(" a ").ok).toBe(false);
    expect(validateName("Orchard")).toEqual({ ok: true, value: "Orchard" });
    expect(validateDate("")).toEqual({ ok: true, value: null });
    expect(validateDate("not a date").ok).toBe(false);
    expect(validateDate("2025-05-01").ok).toBe(true);
    expect(parseFootprintText("").ok).toBe(false);
    expect(parseFootprintText("{").ok).toBe(false);
    expect(parseFootprintText('{"type":"Point","coordinates":[0,0]}')).toMatchObject({ ok: false });
    const polygon = '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}';
    expect(parseFootprintText(polygon).ok).toBe(true);
  });
});
