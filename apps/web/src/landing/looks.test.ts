import { describe, expect, it } from "vitest";

import { LOOKS, lookById, themed } from "./looks";

describe("landing looks", () => {
  it("offers the fixed line drawing first, and it is what an unknown or missing id gets", () => {
    expect(LOOKS[0]?.id).toBe("line");
    expect(lookById(null).id).toBe("line");
    expect(lookById("no-such-look").id).toBe("line");
    expect(lookById("golden").name).toBe("Golden hour");
  });

  it("gives every look its own id, and a stage in both themes or in neither", () => {
    expect(new Set(LOOKS.map((look) => look.id)).size).toBe(LOOKS.length);
    for (const look of LOOKS) expect(Boolean(look.light.stage)).toBe(Boolean(look.dark.stage));
  });

  it("hands the scenes both versions, each with the look's camera", () => {
    const miniature = themed(lookById("miniature"));
    expect(miniature.light.camera).toEqual({ zoom: 0.82, tilt: 12 });
    expect(miniature.dark.camera).toEqual({ zoom: 0.82, tilt: 12 });
    expect(miniature.dark.light?.glow).toBeGreaterThan(0);
  });
});
