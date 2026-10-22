import { describe, expect, it } from "vitest";

import { LOOKS, lookById } from "./looks";

describe("landing looks", () => {
  it("offers the fixed line drawing first, and it is what an unknown or missing id gets", () => {
    expect(LOOKS[0]?.id).toBe("line");
    expect(lookById(null).id).toBe("line");
    expect(lookById("no-such-look").id).toBe("line");
    expect(lookById("golden").name).toBe("Golden hour");
  });

  it("gives every look its own id, and a dark look a stage to put light copy on", () => {
    expect(new Set(LOOKS.map((look) => look.id)).size).toBe(LOOKS.length);
    for (const look of LOOKS.filter((one) => one.dark)) expect(look.stage).toMatch(/^#/);
  });
});
