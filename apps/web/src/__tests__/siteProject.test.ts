import { describe, expect, it } from "vitest";

import { builtinDemoSite } from "@/api/fallback";
import { buildDraftRequest } from "@/missions/planDraft";
import { SITE_ZONE_ID, siteProject } from "@/missions/siteProject";

describe("siteProject", () => {
  it("makes a plannable project from any catalog site", () => {
    const project = siteProject(builtinDemoSite());
    expect(project.siteId).toBe(builtinDemoSite().id);
    expect(project.machines).toEqual([]);
    expect(project.zones.map((z) => z.id)).toEqual([SITE_ZONE_ID]);
    expect(project.zones[0]?.acres).toBeGreaterThan(0);
    expect(project.simulated).toBe(false);
    const request = buildDraftRequest(project, { goal: "Mow the whole site this week" });
    expect(request.zones?.[0]?.id).toBe(SITE_ZONE_ID);
    expect(request.machines).toEqual([]);
  });
});
