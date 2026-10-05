import { describe, expect, it } from "vitest";

import { builtinDemoSite } from "@/api/fallback";
import { buildDraftRequest } from "@/missions/planDraft";
import { SITE_ZONE_ID, siteProject } from "@/missions/siteProject";
import { useMission } from "@/state/mission";

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

  it("takes its site's new name without dropping what is selected or open", () => {
    const site = builtinDemoSite();
    const mission = useMission.getState();
    mission.setProject(siteProject(site));
    mission.select({ kind: "zone", id: SITE_ZONE_ID });
    mission.openPlan("plan-1");

    mission.refreshProject(siteProject({ ...site, name: "North orchard" }));
    expect(useMission.getState()).toMatchObject({
      project: { name: "North orchard", zones: [{ name: "North orchard boundary" }] },
      selection: { kind: "zone", id: SITE_ZONE_ID },
      planId: "plan-1",
    });
    // Another site's project is not this one refreshed.
    mission.refreshProject(siteProject({ ...site, id: "other", name: "Elsewhere" }));
    expect(useMission.getState().project?.name).toBe("North orchard");
    mission.setProject(null);
  });
});
