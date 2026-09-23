/**
 * The project used when no catalog site is active: plans anywhere on Earth from areas the
 * operator draws, with no machines until a robot bridge registers them.
 */

import type { Project } from "./types";

export const ANYWHERE_PROJECT_ID = "anywhere";

export function anywhereProject(): Project {
  return {
    id: ANYWHERE_PROJECT_ID,
    siteId: null,
    name: "Anywhere",
    meta: "plans anywhere on Earth · draw an area to start",
    simulated: false,
    machines: [],
    zones: [],
    plans: [],
    agent: {
      headline: "Agent idle",
      summary: "Nothing running",
      footer: "Draw an area or use the current view, state a goal, and the agent drafts the plan.",
      actions: [],
    },
    fleetStats: [
      { value: "0", label: "Machines" },
      { value: "0", label: "Areas" },
    ],
    fleetNote:
      "No fleet is registered. Plans draft without assignments until a robot bridge brings machines.",
    workLog: [],
    feeds: [],
  };
}
