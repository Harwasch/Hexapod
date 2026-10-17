/**
 * Leaving the plan card takes its draft and its dress rehearsal off the map at once. The drawer
 * animates out and the card unmounts only when that ends (seconds, on a busy main thread), so
 * the lanes, ghosts and hands go as the exit starts, and come back if the card does.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { AnimatePresence, motion } from "motion/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import type { PlanDraft } from "@twin/contracts";

import { api } from "@/api/client";
import { PlanCard } from "@/features/mission/PlanCard";
import { demoRanchProject } from "@/missions/demo";
import type { Project } from "@/missions/types";
import { useMission } from "@/state/mission";

const scene = vi.hoisted(() => ({
  mission: {
    showPlan: vi.fn(),
    flyToZones: vi.fn(),
    flyToProject: vi.fn(),
    restoreLanes: vi.fn(),
    setSelectedZone: vi.fn(),
    machineColor: () => ({ toCssColorString: () => "#7fd8c0" }),
  },
  rehearsal: { stop: vi.fn(), onFrame: () => () => undefined },
  events: { on: () => () => undefined },
  areas: { cancelPick: vi.fn(), edit: vi.fn(), update: vi.fn(), editing: null },
  measurement: { remove: vi.fn() },
}));

vi.mock("@/cesium/SceneContext", () => ({ useScene: () => scene }));

const mow: PlanDraft = {
  title: "Mow Z-08 with one mower",
  objective: "Mow Z-08 with one mower",
  zoneIds: ["Z-08"],
  machineIds: ["TR-04"],
  cadence: "once",
  startDate: "2026-10-17",
  endDate: "2026-10-22",
  estimates: { acres: 44, machineHours: 19, calendarDays: 5 },
  steps: [],
  assumptions: [],
  risks: [],
  questions: [],
  clarifications: [],
  source: "rules",
  model: null,
  note: "",
};

const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });

/** The Plan drawer as PlansPanel draws it: in and out with a (here, long) animation. */
function Drawer({ open, project }: { open: boolean; project: Project }) {
  return (
    <QueryClientProvider client={client}>
      <AnimatePresence>
        {open && (
          <motion.div key="plans" exit={{ opacity: 0 }} transition={{ duration: 30 }}>
            <PlanCard project={project} />
          </motion.div>
        )}
      </AnimatePresence>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.spyOn(api, "GET").mockResolvedValue({
    data: { configured: false, provider: "rules", model: null },
    response: new Response(),
  });
});

afterEach(() => {
  cleanup();
  useMission.setState({ composer: null });
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

it("takes the draft and its rehearsal off the map as the drawer starts to close", () => {
  const project = demoRanchProject();
  useMission.getState().setProject(project);
  useMission.getState().openComposer({ goal: mow.objective, zoneIds: mow.zoneIds });
  useMission.getState().updateComposer({ status: "ready", draft: mow });
  const { rerender } = render(<Drawer open project={project} />);
  expect(screen.getByTestId("plan-review")).toBeInTheDocument();
  expect(scene.mission.showPlan).toHaveBeenLastCalledWith(
    expect.objectContaining({ zones: [expect.objectContaining({ zoneId: "Z-08" })] }),
  );
  scene.rehearsal.stop.mockClear();

  rerender(<Drawer open={false} project={project} />);
  // Still on screen, on its way out: nothing of it is left on the map.
  expect(screen.getByTestId("plan-review")).toBeInTheDocument();
  expect(scene.rehearsal.stop).toHaveBeenCalled();
  expect(scene.mission.showPlan).toHaveBeenLastCalledWith(null);

  // Back before it has gone: the draft is drawn again, without flying the camera there twice.
  scene.mission.flyToZones.mockClear();
  rerender(<Drawer open project={project} />);
  expect(scene.mission.showPlan).toHaveBeenLastCalledWith(
    expect.objectContaining({ zones: [expect.objectContaining({ zoneId: "Z-08" })] }),
  );
  expect(scene.mission.flyToZones).not.toHaveBeenCalled();
});
