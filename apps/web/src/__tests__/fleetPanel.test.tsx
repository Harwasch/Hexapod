/**
 * The Fleet drawer (`features/mission/FleetPanel.tsx`): the machines' table comes first in the
 * scrolling body and what is scheduled follows it, so the rows stay in view however many plans
 * are on the books; each scheduled plan names its crew, or says it has none, or (a demo plan,
 * whose crew is in its facts) says nothing about it.
 */

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";

import { FleetPanel } from "@/features/mission/FleetPanel";
import { demoRanchProject } from "@/missions/demo";
import type { Plan } from "@/missions/types";
import { useMission } from "@/state/mission";

beforeEach(() => {
  const project = demoRanchProject();
  const thistle = project.plans.find((p) => p.id === "thistle")!;
  const plans: Plan[] = [
    ...project.plans,
    { ...thistle, id: "console-crewed", title: "Mow Z-21 weekly", machineIds: ["TR-04", "TR-12"] },
    { ...thistle, id: "console-bare", title: "Walk the fence", machineIds: [] },
    { ...thistle, id: "done", title: "Mowed last spring", state: "Done", machineIds: [] },
  ];
  useMission.setState({ project: { ...project, plans }, view: "fleet", selection: null });
});

afterEach(() => {
  cleanup();
  useMission.setState({ project: null, view: "map" });
});

it("puts the machines first in the scrolling body, and what is scheduled after them", () => {
  const { container } = render(<FleetPanel />);
  const body = container.querySelector(".mc-window__body")!;
  const table = within(body as HTMLElement).getByRole("table", { name: "Machines" });
  const scheduled = within(body as HTMLElement).getByTestId("fleet-scheduled");
  expect(table.compareDocumentPosition(scheduled) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  // Every active plan, and nothing finished.
  expect(within(scheduled).getAllByRole("button")).toHaveLength(7);
  expect(screen.queryByTestId("fleet-scheduled-done")).toBeNull();
});

it("names a plan's crew, says when it has none, and leaves a demo plan's to its facts", () => {
  render(<FleetPanel />);
  expect(screen.getByTestId("fleet-scheduled-console-crewed")).toHaveTextContent(
    "Mar 2 – Nov 20, 2026 · TR-04 Kestrel, TR-12 Pipit",
  );
  expect(screen.getByTestId("fleet-scheduled-console-bare")).toHaveTextContent(
    "Mar 2 – Nov 20, 2026 · unassigned",
  );
  const demo = screen.getByTestId("fleet-scheduled-thistle");
  expect(demo).toHaveTextContent("Mar 2 – Nov 20, 2026");
  expect(demo).not.toHaveTextContent("unassigned");
});
