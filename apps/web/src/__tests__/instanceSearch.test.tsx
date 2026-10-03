import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { InstancePanel, OBJECT_PAGE } from "@/features/sites/InstanceSearch";
import { parseInstances } from "@/lib/instances";
import { useInstances } from "@/state/instances";
import { selectedId, useSceneSelect } from "@/state/sceneSelect";
import { useSettings } from "@/state/settings";

const box = { min: [0, 0, 0], max: [1, 1, 1] };
/**
 * A small camp: a ground region (1) with a conifer in it (2, whose branch 3 has no tags), a
 * second conifer (4), a picnic table (5) and an untagged speck (6) that nothing describes or
 * holds.
 */
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    {
      id: 1,
      bounds: box,
      splats: 5000,
      tags: [{ label: "forest floor", score: 0.6 }],
      properties: { vegetation: 0.7 },
      behaviour: "in-place",
    },
    {
      id: 2,
      parent: 1,
      bounds: box,
      splats: 900,
      tags: [{ label: "conifer", score: 0.5 }],
      properties: { vegetation: 0.9 },
      behaviour: "in-place",
    },
    { id: 3, parent: 2, bounds: box, splats: 300, behaviour: "in-place" },
    {
      id: 4,
      bounds: box,
      splats: 800,
      tags: [{ label: "conifer", score: 0.4 }],
      properties: { vegetation: 0.9 },
      behaviour: "in-place",
    },
    {
      id: 5,
      bounds: box,
      splats: 400,
      tags: [{ label: "picnic table", score: 0.5 }],
      properties: { movable: 0.7 },
      behaviour: "movable",
    },
    // Away from everything: no box holds it.
    { id: 6, bounds: { min: [10, 10, 10], max: [11, 11, 11] }, splats: 3 },
  ],
  tiles: {},
});

const state = () => useInstances.getState().assets.scan;
const sorted = (set: ReadonlySet<number> | undefined): number[] =>
  [...(set ?? [])].sort((a, b) => a - b);

describe("the objects panel", () => {
  beforeEach(() => {
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    useSceneSelect.getState().clear();
    if (DOC) useInstances.getState().setTable("scan", DOC);
  });

  it("renders nothing for a scan without objects", () => {
    const { container } = render(<InstancePanel assetId="other" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists the scan's categories, largest first, with counts and no ids or scores", () => {
    render(<InstancePanel assetId="scan" />);
    expect(screen.getByRole("searchbox", { name: "Search objects" })).toBeInTheDocument();
    const list = screen.getByRole("list", { name: "Object categories" });
    const rows = [...list.querySelectorAll<HTMLElement>(":scope > li")];
    expect(rows.map((r) => r.dataset.category)).toEqual(["ground", "trees", "furniture", "other"]);
    expect(rows[1]).toHaveTextContent("Trees, 2 objects");
    expect(list).not.toHaveTextContent(/Object \d|%|in-place|vegetation/);
    // Nothing hidden or highlighted: no Reset.
    expect(screen.queryByRole("button", { name: "Reset" })).not.toBeInTheDocument();
  });

  it("hides a whole category with its eye, and Reset brings it back", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    await user.click(screen.getByRole("button", { name: "Hide Trees" }));
    // Both trees with the branch; not the ground the first one stands in.
    expect(sorted(state()?.hidden)).toEqual([2, 3, 4]);
    const eye = screen.getByRole("button", { name: "Show Trees" });
    expect(eye).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Trees hidden");
    await user.click(eye);
    expect(state()?.hidden.size).toBe(0);
    await user.click(screen.getByRole("button", { name: "Hide Ground & soil" }));
    expect(sorted(state()?.hidden)).toEqual([1]);
    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(state()?.hidden.size).toBe(0);
    expect(screen.queryByRole("button", { name: "Reset" })).not.toBeInTheDocument();
  });

  it("highlights a category on click, and clears it on a second click", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    const row = screen.getByRole("button", { name: /^Trees/ });
    await user.click(row);
    expect(sorted(state()?.highlighted)).toEqual([2, 3, 4]);
    expect(row).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Trees highlighted");
    await user.click(row);
    expect(state()?.highlighted.size).toBe(0);
    expect(row).toHaveAttribute("aria-pressed", "false");
  });

  it("opens a category onto its objects, each with an eye, by click or arrow key", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    await user.click(screen.getByRole("button", { name: "Expand Trees" }));
    const objects = screen.getByRole("list", { name: "Trees" });
    expect(
      within(objects)
        .getAllByRole("listitem")
        .map((li) => li.textContent),
    ).toEqual(["conifer 1", "conifer 2"]);
    await user.click(within(objects).getByRole("button", { name: "Hide conifer 1" }));
    expect(sorted(state()?.hidden)).toEqual([2, 3]);
    // Part of the category hidden: its eye says so.
    expect(screen.getByRole("button", { name: "Hide Trees" })).toHaveAttribute(
      "aria-pressed",
      "mixed",
    );
    await user.click(within(objects).getByRole("button", { name: /^conifer 2/ }));
    expect(sorted(state()?.highlighted)).toEqual([4]);
    // The untagged speck is named by its category, never by an id.
    const other = screen.getByRole("button", { name: /^Other/ });
    other.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("list", { name: "Other" })).toHaveTextContent("Other 1");
    await user.keyboard("{ArrowLeft}");
    expect(screen.queryByRole("list", { name: "Other" })).not.toBeInTheDocument();
  });

  it("moves through the rows with the arrow keys from the search box", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    await user.click(screen.getByRole("searchbox", { name: "Search objects" }));
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /^Ground & soil/ }));
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /^Trees/ }));
    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(document.activeElement).toBe(screen.getByRole("searchbox"));
  });

  it("shows search results by category, with Hide all and Show only", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    await user.type(screen.getByRole("searchbox", { name: "Search objects" }), "tree");
    expect(screen.getByTestId("instance-count")).toHaveTextContent("2 matches");
    const list = screen.getByRole("list", { name: "Matching objects" });
    expect(
      [...list.querySelectorAll<HTMLElement>(":scope > li")].map((r) => r.dataset.category),
    ).toEqual(["trees"]);
    // Open by default while searching.
    expect(within(list).getByRole("list", { name: "Trees" })).toHaveTextContent("conifer 1");
    await user.click(screen.getByRole("button", { name: "Hide all" }));
    expect(sorted(state()?.hidden)).toEqual([2, 3, 4]);
    await user.click(screen.getByRole("button", { name: "Show only" }));
    expect(sorted(state()?.hidden)).toEqual([1, 5, 6]);
    // Enter highlights every match.
    await user.type(screen.getByRole("searchbox"), "{Enter}");
    expect(sorted(state()?.highlighted)).toEqual([2, 3, 4]);
    await user.clear(screen.getByRole("searchbox"));
    await user.type(screen.getByRole("searchbox"), "submarine");
    expect(screen.getByTestId("instance-count")).toHaveTextContent("No objects match");
    // A typed property filter still works.
    await user.clear(screen.getByRole("searchbox"));
    await user.type(screen.getByRole("searchbox"), "movable > 0.5");
    expect(screen.getByTestId("instance-count")).toHaveTextContent("1 match");
  });

  it("lists a big category fifty objects at a time", async () => {
    const user = userEvent.setup();
    const many = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: Array.from({ length: 80 }, (_, k) => ({
        id: k + 1,
        bounds: box,
        splats: 100 + k,
        tags: [{ label: "cabin", score: 0.6 }],
      })),
      tiles: {},
    });
    if (!many) throw new Error("no document");
    useInstances.getState().setTable("camp", many);
    render(<InstancePanel assetId="camp" />);
    await user.click(screen.getByRole("button", { name: "Expand Buildings" }));
    const objects = screen.getByRole("list", { name: "Buildings" });
    expect(within(objects).getAllByRole("button", { name: /^Hide cabin/ })).toHaveLength(
      OBJECT_PAGE,
    );
    await user.click(screen.getByRole("button", { name: "Show 30 more of 80" }));
    expect(within(objects).getAllByRole("button", { name: /^Hide cabin/ })).toHaveLength(80);
    // The category's eye acts on all 80, listed or not.
    await user.click(screen.getByRole("button", { name: "Hide Buildings" }));
    expect(useInstances.getState().assets.camp?.hidden.size).toBe(80);
  });

  it("follows the scene selection, and selects in the scene from a row", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    // Picking the branch (3) in the scene opens Trees and marks its object, conifer 1 (2).
    act(() => useSceneSelect.getState().select("scan", [3, 2], 2, 0, { x: 10, y: 10 }));
    const trees = screen.getByRole("list", { name: "Trees" });
    const marked = within(trees).getByRole("button", { name: /^conifer 1/ });
    expect(marked).toHaveAttribute("aria-current", "true");
    // Closing it is respected until the next selection.
    await user.click(screen.getByRole("button", { name: "Collapse Trees" }));
    expect(screen.queryByRole("list", { name: "Trees" })).not.toBeInTheDocument();
    // A row selects its object in the scene (the selection card then offers its actions)...
    await user.click(screen.getByRole("button", { name: "Expand Trees" }));
    await user.click(screen.getByRole("button", { name: /^conifer 2/ }));
    expect(selectedId(useSceneSelect.getState())).toBe(4);
    expect(useSceneSelect.getState().assetId).toBe("scan");
    // ...and a second click clears both.
    await user.click(screen.getByRole("button", { name: /^conifer 2/ }));
    expect(selectedId(useSceneSelect.getState())).toBeNull();
    expect(useInstances.getState().assets.scan?.highlighted.size).toBe(0);
  });

  it("says when the renderer cannot hide or highlight, and switches to CesiumJS", async () => {
    const user = userEvent.setup();
    useSettings.setState({ splatRenderer: "playcanvas" });
    useInstances.getState().setGap("scan", { renderer: "playcanvas", reason: "No ids." });
    render(<InstancePanel assetId="scan" />);
    const note = screen.getByTestId("instance-renderer-gap");
    expect(note).toHaveTextContent("Hiding and highlighting need the Cesium renderer");
    await user.click(screen.getByRole("button", { name: "Use the Cesium renderer" }));
    expect(useSettings.getState().splatRenderer).toBe("cesium");
  });
});
