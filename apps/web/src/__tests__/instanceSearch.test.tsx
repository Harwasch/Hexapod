import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { InstancePanel } from "@/features/sites/InstanceSearch";
import { parseInstances } from "@/lib/instances";
import { useInstances } from "@/state/instances";
import { useSettings } from "@/state/settings";

const box = { min: [0, 0, 0], max: [1, 1, 1] };
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    {
      id: 1,
      bounds: box,
      tags: [{ label: "oak tree", score: 0.8 }],
      properties: { vegetation: 0.9 },
      behaviour: "in-place",
    },
    {
      id: 2,
      bounds: box,
      tags: [{ label: "picnic table", score: 0.5 }],
      properties: { movable: 0.7 },
      behaviour: "movable",
    },
  ],
  tiles: {},
});

describe("the object search panel", () => {
  beforeEach(() => {
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    if (DOC) useInstances.getState().setTable("scan", DOC);
  });

  it("renders nothing for a scan without objects", () => {
    const { container } = render(<InstancePanel assetId="other" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("searches by tag, highlights on click, hides per result", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    await user.type(screen.getByLabelText("Find objects"), "tree");
    const list = screen.getByRole("list", { name: "Matching objects" });
    expect(list.querySelectorAll("li")).toHaveLength(1);
    const result = screen.getByRole("button", { name: /^oak tree/ });
    expect(result).toHaveTextContent("in-place");
    // Rows that share a label are told apart by id and size.
    expect(result).toHaveTextContent("#1");
    expect(result).toHaveTextContent("splats");
    await user.click(result);
    expect([...(useInstances.getState().assets.scan?.highlighted ?? [])]).toEqual([1]);
    expect(result).toHaveAttribute("aria-current", "true");
    await user.click(screen.getByRole("button", { name: "Hide oak tree #1" }));
    expect([...(useInstances.getState().assets.scan?.hidden ?? [])]).toEqual([1]);
    expect(screen.getByRole("button", { name: "Show oak tree #1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await user.click(screen.getByRole("button", { name: /Show all/ }));
    expect(useInstances.getState().assets.scan?.hidden.size).toBe(0);
  });

  it("offers quick filters named by the file's properties", async () => {
    const user = userEvent.setup();
    render(<InstancePanel assetId="scan" />);
    const filters = screen.getByRole("group", { name: "Quick filters" });
    expect(filters).toHaveTextContent("movable");
    expect(filters).toHaveTextContent("vegetation");
    await user.click(screen.getAllByRole("button", { name: "vegetation" })[0]!);
    expect(useInstances.getState().assets.scan?.query).toBe("vegetation > 0.5");
    expect(screen.getByRole("button", { name: /^oak tree/ })).toBeInTheDocument();
    await user.clear(screen.getByLabelText("Find objects"));
    await user.type(screen.getByLabelText("Find objects"), "submarine");
    expect(screen.getByRole("status")).toHaveTextContent("No object matches.");
  });

  it("names behaviour filters apart from properties of the same name", () => {
    render(<InstancePanel assetId="scan" />);
    const filters = screen.getByRole("group", { name: "Quick filters" });
    expect(screen.getAllByRole("button", { name: "movable" })).toHaveLength(1);
    expect(filters).toHaveTextContent("behaviour: movable");
  });

  it("acts on every match, not only the fifty listed", async () => {
    const user = userEvent.setup();
    const many = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: Array.from({ length: 80 }, (_, k) => ({
        id: k + 1,
        bounds: box,
        splats: 100 + k,
        tags: [{ label: "cabin", score: 0.6 }],
        properties: { vegetation: k < 70 ? 0.9 : 0.1 },
        behaviour: "static",
      })),
      tiles: {},
    });
    if (!many) throw new Error("no document");
    useInstances.getState().setTable("camp", many);
    render(<InstancePanel assetId="camp" />);
    await user.type(screen.getByLabelText("Find objects"), "vegetation > 0.5");
    expect(screen.getByTestId("instance-count")).toHaveTextContent("50 of 70");
    expect(
      screen.getByRole("list", { name: "Matching objects" }).querySelectorAll("li"),
    ).toHaveLength(50);
    await user.click(screen.getByRole("button", { name: "Hide all 70 matches" }));
    expect(useInstances.getState().assets.camp?.hidden.size).toBe(70);
    await user.click(screen.getByRole("button", { name: "Show only matches" }));
    expect(useInstances.getState().assets.camp?.hidden.size).toBe(10);
    expect(useInstances.getState().assets.camp?.hidden.has(71)).toBe(true);
  });

  it("says when the renderer cannot hide or highlight, and switches to CesiumJS", async () => {
    const user = userEvent.setup();
    useSettings.setState({ splatRenderer: "playcanvas" });
    useInstances.getState().setGap("scan", { renderer: "playcanvas", reason: "No ids." });
    render(<InstancePanel assetId="scan" />);
    const note = screen.getByTestId("instance-renderer-gap");
    expect(note).toHaveTextContent("Highlight and hide need the Cesium renderer");
    await user.click(screen.getByRole("button", { name: "Use the Cesium renderer" }));
    expect(useSettings.getState().splatRenderer).toBe("cesium");
  });
});
