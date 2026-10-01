import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { InstancePanel } from "@/features/sites/InstanceSearch";
import { parseInstances } from "@/lib/instances";
import { useInstances } from "@/state/instances";

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
    useInstances.setState({ assets: {}, dimOthers: true });
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
    await user.click(result);
    expect([...(useInstances.getState().assets.scan?.highlighted ?? [])]).toEqual([1]);
    expect(result).toHaveAttribute("aria-current", "true");
    await user.click(screen.getByRole("button", { name: "Hide oak tree" }));
    expect([...(useInstances.getState().assets.scan?.hidden ?? [])]).toEqual([1]);
    expect(screen.getByRole("button", { name: "Show oak tree" })).toHaveAttribute(
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
});
