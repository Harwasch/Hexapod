import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { CompareMethodsPanel, TODAY } from "@/features/sites/CompareMethods";
import { InferredLegend, InferredStyleControl } from "@/features/sites/InferredStyle";
import { INFERRED_LEGEND } from "@/lib/inferred";
import { variantsOf } from "@/lib/variants";
import { useSettings } from "@/state/settings";
import { useVariants } from "@/state/variants";

const EVIDENCE = {
  kind: "inferred",
  filler: "fixture",
  views: 2,
  gaussians: 9,
  meanConfidence: 0.5,
};

const OFFERED = {
  ...variantsOf({
    variants: {
      objects: [
        { name: "whole", label: "Whole", about: "One object a thing.", instances: "a.json" },
        { name: "parts", label: "Parts", about: "Every part its own.", instances: "b.json" },
      ],
      fill: [
        {
          name: "hedge",
          label: "A long label for a hedge-shaped fill method",
          about: "A hedge.",
          inferredLayers: [{ uri: "h/tileset.json", evidence: EVIDENCE }],
        },
        {
          name: "mound",
          label: "Mound",
          about: "A mound.",
          inferredLayers: [{ uri: "m/tileset.json", evidence: EVIDENCE }],
        },
      ],
    },
  }),
  today: { objects: true, fill: false, skins: true },
};

describe("Compare methods", () => {
  beforeEach(() => {
    useVariants.setState({ offered: { scan: OFFERED }, picks: {}, status: {} });
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("lists a row per system the scan offers, Today first, labels visible", () => {
    render(<CompareMethodsPanel assetId="scan" />);
    const objects = screen.getByRole("group", { name: "Objects" });
    expect(within(objects).getByText("Today")).toBeVisible();
    expect(within(objects).getByText("Whole")).toBeVisible();
    expect(within(objects).getByText(/the objects this scan publishes now/)).toBeVisible();
    expect(screen.getByRole("group", { name: "Fill" })).toBeVisible();
    // No skins variants: no Motion row.
    expect(screen.queryByRole("group", { name: "Motion" })).toBeNull();
  });

  it("picks from the keyboard and says what the pick does", async () => {
    const user = userEvent.setup();
    render(<CompareMethodsPanel assetId="scan" />);
    const objects = screen.getByRole("group", { name: "Objects" });
    const whole = within(objects).getByRole("radio", { name: "Whole" });
    whole.focus();
    await user.keyboard("{Enter}");
    expect(useVariants.getState().picks.scan?.objects).toBe("whole");
    expect(within(objects).getByText("One object a thing.")).toBeVisible();
    expect(objects).toHaveAttribute("data-picked", "whole");
    await user.click(within(objects).getByRole("radio", { name: "Today" }));
    expect(useVariants.getState().picks.scan?.objects).toBeUndefined();
    expect(objects).toHaveAttribute("data-picked", TODAY);
  });

  it("uses a list where the labels do not fit a line, and shows a hidden fill when one is picked", async () => {
    const user = userEvent.setup();
    render(<CompareMethodsPanel assetId="scan" />);
    const fill = screen.getByRole("group", { name: "Fill" });
    expect(within(fill).getByText(/no inferred fill/)).toBeVisible();
    await user.selectOptions(within(fill).getByRole("combobox", { name: "Fill method" }), "mound");
    expect(useVariants.getState().picks.scan?.fill).toBe("mound");
    expect(useSettings.getState().inferredStyle).toBe("show");
  });

  it("says when a pick is loading or did not load", () => {
    useVariants.setState({
      status: { scan: { objects: { state: "error", message: "instances answered 404" } } },
    });
    render(<CompareMethodsPanel assetId="scan" />);
    expect(screen.getByText(/Did not load: instances answered 404/)).toBeVisible();
  });

  it("is nothing for a scan without variants", () => {
    const { container } = render(<CompareMethodsPanel assetId="other" />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Inferred style", () => {
  it("is Show, Highlight or Hide, kept as a setting", async () => {
    const user = userEvent.setup();
    useSettings.getState().set({ inferredStyle: "hide" });
    render(<InferredStyleControl evidence={[{ ...EVIDENCE, kind: "inferred" }]} />);
    const group = screen.getByRole("radiogroup", { name: "Inferred fill" });
    expect(
      within(group)
        .getAllByRole("radio")
        .map((r) => r.textContent),
    ).toEqual(["Show", "Highlight", "Hide"]);
    await user.click(within(group).getByRole("radio", { name: "Highlight inferred fill" }));
    expect(useSettings.getState().inferredStyle).toBe("highlight");
    useSettings.getState().set({ inferredStyle: "hide" });
  });

  it("has a one-line legend", () => {
    render(<InferredLegend style="highlight" />);
    expect(screen.getByTestId("inferred-legend")).toHaveTextContent(INFERRED_LEGEND);
  });
});
