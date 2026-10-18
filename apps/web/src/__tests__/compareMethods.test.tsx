import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { CompareMethodsPanel, TODAY } from "@/features/sites/CompareMethods";
import { InferredLegend, InferredStyleControl } from "@/features/sites/InferredStyle";
import { INFERRED_LEGEND } from "@/lib/inferred";
import { LOOK_FOR, variantsOf } from "@/lib/variants";
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
      // As the spool's motion bake-off publishes them: four methods, long names.
      skins: [
        { name: "freeform", label: "FreeForm · size rule", about: "Size.", skin: "a.json" },
        {
          name: "freeform-stiff",
          label: "FreeForm · stiffness rule",
          about: "Stiffness.",
          skin: "b.json",
        },
        {
          name: "pinned-stiff",
          label: "Pinned FreeForm · stiffness rule",
          about: "Pinned.",
          look: "Watch the base: it should stay put while the top sways.",
          skin: "c.json",
        },
        {
          name: "tetfem-stiff",
          label: "Volume FEM · stiffness rule",
          about: "Volume.",
          skin: "d.json",
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
    expect(screen.getByRole("group", { name: "Motion" })).toBeVisible();
  });

  it("has no row for a system the scan offers no methods for", () => {
    useVariants.setState({
      offered: { scan: { ...OFFERED, skins: [] } },
    });
    render(<CompareMethodsPanel assetId="scan" />);
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
    expect(useVariants.getState().picks.scan?.objects).toBe(TODAY);
    expect(objects).toHaveAttribute("data-picked", TODAY);
  });

  it("lists long method names one a row, in the same radio group, never a native select", async () => {
    const user = userEvent.setup();
    const { container } = render(<CompareMethodsPanel assetId="scan" />);
    expect(container.querySelector("select")).toBeNull();
    expect(screen.queryAllByRole("combobox")).toEqual([]);
    const objects = screen.getByRole("group", { name: "Objects" });
    expect(objects).toHaveAttribute("data-layout", "segments");
    const motion = screen.getByRole("group", { name: "Motion" });
    expect(motion).toHaveAttribute("data-layout", "list");
    const list = within(motion).getByRole("radiogroup", { name: "Motion method" });
    expect(list).toHaveAttribute("aria-orientation", "vertical");
    expect(
      within(list)
        .getAllByRole("radio")
        .map((r) => r.textContent),
    ).toEqual([
      "Today",
      "FreeForm · size rule",
      "FreeForm · stiffness rule",
      "Pinned FreeForm · stiffness rule",
      "Volume FEM · stiffness rule",
    ]);
    // Down the list from the keyboard, Enter to pick.
    within(list).getByRole("radio", { name: "Today" }).focus();
    await user.keyboard("{ArrowDown}");
    await waitFor(() =>
      expect(within(list).getByRole("radio", { name: "FreeForm · size rule" })).toHaveFocus(),
    );
    await user.keyboard("{Enter}");
    expect(useVariants.getState().picks.scan?.skins).toBe("freeform");
    expect(within(list).getByRole("radio", { name: "FreeForm · size rule" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("shows a hidden fill when one is picked", async () => {
    const user = userEvent.setup();
    render(<CompareMethodsPanel assetId="scan" />);
    const fill = screen.getByRole("group", { name: "Fill" });
    expect(fill).toHaveAttribute("data-layout", "list");
    expect(within(fill).getByText(/no inferred fill/)).toBeVisible();
    await user.click(within(fill).getByRole("radio", { name: "Mound" }));
    expect(useVariants.getState().picks.scan?.fill).toBe("mound");
    expect(useSettings.getState().inferredStyle).toBe("show");
  });

  it("says what to look for, per system, unless the method says its own", async () => {
    const user = userEvent.setup();
    render(<CompareMethodsPanel assetId="scan" />);
    for (const [name, system] of [
      ["Objects", "objects"],
      ["Fill", "fill"],
      ["Motion", "skins"],
    ] as const) {
      const row = screen.getByRole("group", { name });
      expect(within(row).getByText(LOOK_FOR[system])).toBeVisible();
      // Read with the row: its description is what the method does and what to look for.
      expect(row).toHaveAccessibleDescription(expect.stringContaining(LOOK_FOR[system]));
    }
    const motion = screen.getByRole("group", { name: "Motion" });
    await user.click(
      within(motion).getByRole("radio", { name: "Pinned FreeForm · stiffness rule" }),
    );
    expect(
      within(motion).getByText("Watch the base: it should stay put while the top sways."),
    ).toBeVisible();
    expect(within(motion).queryByText(LOOK_FOR.skins)).toBeNull();
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
