import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import {
  GlassButton,
  GlassField,
  GlassInput,
  GlassProgress,
  GlassSegmentedControl,
  GlassSheet,
  GlassSwitch,
} from "../index";

describe("GlassButton", () => {
  it("renders an accessible icon-only button and handles clicks", async () => {
    const onClick = vi.fn();
    render(
      <GlassButton iconOnly aria-label="Reset north" onClick={onClick}>
        N
      </GlassButton>,
    );
    const button = screen.getByRole("button", { name: "Reset north" });
    await userEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(button).toHaveClass("glass-button--icon");
  });

  it("exposes loading and active states", () => {
    render(
      <GlassButton loading active>
        Save
      </GlassButton>,
    );
    const button = screen.getByRole("button", { name: /save/i });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button).toHaveAttribute("aria-pressed", "true");
  });
});

describe("GlassSegmentedControl", () => {
  function Harness() {
    const [value, setValue] = useState<"splat" | "mesh" | "points">("splat");
    return (
      <GlassSegmentedControl
        aria-label="Representation"
        value={value}
        onValueChange={setValue}
        options={[
          { value: "splat", label: "Splat" },
          { value: "mesh", label: "Mesh" },
          { value: "points", label: "Points", disabled: true },
        ]}
      />
    );
  }

  it("is keyboard navigable and keeps one selection", async () => {
    render(<Harness />);
    const group = screen.getByRole("radiogroup", { name: "Representation" });
    expect(group).toBeInTheDocument();
    const splat = screen.getByRole("radio", { name: "Splat" });
    const mesh = screen.getByRole("radio", { name: "Mesh" });
    expect(splat).toHaveAttribute("data-state", "on");
    await userEvent.click(mesh);
    expect(mesh).toHaveAttribute("data-state", "on");
    expect(splat).toHaveAttribute("data-state", "off");
    expect(screen.getByRole("radio", { name: "Points" })).toBeDisabled();
    mesh.focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(splat).toHaveFocus();
  });
});

describe("GlassSheet", () => {
  it("opens with a labelled dialog and closes on Escape", async () => {
    const onOpenChange = vi.fn();
    render(
      <GlassSheet open onOpenChange={onOpenChange} title="Settings" description="Tune the world">
        <p>Body</p>
      </GlassSheet>,
    );
    const dialog = screen.getByRole("dialog", { name: "Settings" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText("Tune the world")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(onOpenChange).toHaveBeenCalledWith(false);
    await userEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onOpenChange).toHaveBeenCalledTimes(2);
  });
});

describe("GlassField / GlassSwitch", () => {
  it("associates labels, shows errors, and toggles", async () => {
    const onChange = vi.fn();
    render(
      <>
        <GlassField label="Name" htmlFor="name" error="Required">
          <GlassInput id="name" invalid />
        </GlassField>
        <GlassSwitch aria-label="Visible" checked={false} onCheckedChange={onChange} />
      </>,
    );
    const input = screen.getByLabelText("Name");
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent("Required");
    await userEvent.click(screen.getByRole("switch", { name: "Visible" }));
    expect(onChange).toHaveBeenCalledWith(true);
  });
});

describe("GlassProgress", () => {
  it("reports its position against its own unit, not a percentage", () => {
    render(
      <GlassProgress
        label="Uploading clip.mov"
        value={512}
        max={2048}
        valueText="512 MB of 2 GB"
      />,
    );
    const bar = screen.getByRole("progressbar", { name: "Uploading clip.mov" });
    expect(bar).toHaveAttribute("aria-valuenow", "512");
    expect(bar).toHaveAttribute("aria-valuemax", "2048");
    expect(bar).toHaveAttribute("aria-valuetext", "512 MB of 2 GB");
    expect(bar.querySelector(".glass-progress__bar")).toHaveStyle({ width: "25%" });
  });

  it("clamps out-of-range values and drops aria-valuenow when indeterminate", () => {
    const { rerender } = render(<GlassProgress label="Packaging" value={140} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    rerender(<GlassProgress label="Packaging" value={-5} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
    rerender(<GlassProgress label="Packaging" value={0} indeterminate tone="run" />);
    const bar = screen.getByRole("progressbar", { name: "Packaging" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(bar).toHaveClass("glass-progress--indeterminate", "glass-progress--run");
  });

  it("draws the expected marker where the schedule says it should be", () => {
    const { container } = render(
      <GlassProgress label="Plan" value={30} expected={70} expectedTitle="Schedule expects 70%" />,
    );
    const marker = container.querySelector(".glass-progress__marker");
    expect(marker).toHaveStyle({ left: "70%" });
    expect(marker).toHaveAttribute("title", "Schedule expects 70%");
  });
});
