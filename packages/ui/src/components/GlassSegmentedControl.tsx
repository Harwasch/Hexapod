import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { clsx } from "clsx";
import type { ReactNode } from "react";

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  icon?: ReactNode;
  disabled?: boolean;
  /** Accessible name when the label is an icon or abbreviation. */
  ariaLabel?: string;
}

export interface GlassSegmentedControlProps<T extends string> {
  value: T;
  onValueChange: (value: T) => void;
  options: readonly SegmentOption<T>[];
  /** Accessible group name. */
  "aria-label": string;
  /** Forwarded to the group element, for tests. */
  "data-testid"?: string;
  block?: boolean;
  /**
   * Side by side (the default) or stacked, one segment a row: arrow keys follow it, and
   * assistive technology is told (`aria-orientation`).
   */
  orientation?: "horizontal" | "vertical";
  className?: string;
}

/** Single-select segmented control (Radix ToggleGroup): arrow keys move between segments. */
export function GlassSegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  block = false,
  orientation,
  className,
  ...aria
}: GlassSegmentedControlProps<T>) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(next) => {
        if (next) onValueChange(next as T);
      }}
      orientation={orientation}
      aria-orientation={orientation}
      aria-label={aria["aria-label"]}
      data-testid={aria["data-testid"]}
      className={clsx(
        "glass-segment",
        block && "glass-segment--block",
        orientation === "vertical" && "glass-segment--vertical",
        className,
      )}
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.value}
          value={option.value}
          disabled={option.disabled}
          aria-label={option.ariaLabel}
          className="glass-segment__item"
        >
          {option.icon}
          {option.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
