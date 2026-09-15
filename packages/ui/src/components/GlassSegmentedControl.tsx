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
  block?: boolean;
  className?: string;
}

/** Single-select segmented control (Radix ToggleGroup): arrow keys move between segments. */
export function GlassSegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  block = false,
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
      aria-label={aria["aria-label"]}
      className={clsx("glass-segment", block && "glass-segment--block", className)}
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
