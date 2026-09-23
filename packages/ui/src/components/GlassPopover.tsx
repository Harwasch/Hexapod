import * as Popover from "@radix-ui/react-popover";
import { clsx } from "clsx";
import type { ReactNode } from "react";

export interface GlassPopoverProps {
  trigger: ReactNode;
  children: ReactNode;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  sideOffset?: number;
  className?: string;
  /** Accessible label for the popover content. */
  "aria-label"?: string;
}

export function GlassPopover({
  trigger,
  children,
  open,
  onOpenChange,
  side = "bottom",
  align = "center",
  sideOffset = 8,
  className,
  ...aria
}: GlassPopoverProps) {
  return (
    <Popover.Root open={open} onOpenChange={onOpenChange}>
      <Popover.Trigger asChild>{trigger}</Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          side={side}
          align={align}
          sideOffset={sideOffset}
          collisionPadding={12}
          aria-label={aria["aria-label"]}
          className={clsx("glass glass--strong glass-popover", className)}
        >
          {children}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export const GlassPopoverClose = Popover.Close;
