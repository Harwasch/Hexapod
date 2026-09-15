import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

export interface GlassTooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  /** Keyboard shortcut rendered after the label. */
  shortcut?: string;
}

/** Tooltip provider: mount once near the app root. */
export const GlassTooltipProvider = Tooltip.Provider;

export function GlassTooltip({ content, children, side = "top", shortcut }: GlassTooltipProps) {
  return (
    <Tooltip.Root delayDuration={400}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content side={side} sideOffset={6} className="glass glass--strong glass-tooltip">
          {content}
          {shortcut && (
            <>
              {" "}
              <kbd className="glass-kbd">{shortcut}</kbd>
            </>
          )}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
