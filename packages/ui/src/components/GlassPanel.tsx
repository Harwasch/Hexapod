import { clsx } from "clsx";
import { forwardRef, type ElementType, type HTMLAttributes } from "react";

export interface GlassPanelProps extends HTMLAttributes<HTMLElement> {
  /** Stronger, more opaque material for dense content. */
  strong?: boolean;
  /** Softer shadow for nested surfaces. */
  soft?: boolean;
  /** Fully rounded (pill) shape. */
  pill?: boolean;
  /** Smaller corner radius for compact surfaces. */
  compact?: boolean;
  /** Hover/press affordances for clickable surfaces. */
  interactive?: boolean;
  padding?: "none" | "sm" | "md";
  as?: ElementType;
}

/** The base glass surface. Every floating UI element is built on it. */
export const GlassPanel = forwardRef<HTMLElement, GlassPanelProps>(function GlassPanel(
  {
    strong = false,
    soft = false,
    pill = false,
    compact = false,
    interactive = false,
    padding = "none",
    as: Component = "div",
    className,
    ...rest
  },
  ref,
) {
  return (
    <Component
      ref={ref}
      className={clsx(
        "glass",
        strong && "glass--strong",
        soft && "glass--soft",
        pill && "glass--pill",
        compact && "glass--sm",
        interactive && "glass--interactive",
        padding === "md" && "glass--padded",
        padding === "sm" && "glass--padded-sm",
        className,
      )}
      {...rest}
    />
  );
});
