import { clsx } from "clsx";
import { forwardRef, type HTMLAttributes } from "react";

export type GlassPillProps = HTMLAttributes<HTMLDivElement>;

/** Small floating capsule for status and compact information. */
export const GlassPill = forwardRef<HTMLDivElement, GlassPillProps>(function GlassPill(
  { className, ...rest },
  ref,
) {
  return (
    <div
      ref={ref}
      className={clsx("glass glass--pill glass--soft glass-pill", className)}
      {...rest}
    />
  );
});
