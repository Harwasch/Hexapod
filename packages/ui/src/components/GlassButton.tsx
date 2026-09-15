import { clsx } from "clsx";
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { Spinner } from "./Spinner";

export type GlassButtonVariant = "default" | "primary" | "ghost" | "danger" | "glass";
export type GlassButtonSize = "sm" | "md" | "lg";

export interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: GlassButtonVariant;
  size?: GlassButtonSize;
  /** Square button that only contains an icon. `aria-label` is required. */
  iconOnly?: boolean;
  /** Shows the pressed/selected state (e.g. active tool). */
  active?: boolean;
  block?: boolean;
  loading?: boolean;
  leadingIcon?: ReactNode;
  trailingIcon?: ReactNode;
}

type IconOnlyProps = GlassButtonProps & { iconOnly: true; "aria-label": string };
type LabelledProps = GlassButtonProps & { iconOnly?: false };

export const GlassButton = forwardRef<HTMLButtonElement, IconOnlyProps | LabelledProps>(
  function GlassButton(
    {
      variant = "default",
      size = "md",
      iconOnly = false,
      active = false,
      block = false,
      loading = false,
      leadingIcon,
      trailingIcon,
      className,
      children,
      type = "button",
      disabled,
      ...rest
    },
    ref,
  ) {
    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled ?? loading}
        aria-busy={loading || undefined}
        aria-pressed={active || undefined}
        className={clsx(
          "glass-button",
          variant !== "default" && `glass-button--${variant}`,
          size !== "md" && `glass-button--${size}`,
          iconOnly && "glass-button--icon",
          active && "glass-button--active",
          block && "glass-button--block",
          className,
        )}
        {...rest}
      >
        {loading ? <Spinner className="glass-button__spinner" /> : leadingIcon}
        {!iconOnly && children}
        {iconOnly && !loading && children}
        {trailingIcon}
      </button>
    );
  },
);
