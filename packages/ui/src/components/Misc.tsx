import { clsx } from "clsx";
import type { HTMLAttributes, ReactNode } from "react";

export interface GlassBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: "neutral" | "accent" | "danger" | "success" | "warning";
}

export function GlassBadge({ tone = "neutral", className, ...rest }: GlassBadgeProps) {
  return (
    <span
      className={clsx("glass-badge", tone !== "neutral" && `glass-badge--${tone}`, className)}
      {...rest}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="glass-kbd">{children}</kbd>;
}

export function Divider({ className }: { className?: string }) {
  return <hr className={clsx("glass-divider", className)} />;
}

export interface EmptyStateProps {
  icon?: ReactNode;
  title: ReactNode;
  body?: ReactNode;
  action?: ReactNode;
}

export function EmptyState({ icon, title, body, action }: EmptyStateProps) {
  return (
    <div className="glass-empty">
      {icon && (
        <span className="glass-empty__icon" aria-hidden="true">
          {icon}
        </span>
      )}
      <p className="glass-empty__title">{title}</p>
      {body && <p className="glass-empty__body">{body}</p>}
      {action}
    </div>
  );
}

export function VisuallyHidden({ children }: { children: ReactNode }) {
  return <span className="sr-only">{children}</span>;
}
