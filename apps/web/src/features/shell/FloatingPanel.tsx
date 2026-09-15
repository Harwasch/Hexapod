import { X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";

import { GlassButton, GlassPanel } from "@twin/ui";

import { useSettings } from "@/state/settings";

export interface FloatingPanelProps {
  open: boolean;
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  actions?: ReactNode;
  wide?: boolean;
  flush?: boolean;
  testId?: string;
  /** Slide-in origin. */
  from?: "left" | "right" | "bottom";
  className?: string;
}

const offsets = { left: { x: -12, y: 0 }, right: { x: 12, y: 0 }, bottom: { x: 0, y: 12 } };

/** Floating glass panel with a header; animates in from its edge and respects reduced motion. */
export function FloatingPanel({
  open,
  title,
  onClose,
  children,
  actions,
  wide,
  flush,
  testId,
  from = "left",
  className,
}: FloatingPanelProps) {
  const reducedMotion = useSettings((s) => s.reducedMotion);
  const offset = offsets[from];
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="panel"
          initial={reducedMotion ? false : { opacity: 0, scale: 0.98, ...offset }}
          animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
          exit={reducedMotion ? { opacity: 0 } : { opacity: 0, scale: 0.98, ...offset }}
          transition={{ type: "spring", stiffness: 420, damping: 34, mass: 0.8 }}
          className={className}
        >
          <GlassPanel
            strong
            className={`panel ${wide ? "panel--wide" : ""}`}
            data-testid={testId}
            role="region"
            aria-label={typeof title === "string" ? title : undefined}
          >
            <header className="panel__header">
              <h2 className="panel__title">{title}</h2>
              <div className="panel__actions">
                {actions}
                <GlassButton
                  iconOnly
                  variant="ghost"
                  size="sm"
                  aria-label="Close panel"
                  onClick={onClose}
                >
                  <X size={16} aria-hidden="true" />
                </GlassButton>
              </div>
            </header>
            <div className={`panel__body ${flush ? "panel__body--flush" : ""}`}>{children}</div>
          </GlassPanel>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
