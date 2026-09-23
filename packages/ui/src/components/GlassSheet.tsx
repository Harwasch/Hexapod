import * as Dialog from "@radix-ui/react-dialog";
import { clsx } from "clsx";
import { X } from "lucide-react";
import type { ReactNode } from "react";

import { GlassButton } from "./GlassButton";

export interface GlassSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  side?: "center" | "right" | "bottom";
  /** Hide the visual title but keep it for assistive technology. */
  hideTitle?: boolean;
  className?: string;
  /** Render the overlay (blocks the map). Defaults to true for centre sheets. */
  modal?: boolean;
  testId?: string;
}

/** Modal glass sheet built on Radix Dialog: focus trap, Escape, labelled by title. */
export function GlassSheet({
  open,
  onOpenChange,
  title,
  description,
  children,
  side = "center",
  hideTitle = false,
  className,
  modal,
  testId,
}: GlassSheetProps) {
  const withOverlay = modal ?? side === "center";
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange} modal={withOverlay}>
      <Dialog.Portal>
        {withOverlay && <Dialog.Overlay className="glass-sheet__overlay" />}
        <Dialog.Content
          className={clsx("glass glass--strong glass-sheet", `glass-sheet--${side}`, className)}
          data-testid={testId}
        >
          <div className="glass-sheet__header">
            <div>
              <Dialog.Title className={clsx("glass-sheet__title", hideTitle && "sr-only")}>
                {title}
              </Dialog.Title>
              {description ? (
                <Dialog.Description className="glass-sheet__description">
                  {description}
                </Dialog.Description>
              ) : (
                <Dialog.Description className="sr-only">
                  {typeof title === "string" ? title : "Dialog"}
                </Dialog.Description>
              )}
            </div>
            <Dialog.Close asChild>
              <GlassButton iconOnly variant="ghost" size="sm" aria-label="Close">
                <X size={16} aria-hidden="true" />
              </GlassButton>
            </Dialog.Close>
          </div>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
