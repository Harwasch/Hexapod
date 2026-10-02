import { Info } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { GlassButton } from "@twin/ui";

import { NavControls } from "../nav/NavControls";
import { CreditSlot } from "./CreditSlot";

/**
 * The bottom-right pill: compass and Earth, then the data credits.
 *
 * On a phone the credits do not fit beside the status line, so they fold behind an (i) that
 * opens them above the bar — every credit and the "Data attribution" dialog are one tap away,
 * and the logos are never removed from the page. Wider screens show them inline and never
 * see the button.
 */
export function MapCorner() {
  const [creditsOpen, setCreditsOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!creditsOpen) return;
    const onDown = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setCreditsOpen(false);
    };
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setCreditsOpen(false);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [creditsOpen]);

  return (
    <div
      ref={root}
      className={`glass glass--strong hud-corner ${creditsOpen ? "is-credits-open" : ""}`}
      data-testid="map-corner"
    >
      <NavControls />
      <GlassButton
        iconOnly
        variant="ghost"
        size="sm"
        className="hud-corner__info"
        aria-label={creditsOpen ? "Hide data credits" : "Data credits"}
        aria-expanded={creditsOpen}
        onClick={() => setCreditsOpen(!creditsOpen)}
        data-testid="credits-toggle"
      >
        <Info size={16} aria-hidden="true" />
      </GlassButton>
      <CreditSlot popover={creditsOpen} />
    </div>
  );
}
