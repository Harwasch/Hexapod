import { useEffect } from "react";

import { useSettings } from "@/state/settings";

/** Mirrors appearance settings onto <html> so the CSS token system can react. */
export function useApplyTheme(): void {
  const theme = useSettings((s) => s.theme);
  const reducedMotion = useSettings((s) => s.reducedMotion);
  const highContrast = useSettings((s) => s.highContrast);
  const reducedTransparency = useSettings((s) => s.reducedTransparency);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    root.setAttribute("data-motion", reducedMotion ? "reduced" : "auto");
    root.setAttribute("data-contrast", highContrast ? "high" : "auto");
    root.setAttribute("data-transparency", reducedTransparency ? "reduced" : "auto");
  }, [theme, reducedMotion, highContrast, reducedTransparency]);
}
