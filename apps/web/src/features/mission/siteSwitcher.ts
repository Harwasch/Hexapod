import { useMission } from "@/state/mission";
import { useUi, type SwitcherFocus } from "@/state/ui";

/**
 * Opens the site switcher (the menu behind the site's name, `ProjectCard`) with the keyboard
 * on its sites (`s`, "Switch site") or its saved views (`b`, "Saved views").
 */
export function openSiteSwitcher(focus: SwitcherFocus): void {
  useUi.getState().setSwitcherFocus(focus);
  useMission.getState().setProjectsOpen(true);
}
