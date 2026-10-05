/** How far the map menu keeps from the screen's edges (CSS px). */
export const MENU_EDGE_PX = 8;

/**
 * Where the map menu goes for a press at `at` (client px): below and right of it, as a context
 * menu opens, and turned to the other side of the point where that would leave the screen.
 */
export function placeMenu(
  at: { x: number; y: number },
  size: { width: number; height: number },
  screen: { width: number; height: number },
): { left: number; top: number } {
  const fitsRight = at.x + size.width <= screen.width - MENU_EDGE_PX;
  const fitsBelow = at.y + size.height <= screen.height - MENU_EDGE_PX;
  const left = fitsRight ? at.x : at.x - size.width;
  const top = fitsBelow ? at.y : at.y - size.height;
  return {
    left: Math.max(MENU_EDGE_PX, Math.min(left, screen.width - MENU_EDGE_PX - size.width)),
    top: Math.max(MENU_EDGE_PX, Math.min(top, screen.height - MENU_EDGE_PX - size.height)),
  };
}
