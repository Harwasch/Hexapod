interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/**
 * The path from a card's nearest edge to a point on the map: it leaves the card at right
 * angles and bends onto the point, so it reads as coming out of the card. Null when the point
 * is under the card itself.
 */
export function leaderPath(
  card: Box,
  x: number,
  y: number,
): { d: string; sx: number; sy: number } | null {
  const inset = 14;
  const clampY = Math.min(Math.max(y, card.top + inset), card.bottom - inset);
  const clampX = Math.min(Math.max(x, card.left + inset), card.right - inset);
  let sx: number;
  let sy: number;
  let horizontal = true;
  if (x >= card.right) [sx, sy] = [card.right, clampY];
  else if (x <= card.left) [sx, sy] = [card.left, clampY];
  else if (y <= card.top) [sx, sy, horizontal] = [clampX, card.top, false];
  else if (y >= card.bottom) [sx, sy, horizontal] = [clampX, card.bottom, false];
  else return null;
  const dx = x - sx;
  const dy = y - sy;
  const c1 = horizontal ? [sx + dx * 0.5, sy] : [sx, sy + dy * 0.5];
  const c2 = horizontal ? [x - dx * 0.15, y] : [x, y - dy * 0.15];
  const f = (n: number | undefined) => (n ?? 0).toFixed(1);
  return {
    d: `M${f(sx)},${f(sy)} C${f(c1[0])},${f(c1[1])} ${f(c2[0])},${f(c2[1])} ${f(x)},${f(y)}`,
    sx,
    sy,
  };
}
