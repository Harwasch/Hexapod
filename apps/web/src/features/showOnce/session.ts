/**
 * What a tap on the land does during Show once, everywhere (`state/showOnce.ts`): compare that
 * place with every other cell, or protect the lit patch there. The map's bridge calls these;
 * the card shows what came of them.
 */

import { cellAt } from "@/features/knowledge/grid";
import { describeError } from "@/lib/log";
import { useShowOnce } from "@/state/showOnce";

import { fetchSimilar } from "./api";
import { decodeScores, toggleProtected } from "./grid";

/** Which tap is current: a later tap, or the session ending, drops an earlier answer. */
let tapCounter = 0;

export async function compareAt(longitude: number, latitude: number): Promise<void> {
  const session = useShowOnce.getState().session;
  if (!session) return;
  const mine = ++tapCounter;
  useShowOnce.getState().setPending({ longitude, latitude });
  try {
    const read = await fetchSimilar(session.rasterId, longitude, latitude);
    if (mine !== tapCounter || useShowOnce.getState().session?.rasterId !== session.rasterId)
      return;
    useShowOnce.getState().setResult({ read, grid: decodeScores(read) });
  } catch (error) {
    if (mine !== tapCounter || useShowOnce.getState().session?.rasterId !== session.rasterId)
      return;
    useShowOnce.getState().setMessage(describeError(error));
  }
}

/** Protects the lit patch under the point, or frees the protected patch there. */
export function protectAt(longitude: number, latitude: number): void {
  const { result, guarded, threshold, setGuarded, setMessage } = useShowOnce.getState();
  if (!result) {
    setMessage("Compare a place first: then lit patches can be protected.");
    return;
  }
  const at = cellAt(result.grid, longitude, latitude);
  const toggled = at
    ? toggleProtected(result.grid, guarded, at.row * result.grid.columns + at.col, threshold)
    : null;
  if (!toggled) {
    setMessage("That ground is not lit, so it is already left out. Choose a lit patch to protect.");
    return;
  }
  setGuarded(toggled.mask);
}

/** Forgets any tap still being compared (the session ended). */
export function dropPendingTaps(): void {
  tapCounter += 1;
}
