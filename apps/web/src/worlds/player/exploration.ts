/** Acknowledgment confirms receipt; only worker progress confirms conditioning applied. */
export function explorationProgress(
  queued?: number,
  applied?: number,
  generating?: number | null,
): string {
  if (queued === undefined) return "Commands apply at the next chunk boundary";
  if (queued > (applied ?? 0))
    return `Change ${queued} received · queued for the next chunk${generating !== undefined && generating !== null ? ` · generating change ${generating}` : ""}`;
  return applied !== undefined && applied > 0
    ? `Change ${applied} applied to generated output`
    : "Ready for movement and scene commands";
}
