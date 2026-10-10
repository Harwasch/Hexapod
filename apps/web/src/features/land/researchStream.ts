import type { ResearchEvent } from "@twin/contracts";

export class ResearchAccessError extends Error {
  constructor(public readonly reason: "expired" | "revoked") {
    super(
      reason === "expired"
        ? "Sign in again to continue live research updates."
        : "Access to this research is no longer available.",
    );
  }
}

export function parseResearchEvent(value: unknown): ResearchEvent {
  if (!value || typeof value !== "object") throw new Error("Invalid research event.");
  const item = value as Record<string, unknown>;
  if (
    !Number.isSafeInteger(item.sequence) ||
    (item.sequence as number) < 1 ||
    typeof item.kind !== "string" ||
    !item.kind ||
    item.kind.length > 100 ||
    typeof item.createdAt !== "string" ||
    !Number.isFinite(Date.parse(item.createdAt)) ||
    !item.payload ||
    typeof item.payload !== "object" ||
    Array.isArray(item.payload)
  )
    throw new Error("Invalid research event.");
  return item as unknown as ResearchEvent;
}

/** Parse bounded SSE frames across arbitrary UTF-8 chunks, including split CRLFs. */
export async function readResearchStream(
  stream: ReadableStream<Uint8Array>,
  signal: AbortSignal,
  receive: (event: ResearchEvent) => void,
): Promise<void> {
  const reader = stream.getReader(),
    decoder = new TextDecoder();
  let pending = "",
    eventName = "",
    id = "",
    data: string[] = [],
    size = 0;
  const abort = () => {
    void reader.cancel().catch(() => undefined);
  };
  signal.addEventListener("abort", abort, { once: true });
  const line = (value: string) => {
    if (!value) {
      if (eventName === "access-expired" || eventName === "access-revoked")
        throw new ResearchAccessError(eventName === "access-expired" ? "expired" : "revoked");
      if (eventName === "research" && data.length) {
        const item = parseResearchEvent(JSON.parse(data.join("\n")) as unknown);
        if (id && id !== String(item.sequence)) throw new Error("Research event cursor mismatch.");
        receive(item);
      }
      eventName = "";
      id = "";
      data = [];
      size = 0;
      return;
    }
    if (value.startsWith(":")) return;
    size += value.length;
    if (size > 1_048_576) throw new Error("Research update exceeded the size limit.");
    const split = value.indexOf(":"),
      field = split < 0 ? value : value.slice(0, split);
    const content = split < 0 ? "" : value.slice(split + 1).replace(/^ /, "");
    if (field === "event") eventName = content;
    else if (field === "id") id = content;
    else if (field === "data") data.push(content);
  };
  try {
    while (!signal.aborted) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      if (pending.length > 1_048_576) throw new Error("Research update exceeded the size limit.");
      let start = 0;
      for (let index = 0; index < pending.length; index++) {
        const character = pending[index];
        if (character !== "\r" && character !== "\n") continue;
        if (character === "\r" && index === pending.length - 1 && !done) break;
        line(pending.slice(start, index));
        if (character === "\r" && pending[index + 1] === "\n") index++;
        start = index + 1;
      }
      pending = pending.slice(start);
      if (done) break; // Incomplete final frames are replayed from the last accepted cursor.
    }
  } finally {
    signal.removeEventListener("abort", abort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export function mergeResearchEvents(
  previous: ResearchEvent[],
  incoming: ResearchEvent[],
): ResearchEvent[] {
  const bySequence = new Map(previous.map((event) => [event.sequence, event]));
  for (const event of incoming)
    if (!bySequence.has(event.sequence)) bySequence.set(event.sequence, event);
  return [...bySequence.values()].sort((a, b) => a.sequence - b.sequence).slice(-2000);
}
