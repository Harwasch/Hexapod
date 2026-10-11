import { create } from "zustand";

import type { components } from "@twin/contracts";

type Focus = components["schemas"]["ResearchFocusReference"];

/** A question for the agent, raised from any lens ("Ask about this"). */
export interface AgentRequest {
  id: number;
  question: string;
  /** A research map feature the question is about. */
  focus?: Focus | null;
  /** What the focus is, in words, for the card. */
  focusLabel?: string | null;
}

interface AgentBus {
  request: AgentRequest | null;
  /** Opens the agent card with the question filled in; the person still sends it. */
  ask: (question: string, focus?: Focus | null, focusLabel?: string | null) => void;
  consume: (id: number) => void;
}

let counter = 0;

/**
 * Lenses never start the agent themselves: they hand it a question, and the agent card shows
 * it ready to send. The agent runs only when the person sends it.
 */
export const useAgentBus = create<AgentBus>((set, get) => ({
  request: null,
  ask: (question, focus = null, focusLabel = null) =>
    set({ request: { id: ++counter, question, focus, focusLabel } }),
  consume: (id) => {
    if (get().request?.id === id) set({ request: null });
  },
}));

export function askAgent(question: string, focus?: Focus | null, focusLabel?: string | null) {
  useAgentBus.getState().ask(question, focus, focusLabel);
}
