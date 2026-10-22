/**
 * The words for why research waits or a research button is disabled
 * (features/land/researchAvailability.ts), from what /research/status says about the worker.
 */

import { describe, expect, it } from "vitest";
import type { ResearchStatus } from "@twin/contracts";

import {
  TYPE_A_QUESTION,
  WAKE_GRACE_S,
  analysisBlocker,
  askBlocker,
  formatAgo,
  workerIdleNote,
  workerLine,
} from "@/features/land/researchAvailability";

function status(
  worker: Partial<ResearchStatus["worker"]> = {},
  oldestQueuedSeconds: number | null = 20,
): ResearchStatus {
  return {
    modelConfigured: true,
    model: "fixture",
    workerRequired: true,
    worker: {
      state: "online",
      lastSeenAt: "2026-10-16T18:00:00Z",
      secondsSinceSeen: 4,
      busy: false,
      checkInEverySeconds: 15,
      ...worker,
    },
    queue: { queued: 1, oldestQueuedAt: "2026-10-16T18:00:00Z", oldestQueuedSeconds, running: 0 },
  };
}
const queued = { status: "queued" as const, question: "What is here?", focusLabel: null };
const running = { status: "running" as const, question: "What is here?", focusLabel: null };

describe("formatAgo", () => {
  it("says whole units, rounded down, across each boundary", () => {
    expect(formatAgo(0)).toBe("0 s");
    expect(formatAgo(59.9)).toBe("59 s");
    expect(formatAgo(60)).toBe("1 min");
    expect(formatAgo(840)).toBe("14 min");
    expect(formatAgo(3599)).toBe("59 min");
    expect(formatAgo(3600)).toBe("1 h");
    expect(formatAgo(86_399)).toBe("23 h");
    expect(formatAgo(86_400)).toBe("1 day");
    expect(formatAgo(2 * 86_400 + 5)).toBe("2 days");
    expect(formatAgo(-5)).toBe("0 s");
  });
});

describe("workerLine", () => {
  it("explains a queued run for each worker state", () => {
    const waking =
      "Waking the research worker: it sleeps when there is nothing to do and starts for your request. Your request is saved.";
    expect(workerLine(status({ state: "never", secondsSinceSeen: null }), queued)).toBe(waking);
    expect(workerLine(status({ state: "stale", secondsSinceSeen: 840 }), queued)).toBe(
      "Waiting for the research worker: none has checked in for 14 min. Your request is saved and starts when a worker runs.",
    );
    expect(workerLine(status({ state: "stopped", secondsSinceSeen: 7200 }), queued)).toBe(waking);
    expect(workerLine(status({ busy: true, secondsSinceSeen: 12 }), queued)).toBe(
      "Queued behind another research request. The research worker checked in 12 s ago.",
    );
    expect(workerLine(status({ secondsSinceSeen: 3 }), queued)).toBe(
      "Starting. The research worker checked in 3 s ago.",
    );
  });

  it("calls a sleeping worker asleep, never missing, until the wake has had time", () => {
    for (const state of ["stopped", "never"] as const) {
      const sleeping = { state, secondsSinceSeen: state === "never" ? null : 7200 };
      for (const waited of [null, 0, WAKE_GRACE_S - 1]) {
        const line = workerLine(status(sleeping, waited), queued);
        expect(line).toContain("Waking the research worker");
        expect(line).not.toMatch(/none has checked in|stopped|missing/);
      }
      // A wake that did not happen (no Fly token, a local server with no worker) is said.
      expect(workerLine(status(sleeping, 300), queued)).toBe(
        "The research worker has not started yet: this request has waited 5 min. It is saved and starts when the worker runs.",
      );
    }
  });

  it("says nothing unless the run is queued and the worker is known", () => {
    expect(workerLine(status({ state: "stale", secondsSinceSeen: 840 }), running)).toBeNull();
    expect(workerLine(status(), undefined)).toBeNull();
    expect(workerLine(undefined, queued)).toBeNull();
    // A status from a server without worker presence.
    const older = { modelConfigured: true, model: null } as unknown as ResearchStatus;
    expect(workerLine(older, queued)).toBeNull();
  });

  it("never uses an em dash", () => {
    for (const state of ["online", "stale", "stopped", "never"] as const)
      expect(workerLine(status({ state, secondsSinceSeen: 90 }), queued)).not.toContain("—");
  });
});

describe("workerIdleNote", () => {
  it("says a sleeping worker starts on demand, and warns only when one went quiet", () => {
    expect(workerIdleNote(status())).toBeNull();
    expect(workerIdleNote(undefined)).toBeNull();
    expect(workerIdleNote(status({ state: "stale", secondsSinceSeen: 3 * 86_400 }))).toBe(
      "Analyses run on a separate research worker, which has not checked in for 3 days. New requests will wait in the queue.",
    );
    const sleeping =
      "Analyses run on a separate research worker. It sleeps when there is nothing to do and starts when you ask, so the first request can take a moment longer.";
    expect(workerIdleNote(status({ state: "never", secondsSinceSeen: null }))).toBe(sleeping);
    expect(workerIdleNote(status({ state: "stopped", secondsSinceSeen: 120 }))).toBe(sleeping);
  });
});

describe("analysisBlocker", () => {
  const base = { ready: true, canEdit: true, run: undefined };

  it("gives the first reason: workspace, then role, then the queued or running request", () => {
    expect(analysisBlocker({ ...base, ready: false, canEdit: false, run: queued })).toBe(
      "Connecting to your workspace…",
    );
    expect(analysisBlocker({ ...base, canEdit: false, run: queued })).toBe(
      "Your workspace role can view research but not start it.",
    );
    // The worker's state and age are ResearchWorkerNotice's to say, once.
    expect(analysisBlocker({ ...base, run: queued })).toBe(
      "One request runs at a time. Cancel the queued request to start something else.",
    );
    expect(analysisBlocker({ ...base, run: running })).toBe(
      "One request runs at a time. Wait for “What is here?” to finish, or stop it.",
    );
    expect(analysisBlocker(base)).toBeNull();
  });

  it("names a running request by its focus, and shortens a long question", () => {
    expect(analysisBlocker({ ...base, run: { ...running, focusLabel: "North fence" } })).toContain(
      "“North fence”",
    );
    const long = "Why ".repeat(30);
    expect(analysisBlocker({ ...base, run: { ...running, question: long } })).toContain(
      `“${long.slice(0, 60).trimEnd()}…”`,
    );
  });
});

describe("askBlocker", () => {
  const ask = {
    ready: true,
    canEdit: true,
    run: undefined,
    modelConfigured: true as boolean | null | undefined,
    pending: false,
    recoveryError: false,
    question: "What grew here?",
  };

  it("adds the question's own reasons, in order, after the analysis ones", () => {
    expect(askBlocker({ ...ask, run: running, modelConfigured: false })).toContain(
      "One request runs at a time.",
    );
    expect(askBlocker({ ...ask, modelConfigured: false, pending: true })).toBe(
      "The agent isn't set up on this server. The lens analyses still work.",
    );
    expect(askBlocker({ ...ask, pending: true, recoveryError: true })).toBe(
      "Confirm or discard the captured request above first.",
    );
    expect(askBlocker({ ...ask, recoveryError: true, question: "" })).toBe(
      "Resolve the unreadable saved question above first.",
    );
    expect(askBlocker({ ...ask, question: "   " })).toBe(TYPE_A_QUESTION);
    expect(askBlocker(ask)).toBeNull();
  });

  it("says when it cannot yet tell whether the agent is available", () => {
    expect(askBlocker({ ...ask, modelConfigured: undefined })).toBe(
      "Checking whether the agent is available…",
    );
    expect(askBlocker({ ...ask, modelConfigured: null })).toContain("Could not check");
  });
});
