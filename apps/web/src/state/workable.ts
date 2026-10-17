import { create } from "zustand";

import type { LeftOutGround } from "@/cesium/RehearsalManager";

/**
 * "Work only the ground it can" (`features/mission/workableGround.ts`): what a plan card's
 * redraft onto workable ground left out, for the map's hatch while the card is open, and the
 * rehearsal it is waiting to run again. Never persisted: the ground is a Land's, and private.
 */
export interface WorkableRecord {
  /** The derived zone the plan was redrafted on. */
  zoneId: string;
  /** The zone it was made from, as the rehearsal names it. */
  sourceLabel: string;
  /** The ground left out of it, hatched on the map. */
  leftOut: LeftOutGround[];
  /** Ground heights for the hatch's labels, from the rehearsal that asked. */
  heightAt: (longitude: number, latitude: number) => number;
}

/** The rehearsal a redraft onto workable ground runs again once its draft is ready. */
export interface WorkableRerun {
  zoneIds: string[];
  /** Hands the rehearsal raised before, for "down from" in the card. */
  handsBefore: number;
  /** The zones whose ground was cut, as the rehearsal names them. */
  labels: string[];
}

interface WorkableState {
  records: Record<string, WorkableRecord>;
  rerun: WorkableRerun | null;
  /** What the last rerun answered, once it has run: shown beside its summary. */
  done: WorkableRerun | null;
  remember: (records: readonly WorkableRecord[], rerun: WorkableRerun) => void;
  /**
   * The rerun waiting for a draft on these zones, taken (it runs once); null when none waits
   * or it was for other ground.
   */
  takeRerun: (zoneIds: readonly string[]) => WorkableRerun | null;
  /** Forgets everything: the plan card closed. */
  clear: () => void;
}

export const useWorkable = create<WorkableState>()((set, get) => ({
  records: {},
  rerun: null,
  done: null,
  remember: (records, rerun) =>
    set((state) => ({
      records: {
        ...state.records,
        ...Object.fromEntries(records.map((record) => [record.zoneId, record])),
      },
      rerun,
      done: null,
    })),
  takeRerun: (zoneIds) => {
    const rerun = get().rerun;
    if (!rerun?.zoneIds.every((id) => zoneIds.includes(id))) return null;
    set({ rerun: null, done: rerun });
    return rerun;
  },
  clear: () => set({ records: {}, rerun: null, done: null }),
}));
