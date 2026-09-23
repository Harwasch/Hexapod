import { create } from "zustand";

export type LoadState = "idle" | "loading" | "ready" | "error";

export interface LayerRuntime {
  visible: boolean;
  opacity: number;
  loadState: LoadState;
  error: string | null;
  /** Live credit text read from the provider when the catalog has none. */
  liveCredit?: string;
}

interface LayersState {
  runtime: Record<string, LayerRuntime>;
  ensure: (id: string, initial: Partial<LayerRuntime>) => void;
  update: (id: string, patch: Partial<LayerRuntime>) => void;
  remove: (id: string) => void;
}

export const defaultRuntime: LayerRuntime = {
  visible: false,
  opacity: 1,
  loadState: "idle",
  error: null,
};

export const useLayers = create<LayersState>()((set) => ({
  runtime: {},
  ensure: (id, initial) =>
    set((s) =>
      s.runtime[id] ? s : { runtime: { ...s.runtime, [id]: { ...defaultRuntime, ...initial } } },
    ),
  update: (id, patch) =>
    set((s) => ({
      runtime: { ...s.runtime, [id]: { ...(s.runtime[id] ?? defaultRuntime), ...patch } },
    })),
  remove: (id) =>
    set((s) => ({
      runtime: Object.fromEntries(Object.entries(s.runtime).filter(([key]) => key !== id)),
    })),
}));
