import { create } from "zustand";

export type ToastTone = "info" | "success" | "warning" | "error";

export interface Toast {
  id: string;
  tone: ToastTone;
  title: string;
  body?: string;
  /** Sticky toasts stay until dismissed (setup problems). */
  sticky?: boolean;
  createdAt: number;
}

interface ToastState {
  items: Toast[];
  push: (toast: Omit<Toast, "id" | "createdAt"> & { id?: string }) => string;
  dismiss: (id: string) => void;
}

let counter = 0;

export const useToasts = create<ToastState>()((set) => ({
  items: [],
  push: (toast) => {
    const id = toast.id ?? `toast-${++counter}`;
    set((s) => ({
      items: [...s.items.filter((t) => t.id !== id), { ...toast, id, createdAt: Date.now() }],
    }));
    return id;
  },
  dismiss: (id) => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
}));
