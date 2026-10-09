import { create } from "zustand";
import { env } from "@/app/env";
import { useLand } from "./land";

export const landUsesOidc = Boolean(env.landOidcAuthority ?? env.landOidcClientId);

interface LandIdentityState {
  accessToken: string | null;
  principalId: string | null;
  workspaceId: string | null;
  role: "owner" | "editor" | "viewer" | null;
  ready: boolean;
  error: string | null;
  setSession: (token: string | null, principalId: string | null) => void;
  selectWorkspace: (id: string | null, role?: "owner" | "editor" | "viewer") => void;
  setError: (message: string | null) => void;
}

export const useLandIdentity = create<LandIdentityState>((set, get) => ({
  accessToken: null,
  principalId: null,
  workspaceId: null,
  role: null,
  ready: !landUsesOidc,
  error: null,
  setSession: (accessToken, principalId) => {
    const changed = principalId !== get().principalId;
    if (changed || !accessToken) useLand.getState().clear();
    set({
      accessToken,
      principalId,
      ready: true,
      error: null,
      ...(changed || !accessToken ? { workspaceId: null, role: null } : {}),
    });
  },
  selectWorkspace: (workspaceId, role) => {
    if (workspaceId !== get().workspaceId) useLand.getState().clear();
    set({ workspaceId, role: role ?? null });
  },
  setError: (error) => set({ error, ready: true }),
}));

export function useLandScope() {
  return useLandIdentity((s) =>
    landUsesOidc ? `${s.principalId ?? "anonymous"}/${s.workspaceId ?? "none"}` : "pilot",
  );
}

export function useLandAccessReady() {
  return useLandIdentity(
    (s) => s.ready && (!landUsesOidc || Boolean(s.accessToken && s.workspaceId)),
  );
}

export function useLandCanEdit() {
  return useLandIdentity((s) => !landUsesOidc || s.role === "owner" || s.role === "editor");
}
