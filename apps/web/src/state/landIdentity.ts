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
    set({
      accessToken,
      principalId,
      ready: true,
      error: null,
      ...(changed || !accessToken ? { workspaceId: null, role: null } : {}),
    });
    if (changed || !accessToken) useLand.getState().clear();
  },
  selectWorkspace: (workspaceId, role) => {
    const changed = workspaceId !== get().workspaceId;
    set({ workspaceId, role: role ?? null });
    if (changed) useLand.getState().clear();
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

export function landScope() {
  const s = useLandIdentity.getState();
  return landUsesOidc ? `${s.principalId ?? "anonymous"}/${s.workspaceId ?? "none"}` : "pilot";
}
