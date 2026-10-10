import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { landUsesOidc, useLandIdentity, useLandScope } from "@/state/landIdentity";
import { useUi } from "@/state/ui";
import { signInToLand } from "./identity";
import { captureWorkspaceInvite, clearWorkspaceInvite } from "./workspaceInvite";
import "./workspaceAccess.css";

export function WorkspaceInvitation() {
  const principal = useLandIdentity((state) => state.principalId);
  const [pending, setPending] = useState<{ token: string | null; error: string | null }>(() => {
    try {
      return { token: captureWorkspaceInvite(), error: null };
    } catch {
      return {
        token: null,
        error: "Allow session storage to open this invitation, then reload the link.",
      };
    }
  });
  useEffect(() => {
    const capture = () => {
      try {
        setPending({ token: captureWorkspaceInvite(), error: null });
      } catch {
        setPending({
          token: null,
          error: "Allow session storage to open this invitation, then reload the link.",
        });
      }
    };
    window.addEventListener("hashchange", capture);
    return () => window.removeEventListener("hashchange", capture);
  }, []);
  if (!pending.token && !pending.error) return null;
  return (
    <InvitationDialog
      key={`${pending.token}/${principal}`}
      token={pending.token}
      initialError={pending.error}
      close={() => {
        try {
          clearWorkspaceInvite();
        } catch {
          /* Close remains usable with unavailable storage. */
        }
        setPending({ token: null, error: null });
      }}
    />
  );
}

function InvitationDialog({
  token,
  initialError,
  close,
}: {
  token: string | null;
  initialError: string | null;
  close: () => void;
}) {
  const identity = useLandIdentity();
  const scope = useLandScope();
  const cache = useQueryClient();
  const dialog = useRef<HTMLDialogElement>(null);
  const [previewKey] = useState(() => crypto.randomUUID());
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(initialError);
  const [busy, setBusy] = useState(false);
  const profile = useQuery({
    queryKey: ["land-profile", scope],
    queryFn: () => unwrap(api.GET("/api/v1/workspaces/profile")),
    enabled: landUsesOidc && Boolean(identity.accessToken),
  });
  // The token stays out of query keys, logs, URLs and persisted query caches.
  const preview = useQuery({
    queryKey: ["land-invitation-preview", scope, previewKey],
    queryFn: () =>
      unwrap(api.POST("/api/v1/workspaces/invitations/inspect", { body: { token: token ?? "" } })),
    enabled: landUsesOidc && Boolean(identity.accessToken && token),
    retry: false,
    gcTime: 0,
    staleTime: 0,
  });
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  const join = async () => {
    if (!token) return;
    setBusy(true);
    setError(null);
    const principal = identity.principalId;
    try {
      if (!profile.data?.displayName && !preview.data?.alreadyMember)
        await unwrap(api.PUT("/api/v1/workspaces/profile", { body: { displayName: name } }));
      if (useLandIdentity.getState().principalId !== principal) return;
      const workspace = await unwrap(
        api.POST("/api/v1/workspaces/invitations/accept", { body: { token } }),
      );
      if (useLandIdentity.getState().principalId !== principal) return;
      // Seed before selection so catalog synchronization cannot bounce back to an old workspace.
      cache.setQueryData(
        ["land-workspaces", principal],
        (old: { id: string; name: string; role: "owner" | "editor" | "viewer" }[] | undefined) => [
          ...(old ?? []).filter((item) => item.id !== workspace.id),
          workspace,
        ],
      );
      identity.selectWorkspace(workspace.id, workspace.role);
      useUi.getState().setPanel("land");
      close();
    } catch (cause) {
      setError(describeError(cause));
    } finally {
      setBusy(false);
    }
  };
  return (
    <dialog
      ref={dialog}
      className="workspace-invitation"
      aria-labelledby="workspace-invitation-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) close();
      }}
    >
      <h2 id="workspace-invitation-title">Join a land workspace</h2>
      {!landUsesOidc ? (
        <p>This app needs individual account sign-in configured before invitations can be used.</p>
      ) : !identity.ready ? (
        <p>Restoring your session…</p>
      ) : !identity.accessToken ? (
        <>
          <p>Sign in to review this invitation. Joining requires your confirmation.</p>
          <button
            type="button"
            onClick={() =>
              void signInToLand().catch((cause: unknown) => setError(describeError(cause)))
            }
          >
            Sign in to review invitation
          </button>
        </>
      ) : preview.isPending ? (
        <p>Checking invitation…</p>
      ) : preview.isError ? (
        <p role="alert">
          {describeError(preview.error)}{" "}
          <button type="button" onClick={() => void preview.refetch()}>
            Check again
          </button>
        </p>
      ) : (
        preview.data && (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void join();
            }}
          >
            <h3>{preview.data.workspaceName}</h3>
            <p>
              {preview.data.alreadyMember
                ? "You already belong to this workspace. Your current role will stay the same."
                : `You are invited as ${preview.data.role === "editor" ? "an editor: explore, research and edit land records" : "a viewer: explore and read land records"}.`}
            </p>
            <p>Link expires {new Date(preview.data.expiresAt).toLocaleString()}.</p>
            {!preview.data.alreadyMember && !profile.data?.displayName && (
              <label>
                Your display name
                <input
                  value={name}
                  maxLength={120}
                  required
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
            )}
            {profile.isError && (
              <p role="alert">
                Your profile could not be loaded.{" "}
                <button type="button" onClick={() => void profile.refetch()}>
                  Retry profile
                </button>
              </p>
            )}
            <button
              disabled={
                busy ||
                profile.isPending ||
                profile.isError ||
                (!preview.data.alreadyMember && !profile.data?.displayName && !name.trim())
              }
            >
              {busy ? "Joining…" : preview.data.alreadyMember ? "Open workspace" : "Join workspace"}
            </button>
          </form>
        )
      )}
      {(error ?? identity.error) && <p role="alert">{error ?? identity.error}</p>}
      <button type="button" disabled={busy} onClick={close}>
        Cancel invitation
      </button>
    </dialog>
  );
}
