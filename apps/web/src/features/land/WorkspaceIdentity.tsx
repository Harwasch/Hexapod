import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "@/api/client";
import { landUsesOidc, useLandIdentity } from "@/state/landIdentity";
import { describeError } from "@/lib/log";
import { WorkspaceMembers } from "./WorkspaceMembers";
import { WorkspaceInvitation } from "./WorkspaceInvitation";
import { initializeLandIdentity, signInToLand, signOutOfLand } from "./identity";

export function WorkspaceIdentity() {
  const identity = useLandIdentity();
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const workspaces = useQuery({
    queryKey: ["land-workspaces", identity.principalId],
    queryFn: () => unwrap(api.GET("/api/v1/workspaces")),
    enabled: landUsesOidc && Boolean(identity.accessToken),
    retry: false,
    refetchInterval: 60_000,
  });
  useEffect(() => {
    if (!workspaces.data || !identity.accessToken) return;
    const selected =
      workspaces.data.find((workspace) => workspace.id === identity.workspaceId) ??
      workspaces.data[0];
    if (
      (selected?.id ?? null) !== identity.workspaceId ||
      (selected?.role ?? null) !== identity.role
    )
      identity.selectWorkspace(selected?.id ?? null, selected?.role);
  }, [identity, workspaces.data]);
  if (!landUsesOidc) return null;
  return (
    <section className="land-identity" aria-label="Workspace access">
      {identity.error && <p role="alert">{identity.error}</p>}
      {!identity.ready ? (
        <p>Restoring your workspace…</p>
      ) : !identity.accessToken ? (
        <div className="land-actions">
          <button
            type="button"
            className="land-primary"
            onClick={() =>
              void signInToLand().catch((error: unknown) => identity.setError(describeError(error)))
            }
          >
            Sign in to your workspace
          </button>
        </div>
      ) : (
        <>
          <label className="land-name">
            Workspace
            <select
              value={identity.workspaceId ?? ""}
              onChange={(event) =>
                identity.selectWorkspace(
                  event.target.value || null,
                  workspaces.data?.find((workspace) => workspace.id === event.target.value)?.role,
                )
              }
            >
              <option value="" disabled>
                Choose a workspace
              </option>
              {workspaces.data?.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>
                  {workspace.name} · {workspace.role}
                </option>
              ))}
            </select>
          </label>
          {workspaces.isError && (
            <p role="alert">
              Workspaces could not be loaded.{" "}
              <button type="button" onClick={() => void workspaces.refetch()}>
                Retry
              </button>
            </p>
          )}
          <details>
            <summary>Create a workspace</summary>
            <form
              className="land-actions"
              onSubmit={(event) => {
                event.preventDefault();
                setBusy(true);
                void unwrap(api.POST("/api/v1/workspaces", { body: { name } }))
                  .then(async (workspace) => {
                    await workspaces.refetch();
                    identity.selectWorkspace(workspace.id, workspace.role);
                    setName("");
                  })
                  .catch((error: unknown) => identity.setError(describeError(error)))
                  .finally(() => setBusy(false));
              }}
            >
              <input
                aria-label="New workspace name"
                value={name}
                maxLength={200}
                onChange={(event) => setName(event.target.value)}
                required
              />
              <button disabled={busy || !name.trim()}>Create</button>
            </form>
          </details>
          {identity.workspaceId && (
            <WorkspaceMembers
              key={`${identity.principalId}/${identity.workspaceId}/${identity.role}`}
              workspaceId={identity.workspaceId}
            />
          )}
          <button
            type="button"
            className="land-back"
            onClick={() =>
              void signOutOfLand().catch((error: unknown) =>
                identity.setError(describeError(error)),
              )
            }
          >
            Sign out of this workspace session
          </button>
        </>
      )}
    </section>
  );
}

export function LandIdentityBridge() {
  const cache = useQueryClient();
  useEffect(() => {
    if (landUsesOidc) void initializeLandIdentity();
    return useLandIdentity.subscribe((next, previous) => {
      if (next.principalId === previous.principalId && next.workspaceId === previous.workspaceId)
        return;
      const previousScope = `${previous.principalId ?? "anonymous"}/${previous.workspaceId ?? "none"}`;
      const filter = {
        predicate: (query: { queryKey: readonly unknown[] }) =>
          typeof query.queryKey[0] === "string" &&
          query.queryKey[0].startsWith("land-") &&
          (query.queryKey.includes(previousScope) ||
            (next.principalId !== previous.principalId &&
              query.queryKey[0] === "land-workspaces" &&
              query.queryKey[1] === previous.principalId)),
      };
      void cache.cancelQueries(filter).then(() => cache.removeQueries(filter));
    });
  }, [cache]);
  return <WorkspaceInvitation />;
}
