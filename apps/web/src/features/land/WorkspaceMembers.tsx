import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";
import { useLandIdentity, useLandScope } from "@/state/landIdentity";
import { workspaceInviteLink } from "./workspaceInvite";
import "./workspaceAccess.css";

type Member = components["schemas"]["MemberRead"];
type Role = Member["role"];

export function WorkspaceMembers({ workspaceId }: { workspaceId: string }) {
  const scope = useLandScope();
  const cache = useQueryClient();
  const identity = useLandIdentity();
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const headers = { "X-Workspace-ID": workspaceId };
  const profile = useQuery({
    queryKey: ["land-profile", scope],
    queryFn: () => unwrap(api.GET("/api/v1/workspaces/profile")),
  });
  const members = useQuery({
    queryKey: ["land-members", scope],
    queryFn: () => unwrap(api.GET("/api/v1/workspaces/members", { headers })),
    enabled: identity.role === "owner",
  });
  const [offset, setOffset] = useState(0);
  const invitations = useQuery({
    queryKey: ["land-invitations", scope, offset],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/workspaces/invitations", {
          headers,
          params: { query: { offset, limit: 25 } },
        }),
      ),
    enabled: identity.role === "owner",
  });
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(timer);
  }, []);
  const [name, setName] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [role, setRole] = useState<"viewer" | "editor">("viewer");
  const [days, setDays] = useState(7);
  const [link, setLink] = useState<{ id: string; url: string } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (operation: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await operation();
    } catch (cause) {
      if (active.current) setError(describeError(cause));
    } finally {
      if (active.current) setBusy(false);
    }
  };
  const refresh = async () => {
    await Promise.all([
      cache.invalidateQueries({ queryKey: ["land-members", scope] }),
      cache.invalidateQueries({ queryKey: ["land-invitations", scope] }),
      cache.invalidateQueries({ queryKey: ["land-workspaces", identity.principalId] }),
    ]);
  };
  const onlyOwner = members.data?.filter((member) => member.role === "owner").length === 1;
  return (
    <details>
      <summary>People and access</summary>
      <div className="workspace-access">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void run(async () => {
              const result = await unwrap(
                api.PUT("/api/v1/workspaces/profile", {
                  body: { displayName: name ?? profile.data?.displayName ?? "" },
                }),
              );
              if (!active.current) return;
              cache.setQueryData(["land-profile", scope], result);
              setMessage("Display name saved.");
              await cache.invalidateQueries({ queryKey: ["land-members", scope] });
            });
          }}
        >
          <label>
            Your display name
            <input
              value={name ?? profile.data?.displayName ?? ""}
              onChange={(event) => setName(event.target.value)}
              maxLength={120}
              required
            />
          </label>
          <p>Shown to workspace owners. Display names are chosen by each person.</p>
          <button disabled={busy || !(name ?? profile.data?.displayName ?? "").trim()}>
            Save display name
          </button>
        </form>
        {profile.isError && (
          <p role="alert">
            Your profile could not be loaded.{" "}
            <button type="button" onClick={() => void profile.refetch()}>
              Retry
            </button>
          </p>
        )}
        {identity.role !== "owner" ? (
          <p>Ask a workspace owner to invite people or change access.</p>
        ) : (
          <>
            <p>
              Viewers read land records. Editors also change records and run research. Owners manage
              access.
            </p>
            {members.isPending ? (
              <p>Loading people…</p>
            ) : members.isError ? (
              <p role="alert">
                People could not be loaded.{" "}
                <button type="button" onClick={() => void members.refetch()}>
                  Retry people
                </button>
              </p>
            ) : (
              <ul aria-label="Workspace members">
                {members.data?.map((member) => (
                  <MemberRow
                    key={`${member.principalId}/${member.role}`}
                    member={member}
                    self={member.principalId === profile.data?.principalId}
                    lastOwner={onlyOwner && member.role === "owner"}
                    busy={busy}
                    change={(nextRole) =>
                      run(async () => {
                        await unwrap(
                          api.PUT("/api/v1/workspaces/members", {
                            headers,
                            body: { principalId: member.principalId, role: nextRole },
                          }),
                        );
                        await refresh();
                      })
                    }
                    remove={() =>
                      run(async () => {
                        await unwrap(
                          api.DELETE("/api/v1/workspaces/members/{principal_id}", {
                            headers,
                            params: { path: { principal_id: member.principalId } },
                          }),
                        );
                        await refresh();
                      })
                    }
                  />
                ))}
              </ul>
            )}
            <h3>Invite someone</h3>
            <p>
              Each link admits one person. Share it directly with the intended person; anyone with
              the link can use it before it expires.
            </p>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void run(async () => {
                  const result = await unwrap(
                    api.POST("/api/v1/workspaces/invitations", {
                      headers,
                      body: { label, role, expiresInDays: days },
                    }),
                  );
                  if (!active.current) return;
                  setLink({ id: result.id, url: workspaceInviteLink(result.token) });
                  setLabel("");
                  setOffset(0);
                  await cache.invalidateQueries({ queryKey: ["land-invitations", scope] });
                });
              }}
            >
              <label>
                Invitation label
                <input
                  placeholder="e.g. Ranch field team"
                  value={label}
                  onChange={(event) => setLabel(event.target.value)}
                  required
                  maxLength={120}
                />
              </label>
              <label>
                Invite as
                <select
                  value={role}
                  onChange={(event) => setRole(event.target.value as "viewer" | "editor")}
                >
                  <option value="viewer">Viewer</option>
                  <option value="editor">Editor</option>
                </select>
              </label>
              <label>
                Expires in
                <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
                  <option value={1}>1 day</option>
                  <option value={7}>7 days</option>
                  <option value={30}>30 days</option>
                </select>
              </label>
              <button disabled={busy || !label.trim()}>Create invitation link</button>
            </form>
            {link && (
              <div>
                <label>
                  Invitation link
                  <input readOnly value={link.url} onFocus={(event) => event.target.select()} />
                </label>
                <p>Copy this link now. It is only shown in this session.</p>
                <button
                  type="button"
                  onClick={() =>
                    void run(async () => {
                      await navigator.clipboard.writeText(link.url);
                      setMessage("Invitation link copied.");
                    })
                  }
                >
                  Copy link
                </button>
                <button type="button" onClick={() => setLink(null)}>
                  Hide link
                </button>
              </div>
            )}
            <h3>Invitations</h3>
            <p>Revoking a used link does not remove the person’s membership.</p>
            {invitations.isError ? (
              <p role="alert">
                Invitations could not be loaded.{" "}
                <button type="button" onClick={() => void invitations.refetch()}>
                  Retry invitations
                </button>
              </p>
            ) : invitations.isPending ? (
              <p>Loading invitations…</p>
            ) : !invitations.data?.length ? (
              <p>No invitations yet.</p>
            ) : (
              <ul aria-label="Workspace invitations">
                {invitations.data.map((invite) => (
                  <li key={invite.id}>
                    <strong>{invite.label}</strong>
                    <span>
                      {invite.role} ·{" "}
                      {invite.acceptedAt
                        ? "Used"
                        : invite.revokedAt
                          ? "Revoked"
                          : new Date(invite.expiresAt).getTime() <= now
                            ? "Expired"
                            : `Expires ${new Date(invite.expiresAt).toLocaleString()}`}
                    </span>
                    {!invite.acceptedAt && !invite.revokedAt && (
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void run(async () => {
                            await unwrap(
                              api.DELETE("/api/v1/workspaces/invitations/{invitation_id}", {
                                headers,
                                params: { path: { invitation_id: invite.id } },
                              }),
                            );
                            if (!active.current) return;
                            if (link?.id === invite.id) setLink(null);
                            await invitations.refetch();
                          })
                        }
                      >
                        Revoke {invite.label}
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            <div className="workspace-member-actions" aria-label="Invitation pages">
              <button
                type="button"
                disabled={busy || offset === 0}
                onClick={() => setOffset(Math.max(0, offset - 25))}
              >
                Newer invitations
              </button>
              <span>Page {offset / 25 + 1}</span>
              <button
                type="button"
                disabled={busy || invitations.isPending || (invitations.data?.length ?? 0) < 25}
                onClick={() => setOffset(offset + 25)}
              >
                Older invitations
              </button>
            </div>
          </>
        )}
        {error && (
          <p role="alert">
            {error} Refresh the list before retrying if the result is uncertain.{" "}
            <button type="button" disabled={busy} onClick={() => void run(refresh)}>
              Refresh access lists
            </button>
          </p>
        )}
        {message && <p role="status">{message}</p>}
      </div>
    </details>
  );
}

function MemberRow({
  member,
  self,
  lastOwner,
  busy,
  change,
  remove,
}: {
  member: Member;
  self: boolean;
  lastOwner: boolean;
  busy: boolean;
  change: (role: Role) => Promise<void>;
  remove: () => Promise<void>;
}) {
  const [role, setRole] = useState(member.role);
  const [confirm, setConfirm] = useState(false);
  const label = member.displayName ?? `Account ${member.principalId.slice(0, 8)}`;
  return (
    <li>
      <strong>
        {label}
        {self ? " (you)" : ""}
      </strong>
      <details>
        <summary>Account identifier</summary>
        <code>{member.principalId}</code>
      </details>
      {lastOwner && (
        <p>At least one owner must remain. Promote another person before changing this owner.</p>
      )}
      <div className="workspace-member-actions">
        <label>
          Role for {label}
          <select
            disabled={busy || lastOwner}
            value={role}
            onChange={(event) => setRole(event.target.value as Role)}
          >
            <option value="viewer">Viewer</option>
            <option value="editor">Editor</option>
            <option value="owner">Owner</option>
          </select>
        </label>
        <button
          type="button"
          disabled={busy || lastOwner || role === member.role}
          onClick={() => void change(role)}
        >
          Save role
        </button>
        <button type="button" disabled={busy || lastOwner} onClick={() => setConfirm(true)}>
          Remove access
        </button>
      </div>
      {role === "owner" && member.role !== "owner" && (
        <p>Owners can grant access to all land records in this workspace.</p>
      )}
      {self && role !== "owner" && member.role === "owner" && (
        <p>You will lose the ability to manage access.</p>
      )}
      {confirm && (
        <div>
          <p>
            Remove {label}
            {self ? " (your account)" : ""} from this workspace?
          </p>
          <button type="button" disabled={busy} onClick={() => void remove()}>
            Confirm removal
          </button>
          <button type="button" disabled={busy} onClick={() => setConfirm(false)}>
            Keep access
          </button>
        </div>
      )}
    </li>
  );
}
