import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, auth } from "@/api/client";
import { WorkspaceInvitation } from "@/features/land/WorkspaceInvitation";
import { WorkspaceIdentity } from "@/features/land/WorkspaceIdentity";
import { WorkspaceMembers } from "@/features/land/WorkspaceMembers";
import { captureWorkspaceInvite, workspaceInviteLink } from "@/features/land/workspaceInvite";
import { useLandIdentity } from "@/state/landIdentity";

vi.mock("@/state/landIdentity", async (original) => ({
  ...(await original<object>()),
  landUsesOidc: true,
}));
vi.mock("@/features/land/identity", () => ({ signInToLand: vi.fn() }));
const token = "A".repeat(43);
const account = "a".repeat(64);
const workspace = "11111111-1111-4111-8111-111111111111";
function show(node: React.ReactNode) {
  return render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      {node}
    </QueryClientProvider>,
  );
}
function response<T>(data: T) {
  return { data, response: new Response() };
}
beforeEach(() => {
  useLandIdentity.getState().setSession("test", "issuer/alice");
  useLandIdentity.getState().selectWorkspace(workspace, "owner");
  HTMLDialogElement.prototype.showModal = vi.fn(function (this: HTMLDialogElement) {
    this.open = true;
  });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  sessionStorage.clear();
  history.replaceState({}, "", "/");
});

it("keeps invitation tokens in the fragment, survives redirects and strips the address", () => {
  history.replaceState({}, "", workspaceInviteLink(token));
  expect(captureWorkspaceInvite()).toBe(token);
  expect(location.hash).toBe("");
  history.replaceState({}, "", "/auth/land/callback?code=example");
  expect(captureWorkspaceInvite()).toBe(token);
});

it("requires explicit joining and creates a profile first, then keeps the returned role", async () => {
  history.replaceState({}, "", workspaceInviteLink(token));
  vi.spyOn(api, "GET").mockResolvedValue(
    response({ principalId: account, mode: "oidc", displayName: null }),
  );
  const put = vi
    .spyOn(api, "PUT")
    .mockResolvedValue(response({ displayName: "Field researcher" }) as never);
  const post = vi.spyOn(api, "POST").mockImplementation((path: string) =>
    Promise.resolve(
      response(
        path.endsWith("inspect")
          ? {
              workspaceName: "Ranch",
              role: "viewer",
              expiresAt: "2030-01-01T00:00:00Z",
              alreadyMember: false,
            }
          : { id: workspace, name: "Ranch", role: "viewer" },
      ) as never,
    ),
  );
  show(<WorkspaceInvitation />);
  await screen.findByText("Ranch");
  expect(post).toHaveBeenCalledTimes(1);
  expect(post).toHaveBeenLastCalledWith("/api/v1/workspaces/invitations/inspect", {
    body: { token },
  });
  fireEvent.change(screen.getByRole("textbox", { name: "Your display name" }), {
    target: { value: "Field researcher" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Join workspace" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(put).toHaveBeenCalledOnce();
  expect(post).toHaveBeenCalledWith("/api/v1/workspaces/invitations/accept", { body: { token } });
  expect(useLandIdentity.getState().role).toBe("viewer");
  expect(captureWorkspaceInvite()).toBeNull();
});

it("protects the last owner and sends role changes to the workspace being edited", async () => {
  const bob = "b".repeat(64);
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve(
      response(
        path.endsWith("profile")
          ? { principalId: account, mode: "oidc", displayName: "Alice" }
          : path.endsWith("members")
            ? [
                { principalId: account, role: "owner", displayName: "Alice" },
                { principalId: bob, role: "viewer", displayName: "Bob" },
              ]
            : [],
      ) as never,
    ),
  );
  const put = vi
    .spyOn(api, "PUT")
    .mockResolvedValue(response({ principalId: bob, role: "editor" }) as never);
  show(<WorkspaceMembers workspaceId={workspace} />);
  fireEvent.click(screen.getByText("People and access"));
  expect(await screen.findByRole("combobox", { name: "Role for Alice" })).toBeDisabled();
  fireEvent.change(screen.getByRole("combobox", { name: "Role for Bob" }), {
    target: { value: "editor" },
  });
  fireEvent.click(screen.getAllByRole("button", { name: "Save role" })[1]!);
  await waitFor(() =>
    expect(put).toHaveBeenCalledWith("/api/v1/workspaces/members", {
      headers: { "X-Workspace-ID": workspace },
      body: { principalId: bob, role: "editor" },
    }),
  );
});

it("does not overwrite an explicitly pinned workspace in auth middleware", async () => {
  const request = new Request("https://app.test/api/v1/workspaces/members", {
    headers: { "X-Workspace-ID": "pinned" },
  });
  await auth.onRequest?.({
    request,
    schemaPath: "/api/v1/workspaces/members",
    params: {},
    id: "test",
    options: {},
  } as never);
  expect(request.headers.get("X-Workspace-ID")).toBe("pinned");
  expect(request.headers.get("Authorization")).toBe("Bearer test");
});

it("refreshes changed roles and clears a workspace when access is removed", async () => {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let catalog: { id: string; name: string; role: "owner" | "viewer" }[] = [
    { id: workspace, name: "Ranch", role: "owner" },
  ];
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve(
      response(
        path === "/api/v1/workspaces"
          ? catalog
          : path.endsWith("profile")
            ? { principalId: account, mode: "oidc", displayName: "Alice" }
            : [],
      ) as never,
    ),
  );
  render(
    <QueryClientProvider client={cache}>
      <WorkspaceIdentity />
    </QueryClientProvider>,
  );
  await screen.findByRole("option", { name: "Ranch · owner" });
  catalog = [{ id: workspace, name: "Ranch", role: "viewer" }];
  await cache.invalidateQueries({ queryKey: ["land-workspaces"] });
  await waitFor(() => expect(useLandIdentity.getState().role).toBe("viewer"));
  catalog = [];
  await cache.invalidateQueries({ queryKey: ["land-workspaces"] });
  await waitFor(() => expect(useLandIdentity.getState().workspaceId).toBeNull());
});
