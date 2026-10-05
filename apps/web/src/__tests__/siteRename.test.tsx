/**
 * Renaming a site from the site switcher (ProjectCard `SiteList`): a pencil beside each row the
 * app can rename, an inline field, and the write token asked for when the API wants one.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteSummary } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import { builtinDemoSite, DEMO_SITE_SLUG, toSummary } from "@/api/fallback";
import { ProjectCard } from "@/features/mission/ProjectCard";
import { useMission } from "@/state/mission";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

const ORCHARD_ID = "55555555-5555-4555-8555-555555555555";

const demo: SiteSummary = {
  ...toSummary(builtinDemoSite()),
  id: "11111111-1111-4111-8111-111111111111",
  slug: DEMO_SITE_SLUG,
  name: "Cesium Gaussian splat demo",
};

/** The API's catalog, as the mocked GET answers it; a successful PATCH renames in it. */
let catalog: SiteSummary[] = [];

function record(name: string): Site {
  return { ...builtinDemoSite(), id: ORCHARD_ID, slug: "orchard-block-a", name };
}

const ok = (data: unknown) => ({ data, response: new Response(null, { status: 200 }) });

function mockGet() {
  vi.spyOn(api, "GET").mockImplementation((path: string) =>
    Promise.resolve(path === "/api/v1/sites" ? ok(catalog) : ok(record(catalog[1]?.name ?? ""))),
  );
}

/** A PATCH the API accepts: the catalog takes the name, and the record comes back. */
function accept(body: { name: string }) {
  catalog = catalog.map((s) => (s.id === ORCHARD_ID ? { ...s, name: body.name } : s));
  return ok(record(body.name));
}

function mockPatch(...answers: ((body: { name: string }) => unknown)[]) {
  let call = 0;
  return vi.spyOn(api, "PATCH").mockImplementation(((
    _path: string,
    init: { body: { name: string } },
  ) => {
    const answer = answers[Math.min(call, answers.length - 1)] ?? accept;
    call += 1;
    return Promise.resolve(answer(init.body));
  }) as unknown as typeof api.PATCH);
}

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

async function openSwitcher() {
  useMission.setState({ project: null, projectsOpen: true });
  render(wrap(<ProjectCard />));
  const switcher = screen.getByRole("dialog", { name: "Switch site" });
  await within(switcher).findByText("Orchard block A");
  return switcher;
}

beforeEach(() => {
  catalog = [demo, { ...demo, id: ORCHARD_ID, slug: "orchard-block-a", name: "Orchard block A" }];
  mockGet();
  useSettings.getState().reset();
  useUi.setState({ writeTokenPrompt: false, switcherFocus: "sites" });
});

afterEach(() => {
  useMission.setState({ projectsOpen: false });
  vi.restoreAllMocks();
});

describe("renaming a site in the switcher", () => {
  it("saves the trimmed name and shows it in the row", async () => {
    const user = userEvent.setup();
    const patch = mockPatch(accept);
    const switcher = await openSwitcher();

    await user.click(within(switcher).getByRole("button", { name: "Rename Orchard block A" }));
    const field = within(switcher).getByRole("textbox", { name: "New name for Orchard block A" });
    expect(field).toHaveFocus();
    expect(field).toHaveAttribute("maxLength", "200");
    // A blank name cannot be saved.
    await user.clear(field);
    expect(within(switcher).getByRole("button", { name: "Save" })).toBeDisabled();
    await user.type(field, "  North orchard  {Enter}");

    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch).toHaveBeenCalledWith("/api/v1/sites/{site_id}", {
      params: { path: { site_id: ORCHARD_ID } },
      body: { name: "North orchard" },
    });
    const row = within(switcher).getByTestId("site-row-orchard-block-a");
    await waitFor(() => expect(row).toHaveTextContent("North orchard"));
    expect(within(switcher).queryByRole("textbox", { name: /New name/ })).not.toBeInTheDocument();
    // The keyboard is handed back to the row's pencil.
    expect(within(switcher).getByTestId("site-rename-orchard-block-a")).toHaveFocus();
  });

  it("asks for the write token on a 401, and saving it retries the rename", async () => {
    const user = userEvent.setup();
    const refused = () => ({
      error: { title: "Unauthorized", status: 401, detail: "This endpoint needs the write token." },
      response: new Response(null, { status: 401 }),
    });
    const patch = mockPatch(refused, accept);
    const switcher = await openSwitcher();

    await user.click(within(switcher).getByRole("button", { name: "Rename Orchard block A" }));
    const field = within(switcher).getByRole("textbox", { name: /New name/ });
    await user.clear(field);
    await user.type(field, "North orchard{Enter}");

    const token = await within(switcher).findByTestId("write-token-form");
    expect(token).toHaveTextContent("A write token is needed to rename sites");
    // Not renamed: the row has its old name back.
    expect(within(switcher).getByTestId("site-row-orchard-block-a")).toHaveTextContent(
      "Orchard block A",
    );

    await user.type(within(token).getByTestId("write-token-input"), "s3cret");
    await user.click(within(token).getByTestId("write-token-save"));
    expect(useSettings.getState().writeToken).toBe("s3cret");
    expect(patch).toHaveBeenCalledTimes(2);
    expect(patch.mock.calls[1]?.[1]).toMatchObject({ body: { name: "North orchard" } });
    await waitFor(() =>
      expect(within(switcher).getByTestId("site-row-orchard-block-a")).toHaveTextContent(
        "North orchard",
      ),
    );
    expect(within(switcher).queryByTestId("write-token-form")).not.toBeInTheDocument();
    // Still open: the retry happened where it was asked for.
    expect(useMission.getState().projectsOpen).toBe(true);
  });

  it("says so when the rename fails for another reason", async () => {
    const user = userEvent.setup();
    mockPatch(() => ({
      error: { title: "Validation error", status: 422 },
      response: new Response(null, { status: 422 }),
    }));
    const switcher = await openSwitcher();
    await user.click(within(switcher).getByRole("button", { name: "Rename Orchard block A" }));
    await user.type(within(switcher).getByRole("textbox", { name: /New name/ }), "x{Enter}");
    expect(await within(switcher).findByRole("alert")).toHaveTextContent(
      "Could not rename to “Orchard block Ax”: Validation error",
    );
    expect(within(switcher).queryByTestId("write-token-form")).not.toBeInTheDocument();
  });

  it("offers no rename for the demo, which is known by its project's name", async () => {
    const switcher = await openSwitcher();
    const demoRow = within(switcher).getByTestId(`site-row-${DEMO_SITE_SLUG}`);
    expect(demoRow).toHaveTextContent("Blackrock Mesa");
    expect(within(switcher).queryByTestId(`site-rename-${DEMO_SITE_SLUG}`)).not.toBeInTheDocument();
    expect(within(switcher).queryByRole("button", { name: /Rename Blackrock/ })).toBeNull();
    expect(within(switcher).getByTestId("site-rename-orchard-block-a")).toBeInTheDocument();
  });

  it("offers none either for the built-in catalog the app shows offline", async () => {
    vi.spyOn(api, "GET").mockRejectedValue(new TypeError("offline"));
    useMission.setState({ project: null, projectsOpen: true });
    render(wrap(<ProjectCard />));
    const switcher = screen.getByRole("dialog", { name: "Switch site" });
    await within(switcher).findByText("Built-in demo (API offline)");
    expect(within(switcher).queryByRole("button", { name: /^Rename/ })).toBeNull();
  });

  it("Escape puts the name back and closes only the edit, not the switcher", async () => {
    const user = userEvent.setup();
    const patch = mockPatch(accept);
    const switcher = await openSwitcher();

    await user.click(within(switcher).getByRole("button", { name: "Rename Orchard block A" }));
    await user.type(within(switcher).getByRole("textbox", { name: /New name/ }), "Elsewhere");
    // The app's own Escape listens on the window; the edit's Escape must not reach it either.
    const appEscape = vi.fn();
    window.addEventListener("keydown", appEscape);
    await user.keyboard("{Escape}");
    window.removeEventListener("keydown", appEscape);

    expect(useMission.getState().projectsOpen).toBe(true);
    expect(appEscape).not.toHaveBeenCalled();
    expect(within(switcher).queryByRole("textbox", { name: /New name/ })).not.toBeInTheDocument();
    expect(within(switcher).getByTestId("site-row-orchard-block-a")).toHaveTextContent(
      "Orchard block A",
    );
    expect(patch).not.toHaveBeenCalled();
    // The pencil has the keyboard, so the next Escape closes the switcher as it always did.
    expect(within(switcher).getByTestId("site-rename-orchard-block-a")).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(useMission.getState().projectsOpen).toBe(false);
  });

  it("keeps a changed name when the field loses focus, and leaves an unchanged one", async () => {
    const user = userEvent.setup();
    const patch = mockPatch(accept);
    const switcher = await openSwitcher();
    const pencil = () => within(switcher).getByTestId("site-rename-orchard-block-a");

    await user.click(pencil());
    await user.click(within(switcher).getByText("Sites"));
    expect(patch).not.toHaveBeenCalled();

    await user.click(pencil());
    const field = within(switcher).getByRole("textbox", { name: /New name/ });
    await user.clear(field);
    await user.type(field, "South orchard");
    await user.tab();
    expect(patch).toHaveBeenCalledTimes(1);
    expect(patch.mock.calls[0]?.[1]).toMatchObject({ body: { name: "South orchard" } });
  });
});
