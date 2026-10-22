/**
 * The change map (`GET /sites/{id}/change`, ADR 0018 § API 8) is a read behind the write
 * token, like weak areas and fill targets: the client sends the token on it as on a write, and
 * a 401 on it asks for the token as a write's does. Other site reads stay anonymous.
 */
import { afterEach, describe, expect, it } from "vitest";

import { auth } from "@/api/client";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

const onRequest = auth.onRequest as (input: {
  request: Request;
}) => Promise<Request | undefined> | Request | undefined;
const onResponse = auth.onResponse as (input: { request: Request; response: Response }) => void;

const CHANGE = "https://api.invalid/api/v1/sites/site-1/change?from=e1&to=e2";

afterEach(() => {
  useSettings.getState().reset();
  useUi.setState({ writeTokenPrompt: false });
});

describe("the change map's token", () => {
  it("rides on the change map, and on no other site read", async () => {
    useSettings.getState().set({ writeToken: "pilot" });
    const change = await onRequest({ request: new Request(CHANGE) });
    expect(change?.headers.get("Authorization")).toBe("Bearer pilot");
    for (const path of ["/api/v1/sites/site-1", "/api/v1/sites/site-1/epochs", "/api/v1/sites"]) {
      const read = await onRequest({ request: new Request(`https://api.invalid${path}`) });
      expect(read?.headers.get("Authorization")).toBeNull();
    }
    // A path that only ends like it is not it.
    const other = await onRequest({
      request: new Request("https://api.invalid/api/v1/sites/site-1/exchange"),
    });
    expect(other?.headers.get("Authorization")).toBeNull();
  });

  it("is asked for after a 401 on the change map", () => {
    onResponse({ request: new Request(CHANGE), response: new Response(null, { status: 401 }) });
    expect(useUi.getState().writeTokenPrompt).toBe(true);
  });
});
