/**
 * Set real size (features/inspector/RealSize.tsx, lib/realSize.ts): measure a length on the
 * scan, say how long it really is, preview the scale that says, save it.
 *
 * The scene is faked down to what the tool calls -- `SiteManager.previewScale`/`shownScale` and
 * `ScaleMeasure` -- and the API at `api.PUT`. What is pinned is the request: the body
 * `PUT /assets/{id}/scale` validates (a measured length must imply the scale sent beside it,
 * within a thousandth), the 401 that asks for the write token and the retry after it, the
 * reset, and Escape putting the scan back without sending anything.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Site, SiteAsset } from "@twin/contracts";
import { GlassTooltipProvider } from "@twin/ui";

import { api } from "@/api/client";
import type { CesiumSceneManager } from "@/cesium/CesiumSceneManager";
import type { ScaleMeasureState } from "@/cesium/ScaleMeasure";
import { sceneRegistry } from "@/cesium/SceneContext";
import { InspectorPanel } from "@/features/inspector/InspectorPanel";
import { RealSize } from "@/features/inspector/RealSize";
import {
  formatScale,
  measuredScaleBody,
  parseLength,
  parseScale,
  scalableAsset,
  scaleFromLength,
} from "@/lib/realSize";
import { useSelection } from "@/state/selection";
import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

const SITE_ID = "site-1";
const ASSET_ID = "asset-splat";

function splat(over: Partial<SiteAsset> = {}, renderConfig: Record<string, unknown> = {}) {
  return {
    id: ASSET_ID,
    siteId: SITE_ID,
    provider: "3d-tiles-url",
    name: "Splat",
    representation: "gaussian-splat",
    source: { type: "3d-tiles-url", url: "https://example.invalid/splat/tileset.json" },
    footprint: null,
    provenance: { georefMethod: "exif-gps", scaleSource: "unresolved", uncertaintyM: 10 },
    renderConfig: { maximumScreenSpaceError: 16, clipsWorld: true, ...renderConfig },
    defaultVisible: true,
    attribution: [],
    createdAt: "2026-01-02T00:00:00Z",
    updatedAt: "2026-01-02T00:00:00Z",
    ...over,
  } as unknown as SiteAsset;
}

function registered(assets: SiteAsset[], metadata?: Record<string, unknown>): Site {
  return {
    id: SITE_ID,
    slug: "yard",
    name: "Back yard",
    boundary: { type: "Polygon", coordinates: [] },
    centroid: { longitude: 0, latitude: 0, height: 0 },
    metadata: metadata ?? {
      captureId: "capture-1",
      registration: { georef: { lat: 47.6, lon: -122.1, height: 30 } },
    },
    assets,
    cameraBookmarks: [],
  } as unknown as Site;
}

describe("which scans can be resized", () => {
  it("is the splat a pipeline run registered, as the API decides it", () => {
    const scan = splat();
    expect(scalableAsset(registered([scan]))?.id).toBe(ASSET_ID);
    // No run recorded, or no origin: nothing says where the tiles are placed.
    expect(scalableAsset(registered([scan], {}))).toBeNull();
    expect(
      scalableAsset(registered([scan], { captureId: "c", registration: { georef: {} } })),
    ).toBeNull();
    // A Cesium ion splat, a mesh: 409 not_scalable.
    expect(
      scalableAsset(
        registered([
          splat({ provider: "cesium-ion", source: { type: "cesium-ion", assetId: 4 } as never }),
        ]),
      ),
    ).toBeNull();
    expect(scalableAsset(registered([splat({ representation: "mesh" })]))).toBeNull();
    // Only the site's first splat by creation: the one a re-run repoints.
    const later = splat({ id: "asset-later", createdAt: "2026-02-01T00:00:00Z" });
    expect(scalableAsset(registered([later, scan]))?.id).toBe(ASSET_ID);
    expect(scalableAsset(null)).toBeNull();
  });
});

describe("the arithmetic", () => {
  it("is the API's: drawn-at × true ÷ measured, within its thousandth", () => {
    expect(scaleFromLength(4, 1, 1)).toBe(0.25);
    expect(scaleFromLength(2, 1, 0.5)).toBe(0.25);
    const body = measuredScaleBody({ measuredM: 4.1234, trueM: 1.0, atScale: 0.8 });
    const evidence = body.evidence;
    expect(evidence).toMatchObject({
      method: "measured-length",
      measuredLengthM: 4.1234,
      trueLengthM: 1,
      measuredAtScale: 0.8,
    });
    // apps/api schemas/asset.py `_a_scale_or_a_reset`.
    const implied =
      ((evidence?.measuredAtScale ?? 1) * (evidence?.trueLengthM ?? 0)) /
      (evidence?.measuredLengthM ?? 1);
    expect(Math.abs(implied - (body.scale ?? 0))).toBeLessThanOrEqual(1e-3 * (body.scale ?? 0));
  });

  it("reads lengths and scales as people type them", () => {
    expect(parseLength("1.8")).toBe(1.8);
    expect(parseLength("180 cm")).toBeCloseTo(1.8, 12);
    expect(parseLength("6 ft")).toBeCloseTo(1.8288, 12);
    expect(parseLength('72"')).toBeCloseTo(1.8288, 12);
    expect(parseLength("1,5 m")).toBe(1.5);
    expect(parseLength("0")).toBeNull();
    expect(parseLength("tall")).toBeNull();
    expect(parseScale("0.25")).toBe(0.25);
    expect(parseScale("×0.25")).toBe(0.25);
    expect(parseScale("25%")).toBe(0.25);
    expect(parseScale("0.001")).toBeNull();
    expect(parseScale("1000")).toBeNull();
    expect(formatScale(0.242718)).toBe("×0.2427");
    expect(formatScale(1)).toBe("×1");
  });
});

/** The scene as the tool reaches it: the preview and the two-point measure. */
function fakeScene(shown = 1) {
  let listener: ((state: ScaleMeasureState) => void) | null = null;
  const scene = {
    sites: {
      previewScale: vi.fn(() => true),
      shownScale: vi.fn(() => shown),
    },
    scaleMeasure: {
      start: vi.fn((_siteId: string, onChange: (state: ScaleMeasureState) => void) => {
        listener = onChange;
        onChange({ picking: true, points: 0, lengthM: null, missed: false });
        return true;
      }),
      markCentre: vi.fn(() => true),
      clear: vi.fn(),
    },
  };
  /** Both points placed, `lengthM` apart as drawn. */
  const measure = (lengthM: number) =>
    act(() => listener?.({ picking: false, points: 2, lengthM, missed: false }));
  return { scene, measure };
}

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

function answer(status: number, data?: unknown) {
  return {
    data: status < 300 ? data : undefined,
    error: status < 300 ? undefined : { title: "Unauthorized", status },
    response: new Response(null, { status }),
  };
}

/** The PUTs the tool sent: its path parameter and its body. */
function puts(spy: { mock: { calls: unknown[][] } }): { assetId: string; body: unknown }[] {
  return spy.mock.calls.map((call) => {
    const init = call[1] as { params: { path: { asset_id: string } }; body: unknown };
    return { assetId: init.params.path.asset_id, body: init.body };
  });
}

beforeEach(() => {
  useSettings.setState({ writeToken: "" });
});

afterEach(() => {
  vi.restoreAllMocks();
  sceneRegistry.set(null);
});

describe("RealSize", () => {
  it("is not offered for a scan the API cannot resize", () => {
    render(wrap(<RealSize site={registered([splat()], {})} />));
    expect(screen.queryByTestId("real-size")).not.toBeInTheDocument();
  });

  it("measures on the scan, previews the scale the true length says, and saves it", async () => {
    const user = userEvent.setup();
    const { scene, measure } = fakeScene(1);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const saved = splat({}, { scale: 0.25, scaleEvidence: { method: "measured-length" } });
    const put = vi.spyOn(api, "PUT").mockResolvedValue(answer(200, saved) as never);
    render(wrap(<RealSize site={registered([splat()])} />));
    expect(screen.getByTestId("real-size-current")).toHaveTextContent("as registered");

    await user.click(screen.getByTestId("real-size-open"));
    await user.click(screen.getByTestId("real-size-measure"));
    expect(scene.scaleMeasure.start).toHaveBeenCalledWith(SITE_ID, expect.any(Function));
    expect(screen.getByTestId("real-size-picking")).toHaveTextContent("point 1 of 2");
    // The keyboard's way to place a point.
    await user.click(screen.getByTestId("real-size-centre"));
    expect(scene.scaleMeasure.markCentre).toHaveBeenCalled();

    measure(4);
    expect(screen.getByTestId("real-size-measured")).toHaveTextContent("4");
    await user.type(screen.getByTestId("real-size-true"), "1 m");
    // Previewed live: a quarter of the size.
    await waitFor(() => expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, 0.25));
    expect(screen.getByTestId("real-size-preview")).toHaveTextContent("×0.25");

    const previews = scene.sites.previewScale.mock.calls.length;
    await user.click(screen.getByTestId("real-size-save"));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    expect(put.mock.calls[0]?.[0]).toBe("/api/v1/assets/{asset_id}/scale");
    expect(puts(put)[0]).toEqual({
      assetId: ASSET_ID,
      body: {
        scale: 0.25,
        evidence: {
          method: "measured-length",
          measuredLengthM: 4,
          trueLengthM: 1,
          measuredAtScale: 1,
        },
      },
    });
    // Saved: the record draws the scan at it, so the preview is not undone on the way out.
    await waitFor(() => expect(screen.getByTestId("real-size-open")).toBeInTheDocument());
    expect(scene.sites.previewScale.mock.calls.slice(previews)).not.toContainEqual([SITE_ID, null]);
    expect(scene.scaleMeasure.clear).toHaveBeenCalled();
  });

  it("says the scale it measured at when a preview was showing", async () => {
    const user = userEvent.setup();
    const { scene, measure } = fakeScene(0.5);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const put = vi.spyOn(api, "PUT").mockResolvedValue(answer(200, splat()) as never);
    render(wrap(<RealSize site={registered([splat({}, { scale: 0.5 })])} />));
    await user.click(screen.getByTestId("real-size-open"));
    // The true length typed first, the measurement after: it still counts.
    await user.click(screen.getByTestId("real-size-measure"));
    measure(2);
    await user.type(screen.getByTestId("real-size-true"), "180 cm");
    await waitFor(() =>
      expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, 0.5 * (1.8 / 2)),
    );
    await user.keyboard("{Enter}");
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    expect(puts(put)[0]?.body).toMatchObject({
      scale: 0.45,
      evidence: { measuredLengthM: 2, trueLengthM: 1.8, measuredAtScale: 0.5 },
    });
  });

  it("asks for the write token on a 401, then sends the same request again", async () => {
    const user = userEvent.setup();
    const { scene } = fakeScene(1);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const put = vi
      .spyOn(api, "PUT")
      .mockResolvedValueOnce(answer(401) as never)
      .mockResolvedValueOnce(answer(200, splat({}, { scale: 0.5 })) as never);
    render(wrap(<RealSize site={registered([splat()])} />));
    await user.click(screen.getByTestId("real-size-open"));
    await user.type(screen.getByTestId("real-size-scale"), "0.5");
    await waitFor(() => expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, 0.5));
    await user.click(screen.getByTestId("real-size-save"));

    const form = await screen.findByTestId("write-token-form");
    expect(form).toHaveTextContent("save a scan's size");
    expect(puts(put)[0]?.body).toEqual({ scale: 0.5, evidence: { method: "direct" } });
    await user.type(screen.getByTestId("write-token-input"), "secret");
    await user.click(screen.getByTestId("write-token-save"));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(2));
    expect(useSettings.getState().writeToken).toBe("secret");
    expect(puts(put)[1]?.body).toEqual({ scale: 0.5, evidence: { method: "direct" } });
    await waitFor(() => expect(screen.queryByTestId("write-token-form")).not.toBeInTheDocument());
  });

  it("resets to the model as registered", async () => {
    const user = userEvent.setup();
    const { scene } = fakeScene(0.25);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const put = vi.spyOn(api, "PUT").mockResolvedValue(answer(200, splat()) as never);
    render(wrap(<RealSize site={registered([splat({}, { scale: 0.25 })])} />));
    expect(screen.getByTestId("real-size-current")).toHaveTextContent("×0.25");
    await user.click(screen.getByTestId("real-size-open"));
    await user.click(screen.getByTestId("real-size-reset"));
    await waitFor(() => expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, 1));
    await user.click(screen.getByTestId("real-size-save"));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    expect(puts(put)[0]?.body).toEqual({ reset: true });
  });

  it("puts the scan back as it was on Escape, and sends nothing", async () => {
    const user = userEvent.setup();
    const { scene, measure } = fakeScene(1);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const put = vi.spyOn(api, "PUT");
    render(wrap(<RealSize site={registered([splat()])} />));
    await user.click(screen.getByTestId("real-size-open"));
    await user.click(screen.getByTestId("real-size-measure"));
    measure(3);
    await user.type(screen.getByTestId("real-size-true"), "1.5");
    await waitFor(() => expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, 0.5));

    // From inside the field, too: Escape is the edit's before it is the app's step back.
    const behind = vi.fn();
    window.addEventListener("keydown", behind);
    await user.keyboard("{Escape}");
    expect(scene.sites.previewScale).toHaveBeenLastCalledWith(SITE_ID, null);
    expect(scene.scaleMeasure.clear).toHaveBeenCalled();
    expect(screen.getByTestId("real-size-open")).toBeInTheDocument();
    expect(put).not.toHaveBeenCalled();
    expect((behind.mock.calls[0]?.[0] as KeyboardEvent | undefined)?.defaultPrevented).toBe(true);
    window.removeEventListener("keydown", behind);
  });

  it("does not save a scale the API would refuse", async () => {
    const user = userEvent.setup();
    const { scene } = fakeScene(1);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    render(wrap(<RealSize site={registered([splat()])} />));
    await user.click(screen.getByTestId("real-size-open"));
    await user.type(screen.getByTestId("real-size-scale"), "500");
    expect(screen.getByTestId("real-size-scale")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByTestId("real-size-save")).toBeDisabled();
    // The true length waits for a measurement.
    expect(screen.getByTestId("real-size-true")).toBeDisabled();
  });

  it("sits on the site card, and the card says how the scale was set once it is saved", async () => {
    const user = userEvent.setup();
    const { scene, measure } = fakeScene(1);
    sceneRegistry.set(scene as unknown as CesiumSceneManager);
    const before = registered([splat()]);
    const after = registered([
      splat(
        {
          provenance: { georefMethod: "exif-gps", scaleSource: "manual", uncertaintyM: 10 },
        },
        {
          scale: 0.25,
          scaleEvidence: {
            method: "measured-length",
            measuredLengthM: 4,
            trueLengthM: 1,
            measuredAtScale: 1,
          },
        },
      ),
    ]);
    let record = before;
    const get = vi
      .spyOn(api, "GET")
      .mockImplementation(() => Promise.resolve(answer(200, record)) as never);
    vi.spyOn(api, "PUT").mockImplementation(() => {
      record = after;
      return Promise.resolve(answer(200, after.assets[0])) as never;
    });
    useSelection.getState().setSelection({
      kind: "site",
      title: "Back yard",
      longitude: -122.1,
      latitude: 47.6,
      height: 30,
      terrainHeight: 28,
      siteId: SITE_ID,
      at: Date.now(),
    });
    useUi.getState().setInspectorOpen(true);
    render(wrap(<InspectorPanel />));
    expect(await screen.findByTestId("inspector-placement")).toHaveTextContent("scale unresolved");
    expect(screen.getByTestId("real-size")).toHaveTextContent("Back yard");

    await user.click(screen.getByTestId("real-size-open"));
    await user.click(screen.getByTestId("real-size-measure"));
    measure(4);
    await user.type(screen.getByTestId("real-size-true"), "1");
    const fetched = get.mock.calls.length;
    await user.click(screen.getByTestId("real-size-save"));
    await waitFor(() =>
      expect(screen.getByTestId("inspector-placement")).toHaveTextContent(
        "scaled by hand from a measured length",
      ),
    );
    expect(screen.getByTestId("real-size-current")).toHaveTextContent("×0.25");
    // The record fetched afresh for the boundary the API moved with the scale.
    await waitFor(() => expect(get.mock.calls.length).toBeGreaterThan(fetched));
    useUi.getState().setInspectorOpen(false);
    useSelection.getState().setSelection(null);
  });
});
