/**
 * "More like this" on the selection card (features/sites/ObjectCard,
 * features/sites/MoreLikeThis): offered where the scan's objects carry image embeddings, it
 * reads them, lights the objects that look like the chosen one through the objects store, and
 * moves that set with "How alike"; it always says the match is inferred.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GlassTooltipProvider } from "@twin/ui";

import type * as SplatInstances from "@/cesium/splatInstances";
import type { EmbeddingSource } from "@/cesium/splatInstances";
import { SceneSelectController } from "@/cesium/sceneSelect/SceneSelectController";
import { ObjectCard } from "@/features/sites/ObjectCard";
import { parseInstances, withDescendants } from "@/lib/instances";
import { scanPayloads } from "@/lib/payloadCache";
import { similarText, useInstances } from "@/state/instances";
import { useSceneSelect } from "@/state/sceneSelect";

const ASSET = "camp";
const DIM = 4;

const box = { min: [0, 0, 0], max: [1, 1, 1] };
const tagged = (id: number, label: string, parent: number | null = null) => ({
  id,
  parent,
  bounds: box,
  splats: 100 + id,
  tags: [{ label, score: 0.6 }],
});

/**
 * Shrub A (1) in the forest floor (6), with a described part (7) and one too small to describe
 * (8); shrub B (2) and a bush (11) that look like it, at centred cosines of about 0.86 and
 * 0.48; a rock, a conifer and a stone that do not. Id 5 was never described.
 */
const DOC = parseInstances({
  format: "hexapod.instances",
  version: 1,
  instances: [
    tagged(1, "shrub", 6),
    tagged(2, "shrub"),
    tagged(3, "rock"),
    tagged(4, "conifer"),
    { id: 5, bounds: box, splats: 5, tags: [] },
    tagged(6, "forest floor"),
    tagged(7, "bush", 1),
    { id: 8, parent: 1, bounds: box, splats: 5, tags: [] },
    tagged(9, "rock"),
    tagged(10, "rock"),
    tagged(11, "bush"),
    tagged(12, "rock"),
  ],
  tiles: {},
  embedding: { file: "instances.emb", model: "siglip:google/siglip2-base-patch16-224", dim: DIM },
})!;

const VECTORS: Record<number, number[]> = {
  1: [4, 1, 0, 0],
  2: [4, 0.8, 0.3, 0],
  3: [4, 0, 0, 1],
  4: [4, -0.3, 1, 0.2],
  6: [4, 0.3, 0.3, 0.3],
  7: [4, 1, 0.1, 0],
  11: [4, 0.6, 0.4, 0.1],
  12: [4, 0.5, 0.3, 0.5],
};

/** A number as IEEE 754 half bits. */
function toHalf(value: number): number {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0] ?? 0;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  if (exponent <= 0) return (bits >>> 16) & 0x8000;
  return ((bits >>> 16) & 0x8000) | ((exponent << 10) + (((bits & 0x7fffff) + 0x1000) >> 13));
}

function embeddingBytes(): ArrayBuffer {
  const out = new Uint16Array(12 * DIM);
  for (const [id, vector] of Object.entries(VECTORS)) {
    const norm = Math.hypot(...vector);
    vector.forEach((v, j) => (out[(Number(id) - 1) * DIM + j] = toHalf(v / norm)));
  }
  return out.buffer;
}

const SOURCE: EmbeddingSource = {
  url: "https://bucket.test/camp/instances.emb",
  rows: 12,
  dim: DIM,
  model: "siglip:google/siglip2-base-patch16-224",
};
let source: EmbeddingSource | null = SOURCE;

vi.mock("@/cesium/splatInstances", async (original) => ({
  ...(await original<typeof SplatInstances>()),
  instancesDocOf: (assetId: string) => (assetId === ASSET ? DOC : undefined),
  embeddingOf: (assetId: string) => (assetId === ASSET ? source : null),
}));

function viewer() {
  return {
    scene: {
      screenSpaceCameraController: { enableInputs: true },
      requestRender: () => undefined,
    },
    camera: {},
    canvas: document.body.appendChild(document.createElement("canvas")),
  };
}

function wrap(children: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return (
    <QueryClientProvider client={client}>
      <GlassTooltipProvider>{children}</GlassTooltipProvider>
    </QueryClientProvider>
  );
}

const lit = (): number[] =>
  [...(useInstances.getState().assets[ASSET]?.highlighted ?? [])].sort((a, b) => a - b);

let controller: SceneSelectController;

beforeEach(() => {
  source = SOURCE;
  scanPayloads.clear();
  useSceneSelect.getState().clear();
  useSceneSelect.getState().setMode("pick");
  useInstances.setState({ assets: {}, gaps: {} });
  useInstances.getState().setTable(ASSET, DOC);
  controller = new SceneSelectController(viewer() as never, { ownKeys: false });
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(embeddingBytes()))),
  );
});

afterEach(() => {
  controller.destroy();
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Chooses `id` in the scene, as a click on it does (with the forest floor above it). */
function choose(id: number): void {
  act(() => useSceneSelect.getState().select(ASSET, [id, 6], 2, 0, null));
}

describe("More like this", () => {
  it("is offered only where the scan's objects carry embeddings", () => {
    choose(1);
    source = null;
    const { unmount } = render(wrap(<ObjectCard controller={controller} />));
    expect(screen.queryByRole("button", { name: /More like this/ })).not.toBeInTheDocument();
    unmount();
    source = SOURCE;
    render(wrap(<ObjectCard controller={controller} />));
    expect(screen.getByRole("button", { name: /More like this/ })).toBeInTheDocument();
  });

  it("is not offered for a combination the brush chose", () => {
    act(() => useSceneSelect.getState().selectSet(ASSET, { ids: [1, 2], iou: 0.9 }, [], null));
    render(wrap(<ObjectCard controller={controller} />));
    expect(screen.queryByRole("button", { name: /More like this/ })).not.toBeInTheDocument();
  });

  it("lights what looks alike, moves it with How alike, and always says it is inferred", async () => {
    const user = userEvent.setup();
    choose(1);
    render(wrap(<ObjectCard controller={controller} />));
    // The selection is lit as ever until then.
    expect(lit()).toEqual([...withDescendants(DOC, [1])].sort((a, b) => a - b));
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    const count = await screen.findByTestId("more-like-this-count");
    expect(fetch).toHaveBeenCalledWith(SOURCE.url);
    // At 60%: shrub B, with the shrub compared and its parts.
    expect(screen.getByTestId("more-like-this-threshold")).toHaveTextContent("60%");
    expect(count).toHaveTextContent("1 object · Shrubs & bushes");
    expect(lit()).toEqual([1, 2, 7, 8]);
    expect(useInstances.getState().assets[ASSET]?.focus).toEqual({ kind: "similar", seedId: 1 });
    const inferred = screen.getByTestId("more-like-this-inferred");
    expect(inferred).toHaveTextContent(/Inferred/);
    expect(inferred).toHaveTextContent(/SigLIP 2/);
    expect(inferred).toHaveTextContent(/not a species ID/);

    const thumb = screen.getByRole("slider", { name: "How alike" });
    act(() => thumb.focus());
    await user.keyboard("{Home}");
    expect(screen.getByTestId("more-like-this-threshold")).toHaveTextContent("30%");
    expect(count).toHaveTextContent("2 objects · mostly Shrubs & bushes (2)");
    expect(lit()).toEqual([1, 2, 7, 8, 11]);
    expect(screen.getByTestId("more-like-this-inferred")).toHaveTextContent(/Inferred/);

    await user.keyboard("{End}");
    expect(screen.getByTestId("more-like-this-threshold")).toHaveTextContent("95%");
    expect(count).toHaveTextContent("No other object is this alike");
    // Only what was compared stays lit.
    expect(lit()).toEqual([1, 7, 8]);
    expect(screen.getByTestId("more-like-this-inferred")).toHaveTextContent(/Inferred/);
  });

  it("shows only the matches, and Clear gives the highlight back to the selection", async () => {
    const user = userEvent.setup();
    choose(1);
    render(wrap(<ObjectCard controller={controller} />));
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    await screen.findByTestId("more-like-this-count");
    await user.click(screen.getByRole("button", { name: /Show only these/ }));
    // Everything but the lit set and what holds it (the forest floor holds shrub A).
    expect(
      [...(useInstances.getState().assets[ASSET]?.hidden ?? [])].sort((a, b) => a - b),
    ).toEqual([3, 4, 5, 9, 10, 11, 12]);
    await user.click(screen.getByRole("button", { name: /Clear/ }));
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();
    expect(lit()).toEqual([1, 7, 8]);
    expect(screen.getByRole("button", { name: /More like this/ })).toBeInTheDocument();
  });

  it("compares a part too small to describe by the object it is part of", async () => {
    const user = userEvent.setup();
    choose(8);
    render(wrap(<ObjectCard controller={controller} />));
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    expect(await screen.findByTestId("more-like-this-compared-by")).toHaveTextContent(
      "Compared by Shrub, the object it is part of.",
    );
    expect(useInstances.getState().assets[ASSET]?.similar).toMatchObject({
      seedId: 8,
      seedUsed: 1,
    });
    expect(lit()).toEqual([1, 2, 7, 8]);
  });

  it("says how much it reads, and what went wrong when it cannot", async () => {
    const user = userEvent.setup();
    let answer: (response: Response) => void = () => undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            answer = resolve;
          }),
      ),
    );
    choose(1);
    render(wrap(<ObjectCard controller={controller} />));
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    const status = screen.getByTestId("more-like-this-status");
    expect(status).toHaveTextContent("Reading object signatures · 1 kB");
    expect(
      screen.getByRole("progressbar", { name: "Reading object signatures" }),
    ).toBeInTheDocument();
    act(() => answer(new Response("gone", { status: 404 })));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Could not read this scan's object signatures (object signatures answered 404). Try again.",
    );
    // And it can be asked again.
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(new Response(embeddingBytes()))),
    );
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    await waitFor(() => expect(screen.getByTestId("more-like-this-count")).toBeInTheDocument());
    expect(within(screen.getByTestId("selection-card")).queryByRole("alert")).toBeNull();
  });

  it("drops a read when another object is chosen meanwhile", async () => {
    const user = userEvent.setup();
    let answer: (response: Response) => void = () => undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            answer = resolve;
          }),
      ),
    );
    choose(1);
    render(wrap(<ObjectCard controller={controller} />));
    await user.click(screen.getByRole("button", { name: /More like this/ }));
    choose(2);
    await act(async () => {
      answer(new Response(embeddingBytes()));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(useInstances.getState().assets[ASSET]?.similar).toBeNull();
    expect(lit()).toEqual([2]);
    expect(screen.getByRole("button", { name: /More like this/ })).toBeInTheDocument();
  });
});

describe("similarText", () => {
  const category = (id: string, name: string) => ({ id, name, color: "#000" });
  it("counts objects and names what they mostly are", () => {
    expect(similarText({ objects: 0, categories: [] })).toMatch(/^No other object/);
    expect(
      similarText({
        objects: 48,
        categories: [{ category: category("shrubs", "Shrubs"), count: 37 }],
      }),
    ).toBe("48 objects · mostly Shrubs (37)");
    expect(
      similarText({
        objects: 10,
        categories: [
          { category: category("trees", "Trees"), count: 4 },
          { category: category("ground", "Ground"), count: 3 },
          { category: category("shrubs", "Shrubs"), count: 3 },
        ],
      }),
    ).toBe("10 objects · Trees (4), Ground (3), …");
  });
});
