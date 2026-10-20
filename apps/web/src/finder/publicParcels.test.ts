import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CandidateResult, LandCandidate } from "@twin/contracts";

import type * as PublicParcelsModule from "./publicParcels";

type Module = typeof PublicParcelsModule;

/** The module loaded against a stand-in API at an absolute origin (Node's fetch needs one). */
async function load(): Promise<Module> {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE_URL", "http://api.test");
  return import("./publicParcels");
}

const answer: CandidateResult = {
  status: "available",
  message: "Parcels from Kern County records near your click.",
  candidates: [],
  truncated: false,
  notices: [{ source: "Kern County parcels", text: "SUMMARY: This disclaimer statement…" }],
};

let fetched: Request[] = [];
function respond(response: Response) {
  vi.stubGlobal(
    "fetch",
    vi.fn((request: Request) => {
      fetched.push(request);
      return Promise.resolve(response);
    }),
  );
}

beforeEach(() => {
  fetched = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("findPublicParcels", () => {
  it("asks the public route, with no credential, and returns its answer and notices", async () => {
    const { findPublicParcels } = await load();
    respond(Response.json(answer));
    const result = await findPublicParcels([-119.02, 35.37], new AbortController().signal);
    expect(result.notices).toEqual(answer.notices);
    const [request] = fetched;
    expect(request?.method).toBe("POST");
    expect(request?.url).toBe("http://api.test/api/v1/public/parcels/near");
    expect(request?.headers.get("Authorization")).toBeNull();
    expect(await request?.json()).toEqual({
      point: { type: "Point", coordinates: [-119.02, 35.37] },
    });
  });

  it("says to wait when this visitor has asked too often (429)", async () => {
    const { findPublicParcels } = await load();
    respond(Response.json({ title: "Too many requests" }, { status: 429 }));
    await expect(findPublicParcels([-122.27, 37.8], new AbortController().signal)).rejects.toThrow(
      /Wait a few seconds/,
    );
  });

  it("says the lookups are busy when every slot is taken (503)", async () => {
    const { findPublicParcels } = await load();
    respond(Response.json({ title: "Busy" }, { status: 503, headers: { "Retry-After": "5" } }));
    await expect(findPublicParcels([-122.27, 37.8], new AbortController().signal)).rejects.toThrow(
      /busy right now/,
    );
  });

  it("names the status of any other failure", async () => {
    const { findPublicParcels } = await load();
    respond(Response.json({ title: "Server error" }, { status: 500 }));
    await expect(findPublicParcels([-122.27, 37.8], new AbortController().signal)).rejects.toThrow(
      "Parcel records could not be looked up (HTTP 500).",
    );
  });
});

describe("foundParcels", () => {
  const parcel = (id: string, county: string): LandCandidate => ({
    id,
    label: `Parcel ${id}`,
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [-119.02, 35.37],
          [-119.01, 35.37],
          [-119.01, 35.38],
          [-119.02, 35.37],
        ],
      ],
    },
    distanceM: 0,
    properties: {},
    source: {
      method: "parcel",
      meaning: "recorded-parcel",
      label: `${county} County assessor parcel`,
      attribution: `${county} County`,
    },
  });

  it("carries each source's notice once, with the parcels it came with", async () => {
    const { foundParcels } = await load();
    const kern = [parcel("ca-parcel/06029/1", "Kern"), parcel("ca-parcel/06029/2", "Kern")];
    const land = foundParcels(kern, () => answer.notices ?? []);
    expect(land.notices).toEqual(answer.notices);
    expect(land.parcelIds).toEqual(["ca-parcel/06029/1", "ca-parcel/06029/2"]);
    expect(land.boundary.type).toBe("MultiPolygon");
    expect(foundParcels([parcel("ca-parcel/06001/1", "Alameda")]).notices).toEqual([]);
  });
});
