import {
  buildPayload,
  formatAcres,
  parseAcres,
  sendRequest,
  validateRequest,
  type SiteRequest,
} from "./form";

const complete: SiteRequest = {
  name: "Ada",
  organization: "",
  email: "ada@example.com",
  acres: "",
  message: "",
  hasBoundary: true,
};

describe("validateRequest", () => {
  it("accepts a name, an email and an area on the map", () => {
    expect(validateRequest(complete)).toEqual({});
  });

  it("asks for a name and an email", () => {
    const errors = validateRequest({ ...complete, name: "  ", email: "" });
    expect(Object.keys(errors).sort()).toEqual(["email", "name"]);
  });

  it("rejects an email without a domain", () => {
    expect(validateRequest({ ...complete, email: "ada@" }).email).toMatch(/name@example\.com/);
  });

  it("takes typed acres in place of a map selection", () => {
    expect(validateRequest({ ...complete, hasBoundary: false, acres: "1,200" })).toEqual({});
  });

  it("asks for an area when there is neither a selection nor acres", () => {
    expect(validateRequest({ ...complete, hasBoundary: false }).area).toBeDefined();
  });

  it("rejects acres that are not a positive number", () => {
    expect(validateRequest({ ...complete, acres: "lots" }).acres).toBeDefined();
    expect(validateRequest({ ...complete, acres: "0" }).acres).toBeDefined();
  });
});

describe("parseAcres", () => {
  it("ignores thousands separators and reads blanks as no number", () => {
    expect(parseAcres(" 1,200.5 ")).toBe(1200.5);
    expect(parseAcres("")).toBeNaN();
  });
});

describe("buildPayload", () => {
  const boundary = {
    type: "Polygon" as const,
    coordinates: [
      [
        [-122.27, 37.8],
        [-122.26, 37.8],
        [-122.26, 37.81],
        [-122.27, 37.8],
      ],
    ],
  };

  it("carries exactly the name, email, notes, boundary and parcel ids", () => {
    const payload = buildPayload(
      { ...complete, name: " Ada ", message: "Broom on the north slope." },
      { boundary, parcelIds: ["ca-parcel/06001/2-51-1"] },
    );
    expect(Object.keys(payload).sort()).toEqual([
      "boundary",
      "email",
      "name",
      "notes",
      "parcelIds",
    ]);
    expect(payload).toEqual({
      name: "Ada",
      email: "ada@example.com",
      notes: "Broom on the north slope.",
      boundary,
      parcelIds: ["ca-parcel/06001/2-51-1"],
    });
  });

  it("opens the notes with the fields that have no place of their own", () => {
    const payload = buildPayload(
      { ...complete, organization: "Oak Bench Ranch", acres: "40", message: "Weeds." },
      null,
    );
    expect(payload.notes).toBe("Organization: Oak Bench Ranch\nArea (acres): 40\nWeeds.");
    expect(payload.boundary).toBeNull();
    expect(payload.parcelIds).toEqual([]);
  });
});

describe("formatAcres", () => {
  it("keeps a small lot from rounding to zero, so the acres check accepts it", () => {
    expect(formatAcres(4046.856 * 250)).toBe("250");
    expect(formatAcres(4046.856 * 12.44)).toBe("12.4");
    expect(formatAcres(4046.856 * 0.0342)).toBe("0.034");
    expect(formatAcres(150)).toBe("0.037");
    expect(validateRequest({ ...complete, hasBoundary: false, acres: formatAcres(150) })).toEqual(
      {},
    );
  });
});

describe("sendRequest", () => {
  const payload = buildPayload({ ...complete, acres: "12" }, null);
  const answering = (response: Response | Error) => {
    const calls: { input: string; init: RequestInit }[] = [];
    const fetchImpl = (input: string, init: RequestInit) => {
      calls.push({ input, init });
      return response instanceof Error ? Promise.reject(response) : Promise.resolve(response);
    };
    return { calls, fetchImpl };
  };

  it("posts the payload and the honeypot as JSON to the API's scan requests", async () => {
    const { calls, fetchImpl } = answering(Response.json({ status: "received" }, { status: 202 }));
    const outcome = await sendRequest(payload, "", { base: "https://api.example", fetchImpl });
    expect(outcome).toEqual({ kind: "received" });
    const [call] = calls;
    expect(call?.input).toBe("https://api.example/api/v1/scan-requests");
    expect(call?.init.method).toBe("POST");
    expect(JSON.parse(call?.init.body as string)).toEqual({ ...payload, website: "" });
  });

  it("says what to do when the request was refused", async () => {
    const said = async (response: Response | Error) =>
      sendRequest(payload, "", answering(response)).then((outcome) =>
        outcome.kind === "error" ? outcome.message : "received",
      );
    expect(await said(new Response(null, { status: 429 }))).toMatch(/Wait a minute/);
    expect(await said(new Response(null, { status: 413 }))).toMatch(/too detailed/);
    expect(
      await said(
        Response.json(
          { errors: [{ msg: "Value error, must be an email address like name@example.com" }] },
          { status: 422 },
        ),
      ),
    ).toBe(
      "Part of this could not be accepted: must be an email address like name@example.com. Check it and send it again.",
    );
    expect(await said(new Response(null, { status: 500 }))).toMatch(/HTTP 500/);
    expect(await said(new TypeError("Failed to fetch"))).toMatch(/could not reach the server/);
  });
});
