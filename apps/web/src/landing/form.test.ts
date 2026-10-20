import { buildPayload, parseAcres, validateRequest, type SiteRequest } from "./form";

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
