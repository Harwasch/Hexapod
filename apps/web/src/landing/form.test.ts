import { parseAcres, validateRequest, type SiteRequest } from "./form";

const complete: SiteRequest = {
  name: "Ada",
  organization: "",
  email: "ada@example.com",
  acres: "",
  message: "",
  areaCount: 1,
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
    expect(validateRequest({ ...complete, areaCount: 0, acres: "1,200" })).toEqual({});
  });

  it("asks for an area when there is neither a selection nor acres", () => {
    expect(validateRequest({ ...complete, areaCount: 0 }).area).toBeDefined();
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
