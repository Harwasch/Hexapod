import { afterEach, describe, expect, it } from "vitest";

import {
  requestedSkinVariant,
  skinVariantFor,
  skinVariantsOf,
  useSkinVariant,
} from "@/state/skinVariant";

const EXTRAS = {
  skin: { uri: "skin.json", count: 2 },
  variants: {
    objects: [{ name: "ground-first", instances: "variants/objects/g/instances.json" }],
    skins: [
      {
        name: "freeform",
        label: "FreeForm · size rule",
        about: "Today's method.",
        skin: "variants/skins/freeform/skin.json",
      },
      { name: "tetfem-stiff", skin: "variants/skins/tetfem-stiff/skin.json" },
      { name: "broken" },
      "not an entry",
    ],
  },
};

afterEach(() => useSkinVariant.setState({ chosen: {} }));

describe("skin variants", () => {
  it("reads extras.variants.skins, skipping what is malformed", () => {
    const found = skinVariantsOf(EXTRAS);
    expect(found.map((v) => v.name)).toEqual(["freeform", "tetfem-stiff"]);
    expect(found[1]?.label).toBe("tetfem-stiff");
    expect(skinVariantsOf({})).toEqual([]);
    expect(skinVariantsOf(null)).toEqual([]);
  });

  it("takes the temporary URL parameter, and nothing from an empty one", () => {
    expect(requestedSkinVariant("?skinVariant=tetfem-stiff")).toBe("tetfem-stiff");
    expect(requestedSkinVariant("?skinVariant=")).toBeNull();
    expect(requestedSkinVariant("?other=1")).toBeNull();
  });

  it("draws the chosen candidate, else the URL's, else the scan's own skin", () => {
    expect(skinVariantFor("a", EXTRAS, "")).toBeNull();
    expect(skinVariantFor("a", EXTRAS, "?skinVariant=freeform")?.skin).toBe(
      "variants/skins/freeform/skin.json",
    );
    // A name the scan does not declare changes nothing.
    expect(skinVariantFor("a", EXTRAS, "?skinVariant=pinned-stiff")).toBeNull();
    // A choice for the scan wins over the URL; choosing null is the scan's own skin.
    useSkinVariant.getState().select("a", "tetfem-stiff");
    expect(skinVariantFor("a", EXTRAS, "?skinVariant=freeform")?.name).toBe("tetfem-stiff");
    useSkinVariant.getState().select("a", null);
    expect(skinVariantFor("a", EXTRAS, "?skinVariant=freeform")).toBeNull();
    // Another scan still follows the URL.
    expect(skinVariantFor("b", EXTRAS, "?skinVariant=freeform")?.name).toBe("freeform");
  });
});
