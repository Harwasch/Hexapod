import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";

import {
  assignCategories,
  CATEGORIES,
  categoryById,
  categoryOfLabel,
  indexCategories,
  matchCategory,
  OTHER_CATEGORY,
  tagCategory,
  type CategorisedInstance,
} from "@/lib/categories";
import { parseInstances } from "@/lib/instances";
import { matchObjects, useInstances } from "@/state/instances";

const LABELS = new Map([
  ["oak", "trees"],
  ["pine", "trees"],
  ["lawn", "grass"],
  ["bench", "furniture"],
  ["dirt", "ground"],
]);

function inst(
  id: number,
  parent: number | null,
  splats: number,
  tags: [string, number][] = [],
): CategorisedInstance {
  return { id, parent, splats, tags: tags.map(([label, score]) => ({ label, score })) };
}

describe("the category list and label mapping", () => {
  it("is the committed file the pipeline writes", () => {
    const file = JSON.parse(
      readFileSync(resolve(__dirname, "../../../../tools/captures/data/categories.json"), "utf8"),
    ) as { categories: { id: string }[]; labels: Record<string, string> };
    expect(CATEGORIES.map((c) => c.id)).toEqual(file.categories.map((c) => c.id));
    expect(CATEGORIES.length).toBeGreaterThanOrEqual(20);
    expect(CATEGORIES.length).toBeLessThanOrEqual(30);
    expect(CATEGORIES.at(-1)?.id).toBe(OTHER_CATEGORY);
    // Every label of the open vocabulary has a category this build knows.
    const vocabulary = readFileSync(
      resolve(__dirname, "../../../../tools/captures/data/open_vocabulary.txt"),
      "utf8",
    )
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"));
    for (const label of vocabulary) {
      const id = categoryOfLabel(label);
      expect(id, label).toBeDefined();
      expect(categoryById(id ?? "").id).toBe(id);
    }
  });

  it("puts common scene words where a person would look", () => {
    expect(categoryOfLabel("pumpkin")).toBe("produce");
    expect(categoryOfLabel("Conifer")).toBe("trees");
    expect(categoryOfLabel("bush")).toBe("shrubs");
    expect(categoryOfLabel("forest floor")).toBe("ground");
    expect(categoryOfLabel("stream")).toBe("water");
    expect(categoryOfLabel("bench")).toBe("furniture");
    expect(categoryOfLabel("cabin")).toBe("buildings");
    expect(categoryOfLabel("no such label")).toBeUndefined();
    expect(categoryById("nope").id).toBe(OTHER_CATEGORY);
  });
});

describe("assigning categories", () => {
  it("lets the tags vote by score", () => {
    const tags = [
      { label: "lawn", score: 0.4 },
      { label: "oak", score: 0.3 },
      { label: "pine", score: 0.2 },
      { label: "unknown", score: 0.9 },
    ];
    expect(tagCategory(tags, LABELS)).toBe("trees");
    expect(tagCategory([{ label: "unknown", score: 1 }], LABELS)).toBeNull();
    expect(tagCategory([], LABELS)).toBeNull();
  });

  it("falls back to tagged siblings, the nearest tagged ancestor, the parts below, then Other", () => {
    const instances = [
      inst(1, null, 10, [["oak", 0.5]]),
      inst(2, 1, 5),
      inst(3, 2, 5, [["bench", 0.4]]),
      inst(4, null, 1),
      inst(5, 4, 900, [["lawn", 0.3]]),
      inst(6, 4, 100, [["pine", 0.3]]),
      inst(7, null, 3),
      // A coarse "oak" region whose untagged part sits beside a lawn part: lawn.
      inst(10, null, 50, [["oak", 0.3]]),
      inst(11, 10, 20),
      {
        ...inst(12, 10, 80, [["lawn", 0.4]]),
        bounds: { min: [0, 0, 0] as const, max: [1, 1, 1] as const },
      },
      // A fragment with no tagged relative, inside the lawn's box: lawn.
      { ...inst(13, null, 2), centroid: [0.5, 0.5, 0.5] as const },
    ];
    expect(Object.fromEntries(assignCategories(instances, LABELS))).toEqual({
      1: "trees",
      2: "trees",
      3: "furniture",
      4: "grass",
      5: "grass",
      6: "trees",
      7: OTHER_CATEGORY,
      10: "trees",
      11: "grass",
      12: "grass",
      13: "grass",
    });
  });

  it("takes a category the file names, when this build knows it", () => {
    const instances = [
      { ...inst(1, null, 10, [["oak", 0.5]]), category: "water" },
      { ...inst(2, null, 10, [["oak", 0.5]]), category: "no-such-category" },
    ];
    const out = assignCategories(instances, LABELS);
    expect(out.get(1)).toBe("water");
    expect(out.get(2)).toBe("trees");
  });
});

describe("objects and groups", () => {
  // A ground region (1) holding a tree (2, with an untagged branch 3) and more ground (4);
  // two separate trees (5, 6) and an untagged root with nothing below it (7).
  const instances = [
    inst(1, null, 100, [["dirt", 0.6]]),
    inst(2, 1, 50, [
      ["lawn", 0.1],
      ["oak", 0.5],
    ]),
    inst(3, 2, 20),
    inst(4, 1, 30, [["dirt", 0.4]]),
    inst(5, null, 40, [["oak", 0.4]]),
    inst(6, null, 10, [["oak", 0.3]]),
    inst(7, null, 5),
  ];

  it("makes an object of each instance whose parent is another category, with its parts", () => {
    const index = indexCategories(instances, LABELS);
    expect(index.groups.map((g) => [g.category.id, g.objects.length, g.splats])).toEqual([
      ["ground", 1, 130],
      ["trees", 3, 120],
      [OTHER_CATEGORY, 1, 5],
    ]);
    const trees = index.groups[1];
    expect(trees?.objects.map((o) => [o.id, [...o.members].sort()])).toEqual([
      [2, [2, 3]],
      [5, [5]],
      [6, [6]],
    ]);
    // The ground object does not contain the tree inside it.
    expect([...(index.objects.get(1)?.members ?? [])].sort()).toEqual([1, 4]);
    expect(index.objectOf.get(3)).toBe(2);
    expect(index.objectOf.get(4)).toBe(1);
  });

  it("names objects by their own category's best tag, numbered when shared, never by id", () => {
    const index = indexCategories(instances, LABELS);
    // Object 2's best tag is "oak" (its higher "lawn" is another category's).
    expect(index.groups[1]?.objects.map((o) => o.name)).toEqual(["oak 1", "oak 2", "oak 3"]);
    expect(index.groups[0]?.objects.map((o) => o.name)).toEqual(["dirt"]);
    expect(index.objects.get(7)?.name).toBe("Other 1");
    for (const object of index.objects.values()) expect(object.name).not.toMatch(/Object/);
  });

  it("names an object by the file's own name first, numbered when shared", () => {
    // A concept-first file: the ground's cover classes as objects of Ground, named.
    const named: CategorisedInstance[] = [
      { ...inst(1, null, 100, [["lawn", 0.4]]), category: "ground", name: "Grass" },
      { ...inst(2, null, 50), category: "ground", name: "Gravel" },
      { ...inst(3, null, 40, [["oak", 0.4]]), category: "produce", name: "pumpkin" },
      { ...inst(4, 3, 10), category: "produce" },
      { ...inst(5, null, 30), category: "produce", name: "pumpkin" },
    ];
    const index = indexCategories(named, LABELS);
    expect(index.groups.map((g) => [g.category.id, g.objects.map((o) => o.name)])).toEqual([
      ["ground", ["Grass", "Gravel"]],
      ["produce", ["pumpkin 1", "pumpkin 2"]],
    ]);
    expect([...(index.objects.get(3)?.members ?? [])].sort()).toEqual([3, 4]);
  });

  it("keeps a file's name when it parses one", () => {
    const box = { min: [0, 0, 0], max: [1, 1, 1] };
    const doc = parseInstances({
      format: "hexapod.instances",
      version: 1,
      instances: [
        { id: 1, bounds: box, splats: 9, tags: [], name: " cable spool " },
        { id: 2, bounds: box, splats: 9, tags: [], name: "" },
      ],
      tiles: {},
    });
    expect(doc?.instances.map((i) => i.name)).toEqual(["cable spool", undefined]);
  });

  it("finds categories by name, plural or not", () => {
    const trees = categoryById("trees");
    expect(matchCategory(["trees"], trees)).toBe(1);
    expect(matchCategory(["tree"], trees)).toBe(1);
    expect(matchCategory(["rock"], categoryById("rock"))).toBe(1);
    expect(matchCategory(["bench"], trees)).toBe(0);
    expect(matchCategory([], trees)).toBe(0);
  });
});

describe("the store, by category", () => {
  const box = { min: [0, 0, 0], max: [1, 1, 1] };
  const doc = parseInstances({
    format: "hexapod.instances",
    version: 1,
    instances: [
      { id: 1, bounds: box, splats: 100, tags: [{ label: "dirt", score: 0.6 }] },
      { id: 2, parent: 1, bounds: box, splats: 50, tags: [{ label: "conifer", score: 0.5 }] },
      { id: 3, parent: 2, bounds: box, splats: 20 },
      { id: 4, bounds: box, splats: 40, tags: [{ label: "conifer", score: 0.4 }] },
      { id: 5, bounds: box, splats: 30, tags: [{ label: "picnic table", score: 0.4 }] },
    ],
    tiles: {},
  });
  const s = (): ReturnType<typeof useInstances.getState> => useInstances.getState();
  const sorted = (set: ReadonlySet<number> | undefined): number[] =>
    [...(set ?? [])].sort((a, b) => a - b);

  beforeEach(() => {
    useInstances.setState({ assets: {}, dimOthers: true, gaps: {} });
    if (doc) s().setTable("a", doc);
  });

  it("hides a category's members and nothing of another category under them", () => {
    s().setCategoryHidden("a", "ground", true);
    expect(sorted(s().assets.a?.hidden)).toEqual([1]);
    s().setCategoryHidden("a", "trees", true);
    expect(sorted(s().assets.a?.hidden)).toEqual([1, 2, 3, 4]);
    s().setCategoryHidden("a", "ground", false);
    expect(sorted(s().assets.a?.hidden)).toEqual([2, 3, 4]);
    s().setObjectsHidden("a", [4], false);
    expect(sorted(s().assets.a?.hidden)).toEqual([2, 3]);
    // An instance by id still takes everything below it.
    s().showAll("a");
    s().setHidden("a", [1], true);
    expect(sorted(s().assets.a?.hidden)).toEqual([1, 2, 3]);
  });

  it("highlights a category or an object, and the same click again clears it", () => {
    s().toggleFocus("a", { kind: "category", id: "trees" });
    expect(sorted(s().assets.a?.highlighted)).toEqual([2, 3, 4]);
    s().toggleFocus("a", { kind: "object", id: 4 });
    expect(sorted(s().assets.a?.highlighted)).toEqual([4]);
    s().toggleFocus("a", { kind: "object", id: 4 });
    expect(s().assets.a?.highlighted.size).toBe(0);
    expect(s().assets.a?.focus).toBeNull();
    s().toggleFocus("a", { kind: "category", id: "trees" });
    s().setCategoryHidden("a", "furniture", true);
    s().reset("a");
    expect(s().assets.a?.hidden.size).toBe(0);
    expect(s().assets.a?.highlighted.size).toBe(0);
  });

  it("searches objects by tag and by category name, and acts on every match", () => {
    const entry = s().assets.a;
    if (!entry) throw new Error("no table");
    // A part's match is its object's: "conifer" finds both tree objects, not the branch.
    expect(matchObjects(entry, "conifer")).toEqual([2, 4]);
    expect(matchObjects(entry, "trees")).toEqual([2, 4]);
    expect(matchObjects(entry, "table")).toEqual([5]);
    expect(matchObjects(entry, "")).toEqual([]);
    s().setQuery("a", "trees");
    s().hideMatches("a");
    expect(sorted(s().assets.a?.hidden)).toEqual([2, 3, 4]);
    s().showOnlyMatches("a");
    expect(sorted(s().assets.a?.hidden)).toEqual([1, 5]);
    s().toggleFocus("a", { kind: "matches" });
    expect(sorted(s().assets.a?.highlighted)).toEqual([2, 3, 4]);
    // The highlight of the matches follows the query.
    s().setQuery("a", "table");
    expect(sorted(s().assets.a?.highlighted)).toEqual([5]);
    s().setQuery("a", "nothing at all");
    expect(s().assets.a?.focus).toBeNull();
  });
});
