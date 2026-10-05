/**
 * Broad scene categories for a scan's objects (docs/SCENE_OBJECTS.md §3 step 5b): trees,
 * water, buildings, ... -- what a person hides or highlights all at once.
 *
 * The mapping from every open-vocabulary label to one category is data, made by the text
 * encoder that scored the tags (`tools/captures/scene_categories.py`, which writes
 * `tools/captures/data/categories.json`; the same file is bundled here, so the pipeline and the
 * viewer agree). An instance's category comes from its tags, each tag's score a vote for its
 * label's category; one with no tags takes what most of its tagged siblings are (a coarse
 * parent is often a mixed region -- a pumpkin's crop with the hay around it -- and a part is
 * more like the parts beside it), else its nearest tagged ancestor's (a part is what it is
 * part of), else what most of the splats below it are, else what the smallest categorised
 * instance around it is, else "Other". It is computed here, so
 * a scan published before categories existed needs no republish; a file that carries a
 * `category` per instance (newer runs) is taken as it is.
 *
 * Categories act on the hierarchy without spilling over it: an **object** is an instance whose
 * parent is in another category (or none), with every descendant reached through instances of
 * its own category. A category is the union of its objects, so hiding "Ground & soil" hides
 * the ground and the untagged bits of it, not the trees a ground region happens to contain.
 */

import data from "../../../../tools/captures/data/categories.json";

import type { Instance, InstanceTag } from "./instances";

export interface SceneCategory {
  id: string;
  name: string;
  /** sRGB hex. */
  color: string;
}

interface CategoriesFile {
  format: string;
  version: number;
  other: string;
  categories: SceneCategory[];
  labels: Record<string, string>;
}

const FILE = data as CategoriesFile;

/** Every category, in display order ("Other" last). */
export const CATEGORIES: readonly SceneCategory[] = FILE.categories;
/** The category of what nothing describes. */
export const OTHER_CATEGORY: string = FILE.other;

const BY_ID = new Map(CATEGORIES.map((c) => [c.id, c]));
const ORDER = new Map(CATEGORIES.map((c, i) => [c.id, i]));
const LABELS: ReadonlyMap<string, string> = new Map(
  Object.entries(FILE.labels).map(([label, id]) => [label.toLowerCase(), id]),
);

/** The category record for an id; "Other" for one this build does not know. */
export function categoryById(id: string): SceneCategory {
  return BY_ID.get(id) ?? BY_ID.get(OTHER_CATEGORY) ?? { id, name: id, color: "#9e9e9e" };
}

/** The category a vocabulary label belongs to, or undefined for a label outside it. */
export function categoryOfLabel(
  label: string,
  labels: ReadonlyMap<string, string> = LABELS,
): string | undefined {
  return labels.get(label.trim().toLowerCase());
}

/**
 * The category an instance's tags vote for: each tag's score added to its label's category,
 * the most votes winning (ties to the earlier category). Null without a known tag.
 */
export function tagCategory(
  tags: readonly InstanceTag[],
  labels: ReadonlyMap<string, string> = LABELS,
): string | null {
  const votes = new Map<string, number>();
  for (const tag of tags) {
    const id = categoryOfLabel(tag.label, labels);
    if (id === undefined || !(tag.score > 0)) continue;
    votes.set(id, (votes.get(id) ?? 0) + tag.score);
  }
  let best: string | null = null;
  let most = -1;
  for (const [id, vote] of votes) {
    const earlier = best !== null && (ORDER.get(id) ?? 99) < (ORDER.get(best) ?? 99);
    if (vote > most || (vote === most && earlier)) {
      best = id;
      most = vote;
    }
  }
  return best;
}

/** What `assignCategories` reads of an instance (and the file's own `category`, if any). */
export type CategorisedInstance = Pick<Instance, "id" | "parent" | "tags" | "splats"> &
  Partial<Pick<Instance, "bounds" | "centroid" | "name">> & {
    category?: string | null;
  };

/**
 * Per instance id, its category: the file's own when it names a known one; else its tags'
 * vote (`tagCategory`); else its tagged siblings' (by splats); else its nearest tagged
 * ancestor's; else the category that most of the splats below it are in; else that of the
 * smallest categorised instance whose bounds hold its centroid; else "Other". The same rule as `scene_categories.instance_categories`.
 */
export function assignCategories(
  instances: readonly CategorisedInstance[],
  labels: ReadonlyMap<string, string> = LABELS,
): Map<number, string> {
  const byId = new Map(instances.map((i) => [i.id, i]));
  const own = new Map<number, string>();
  for (const instance of instances) {
    const given = instance.category;
    if (typeof given === "string" && BY_ID.has(given)) {
      own.set(instance.id, given);
      continue;
    }
    const voted = tagCategory(instance.tags, labels);
    if (voted !== null) own.set(instance.id, voted);
  }
  const parentOf = (id: number): number | null => {
    const p = byId.get(id)?.parent ?? null;
    return p !== null && p !== id && byId.has(p) ? p : null;
  };
  const out = new Map<number, string>();
  const children = childrenOf(instances);
  for (const instance of instances) {
    const mine = own.get(instance.id);
    if (mine !== undefined) {
      out.set(instance.id, mine);
      continue;
    }
    // Its tagged siblings first: a coarse parent is often a mixed region (a pumpkin's crop
    // with the hay around it), and an untagged part is more like the parts beside it.
    const parent = parentOf(instance.id);
    if (parent !== null) {
      const sibling = heaviest(
        (children.get(parent) ?? []).filter((id) => id !== instance.id),
        (id) => own.get(id),
        (id) => byId.get(id)?.splats ?? 0,
      );
      if (sibling !== null) {
        out.set(instance.id, sibling);
        continue;
      }
    }
    const seen = new Set<number>([instance.id]);
    let up: number | null = instance.id;
    while (up !== null && !own.has(up)) {
      up = parentOf(up);
      if (up !== null && seen.has(up)) up = null;
      if (up !== null) seen.add(up);
    }
    if (up !== null) out.set(instance.id, own.get(up) ?? OTHER_CATEGORY);
  }
  if (out.size === instances.length) return out;
  for (const instance of instances) {
    if (out.has(instance.id)) continue;
    const below = new Set<number>();
    const stack = [...(children.get(instance.id) ?? [])];
    for (let id = stack.pop(); id !== undefined; id = stack.pop()) {
      if (below.has(id)) continue;
      below.add(id);
      stack.push(...(children.get(id) ?? []));
    }
    const best = heaviest(
      below,
      (id) => out.get(id),
      (id) => byId.get(id)?.splats ?? 0,
    );
    if (best !== null) out.set(instance.id, best);
  }
  // Fragments with no tagged instance above, beside or below them (a speck the crops never
  // showed): the category of the smallest categorised instance whose box holds their centre.
  const placed = instances.filter((i) => out.has(i.id) && i.bounds !== undefined);
  for (const instance of instances) {
    if (out.has(instance.id)) continue;
    const centre = instance.centroid;
    let best = OTHER_CATEGORY;
    let smallest = Number.POSITIVE_INFINITY;
    if (centre) {
      for (const other of placed) {
        if (!other.bounds) continue;
        const [x0, y0, z0] = other.bounds.min;
        const [x1, y1, z1] = other.bounds.max;
        const [cx, cy, cz] = centre;
        if (cx < x0 || cx > x1 || cy < y0 || cy > y1 || cz < z0 || cz > z1) continue;
        const volume = Math.max(x1 - x0, 1e-6) * Math.max(y1 - y0, 1e-6) * Math.max(z1 - z0, 1e-6);
        if (volume < smallest) {
          smallest = volume;
          best = out.get(other.id) ?? OTHER_CATEGORY;
        }
      }
    }
    out.set(instance.id, best);
  }
  return out;
}

/** The category with the most splats among `ids` that have one; null when none has. */
function heaviest(
  ids: Iterable<number>,
  categoryOf: (id: number) => string | undefined,
  splatsOf: (id: number) => number,
): string | null {
  const weight = new Map<string, number>();
  for (const id of ids) {
    const category = categoryOf(id);
    if (category === undefined) continue;
    weight.set(category, (weight.get(category) ?? 0) + splatsOf(id) + 1e-9);
  }
  let best: string | null = null;
  let most = 0;
  for (const [category, w] of weight) {
    if (w > most) {
      best = category;
      most = w;
    }
  }
  return best;
}

function childrenOf(instances: readonly Pick<Instance, "id" | "parent">[]): Map<number, number[]> {
  const children = new Map<number, number[]>();
  for (const instance of instances) {
    if (instance.parent === null || instance.parent === instance.id) continue;
    const list = children.get(instance.parent) ?? [];
    list.push(instance.id);
    children.set(instance.parent, list);
  }
  return children;
}

// ---- Objects and groups ------------------------------------------------------------------

/** One object: an instance with its parts of the same category. */
export interface SceneObject {
  id: number;
  category: string;
  /**
   * What it is called: its best tag of its own category, else its category's name; numbered
   * when several objects of the category share it ("bush 2", "Trees 1"). Never an id.
   */
  name: string;
  /** Its instance and every part of it (the ids hide and highlight act on). */
  members: readonly number[];
  /** Gaussians over all members. */
  splats: number;
}

/** One category present in a scan. */
export interface CategoryGroup {
  category: SceneCategory;
  /** Its objects, largest first. */
  objects: readonly SceneObject[];
  /** Every instance id in it. */
  members: readonly number[];
  /** Gaussians over all members: how much of the scan it covers. */
  splats: number;
}

/** A scan's objects by category: what the objects panel lists. */
export interface CategoryIndex {
  /** Categories present, by how much of the scan they cover (largest first). */
  groups: readonly CategoryGroup[];
  /** Per instance id, its category id. */
  categoryOf: ReadonlyMap<number, string>;
  /** Per instance id, the object it is part of (itself, for an object). */
  objectOf: ReadonlyMap<number, number>;
  /** By object id. */
  objects: ReadonlyMap<number, SceneObject>;
}

/** The label an object goes by: its name when the file gives one, else its best tag whose
 * label is in its own category. */
function objectName(
  instance: Pick<Instance, "tags"> & Partial<Pick<Instance, "name">>,
  category: string,
  labels: ReadonlyMap<string, string>,
): string | null {
  if (instance.name !== undefined && instance.name !== "") return instance.name;
  for (const tag of instance.tags) {
    if (categoryOfLabel(tag.label, labels) === category) return tag.label;
  }
  return null;
}

/** Groups a scan's instances into categories and objects (see the module comment). */
export function indexCategories(
  instances: readonly CategorisedInstance[],
  labels: ReadonlyMap<string, string> = LABELS,
): CategoryIndex {
  const categoryOf = assignCategories(instances, labels);
  const byId = new Map(instances.map((i) => [i.id, i]));
  const children = childrenOf(instances);
  const objectOf = new Map<number, number>();
  const roots: CategorisedInstance[] = [];
  for (const instance of instances) {
    const parent = instance.parent !== null ? byId.get(instance.parent) : undefined;
    const mine = categoryOf.get(instance.id);
    if (!parent || parent.id === instance.id || categoryOf.get(parent.id) !== mine) {
      roots.push(instance);
    }
  }
  const draft = new Map<
    string,
    { root: CategorisedInstance; members: number[]; splats: number }[]
  >();
  for (const root of roots) {
    const category = categoryOf.get(root.id) ?? OTHER_CATEGORY;
    const members: number[] = [];
    let splats = 0;
    const stack = [root.id];
    for (let id = stack.pop(); id !== undefined; id = stack.pop()) {
      if (objectOf.has(id)) continue;
      objectOf.set(id, root.id);
      members.push(id);
      splats += byId.get(id)?.splats ?? 0;
      for (const child of children.get(id) ?? []) {
        if (categoryOf.get(child) === category) stack.push(child);
      }
    }
    const list = draft.get(category) ?? [];
    list.push({ root, members, splats });
    draft.set(category, list);
  }
  const objects = new Map<number, SceneObject>();
  const groups: CategoryGroup[] = [];
  for (const [id, list] of draft) {
    const category = categoryById(id);
    list.sort((a, b) => b.splats - a.splats || a.root.id - b.root.id);
    // Names: the best tag of the object's own category, else the category's; a name that
    // several objects share is numbered, largest first ("bush 1", "bush 2", "Trees 1").
    const names = list.map(({ root }) => objectName(root, id, labels) ?? category.name);
    const shared = new Map<string, number>();
    for (const name of names) shared.set(name, (shared.get(name) ?? 0) + 1);
    const seen = new Map<string, number>();
    const made = list.map(({ root, members, splats }, k) => {
      const base = names[k] ?? category.name;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      const numbered = (shared.get(base) ?? 0) > 1 || base === category.name;
      const name = numbered ? `${base} ${String(n)}` : base;
      const object: SceneObject = { id: root.id, category: id, name, members, splats };
      objects.set(root.id, object);
      return object;
    });
    groups.push({
      category,
      objects: made,
      members: made.flatMap((o) => o.members),
      splats: made.reduce((sum, o) => sum + o.splats, 0),
    });
  }
  groups.sort(
    (a, b) =>
      Number(a.category.id === OTHER_CATEGORY) - Number(b.category.id === OTHER_CATEGORY) ||
      b.splats - a.splats ||
      (ORDER.get(a.category.id) ?? 99) - (ORDER.get(b.category.id) ?? 99),
  );
  return { groups, categoryOf, objectOf, objects };
}

/**
 * How well query words name a category, 0..1: each word must begin a word of its name or id
 * ("tree" finds "Trees", "rock" "Rock & stone"). Plural-tolerant, nothing more.
 */
export function matchCategory(terms: readonly string[], category: SceneCategory): number {
  if (terms.length === 0) return 0;
  const words = `${category.name} ${category.id}`
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((w) => w.length > 0);
  for (const term of terms) {
    const stem = term.length > 3 && term.endsWith("s") ? term.slice(0, -1) : term;
    if (stem.length < 2 || !words.some((w) => w.startsWith(stem))) return 0;
  }
  return 1;
}
