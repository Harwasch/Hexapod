"""Broad scene categories for the open vocabulary (docs/SCENE_OBJECTS.md §3 step 5b).

Tags come from a 1,300-label open vocabulary (`data/open_vocabulary.txt`), which is too fine
to act on: a camp scan lists 650 "forest floor" pieces, 238 "bush"es and 51 "conifer"s. The
viewer groups them under a compact, fixed set of broad scene categories (trees, water,
buildings, ...) so a person can hide or highlight all of one kind at once.

The mapping is data, made by the same text encoder that scored the tags: each label's prompt
(`segment_models.tag_prompt`) is embedded, each category's few phrasings are embedded and
averaged (`segment_models.text_bank`), and the label goes to the nearest category by cosine.
The outliers that inspection found are corrected in `OVERRIDES` (every one is a label the
encoder put somewhere a person would not look for it). Nothing here is per scan.

    python scene_categories.py            # rewrites data/categories.json (SigLIP 2, CPU)
    python scene_categories.py --check    # exits 1 when the file is stale or incomplete

`instance_categories` assigns a category per instance from its tags, as the viewer does
(`apps/web/src/lib/categories.ts`), for `segment_scene` to write beside the tags.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
CATEGORIES_FILE = DATA / "categories.json"
FORMAT = "hexapod.categories"
VERSION = 1
#: Where a label lands when nothing fits, and an instance with no tag anywhere near it.
OTHER = "other"


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    #: sRGB hex, distinct between neighbours in the list.
    color: str
    #: Phrasings averaged into one text embedding. Empty for `OTHER` (overrides only).
    prompts: tuple[str, ...]


#: Nature first, then what is built, what lives, and loose things; `OTHER` last.
CATEGORIES: tuple[Category, ...] = (
    Category("trees", "Trees", "#2f8f46", (
        "a photo of a tree", "a photo of trees in a forest", "a photo of a tree trunk",
        "a photo of a conifer", "a photo of a palm tree",
    )),
    Category("shrubs", "Shrubs & bushes", "#7cb342", (
        "a photo of a bush", "a photo of a shrub", "a photo of a hedge",
        "a photo of dense undergrowth",
    )),
    Category("grass", "Grass & ground cover", "#a5c94a", (
        "a photo of grass", "a photo of a lawn", "a photo of moss", "a photo of ferns",
        "a photo of fallen leaves on the ground",
    )),
    Category("flowers", "Flowers & plants", "#d46fb1", (
        "a photo of flowers", "a photo of a potted plant", "a photo of a houseplant",
        "a photo of a flower bed",
    )),
    Category("produce", "Fruit, vegetables & crops", "#f08a24", (
        "a photo of a fruit", "a photo of a vegetable", "a photo of a pumpkin",
        "a photo of a crop growing in a field", "a photo of harvested produce",
    )),
    Category("wood", "Logs & dead wood", "#8d6e4c", (
        "a photo of a log", "a photo of a tree stump", "a photo of a fallen tree",
        "a photo of firewood", "a photo of dead branches",
    )),
    Category("ground", "Ground & soil", "#a1887f", (
        "a photo of dirt", "a photo of the ground", "a photo of mud", "a photo of sand",
        "a photo of the forest floor", "a photo of gravel",
    )),
    Category("rock", "Rock & stone", "#90a4ae", (
        "a photo of a rock", "a photo of stones", "a photo of a boulder",
        "a photo of a cliff", "a photo of a mountain",
    )),
    Category("water", "Water", "#2f86d6", (
        "a photo of water", "a photo of a river", "a photo of a lake", "a photo of the sea",
        "a photo of a stream", "a photo of a waterfall",
    )),
    Category("snow", "Snow & ice", "#c9e3f2", (
        "a photo of snow", "a photo of ice", "a photo of frost",
    )),
    Category("sky", "Sky & weather", "#8fb8e8", (
        "a photo of the sky", "a photo of clouds", "a photo of fog", "a photo of smoke",
    )),
    Category("buildings", "Buildings", "#c0623b", (
        "a photo of a building", "a photo of a house", "a photo of a cabin",
        "a photo of a roof", "a photo of a shed", "a photo of a door or a window",
    )),
    Category("walls", "Walls & fences", "#b39b72", (
        "a photo of a wall", "a photo of a fence", "a photo of a stone wall",
        "a photo of a gate", "a photo of a railing",
    )),
    Category("paths", "Paths & roads", "#6d6f73", (
        "a photo of a road", "a photo of a path", "a photo of pavement",
        "a photo of a sidewalk", "a photo of a trail", "a photo of stairs",
    )),
    Category("fixtures", "Poles, signs & fixtures", "#e0b030", (
        "a photo of a pole", "a photo of a street sign", "a photo of a street lamp",
        "a photo of a fire hydrant", "a photo of a manhole cover", "a photo of a traffic cone",
    )),
    Category("vehicles", "Vehicles", "#d64545", (
        "a photo of a car", "a photo of a truck", "a photo of a bicycle",
        "a photo of a boat", "a photo of an airplane", "a photo of a vehicle",
    )),
    Category("people", "People", "#e57373", (
        "a photo of a person", "a photo of people", "a photo of a man or a woman",
    )),
    Category("animals", "Animals", "#ba68c8", (
        "a photo of an animal", "a photo of a dog", "a photo of a bird",
        "a photo of a wild animal", "a photo of an insect",
    )),
    Category("furniture", "Furniture", "#9c6b3f", (
        "a photo of furniture", "a photo of a bench", "a photo of a table",
        "a photo of a chair", "a photo of a bed", "a photo of a picnic table",
    )),
    Category("containers", "Containers & bins", "#5c8a8a", (
        "a photo of a container", "a photo of a barrel", "a photo of a trash can",
        "a photo of a box", "a photo of a bucket", "a photo of a water tank",
    )),
    Category("equipment", "Tools & equipment", "#7e7f9a", (
        "a photo of a tool", "a photo of equipment", "a photo of a machine",
        "a photo of a ladder", "a photo of a hose",
    )),
    Category("electronics", "Electronics & appliances", "#4f6fb3", (
        "a photo of an electronic device", "a photo of a household appliance",
        "a photo of a computer", "a photo of a lamp",
    )),
    Category("household", "Household items", "#b0803c", (
        "a photo of a household object", "a photo of kitchenware", "a photo of a cup",
        "a photo of a bottle", "a photo of a book", "a photo of a towel",
    )),
    Category("clothing", "Clothing & bags", "#8e5aa8", (
        "a photo of clothing", "a photo of a hat", "a photo of shoes", "a photo of a bag",
        "a photo of jewelry",
    )),
    Category("food", "Food & drink", "#e2725b", (
        "a photo of food", "a photo of a meal", "a photo of a drink", "a photo of a dessert",
    )),
    Category("toys", "Toys & sports", "#4db6ac", (
        "a photo of a toy", "a photo of sports equipment", "a photo of a ball",
        "a photo of a musical instrument",
    )),
    Category(OTHER, "Other", "#9e9e9e", ()),
)  # fmt: skip

#: Labels the encoder misplaced, reviewed by hand (`python scene_categories.py --review`
#: lists each label with its nearest categories). A label not here keeps its nearest.
OVERRIDES: dict[str, str] = {
    # Ground the encoder read as something else.
    "forest floor": "ground",
    "mound (baseball)": "ground",
    "playing field": "grass",
    # Smoke over a scan is weather, not a feather.
    "plume": "sky",
    "campfire": "fixtures",
    "fire pit": "fixtures",
    "snowman": "snow",
    # Pulled toward a category by one word of the label ("water ski", "wooden leg", ...).
    "aquarium": "containers",
    "eel": "animals",
    "manatee": "animals",
    "raft": "vehicles",
    "houseboat": "vehicles",
    "water scooter": "vehicles",
    "water faucet": "household",
    "water ski": "toys",
    "wet suit": "clothing",
    "skullcap": "clothing",
    "wooden leg": "furniture",
    "trunk": "containers",
    "clippers (for plants)": "equipment",
    "corkboard": "household",
    "crumb": "food",
    "heart": OTHER,
    "seashell": "animals",
    "gemstone": "clothing",
    "birdcage": "containers",
    "dollhouse": "toys",
    "map": "household",
    "nest": "animals",
    "solar array": "fixtures",
    "tarp": "equipment",
    "bulletin board": "household",
    "headboard": "furniture",
    "playpen": "furniture",
    "bridge": "paths",
    "doormat": "household",
    "road map": "household",
    "runner (carpet)": "household",
    "bagpipe": "toys",
    "basketball backboard": "toys",
    "bullhorn": "electronics",
    "walking cane": "equipment",
    "crucifix": "household",
    "fire hose": "equipment",
    "home plate (baseball)": "toys",
    "ski pole": "toys",
    "scarecrow": OTHER,
    "peeler (tool for fruit and vegetables)": "household",
    "bow-tie": "clothing",
    "tux": "clothing",
    "flap": OTHER,
    "gargoyle": "buildings",
    "kennel": "buildings",
    "mascot": "people",
    "nosebag (for animals)": "equipment",
    "noseband (for animals)": "equipment",
    "saddle (on an animal)": "equipment",
    "blinder (for horses)": "equipment",
    "bait": "equipment",
    "teddy bear": "toys",
    "car battery": "equipment",
    "Ferris wheel": "toys",
    "machine gun": "equipment",
    "rifle": "equipment",
    "motor": "equipment",
    "lawn mower": "equipment",
    "ski": "toys",
    "surfboard": "toys",
    "parasail (sports)": "toys",
    "chessboard": "toys",
    "ironing board": "household",
    "tablecloth": "household",
    "sawhorse": "equipment",
    "bass horn": "toys",
    "cowbell": "toys",
    "crock pot": "electronics",
    "ice maker": "electronics",
    "shredder (for paper)": "electronics",
    "cube": OTHER,
    "cylinder": OTHER,
    "dustpan": "household",
    "pencil box": "household",
    "pencil sharpener": "household",
    "silo": "buildings",
    "bathtub": "household",
    "sink": "household",
    "toilet": "household",
    "urinal": "household",
    "washbasin": "household",
    "newsstand": "buildings",
    "flag": "fixtures",
    "shield": "equipment",
    "stirrup": "equipment",
    "boxing glove": "toys",
    "chandelier": "electronics",
    "cupboard": "furniture",
    "bookcase": "furniture",
    "shelf": "furniture",
    "drawer": "furniture",
    "deadbolt": "equipment",
    "inkpad": "household",
    "pad": "household",
    "phonebook": "household",
    "thumbtack": "household",
    "stylus": "household",
    "beanbag": "furniture",
    "coat hanger": "household",
    "saddle blanket": "equipment",
    "headstall (for horses)": "equipment",
    "wardrobe": "furniture",
    "canteen": "containers",
    "dish": "household",
    "plate": "household",
    "platter": "household",
    "salad plate": "household",
    "candy cane": "food",
    "checkerboard": "toys",
    "underdrawers": "clothing",
    "flip-flop (sandal)": "clothing",
    "cornet": "toys",
    "crayon": "household",
    "faucet": "household",
    "fork": "household",
    "tinsel": "household",
    "egg": "food",
    "bead": "clothing",
    "bubble gum": "food",
    "jelly bean": "food",
    "statue (sculpture)": "household",
    # Food the encoder filed with what it is made of or kept in.
    "bean curd": "food",
    "crisp (potato chip)": "food",
    "egg yolk": "food",
    "string cheese": "food",
    "pickle": "food",
    "cayenne (spice)": "food",
    "crouton": "food",
    "escargot": "food",
    "fish (food)": "food",
    "ginger": "food",
    "honey": "food",
    "lamb-chop": "food",
    "octopus (food)": "food",
    "lime": "produce",
    "ham": "food",
    "baguet": "food",
    "chocolate bar": "food",
    "cracker": "food",
    "lollipop": "food",
    "mint candy": "food",
    "popsicle": "food",
    "jam": "food",
    "olive oil": "food",
    "condiment": "food",
    "vinegar": "food",
    "cocoa (beverage)": "food",
    "alcohol": "food",
    "liquor": "food",
    "vodka": "food",
    "pop (soda)": "food",
}


def category_ids() -> list[str]:
    return [c.id for c in CATEGORIES]


def assign_labels(label_emb: np.ndarray, category_emb: np.ndarray, ids: Sequence[str]) -> list[str]:
    """Per label row, the id of the nearest category row by cosine (rows L2-normalised)."""
    sim = np.atleast_2d(label_emb) @ np.atleast_2d(category_emb).T
    return [ids[j] for j in np.argmax(sim, axis=1)]


def build(embedder: object, labels: Sequence[str]) -> tuple[dict, np.ndarray]:
    """The categories document for `labels`, and the label-by-category similarity matrix."""
    import segment_models as sm

    prompted = [c for c in CATEGORIES if c.prompts]
    bank = sm.text_bank(embedder, [c.prompts for c in prompted])
    label_emb = sm.vocabulary_bank(embedder, list(labels))
    nearest = assign_labels(label_emb, bank, [c.id for c in prompted])
    mapping = {label: OVERRIDES.get(label, near) for label, near in zip(labels, nearest)}
    return document(mapping, getattr(embedder, "name", "unknown")), label_emb @ bank.T


def document(mapping: Mapping[str, str], model: str) -> dict:
    """`categories.json`: the categories in display order, then every label's category."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": model,
        "vocabulary": "open_vocabulary.txt",
        "other": OTHER,
        "categories": [{"id": c.id, "name": c.name, "color": c.color} for c in CATEGORIES],
        "labels": dict(mapping),
    }


def load(path: Path = CATEGORIES_FILE) -> dict[str, str]:
    """Label to category id, from the committed file."""
    return dict(json.loads(path.read_text(encoding="utf-8"))["labels"])


# ------------------------------------------------------------------ instances to categories


def tag_category(
    tags: Sequence[Mapping[str, object]], labels: Mapping[str, str]
) -> tuple[str, float] | None:
    """The category an instance's tags vote for: each tag's score added to its label's
    category; the best and its share of the tags' total. None without a known tag."""
    votes: dict[str, float] = {}
    for tag in tags:
        category = labels.get(str(tag.get("label", "")).strip().lower())
        score = tag.get("score", 0.0)
        if category is None or not isinstance(score, int | float) or score <= 0:
            continue
        votes[category] = votes.get(category, 0.0) + float(score)
    if not votes:
        return None
    total = sum(votes.values())
    best = max(votes, key=lambda c: (votes[c], -category_ids().index(c)))
    return best, votes[best] / total


def _heaviest(
    ids: Iterable[int], category_of: Mapping[int, str], splats_of: Mapping[int, float]
) -> str | None:
    """The category with the most splats among `ids` that have one; None when none has."""
    weight: dict[str, float] = {}
    for k in ids:
        if k in category_of:
            c = category_of[k]
            weight[c] = weight.get(c, 0.0) + float(splats_of.get(k, 0)) + 1e-9
    best: str | None = None
    most = 0.0
    for c, w in weight.items():
        if w > most:
            best, most = c, w
    return best


def instance_categories(
    instances: Sequence[Mapping[str, object]], labels: Mapping[str, str]
) -> dict[int, str]:
    """Per instance id, its category: from its own tags; else from its tagged siblings (most
    splats; a coarse parent is often a mixed region, such as a pumpkin's crop with the hay
    around it, and an untagged part is more like the parts beside it); else its nearest tagged
    ancestor's (a part is what it is part of); else the category most of its descendants'
    splats are in; else `OTHER`. Instances are the `instances.json` records (`id`, `parent`,
    `tags`, `splats`). The viewer's `assignCategories` is the same rule."""
    by_id = {int(i["id"]): i for i in instances}
    splats = {k: float(i.get("splats") or 0) for k, i in by_id.items()}  # type: ignore[arg-type]
    own: dict[int, str] = {}
    for i in instances:
        voted = tag_category(i.get("tags") or [], labels)  # type: ignore[arg-type]
        if voted is not None:
            own[int(i["id"])] = voted[0]

    def parent_of(k: int) -> int | None:
        p = by_id[k].get("parent")
        return int(p) if isinstance(p, int) and p in by_id and p != k else None

    children: dict[int, list[int]] = {}
    for k in by_id:
        p = parent_of(k)
        if p is not None:
            children.setdefault(p, []).append(k)
    out: dict[int, str] = {}
    for k in by_id:
        if k in own:
            out[k] = own[k]
            continue
        p = parent_of(k)
        if p is not None:
            sibling = _heaviest((c for c in children[p] if c != k), own, splats)
            if sibling is not None:
                out[k] = sibling
                continue
        seen = {k}
        up: int | None = k
        while up is not None and up not in own:
            up = parent_of(up)
            if up in seen:
                up = None
            elif up is not None:
                seen.add(up)
        if up is not None:
            out[k] = own[up]
    for k in by_id:
        if k in out:
            continue
        below: set[int] = set()
        stack = list(children.get(k, []))
        while stack:
            c = stack.pop()
            if c not in below:
                below.add(c)
                stack.extend(children.get(c, []))
        out[k] = _heaviest(below, out, splats) or OTHER
    return out


# ----------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    import segment_models as sm

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=CATEGORIES_FILE)
    parser.add_argument("--review", action="store_true", help="print each label's top 3")
    parser.add_argument("--check", action="store_true", help="the committed file is complete")
    args = parser.parse_args(argv)
    labels = sm.default_vocabulary()
    if args.check:
        committed = load(args.out)
        missing = [label for label in labels if label not in committed]
        unknown = {c for c in committed.values()} - set(category_ids())
        if missing or unknown:
            print(f"categories.json: {len(missing)} labels missing, unknown {sorted(unknown)}")
            return 1
        return 0
    doc, sim = build(sm.SiglipEmbedder(), labels)
    if args.review:
        prompted = [c.id for c in CATEGORIES if c.prompts]
        for k, label in enumerate(labels):
            top = np.argsort(-sim[k])[:3]
            near = ", ".join(f"{prompted[j]} {sim[k, j]:.3f}" for j in top)
            print(f"{label:32s} -> {doc['labels'][label]:12s} | {near}")
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for c in doc["labels"].values():
        counts[c] = counts.get(c, 0) + 1
    print(json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
