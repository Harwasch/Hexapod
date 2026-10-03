import { Boxes, ChevronRight, Eye, EyeOff, Search } from "lucide-react";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type RefObject,
} from "react";

import { GlassButton, GlassInput, GlassPopover } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import { instanceSphere } from "@/cesium/splatInstances";
import { categoryById, type SceneCategory, type SceneObject } from "@/lib/categories";
import { sameFocus, type AssetInstances, type Focus, useInstances } from "@/state/instances";
import { selectedId, useSceneSelect } from "@/state/sceneSelect";
import { useSettings } from "@/state/settings";

import { MotionRendererNote } from "./MotionRendererNote";

/** Objects listed under an open category before "Show more". */
export const OBJECT_PAGE = 50;

type Visibility = "shown" | "hidden" | "mixed";

/** Whether all, none or some of `members` are hidden. */
function visibilityOf(hidden: ReadonlySet<number>, members: readonly number[]): Visibility {
  if (hidden.size === 0 || members.length === 0) return "shown";
  let count = 0;
  for (const id of members) if (hidden.has(id)) count += 1;
  return count === 0 ? "shown" : count === members.length ? "hidden" : "mixed";
}

/** One row the panel lists: a category with the objects under it (all, or the matches). */
interface Row {
  category: SceneCategory;
  objects: readonly SceneObject[];
  members: readonly number[];
  /** What clicking the row highlights. */
  focus: Focus;
}

/** The rows: every category present, or, while searching, the matches by category. */
function rowsOf(entry: AssetInstances): Row[] {
  if (entry.query.trim() === "") {
    return entry.index.groups.map((group) => ({
      category: group.category,
      objects: group.objects,
      members: group.members,
      focus: { kind: "category", id: group.category.id },
    }));
  }
  const byCategory = new Map<string, SceneObject[]>();
  for (const id of entry.matches) {
    const object = entry.index.objects.get(id);
    if (!object) continue;
    const list = byCategory.get(object.category) ?? [];
    list.push(object);
    byCategory.set(object.category, list);
  }
  // In the order the scan's categories are listed (largest first).
  return entry.index.groups
    .filter((group) => byCategory.has(group.category.id))
    .map((group) => {
      const objects = byCategory.get(group.category.id) ?? [];
      return {
        category: group.category,
        objects,
        members: objects.flatMap((o) => o.members),
        focus: { kind: "matches", category: group.category.id },
      };
    });
}

/**
 * Arrow keys move between the rows' main buttons; Right opens a category, Left closes it (or,
 * on an object, goes back up to its category); Up from the first row returns to the search box.
 */
function onRowKey(
  event: KeyboardEvent<HTMLElement>,
  list: HTMLElement | null,
  toggle?: (open: boolean) => void,
): void {
  if (!list) return;
  const buttons = [...list.querySelectorAll<HTMLButtonElement>("[data-row]")];
  const at = buttons.indexOf(event.currentTarget as HTMLButtonElement);
  if (event.key === "ArrowDown") {
    event.preventDefault();
    buttons[Math.min(buttons.length - 1, at + 1)]?.focus();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    if (at <= 0) list.closest(".objects-panel")?.querySelector<HTMLInputElement>("input")?.focus();
    else buttons[at - 1]?.focus();
  } else if (event.key === "ArrowRight" && toggle) {
    event.preventDefault();
    toggle(true);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    if (toggle) toggle(false);
    else
      event.currentTarget
        .closest(".objects-row--category")
        ?.querySelector<HTMLButtonElement>("[data-row]")
        ?.focus();
  }
}

function EyeToggle({
  name,
  visibility,
  onToggle,
}: {
  name: string;
  visibility: Visibility;
  onToggle: (hide: boolean) => void;
}) {
  const hidden = visibility === "hidden";
  return (
    <GlassButton
      iconOnly
      size="sm"
      variant="ghost"
      className="objects-row__eye"
      aria-label={`${hidden ? "Show" : "Hide"} ${name}`}
      aria-pressed={visibility === "mixed" ? "mixed" : hidden}
      title={hidden ? "Show" : visibility === "mixed" ? "Partly hidden: hide all" : "Hide"}
      data-visibility={visibility}
      onClick={() => onToggle(!hidden)}
    >
      {visibility === "shown" ? (
        <Eye size={14} aria-hidden="true" />
      ) : (
        <EyeOff size={14} aria-hidden="true" />
      )}
    </GlassButton>
  );
}

function CategoryRow({
  assetId,
  row,
  entry,
  open,
  onOpen,
  listRef,
  searching,
  selected,
}: {
  assetId: string;
  row: Row;
  entry: AssetInstances;
  open: boolean;
  onOpen: (open: boolean) => void;
  listRef: RefObject<HTMLUListElement | null>;
  searching: boolean;
  /** The object selected in the scene (cesium/sceneSelect), if it is one of this row's. */
  selected: number | null;
}) {
  const scene = useScene();
  const toggleFocus = useInstances((s) => s.toggleFocus);
  const setObjectsHidden = useInstances((s) => s.setObjectsHidden);
  const setCategoryHidden = useInstances((s) => s.setCategoryHidden);
  const hideMatchesIn = (hide: boolean): void =>
    setObjectsHidden(
      assetId,
      row.objects.map((o) => o.id),
      hide,
    );
  const [shown, setShown] = useState(OBJECT_PAGE);
  const objectsId = useId();
  const { category, objects } = row;
  const active = sameFocus(entry.focus, row.focus);
  const visibility = visibilityOf(entry.hidden, row.members);
  const count = objects.length;

  // The object selected in the scene: listed (past the page if need be) and in view.
  const selectedAt = selected === null ? -1 : objects.findIndex((o) => o.id === selected);
  const selectedRef = useRef<HTMLLIElement>(null);
  const listed = Math.max(shown, Math.ceil((selectedAt + 1) / OBJECT_PAGE) * OBJECT_PAGE);
  useEffect(() => {
    if (selectedAt >= 0) selectedRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [selectedAt, open]);

  /**
   * Highlights the object, selects it in the scene (the chip offers its actions), and flies to
   * it as the chip's Fly to does (`CameraController.flyToObject`).
   */
  const selectObject = (object: SceneObject): void => {
    const focus: Focus = { kind: "object", id: object.id };
    const clearing = sameFocus(entry.focus, focus) || selected === object.id;
    const picking = useSceneSelect.getState();
    if (clearing) {
      if (selected === object.id) picking.clear();
      if (sameFocus(entry.focus, focus)) toggleFocus(assetId, focus);
      return;
    }
    toggleFocus(assetId, focus);
    picking.select(assetId, [object.id], 1, 0, null);
    const sphere = instanceSphere(assetId, object.id);
    if (sphere && scene) scene.camera.flyToObject(sphere);
  };

  return (
    <li
      className="objects-row objects-row--category"
      data-category={category.id}
      data-hidden={visibility === "hidden" || undefined}
    >
      <div className="objects-row__line">
        <button
          type="button"
          className="objects-row__chevron"
          aria-expanded={open}
          aria-controls={objectsId}
          aria-label={`${open ? "Collapse" : "Expand"} ${category.name}`}
          tabIndex={-1}
          onClick={() => onOpen(!open)}
        >
          <ChevronRight size={14} aria-hidden="true" />
        </button>
        <button
          type="button"
          data-row
          className="objects-row__main"
          aria-pressed={active}
          aria-expanded={open}
          title={active ? "Clear the highlight" : `Highlight ${category.name}`}
          onClick={() => toggleFocus(assetId, row.focus)}
          onKeyDown={(event) => onRowKey(event, listRef.current, onOpen)}
        >
          <span
            className="objects-row__swatch"
            style={{ background: category.color }}
            aria-hidden="true"
          />
          <span className="objects-row__name">{category.name}</span>
          <span className="objects-row__count">
            <span className="sr-only">, </span>
            {count.toLocaleString()}
            <span className="sr-only"> {count === 1 ? "object" : "objects"}</span>
          </span>
        </button>
        <EyeToggle
          name={category.name}
          visibility={visibility}
          onToggle={(hide) =>
            searching ? hideMatchesIn(hide) : setCategoryHidden(assetId, category.id, hide)
          }
        />
      </div>
      {open && (
        <ul id={objectsId} className="objects-panel__objects" aria-label={category.name}>
          {objects.slice(0, listed).map((object) => {
            const objectVisibility = visibilityOf(entry.hidden, object.members);
            const isSelected = object.id === selected;
            const lit = isSelected || sameFocus(entry.focus, { kind: "object", id: object.id });
            return (
              <li
                key={object.id}
                ref={isSelected ? selectedRef : undefined}
                className="objects-row objects-row--object"
                data-hidden={objectVisibility === "hidden" || undefined}
                data-selected={isSelected || undefined}
              >
                <div className="objects-row__line">
                  <button
                    type="button"
                    data-row
                    className="objects-row__main"
                    aria-pressed={lit}
                    aria-current={isSelected || undefined}
                    title={lit ? "Clear the highlight" : `Highlight and go to ${object.name}`}
                    onClick={() => selectObject(object)}
                    onKeyDown={(event) => onRowKey(event, listRef.current)}
                  >
                    <span className="objects-row__name">{object.name}</span>
                  </button>
                  <EyeToggle
                    name={object.name}
                    visibility={objectVisibility}
                    onToggle={(hide) => setObjectsHidden(assetId, [object.id], hide)}
                  />
                </div>
              </li>
            );
          })}
          {objects.length > listed && (
            <li className="objects-row objects-row--more">
              <GlassButton size="sm" variant="ghost" onClick={() => setShown(listed + OBJECT_PAGE)}>
                Show {Math.min(OBJECT_PAGE, objects.length - listed)} more of{" "}
                {objects.length.toLocaleString()}
              </GlassButton>
            </li>
          )}
        </ul>
      )}
    </li>
  );
}

/** What is hidden, in words: "Trees hidden", "Trees and 2 more hidden", "3 objects hidden". */
function hiddenSummary(entry: AssetInstances): string | null {
  if (entry.hidden.size === 0) return null;
  const whole: string[] = [];
  let partly = 0;
  for (const group of entry.index.groups) {
    const v = visibilityOf(entry.hidden, group.members);
    if (v === "hidden") whole.push(group.category.name);
    else if (v === "mixed")
      partly += group.objects.filter(
        (o) => visibilityOf(entry.hidden, o.members) === "hidden",
      ).length;
  }
  const parts: string[] = [];
  if (whole.length === 1) parts.push(whole[0] ?? "");
  else if (whole.length > 1) parts.push(`${whole[0] ?? ""} and ${String(whole.length - 1)} more`);
  if (partly > 0) parts.push(`${String(partly)} object${partly === 1 ? "" : "s"}`);
  return parts.length ? `${parts.join(", ")} hidden` : "Parts hidden";
}

/** What is highlighted, in words, or null. */
function highlightSummary(entry: AssetInstances): string | null {
  const focus = entry.focus;
  if (entry.highlighted.size === 0 || focus === null) return null;
  switch (focus.kind) {
    case "category":
      return `${categoryById(focus.id).name} highlighted`;
    case "object":
      return `${entry.index.objects.get(focus.id)?.name ?? "Object"} highlighted`;
    case "matches":
      return focus.category === undefined
        ? "Matches highlighted"
        : `Matching ${categoryById(focus.category).name.toLowerCase()} highlighted`;
    case "ids":
      return "Highlighted";
  }
}

/** The objects panel for one scan (lib/categories.ts, state/instances.ts). */
export function InstancePanel({ assetId }: { assetId: string }) {
  const entry = useInstances((s) => s.assets[assetId]);
  const setQuery = useInstances((s) => s.setQuery);
  const hideMatches = useInstances((s) => s.hideMatches);
  const showOnlyMatches = useInstances((s) => s.showOnlyMatches);
  const toggleFocus = useInstances((s) => s.toggleFocus);
  const reset = useInstances((s) => s.reset);
  const gap = useInstances((s) => s.gaps[assetId]);
  const setSettings = useSettings((s) => s.set);
  const inputId = useId();
  const listRef = useRef<HTMLUListElement>(null);
  /** Categories opened while browsing, and closed while searching (open by default there). */
  const [opened, setOpened] = useState<ReadonlySet<string>>(new Set());
  const [closed, setClosed] = useState<ReadonlySet<string>>(new Set());
  const rows = useMemo(() => (entry ? rowsOf(entry) : []), [entry]);
  // What is selected in the scene, as the object (and category) the panel lists it under.
  const picked = useSceneSelect((s) => (s.assetId === assetId ? selectedId(s) : null));
  const pickedObject = picked === null ? null : (entry?.index.objectOf.get(picked) ?? null);
  const pickedCategory =
    pickedObject === null ? undefined : entry?.index.objects.get(pickedObject)?.category;
  /** The scene selection a person closed the category of: it stays closed until the next. */
  const [dismissed, setDismissed] = useState<number | null>(null);
  if (!entry) return null;

  const searching = entry.query.trim() !== "";
  const pickOpens = (id: string): boolean => id === pickedCategory && dismissed !== pickedObject;
  const isOpen = (id: string): boolean =>
    pickOpens(id) || (searching ? !closed.has(id) : opened.has(id));
  const setOpen = (id: string, open: boolean): void => {
    if (!open && pickOpens(id)) setDismissed(pickedObject);
    const update = (s: ReadonlySet<string>, add: boolean): ReadonlySet<string> => {
      const next = new Set(s);
      if (add) next.add(id);
      else next.delete(id);
      return next;
    };
    if (searching) setClosed((s) => update(s, !open));
    else setOpened((s) => update(s, open));
  };

  const onInputKey = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Enter" && entry.matches.length > 0) {
      toggleFocus(assetId, { kind: "matches" });
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      listRef.current?.querySelector<HTMLButtonElement>("[data-row]")?.focus();
    }
  };

  const summary = [hiddenSummary(entry), highlightSummary(entry)]
    .filter((s): s is string => s !== null)
    .join(" · ");
  const changed = entry.hidden.size > 0 || entry.highlighted.size > 0;
  const matchCount = entry.matches.length;

  return (
    <div className="objects-panel" data-testid="instance-panel">
      <div className="objects-panel__search">
        <Search size={14} aria-hidden="true" className="objects-panel__search-icon" />
        <GlassInput
          id={inputId}
          type="search"
          autoComplete="off"
          aria-label="Search objects"
          placeholder="Search objects"
          value={entry.query}
          aria-describedby={`${inputId}-hint`}
          onChange={(event) => setQuery(assetId, event.target.value)}
          onKeyDown={onInputKey}
        />
      </div>
      <span id={`${inputId}-hint`} className="sr-only">
        Type what you are looking for, such as tree, water or bench. Enter highlights every match;
        the arrow keys move through the list. A filter such as vegetation &gt; 0.5 also works.
      </span>
      {gap && (
        <div className="objects-panel__gap" role="note" data-testid="instance-renderer-gap">
          <p>Hiding and highlighting need the Cesium renderer here. {gap.reason}</p>
          <GlassButton
            size="sm"
            variant="ghost"
            onClick={() => setSettings({ splatRenderer: "cesium" })}
          >
            Use the Cesium renderer
          </GlassButton>
        </div>
      )}
      <MotionRendererNote assetId={assetId} />
      {searching && (
        <div className="objects-panel__bar" role="group" aria-label="Matches">
          <span className="objects-panel__count" role="status" data-testid="instance-count">
            {matchCount === 0
              ? "No objects match"
              : `${matchCount.toLocaleString()} ${matchCount === 1 ? "match" : "matches"}`}
          </span>
          {matchCount > 0 && (
            <span className="objects-panel__actions">
              <GlassButton size="sm" variant="ghost" onClick={() => hideMatches(assetId)}>
                Hide all
              </GlassButton>
              <GlassButton size="sm" variant="ghost" onClick={() => showOnlyMatches(assetId)}>
                Show only
              </GlassButton>
            </span>
          )}
        </div>
      )}
      <ul
        ref={listRef}
        className="objects-panel__list"
        aria-label={searching ? "Matching objects" : "Object categories"}
      >
        {rows.map((row) => (
          <CategoryRow
            key={`${searching ? "q" : "c"}:${row.category.id}`}
            assetId={assetId}
            row={row}
            entry={entry}
            open={isOpen(row.category.id)}
            onOpen={(open) => setOpen(row.category.id, open)}
            listRef={listRef}
            searching={searching}
            selected={pickedCategory === row.category.id ? pickedObject : null}
          />
        ))}
      </ul>
      {changed && (
        <div className="objects-panel__footer">
          <span className="objects-panel__summary" role="status">
            {summary}
          </span>
          <GlassButton size="sm" variant="ghost" onClick={() => reset(assetId)}>
            Reset
          </GlassButton>
        </div>
      )}
    </div>
  );
}

/** A compact button beside the representation switcher that opens the objects panel. */
export function InstanceSearch({ assetId }: { assetId: string }) {
  const count = useInstances((s) => s.assets[assetId]?.instances.length ?? 0);
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <GlassPopover
      open={open}
      onOpenChange={setOpen}
      side="top"
      aria-label="Objects in this scan"
      className="instance-popover"
      trigger={
        <GlassButton
          size="sm"
          variant="ghost"
          data-testid="instance-search"
          leadingIcon={<Boxes size={14} aria-hidden="true" />}
          aria-label="Objects: hide, highlight and search what is in this scan"
        >
          Objects
        </GlassButton>
      }
    >
      <InstancePanel assetId={assetId} />
    </GlassPopover>
  );
}
