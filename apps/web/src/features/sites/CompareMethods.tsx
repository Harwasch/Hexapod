import { clsx } from "clsx";
import { Eye, GitCompareArrows, Loader2 } from "lucide-react";
import { useId, useState } from "react";

import { GlassButton, GlassPopover, GlassSegmentedControl } from "@twin/ui";

import {
  findVariant,
  resolvePick,
  TODAY_PICK,
  lookFor,
  SYSTEM_LABELS,
  VARIANT_SYSTEMS,
  type VariantSystem,
} from "@/lib/variants";
import { useSettings } from "@/state/settings";
import { useVariants, type OfferedVariants, type VariantStatus } from "@/state/variants";

/** The value that stands for Today in a row's control (a variant's name never starts with ":"). */
export const TODAY = TODAY_PICK;

/**
 * Segments side by side while the choices fit one line; beyond that the same radio group as a
 * list, one row a choice (real bake-offs have five methods with labels like "Pinned FreeForm ·
 * stiffness rule").
 */
const SEGMENTS_MAX = 4;
const SEGMENT_LABEL_CHARS = 34;

/** What Today is, per system, said in the row when it is the one drawn. */
function todayAbout(system: VariantSystem, published: boolean): string {
  if (!published) {
    return {
      objects: "Today: this scan publishes no objects.",
      fill: "Today: no inferred fill.",
      skins: "Today: nothing moves.",
    }[system];
  }
  return {
    objects: "Today: the objects this scan publishes now.",
    fill: "Today: the inferred fill this scan publishes now.",
    skins: "Today: the motion skin this scan publishes now.",
  }[system];
}

function StatusNote({ status }: { status: VariantStatus | undefined }) {
  if (status?.state === "loading") {
    return (
      <span className="compare__status" data-state="loading">
        <Loader2
          size={12}
          aria-hidden="true"
          style={{ animation: "glass-spin 0.8s linear infinite" }}
        />
        Loading
      </span>
    );
  }
  if (status?.state === "error") {
    return (
      <span className="compare__status" data-state="error">
        Did not load: {status.message}
      </span>
    );
  }
  return null;
}

/** One system's row: Today and the variants, the pick's `about` beneath. */
function SystemRow({
  assetId,
  system,
  offered,
}: {
  assetId: string;
  system: VariantSystem;
  offered: OfferedVariants;
}) {
  const id = useId();
  const picked = useVariants((s) => s.picks[assetId]?.[system]);
  const status = useVariants((s) => s.status[assetId]?.[system]);
  const pick = useVariants((s) => s.pick);
  const variants = offered[system];
  const variant = findVariant(offered, system, resolvePick(system, picked));
  const value = variant?.name ?? TODAY;
  const about = variant ? variant.about : todayAbout(system, offered.today[system]);
  const options = [
    { value: TODAY, label: "Today" },
    ...variants.map((v) => ({ value: v.name, label: v.label })),
  ];
  const oneLine =
    options.length <= SEGMENTS_MAX &&
    options.reduce((n, o) => n + o.label.length, 0) <= SEGMENT_LABEL_CHARS;
  const look = lookFor(system, variant);
  const name = SYSTEM_LABELS[system];
  const choose = (next: string): void => {
    pick(assetId, system, next === TODAY ? null : next);
    // A fill picked while inferred layers are hidden is asked to be seen.
    if (system === "fill" && next !== TODAY && useSettings.getState().inferredStyle === "hide") {
      useSettings.getState().set({ inferredStyle: "show" });
    }
  };
  return (
    <div
      className="compare__row"
      role="group"
      aria-labelledby={`${id}-label`}
      aria-describedby={`${id}-about ${id}-look`}
      data-testid={`compare-${system}`}
      data-picked={value}
      data-layout={oneLine ? "segments" : "list"}
    >
      <div className="compare__head">
        <span id={`${id}-label`} className="compare__label">
          {name}
        </span>
        <StatusNote status={status} />
      </div>
      {/* One radio group either way: segments while the labels fit a line, else one row each,
          the picked one lit as the segment is. Arrow keys move, Space or Enter picks. */}
      <GlassSegmentedControl
        aria-label={`${name} method`}
        block
        orientation={oneLine ? "horizontal" : "vertical"}
        className={clsx("compare__control", !oneLine && "compare__control--list")}
        value={value}
        onValueChange={choose}
        options={options}
      />
      <p id={`${id}-about`} className="compare__about" aria-live="polite">
        {about || "No description given."}
      </p>
      <p id={`${id}-look`} className="compare__look">
        <Eye size={12} aria-hidden="true" className="compare__look-icon" />
        <span>
          <span className="compare__look-label">What to look for: </span>
          {look}
        </span>
      </p>
    </div>
  );
}

/**
 * The scan's method variants (lib/variants.ts), one row per system that offers any -- Objects,
 * Fill, Motion -- each with Today and the variants by their visible labels. A pick swaps what
 * is drawn in place, the camera where it is (cesium/splatInstances.ts, inferredLayers.ts,
 * splatSkin.ts), and is kept for the session, per scan (`state/variants.ts`).
 */
export function CompareMethodsPanel({ assetId }: { assetId: string }) {
  const offered = useVariants((s) => s.offered[assetId]);
  const headingId = useId();
  if (!offered) return null;
  const systems = VARIANT_SYSTEMS.filter((system) => offered[system].length > 0);
  return (
    <section className="compare" aria-labelledby={headingId} data-testid="compare-methods">
      <h2 id={headingId} className="compare__title">
        Compare methods
      </h2>
      {systems.map((system) => (
        <SystemRow key={system} assetId={assetId} system={system} offered={offered} />
      ))}
    </section>
  );
}

/**
 * "Methods" beside the representation switcher, while the scan shown declares variants (not
 * "Compare": that is the Layers panel's swipe): opens the panel over the HUD
 * (`data-hud-popover`, as the objects panel), closed with Escape or a click away.
 */
export function CompareMethods({ assetId }: { assetId: string }) {
  const offered = useVariants((s) => s.offered[assetId] !== undefined);
  const [open, setOpen] = useState(false);
  if (!offered) return null;
  return (
    <GlassPopover
      open={open}
      onOpenChange={setOpen}
      side="top"
      aria-label="Compare methods on this scan"
      className="compare-popover"
      data-hud-popover=""
      data-testid="compare-popover"
      trigger={
        <GlassButton
          size="sm"
          variant="ghost"
          data-testid="compare-methods-button"
          leadingIcon={<GitCompareArrows size={14} aria-hidden="true" />}
          aria-label="Compare methods: switch objects, fill and motion between methods on this scan"
        >
          Methods
        </GlassButton>
      }
    >
      <CompareMethodsPanel assetId={assetId} />
    </GlassPopover>
  );
}
