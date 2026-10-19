import { useEffect, useRef, useState, type ElementType, type ReactNode } from "react";
import { clsx } from "clsx";

import { useLandMap, type MapTone } from "@/state/landMap";

import { anchorOf } from "./geometryAnchor";
import { useFlyTo, useLeader, type MapTarget } from "./mapHooks";

export type { MapTarget } from "./mapHooks";

/**
 * Something in a card that stands for a place: hovering or focusing it outlines the place on
 * the map and draws a line to it; clicking flies there.
 */
export function MapLink({
  target,
  tone = "accent",
  as: Tag = "div",
  className,
  children,
  label,
  fly: flies = true,
}: {
  target: MapTarget | null;
  tone?: MapTone;
  as?: ElementType;
  className?: string;
  children: ReactNode;
  /** Accessible name for the fly-to action. */
  label?: string;
  /**
   * Whether a click flies to the place. Off when the element holds controls of its own (a
   * checkbox): hovering or focusing inside it still points at the map.
   */
  fly?: boolean;
}) {
  const ref = useRef<HTMLElement>(null);
  const [hot, setHot] = useState(false);
  const setFocus = useLandMap((s) => s.setFocus);
  const releaseFocus = useLandMap((s) => s.releaseFocus);
  const fly = useFlyTo();
  const anchor = target ? (target.anchor ?? anchorOf(target.geometry)) : null;
  useLeader(ref, anchor, tone, hot);
  useEffect(() => {
    if (!hot || !target || !anchor) return;
    setFocus({ key: target.key, geometry: target.geometry, anchor, tone });
    return () => releaseFocus(target.key);
    // The anchor follows the target; re-running on its identity would flicker the outline.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hot, target?.key, tone, setFocus, releaseFocus]);
  if (!target)
    return (
      <Tag ref={ref} className={className}>
        {children}
      </Tag>
    );
  if (!flies)
    return (
      <Tag
        ref={ref}
        className={clsx(className, "land-map-link", hot && "is-hot")}
        onMouseEnter={() => setHot(true)}
        onMouseLeave={() => setHot(false)}
        onFocus={() => setHot(true)}
        onBlur={() => setHot(false)}
      >
        {children}
      </Tag>
    );
  return (
    <Tag
      ref={ref}
      className={clsx(className, "land-map-link", hot && "is-hot")}
      tabIndex={0}
      role="button"
      aria-label={label}
      onMouseEnter={() => setHot(true)}
      onMouseLeave={() => setHot(false)}
      onFocus={() => setHot(true)}
      onBlur={() => setHot(false)}
      onClick={() => fly(target.geometry)}
      onKeyDown={(event: React.KeyboardEvent) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        fly(target.geometry);
      }}
    >
      {children}
    </Tag>
  );
}

/** One short fact about the land, tied to where it is true. */
export function Fact({
  label,
  value,
  target,
  tone = "accent",
  note,
}: {
  label: string;
  value: ReactNode;
  target?: MapTarget | null;
  tone?: MapTone;
  note?: ReactNode;
}) {
  return (
    <MapLink
      as="li"
      target={target ?? null}
      tone={tone}
      className={clsx("land-fact", tone === "warn" && "land-fact--warn")}
      label={target ? `${label}: show on map` : undefined}
    >
      <span className="land-fact__label">{label}</span>
      {target && (
        <span className="land-fact__go" aria-hidden="true">
          on map
        </span>
      )}
      <span className="land-fact__value">{value}</span>
      {note && <span className="land-fact__note">{note}</span>}
    </MapLink>
  );
}
