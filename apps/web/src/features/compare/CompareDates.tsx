import { useState } from "react";

import { GlassButton, GlassField, GlassSelect, GlassSwitch, useFieldId } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import {
  epochOptions,
  formatEpochDate,
  platformLabel,
  siteEpochs,
  type CompareDates as Dates,
} from "@/lib/epochs";
import { DRAW_WITH_CESIUM, drawWithCesium, NEEDS_CESIUM_NOTICE } from "@/lib/viewModes";
import { recordAction } from "@/state/history";
import { useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";

/**
 * Compare › Dates: two of the active site's dated captures (epochs), one each side of the
 * divider (SiteManager `setCompareDates`; ADR 0017). Offered once the representation shown has
 * two dates or more; nothing is faked, so one capture means no section.
 *
 * The scene owns the comparison: the switch asks it, and the section shows what it then says
 * it draws (`compareDates`, from the `compare-dates` event), never its own click. The split is
 * CesiumJS's: under another splat renderer the section is disabled, says so, and offers the
 * one-click switch (lib/viewModes.ts), rather than a switch that silently does nothing.
 */
export function CompareDates({ siteId }: { siteId: string }) {
  const scene = useScene();
  const site = useSite(siteId).data;
  const representation = useSites((s) => s.representation[siteId]);
  const pair = useSites((s) => (s.compareDates?.siteId === siteId ? s.compareDates : null));
  const renderer = useSplatRenderer();
  const [choice, setChoice] = useState<Partial<Dates>>({});
  const leftId = useFieldId("compare-date-left");
  const rightId = useFieldId("compare-date-right");
  if (!site || !representation) return null;
  const options = epochOptions(siteEpochs(site), site.assets, representation);
  const [before, latest] = [options.at(-2), options.at(-1)];
  if (!before || !latest) return null;
  const offered = (id: string | undefined) =>
    options.some((option) => option.epoch.id === id) ? id : undefined;
  // What the scene draws, else the choice made here, else the latest against the one before.
  const left = pair?.left ?? offered(choice.left) ?? before.epoch.id;
  const right = pair?.right ?? offered(choice.right) ?? latest.epoch.id;
  const needsCesium = representation === "gaussian-splat" && renderer !== "cesium";
  const ask = (dates: Dates | null) => void scene?.sites.setCompareDates(dates);
  const choose = (side: keyof Dates, epochId: string) => {
    const next = { left, right, [side]: epochId };
    setChoice(next);
    if (pair && next.left !== next.right) ask(next);
  };
  const select = (side: keyof Dates, id: string, other: string) => (
    <GlassSelect
      id={id}
      value={side === "left" ? left : right}
      disabled={needsCesium}
      onChange={(e) => choose(side, e.target.value)}
    >
      {options.map(({ epoch }) => (
        <option key={epoch.id} value={epoch.id} disabled={epoch.id === other}>
          {formatEpochDate(epoch.observedAt)} · {platformLabel(epoch.platform)}
        </option>
      ))}
    </GlassSelect>
  );
  return (
    <section className="glass-stack" aria-label="Compare dates" data-testid="compare-dates">
      <p className="glass-eyebrow">Dates · {site.name}</p>
      <GlassField label="Left of the divider" htmlFor={leftId}>
        {select("left", leftId, right)}
      </GlassField>
      <GlassField label="Right of the divider" htmlFor={rightId}>
        {select("right", rightId, left)}
      </GlassField>
      <div className="setting">
        <span className="setting__label" id="compare-dates-label">
          Swipe between dates
        </span>
        <GlassSwitch
          checked={pair !== null}
          disabled={needsCesium || left === right}
          onCheckedChange={(on) => {
            const dates = { left, right };
            recordAction(
              `Turn date comparison ${on ? "on" : "off"}`,
              () => ask(on ? dates : null),
              () => ask(on ? null : dates),
              // The active site's dates: never undone from another site's view.
              { scope: "site" },
            );
          }}
          aria-labelledby="compare-dates-label"
        />
      </div>
      {needsCesium ? (
        <div
          className="views__notice"
          style={{ flexWrap: "wrap" }}
          role="note"
          data-testid="compare-dates-needs-cesium"
        >
          <span>{NEEDS_CESIUM_NOTICE}</span>
          <GlassButton size="sm" variant="ghost" onClick={drawWithCesium}>
            {DRAW_WITH_CESIUM}
          </GlassButton>
        </div>
      ) : (
        <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
          Each side draws one date, both with the site&apos;s one placement.
        </p>
      )}
    </section>
  );
}
