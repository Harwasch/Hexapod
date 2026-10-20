import { useState } from "react";

import { GlassButton, GlassField, GlassSelect, GlassSwitch, useFieldId } from "@twin/ui";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { changeOffer } from "@/lib/changeDrape";
import {
  epochOptions,
  formatEpochDate,
  platformLabel,
  siteEpochs,
  type CompareDates as Dates,
} from "@/lib/epochs";
import {
  compareDrawable,
  DRAW_WITH_PLAYCANVAS,
  drawWithPlaycanvas,
  NEEDS_CESIUM_OR_PLAYCANVAS,
} from "@/lib/viewModes";
import { useChangeDrape, type ChangeDrapeRequest } from "@/state/changeDrape";
import { recordAction } from "@/state/history";
import { useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { compareRoomKey, useSplatRoom } from "@/state/splatRoom";
import { useViewRule } from "@/state/viewRule";

/**
 * Compare › Dates: two of the active site's dated captures (epochs), one each side of the
 * divider (SiteManager `setCompareDates`; ADR 0017). Offered once the representation shown has
 * two dates or more; nothing is faked, so one capture means no section.
 *
 * The scene owns the comparison: the switch asks it, and the section shows what it then says
 * it draws (`compareDates`, from the `compare-dates` event), never its own click. CesiumJS and
 * PlayCanvas draw the cut (lib/viewModes.ts `compareDrawable`): under Spark or the WebGPU trial
 * the section is disabled, says so, and offers the one-click switch to PlayCanvas, the default
 * renderer, rather than a switch that silently does nothing.
 *
 * Below it, the change map between the two dates (features/change, ADR 0018): offered once
 * both measured a height surface, under any renderer (it lies on the ground; it is not
 * splats), and beside the swipe or a per-cell view rule, which draw splats. It follows a date
 * chosen while it is on, and ends, saying why, when the new pair cannot have one.
 */
/** Whether the scene draws `dates` for `siteId` now (null: one date). */
function isShown(siteId: string, dates: Dates | null): boolean {
  const pair = useSites.getState().compareDates;
  const own = pair?.siteId === siteId ? pair : null;
  return dates ? own?.left === dates.left && own.right === dates.right : own === null;
}

/** Whether the change map asked for is `request` (null: none) on `siteId`. */
function isDraped(siteId: string, request: ChangeDrapeRequest | null): boolean {
  const asked = useChangeDrape.getState().request;
  const own = asked?.siteId === siteId ? asked : null;
  return request ? own?.from === request.from && own.to === request.to : own === null;
}

function askDrape(request: ChangeDrapeRequest | null): void {
  useChangeDrape.setState({ request, error: null });
}

export function CompareDates({ siteId }: { siteId: string }) {
  const scene = useScene();
  const site = useSite(siteId).data;
  const representation = useSites((s) => s.representation[siteId]);
  const pair = useSites((s) => (s.compareDates?.siteId === siteId ? s.compareDates : null));
  const renderer = useSplatRenderer();
  const rule = useViewRule((s) => s.active?.siteId === siteId);
  const room = useSplatRoom((s) => s.compare);
  const drape = useChangeDrape((s) => (s.request?.siteId === siteId ? s.request : null));
  const refused = useChangeDrape((s) => s.error);
  const [choice, setChoice] = useState<Partial<Dates>>({});
  const leftId = useFieldId("compare-date-left");
  const rightId = useFieldId("compare-date-right");
  if (!site || !representation) return null;
  const epochs = siteEpochs(site);
  const options = epochOptions(epochs, site.assets, representation);
  const [before, latest] = [options.at(-2), options.at(-1)];
  if (!before || !latest) return null;
  const offered = (id: string | undefined) =>
    options.some((option) => option.epoch.id === id) ? id : undefined;
  // What the scene draws, else the choice made here, else the latest against the one before.
  const left = pair?.left ?? offered(choice.left) ?? before.epoch.id;
  const right = pair?.right ?? offered(choice.right) ?? latest.epoch.id;
  const needsRenderer = representation === "gaussian-splat" && !compareDrawable(renderer);
  // Refused by PlayCanvas for want of room: this pair's root tiles alone pass the device's
  // budget (state/splatRoom.ts). Held off, saying so, until another pair or renderer.
  const noRoom =
    !pair && renderer === "playcanvas" && room?.key === compareRoomKey(siteId, left, right)
      ? room
      : null;
  const ask = (dates: Dates | null) => void scene?.sites.setCompareDates(dates);
  const choose = (side: keyof Dates, epochId: string) => {
    const next = { left, right, [side]: epochId };
    setChoice(next);
    if (pair && next.left !== next.right) ask(next);
    if (drape) {
      const follows = changeOffer(epochs, next.left, next.right);
      askDrape(follows.offered ? { siteId, from: follows.from, to: follows.to } : null);
    }
  };
  const offer = changeOffer(epochs, left, right);
  const select = (side: keyof Dates, id: string, other: string) => (
    <GlassSelect
      id={id}
      value={side === "left" ? left : right}
      disabled={needsRenderer}
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
          disabled={needsRenderer || left === right || noRoom !== null}
          onCheckedChange={(on) => {
            const dates = { left, right };
            const [after, before] = on ? [dates, null] : [null, dates];
            let applied = true;
            recordAction(
              `Turn date comparison ${on ? "on" : "off"}`,
              () => {
                applied = true;
                ask(after);
              },
              () => {
                applied = false;
                ask(before);
              },
              {
                // The active site's dates: never undone from another site's view.
                scope: "site",
                // Undone or redone only from where it left the scene: a comparison a date
                // choice (or the renderer) has ended since leaves nothing to take back, and the
                // step is dropped rather than replayed as a no-op.
                alive: () => isShown(siteId, applied ? after : before),
              },
            );
          }}
          aria-labelledby="compare-dates-label"
        />
      </div>
      {needsRenderer ? (
        <div
          className="views__notice"
          style={{ flexWrap: "wrap" }}
          role="note"
          data-testid="compare-dates-needs-renderer"
        >
          <span>{NEEDS_CESIUM_OR_PLAYCANVAS}</span>
          <GlassButton size="sm" variant="ghost" onClick={drawWithPlaycanvas}>
            {DRAW_WITH_PLAYCANVAS}
          </GlassButton>
        </div>
      ) : (
        <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
          Each side draws one date, both with the site&apos;s one placement.
        </p>
      )}
      {noRoom && (
        <p className="views__notice" role="note" data-testid="compare-dates-room">
          Not enough room to draw {noRoom.dates} dates on this device: their coarsest tiles alone
          pass its splat budget
        </p>
      )}
      <div className="setting">
        <span className="setting__label" id="compare-change-label">
          Change map
        </span>
        <GlassSwitch
          checked={drape !== null}
          disabled={!offer.offered}
          onCheckedChange={(on) => {
            const request = offer.offered ? { siteId, from: offer.from, to: offer.to } : null;
            const [after, before] = on ? [request, null] : [null, drape];
            let applied = true;
            recordAction(
              `Turn the change map ${on ? "on" : "off"}`,
              () => {
                applied = true;
                askDrape(after);
              },
              () => {
                applied = false;
                askDrape(before);
              },
              { scope: "site", alive: () => isDraped(siteId, applied ? after : before) },
            );
          }}
          aria-labelledby="compare-change-label"
        />
      </div>
      {!offer.offered ? (
        <p className="views__notice" role="note" data-testid="compare-change-not-offered">
          {offer.why}
        </p>
      ) : drape && refused ? (
        <p className="views__notice" role="note" data-testid="compare-change-refused">
          {refused}
        </p>
      ) : (
        <p className="glass-subtle" style={{ fontSize: "var(--text-xs)", margin: 0 }}>
          Where the ground rose or fell between the two dates, from their height surfaces; on the
          ground under any renderer.
        </p>
      )}
      {rule && !pair && (
        <p className="views__notice" role="note" data-testid="compare-dates-ends-rule">
          Swiping ends the per-cell dates (Splat view › Per cell): the two are one or the other.
        </p>
      )}
    </section>
  );
}
