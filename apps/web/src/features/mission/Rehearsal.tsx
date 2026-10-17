import {
  ChevronDown,
  Hand,
  LandPlot,
  Pause,
  Play,
  Scissors,
  SkipForward,
  Square,
  Undo2,
} from "lucide-react";
import { useState } from "react";

import { env } from "@/app/env";
import {
  formatSimDuration,
  REHEARSAL_SPEEDS,
  type HandKind,
  type RaisedHand,
  type SummaryLine,
} from "@/missions/rehearsal";
import { HAND_KIND_TITLE, HAND_KINDS } from "@/missions/rehearsalClusters";
import { useUi } from "@/state/ui";

import type { DressRehearsal, WorkableOffer } from "./dressRehearsal";

import "./rehearsal.css";

/** How each kind of summary line reads: hands and gaps warn, what was driven and clear do not. */
const TONE: Record<SummaryLine["tone"], string> = {
  info: "mc-note--agent",
  hand: "mc-note--warn",
  clear: "mc-note--agent",
  gap: "mc-note--warn mc-rehearsal__gap",
};

/**
 * The lanes left out of the preview, and why their hands still count: only the preview lost
 * them, and the approved plan still drives them.
 */
function leftOutText(leftOut: DressRehearsal["leftOut"]): string {
  const one = leftOut.length === 1;
  const hands = leftOut.reduce((sum, lane) => sum + lane.hands, 0);
  const lanes = leftOut.map((lane) => `${lane.label} lane ${lane.number}`).join(", ");
  return `Left out of this draft's preview: ${lanes}. The approved plan still drives ${one ? "it" : "them"}, so ${one ? "its" : "their"} ${hands} hand${hands === 1 ? " still counts" : "s still count"} on Approve; the plan's zones and hours are unchanged.`;
}

function HandItem({
  hand,
  open,
  onOpen,
  onLeaveOut,
}: {
  hand: RaisedHand;
  open: boolean;
  onOpen: () => void;
  onLeaveOut: (() => void) | null;
}) {
  return (
    <li className={`mc-rehearsal__hand ${open ? "is-open" : ""}`} data-testid="rehearsal-hand">
      <button
        type="button"
        className="mc-rehearsal__hand-head"
        onClick={onOpen}
        aria-expanded={open}
      >
        <Hand size={13} aria-hidden="true" />
        <span>{hand.label}</span>
        <span className="mc-muted mc-mono">
          {hand.zoneLabel} lane {hand.number} · {hand.machineId}
        </span>
      </button>
      {open && (
        <div className="mc-rehearsal__hand-body">
          <p>{hand.rule}</p>
          <span className="mc-rehearsal__source">{hand.source}</span>
          {onLeaveOut && (
            <button
              type="button"
              className="mc-btn mc-btn--sm"
              onClick={onLeaveOut}
              data-testid="rehearsal-leave-out"
            >
              Leave this lane out
            </button>
          )}
        </div>
      )}
    </li>
  );
}

/** The raised hands by kind, each in lane order: a group per rule, its count, every lane. */
function handGroups(raised: readonly RaisedHand[]): { kind: HandKind; hands: RaisedHand[] }[] {
  return HAND_KINDS.map((kind) => ({
    kind,
    hands: raised
      .filter((h) => h.kind === kind)
      .sort((a, b) => a.zoneLabel.localeCompare(b.zoneLabel) || a.number - b.number),
  })).filter((group) => group.hands.length > 0);
}

/**
 * "Work only the ground it can": the plan redrafted on the ground its machines can work, with
 * what that cuts. Or, where it cannot be offered, what would make it possible.
 */
function WorkableAction({ offer }: { offer: WorkableOffer }) {
  if (offer.state === "none") return null;
  const needsLand = offer.state === "needs-land" || offer.state === "needs-terrain";
  return (
    <div
      className="mc-rehearsal__workable"
      data-testid="rehearsal-workable"
      data-state={offer.state}
    >
      {(offer.state === "ready" || offer.state === "loading") && (
        <button
          type="button"
          className="mc-btn mc-btn--accent mc-rehearsal__workable-btn"
          onClick={offer.apply}
          disabled={offer.state !== "ready"}
          data-testid="rehearsal-workable-apply"
        >
          <span className="mc-rehearsal__workable-title">
            <Scissors size={13} aria-hidden="true" /> Work only the ground it can
          </span>
          <span className="mc-mono mc-rehearsal__workable-sum">
            {offer.state === "ready" ? offer.text : "Reading the terrain and creeks…"}
          </span>
        </button>
      )}
      {offer.state === "ready" && (
        <p className="mc-muted mc-rehearsal__note">
          Redrafts the plan on what is left of {offer.labels.join(" and ")}, hatches what is left
          out, and rehearses again. Derived from {offer.elevation ?? "the Land's terrain"}
          {offer.creeksRead ? " and USGS NHD creeks" : ""}; the limits are assumed.
          {offer.kept.length > 0 && ` ${keptText(offer.kept)}`}
        </p>
      )}
      {offer.state !== "loading" && offer.message && (
        <p
          className={`mc-rehearsal__note ${offer.state === "error" ? "mc-note mc-note--warn" : "mc-muted"}`}
        >
          {offer.message}
        </p>
      )}
      {offer.state === "error" && (
        <button type="button" className="mc-btn mc-btn--sm" onClick={offer.retry}>
          Try again
        </button>
      )}
      {needsLand && env.landExplorationEnabled && (
        <button
          type="button"
          className="mc-btn mc-btn--sm"
          onClick={() => useUi.getState().setPanel("land")}
          data-testid="rehearsal-workable-open-land"
        >
          <LandPlot size={12} aria-hidden="true" /> Open in Land
        </button>
      )}
    </div>
  );
}

function handCount(hands: number): string {
  return hands === 0 ? "no hands" : `${hands} hand${hands === 1 ? "" : "s"}`;
}

/** "3 hands stay", "1 hand stays". */
function stay(hands: number): string {
  return `${hands} hand${hands === 1 ? " stays" : "s stay"}`;
}

/**
 * Why some zones' hands stay: "Perimeter fence follows a line, so its 3 hands stay." or
 * "Z-08 has every lane left out of the preview, so its 4 hands stay."
 */
function keptText(kept: WorkableOffer["kept"]): string {
  const sentence = (zones: WorkableOffer["kept"], one: string, many: string) => {
    if (zones.length === 0) return null;
    const hands = zones.reduce((sum, zone) => sum + zone.hands, 0);
    const single = zones.length === 1;
    const owner = single ? "its" : "their";
    return `${zones.map((zone) => zone.label).join(" and ")} ${single ? one : many}, so ${owner} ${stay(hands)}.`;
  };
  return [
    sentence(
      kept.filter((zone) => zone.why === "line"),
      "follows a line",
      "follow lines",
    ),
    sentence(
      kept.filter((zone) => zone.why === "left-out"),
      "has every lane left out of the preview",
      "have every lane left out of the preview",
    ),
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * After a redraft onto workable ground: the hands on the workable areas against the hands on
 * the ground they replaced, why some stay at its edges, then the hands elsewhere (a followed
 * line, ground that was not reworked) and the total Approve counts, against the total before.
 */
function afterText(
  after: NonNullable<DressRehearsal["after"]>,
  run: Pick<DressRehearsal, "rehearsal" | "leftOut" | "workable">,
  spacing: string | null,
): string {
  const hands = run.rehearsal?.hands ?? [];
  const onArea = (zoneId: string) => after.zoneIds.includes(zoneId);
  const leftOut = (keep: (zoneId: string) => boolean) =>
    run.leftOut.filter((lane) => keep(lane.zoneId)).reduce((sum, lane) => sum + lane.hands, 0);
  const now = hands.filter((h) => onArea(h.zoneId)).length + leftOut(onArea);
  const total = hands.length + leftOut(() => true);
  const elsewhere = total - now;
  const there = elsewhere > 0 || after.handsBefore !== after.reworkedBefore ? " there" : "";
  const change =
    now < after.reworkedBefore
      ? `${handCount(now)}${there}, down from ${after.reworkedBefore}`
      : `${handCount(now)}${there} (${after.reworkedBefore} before)`;
  // Slope hands stay at the edges of the left-out ground; creek hands, where the cut had no
  // creeks to buffer (NHD not read then) or did not count these ones.
  const areaHands = hands.filter((h) => onArea(h.zoneId));
  const creek = areaHands.filter((h) => h.kind === "creek-buffer").length;
  const slope = areaHands.length - creek;
  const slopeWhy =
    slope > 0
      ? `The drawn lanes${spacing ? ` (${spacing})` : ""} run up to the left-out ground, and slope there is read from heights a cell either side, steep cells included, so ${slope === now ? "hands stay" : `${slope} slope ${slope === 1 ? "hand stays" : "hands stay"}`} at its edges: open them to see where.`
      : null;
  const are = creek === 1 ? "is" : "are";
  const creekWhy =
    creek > 0
      ? after.creeksRead
        ? `${creek} ${are} inside a creek buffer, on creeks the cut did not count: open them to see which.`
        : `${creek} ${are} inside a creek buffer the cut did not apply, because USGS NHD was not read then.`
      : null;
  // Hands on other ground: a kept line says why; the rest was not reworked.
  const kept = run.workable.kept.filter((zone) => !onArea(zone.zoneId));
  const keptHands = kept.reduce((sum, zone) => sum + zone.hands, 0);
  const notReworked = [
    ...new Set(
      hands
        .filter((h) => !onArea(h.zoneId) && !kept.some((zone) => zone.zoneId === h.zoneId))
        .map((h) => h.zoneLabel),
    ),
  ];
  const other = elsewhere - keptHands;
  const rest =
    other > 0
      ? `${notReworked.length ? `${notReworked.join(" and ")}: ` : ""}${stay(other)} on ground that was not reworked.`
      : null;
  const totals = there
    ? `Approve now counts ${handCount(total)}, ${total < after.handsBefore ? "down from" : "against"} ${after.handsBefore} before.`
    : null;
  return [
    `Rehearsed again on the workable ground of ${after.labels.join(" and ")}: ${change}.`,
    slopeWhy,
    creekWhy,
    kept.length ? keptText(kept) : null,
    rest,
    totals,
    "The ground between drawn lanes is not checked.",
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * The rehearsal in the plan card, under the steps: the banner, the simulated clock and its
 * controls, the hands as they go up (grouped by rule, each group open to every lane), the
 * one-click answer to them, and at the end the summary with where each check's numbers came
 * from. Renders nothing until "Rehearse" is pressed.
 */
export function RehearsalPanel({ run }: { run: DressRehearsal }) {
  const { status, rehearsal, frame, ground } = run;
  const [open, setOpen] = useState<ReadonlySet<HandKind>>(new Set());
  // A hand opened from the map (or in the list) keeps its group open.
  const focusedKind = rehearsal?.hands.find((h) => h.id === run.focused)?.kind ?? null;
  if (status === "idle") return null;
  const raised = rehearsal && frame ? rehearsal.hands.filter((h) => frame.t >= h.t) : [];
  const groups = handGroups(raised);
  const expanded = (kind: HandKind) => open.has(kind) || kind === focusedKind;
  const toggle = (kind: HandKind) => {
    const closing = expanded(kind);
    if (closing && kind === focusedKind) run.focus(null);
    setOpen((kinds) => {
      const next = new Set(kinds);
      if (closing) next.delete(kind);
      else next.add(kind);
      return next;
    });
  };
  const spacing = rehearsal?.summary[0]?.text.match(/preview spacing, ([^)]*apart)/)?.[1] ?? null;
  const ephemeral = ground?.creeks?.lines.some((line) => line.kind === "ephemeral") ?? false;
  return (
    <section className="mc-rehearsal" data-testid="plan-rehearsal" aria-label="Simulated rehearsal">
      <div className="mc-rehearsal__banner" role="status">
        <span>SIMULATED REHEARSAL</span>
        {rehearsal && (
          <span className="mc-mono">
            {rehearsal.machines.length} ghost{rehearsal.machines.length === 1 ? "" : "s"}
          </span>
        )}
      </div>
      {status === "loading" && (
        <p className="mc-muted" data-testid="rehearsal-loading">
          Reading the ground under the lanes: elevation and mapped creeks…
        </p>
      )}
      {status === "error" && (
        <div className="mc-note mc-note--warn" role="alert">
          {run.error}
          <div className="mc-actions mc-actions--start">
            <button type="button" className="mc-btn mc-btn--sm" onClick={run.start}>
              Try again
            </button>
          </div>
        </div>
      )}
      {status === "ready" && rehearsal && frame && (
        <>
          <div className="mc-rehearsal__clock mc-mono" data-testid="rehearsal-clock">
            {formatSimDuration(frame.t)} of {formatSimDuration(frame.durationS)} sim · {frame.speed}
            ×
          </div>
          <input
            type="range"
            className="mc-rehearsal__scrub"
            min={0}
            max={Math.max(1, frame.durationS)}
            step={1}
            value={Math.round(frame.t)}
            onChange={(e) => run.seek(Number(e.target.value))}
            aria-label="Rehearsal time"
          />
          <div className="mc-rehearsal__controls">
            {frame.playing ? (
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={run.pause}
                data-testid="rehearsal-pause"
              >
                <Pause size={12} aria-hidden="true" /> Pause
              </button>
            ) : (
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={run.play}
                data-testid="rehearsal-play"
              >
                <Play size={12} aria-hidden="true" /> {frame.ended ? "Play again" : "Play"}
              </button>
            )}
            <span className="mc-tags mc-rehearsal__speeds" role="group" aria-label="Speed">
              {REHEARSAL_SPEEDS.map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`mc-tag ${frame.speed === value ? "is-on" : ""}`}
                  aria-pressed={frame.speed === value}
                  onClick={() => run.setSpeed(value)}
                  data-testid={`rehearsal-speed-${value}`}
                >
                  {value}×
                </button>
              ))}
            </span>
            {!frame.ended && (
              <button
                type="button"
                className="mc-btn mc-btn--sm"
                onClick={run.skipToEnd}
                data-testid="rehearsal-skip"
              >
                <SkipForward size={12} aria-hidden="true" /> Skip to end
              </button>
            )}
            <button
              type="button"
              className="mc-btn mc-btn--sm"
              onClick={run.stop}
              data-testid="rehearsal-stop"
            >
              <Square size={12} aria-hidden="true" /> Stop
            </button>
          </div>
          <p className="mc-muted mc-rehearsal__note">
            Ghosts drive the lanes drawn on the map at their class's assumed speed; the clock is the
            time to drive those lanes, not the plan's machine-hours.
            {!frame.ended && " Approve counts the hands raised so far; skip to the end for all."}
          </p>
          {raised.length > 0 && (
            <div className="mc-rehearsal__raised" data-testid="rehearsal-hands">
              <span className="mc-eyebrow">
                {raised.length} HAND{raised.length === 1 ? "" : "S"} RAISED
              </span>
              <ul className="mc-list mc-rehearsal__groups">
                {groups.map((group) => {
                  const shown = expanded(group.kind);
                  const lanes = group.hands.length;
                  return (
                    <li
                      key={group.kind}
                      className="mc-rehearsal__group"
                      data-testid="rehearsal-hand-group"
                      data-kind={group.kind}
                    >
                      <button
                        type="button"
                        className="mc-rehearsal__group-head"
                        onClick={() => toggle(group.kind)}
                        aria-expanded={shown}
                      >
                        <Hand size={13} aria-hidden="true" />
                        <span>{HAND_KIND_TITLE[group.kind]}</span>
                        <span className="mc-mono">
                          {lanes} lane{lanes === 1 ? "" : "s"}
                        </span>
                        <ChevronDown
                          size={13}
                          aria-hidden="true"
                          className="mc-rehearsal__chevron"
                        />
                      </button>
                      {shown && (
                        <ul className="mc-list mc-rehearsal__hands">
                          {group.hands.map((hand) => (
                            <HandItem
                              key={hand.id}
                              hand={hand}
                              open={hand.id === run.focused}
                              onOpen={() => run.focus(hand.id === run.focused ? null : hand.id)}
                              onLeaveOut={
                                rehearsal.lanes.find(
                                  (l) => l.zoneId === hand.zoneId && l.index === hand.index,
                                )?.removable
                                  ? () => run.leaveOut(hand)
                                  : null
                              }
                            />
                          ))}
                        </ul>
                      )}
                    </li>
                  );
                })}
              </ul>
              <p className="mc-muted mc-rehearsal__note">
                One hand per lane and rule, at its worst point. On the map, hands close together
                share a marker that counts them; click one to open its first hand here.
              </p>
              {!frame.ended && <WorkableAction offer={run.workable} />}
            </div>
          )}
          {run.leftOut.length > 0 && (
            <div className="mc-note mc-note--warn" data-testid="rehearsal-left-out">
              <span>{leftOutText(run.leftOut)}</span>
              <button type="button" className="mc-btn mc-btn--sm" onClick={run.restoreLanes}>
                <Undo2 size={12} aria-hidden="true" /> Put back
              </button>
            </div>
          )}
          {frame.ended && (
            <div className="mc-rehearsal__summary" data-testid="rehearsal-summary">
              <span className="mc-eyebrow">END OF REHEARSAL</span>
              {run.after && (
                <p className="mc-note mc-note--agent" data-testid="rehearsal-after">
                  {afterText(run.after, run, spacing)}
                </p>
              )}
              <ul className="mc-list">
                {rehearsal.summary.map((line) => (
                  <li
                    key={line.text}
                    className={`mc-note ${TONE[line.tone]}`}
                    data-tone={line.tone}
                  >
                    {line.text}
                  </li>
                ))}
              </ul>
              <div className="mc-rehearsal__sources" aria-label="Sources">
                {rehearsal.sources.map((source) => (
                  <span key={source} className="mc-rehearsal__source">
                    {source}
                  </span>
                ))}
              </div>
              <WorkableAction offer={run.workable} />
              {ground?.better && (
                <div className="mc-rehearsal__better">
                  <p className="mc-muted mc-rehearsal__note">{ground.better}</p>
                  {env.landExplorationEnabled && (
                    <button
                      type="button"
                      className="mc-btn mc-btn--sm"
                      onClick={() => useUi.getState().setPanel("land")}
                      data-testid="rehearsal-open-land"
                    >
                      <LandPlot size={12} aria-hidden="true" /> Open in Land
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
          {ephemeral && (
            <label className="mc-rehearsal__toggle">
              <input
                type="checkbox"
                checked={run.includeEphemeral}
                onChange={(e) => run.setIncludeEphemeral(e.target.checked)}
              />
              Count ephemeral drainages as creeks
            </label>
          )}
          <details className="mc-details">
            <summary className="mc-eyebrow">WHAT THIS REHEARSAL DOES NOT CHECK</summary>
            <ul className="mc-list mc-list--plain">
              {rehearsal.skipped.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </details>
        </>
      )}
    </section>
  );
}
