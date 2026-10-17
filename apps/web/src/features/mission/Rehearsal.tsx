import { Hand, LandPlot, Pause, Play, SkipForward, Square, Undo2 } from "lucide-react";

import { env } from "@/app/env";
import {
  formatSimDuration,
  REHEARSAL_SPEEDS,
  type RaisedHand,
  type SummaryLine,
} from "@/missions/rehearsal";
import { useUi } from "@/state/ui";

import type { DressRehearsal } from "./dressRehearsal";

import "./rehearsal.css";

/** Raised hands listed while the ghosts drive; the summary names every lane at the end. */
const LISTED_HANDS = 8;

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

/**
 * The rehearsal in the plan card, under the steps: the banner, the simulated clock and its
 * controls, the hands as they go up, and at the end the summary with where each check's
 * numbers came from. Renders nothing until "Rehearse" is pressed.
 */
export function RehearsalPanel({ run }: { run: DressRehearsal }) {
  const { status, rehearsal, frame, ground } = run;
  if (status === "idle") return null;
  const raised = rehearsal && frame ? rehearsal.hands.filter((h) => frame.t >= h.t) : [];
  const listed = [...raised].reverse().slice(0, LISTED_HANDS);
  const focusedHand = rehearsal?.hands.find((h) => h.id === run.focused) ?? null;
  if (focusedHand && !listed.includes(focusedHand)) listed.unshift(focusedHand);
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
            <div>
              <span className="mc-eyebrow">
                {raised.length} HAND{raised.length === 1 ? "" : "S"} RAISED
              </span>
              <ul className="mc-list mc-rehearsal__hands">
                {listed.map((hand) => (
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
              {raised.length > listed.length && (
                <p className="mc-muted mc-rehearsal__note">
                  {raised.length - listed.length} earlier hand
                  {raised.length - listed.length === 1 ? "" : "s"} are on the map; the summary names
                  every lane.
                </p>
              )}
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
