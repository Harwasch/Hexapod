import { useState, type ReactNode } from "react";
import { ArrowUpRight, MessageCircleQuestion } from "lucide-react";

import { GlassButton } from "@twin/ui";

import { ResearchWorkerNotice } from "../../../ResearchWorkerNotice";
import { askAgent } from "../../agentBus";
import { LensEmpty } from "../../LensCard";
import {
  SOURCE_SPECS,
  type OverviewBundle,
  type OverviewBundleState,
  type OverviewProvider,
  type SourceState,
} from "../overviewBundle";
import "./bundle.css";

/**
 * The parts the Water and Soil lenses share: both read the one overview bundle, so they
 * start it with the same button, show the same progress and failure lines, and list their
 * sources the same way.
 */

/** Nothing checked yet: what the check reads, and the one button that runs it. */
export function OverviewEmpty({ state }: { state: OverviewBundleState }) {
  return (
    <>
      <LensEmpty
        action={
          state.canEdit ? (
            <GlassButton variant="primary" loading={state.busy} onClick={state.start}>
              Check water, soil and climate
            </GlassButton>
          ) : undefined
        }
      >
        {state.canEdit
          ? "Look up flood zones, soil, rainfall and elevation for this land from public maps."
          : "Flood zones, soil, rainfall and elevation haven't been checked here yet. An editor can run the check."}
      </LensEmpty>
      {state.error && (
        <p className="land-alert" role="alert">
          The check didn't start. {state.error}
        </p>
      )}
    </>
  );
}

/** Loading the saved result, or a failure to load it. */
export function OverviewLoading({ state }: { state: OverviewBundleState }) {
  if (state.loadError)
    return (
      <p className="land-alert" role="alert">
        Saved results couldn't be loaded.{" "}
        <button type="button" className="land-link" onClick={state.reload}>
          Try again
        </button>
      </p>
    );
  return (
    <p className="land-muted" aria-live="polite">
      Loading saved results…
    </p>
  );
}

/**
 * The one state line above the facts: progress while it runs (with Stop), a note when it was
 * made for an earlier outline, or what failed and how to retry. `unreachable` names the
 * lens's own sources that could not be reached, so a partial run offers to try again.
 */
export function OverviewNotices({
  state,
  unreachable,
}: {
  state: OverviewBundleState;
  unreachable: string | null;
}) {
  let line: ReactNode = null;
  if (state.running)
    line = (
      <>
        <div className="bundle-progress" role="status">
          <span className="bundle-progress__text">
            Checking public maps · {state.progress.done} of {state.progress.total} done
          </span>
          <span className="bundle-progress__bar" aria-hidden="true">
            <i style={{ width: `${(state.progress.done / state.progress.total) * 100}%` }} />
          </span>
          {state.canEdit && (
            <GlassButton size="sm" variant="ghost" disabled={state.busy} onClick={state.stop}>
              {state.latest?.status === "queued" ? "Cancel queued request" : "Stop"}
            </GlassButton>
          )}
        </div>
        <ResearchWorkerNotice run={state.latest ?? undefined} className="land-muted" />
      </>
    );
  else if (state.stale)
    line = (
      <p className="land-note bundle-note">
        <span>Checked for an earlier outline of this land.</span>
        {state.canEdit && (
          <button type="button" className="land-link" disabled={state.busy} onClick={state.start}>
            Run again
          </button>
        )}
      </p>
    );
  else if (state.failed)
    line = (
      <div className="land-alert bundle-note" role="alert">
        <span>
          {state.latest?.status === "cancelled"
            ? "The last check was stopped before it finished."
            : "The last check failed before it finished."}
          {state.latest?.error ? ` ${state.latest.error}` : ""}
        </span>
        {state.canEdit && (
          <button type="button" className="land-link" disabled={state.busy} onClick={state.retry}>
            Try again
          </button>
        )}
      </div>
    );
  else if (unreachable)
    line = (
      <p className="land-note bundle-note">
        <span>{unreachable}</span>
        {state.canEdit && (
          <button type="button" className="land-link" disabled={state.busy} onClick={state.retry}>
            Try again
          </button>
        )}
      </p>
    );
  return (
    <>
      {line}
      {state.error && (
        <p className="land-alert" role="alert">
          {state.error}
        </p>
      )}
    </>
  );
}

/** The ask button in a lens card's top corner: hands the agent a question to send. */
export function AskButton({ question, label }: { question: string; label: string }) {
  return (
    <button
      type="button"
      className="land-icon-button"
      aria-label={label}
      title={label}
      onClick={() => askAgent(question)}
    >
      <MessageCircleQuestion size={16} aria-hidden="true" />
    </button>
  );
}

export interface Column<Row> {
  key: string;
  label: string;
  cell: (row: Row) => ReactNode;
  numeric?: boolean;
}

/** A small table that shows its first rows and folds the rest behind "Show all". */
export function DataTable<Row>({
  label,
  columns,
  rows,
  rowKey,
  pageSize = 8,
}: {
  label: string;
  columns: Column<Row>[];
  rows: readonly Row[];
  rowKey: (row: Row, index: number) => string;
  pageSize?: number;
}) {
  const [all, setAll] = useState(false);
  const shown = all ? rows : rows.slice(0, pageSize);
  return (
    <div className="bundle-table-block">
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- wide tables scroll by keyboard */}
      <div className="bundle-table" role="region" aria-label={label} tabIndex={0}>
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key} scope="col" className={column.numeric ? "is-number" : ""}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, index) => (
              <tr key={rowKey(row, index)}>
                {columns.map((column) => (
                  <td key={column.key} className={column.numeric ? "is-number" : ""}>
                    {column.cell(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > pageSize && (
        <button type="button" className="land-link" onClick={() => setAll((value) => !value)}>
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}

/** What a source's status means, in a few words. */
function statusWords(source: SourceState, running: boolean): string {
  switch (source.status) {
    case "available":
      return "Found";
    case "empty":
      return source.provider === "fema-flood-zones"
        ? "No flood zone mapped"
        : source.provider === "usda-soils"
          ? "No soil mapped at the point"
          : "Nothing found";
    case "uncovered":
      return "Not covered here";
    case "unavailable":
      return "Couldn't be reached";
    default:
      return running ? "Waiting" : "No result";
  }
}

/** Each source behind a lens: what it returned, its licence, and its raw record. */
export function SourceList({
  bundle,
  providers,
  running,
  note,
}: {
  bundle: OverviewBundle | null;
  providers: readonly OverviewProvider[];
  running: boolean;
  note?: string | null;
}) {
  return (
    <section className="bundle-section" aria-label="Sources">
      <h4 className="bundle-heading">Sources</h4>
      {note && <p className="land-muted">{note}</p>}
      <ul className="bundle-sources">
        {providers.map((provider) => {
          const spec = SOURCE_SPECS[provider];
          const source = bundle?.sources[provider];
          const evidence = source?.evidence ?? null;
          return (
            <li key={provider} className="bundle-source">
              <div className="bundle-source__head">
                <strong>{spec.name}</strong>
                <span className="land-muted">{spec.publisher}</span>
                <span className={`bundle-status bundle-status--${source?.status ?? "none"}`}>
                  {source ? statusWords(source, running) : running ? "Waiting" : "No result"}
                </span>
              </div>
              {source?.message && <p className="bundle-source__message">{source.message}</p>}
              <p className="land-muted">
                {evidence?.license ?? spec.license} · {evidence?.attribution ?? spec.attribution}
                {evidence && ` · Retrieved ${new Date(evidence.retrievedAt).toLocaleDateString()}`}
              </p>
              {evidence?.relevanceNote && <p className="land-muted">{evidence.relevanceNote}</p>}
              <div className="land-row">
                <a
                  className="land-link bundle-source__link"
                  href={evidence?.url ?? spec.documentation}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {evidence?.url ? "Open original source" : "About this source"}
                  <ArrowUpRight size={13} aria-hidden="true" />
                </a>
              </div>
              {evidence && (
                <details className="bundle-raw">
                  <summary>Raw record</summary>
                  <pre>
                    {evidence.excerpt.length > 6000
                      ? `${evidence.excerpt.slice(0, 6000)}…`
                      : evidence.excerpt}
                  </pre>
                </details>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/** Suggested follow-up questions, handed to the agent card to send. */
export function AskLinks({ questions }: { questions: readonly string[] }) {
  const unique = [...new Set(questions)].slice(0, 3);
  if (!unique.length) return null;
  return (
    <section className="bundle-section" aria-label="Ask the agent">
      <h4 className="bundle-heading">Ask the agent</h4>
      <ul className="bundle-asks">
        {unique.map((question) => (
          <li key={question}>
            <button type="button" className="land-link" onClick={() => askAgent(question)}>
              {question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
