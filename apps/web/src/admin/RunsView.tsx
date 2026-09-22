/**
 * Runs: every job, sortable, and comparable side by side.
 *
 * The comparison is the reason this view is a table and not a list of cards. B3's
 * three-way `none` / `robust` / `imc` and B5's null control are both "run the same
 * capture through different parameters and read the difference", and a difference is only
 * readable when the runs are in columns next to each other.
 */
import { Fragment, useMemo, useState } from "react";

import { Ban, Columns3, FileText, RotateCw } from "lucide-react";

import type { Job } from "@twin/contracts";

import { EmptyState, GlassBadge, GlassButton, Spinner } from "@twin/ui";

import { compareRuns } from "./compare";
import { elapsed, formatCost, formatDate, formatDuration, shortId } from "./format";
import { useCancelRun, useCaptures, useJobs, useRetryRun, useStepLog } from "./queries";

type SortKey = "created" | "capture" | "recipe" | "status" | "duration" | "cost";

const STATUS_TONE: Record<string, "neutral" | "accent" | "danger" | "success" | "warning"> = {
  "not-started": "neutral",
  "in-progress": "accent",
  complete: "success",
  error: "danger",
  cancelled: "warning",
};

function sortJobs(jobs: Job[], key: SortKey, ascending: boolean, names: Record<string, string>) {
  const direction = ascending ? 1 : -1;
  const value = (job: Job): string | number => {
    switch (key) {
      case "capture":
        return names[job.captureId] ?? job.captureId;
      case "recipe":
        return `${job.recipe} ${job.recipeVersion}`;
      case "status":
        return job.status;
      case "duration":
        return job.durationS ?? -1;
      case "cost":
        return Number(job.costUsd ?? -1);
      case "created":
        return job.createdAt;
    }
  };
  return [...jobs].sort((a, b) => {
    const left = value(a);
    const right = value(b);
    if (left === right) return 0;
    return (left < right ? -1 : 1) * direction;
  });
}

function StepLog({ jobId, stepId }: { jobId: string; stepId: string }) {
  const log = useStepLog(jobId, stepId);
  if (log.isPending) return <Spinner />;
  if (log.isError) return <p className="admin-note">No log was written for this stage.</p>;
  return (
    <pre className="admin-log" data-testid="step-log">
      {log.data.text}
    </pre>
  );
}

function Steps({ job }: { job: Job }) {
  const [openLog, setOpenLog] = useState<string | null>(null);
  return (
    <>
      <table className="admin-subtable">
        <caption className="sr-only">Stages of run {shortId(job.id)}</caption>
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Stage</th>
            <th scope="col">Impl</th>
            <th scope="col">Status</th>
            <th scope="col">Wall time</th>
            <th scope="col">Attempt</th>
            <th scope="col">Preempted</th>
            <th scope="col">Artifacts</th>
            <th scope="col">
              <span className="sr-only">Log</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {job.steps.map((step) => (
            <tr key={step.id}>
              <td className="admin-num">{step.ordinal}</td>
              <td>{step.stageId}</td>
              <td className="admin-mono">{step.impl}</td>
              <td>
                <GlassBadge tone={STATUS_TONE[step.status] ?? "neutral"}>{step.status}</GlassBadge>
              </td>
              <td className="admin-num">
                {formatDuration(elapsed(step.startedAt, step.finishedAt))}
              </td>
              <td className="admin-num">{step.attempt}</td>
              <td className="admin-num">{step.preemptedAt ? formatDate(step.preemptedAt) : "—"}</td>
              <td className="admin-num">{step.artifacts.length}</td>
              <td>
                {step.logKey && (
                  <GlassButton
                    size="sm"
                    variant="ghost"
                    leadingIcon={<FileText size={12} aria-hidden="true" />}
                    aria-expanded={openLog === step.id}
                    onClick={() => {
                      setOpenLog(openLog === step.id ? null : step.id);
                    }}
                  >
                    Log
                  </GlassButton>
                )}
              </td>
            </tr>
          ))}
          {job.steps.length === 0 && (
            <tr>
              <td colSpan={9}>Queued. No worker has claimed it yet.</td>
            </tr>
          )}
        </tbody>
      </table>
      {openLog && <StepLog jobId={job.id} stepId={openLog} />}
      {job.error && (
        <p className="admin-error" role="alert">
          {job.error}
        </p>
      )}
    </>
  );
}

function Comparison({ runs, names }: { runs: Job[]; names: Record<string, string> }) {
  const comparison = useMemo(() => compareRuns(runs), [runs]);
  const sections = [
    { title: "The run", rows: comparison.facts },
    { title: "Parameters", rows: comparison.params },
    { title: "Per-stage wall time", rows: comparison.stages },
  ];

  return (
    <section className="admin-card" aria-labelledby="compare-heading" data-testid="comparison">
      <div className="admin-card__head">
        <h2 id="compare-heading">Comparing {runs.length} runs</h2>
        <span className="admin-note">
          {comparison.differences} row{comparison.differences === 1 ? "" : "s"} differ
        </span>
      </div>
      {!comparison.sameCapture && (
        <p className="admin-note" data-testid="cross-capture">
          These runs are over different captures, so what differs between them is not only the
          parameters.
        </p>
      )}
      <table className="admin-table admin-table--compare">
        <caption className="sr-only">Selected runs side by side</caption>
        <thead>
          <tr>
            <th scope="col">Field</th>
            {runs.map((job) => (
              <th scope="col" key={job.id}>
                <span className="admin-mono">{shortId(job.id)}</span>
                <br />
                <span className="admin-dim">{names[job.captureId] ?? "—"}</span>
              </th>
            ))}
          </tr>
        </thead>
        {sections.map((section) => (
          <tbody key={section.title}>
            <tr className="admin-table__group">
              <th scope="colgroup" colSpan={runs.length + 1}>
                {section.title}
              </th>
            </tr>
            {section.rows.map((row) => (
              <tr key={row.key} className={row.differs ? "admin-row--differs" : undefined}>
                <th scope="row">{row.label}</th>
                {row.values.map((value, index) => (
                  <td
                    key={`${row.key}:${runs[index]?.id ?? String(index)}`}
                    className="admin-mono"
                    data-differs={row.differs ? "true" : undefined}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
            {section.rows.length === 0 && (
              <tr>
                <td colSpan={runs.length + 1} className="admin-note">
                  Nothing recorded.
                </td>
              </tr>
            )}
          </tbody>
        ))}
      </table>
    </section>
  );
}

export function RunsView() {
  const jobs = useJobs();
  const captures = useCaptures();
  const cancel = useCancelRun();
  const retry = useRetryRun();
  const [sort, setSort] = useState<{ key: SortKey; ascending: boolean }>({
    key: "created",
    ascending: false,
  });
  const [selected, setSelected] = useState<string[]>([]);
  const [open, setOpen] = useState<string | null>(null);

  const names = useMemo(() => {
    const index: Record<string, string> = {};
    for (const capture of captures.data ?? []) index[capture.id] = capture.name;
    return index;
  }, [captures.data]);

  const rows = useMemo(
    () => sortJobs(jobs.data ?? [], sort.key, sort.ascending, names),
    [jobs.data, sort, names],
  );
  const chosen = rows.filter((job) => selected.includes(job.id));

  if (jobs.isPending) return <Spinner />;
  if (jobs.isError) {
    return (
      <p className="admin-error" role="alert">
        The API did not answer, so there are no runs to show.
      </p>
    );
  }

  const header = (key: SortKey, label: string) => (
    <th
      scope="col"
      aria-sort={sort.key === key ? (sort.ascending ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        className="admin-sort"
        onClick={() => {
          setSort((current) =>
            current.key === key
              ? { key, ascending: !current.ascending }
              : { key, ascending: false },
          );
        }}
      >
        {label}
        {sort.key === key && <span aria-hidden="true">{sort.ascending ? " ▲" : " ▼"}</span>}
      </button>
    </th>
  );

  return (
    <>
      <section className="admin-card" aria-labelledby="runs-heading">
        <div className="admin-card__head">
          <h2 id="runs-heading">Runs</h2>
          <div className="admin-card__actions">
            <span className="admin-note">{rows.length} runs</span>
            <GlassButton
              size="sm"
              variant={chosen.length >= 2 ? "primary" : "ghost"}
              disabled={chosen.length < 2}
              leadingIcon={<Columns3 size={13} aria-hidden="true" />}
              onClick={() => {
                setSelected([]);
              }}
              data-testid="clear-selection"
            >
              {chosen.length >= 2
                ? `Clear ${String(chosen.length)} selected`
                : "Select two runs to compare"}
            </GlassButton>
          </div>
        </div>

        {rows.length === 0 ? (
          <EmptyState title="No runs yet" body="Launch one from a capture." />
        ) : (
          <table className="admin-table" data-testid="runs-table">
            <caption className="sr-only">Every pipeline run</caption>
            <thead>
              <tr>
                <th scope="col">
                  <span className="sr-only">Compare</span>
                </th>
                {header("created", "Queued")}
                {header("capture", "Capture")}
                {header("recipe", "Recipe")}
                {header("status", "Status")}
                <th scope="col">Provider</th>
                <th scope="col">Tier</th>
                {header("duration", "Wall time")}
                {header("cost", "Cost")}
                <th scope="col">Stages</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((job) => {
                const expanded = open === job.id;
                const preemptions = job.steps.filter((step) => step.preemptedAt !== null).length;
                return (
                  <Fragment key={job.id}>
                    <tr data-testid="run-row">
                      <td>
                        <label className="admin-check">
                          <input
                            type="checkbox"
                            checked={selected.includes(job.id)}
                            onChange={(event) => {
                              setSelected((current) =>
                                event.target.checked
                                  ? [...current, job.id]
                                  : current.filter((id) => id !== job.id),
                              );
                            }}
                          />
                          <span className="sr-only">Compare run {shortId(job.id)}</span>
                        </label>
                      </td>
                      <td className="admin-num">{formatDate(job.createdAt)}</td>
                      <td>{names[job.captureId] ?? shortId(job.captureId)}</td>
                      <td>
                        {job.recipe} <span className="admin-dim">v{job.recipeVersion}</span>
                      </td>
                      <td>
                        <GlassBadge tone={STATUS_TONE[job.status] ?? "neutral"}>
                          {job.status}
                        </GlassBadge>
                      </td>
                      <td>{job.provider ?? "—"}</td>
                      <td>{job.tier ?? "—"}</td>
                      <td className="admin-num">{formatDuration(job.durationS)}</td>
                      <td className="admin-num">{formatCost(job.costUsd)}</td>
                      <td className="admin-num">
                        <button
                          type="button"
                          className="admin-sort"
                          aria-expanded={expanded}
                          onClick={() => {
                            setOpen(expanded ? null : job.id);
                          }}
                        >
                          {job.steps.filter((step) => step.status === "complete").length}/
                          {job.steps.length}
                          {preemptions > 0 && ` · ${String(preemptions)} preempted`}
                        </button>
                      </td>
                      <td className="admin-actions">
                        {(job.status === "not-started" || job.status === "in-progress") && (
                          <GlassButton
                            size="sm"
                            variant="ghost"
                            leadingIcon={<Ban size={12} aria-hidden="true" />}
                            onClick={() => {
                              cancel.mutate(job.id);
                            }}
                          >
                            Cancel
                          </GlassButton>
                        )}
                        {(job.status === "error" || job.status === "cancelled") &&
                          job.steps.length > 0 && (
                            <GlassButton
                              size="sm"
                              variant="ghost"
                              leadingIcon={<RotateCw size={12} aria-hidden="true" />}
                              onClick={() => {
                                retry.mutate(job.id);
                              }}
                            >
                              Retry
                            </GlassButton>
                          )}
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="admin-table__detail">
                        <td colSpan={11}>
                          <Steps job={job} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      {chosen.length >= 2 && <Comparison runs={chosen} names={names} />}
    </>
  );
}
