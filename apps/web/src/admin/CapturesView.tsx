/**
 * Captures: everything in storage, and whether the database and the bucket agree.
 *
 * The table is the easy half. The reconciliation panel underneath it is the half that
 * exists because failures here are silent by construction — a run that uploaded three
 * artifacts and died before committing its step leaves bytes nobody is billed for on
 * purpose, and a deleted object leaves a site pointing at a 404 that only a visitor
 * would ever notice.
 */
import { Fragment, useState } from "react";

import { AlertTriangle, ChevronDown, ChevronRight, Play, ScanSearch } from "lucide-react";

import type { Capture, Job } from "@twin/contracts";

import { EmptyState, GlassBadge, GlassButton, Spinner } from "@twin/ui";

import { formatBytes, formatDate, shortId } from "./format";
import { siteIndex, useCaptures, useJobs, useReconciliation, useSites } from "./queries";

function runsOf(jobs: Job[] | undefined, captureId: string): Job[] {
  return (jobs ?? []).filter((job) => job.captureId === captureId);
}

function CaptureFiles({ capture }: { capture: Capture }) {
  return (
    <table className="admin-subtable">
      <caption className="sr-only">Source files of {capture.name}</caption>
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Bytes</th>
          <th scope="col">Checksum</th>
          <th scope="col">Storage key</th>
          <th scope="col">Upload</th>
        </tr>
      </thead>
      <tbody>
        {capture.files.map((file) => (
          <tr key={file.id}>
            <td>{file.filename}</td>
            <td className="admin-num">{formatBytes(file.bytes)}</td>
            <td className="admin-mono admin-ellipsis">{file.checksum ?? "—"}</td>
            <td className="admin-mono admin-ellipsis">{file.storageKey}</td>
            <td>{file.status}</td>
          </tr>
        ))}
        {capture.files.length === 0 && (
          <tr>
            <td colSpan={5}>No files uploaded yet.</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

function Reconciliation() {
  const [asked, setAsked] = useState(false);
  const query = useReconciliation(asked);

  return (
    <section className="admin-card" aria-labelledby="reconcile-heading">
      <div className="admin-card__head">
        <h2 id="reconcile-heading">Storage against the database</h2>
        <GlassButton
          size="sm"
          leadingIcon={<ScanSearch size={13} aria-hidden="true" />}
          loading={query.isFetching}
          onClick={() => {
            if (asked) void query.refetch();
            else setAsked(true);
          }}
          data-testid="reconcile"
        >
          {asked ? "Check again" : "Reconcile"}
        </GlassButton>
      </div>

      <p className="admin-note">
        Finds files in storage that no record points to, and records whose file is missing. Runs
        only when you ask.
      </p>

      {query.isError && (
        <p className="admin-error" role="alert" data-testid="reconcile-error">
          Could not reconcile: {String(query.error)}
        </p>
      )}

      {query.data && (
        <div data-testid="reconcile-result">
          <p className="admin-summary">
            <strong>{query.data.scanned}</strong> objects walked,{" "}
            <strong>{query.data.matched}</strong> claimed by a row.{" "}
            <strong>{query.data.rowsChecked}</strong> rows checked against storage.
            {query.data.truncated && " The walk hit its ceiling, so the orphan list is a sample."}
          </p>

          <h3 className="admin-subhead">
            Orphans — in the bucket, in no row ({query.data.orphans.length},{" "}
            {formatBytes(query.data.bytesOrphaned)})
          </h3>
          {query.data.orphans.length === 0 ? (
            <p className="admin-note" data-testid="no-orphans">
              Nothing unaccounted for.
            </p>
          ) : (
            <table className="admin-subtable" data-testid="orphans">
              <caption className="sr-only">Objects in storage with no database row</caption>
              <thead>
                <tr>
                  <th scope="col">Key</th>
                  <th scope="col">Bytes</th>
                  <th scope="col">Written</th>
                  <th scope="col">Why</th>
                </tr>
              </thead>
              <tbody>
                {query.data.orphans.map((orphan) => (
                  <tr key={orphan.key}>
                    <td className="admin-mono admin-ellipsis">{orphan.key}</td>
                    <td className="admin-num">{formatBytes(orphan.bytes)}</td>
                    <td className="admin-num">{formatDate(orphan.lastModified)}</td>
                    <td>{orphan.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h3 className="admin-subhead">
            Missing — a row says it is there, and it is not ({query.data.missing.length})
          </h3>
          {query.data.missing.length === 0 ? (
            <p className="admin-note" data-testid="no-missing">
              Every row&apos;s object is where it says it is.
            </p>
          ) : (
            <table className="admin-subtable" data-testid="missing">
              <caption className="sr-only">Database rows whose storage object is missing</caption>
              <thead>
                <tr>
                  <th scope="col">Row</th>
                  <th scope="col">What</th>
                  <th scope="col">Key</th>
                  <th scope="col">Run</th>
                </tr>
              </thead>
              <tbody>
                {query.data.missing.map((row) => (
                  <tr key={`${row.kind}:${row.key}`}>
                    <td>
                      <GlassBadge tone="danger">
                        <AlertTriangle size={11} aria-hidden="true" /> {row.kind}
                      </GlassBadge>
                    </td>
                    <td>{row.label}</td>
                    <td className="admin-mono admin-ellipsis">{row.key}</td>
                    <td className="admin-mono">{row.jobId ? shortId(row.jobId) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </section>
  );
}

export function CapturesView({ onLaunch }: { onLaunch: (captureId: string) => void }) {
  const captures = useCaptures();
  const jobs = useJobs();
  const sites = useSites();
  const [open, setOpen] = useState<string | null>(null);
  const names = siteIndex(sites.data);

  if (captures.isPending) return <Spinner />;
  if (captures.isError) {
    return (
      <p className="admin-error" role="alert">
        The API did not answer. This page shows what is really in storage, so there is nothing to
        show without it.
      </p>
    );
  }

  return (
    <>
      <section className="admin-card" aria-labelledby="captures-heading">
        <div className="admin-card__head">
          <h2 id="captures-heading">Captures</h2>
          <span className="admin-note">{captures.data.length} in the catalog</span>
        </div>

        {captures.data.length === 0 ? (
          <EmptyState
            title="No captures yet"
            body="Drop a file into the globe app's Captures panel, or send one from a phone."
          />
        ) : (
          <table className="admin-table" data-testid="captures-table">
            <caption className="sr-only">Every capture, newest first</caption>
            <thead>
              <tr>
                <th scope="col">
                  <span className="sr-only">Expand</span>
                </th>
                <th scope="col">Capture</th>
                <th scope="col">Kind</th>
                <th scope="col">Sensor</th>
                <th scope="col">Files</th>
                <th scope="col">Bytes</th>
                <th scope="col">Uploaded</th>
                <th scope="col">Status</th>
                <th scope="col">Runs</th>
                <th scope="col">Site</th>
                <th scope="col">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {captures.data.map((capture) => {
                const runs = runsOf(jobs.data, capture.id);
                const bytes = capture.files.reduce((sum, file) => sum + (file.bytes ?? 0), 0);
                const expanded = open === capture.id;
                return (
                  <Fragment key={capture.id}>
                    <tr data-testid="capture-row">
                      <td>
                        <GlassButton
                          size="sm"
                          variant="ghost"
                          iconOnly
                          aria-label={expanded ? `Hide ${capture.name}` : `Show ${capture.name}`}
                          aria-expanded={expanded}
                          onClick={() => {
                            setOpen(expanded ? null : capture.id);
                          }}
                        >
                          {expanded ? (
                            <ChevronDown size={14} aria-hidden="true" />
                          ) : (
                            <ChevronRight size={14} aria-hidden="true" />
                          )}
                        </GlassButton>
                      </td>
                      <td>
                        <span className="admin-strong">{capture.name}</span>
                        <br />
                        <span className="admin-mono admin-dim">{capture.slug}</span>
                      </td>
                      <td>{capture.kind}</td>
                      <td>{capture.sensor ?? capture.device ?? "—"}</td>
                      <td className="admin-num">{capture.files.length}</td>
                      <td className="admin-num">{formatBytes(bytes)}</td>
                      <td className="admin-num">{formatDate(capture.createdAt)}</td>
                      <td>
                        <GlassBadge
                          tone={
                            capture.status === "complete"
                              ? "success"
                              : capture.status === "error"
                                ? "danger"
                                : "neutral"
                          }
                        >
                          {capture.status}
                        </GlassBadge>
                      </td>
                      <td className="admin-num">{runs.length}</td>
                      <td>{capture.siteId ? (names[capture.siteId] ?? "registered") : "—"}</td>
                      <td>
                        <GlassButton
                          size="sm"
                          variant="ghost"
                          leadingIcon={<Play size={12} aria-hidden="true" />}
                          onClick={() => {
                            onLaunch(capture.id);
                          }}
                          data-testid={`run-${capture.slug}`}
                        >
                          Run
                        </GlassButton>
                      </td>
                    </tr>
                    {expanded && (
                      <tr className="admin-table__detail">
                        <td colSpan={11}>
                          <CaptureFiles capture={capture} />
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

      <Reconciliation />
    </>
  );
}
