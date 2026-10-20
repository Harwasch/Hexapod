/**
 * Requests: the site scans visitors asked for on the landing page, newest first. Read-only:
 * replying is by email for now, and nothing here changes a request.
 */
import { EmptyState, GlassBadge, Spinner } from "@twin/ui";

import { ApiError } from "@/api/client";

import { areaLabel, formatDate, mailtoHref } from "./format";
import { useScanRequests } from "./queries";

export function RequestsView() {
  const requests = useScanRequests();

  if (requests.isPending) return <Spinner />;
  if (requests.isError) {
    const needsToken = requests.error instanceof ApiError && requests.error.status === 401;
    return (
      <p className="admin-error" role="alert">
        {needsToken
          ? "Scan requests hold contact details: enter the write token to see them."
          : "The API did not answer, so there are no requests to show."}
      </p>
    );
  }

  const rows = requests.data;
  return (
    <section className="admin-card" aria-labelledby="requests-heading">
      <div className="admin-card__head">
        <h2 id="requests-heading">Scan requests</h2>
        <div className="admin-card__actions">
          <span className="admin-note">{rows.length} requests, newest first</span>
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState
          title="No requests yet"
          body="Requests sent from the landing page's form appear here."
        />
      ) : (
        <table className="admin-table" data-testid="requests-table">
          <caption className="sr-only">Site scans requested from the landing page</caption>
          <thead>
            <tr>
              <th scope="col">Received</th>
              <th scope="col">Name</th>
              <th scope="col">Email</th>
              <th scope="col">Land</th>
              <th scope="col">Parcels</th>
              <th scope="col">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} data-testid="request-row">
                <td className="admin-num">{formatDate(row.createdAt)}</td>
                <td>{row.name}</td>
                <td>
                  {mailtoHref(row.email) ? (
                    <a href={mailtoHref(row.email) ?? undefined}>{row.email}</a>
                  ) : (
                    row.email
                  )}
                </td>
                <td className="admin-num">
                  {row.boundary ? (
                    areaLabel(row.areaM2)
                  ) : row.boundaryUnreadable ? (
                    <GlassBadge tone="danger">outline unreadable</GlassBadge>
                  ) : (
                    <GlassBadge tone="warning">no outline</GlassBadge>
                  )}
                </td>
                <td className="admin-mono">
                  {row.parcelIds.length === 0 ? "—" : row.parcelIds.join(", ")}
                </td>
                <td className="admin-notes">{row.notes || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
