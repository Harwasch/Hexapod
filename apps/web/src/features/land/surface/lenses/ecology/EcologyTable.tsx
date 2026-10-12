import { useState, type ReactNode } from "react";

import type { TableColumn, TableRow } from "./ecologyData";

/** "HUMAN_OBSERVATION" → "Human observation". */
function humanize(value: string): string {
  if (!/^[A-Z]+(_[A-Z]+)*$/.test(value) || value.length < 4) return value;
  const words = value.toLowerCase().replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function cell(column: TableColumn, value: TableRow[string]): ReactNode {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number")
    return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${column.unit ? ` ${column.unit}` : ""}`;
  if (/^https?:\/\//.test(value))
    return (
      <a href={value} target="_blank" rel="noopener noreferrer">
        Open
      </a>
    );
  return humanize(value);
}

/** A source table as the source gave it, a page at a time. */
export function EcologyTable({
  label,
  columns,
  rows,
  pageSize = 20,
  italic,
}: {
  label: string;
  columns: TableColumn[];
  rows: TableRow[];
  pageSize?: number;
  /** A column of scientific names, set in italics. */
  italic?: string;
}) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const at = Math.min(page, pages - 1);
  const shown = rows.slice(at * pageSize, at * pageSize + pageSize);
  return (
    <div className="land-eco-table">
      <div className="land-eco-table__scroll">
        <table aria-label={label}>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key} scope="col">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, index) => (
              <tr key={at * pageSize + index}>
                {columns.map((column) => (
                  <td key={column.key}>
                    {column.key === italic && typeof row[column.key] === "string" ? (
                      <i>{row[column.key]}</i>
                    ) : (
                      cell(column, row[column.key] ?? null)
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="land-row">
          <button
            type="button"
            className="land-link"
            disabled={at === 0}
            onClick={() => setPage(at - 1)}
          >
            Previous rows
          </button>
          <span className="land-muted">
            {at + 1} of {pages}
          </span>
          <button
            type="button"
            className="land-link"
            disabled={at >= pages - 1}
            onClick={() => setPage(at + 1)}
          >
            Next rows
          </button>
        </div>
      )}
    </div>
  );
}
