import { Credit } from "cesium";

import type { Attribution } from "@twin/contracts";

/** Builds a Cesium credit from catalog attribution so provider credits stay visible. */
export function creditFor(attribution: Attribution[] | undefined, fallback: string): Credit {
  const joined = (attribution ?? [])
    .map((a) => a.text)
    .filter(Boolean)
    .join(" · ");
  const text = joined.length > 0 ? joined : fallback;
  const url = attribution?.find((a) => a.url)?.url;
  const html = url
    ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(text)}</a>`
    : escapeHtml(text);
  return new Credit(html, false);
}

function escapeHtml(value: string): string {
  return value.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c,
  );
}
