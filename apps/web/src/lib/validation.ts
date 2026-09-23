/** Client-side validation for the Add Data workflow. The API validates again. */

import { validateFootprint, type Footprint } from "@twin/geo";

export type Validation<T> = { ok: true; value: T } | { ok: false; error: string };

export function validateAssetId(value: string): Validation<number> {
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed))
    return { ok: false, error: "Enter a numeric Cesium ion asset ID (e.g. 4547222)." };
  const parsed = Number.parseInt(trimmed, 10);
  if (parsed <= 0) return { ok: false, error: "Asset IDs are positive integers." };
  return { ok: true, value: parsed };
}

export function validateHttpUrl(
  value: string,
  options: { endsWith?: string[]; placeholders?: string[] } = {},
): Validation<string> {
  const trimmed = value.trim();
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return { ok: false, error: "Enter a full URL starting with https://." };
  }
  if (url.protocol !== "https:" && url.protocol !== "http:")
    return { ok: false, error: "Only http(s) URLs are supported." };
  if (url.username || url.password) return { ok: false, error: "URLs must not embed credentials." };
  if (options.placeholders) {
    for (const placeholder of options.placeholders) {
      if (!trimmed.includes(placeholder))
        return { ok: false, error: `The template must contain ${placeholder}.` };
    }
  }
  if (
    options.endsWith &&
    !options.endsWith.some((suffix) => url.pathname.toLowerCase().endsWith(suffix))
  ) {
    return { ok: false, error: `Expected a URL ending in ${options.endsWith.join(" or ")}.` };
  }
  return { ok: true, value: trimmed };
}

export function validateName(value: string): Validation<string> {
  const trimmed = value.trim();
  if (trimmed.length < 2) return { ok: false, error: "Give it a name (at least 2 characters)." };
  if (trimmed.length > 200) return { ok: false, error: "Names are limited to 200 characters." };
  return { ok: true, value: trimmed };
}

export function parseFootprintText(text: string): Validation<Footprint> {
  if (!text.trim()) return { ok: false, error: "Paste or upload a GeoJSON Polygon footprint." };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, error: "That is not valid JSON." };
  }
  const result = validateFootprint(parsed);
  return result.ok ? { ok: true, value: result.footprint } : { ok: false, error: result.error };
}

export function validateDate(value: string): Validation<string | null> {
  if (!value.trim()) return { ok: true, value: null };
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return { ok: false, error: "Enter a valid date." };
  return { ok: true, value: date.toISOString() };
}
