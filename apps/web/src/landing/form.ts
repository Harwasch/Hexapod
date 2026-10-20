/**
 * The "Request a site scan" form: what it collects, what makes a request complete, what a
 * submission carries, and how it is sent: to `POST /api/v1/scan-requests`, our own API and
 * database (Harrison's decision, 2026-10-20; apps/api/app/api/v1/scan_requests.py).
 */
import type { Footprint } from "@twin/contracts";

/** Where a request goes, on the API's origin (`VITE_API_BASE_URL`; the page's own in dev). */
export const REQUEST_ENDPOINT = "/api/v1/scan-requests";

export interface SiteRequest {
  name: string;
  organization: string;
  email: string;
  /** As typed, or as the site finder filled it in. */
  acres: string;
  message: string;
  /** Whether the site finder handed back land (parcels or a drawn outline). */
  hasBoundary: boolean;
}

export type RequestField = "name" | "email" | "acres" | "area";
export type RequestErrors = Partial<Record<RequestField, string>>;

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const M2_PER_ACRE = 4046.856;

/**
 * An area in acres as the form shows and fills it: whole acres from 100, one decimal from 1,
 * and two significant figures below that, so a small lot reads 0.034 rather than 0.0 (which
 * the acres check would refuse).
 */
export function formatAcres(m2: number): string {
  const acres = m2 / M2_PER_ACRE;
  if (acres >= 100) return acres.toFixed(0);
  if (acres >= 1) return acres.toFixed(1);
  return Number(acres.toPrecision(2)).toString();
}

/** The acres as a number, ignoring thousands separators; NaN when it is not one. */
export function parseAcres(value: string): number {
  const text = value.trim().replace(/,/g, "");
  return text === "" ? NaN : Number(text);
}

/** What is missing or malformed in a request; an empty object when it is complete. */
export function validateRequest(request: SiteRequest): RequestErrors {
  const errors: RequestErrors = {};
  if (request.name.trim() === "") errors.name = "Enter your name.";
  const email = request.email.trim();
  if (email === "") errors.email = "Enter your email so we can reply.";
  else if (!EMAIL.test(email)) errors.email = "Enter an email address like name@example.com.";
  const typed = request.acres.trim() !== "";
  const acres = parseAcres(request.acres);
  if (typed && !(acres > 0)) errors.acres = "Enter the area in acres, as a number.";
  if (!request.hasBoundary && !typed) {
    errors.area = "Find your land on the map, or enter its acres.";
  }
  return errors;
}

/**
 * What a submission carries, exactly: who is asking, their notes, and the land as GeoJSON with
 * the parcel records it came from. The form's other fields have no place of their own in it,
 * so they open the notes, labelled.
 */
export interface ScanRequestPayload {
  name: string;
  email: string;
  notes: string;
  /** The land found on the map; null when only its acres were given. */
  boundary: Footprint | null;
  /** Candidate ids of the parcels chosen (`ca-parcel/<county fips>/<parcel number>`). */
  parcelIds: string[];
}

export function buildPayload(
  request: SiteRequest,
  land: { boundary: Footprint; parcelIds: string[] } | null,
): ScanRequestPayload {
  const lines = [
    request.organization.trim() && `Organization: ${request.organization.trim()}`,
    request.acres.trim() && `Area (acres): ${request.acres.trim()}`,
    request.message.trim(),
  ].filter(Boolean);
  return {
    name: request.name.trim(),
    email: request.email.trim(),
    notes: lines.join("\n"),
    boundary: land?.boundary ?? null,
    parcelIds: land ? [...land.parcelIds] : [],
  };
}

/** How a send ended: received, or a message for the visitor saying what to do. */
export type SendOutcome = { kind: "received" } | { kind: "error"; message: string };

/**
 * Sends a complete request. `website` is the form's honeypot, sent as typed (empty, unless a
 * bot filled it): the API answers it like any other request and drops it. Every accepted
 * request is answered the same way, so a 202 is all there is to read.
 */
export async function sendRequest(
  payload: ScanRequestPayload,
  website: string,
  { base = "", fetchImpl = (input: string, init: RequestInit) => fetch(input, init) } = {},
): Promise<SendOutcome> {
  let response: Response;
  try {
    response = await fetchImpl(`${base}${REQUEST_ENDPOINT}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ ...payload, website }),
    });
  } catch {
    return {
      kind: "error",
      message: "We could not reach the server. Check your connection, then send it again.",
    };
  }
  if (response.ok) return { kind: "received" };
  if (response.status === 429)
    return {
      kind: "error",
      message: "Several requests came from here just now. Wait a minute, then send it again.",
    };
  if (response.status === 413)
    return {
      kind: "error",
      message:
        "The outline is too detailed to send. Choose fewer parcels, or remove the land and enter its acres.",
    };
  if (response.status === 422) {
    const detail = await problemDetail(response);
    return {
      kind: "error",
      message: `Part of this could not be accepted${detail ? `: ${detail}` : ""}. Check it and send it again.`,
    };
  }
  return {
    kind: "error",
    message: `Something went wrong on our side (HTTP ${response.status}). Try again in a few minutes.`,
  };
}

/** The first thing a Problem response says, if it says anything readable. */
async function problemDetail(response: Response): Promise<string> {
  try {
    const problem = (await response.json()) as {
      detail?: unknown;
      errors?: { msg?: unknown }[];
    };
    const first = problem.errors?.[0]?.msg;
    if (typeof first === "string") return first.replace(/^Value error, /, "");
    return typeof problem.detail === "string" ? problem.detail : "";
  } catch {
    return "";
  }
}
