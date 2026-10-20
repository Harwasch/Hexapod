/**
 * The "Request a site scan" form: what it collects, what makes a request complete, and what a
 * submission will carry.
 *
 * Nothing is sent yet. `REQUEST_ENDPOINT` is where a submission would go (the proposal is a
 * public `POST /api/v1/scan-requests`); it is null until that is decided, and while it is, a
 * complete request shows a "not connected" notice instead of pretending to submit.
 */
import type { Footprint } from "@twin/contracts";

export const REQUEST_ENDPOINT: string | null = null;

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
