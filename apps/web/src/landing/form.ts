/**
 * The "Request a site scan" form: what it collects, and what makes a request complete.
 *
 * Nothing is sent yet. `REQUEST_ENDPOINT` is where a submission would go (an API route, a form
 * service, an email relay); it is null until that is decided, and while it is, a complete request
 * shows a "not connected" notice instead of pretending to submit.
 */
export const REQUEST_ENDPOINT: string | null = null;

export interface SiteRequest {
  name: string;
  organization: string;
  email: string;
  /** As typed, or as the site picker filled it in. */
  acres: string;
  message: string;
  /** Parcels selected plus areas drawn on the site picker. */
  areaCount: number;
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
  if (request.areaCount === 0 && !typed) {
    errors.area = "Pick a parcel or draw an area on the map, or enter the acres.";
  }
  return errors;
}
