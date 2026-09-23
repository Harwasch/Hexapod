import { ArcGisMapService, Ion } from "cesium";

import type { TokenState } from "@/state/viewer";

/**
 * Configures the Cesium ion token used by the browser.
 *
 * With no token configured we keep CesiumJS's built-in evaluation token so the
 * public demo still works; the UI labels this state and the console warning
 * Cesium prints in that case is expected.
 */
export function configureIonToken(token: string | undefined): TokenState {
  // Before the token is swapped: the demotion below looks the credit up by the built-in
  // token, which is what `Ion.defaultAccessToken` still holds at this point.
  demoteDefaultTokenNotices();
  if (token) {
    Ion.defaultAccessToken = token;
    return "custom";
  }
  return Ion.defaultAccessToken ? "default" : "invalid";
}

/** The shape of the two CesiumJS namespaces that carry a default-key notice. */
interface DefaultKeyedService {
  defaultAccessToken: string;
  getDefaultTokenCredit?: (providedKey: string) => { showOnScreen: boolean } | undefined;
}

/**
 * Moves CesiumJS's "you are using the default access token" advice off the map.
 *
 * When an app runs on the bundled evaluation token, CesiumJS pushes a paragraph of setup
 * advice into the *on-screen* credit container — a `Credit` built with `showOnScreen: true`.
 * That container is a small glass chip here, so the paragraph inflates it into a slab of
 * bold text sitting over the globe. The paragraph is a developer setup hint, not
 * attribution, and `SetupNotices` already says the same thing in the app's own voice; this
 * flips the credit to the collapsed side of the display, so it moves into the "Data
 * attribution" dialog and stays reachable there rather than disappearing.
 *
 * Attribution proper is untouched: the Cesium ion logo, Google's Photorealistic 3D Tiles
 * credit and every layer credit are separate `Credit`s and keep whatever visibility their
 * terms require.
 *
 * `getDefaultTokenCredit` is undeclared in `cesium`'s type definitions (and declared with
 * the wrong return type for ArcGIS), so it is reached through a structural type and an
 * optional call. `ion.test.ts` pins the assumption from both ends.
 */
export function demoteDefaultTokenNotices(): void {
  demote(Ion);
  // ArcGIS declares the same helper, with `string` for what is really a `Credit`.
  demote(ArcGisMapService as unknown as DefaultKeyedService);
}

function demote(service: DefaultKeyedService): void {
  // CesiumJS memoises one credit per service, so demoting it once covers every clone the
  // providers make from it afterwards.
  const credit = service.getDefaultTokenCredit?.(service.defaultAccessToken);
  if (credit) credit.showOnScreen = false;
}

/** True when an error from an ion-backed request looks like an auth/entitlement failure. */
export function isIonAuthError(error: unknown): boolean {
  const text = describe(error);
  return /\b(401|403|invalid.*token|not authorized|unauthorized|forbidden)\b/i.test(text);
}

export function isIonNotFound(error: unknown): boolean {
  return /\b404\b/.test(describe(error));
}

function describe(error: unknown): string {
  if (error instanceof Error)
    return `${error.message} ${String((error as { statusCode?: number }).statusCode ?? "")}`;
  return String(error);
}
