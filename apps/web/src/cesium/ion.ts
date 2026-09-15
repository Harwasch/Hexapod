import { Ion } from "cesium";

import type { TokenState } from "@/state/viewer";

/**
 * Configures the Cesium ion token used by the browser.
 *
 * With no token configured we keep CesiumJS's built-in evaluation token so the
 * public demo still works; the UI labels this state and the console warning
 * Cesium prints in that case is expected.
 */
export function configureIonToken(token: string | undefined): TokenState {
  if (token) {
    Ion.defaultAccessToken = token;
    return "custom";
  }
  return Ion.defaultAccessToken ? "default" : "invalid";
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
