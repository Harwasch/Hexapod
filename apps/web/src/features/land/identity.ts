import { UserManager, WebStorageStateStore, type User } from "oidc-client-ts";
import { useUi } from "@/state/ui";
import { env } from "@/app/env";
import { useLandIdentity } from "@/state/landIdentity";

let manager: UserManager | undefined;
let initialization: Promise<void> | undefined;

function identityManager() {
  if (!env.landOidcAuthority || !env.landOidcClientId) {
    throw new Error("Workspace sign-in needs the configured identity provider and client ID.");
  }
  if (!manager) {
    manager = new UserManager({
      authority: env.landOidcAuthority,
      client_id: env.landOidcClientId,
      redirect_uri: `${window.location.origin}/auth/land/callback`,
      post_logout_redirect_uri: window.location.origin,
      response_type: "code",
      scope: "openid profile",
      automaticSilentRenew: true,
      userStore: new WebStorageStateStore({ store: window.sessionStorage }),
      ...(env.landOidcAudience ? { extraQueryParams: { audience: env.landOidcAudience } } : {}),
    });
    manager.events.addUserLoaded(applyUser);
    manager.events.addUserUnloaded(() => applyUser(null));
    manager.events.addAccessTokenExpired(() => applyUser(null));
    manager.events.addSilentRenewError(() => {
      applyUser(null);
      useLandIdentity.getState().setError("Your workspace session expired. Sign in to continue.");
    });
  }
  return manager;
}

function applyUser(user: User | null) {
  useLandIdentity
    .getState()
    .setSession(
      user && !user.expired ? user.access_token : null,
      user && !user.expired ? `${user.profile.iss}/${user.profile.sub}` : null,
    );
}

export function initializeLandIdentity(): Promise<void> {
  initialization ??= (async () => {
    try {
      const auth = identityManager();
      if (window.location.pathname === "/auth/land/callback") {
        applyUser(await auth.signinRedirectCallback());
        window.history.replaceState({}, "", "/");
        useUi.getState().setPanel("land");
      } else {
        applyUser(await auth.getUser());
      }
    } catch {
      useLandIdentity
        .getState()
        .setError(
          "Workspace sign-in could not be completed. Check the identity configuration or try signing in again.",
        );
    }
  })();
  return initialization;
}

export async function signInToLand() {
  await identityManager().signinRedirect();
}
export async function signOutOfLand() {
  await identityManager().removeUser();
}
