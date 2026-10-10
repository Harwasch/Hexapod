const KEY = "living-world-land-invitation";
const VALID = /^[A-Za-z0-9_-]{43}$/;

/** Fragments never go to the server. Preserve across the explicit OIDC redirect. */
export function captureWorkspaceInvite(): string | null {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("land-invite");
  if (token && VALID.test(token)) {
    sessionStorage.setItem(KEY, token);
    window.history.replaceState(
      window.history.state,
      "",
      window.location.pathname + window.location.search,
    );
    return token;
  }
  try {
    const saved = sessionStorage.getItem(KEY);
    return saved && VALID.test(saved) ? saved : null;
  } catch {
    return null;
  }
}

export function clearWorkspaceInvite() {
  sessionStorage.removeItem(KEY);
}

export function workspaceInviteLink(token: string) {
  return `${window.location.origin}/#land-invite=${encodeURIComponent(token)}`;
}
