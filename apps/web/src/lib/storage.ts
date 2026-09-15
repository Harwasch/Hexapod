/** localStorage that never throws (private mode, quota, SSR). */

export function readJson(key: string): unknown {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as unknown) : undefined;
  } catch (error) {
    console.warn(`storage: could not read ${key}`, error);
    return undefined;
  }
}

export function writeJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`storage: could not write ${key}`, error);
  }
}
