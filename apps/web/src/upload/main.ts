/**
 * The page a phone opens after scanning the QR code in the Captures panel.
 *
 * Deliberately not the console: no React, no stores, no CesiumJS. A phone should not
 * download a 3D globe over cellular to pick one file out of its camera roll. What it
 * shares with the console is the part that is actually hard — `putPart`, the direct-to-
 * storage PUT with byte progress — and nothing else.
 *
 * The handoff token arrives in the URL *fragment*, which is why this reads
 * `location.hash` and never a query string: a fragment is not sent to any server, so the
 * token stays out of access logs and out of `Referer` headers on the way here.
 */
import type { CaptureFile, CaptureFileUpload, UploadWindow } from "@twin/contracts";

import { ApiError, isAbort } from "@/api/error";
import { partSizeFor, putPart } from "@/api/putPart";
import { SUPPORTED_TEXT, unsupported } from "@/features/captures/recipes";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/** `version.captureIdHex.expiresAt.ceiling.signature` — signed, not secret. */
function captureIdFromToken(token: string): string | null {
  const parts = token.split(".");
  const hex = parts[1];
  if (parts.length < 5 || !hex || !/^[0-9a-f]{32}$/.test(hex)) return null;
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}

function expiryFromToken(token: string): number | null {
  const at = Number(token.split(".")[2]);
  return Number.isFinite(at) ? at : null;
}

async function call<T>(token: string, path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const problem: unknown = await response.json().catch(() => undefined);
    throw new ApiError(
      response.status,
      problem as never,
      response.status === 401
        ? "This link has expired. Ask the console for a fresh QR code."
        : `The server refused the request (${String(response.status)}).`,
    );
  }
  return (await response.json()) as T;
}

interface Ui {
  /** Carries `data-state`, which the page's CSS reads: idle, invalid, uploading, done, error. */
  main: HTMLElement | null;
  status: HTMLElement;
  bar: HTMLElement;
  detail: HTMLElement;
  input: HTMLInputElement;
  form: HTMLElement;
}

type PageState = "idle" | "invalid" | "uploading" | "done" | "error";

function setState(ui: Ui, state: PageState): void {
  ui.main?.setAttribute("data-state", state);
}

function setProgress(ui: Ui, uploaded: number, total: number): void {
  const pct = total > 0 ? Math.min(100, Math.round((uploaded / total) * 100)) : 0;
  ui.bar.style.width = `${String(pct)}%`;
  ui.bar.parentElement?.setAttribute("aria-valuenow", String(pct));
  ui.detail.textContent = `${String(pct)}% — ${formatBytes(uploaded)} of ${formatBytes(total)}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${String(units[unit])}`;
}

async function upload(token: string, captureId: string, file: File, ui: Ui): Promise<void> {
  const parts: { partNumber: number; etag: string }[] = [];
  let uploaded = 0;

  const registered = await call<CaptureFileUpload>(token, `/api/v1/captures/${captureId}/files`, {
    filename: file.name,
    contentType: file.type || "application/octet-stream",
    bytes: file.size,
  });
  const fileId = registered.file.id;
  let window: UploadWindow = registered.upload;

  for (;;) {
    for (const part of window.parts) {
      const size = partSizeFor(file.size, window.partSize, part.partNumber);
      const start = (part.partNumber - 1) * window.partSize;
      const base = uploaded;
      const etag = await putPart(part.url, file.slice(start, start + size), {
        onProgress: (loaded) => {
          setProgress(ui, base + loaded, file.size);
        },
      });
      parts.push({ partNumber: part.partNumber, etag });
      uploaded = base + size;
      setProgress(ui, uploaded, file.size);
    }
    if (window.nextPartNumber === null || window.nextPartNumber === undefined) break;
    window = await call<UploadWindow>(
      token,
      `/api/v1/captures/${captureId}/files/${fileId}/parts`,
      { fromPartNumber: window.nextPartNumber },
    );
    if (window.parts.length === 0) break;
  }

  await call<CaptureFile>(token, `/api/v1/captures/${captureId}/files/${fileId}/complete`, {
    parts,
  });
}

/** Returns null rather than asserting: a missing element means the page is not this page. */
function collectUi(root: Document): Ui | null {
  const status = root.getElementById("status");
  const bar = root.getElementById("bar");
  const detail = root.getElementById("detail");
  const form = root.getElementById("form");
  const input = root.getElementById("file");
  if (!status || !bar || !detail || !form || !(input instanceof HTMLInputElement)) return null;
  return { main: root.querySelector("main"), status, bar, detail, form, input };
}

export function start(root: Document = document): void {
  const ui = collectUi(root);
  if (!ui) return;

  const token = root.location.hash.replace(/^#/, "").trim();
  const captureId = token ? captureIdFromToken(token) : null;
  if (!captureId) {
    ui.form.hidden = true;
    setState(ui, "invalid");
    ui.status.textContent =
      "This link is not a valid handoff. Scan the QR code in the Captures panel again.";
    return;
  }

  const expiresAt = expiryFromToken(token);
  if (expiresAt !== null && expiresAt * 1000 < Date.now()) {
    ui.form.hidden = true;
    setState(ui, "invalid");
    ui.status.textContent = "This link has expired. Ask the console for a fresh QR code.";
    return;
  }

  ui.input.addEventListener("change", () => {
    const files = Array.from(ui.input.files ?? []);
    if (files.length === 0) return;
    const refused = unsupported(files);
    if (refused.length > 0) {
      ui.status.textContent = `Can't read ${refused.join(", ")}. Choose ${SUPPORTED_TEXT}.`;
      ui.input.value = "";
      return;
    }
    ui.input.disabled = true;
    setState(ui, "uploading");
    uploadAll(token, captureId, files, ui)
      .then(() => {
        setState(ui, "done");
        const what =
          files.length === 1 ? (files[0]?.name ?? "The file") : `${String(files.length)} files`;
        ui.status.textContent = `${what} ${files.length === 1 ? "is on its way" : "are on their way"}. You can close this page.`;
        ui.detail.textContent = "The console has it.";
      })
      .catch((error: unknown) => {
        ui.input.disabled = false;
        if (isAbort(error)) {
          setState(ui, "idle");
          ui.status.textContent = "Upload cancelled.";
          return;
        }
        setState(ui, "error");
        ui.status.textContent =
          error instanceof Error ? error.message : "The upload failed. Try again.";
      });
  });
}

/**
 * One after another, like the console: parallel uploads would share one cellular uplink
 * and make the progress bar a lie. A failure stops the batch rather than skipping ahead,
 * so "the console has it" is never said about a set with a hole in it.
 */
async function uploadAll(token: string, captureId: string, files: File[], ui: Ui): Promise<void> {
  for (const [index, file] of files.entries()) {
    ui.status.textContent =
      files.length === 1
        ? `Uploading ${file.name}`
        : `Uploading ${file.name} (${String(index + 1)} of ${String(files.length)})`;
    setProgress(ui, 0, file.size);
    await upload(token, captureId, file, ui);
  }
}

if (typeof document !== "undefined" && document.getElementById("file")) start();
