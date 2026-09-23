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
  status: HTMLElement;
  bar: HTMLElement;
  detail: HTMLElement;
  input: HTMLInputElement;
  form: HTMLElement;
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
  return { status, bar, detail, form, input };
}

export function start(root: Document = document): void {
  const ui = collectUi(root);
  if (!ui) return;

  const token = root.location.hash.replace(/^#/, "").trim();
  const captureId = token ? captureIdFromToken(token) : null;
  if (!captureId) {
    ui.form.hidden = true;
    ui.status.textContent =
      "This link is not a valid handoff. Open the Captures panel on the console and scan the QR code again.";
    return;
  }

  const expiresAt = expiryFromToken(token);
  if (expiresAt !== null && expiresAt * 1000 < Date.now()) {
    ui.form.hidden = true;
    ui.status.textContent = "This link has expired. Ask the console for a fresh QR code.";
    return;
  }

  ui.input.addEventListener("change", () => {
    const file = ui.input.files?.[0];
    if (!file) return;
    ui.input.disabled = true;
    ui.status.textContent = `Uploading ${file.name}`;
    setProgress(ui, 0, file.size);
    upload(token, captureId, file, ui)
      .then(() => {
        ui.status.textContent = `${file.name} is on its way. You can close this page.`;
        ui.detail.textContent = "The console has it.";
      })
      .catch((error: unknown) => {
        ui.input.disabled = false;
        if (isAbort(error)) {
          ui.status.textContent = "Upload cancelled.";
          return;
        }
        ui.status.textContent =
          error instanceof Error ? error.message : "The upload failed. Try again.";
      });
  });
}

if (typeof document !== "undefined" && document.getElementById("file")) start();
