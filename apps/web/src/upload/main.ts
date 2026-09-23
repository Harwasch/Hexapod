/**
 * The page a phone opens: from a QR code in the Captures panel, or on its own with the
 * phone key (app/services/phone_key.py), which lets the phone start its own captures.
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
import type {
  CaptureDetail,
  CaptureFile,
  CaptureFileUpload,
  Job,
  PhoneCapture,
  UploadWindow,
} from "@twin/contracts";

import { ApiError, isAbort } from "@/api/error";
import { partSizeFor, putPart } from "@/api/putPart";
import { classify, SUPPORTED_TEXT, unsupported } from "@/features/captures/recipes";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/** Where the phone key is remembered. Only on this phone, only in this browser. */
const KEY_STORAGE = "twin.phoneKey";
/** How often the page asks how processing is going, once it has started a run. */
const POLL_MS = 20_000;

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

/**
 * The upload credential, which changes as the upload goes.
 *
 * Every handoff-authorised response carries the next token in `X-Handoff-Token` (see
 * app/services/handoff.py): each one lives ten minutes, so a long video on cellular that
 * kept using the first would be refused at its next window. Holding it here, and taking
 * the newest one from every response, is what lets an upload run for hours.
 */
interface Upload {
  token: string;
  captureId: string;
}

function readStoredKey(): string | null {
  try {
    return window.localStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

function storeKey(key: string | null): void {
  try {
    if (key) window.localStorage.setItem(KEY_STORAGE, key);
    else window.localStorage.removeItem(KEY_STORAGE);
  } catch {
    // A private window: the key works for this visit and is asked for again next time.
  }
}

async function post<T>(
  bearer: string,
  path: string,
  body: unknown,
  { unauthorized, upload }: { unauthorized: string; upload?: Upload },
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${bearer}` },
    body: JSON.stringify(body ?? {}),
  });
  const renewed = response.headers.get("x-handoff-token");
  if (upload && renewed) upload.token = renewed;
  if (!response.ok) {
    const problem: unknown = await response.json().catch(() => undefined);
    const detail =
      typeof problem === "object" && problem !== null && "detail" in problem
        ? String(problem.detail)
        : null;
    throw new ApiError(
      response.status,
      problem as never,
      response.status === 401
        ? unauthorized
        : (detail ?? `The server refused the request (${String(response.status)}).`),
    );
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const LINK_EXPIRED = "This link has expired. Ask the console for a fresh QR code.";
const KEY_WRONG = "That phone key isn't right. Check it and try again.";

interface Ui {
  /** Carries `data-state`, which the page's CSS reads: idle, invalid, uploading, done, error. */
  main: HTMLElement | null;
  status: HTMLElement;
  bar: HTMLElement;
  detail: HTMLElement;
  input: HTMLInputElement;
  form: HTMLElement;
  keyForm: HTMLFormElement | null;
  keyInput: HTMLInputElement | null;
  foot: HTMLElement | null;
  forget: HTMLElement | null;
  doneLink: HTMLAnchorElement | null;
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

async function uploadOne(upload: Upload, file: File, ui: Ui): Promise<void> {
  const parts: { partNumber: number; etag: string }[] = [];
  let uploaded = 0;
  const base = `/api/v1/captures/${upload.captureId}/files`;
  const opts = { unauthorized: LINK_EXPIRED, upload };

  const registered = await post<CaptureFileUpload>(
    upload.token,
    base,
    {
      filename: file.name,
      contentType: file.type || "application/octet-stream",
      bytes: file.size,
    },
    opts,
  );
  const fileId = registered.file.id;
  let window: UploadWindow = registered.upload;

  for (;;) {
    for (const part of window.parts) {
      const size = partSizeFor(file.size, window.partSize, part.partNumber);
      const start = (part.partNumber - 1) * window.partSize;
      const done = uploaded;
      const etag = await putPart(part.url, file.slice(start, start + size), {
        onProgress: (loaded) => {
          setProgress(ui, done + loaded, file.size);
        },
      });
      parts.push({ partNumber: part.partNumber, etag });
      uploaded = done + size;
      setProgress(ui, uploaded, file.size);
    }
    if (window.nextPartNumber === null || window.nextPartNumber === undefined) break;
    window = await post<UploadWindow>(
      upload.token,
      `${base}/${fileId}/parts`,
      { fromPartNumber: window.nextPartNumber },
      opts,
    );
    if (window.parts.length === 0) break;
  }

  await post<CaptureFile>(upload.token, `${base}/${fileId}/complete`, { parts }, opts);
}

/**
 * One after another, like the console: parallel uploads would share one cellular uplink
 * and make the progress bar a lie. A failure stops the batch rather than skipping ahead,
 * so "the console has it" is never said about a set with a hole in it.
 */
async function uploadAll(upload: Upload, files: File[], ui: Ui): Promise<void> {
  for (const [index, file] of files.entries()) {
    ui.status.textContent =
      files.length === 1
        ? `Uploading ${file.name}`
        : `Uploading ${file.name} (${String(index + 1)} of ${String(files.length)})`;
    setProgress(ui, 0, file.size);
    await uploadOne(upload, file, ui);
  }
}

function sent(files: File[]): string {
  const what =
    files.length === 1 ? (files[0]?.name ?? "The file") : `${String(files.length)} files`;
  return `${what} ${files.length === 1 ? "is on its way" : "are on their way"}`;
}

/** The phone's own position, or null if it is refused or slow. Never blocks for long. */
function locate(): Promise<GeolocationPosition | null> {
  if (!("geolocation" in navigator)) return Promise.resolve(null);
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(resolve, () => resolve(null), {
      enableHighAccuracy: true,
      timeout: 10_000,
      maximumAge: 60_000,
    });
  });
}

function describeJob(job: Job): string {
  const steps = [...job.steps].sort((a, b) => a.ordinal - b.ordinal);
  const done = steps.filter((step) => step.status === "complete").length;
  if (job.status === "complete") return "Done. It's on the map.";
  if (job.status === "error" || job.status === "cancelled") {
    const failed = steps.find((step) => step.status === "error" || step.status === "cancelled");
    return `Processing stopped${failed ? ` at ${failed.stageId}` : ""}. Open the map to retry it.`;
  }
  const current = steps.find((step) => step.status === "in-progress");
  if (!current) return "Queued. Processing starts in a moment.";
  return `Processing: ${current.stageId} (${String(done + 1)} of ${String(steps.length)})`;
}

/** Report on the run until it ends. Reads are open, so this needs no credential. */
function follow(captureId: string, ui: Ui): void {
  const tick = async (): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/captures/${captureId}`);
      if (!response.ok) return;
      const capture = (await response.json()) as CaptureDetail;
      const job = [...capture.jobs].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
      if (!job) return;
      ui.detail.textContent = describeJob(job);
      if (job.status === "complete" || job.status === "error" || job.status === "cancelled") {
        if (ui.doneLink) ui.doneLink.hidden = false;
        return;
      }
    } catch {
      // A dropped connection on a phone is normal; the next tick tries again.
    }
    window.setTimeout(() => void tick(), POLL_MS);
  };
  window.setTimeout(() => void tick(), 2_000);
}

function collectUi(root: Document): Ui | null {
  const status = root.getElementById("status");
  const bar = root.getElementById("bar");
  const detail = root.getElementById("detail");
  const form = root.getElementById("form");
  const input = root.getElementById("file");
  if (!status || !bar || !detail || !form || !(input instanceof HTMLInputElement)) return null;
  const keyForm = root.getElementById("keyform");
  const keyInput = root.getElementById("key");
  const doneLink = root.getElementById("done-link");
  return {
    main: root.querySelector("main"),
    status,
    bar,
    detail,
    form,
    input,
    keyForm: keyForm instanceof HTMLFormElement ? keyForm : null,
    keyInput: keyInput instanceof HTMLInputElement ? keyInput : null,
    foot: root.getElementById("foot"),
    forget: root.getElementById("forget"),
    doneLink: doneLink instanceof HTMLAnchorElement ? doneLink : null,
  };
}

/** Files picked, checked; null (and a message) when some cannot be read. */
function picked(ui: Ui): File[] | null {
  const files = Array.from(ui.input.files ?? []);
  if (files.length === 0) return null;
  const refused = unsupported(files);
  if (refused.length > 0) {
    ui.status.textContent = `Can't read ${refused.join(", ")}. Choose ${SUPPORTED_TEXT}.`;
    ui.input.value = "";
    return null;
  }
  return files;
}

function failed(ui: Ui, error: unknown): void {
  ui.input.disabled = false;
  if (isAbort(error)) {
    setState(ui, "idle");
    ui.status.textContent = "Upload cancelled.";
    return;
  }
  setState(ui, "error");
  ui.status.textContent = error instanceof Error ? error.message : "The upload failed. Try again.";
}

/** A QR code from the console: one capture that already exists. */
function startHandoff(ui: Ui, token: string, captureId: string): void {
  const upload: Upload = { token, captureId };
  ui.input.addEventListener("change", () => {
    const files = picked(ui);
    if (!files) return;
    ui.input.disabled = true;
    setState(ui, "uploading");
    uploadAll(upload, files, ui)
      .then(() => {
        setState(ui, "done");
        ui.status.textContent = `${sent(files)}. You can close this page.`;
        ui.detail.textContent = "The console has it.";
      })
      .catch((error: unknown) => failed(ui, error));
  });
}

/** The phone key: start a capture here, upload it, and start processing it. */
function startWithKey(ui: Ui): void {
  if (ui.foot) {
    ui.foot.textContent =
      "Each pick becomes its own capture, placed where this phone is (if you allow location).";
  }
  const showPicker = (): void => {
    if (ui.keyForm) ui.keyForm.hidden = true;
    ui.form.hidden = false;
    if (ui.forget) ui.forget.hidden = false;
    setState(ui, "idle");
    ui.status.textContent = "Pick a video, photos or a scan. You can pick several photos at once.";
  };
  const askForKey = (message: string): void => {
    ui.form.hidden = true;
    if (ui.forget) ui.forget.hidden = true;
    if (ui.keyForm) ui.keyForm.hidden = false;
    setState(ui, "idle");
    ui.status.textContent = message;
    ui.keyInput?.focus();
  };

  ui.forget?.addEventListener("click", () => {
    storeKey(null);
    askForKey("Key forgotten on this phone.");
  });

  ui.keyForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const key = (ui.keyInput?.value ?? "").trim().toLowerCase();
    if (!key) return;
    ui.status.textContent = "Checking the key…";
    post<undefined>(key, "/api/v1/phone/check", {}, { unauthorized: KEY_WRONG })
      .then(() => {
        storeKey(key);
        showPicker();
      })
      .catch((error: unknown) => {
        ui.status.textContent = error instanceof Error ? error.message : KEY_WRONG;
      });
  });

  ui.input.addEventListener("change", () => {
    const files = picked(ui);
    const key = readStoredKey();
    if (!files) return;
    if (!key) {
      askForKey("Enter the phone key first.");
      return;
    }
    ui.input.disabled = true;
    setState(ui, "uploading");
    const proposal = classify(files.map((file) => ({ name: file.name, size: file.size })));
    void (async () => {
      ui.status.textContent = "Finding where you are…";
      const position = await locate();
      ui.status.textContent = "Starting the capture…";
      const coords = position?.coords;
      const created = await post<PhoneCapture>(
        key,
        "/api/v1/phone/captures",
        coords ? { lat: coords.latitude, lon: coords.longitude, accuracyM: coords.accuracy } : {},
        { unauthorized: KEY_WRONG },
      );
      const upload: Upload = { token: created.uploadToken, captureId: created.capture.id };
      await uploadAll(upload, files, ui);
      ui.status.textContent = "Starting processing…";
      await post<Job>(
        key,
        `/api/v1/phone/captures/${upload.captureId}/process`,
        { recipe: proposal.recipe },
        { unauthorized: KEY_WRONG },
      );
      setState(ui, "done");
      ui.input.disabled = false;
      ui.input.value = "";
      ui.status.textContent = `${sent(files)} and processing has started. ${proposal.estimate}.`;
      ui.detail.textContent = position
        ? "Placed where this phone is. You can close this page."
        : "Location was not shared, so it is placed from the file (or where the map was).";
      follow(upload.captureId, ui);
    })().catch((error: unknown) => {
      if (error instanceof ApiError && error.status === 401) {
        storeKey(null);
        ui.input.disabled = false;
        askForKey(KEY_WRONG);
        return;
      }
      failed(ui, error);
    });
  });

  if (readStoredKey()) showPicker();
  else askForKey("Enter the phone key. You only need to do this once on this phone.");
}

export function start(root: Document = document): void {
  const ui = collectUi(root);
  if (!ui) return;

  const token = root.location.hash.replace(/^#/, "").trim();
  if (!token) {
    startWithKey(ui);
    return;
  }

  const captureId = captureIdFromToken(token);
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
    ui.status.textContent = LINK_EXPIRED;
    return;
  }

  startHandoff(ui, token, captureId);
}

if (typeof document !== "undefined" && document.getElementById("file")) start();
