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
  Capture,
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
const POLL_MS = 10_000;

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
  resume: HTMLButtonElement | null;
  mine: HTMLElement | null;
  mineList: HTMLElement | null;
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

/** How far one file got, so a resume continues it rather than starting it again. */
interface FileProgress {
  fileId: string | null;
  parts: { partNumber: number; etag: string }[];
  done: boolean;
}

/** Tries per step before the page gives up and offers "Resume upload" instead. */
const ATTEMPTS = 8;

/** A failure worth trying again: no connection, a gateway, a busy server. Not a refusal. */
function transient(error: unknown): boolean {
  if (error instanceof TypeError) return true; // fetch's own "network error"
  if (!(error instanceof ApiError)) return false;
  return error.status === 0 || error.status === 408 || error.status === 429 || error.status >= 500;
}

/** Resolves once the page is on screen and the phone thinks it is online. */
function whenReachable(): Promise<void> {
  if (document.visibilityState === "visible" && navigator.onLine) return Promise.resolve();
  return new Promise((resolve) => {
    const check = (): void => {
      if (document.visibilityState === "visible" && navigator.onLine) {
        document.removeEventListener("visibilitychange", check);
        window.removeEventListener("online", check);
        resolve();
      }
    };
    document.addEventListener("visibilitychange", check);
    window.addEventListener("online", check);
  });
}

/**
 * One step of an upload, tried again when the connection drops.
 *
 * A phone that locks mid-upload suspends the page: the request in flight dies with a
 * network error and nothing more is sent until it is unlocked. So a transient failure
 * waits for the page to be visible and online again, backs off, and repeats the same
 * step -- a part is the unit, so at most one part's bytes are sent twice.
 */
async function persist<T>(ui: Ui, step: () => Promise<T>): Promise<T> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await step();
    } catch (error) {
      if (!transient(error) || attempt >= ATTEMPTS) throw error;
      const hidden = document.visibilityState !== "visible" || !navigator.onLine;
      ui.detail.textContent = hidden
        ? "Paused while the phone was asleep or offline. It carries on when you come back."
        : `Connection dropped. Trying again (${String(attempt)} of ${String(ATTEMPTS - 1)})…`;
      await whenReachable();
      await new Promise((resolve) =>
        window.setTimeout(resolve, Math.min(30_000, 1_000 * 2 ** attempt)),
      );
    }
  }
}

/**
 * Keep the screen on while bytes are moving. iOS Safari 16.4+ and Chrome honour it;
 * where it is refused the upload still works, and `persist` covers a lock.
 */
function keepAwake(): () => void {
  let lock: WakeLockSentinel | null = null;
  let wanted = true;
  const request = async (): Promise<void> => {
    try {
      if (wanted && document.visibilityState === "visible" && "wakeLock" in navigator) {
        lock = await navigator.wakeLock.request("screen");
      }
    } catch {
      lock = null;
    }
  };
  // The system drops the lock whenever the page is hidden; take it again on return.
  const onVisible = (): void => void request();
  document.addEventListener("visibilitychange", onVisible);
  void request();
  return () => {
    wanted = false;
    document.removeEventListener("visibilitychange", onVisible);
    void lock?.release().catch(() => undefined);
  };
}

async function uploadOne(
  upload: Upload,
  file: File,
  progress: FileProgress,
  ui: Ui,
): Promise<void> {
  const base = `/api/v1/captures/${upload.captureId}/files`;
  const opts = { unauthorized: LINK_EXPIRED, upload };
  const partBytes = (partNumber: number, partSize: number): number =>
    partSizeFor(file.size, partSize, partNumber);

  let window: UploadWindow;
  if (progress.fileId === null) {
    const registered = await persist(ui, () =>
      post<CaptureFileUpload>(
        upload.token,
        base,
        {
          filename: file.name,
          contentType: file.type || "application/octet-stream",
          bytes: file.size,
        },
        opts,
      ),
    );
    progress.fileId = registered.file.id;
    window = registered.upload;
  } else {
    // A resume: ask for fresh URLs from the first part that has not landed.
    const fileId = progress.fileId;
    window = await persist(ui, () =>
      post<UploadWindow>(
        upload.token,
        `${base}/${fileId}/parts`,
        { fromPartNumber: progress.parts.length + 1 },
        opts,
      ),
    );
  }
  const fileId = progress.fileId;
  const landed = (): number =>
    progress.parts.reduce((sum, part) => sum + partBytes(part.partNumber, window.partSize), 0);

  for (;;) {
    for (const part of window.parts) {
      if (progress.parts.some((done) => done.partNumber === part.partNumber)) continue;
      const size = partBytes(part.partNumber, window.partSize);
      const start = (part.partNumber - 1) * window.partSize;
      const before = landed();
      const etag = await persist(ui, () =>
        putPart(part.url, file.slice(start, start + size), {
          onProgress: (loaded) => {
            setProgress(ui, before + loaded, file.size);
          },
        }),
      );
      progress.parts.push({ partNumber: part.partNumber, etag });
      setProgress(ui, landed(), file.size);
    }
    if (window.nextPartNumber === null || window.nextPartNumber === undefined) break;
    const from = window.nextPartNumber;
    window = await persist(ui, () =>
      post<UploadWindow>(upload.token, `${base}/${fileId}/parts`, { fromPartNumber: from }, opts),
    );
    if (window.parts.length === 0) break;
  }

  const parts = [...progress.parts].sort((a, b) => a.partNumber - b.partNumber);
  await persist(ui, () =>
    post<CaptureFile>(upload.token, `${base}/${fileId}/complete`, { parts }, opts),
  );
  progress.done = true;
}

/**
 * One after another, like the console: parallel uploads would share one cellular uplink
 * and make the progress bar a lie. A file already done is skipped, which is what makes
 * calling this again a resume.
 */
async function uploadAll(
  upload: Upload,
  files: File[],
  progress: Map<File, FileProgress>,
  ui: Ui,
): Promise<void> {
  const release = keepAwake();
  try {
    for (const [index, file] of files.entries()) {
      let state = progress.get(file);
      if (!state) {
        state = { fileId: null, parts: [], done: false };
        progress.set(file, state);
      }
      if (state.done) continue;
      ui.status.textContent =
        files.length === 1
          ? `Uploading ${file.name}`
          : `Uploading ${file.name} (${String(index + 1)} of ${String(files.length)})`;
      setProgress(ui, 0, file.size);
      await uploadOne(upload, file, state, ui);
    }
  } finally {
    release();
  }
}

function sent(files: File[]): string {
  const what =
    files.length === 1 ? (files[0]?.name ?? "The file") : `${String(files.length)} files`;
  return `${what} ${files.length === 1 ? "is on its way" : "are on their way"}`;
}

/**
 * The phone's own position, or null -- in at most `LOCATE_MS`, whatever the browser does.
 *
 * The API's own `timeout` is not enough on iOS: it only starts once permission has been
 * granted, so a permission prompt that is never answered (or an in-app browser that
 * never shows one) leaves the callbacks uncalled for ever, and the page sat on "Finding
 * where you are…" with the files never sent. Location is a better placement, not a
 * requirement, so after a few seconds the capture goes ahead without it.
 */
const LOCATE_MS = 8_000;

function locate(): Promise<GeolocationPosition | null> {
  if (!("geolocation" in navigator)) return Promise.resolve(null);
  return new Promise((resolve) => {
    const giveUp = window.setTimeout(() => resolve(null), LOCATE_MS);
    const settle = (position: GeolocationPosition | null): void => {
      window.clearTimeout(giveUp);
      resolve(position);
    };
    try {
      navigator.geolocation.getCurrentPosition(settle, () => settle(null), {
        enableHighAccuracy: true,
        timeout: LOCATE_MS,
        maximumAge: 60_000,
      });
    } catch {
      settle(null);
    }
  });
}

/** What each stage is, in words a person watching a phone would use. */
const STAGE_LABEL: Record<string, string> = {
  normalize: "Preparing the frames",
  pose: "Working out where each photo was taken",
  mask: "Masking moving things",
  train: "Training the 3D model (the long part)",
  compensate: "Evening out the exposure",
  georeference: "Placing it on the map",
  place: "Placing it on the map",
  package: "Packaging it for viewing",
  thumbnail: "Making a preview",
  ground_samples: "Measuring the ground",
  manifest: "Writing it up",
  register: "Publishing",
};

function stageLabel(stageId: string): string {
  return STAGE_LABEL[stageId] ?? stageId;
}

function minutesSince(iso: string | null | undefined): string {
  if (!iso) return "";
  const minutes = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60_000));
  return minutes < 1 ? "under a minute" : `${String(minutes)} min`;
}

/** Every recipe's stage ids in order, fetched once: steps appear only as they start. */
let recipeStages: Promise<Map<string, string[]>> | null = null;
function stagesOf(recipe: string): Promise<string[]> {
  recipeStages ??= fetch(`${API_BASE}/api/v1/recipes`)
    .then((response) => response.json())
    .then(
      (body: { recipes?: { name: string; stages: { id: string }[] }[] }) =>
        new Map((body.recipes ?? []).map((r) => [r.name, r.stages.map((stage) => stage.id)])),
    )
    .catch(() => {
      recipeStages = null;
      return new Map<string, string[]>();
    });
  return recipeStages.then((all) => all.get(recipe) ?? []);
}

function latestJob(jobs: Job[]): Job | undefined {
  return [...jobs].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
}

const ENDED = new Set(["complete", "error", "cancelled"]);

/** Headline, detail and a 0-1 fraction for one run, from its steps and its recipe. */
async function describeRun(job: Job): Promise<{ headline: string; detail: string; done: number }> {
  const order = await stagesOf(job.recipe);
  const steps = [...job.steps].sort((a, b) => a.ordinal - b.ordinal);
  const total = Math.max(order.length, steps.length, 1);
  const finished = steps.filter((step) => step.status === "complete").length;
  if (job.status === "complete") {
    return { headline: "Done.", detail: "Tap View in 3D.", done: 1 };
  }
  if (job.status === "error" || job.status === "cancelled") {
    const stopped = steps.find((step) => step.status === "error" || step.status === "cancelled");
    return {
      headline: `Processing stopped${stopped ? `: ${stageLabel(stopped.stageId)}` : ""}.`,
      detail: job.error ?? "It can be retried from the console.",
      done: finished / total,
    };
  }
  const current = steps.find((step) => step.status === "in-progress");
  if (!current) {
    return {
      headline: finished === 0 ? "Queued." : "Between steps…",
      detail:
        finished === 0
          ? "Waiting for the worker to pick it up, usually under a minute."
          : `${String(finished)} of ${String(total)} steps done.`,
      done: finished / total,
    };
  }
  const long = current.stageId === "train" ? " Usually 10–30 minutes." : "";
  return {
    headline: `${stageLabel(current.stageId)}…`,
    detail: `Step ${String(finished + 1)} of ${String(total)} · running ${minutesSince(current.startedAt)}.${long}`,
    done: finished / total,
  };
}

/** The standalone scan viewer (view.html) on one site. */
function viewerLink(siteId: string): string {
  return `/view.html#${encodeURIComponent(siteId)}`;
}

/** The one run the status panel is following; starting another stops this one. */
let following: (() => void) | null = null;

/**
 * Show a capture's run in the status panel until it ends.
 *
 * It keeps asking whatever goes wrong -- a phone drops requests, and one bad answer
 * once ended the polling for good, which is how the page sat on "Queued" while the GPU
 * was training. It also asks again the moment the page is back on screen, because a
 * phone suspends timers while it is locked or in another app. Reads are open in this
 * API, so this needs no credential.
 */
function follow(captureId: string, ui: Ui): void {
  following?.();
  let timer = 0;
  let stopped = false;
  const tick = async (): Promise<void> => {
    window.clearTimeout(timer);
    if (stopped) return;
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/jobs?captureId=${encodeURIComponent(captureId)}&limit=5`,
      );
      const job = response.ok ? latestJob((await response.json()) as Job[]) : undefined;
      if (job && !stopped) {
        const run = await describeRun(job);
        ui.status.textContent = run.headline;
        ui.detail.textContent = run.detail;
        ui.bar.style.width = `${String(Math.round(run.done * 100))}%`;
        if (ENDED.has(job.status)) {
          setState(ui, job.status === "complete" ? "done" : "error");
          const capture = (await fetch(`${API_BASE}/api/v1/captures/${captureId}`).then((r) =>
            r.json(),
          )) as CaptureDetail;
          if (ui.doneLink && capture.siteId) {
            ui.doneLink.href = viewerLink(capture.siteId);
            ui.doneLink.hidden = false;
          }
          refreshMine(ui);
          stop();
          return;
        }
      }
    } catch {
      // A dropped connection on a phone is normal; the next tick tries again.
    }
    if (!stopped) timer = window.setTimeout(() => void tick(), POLL_MS);
  };
  const onVisible = (): void => {
    if (document.visibilityState === "visible") void tick();
  };
  const stop = (): void => {
    stopped = true;
    window.clearTimeout(timer);
    document.removeEventListener("visibilitychange", onVisible);
    if (following === stop) following = null;
  };
  following = stop;
  document.addEventListener("visibilitychange", onVisible);
  void tick();
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
  const resume = root.getElementById("resume");
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
    resume: resume instanceof HTMLButtonElement ? resume : null,
    mine: root.getElementById("mine"),
    mineList: root.getElementById("mine-list"),
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

/**
 * Offer to carry on from where an upload stopped, in the same capture. The `File`s are
 * still held by this page, so nothing has to be picked again -- which is exactly what
 * is lost if the page is closed, so the message says to keep it open.
 */
function offerResume(ui: Ui, error: unknown, resume: () => void): void {
  failed(ui, error);
  if (isAbort(error) || !ui.resume) return;
  ui.detail.textContent = "What already arrived is kept. Keep this page open and resume.";
  ui.resume.hidden = false;
  ui.resume.onclick = () => {
    if (ui.resume) ui.resume.hidden = true;
    resume();
  };
}

/** A QR code from the console: one capture that already exists. */
function startHandoff(ui: Ui, token: string, captureId: string): void {
  const upload: Upload = { token, captureId };
  const progress = new Map<File, FileProgress>();
  const run = (files: File[]): void => {
    ui.input.disabled = true;
    setState(ui, "uploading");
    uploadAll(upload, files, progress, ui)
      .then(() => {
        setState(ui, "done");
        ui.status.textContent = `${sent(files)}. You can close this page.`;
        ui.detail.textContent = "The console has it.";
      })
      .catch((error: unknown) => offerResume(ui, error, () => run(files)));
  };
  ui.input.addEventListener("change", () => {
    const files = picked(ui);
    if (files) run(files);
  });
}

// --- this phone's captures -----------------------------------------------------------

/** Refreshes the list while anything in it is still processing. */
let mineTimer = 0;

/**
 * What this key has sent, newest first, with what can be done about each. The state
 * comes from each capture's latest run, not from the capture's own status, which does
 * not change while a run is going -- reading that is how a capture that was training
 * still offered "Process". Reads are open in this API, so listing needs no key.
 */
function refreshMine(ui: Ui): void {
  const list = ui.mineList;
  const section = ui.mine;
  if (!list || !section) return;
  window.clearTimeout(mineTimer);
  void (async () => {
    try {
      const [captureResponse, jobResponse] = await Promise.all([
        fetch(`${API_BASE}/api/v1/captures?limit=100`),
        fetch(`${API_BASE}/api/v1/jobs?limit=100`),
      ]);
      if (!captureResponse.ok) return;
      const captures = ((await captureResponse.json()) as Capture[])
        .filter((capture) => capture.metadata.origin === "phone-key")
        .slice(0, 12);
      const jobs = jobResponse.ok ? ((await jobResponse.json()) as Job[]) : [];
      const byCapture = new Map<string, Job[]>();
      for (const job of jobs) {
        byCapture.set(job.captureId, [...(byCapture.get(job.captureId) ?? []), job]);
      }
      section.hidden = captures.length === 0;
      const rows = await Promise.all(
        captures.map((capture) =>
          captureRow(ui, capture, latestJob(byCapture.get(capture.id) ?? [])),
        ),
      );
      list.replaceChildren(...rows);
      const busy = captures.some((capture) => {
        const job = latestJob(byCapture.get(capture.id) ?? []);
        return job !== undefined && !ENDED.has(job.status);
      });
      if (busy) mineTimer = window.setTimeout(() => refreshMine(ui), POLL_MS);
    } catch {
      // Offline: the list simply stays as it was.
    }
  })();
}

function rowButton(label: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rowbtn";
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

async function captureRow(ui: Ui, capture: Capture, job: Job | undefined): Promise<HTMLElement> {
  const row = document.createElement("li");
  row.dataset.state = job?.status ?? "none";
  const text = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = capture.name;
  const state = document.createElement("span");
  text.append(name, state);
  row.append(text);

  const complete = capture.files.filter((file) => file.status === "complete");
  const everything = capture.files.length > 0 && complete.length === capture.files.length;

  const process = (): void => {
    const key = readStoredKey();
    if (!key) return;
    const recipe = classify(
      complete.map((file) => ({ name: file.filename, size: file.bytes ?? 0 })),
    ).recipe;
    post<Job>(
      key,
      `/api/v1/phone/captures/${capture.id}/process`,
      { recipe },
      {
        unauthorized: KEY_WRONG,
      },
    )
      .catch((error: unknown) => {
        // Already running is not a failure: follow the run that is.
        if (error instanceof ApiError && error.status === 409) return undefined;
        throw error;
      })
      .then(() => {
        follow(capture.id, ui);
        refreshMine(ui);
      })
      .catch((error: unknown) => {
        ui.status.textContent = error instanceof Error ? error.message : "Could not start it.";
      });
  };

  if (capture.siteId && (!job || job.status === "complete")) {
    state.textContent = "Ready";
    const view = document.createElement("a");
    view.className = "rowbtn";
    view.href = viewerLink(capture.siteId);
    view.textContent = "View in 3D";
    row.append(view);
  } else if (job && !ENDED.has(job.status)) {
    const run = await describeRun(job);
    state.textContent = `${run.headline} ${run.detail}`;
    row.append(rowButton("Progress", () => follow(capture.id, ui)));
  } else if (job) {
    const run = await describeRun(job);
    state.textContent = run.headline;
    if (everything) row.append(rowButton("Try again", process));
  } else if (!everything) {
    // Deliberately no Process: a reconstruction from part of a capture is not the
    // capture, and it would be spent GPU time on a result nobody asked for.
    state.textContent = `Upload didn't finish · ${String(complete.length)} of ${String(capture.files.length)} files. Send it again.`;
  } else {
    state.textContent = `Uploaded, not processed · ${String(complete.length)} files`;
    row.append(rowButton("Process", process));
  }
  return row;
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
    refreshMine(ui);
  };
  const askForKey = (message: string): void => {
    ui.form.hidden = true;
    if (ui.forget) ui.forget.hidden = true;
    if (ui.keyForm) ui.keyForm.hidden = false;
    if (ui.mine) ui.mine.hidden = true;
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

  /** One pick: a capture, its uploads, its run. Resumable from whichever step failed. */
  const send = (key: string, files: File[]): void => {
    const proposal = classify(files.map((file) => ({ name: file.name, size: file.size })));
    const progress = new Map<File, FileProgress>();
    let upload: Upload | null = null;
    let located = false;
    let processed = false;

    const attempt = async (): Promise<void> => {
      ui.input.disabled = true;
      setState(ui, "uploading");
      if (!upload) {
        ui.status.textContent = "Finding where you are (a few seconds at most)…";
        const coords = (await locate())?.coords;
        located = coords !== undefined;
        ui.status.textContent = "Starting the capture…";
        const created = await persist(ui, () =>
          post<PhoneCapture>(
            key,
            "/api/v1/phone/captures",
            coords
              ? { lat: coords.latitude, lon: coords.longitude, accuracyM: coords.accuracy }
              : {},
            { unauthorized: KEY_WRONG },
          ),
        );
        upload = { token: created.uploadToken, captureId: created.capture.id };
        refreshMine(ui);
      }
      const current = upload;
      await uploadAll(current, files, progress, ui);
      if (!processed) {
        ui.status.textContent = "Starting processing…";
        await persist(ui, () =>
          post<Job>(
            key,
            `/api/v1/phone/captures/${current.captureId}/process`,
            { recipe: proposal.recipe },
            { unauthorized: KEY_WRONG },
          ),
        );
        processed = true;
      }
      setState(ui, "done");
      ui.input.disabled = false;
      ui.input.value = "";
      ui.status.textContent = `${sent(files)} and processing has started. ${proposal.estimate}.`;
      ui.detail.textContent = located
        ? "Placed where this phone is. You can close this page; it will be in Your captures."
        : "Location was not shared, so it is placed from the file (or where the map was).";
      follow(current.captureId, ui);
      refreshMine(ui);
    };

    const run = (): void => {
      attempt().catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 401 && !upload) {
          storeKey(null);
          ui.input.disabled = false;
          askForKey(KEY_WRONG);
          return;
        }
        offerResume(ui, error, run);
        refreshMine(ui);
      });
    };
    run();
  };

  ui.input.addEventListener("change", () => {
    const files = picked(ui);
    const key = readStoredKey();
    if (!files) return;
    if (!key) {
      askForKey("Enter the phone key first.");
      return;
    }
    if (ui.resume) ui.resume.hidden = true;
    send(key, files);
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
