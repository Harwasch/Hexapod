/**
 * The phone's processing options: the few choices worth making per scan, turned into
 * the per-stage `params` the API accepts (`PHONE_OPTIONS` in apps/api's phone routes,
 * which refuses anything not listed there).
 *
 * The pipeline itself is not a choice: photos and video can only be reconstructed, and
 * a splat file can only be placed, so the panel says which one the files will take
 * rather than offering a switch that could only be set one way. The same goes for
 * outputs: every finished scan gets the 3D view and the map, and a .ply download.
 *
 * The choices are remembered on this phone (a convenience; losing them loses nothing).
 */

export type Recipe = "photo-reconstruct" | "splat-ingest";
export type StageParams = Record<string, Record<string, number | string>>;

interface Choice<V extends string> {
  value: V;
  label: string;
  hint: string;
}

const QUALITY = [
  {
    value: "quick",
    label: "Quick",
    hint: "A short training run and at most 250k splats. A fast preview.",
  },
  {
    value: "standard",
    label: "Standard",
    hint: "Training sized to the number of photos, at most 500k splats.",
  },
  {
    value: "best",
    label: "Best",
    hint: "The full 30,000-step training, up to 1M splats. Slowest and sharpest.",
  },
] as const satisfies readonly Choice<string>[];

const PHOTO_SIZE = [
  { value: "1200", label: "1200 px", hint: "Faster; softer detail." },
  { value: "1600", label: "1600 px", hint: "The balance most scans want." },
  { value: "2400", label: "2400 px", hint: "Sharper; about twice the GPU time." },
] as const satisfies readonly Choice<string>[];

const VIDEO_FPS = [
  { value: "2", label: "2 / s", hint: "For a slow walk-around." },
  { value: "4", label: "4 / s", hint: "Frames taken from a video, per second of it." },
  { value: "8", label: "8 / s", hint: "For fast movement; more frames to match." },
] as const satisfies readonly Choice<string>[];

const UP_AXIS = [
  { value: "", label: "Auto", hint: "The file format's usual convention." },
  { value: "y", label: "Y up", hint: "Most .spz files, and three.js exports." },
  { value: "-y", label: "Y down", hint: "Most .ply exports (COLMAP's convention)." },
  { value: "z", label: "Z up", hint: "Already upright, as maps and CAD are." },
] as const satisfies readonly Choice<string>[];

const DETAIL = [
  { value: "200000", label: "Light", hint: "200k splats on the map. Loads fastest." },
  { value: "400000", label: "Standard", hint: "400k splats on the map and in the viewer." },
  { value: "800000", label: "Full", hint: "800k splats. Heavier to load on a phone." },
] as const satisfies readonly Choice<string>[];

export interface Options {
  quality: (typeof QUALITY)[number]["value"];
  photoSize: (typeof PHOTO_SIZE)[number]["value"];
  videoFps: (typeof VIDEO_FPS)[number]["value"];
  upAxis: (typeof UP_AXIS)[number]["value"];
  headingDeg: number;
  detail: (typeof DETAIL)[number]["value"];
}

export const DEFAULTS: Options = {
  quality: "standard",
  photoSize: "1600",
  videoFps: "4",
  upAxis: "",
  headingDeg: 0,
  detail: "400000",
};

const TRAIN: Record<Options["quality"], Record<string, number>> = {
  quick: { schedule_full_at: 240, schedule_floor: 0.1, cap_max: 250_000 },
  // The recipe's own defaults.
  standard: {},
  best: { schedule_floor: 1, cap_max: 1_000_000 },
};

/** The per-stage params a run of `recipe` is started with. */
export function paramsFor(recipe: Recipe, options: Options): StageParams {
  const packaged = { max_gaussians: Number(options.detail) };
  if (recipe === "splat-ingest") {
    return {
      normalize: { up_axis: options.upAxis, heading_deg: options.headingDeg },
      package: packaged,
    };
  }
  const params: StageParams = {
    normalize: { max_side: Number(options.photoSize), fps: Number(options.videoFps) },
    package: packaged,
  };
  const train = TRAIN[options.quality];
  if (Object.keys(train).length > 0) params.train = train;
  return params;
}

/** One line for the panel's closed state, e.g. "Standard · 1600 px · Standard detail". */
export function summarise(options: Options): string {
  const label = <V extends string>(choices: readonly Choice<V>[], value: V): string =>
    choices.find((choice) => choice.value === value)?.label ?? value;
  return [
    label(QUALITY, options.quality),
    label(PHOTO_SIZE, options.photoSize),
    `${label(DETAIL, options.detail)} detail`,
  ].join(" · ");
}

const STORAGE = "twin.phoneOptions";

export function loadOptions(): Options {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE) ?? "{}") as Partial<Options>;
    return { ...DEFAULTS, ...stored };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveOptions(options: Options): void {
  try {
    localStorage.setItem(STORAGE, JSON.stringify(options));
  } catch {
    // Private browsing: the choices last for this visit only.
  }
}

function segmented(
  key: keyof Options,
  title: string,
  choices: readonly Choice<string>[],
  options: Options,
  changed: () => void,
): HTMLFieldSetElement {
  const set = document.createElement("fieldset");
  const legend = document.createElement("legend");
  legend.textContent = title;
  const row = document.createElement("div");
  row.className = "seg";
  const hint = document.createElement("p");
  hint.className = "hint";
  const showHint = (): void => {
    hint.textContent = choices.find((choice) => choice.value === String(options[key]))?.hint ?? "";
  };
  for (const choice of choices) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `opt-${key}`;
    input.value = choice.value;
    input.checked = String(options[key]) === choice.value;
    input.addEventListener("change", () => {
      (options as unknown as Record<string, unknown>)[key] = choice.value;
      showHint();
      changed();
    });
    label.append(input, choice.label);
    row.append(label);
  }
  showHint();
  set.append(legend, row, hint);
  return set;
}

export interface OptionsPanel {
  /** The choices as they stand now. */
  current(): Options;
  /** Show the options that apply to what was picked; null before anything is. */
  showFor(recipe: Recipe | null): void;
}

/** Fill `root` (a `<details>`) with the panel, and return a handle on its state. */
export function mountOptions(root: HTMLDetailsElement): OptionsPanel {
  const options = loadOptions();
  const summary = document.createElement("summary");
  const title = document.createElement("strong");
  title.textContent = "Processing options";
  const glance = document.createElement("span");
  summary.append(title, glance);

  const changed = (): void => {
    glance.textContent = summarise(options);
    saveOptions(options);
  };

  const route = document.createElement("p");
  route.className = "route";

  const groupTitle = (text: string): HTMLElement => {
    const heading = document.createElement("p");
    heading.className = "group-title";
    heading.textContent = text;
    return heading;
  };

  const photos = document.createElement("div");
  photos.className = "group";
  photos.append(
    groupTitle("For photos and video"),
    segmented("quality", "Quality", QUALITY, options, changed),
    segmented("photoSize", "Photo size for training", PHOTO_SIZE, options, changed),
    segmented("videoFps", "Frames from a video", VIDEO_FPS, options, changed),
  );

  const splat = document.createElement("div");
  splat.className = "group";
  const heading = document.createElement("fieldset");
  const headingLegend = document.createElement("legend");
  headingLegend.textContent = "Turn it (degrees clockwise)";
  const headingInput = document.createElement("input");
  headingInput.type = "number";
  headingInput.inputMode = "numeric";
  headingInput.min = "-359";
  headingInput.max = "359";
  headingInput.step = "15";
  headingInput.value = String(options.headingDeg);
  headingInput.className = "number";
  headingInput.setAttribute("aria-label", "Turn it, in degrees clockwise");
  headingInput.addEventListener("change", () => {
    const value = Math.max(-359, Math.min(359, Math.round(Number(headingInput.value) || 0)));
    headingInput.value = String(value);
    options.headingDeg = value;
    changed();
  });
  heading.append(headingLegend, headingInput);
  splat.append(
    groupTitle("For splat files"),
    segmented("upAxis", "Which way is up in the file", UP_AXIS, options, changed),
    heading,
  );

  const detail = document.createElement("div");
  detail.className = "group";
  detail.append(
    groupTitle("For every scan"),
    segmented("detail", "Detail on the map and in the viewer", DETAIL, options, changed),
  );

  const outputs = document.createElement("div");
  outputs.className = "outputs";
  const outputsTitle = document.createElement("p");
  outputsTitle.className = "legend";
  outputsTitle.textContent = "You get";
  const outputList = document.createElement("ul");
  for (const text of [
    "A 3D view, and the scan placed on the map",
    "The full splat as a .ply download",
  ]) {
    const item = document.createElement("li");
    item.textContent = text;
    outputList.append(item);
  }
  outputs.append(outputsTitle, outputList);

  const body = document.createElement("div");
  body.className = "body";
  body.append(route, photos, splat, detail, outputs);
  root.replaceChildren(summary, body);
  changed();

  const showFor = (recipe: Recipe | null): void => {
    photos.hidden = recipe === "splat-ingest";
    splat.hidden = recipe === "photo-reconstruct";
    route.textContent =
      recipe === "splat-ingest"
        ? "A splat file (.ply or .spz, e.g. from Scaniverse or Polycam) is placed on the map as it is. No GPU training."
        : recipe === "photo-reconstruct"
          ? "Photos or a video are turned into a 3D model: camera positions on a CPU, then Gaussian-splat training on a cloud GPU."
          : "Photos or a video become a 3D model on a cloud GPU; a splat file (.ply, .spz) is placed as it is. The route is chosen from the files you pick.";
  };
  showFor(null);
  return { current: () => ({ ...options }), showFor };
}
