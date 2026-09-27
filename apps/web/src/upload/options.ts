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
 * Photos and video go through two passes, Scaniverse-style: a cheap **preview** that
 * trains briefly and forecasts how much of the scan the capture can support, then --
 * when the forecast is worth it -- **Refine**, the full-quality pass over the same frames
 * and camera positions, trained only inside the region the cameras were pointed at. The
 * quality bar decides what of the result is kept (apps/api `quality` stage).
 *
 * The choices are remembered on this phone (a convenience; losing them loses nothing).
 */

export type Recipe = "photo-reconstruct" | "splat-ingest";
export type StageParams = Record<string, Record<string, number | string | boolean>>;

interface Choice<V extends string> {
  value: V;
  label: string;
  hint: string;
}

const QUALITY = [
  {
    value: "quick",
    label: "Quick",
    hint: "Refine with a short training run and half the splats the scan's detail calls for.",
  },
  {
    value: "standard",
    label: "Standard",
    hint: "Refine with as many splats as the scan's detail calls for, trained until it stops improving.",
  },
  {
    value: "best",
    label: "Best",
    hint: "Refine with the full training and twice the splats, as far as the GPU holds. Slowest and sharpest.",
  },
] as const satisfies readonly Choice<string>[];

/** What is kept of the reconstruction: tools/pipeline/quality.py's `bar`. */
const BAR = [
  {
    value: "strict",
    label: "Strict",
    hint: "Only what enough frames saw well, from enough angles. Everything else is removed. The default.",
  },
  {
    value: "balanced",
    label: "Balanced",
    hint: "The well-seen parts, plus the fringes faded, for context.",
  },
  {
    value: "everything",
    label: "Everything",
    hint: "No quality bar: everything the training made, spikes and all.",
  },
] as const satisfies readonly Choice<string>[];

/**
 * `auto` is tools/pipeline/resolution.py: 1600 px unless a sample of the scan's sharpest
 * frames measurably holds detail above it (4K video of something far away, full-size
 * photos), then up to 2400. A number overrides the measurement.
 */
const PHOTO_SIZE = [
  {
    value: "auto",
    label: "Auto",
    hint: "The default. 1600 px, or up to 2400 when the scan measurably holds finer detail.",
  },
  { value: "1200", label: "1200 px", hint: "Faster; softer detail." },
  { value: "1600", label: "1600 px", hint: "The balance most scans want." },
  { value: "2400", label: "2400 px", hint: "Sharper; about twice the GPU time." },
] as const satisfies readonly Choice<string>[];

/**
 * A video's frames are chosen by how far the camera moved (tools/pipeline/keyframes.py):
 * a slow close-up keeps few, a long walk many. A number here caps how many a second can
 * be considered; "" sends nothing, which is the recipe's own rate (15 a second).
 */
const VIDEO_FPS = [
  {
    value: "",
    label: "By motion",
    hint: "The default. More frames where the camera moved, fewer where it barely did.",
  },
  {
    value: "4",
    label: "Up to 4 / s",
    hint: "At most 4 a second: faster camera solving, less detail.",
  },
  {
    value: "8",
    label: "Up to 8 / s",
    hint: "At most 8 a second, still fewer where the camera paused.",
  },
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
  bar: (typeof BAR)[number]["value"];
  photoSize: (typeof PHOTO_SIZE)[number]["value"];
  videoFps: (typeof VIDEO_FPS)[number]["value"];
  upAxis: (typeof UP_AXIS)[number]["value"];
  headingDeg: number;
  detail: (typeof DETAIL)[number]["value"];
}

export const DEFAULTS: Options = {
  quality: "standard",
  bar: "strict",
  photoSize: "auto",
  videoFps: "",
  upAxis: "",
  headingDeg: 0,
  detail: "400000",
};

/**
 * The Refine's training per quality tier. The recipe sizes the gaussian budget to the
 * capture -- its surface in its own finest pixels (tools/pipeline/gaussian_budget.py) --
 * so a tier scales that measured budget (`density_scale`) rather than naming a count:
 * a fixed 500k was too few for a table and too many for a teacup. The pipeline's floor
 * (the preview's 200k) and its ceilings (GPU memory, what the worker can package) apply
 * to every tier.
 */
const TRAIN: Record<Options["quality"], Record<string, number>> = {
  quick: { schedule_full_at: 240, schedule_floor: 0.1, density_scale: 0.5 },
  // The recipe's own defaults.
  standard: {},
  best: { schedule_floor: 1, density_scale: 2 },
};

/**
 * The preview's training: a tenth of the schedule, at most 200k splats, on 800 px images.
 * Minutes on the GPU rather than most of an hour, and enough to see what the capture can
 * support -- which is what the quality stage forecasts from.
 */
export const PREVIEW_TRAIN = { schedule_scale: 0.1, cap_max: 200_000, train_max_side: 800 };

/** The per-stage params of a full-quality run of `recipe`: what Refine sends. */
export function paramsFor(recipe: Recipe, options: Options): StageParams {
  const packaged = { max_gaussians: Number(options.detail) };
  if (recipe === "splat-ingest") {
    return {
      normalize: { up_axis: options.upAxis, heading_deg: options.headingDeg },
      package: packaged,
    };
  }
  const normalize: Record<string, number | string> = {
    max_side: options.photoSize === "auto" ? "auto" : Number(options.photoSize),
  };
  if (options.videoFps !== "") normalize.fps = Number(options.videoFps);
  const params: StageParams = {
    normalize,
    package: packaged,
    quality: { bar: options.bar },
  };
  const train = TRAIN[options.quality];
  if (Object.keys(train).length > 0) params.train = train;
  return params;
}

/**
 * The params a new capture is first processed with: the preview. The frames and the
 * camera positions it makes are the ones Refine keeps, so the photo size and frame rate
 * are the phone's choices here; the training is the preview's own.
 */
export function previewParamsFor(recipe: Recipe, options: Options): StageParams {
  const params = paramsFor(recipe, options);
  if (recipe === "splat-ingest") return params;
  return { ...params, train: { ...PREVIEW_TRAIN }, quality: { mode: "preview", bar: options.bar } };
}

/** One line for the panel's closed state, e.g. "Standard · 1600 px · Standard detail". */
export function summarise(options: Options): string {
  const label = <V extends string>(choices: readonly Choice<V>[], value: V): string =>
    choices.find((choice) => choice.value === value)?.label ?? value;
  return [
    label(QUALITY, options.quality),
    options.photoSize === "auto" ? "Auto size" : label(PHOTO_SIZE, options.photoSize),
    `${label(DETAIL, options.detail)} detail`,
    `${label(BAR, options.bar)} bar`,
  ].join(" · ");
}

// Versioned: every choice is saved, defaults included, so a phone that had opened the
// page keeps an old default after it changes. v1 kept 4 / s after 8 / s became the
// default (measured sharper, 2026-09-27); v2 kept 1600 px and 8 / s after Auto and
// frames by camera motion did.
const STORAGE = "twin.phoneOptions.v3";

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
    segmented("bar", "Quality bar", BAR, options, changed),
    segmented("quality", "Refine training", QUALITY, options, changed),
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
    "A quick preview first, with a forecast of how much will be high quality",
    "Refine for the full-quality pass, kept to what the photos support",
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
