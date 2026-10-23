/**
 * Looks for the landing page's 3D scenes, for Harrison to compare (review of 2026-10-22: "I love
 * the high-level idea, but maybe we need to try some different sort of design feels for it").
 *
 * Every look draws the same world and the same crews (scene.js); a look is only data: colours
 * (the --anatomy-* tokens the renderer reads), the ground's own colours, scan dots,
 * ink outlines on or off, real light with soft shadows, the camera, and a finish (grain,
 * tilt-shift, vignette, tint). `?look=<id>` picks one; `?review` adds a switcher. Once one is
 * chosen, the rest go.
 */

export interface SceneLight {
  /** Sun and sky strength. */
  sun: number;
  sky: number;
  sunColor: string;
  skyColor: string;
  groundColor: string;
  /** Sun direction: degrees around from the camera, and above the horizon. */
  az: number;
  el: number;
  object: string;
  robot: string;
  dark: string;
  solar: string;
  ground: string;
  water: string;
  road: string;
  mulch: string;
}

/** What scene.js applies (Engine.setLook). */
export interface SceneLook {
  tokens?: Record<string, string>;
  /** The ground's own colours: its flat tops, and its slopes and sides. */
  terrain?: { top: string; side: string };
  /** Scan dots: spacing in metres. */
  stipple?: number;
  edges?: boolean;
  edgeAlpha?: number;
  lines?: boolean;
  light?: SceneLight;
  flat?: Partial<Record<"water" | "road" | "mulch", string>>;
  camera?: { el?: number; tilt?: number; zoom?: number };
  post?: {
    tilt?: number;
    vignette?: number;
    grain?: number;
    tint?: { color: string; amount: number };
  };
}

export interface Look {
  id: string;
  name: string;
  /** One line on what it is going for. */
  blurb: string;
  /** The hero behind the copy: a dark look turns the copy light. */
  stage?: string;
  dark?: boolean;
  scene: SceneLook;
}

/** The line drawing, smoothed: the page's look until another is chosen. */
const LINE: Look = {
  id: "line",
  name: "Line drawing (fixed)",
  blurb: "The current look, smoothed: more supersampling, GPU anti-aliasing, no bands or jumps.",
  scene: {},
};

export const LOOKS: Look[] = [
  LINE,
  {
    id: "clay",
    name: "Clay",
    blurb: "One soft material under daylight with real shadows; the robots in one warm colour.",
    stage: "#ECE8E1",
    scene: {
      tokens: { "--anatomy-card": "#ECE8E1" },
      edges: false,
      lines: false,
      light: {
        sun: 2.3,
        sky: 1.15,
        sunColor: "#FFF5E8",
        skyColor: "#F2F1EE",
        groundColor: "#B7B0A3",
        az: 55,
        el: 42,
        object: "#E7E2D9",
        robot: "#DD7448",
        dark: "#8E8981",
        solar: "#3D4859",
        ground: "#EAE5DC",
        water: "#B8CDD5",
        road: "#DAD4C9",
        mulch: "#CFC4B2",
      },
      post: { grain: 0.012 },
    },
  },
  {
    id: "golden",
    name: "Golden hour",
    blurb: "Late sun from the side, long shadows, a warm grade: closer to a photograph.",
    stage: "#EFDDBD",
    scene: {
      tokens: { "--anatomy-card": "#EFDDBD" },
      edges: false,
      lines: false,
      light: {
        sun: 3.1,
        sky: 0.85,
        sunColor: "#FFC488",
        skyColor: "#9DB6D4",
        groundColor: "#C6A472",
        az: 105,
        el: 15,
        object: "#E3D4B7",
        robot: "#FBF7F0",
        dark: "#6A5947",
        solar: "#2E3A4D",
        ground: "#CDBA8B",
        water: "#7EA2B6",
        road: "#D8C39E",
        mulch: "#B79B72",
      },
      post: { vignette: 0.28, tint: { color: "#FFE2BC", amount: 0.12 }, grain: 0.012 },
    },
  },
  {
    id: "miniature",
    name: "Miniature",
    blurb: "A model railway: bright toy colours, a higher camera, tilt-shift blur top and bottom.",
    stage: "#DCE7CF",
    scene: {
      tokens: { "--anatomy-card": "#DCE7CF" },
      edges: false,
      lines: false,
      camera: { zoom: 0.82, tilt: 12 },
      light: {
        sun: 2.6,
        sky: 1.0,
        sunColor: "#FFF3DF",
        skyColor: "#E6EFFF",
        groundColor: "#91A878",
        az: 40,
        el: 50,
        object: "#EADDC5",
        robot: "#F2BE45",
        dark: "#55694A",
        solar: "#2A3C56",
        ground: "#9DC37B",
        water: "#6EA7C8",
        road: "#E4DAC3",
        mulch: "#B48D61",
      },
      post: { tilt: 1.0, vignette: 0.18 },
    },
  },
  {
    id: "color",
    name: "Flat colour",
    blurb: "The flat-shaded models in a real palette: green ground, cream objects, orange robots.",
    stage: "#EAF0E4",
    scene: {
      tokens: {
        "--anatomy-card": "#EAF0E4",
        "--anatomy-top": "#F2EADA",
        "--anatomy-shade-3": "#E8DEC9",
        "--anatomy-shade-2": "#DDD1B8",
        "--anatomy-shade-1": "#CDBFA3",
        "--anatomy-shade-0": "#B9A88A",
        "--anatomy-lit-top": "#F08A3C",
        "--anatomy-lit-shade": "#D6702A",
        "--anatomy-hi": "#3B4A36",
        "--anatomy-mid": "#6E7F66",
        "--anatomy-lo": "#9FB094",
        "--anatomy-faint": "#C5D2BA",
        "--anatomy-lit": "#F08A3C",
      },
      terrain: { top: "#A8C686", side: "#7FA061" },
      flat: { water: "#86B9D8", road: "#E2D7BF", mulch: "#B5946B" },
      edgeAlpha: 0.5,
    },
  },
  {
    id: "paper",
    name: "Ink and wash",
    blurb: "Pen lines over watercolour washes on warm paper: a field sketchbook.",
    stage: "#F1E9D8",
    scene: {
      tokens: {
        "--anatomy-card": "#F1E9D8",
        "--anatomy-top": "#F1E9D8",
        "--anatomy-shade-3": "#E7DCC4",
        "--anatomy-shade-2": "#DCCDAE",
        "--anatomy-shade-1": "#CDB993",
        "--anatomy-shade-0": "#BCA57D",
        "--anatomy-lit-top": "#C9D8E6",
        "--anatomy-lit-shade": "#A9BED3",
        "--anatomy-hi": "#2A241D",
        "--anatomy-mid": "#5E5444",
        "--anatomy-lo": "#8E8269",
        "--anatomy-faint": "#BFB296",
        "--anatomy-lit": "#2A241D",
      },
      terrain: { top: "#EFE6D2", side: "#D3C29F" },
      flat: { water: "#BCD0D6", road: "#EADFC8" },
      edgeAlpha: 1,
      post: { grain: 0.05, vignette: 0.2, tint: { color: "#F7EBD2", amount: 0.15 } },
    },
  },
  {
    id: "blueprint",
    name: "Blueprint",
    blurb: "An engineering drawing: white linework on deep blue, the robots picked out.",
    stage: "#0F2D4A",
    dark: true,
    scene: {
      tokens: {
        "--anatomy-card": "#0F2D4A",
        "--anatomy-top": "#133659",
        "--anatomy-shade-3": "#12324F",
        "--anatomy-shade-2": "#10304B",
        "--anatomy-shade-1": "#0E2B44",
        "--anatomy-shade-0": "#0C253B",
        "--anatomy-lit-top": "#1F4D7E",
        "--anatomy-lit-shade": "#1A4472",
        "--anatomy-hi": "#D6ECFF",
        "--anatomy-mid": "#8FB6D9",
        "--anatomy-lo": "#5C86AD",
        "--anatomy-faint": "#2E5378",
        "--anatomy-lit": "#FFFFFF",
      },
      flat: { water: "#0B2236", road: "#163A5E" },
      edgeAlpha: 0.95,
      post: { grain: 0.02 },
    },
  },
  {
    id: "night",
    name: "Night survey",
    blurb: "Dark ground, the robots glowing as a sensor sees them working through the night.",
    stage: "#06090A",
    dark: true,
    scene: {
      tokens: {
        "--anatomy-card": "#06090A",
        "--anatomy-top": "#0B1110",
        "--anatomy-shade-3": "#0A0F0E",
        "--anatomy-shade-2": "#090D0C",
        "--anatomy-shade-1": "#080B0B",
        "--anatomy-shade-0": "#070A09",
        "--anatomy-lit-top": "#8BFFC2",
        "--anatomy-lit-shade": "#4FD68D",
        "--anatomy-hi": "#5C8E77",
        "--anatomy-mid": "#3F6253",
        "--anatomy-lo": "#2A4238",
        "--anatomy-faint": "#14201B",
        "--anatomy-lit": "#8BFFC2",
      },
      flat: { water: "#040607", road: "#0E1513" },
      edgeAlpha: 0.95,
      post: { vignette: 0.45, grain: 0.03 },
    },
  },
  {
    id: "scan",
    name: "Scan points",
    blurb: "The land as the scan sees it: everything drawn as measured points, robots in amber.",
    stage: "#070B0F",
    dark: true,
    scene: {
      tokens: {
        "--anatomy-card": "#070B0F",
        "--anatomy-top": "#9FD8FF",
        "--anatomy-shade-3": "#8CBFE0",
        "--anatomy-shade-2": "#71A3C4",
        "--anatomy-shade-1": "#5687A8",
        "--anatomy-shade-0": "#3E6E8F",
        "--anatomy-lit-top": "#FFB347",
        "--anatomy-lit-shade": "#E0902A",
        "--anatomy-hi": "#6FA8D0",
        "--anatomy-mid": "#3E6E8F",
        "--anatomy-lo": "#284B63",
        "--anatomy-faint": "#1A3245",
        "--anatomy-lit": "#FFB347",
      },
      terrain: { top: "#4E86AB", side: "#2E5875" },
      flat: { water: "#1D3A50", road: "#4E86AB" },
      stipple: 0.22,
      edges: false,
      post: { vignette: 0.45, grain: 0.02 },
    },
  },
];

/** The look a page asks for (`?look=`), or the default. */
export function lookById(id: string | null): Look {
  return LOOKS.find((look) => look.id === id) ?? LINE;
}
