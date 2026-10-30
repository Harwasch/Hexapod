/**
 * Looks for the landing page's 3D scenes, for Harrison to compare (review of 2026-10-22: "I love
 * the high-level idea, but maybe we need to try some different sort of design feels for it").
 *
 * Every look draws the same world and the same crews (scene.js); a look is only data: colours
 * (the --anatomy-* tokens the renderer reads), the ground's own colours, scan dots,
 * ink outlines on or off, real light with soft shadows, the camera, and a finish (grain,
 * tilt-shift, vignette, tint). Each look has a light and a dark version, and the scenes follow
 * the page's theme between them. `?look=<id>` picks one; `?review` adds a switcher. Once one is
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
  /** How much light the robots give off themselves (a dark look's), 0 to 1. */
  glow?: number;
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

/** One theme's version of a look: what the scenes draw, and the hero's ground behind the copy. */
export interface LookVersion {
  /** The stage colour behind the hero copy and the fleet cards; none keeps the page's own. */
  stage?: string;
  scene: SceneLook;
}

export interface Look {
  id: string;
  name: string;
  /** One line on what it is going for. */
  blurb: string;
  camera?: SceneLook["camera"];
  light: LookVersion;
  dark: LookVersion;
}

/** What the scenes take (scene.d.ts): both versions, picked between by the page's theme. */
export interface ThemedLook {
  light: SceneLook;
  dark: SceneLook;
}

/** The line drawing, smoothed: the page's look until another is chosen. Its colours are the
 * page's own --anatomy-* palette, which already has a light and a dark set. */
const LINE: Look = {
  id: "line",
  name: "Line drawing (fixed)",
  blurb: "The current look, smoothed: more supersampling, GPU anti-aliasing, no bands or jumps.",
  light: { scene: {} },
  dark: { scene: {} },
};

/** A toon palette: the --anatomy-* tokens the line renderer reads, in one call. */
function palette(p: {
  card: string;
  top: string;
  shades: [string, string, string, string];
  litTop: string;
  litShade: string;
  hi: string;
  mid: string;
  lo: string;
  faint: string;
  lit: string;
}): Record<string, string> {
  return {
    "--anatomy-card": p.card,
    "--anatomy-top": p.top,
    "--anatomy-shade-3": p.shades[0],
    "--anatomy-shade-2": p.shades[1],
    "--anatomy-shade-1": p.shades[2],
    "--anatomy-shade-0": p.shades[3],
    "--anatomy-lit-top": p.litTop,
    "--anatomy-lit-shade": p.litShade,
    "--anatomy-hi": p.hi,
    "--anatomy-mid": p.mid,
    "--anatomy-lo": p.lo,
    "--anatomy-faint": p.faint,
    "--anatomy-lit": p.lit,
  };
}

export const LOOKS: Look[] = [
  LINE,
  {
    id: "clay",
    name: "Clay",
    blurb: "One soft material under real light and shadow; the robots in one warm colour.",
    light: {
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
    dark: {
      stage: "#1A1916",
      scene: {
        tokens: { "--anatomy-card": "#1A1916" },
        edges: false,
        lines: false,
        light: {
          sun: 1.8,
          sky: 0.9,
          sunColor: "#A9BBE0",
          skyColor: "#2B3550",
          groundColor: "#15130F",
          az: 55,
          el: 34,
          object: "#77736A",
          robot: "#FF8A4C",
          dark: "#3A3731",
          solar: "#1B2230",
          ground: "#4A463F",
          water: "#1F2E38",
          road: "#3E3A33",
          mulch: "#433A2F",
          glow: 0.8,
        },
        post: { grain: 0.015, vignette: 0.25 },
      },
    },
  },
  {
    id: "golden",
    name: "Golden hour",
    blurb: "Late sun from the side and long shadows by day; the blue hour after it by night.",
    light: {
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
    dark: {
      stage: "#121A2A",
      scene: {
        tokens: { "--anatomy-card": "#121A2A" },
        edges: false,
        lines: false,
        light: {
          sun: 2.4,
          sky: 1.2,
          sunColor: "#FF9A5C",
          skyColor: "#3D5A8C",
          groundColor: "#1B1A22",
          az: 105,
          el: 8,
          object: "#9298AC",
          robot: "#FFE2B0",
          dark: "#2A2C36",
          solar: "#182032",
          ground: "#4C5672",
          water: "#22334D",
          road: "#454B5E",
          mulch: "#433E4A",
          glow: 0.6,
        },
        post: { vignette: 0.35, tint: { color: "#9BB4FF", amount: 0.08 }, grain: 0.015 },
      },
    },
  },
  {
    id: "miniature",
    name: "Miniature",
    blurb: "A model railway: toy colours, a higher camera, tilt-shift blur; lamps lit at night.",
    camera: { zoom: 0.82, tilt: 12 },
    light: {
      stage: "#DCE7CF",
      scene: {
        tokens: { "--anatomy-card": "#DCE7CF" },
        edges: false,
        lines: false,
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
    dark: {
      stage: "#0F1A12",
      scene: {
        tokens: { "--anatomy-card": "#0F1A12" },
        edges: false,
        lines: false,
        light: {
          sun: 1.5,
          sky: 0.85,
          sunColor: "#B8C8FF",
          skyColor: "#1E2A40",
          groundColor: "#0E140E",
          az: 40,
          el: 55,
          object: "#7A7466",
          robot: "#FFC640",
          dark: "#22301F",
          solar: "#16202E",
          ground: "#3E5A31",
          water: "#1A2E40",
          road: "#423F36",
          mulch: "#3A2E20",
          glow: 0.85,
        },
        post: { tilt: 1.0, vignette: 0.3 },
      },
    },
  },
  {
    id: "color",
    name: "Flat colour",
    blurb: "The flat-shaded models in a real palette: green ground, cream objects, orange robots.",
    light: {
      stage: "#EAF0E4",
      scene: {
        tokens: palette({
          card: "#EAF0E4",
          top: "#F2EADA",
          shades: ["#E8DEC9", "#DDD1B8", "#CDBFA3", "#B9A88A"],
          litTop: "#F08A3C",
          litShade: "#D6702A",
          hi: "#3B4A36",
          mid: "#6E7F66",
          lo: "#9FB094",
          faint: "#C5D2BA",
          lit: "#F08A3C",
        }),
        terrain: { top: "#A8C686", side: "#7FA061" },
        flat: { water: "#86B9D8", road: "#E2D7BF", mulch: "#B5946B" },
        edgeAlpha: 0.5,
      },
    },
    dark: {
      stage: "#12180F",
      scene: {
        tokens: palette({
          card: "#12180F",
          top: "#3A3D33",
          shades: ["#34372E", "#2D3028", "#262922", "#20221C"],
          litTop: "#F08A3C",
          litShade: "#C76A28",
          hi: "#C9D6BF",
          mid: "#7E8C74",
          lo: "#4C5844",
          faint: "#2E3829",
          lit: "#F08A3C",
        }),
        terrain: { top: "#2E4424", side: "#22341B" },
        flat: { water: "#1F3A4D", road: "#3A3A30", mulch: "#4A3A28" },
        edgeAlpha: 0.5,
      },
    },
  },
  {
    id: "paper",
    name: "Ink and wash",
    blurb: "A field sketchbook: pen over watercolour by day, chalk on slate by night.",
    light: {
      stage: "#F1E9D8",
      scene: {
        tokens: palette({
          card: "#F1E9D8",
          top: "#F1E9D8",
          shades: ["#E7DCC4", "#DCCDAE", "#CDB993", "#BCA57D"],
          litTop: "#C9D8E6",
          litShade: "#A9BED3",
          hi: "#2A241D",
          mid: "#5E5444",
          lo: "#8E8269",
          faint: "#BFB296",
          lit: "#2A241D",
        }),
        terrain: { top: "#EFE6D2", side: "#D3C29F" },
        flat: { water: "#BCD0D6", road: "#EADFC8" },
        edgeAlpha: 1,
        post: { grain: 0.05, vignette: 0.2, tint: { color: "#F7EBD2", amount: 0.15 } },
      },
    },
    dark: {
      stage: "#1E2324",
      scene: {
        tokens: palette({
          card: "#1E2324",
          top: "#262C2D",
          shades: ["#242A2B", "#212627", "#1E2223", "#1B1F20"],
          litTop: "#3A4A57",
          litShade: "#33424E",
          hi: "#EDEBE4",
          mid: "#A9A79F",
          lo: "#6F6E68",
          faint: "#3E4243",
          lit: "#F4F1E8",
        }),
        terrain: { top: "#262C2D", side: "#1F2425" },
        flat: { water: "#1C2B33", road: "#2B3132" },
        edgeAlpha: 1,
        post: { grain: 0.06, vignette: 0.25 },
      },
    },
  },
  {
    id: "blueprint",
    name: "Blueprint",
    blurb: "An engineering drawing: blue lines on white print by day, white on blue by night.",
    light: {
      stage: "#F4F7FA",
      scene: {
        tokens: palette({
          card: "#F4F7FA",
          top: "#FFFFFF",
          shades: ["#EEF3F8", "#E6EDF4", "#DCE5EF", "#D0DCE8"],
          litTop: "#D6E6FF",
          litShade: "#C2D8F7",
          hi: "#1D4E89",
          mid: "#4E77A8",
          lo: "#8EABCB",
          faint: "#C3D3E6",
          lit: "#0B3D7A",
        }),
        flat: { water: "#E2ECF5", road: "#EDF2F7" },
        edgeAlpha: 0.95,
        post: { grain: 0.02 },
      },
    },
    dark: {
      stage: "#0F2D4A",
      scene: {
        tokens: palette({
          card: "#0F2D4A",
          top: "#133659",
          shades: ["#12324F", "#10304B", "#0E2B44", "#0C253B"],
          litTop: "#1F4D7E",
          litShade: "#1A4472",
          hi: "#D6ECFF",
          mid: "#8FB6D9",
          lo: "#5C86AD",
          faint: "#2E5378",
          lit: "#FFFFFF",
        }),
        flat: { water: "#0B2236", road: "#163A5E" },
        edgeAlpha: 0.95,
        post: { grain: 0.02 },
      },
    },
  },
  {
    id: "sensor",
    name: "Sensor view",
    blurb: "The land as the fleet's sensors see it: quiet ground, every robot lit up.",
    light: {
      stage: "#EEF2EF",
      scene: {
        tokens: palette({
          card: "#EEF2EF",
          top: "#F7FAF8",
          shades: ["#EDF1EE", "#E4EAE6", "#DAE2DD", "#CFD9D3"],
          litTop: "#19C27A",
          litShade: "#12A366",
          hi: "#2F5A48",
          mid: "#6E8C7F",
          lo: "#A3B8AE",
          faint: "#CAD7D0",
          lit: "#0E9F62",
        }),
        flat: { water: "#DCE8E6", road: "#E7ECE9" },
        edgeAlpha: 0.9,
        post: { vignette: 0.15, grain: 0.015 },
      },
    },
    dark: {
      stage: "#06090A",
      scene: {
        tokens: palette({
          card: "#06090A",
          top: "#0B1110",
          shades: ["#0A0F0E", "#090D0C", "#080B0B", "#070A09"],
          litTop: "#8BFFC2",
          litShade: "#4FD68D",
          hi: "#5C8E77",
          mid: "#3F6253",
          lo: "#2A4238",
          faint: "#14201B",
          lit: "#8BFFC2",
        }),
        flat: { water: "#040607", road: "#0E1513" },
        edgeAlpha: 0.95,
        post: { vignette: 0.45, grain: 0.03 },
      },
    },
  },
  {
    id: "scan",
    name: "Scan points",
    blurb: "The land as the scan sees it: everything drawn as measured points, robots in amber.",
    light: {
      stage: "#F3F5F7",
      scene: {
        tokens: palette({
          card: "#F3F5F7",
          top: "#2D5F86",
          shades: ["#3A6C92", "#4F7FA3", "#6690B2", "#7FA2BF"],
          litTop: "#E2731C",
          litShade: "#C55F12",
          hi: "#2D5F86",
          mid: "#6690B2",
          lo: "#A3BCD0",
          faint: "#D3DFE9",
          lit: "#E2731C",
        }),
        terrain: { top: "#5C88AB", side: "#86A6C0" },
        flat: { water: "#B9CCDB", road: "#5C88AB" },
        stipple: 0.22,
        edges: false,
        post: { grain: 0.015 },
      },
    },
    dark: {
      stage: "#070B0F",
      scene: {
        tokens: palette({
          card: "#070B0F",
          top: "#9FD8FF",
          shades: ["#8CBFE0", "#71A3C4", "#5687A8", "#3E6E8F"],
          litTop: "#FFB347",
          litShade: "#E0902A",
          hi: "#6FA8D0",
          mid: "#3E6E8F",
          lo: "#284B63",
          faint: "#1A3245",
          lit: "#FFB347",
        }),
        terrain: { top: "#4E86AB", side: "#2E5875" },
        flat: { water: "#1D3A50", road: "#4E86AB" },
        stipple: 0.22,
        edges: false,
        post: { vignette: 0.45, grain: 0.02 },
      },
    },
  },
];

/** The look a page asks for (`?look=`), or the default. */
export function lookById(id: string | null): Look {
  return LOOKS.find((look) => look.id === id) ?? LINE;
}

/** Both of a look's versions as the scenes take them, each with the look's camera. */
export function themed(look: Look): ThemedLook {
  return {
    light: { ...look.light.scene, camera: look.camera },
    dark: { ...look.dark.scene, camera: look.camera },
  };
}
