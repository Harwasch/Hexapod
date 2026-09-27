/**
 * The scan viewer: one finished capture on its own, orbitable, on a phone.
 *
 * Not the console. The console is a globe, and a scan seen there sits in the world at
 * city scale behind imagery, terrain and panels; this page shows just the splat, framed,
 * with touch controls made for turning an object over. It renders with Spark (three.js,
 * MIT) rather than CesiumJS. Not for size -- measured, this page's script is 1.0 MB
 * gzipped (Spark embeds its WebAssembly sorter) against 1.1 MB for Cesium's main chunk
 * before its workers -- but because Spark is built for exactly this: splat sorting tuned
 * for phones, and orbit controls meant for an object rather than for the Earth.
 *
 * `view.html` lists every scan; `view.html#<siteId>` opens one; `view.html#live/<captureId>`
 * watches a capture's run as it happens (live.ts). Reads are open in this API, so the page
 * needs no key.
 */
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { Capture, Site, SiteSummary } from "@twin/contracts";

import { splatBudget } from "@/lib/detail";

import { parseCoverage } from "./coverage";
import { spzFromGlb } from "./glb";
import { showLive } from "./live";
import {
  LOAD_FACTOR,
  countTiles,
  parseTileset,
  planLoads,
  plannedGaussians,
  type TileNode,
  type TileTree,
} from "./tiles";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`#${id} is missing from view.html`);
  return found;
}

function date(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${String(response.status)}`);
  return (await response.json()) as T;
}

// --- gallery --------------------------------------------------------------------------

async function showGallery(): Promise<void> {
  el("viewer").hidden = true;
  el("gallery").hidden = false;
  document.title = "Scans";
  const status = el("gallery-status");
  const cards = el("cards");
  try {
    // Scans are captures that became a site: what you sent, not the demo sites the
    // console seeds (several of which are on Cesium ion, which this page does not read).
    const [captures, sites] = await Promise.all([
      json<Capture[]>(`${API_BASE}/api/v1/captures?limit=200`),
      json<SiteSummary[]>(`${API_BASE}/api/v1/sites`),
    ]);
    const splatSites = new Map(
      sites
        .filter((site) => site.representations.includes("gaussian-splat"))
        .map((site) => [site.id, site]),
    );
    const seen = new Set<string>();
    const scans = captures
      .filter((capture) => capture.siteId !== null && splatSites.has(capture.siteId))
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .filter((capture) => {
        const id = capture.siteId ?? "";
        if (seen.has(id)) return false;
        seen.add(id);
        return true;
      })
      .map((capture) => ({
        id: capture.siteId ?? "",
        name: capture.name,
        createdAt: capture.createdAt,
        thumbnailUrl: splatSites.get(capture.siteId ?? "")?.thumbnailUrl ?? null,
      }));
    status.textContent = scans.length
      ? `${String(scans.length)} ${scans.length === 1 ? "scan" : "scans"}, newest first. Tap one to view it in 3D.`
      : "No scans yet. Send one from the phone page and it appears here when it's done.";
    cards.replaceChildren(
      ...scans.map((site) => {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.className = "card";
        link.href = `#${site.id}`;
        const picture = site.thumbnailUrl
          ? document.createElement("img")
          : document.createElement("div");
        if (picture instanceof HTMLImageElement) {
          picture.src = site.thumbnailUrl ?? "";
          picture.alt = "";
          picture.loading = "lazy";
        } else {
          picture.className = "blank";
        }
        const meta = document.createElement("span");
        meta.className = "meta";
        const name = document.createElement("strong");
        name.textContent = site.name;
        const when = document.createElement("span");
        when.textContent = date(site.createdAt);
        meta.append(name, when);
        link.append(picture, meta);
        item.append(link);
        return item;
      }),
    );
  } catch {
    status.textContent = "Couldn't reach the server. Check your connection and reload.";
  }
}

// --- viewer ---------------------------------------------------------------------------

let active: { stop: () => void } | null = null;

/** A site's gaussian-splat tileset: where it is, and its tile tree (see tiles.ts). */
async function splatTileset(site: Site): Promise<{ url: string; tree: TileTree }> {
  const source = site.assets.find(
    (candidate) => candidate.representation === "gaussian-splat",
  )?.source;
  if (source?.type !== "3d-tiles-url") throw new Error("This site has no splat scan to show.");
  return { url: source.url, tree: parseTileset(await json<unknown>(source.url)) };
}

/**
 * One tile as a Spark mesh: the SPZ inside the tile's GLB, handed over as it is, with Spark's
 * level of detail on. `lod: true` has Spark build a merged LoD tree over the tile in a worker
 * ("quick" tiny-lod: 1-3 s per million splats, per Spark's lod-getting-started docs; a 100k
 * tile is a fraction of a second) and draw from it within `SparkRenderer.lodSplatCount`,
 * which is shared by every mesh in the scene.
 */
async function tileMesh(tilesetUrl: string, tile: TileNode): Promise<SplatMesh> {
  const response = await fetch(new URL(tile.uri, tilesetUrl).toString());
  if (!response.ok) throw new Error(`The scan's data answered ${String(response.status)}.`);
  const bytes = spzFromGlb(await response.arrayBuffer());
  const mesh = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.SPZ, lod: true });
  await mesh.initialized;
  return mesh;
}

/**
 * A box around where most of the splats are. Percentiles rather than min/max because
 * some captures carry a sky dome or stray floaters hundreds of metres out, and framing
 * those would leave the subject a dot in the middle of the screen. A LoD mesh keeps only its
 * LoD tree (`packedSplats.lodSplats`), whose leaves are the tile's own splats and whose inner
 * nodes are merged from them, so the percentiles are read from that.
 */
function robustBounds(mesh: SplatMesh): THREE.Box3 {
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  let seen = 0;
  const splats = mesh.packedSplats?.lodSplats ?? mesh;
  splats.forEachSplat((_index, center, _scales, _quaternion, opacity) => {
    seen += 1;
    // Every splat up to 50k, then a stride: plenty for a percentile, cheap on a phone.
    if (opacity < 0.1 || (seen > 50_000 && seen % 4 !== 0)) return;
    xs.push(center.x);
    ys.push(center.y);
    zs.push(center.z);
  });
  if (xs.length === 0) return mesh.getBoundingBox(true);
  const pick = (values: number[], p: number): number => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] ?? 0;
  };
  return new THREE.Box3(
    new THREE.Vector3(pick(xs, 0.03), pick(ys, 0.03), pick(zs, 0.03)),
    new THREE.Vector3(pick(xs, 0.97), pick(ys, 0.97), pick(zs, 0.97)),
  );
}

async function showScan(siteId: string): Promise<void> {
  el("gallery").hidden = true;
  const section = el("viewer");
  section.hidden = false;
  const status = el("viewer-status");
  status.hidden = false;
  status.textContent = "Loading the scan…";
  el("scan-name").textContent = "";
  el("scan-date").textContent = "";
  el("coverage").hidden = true;
  el("coverage-legend").hidden = true;
  delete section.dataset.tiles;

  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x0b0c0a);
  section.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.01,
    5000,
  );
  // The phone's Detail choice is Spark's splat budget: its LoD draws at most this many a
  // frame, the merged coarse splats far away and the originals close up, across every tile
  // loaded (lib/detail.ts; SparkRenderer `lodSplatCount`, whose own default -- 1-1.5M on a
  // phone -- would ignore the choice).
  const budget = splatBudget();
  scene.add(new SparkRenderer({ renderer, lodSplatCount: budget }));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;
  // The first touch stops the slow spin for good: the person is looking now.
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
  });

  const resize = (): void => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener("resize", resize);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
  // Tiles keep arriving after the first one is on screen; leaving the scan stops them.
  let alive = true;
  active = {
    stop: () => {
      alive = false;
      window.removeEventListener("resize", resize);
      renderer.setAnimationLoop(null);
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };

  try {
    const site = await json<Site>(`${API_BASE}/api/v1/sites/${encodeURIComponent(siteId)}`);
    el("scan-name").textContent = site.name;
    el("scan-date").textContent = date(site.createdAt);
    document.title = site.name;

    // A scan is a level-of-detail tileset holding every gaussian. This phone downloads all
    // of it, or the finest cut of it that fits LOAD_FACTOR times its Detail budget
    // (tiles.ts), and Spark's LoD decides what of that to draw each frame. The root comes
    // first and is shown on its own -- the whole scan, merged coarse -- and each later step
    // swaps a tile for its children once all of them have arrived.
    const tileset = await splatTileset(site);
    const steps = planLoads(tileset.tree, budget * LOAD_FACTOR);
    const scan = new THREE.Group();
    // The pipeline's splats are east/north/up with z up (see splat_tiles.py, whose glTF
    // node matrix makes the same turn for Cesium); three.js is y-up.
    scan.rotation.x = -Math.PI / 2;
    scene.add(scan);
    const [first, ...rest] = steps;
    const root = first?.add[0];
    if (!root) throw new Error("The scan's tileset names no content.");
    const mesh = await tileMesh(tileset.url, root);
    const shown = new Map<TileNode, SplatMesh>([[root, mesh]]);
    scan.add(mesh);
    scan.updateMatrixWorld(true);
    section.dataset.tiles = root.uri;

    const box = robustBounds(mesh).applyMatrix4(mesh.matrixWorld);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    // Far enough back that the whole box fits the *narrower* of the two fields of view:
    // on a phone held upright that is the horizontal one, at about half the vertical.
    const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    // The diagonal overstates what the box projects to, so 0.8 of the fit, not all of it.
    const distance = (size / 2 / Math.tan(Math.min(halfV, halfH))) * 0.8;
    controls.target.copy(center);
    const lookFrom = new THREE.Vector3(0, 0.45, 1).normalize().multiplyScalar(distance);
    camera.position.copy(center).add(lookFrom);
    camera.near = size / 1000;
    camera.far = size * 100;
    camera.updateProjectionMatrix();
    controls.minDistance = size * 0.05;
    controls.maxDistance = distance * 4;
    offerCoverage(site, scene, scan, size);
    status.textContent = "Drag to turn · pinch to zoom · two fingers to move";
    window.setTimeout(() => {
      status.hidden = true;
    }, 4_000);

    // One tile at a time: a tile is 1-2 MB, and a phone's connection and decoder are better
    // spent finishing one than starting four. A step's tiles go on screen together, and
    // only then does the tile they replace come off.
    for (const step of rest) {
      const arrived: [TileNode, SplatMesh][] = [];
      for (const tile of step.add) {
        if (!alive) break;
        arrived.push([tile, await tileMesh(tileset.url, tile)]);
      }
      if (!alive) {
        arrived.forEach(([, more]) => more.dispose());
        return;
      }
      for (const [tile, more] of arrived) {
        more.opacity = mesh.opacity;
        scan.add(more);
        shown.set(tile, more);
      }
      for (const tile of step.remove) {
        const gone = shown.get(tile);
        if (!gone) continue;
        scan.remove(gone);
        gone.dispose();
        shown.delete(tile);
      }
      // What is on screen, for the page's tests: a REPLACE parent is gone once its children
      // are up.
      section.dataset.tiles = [...shown.keys()].map((tile) => tile.uri).join(" ");
    }
    const loaded = plannedGaussians(steps);
    const whole = countTiles(tileset.tree).gaussians;
    if (whole > loaded) {
      el("scan-date").textContent =
        `${date(site.createdAt)} · ${loaded.toLocaleString("en-US")} of ` +
        `${whole.toLocaleString("en-US")} splats loaded · more with Detail on the phone page`;
    }
  } catch (error) {
    // Also after the first tile is up: a later one failing leaves what arrived on screen.
    status.hidden = false;
    status.textContent = error instanceof Error ? error.message : "The scan could not be shown.";
  }
}

/**
 * The Coverage toggle: the quality bar's tier-coloured cloud and the camera path, laid
 * over the scan. Offered only when the run that made this site measured one (its URL is
 * on the site's metadata, set by the worker's registration). Fetched on the first tap,
 * not before: it is up to a couple of megabytes a phone need not spend unasked.
 */
function offerCoverage(site: Site, scene: THREE.Scene, scan: THREE.Group, size: number): void {
  // Read defensively: a site registered before the quality bar has no such key, and an
  // older API may send no metadata at all.
  const metadata = site.metadata as Record<string, unknown> | null | undefined;
  const url = metadata?.coverageUrl;
  if (typeof url !== "string" || !url) return;
  // How much of the high-quality tier frames held back from training confirmed (the
  // pipeline's keepVerifiedPct); absent when accuracy was not measured.
  const verified = metadata?.keepVerifiedPct;
  const button = el("coverage");
  const legend = el("coverage-legend");
  button.hidden = false;
  button.setAttribute("aria-pressed", "false");
  let overlay: THREE.Group | null = null;
  let loading = false;

  const show = (on: boolean): void => {
    button.setAttribute("aria-pressed", String(on));
    legend.hidden = !on;
    if (overlay) overlay.visible = on;
    // The splat stays, faded, so each point can be told apart from what it describes.
    // Every tile's mesh: tiles still arriving copy the first one's opacity.
    for (const child of scan.children) {
      if (child instanceof SplatMesh) child.opacity = on ? 0.25 : 1;
    }
  };

  button.onclick = () => {
    const on = button.getAttribute("aria-pressed") !== "true";
    if (!on || overlay) {
      show(on);
      return;
    }
    if (loading) return;
    loading = true;
    button.textContent = "Coverage…";
    void fetch(url)
      .then(async (response) => {
        if (!response.ok) throw new Error(`The coverage answered ${String(response.status)}.`);
        const coverage = parseCoverage(await response.arrayBuffer());
        overlay = coverageOverlay(coverage, size);
        scene.add(overlay);
        el("coverage-counts").textContent =
          `${coverage.counts.keep.toLocaleString("en-US")} kept · ` +
          `${coverage.counts.context.toLocaleString("en-US")} context · ` +
          `${coverage.counts.drop.toLocaleString("en-US")} dropped (a sample)` +
          (typeof verified === "number"
            ? ` · ${String(Math.round(verified))}% of the scene verified by held-out frames`
            : "");
        show(true);
      })
      .catch((error: unknown) => {
        const status = el("viewer-status");
        status.hidden = false;
        status.textContent =
          error instanceof Error ? error.message : "The coverage could not be shown.";
      })
      .finally(() => {
        loading = false;
        button.textContent = "Coverage";
      });
  };
}

/** Points in their tier colours, and the camera path as a line, turned z-up like the mesh. */
/** The camera path: white, so it can't be mistaken for any tier's colour. */
const CAMERA_PATH = 0xffffff;

function coverageOverlay(coverage: ReturnType<typeof parseCoverage>, size: number): THREE.Group {
  const group = new THREE.Group();
  group.rotation.x = -Math.PI / 2;
  const cloud = new THREE.BufferGeometry();
  cloud.setAttribute("position", new THREE.BufferAttribute(coverage.positions, 3));
  // The file's colours are sRGB bytes; three.js treats vertex colours as linear and encodes
  // on output, which washed the tier colours out (keep read as the camera path's mint).
  const linear = coverage.colors.map((c) =>
    c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4),
  );
  cloud.setAttribute("color", new THREE.BufferAttribute(linear, 3));
  // Drawn over the splat rather than hidden in it: this is an x-ray of the scan's support.
  const points = new THREE.Points(
    cloud,
    new THREE.PointsMaterial({
      // Small and slightly see-through, so a dense surface reads as a tinted surface
      // instead of a solid block, and the scan stays legible under it.
      size: 1.5,
      sizeAttenuation: false,
      vertexColors: true,
      depthTest: false,
      transparent: true,
      opacity: 0.75,
    }),
  );
  points.renderOrder = 1;
  group.add(points);
  if (coverage.cameraPath.length >= 6) {
    const path = new THREE.BufferGeometry();
    path.setAttribute("position", new THREE.BufferAttribute(coverage.cameraPath, 3));
    const line = new THREE.Line(
      path,
      new THREE.LineBasicMaterial({ color: CAMERA_PATH, depthTest: false, transparent: true }),
    );
    line.renderOrder = 2;
    const cameras = new THREE.Points(
      path,
      new THREE.PointsMaterial({
        color: CAMERA_PATH,
        size: Math.max(size / 150, 1e-3),
        depthTest: false,
        transparent: true,
      }),
    );
    cameras.renderOrder = 2;
    group.add(line, cameras);
  }
  group.updateMatrixWorld(true);
  return group;
}

function route(): void {
  active?.stop();
  active = null;
  const siteId = window.location.hash.replace(/^#/, "");
  // `#live/<captureId>`: a run as it happens (live.ts).
  const live = /^live\/([A-Za-z0-9-]{1,64})$/.exec(siteId);
  if (live?.[1]) active = showLive(live[1]);
  else if (/^[A-Za-z0-9-]{1,64}$/.test(siteId)) void showScan(siteId);
  else void showGallery();
}

window.addEventListener("hashchange", route);
route();
