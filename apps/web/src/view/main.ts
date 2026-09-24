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
 * `view.html` lists every scan; `view.html#<siteId>` opens one. Reads are open in this
 * API, so the page needs no key.
 */
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { Capture, Site, SiteSummary } from "@twin/contracts";

import { spzFromGlb } from "./glb";

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

/** The URL of the splat tile inside a site's gaussian-splat tileset. */
async function splatTileUrl(site: Site): Promise<string> {
  const source = site.assets.find(
    (candidate) => candidate.representation === "gaussian-splat",
  )?.source;
  if (source?.type !== "3d-tiles-url") throw new Error("This site has no splat scan to show.");
  const tilesetUrl = source.url;
  const tileset = await json<{ root?: { content?: { uri?: string } } }>(tilesetUrl);
  const uri = tileset.root?.content?.uri;
  if (!uri) throw new Error("The scan's tileset names no content.");
  return new URL(uri, tilesetUrl).toString();
}

/**
 * A box around where most of the splats are. Percentiles rather than min/max because
 * some captures carry a sky dome or stray floaters hundreds of metres out, and framing
 * those would leave the subject a dot in the middle of the screen.
 */
function robustBounds(mesh: SplatMesh): THREE.Box3 {
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  let seen = 0;
  mesh.forEachSplat((_index, center, _scales, _quaternion, opacity) => {
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
  scene.add(new SparkRenderer({ renderer }));
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
  active = {
    stop: () => {
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

    const tile = await fetch(await splatTileUrl(site));
    if (!tile.ok) throw new Error(`The scan's data answered ${String(tile.status)}.`);
    const bytes = spzFromGlb(await tile.arrayBuffer());

    const mesh = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.SPZ });
    // The pipeline's splats are east/north/up with z up (see splat_tiles.py, whose glTF
    // node matrix makes the same turn for Cesium); three.js is y-up.
    mesh.rotation.x = -Math.PI / 2;
    scene.add(mesh);
    await mesh.initialized;
    mesh.updateMatrixWorld(true);

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
    status.textContent = "Drag to turn · pinch to zoom · two fingers to move";
    window.setTimeout(() => {
      status.hidden = true;
    }, 4_000);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "The scan could not be shown.";
  }
}

function route(): void {
  active?.stop();
  active = null;
  const siteId = window.location.hash.replace(/^#/, "");
  if (/^[A-Za-z0-9-]{1,64}$/.test(siteId)) void showScan(siteId);
  else void showGallery();
}

window.addEventListener("hashchange", route);
route();
