/**
 * Live mode (`view.html#live/<captureId>`): a reconstruction as it happens.
 *
 * Polls `GET /captures/{id}/live` every few seconds and draws what the run has so far:
 * while the pose stage solves, each registered camera as a small frustum and a sample of
 * the sparse points in their own colours; while training runs, each intermediate splat
 * as it arrives, faded in over the last one, among the solved cameras. The person's view
 * is never reset by an update -- only the first thing drawn frames the camera.
 *
 * Everything the run shows here is in the pose solve's own frame (COLMAP's), not yet
 * placed east/north/up: it is turned so the solve's up estimate faces the sky, and centred
 * on the points. When the run finishes, the status panel links to the final scan.
 */
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { LiveCameras, LiveState } from "@twin/contracts";

import { deviceSplatBudget } from "@/lib/detail";

import {
  decodeCameras,
  decodePoints,
  describeLive,
  medianCentre,
  upOf,
  type DecodedCameras,
} from "./liveData";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const POLL_MS = 3_000;
const FADE_MS = 900;
const CAMERA_COLOUR = 0x7fd8c0;

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`#${id} is missing from view.html`);
  return found;
}

/** Frustum lines for every camera: apex to the four corners, and the far rectangle. */
function frustums(cameras: DecodedCameras, size: number, aspect: number): THREE.BufferGeometry {
  const positions = new Float32Array(cameras.count * 16 * 3);
  const depth = size;
  const halfH = size * 0.35;
  const halfW = halfH * (aspect > 0 ? aspect : 4 / 3);
  const c = new THREE.Vector3();
  const f = new THREE.Vector3();
  const u = new THREE.Vector3();
  const r = new THREE.Vector3();
  let at = 0;
  const push = (v: THREE.Vector3): void => {
    positions[at] = v.x;
    positions[at + 1] = v.y;
    positions[at + 2] = v.z;
    at += 3;
  };
  for (let i = 0; i < cameras.count; i += 1) {
    c.fromArray(cameras.centres, i * 3);
    f.fromArray(cameras.forward, i * 3);
    u.fromArray(cameras.up, i * 3);
    // COLMAP's camera x is right, y down, z forward: right = forward x up.
    r.crossVectors(f, u).normalize();
    const corner = (sx: number, sy: number): THREE.Vector3 =>
      c
        .clone()
        .addScaledVector(f, depth)
        .addScaledVector(r, sx * halfW)
        .addScaledVector(u, sy * halfH);
    const corners = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)];
    for (const corner of corners) {
      push(c);
      push(corner);
    }
    for (let k = 0; k < 4; k += 1) {
      push(corners[k] ?? c);
      push(corners[(k + 1) % 4] ?? c);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return geometry;
}

function srgbToLinear(values: Float32Array): Float32Array {
  return values.map((c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
}

/** Median spacing between consecutive cameras: how big a frustum should be. */
function spacing(cameras: DecodedCameras, fallback: number): number {
  const gaps: number[] = [];
  for (let i = 1; i < cameras.count; i += 1) {
    const dx = (cameras.centres[i * 3] ?? 0) - (cameras.centres[i * 3 - 3] ?? 0);
    const dy = (cameras.centres[i * 3 + 1] ?? 0) - (cameras.centres[i * 3 - 2] ?? 0);
    const dz = (cameras.centres[i * 3 + 2] ?? 0) - (cameras.centres[i * 3 - 1] ?? 0);
    gaps.push(Math.hypot(dx, dy, dz));
  }
  gaps.sort((a, b) => a - b);
  const median = gaps[Math.floor(gaps.length / 2)] ?? 0;
  return median > 0 ? median : fallback;
}

export function showLive(captureId: string): { stop: () => void } {
  el("gallery").hidden = true;
  el("viewer").hidden = true;
  const section = el("live");
  section.hidden = false;
  section.dataset.cameras = "0";
  section.dataset.splatStep = "";
  const status = el("live-status");
  const detail = el("live-detail");
  const bar = el("live-bar");
  const track = el("live-track");
  const finalLink = el("live-final") as HTMLAnchorElement;
  finalLink.hidden = true;
  status.textContent = "Connecting…";
  detail.textContent = "";
  el("live-name").textContent = "Live";
  document.title = "Live";

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
  // This device's splat budget is Spark's here too (see main.ts).
  const budget = deviceSplatBudget();
  scene.add(new SparkRenderer({ renderer, lodSplatCount: budget }));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  let touched = false;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
    touched = true;
  });

  // The solve's frame, turned so its up is three.js's +y and centred on the points.
  const frame = new THREE.Group();
  scene.add(frame);
  let cameraLines: THREE.LineSegments | null = null;
  let cloud: THREE.Points | null = null;
  let splat: { mesh: SplatMesh; name: string } | null = null;
  let loadingSplat: string | null = null;
  let framed = false;
  let lastSeq = -1;
  let camerasFinal = false;

  /**
   * A mid-solve snapshot is in the solve's frame of that moment; a splat is trained in the
   * finished model's, which COLMAP has re-centred and re-scaled since. Drawn together they
   * show the scene twice, at two sizes. So once a splat is up, only finished cameras stay.
   */
  const showOverlay = (): void => {
    const visible = splat === null || camerasFinal;
    if (cameraLines) cameraLines.visible = visible;
    if (cloud) cloud.visible = visible;
  };
  let stopped = false;
  let timer = 0;
  const fades: { mesh: SplatMesh; from: number; to: number; start: number; done?: () => void }[] =
    [];

  const resize = (): void => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener("resize", resize);
  renderer.setAnimationLoop((time: number) => {
    for (let i = fades.length - 1; i >= 0; i -= 1) {
      const fade = fades[i];
      if (!fade) continue;
      const t = Math.min(1, (time - fade.start) / FADE_MS);
      fade.mesh.opacity = fade.from + (fade.to - fade.from) * t;
      if (t >= 1) {
        fades.splice(i, 1);
        fade.done?.();
      }
    }
    controls.update();
    renderer.render(scene, camera);
  });

  /** Turn the solve's frame so `up` faces the sky, centred on `centre`. */
  const orient = (up: [number, number, number], centre: [number, number, number]): void => {
    frame.quaternion.setFromUnitVectors(new THREE.Vector3(...up), new THREE.Vector3(0, 1, 0));
    frame.position.copy(new THREE.Vector3(...centre).applyQuaternion(frame.quaternion).negate());
    frame.updateMatrixWorld(true);
  };

  /**
   * Fit the view to a sphere of `radius` around the origin. Again whenever the scene has
   * grown or shrunk by half since the last fit -- the solve starts with a handful of
   * cameras -- until the user takes the camera, after which it is theirs.
   */
  let framedRadius = 0;
  const frameView = (radius: number): void => {
    if (touched) return;
    if (framed && radius < framedRadius * 1.5 && radius > framedRadius / 1.5) return;
    framed = true;
    framedRadius = radius;
    const size = Math.max(radius, 1e-3);
    const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    const distance = (size / Math.tan(Math.min(halfV, halfH))) * 1.1;
    controls.target.set(0, 0, 0);
    camera.position.copy(new THREE.Vector3(0, 0.5, 1).normalize().multiplyScalar(distance));
    camera.near = size / 1000;
    camera.far = size * 200;
    camera.updateProjectionMatrix();
    controls.minDistance = size * 0.02;
    controls.maxDistance = distance * 6;
  };

  /** The 90th-percentile distance of these points from `centre`: robust to strays. */
  const spreadRadius = (sets: Float32Array[], centre: readonly number[]): number | null => {
    const distances: number[] = [];
    for (const set of sets) {
      for (let i = 0; i + 2 < set.length; i += 3) {
        const dx = (set[i] ?? 0) - (centre[0] ?? 0);
        const dy = (set[i + 1] ?? 0) - (centre[1] ?? 0);
        const dz = (set[i + 2] ?? 0) - (centre[2] ?? 0);
        distances.push(Math.hypot(dx, dy, dz));
      }
    }
    if (distances.length < 3) return null;
    distances.sort((a, b) => a - b);
    return distances[Math.floor(distances.length * 0.9)] ?? null;
  };

  const drawCameras = (payload: LiveCameras, dimPoints: boolean): void => {
    const decoded = decodeCameras(payload);
    const points = decodePoints(payload);
    const centre = medianCentre(points.count ? points.positions : decoded.centres) ?? [
      payload.origin[0] ?? 0,
      payload.origin[1] ?? 0,
      payload.origin[2] ?? 0,
    ];
    // Until a splat is on screen the frame follows the solve, which re-normalises itself
    // as it grows; once one is, the frame holds still so the splat does not jump.
    if (!splat) orient(upOf({ cameras: payload, splat: null }), centre);
    cameraLines?.geometry.dispose();
    if (cameraLines) frame.remove(cameraLines);
    const size = spacing(decoded, payload.scale * 0.05) * 0.6;
    cameraLines = new THREE.LineSegments(
      frustums(decoded, size, payload.aspect ?? 4 / 3),
      new THREE.LineBasicMaterial({ color: CAMERA_COLOUR, transparent: true, opacity: 0.9 }),
    );
    frame.add(cameraLines);
    cloud?.geometry.dispose();
    if (cloud) frame.remove(cloud);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(srgbToLinear(points.colors), 3));
    cloud = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 2.5,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: dimPoints ? 0.25 : 0.95,
      }),
    );
    frame.add(cloud);
    frame.updateMatrixWorld(true);
    section.dataset.cameras = String(decoded.count);
    section.dataset.points = String(points.count);
    // The solve's own spread, not `payload.scale` (the quantisation range, far larger).
    frameView(spreadRadius([points.positions, decoded.centres], centre) ?? payload.scale * 0.8);
    camerasFinal = payload.final;
    showOverlay();
  };

  const showSplat = async (
    name: string,
    step: number,
    url: string,
    up: [number, number, number],
  ): Promise<void> => {
    loadingSplat = name;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`The splat answered ${String(response.status)}.`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    // A snapshot within the Detail budget is drawn whole, as it arrives; only a bigger one
    // waits for Spark to build its LoD tree (`lodAbove`), and is then drawn within it.
    const mesh = new SplatMesh({
      fileBytes: bytes,
      fileType: SplatFileType.SPZ,
      lod: true,
      lodAbove: budget,
    });
    await mesh.initialized;
    if (stopped) {
      mesh.dispose();
      return;
    }
    if (!cameraLines) {
      // No cameras to take the frame from: the splat's own centre and the pose up. A LoD
      // mesh keeps only its LoD tree, so the box is that tree's (its leaves are the splats).
      const lodSplats = mesh.packedSplats?.lodSplats;
      const box = lodSplats ? new THREE.Box3() : mesh.getBoundingBox(true);
      lodSplats?.forEachSplat((_index, centre) => box.expandByPoint(centre));
      const centre = box.getCenter(new THREE.Vector3());
      orient(up, [centre.x, centre.y, centre.z]);
      frameView(box.getSize(new THREE.Vector3()).length() / 2);
    }
    mesh.opacity = 0;
    frame.add(mesh);
    const now = performance.now();
    fades.push({ mesh, from: 0, to: 1, start: now });
    const previous = splat;
    if (previous) {
      fades.push({
        mesh: previous.mesh,
        from: previous.mesh.opacity,
        to: 0,
        start: now,
        done: () => {
          frame.remove(previous.mesh);
          previous.mesh.dispose();
        },
      });
    }
    splat = { mesh, name };
    if (cloud) (cloud.material as THREE.PointsMaterial).opacity = 0.25;
    showOverlay();
    section.dataset.splat = name;
    section.dataset.splatStep = String(step);
  };

  const render = (state: LiveState): boolean => {
    el("live-name").textContent = state.captureName;
    document.title = `Live · ${state.captureName}`;
    const described = describeLive(state);
    status.textContent = described.headline;
    detail.textContent = described.detail;
    track.hidden = described.fraction === null;
    bar.style.width = `${String(Math.round((described.fraction ?? 0) * 100))}%`;
    track.setAttribute("aria-valuenow", String(Math.round((described.fraction ?? 0) * 100)));
    if (state.cameras && state.cameras.seq !== lastSeq) {
      lastSeq = state.cameras.seq;
      drawCameras(state.cameras, splat !== null);
    }
    const next = state.splat;
    if (next?.url && next.name !== splat?.name && next.name !== loadingSplat) {
      void showSplat(next.name, next.step, next.url, upOf(state)).catch(() => {
        // The next poll signs a fresh URL and tries again.
        loadingSplat = null;
      });
    }
    if (described.ended && state.siteId) {
      finalLink.href = `#${state.siteId}`;
      finalLink.hidden = false;
    }
    section.dataset.status = state.status;
    return described.ended;
  };

  const tick = async (): Promise<void> => {
    window.clearTimeout(timer);
    if (stopped) return;
    let ended = false;
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/captures/${encodeURIComponent(captureId)}/live`,
        { cache: "no-store" },
      );
      if (response.status === 404) {
        status.textContent = "Nothing to watch yet";
        detail.textContent = "This capture has no run. Start one from the phone page.";
      } else if (response.ok) {
        ended = render((await response.json()) as LiveState);
      }
    } catch {
      // A dropped request on a phone is normal; the next tick asks again.
    }
    if (!stopped && !ended) timer = window.setTimeout(() => void tick(), POLL_MS);
  };
  const onVisible = (): void => {
    if (document.visibilityState === "visible") void tick();
  };
  document.addEventListener("visibilitychange", onVisible);
  void tick();

  return {
    stop: () => {
      stopped = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("resize", resize);
      renderer.setAnimationLoop(null);
      controls.dispose();
      splat?.mesh.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      section.hidden = true;
    },
  };
}
