import { Cartesian2, Cartesian3, Cartographic, SceneTransforms } from "cesium";

import type { LandPoint } from "@/state/land";

import type { CesiumSceneManager } from "./CesiumSceneManager";

export interface ProjectedPoint {
  x: number;
  y: number;
  visible: boolean;
}

/** Scene samples per frame: each is a render pass, so a burst of new pins is spread out. */
const SAMPLES_PER_FRAME = 3;
/** Where the scene has no surface to sample (nothing loaded there), stop asking after this. */
const MAX_ATTEMPTS = 12;
/** Below the Dead Sea shore or above Everest, a sample is not the ground. */
const LOWEST_GROUND_M = -500;
const HIGHEST_GROUND_M = 9_000;
/** Drawn terrain and 3D tiles agree far closer than this; a sample further off is not the ground. */
const MAX_TERRAIN_GAP_M = 1_000;

const key = (point: LandPoint) => `${point[0].toFixed(7)},${point[1].toFixed(7)}`;

/**
 * Places land pins and leader ends on the drawn surface. Photorealistic tiles hide the globe,
 * so a point at the globe's height floats off its spot as the camera tilts; each point is
 * sampled once from the scene while the camera rests, and the globe stands in until then.
 */
export class LandAnchorProjector {
  private readonly heights = new Map<
    string,
    { height: number; sampled: boolean; attempts: number }
  >();
  private readonly scratchWindow = new Cartesian2();
  private readonly scratchNormal = new Cartesian3();
  private readonly scratchToCamera = new Cartesian3();
  private samplesLeft = SAMPLES_PER_FRAME;
  private readonly removeFrame: () => void;
  private readonly listeners = new Set<() => void>();

  constructor(private readonly host: CesiumSceneManager) {
    this.removeFrame = host.scene.postRender.addEventListener(() => {
      this.samplesLeft = SAMPLES_PER_FRAME;
      this.listeners.forEach((listener) => listener());
    });
  }

  /** Runs after every rendered frame; returns the unsubscribe. */
  onFrame(listener: () => void): () => void {
    this.listeners.add(listener);
    this.requestFrame();
    return () => this.listeners.delete(listener);
  }

  requestFrame(): void {
    if (!this.host.isDestroyed) this.host.scene.requestRender();
  }

  /** Window position of a ground point, in CSS pixels from the canvas's top left. */
  project(point: LandPoint, lift = 0): ProjectedPoint {
    const height = this.heightAt(point) + lift;
    const world = Cartesian3.fromDegrees(point[0], point[1], height);
    if (!this.facesCamera(world)) return { x: -9999, y: -9999, visible: false };
    const window = SceneTransforms.worldToWindowCoordinates(
      this.host.scene,
      world,
      this.scratchWindow,
    );
    if (!window) return { x: -9999, y: -9999, visible: false };
    return { x: window.x, y: window.y, visible: true };
  }

  private heightAt(point: LandPoint): number {
    const id = key(point);
    const cached = this.heights.get(id);
    if (cached?.sampled) return cached.height;
    const { scene, viewer } = this.host;
    const carto = Cartographic.fromDegrees(point[0], point[1]);
    let attempts = cached?.attempts ?? 0;
    const resting = !this.host.camera.isMoving;
    if (scene.sampleHeightSupported && resting && this.samplesLeft > 0) {
      this.samplesLeft -= 1;
      attempts += 1;
      try {
        const sampled = scene.sampleHeight(carto, viewer.entities.values);
        const terrain = scene.globe.getHeight(carto);
        // A sample read back before the surface there was drawn can land far underground,
        // which puts the pin between the screen's centre and its spot; such a reading is
        // retried rather than kept.
        if (
          sampled !== undefined &&
          Number.isFinite(sampled) &&
          sampled > LOWEST_GROUND_M &&
          sampled < HIGHEST_GROUND_M &&
          (terrain === undefined || Math.abs(sampled - terrain) < MAX_TERRAIN_GAP_M)
        ) {
          this.heights.set(id, { height: sampled, sampled: true, attempts });
          return sampled;
        }
      } catch {
        // No surface drawn there yet; the globe or the last estimate stands in.
      }
    }
    const height = scene.globe.getHeight(carto) ?? cached?.height ?? 0;
    const settled = attempts >= MAX_ATTEMPTS || !scene.sampleHeightSupported;
    this.heights.set(id, { height, sampled: settled, attempts });
    // Not sampled yet: another frame takes the sample once the camera rests.
    if (!settled && resting) this.requestFrame();
    return height;
  }

  /** False when the point is behind the planet's limb. */
  private facesCamera(point: Cartesian3): boolean {
    const camera = this.host.viewer.camera.positionWC;
    const normal = Cartesian3.normalize(point, this.scratchNormal);
    const toCamera = Cartesian3.normalize(
      Cartesian3.subtract(camera, point, this.scratchToCamera),
      this.scratchToCamera,
    );
    return Cartesian3.dot(normal, toCamera) > -0.05;
  }

  destroy(): void {
    this.removeFrame();
    this.listeners.clear();
  }
}
