/**
 * Who may move the camera: the rule every camera correction asks first.
 *
 * Once the person has touched the camera -- a wheel, a press or drag, a pinch, a key, a walk --
 * nothing moves it but them. A fly-to may still be corrected on the way and settle on a better
 * pose once it lands, and a camera the app put somewhere may be eased off a surface it ended
 * up under, but only until a hand is on the map: from then the camera stays exactly where it
 * was left. The app takes it back only by flying it again (a site picked, a search result,
 * Home), which the person asked for.
 *
 * A floor may still stop the camera, never move it. CesiumJS's terrain collision runs every
 * frame (ScreenSpaceCameraController's `adjustHeightForTerrain`), and also on a camera nobody
 * is moving: when a finer terrain tile lands under a close-up a moment after the wheel stops,
 * it lifts the camera onto the new tile. So collision is on while the person is moving the
 * camera -- the gesture, and its inertia -- where it stops a zoom or a pan at the ground, and
 * off for an idle camera of theirs (`collisionAllowed`). Before anybody touches it, it acts as
 * it always did.
 *
 * Pure: the time is passed in, so it is tested without a viewer (cameraOwnership.test.ts).
 */

/**
 * How long after the last wheel notch, press or key a gesture is still moving the camera (ms):
 * CesiumJS's inertia (zoom 0.8, spin and translate 0.85) has decayed to under a thousandth by
 * then. A controller that is still moving the camera past it counts as moving too.
 */
export const INPUT_SETTLE_MS = 1500;

export class CameraOwnership {
  private user = false;
  private lastInputAt = Number.NEGATIVE_INFINITY;
  private held = false;
  private controllerMoving = false;

  constructor(private readonly settleMs = INPUT_SETTLE_MS) {}

  /** The person moved the camera, or is moving it: a wheel notch, a drag, a key, a pinch. */
  input(now: number): void {
    this.user = true;
    this.lastInputAt = now;
  }

  /** A pointer pressed on (or released from) the map. A press takes the camera. */
  hold(down: boolean, now: number): void {
    this.held = down;
    if (down) this.input(now);
    else this.lastInputAt = Math.max(this.lastInputAt, now);
  }

  /**
   * CesiumJS's controller moved the camera in its last update (the person's gesture or its
   * inertia), or did not: a camera still coasting is still theirs to move.
   */
  controllerMoved(moved: boolean): void {
    this.controllerMoving = moved;
  }

  /** The app flies the camera (a fly-to, a search result, Home): it has it again. */
  appFlew(): void {
    this.user = false;
  }

  /** Whether the person has touched the camera since the app last flew it. */
  get userHasCamera(): boolean {
    return this.user;
  }

  /** Whether the person is moving the camera now: a gesture, or its inertia. */
  moving(now: number): boolean {
    return this.held || this.controllerMoving || now - this.lastInputAt < this.settleMs;
  }

  /**
   * Whether the app may correct the camera now -- a floor lift, a settle on a better pose,
   * surfacing from underground: only one nobody has touched since the app flew it.
   */
  get mayCorrect(): boolean {
    return !this.user;
  }

  /**
   * Whether CesiumJS's terrain collision may act this frame: on the person's own motion, where
   * it stops the camera at the ground, and on a camera they have not touched; never on an idle
   * camera of theirs, which it would lift as finer terrain lands under it.
   */
  collisionAllowed(now: number): boolean {
    return !this.user || this.moving(now);
  }
}
