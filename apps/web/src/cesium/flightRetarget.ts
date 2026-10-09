/**
 * The poses a fly-to arrives at, and when two of them are the same arrival.
 *
 * A fly-to leaves on the click, towards the best pose known at that moment (the catalog
 * summary's centre), while the site's details and model load (SiteManager.flyTo); better
 * poses arrive on the way and move the flight's destination (glide.ts). Re-pointing used to be
 * done here, by replacing a Cesium flight with a new leg whose easing left at the speed the
 * camera had: the speed was continuous, but every leg was a new height curve and a new pitch
 * curve, and the camera lurched at each one. What is left is what both ways share.
 */

/** A camera pose to arrive at, in degrees and metres. */
export interface ArrivalPose {
  longitude: number;
  latitude: number;
  height: number;
  heading: number;
  pitch: number;
}

/**
 * Whether two arrival poses are close enough that steering from one to the other is not
 * worth doing: within `toleranceM` of each other and within 2° of heading and pitch.
 */
export function samePose(
  a: ArrivalPose,
  b: ArrivalPose,
  separationM: number,
  toleranceM: number,
): boolean {
  const turn = Math.abs(((a.heading - b.heading + 540) % 360) - 180);
  return separationM <= toleranceM && turn <= 2 && Math.abs(a.pitch - b.pitch) <= 2;
}
