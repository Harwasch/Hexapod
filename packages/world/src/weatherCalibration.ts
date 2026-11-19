/** Cardona & Dabiri (2022), DOI 10.1017/flo.2021.15; supporting MAT data at
 * https://purl.stanford.edu/mg505cc6103 . Standard deviation over 900 frames.
 * Pixel-to-metre transfer uses approximate 5 m height / 265 +/-10 source pixels,
 * divided by cos(50deg) for the reported wind/image-plane angle. Allow 20% metric
 * uncertainty; Minnetonka species/architecture transfer is additionally unverified.
 */
export const MAGNOLIA_SWAY = [
  [8.27, 3.5140446],
  [9.17, 3.2954653],
  [8.85, 3.6494],
  [12.9, 4.19906],
  [13.14, 5.4096472],
  [7.937, 1.8240948],
  [6.605, 1.3041656],
  [9.05, 3.3781379],
  [7.409, 1.874277],
  [10.9, 3.9961049],
  [15.79, 5.343027],
  [13.51, 6.0599816],
  [15.28, 7.2785002],
  [14.28, 6.8690824],
  [11.02, 3.5158282],
  [12.05, 6.8201714],
  [14.28, 6.306992],
  [11.18, 4.5286678],
  [10.2, 3.7898855],
  [5.322, 1.8420711],
] as const;
/** Published OLS form U = slope * sqrt(sigma pixels). */
export function fitMagnoliaSway() {
  const slope =
    MAGNOLIA_SWAY.reduce((sum, [u, s]) => sum + u * Math.sqrt(s), 0) /
    MAGNOLIA_SWAY.reduce((sum, [, s]) => sum + s, 0);
  const metresPerPixel = 5 / 265 / Math.cos((50 * Math.PI) / 180);
  return { slope, metresPerPixel, sigmaMetresPerMpsSquared: metresPerPixel / slope ** 2 };
}

/** Field-constrained target under our simulated gust field, not observed hourly TI.
 * For w=.6 sin(a)+.4 sin(b): E[w²]=.26, E[w⁴]=.1446, Var(w²)=.077.
 * Hence SD[(U+(G-U)w)²] = sqrt(1.04 U²(G-U)² + .077(G-U)^4).
 * Normalize at the study's ~11% TI (G-U=.216 U). This omits clipping,
 * saturation and independent branch modes; the pose probe checks their effect.
 */
export function weatherGustSigma(meanMps: number, gustMps: number): number {
  const u = Math.max(0, meanMps),
    d = Math.max(0, gustMps - u);
  const reference = Math.sqrt(1.04 * 0.216 ** 2 + 0.077 * 0.216 ** 4);
  return (
    (fitMagnoliaSway().sigmaMetresPerMpsSquared *
      Math.sqrt(1.04 * u * u * d * d + 0.077 * d ** 4)) /
    reference
  );
}
