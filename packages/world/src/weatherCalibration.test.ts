import { expect, it } from "vitest";
import { MAGNOLIA_SWAY, fitMagnoliaSway, weatherGustSigma } from "./weatherCalibration";
it("fits the published broadleaf field data and reproduces the reported R²=.81", () => {
  const fit = fitMagnoliaSway();
  const avg = MAGNOLIA_SWAY.reduce((a, [u]) => a + u, 0) / MAGNOLIA_SWAY.length;
  const residual = MAGNOLIA_SWAY.reduce((a, [u, s]) => a + (u - fit.slope * Math.sqrt(s)) ** 2, 0);
  const total = MAGNOLIA_SWAY.reduce((a, [u]) => a + (u - avg) ** 2, 0);
  expect(1 - residual / total).toBeCloseTo(0.81, 2);
  const sigmaScatter = MAGNOLIA_SWAY.map(([u, s]) => Math.abs((u / fit.slope) ** 2 / s - 1));
  // Record the observed inverse-fit scatter before using its 60% envelope in
  // the pose probe. The tighter 20% bound applies to the fitted reference curve.
  expect(Math.max(...sigmaScatter)).toBeGreaterThan(0.58);
  expect(Math.max(...sigmaScatter)).toBeLessThan(0.6);
  // Independent figure digitization: slope about 5.4 m/s / sqrt(pixel).
  expect(fit.slope).toBeGreaterThan(5.3);
  expect(fit.slope).toBeLessThan(5.6);
  expect(fit.sigmaMetresPerMpsSquared * 9.17 ** 2 * 100).toBeCloseTo(8.38, 1);
  expect(weatherGustSigma(9.17, 9.17 * 1.216) * 100).toBeCloseTo(8.38, 1);
  expect(weatherGustSigma(11.49, 22.9) * 100).toBeGreaterThan(60);
  expect(weatherGustSigma(11.49, 22.9) * 100).toBeLessThan(65);
});
