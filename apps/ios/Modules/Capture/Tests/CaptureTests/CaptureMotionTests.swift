import Foundation
import Testing

@testable import Capture

private func pose(yaw: Double, x: Double = 0) -> [Double] {
  [cos(yaw), 0, sin(yaw), x, 0, 1, 0, 0, -sin(yaw), 0, cos(yaw), 0, 0, 0, 0, 1]
}

/// Feed a steady motion long enough for the smoothing to settle.
private func settle(_ omega: Double, speed: Double = 0, focal: Double, depth: Double = 1.5)
  -> MotionBlurMeter
{
  var meter = MotionBlurMeter()
  for step in 0..<60 {
    let t = Double(step) / 15
    meter.update(
      time: t, transform: pose(yaw: omega * t, x: speed * t), focalPx: focal,
      shutterS: 1.0 / 250, depthM: depth)
  }
  return meter
}

@Test func blurMatchesTheDerivationAt4K() {
  // One pixel at 1/250 s and fx 2830 px is 1 / (2830 / 250) = 0.0883 rad/s.
  let limit = 250.0 / 2830
  #expect(abs(settle(limit, focal: 2830).blurPixels - 1) < 0.01)
  #expect(settle(limit * 0.8, focal: 2830).level == .steady)
  #expect(settle(limit * 2, focal: 2830).level == .slowDown)
  #expect(settle(limit * 4, focal: 2830).level == .tooFast)
  // A slow 10°/s sweep is steady at 1440p and not at 4K.
  let tenDegrees = 10 * Double.pi / 180
  #expect(settle(tenDegrees * 0.9, focal: 1414).level == .steady)
  #expect(settle(tenDegrees, focal: 2830).level == .slowDown)
}

@Test func translationBlurScalesWithDepth() {
  let near = settle(0, speed: 0.13, focal: 2830, depth: 1.5).blurPixels
  let far = settle(0, speed: 0.13, focal: 2830, depth: 3).blurPixels
  #expect(abs(near - 0.98) < 0.02)
  #expect(abs(far - near / 2) < 0.01)
}

@Test func meterIgnoresBadSamplesAndUsesNominals() {
  var meter = MotionBlurMeter()
  meter.update(time: 0, transform: pose(yaw: 0), focalPx: 2830, shutterS: nil, depthM: nil)
  meter.update(time: 0, transform: pose(yaw: 1), focalPx: 2830, shutterS: nil, depthM: nil)
  #expect(meter.blurPixels == 0)
  meter.update(time: 0.1, transform: [.nan], focalPx: 2830, shutterS: nil, depthM: nil)
  #expect(meter.blurPixels == 0)
  meter.update(time: 0.1, transform: pose(yaw: 0.05), focalPx: 2830, shutterS: 0, depthM: 0)
  #expect(meter.blurPixels > 0)
}

@Test func exposurePlanKeepsShortShuttersAndCapsLongOnes() {
  let bright = ExposurePlan.plan(
    meteredDurationS: 1.0 / 500, meteredISO: 50, minDurationS: 1.0 / 20000, maxDurationS: 1.0 / 3,
    minISO: 32, maxISO: 2304)
  #expect(bright == ExposurePlan(mode: "locked", durationS: 1.0 / 500, iso: 50, shortfallEV: 0))
  let indoor = ExposurePlan.plan(
    meteredDurationS: 1.0 / 60, meteredISO: 200, minDurationS: 1.0 / 20000,
    maxDurationS: 1.0 / 3, minISO: 32, maxISO: 2304)
  #expect(indoor.mode == "custom" && indoor.durationS == 1.0 / 250)
  #expect(abs(indoor.iso - 200 * 250.0 / 60) < 1e-9)
  #expect(indoor.shortfallEV == 0)
  let dim = ExposurePlan.plan(
    meteredDurationS: 1.0 / 30, meteredISO: 1600, minDurationS: 1.0 / 20000,
    maxDurationS: 1.0 / 3, minISO: 32, maxISO: 2304)
  #expect(dim.iso == 2304)
  #expect(abs(dim.shortfallEV - log2(1600 * 250.0 / 30 / 2304)) < 1e-9)
}
