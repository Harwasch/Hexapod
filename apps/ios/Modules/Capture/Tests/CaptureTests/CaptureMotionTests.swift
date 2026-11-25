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

private func plan(_ duration: Double, _ iso: Double, maxISO: Double = 2304) -> ExposurePlan {
  ExposurePlan.plan(
    meteredDurationS: duration, meteredISO: iso, minDurationS: 1.0 / 20000,
    maxDurationS: 1.0 / 3, minISO: 32, maxISO: maxISO)
}

@Test func exposurePlanLocksShortShutters() {
  #expect(
    plan(1.0 / 500, 50)
      == ExposurePlan(mode: "locked", durationS: 1.0 / 500, iso: 50, shortfallEV: 0))
}

@Test func exposurePlanShortensOnlyAsFarAsTheNoiseCeilingAllows() {
  // Bright indoor: 1/60 s at ISO 200 reaches 1/250 s at ISO 833, under the ceiling.
  let bright = plan(1.0 / 60, 200)
  #expect(bright.mode == "custom" && bright.durationS == 1.0 / 250)
  #expect(abs(bright.iso - 200 * 250.0 / 60) < 1e-9)
  // Dim indoor: 1/30 s at ISO 400 would need ISO 3333 at 1/250 s; stop at ISO 1600, 1/120 s.
  let dim = plan(1.0 / 30, 400)
  #expect(dim.mode == "custom" && dim.iso == ExposurePlan.noiseCeilingISO)
  #expect(abs(dim.durationS - 1.0 / 120) < 1e-12 && dim.shortfallEV == 0)
  // Already at or over the ceiling: keep what was metered, never a longer shutter.
  #expect(
    plan(1.0 / 30, 1600)
      == ExposurePlan(mode: "locked", durationS: 1.0 / 30, iso: 1600, shortfallEV: 0))
  #expect(plan(1.0 / 30, 2000).durationS == 1.0 / 30)
  // A sensor whose ISO tops out below the ceiling uses its own maximum.
  let small = plan(1.0 / 30, 400, maxISO: 1000)
  #expect(small.iso == 1000 && abs(small.durationS - 1.0 / 75) < 1e-12)
}

@Test func probeWaitsForDepthAndCountsOnlyFramesWithIt() {
  var probe = FourKProbe()
  for step in 0..<40 {  // 2.6 s of depth-less frames at 15 Hz: still pending
    #expect(
      probe.observe(
        time: Double(step) / 15, imageWidth: 3840, imageHeight: 2160, depthWidth: nil,
        depthHeight: nil) == .pending)
  }
  for step in 0..<FourKProbe.depthFramesNeeded {
    probe.observe(
      time: 2.7 + Double(step) / 15, imageWidth: 3840, imageHeight: 2160, depthWidth: 256,
      depthHeight: 144)
  }
  #expect(probe.verdict == .accepted)
}

@Test func probeRejectsMeasuredAspectForGoodAndMissingDepthForTheSession() {
  var aspect = FourKProbe()
  aspect.observe(time: 0, imageWidth: 3840, imageHeight: 2160, depthWidth: nil, depthHeight: nil)
  #expect(
    aspect.observe(
      time: 0.5, imageWidth: 3840, imageHeight: 2160, depthWidth: 256, depthHeight: 192)
      == .rejected(reason: "depthAspect", persist: true))
  var missing = FourKProbe()
  missing.observe(time: 10, imageWidth: 3840, imageHeight: 2160, depthWidth: nil, depthHeight: nil)
  #expect(
    missing.observe(
      time: 13, imageWidth: 3840, imageHeight: 2160, depthWidth: nil, depthHeight: nil)
      == .rejected(reason: "noDepth", persist: false))
}
