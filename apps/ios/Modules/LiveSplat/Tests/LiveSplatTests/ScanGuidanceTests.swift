import Foundation
import Testing

@testable import LiveSplat

@Suite struct ScanGuidanceTests {
  private func sample(
    angle: Float, elevation: Float = 0, radius: Float = 2, time: Double,
    spread: Float = 0
  ) -> ScanGuidanceSample {
    let a = angle * .pi / 180
    let e = elevation * .pi / 180
    return ScanGuidanceSample(
      center: .zero,
      camera: SIMD3(radius * cos(e) * cos(a), radius * sin(e), radius * cos(e) * sin(a)),
      observedCells: 100, shortSpreadFraction: spread, timestamp: time)
  }
  @Test func halfOrbitAsksAroundAndFullLevelOrbitAsksHigher() {
    var half = ScanGuidance()
    var full = ScanGuidance()
    for i in 0...240 {
      let t = Double(i) / 10
      half.update(sample(angle: Float(min(i, 120)) * 1.5, time: t), blur: 0, now: t)
      full.update(sample(angle: Float(min(i, 240)) * 1.5, time: t), blur: 0, now: t)
    }
    #expect(half.tip == .around && half.largestGap >= 170)
    // Let the new candidate pass the dwell after closing the loop.
    for i in 241...275 {
      let t = Double(i) / 10
      full.update(sample(angle: 360, time: t), blur: 0, now: t)
    }
    #expect(full.largestGap < 15 && full.tip == .higher)
  }
  @Test func farAndBlurUseHysteresisAndMinimumDwell() {
    var guidance = ScanGuidance()
    for i in 0...10 {
      let t = Double(i) / 10
      guidance.update(sample(angle: 0, radius: 4, time: t), blur: 0, now: t)
    }
    #expect(guidance.tip == .closer && guidance.completedSectors == 0)
    // Range jitter within the Schmitt band retains Closer.
    for i in 11...35 {
      let t = Double(i) / 10
      guidance.update(sample(angle: 0, radius: i % 2 == 0 ? 3.1 : 3.4, time: t), blur: 0, now: t)
    }
    #expect(guidance.tip == .closer)
    // A transient blur spike cannot flicker the pill.
    guidance.update(nil, blur: 2, now: 3.6)
    guidance.update(nil, blur: 0, now: 3.8)
    #expect(guidance.tip == .closer)
    for i in 40...50 { guidance.update(nil, blur: 2, now: Double(i) / 10) }
    #expect(guidance.tip == .slower)
    guidance.update(sample(angle: 0, radius: 2, time: 5.1), blur: 0, now: 5.1)
    #expect(guidance.tip == .slower)
    for i in 52...75 {
      let t = Double(i) / 10
      guidance.update(sample(angle: 0, time: t), blur: 0, now: t)
    }
    #expect(guidance.tip == .around)
  }
  @Test func sectorCreditIsBoundedDeduplicatedAndResetsForNewScan() {
    var guidance = ScanGuidance()
    var notifications = 0
    for sector in 0..<24 {
      for step in 0..<8 {
        let t = Double(sector * 8 + step) / 10
        let fresh = guidance.update(sample(angle: Float(sector * 15 + 7), time: t), blur: 0, now: t)
        notifications += fresh.nonzeroBitCount
      }
    }
    #expect(guidance.fraction == 1 && notifications == 24)
    #expect(guidance.completedSectors == (1 << 24) - 1)
    let old = guidance.completedSectors
    // Repeated pose timestamps never earn credit twice or retrigger a haptic.
    #expect(guidance.update(sample(angle: 7, time: 0), blur: 0, now: 100) == 0)
    #expect(guidance.completedSectors == old)
    guidance = ScanGuidance()
    #expect(guidance.fraction == 0 && guidance.tip == nil)
  }
  @Test func raisedLoopClearsHigherAndPoorSurfaceSpreadRequestsMoreAngles() {
    var guidance = ScanGuidance()
    for i in 0...300 {
      let t = Double(i) / 10
      guidance.update(sample(angle: Float(i) * 1.5, elevation: 50, time: t), blur: 0, now: t)
    }
    #expect(guidance.tip == nil)
    for i in 301...335 {
      let t = Double(i) / 10
      guidance.update(
        sample(angle: Float(i) * 1.5, elevation: 50, time: t, spread: 0.5), blur: 0, now: t)
    }
    #expect(guidance.tip == .around)
    for i in 336...370 {
      let t = Double(i) / 10
      guidance.update(
        sample(angle: Float(i) * 1.5, elevation: 50, time: t, spread: 0.2), blur: 0, now: t)
    }
    #expect(guidance.tip == .around)
    for i in 371...405 {
      let t = Double(i) / 10
      guidance.update(
        sample(angle: Float(i) * 1.5, elevation: 50, time: t, spread: 0.1), blur: 0, now: t)
    }
    #expect(guidance.tip == nil)
  }
  @Test func stationaryViewsCannotCompleteAnOrbit() {
    var guidance = ScanGuidance()
    var events = 0
    for i in 0...200 {
      let t = Double(i) / 10
      events += guidance.update(sample(angle: 7, time: t), blur: 0, now: t).nonzeroBitCount
    }
    #expect(guidance.completedSectors.nonzeroBitCount == 1 && events == 1)
    #expect(guidance.fraction == 1.0 / 24 && guidance.tip == .around)
  }
  @Test func missingGeometryAndBlurredViewsDoNotFillRing() {
    var guidance = ScanGuidance()
    for i in 0...100 {
      let t = Double(i) / 10
      guidance.update(sample(angle: Float(i) * 4, time: t), blur: 1, now: t)
    }
    #expect(guidance.fraction == 0 && guidance.tip == .slower)
    let empty = ScanGuidanceSample(
      center: .zero, camera: SIMD3(2, 0, 0), observedCells: 0,
      shortSpreadFraction: 1, timestamp: 30)
    guidance = ScanGuidance()
    guidance.update(empty, blur: 0, now: 30)
    guidance.update(nil, blur: 0, now: 31)
    #expect(guidance.tip == nil && guidance.fraction == 0)
  }
}
