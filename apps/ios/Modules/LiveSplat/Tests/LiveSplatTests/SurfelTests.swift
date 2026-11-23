import Foundation
import Testing

@testable import LiveSplat

@Test func surfelQuantizationUsesFloorAndIncludesResolutionLevel() {
  let near = SurfelKey(position: SIMD3(-0.01, 0.03, -2), depth: 2)
  #expect(near.x == -1 && near.y == 1 && near.z == -100 && near.level == 0)
  let far = SurfelKey(position: SIMD3(-0.01, 0.03, -2), depth: 4)
  #expect(far.level == 1 && far.z == -50)
  #expect(near != far)
  #expect(near.bucket(count: 65_536) < 65_536)
}
@Test func surfelFusionAveragesAndEvictsOldestThenLowestConfidence() {
  var hash = SurfelHash(capacity: 4)  // Force every sample to collide.
  func p(_ x: Float) -> SIMD3<Float> { SIMD3(x, 0, -2) }
  for i in 0..<4 {
    let accepted = hash.fuse(
      position: p(Float(i)), color: .zero, depth: 2, confidence: 2, radius: 0.02,
      epoch: UInt32(i + 1))
    #expect(accepted)
  }
  hash.fuse(
    position: p(0), color: SIMD3(repeating: 1), depth: 2, confidence: 2, radius: 0.03, epoch: 5)
  #expect(hash.cells[0]?.color == SIMD3(repeating: 0.5))
  #expect(hash.cells[0]?.radius == 0.03)
  hash.fuse(position: p(5), color: .zero, depth: 2, confidence: 2, radius: 0.02, epoch: 6)
  #expect(hash.count == 4)
  #expect(!hash.cells.contains { $0?.position == p(1) })
  #expect(hash.cells.contains { $0?.position == p(0) })
  let rejectedInvalidDepth = !hash.fuse(
    position: p(9), color: .zero, depth: .nan, confidence: 2, radius: 0.02, epoch: 7)
  #expect(rejectedInvalidDepth)
  let rejectedConfidence = !hash.fuse(
    position: p(9), color: .zero, depth: 2, confidence: 1, radius: 0.02, epoch: 7)
  #expect(rejectedConfidence)
}
@Test func surfelSizingCoversDiagonalSampleSpacingAndIsBounded() {
  let radius = SurfelHash.radius(depth: 2, focalPixels: 100)
  #expect(radius > sqrt(2) * 2 / 100)  // 2-pixel grid: half its diagonal spacing
  #expect(radius == Float(0.85) * 2 * 2 / 100)
  #expect(SurfelHash.radius(depth: 0.1, focalPixels: 1000) == Float(0.85) * 0.02)
  #expect(SurfelHash.radius(depth: 4, focalPixels: 1000) > sqrt(2) * 0.04 / 2)
  #expect(SurfelHash.radius(depth: 8, focalPixels: 10) == 0.12)
}
@Test func qualityRecoversAfterOverrunsAndPowerConstraints() {
  var q = PreviewQuality()
  q.record(seconds: 0.009, constrained: false)
  #expect(q.level == 0)
  for _ in 0..<5 { q.record(seconds: 0.009, constrained: false) }
  #expect(q.level == 2 && q.fusionInterval == 0.4 && q.visibleStride == 4)
  for _ in 0..<240 { q.record(seconds: 0.002, constrained: false) }
  #expect(q.level == 0)
  q.record(seconds: 0.002, constrained: true)
  #expect(q.level == 1)
  for _ in 0..<120 { q.record(seconds: 0.002, constrained: false) }
  #expect(q.level == 0)
}
@Test func fiveMinuteFusionReferenceStaysFlatAtCapacity() {
  var hash = SurfelHash(capacity: 1024)
  let initialBytes = hash.cells.count * MemoryLayout<Surfel?>.stride
  var peak = 0
  let start = DispatchTime.now().uptimeNanoseconds
  // 3000 keyframes at 10 Hz. Fixture sample subset; shader dispatch has the separate 12,288 cap.
  for epoch in 1...3000 {
    for i in 0..<128 {
      let x = Float((epoch * 128 + i) % 10000) / 10
      hash.fuse(
        position: SIMD3(x, 0, -2), color: .zero, depth: 2, confidence: 2, radius: 0.02,
        epoch: UInt32(epoch))
    }
    peak = max(peak, hash.count)
    #expect(hash.cells.count * MemoryLayout<Surfel?>.stride == initialBytes)
  }
  let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
  #expect(peak == 1024 && hash.count == 1024)
  #expect(seconds / 3000 < 0.008)  // reference subset CPU budget, not a Metal timing claim
}

@Test func fusionDispatchBudgetAndRotatingSamplesMatchShader() {
  for (w, h) in [(256, 192), (512, 512), (1, 1), (49, 37)] {
    let plan = SurfelSampling(width: w, height: h)
    #expect(plan.count <= 12_288)
    var pixels = Set<SIMD2<Int>>()
    for epoch in 0..<plan.stride * plan.stride {
      for i in 0..<plan.count {
        if let p = plan.pixel(sample: i, epoch: UInt32(epoch)) { pixels.insert(p) }
      }
    }
    #expect(pixels.count == w * h)
    #expect(plan.pixel(sample: plan.count, epoch: 0) == nil)
  }
}
