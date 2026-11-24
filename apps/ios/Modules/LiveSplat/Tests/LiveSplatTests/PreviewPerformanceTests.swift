import Foundation
import Testing

@testable import LiveSplat

@Test func fiveMinuteGaussianAndInsetReferenceKeepsCapacityAndFrameBudget() throws {
  var hash = GaussianVoxelHash(capacity: 1024)
  let seed = try #require(
    GaussianSeed.make(
      position: SplatVector(0, 0, -2), normal: SplatVector(0, 0, 1), depth: 2, focalPixels: 100,
      confidence: 2, color: .zero))
  let initialCells = hash.voxels.cells.count
  let initialSeeds = hash.seeds.count
  let start = DispatchTime.now().uptimeNanoseconds
  var orbit = ScanOrbit()
  var order = (0..<128).map { GaussianSort.Entry(depth: Float($0 + 1), index: UInt32($0)) }
  // Same 3000-keyframe, 128-sample subset as the surfel regression. CPU reference
  // only: bounds fixed storage/work, never claims GPU fps or a resident-memory measurement.
  for epoch in 1...3000 {
    for i in 0..<128 {
      var g = seed
      g.parameters[0] = Double((epoch * 128 + i) % 10000) / 10
      hash.fuse(g, depth: 2, confidence: 2, epoch: UInt32(epoch))
    }
    GaussianSort.sort(&order)
    orbit.rotate(horizontal: 0.001, vertical: 0)
    #expect(orbit.angle(at: Double(epoch)).isFinite)
    #expect(hash.voxels.cells.count == initialCells && hash.seeds.count == initialSeeds)
  }
  let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
  #expect(hash.voxels.count == 1024)
  #expect(seconds / 3000 < 0.008)
}
