import Foundation
import Testing
@testable import LiveSplat

@Test func savedSplatRoundTripsWorldPoseAndRejectsBrokenPLY() throws {
  let g = try Gaussian(position: SplatVector(1, 2, 3), scale: SplatVector(0.02, 0.03, 0.01),
    rotation: [1, 0, 0, 0], opacity: 0.8, color: SplatVector(0.2, 0.4, 0.8))
  #expect(MemoryLayout<SavedGaussian>.stride == 80)
  let data = try SplatPLY.encode([g])
  let saved = try SavedSplat(data: data)
  #expect(saved.gaussians.count == 1)
  #expect(saved.center == SIMD3<Float>(1, 2, 3))
  #expect(abs(saved.gaussians[0].color.w - 0.8) < 1e-6)
  #expect(abs(saved.gaussians[0].axisX.x - 0.02) < 1e-6)
  #expect(throws: LiveSplatError.self) { try SavedSplat(data: data.dropLast()) }
  #expect(throws: LiveSplatError.self) { try SavedSplat(data: Data("bad".utf8)) }
}
@Test func savedOrbitIsOrthonormalAndZoomIsBounded() {
  var orbit = SplatOrbit()
  orbit.drag(x: 123, y: 100_000)
  orbit.zoom(factor: 1_000_000)
  #expect(orbit.distance == 0.1)
  let p = orbit.pose
  var length: Float = 0
  var dot: Float = 0
  for row in 0..<3 { length += p[row*4]*p[row*4]; dot += p[row*4]*p[row*4+1] }
  #expect(abs(length - 1) < 1e-6)
  #expect(abs(dot) < 1e-6)
  let previous = orbit.distance
  orbit.zoom(factor: .nan)
  #expect(orbit.distance == previous)
}

@Test func savedSplatSortsFarToNearAndPreservesRotatedCovariance() throws {
  let near = try Gaussian(position: SplatVector(0, 0, -1), scale: SplatVector(0.02, 0.03, 0.01),
    rotation: [1, 0, 0, 0], opacity: 0.8, color: SplatVector(1, 0, 0))
  let far = try Gaussian(position: SplatVector(0, 0, -3), scale: SplatVector(0.02, 0.03, 0.01),
    rotation: [1, 0, 0, 1], opacity: 0.8, color: SplatVector(0, 1, 0))
  let saved = try SavedSplat(data: SplatPLY.encode([near, far]))
  let identity: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
  let indices = saved.sortedIndices(viewRows: identity)
  #expect(indices.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } == 1)
  #expect(abs(saved.gaussians[1].axisX.x) < 1e-6)
  #expect(abs(saved.gaussians[1].axisX.y - 0.02) < 1e-6)
  #expect(saved.sortedIndices(viewRows: []).isEmpty)
}
