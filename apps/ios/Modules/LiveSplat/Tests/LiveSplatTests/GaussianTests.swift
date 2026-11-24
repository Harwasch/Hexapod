import Foundation
import Testing

@testable import LiveSplat

@Test func surfaceSeedCovarianceFollowsNormalAndDepthFootprint() throws {
  for normal in [
    SplatVector(0, 0, 1), SplatVector(0, 0, -1), SplatVector(1, 0, 0), SplatVector(1, 2, 3),
  ] {
    let g = try #require(
      GaussianSeed.make(
        position: SplatVector(0, 0, -2), normal: normal, depth: 2, focalPixels: 100,
        confidence: 2, color: .zero))
    let n = normal * (1 / sqrt(normal.squaredLength))
    let sigma = Double(SurfelHash.radius(depth: 2, focalPixels: 100)) * 0.5
    let thin = max(0.001, sigma * 0.2)
    // Analytic eigenvector identity Sigma*n = s_normal^2*n. Bound is for
    // Double operations/quaternion normalization, not GPU parity calibration.
    let error = g.covariance * n - n * (thin * thin)
    #expect(error.squaredLength < 1e-24)
    let trace = (0..<3).reduce(0.0) { $0 + g.covariance[$1, $1] }
    #expect(abs(trace - (2 * sigma * sigma + thin * thin)) < 1e-12)
    #expect(abs(g.opacity - 0.85) < 1e-12)
  }
  let near = try #require(
    GaussianSeed.make(
      position: .zero, normal: SplatVector(0, 0, 1), depth: 1, focalPixels: 100, confidence: 1,
      color: .zero))
  let far = try #require(
    GaussianSeed.make(
      position: .zero, normal: SplatVector(0, 0, 1), depth: 2, focalPixels: 100, confidence: 2,
      color: .zero))
  #expect(far.covariance[0, 0] > near.covariance[0, 0])
  #expect(near.opacity < far.opacity)
  #expect(
    GaussianSeed.make(
      position: .zero, normal: .zero, depth: 2, focalPixels: 100, confidence: 2, color: .zero)
      == nil)
  #expect(
    GaussianSeed.make(
      position: .zero, normal: SplatVector(0, 0, 1), depth: .nan, focalPixels: 100, confidence: 2,
      color: .zero) == nil)
  #expect(
    GaussianSeed.make(
      position: .zero, normal: SplatVector(0, 0, 1), depth: 2, focalPixels: 100, confidence: 0,
      color: .zero) == nil)
}

@Test func surfaceDepthNormalAndProjectionHaveAnalyticCovariance() throws {
  let camera = try SplatCamera(fx: 100, fy: 100, cx: 0, cy: 0, width: 3, height: 3)
  let frame = try SplatFrame(
    timestamp: 0, camera: camera, width: 3, height: 3,
    depths: Array(repeating: 2, count: 9), confidence: Array(repeating: 2, count: 9),
    colors: Array(repeating: .zero, count: 9))
  let g = try #require(GaussianSeed.surface(frame: frame, x: 0, y: 0))
  let p = try #require(SplatMath.project(g, camera: camera))
  #expect(p.u == 0 && p.v == 0 && p.depth == 2)
  #expect(abs(p.xx - (2500 * g.covariance[0, 0] + 0.3)) < 1e-12)
  #expect(abs(p.yy - (2500 * g.covariance[1, 1] + 0.3)) < 1e-12)
  #expect(p.xy == 0)
  let angled = try #require(
    GaussianSeed.make(
      position: SplatVector(0.1, 0.2, -2), normal: SplatVector(1, 1, 1), depth: 2, focalPixels: 100,
      confidence: 2, color: .zero))
  let a = try #require(SplatMath.project(angled, camera: camera))
  #expect(a.xx > 0 && a.yy > 0 && a.xx * a.yy > a.xy * a.xy)
  #expect(abs(a.xy) > 0)
  #expect(GaussianSeed.surface(frame: frame, x: 3, y: 0) == nil)
}

@Test func gaussianBitonicOrderMatchesComparisonIncludingInvalidAndTiedDepths() {
  for count in [1, 2, 16, 256, 1024, GaussianSeed.drawCapacity] {
    var entries = (0..<count).map {
      GaussianSort.Entry(depth: Float(($0 * 37) % 19), index: UInt32($0))
    }
    if count > 2 {
      entries[0] = .init(depth: .nan, index: 0)
      entries[1] = .init(depth: .infinity, index: 1)
      entries[2] = .init(depth: -2, index: 2)
    }
    let expected = entries.sorted(by: GaussianSort.precedes)
    GaussianSort.sort(&entries)
    #expect(entries == expected)
  }
}

@Test func gaussianDedupeAndEvictionReuseSurfelKeys() throws {
  var hash = GaussianVoxelHash(capacity: 4)
  func seed(_ x: Double, _ confidence: UInt8 = 2) throws -> Gaussian {
    try #require(
      GaussianSeed.make(
        position: SplatVector(x, 0, -2), normal: SplatVector(0, 0, 1), depth: 2, focalPixels: 100,
        confidence: confidence, color: .zero))
  }
  for epoch in 1...8 {
    #expect(hash.fuse(try seed(0), depth: 2, confidence: 2, epoch: UInt32(epoch)))
  }
  #expect(hash.voxels.count == 1)
  #expect(hash.voxels.cells.compactMap { $0 }.first?.observations == 8)
  for i in 1...8 {
    hash.fuse(try seed(Double(i), 1), depth: 2, confidence: 1, epoch: UInt32(i + 8))
  }
  #expect(hash.voxels.count == 4)
  for i in hash.seeds.indices {
    let g = try #require(hash.gaussian(at: i))
    let key = SurfelKey(
      position: SIMD3(Float(g.position.x), Float(g.position.y), Float(g.position.z)), depth: 2)
    #expect(key == hash.voxels.cells[i]?.key)
    #expect(g.covariance[2, 2] < g.covariance[0, 0])
  }
  #expect(GaussianSeed.drawCapacity == 32_768)
  #expect(GaussianSeed.attributeBytes + GaussianSeed.sortBytes == 4_456_448)
}

@Test func depthDiscontinuitiesAndInvalidNeighborsUseViewingNormal() throws {
  let camera = try SplatCamera(fx: 100, fy: 100, cx: 1, cy: 1, width: 3, height: 3)
  for (depth, neighbor) in [(2.0, 4.0), (8.0, 8.01), (2.0, Double.nan)] {
    var depths = Array(repeating: Float(depth), count: 9)
    depths[5] = Float(neighbor)
    let frame = try SplatFrame(
      timestamp: 0, camera: camera, width: 3, height: 3,
      depths: depths, confidence: Array(repeating: 2, count: 9),
      colors: Array(repeating: .zero, count: 9))
    let g = try #require(GaussianSeed.surface(frame: frame, x: 1, y: 1))
    #expect(g.covariance[0, 2] == 0 && g.covariance[1, 2] == 0)
    #expect(g.covariance[2, 2] < g.covariance[0, 0])
  }
  #expect(GaussianVoxelHash(capacity: 1).seeds.count == 4)
  let capped = GaussianVoxelHash(capacity: SurfelHash.capacity + 4)
  #expect(
    capped.seeds.count == SurfelHash.capacity && capped.voxels.cells.count == SurfelHash.capacity)
}
