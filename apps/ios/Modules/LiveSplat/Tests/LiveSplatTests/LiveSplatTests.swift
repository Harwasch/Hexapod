import Foundation
import Testing

@testable import LiveSplat

// Bounds chosen from Double roundoff and central-difference truncation, not fitted
// to device arithmetic: analytic 1e-10.
private func near(
  _ a: Double, _ b: Double, _ tolerance: Double = 1e-10,
  sourceLocation: SourceLocation = #_sourceLocation
) {
  #expect(abs(a - b) <= tolerance, "\(a) != \(b)", sourceLocation: sourceLocation)
}
private func camera(rotation: SplatMatrix = .identity, position: SplatVector = .zero) throws
  -> SplatCamera
{
  try SplatCamera(
    cameraToWorld: rotation, position: position, fx: 100, fy: 120, cx: 32, cy: 24, width: 64,
    height: 48)
}
private func gaussian(position: SplatVector = SplatVector(0, 0, -2)) throws -> Gaussian {
  try Gaussian(
    position: position, scale: SplatVector(0.1, 0.2, 0.3), color: SplatVector(0.8, 0.3, 0.1))
}
private func frame(timestamp: Double = 1, depth: Float = 2, confidence: UInt8 = 2) throws
  -> SplatFrame
{
  try SplatFrame(
    timestamp: timestamp, camera: camera(), width: 32, height: 24,
    depths: Array(repeating: depth, count: 768),
    confidence: Array(repeating: confidence, count: 768),
    colors: Array(repeating: SplatVector(0.8, 0.3, 0.1), count: 768))
}

@Test func arkitProjectionAndRoundTrip() throws {
  let c = try camera()
  let p = try #require(c.project(SplatVector(0.2, 0.1, -2)))
  near(p.x, 42)
  near(p.y, 18)
  near(p.z, 2)
  let back = c.unproject(u: p.x, v: p.y, depth: p.z)
  near(back.x, 0.2)
  near(back.y, 0.1)
  near(back.z, -2)
  #expect(c.project(SplatVector(0, 0, 2)) == nil)
  #expect(c.project(SplatVector(0, 0, -0.001)) == nil)
}

@Test func cameraAxesFlipAfterPoseInversion() throws {
  let rotation = SplatMatrix([0, 0, 1, 0, 1, 0, -1, 0, 0])
  let c = try camera(rotation: rotation, position: SplatVector(10, 3, 4))
  // Camera (0.2,0.1,-2) -> world (8,3.1,3.8).
  let p = try #require(c.project(SplatVector(8, 3.1, 3.8)))
  near(p.x, 42)
  near(p.y, 18)
  near(p.z, 2)
  let back = c.unproject(u: 42, v: 18, depth: 2)
  near(back.x, 8)
  near(back.y, 3.1)
  near(back.z, 3.8)
}

@Test func cameraRejectsReflectionAndInvalidInputs() throws {
  #expect(throws: LiveSplatError.self) {
    try camera(rotation: SplatMatrix([1, 0, 0, 0, 1, 0, 0, 0, -1]))
  }
  #expect(throws: LiveSplatError.self) { try camera(position: SplatVector(.nan, 0, 0)) }
  #expect(throws: LiveSplatError.self) {
    try SplatCamera(fx: 0, fy: 1, cx: 0, cy: 0, width: 1, height: 1)
  }
}

@Test func covarianceFromScaleAndHamiltonRotation() throws {
  var g = try gaussian()
  let expected = [0.01, 0, 0, 0, 0.04, 0, 0, 0, 0.09]
  for (a, b) in zip(g.covariance.values, expected) { near(a, b) }
  g.parameters.replaceSubrange(6..<10, with: [sqrt(0.5), 0, 0, sqrt(0.5)])
  for (a, b) in zip(g.covariance.values, [0.04, 0, 0, 0, 0.01, 0, 0, 0, 0.09]) { near(a, b) }
  // Quaternion normalization must leave covariance invariant.
  for i in 6..<10 { g.parameters[i] *= 3 }
  for (a, b) in zip(g.covariance.values, [0.04, 0, 0, 0, 0.01, 0, 0, 0, 0.09]) { near(a, b) }
}

@Test func projectedCovarianceHandComputedOffAxis() throws {
  let g = try gaussian(position: SplatVector(0.2, 0.1, -2))
  let p = try #require(SplatMath.project(g, camera: camera()))
  // Optical p=(.2,-.1,2), J=[50,0,-5;0,60,3].
  // Diagonal world covariance .01,.04,.09; camera axis flips preserve it.
  near(p.xx, 25 + 2.25 + 0.3)
  near(p.xy, -1.35)
  near(p.yy, 144 + 0.81 + 0.3)
  #expect(p.radius > 0)
}

@Test func cameraRotationTransformsCovariance() throws {
  let c = try camera(rotation: SplatMatrix([0, 0, 1, 0, 1, 0, -1, 0, 0]))
  let g = try gaussian(position: SplatVector(-2, 0, 0))
  let p = try #require(SplatMath.project(g, camera: c))
  near(p.xx, 225.3)
  near(p.yy, 144.3)
  near(p.xy, 0)
}

@Test func seedUsesDepthResolutionAndConfidence() throws {
  let f = try frame()
  let g = try #require(f.seed(x: 21, y: 9))
  near(g.position.x, 0.2)
  near(g.position.y, 0.1)
  near(g.position.z, -2)
  near(exp(g.parameters[3]), 0.02)
  #expect(try frame(confidence: 1).seed(x: 0, y: 0) == nil)
  #expect(try frame(depth: .nan).seed(x: 0, y: 0) == nil)
  #expect(try frame(depth: -1).seed(x: 0, y: 0) == nil)
  #expect(try frame(depth: 9).seed(x: 0, y: 0) == nil)
  #expect(f.seed(x: -1, y: 0) == nil)
}

@Test func frameRejectsMalformedBuffers() throws {
  #expect(throws: LiveSplatError.self) {
    try SplatFrame(
      timestamp: 1, camera: camera(), width: 1, height: 1, depths: [], confidence: [2],
      colors: [.zero])
  }
}

@Test func coverageClipsAndRequiresDepthAgreement() throws {
  var grid = CoverageGrid(columns: 2, rows: 2)
  #expect(grid.cell(u: -0.1, v: 0, width: 64, height: 48) == nil)
  #expect(grid.cell(u: 64, v: 0, width: 64, height: 48) == nil)
  #expect(grid.cell(u: .nan, v: 0, width: 64, height: 48) == nil)
  #expect(grid.observe(try gaussian(), in: try frame()))
  near(grid.fraction, 0.25)
  #expect(!grid.observe(try gaussian(position: SplatVector(0, 0, -1)), in: try frame()))
  #expect(!grid.observe(try gaussian(), in: try frame(confidence: 0)))
}

@Test @MainActor func rendererCapsSeedsAndResets() throws {
  var config = LiveSplatConfiguration()
  config.maxSplats = 4
  config.maxSeedsPerFrame = 3
  let renderer = DefaultLiveSplatRenderer(configuration: config)
  renderer.start()
  renderer.ingest(try frame())
  #expect(renderer.splats.count == 3)
  renderer.ingest(try frame(timestamp: 2))
  #expect(renderer.splats.count == 4)
  let before = renderer.splats
  renderer.ingest(try frame(timestamp: 2))
  #expect(renderer.splats == before)
  renderer.stop()
  renderer.start()
  #expect(renderer.splats.isEmpty)
}

@Test @MainActor func stationaryFusionIsNotLimitedByCoverageGrid() throws {
  let c = try SplatCamera(fx: 100, fy: 100, cx: 32, cy: 24, width: 64, height: 48)
  let f = try SplatFrame(
    timestamp: 0, camera: c, width: 64, height: 48,
    depths: Array(repeating: 2, count: 3072), confidence: Array(repeating: 2, count: 3072),
    colors: Array(repeating: .zero, count: 3072))
  let renderer = DefaultLiveSplatRenderer()
  renderer.start()
  renderer.ingest(f)
  #expect(renderer.splats.count > 768)
}

@Test func binaryPLYUsesStandardActivationsAndOrder() throws {
  let g = try gaussian()
  let data = try SplatPLY.encode([g])
  let marker = Data("end_header\n".utf8)
  let range = try #require(data.range(of: marker))
  let header = String(decoding: data[..<range.upperBound], as: UTF8.self)
  #expect(header.contains("element vertex 1\n"))
  #expect(header.contains("property float f_dc_0\n"))
  let payload = Array(data[range.upperBound...])
  #expect(payload.count == 17 * 4)
  let floats = stride(from: 0, to: payload.count, by: 4).map { i -> Double in
    let bits =
      UInt32(payload[i]) | UInt32(payload[i + 1]) << 8 | UInt32(payload[i + 2]) << 16 | UInt32(
        payload[i + 3]) << 24
    return Double(Float(bitPattern: bits))
  }
  near(floats[2], -2)
  near(floats[6], (0.8 - 0.5) / 0.28209479177387814, 1e-6)
  near(floats[9], log(4), 1e-6)
  near(floats[10], log(0.1), 1e-6)
  near(floats[13], 1)
  var bad = g
  bad.parameters[0] = .nan
  #expect(throws: LiveSplatError.self) { try SplatPLY.encode([bad]) }
}

@Test @MainActor func exportRequiresStopAndWritesInsideFolder() throws {
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let r = DefaultLiveSplatRenderer()
  r.start()
  r.ingest(try frame())
  #expect(throws: LiveSplatError.self) { try r.exportPLY(to: directory) }
  r.stop()
  let url = try r.exportPLY(to: directory)
  #expect(url.lastPathComponent == "livesplat.ply")
  #expect(try Data(contentsOf: url) == SplatPLY.encode(r.splats))
}

@Test @MainActor func exportDoesNotInvalidateAFinalizedManifest() throws {
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  try Data("{}".utf8).write(to: directory.appendingPathComponent("manifest.json"))
  let r = DefaultLiveSplatRenderer()
  #expect(throws: LiveSplatError.self) { try r.exportPLY(to: directory) }
  #expect(
    !FileManager.default.fileExists(atPath: directory.appendingPathComponent("livesplat.ply").path)
  )
}

@Test @MainActor func trackingLossClearsStaleOverlayAndKeyframes() throws {
  let r = DefaultLiveSplatRenderer()
  r.start()
  r.ingest(try frame())
  r.trackingUnavailable()
  #expect(r.latestFrame == nil)
  near(r.coverageFraction, 0)
  #expect(!r.splats.isEmpty)
}

@Test func gaussianRejectsUnrepresentableGPUInputs() {
  #expect(throws: LiveSplatError.self) {
    try Gaussian(
      position: SplatVector(1e100, 0, -2), scale: SplatVector(0.1, 0.1, 0.1), color: .zero)
  }
  #expect(throws: LiveSplatError.self) {
    try Gaussian(
      position: SplatVector(0, 0, -2), scale: SplatVector(0.1, 0.1, 0.1),
      rotation: [1e300, 0, 0, 0], color: .zero)
  }
}

@Test func coverageHintSelectsAVisibleUncoveredCellAndDisappearsWhenFilled() throws {
  var grid = CoverageGrid(columns: 4, rows: 4)
  let identity: [Double] = [1, 0, 0, 1, 0, 0]
  let hint = try #require(grid.gapHint(displayTransform: identity))
  let index = try #require(grid.cell(u: hint.x, v: hint.y, width: 1, height: 1))
  #expect(!grid.covered[index])
  grid.mark(index)
  let next = try #require(grid.gapHint(displayTransform: identity))
  #expect(next != hint)
  for cell in grid.covered.indices { grid.mark(cell) }
  #expect(grid.gapHint(displayTransform: identity) == nil)
  #expect(grid.gapHint(displayTransform: []) == nil)
  #expect(CoverageGrid().gapHint(displayTransform: [1, 0, 0, 1, 5, 0]) == nil)
}
