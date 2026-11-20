import Foundation
import Testing

@testable import LiveSplat

// Bounds chosen from Double roundoff and central-difference truncation, not fitted
// to device arithmetic: analytic 1e-10; finite differences h=1e-5, error 2e-6.
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

@Test func allParameterGradientsMatchIndependentFiniteDifferences() throws {
  var g = try gaussian(position: SplatVector(0.06, -0.04, -2))
  g.parameters.replaceSubrange(6..<10, with: [0.9, 0.15, -0.2, 0.3])
  let c = try camera(rotation: SplatMatrix([0.96, 0, 0.28, 0, 1, 0, -0.28, 0, 0.96]))
  let p = try #require(SplatMath.project(g, camera: c))
  let samples = [
    PhotometricSample(
      u: p.u + 3, v: p.v - 4, target: SplatVector(0.2, 0.4, 0.7),
      background: SplatVector(0.1, 0.05, 0.2)),
    PhotometricSample(u: p.u - 2, v: p.v + 2, target: SplatVector(0.6, 0.1, 0.2)),
  ]
  let gradient = SplatMath.gradient(g, camera: c, samples: samples)
  let h = 1e-5
  for i in 0..<14 {
    var plus = g
    var minus = g
    plus.parameters[i] += h
    minus.parameters[i] -= h
    let fd =
      (SplatMath.loss(plus, camera: c, samples: samples)
        - SplatMath.loss(minus, camera: c, samples: samples)) / (2 * h)
    near(gradient[i], fd, 2e-6)
  }
  let updated = SplatMath.step(g, gradient: gradient, learningRate: 1e-4)
  #expect(
    SplatMath.loss(updated, camera: c, samples: samples)
      < SplatMath.loss(g, camera: c, samples: samples))
}

@Test func gradientCenterColorAndOpacityHandValues() throws {
  let g = try Gaussian(
    position: SplatVector(0, 0, -2), scale: SplatVector(0.1, 0.1, 0.1), opacity: 0.5,
    color: SplatVector(0.8, 0.4, 0.2))
  let samples = [PhotometricSample(u: 32, v: 24, target: .zero)]
  near(SplatMath.loss(g, camera: try camera(), samples: samples), 0.105)
  let grad = SplatMath.gradient(g, camera: try camera(), samples: samples)
  near(grad[10], 0.105)
  near(grad[11], 0.2)
  near(grad[12], 0.1)
  near(grad[13], 0.05)
  for i in 0..<10 { near(grad[i], 0) }
}

@Test func invalidGradientCannotPoisonState() throws {
  let g = try gaussian()
  #expect(SplatMath.step(g, gradient: Array(repeating: .nan, count: 14)) == g)
  #expect(SplatMath.step(g, gradient: [0]) == g)
  #expect(SplatMath.step(g, gradient: Array(repeating: 1, count: 14), learningRate: .infinity) == g)
  let bounded = SplatMath.step(g, gradient: Array(repeating: 1e30, count: 14))
  #expect(bounded.isValid)
  near(bounded.position.x, g.position.x - 0.001)
}

@Test func alphaCompositingIsFrontToBack() {
  let (color, alpha) = SplatMath.composite([
    (SplatVector(1, 0, 0), 0.5), (SplatVector(0, 0, 1), 0.5),
  ])
  near(color.x, 0.5)
  near(color.z, 0.25)
  near(alpha, 0.75)
  let (_, emptyAlpha) = SplatMath.composite([])
  near(emptyAlpha, 0)
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

@Test @MainActor func rendererCapsSeedsPrunesAndResets() throws {
  var config = LiveSplatConfiguration()
  config.maxSplats = 5
  config.maxSeedsPerFrame = 3
  config.maxAgeFrames = 1
  let r = DefaultLiveSplatRenderer(configuration: config)
  r.ingest(try frame())
  #expect(r.splats.isEmpty)
  r.start()
  r.ingest(try frame())
  #expect(r.splats.count == 3)
  r.ingest(try frame(timestamp: 2))
  #expect(r.splats.count == 5)
  let version = r.generation
  r.ingest(try frame(timestamp: 2))
  #expect(r.generation == version)
  r.ingest(try frame(timestamp: 3, confidence: 0))
  #expect(r.splats.count == 5)
  r.ingest(try frame(timestamp: 4, confidence: 0))
  #expect(r.splats.isEmpty)
  for i in 5..<20 { r.ingest(try frame(timestamp: Double(i))) }
  #expect(r.recentKeyframes.count == 3)
  #expect(r.splats.count == 5)
  r.stop()
  #expect(r.status == .stopped)
  r.start()
  #expect(r.splats.isEmpty)
  near(r.coverageFraction, 0)
}

@Test @MainActor func stationaryCoverageDoesNotReseedSameCells() throws {
  let r = DefaultLiveSplatRenderer()
  r.start()
  r.ingest(try frame())
  let first = r.splats.count
  for i in 2..<10 { r.ingest(try frame(timestamp: Double(i))) }
  #expect(first == 128)
  #expect(r.splats.count == 768)
  near(r.coverageFraction, 1)
  r.ingest(try frame(timestamp: 10))
  #expect(r.splats.count == 768)
}

@Test @MainActor func gpuResultsFollowStableIDsAndRejectPreviousSessions() throws {
  var config = LiveSplatConfiguration()
  config.maxAgeFrames = 1
  let r = DefaultLiveSplatRenderer(configuration: config)
  r.start()
  r.ingest(try frame())
  r.activateGaussian()
  let session = r.sessionGeneration
  let ids = r.splatIDs
  var changed = r.splats
  changed[0].parameters[11] = 0.2
  r.ingest(try frame(timestamp: 2))
  r.acceptGPU(changed, ids: ids, seconds: 0.001, session: session)
  near(r.splats[0].color.x, 0.2)  // A newer camera view does not discard training.
  r.ingest(try frame(timestamp: 3, confidence: 0))
  r.ingest(try frame(timestamp: 4, confidence: 0))
  r.ingest(try frame(timestamp: 5))
  let newSeeds = r.splats
  r.acceptGPU(changed, ids: ids, seconds: 0.001, session: session)
  #expect(r.splats == newSeeds)  // Removed IDs cannot overwrite newly inserted seeds.
  r.stop()
  r.start()
  r.ingest(try frame())
  r.activateGaussian()
  let restarted = r.splats
  r.acceptGPU(changed, ids: ids, seconds: 0.001, session: session)
  #expect(r.splats == restarted)
}

@Test func budgetFailsClosedOnOverrunOrMissingTiming() {
  var b = SplatBudget()
  b.record(seconds: 0.007)
  #expect(b.optimizationSteps == 2)
  b.record(seconds: 0.009)
  #expect(b.exceeded)
  #expect(b.optimizationSteps == 0)
  b.record(seconds: 0.001)
  #expect(b.exceeded)
  var missing = SplatBudget()
  missing.record(seconds: 0)
  #expect(missing.exceeded)
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

@Test func tileBinningRetainsNearestAndReportsOverflow() {
  var projections: [ProjectedGaussian?] = (0..<70).map {
    ProjectedGaussian(u: 8, v: 8, depth: Double(70 - $0), xx: 1, xy: 0, yy: 1)
  }
  projections += [nil, ProjectedGaussian(u: 100, v: 100, depth: 0.1, xx: 1, xy: 0, yy: 1)]
  let tile = SplatTile.bin(projections, column: 0, row: 0)
  #expect(tile.indices == Array((6..<70).reversed()))
  #expect(tile.overflow == 6)
  let tied = ProjectedGaussian(u: 8, v: 8, depth: 1, xx: 1, xy: 0, yy: 1)
  #expect(SplatTile.bin([tied, tied], column: 0, row: 0).indices == [0, 1])
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

@Test @MainActor func overBudgetGPUResultTriggersFallbackEvenAfterNewCameraFrame() throws {
  let r = DefaultLiveSplatRenderer()
  r.start()
  r.ingest(try frame())
  r.activateGaussian()
  let snapshot = r.splats
  let ids = r.splatIDs
  let session = r.sessionGeneration
  r.ingest(try frame(timestamp: 2))
  r.acceptGPU(snapshot, ids: ids, seconds: 0.009, session: session)
  #expect(r.status == .fallback("GPU budget exceeded"))
  #expect(r.budget.optimizationSteps == 0)
  r.start()
  #expect(!r.budget.exceeded)
  #expect(r.status == .training)
}

@Test @MainActor func trackingLossClearsStaleOverlayAndKeyframes() throws {
  let r = DefaultLiveSplatRenderer()
  r.start()
  r.ingest(try frame())
  r.trackingUnavailable()
  #expect(r.latestFrame == nil)
  #expect(r.recentKeyframes.isEmpty)
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
