import Foundation
import Testing

@testable import LiveSplat

private func fixture(_ name: String) -> URL {
  URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/bundles/\(name)")
}

private struct GroundTruth: Decodable {
  struct Object: Decodable {
    let label: String
    let centre: [Double]
    let extents: [Double]

    func distance(to p: SplatVector) -> Double {
      let d = p - SplatVector(centre[0], centre[1], centre[2])
      switch label {
      case "crown": return abs(sqrt(d.squaredLength) - extents[0] / 2)
      case "trunk":
        // Distance to a capped cylinder, including its rims and end disks.
        let radial = sqrt(d.x * d.x + d.z * d.z) - extents[0] / 2
        let vertical = abs(d.y) - extents[1] / 2
        return abs(hypot(max(radial, 0), max(vertical, 0)) + min(max(radial, vertical), 0))
      default:
        // Box signed distance; zero-height ground is a bounded rectangle.
        let q = zip(d.values, extents).map { abs($0) - $1 / 2 }
        let outside = sqrt(q.reduce(0) { $0 + pow(max($1, 0), 2) })
        return abs(outside + min(q.max()!, 0))
      }
    }
  }
  struct Probe: Decodable {
    let frame: Int
    let label: String
    let pixel: [Int]
    let world: [Double]
  }
  let objects: [Object]
  let probes: [Probe]
  let normalFrames: Int
}

// IEEE binary16 has 10 fraction bits. Half an ULP is the maximum rounding error
// in optical-axis depth. Unprojection scales it by the unit-depth ray length.
// Integer pixel centres are exact: no half-pixel positional allowance is needed.
// 1e-10 covers Double pose/unprojection roundoff, as in the existing math tests.
private func depthTolerance(_ f: SplatFrame, x: Int, y: Int, depth: Double) -> Double {
  let halfULP = pow(2, floor(log2(depth)) - 10) / 2
  let ray =
    f.camera.unproject(
      u: Double(x) * Double(f.camera.width) / Double(f.width),
      v: Double(y) * Double(f.camera.height) / Double(f.height), depth: 1) - f.camera.position
  return halfULP * sqrt(ray.squaredLength) + 1e-10
}

@Test @MainActor func replayYardSmallSeedsGroundTruthAndCoverage() async throws {
  let root = fixture("yard-small")
  let truth = try JSONDecoder().decode(
    GroundTruth.self, from: Data(contentsOf: root.appendingPathComponent("ground_truth.json")))
  let source: any SplatFrameSource = try BundleReplayFrameSource(
    bundleDirectory: root.appendingPathComponent("bundle"), decodeRGB: false)
  let renderer = DefaultLiveSplatRenderer()
  renderer.start()
  var index = 0
  var tracked = 0
  var firstCoverage = 0.0
  var checkedProbes: Set<String> = []
  var surfaceHits: Set<String> = []
  while let event = try await source.next() {
    renderer.ingest(event)
    defer { index += 1 }
    guard case .frame(let frame) = event else {
      #expect(renderer.splats.isEmpty)
      #expect(renderer.coverageFraction == 0)
      continue
    }
    tracked += 1
    #expect(frame.width == 48 && frame.height == 36)
    #expect(frame.colors.allSatisfy { $0 == BundleReplayFrameSource.flatColor })
    #expect(frame.depths.contains(.infinity))
    #expect(frame.confidence.contains(0) && frame.confidence.contains(2))
    if tracked == 1 { firstCoverage = renderer.coverageFraction }
    for y in 0..<frame.height {
      for x in 0..<frame.width {
        guard let seed = frame.seed(x: x, y: y) else { continue }
        let tolerance = depthTolerance(
          frame, x: x, y: y, depth: Double(frame.depths[y * frame.width + x]))
        let nearest = try #require(
          truth.objects.min {
            $0.distance(to: seed.position) < $1.distance(to: seed.position)
          })
        #expect(nearest.distance(to: seed.position) <= tolerance)
        surfaceHits.insert(nearest.label)
      }
    }
    for probe in truth.probes where probe.frame == index {
      let x = probe.pixel[0]
      let y = probe.pixel[1]
      let seed = try #require(frame.seed(x: x, y: y))
      let exact = SplatVector(probe.world[0], probe.world[1], probe.world[2])
      let tolerance = depthTolerance(
        frame, x: x, y: y, depth: Double(frame.depths[y * frame.width + x]))
      #expect(sqrt((seed.position - exact).squaredLength) <= tolerance)
      checkedProbes.insert(probe.label)
    }
    // The retained pipeline seeds must also remain on surfaces; bound all rays
    // by the farthest image corner and the seeding cap of 8 metres (half ULP=1/256 m).
    let tolerance = depthTolerance(frame, x: 0, y: 0, depth: 8)
    #expect(
      renderer.splats.allSatisfy { point in
        truth.objects.contains { $0.distance(to: point.position) <= tolerance }
      })
    var independentCoverage = CoverageGrid()
    for seed in renderer.splats { _ = independentCoverage.observe(seed, in: frame) }
    #expect(renderer.coverage.covered == independentCoverage.covered)
  }
  #expect(index == 24)
  #expect(tracked == truth.normalFrames)
  #expect(checkedProbes == Set(["ground", "shed", "trunk", "crown"]))
  #expect(surfaceHits == checkedProbes)
  #expect(firstCoverage > 0)
  #expect(renderer.coverageFraction > firstCoverage)
  #expect(renderer.splats.count > renderer.configuration.maxSeedsPerFrame)
  #expect(try await source.next() == nil)
}

@Test @MainActor func replayLimitedFramesNeverSeedAndClearCoverage() async throws {
  let root = fixture("yard-limited")
  let truth = try JSONDecoder().decode(
    GroundTruth.self, from: Data(contentsOf: root.appendingPathComponent("ground_truth.json")))
  let source = try BundleReplayFrameSource(bundleDirectory: root.appendingPathComponent("bundle"))
  let renderer = DefaultLiveSplatRenderer()
  renderer.start()
  var accepted = 0
  var skipped = 0
  while let event = try await source.next() {
    let before = renderer.splats
    renderer.ingest(event)
    switch event {
    case .frame:
      accepted += 1
      #expect(renderer.coverageFraction > 0)
    case .trackingUnavailable:
      skipped += 1
      #expect(renderer.splats == before)
      #expect(renderer.coverageFraction == 0)
      #expect(renderer.latestFrame == nil)
      #expect(renderer.recentKeyframes.isEmpty)
    }
  }
  #expect(accepted == truth.normalFrames && accepted == 3)
  #expect(skipped == 21)
  #expect(!renderer.splats.isEmpty)
  #expect(source.consumedFrameCount == source.frameCount)
  source.reset()
  #expect(source.consumedFrameCount == 0)
  guard case .trackingUnavailable = try await source.next() else {
    Issue.record("Reset must return the initializing frame")
    return
  }
}

@Test @MainActor func fakeCoverageIsDeterministicCappedAndRestartable() async throws {
  let source: any SplatFrameSource = try FakeFrameSource.plane()
  let renderer: any LiveSplatRenderer = FakeLiveSplat(cellsPerFrame: 64)
  let first = try #require(try await source.next())
  renderer.ingest(first)
  #expect(renderer.coverageFraction == 0)
  var runs: [[Double]] = []
  for _ in 0..<2 {
    source.reset()
    renderer.start()
    var fractions: [Double] = []
    while let event = try await source.next() {
      renderer.ingest(event)
      let fraction = renderer.coverageFraction
      renderer.ingest(event)  // Duplicate timestamp must not advance the fake.
      #expect(renderer.coverageFraction == fraction)
      fractions.append(fraction)
    }
    runs.append(fractions)
    renderer.stop()
    renderer.ingest(.trackingUnavailable)
    #expect(renderer.coverageFraction == 1)
    #expect(renderer.status == .stopped)
  }
  #expect(runs[0] == runs[1])
  #expect(runs[0] == (1...16).map { Double(min($0 * 64, 768)) / 768 })
  renderer.start()
  renderer.ingest(first)
  renderer.ingest(.trackingUnavailable)
  #expect(renderer.coverageFraction == 0)
  renderer.ingest(first)
  #expect(renderer.coverageFraction == 64.0 / 768)
}

@Test @MainActor func fakeSourcePreservesLossEventsAndEOF() async throws {
  let source: any SplatFrameSource = FakeFrameSource(events: [.trackingUnavailable])
  guard case .trackingUnavailable = try await source.next() else {
    Issue.record("Fake must preserve tracking loss")
    return
  }
  #expect(try await source.next() == nil)
  source.reset()
  #expect(try await source.next() != nil)
}

@Test @MainActor func replayResetProducesIdenticalGeometry() async throws {
  let source = try BundleReplayFrameSource(
    bundleDirectory: fixture("yard-small").appendingPathComponent("bundle"), decodeRGB: false)
  let renderer = DefaultLiveSplatRenderer()
  var snapshots: [[Gaussian]] = []
  for _ in 0..<2 {
    renderer.start()
    source.reset()
    while let event = try await source.next() { renderer.ingest(event) }
    snapshots.append(renderer.splats)
  }
  #expect(snapshots[0] == snapshots[1])
}

@Test @MainActor func rendererProtocolSharesLifecycleAndExportGuards() async throws {
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let renderers: [any LiveSplatRenderer] = [FakeLiveSplat(), DefaultLiveSplatRenderer()]
  for renderer in renderers {
    let source = try FakeFrameSource.plane(frameCount: 1)
    renderer.start()
    renderer.ingest(try #require(try await source.next()))
    #expect(throws: LiveSplatError.exportWhileRunning) { try renderer.exportPLY(to: directory) }
    let coverage = renderer.coverageFraction
    renderer.stop()
    renderer.ingest(.trackingUnavailable)
    #expect(renderer.coverageFraction == coverage)
    let output = try renderer.exportPLY(to: directory)
    #expect(try Data(contentsOf: output) == SplatPLY.encode(renderer.splats))
    let manifest = directory.appendingPathComponent("manifest.json")
    try Data("{}".utf8).write(to: manifest)
    #expect(throws: LiveSplatError.bundleFinalized) { try renderer.exportPLY(to: directory) }
    try FileManager.default.removeItem(at: manifest)
  }
}

@Test @MainActor func replayRejectsMalformedRecordsAndPayloadsWithoutAdvancing() async throws {
  let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: directory) }
  let fixtureBundle = fixture("yard-small").appendingPathComponent("bundle")
  let log = try String(
    contentsOf: fixtureBundle.appendingPathComponent("frames.jsonl"), encoding: .utf8)
  let json = try #require(log.split(separator: "\n").first)
  var record = try #require(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
  record["trackingState"] = "normal"
  func write(_ value: [String: Any]) throws {
    try JSONSerialization.data(withJSONObject: value).write(
      to: directory.appendingPathComponent("frames.jsonl"))
  }
  var bad = record
  bad["transform"] = [1, 0]
  try write(bad)
  #expect(throws: BundleReplayError.invalidRecord(0)) {
    try BundleReplayFrameSource(bundleDirectory: directory, decodeRGB: false)
  }
  var depth = try #require(record["depth"] as? [String: Any])
  depth["file"] = "../outside.f16"
  bad = record
  bad["depth"] = depth
  try write(bad)
  let traversal = try BundleReplayFrameSource(bundleDirectory: directory, decodeRGB: false)
  await #expect(throws: BundleReplayError.unsafePath("../outside.f16")) {
    try await traversal.next()
  }
  #expect(traversal.consumedFrameCount == 0)

  try write(record)
  for name in ["depth_00000000.f16", "depth_00000000.u8"] {
    try FileManager.default.copyItem(
      at: fixtureBundle.appendingPathComponent(name), to: directory.appendingPathComponent(name))
  }
  let shortFile = directory.appendingPathComponent("depth_00000000.f16")
  try Data([0]).write(to: shortFile)
  let truncated = try BundleReplayFrameSource(bundleDirectory: directory, decodeRGB: false)
  await #expect(throws: BundleReplayError.invalidPayload("depth_00000000.f16")) {
    try await truncated.next()
  }
  #expect(truncated.consumedFrameCount == 0)
  try Data(contentsOf: fixtureBundle.appendingPathComponent("depth_00000000.f16")).write(
    to: shortFile)
  try Data(repeating: 3, count: 48 * 36).write(
    to: directory.appendingPathComponent("depth_00000000.u8"))
  await #expect(throws: BundleReplayError.invalidPayload("depth_00000000.u8")) {
    try await truncated.next()
  }
  #expect(truncated.consumedFrameCount == 0)
}

#if canImport(AVFoundation) && canImport(CoreGraphics)
  import CoreGraphics

  @Test @MainActor func replayAppleRGBPreservesRowsAndIntegerPixelCentres() throws {
    // Distinct colours at the four sampled corners detect vertical flips, channel
    // swaps and half-pixel/resizing filters without any codec error tolerance.
    var pixels = [UInt8](repeating: 0, count: 4 * 4 * 4)
    for i in 0..<16 { pixels[4 * i + 3] = 255 }
    let samples: [(Int, [UInt8], SplatVector)] = [
      (0, [255, 0, 0, 255], SplatVector(1, 0, 0)),
      (2, [0, 255, 0, 255], SplatVector(0, 1, 0)),
      (8, [0, 0, 255, 255], SplatVector(0, 0, 1)),
      (10, [255, 255, 255, 255], SplatVector(1, 1, 1)),
    ]
    for (index, color, _) in samples {
      pixels.replaceSubrange((4 * index)..<(4 * index + 4), with: color)
    }
    let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
    let image = try #require(
      CGImage(
        width: 4, height: 4, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 16,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
          .union(.byteOrder32Big),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    #expect(try ReplayRGB.sample(image: image, width: 2, height: 2) == samples.map { $0.2 })
  }

  @Test @MainActor func replayAppleRGBMatchesSyntheticColorClasses() async throws {
    let root = fixture("yard-small")
    let truth = try JSONDecoder().decode(
      GroundTruth.self, from: Data(contentsOf: root.appendingPathComponent("ground_truth.json")))
    let source = try BundleReplayFrameSource(bundleDirectory: root.appendingPathComponent("bundle"))
    var checked: Set<String> = []
    var index = 0
    while let event = try await source.next() {
      defer { index += 1 }
      guard case .frame(let frame) = event else { continue }
      for probe in truth.probes where probe.frame == index {
        let color = frame.colors[probe.pixel[1] * frame.width + probe.pixel[0]]
        // Channel ordering, not a fitted codec error tolerance. Crown is green;
        // the other fixture surfaces have red > green > blue.
        if probe.label == "crown" {
          #expect(color.y > color.x && color.y > color.z)
        } else {
          #expect(color.x > color.y && color.y > color.z)
        }
        checked.insert(probe.label)
      }
    }
    #expect(checked == Set(["ground", "shed", "trunk", "crown"]))
  }
#endif
