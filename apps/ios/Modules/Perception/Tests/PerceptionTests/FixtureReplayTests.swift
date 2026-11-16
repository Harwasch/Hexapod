import Foundation
import Testing

@testable import Perception

private func fixture(_ name: String = "yard-small") -> URL {
  URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("Fixtures/bundles/" + name)
}

@Test func replayDepthAndPosesAgreeWithIndependentSurfaceProbes() throws {
  for name in ["yard-small", "yard-limited"] {
    let replay = try FixtureReplay(directory: fixture(name))
    #expect(replay.frames.count == 24)
    #expect(
      replay.frames.filter { $0.trackingState == "normal" }.count == replay.truth.normalFrames)
    for probe in replay.truth.probes {
      let frame = try replay.snapshot(at: probe.frame)
      let x = probe.pixel[0]
      let y = probe.pixel[1]
      let p = try #require(
        frame.camera.worldPoint(
          pixel: .init(
            x: (Double(x) + 0.5) / Double(frame.depth.width),
            y: (Double(y) + 0.5) / Double(frame.depth.height)),
          depth: Double(frame.depth.metres[y * frame.depth.width + x])))
      // Float16 roundoff at fixture distances (<10m), independent of host arithmetic.
      for (actual, expected) in zip([p.x, p.y, p.z], probe.world) {
        #expect(abs(actual - expected) < 0.005)
      }
    }
  }
}

@Test func yardSmallMasksLiftVisibleEntitiesWithinVisibilityBounds() async throws {
  let replay = try FixtureReplay(directory: fixture())
  let scene = FakePerception.makeScene()
  for frame in replay.frames where frame.trackingState == "normal" {
    await scene.update(
      try replay.snapshot(at: frame.index), viewportToImage: [1, 0, 0, 1, 0, 0],
      capturedAt: frame.timestamp)
    let entities = try await scene.inspect(subject: "subject", count: true)
    #expect(entities.count == replay.truth.counts.values.reduce(0, +))
    let shed = try #require(entities.first { $0.label == "shed" })
    let expected = try #require(replay.truth.objects.first { $0.label == "shed" })
    // <= 0.4m lateral/height loss: up to two <=0.15m depth texels plus boundaries.
    // The rear is occluded: centre Z may lie on the front face (half depth),
    // and the measured Z extent may lose the entire hidden 1.4m depth.
    for (actual, target, tolerance) in zip3(
      [shed.anchor.world.x, shed.anchor.world.y, shed.anchor.world.z], expected.centre,
      [0.4, 0.4, expected.extents[2] / 2 + 0.15])
    {
      #expect(abs(actual - target) <= tolerance)
    }
    for (actual, target, tolerance) in zip3(
      [shed.extent.x, shed.extent.y, shed.extent.z], expected.extents,
      [0.4, 0.4, expected.extents[2] + 0.15])
    {
      #expect(abs(actual - target) <= tolerance)
    }
    let tree = try #require(entities.first { $0.label == "tree" })
    let crown = try #require(replay.truth.objects.first { $0.label == "crown" })
    #expect(abs(tree.anchor.world.x - crown.centre[0]) <= 0.4)
    // Median surface centre is crown-weighted, within one crown radius of full-tree centre.
    #expect(abs(tree.anchor.world.y - replay.truth.treeHeight / 2) <= crown.extents[1] / 2)
    #expect(abs(tree.anchor.world.z - crown.centre[2]) <= crown.extents[2] / 2 + 0.15)
    #expect(abs(tree.extent.x - crown.extents[0]) <= 0.4)
    #expect(abs(tree.extent.y - replay.truth.treeHeight) <= 0.6)
    #expect(abs(tree.extent.z - crown.extents[2]) <= crown.extents[2] + 0.15)
    #expect(tree.attrs["identifiedOnDevice"] == .bool(true))
  }
}
private func zip3(_ a: [Double], _ b: [Double], _ c: [Double]) -> [(Double, Double, Double)] {
  zip(zip(a, b), c).map { ($0.0.0, $0.0.1, $0.1) }
}
