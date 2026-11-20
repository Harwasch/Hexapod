import Foundation
import Testing

@testable import Capture

@Test(arguments: ["yard-small", "yard-limited"])
func sharedSyntheticReplay(name: String) throws {
  let fixtures = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().appendingPathComponent("Fixtures/bundles")
  let fixture = fixtures.appendingPathComponent(name)
  let bundle = fixture.appendingPathComponent("bundle")
  let decoder = JSONDecoder()
  let manifest = try decoder.decode(
    CaptureManifest.self, from: Data(contentsOf: bundle.appendingPathComponent("manifest.json")))
  let lines = try String(
    contentsOf: bundle.appendingPathComponent(manifest.frameLog), encoding: .utf8
  ).split(separator: "\n")
  let frames = try lines.map { try decoder.decode(CaptureFrame.self, from: Data($0.utf8)) }
  #expect(manifest.version == 1)
  #expect(manifest.format == "hex-capture")
  #expect(manifest.device == "SYNTHETIC")
  #expect(manifest.frameCount == 24)
  #expect(frames.count == manifest.frameCount)
  #expect(manifest.depthFrameCount == frames.count)
  #expect(manifest.smoothedDepthFrameCount == 0)
  #expect(frames.prefix(2).allSatisfy { $0.trackingState == "limited:initializing" })
  #expect(frames.filter { $0.trackingState == "normal" }.count == (name == "yard-small" ? 22 : 3))
  for (index, frame) in frames.enumerated() {
    #expect(frame.index == index)
    #expect(frame.timestamp == Double(index) / 12)
    #expect(frame.transform.count == 16)
    #expect(frame.intrinsics.count == 9)
    #expect(frame.imageWidth == 192 && frame.imageHeight == 144)
    let depth = try #require(frame.depth)
    #expect(depth.width == 48 && depth.height == 36)
    #expect(try Data(contentsOf: bundle.appendingPathComponent(depth.file)).count == 48 * 36 * 2)
    #expect(try Data(contentsOf: bundle.appendingPathComponent(depth.confidence)).count == 48 * 36)
    #expect(frame.smoothedDepth == nil)
  }
  let locationLog = try #require(manifest.locationLog)
  let locations = try String(
    contentsOf: bundle.appendingPathComponent(locationLog), encoding: .utf8
  ).split(separator: "\n").map {
    try decoder.decode(CaptureLocation.self, from: Data($0.utf8))
  }
  #expect(locations.count == 2)
  #expect(manifest.location == locations.first)
  for file in manifest.files {
    #expect(try UInt64(Data(contentsOf: bundle.appendingPathComponent(file.path)).count) == file.bytes)
  }
  #expect(try CaptureArchive.readManifest(from: fixture.appendingPathComponent("capture.hexcap")) == manifest)
}
