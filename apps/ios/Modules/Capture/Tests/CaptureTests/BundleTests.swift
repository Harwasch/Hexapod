import Foundation
import Testing

@testable import Capture

@Test func roundTripAndArchive() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let folder = root.appendingPathComponent("capture")
  let writer = try CaptureBundleWriter(
    directory: folder, device: "fixture", startedAt: Date(timeIntervalSince1970: 0))
  try Data([1, 2, 3]).write(to: folder.appendingPathComponent("rgb.mov"))
  let frame = CaptureFrame(
    index: 0, timestamp: 0,
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    intrinsics: [100, 0, 50, 0, 100, 40, 0, 0, 1], imageWidth: 100, imageHeight: 80,
    trackingState: "normal")
  try writer.append(frame)
  #expect(throws: CaptureError.self) { try writer.append(frame) }
  let manifest = try writer.finish()
  #expect(
    manifest
      == (try JSONDecoder().decode(
        CaptureManifest.self, from: Data(contentsOf: folder.appendingPathComponent("manifest.json"))
      )))
  #expect(
    frame
      == (try JSONDecoder().decode(
        CaptureFrame.self, from: Data(contentsOf: folder.appendingPathComponent("frames.jsonl")))))
  let archive = root.appendingPathComponent("capture.hexcap")
  try CaptureArchive.write(bundle: folder, to: archive)
  let data = try Data(contentsOf: archive)
  #expect(data.count % 512 == 0)
  #expect(String(decoding: data[257..<262], as: UTF8.self) == "ustar")
  #expect(data.suffix(1024).allSatisfy { $0 == 0 })
  #if os(Linux)
    let tar = Process()
    tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
    tar.arguments = ["-tf", archive.path]
    let pipe = Pipe()
    tar.standardOutput = pipe
    try tar.run()
    let listing = pipe.fileHandleForReading.readDataToEndOfFile()
    tar.waitUntilExit()
    #expect(tar.terminationStatus == 0)
    #expect(String(decoding: listing, as: UTF8.self) == "manifest.json\nframes.jsonl\nrgb.mov\n")
  #endif
  #expect(manifest.frameCount == 1)
  #expect(throws: CaptureError.self) { try writer.append(frame) }
}

@Test func rejectsMissingDepthAndUnsafeNames() throws {
  #expect(!CaptureBundleWriter.safeName("../depth.f16"))
  #expect(!CaptureBundleWriter.safeName("/absolute"))
  let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: folder) }
  let writer = try CaptureBundleWriter(directory: folder, device: "fixture")
  let frame = CaptureFrame(
    index: 0, timestamp: 0, transform: Array(repeating: 0, count: 16),
    intrinsics: Array(repeating: 0, count: 9), imageWidth: 1, imageHeight: 1,
    trackingState: "normal",
    depth: DepthReference(file: "missing.f16", confidence: "missing.u8", width: 1, height: 1))
  #expect(throws: (any Error).self) { try writer.append(frame) }
  #expect(throws: CaptureError.self) { try writer.finish() }
}
