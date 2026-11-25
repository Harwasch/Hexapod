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
  #expect(try CaptureArchive.readManifest(from: archive) == manifest)
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

@Test func locationLogAndSummaryRoundTrip() throws {
  let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: folder) }
  let writer = try CaptureBundleWriter(directory: folder, device: "fixture")
  try Data([1]).write(to: folder.appendingPathComponent("rgb.mov"))
  try writer.append(
    CaptureFrame(
      index: 0, timestamp: 0, transform: Array(repeating: 0, count: 16),
      intrinsics: Array(repeating: 0, count: 9), imageWidth: 1, imageHeight: 1,
      trackingState: "normal"))
  let fixes = [-1.0, 20, 5, 5].enumerated().map { index, accuracy in
    CaptureLocation(
      timestamp: 1_800_000_000 + Double(index), latitude: 37 + Double(index) / 1000,
      longitude: -122, altitudeM: 100, horizontalAccuracyM: accuracy, verticalAccuracyM: -1)
  }
  for fix in fixes { try writer.appendLocation(fix) }
  let manifest = try writer.finish()
  let lines = try String(
    contentsOf: folder.appendingPathComponent("location.jsonl"), encoding: .utf8
  )
  .split(separator: "\n")
  let decoded = try lines.map {
    try JSONDecoder().decode(CaptureLocation.self, from: Data($0.utf8))
  }
  #expect(decoded == fixes)
  #expect(manifest.locationLog == "location.jsonl")
  #expect(manifest.location == fixes[2])
  #expect(manifest.files.contains { $0.path == "location.jsonl" })
  #expect(throws: CaptureError.self) { try writer.appendLocation(fixes[0]) }
  var invalid = fixes[0]
  invalid.lat = 91
  #expect(!invalid.isValid)
  let archive = folder.deletingLastPathComponent().appendingPathComponent(
    UUID().uuidString + ".hexcap")
  defer { try? FileManager.default.removeItem(at: archive) }
  try CaptureArchive.write(bundle: folder, to: archive)
  #expect(try CaptureArchive.readManifest(from: archive) == manifest)
}

@Test func manifestReaderSkipsPayloadAndRejectsBrokenArchives() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let folder = root.appendingPathComponent("bundle")
  let writer = try CaptureBundleWriter(directory: folder, device: "fixture")
  try Data([1]).write(to: folder.appendingPathComponent("rgb.mov"))
  try writer.append(
    CaptureFrame(
      index: 0, timestamp: 0, transform: Array(repeating: 0, count: 16),
      intrinsics: Array(repeating: 0, count: 9), imageWidth: 1, imageHeight: 1,
      trackingState: "normal"))
  let manifest = try writer.finish()
  let archive = root.appendingPathComponent("source.hexcap")
  try CaptureArchive.write(bundle: folder, to: archive)
  let data = try Data(contentsOf: archive)
  let size = try Data(contentsOf: folder.appendingPathComponent("manifest.json")).count
  let firstMemberBytes = 512 + ((size + 511) / 512) * 512
  // USTAR does not require the manifest to precede the payload.
  let reordered =
    Data(data[firstMemberBytes..<(data.count - 1024)])
    + Data(data.prefix(firstMemberBytes)) + Data(repeating: 0, count: 1024)
  try reordered.write(to: archive)
  #expect(try CaptureArchive.readManifest(from: archive) == manifest)
  var corrupt = data
  corrupt[0] ^= 1
  try corrupt.write(to: archive)
  #expect(throws: CaptureError.self) { try CaptureArchive.readManifest(from: archive) }
  try data.prefix(520).write(to: archive)
  #expect(throws: CaptureError.self) { try CaptureArchive.readManifest(from: archive) }
}

@Test func optionalLiveSplatIsReferencedBeforeManifestPublication() throws {
  let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: folder) }
  let writer = try CaptureBundleWriter(directory: folder, device: "fixture")
  try Data([1]).write(to: folder.appendingPathComponent("rgb.mov"))
  try Data("ply\n".utf8).write(to: folder.appendingPathComponent("livesplat.ply"))
  try writer.append(
    CaptureFrame(
      index: 0, timestamp: 0, transform: Array(repeating: 0, count: 16),
      intrinsics: Array(repeating: 0, count: 9), imageWidth: 1, imageHeight: 1,
      trackingState: "normal"))
  let manifest = try writer.finish()
  #expect(manifest.liveSplat == "livesplat.ply")
  #expect(manifest.files.contains { $0.path == "livesplat.ply" && $0.bytes == 4 })
  let archive = folder.deletingLastPathComponent().appendingPathComponent(
    "\(UUID().uuidString).hexcap")
  defer { try? FileManager.default.removeItem(at: archive) }
  try CaptureArchive.write(bundle: folder, to: archive)
  #expect(try CaptureArchive.readManifest(from: archive).liveSplat == "livesplat.ply")
}

@Test func exposureAndFormatRecordsRoundTripAndOldFramesDecode() throws {
  let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
  defer { try? FileManager.default.removeItem(at: root) }
  let folder = root.appendingPathComponent("capture")
  let writer = try CaptureBundleWriter(directory: folder, device: "fixture")
  writer.videoFormat = CaptureVideoFormat(kind: "4k", width: 3840, height: 2160, framesPerSecond: 30)
  writer.exposureControl = CaptureExposureControl(
    mode: "custom", whiteBalance: "locked", durationS: 0.004, iso: 640)
  try Data([1]).write(to: folder.appendingPathComponent("rgb.mov"))
  let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  let k: [Double] = [2830, 0, 1920, 0, 2830, 1080, 0, 0, 1]
  let exposed = CaptureFrame(
    index: 0, timestamp: 0, transform: identity, intrinsics: k, imageWidth: 3840,
    imageHeight: 2160, trackingState: "normal",
    exposure: CaptureExposure(
      durationS: 0.004, offsetEV: -0.3, iso: 640, whiteBalanceGains: [1.9, 1, 1.6]))
  try writer.append(exposed)
  var broken = exposed
  broken.index = 1
  broken.timestamp = 0.1
  broken.whiteBalanceGains = [1, 1]
  #expect(throws: CaptureError.invalidFrame) { try writer.append(broken) }
  broken.whiteBalanceGains = nil
  broken.exposureDurationS = 2
  #expect(throws: CaptureError.invalidFrame) { try writer.append(broken) }
  let manifest = try writer.finish()
  #expect(manifest.videoFormat?.kind == "4k")
  #expect(manifest.exposureControl?.iso == 640)
  let line = try Data(contentsOf: folder.appendingPathComponent("frames.jsonl"))
  #expect(try JSONDecoder().decode(CaptureFrame.self, from: line) == exposed)
  // A frame from an older recorder carries none of the new keys.
  let old = #"{"index":0,"timestamp":0,"transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],"intrinsics":[1,0,0,0,1,0,0,0,1],"imageWidth":1,"imageHeight":1,"trackingState":"normal"}"#
  let decoded = try JSONDecoder().decode(CaptureFrame.self, from: Data(old.utf8))
  #expect(decoded.iso == nil && decoded.exposureDurationS == nil)
  #expect(!String(decoding: try JSONEncoder().encode(decoded), as: UTF8.self).contains("iso"))
}

@Test func defaultFormatCarriesTheFourKProbeEvidence() throws {
  let evidence = CaptureFormatFallback(
    reason: "depthAspect", imageWidth: 3840, imageHeight: 2160, depthWidth: 256, depthHeight: 192)
  let format = CaptureVideoFormat(
    kind: "default", width: 1920, height: 1440, framesPerSecond: 60, fallback: evidence)
  let json = try JSONEncoder().encode(format)
  #expect(try JSONDecoder().decode(CaptureVideoFormat.self, from: json) == format)
  let control = CaptureExposureControl(
    mode: "custom", whiteBalance: "locked", durationS: 1.0 / 120, iso: 1600, shortfallEV: 0)
  #expect(
    try JSONDecoder().decode(CaptureExposureControl.self, from: JSONEncoder().encode(control))
      == control)
  // A 4K scan carries no fallback key at all.
  let plain = CaptureVideoFormat(kind: "4k", width: 3840, height: 2160, framesPerSecond: 30)
  #expect(!String(decoding: try JSONEncoder().encode(plain), as: UTF8.self).contains("fallback"))
}
