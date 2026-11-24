import Capture
import Foundation
import Testing

@testable import RecorderSupport

@Test func generateRealWriterBundles() async throws {
  let root = try URL(
    fileURLWithPath: #require(ProcessInfo.processInfo.environment["SWIFT_BUNDLE_OUT"]))
  for seconds in [1, 18, 48] {
    for details in [false, true] {
      let name = "\(seconds)s-" + (details ? "details" : "control")
      let directory = root.appendingPathComponent(name)
      if FileManager.default.fileExists(atPath: directory.path) {
        try FileManager.default.removeItem(at: directory)
      }
      let archive = root.appendingPathComponent(name + ".hexcap")
      if FileManager.default.fileExists(atPath: archive.path) {
        try FileManager.default.removeItem(at: archive)
      }
      let disk = try await RecordingDiskWriter.create(
        directory: directory, device: "Linux synthetic test")
      let writer = BackgroundWriter<RecordingPacket>(capacity: 4) { try await disk.write($0) }
      var cadence = CaptureCadence()
      var count = 0
      for tick in 0..<(seconds * 60) {
        let time = 10_000 + Double(tick) / 60
        guard cadence.admit(time) else { continue }
        var files: [RecordedFile] = []
        func depth(_ prefix: String) -> DepthReference {
          let stem = String(format: "%@_%08d", prefix, count)
          files.append(.init(name: stem + ".f16", data: Data([0, 60, 0, 64])))
          files.append(.init(name: stem + ".u8", data: Data([2, 2])))
          return DepthReference(file: stem + ".f16", confidence: stem + ".u8", width: 2, height: 1)
        }
        let raw = depth("depth")
        let smooth = depth("smooth")
        let frame = CaptureFrame(
          index: count, timestamp: time - 10_000,
          transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
          intrinsics: [32, 0, 16, 0, 32, 16, 0, 0, 1],
          imageWidth: 32, imageHeight: 32, trackingState: "normal",
          depth: raw, smoothedDepth: smooth)
        // Accelerate logical time without turning disk speed into artificial capture drops.
        while !writer.submit(.frame(frame, files)) {
          try #require(!writer.hasFailed)
          await Task.yield()
        }
        count += 1
      }
      let fix = CaptureLocation(
        timestamp: 1_800_000_000, latitude: 37, longitude: -122,
        altitudeM: 100, horizontalAccuracyM: 5, verticalAccuracyM: 10)
      while !writer.submit(.location(fix)) {
        try #require(!writer.hasFailed)
        await Task.yield()
      }
      try await writer.close()
      #expect(count == seconds * 15)
      #expect(writer.outstanding == 0)
      #expect(
        !FileManager.default.fileExists(
          atPath: directory.appendingPathComponent("manifest.json").path))
      // read_bundle never decodes RGB: AVFoundation is replaced only at this boundary.
      try Data("synthetic finalized video; not codec validation".utf8).write(
        to: directory.appendingPathComponent("rgb.mov"))
      if details {
        try await disk.writeDetails(
          CaptureDetails(siteName: "Synthetic Linux test", siteID: nil, duration: Double(seconds)))
      }
      try await disk.finish()
      try CaptureArchive.write(bundle: directory, to: archive)
      let manifest = try CaptureArchive.readManifest(from: archive)
      #expect(manifest.frameCount == count)
      #expect(manifest.depthFrameCount == count && manifest.smoothedDepthFrameCount == count)
      let lines = try Data(contentsOf: directory.appendingPathComponent("frames.jsonl")).split(
        separator: 10)
      #expect(lines.count == count)
      print("GENERATED \(name): frames=\(count), inventory=\(manifest.files.count)")
    }
  }
}
