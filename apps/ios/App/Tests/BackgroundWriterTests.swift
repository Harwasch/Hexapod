#if !os(iOS)
  import Foundation
  import Testing
  @testable import AppSupport

  private actor Gate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var entered: CheckedContinuation<Void, Never>?
    private var isEntered = false
    private var open = false
    func wait() async {
      isEntered = true
      entered?.resume()
      entered = nil
      if !open { await withCheckedContinuation { continuation = $0 } }
    }
    func waitForEntry() async {
      if !isEntered { await withCheckedContinuation { entered = $0 } }
    }
    func release() {
      open = true
      continuation?.resume()
      continuation = nil
    }
  }
  private actor Log {
    var values: [Int] = []
    func append(_ value: Int) { values.append(value) }
  }
  @Test func boundedWriterRejectsOverflowAndDrainsFIFOIncludingInFlightJob() async throws {
    let gate = Gate()
    let log = Log()
    let writer = BackgroundWriter<Int>(capacity: 2) { value in
      if value == 0 { await gate.wait() }
      await log.append(value)
    }
    #expect(writer.submit(0))
    await gate.waitForEntry()
    #expect(writer.submit(1))
    #expect(writer.outstanding == 2)
    #expect(!writer.submit(2))
    let closing = Task { try await writer.close() }
    await gate.release()
    try await closing.value
    #expect(await log.values == [0, 1])
    #expect(writer.outstanding == 0)
    #expect(!writer.submit(3))
  }
  @Test func writerFailureClosesIngressAndIsReportedByDrain() async {
    struct Failure: Error {}
    let gate = Gate()
    let writer = BackgroundWriter<Int>(capacity: 2) { _ in
      await gate.wait()
      throw Failure()
    }
    #expect(writer.submit(0))
    await gate.waitForEntry()
    #expect(writer.submit(1))
    await gate.release()
    await #expect(throws: Failure.self) { try await writer.close() }
    #expect(writer.outstanding == 0)
    #expect(!writer.submit(2))
  }
#endif
#if !os(iOS)
  import Capture
  @Test func backgroundDiskWriterPublishesManifestAfterOrderedDepthAndFrameWrites() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let disk = try await RecordingDiskWriter.create(directory: root, device: "synthetic-test")
    let writer = BackgroundWriter<RecordingPacket>(capacity: 4) { try await disk.write($0) }
    let identity: [Double] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    for index in 0..<2 {
      let name = "depth_\(index)"
      let frame = CaptureFrame(
        index: index, timestamp: Double(index), transform: identity,
        intrinsics: [1, 0, 0, 0, 1, 0, 0, 0, 1], imageWidth: 1, imageHeight: 1,
        trackingState: "normal",
        depth: .init(file: name + ".f16", confidence: name + ".u8", width: 1, height: 1))
      #expect(
        writer.submit(
          .frame(
            frame,
            [
              .init(name: name + ".f16", data: Data([0, 60])),
              .init(name: name + ".u8", data: Data([2])),
            ])))
    }
    try await writer.close()
    #expect(
      !FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.json").path))
    // A fake finalized video is sufficient to exercise bundle publication, not codec validity.
    try Data([1]).write(to: root.appendingPathComponent("rgb.mov"))
    let siteID = UUID()
    try await disk.writeDetails(CaptureDetails(siteName: "Field", siteID: siteID, duration: 1))
    try await disk.finish()
    let local = LocalCapture(directory: root, upload: nil)
    #expect(local.details?.siteID == siteID && local.details?.siteName == "Field")
    let archive = root.deletingLastPathComponent().appendingPathComponent(
      root.lastPathComponent + ".hexcap")
    defer { try? FileManager.default.removeItem(at: archive) }
    try CaptureArchive.write(bundle: root, to: archive)
    let archived = try CaptureArchive.readManifest(from: archive)
    #expect(archived.frameCount == 2 && archived.depthFrameCount == 2)
    #expect(archived.files.contains { $0.path == "capture-info.json" })
    let manifest = try JSONDecoder().decode(
      CaptureManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
    #expect(manifest.frameCount == 2 && manifest.depthFrameCount == 2)
    let frames = try Data(contentsOf: root.appendingPathComponent("frames.jsonl")).split(
      separator: 10
    )
    .map { try JSONDecoder().decode(CaptureFrame.self, from: Data($0)) }
    #expect(frames.map(\.index) == [0, 1])
    #expect(manifest.files.contains { $0.path == "depth_1.f16" && $0.bytes == 2 })
  }
#endif
