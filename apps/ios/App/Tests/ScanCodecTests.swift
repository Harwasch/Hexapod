#if canImport(ARKit) && os(iOS)
  @preconcurrency import AVFoundation
  import Capture
  import CoreVideo
  import Foundation
  import Testing
  @testable import AppSupport

  @Test func recordedHEVCFramesMatchJSONLAndDepthPackets() async throws {
    func buffer(_ format: OSType) throws -> CVPixelBuffer {
      var value: CVPixelBuffer?
      let status = CVPixelBufferCreate(nil, 32, 32, format, nil, &value)
      #expect(status == kCVReturnSuccess)
      let result = try #require(value)
      CVPixelBufferLockBaseAddress(result, [])
      defer { CVPixelBufferUnlockBaseAddress(result, []) }
      let base = try #require(CVPixelBufferGetBaseAddress(result))
      for row in 0..<32 {
        let address = base.advanced(by: row * CVPixelBufferGetBytesPerRow(result))
        if format == kCVPixelFormatType_DepthFloat32 {
          address.assumingMemoryBound(to: Float.self).initialize(repeating: 1, count: 32)
        } else {
          address.initializeMemory(
            as: UInt8.self, repeating: format == kCVPixelFormatType_OneComponent8 ? 2 : 128,
            count: CVPixelBufferGetBytesPerRow(result))
        }
      }
      return result
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let worker = try await ScanRecordingWorker.create(
      directory: root, device: "synthetic-codec-test")
    let image = try buffer(kCVPixelFormatType_32BGRA)
    let depth = try ScanDepth(
      map: buffer(kCVPixelFormatType_DepthFloat32),
      confidence: buffer(kCVPixelFormatType_OneComponent8))
    var cadence = CaptureCadence()
    for i in 0..<300 where cadence.admit(Double(i) / 60) {
      try await worker.append(ScanSample(image: image, depth: depth, timestamp: Double(i) / 60))
    }
    try await worker.finish(details: nil)
    let frames = try Data(contentsOf: root.appendingPathComponent("frames.jsonl")).split(
      separator: 10
    )
    .map { try JSONDecoder().decode(CaptureFrame.self, from: Data($0)) }
    #expect(!frames.isEmpty && frames.count <= 75)
    #expect(frames.map(\.index) == Array(0..<frames.count))
    let asset = AVURLAsset(url: root.appendingPathComponent("rgb.mov"))
    let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    #expect(reader.startReading())
    var decoded = 0
    while let sample = output.copyNextSampleBuffer() {
      #expect(decoded < frames.count)
      if decoded < frames.count {
        let nanoseconds = CMTimeConvertScale(
          CMSampleBufferGetPresentationTimeStamp(sample), timescale: 1_000_000_000,
          method: .roundHalfAwayFromZero
        ).value
        #expect(nanoseconds == Int64((frames[decoded].timestamp * 1e9).rounded()))
      }
      decoded += 1
    }
    #expect(reader.status == .completed)
    #expect(decoded == frames.count)
    let manifest = try JSONDecoder().decode(
      CaptureManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
    #expect(
      manifest.frameCount == decoded && manifest.depthFrameCount == decoded
        && manifest.smoothedDepthFrameCount == decoded)
    let raw = try #require(frames.first?.depth)
    let values = try Data(contentsOf: root.appendingPathComponent(raw.file))
    #expect(values == Data(Array(repeating: [UInt8(0), 60], count: 32 * 32).flatMap { $0 }))
  }
#endif
