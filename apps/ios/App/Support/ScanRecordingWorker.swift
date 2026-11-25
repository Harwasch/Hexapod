#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Accelerate
  import Capture
  import Foundation

  /// ARFrame and pixel buffers are immutable snapshots. Only a reserved queue slot retains one.
  final class ScanFrame: @unchecked Sendable {
    let frame: ARFrame
    let exposure: CaptureExposure
    init(_ frame: ARFrame, exposure: CaptureExposure = CaptureExposure()) {
      self.frame = frame
      self.exposure = exposure
    }
  }
  enum ScanPacket: Sendable {
    case frame(ScanFrame)
    case location(CaptureLocation)
  }

  struct ScanDepth: @unchecked Sendable {
    let map: CVPixelBuffer
    let confidence: CVPixelBuffer?
    init(_ depth: ARDepthData) {
      map = depth.depthMap
      confidence = depth.confidenceMap
    }
    init(map: CVPixelBuffer, confidence: CVPixelBuffer) {
      self.map = map
      self.confidence = confidence
    }
  }
  /// Test seam uses the same codec/conversion/manifest path as ARFrame packets.
  struct ScanSample: @unchecked Sendable {
    let image: CVPixelBuffer
    let raw: ScanDepth?, smooth: ScanDepth?
    let timestamp: Double
    let transform: [Double], intrinsics: [Double]
    let tracking: String
    var exposure = CaptureExposure()
    init(_ frame: ARFrame, exposure: CaptureExposure = CaptureExposure()) {
      self.exposure = exposure
      image = frame.capturedImage
      raw = frame.sceneDepth.map(ScanDepth.init)
      smooth = frame.smoothedSceneDepth.map(ScanDepth.init)
      timestamp = frame.timestamp
      transform = (0..<4).flatMap { row in (0..<4).map { Double(frame.camera.transform[$0][row]) } }
      intrinsics = (0..<3).flatMap { row in (0..<3).map { Double(frame.camera.intrinsics[$0][row]) }
      }
      switch frame.camera.trackingState {
      case .normal: tracking = "normal"
      case .notAvailable: tracking = "notAvailable"
      case .limited(let reason): tracking = "limited:\(reason)"
      @unknown default: tracking = "notAvailable"
      }
    }
    init(image: CVPixelBuffer, depth: ScanDepth?, timestamp: Double) {
      self.image = image
      raw = depth
      smooth = depth
      self.timestamp = timestamp
      tracking = "normal"
      transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
      intrinsics = [32, 0, 16, 0, 32, 16, 0, 0, 1]
    }
  }

  /// The delegate does admission + retain only. Lock also orders stop against the last admission.
  final class ScanIngress: @unchecked Sendable {
    let camera = CaptureCamera()
    private let lock = NSLock()
    private var writer: BackgroundWriter<ScanPacket>?
    private var cadence = CaptureCadence()
    func start(_ writer: BackgroundWriter<ScanPacket>) {
      lock.withLock {
        self.writer = writer
        cadence = CaptureCadence()
      }
    }
    func stop() { lock.withLock { writer = nil } }
    func offer(_ frame: ARFrame) {
      lock.withLock {
        guard let writer, frame.sceneDepth?.confidenceMap != nil,
          frame.smoothedSceneDepth?.confidenceMap != nil, cadence.admit(frame.timestamp)
        else { return }
        let exposure = camera.exposure(frame)
        writer.submit { .frame(ScanFrame(frame, exposure: exposure)) }
      }
    }
    func location(_ fix: CaptureLocation) {
      lock.withLock { _ = writer?.submit { .location(fix) } }
    }
  }

  /// All codec, conversion and disk work is serialized here, away from MainActor.
  /// Any partial I/O/codec failure prevents manifest publication. Backpressure drops before work.
  actor ScanRecordingWorker {
    private let bundle: CaptureBundleWriter
    private let directory: URL
    private var video: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var pixels: AVAssetWriterInputPixelBufferAdaptor?
    private var origin: Double?
    private var index = 0
    private var width = 0, height = 0
    init(directory: URL, device: String) throws {
      self.directory = directory
      bundle = try CaptureBundleWriter(directory: directory, device: device)
    }
    static func create(directory: URL, device: String) async throws -> ScanRecordingWorker {
      try await Task.detached { try ScanRecordingWorker(directory: directory, device: device) }
        .value
    }
    func write(_ packet: ScanPacket) throws {
      switch packet {
      case .location(let fix): try bundle.appendLocation(fix)
      case .frame(let transfer): try append(ScanSample(transfer.frame, exposure: transfer.exposure))
      }
    }
    func append(_ frame: ScanSample) throws {
      if video == nil {
        guard frame.tracking == "normal" else { return }
        width = CVPixelBufferGetWidth(frame.image)
        height = CVPixelBufferGetHeight(frame.image)
        let writer = try AVAssetWriter(
          outputURL: directory.appendingPathComponent("rgb.mov"), fileType: .mov)
        let input = AVAssetWriterInput(
          mediaType: .video,
          outputSettings: CaptureVideoSettings(width: width, height: height).outputSettings)
        input.transform = CGAffineTransform(rotationAngle: .pi / 2)
        input.expectsMediaDataInRealTime = true
        input.mediaTimeScale = 1_000_000_000
        guard writer.canAdd(input) else { throw CaptureError.recordingFailed }
        writer.add(input)
        pixels = AVAssetWriterInputPixelBufferAdaptor(
          assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.startWriting() else { throw CaptureError.recordingFailed }
        writer.startSession(atSourceTime: .zero)
        video = writer
        self.input = input
      }
      guard let video, video.status == .writing, let input, let pixels else {
        throw CaptureError.recordingFailed
      }
      // Drop the WHOLE packet before converting or appending either stream.
      guard input.isReadyForMoreMediaData else { return }
      guard CVPixelBufferGetWidth(frame.image) == width,
        CVPixelBufferGetHeight(frame.image) == height
      else { throw CaptureError.invalidFrame }
      var files: [RecordedFile] = []
      let raw = try depth(frame.raw, prefix: "depth", files: &files)
      let smooth = try depth(frame.smooth, prefix: "smooth", files: &files)
      let timestamp = frame.timestamp - (origin ?? frame.timestamp)
      let metadata = CaptureFrame(
        index: index, timestamp: timestamp,
        transform: frame.transform, intrinsics: frame.intrinsics,
        imageWidth: width, imageHeight: height, trackingState: frame.tracking, depth: raw,
        smoothedDepth: smooth, exposure: frame.exposure)
      guard
        pixels.append(
          frame.image,
          withPresentationTime: CMTime(value: Int64((timestamp * 1e9).rounded()), timescale: 1_000_000_000))
      else {
        throw CaptureError.recordingFailed
      }
      for file in files { try file.data.write(to: directory.appendingPathComponent(file.name)) }
      try bundle.append(metadata)
      origin = origin ?? frame.timestamp
      index += 1
    }
    private func depth(_ depth: ScanDepth?, prefix: String, files: inout [RecordedFile]) throws
      -> DepthReference?
    {
      guard let depth else { return nil }
      let map = depth.map
      guard let confidence = depth.confidence,
        CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8
      else {
        throw CaptureError.invalidFrame
      }
      let w = CVPixelBufferGetWidth(map)
      let h = CVPixelBufferGetHeight(map)
      guard CVPixelBufferGetWidth(confidence) == w, CVPixelBufferGetHeight(confidence) == h,
        CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess
      else { throw CaptureError.invalidFrame }
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      guard CVPixelBufferLockBaseAddress(confidence, .readOnly) == kCVReturnSuccess else {
        throw CaptureError.invalidFrame
      }
      defer { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
      guard let base = CVPixelBufferGetBaseAddress(map),
        let conf = CVPixelBufferGetBaseAddress(confidence)
      else {
        throw CaptureError.invalidFrame
      }
      var values = Data(count: w * h * 2)
      let error = values.withUnsafeMutableBytes { bytes in
        var source = vImage_Buffer(
          data: base, height: vImagePixelCount(h), width: vImagePixelCount(w),
          rowBytes: CVPixelBufferGetBytesPerRow(map))
        var target = vImage_Buffer(
          data: bytes.baseAddress!, height: vImagePixelCount(h), width: vImagePixelCount(w),
          rowBytes: w * 2)
        return vImageConvert_PlanarFtoPlanar16F(&source, &target, vImage_Flags(kvImageDoNotTile))
      }
      guard error == kvImageNoError else { throw CaptureError.invalidFrame }
      var confidences = Data(count: w * h)
      confidences.withUnsafeMutableBytes { bytes in
        for row in 0..<h {
          bytes.baseAddress!.advanced(by: row * w).copyMemory(
            from: conf.advanced(by: row * CVPixelBufferGetBytesPerRow(confidence)), byteCount: w)
        }
      }
      let name = String(format: "%@_%08d", prefix, index)
      files.append(.init(name: name + ".f16", data: values))
      files.append(.init(name: name + ".u8", data: confidences))
      return DepthReference(file: name + ".f16", confidence: name + ".u8", width: w, height: h)
    }
    func finish(
      details: CaptureDetails?,
      settings: (CaptureVideoFormat?, CaptureExposureControl)? = nil
    ) async throws {
      guard let video, index > 0 else { throw CaptureError.incomplete }
      input?.markAsFinished()
      await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
        video.finishWriting { c.resume() }
      }
      guard video.status == .completed else { throw CaptureError.recordingFailed }
      // Count finalized samples without decoding/copying pixels. A codec-side drop must
      // never publish a bundle that normalize would reject. The simulator test also decodes.
      let asset = AVURLAsset(url: directory.appendingPathComponent("rgb.mov"))
      guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw CaptureError.incomplete
      }
      let reader = try AVAssetReader(asset: asset)
      let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
      output.alwaysCopiesSampleData = false
      guard reader.canAdd(output) else { throw CaptureError.recordingFailed }
      reader.add(output)
      guard reader.startReading() else { throw CaptureError.recordingFailed }
      var samples = 0
      while let sample = output.copyNextSampleBuffer() {
        samples += CMSampleBufferGetNumSamples(sample)
      }
      guard reader.status == .completed, samples == index else {
        throw CaptureError.recordingFailed
      }
      if let details {
        try JSONEncoder().encode(details).write(
          to: directory.appendingPathComponent("capture-info.json"), options: .atomic)
      }
      if let (format, control) = settings {
        // The format is what ARKit was asked for; record it only if the frames match it.
        if let format, format.width == width, format.height == height {
          bundle.videoFormat = format
        }
        bundle.exposureControl = control
      }
      _ = try bundle.finish()
    }
    func cancel() { video?.cancelWriting() }
  }
#endif
