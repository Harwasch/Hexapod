#if canImport(ARKit) && canImport(AVFoundation) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Foundation
  import Darwin

  /// Device-only implementation: requires Xcode compilation and an iPhone acceptance run.
  /// The app owns camera permission and foreground lifecycle. Every mutable field is
  /// confined to queue, including ARSessionDelegate calls. Backpressure fails the capture
  /// rather than silently dropping a frame or publishing a misleading manifest.
  public final class ARCaptureRecorder: NSObject, ARSessionDelegate, @unchecked Sendable {
    private let queue = DispatchQueue(label: "hex.capture.recording")
    private let session = ARSession()
    private struct PendingStart {
      let directory: URL
      let device: String
      let completion: @Sendable (Result<Void, Error>) -> Void
    }
    private var pendingStart: PendingStart?
    private var bundle: CaptureBundleWriter?
    private var video: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var firstTimestamp: TimeInterval?
    private var index = 0
    private var failure: Error?
    private var finishing = false
    private var width = 0
    private var height = 0
    #if canImport(CoreLocation)
      @MainActor private var locationRecorder: CaptureLocationRecorder?
    #endif

    public override init() {
      super.init()
      session.delegate = self
      session.delegateQueue = queue
    }

    /// Completes when tracking is normal and recording starts. Initialization is
    /// pre-roll, not a recording. Use a new recorder for each finished bundle.
    public func start(directory: URL, completion: @escaping @Sendable (Result<Void, Error>) -> Void)
    {
      queue.async {
        do {
          guard self.bundle == nil, self.pendingStart == nil, !self.finishing else {
            throw CaptureError.recordingFailed
          }
          var system = utsname()
          uname(&system)
          let model = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
          }
          let generation =
            Int(
              model.replacingOccurrences(of: "iPhone", with: "").split(separator: ",").first ?? "")
            ?? 0
          let semantics: ARConfiguration.FrameSemantics = [.sceneDepth, .smoothedSceneDepth]
          guard model.hasPrefix("iPhone"), generation >= 16,
            ARWorldTrackingConfiguration.isSupported,
            ARWorldTrackingConfiguration.supportsFrameSemantics(semantics)
          else { throw CaptureError.unsupported }
          let config = ARWorldTrackingConfiguration()
          config.frameSemantics = semantics
          config.worldAlignment = .gravity
          if let format = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution {
            config.videoFormat = format
          }
          self.pendingStart = PendingStart(
            directory: directory, device: model, completion: completion)
          self.failure = nil
          self.firstTimestamp = nil
          self.index = 0
          #if canImport(CoreLocation)
            DispatchQueue.main.async {
              let recorder = CaptureLocationRecorder { [weak self] fixes in
                guard let self else { return }
                self.queue.async {
                  guard let bundle = self.bundle, self.failure == nil, !self.finishing else {
                    return
                  }
                  do {
                    for fix in fixes { try bundle.appendLocation(fix) }
                  } catch { self.fail(error) }
                }
              }
              self.locationRecorder = recorder
              recorder.start()
            }
          #endif
          self.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        } catch { completion(.failure(error)) }
      }
    }

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
      guard failure == nil, !finishing else { return }
      do {
        if let pending = pendingStart {
          guard case .normal = frame.camera.trackingState else { return }
          bundle = try CaptureBundleWriter(directory: pending.directory, device: pending.device)
          pendingStart = nil
          pending.completion(.success(()))
        }
        guard let bundle else { return }
        let pixel = frame.capturedImage
        if video == nil {
          width = CVPixelBufferGetWidth(pixel)
          height = CVPixelBufferGetHeight(pixel)
          let writer = try AVAssetWriter(
            outputURL: bundle.directory.appendingPathComponent("rgb.mov"), fileType: .mov)
          let track = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
              AVVideoCodecKey: AVVideoCodecType.hevc,
              AVVideoWidthKey: width, AVVideoHeightKey: height,
              AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
            ])
          track.expectsMediaDataInRealTime = true
          track.mediaTimeScale = 1_000_000_000
          guard writer.canAdd(track) else { throw CaptureError.recordingFailed }
          writer.add(track)
          let pixels = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: track, sourcePixelBufferAttributes: nil)
          guard writer.startWriting() else { throw writer.error ?? CaptureError.recordingFailed }
          writer.startSession(atSourceTime: .zero)
          video = writer
          input = track
          adaptor = pixels
          firstTimestamp = frame.timestamp
        }
        guard CVPixelBufferGetWidth(pixel) == width, CVPixelBufferGetHeight(pixel) == height,
          let firstTimestamp, let input, input.isReadyForMoreMediaData,
          let adaptor
        else { throw CaptureError.recordingFailed }
        let timestamp = frame.timestamp - firstTimestamp
        guard
          adaptor.append(
            pixel,
            withPresentationTime: CMTime(value: Int64((timestamp * 1e9).rounded()), timescale: 1_000_000_000))
        else { throw video?.error ?? CaptureError.recordingFailed }
        let raw = try writeDepth(frame.sceneDepth, prefix: "depth", directory: bundle.directory)
        let smoothed = try writeDepth(
          frame.smoothedSceneDepth, prefix: "smooth", directory: bundle.directory)
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .notAvailable: tracking = "notAvailable"
        case .limited(let reason): tracking = "limited:\(reason)"
        @unknown default: tracking = "notAvailable"
        }
        let transform = (0..<4).flatMap { row in
          (0..<4).map { Double(frame.camera.transform[$0][row]) }
        }
        let intrinsics = (0..<3).flatMap { row in
          (0..<3).map { Double(frame.camera.intrinsics[$0][row]) }
        }
        try bundle.append(
          CaptureFrame(
            index: index, timestamp: timestamp, transform: transform,
            intrinsics: intrinsics, imageWidth: width, imageHeight: height, trackingState: tracking,
            depth: raw, smoothedDepth: smoothed))
        index += 1
      } catch { fail(error) }
    }

    private func writeDepth(_ depth: ARDepthData?, prefix: String, directory: URL) throws
      -> DepthReference?
    {
      guard let depth else { return nil }
      guard let confidence = depth.confidenceMap else { throw CaptureError.incomplete }
      let map = depth.depthMap
      let w = CVPixelBufferGetWidth(map)
      let h = CVPixelBufferGetHeight(map)
      guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8,
        CVPixelBufferGetWidth(confidence) == w, CVPixelBufferGetHeight(confidence) == h
      else { throw CaptureError.unsupported }
      CVPixelBufferLockBaseAddress(map, .readOnly)
      CVPixelBufferLockBaseAddress(confidence, .readOnly)
      defer {
        CVPixelBufferUnlockBaseAddress(map, .readOnly)
        CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
      }
      guard let base = CVPixelBufferGetBaseAddress(map),
        let conf = CVPixelBufferGetBaseAddress(confidence)
      else { throw CaptureError.incomplete }
      var values = Data(capacity: w * h * 2)
      var confidences = Data(capacity: w * h)
      for y in 0..<h {
        let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(
          to: Float.self)
        for x in 0..<w {
          var bits = Float16(row[x]).bitPattern.littleEndian
          withUnsafeBytes(of: &bits) { values.append(contentsOf: $0) }
        }
        confidences.append(
          conf.advanced(by: y * CVPixelBufferGetBytesPerRow(confidence)).assumingMemoryBound(
            to: UInt8.self), count: w)
      }
      let name = String(format: "%@_%08d", prefix, index)
      try values.write(to: directory.appendingPathComponent(name + ".f16"))
      try confidences.write(to: directory.appendingPathComponent(name + ".u8"))
      return DepthReference(file: name + ".f16", confidence: name + ".u8", width: w, height: h)
    }

    public func session(_ session: ARSession, didFailWithError error: Error) { fail(error) }
    public func sessionWasInterrupted(_ session: ARSession) { fail(CaptureError.incomplete) }
    private func stopLocation() {
      #if canImport(CoreLocation)
        DispatchQueue.main.async {
          self.locationRecorder?.stop()
          self.locationRecorder = nil
        }
      #endif
    }

    private func fail(_ error: Error) {
      stopLocation()
      failure = error
      session.pause()
      let pending = pendingStart
      pendingStart = nil
      pending?.completion(.failure(error))
    }

    public func stop(completion: @escaping @Sendable (Result<URL, Error>) -> Void) {
      queue.async {
        self.session.pause()
        self.stopLocation()
        if self.pendingStart != nil { self.fail(CaptureError.incomplete) }
        guard !self.finishing else {
          completion(.failure(CaptureError.closed))
          return
        }
        self.finishing = true
        guard self.failure == nil, let writer = self.video, self.bundle != nil else {
          self.video?.cancelWriting()
          completion(.failure(self.failure ?? CaptureError.incomplete))
          return
        }
        self.input?.markAsFinished()
        writer.finishWriting {
          self.queue.async {
            do {
              guard let writer = self.video, let bundle = self.bundle else {
                throw CaptureError.incomplete
              }
              guard writer.status == .completed else {
                throw writer.error ?? CaptureError.recordingFailed
              }
              _ = try bundle.finish()
              completion(.success(bundle.directory))
            } catch { completion(.failure(error)) }
          }
        }
      }
    }
  }
#endif
