#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Capture
  import CoreLocation
  import CoreMedia
  import CoreVideo
  import Darwin
  import Foundation

  /// Uses the displayed ARSession so recording, perception and answer anchors share one world.
  /// Device-only: HEVC/backpressure, location permission and bundle playback need an iPhone run.
  /// This adapter is needed because M1's ARCaptureRecorder owns a private, separate ARSession.
  @MainActor
  final class SharedSessionRecorder: NSObject, RecordingControl,
    @preconcurrency CLLocationManagerDelegate
  {
    private(set) var isRecording = false
    var exportBundle: (@MainActor (URL) async throws -> Void)?
    var captureDetails: (() -> CaptureDetails)?
    var onPhase: ((CapturePhase) -> Void)?
    var onChange: ((Bool, Bool) -> Void)?
    private var disk: RecordingDiskWriter?
    private var writer: BackgroundWriter<RecordingPacket>?
    private var directory: URL?
    private var video: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var pixels: AVAssetWriterInputPixelBufferAdaptor?
    private var firstTimestamp: TimeInterval?
    private var index = 0
    private var finishing = false
    private var starting: Task<Void, Never>?
    private var generation = UUID()
    private var width = 0, height = 0
    // Core Location delivers callbacks on the run loop where this manager is created.
    private let location = CLLocationManager()

    func toggleRecording() {
      if isRecording {
        stop()
        return
      }
      guard !finishing else { return }
      finishing = true
      onPhase?(.starting)
      let id = UUID()
      generation = id
      starting = Task {
        defer { if generation == id { starting = nil } }
        do {
          let root = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Captures")
          var system = utsname()
          uname(&system)
          let device = withUnsafeBytes(of: &system.machine) {
            String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self)
          }
          let directory = root.appendingPathComponent(UUID().uuidString)
          let disk = try await RecordingDiskWriter.create(directory: directory, device: device)
          guard generation == id, !Task.isCancelled else { return }
          self.disk = disk
          self.directory = directory
          // At most four complete depth/frame packets, including the active disk write.
          writer = BackgroundWriter(capacity: 4) { try await disk.write($0) }
          firstTimestamp = nil
          index = 0
          finishing = false
          isRecording = true
          location.delegate = self
          location.requestWhenInUseAuthorization()
          location.startUpdatingLocation()
          onChange?(true, false)
          onPhase?(.recording(Date()))
        } catch { if generation == id { fail() } }
      }
    }

    func append(_ frame: ARFrame) {
      guard isRecording, let writer, let directory else { return }
      do {
        // Tracking initialization is pre-roll, just as in M1.
        if firstTimestamp == nil {
          guard case .normal = frame.camera.trackingState else { return }
          let buffer = frame.capturedImage
          width = CVPixelBufferGetWidth(buffer)
          height = CVPixelBufferGetHeight(buffer)
          let writer = try AVAssetWriter(
            outputURL: directory.appendingPathComponent("rgb.mov"), fileType: .mov)
          let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
              AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width,
              AVVideoHeightKey: height,
              AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
            ])
          // The capture buffers/intrinsics stay in sensor coordinates; this is
          // presentation metadata only, for the portrait iPhone replay/thumbnail.
          input.transform = CGAffineTransform(rotationAngle: .pi / 2)
          input.expectsMediaDataInRealTime = true
          input.mediaTimeScale = 1_000_000_000
          guard writer.canAdd(input) else { throw CaptureError.recordingFailed }
          writer.add(input)
          let pixels = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: nil)
          guard writer.startWriting() else { throw CaptureError.recordingFailed }
          writer.startSession(atSourceTime: .zero)
          self.video = writer
          self.input = input
          self.pixels = pixels
          firstTimestamp = frame.timestamp
        }
        guard let input, input.isReadyForMoreMediaData, let pixels, let firstTimestamp,
          CVPixelBufferGetWidth(frame.capturedImage) == width,
          CVPixelBufferGetHeight(frame.capturedImage) == height
        else { throw CaptureError.recordingFailed }
        let timestamp = frame.timestamp - firstTimestamp
        guard
          pixels.append(
            frame.capturedImage,
            withPresentationTime: CMTime(seconds: timestamp, preferredTimescale: 1_000_000_000))
        else { throw CaptureError.recordingFailed }
        var files: [RecordedFile] = []
        let raw = try copyDepth(frame.sceneDepth, prefix: "depth", files: &files)
        let smooth = try copyDepth(frame.smoothedSceneDepth, prefix: "smooth", files: &files)
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .notAvailable: tracking = "notAvailable"
        case .limited(let reason): tracking = "limited:\(reason)"
        @unknown default: tracking = "notAvailable"
        }
        let metadata = CaptureFrame(
          index: index, timestamp: timestamp,
          transform: (0..<4).flatMap { row in
            (0..<4).map { Double(frame.camera.transform[$0][row]) }
          },
          intrinsics: (0..<3).flatMap { row in
            (0..<3).map { Double(frame.camera.intrinsics[$0][row]) }
          },
          imageWidth: width, imageHeight: height, trackingState: tracking, depth: raw,
          smoothedDepth: smooth)
        guard writer.submit(.frame(metadata, files)) else { throw CaptureError.recordingFailed }
        index += 1
      } catch { fail() }
    }

    private func copyDepth(_ depth: ARDepthData?, prefix: String, files: inout [RecordedFile])
      throws
      -> DepthReference?
    {
      guard let depth else { return nil }
      let map = depth.depthMap
      guard let confidence = depth.confidenceMap,
        CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        CVPixelBufferGetPixelFormatType(confidence) == kCVPixelFormatType_OneComponent8
      else { throw CaptureError.incomplete }
      let width = CVPixelBufferGetWidth(map)
      let height = CVPixelBufferGetHeight(map)
      guard CVPixelBufferGetWidth(confidence) == width, CVPixelBufferGetHeight(confidence) == height
      else {
        throw CaptureError.invalidFrame
      }
      CVPixelBufferLockBaseAddress(map, .readOnly)
      CVPixelBufferLockBaseAddress(confidence, .readOnly)
      defer {
        CVPixelBufferUnlockBaseAddress(map, .readOnly)
        CVPixelBufferUnlockBaseAddress(confidence, .readOnly)
      }
      guard let base = CVPixelBufferGetBaseAddress(map),
        let conf = CVPixelBufferGetBaseAddress(confidence)
      else {
        throw CaptureError.incomplete
      }
      var values = Data(capacity: width * height * 2)
      var confidences = Data(capacity: width * height)
      for y in 0..<height {
        let row = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map)).assumingMemoryBound(
          to: Float.self)
        for x in 0..<width {
          var bits = Float16(row[x]).bitPattern.littleEndian
          withUnsafeBytes(of: &bits) { values.append(contentsOf: $0) }
        }
        confidences.append(
          conf.advanced(by: y * CVPixelBufferGetBytesPerRow(confidence)).assumingMemoryBound(
            to: UInt8.self), count: width)
      }
      let name = String(format: "%@_%08d", prefix, index)
      files.append(RecordedFile(name: name + ".f16", data: values))
      files.append(RecordedFile(name: name + ".u8", data: confidences))
      return DepthReference(
        file: name + ".f16", confidence: name + ".u8", width: width, height: height)
    }

    func stop() {
      if starting != nil {
        generation = UUID()
        starting?.cancel()
        starting = nil
        finishing = false
        onPhase?(.idle)
      }
      guard isRecording else { return }
      isRecording = false
      location.stopUpdatingLocation()
      onChange?(false, false)
      guard let video, let disk, let writer, index > 0 else {
        fail()
        return
      }
      finishing = true
      onPhase?(.stopping)
      let details = captureDetails?()
      let savedDirectory = directory
      let export = exportBundle
      input?.markAsFinished()
      Task { @MainActor [weak self] in
        // The AVFoundation callback is Sendable; only the continuation crosses to it.
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
          video.finishWriting { continuation.resume() }
        }
        do {
          guard video.status == .completed else { throw CaptureError.recordingFailed }
          // Keep finalization alive even when provider switching releases the camera model.
          try await writer.close()
          if let savedDirectory { try await export?(savedDirectory) }
          if let details { try await disk.writeDetails(details) }
          try await disk.finish()
          if self?.video === video {
            self?.reset()
            if let savedDirectory { self?.onPhase?(.saved(savedDirectory)) }
          }
        } catch {
          if self?.video === video { self?.fail() }
        }
      }
    }

    private func fail() {
      // Incomplete bundles have no manifest and are never offered for upload.
      video?.cancelWriting()
      location.stopUpdatingLocation()
      reset()
      onChange?(false, true)
      onPhase?(.failed)
    }
    private func reset() {
      generation = UUID()
      starting?.cancel()
      starting = nil
      isRecording = false
      finishing = false
      disk = nil
      writer = nil
      directory = nil
      video = nil
      input = nil
      pixels = nil
      firstTimestamp = nil
      index = 0
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
      guard isRecording, let writer else { return }
      do {
        for fix in locations {
          guard
            writer.submit(
              .location(
                CaptureLocation(
                  timestamp: fix.timestamp.timeIntervalSince1970,
                  latitude: fix.coordinate.latitude, longitude: fix.coordinate.longitude,
                  altitudeM: fix.altitude,
                  horizontalAccuracyM: fix.horizontalAccuracy,
                  verticalAccuracyM: fix.verticalAccuracy)))
          else { throw CaptureError.recordingFailed }
        }
      } catch { fail() }
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
  }
#endif
