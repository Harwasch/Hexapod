#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Capture
  import CoreLocation
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
    var onChange: ((Bool, Bool) -> Void)?
    private var bundle: CaptureBundleWriter?
    private var video: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var pixels: AVAssetWriterInputPixelBufferAdaptor?
    private var firstTimestamp: TimeInterval?
    private var index = 0
    private var finishing = false
    private var width = 0, height = 0
    private let location = CLLocationManager()

    func toggleRecording() {
      if isRecording {
        stop()
        return
      }
      guard !finishing else { return }
      do {
        let root = try FileManager.default.url(
          for: .applicationSupportDirectory, in: .userDomainMask,
          appropriateFor: nil, create: true
        ).appendingPathComponent("Captures")
        var system = utsname()
        uname(&system)
        let device = withUnsafeBytes(of: &system.machine) {
          String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        bundle = try CaptureBundleWriter(
          directory: root.appendingPathComponent(UUID().uuidString), device: device)
        firstTimestamp = nil
        index = 0
        isRecording = true
        location.delegate = self
        location.requestWhenInUseAuthorization()
        location.startUpdatingLocation()
        onChange?(true, false)
      } catch { fail() }
    }

    func append(_ frame: ARFrame) {
      guard isRecording, let bundle else { return }
      do {
        // Tracking initialization is pre-roll, just as in M1.
        if firstTimestamp == nil {
          guard case .normal = frame.camera.trackingState else { return }
          let buffer = frame.capturedImage
          width = CVPixelBufferGetWidth(buffer)
          height = CVPixelBufferGetHeight(buffer)
          let writer = try AVAssetWriter(
            outputURL: bundle.directory.appendingPathComponent("rgb.mov"), fileType: .mov)
          let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
              AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: width,
              AVVideoHeightKey: height,
              AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
            ])
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
        let raw = try writeDepth(frame.sceneDepth, prefix: "depth", directory: bundle.directory)
        let smooth = try writeDepth(
          frame.smoothedSceneDepth, prefix: "smooth", directory: bundle.directory)
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .notAvailable: tracking = "notAvailable"
        case .limited(let reason): tracking = "limited:\(reason)"
        @unknown default: tracking = "notAvailable"
        }
        try bundle.append(
          CaptureFrame(
            index: index, timestamp: timestamp,
            transform: (0..<4).flatMap { row in
              (0..<4).map { Double(frame.camera.transform[$0][row]) }
            },
            intrinsics: (0..<3).flatMap { row in
              (0..<3).map { Double(frame.camera.intrinsics[$0][row]) }
            },
            imageWidth: width, imageHeight: height, trackingState: tracking, depth: raw,
            smoothedDepth: smooth))
        index += 1
      } catch { fail() }
    }

    private func writeDepth(_ depth: ARDepthData?, prefix: String, directory: URL) throws
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
      try values.write(to: directory.appendingPathComponent(name + ".f16"))
      try confidences.write(to: directory.appendingPathComponent(name + ".u8"))
      return DepthReference(
        file: name + ".f16", confidence: name + ".u8", width: width, height: height)
    }

    func stop() {
      guard isRecording else { return }
      isRecording = false
      location.stopUpdatingLocation()
      onChange?(false, false)
      guard let video, index > 0 else {
        fail()
        return
      }
      finishing = true
      input?.markAsFinished()
      video.finishWriting { [weak self] in
        Task { @MainActor in
          guard let self else { return }
          do {
            guard self.video?.status == .completed, let bundle = self.bundle else {
              throw CaptureError.recordingFailed
            }
            _ = try bundle.finish()
            self.reset()
          } catch { self.fail() }
        }
      }
    }

    private func fail() {
      // Incomplete bundles have no manifest and are never offered for upload.
      video?.cancelWriting()
      location.stopUpdatingLocation()
      reset()
      onChange?(false, true)
    }
    private func reset() {
      isRecording = false
      finishing = false
      bundle = nil
      video = nil
      input = nil
      pixels = nil
      firstTimestamp = nil
      index = 0
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
      guard isRecording, let bundle else { return }
      do {
        for fix in locations {
          try bundle.appendLocation(
            CaptureLocation(
              timestamp: fix.timestamp.timeIntervalSince1970,
              latitude: fix.coordinate.latitude, longitude: fix.coordinate.longitude,
              altitudeM: fix.altitude,
              horizontalAccuracyM: fix.horizontalAccuracy, verticalAccuracyM: fix.verticalAccuracy))
        }
      } catch { fail() }
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}
  }
#endif
