#if canImport(ARKit) && os(iOS)
  @preconcurrency import ARKit
  @preconcurrency import AVFoundation
  import Capture
  import Foundation

  /// The camera side of a scan: which video format the session runs, the exposure lock held
  /// while recording, each frame's exposure, and the motion-blur cue.
  ///
  /// `observe` and `exposure` run on the ARSession delegate queue; `configure`, `holdExposure` and
  /// `releaseExposure` on the MainActor. Everything shared sits behind one lock.
  final class CaptureCamera: @unchecked Sendable {
    /// Set once a device has shown that ARKit's 4K format gives no usable depth beside it.
    static let fourKUnsupportedKey = "hex.capture.fourKUnsupported"
    /// Frames to watch after the session starts before trusting the 4K format.
    static let probeFrames = 10

    private let mutex = NSLock()
    private var format: CaptureVideoFormat?
    private var control = CaptureExposureControl(mode: "auto", whiteBalance: "auto")
    private var device: AVCaptureDevice?
    private var meter = MotionBlurMeter()
    private var probed = 0
    private var sessionStarted = Date.distantPast
    /// Called once, off the main thread, when the 4K format must be abandoned.
    var onFourKRejected: (@Sendable () -> Void)?

    var motionLevel: MotionBlurMeter.Level { mutex.withLock { meter.level } }

    /// The format the session will run: ARKit's 4K one unless this device ruled it out.
    @MainActor static func videoFormat() -> (ARConfiguration.VideoFormat?, String) {
      if !UserDefaults.standard.bool(forKey: fourKUnsupportedKey),
        let format = ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution
      {
        return (format, "4k")
      }
      return (nil, "default")
    }

    @MainActor func configure(_ configuration: ARWorldTrackingConfiguration, kind: String) {
      let format = configuration.videoFormat
      mutex.withLock {
        self.format = CaptureVideoFormat(
          kind: kind, width: Int(format.imageResolution.width),
          height: Int(format.imageResolution.height), framesPerSecond: format.framesPerSecond)
        device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
        meter = MotionBlurMeter()
        probed = 0
        sessionStarted = Date()
      }
    }

    /// What finish() writes into the manifest.
    var settings: (CaptureVideoFormat?, CaptureExposureControl) {
      mutex.withLock { (format, control) }
    }

    /// Hold what auto-exposure metered, capping the shutter at 1/250 s (ExposurePlan), and
    /// lock white balance. Waits out the first second of a fresh session so it has metered.
    @MainActor func holdExposure() async {
      let age = mutex.withLock { Date().timeIntervalSince(sessionStarted) }
      if age < ExposurePlan.meteringS {
        try? await Task.sleep(for: .seconds(ExposurePlan.meteringS - age))
      }
      guard let device = mutex.withLock({ device }) else { return }
      do {
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        let format = device.activeFormat
        let plan = ExposurePlan.plan(
          meteredDurationS: device.exposureDuration.seconds, meteredISO: Double(device.iso),
          minDurationS: format.minExposureDuration.seconds,
          maxDurationS: format.maxExposureDuration.seconds,
          minISO: Double(format.minISO), maxISO: Double(format.maxISO))
        var mode = "auto"
        if plan.mode == "custom", device.isExposureModeSupported(.custom) {
          device.setExposureModeCustom(
            duration: CMTime(seconds: plan.durationS, preferredTimescale: 1_000_000),
            iso: Float(plan.iso))
          mode = "custom"
        } else if device.isExposureModeSupported(.locked) {
          device.exposureMode = .locked
          mode = "locked"
        }
        var whiteBalance = "auto"
        if device.isWhiteBalanceModeSupported(.locked) {
          device.whiteBalanceMode = .locked
          whiteBalance = "locked"
        }
        let applied = CaptureExposureControl(
          mode: mode, whiteBalance: whiteBalance,
          durationS: mode == "auto" ? nil : plan.durationS, iso: mode == "auto" ? nil : plan.iso)
        mutex.withLock { control = applied }
      } catch {
        mutex.withLock { control = CaptureExposureControl(mode: "auto", whiteBalance: "auto") }
      }
    }

    /// Back to continuous auto-exposure and white balance once a scan ends.
    @MainActor func releaseExposure() {
      guard let device = mutex.withLock({ device }), (try? device.lockForConfiguration()) != nil
      else { return }
      if device.isExposureModeSupported(.continuousAutoExposure) {
        device.exposureMode = .continuousAutoExposure
      }
      if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
        device.whiteBalanceMode = .continuousAutoWhiteBalance
      }
      device.unlockForConfiguration()
      mutex.withLock { control = CaptureExposureControl(mode: "auto", whiteBalance: "auto") }
    }

    /// One frame's exposure for the frame log. ISO and gains are the device's current values.
    func exposure(_ frame: ARFrame) -> CaptureExposure {
      let device = mutex.withLock { self.device }
      let gains = device?.deviceWhiteBalanceGains
      return CaptureExposure(
        durationS: frame.camera.exposureDuration > 0 ? frame.camera.exposureDuration : nil,
        offsetEV: Double(frame.camera.exposureOffset),
        iso: device.map { Double($0.iso) },
        whiteBalanceGains: gains.map {
          [Double($0.redGain), Double($0.greenGain), Double($0.blueGain)]
        })
    }

    /// Every delegate frame: update the blur cue, and in the first frames of a 4K session
    /// confirm that depth arrives covering the same view as the image.
    func observe(_ frame: ARFrame) {
      let k = frame.camera.intrinsics
      let depth = Self.centreDepth(frame)
      let reject = mutex.withLock { () -> Bool in
        meter.update(
          time: frame.timestamp,
          transform: (0..<4).flatMap { row in
            (0..<4).map { Double(frame.camera.transform[$0][row]) }
          },
          focalPx: Double(k[0][0]), shutterS: frame.camera.exposureDuration, depthM: depth)
        guard format?.kind == "4k", probed < Self.probeFrames else { return false }
        probed += 1
        let fits = Self.depthFits(frame)
        if fits == false || (fits == nil && probed == Self.probeFrames) {
          probed = Self.probeFrames
          return true
        }
        if fits == true { probed = Self.probeFrames }
        return false
      }
      if reject {
        UserDefaults.standard.set(true, forKey: Self.fourKUnsupportedKey)
        onFourKRejected?()
      }
    }

    /// nil while there is no depth yet; false when depth and image differ in aspect, so the
    /// depth covers another field of view than the image (normalized intrinsics would lie).
    static func depthFits(_ frame: ARFrame) -> Bool? {
      guard let map = frame.sceneDepth?.depthMap, frame.smoothedSceneDepth != nil else {
        return nil
      }
      let image = frame.camera.imageResolution
      let depthAspect = Double(CVPixelBufferGetWidth(map)) / Double(CVPixelBufferGetHeight(map))
      return abs(depthAspect / (image.width / image.height) - 1) < 0.02
    }

    /// The smoothed depth at the image centre, metres, when ARKit is confident of it.
    static func centreDepth(_ frame: ARFrame) -> Double? {
      guard let data = frame.smoothedSceneDepth, let confidence = data.confidenceMap else {
        return nil
      }
      let map = data.depthMap
      guard CVPixelBufferLockBaseAddress(map, .readOnly) == kCVReturnSuccess else { return nil }
      defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
      guard CVPixelBufferLockBaseAddress(confidence, .readOnly) == kCVReturnSuccess else {
        return nil
      }
      defer { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
      let x = CVPixelBufferGetWidth(map) / 2
      let y = CVPixelBufferGetHeight(map) / 2
      guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
        let base = CVPixelBufferGetBaseAddress(map),
        let conf = CVPixelBufferGetBaseAddress(confidence),
        conf.advanced(by: y * CVPixelBufferGetBytesPerRow(confidence) + x)
          .load(as: UInt8.self) >= UInt8(ARConfidenceLevel.medium.rawValue)
      else { return nil }
      let value = base.advanced(by: y * CVPixelBufferGetBytesPerRow(map) + x * 4)
        .load(as: Float32.self)
      return value.isFinite && value > 0 ? Double(value) : nil
    }
  }
#endif
